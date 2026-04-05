"""
v23.2 BTC/USDT Futures - MASSIVE Parameter Optimization (2,000,000 combos)
==========================================================================
GOAL: Find HIGHEST RETURN combination. Trade frequency doesn't matter.
KEY FOCUS: SL, TP, Trailing Stop, Partial Exit (exit management critical)

Architecture:
  - Main process: loads data, precomputes indicators, saves to .npz
  - Worker processes: load .npz, run JIT-compiled backtests
  - multiprocessing.Pool with initializer for each worker
"""

import numpy as np
import pandas as pd
import time
import os
import sys
import random
from math import log10
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool, cpu_count
import pickle

try:
    from numba import njit
    import numba
    HAS_NUMBA = True
    print(f"Numba {numba.__version__}", flush=True)
except ImportError:
    HAS_NUMBA = False
    print("No Numba", flush=True)
    def njit(*a, **kw):
        def w(f): return f
        if a and callable(a[0]): return a[0]
        return w

print(f"Python {sys.version}", flush=True)
NCPU = cpu_count()
print(f"CPUs: {NCPU}", flush=True)
sys.stdout.flush()

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = r"D:\filesystem\futures\btc_V1\test3"
CSV_FILES = [
    os.path.join(DATA_DIR, f"btc_usdt_5m_2020_to_now_part{i}.csv")
    for i in range(1, 4)
]
INITIAL_CAPITAL = 3000.0
FEE_RATE = 0.0004
NUM_SAMPLES = 2_000_000
SEED = 42
PROGRESS_INTERVAL = 100_000
NWORKERS = max(1, NCPU - 2)
CHUNK_PER_WORKER = 2000

CACHE_FILE = os.path.join(DATA_DIR, "_v23_2_cache.pkl")

# ============================================================
# JIT INDICATORS
# ============================================================
@njit(cache=True)
def nb_ema(data, period):
    n = len(data); r = np.empty(n); r[:] = np.nan
    a = 2.0/(period+1); s=-1
    for i in range(n):
        if not np.isnan(data[i]): s=i; break
    if s<0 or s+period>n: return r
    v=0.0
    for j in range(s,s+period): v+=data[j]
    r[s+period-1]=v/period
    for i in range(s+period,n):
        if not np.isnan(data[i]): r[i]=a*data[i]+(1-a)*r[i-1]
        else: r[i]=r[i-1]
    return r

@njit(cache=True)
def nb_sma(data, period):
    n=len(data); r=np.empty(n); r[:]=np.nan
    for i in range(period-1,n):
        s=0.0; ok=True
        for j in range(i-period+1,i+1):
            if np.isnan(data[j]): ok=False; break
            s+=data[j]
        if ok: r[i]=s/period
    return r

@njit(cache=True)
def nb_wma(data, period):
    n=len(data); r=np.empty(n); r[:]=np.nan
    ws=period*(period+1)/2.0
    for i in range(period-1,n):
        s=0.0; ok=True
        for j in range(period):
            idx=i-period+1+j
            if np.isnan(data[idx]): ok=False; break
            s+=data[idx]*(j+1)
        if ok: r[i]=s/ws
    return r

@njit(cache=True)
def nb_hma(data, period):
    hp=max(period//2,1); sp=max(int(np.sqrt(period)),1)
    wh=nb_wma(data,hp); wf=nb_wma(data,period)
    n=len(data); d=np.empty(n); d[:]=np.nan
    for i in range(n):
        if not np.isnan(wh[i]) and not np.isnan(wf[i]):
            d[i]=2.0*wh[i]-wf[i]
    return nb_wma(d,sp)

@njit(cache=True)
def nb_rsi(c, p):
    n=len(c); r=np.empty(n); r[:]=np.nan
    if n<p+1: return r
    gs=0.0; ls=0.0
    for i in range(1,p+1):
        d=c[i]-c[i-1]
        if d>0: gs+=d
        else: ls-=d
    ag=gs/p; al=ls/p
    r[p]=100.0 if al==0 else 100.0-100.0/(1+ag/al)
    for i in range(p+1,n):
        d=c[i]-c[i-1]
        g=d if d>0 else 0.0; l=-d if d<0 else 0.0
        ag=(ag*(p-1)+g)/p; al=(al*(p-1)+l)/p
        r[i]=100.0 if al==0 else 100.0-100.0/(1+ag/al)
    return r

@njit(cache=True)
def wilder(v, p):
    n=len(v); r=np.empty(n); r[:]=np.nan; s=-1
    for i in range(n):
        if not np.isnan(v[i]): s=i; break
    if s<0 or s+p>n: return r
    t=0.0
    for j in range(s,s+p): t+=v[j]
    r[s+p-1]=t/p
    for i in range(s+p,n):
        if not np.isnan(v[i]) and not np.isnan(r[i-1]):
            r[i]=(r[i-1]*(p-1)+v[i])/p
        elif not np.isnan(r[i-1]): r[i]=r[i-1]
    return r

@njit(cache=True)
def nb_adx(h, l, c, p):
    n=len(c)
    tr=np.empty(n); pdm=np.empty(n); mdm=np.empty(n)
    tr[0]=h[0]-l[0]; pdm[0]=0.0; mdm[0]=0.0
    for i in range(1,n):
        hl=h[i]-l[i]; hpc=abs(h[i]-c[i-1]); lpc=abs(l[i]-c[i-1])
        tr[i]=max(hl,max(hpc,lpc))
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        pdm[i]=up if up>dn and up>0 else 0.0
        mdm[i]=dn if dn>up and dn>0 else 0.0
    atr=wilder(tr,p); sp=wilder(pdm,p); sm=wilder(mdm,p)
    pdi=np.empty(n); mdi=np.empty(n); dx=np.empty(n)
    pdi[:]=np.nan; mdi[:]=np.nan; dx[:]=np.nan
    for i in range(n):
        if not np.isnan(atr[i]) and atr[i]>0:
            pdi[i]=100*sp[i]/atr[i]; mdi[i]=100*sm[i]/atr[i]
            ds=pdi[i]+mdi[i]
            if ds>0: dx[i]=100*abs(pdi[i]-mdi[i])/ds
    return wilder(dx,p), pdi, mdi


# ============================================================
# BACKTEST ENGINE v23.2
# ============================================================
@njit(cache=True)
def bt_v23_2(close, high, low, fast_ma, slow_ma, adx, rsi,
             adx_th, rsi_lo, rsi_hi,
             sl_pct, trail_act, trail_w,
             tp1_lev, tp1_sz,
             delay, margin, lev,
             cap0, fee):
    n=len(close)
    # cross signals
    cu=np.zeros(n, dtype=np.int8); cd=np.zeros(n, dtype=np.int8)
    for i in range(1,n):
        if (not np.isnan(fast_ma[i]) and not np.isnan(fast_ma[i-1]) and
            not np.isnan(slow_ma[i]) and not np.isnan(slow_ma[i-1])):
            if fast_ma[i]>slow_ma[i] and fast_ma[i-1]<=slow_ma[i-1]: cu[i]=1
            if fast_ma[i]<slow_ma[i] and fast_ma[i-1]>=slow_ma[i-1]: cd[i]=1
    # delay
    if delay>0:
        cu2=np.zeros(n, dtype=np.int8); cd2=np.zeros(n, dtype=np.int8)
        for i in range(delay,n): cu2[i]=cu[i-delay]; cd2[i]=cd[i-delay]
        cu=cu2; cd=cd2
    # filters
    vl=np.zeros(n, dtype=np.int8); vs=np.zeros(n, dtype=np.int8)
    for i in range(n):
        if np.isnan(adx[i]) or np.isnan(rsi[i]): continue
        if adx[i]>=adx_th and rsi_lo<=rsi[i]<=rsi_hi:
            if not np.isnan(fast_ma[i]) and not np.isnan(slow_ma[i]):
                if fast_ma[i]>slow_ma[i]: vl[i]=1
                if fast_ma[i]<slow_ma[i]: vs[i]=1

    cap=cap0; pk=cap0; mdd=0.0
    pos=0; ep=0.0; psz=0.0; pmg=0.0
    ta=False; tp_=0.0; ts=0.0
    tp1d=False; slbe=False
    tr=0; w=0; lo=0; gp=0.0; gl=0.0
    slh=0; tp1h=0; trh=0; rvh=0; liq=False; ddp=False

    sb=50
    for i in range(n):
        if (not np.isnan(fast_ma[i]) and not np.isnan(slow_ma[i]) and
            not np.isnan(adx[i]) and not np.isnan(rsi[i])):
            sb=max(i,50); break

    for i in range(sb,n):
        if liq: break
        p=close[i]; bh=high[i]; bl=low[i]
        if pk>0:
            cdd=(pk-cap)/pk*100
            if cdd>30: ddp=True
            elif cdd<15: ddp=False

        if pos!=0:
            if pos==1:
                wr=(bl-ep)/ep*lev; cr=(p-ep)/ep*lev
            else:
                wr=(ep-bh)/ep*lev; cr=(ep-p)/ep*lev
            cp_=False; xp=p; rsn=0
            esl=0.0 if slbe else sl_pct
            # SL
            if wr<=esl:
                cp_=True; rsn=1; slh+=1
                if slbe: xp=ep
                elif pos==1: xp=ep*(1+sl_pct/lev)
                else: xp=ep*(1-sl_pct/lev)
            # Liq
            if not cp_ and wr<=-0.9:
                cp_=True; rsn=3; liq=True
                xp=bl if pos==1 else bh
            # TP1
            if not cp_ and not tp1d and tp1_lev>0 and tp1_sz>0:
                if cr>=tp1_lev:
                    psz_p=psz*tp1_sz; pmg_p=pmg*tp1_sz
                    pp=cr
                    f_=psz_p*p*fee; pd_=pmg_p*pp-f_
                    cap+=pd_
                    if pd_>0: gp+=pd_
                    else: gl+=abs(pd_)
                    psz-=psz_p; pmg-=pmg_p
                    slbe=True; tp1d=True; tp1h+=1
                    if cap>pk: pk=cap
            # Trail
            if not cp_:
                if not ta and cr>=trail_act: ta=True; tp_=cr; ts=tp_+trail_w
                if ta:
                    if cr>tp_: tp_=cr; ts=tp_+trail_w
                    if cr<=ts: cp_=True; rsn=2; xp=p; trh+=1
            # Reverse
            if not cp_:
                if pos==1 and cd[i]==1 and vs[i]==1: cp_=True; rsn=4; xp=p; rvh+=1
                elif pos==-1 and cu[i]==1 and vl[i]==1: cp_=True; rsn=4; xp=p; rvh+=1

            if cp_:
                if pos==1: pp_=(xp-ep)/ep*lev
                else: pp_=(ep-xp)/ep*lev
                f_=psz*xp*fee; pd_=pmg*pp_-f_
                cap+=pd_; tr+=1
                if pd_>0: w+=1; gp+=pd_
                else: lo+=1; gl+=abs(pd_)
                if cap>pk: pk=cap
                dd=(pk-cap)/pk*100 if pk>0 else 0
                if dd>mdd: mdd=dd
                if rsn==4 and cap>0:
                    pos=-pos; ep=p
                    em=margin*0.5 if ddp else margin
                    pmg=cap*em; psz=pmg*lev/p
                    cap-=psz*p*fee
                    ta=False; tp_=0; ts=0; tp1d=False; slbe=False
                else: pos=0
                if cap<=0: liq=True; break
                continue

        if pos==0 and cap>100:
            el_=cu[i]==1 and vl[i]==1; es_=cd[i]==1 and vs[i]==1
            if el_ or es_:
                pos=1 if el_ else -1; ep=p
                em=margin*0.5 if ddp else margin
                pmg=cap*em; psz=pmg*lev/p; cap-=psz*p*fee
                ta=False; tp_=0; ts=0; tp1d=False; slbe=False

    if pos!=0 and not liq and n>0:
        xp=close[n-1]
        if pos==1: pp_=(xp-ep)/ep*lev
        else: pp_=(ep-xp)/ep*lev
        f_=psz*xp*fee; pd_=pmg*pp_-f_; cap+=pd_; tr+=1
        if pd_>0: w+=1; gp+=pd_
        else: lo+=1; gl+=abs(pd_)
    if cap>pk: pk=cap
    dd=(pk-cap)/pk*100 if pk>0 else 0
    if dd>mdd: mdd=dd
    return (tr,w,lo,gp,gl,cap,mdd,slh,liq,tp1h,trh,rvh)


# ============================================================
# PARAMETER SPACE
# ============================================================
FAST_MA_CONFIGS = [('WMA',3),('HMA',3),('HMA',5),('HMA',21),('EMA',3),('EMA',5),('EMA',7)]
SLOW_MA_CONFIGS = [('EMA',100),('EMA',150),('EMA',200),('EMA',250),('SMA',200),('SMA',300)]
ADX_PERIODS = [14,20]
TIMEFRAMES = ['10m','15m','30m']

FAST_KEYS = [f"{t}_{p}" for t,p in FAST_MA_CONFIGS]
SLOW_KEYS = [f"{t}_{p}" for t,p in SLOW_MA_CONFIGS]
ADX_THS = [25,30,35,40,45]
RSI_RS = [(25,65),(30,65),(30,70),(35,65),(35,70),(40,75)]
DELAYS = [0,1,2,3,5]
SLS = [-3,-4,-5,-6,-7,-8,-9,-10]
TACTS = [2,3,4,5,7,8,10,15]
TWS = [-1,-2,-3,-4,-5,-6]
TP1LS = [0,5,8,10,15,20,30]
TP1SS = [0,20,30,40,50]
MGNS = [20,30,40,50,60]
LEVS = [5,7,10,15,20]

ALL_LISTS = [FAST_KEYS, SLOW_KEYS, TIMEFRAMES, ADX_PERIODS, ADX_THS,
             RSI_RS, DELAYS, SLS, TACTS, TWS, TP1LS, TP1SS, MGNS, LEVS]

def total_space():
    t=1
    for l in ALL_LISTS: t*=len(l)
    return t


# ============================================================
# DATA / INDICATOR PREP
# ============================================================
def load_and_prep():
    print("Loading CSVs...", flush=True)
    t0=time.time()
    dfs=[]
    for f in CSV_FILES:
        df=pd.read_csv(f,parse_dates=['timestamp']); dfs.append(df)
        print(f"  {os.path.basename(f)}: {len(df):,}", flush=True)
    df=pd.concat(dfs,ignore_index=True)
    df.drop_duplicates(subset='timestamp',keep='first',inplace=True)
    df.sort_values('timestamp',inplace=True)
    df.reset_index(drop=True,inplace=True)
    df.set_index('timestamp',inplace=True)
    for c in ['open','high','low','close','volume']: df[c]=df[c].astype(np.float64)
    print(f"Total: {len(df):,} | {df.index[0]} - {df.index[-1]} | {time.time()-t0:.1f}s", flush=True)
    return df


def calc_fm(c, t, p):
    if t=='EMA': return nb_ema(c,p)
    if t=='WMA': return nb_wma(c,p)
    if t=='HMA': return nb_hma(c,p)
    return nb_ema(c,p)

def calc_sm(c, t, p):
    if t=='EMA': return nb_ema(c,p)
    if t=='SMA': return nb_sma(c,p)
    return nb_ema(c,p)


def precompute_all(df5m):
    """Precompute all indicators for all TFs. Store in a flat dict for pickling."""
    caches = {}
    for tf in TIMEFRAMES:
        t0=time.time()
        rm = {'10m':'10min','15m':'15min','30m':'30min'}
        dfr = df5m.resample(rm[tf]).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        c=dfr['close'].values.astype(np.float64)
        h=dfr['high'].values.astype(np.float64)
        l=dfr['low'].values.astype(np.float64)

        d = {'close':c, 'high':h, 'low':l, 'rsi':nb_rsi(c,14)}
        for mt,mp in FAST_MA_CONFIGS:
            d[f"fm_{mt}_{mp}"] = calc_fm(c,mt,mp)
        for mt,mp in SLOW_MA_CONFIGS:
            d[f"sm_{mt}_{mp}"] = calc_sm(c,mt,mp)
        for ap in ADX_PERIODS:
            adx_v,_,_ = nb_adx(h,l,c,ap)
            d[f"adx_{ap}"] = adx_v
        caches[tf] = d
        print(f"  [{tf}] {len(c):,} bars, {time.time()-t0:.1f}s", flush=True)
    return caches


# ============================================================
# WORKER
# ============================================================
_W = None  # worker-local cache

def _init_worker(cache_file):
    global _W
    with open(cache_file, 'rb') as f:
        _W = pickle.load(f)
    # Warmup JIT in each worker
    c = _W['30m']
    _ = bt_v23_2(
        c['close'][:100], c['high'][:100], c['low'][:100],
        c['fm_WMA_3'][:100], c['sm_EMA_100'][:100],
        c['adx_14'][:100], c['rsi'][:100],
        30.0, 30.0, 70.0, -0.05, 0.05, -0.03, 0.10, 0.30,
        0, 0.20, 10.0, 3000.0, 0.0004
    )


def _run_chunk(combo_indices):
    """Run a chunk of combo indices. Returns list of result tuples."""
    global _W
    results = []
    for row in combo_indices:
        fk = FAST_KEYS[row[0]]
        sk = SLOW_KEYS[row[1]]
        tf = TIMEFRAMES[row[2]]
        ap = ADX_PERIODS[row[3]]
        at = float(ADX_THS[row[4]])
        rl, rh = RSI_RS[row[5]]
        dl = DELAYS[row[6]]
        sl = SLS[row[7]] / 100.0
        ta = TACTS[row[8]] / 100.0
        tw = TWS[row[9]] / 100.0
        t1l = TP1LS[row[10]] / 100.0
        t1s = TP1SS[row[11]] / 100.0
        mg = MGNS[row[12]] / 100.0
        lv = float(LEVS[row[13]])

        d = _W[tf]
        # Lookup indicator arrays by constructed key
        fmt, fmp = fk.split('_', 1)
        smt_sp = sk.split('_', 1)

        res = bt_v23_2(
            d['close'], d['high'], d['low'],
            d[f"fm_{fk}"], d[f"sm_{sk}"],
            d[f"adx_{ap}"], d['rsi'],
            at, float(rl), float(rh),
            sl, ta, tw, t1l, t1s,
            dl, mg, lv,
            INITIAL_CAPITAL, FEE_RATE
        )
        trades,w,l,gp,gl,fb,mdd,slh,liq,tp1h,trh,rvh = res
        if trades<5 or liq: continue
        ret = (fb - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        pf = gp/gl if gl>0 else (999.0 if gp>0 else 0.0)
        wr = w/trades*100
        sc = ret + min(pf,100)*10 - mdd/2

        results.append((
            row[0],row[1],row[2],row[3],row[4],row[5],row[6],
            row[7],row[8],row[9],row[10],row[11],row[12],row[13],
            trades, w, l, round(wr,1), round(pf,2),
            round(ret,2), round(fb,2), round(mdd,2),
            slh, tp1h, trh, rvh, round(sc,2)
        ))
    return results


# ============================================================
# MAIN
# ============================================================
def main():
    T0 = time.time()
    ts = total_space()
    print(f"\nSpace: {ts:,} | Sampling: {NUM_SAMPLES:,} | Seed: {SEED}", flush=True)
    print(f"Workers: {NWORKERS}\n", flush=True)

    df5m = load_and_prep()
    print(flush=True)

    print("Precomputing indicators...", flush=True)
    t0=time.time()
    caches = precompute_all(df5m)
    it = time.time()-t0
    print(f"Indicators: {it:.1f}s\n", flush=True)

    # Save cache for workers
    print("Saving cache for workers...", flush=True)
    t0=time.time()
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(caches, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cache saved: {os.path.getsize(CACHE_FILE)/1e6:.1f}MB in {time.time()-t0:.1f}s\n", flush=True)

    # Free main process memory
    del df5m, caches

    # Generate combos
    print(f"Generating {NUM_SAMPLES:,} combos...", flush=True)
    rng = np.random.RandomState(SEED)
    lens = [len(l) for l in ALL_LISTS]
    combos = np.column_stack([rng.randint(0, l, size=NUM_SAMPLES) for l in lens]).astype(np.int32)
    print(f"Done: {combos.shape}\n", flush=True)

    # Split into chunks for pool
    chunks = []
    for i in range(0, NUM_SAMPLES, CHUNK_PER_WORKER):
        chunks.append(combos[i:i+CHUNK_PER_WORKER])
    print(f"Chunks: {len(chunks)} x ~{CHUNK_PER_WORKER}\n", flush=True)

    # Run with pool
    print("="*70, flush=True)
    print("RUNNING OPTIMIZATION", flush=True)
    print("="*70, flush=True)

    all_results = []
    bt_start = time.time()
    processed = 0

    with Pool(processes=NWORKERS, initializer=_init_worker, initargs=(CACHE_FILE,)) as pool:
        for chunk_res in pool.imap_unordered(_run_chunk, chunks, chunksize=1):
            all_results.extend(chunk_res)
            processed += CHUNK_PER_WORKER  # approximate

            if processed % PROGRESS_INTERVAL < CHUNK_PER_WORKER or processed >= NUM_SAMPLES:
                elapsed = time.time() - bt_start
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (NUM_SAMPLES - processed) / rate if rate > 0 else 0
                valid = len(all_results)
                if all_results:
                    best = max(all_results, key=lambda x: x[26])
                    bi = best
                    bret = bi[19]
                    bpf = bi[18]
                    bmdd = bi[21]
                    bsc = bi[26]
                    print(f"  [{min(processed,NUM_SAMPLES):>10,}/{NUM_SAMPLES:,}] "
                          f"{elapsed:7.1f}s | {rate:,.0f}/s | ETA {eta:5.0f}s | "
                          f"Valid: {valid:,} | "
                          f"Best: {bret:+,.1f}% PF={bpf:.1f} MDD={bmdd:.1f}% Sc={bsc:.1f}",
                          flush=True)
                else:
                    print(f"  [{min(processed,NUM_SAMPLES):>10,}/{NUM_SAMPLES:,}] "
                          f"{elapsed:7.1f}s | {rate:,.0f}/s | Valid: {valid:,}", flush=True)

    bt_elapsed = time.time() - bt_start
    print(f"\nBacktest: {bt_elapsed:.1f}s | {NUM_SAMPLES/bt_elapsed:,.0f}/s | Valid: {len(all_results):,}\n", flush=True)

    # Cleanup cache
    try: os.remove(CACHE_FILE)
    except: pass

    if not all_results:
        print("ERROR: No valid results!"); return

    # Decode and sort
    COL = ['fast_ma','slow_ma','tf','adx_period','adx_thresh',
           'rsi_lo','rsi_hi','delay','sl_pct','trail_act','trail_width',
           'tp1_level','tp1_size','margin_pct','leverage',
           'trades','wins','losses','win_rate','pf','return_pct','final_bal',
           'max_dd','sl_hits','tp1_hits','trail_hits','reverse_hits','score']

    def decode(r):
        return {
            'fast_ma': FAST_KEYS[r[0]], 'slow_ma': SLOW_KEYS[r[1]],
            'tf': TIMEFRAMES[r[2]], 'adx_period': ADX_PERIODS[r[3]],
            'adx_thresh': ADX_THS[r[4]],
            'rsi_lo': RSI_RS[r[5]][0], 'rsi_hi': RSI_RS[r[5]][1],
            'delay': DELAYS[r[6]],
            'sl_pct': SLS[r[7]], 'trail_act': TACTS[r[8]], 'trail_width': TWS[r[9]],
            'tp1_level': TP1LS[r[10]], 'tp1_size': TP1SS[r[11]],
            'margin_pct': MGNS[r[12]], 'leverage': LEVS[r[13]],
            'trades': r[14], 'wins': r[15], 'losses': r[16], 'win_rate': r[17],
            'pf': r[18], 'return_pct': r[19], 'final_bal': r[20],
            'max_dd': r[21], 'sl_hits': r[22], 'tp1_hits': r[23],
            'trail_hits': r[24], 'reverse_hits': r[25], 'score': r[26],
        }

    all_results.sort(key=lambda x: x[26], reverse=True)
    result_dicts = [decode(r) for r in all_results[:500]]

    # Save top 200 CSV
    top200 = result_dicts[:200]
    df_top = pd.DataFrame(top200)
    csv_p = os.path.join(DATA_DIR, "v23_2_opt_top200.csv")
    df_top.to_csv(csv_p, index=False)
    print(f"Saved: {csv_p}", flush=True)

    # Summary file
    sum_p = os.path.join(DATA_DIR, "v23_2_opt_summary.txt")
    with open(sum_p, 'w', encoding='utf-8') as f:
        f.write("="*80+"\n")
        f.write("v23.2 MASSIVE PARAMETER OPTIMIZATION RESULTS\n")
        f.write("="*80+"\n\n")
        f.write(f"Total space:      {ts:,}\n")
        f.write(f"Sampled:          {NUM_SAMPLES:,}\n")
        f.write(f"Valid:            {len(all_results):,}\n")
        f.write(f"Capital:          ${INITIAL_CAPITAL:,.0f}\n")
        f.write(f"Fee:              {FEE_RATE*100:.2f}%\n")
        f.write(f"Indicator time:   {it:.1f}s\n")
        f.write(f"Backtest time:    {bt_elapsed:.1f}s\n")
        f.write(f"Total time:       {time.time()-T0:.1f}s\n")
        f.write(f"Speed:            {NUM_SAMPLES/bt_elapsed:,.0f}/s\n")
        f.write(f"Workers:          {NWORKERS}\n\n")

        f.write("="*80+"\n")
        f.write("TOP 30 (Score = Return% + PF*10 - MDD%/2)\n")
        f.write("="*80+"\n\n")
        for i,r in enumerate(top200[:30], 1):
            f.write(f"--- #{i} ---\n")
            f.write(f"  Score:    {r['score']:.2f}\n")
            f.write(f"  Return:   {r['return_pct']:+,.2f}% (${r['final_bal']:,.2f})\n")
            f.write(f"  PF:       {r['pf']:.2f} | WR: {r['win_rate']:.1f}%\n")
            f.write(f"  MDD:      {r['max_dd']:.2f}%\n")
            f.write(f"  Trades:   {r['trades']} (W:{r['wins']} L:{r['losses']})\n")
            f.write(f"  Exits:    SL={r['sl_hits']} TP1={r['tp1_hits']} Trail={r['trail_hits']} Rev={r['reverse_hits']}\n")
            f.write(f"  Params:   {r['fast_ma']}/{r['slow_ma']} {r['tf']}\n")
            f.write(f"            ADX({r['adx_period']})>={r['adx_thresh']} RSI {r['rsi_lo']}-{r['rsi_hi']} Delay={r['delay']}\n")
            f.write(f"            SL={r['sl_pct']}% Trail={r['trail_act']}%/w{r['trail_width']}%\n")
            f.write(f"            TP1={r['tp1_level']}%@{r['tp1_size']}%\n")
            f.write(f"            Margin={r['margin_pct']}% Lev={r['leverage']}x\n\n")

        # Distribution
        f.write("="*80+"\n")
        f.write("PARAMETER DISTRIBUTION (Top 100)\n")
        f.write("="*80+"\n\n")
        top100 = result_dicts[:100]
        for pk in ['fast_ma','slow_ma','tf','adx_period','adx_thresh',
                    'sl_pct','trail_act','trail_width','tp1_level','tp1_size',
                    'margin_pct','leverage','delay']:
            cts = defaultdict(int)
            for r in top100: cts[r[pk]]+=1
            sc = sorted(cts.items(), key=lambda x:x[1], reverse=True)
            f.write(f"  {pk}:\n")
            for v,c in sc:
                f.write(f"    {str(v):>15}: {c:3d} {'#'*c}\n")
            f.write("\n")
        # RSI
        cts = defaultdict(int)
        for r in top100: cts[f"{r['rsi_lo']}-{r['rsi_hi']}"]+=1
        sc = sorted(cts.items(), key=lambda x:x[1], reverse=True)
        f.write(f"  RSI range:\n")
        for v,c in sc: f.write(f"    {v:>15}: {c:3d} {'#'*c}\n")
        f.write("\n")

        # Exit analysis
        f.write("="*80+"\n")
        f.write("EXIT TYPE ANALYSIS (Top 100)\n")
        f.write("="*80+"\n\n")
        tsl=sum(r['sl_hits'] for r in top100)
        ttp=sum(r['tp1_hits'] for r in top100)
        ttr=sum(r['trail_hits'] for r in top100)
        trv=sum(r['reverse_hits'] for r in top100)
        ta_=max(tsl+ttr+trv,1)
        f.write(f"  SL:    {tsl:,} ({tsl/ta_*100:.1f}%)\n")
        f.write(f"  TP1:   {ttp:,} (partial)\n")
        f.write(f"  Trail: {ttr:,} ({ttr/ta_*100:.1f}%)\n")
        f.write(f"  Rev:   {trv:,} ({trv/ta_*100:.1f}%)\n\n")

        # Stats
        f.write("="*80+"\n")
        f.write("STATS (all valid)\n")
        f.write("="*80+"\n\n")
        # Use raw tuples for stats (all_results)
        rets = [r[19] for r in all_results]
        pfs = [r[18] for r in all_results]
        mdds = [r[21] for r in all_results]
        f.write(f"  Return: mean={np.mean(rets):+.1f}% med={np.median(rets):+.1f}% "
                f"max={np.max(rets):+,.1f}% min={np.min(rets):+,.1f}%\n")
        f.write(f"  PF:     mean={np.mean(pfs):.2f} med={np.median(pfs):.2f} max={min(np.max(pfs),999):.2f}\n")
        f.write(f"  MDD:    mean={np.mean(mdds):.1f}% med={np.median(mdds):.1f}% max={np.max(mdds):.1f}%\n\n")
        bkts = [(-100,0),(0,100),(100,500),(500,1000),(1000,5000),(5000,10000),(10000,50000),(50000,1e9)]
        f.write("  Return distribution:\n")
        for lo,hi in bkts:
            cnt=sum(1 for r in rets if lo<=r<hi)
            pct=cnt/max(len(rets),1)*100
            lb=f"{int(lo):>8,}%+" if hi>=1e9 else f"{int(lo):>8,}%-{int(hi):>8,}%"
            f.write(f"    {lb}: {cnt:>6,} ({pct:5.1f}%) {'#'*int(pct)}\n")

    print(f"Saved: {sum_p}\n", flush=True)

    # Print top 10
    print("="*70, flush=True)
    print("TOP 10", flush=True)
    print("="*70, flush=True)
    for i,r in enumerate(top200[:10],1):
        print(f"\n  #{i}: Sc={r['score']:.1f} | "
              f"Ret={r['return_pct']:+,.1f}% (${r['final_bal']:,.0f}) | "
              f"PF={r['pf']:.1f} WR={r['win_rate']:.0f}% MDD={r['max_dd']:.1f}% "
              f"Tr={r['trades']}", flush=True)
        print(f"      {r['fast_ma']}/{r['slow_ma']} {r['tf']} "
              f"ADX({r['adx_period']})>={r['adx_thresh']} "
              f"RSI {r['rsi_lo']}-{r['rsi_hi']} D={r['delay']}", flush=True)
        print(f"      SL={r['sl_pct']}% Trail={r['trail_act']}%/w{r['trail_width']}% "
              f"TP1={r['tp1_level']}%@{r['tp1_size']}% "
              f"Mg={r['margin_pct']}% Lv={r['leverage']}x", flush=True)
        print(f"      Exits: SL={r['sl_hits']} TP1={r['tp1_hits']} "
              f"Trail={r['trail_hits']} Rev={r['reverse_hits']}", flush=True)

    te=time.time()-T0
    print(f"\n{'='*70}", flush=True)
    print(f"TOTAL: {te:.1f}s ({te/60:.1f} min)", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
