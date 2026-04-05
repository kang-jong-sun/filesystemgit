"""
v23.3 BTC/USDT Futures - Complete Optimization + Backtest + Verification Pipeline
==================================================================================
KEY CHANGES from v23.2:
  - Initial capital: $5,000 (was $3,000)
  - CROSS margin mode (was ISOLATED) - position_size = balance * margin_eff * leverage
  - PF target: >= 5 (was >= 8)
  - Wider indicator set (WMA(250) added to slow MAs)
  - Score = PF*5 + log10(max(Return%,1))*10 + min(Trades,300)/30 - MDD%/3
  - Entry delay: 0,1,2,3,5 candles
  - Margin effective: 10,15,20,25% (CROSS margin with small positions)
  - Leverage: 5,7,10
  - 1,200,000 random sample combinations

PHASE 1: Optimization (1,200,000 combos)
PHASE 2: Top 3 Backtest with full details
PHASE 3: 30-run Verification per top 3 strategy
"""

import numpy as np
import pandas as pd
import time
import os
import sys
import io
import random
from math import log10
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool, cpu_count
import pickle

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    from numba import njit
    import numba
    HAS_NUMBA = True
    print(f"Numba {numba.__version__}", flush=True)
except ImportError:
    HAS_NUMBA = False
    print("No Numba - running pure Python (slower)", flush=True)
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
INITIAL_CAPITAL = 5000.0
FEE_RATE = 0.0004
NUM_SAMPLES = 1_200_000
SEED = 42
PROGRESS_INTERVAL = 100_000
NWORKERS = max(1, NCPU - 2)
CHUNK_PER_WORKER = 2000

CACHE_FILE = os.path.join(DATA_DIR, "_v23_3_cache.pkl")

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
# BACKTEST ENGINE v23.3 - CROSS MARGIN
# ============================================================
@njit(cache=True)
def bt_v23_3(close, high, low, fast_ma, slow_ma, adx, rsi,
             adx_th, rsi_lo, rsi_hi,
             sl_pct, trail_act, trail_w,
             delay, margin_eff, lev,
             cap0, fee):
    """
    CROSS margin backtest.
    position_size = balance * margin_eff * leverage
    SL loss is capped at balance (no liquidation beyond balance in cross mode).
    No TP1/partial exit in optimization - simplifies for speed.
    Trail on CLOSE, SL on HIGH/LOW.
    REVERSE: opposite cross + ADX + RSI -> close + flip.
    Same-direction skip.
    NO DD protection or monthly limits in optimization.
    """
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
    tr=0; w=0; lo=0; gp=0.0; gl=0.0
    slh=0; trh=0; rvh=0; liq=False

    sb=50
    for i in range(n):
        if (not np.isnan(fast_ma[i]) and not np.isnan(slow_ma[i]) and
            not np.isnan(adx[i]) and not np.isnan(rsi[i])):
            sb=max(i,50); break

    for i in range(sb,n):
        if liq: break
        p=close[i]; bh=high[i]; bl=low[i]

        if pos!=0:
            if pos==1:
                wr=(bl-ep)/ep*lev; cr=(p-ep)/ep*lev
            else:
                wr=(ep-bh)/ep*lev; cr=(ep-p)/ep*lev

            cp_=False; xp=p; rsn=0

            # SL check on HIGH/LOW
            if wr<=sl_pct:
                cp_=True; rsn=1; slh+=1
                if pos==1: xp=ep*(1+sl_pct/lev)
                else: xp=ep*(1-sl_pct/lev)

            # Cross margin: cap SL loss at balance
            if not cp_ and wr<=-0.95:
                cp_=True; rsn=3; liq=True
                xp=bl if pos==1 else bh

            # Trail on CLOSE
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
                f_=psz*xp*fee
                # CROSS margin: PnL = margin * roi - fees, capped at -balance
                pd_=pmg*pp_-f_
                if pd_ < -cap: pd_ = -cap  # cross margin cap
                cap+=pd_; tr+=1
                if pd_>0: w+=1; gp+=pd_
                else: lo+=1; gl+=abs(pd_)
                if cap>pk: pk=cap
                dd=(pk-cap)/pk*100 if pk>0 else 0
                if dd>mdd: mdd=dd

                if rsn==4 and cap>100:
                    # Reverse: flip position
                    pos=-pos; ep=p
                    pmg=cap*margin_eff; psz=pmg*lev/p
                    cap-=psz*p*fee
                    ta=False; tp_=0; ts=0
                else:
                    pos=0
                if cap<=0: liq=True; break
                continue

        # Entry
        if pos==0 and cap>100:
            el_=cu[i]==1 and vl[i]==1; es_=cd[i]==1 and vs[i]==1
            if el_ or es_:
                pos=1 if el_ else -1; ep=p
                # CROSS margin: position_size = balance * margin_eff * leverage
                pmg=cap*margin_eff; psz=pmg*lev/p
                cap-=psz*p*fee
                ta=False; tp_=0; ts=0

    # Close remaining position at end
    if pos!=0 and not liq and n>0:
        xp=close[n-1]
        if pos==1: pp_=(xp-ep)/ep*lev
        else: pp_=(ep-xp)/ep*lev
        f_=psz*xp*fee; pd_=pmg*pp_-f_
        if pd_ < -cap: pd_ = -cap
        cap+=pd_; tr+=1
        if pd_>0: w+=1; gp+=pd_
        else: lo+=1; gl+=abs(pd_)
    if cap>pk: pk=cap
    dd=(pk-cap)/pk*100 if pk>0 else 0
    if dd>mdd: mdd=dd
    return (tr,w,lo,gp,gl,cap,mdd,slh,liq,trh,rvh)


# ============================================================
# PARAMETER SPACE
# ============================================================
FAST_MA_CONFIGS = [('WMA',3),('HMA',3),('HMA',5),('EMA',3),('EMA',5),('EMA',7)]
SLOW_MA_CONFIGS = [('EMA',100),('EMA',150),('EMA',200),('EMA',250),('WMA',250),('SMA',300)]
ADX_PERIODS = [14,20]
TIMEFRAMES = ['10m','15m','30m']

FAST_KEYS = [f"{t}_{p}" for t,p in FAST_MA_CONFIGS]
SLOW_KEYS = [f"{t}_{p}" for t,p in SLOW_MA_CONFIGS]
ADX_THS = [25,30,35,40,45]
RSI_RS = [(25,65),(30,65),(30,70),(35,65),(35,70),(40,75)]
DELAYS = [0,1,2,3,5]
SLS = [-5,-6,-7,-8,-10,-12]
TACTS = [2,3,4,5,7,10]
TWS = [-1,-2,-3,-4,-5]
MGNS = [10,15,20,25]  # CROSS margin with small positions
LEVS = [5,7,10]

ALL_LISTS = [FAST_KEYS, SLOW_KEYS, TIMEFRAMES, ADX_PERIODS, ADX_THS,
             RSI_RS, DELAYS, SLS, TACTS, TWS, MGNS, LEVS]

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
    if t=='WMA': return nb_wma(c,p)
    return nb_ema(c,p)


def precompute_all(df5m):
    """Precompute all indicators for all TFs."""
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
    _ = bt_v23_3(
        c['close'][:100], c['high'][:100], c['low'][:100],
        c['fm_WMA_3'][:100], c['sm_EMA_100'][:100],
        c['adx_14'][:100], c['rsi'][:100],
        30.0, 30.0, 70.0, -0.05, 0.05, -0.03,
        0, 0.20, 10.0, 5000.0, 0.0004
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
        mg = MGNS[row[10]] / 100.0
        lv = float(LEVS[row[11]])

        d = _W[tf]

        res = bt_v23_3(
            d['close'], d['high'], d['low'],
            d[f"fm_{fk}"], d[f"sm_{sk}"],
            d[f"adx_{ap}"], d['rsi'],
            at, float(rl), float(rh),
            sl, ta, tw,
            dl, mg, lv,
            INITIAL_CAPITAL, FEE_RATE
        )
        trades,w,l,gp,gl,fb,mdd,slh,liq,trh,rvh = res
        if trades<10 or liq: continue
        if fb <= INITIAL_CAPITAL: continue
        pf = gp/gl if gl>0 else (999.0 if gp>0 else 0.0)
        if pf <= 1.0: continue
        ret = (fb - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        wr = w/trades*100

        # Score = PF*5 + log10(max(Return%,1))*10 + min(Trades,300)/30 - MDD%/3
        sc = pf*5.0 + log10(max(ret, 1.0))*10.0 + min(trades, 300)/30.0 - mdd/3.0

        results.append((
            row[0],row[1],row[2],row[3],row[4],row[5],row[6],
            row[7],row[8],row[9],row[10],row[11],
            trades, w, l, round(wr,1), round(pf,2),
            round(ret,2), round(fb,2), round(mdd,2),
            slh, trh, rvh, round(sc,2)
        ))
    return results


# ============================================================
# PHASE 2: DETAILED BACKTEST ENGINE (pure Python, with trades)
# ============================================================
def wilder_py(arr, p):
    out = np.full(len(arr), np.nan)
    s = 0
    while s < len(arr) and np.isnan(arr[s]): s += 1
    if s + p > len(arr): return out
    out[s+p-1] = np.nanmean(arr[s:s+p])
    for i in range(s+p, len(arr)):
        if not np.isnan(arr[i]) and not np.isnan(out[i-1]):
            out[i] = (out[i-1]*(p-1) + arr[i]) / p
    return out

def calc_adx_wilder(high, low, close, period=14):
    n = len(high)
    tr = np.full(n, np.nan)
    pdm = np.full(n, np.nan)
    mdm = np.full(n, np.nan)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
        up = high[i] - high[i-1]
        dn = low[i-1] - low[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
    atr = wilder_py(tr, period)
    spdm = wilder_py(pdm, period)
    smdm = wilder_py(mdm, period)
    pdi = np.full(n, np.nan)
    mdi = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr[i]) and atr[i] > 0:
            pdi[i] = 100 * spdm[i] / atr[i]
            mdi[i] = 100 * smdm[i] / atr[i]
            s = pdi[i] + mdi[i]
            dx[i] = 100 * abs(pdi[i] - mdi[i]) / s if s > 0 else 0
    adx = wilder_py(dx, period)
    return adx, atr

def calc_wma_py(close, period):
    n = len(close)
    out = np.full(n, np.nan)
    weights = np.arange(1, period+1, dtype=float)
    wsum = weights.sum()
    for i in range(period-1, n):
        sl = close[i-period+1:i+1]
        if not np.any(np.isnan(sl)):
            out[i] = np.dot(sl, weights) / wsum
    return out

def calc_ema_py(close, period):
    out = np.full(len(close), np.nan)
    s = 0
    while s < len(close) and np.isnan(close[s]): s += 1
    if s >= len(close): return out
    out[s] = close[s]
    m = 2.0 / (period + 1)
    for i in range(s+1, len(close)):
        if not np.isnan(close[i]) and not np.isnan(out[i-1]):
            out[i] = close[i] * m + out[i-1] * (1 - m)
    return out

def calc_sma_py(close, period):
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period-1, n):
        sl = close[i-period+1:i+1]
        if not np.any(np.isnan(sl)):
            out[i] = np.mean(sl)
    return out

def calc_hma_py(close, period):
    hp = max(period//2, 1)
    sp = max(int(np.sqrt(period)), 1)
    wh = calc_wma_py(close, hp)
    wf = calc_wma_py(close, period)
    d = 2.0 * wh - wf
    return calc_wma_py(d, sp)

def calc_rsi_py(close, period=14):
    n = len(close)
    out = np.full(n, np.nan)
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        d = close[i] - close[i-1]
        if d > 0: gains[i] = d
        else: losses[i] = -d
    avg_g = wilder_py(gains, period)
    avg_l = wilder_py(losses, period)
    for i in range(n):
        if not np.isnan(avg_g[i]) and not np.isnan(avg_l[i]):
            if avg_l[i] == 0:
                out[i] = 100.0
            else:
                out[i] = 100 - 100 / (1 + avg_g[i] / avg_l[i])
    return out

def calc_fast_ma_py(close, ma_type, period):
    if ma_type == 'WMA': return calc_wma_py(close, period)
    if ma_type == 'HMA': return calc_hma_py(close, period)
    if ma_type == 'EMA': return calc_ema_py(close, period)
    return calc_ema_py(close, period)

def calc_slow_ma_py(close, ma_type, period):
    if ma_type == 'EMA': return calc_ema_py(close, period)
    if ma_type == 'SMA': return calc_sma_py(close, period)
    if ma_type == 'WMA': return calc_wma_py(close, period)
    return calc_ema_py(close, period)


def load_data_phase2(tf_str):
    """Load 5m data and resample to target timeframe."""
    t0 = time.time()
    fs = [os.path.join(DATA_DIR, f"btc_usdt_5m_2020_to_now_part{i}.csv") for i in range(1, 4)]
    df = pd.concat([pd.read_csv(f, parse_dates=['timestamp']) for f in fs], ignore_index=True)
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df.set_index('timestamp', inplace=True)

    rm = {'5m': None, '10m':'10min','15m':'15min','30m':'30min'}
    if tf_str == '5m':
        out = df[['open','high','low','close','volume']].copy()
    else:
        out = df.resample(rm[tf_str]).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

    print(f"  [Data] 5m:{len(df):,} -> {tf_str}:{len(out):,} | {time.time()-t0:.1f}s", flush=True)
    return out


def run_detailed_backtest(cfg, opens, highs, lows, closes, adx, rsi_arr, fast_ma, slow_ma, timestamps, return_trades=True):
    """
    Detailed backtest with CROSS margin, DD protection, monthly limits.
    Phase 2 engine.
    """
    n = len(closes)
    bal = INITIAL_CAPITAL
    peak_bal = INITIAL_CAPITAL
    pos_dir = 0
    pos_entry = 0.0
    pos_size = 0.0  # notional
    pos_margin = 0.0
    pos_time_idx = 0
    pos_highest = 0.0
    pos_lowest = 0.0
    pos_trail_active = False
    pos_trail_sl = 0.0
    pos_sl = 0.0
    pos_rem = 1.0

    pending_signal = 0
    pending_bar = 0

    trades = []
    consec_loss = 0
    monthly_start = {}

    _n_trades = 0; _n_wins = 0
    _gross_profit = 0.0; _gross_loss = 0.0
    _win_roi_sum = 0.0; _loss_roi_sum = 0.0
    _sl_count = 0; _trail_count = 0; _rev_count = 0; _end_count = 0
    _max_consec_loss = 0; _cur_consec_loss = 0
    _mdd = 0.0; _mdd_peak = INITIAL_CAPITAL

    tf_hours = {'5m': 1/12, '10m': 1/6, '15m': 0.25, '30m': 0.5}
    bar_h = tf_hours.get(cfg['tf'], 0.25)
    warmup = max(300, int(cfg['slow_period'] * 1.5))

    dd_halved = False  # DD protection: if DD > 30%, halve position

    def close_pos(exit_price, exit_reason, bar_idx):
        nonlocal bal, peak_bal, pos_dir, consec_loss
        nonlocal _n_trades, _n_wins, _gross_profit, _gross_loss
        nonlocal _win_roi_sum, _loss_roi_sum, _sl_count, _trail_count, _rev_count
        nonlocal _end_count, _max_consec_loss, _cur_consec_loss
        nonlocal _mdd, _mdd_peak

        rs = pos_size * pos_rem
        if pos_dir == 1:
            rpnl = rs * (exit_price - pos_entry) / pos_entry
        else:
            rpnl = rs * (pos_entry - exit_price) / pos_entry
        fee_cost = rs * FEE_RATE
        total_pnl = rpnl - fee_cost
        # CROSS margin: loss capped at balance
        if total_pnl < -bal:
            total_pnl = -bal

        bal += total_pnl
        peak_bal = max(peak_bal, bal)
        consec_loss = consec_loss + 1 if total_pnl < 0 else 0

        margin = pos_margin * pos_rem
        roi = total_pnl / margin * 100 if margin > 0 else 0
        hold_bars = bar_idx - pos_time_idx

        _n_trades += 1
        if total_pnl > 0:
            _n_wins += 1
            _gross_profit += total_pnl
            _win_roi_sum += roi
            _cur_consec_loss = 0
        else:
            _gross_loss += abs(total_pnl)
            _loss_roi_sum += roi
            _cur_consec_loss += 1
            _max_consec_loss = max(_max_consec_loss, _cur_consec_loss)

        if exit_reason == 'SL': _sl_count += 1
        elif exit_reason == 'TRAIL': _trail_count += 1
        elif exit_reason == 'REV': _rev_count += 1
        elif exit_reason == 'END': _end_count += 1

        _mdd_peak = max(_mdd_peak, bal)
        dd_now = (bal - _mdd_peak) / _mdd_peak
        _mdd = min(_mdd, dd_now)

        if return_trades:
            trades.append({
                'direction': 'LONG' if pos_dir == 1 else 'SHORT',
                'entry_time': str(timestamps[pos_time_idx]),
                'exit_time': str(timestamps[bar_idx]),
                'entry_price': round(pos_entry, 2),
                'exit_price': round(exit_price, 2),
                'exit_reason': exit_reason,
                'margin': round(margin, 2),
                'pnl': round(total_pnl, 2),
                'roi_pct': round(roi, 2),
                'peak_roi_pct': round(
                    ((pos_highest - pos_entry) / pos_entry * cfg['lev'] * 100) if pos_dir == 1 else
                    ((pos_entry - pos_lowest) / pos_entry * cfg['lev'] * 100), 2),
                'hold_bars': hold_bars,
                'hold_hours': round(hold_bars * bar_h, 2),
                'balance': round(bal, 2),
            })
        pos_dir = 0
        return total_pnl

    for i in range(warmup, n):
        h = highs[i]; l = lows[i]; c = closes[i]; o = opens[i]

        # DD protection check
        if peak_bal > 0:
            cur_dd = (peak_bal - bal) / peak_bal
            if cur_dd > 0.30:
                dd_halved = True
            elif cur_dd < 0.15:
                dd_halved = False

        # ---- Position management ----
        if pos_dir != 0:
            exited = False

            if pos_dir == 1:
                pos_highest = max(pos_highest, h)

                # SL check on intrabar LOW
                if l <= pos_sl:
                    close_pos(pos_sl, 'SL', i)
                    exited = True
                else:
                    # Trail check on CLOSE
                    cur_roi = (c - pos_entry) / pos_entry
                    if cur_roi >= cfg['trail_act']:
                        pos_trail_active = True
                    if pos_trail_active:
                        new_tsl = pos_highest * (1 - cfg['trail_width'])
                        if new_tsl > pos_trail_sl:
                            pos_trail_sl = max(new_tsl, pos_sl)
                        if c <= pos_trail_sl:
                            close_pos(c, 'TRAIL', i)
                            exited = True

            elif pos_dir == -1:
                pos_lowest = min(pos_lowest, l)

                # SL check on intrabar HIGH
                if h >= pos_sl:
                    close_pos(pos_sl, 'SL', i)
                    exited = True
                else:
                    # Trail check on CLOSE
                    cur_roi = (pos_entry - c) / pos_entry
                    if cur_roi >= cfg['trail_act']:
                        pos_trail_active = True
                    if pos_trail_active:
                        new_tsl = pos_lowest * (1 + cfg['trail_width'])
                        if new_tsl < pos_trail_sl:
                            pos_trail_sl = min(new_tsl, pos_sl)
                        if c >= pos_trail_sl:
                            close_pos(c, 'TRAIL', i)
                            exited = True

        # ---- Signal detection ----
        if i < warmup + 1: continue
        if np.isnan(fast_ma[i]) or np.isnan(fast_ma[i-1]): continue
        if np.isnan(slow_ma[i]) or np.isnan(slow_ma[i-1]): continue

        cross_up = (fast_ma[i-1] <= slow_ma[i-1]) and (fast_ma[i] > slow_ma[i])
        cross_down = (fast_ma[i-1] >= slow_ma[i-1]) and (fast_ma[i] < slow_ma[i])

        if cross_up:
            pending_signal = 1
            pending_bar = i
        elif cross_down:
            pending_signal = -1
            pending_bar = i

        # ---- Delayed entry check ----
        if pending_signal != 0 and (i - pending_bar) == cfg['delay']:
            sig = pending_signal
            pending_signal = 0

            # Verify MA still aligned after delay
            if sig == 1 and fast_ma[i] <= slow_ma[i]: continue
            if sig == -1 and fast_ma[i] >= slow_ma[i]: continue

            # ADX filter
            if np.isnan(adx[i]) or adx[i] < cfg['adx_min']: continue
            # RSI filter
            if np.isnan(rsi_arr[i]) or not (cfg['rsi_lo'] <= rsi_arr[i] <= cfg['rsi_hi']): continue

            direction = sig

            # Same-direction skip
            if pos_dir == direction:
                continue

            # REVERSE: close opposite position
            if pos_dir != 0 and pos_dir != direction:
                close_pos(c, 'REV', i)

            # Monthly loss limit: -15%
            mk = f"{timestamps[i].year}-{timestamps[i].month:02d}"
            if mk not in monthly_start: monthly_start[mk] = bal
            if monthly_start[mk] > 0 and (bal - monthly_start[mk]) / monthly_start[mk] < -0.15:
                continue

            if bal < 50: continue

            # Position sizing - CROSS margin
            margin_mult = 0.5 if dd_halved else 1.0
            sz = bal * cfg['margin'] * margin_mult
            if sz < 5: continue
            notional = sz * cfg['lev']

            # SL price
            if direction == 1:
                pos_sl = c * (1 - cfg['sl_pct'])
            else:
                pos_sl = c * (1 + cfg['sl_pct'])

            # Entry fee
            bal -= notional * FEE_RATE

            # Open position
            pos_dir = direction
            pos_entry = c
            pos_size = notional
            pos_margin = sz
            pos_time_idx = i
            pos_highest = c
            pos_lowest = c
            pos_trail_active = False
            pos_trail_sl = pos_sl
            pos_rem = 1.0

    # Close remaining position
    if pos_dir != 0:
        close_pos(closes[-1], 'END', n-1)

    total_ret = (bal - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    n_losses = _n_trades - _n_wins
    avg_win_roi = _win_roi_sum / _n_wins if _n_wins > 0 else 0
    avg_loss_roi = _loss_roi_sum / n_losses if n_losses > 0 else 0

    metrics = {
        'initial': INITIAL_CAPITAL,
        'final': round(bal, 2),
        'return_pct': round(total_ret, 2),
        'trades': _n_trades,
        'pf': round(_gross_profit / _gross_loss, 4) if _gross_loss > 0 else 999,
        'mdd': round(_mdd * 100, 2),
        'win_rate': round(_n_wins / _n_trades * 100, 2) if _n_trades > 0 else 0,
        'avg_win': round(avg_win_roi, 2),
        'avg_loss': round(avg_loss_roi, 2),
        'rr': round(abs(avg_win_roi / avg_loss_roi), 2) if avg_loss_roi != 0 else 999,
        'max_consec_loss': _max_consec_loss,
        'gross_profit': round(_gross_profit, 2),
        'gross_loss': round(_gross_loss, 2),
        'sl_count': _sl_count,
        'trail_count': _trail_count,
        'rev_count': _rev_count,
        'end_count': _end_count,
    }

    return metrics, trades


# ============================================================
# REPORT FUNCTIONS
# ============================================================
def generate_report_text(label, cfg, metrics, trades, timestamps_index):
    """Generate full report text and monthly rows."""
    lines = []
    T = metrics['trades']

    lines.append(f"\n{'='*100}")
    lines.append(f"  PORTFOLIO SUMMARY  [{label}]")
    lines.append(f"{'='*100}")
    lines.append(f"  Strategy:      {cfg['fast_name']}/{cfg['slow_name']} | {cfg['tf']}")
    lines.append(f"  ADX({cfg['adx_period']})>={cfg['adx_min']} | RSI {cfg['rsi_lo']}-{cfg['rsi_hi']}")
    lines.append(f"  Delay: {cfg['delay']} | SL: -{cfg['sl_pct']*100:.0f}% | Trail: +{cfg['trail_act']*100:.0f}%/-{cfg['trail_width']*100:.0f}%")
    lines.append(f"  Margin: {cfg['margin']*100:.0f}% | Leverage: {cfg['lev']}x | CROSS margin")
    lines.append(f"  Fee: {FEE_RATE*100:.2f}% | REVERSE + Same-dir skip")
    lines.append(f"  DD Protection: halve size if DD>30% | Monthly limit: -15%")
    lines.append(f"{'='*100}")
    lines.append(f"  Initial:       ${INITIAL_CAPITAL:,.0f}")
    lines.append(f"  Final:         ${metrics['final']:,.2f}")
    lines.append(f"  Return:        {metrics['return_pct']:+.1f}%")
    lines.append(f"  PF:            {metrics.get('pf', 0):.2f}")
    lines.append(f"  MDD:           {metrics.get('mdd', 0):.1f}%")
    lines.append(f"  Trades:        {T}")
    lines.append(f"  Win Rate:      {metrics.get('win_rate', 0):.1f}%")
    lines.append(f"  Avg Win:       {metrics.get('avg_win', 0):+.2f}%")
    lines.append(f"  Avg Loss:      {metrics.get('avg_loss', 0):+.2f}%")
    lines.append(f"  R:R:           {metrics.get('rr', 0):.2f}")
    lines.append(f"  Max Consec L:  {metrics.get('max_consec_loss', 0)}")
    lines.append(f"  Gross Profit:  ${metrics.get('gross_profit', 0):,.2f}")
    lines.append(f"  Gross Loss:    ${metrics.get('gross_loss', 0):,.2f}")
    lines.append(f"  SL:            {metrics.get('sl_count', 0)}")
    lines.append(f"  TRAIL:         {metrics.get('trail_count', 0)}")
    lines.append(f"  REV:           {metrics.get('rev_count', 0)}")
    lines.append(f"  END (open):    {metrics.get('end_count', 0)}")
    lines.append("")

    if not trades:
        lines.append("  NO TRADES")
        return "\n".join(lines), []

    df = pd.DataFrame(trades)

    # Direction analysis
    lines.append("  DIRECTION ANALYSIS")
    lines.append("  " + "-" * 70)
    for d in ['LONG', 'SHORT']:
        sub = df[df['direction'] == d]
        if len(sub) == 0: continue
        sw = sub[sub['pnl'] > 0]
        lines.append(f"  {d:>5}: {len(sub):>4} ({len(sub)/T*100:.0f}%) WR:{len(sw)/len(sub)*100:.0f}% AvgROI:{sub['roi_pct'].mean():+.2f}% PnL:${sub['pnl'].sum():+,.0f}")

    # Exit reason analysis
    lines.append(f"\n  EXIT REASON ANALYSIS")
    lines.append("  " + "-" * 70)
    for r in ['TRAIL', 'REV', 'SL', 'END']:
        rt = df[df['exit_reason'] == r]
        if len(rt) == 0: continue
        rw = rt[rt['pnl'] > 0]
        wr_val = len(rw)/len(rt)*100 if len(rt) else 0
        lines.append(f"  {r:>5}: {len(rt):>4} ({len(rt)/T*100:.0f}%) WR:{wr_val:.0f}% AvgROI:{rt['roi_pct'].mean():+.2f}% PnL:${rt['pnl'].sum():+,.0f}")

    # Hold time analysis
    lines.append(f"\n  HOLD TIME ANALYSIS")
    lines.append("  " + "-" * 70)
    for a, b, lb in [(0,2,'<2h'),(2,8,'2-8h'),(8,24,'8-24h'),(24,72,'1-3d'),(72,168,'3-7d'),(168,9999,'7d+')]:
        ht = df[(df['hold_hours'] >= a) & (df['hold_hours'] < b)]
        if len(ht):
            hw = ht[ht['pnl'] > 0]
            lines.append(f"  {lb:>6}: {len(ht):>4} WR:{len(hw)/len(ht)*100:.0f}% AvgROI:{ht['roi_pct'].mean():+.2f}% PnL:${ht['pnl'].sum():+,.0f}")

    # Monthly table
    df['exit_dt'] = pd.to_datetime(df['exit_time'])
    df['month'] = df['exit_dt'].dt.to_period('M')
    all_months = pd.period_range('2020-01', '2026-03', freq='M')
    mg = df.groupby('month')

    lines.append(f"\n{'='*100}")
    lines.append(f"  MONTHLY PERFORMANCE (2020-01 to 2026-03)")
    lines.append(f"{'='*100}")
    lines.append(f"  {'Month':>8} {'Trd':>4} {'W':>3} {'L':>3} {'WR%':>5} {'GrossP':>9} {'GrossL':>9} {'NetPnL':>9} {'PF':>6} {'Bal':>10} {'Ret%':>7}")
    lines.append("  " + "-" * 95)

    rb = INITIAL_CAPITAL; yearly = {}; lm = 0; tm = 0; pm = 0
    monthly_rows = []

    for mo in all_months:
        if mo in mg.groups:
            g = mg.get_group(mo)
            nt = len(g); nw = len(g[g['pnl'] > 0]); nl = nt - nw
            wr = nw/nt*100 if nt else 0
            gp2 = g[g['pnl'] > 0]['pnl'].sum() if nw else 0
            gl2 = abs(g[g['pnl'] <= 0]['pnl'].sum()) if nl else 0
            net = g['pnl'].sum()
            mpf = gp2/gl2 if gl2 > 0 else (999 if gp2 > 0 else 0)
        else:
            nt = nw = nl = 0; wr = 0; gp2 = gl2 = net = 0; mpf = 0

        sbr = rb; rb += net
        mr = net/sbr*100 if sbr > 0 else 0
        tm += 1
        if net < 0: lm += 1
        if net > 0: pm += 1

        y = str(mo)[:4]
        if y not in yearly: yearly[y] = {'p': 0, 't': 0, 'w': 0, 'l': 0, 'gp': 0, 'gl': 0, 'sb': sbr}
        yearly[y]['p'] += net; yearly[y]['t'] += nt; yearly[y]['w'] += nw; yearly[y]['l'] += nl
        yearly[y]['gp'] += gp2; yearly[y]['gl'] += gl2; yearly[y]['eb'] = rb

        pfs = f"{mpf:.1f}" if mpf < 999 else "INF"
        if mpf == 0 and net == 0: pfs = "-"
        mk = " <<" if net < 0 else ""
        lines.append(f"  {str(mo):>8} {nt:>4} {nw:>3} {nl:>3} {wr:>4.0f}% ${gp2:>7,.0f} ${gl2:>7,.0f} ${net:>+7,.0f} {pfs:>5} ${rb:>8,.0f} {mr:>+6.1f}%{mk}")

        monthly_rows.append({
            'month': str(mo), 'trades': nt, 'wins': nw, 'losses': nl,
            'win_rate': round(wr, 1), 'gross_profit': round(gp2, 2),
            'gross_loss': round(gl2, 2), 'pnl': round(net, 2),
            'pf': round(mpf, 2) if mpf < 999 else 999,
            'balance': round(rb, 2), 'return_pct': round(mr, 2)
        })

    # Yearly summary
    lines.append(f"\n{'='*100}")
    lines.append(f"  YEARLY PERFORMANCE")
    lines.append(f"{'='*100}")
    lines.append(f"  {'Year':>6} {'Trd':>4} {'W':>4} {'L':>4} {'WR%':>5} {'GrossP':>10} {'GrossL':>10} {'NetPnL':>10} {'PF':>6} {'YrRet%':>8}")
    lines.append("  " + "-" * 80)
    for y2 in sorted(yearly):
        yd = yearly[y2]; ywr = yd['w']/yd['t']*100 if yd['t'] else 0
        ypf = yd['gp']/yd['gl'] if yd['gl'] > 0 else (999 if yd['gp'] > 0 else 0)
        yret = yd['p']/yd['sb']*100 if yd['sb'] > 0 else 0
        pfs = f"{ypf:.2f}" if ypf < 999 else "INF"
        lines.append(f"  {y2:>6} {yd['t']:>3} {yd['w']:>4} {yd['l']:>4} {ywr:>4.0f}% ${yd['gp']:>8,.0f} ${yd['gl']:>8,.0f} ${yd['p']:>+8,.0f} {pfs:>5} {yret:>+7.1f}%")
    pyrs = sum(1 for v in yearly.values() if v['p'] > 0)
    lines.append(f"\n  Profit Months: {pm}/{tm} ({pm/max(1,tm)*100:.0f}%)")
    lines.append(f"  Loss Months:   {lm}/{tm} ({lm/max(1,tm)*100:.0f}%)")
    lines.append(f"  Profit Years:  {pyrs}/{len(yearly)}")

    # Top/Bottom trades
    ds = df.sort_values('pnl', ascending=False)
    lines.append(f"\n  TOP 10 TRADES")
    lines.append("  " + "-" * 100)
    for idx, (_, r) in enumerate(ds.head(10).iterrows()):
        lines.append(f"  {idx+1:>2} {r['direction']:>5} {r['entry_time'][:16]} -> {r['exit_time'][:16]} {r['exit_reason']:>5} ROI:{r['roi_pct']:>+7.1f}% PeakROI:{r['peak_roi_pct']:>+7.1f}% PnL:${r['pnl']:>+8,.0f}")
    lines.append(f"\n  BOTTOM 10 TRADES")
    lines.append("  " + "-" * 100)
    for idx, (_, r) in enumerate(ds.tail(10).iterrows()):
        lines.append(f"  {idx+1:>2} {r['direction']:>5} {r['entry_time'][:16]} -> {r['exit_time'][:16]} {r['exit_reason']:>5} ROI:{r['roi_pct']:>+7.1f}% PeakROI:{r['peak_roi_pct']:>+7.1f}% PnL:${r['pnl']:>+8,.0f}")

    return "\n".join(lines), monthly_rows


# ============================================================
# MAIN
# ============================================================
def main():
    T0 = time.time()
    ts = total_space()
    print(f"\n{'='*70}", flush=True)
    print(f"  v23.3 BTC/USDT FUTURES - OPTIMIZATION + BACKTEST + VERIFICATION", flush=True)
    print(f"  Initial: ${INITIAL_CAPITAL:,.0f} | CROSS margin | PF target >= 5", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\nSpace: {ts:,} | Sampling: {NUM_SAMPLES:,} | Seed: {SEED}", flush=True)
    print(f"Workers: {NWORKERS}\n", flush=True)

    # ================================================================
    # PHASE 1: OPTIMIZATION
    # ================================================================
    print(f"{'#'*70}", flush=True)
    print(f"  PHASE 1: OPTIMIZATION ({NUM_SAMPLES:,} combos)", flush=True)
    print(f"{'#'*70}\n", flush=True)

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

    del df5m, caches

    # Generate combos
    print(f"Generating {NUM_SAMPLES:,} combos...", flush=True)
    rng = np.random.RandomState(SEED)
    lens = [len(l) for l in ALL_LISTS]
    combos = np.column_stack([rng.randint(0, l, size=NUM_SAMPLES) for l in lens]).astype(np.int32)
    print(f"Done: {combos.shape}\n", flush=True)

    # Split into chunks
    chunks = []
    for i in range(0, NUM_SAMPLES, CHUNK_PER_WORKER):
        chunks.append(combos[i:i+CHUNK_PER_WORKER])
    print(f"Chunks: {len(chunks)} x ~{CHUNK_PER_WORKER}\n", flush=True)

    # Run optimization
    print("="*70, flush=True)
    print("RUNNING OPTIMIZATION", flush=True)
    print("="*70, flush=True)

    all_results = []
    bt_start = time.time()
    processed = 0

    with Pool(processes=NWORKERS, initializer=_init_worker, initargs=(CACHE_FILE,)) as pool:
        for chunk_res in pool.imap_unordered(_run_chunk, chunks, chunksize=1):
            all_results.extend(chunk_res)
            processed += CHUNK_PER_WORKER

            if processed % PROGRESS_INTERVAL < CHUNK_PER_WORKER or processed >= NUM_SAMPLES:
                elapsed = time.time() - bt_start
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (NUM_SAMPLES - processed) / rate if rate > 0 else 0
                valid = len(all_results)
                if all_results:
                    best = max(all_results, key=lambda x: x[23])
                    bi = best
                    bret = bi[17]
                    bpf = bi[16]
                    bmdd = bi[19]
                    bsc = bi[23]
                    btr = bi[12]
                    print(f"  [{min(processed,NUM_SAMPLES):>10,}/{NUM_SAMPLES:,}] "
                          f"{elapsed:7.1f}s | {rate:,.0f}/s | ETA {eta:5.0f}s | "
                          f"Valid: {valid:,} | "
                          f"Best: {bret:+,.1f}% PF={bpf:.1f} MDD={bmdd:.1f}% T={btr} Sc={bsc:.1f}",
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
    def decode(r):
        rsi_lo, rsi_hi = RSI_RS[r[5]]
        return {
            'fast_ma': FAST_KEYS[r[0]], 'slow_ma': SLOW_KEYS[r[1]],
            'tf': TIMEFRAMES[r[2]], 'adx_period': ADX_PERIODS[r[3]],
            'adx_thresh': ADX_THS[r[4]],
            'rsi_lo': rsi_lo, 'rsi_hi': rsi_hi,
            'delay': DELAYS[r[6]],
            'sl_pct': SLS[r[7]], 'trail_act': TACTS[r[8]], 'trail_width': TWS[r[9]],
            'margin_pct': MGNS[r[10]], 'leverage': LEVS[r[11]],
            'trades': r[12], 'wins': r[13], 'losses': r[14], 'win_rate': r[15],
            'pf': r[16], 'return_pct': r[17], 'final_bal': r[18],
            'max_dd': r[19], 'sl_hits': r[20], 'trail_hits': r[21],
            'reverse_hits': r[22], 'score': r[23],
        }

    all_results.sort(key=lambda x: x[23], reverse=True)
    result_dicts = [decode(r) for r in all_results[:500]]

    # Save top 100 CSV
    top100 = result_dicts[:100]
    df_top = pd.DataFrame(top100)
    csv_p = os.path.join(DATA_DIR, "v23_3_opt_top100.csv")
    df_top.to_csv(csv_p, index=False)
    print(f"Saved: {csv_p}", flush=True)

    # Summary file
    sum_p = os.path.join(DATA_DIR, "v23_3_opt_summary.txt")
    with open(sum_p, 'w', encoding='utf-8') as f:
        f.write("="*80+"\n")
        f.write("v23.3 PARAMETER OPTIMIZATION RESULTS\n")
        f.write("="*80+"\n\n")
        f.write(f"Initial capital:  ${INITIAL_CAPITAL:,.0f}\n")
        f.write(f"Margin mode:      CROSS\n")
        f.write(f"Total space:      {ts:,}\n")
        f.write(f"Sampled:          {NUM_SAMPLES:,}\n")
        f.write(f"Valid:            {len(all_results):,}\n")
        f.write(f"Fee:              {FEE_RATE*100:.2f}%\n")
        f.write(f"Indicator time:   {it:.1f}s\n")
        f.write(f"Backtest time:    {bt_elapsed:.1f}s\n")
        f.write(f"Total time:       {time.time()-T0:.1f}s\n")
        f.write(f"Speed:            {NUM_SAMPLES/bt_elapsed:,.0f}/s\n")
        f.write(f"Workers:          {NWORKERS}\n")
        f.write(f"Score formula:    PF*5 + log10(max(Ret%,1))*10 + min(Trades,300)/30 - MDD%/3\n\n")

        f.write("="*80+"\n")
        f.write("TOP 30\n")
        f.write("="*80+"\n\n")
        for i,r in enumerate(result_dicts[:30], 1):
            f.write(f"--- #{i} ---\n")
            f.write(f"  Score:    {r['score']:.2f}\n")
            f.write(f"  Return:   {r['return_pct']:+,.2f}% (${r['final_bal']:,.2f})\n")
            f.write(f"  PF:       {r['pf']:.2f} | WR: {r['win_rate']:.1f}%\n")
            f.write(f"  MDD:      {r['max_dd']:.2f}%\n")
            f.write(f"  Trades:   {r['trades']} (W:{r['wins']} L:{r['losses']})\n")
            f.write(f"  Exits:    SL={r['sl_hits']} Trail={r['trail_hits']} Rev={r['reverse_hits']}\n")
            f.write(f"  Params:   {r['fast_ma']}/{r['slow_ma']} {r['tf']}\n")
            f.write(f"            ADX({r['adx_period']})>={r['adx_thresh']} RSI {r['rsi_lo']}-{r['rsi_hi']} Delay={r['delay']}\n")
            f.write(f"            SL={r['sl_pct']}% Trail={r['trail_act']}%/w{r['trail_width']}%\n")
            f.write(f"            Margin={r['margin_pct']}% Lev={r['leverage']}x | CROSS\n\n")

        # Parameter frequency analysis
        f.write("="*80+"\n")
        f.write("PARAMETER FREQUENCY ANALYSIS (Top 100)\n")
        f.write("="*80+"\n\n")
        for pk in ['fast_ma','slow_ma','tf','adx_period','adx_thresh',
                    'sl_pct','trail_act','trail_width',
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
        ttr=sum(r['trail_hits'] for r in top100)
        trv=sum(r['reverse_hits'] for r in top100)
        ta_=max(tsl+ttr+trv,1)
        f.write(f"  SL:    {tsl:,} ({tsl/ta_*100:.1f}%)\n")
        f.write(f"  Trail: {ttr:,} ({ttr/ta_*100:.1f}%)\n")
        f.write(f"  Rev:   {trv:,} ({trv/ta_*100:.1f}%)\n\n")

        # Stats
        f.write("="*80+"\n")
        f.write("STATS (all valid)\n")
        f.write("="*80+"\n\n")
        rets = [r[17] for r in all_results]
        pfs = [r[16] for r in all_results]
        mdds = [r[19] for r in all_results]
        trs = [r[12] for r in all_results]
        f.write(f"  Return: mean={np.mean(rets):+.1f}% med={np.median(rets):+.1f}% "
                f"max={np.max(rets):+,.1f}% min={np.min(rets):+,.1f}%\n")
        f.write(f"  PF:     mean={np.mean(pfs):.2f} med={np.median(pfs):.2f} max={min(np.max(pfs),999):.2f}\n")
        f.write(f"  MDD:    mean={np.mean(mdds):.1f}% med={np.median(mdds):.1f}% max={np.max(mdds):.1f}%\n")
        f.write(f"  Trades: mean={np.mean(trs):.0f} med={np.median(trs):.0f} max={np.max(trs)}\n\n")

        bkts = [(-100,0),(0,100),(100,500),(500,1000),(1000,5000),(5000,10000),(10000,50000),(50000,1e9)]
        f.write("  Return distribution:\n")
        for lo,hi in bkts:
            cnt=sum(1 for r in rets if lo<=r<hi)
            pct=cnt/max(len(rets),1)*100
            lb=f"{int(lo):>8,}%+" if hi>=1e9 else f"{int(lo):>8,}%-{int(hi):>8,}%"
            f.write(f"    {lb}: {cnt:>6,} ({pct:5.1f}%) {'#'*int(pct)}\n")

        # PF distribution
        f.write("\n  PF distribution:\n")
        pf_bkts = [(1,2),(2,3),(3,5),(5,8),(8,10),(10,50),(50,1000)]
        for lo,hi in pf_bkts:
            cnt=sum(1 for r in pfs if lo<=r<hi)
            pct=cnt/max(len(pfs),1)*100
            f.write(f"    PF {lo:>3}-{hi:>4}: {cnt:>6,} ({pct:5.1f}%) {'#'*int(pct)}\n")

    print(f"Saved: {sum_p}\n", flush=True)

    # Print top 10
    print("="*70, flush=True)
    print("OPTIMIZATION TOP 10", flush=True)
    print("="*70, flush=True)
    for i,r in enumerate(result_dicts[:10],1):
        print(f"\n  #{i}: Sc={r['score']:.1f} | "
              f"Ret={r['return_pct']:+,.1f}% (${r['final_bal']:,.0f}) | "
              f"PF={r['pf']:.1f} WR={r['win_rate']:.0f}% MDD={r['max_dd']:.1f}% "
              f"Tr={r['trades']}", flush=True)
        print(f"      {r['fast_ma']}/{r['slow_ma']} {r['tf']} "
              f"ADX({r['adx_period']})>={r['adx_thresh']} "
              f"RSI {r['rsi_lo']}-{r['rsi_hi']} D={r['delay']}", flush=True)
        print(f"      SL={r['sl_pct']}% Trail={r['trail_act']}%/w{r['trail_width']}% "
              f"Mg={r['margin_pct']}% Lv={r['leverage']}x", flush=True)
        print(f"      Exits: SL={r['sl_hits']} Trail={r['trail_hits']} Rev={r['reverse_hits']}", flush=True)

    # ================================================================
    # PHASE 2: TOP 3 DETAILED BACKTEST
    # ================================================================
    print(f"\n\n{'#'*70}", flush=True)
    print(f"  PHASE 2: TOP 3 DETAILED BACKTEST", flush=True)
    print(f"{'#'*70}\n", flush=True)

    top3 = result_dicts[:3]
    full_report_text = []
    best_trades = None
    best_monthly = None
    best_cfg = None
    best_metrics = None

    # Data cache for phase 2
    data_cache_p2 = {}

    for rank, top_r in enumerate(top3, 1):
        # Parse config from optimization result
        fm_parts = top_r['fast_ma'].split('_')
        sm_parts = top_r['slow_ma'].split('_')

        cfg = {
            'name': f"TOP{rank}_{top_r['fast_ma']}_{top_r['slow_ma']}_{top_r['tf']}",
            'fast_name': f"{fm_parts[0]}({fm_parts[1]})",
            'slow_name': f"{sm_parts[0]}({sm_parts[1]})",
            'fast_type': fm_parts[0].lower() if fm_parts[0] != 'HMA' else 'hma',
            'fast_period': int(fm_parts[1]),
            'slow_type': sm_parts[0].lower() if sm_parts[0] != 'SMA' else 'sma',
            'slow_period': int(sm_parts[1]),
            'tf': top_r['tf'],
            'adx_period': top_r['adx_period'],
            'adx_min': top_r['adx_thresh'],
            'rsi_lo': top_r['rsi_lo'],
            'rsi_hi': top_r['rsi_hi'],
            'delay': top_r['delay'],
            'sl_pct': abs(top_r['sl_pct']) / 100.0,
            'trail_act': top_r['trail_act'] / 100.0,
            'trail_width': abs(top_r['trail_width']) / 100.0,
            'margin': top_r['margin_pct'] / 100.0,
            'lev': top_r['leverage'],
        }

        print(f"\n{'='*70}", flush=True)
        print(f"  TOP {rank}: {cfg['fast_name']}/{cfg['slow_name']} | {cfg['tf']}", flush=True)
        print(f"  ADX({cfg['adx_period']})>={cfg['adx_min']} RSI {cfg['rsi_lo']}-{cfg['rsi_hi']} D={cfg['delay']}", flush=True)
        print(f"  SL={top_r['sl_pct']}% Trail={top_r['trail_act']}%/w{top_r['trail_width']}%", flush=True)
        print(f"  Margin={top_r['margin_pct']}% Lev={cfg['lev']}x CROSS", flush=True)
        print(f"  Opt Score={top_r['score']:.1f} Ret={top_r['return_pct']:+,.1f}% PF={top_r['pf']:.1f}", flush=True)
        print(f"{'='*70}", flush=True)

        # Load data
        tf = cfg['tf']
        if tf not in data_cache_p2:
            print(f"\n  Loading data for {tf}...", flush=True)
            data_cache_p2[tf] = load_data_phase2(tf)
        dfr = data_cache_p2[tf]

        o = dfr['open'].values; h = dfr['high'].values; l = dfr['low'].values
        c = dfr['close'].values; tsi = dfr.index

        # Compute indicators
        print(f"  Computing indicators...", flush=True)
        t0 = time.time()
        fast_ma = calc_fast_ma_py(c, fm_parts[0], int(fm_parts[1]))
        slow_ma = calc_slow_ma_py(c, sm_parts[0], int(sm_parts[1]))
        adx_arr, _ = calc_adx_wilder(h, l, c, cfg['adx_period'])
        rsi_arr = calc_rsi_py(c, 14)
        print(f"  Indicators done | {time.time()-t0:.1f}s", flush=True)

        # Run detailed backtest
        print(f"  Running detailed backtest...", flush=True)
        t0 = time.time()
        metrics, trades = run_detailed_backtest(
            cfg, o, h, l, c, adx_arr, rsi_arr, fast_ma, slow_ma, tsi, return_trades=True
        )
        elapsed = time.time() - t0
        print(f"  Backtest done | {elapsed:.1f}s", flush=True)
        print(f"  Result: ${metrics['final']:,.2f} ({metrics['return_pct']:+.1f}%) "
              f"PF={metrics['pf']:.2f} MDD={metrics['mdd']:.1f}% "
              f"Trades={metrics['trades']} WR={metrics['win_rate']:.1f}%", flush=True)

        # Generate report
        report_text, monthly_rows = generate_report_text(
            f"TOP {rank}", cfg, metrics, trades, tsi
        )
        print(report_text, flush=True)
        full_report_text.append(report_text)

        if rank == 1:
            best_trades = trades
            best_monthly = monthly_rows
            best_cfg = cfg
            best_metrics = metrics

    # ================================================================
    # PHASE 3: 30-RUN VERIFICATION
    # ================================================================
    print(f"\n\n{'#'*70}", flush=True)
    print(f"  PHASE 3: 30-RUN VERIFICATION", flush=True)
    print(f"{'#'*70}\n", flush=True)

    all_verify_rows = []

    for rank, top_r in enumerate(top3, 1):
        fm_parts = top_r['fast_ma'].split('_')
        sm_parts = top_r['slow_ma'].split('_')

        cfg = {
            'name': f"TOP{rank}_{top_r['fast_ma']}_{top_r['slow_ma']}_{top_r['tf']}",
            'fast_name': f"{fm_parts[0]}({fm_parts[1]})",
            'slow_name': f"{sm_parts[0]}({sm_parts[1]})",
            'fast_type': fm_parts[0].lower(),
            'fast_period': int(fm_parts[1]),
            'slow_type': sm_parts[0].lower(),
            'slow_period': int(sm_parts[1]),
            'tf': top_r['tf'],
            'adx_period': top_r['adx_period'],
            'adx_min': top_r['adx_thresh'],
            'rsi_lo': top_r['rsi_lo'],
            'rsi_hi': top_r['rsi_hi'],
            'delay': top_r['delay'],
            'sl_pct': abs(top_r['sl_pct']) / 100.0,
            'trail_act': top_r['trail_act'] / 100.0,
            'trail_width': abs(top_r['trail_width']) / 100.0,
            'margin': top_r['margin_pct'] / 100.0,
            'lev': top_r['leverage'],
        }

        tf = cfg['tf']
        if tf not in data_cache_p2:
            data_cache_p2[tf] = load_data_phase2(tf)
        dfr = data_cache_p2[tf]

        o = dfr['open'].values; h = dfr['high'].values; l = dfr['low'].values
        c = dfr['close'].values; tsi = dfr.index

        fast_ma = calc_fast_ma_py(c, fm_parts[0], int(fm_parts[1]))
        slow_ma = calc_slow_ma_py(c, sm_parts[0], int(sm_parts[1]))
        adx_arr, _ = calc_adx_wilder(h, l, c, cfg['adx_period'])
        rsi_arr = calc_rsi_py(c, 14)

        print(f"\n  TOP {rank}: {cfg['fast_name']}/{cfg['slow_name']} | {cfg['tf']} - 30 runs", flush=True)
        finals = []
        for run in range(1, 31):
            t0 = time.time()
            m, _ = run_detailed_backtest(
                cfg, o, h, l, c, adx_arr, rsi_arr, fast_ma, slow_ma, tsi, return_trades=False
            )
            elapsed = time.time() - t0
            finals.append(m['final'])
            all_verify_rows.append({
                'strategy': f"TOP{rank}",
                'run': run,
                'final': m['final'],
                'return_pct': m['return_pct'],
                'trades': m['trades'],
                'pf': m['pf'],
                'mdd': m['mdd'],
                'win_rate': m['win_rate'],
                'time_s': round(elapsed, 2),
            })
            if run <= 5 or run % 10 == 0:
                print(f"    Run {run:>2} | ${m['final']:>10,.2f} | {m['return_pct']:>+8.1f}% | "
                      f"PF={m['pf']:.2f} | T={m['trades']} | {elapsed:.1f}s", flush=True)

        std_f = np.std(finals)
        mean_f = np.mean(finals)
        print(f"  RESULT: mean=${mean_f:,.2f} std=${std_f:.4f} "
              f"{'PASS (std=0)' if std_f < 0.01 else 'FAIL (std!=0)'}", flush=True)

    # ================================================================
    # SAVE ALL OUTPUT FILES
    # ================================================================
    print(f"\n\n{'#'*70}", flush=True)
    print(f"  SAVING OUTPUT FILES", flush=True)
    print(f"{'#'*70}\n", flush=True)

    # v23_3_FINAL_report.txt
    report_path = os.path.join(DATA_DIR, "v23_3_FINAL_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("v23.3 BTC/USDT FUTURES - FINAL BACKTEST REPORT\n")
        f.write(f"Initial: ${INITIAL_CAPITAL:,.0f} | CROSS margin | PF target >= 5\n")
        f.write(f"Score = PF*5 + log10(max(Ret%,1))*10 + min(Trades,300)/30 - MDD%/3\n")
        f.write("="*100 + "\n")
        for rt in full_report_text:
            f.write(rt + "\n")
    print(f"  Saved: {report_path}", flush=True)

    # v23_3_FINAL_trades.csv
    if best_trades:
        trades_path = os.path.join(DATA_DIR, "v23_3_FINAL_trades.csv")
        pd.DataFrame(best_trades).to_csv(trades_path, index=False)
        print(f"  Saved: {trades_path}", flush=True)

    # v23_3_FINAL_monthly.csv
    if best_monthly:
        monthly_path = os.path.join(DATA_DIR, "v23_3_FINAL_monthly.csv")
        pd.DataFrame(best_monthly).to_csv(monthly_path, index=False)
        print(f"  Saved: {monthly_path}", flush=True)

    # v23_3_FINAL_30run.csv
    run_path = os.path.join(DATA_DIR, "v23_3_FINAL_30run.csv")
    pd.DataFrame(all_verify_rows).to_csv(run_path, index=False)
    print(f"  Saved: {run_path}", flush=True)

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    te = time.time() - T0
    print(f"\n{'='*70}", flush=True)
    print(f"  v23.3 PIPELINE COMPLETE", flush=True)
    print(f"  Total time: {te:.1f}s ({te/60:.1f} min)", flush=True)
    print(f"  Optimization: {NUM_SAMPLES:,} combos | Valid: {len(all_results):,}", flush=True)
    if best_metrics:
        print(f"  Best (TOP 1): ${best_metrics['final']:,.2f} ({best_metrics['return_pct']:+.1f}%)", flush=True)
        print(f"                PF={best_metrics['pf']:.2f} MDD={best_metrics['mdd']:.1f}% Trades={best_metrics['trades']}", flush=True)
    print(f"  Output files:", flush=True)
    print(f"    - v23_3_opt_top100.csv", flush=True)
    print(f"    - v23_3_opt_summary.txt", flush=True)
    print(f"    - v23_3_FINAL_report.txt", flush=True)
    print(f"    - v23_3_FINAL_trades.csv", flush=True)
    print(f"    - v23_3_FINAL_monthly.csv", flush=True)
    print(f"    - v23_3_FINAL_30run.csv", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
