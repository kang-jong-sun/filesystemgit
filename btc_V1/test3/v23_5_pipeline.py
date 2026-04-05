"""
v23.5 BTC/USDT Futures - MDD-Focused Optimization Pipeline (1,500,000 combos)
===============================================================================
HARD CONSTRAINTS:
  - Initial: $5,000
  - Margin: <= 25%
  - Leverage: <= 15x
  - MDD TARGET: <= 20%
  - PF TARGET: >= 5
  - Return TARGET: >= 100,000% ($5K -> $5M+)
  - Fee: 0.04%
  - ISOLATED margin (no DD protection in optimizer)
  - Trail on CLOSE, SL on HIGH/LOW
  - REVERSE signal + same-dir skip
  - Wilder ADX (correct implementation)

SCORING (MDD-penalized):
  Score = PF*10 + log10(max(Return%,1))*15 - max(0, MDD%-20)*5 + min(Trades,200)/20

PHASES:
  1. Optimization (1,500,000 random combos via multiprocessing)
  2. Top-N analysis tables
  3. Winner detailed backtest + 30-run verification
  4. Parameter frequency analysis
  5. Save all output files
"""

import numpy as np
import pandas as pd
import time
import os
import sys
import io
import random
from math import log10
from collections import defaultdict, Counter
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
DATA_DIR = r"D:\filesystem\futures\btc_V1\test"
OUT_DIR = r"D:\filesystem\futures\btc_V1\test3"
CSV_FILES = [
    os.path.join(DATA_DIR, f"btc_usdt_5m_2020_to_now_part{i}.csv")
    for i in range(1, 4)
]
INITIAL_CAPITAL = 5000.0
FEE_RATE = 0.0004  # 0.04%
NUM_SAMPLES = 1_500_000
SEED = 42
PROGRESS_INTERVAL = 100_000
NWORKERS = max(1, NCPU - 2)
CHUNK_PER_WORKER = 2000

CACHE_FILE = os.path.join(OUT_DIR, "_v23_5_cache.pkl")

# ============================================================
# PARAMETER SPACE
# ============================================================
FAST_MA_CONFIGS = [('WMA',3),('HMA',3),('HMA',5),('EMA',3),('EMA',5),('EMA',7),('EMA',10)]
SLOW_MA_CONFIGS = [('EMA',100),('EMA',150),('EMA',200),('EMA',250),('SMA',200),('SMA',300)]
ADX_PERIODS = [14, 20]
TIMEFRAMES = ['10m', '15m', '30m', '1h']

FAST_KEYS = [f"{t}_{p}" for t,p in FAST_MA_CONFIGS]
SLOW_KEYS = [f"{t}_{p}" for t,p in SLOW_MA_CONFIGS]
ADX_THS = [25, 30, 35, 40, 45]
RSI_RS = [(25,65), (30,65), (30,70), (35,65), (35,70), (40,75)]
DELAYS = [0, 1, 2, 3, 5]
SLS = [-3, -4, -5, -6, -7, -8, -10]       # 7 values
TACTS = [2, 3, 4, 5, 6, 8, 10, 15]        # 8 values
TWS = [-1, -2, -3, -4, -5]                 # 5 values
MGNS = [5, 8, 10, 12, 15, 18, 20, 25]     # 8 values (small for low MDD)
LEVS = [3, 5, 7, 10, 12, 15]              # 6 values

ALL_LISTS = [FAST_KEYS, SLOW_KEYS, TIMEFRAMES, ADX_PERIODS, ADX_THS,
             RSI_RS, DELAYS, SLS, TACTS, TWS, MGNS, LEVS]

def total_space():
    t = 1
    for l in ALL_LISTS: t *= len(l)
    return t

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
    """Wilder ADX - correct implementation."""
    n=len(c)
    tr=np.empty(n); pdm=np.empty(n); mdm=np.empty(n)
    tr[0]=h[0]-l[0]; pdm[0]=0.0; mdm[0]=0.0
    for i in range(1,n):
        hl=h[i]-l[i]; hpc=abs(h[i]-c[i-1]); lpc=abs(l[i]-c[i-1])
        tr[i]=max(hl,max(hpc,lpc))
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        pdm[i]=up if up>dn and up>0 else 0.0
        mdm[i]=dn if dn>up and dn>0 else 0.0
    atr=wilder(tr,p); sp_=wilder(pdm,p); sm_=wilder(mdm,p)
    pdi=np.empty(n); mdi=np.empty(n); dx=np.empty(n)
    pdi[:]=np.nan; mdi[:]=np.nan; dx[:]=np.nan
    for i in range(n):
        if not np.isnan(atr[i]) and atr[i]>0:
            pdi[i]=100*sp_[i]/atr[i]; mdi[i]=100*sm_[i]/atr[i]
            ds=pdi[i]+mdi[i]
            if ds>0: dx[i]=100*abs(pdi[i]-mdi[i])/ds
    return wilder(dx,p), pdi, mdi


# ============================================================
# BACKTEST ENGINE v23.5 - ISOLATED MARGIN (optimizer)
# ============================================================
@njit(cache=True)
def bt_v23_5(close, high, low, fast_ma, slow_ma, adx, rsi,
             adx_th, rsi_lo, rsi_hi,
             sl_pct, trail_act, trail_w,
             delay, margin_eff, lev,
             cap0, fee):
    """
    ISOLATED margin backtest - optimizer version (no trades list).
    - SL on HIGH/LOW (intrabar)
    - Trail on CLOSE
    - REVERSE: opposite cross + ADX + RSI -> close + flip
    - Same-direction skip
    - NO DD protection (want raw MDD)
    - Loss capped at allocated margin (ISOLATED)
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
    # filters: valid long / valid short
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
    sl_price=0.0
    hi_=0.0; lo_=0.0
    tr=0; w=0; lo2=0; gp=0.0; gl=0.0
    slh=0; trh=0; rvh=0; liq=0

    sb=50
    for i in range(n):
        if (not np.isnan(fast_ma[i]) and not np.isnan(slow_ma[i]) and
            not np.isnan(adx[i]) and not np.isnan(rsi[i])):
            sb=max(i,50); break

    for i in range(sb,n):
        if cap<=0: break
        p=close[i]; bh=high[i]; bl=low[i]

        if pos!=0:
            # Track high/low
            if pos==1: hi_=max(hi_,bh)
            else: lo_=min(lo_,bl)

            cp_=False; xp=p; rsn=0

            # SL check on HIGH/LOW (intrabar)
            if pos==1 and bl<=sl_price:
                cp_=True; rsn=1; slh+=1; xp=sl_price
            elif pos==-1 and bh>=sl_price:
                cp_=True; rsn=1; slh+=1; xp=sl_price

            # Trail on CLOSE
            if not cp_:
                if pos==1:
                    cr=(p-ep)/ep*lev
                else:
                    cr=(ep-p)/ep*lev
                if not ta and cr>=trail_act: ta=True; tp_=cr; ts=tp_+trail_w
                if ta:
                    if cr>tp_: tp_=cr; ts=tp_+trail_w
                    if cr<=ts: cp_=True; rsn=2; xp=p; trh+=1

            # Reverse signal
            if not cp_:
                if pos==1 and cd[i]==1 and vs[i]==1: cp_=True; rsn=4; xp=p; rvh+=1
                elif pos==-1 and cu[i]==1 and vl[i]==1: cp_=True; rsn=4; xp=p; rvh+=1

            if cp_:
                # PnL calc
                if pos==1: pp_=(xp-ep)/ep*lev
                else: pp_=(ep-xp)/ep*lev
                f_=psz*xp*fee
                pd_=pmg*pp_-f_
                # ISOLATED: loss capped at allocated margin
                if pd_<-pmg:
                    pd_=-pmg; liq+=1
                cap+=pd_; tr+=1
                if pd_>0: w+=1; gp+=pd_
                else: lo2+=1; gl+=abs(pd_)
                if cap>pk: pk=cap
                dd=(pk-cap)/pk*100 if pk>0 else 0
                if dd>mdd: mdd=dd

                if rsn==4 and cap>100:
                    # Reverse: flip position
                    pos=-pos; ep=p
                    pmg=cap*margin_eff; psz=pmg*lev/p
                    cap-=psz*p*fee
                    if pos==1:
                        sl_price=p*(1+sl_pct/lev)
                        hi_=p; lo_=p
                    else:
                        sl_price=p*(1-sl_pct/lev)
                        hi_=p; lo_=p
                    ta=False; tp_=0; ts=0
                else:
                    pos=0
                if cap<=0: break
                continue

        # Entry
        if pos==0 and cap>100:
            el_=cu[i]==1 and vl[i]==1; es_=cd[i]==1 and vs[i]==1
            if el_ or es_:
                pos=1 if el_ else -1; ep=p
                pmg=cap*margin_eff; psz=pmg*lev/p
                cap-=psz*p*fee
                if pos==1:
                    sl_price=p*(1+sl_pct/lev)  # sl_pct is negative, so this is below
                    hi_=p; lo_=p
                else:
                    sl_price=p*(1-sl_pct/lev)  # sl_pct is negative, so this is above
                    hi_=p; lo_=p
                ta=False; tp_=0; ts=0

    # Close remaining
    if pos!=0 and cap>0 and n>0:
        xp=close[n-1]
        if pos==1: pp_=(xp-ep)/ep*lev
        else: pp_=(ep-xp)/ep*lev
        f_=psz*xp*fee; pd_=pmg*pp_-f_
        if pd_<-pmg: pd_=-pmg; liq+=1
        cap+=pd_; tr+=1
        if pd_>0: w+=1; gp+=pd_
        else: lo2+=1; gl+=abs(pd_)
    if cap>pk: pk=cap
    dd=(pk-cap)/pk*100 if pk>0 else 0
    if dd>mdd: mdd=dd
    return (tr, w, lo2, gp, gl, cap, mdd, slh, liq, trh, rvh)


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
        rm = {'10m':'10min','15m':'15min','30m':'30min','1h':'1h'}
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
# WORKER (multiprocessing)
# ============================================================
_W = None

def _init_worker(cache_file):
    global _W
    with open(cache_file, 'rb') as f:
        _W = pickle.load(f)
    # Warmup JIT
    tf0 = list(_W.keys())[0]
    c = _W[tf0]
    _ = bt_v23_5(
        c['close'][:100], c['high'][:100], c['low'][:100],
        c['fm_WMA_3'][:100], c['sm_EMA_100'][:100],
        c['adx_14'][:100], c['rsi'][:100],
        30.0, 30.0, 70.0, -0.05, 0.05, -0.03,
        0, 0.10, 5.0, 5000.0, 0.0004
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
        sl = SLS[row[7]] / 100.0      # negative: e.g. -5 -> -0.05
        ta = TACTS[row[8]] / 100.0     # positive: e.g. 5 -> 0.05
        tw = TWS[row[9]] / 100.0       # negative: e.g. -3 -> -0.03
        mg = MGNS[row[10]] / 100.0     # e.g. 10 -> 0.10
        lv = float(LEVS[row[11]])

        d = _W[tf]

        res = bt_v23_5(
            d['close'], d['high'], d['low'],
            d[f"fm_{fk}"], d[f"sm_{sk}"],
            d[f"adx_{ap}"], d['rsi'],
            at, float(rl), float(rh),
            sl, ta, tw,
            dl, mg, lv,
            INITIAL_CAPITAL, FEE_RATE
        )
        trades,w,l,gp_,gl_,fb,mdd_,slh,liq_,trh,rvh = res

        # Strict filter: trades >= 5, PF > 1.0, no liquidation, MDD <= 25% (slight overshoot allowed)
        if trades < 5: continue
        if liq_ > 0: continue
        pf = gp_/gl_ if gl_>0 else (999.0 if gp_>0 else 0.0)
        if pf <= 1.0: continue
        if mdd_ > 25.0: continue
        ret = (fb - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        if ret <= 0: continue
        wr = w/trades*100

        # Score = PF*10 + log10(max(Return%,1))*15 - max(0, MDD%-20)*5 + min(Trades,200)/20
        sc = pf*10.0 + log10(max(ret, 1.0))*15.0 - max(0.0, mdd_-20.0)*5.0 + min(trades, 200)/20.0

        results.append((
            row[0],row[1],row[2],row[3],row[4],row[5],row[6],
            row[7],row[8],row[9],row[10],row[11],
            trades, w, l, round(wr,1), round(pf,2),
            round(ret,2), round(fb,2), round(mdd_,2),
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

def calc_adx_wilder_py(high, low, close, period=14):
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
    return adx

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
    if s + period > len(close): return out
    out[s+period-1] = np.nanmean(close[s:s+period])
    m = 2.0 / (period + 1)
    for i in range(s+period, len(close)):
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
    d = np.where(np.isnan(wh) | np.isnan(wf), np.nan, 2.0 * wh - wf)
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
            if avg_l[i] == 0: out[i] = 100.0
            else: out[i] = 100 - 100 / (1 + avg_g[i] / avg_l[i])
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

    rm = {'5m': None, '10m':'10min','15m':'15min','30m':'30min','1h':'1h'}
    if tf_str == '5m':
        out = df[['open','high','low','close','volume']].copy()
    else:
        out = df.resample(rm[tf_str]).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

    print(f"  [Data] 5m:{len(df):,} -> {tf_str}:{len(out):,} | {time.time()-t0:.1f}s", flush=True)
    return out


def run_detailed_backtest(cfg, opens, highs, lows, closes, adx, rsi_arr, fast_ma, slow_ma, timestamps, return_trades=True):
    """
    ISOLATED margin backtest - detailed version with trade-by-trade tracking.
    NO DD protection (raw MDD measurement).
    """
    n = len(closes)
    bal = INITIAL_CAPITAL
    peak_bal = INITIAL_CAPITAL
    pos_dir = 0
    pos_entry = 0.0
    pos_size = 0.0
    pos_margin = 0.0
    pos_time_idx = 0
    pos_highest = 0.0
    pos_lowest = 0.0
    pos_trail_active = False
    pos_trail_peak_roi = 0.0
    pos_trail_sl_roi = 0.0
    pos_sl = 0.0

    pending_signal = 0
    pending_bar = 0

    trades = []

    _n_trades = 0; _n_wins = 0
    _gross_profit = 0.0; _gross_loss = 0.0
    _win_roi_sum = 0.0; _loss_roi_sum = 0.0
    _sl_count = 0; _trail_count = 0; _rev_count = 0; _end_count = 0
    _max_consec_loss = 0; _cur_consec_loss = 0
    _mdd = 0.0; _mdd_peak = INITIAL_CAPITAL
    _liq_events = 0

    tf_hours = {'5m': 1/12, '10m': 1/6, '15m': 0.25, '30m': 0.5, '1h': 1.0}
    bar_h = tf_hours.get(cfg['tf'], 0.25)
    warmup = max(300, int(cfg['slow_period'] * 1.5))

    def close_pos(exit_price, exit_reason, bar_idx):
        nonlocal bal, peak_bal, pos_dir
        nonlocal _n_trades, _n_wins, _gross_profit, _gross_loss
        nonlocal _win_roi_sum, _loss_roi_sum, _sl_count, _trail_count, _rev_count
        nonlocal _end_count, _max_consec_loss, _cur_consec_loss
        nonlocal _mdd, _mdd_peak, _liq_events

        if pos_dir == 1:
            rpnl = pos_size * (exit_price - pos_entry)
        else:
            rpnl = pos_size * (pos_entry - exit_price)
        fee_cost = pos_size * exit_price * FEE_RATE
        total_pnl = rpnl - fee_cost

        # ISOLATED: loss capped at allocated margin
        if total_pnl < -pos_margin:
            total_pnl = -pos_margin
            _liq_events += 1

        bal += total_pnl
        if bal < 0: bal = 0
        peak_bal = max(peak_bal, bal)

        margin = pos_margin
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
        if _mdd_peak > 0:
            dd_now = (_mdd_peak - bal) / _mdd_peak
            _mdd = max(_mdd, dd_now)

        if return_trades:
            lev = cfg['lev']
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
                    ((pos_highest - pos_entry) / pos_entry * lev * 100) if pos_dir == 1 else
                    ((pos_entry - pos_lowest) / pos_entry * lev * 100), 2),
                'hold_bars': hold_bars,
                'hold_hours': round(hold_bars * bar_h, 2),
                'balance': round(bal, 2),
            })
        pos_dir = 0
        return total_pnl

    for i in range(warmup, n):
        if bal <= 0:
            break

        c = closes[i]
        h = highs[i]
        l = lows[i]

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
                    cur_roi = (c - pos_entry) / pos_entry * cfg['lev']
                    if cur_roi >= cfg['trail_act']:
                        pos_trail_active = True
                    if pos_trail_active:
                        if cur_roi > pos_trail_peak_roi:
                            pos_trail_peak_roi = cur_roi
                            pos_trail_sl_roi = pos_trail_peak_roi + cfg['trail_width']  # trail_width is negative
                        if cur_roi <= pos_trail_sl_roi:
                            close_pos(c, 'TRAIL', i)
                            exited = True

            elif pos_dir == -1:
                pos_lowest = min(pos_lowest, l)

                # SL check on intrabar HIGH
                if h >= pos_sl:
                    close_pos(pos_sl, 'SL', i)
                    exited = True
                else:
                    cur_roi = (pos_entry - c) / pos_entry * cfg['lev']
                    if cur_roi >= cfg['trail_act']:
                        pos_trail_active = True
                    if pos_trail_active:
                        if cur_roi > pos_trail_peak_roi:
                            pos_trail_peak_roi = cur_roi
                            pos_trail_sl_roi = pos_trail_peak_roi + cfg['trail_width']
                        if cur_roi <= pos_trail_sl_roi:
                            close_pos(c, 'TRAIL', i)
                            exited = True

            if exited:
                if bal <= 0: break
                continue

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

            if sig == 1 and fast_ma[i] <= slow_ma[i]: continue
            if sig == -1 and fast_ma[i] >= slow_ma[i]: continue

            if np.isnan(adx[i]) or adx[i] < cfg['adx_min']: continue
            if np.isnan(rsi_arr[i]) or not (cfg['rsi_lo'] <= rsi_arr[i] <= cfg['rsi_hi']): continue

            direction = sig

            # Same-direction skip
            if pos_dir == direction: continue

            # REVERSE: close opposite position
            if pos_dir != 0 and pos_dir != direction:
                close_pos(c, 'REV', i)
                if bal <= 0: break

            if bal < 50: continue

            # Position sizing (NO DD protection)
            sz = bal * cfg['margin']
            if sz < 5: continue
            notional = sz * cfg['lev']

            # SL price
            sl_pct_abs = abs(cfg['sl_pct'])
            if direction == 1:
                pos_sl = c * (1 - sl_pct_abs / cfg['lev'])
            else:
                pos_sl = c * (1 + sl_pct_abs / cfg['lev'])

            # Entry fee
            entry_fee = notional * FEE_RATE
            bal -= entry_fee

            # Open position
            pos_dir = direction
            pos_entry = c
            pos_size = notional / c
            pos_margin = sz
            pos_time_idx = i
            pos_highest = c
            pos_lowest = c
            pos_trail_active = False
            pos_trail_peak_roi = 0.0
            pos_trail_sl_roi = 0.0

    # Close remaining position
    if pos_dir != 0 and bal > 0:
        close_pos(closes[-1], 'END', n - 1)

    total_ret = (bal - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    n_losses = _n_trades - _n_wins
    avg_win_roi = _win_roi_sum / _n_wins if _n_wins > 0 else 0
    avg_loss_roi = _loss_roi_sum / n_losses if n_losses > 0 else 0

    metrics = {
        'initial': INITIAL_CAPITAL,
        'final': round(bal, 2),
        'return_pct': round(total_ret, 2),
        'trades': _n_trades,
        'wins': _n_wins,
        'losses': n_losses,
        'pf': round(_gross_profit / _gross_loss, 4) if _gross_loss > 0 else (999 if _gross_profit > 0 else 0),
        'mdd_pct': round(_mdd * 100, 2),
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
        'liq_events': _liq_events,
    }

    return metrics, trades


# ============================================================
# REPORT GENERATION
# ============================================================
def generate_report_text(label, cfg, metrics, trades_list, timestamps_idx):
    """Generate full report text with monthly/yearly tables."""
    lines = []
    T = metrics['trades']

    lines.append(f"\n{'='*100}")
    lines.append(f"  REPORT: [{label}]")
    lines.append(f"{'='*100}")
    lines.append(f"  Strategy:      {cfg['fast_name']}/{cfg['slow_name']} | {cfg['tf']}")
    lines.append(f"  ADX({cfg['adx_period']})>={cfg['adx_min']} RSI {cfg['rsi_lo']}-{cfg['rsi_hi']} Delay={cfg['delay']}")
    lines.append(f"  Margin Mode:   ISOLATED")
    lines.append(f"  Margin:        {cfg['margin']*100:.0f}% | Leverage: {cfg['lev']}x")
    lines.append(f"  SL: -{cfg['sl_pct']*100:.0f}% | Trail Act: +{cfg['trail_act']*100:.0f}% | Trail Width: {cfg['trail_width']*100:+.0f}%")
    lines.append(f"  Fee: {FEE_RATE*100:.2f}% | ISOLATED | REVERSE + Same-dir skip | NO DD Protection")
    lines.append(f"{'='*100}")
    lines.append(f"  Initial:        ${INITIAL_CAPITAL:,.0f}")
    lines.append(f"  Final:          ${metrics['final']:,.2f}")
    lines.append(f"  Return:         {metrics['return_pct']:+,.1f}%")
    lines.append(f"  PF:             {metrics['pf']:.2f}")
    lines.append(f"  MDD:            {metrics['mdd_pct']:.1f}%")
    lines.append(f"  Trades:         {T}")
    lines.append(f"  Win Rate:       {metrics['win_rate']:.1f}%")
    lines.append(f"  Avg Win:        {metrics['avg_win']:+.2f}%")
    lines.append(f"  Avg Loss:       {metrics['avg_loss']:+.2f}%")
    lines.append(f"  R:R:            {metrics['rr']:.2f}")
    lines.append(f"  Max Consec L:   {metrics['max_consec_loss']}")
    lines.append(f"  SL Hits:        {metrics['sl_count']}")
    lines.append(f"  Trail Hits:     {metrics['trail_count']}")
    lines.append(f"  Rev Hits:       {metrics['rev_count']}")
    lines.append(f"  Liquidations:   {metrics['liq_events']}")

    if not trades_list:
        lines.append("  NO TRADES")
        return "\n".join(lines), []

    df = pd.DataFrame(trades_list)

    # Direction analysis
    lines.append(f"\n  DIRECTION ANALYSIS")
    lines.append("  " + "-"*70)
    for d in ['LONG', 'SHORT']:
        sub = df[df['direction'] == d]
        if len(sub) == 0: continue
        sw = sub[sub['pnl'] > 0]
        lines.append(f"  {d:>5}: {len(sub):>4} ({len(sub)/T*100:.0f}%) "
                      f"WR:{len(sw)/len(sub)*100:.0f}% "
                      f"AvgROI:{sub['roi_pct'].mean():+.2f}% "
                      f"PnL:${sub['pnl'].sum():+,.0f}")

    # Exit reason analysis
    lines.append(f"\n  EXIT REASON ANALYSIS")
    lines.append("  " + "-"*70)
    for r in ['TRAIL', 'REV', 'SL', 'END']:
        rt = df[df['exit_reason'] == r]
        if len(rt) == 0: continue
        rw = rt[rt['pnl'] > 0]
        wr_val = len(rw)/len(rt)*100 if len(rt) else 0
        lines.append(f"  {r:>5}: {len(rt):>4} ({len(rt)/T*100:.0f}%) "
                      f"WR:{wr_val:.0f}% AvgROI:{rt['roi_pct'].mean():+.2f}% "
                      f"PnL:${rt['pnl'].sum():+,.0f}")

    # Hold time analysis
    lines.append(f"\n  HOLD TIME ANALYSIS")
    lines.append("  " + "-"*70)
    for a, b, lb in [(0,2,'<2h'),(2,8,'2-8h'),(8,24,'8-24h'),
                      (24,72,'1-3d'),(72,168,'3-7d'),(168,9999,'7d+')]:
        ht = df[(df['hold_hours']>=a) & (df['hold_hours']<b)]
        if len(ht):
            hw = ht[ht['pnl']>0]
            lines.append(f"  {lb:>6}: {len(ht):>4} "
                          f"WR:{len(hw)/len(ht)*100:.0f}% "
                          f"AvgROI:{ht['roi_pct'].mean():+.2f}% "
                          f"PnL:${ht['pnl'].sum():+,.0f}")

    # Monthly table
    df['exit_dt'] = pd.to_datetime(df['exit_time'])
    df['month'] = df['exit_dt'].dt.to_period('M')
    all_months = pd.period_range('2020-01', '2026-03', freq='M')
    mg = df.groupby('month')

    lines.append(f"\n{'='*100}")
    lines.append(f"  MONTHLY PERFORMANCE (2020-01 to 2026-03)")
    lines.append(f"{'='*100}")
    lines.append(f"  {'Month':>8} {'Trd':>4} {'W':>3} {'L':>3} {'WR%':>5} "
                  f"{'GrossP':>12} {'GrossL':>12} {'NetPnL':>12} {'PF':>6} "
                  f"{'Bal':>14} {'Ret%':>7}")
    lines.append("  " + "-"*105)

    rb = INITIAL_CAPITAL
    yearly = {}
    lm = 0; pm = 0; tm = 0
    monthly_rows = []

    for mo in all_months:
        if mo in mg.groups:
            g = mg.get_group(mo)
            nt = len(g); nw = len(g[g['pnl']>0]); nl = nt - nw
            wr = nw/nt*100 if nt else 0
            gp2 = g[g['pnl']>0]['pnl'].sum() if nw else 0
            gl2 = abs(g[g['pnl']<=0]['pnl'].sum()) if nl else 0
            net = g['pnl'].sum()
            mpf = gp2/gl2 if gl2>0 else (999 if gp2>0 else 0)
        else:
            nt=nw=nl=0; wr=0; gp2=gl2=net=0; mpf=0

        sbr = rb; rb += net
        mr = net/sbr*100 if sbr>0 else 0
        tm += 1
        if net<0: lm+=1
        if net>0: pm+=1

        y = str(mo)[:4]
        if y not in yearly:
            yearly[y] = {'p':0,'t':0,'w':0,'l':0,'gp':0,'gl':0,'sb':sbr}
        yearly[y]['p'] += net; yearly[y]['t'] += nt
        yearly[y]['w'] += nw; yearly[y]['l'] += nl
        yearly[y]['gp'] += gp2; yearly[y]['gl'] += gl2
        yearly[y]['eb'] = rb

        pfs = f"{mpf:.1f}" if mpf<999 else "INF"
        if mpf==0 and net==0: pfs="-"
        mk = " <<" if net<0 else ""
        lines.append(f"  {str(mo):>8} {nt:>4} {nw:>3} {nl:>3} {wr:>4.0f}% "
                      f"${gp2:>10,.0f} ${gl2:>10,.0f} ${net:>+10,.0f} {pfs:>5} "
                      f"${rb:>12,.0f} {mr:>+6.1f}%{mk}")

        monthly_rows.append({
            'month': str(mo), 'trades': nt, 'wins': nw, 'losses': nl,
            'win_rate': round(wr,1), 'gross_profit': round(gp2,2),
            'gross_loss': round(gl2,2), 'pnl': round(net,2),
            'pf': round(mpf,2) if mpf<999 else 999,
            'balance': round(rb,2), 'return_pct': round(mr,2)
        })

    # Yearly summary
    lines.append(f"\n{'='*100}")
    lines.append(f"  YEARLY PERFORMANCE")
    lines.append(f"{'='*100}")
    lines.append(f"  {'Year':>6} {'Trd':>4} {'W':>4} {'L':>4} {'WR%':>5} "
                  f"{'GrossP':>12} {'GrossL':>12} {'NetPnL':>12} {'PF':>6} {'YrRet%':>8}")
    lines.append("  " + "-"*90)
    for y2 in sorted(yearly):
        yd = yearly[y2]
        ywr = yd['w']/yd['t']*100 if yd['t'] else 0
        ypf = yd['gp']/yd['gl'] if yd['gl']>0 else (999 if yd['gp']>0 else 0)
        yret = yd['p']/yd['sb']*100 if yd['sb']>0 else 0
        pfs = f"{ypf:.2f}" if ypf<999 else "INF"
        lines.append(f"  {y2:>6} {yd['t']:>3} {yd['w']:>4} {yd['l']:>4} {ywr:>4.0f}% "
                      f"${yd['gp']:>10,.0f} ${yd['gl']:>10,.0f} ${yd['p']:>+10,.0f} "
                      f"{pfs:>5} {yret:>+7.1f}%")

    pyrs = sum(1 for v in yearly.values() if v['p']>0)
    lines.append(f"\n  Profit Months: {pm}/{tm} ({pm/max(1,tm)*100:.0f}%)")
    lines.append(f"  Loss Months:   {lm}/{tm} ({lm/max(1,tm)*100:.0f}%)")
    lines.append(f"  Profit Years:  {pyrs}/{len(yearly)}")

    # Top/Bottom trades
    ds = df.sort_values('pnl', ascending=False)
    lines.append(f"\n  TOP 10 TRADES")
    lines.append("  " + "-"*110)
    for idx, (_, r) in enumerate(ds.head(10).iterrows()):
        lines.append(f"  {idx+1:>2} {r['direction']:>5} {r['entry_time'][:16]} -> "
                      f"{r['exit_time'][:16]} {r['exit_reason']:>5} "
                      f"ROI:{r['roi_pct']:>+7.1f}% PeakROI:{r['peak_roi_pct']:>+7.1f}% "
                      f"PnL:${r['pnl']:>+10,.0f}")
    lines.append(f"\n  BOTTOM 10 TRADES")
    lines.append("  " + "-"*110)
    for idx, (_, r) in enumerate(ds.tail(10).iterrows()):
        lines.append(f"  {idx+1:>2} {r['direction']:>5} {r['entry_time'][:16]} -> "
                      f"{r['exit_time'][:16]} {r['exit_reason']:>5} "
                      f"ROI:{r['roi_pct']:>+7.1f}% PeakROI:{r['peak_roi_pct']:>+7.1f}% "
                      f"PnL:${r['pnl']:>+10,.0f}")

    return "\n".join(lines), monthly_rows


# ============================================================
# MAIN
# ============================================================
def main():
    T0 = time.time()

    ts = total_space()
    print(f"\n{'='*100}", flush=True)
    print(f"  v23.5 BTC/USDT FUTURES - MDD-FOCUSED OPTIMIZATION PIPELINE", flush=True)
    print(f"  ISOLATED Margin | Wilder ADX | 1,500,000 Random Samples", flush=True)
    print(f"  Total parameter space: {ts:,}", flush=True)
    print(f"  Initial: ${INITIAL_CAPITAL:,.0f} | Fee: {FEE_RATE*100:.2f}% | Workers: {NWORKERS}", flush=True)
    print(f"  Targets: MDD<=20% | PF>=5 | Return>=100,000%", flush=True)
    print(f"  Score = PF*10 + log10(max(Ret%,1))*15 - max(0,MDD%-20)*5 + min(Trades,200)/20", flush=True)
    print(f"{'='*100}\n", flush=True)

    # ================================================================
    # PHASE 1: PRECOMPUTE + OPTIMIZATION
    # ================================================================
    print(f"{'#'*100}", flush=True)
    print(f"  PHASE 1: DATA LOADING + INDICATOR PRECOMPUTATION", flush=True)
    print(f"{'#'*100}\n", flush=True)

    df5m = load_and_prep()
    print(f"\nPrecomputing indicators for all TFs...", flush=True)
    caches = precompute_all(df5m)

    # Save cache for workers
    print(f"\nSaving indicator cache...", flush=True)
    t0 = time.time()
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(caches, f, protocol=pickle.HIGHEST_PROTOCOL)
    fsize = os.path.getsize(CACHE_FILE) / 1024 / 1024
    print(f"  Cache saved: {fsize:.1f} MB | {time.time()-t0:.1f}s", flush=True)

    # Generate random combos
    print(f"\nGenerating {NUM_SAMPLES:,} random combos (seed={SEED})...", flush=True)
    rng = random.Random(SEED)
    dims = [len(l) for l in ALL_LISTS]
    combos = []
    seen = set()
    while len(combos) < NUM_SAMPLES:
        c = tuple(rng.randint(0, d-1) for d in dims)
        if c not in seen:
            seen.add(c)
            combos.append(c)
    print(f"  Generated {len(combos):,} unique combos", flush=True)

    # Chunk for workers
    chunks = []
    for ci in range(0, len(combos), CHUNK_PER_WORKER):
        chunks.append(combos[ci:ci+CHUNK_PER_WORKER])
    print(f"  {len(chunks):,} chunks of ~{CHUNK_PER_WORKER}", flush=True)

    # ================================================================
    # RUN OPTIMIZATION
    # ================================================================
    print(f"\n{'#'*100}", flush=True)
    print(f"  PHASE 1b: RUNNING OPTIMIZATION ({NUM_SAMPLES:,} combos)", flush=True)
    print(f"{'#'*100}\n", flush=True)

    all_results = []
    t_start = time.time()

    pool = Pool(NWORKERS, initializer=_init_worker, initargs=(CACHE_FILE,))
    done = 0
    for batch_res in pool.imap_unordered(_run_chunk, chunks, chunksize=4):
        all_results.extend(batch_res)
        done += CHUNK_PER_WORKER
        if done % PROGRESS_INTERVAL < CHUNK_PER_WORKER:
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(combos) - done) / rate if rate > 0 else 0
            print(f"  {done:>10,}/{NUM_SAMPLES:,} | Valid: {len(all_results):>6,} | "
                  f"{rate:,.0f}/s | ETA: {eta:.0f}s | Elapsed: {elapsed:.0f}s", flush=True)
    pool.close()
    pool.join()

    opt_time = time.time() - t_start
    print(f"\n  Optimization complete: {len(all_results):,} valid results in {opt_time:.1f}s "
          f"({opt_time/60:.1f} min)", flush=True)

    if not all_results:
        print("  NO VALID RESULTS FOUND. Exiting.", flush=True)
        return

    # Build results DataFrame
    cols = ['fast_ma_idx','slow_ma_idx','tf_idx','adx_period_idx','adx_thresh_idx',
            'rsi_idx','delay_idx','sl_idx','trail_act_idx','trail_w_idx','margin_idx','lev_idx',
            'trades','wins','losses','win_rate','pf',
            'return_pct','final_bal','mdd_pct',
            'sl_hits','trail_hits','rev_hits','score']
    rdf = pd.DataFrame(all_results, columns=cols)

    # Decode indices to names
    rdf['fast_ma'] = rdf['fast_ma_idx'].map(lambda x: FAST_KEYS[x])
    rdf['slow_ma'] = rdf['slow_ma_idx'].map(lambda x: SLOW_KEYS[x])
    rdf['tf'] = rdf['tf_idx'].map(lambda x: TIMEFRAMES[x])
    rdf['adx_period'] = rdf['adx_period_idx'].map(lambda x: ADX_PERIODS[x])
    rdf['adx_thresh'] = rdf['adx_thresh_idx'].map(lambda x: ADX_THS[x])
    rdf['rsi_range'] = rdf['rsi_idx'].map(lambda x: f"{RSI_RS[x][0]}-{RSI_RS[x][1]}")
    rdf['rsi_lo'] = rdf['rsi_idx'].map(lambda x: RSI_RS[x][0])
    rdf['rsi_hi'] = rdf['rsi_idx'].map(lambda x: RSI_RS[x][1])
    rdf['delay'] = rdf['delay_idx'].map(lambda x: DELAYS[x])
    rdf['sl_pct'] = rdf['sl_idx'].map(lambda x: SLS[x])
    rdf['trail_act'] = rdf['trail_act_idx'].map(lambda x: TACTS[x])
    rdf['trail_width'] = rdf['trail_w_idx'].map(lambda x: TWS[x])
    rdf['margin_pct'] = rdf['margin_idx'].map(lambda x: MGNS[x])
    rdf['leverage'] = rdf['lev_idx'].map(lambda x: LEVS[x])

    # Save top 100
    top100 = rdf.sort_values('score', ascending=False).head(100)
    top100_path = os.path.join(OUT_DIR, "v23_5_opt_top100.csv")
    save_cols = ['fast_ma','slow_ma','tf','adx_period','adx_thresh','rsi_range',
                 'delay','sl_pct','trail_act','trail_width','margin_pct','leverage',
                 'trades','wins','losses','win_rate','pf','return_pct','final_bal','mdd_pct',
                 'sl_hits','trail_hits','rev_hits','score']
    top100[save_cols].to_csv(top100_path, index=False)
    print(f"\n  Saved: {top100_path}", flush=True)

    # ================================================================
    # PHASE 2: ANALYSIS TABLES
    # ================================================================
    print(f"\n\n{'#'*100}", flush=True)
    print(f"  PHASE 2: ANALYSIS TABLES", flush=True)
    print(f"{'#'*100}\n", flush=True)

    summary_lines = []
    summary_lines.append(f"v23.5 OPTIMIZATION SUMMARY")
    summary_lines.append(f"Total combos: {NUM_SAMPLES:,} | Valid: {len(rdf):,}")
    summary_lines.append(f"Time: {opt_time:.1f}s ({opt_time/60:.1f} min)")
    summary_lines.append("")

    # TABLE 1: TOP 30 sorted by return (MDD <= 25%)
    t1 = rdf[rdf['mdd_pct'] <= 25.0].sort_values('return_pct', ascending=False).head(30)
    header = f"  {'#':>3} {'Fast':>8} {'Slow':>8} {'TF':>4} {'ADXp':>4} {'ADXt':>4} {'RSI':>7} " \
             f"{'D':>2} {'SL%':>5} {'TA%':>4} {'TW%':>4} {'M%':>4} {'Lev':>3} " \
             f"{'Trd':>4} {'WR%':>5} {'PF':>6} {'Ret%':>12} {'MDD%':>6} {'Score':>7}"
    sep = "  " + "-"*130

    print(f"\n  TABLE 1: TOP 30 BY RETURN (MDD <= 25%)", flush=True)
    print(sep, flush=True)
    print(header, flush=True)
    print(sep, flush=True)
    summary_lines.append("TABLE 1: TOP 30 BY RETURN (MDD <= 25%)")
    summary_lines.append(header)
    summary_lines.append(sep)
    for rank, (_, r) in enumerate(t1.iterrows(), 1):
        line = (f"  {rank:>3} {r['fast_ma']:>8} {r['slow_ma']:>8} {r['tf']:>4} "
                f"{r['adx_period']:>4} {r['adx_thresh']:>4} {r['rsi_range']:>7} "
                f"{r['delay']:>2} {r['sl_pct']:>5} {r['trail_act']:>4} {r['trail_width']:>4} "
                f"{r['margin_pct']:>4} {r['leverage']:>3} "
                f"{r['trades']:>4} {r['win_rate']:>4.0f}% {r['pf']:>5.1f} "
                f"{r['return_pct']:>+11,.1f}% {r['mdd_pct']:>5.1f}% {r['score']:>6.1f}")
        print(line, flush=True)
        summary_lines.append(line)

    # TABLE 2: TOP 10 with MDD <= 20%
    t2 = rdf[rdf['mdd_pct'] <= 20.0].sort_values('return_pct', ascending=False).head(10)
    print(f"\n\n  TABLE 2: TOP 10 WITH MDD <= 20%", flush=True)
    print(sep, flush=True)
    print(header, flush=True)
    print(sep, flush=True)
    summary_lines.append("")
    summary_lines.append("TABLE 2: TOP 10 WITH MDD <= 20%")
    summary_lines.append(header)
    summary_lines.append(sep)
    for rank, (_, r) in enumerate(t2.iterrows(), 1):
        line = (f"  {rank:>3} {r['fast_ma']:>8} {r['slow_ma']:>8} {r['tf']:>4} "
                f"{r['adx_period']:>4} {r['adx_thresh']:>4} {r['rsi_range']:>7} "
                f"{r['delay']:>2} {r['sl_pct']:>5} {r['trail_act']:>4} {r['trail_width']:>4} "
                f"{r['margin_pct']:>4} {r['leverage']:>3} "
                f"{r['trades']:>4} {r['win_rate']:>4.0f}% {r['pf']:>5.1f} "
                f"{r['return_pct']:>+11,.1f}% {r['mdd_pct']:>5.1f}% {r['score']:>6.1f}")
        print(line, flush=True)
        summary_lines.append(line)

    # TABLE 3: TOP 10 with PF >= 5
    t3 = rdf[rdf['pf'] >= 5.0].sort_values('return_pct', ascending=False).head(10)
    print(f"\n\n  TABLE 3: TOP 10 WITH PF >= 5", flush=True)
    print(sep, flush=True)
    print(header, flush=True)
    print(sep, flush=True)
    summary_lines.append("")
    summary_lines.append("TABLE 3: TOP 10 WITH PF >= 5")
    summary_lines.append(header)
    summary_lines.append(sep)
    for rank, (_, r) in enumerate(t3.iterrows(), 1):
        line = (f"  {rank:>3} {r['fast_ma']:>8} {r['slow_ma']:>8} {r['tf']:>4} "
                f"{r['adx_period']:>4} {r['adx_thresh']:>4} {r['rsi_range']:>7} "
                f"{r['delay']:>2} {r['sl_pct']:>5} {r['trail_act']:>4} {r['trail_width']:>4} "
                f"{r['margin_pct']:>4} {r['leverage']:>3} "
                f"{r['trades']:>4} {r['win_rate']:>4.0f}% {r['pf']:>5.1f} "
                f"{r['return_pct']:>+11,.1f}% {r['mdd_pct']:>5.1f}% {r['score']:>6.1f}")
        print(line, flush=True)
        summary_lines.append(line)

    # TABLE 4: BEST RETURN where MDD <= 20% AND PF >= 3
    t4 = rdf[(rdf['mdd_pct'] <= 20.0) & (rdf['pf'] >= 3.0)].sort_values('return_pct', ascending=False).head(10)
    print(f"\n\n  TABLE 4: BEST RETURN WHERE MDD <= 20% AND PF >= 3", flush=True)
    print(sep, flush=True)
    print(header, flush=True)
    print(sep, flush=True)
    summary_lines.append("")
    summary_lines.append("TABLE 4: BEST RETURN WHERE MDD <= 20% AND PF >= 3")
    summary_lines.append(header)
    summary_lines.append(sep)
    for rank, (_, r) in enumerate(t4.iterrows(), 1):
        line = (f"  {rank:>3} {r['fast_ma']:>8} {r['slow_ma']:>8} {r['tf']:>4} "
                f"{r['adx_period']:>4} {r['adx_thresh']:>4} {r['rsi_range']:>7} "
                f"{r['delay']:>2} {r['sl_pct']:>5} {r['trail_act']:>4} {r['trail_width']:>4} "
                f"{r['margin_pct']:>4} {r['leverage']:>3} "
                f"{r['trades']:>4} {r['win_rate']:>4.0f}% {r['pf']:>5.1f} "
                f"{r['return_pct']:>+11,.1f}% {r['mdd_pct']:>5.1f}% {r['score']:>6.1f}")
        print(line, flush=True)
        summary_lines.append(line)

    # PARAMETER FREQUENCY ANALYSIS (top 100 by score)
    print(f"\n\n  PARAMETER FREQUENCY ANALYSIS (Top 100 by score)", flush=True)
    print("  " + "="*80, flush=True)
    summary_lines.append("")
    summary_lines.append("PARAMETER FREQUENCY ANALYSIS (Top 100 by score)")
    summary_lines.append("="*80)

    freq_df = rdf.sort_values('score', ascending=False).head(100)
    for col_name, col_key in [
        ('Fast MA', 'fast_ma'), ('Slow MA', 'slow_ma'), ('Timeframe', 'tf'),
        ('ADX Period', 'adx_period'), ('ADX Threshold', 'adx_thresh'),
        ('RSI Range', 'rsi_range'), ('Entry Delay', 'delay'),
        ('SL %', 'sl_pct'), ('Trail Act %', 'trail_act'),
        ('Trail Width %', 'trail_width'),
        ('Margin %', 'margin_pct'), ('Leverage', 'leverage')
    ]:
        vc = freq_df[col_key].value_counts()
        top_vals = vc.head(5)
        vals_str = " | ".join([f"{v}:{c}" for v, c in top_vals.items()])
        line = f"  {col_name:>16}: {vals_str}"
        print(line, flush=True)
        summary_lines.append(line)

    # Save summary
    summary_path = os.path.join(OUT_DIR, "v23_5_opt_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))
    print(f"\n  Saved: {summary_path}", flush=True)

    # ================================================================
    # PHASE 3: WINNER DETAILED BACKTEST + 30-RUN VERIFICATION
    # ================================================================
    # WINNER = best return with MDD <= 20%
    winner_pool = rdf[rdf['mdd_pct'] <= 20.0].sort_values('return_pct', ascending=False)
    if len(winner_pool) == 0:
        # Fallback: best return with MDD <= 25%
        print("\n  WARNING: No results with MDD<=20%. Using MDD<=25% fallback.", flush=True)
        winner_pool = rdf[rdf['mdd_pct'] <= 25.0].sort_values('return_pct', ascending=False)
    if len(winner_pool) == 0:
        print("  NO WINNER FOUND. Skipping detailed analysis.", flush=True)
        return

    winner = winner_pool.iloc[0]

    print(f"\n\n{'#'*100}", flush=True)
    print(f"  PHASE 3: WINNER DETAILED BACKTEST + 30-RUN VERIFICATION", flush=True)
    print(f"{'#'*100}\n", flush=True)

    fm_parts = winner['fast_ma'].split('_')
    sm_parts = winner['slow_ma'].split('_')

    winner_cfg = {
        'name': f"{winner['fast_ma']}/{winner['slow_ma']} {winner['tf']} "
                f"ADX({winner['adx_period']})>={winner['adx_thresh']} "
                f"RSI{winner['rsi_range']} D{winner['delay']}",
        'fast_name': f"{fm_parts[0]}({fm_parts[1]})",
        'slow_name': f"{sm_parts[0]}({sm_parts[1]})",
        'fast_type': fm_parts[0],
        'fast_period': int(fm_parts[1]),
        'slow_type': sm_parts[0],
        'slow_period': int(sm_parts[1]),
        'tf': winner['tf'],
        'adx_period': int(winner['adx_period']),
        'adx_min': int(winner['adx_thresh']),
        'rsi_lo': int(winner['rsi_lo']),
        'rsi_hi': int(winner['rsi_hi']),
        'delay': int(winner['delay']),
        'sl_pct': abs(winner['sl_pct']) / 100.0,
        'trail_act': winner['trail_act'] / 100.0,
        'trail_width': winner['trail_width'] / 100.0,  # negative: e.g. -4 -> -0.04
        'margin': winner['margin_pct'] / 100.0,
        'lev': int(winner['leverage']),
    }

    print(f"  WINNER: {winner_cfg['fast_name']}/{winner_cfg['slow_name']} | {winner_cfg['tf']}", flush=True)
    print(f"  ADX({winner_cfg['adx_period']})>={winner_cfg['adx_min']} RSI {winner_cfg['rsi_lo']}-{winner_cfg['rsi_hi']} D={winner_cfg['delay']}", flush=True)
    print(f"  SL={winner['sl_pct']}% Trail={winner['trail_act']}%/w{winner['trail_width']}%", flush=True)
    print(f"  Margin={winner['margin_pct']}% Lev={winner_cfg['lev']}x ISOLATED", flush=True)
    print(f"  Opt: Ret={winner['return_pct']:+,.1f}% PF={winner['pf']:.1f} MDD={winner['mdd_pct']:.1f}%", flush=True)

    # Load data for winner's timeframe
    tf = winner_cfg['tf']
    print(f"\n  Loading data for {tf}...", flush=True)
    dfr = load_data_phase2(tf)

    o = dfr['open'].values; h = dfr['high'].values; l = dfr['low'].values
    c = dfr['close'].values; tsi = dfr.index

    # Compute indicators
    print(f"  Computing indicators...", flush=True)
    t0 = time.time()
    fast_ma = calc_fast_ma_py(c, fm_parts[0], int(fm_parts[1]))
    slow_ma = calc_slow_ma_py(c, sm_parts[0], int(sm_parts[1]))
    adx_arr = calc_adx_wilder_py(h, l, c, winner_cfg['adx_period'])
    rsi_arr = calc_rsi_py(c, 14)
    print(f"  Indicators done | {time.time()-t0:.1f}s", flush=True)

    # Detailed backtest
    print(f"  Running detailed backtest...", flush=True)
    t0 = time.time()
    metrics, best_trades = run_detailed_backtest(
        winner_cfg, o, h, l, c, adx_arr, rsi_arr, fast_ma, slow_ma, tsi, return_trades=True
    )
    elapsed = time.time() - t0
    print(f"  Backtest done | {elapsed:.1f}s", flush=True)
    print(f"  Result: ${metrics['final']:,.2f} ({metrics['return_pct']:+,.1f}%) "
          f"PF={metrics['pf']:.2f} MDD={metrics['mdd_pct']:.1f}% "
          f"Trades={metrics['trades']} WR={metrics['win_rate']:.1f}%", flush=True)

    # Generate full report
    report_text, monthly_rows = generate_report_text("WINNER", winner_cfg, metrics, best_trades, tsi)
    print(report_text, flush=True)

    # ---- 30-RUN VERIFICATION ----
    print(f"\n\n{'='*100}", flush=True)
    print(f"  30-RUN VERIFICATION", flush=True)
    print(f"{'='*100}\n", flush=True)

    verify_rows = []
    for run in range(1, 31):
        t0 = time.time()
        m, _ = run_detailed_backtest(
            winner_cfg, o, h, l, c, adx_arr, rsi_arr, fast_ma, slow_ma, tsi, return_trades=False
        )
        elapsed = time.time() - t0
        verify_rows.append({
            'strategy': 'WINNER',
            'run': run,
            'final': m['final'],
            'return_pct': m['return_pct'],
            'trades': m['trades'],
            'pf': m['pf'],
            'mdd_pct': m['mdd_pct'],
            'win_rate': m['win_rate'],
            'time_s': round(elapsed, 2),
        })
        if run <= 5 or run % 10 == 0:
            print(f"    Run {run:>2} | ${m['final']:>12,.2f} | {m['return_pct']:>+10.1f}% | "
                  f"PF={m['pf']:.2f} | MDD={m['mdd_pct']:.1f}% | T={m['trades']} | {elapsed:.1f}s", flush=True)

    finals = [r['final'] for r in verify_rows]
    std_f = np.std(finals)
    mean_f = np.mean(finals)
    print(f"\n  VERIFICATION RESULT: mean=${mean_f:,.2f} std=${std_f:.4f} "
          f"{'PASS (deterministic)' if std_f < 0.01 else 'INVESTIGATE (std!=0)'}", flush=True)

    # ================================================================
    # PHASE 4: SAVE ALL OUTPUT FILES
    # ================================================================
    print(f"\n\n{'#'*100}", flush=True)
    print(f"  PHASE 4: SAVING OUTPUT FILES", flush=True)
    print(f"{'#'*100}\n", flush=True)

    # v23_5_FINAL_report.txt
    report_path = os.path.join(OUT_DIR, "v23_5_FINAL_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("v23.5 BTC/USDT FUTURES - FINAL WINNER REPORT\n")
        f.write(f"Initial: ${INITIAL_CAPITAL:,.0f} | ISOLATED margin | Wilder ADX\n")
        f.write(f"Score = PF*10 + log10(max(Ret%,1))*15 - max(0,MDD%-20)*5 + min(Trades,200)/20\n")
        f.write(f"Targets: MDD<=20% | PF>=5 | Return>=100,000%\n")
        f.write("="*100 + "\n")
        f.write(report_text + "\n")
        f.write("\n" + "="*100 + "\n")
        f.write("30-RUN VERIFICATION\n")
        f.write("="*100 + "\n")
        for vr in verify_rows:
            f.write(f"  Run {vr['run']:>2} | ${vr['final']:>12,.2f} | {vr['return_pct']:>+10.1f}% | "
                    f"PF={vr['pf']:.2f} | MDD={vr['mdd_pct']:.1f}% | T={vr['trades']}\n")
        f.write(f"\n  mean=${mean_f:,.2f} std=${std_f:.4f}\n")
    print(f"  Saved: {report_path}", flush=True)

    # v23_5_FINAL_trades.csv
    if best_trades:
        trades_path = os.path.join(OUT_DIR, "v23_5_FINAL_trades.csv")
        pd.DataFrame(best_trades).to_csv(trades_path, index=False)
        print(f"  Saved: {trades_path}", flush=True)

    # v23_5_FINAL_monthly.csv
    if monthly_rows:
        monthly_path = os.path.join(OUT_DIR, "v23_5_FINAL_monthly.csv")
        pd.DataFrame(monthly_rows).to_csv(monthly_path, index=False)
        print(f"  Saved: {monthly_path}", flush=True)

    # v23_5_FINAL_30run.csv
    run_path = os.path.join(OUT_DIR, "v23_5_FINAL_30run.csv")
    pd.DataFrame(verify_rows).to_csv(run_path, index=False)
    print(f"  Saved: {run_path}", flush=True)

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    te = time.time() - T0
    print(f"\n{'='*100}", flush=True)
    print(f"  v23.5 PIPELINE COMPLETE", flush=True)
    print(f"  Total time: {te:.1f}s ({te/60:.1f} min)", flush=True)
    print(f"  Optimization: {NUM_SAMPLES:,} combos | Valid: {len(rdf):,}", flush=True)
    print(f"  WINNER: {winner_cfg['name']}", flush=True)
    print(f"    Final: ${metrics['final']:,.2f} ({metrics['return_pct']:+,.1f}%)", flush=True)
    print(f"    PF={metrics['pf']:.2f} MDD={metrics['mdd_pct']:.1f}% Trades={metrics['trades']} WR={metrics['win_rate']:.1f}%", flush=True)
    print(f"    Margin={winner['margin_pct']}% Lev={winner_cfg['lev']}x ISOLATED", flush=True)

    # Check against targets
    print(f"\n  TARGET CHECK:", flush=True)
    checks = [
        ('MDD <= 20%', metrics['mdd_pct'] <= 20.0, f"{metrics['mdd_pct']:.1f}%"),
        ('PF >= 5', metrics['pf'] >= 5.0, f"{metrics['pf']:.2f}"),
        ('Return >= 100,000%', metrics['return_pct'] >= 100000, f"{metrics['return_pct']:+,.1f}%"),
        ('Margin <= 25%', winner['margin_pct'] <= 25, f"{winner['margin_pct']}%"),
        ('Leverage <= 15x', winner_cfg['lev'] <= 15, f"{winner_cfg['lev']}x"),
        ('No Liquidation', metrics['liq_events'] == 0, f"{metrics['liq_events']} events"),
    ]
    for label, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {label}: {val}", flush=True)

    print(f"\n  Output files:", flush=True)
    print(f"    - v23_5_opt_top100.csv", flush=True)
    print(f"    - v23_5_opt_summary.txt", flush=True)
    print(f"    - v23_5_FINAL_report.txt", flush=True)
    print(f"    - v23_5_FINAL_trades.csv", flush=True)
    print(f"    - v23_5_FINAL_monthly.csv", flush=True)
    print(f"    - v23_5_FINAL_30run.csv", flush=True)
    print(f"{'='*100}", flush=True)


if __name__ == '__main__':
    main()
