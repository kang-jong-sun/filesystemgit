"""
v23.5b FAST: MDD<=30% Optimization - 30m ONLY + Numba JIT + Multiprocessing
=============================================================================
Uses Numba JIT for the backtest loop (~100x faster than pure Python).
Single TF (30m ~109K bars), pre-computed indicators, 10 workers.
Target: 1,000,000 combos in < 10 minutes.
"""
import numpy as np
import pandas as pd
import time
import os
import sys
import io
import random
import pickle
from collections import Counter
from multiprocessing import Pool, cpu_count, freeze_support
from numba import njit
import numba

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print(f"Python {sys.version.split()[0]} | Numba {numba.__version__} | CPUs: {cpu_count()}", flush=True)

# ============================================================
# CONFIG
# ============================================================
DIR = r"D:\filesystem\futures\btc_V1\test3"
CSV_FILES = [os.path.join(DIR, f"btc_usdt_5m_2020_to_now_part{i}.csv") for i in range(1, 4)]
INIT = 5000.0
FEE = 0.0004
N_COMBOS = 1_000_000
SEED = 42
NWORKERS = min(10, max(1, cpu_count() - 2))
BATCH_SIZE = 2000

# Parameter space (indices for compact combo representation)
FAST_NAMES = ['ema3', 'ema5', 'ema7', 'wma3', 'hma3', 'hma5']
SLOW_NAMES = ['ema100', 'ema150', 'ema200', 'ema250', 'sma200', 'sma300']
ADX_P_NAMES = ['adx14', 'adx20']
ADX_THS = np.array([25.0, 30.0, 35.0, 40.0, 45.0], dtype=np.float64)
RSI_LOS = np.array([25.0, 30.0, 30.0, 35.0, 35.0, 40.0], dtype=np.float64)
RSI_HIS = np.array([65.0, 65.0, 70.0, 65.0, 70.0, 75.0], dtype=np.float64)
RSI_LABELS = ['25-65', '30-65', '30-70', '35-65', '35-70', '40-75']
DELAYS = np.array([0, 1, 2, 3, 5], dtype=np.int64)
SLS = np.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.10], dtype=np.float64)
TAS = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15], dtype=np.float64)
TWS = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float64)
MGNS = np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25], dtype=np.float64)
LEVS = np.array([3.0, 5.0, 7.0, 10.0, 12.0, 15.0], dtype=np.float64)

N_FAST = len(FAST_NAMES)
N_SLOW = len(SLOW_NAMES)
N_ADXP = len(ADX_P_NAMES)
N_ADXTH = len(ADX_THS)
N_RSI = len(RSI_LOS)
N_DELAY = len(DELAYS)
N_SL = len(SLS)
N_TA = len(TAS)
N_TW = len(TWS)
N_MGN = len(MGNS)
N_LEV = len(LEVS)

CACHE_FILE = os.path.join(DIR, "_v23_5b_fast_cache.pkl")


# ============================================================
# INDICATOR FUNCTIONS (numpy-based, run once in main process)
# ============================================================
def ema_np(c, p):
    n = len(c)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    m = 2.0 / (p + 1)
    s = 0
    while s < n and np.isnan(c[s]):
        s += 1
    if s >= n:
        return out
    out[s] = c[s]
    for i in range(s + 1, n):
        if not np.isnan(c[i]):
            out[i] = c[i] * m + out[i-1] * (1.0 - m)
        else:
            out[i] = out[i-1]
    return out


def sma_np(c, p):
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    cs = np.nancumsum(c)
    out[p-1:] = (cs[p-1:] - np.concatenate([[0.0], cs[:n-p]])) / p
    return out


def wma_np(c, p):
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    w = np.arange(1, p + 1, dtype=np.float64)
    ws = w.sum()
    conv = np.convolve(c, w[::-1], mode='valid')
    out[p-1:p-1+len(conv)] = conv / ws
    return out


def hma_np(c, p):
    hp = max(p // 2, 1)
    sq = max(int(np.sqrt(p)), 1)
    wh = wma_np(c, hp)
    wf = wma_np(c, p)
    d = 2.0 * wh - wf
    return wma_np(d, sq)


def wilder_smooth(arr, p):
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    s = 0
    while s < n and np.isnan(arr[s]):
        s += 1
    if s + p > n:
        return out
    out[s + p - 1] = np.nanmean(arr[s:s+p])
    for i in range(s + p, n):
        if not np.isnan(arr[i]) and not np.isnan(out[i-1]):
            out[i] = (out[i-1] * (p - 1) + arr[i]) / p
        elif not np.isnan(out[i-1]):
            out[i] = out[i-1]
    return out


def adx_wilder(h, l, c, p=14):
    n = len(c)
    tr = np.empty(n, dtype=np.float64)
    pdm = np.empty(n, dtype=np.float64)
    mdm = np.empty(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    pdm[0] = 0.0
    mdm[0] = 0.0
    for i in range(1, n):
        hl = h[i] - l[i]
        hpc = abs(h[i] - c[i-1])
        lpc = abs(l[i] - c[i-1])
        tr[i] = max(hl, hpc, lpc)
        up = h[i] - h[i-1]
        dn = l[i-1] - l[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
    atr = wilder_smooth(tr, p)
    spdm = wilder_smooth(pdm, p)
    smdm = wilder_smooth(mdm, p)
    pdi = np.full(n, np.nan, dtype=np.float64)
    mdi = np.full(n, np.nan, dtype=np.float64)
    dx = np.full(n, np.nan, dtype=np.float64)
    valid = (~np.isnan(atr)) & (atr > 0)
    pdi[valid] = 100.0 * spdm[valid] / atr[valid]
    mdi[valid] = 100.0 * smdm[valid] / atr[valid]
    s = pdi + mdi
    s_valid = valid & (s > 0)
    dx[s_valid] = 100.0 * np.abs(pdi[s_valid] - mdi[s_valid]) / s[s_valid]
    return wilder_smooth(dx, p)


def rsi_wilder(c, p=14):
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < p + 1:
        return out
    delta = np.diff(c)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    ag = np.mean(gains[:p])
    al = np.mean(losses[:p])
    out[p] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(p, len(delta)):
        ag = (ag * (p - 1) + gains[i]) / p
        al = (al * (p - 1) + losses[i]) / p
        out[i + 1] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


# ============================================================
# NUMBA JIT BACKTEST ENGINE
# ============================================================
@njit(cache=True)
def bt_jit(hi, lo, cl, fast, slow, adx, rsi,
           M, L, SL, TA, TW, ADX_TH, RSI_LO, RSI_HI, DELAY,
           init_cap, fee):
    """
    Numba JIT backtest. Isolated margin.
    SL on HIGH/LOW. Trail on CLOSE (leveraged ROI). REVERSE + same-dir skip.
    Returns: (balance, return%, PF, MDD%, trades, wins, win_rate, sl_hits)
    """
    n = len(cl)
    bal = init_cap
    pkb = init_cap
    pos = 0
    ep = 0.0
    pmg = 0.0
    psz = 0.0
    sl_price = 0.0
    ta_active = False
    peak_roi = 0.0
    trail_sl_roi = 0.0
    mdd = 0.0
    trd = 0
    wins = 0
    gp = 0.0
    gl = 0.0
    slh = 0
    pending = 0
    pbar = 0
    w = 300
    if DELAY + 10 > w:
        w = DELAY + 10

    for i in range(w, n):
        h = hi[i]
        lv_ = lo[i]
        c = cl[i]

        if pos != 0:
            # SL on HIGH/LOW
            if pos == 1:
                if lv_ <= sl_price:
                    pnl_pct = (sl_price - ep) / ep * L
                    pnl = pmg * pnl_pct - psz * sl_price * fee
                    if pnl < -pmg:
                        pnl = -pmg
                    bal += pnl
                    if bal > pkb:
                        pkb = bal
                    if pkb > 0.0:
                        dd = (pkb - bal) / pkb
                        if dd > mdd:
                            mdd = dd
                    if pnl > 0.0:
                        gp += pnl
                        wins += 1
                    else:
                        gl += abs(pnl)
                    trd += 1
                    slh += 1
                    pos = 0
                    continue
            else:
                if h >= sl_price:
                    pnl_pct = (ep - sl_price) / ep * L
                    pnl = pmg * pnl_pct - psz * sl_price * fee
                    if pnl < -pmg:
                        pnl = -pmg
                    bal += pnl
                    if bal > pkb:
                        pkb = bal
                    if pkb > 0.0:
                        dd = (pkb - bal) / pkb
                        if dd > mdd:
                            mdd = dd
                    if pnl > 0.0:
                        gp += pnl
                        wins += 1
                    else:
                        gl += abs(pnl)
                    trd += 1
                    slh += 1
                    pos = 0
                    continue

            # Trail on CLOSE (leveraged ROI)
            if pos == 1:
                cur_roi = (c - ep) / ep * L
            else:
                cur_roi = (ep - c) / ep * L
            if not ta_active and cur_roi >= TA:
                ta_active = True
                peak_roi = cur_roi
                trail_sl_roi = peak_roi - TW
            if ta_active:
                if cur_roi > peak_roi:
                    peak_roi = cur_roi
                    trail_sl_roi = peak_roi - TW
                if cur_roi <= trail_sl_roi:
                    pnl = pmg * cur_roi - psz * c * fee
                    if pnl < -pmg:
                        pnl = -pmg
                    bal += pnl
                    if bal > pkb:
                        pkb = bal
                    if pkb > 0.0:
                        dd = (pkb - bal) / pkb
                        if dd > mdd:
                            mdd = dd
                    if pnl > 0.0:
                        gp += pnl
                        wins += 1
                    else:
                        gl += abs(pnl)
                    trd += 1
                    pos = 0
                    continue

        # Signal detection (cross)
        fi = fast[i]
        fi1 = fast[i - 1]
        si = slow[i]
        si1 = slow[i - 1]
        if fi == fi and fi1 == fi1 and si == si and si1 == si1:  # NaN check
            if fi1 <= si1 and fi > si:
                pending = 1
                pbar = i
            elif fi1 >= si1 and fi < si:
                pending = -1
                pbar = i

        # Delayed entry
        if pending != 0 and i - pbar == DELAY:
            sig = pending
            pending = 0
            fi2 = fast[i]
            si2 = slow[i]
            if fi2 != fi2 or si2 != si2:
                continue
            if sig == 1 and fi2 <= si2:
                continue
            if sig == -1 and fi2 >= si2:
                continue
            ai = adx[i]
            ri = rsi[i]
            if ai != ai or ai < ADX_TH:
                continue
            if ri != ri or ri < RSI_LO or ri > RSI_HI:
                continue
            if pos == sig:
                continue
            # REVERSE close
            if pos != 0 and pos != sig:
                if pos == 1:
                    pnl_pct = (c - ep) / ep * L
                else:
                    pnl_pct = (ep - c) / ep * L
                pnl = pmg * pnl_pct - psz * c * fee
                if pnl < -pmg:
                    pnl = -pmg
                bal += pnl
                if bal > pkb:
                    pkb = bal
                if pkb > 0.0:
                    dd = (pkb - bal) / pkb
                    if dd > mdd:
                        mdd = dd
                if pnl > 0.0:
                    gp += pnl
                    wins += 1
                else:
                    gl += abs(pnl)
                trd += 1
                pos = 0
            # New entry
            if bal <= 100.0:
                continue
            pmg = bal * M
            psz = pmg * L / c
            bal -= psz * c * fee
            ep = c
            pos = sig
            ta_active = False
            peak_roi = 0.0
            trail_sl_roi = 0.0
            if sig == 1:
                sl_price = c * (1.0 - SL / L)
            else:
                sl_price = c * (1.0 + SL / L)

        if bal <= 0.0:
            break

    # Close remaining
    if pos != 0 and bal > 0.0 and n > 0:
        c = cl[n - 1]
        if pos == 1:
            pnl_pct = (c - ep) / ep * L
        else:
            pnl_pct = (ep - c) / ep * L
        pnl = pmg * pnl_pct - psz * c * fee
        if pnl < -pmg:
            pnl = -pmg
        bal += pnl
        trd += 1
        if pnl > 0.0:
            gp += pnl
            wins += 1
        else:
            gl += abs(pnl)
    if bal > pkb:
        pkb = bal
    if pkb > 0.0:
        dd = (pkb - bal) / pkb
        if dd > mdd:
            mdd = dd

    pf = 999.0
    if gl > 0.0:
        pf = gp / gl
    elif gp == 0.0:
        pf = 0.0
    ret = (bal - init_cap) / init_cap * 100.0
    wr = 0.0
    if trd > 0:
        wr = wins / trd * 100.0
    return bal, ret, pf, mdd * 100.0, trd, wins, wr, slh


# ============================================================
# WORKER (multiprocessing)
# ============================================================
_WDATA = None


def _init_worker(cache_file):
    """Load cache and JIT-warmup."""
    global _WDATA
    with open(cache_file, 'rb') as f:
        _WDATA = pickle.load(f)
    # JIT warmup: run once on small slice to compile
    d = _WDATA
    n = min(500, len(d['cl']))
    _ = bt_jit(
        d['hi'][:n], d['lo'][:n], d['cl'][:n],
        d['fast'][0][:n], d['slow'][0][:n],
        d['adx'][0][:n], d['rsi'][:n],
        0.10, 5.0, 0.05, 0.05, 0.03, 30.0, 30.0, 70.0, 1,
        5000.0, 0.0004
    )


def _run_batch(combo_arr):
    """
    combo_arr: numpy int32 array, shape (batch, 11)
    Columns: fast_idx, slow_idx, adxp_idx, adxth_idx, rsi_idx,
             delay_idx, sl_idx, ta_idx, tw_idx, mgn_idx, lev_idx
    """
    global _WDATA
    d = _WDATA
    results = []
    n_combos = combo_arr.shape[0]

    for k in range(n_combos):
        fi = combo_arr[k, 0]
        si = combo_arr[k, 1]
        api = combo_arr[k, 2]
        athi = combo_arr[k, 3]
        rsii = combo_arr[k, 4]
        dlyi = combo_arr[k, 5]
        sli = combo_arr[k, 6]
        tai = combo_arr[k, 7]
        twi = combo_arr[k, 8]
        mgi = combo_arr[k, 9]
        lvi = combo_arr[k, 10]

        b, r, pf, mdd_pct, trd, w, wr, slh_ = bt_jit(
            d['hi'], d['lo'], d['cl'],
            d['fast'][fi], d['slow'][si],
            d['adx'][api], d['rsi'],
            MGNS[mgi], LEVS[lvi], SLS[sli],
            TAS[tai], TWS[twi],
            ADX_THS[athi], RSI_LOS[rsii], RSI_HIS[rsii],
            DELAYS[dlyi],
            INIT, FEE
        )

        # Filter: trades>=5, PF>1.0, MDD<35%, balance>initial
        if trd >= 5 and pf > 1.0 and mdd_pct < 35.0 and b > INIT:
            score = r + pf * 20.0 - max(0.0, mdd_pct - 30.0) * 10.0 + min(trd, 200) / 10.0
            results.append((
                fi, si, api, athi, rsii, dlyi,
                sli, tai, twi, mgi, lvi,
                round(b, 2), round(r, 2), round(pf, 2),
                round(-mdd_pct, 2), trd, w, round(wr, 1), slh_,
                round(score, 2)
            ))
    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 120)
    print("  v23.5b FAST: MDD<=30% Optimization | 30m ONLY | Numba JIT + Multiprocessing")
    print("  Initial: $5,000 | Margin<=25% | Leverage<=15x | Fee: 0.04%")
    print(f"  {N_COMBOS:,} random combinations | {NWORKERS} workers")
    print("=" * 120)
    t_start = time.time()

    # ----------------------------------------------------------
    # PHASE 1: Load data - 30m ONLY
    # ----------------------------------------------------------
    print("\n[1/4] Loading data (5m -> 30m resample)...", flush=True)
    t0 = time.time()
    dfs = []
    for f in CSV_FILES:
        df = pd.read_csv(f, parse_dates=['timestamp'])
        dfs.append(df)
        print(f"  {os.path.basename(f)}: {len(df):,} rows", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(np.float64)

    df30 = df.resample('30min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    print(f"  5m: {len(df):,} -> 30m: {len(df30):,} bars", flush=True)
    print(f"  Period: {df30.index[0]} ~ {df30.index[-1]}", flush=True)
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)
    del df, dfs

    # ----------------------------------------------------------
    # PHASE 2: Pre-compute ALL indicators
    # ----------------------------------------------------------
    print("\n[2/4] Pre-computing indicators...", flush=True)
    t0 = time.time()
    hi = df30['high'].values.astype(np.float64).copy()
    lo = df30['low'].values.astype(np.float64).copy()
    cl = df30['close'].values.astype(np.float64).copy()

    # Store as indexed arrays for compact worker access
    fast_arrs = [
        ema_np(cl, 3),   # 0: ema3
        ema_np(cl, 5),   # 1: ema5
        ema_np(cl, 7),   # 2: ema7
        wma_np(cl, 3),   # 3: wma3
        hma_np(cl, 3),   # 4: hma3
        hma_np(cl, 5),   # 5: hma5
    ]
    print(f"  Fast MAs: {len(fast_arrs)} done", flush=True)

    slow_arrs = [
        ema_np(cl, 100),  # 0: ema100
        ema_np(cl, 150),  # 1: ema150
        ema_np(cl, 200),  # 2: ema200
        ema_np(cl, 250),  # 3: ema250
        sma_np(cl, 200),  # 4: sma200
        sma_np(cl, 300),  # 5: sma300
    ]
    print(f"  Slow MAs: {len(slow_arrs)} done", flush=True)

    adx_arrs = [
        adx_wilder(hi, lo, cl, 14),  # 0: adx14
        adx_wilder(hi, lo, cl, 20),  # 1: adx20
    ]
    print(f"  ADX: {len(adx_arrs)} done", flush=True)

    rsi_arr = rsi_wilder(cl, 14)
    print(f"  RSI: done", flush=True)

    cache = {
        'hi': hi, 'lo': lo, 'cl': cl,
        'fast': fast_arrs, 'slow': slow_arrs,
        'adx': adx_arrs, 'rsi': rsi_arr
    }

    nbars = len(cl)
    n_total = 3 + len(fast_arrs) + len(slow_arrs) + len(adx_arrs) + 1
    mem_mb = nbars * 8 * n_total / 1024 / 1024
    print(f"  {n_total} arrays x {nbars:,} bars = {mem_mb:.1f} MB", flush=True)
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)

    # Save cache
    print("  Saving cache...", flush=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cache: {os.path.getsize(CACHE_FILE)/1024/1024:.1f} MB", flush=True)

    # JIT warmup in main process
    print("  JIT warmup...", flush=True)
    t0 = time.time()
    ns = min(500, nbars)
    _ = bt_jit(hi[:ns], lo[:ns], cl[:ns], fast_arrs[0][:ns], slow_arrs[0][:ns],
               adx_arrs[0][:ns], rsi_arr[:ns],
               0.10, 5.0, 0.05, 0.05, 0.03, 30.0, 30.0, 70.0, 1, 5000.0, 0.0004)
    print(f"  JIT compiled: {time.time()-t0:.1f}s", flush=True)

    # ----------------------------------------------------------
    # PHASE 3: Random search with multiprocessing
    # ----------------------------------------------------------
    print(f"\n[3/4] Optimization: {N_COMBOS:,} combos x {NWORKERS} workers...", flush=True)
    t0 = time.time()
    random.seed(SEED)
    np.random.seed(SEED)

    # Generate all combos as numpy int32 array for efficiency
    combo_data = np.empty((N_COMBOS, 11), dtype=np.int32)
    combo_data[:, 0] = np.random.randint(0, N_FAST, N_COMBOS)
    combo_data[:, 1] = np.random.randint(0, N_SLOW, N_COMBOS)
    combo_data[:, 2] = np.random.randint(0, N_ADXP, N_COMBOS)
    combo_data[:, 3] = np.random.randint(0, N_ADXTH, N_COMBOS)
    combo_data[:, 4] = np.random.randint(0, N_RSI, N_COMBOS)
    combo_data[:, 5] = np.random.randint(0, N_DELAY, N_COMBOS)
    combo_data[:, 6] = np.random.randint(0, N_SL, N_COMBOS)
    combo_data[:, 7] = np.random.randint(0, N_TA, N_COMBOS)
    combo_data[:, 8] = np.random.randint(0, N_TW, N_COMBOS)
    combo_data[:, 9] = np.random.randint(0, N_MGN, N_COMBOS)
    combo_data[:, 10] = np.random.randint(0, N_LEV, N_COMBOS)

    # Split into batches
    batches = []
    for i in range(0, N_COMBOS, BATCH_SIZE):
        batches.append(combo_data[i:i+BATCH_SIZE])
    print(f"  {N_COMBOS:,} combos in {len(batches)} batches (size {BATCH_SIZE})", flush=True)

    # Run
    raw_results = []
    done = 0
    with Pool(processes=NWORKERS, initializer=_init_worker, initargs=(CACHE_FILE,)) as pool:
        for batch_result in pool.imap_unordered(_run_batch, batches):
            raw_results.extend(batch_result)
            done += BATCH_SIZE
            if done % 100000 < BATCH_SIZE:
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                eta = (N_COMBOS - done) / speed if speed > 0 else 0
                print(f"  {done:>10,}/{N_COMBOS:,} | Valid: {len(raw_results):>6,} | "
                      f"{elapsed:>5.0f}s | {speed:>6.0f}/s | ETA: {eta:>4.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  DONE: {len(raw_results):,} valid | {elapsed:.0f}s | {N_COMBOS/elapsed:.0f}/s", flush=True)

    # Cleanup
    try:
        os.remove(CACHE_FILE)
    except:
        pass

    # ----------------------------------------------------------
    # PHASE 4: Analysis and output
    # ----------------------------------------------------------
    print(f"\n[4/4] Analysis...", flush=True)

    # Convert raw tuples to dicts
    all_results = []
    for r in raw_results:
        (fi, si, api, athi, rsii, dlyi,
         sli, tai, twi, mgi, lvi,
         final, ret, pf, mdd_val, trades, w, wr, slh_, score) = r
        all_results.append({
            'fast': FAST_NAMES[fi], 'slow': SLOW_NAMES[si],
            'adx_p': ADX_P_NAMES[api], 'adx_th': int(ADX_THS[athi]),
            'rsi': RSI_LABELS[rsii], 'delay': int(DELAYS[dlyi]),
            'sl': SLS[sli], 'ta': TAS[tai], 'tw': TWS[twi],
            'margin': MGNS[mgi], 'lev': int(LEVS[lvi]),
            'final': final, 'return': ret, 'pf': pf,
            'mdd': mdd_val, 'trades': trades, 'wins': w,
            'wr': wr, 'sl_hits': slh_, 'score': score
        })

    # MDD <= 30%
    mdd30 = [r for r in all_results if r['mdd'] >= -30.0]
    mdd30.sort(key=lambda x: -x['return'])

    print(f"\n{'='*120}")
    print(f"  RESULTS: Total valid={len(all_results):,} | MDD<=30%={len(mdd30):,}")
    print(f"{'='*120}")

    # TOP 30 by return
    hdr = (f"  {'#':>3} {'Final$':>10} {'Ret%':>9} {'PF':>6} {'MDD%':>7} {'Trd':>5} "
           f"{'WR%':>5} {'SL':>3} {'Fast':>6} {'Slow':>7} {'ADX':>6} {'RSI':>6} "
           f"{'D':>2} {'SL%':>5} {'TA%':>5} {'TW%':>5} {'M%':>4} {'Lv':>3} {'Score':>7}")

    print(f"\n  TOP 30 BY RETURN (MDD<=30%)")
    print(hdr)
    print("  " + "-" * 118)
    for i, r in enumerate(mdd30[:30]):
        pfs = f"{r['pf']:.1f}" if r['pf'] < 999 else "INF"
        print(f"  {i+1:>3} ${r['final']:>8,.0f} {r['return']:>+8.0f}% {pfs:>5} "
              f"{r['mdd']:>6.1f}% {r['trades']:>4} {r['wr']:>4.0f}% {r['sl_hits']:>3} "
              f"{r['fast']:>6} {r['slow']:>7} {r['adx_p'][-2:]}/{r['adx_th']:>2} "
              f"{r['rsi']:>6} {r['delay']:>2} {r['sl']*100:>4.0f}% {r['ta']*100:>4.0f}% "
              f"{r['tw']*100:>4.0f}% {r['margin']*100:>3.0f}% {r['lev']:>2}x {r['score']:>6.0f}")

    # TOP 10 PF>=3
    pf3 = [r for r in mdd30 if r['pf'] >= 3.0]
    pf3.sort(key=lambda x: -x['return'])
    print(f"\n  TOP 10 PF>=3 AND MDD<=30% ({len(pf3):,} combos)")
    print(hdr)
    print("  " + "-" * 118)
    for i, r in enumerate(pf3[:10]):
        pfs = f"{r['pf']:.1f}" if r['pf'] < 999 else "INF"
        print(f"  {i+1:>3} ${r['final']:>8,.0f} {r['return']:>+8.0f}% {pfs:>5} "
              f"{r['mdd']:>6.1f}% {r['trades']:>4} {r['wr']:>4.0f}% {r['sl_hits']:>3} "
              f"{r['fast']:>6} {r['slow']:>7} {r['adx_p'][-2:]}/{r['adx_th']:>2} "
              f"{r['rsi']:>6} {r['delay']:>2} {r['sl']*100:>4.0f}% {r['ta']*100:>4.0f}% "
              f"{r['tw']*100:>4.0f}% {r['margin']*100:>3.0f}% {r['lev']:>2}x {r['score']:>6.0f}")

    # Parameter frequency
    print(f"\n  PARAMETER FREQUENCY (Top 100 by return, MDD<=30%)")
    print("  " + "-" * 90)
    top100 = mdd30[:100]
    if top100:
        for key in ['fast', 'slow', 'adx_p', 'adx_th', 'rsi', 'delay',
                     'sl', 'ta', 'tw', 'margin', 'lev']:
            vals = [r[key] for r in top100]
            if key in ['sl', 'ta', 'tw', 'margin']:
                vals = [f"{v*100:.0f}%" for v in vals]
            elif key == 'lev':
                vals = [f"{v}x" for v in vals]
            ctr = Counter(vals).most_common(6)
            items = " | ".join(f"{v}:{c}" for v, c in ctr)
            print(f"  {key:>8}: {items}")

    # Save CSV
    out_csv = os.path.join(DIR, "v23_5b_top100.csv")
    if mdd30:
        pd.DataFrame(mdd30[:100]).to_csv(out_csv, index=False)
        print(f"\n  Saved: {out_csv} ({min(100, len(mdd30))} rows)")

    total_time = time.time() - t_start
    print(f"\n{'='*120}")
    print(f"  COMPLETE | Total: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Combos: {N_COMBOS:,} | Valid: {len(all_results):,} | MDD<=30%: {len(mdd30):,}")
    print(f"{'='*120}")


if __name__ == '__main__':
    freeze_support()
    main()
