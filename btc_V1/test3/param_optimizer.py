"""
BTC/USDT Futures Backtest Parameter Optimizer
==============================================
Random-samples 55,000+ parameter combinations across 16 dimensions,
runs a single-engine backtest on 75 months of 5-min data (resampled to 15m/30m),
and ranks results by a composite score.

Uses numba JIT for the critical backtest loop (~20x speedup).
"""
import os, sys, time, random, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import numba as nb

warnings.filterwarnings("ignore")

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)

# ────────────────────────────────────────────
DATA_DIR   = Path(r"D:\filesystem\futures\btc_V1\test3")
NUM_COMBOS = 55_000
INIT_CAP   = 3_000.0
FEE_RATE   = 0.0004
MAX_SL_PCT = 0.05
SEED       = 42

PARAM_SPACE = {
    "fast_ma_type":      ["WMA", "HMA", "EMA"],
    "fast_ma_len":       [2, 3, 4, 5, 7],
    "slow_ma_type":      ["EMA", "SMA"],
    "slow_ma_len":       [100, 150, 200, 250, 300],
    "adx_period":        [14, 20],
    "adx_threshold":     [25, 30, 35, 40, 45],
    "rsi_lower":         [25, 30, 35],
    "rsi_upper":         [65, 70, 75],
    "atr_sl_mult":       [2.0, 2.5, 3.0, 3.5, 4.0],
    "atr_tp1_mult":      [4.0, 5.0, 6.0, 8.0],
    "atr_trail_mult":    [2.0, 2.5, 3.0, 3.5],
    "trail_activation":  [0.04, 0.06, 0.08, 0.10],
    "timeframe":         ["15m", "30m"],
    "leverage":          [5, 7, 10],
    "position_ratio":    [0.15, 0.20, 0.25],
    "min_gap_bars":      [12, 24, 36, 48],
}


# ────────────────────────────────────────────
# NUMBA-ACCELERATED INDICATOR FUNCTIONS
# ────────────────────────────────────────────

@nb.njit(cache=True)
def ema_nb(src, length):
    n = len(src)
    alpha = 2.0 / (length + 1)
    out = np.empty(n, dtype=np.float64)
    out[0] = src[0]
    for i in range(1, n):
        out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1]
    return out


@nb.njit(cache=True)
def sma_nb(src, length):
    n = len(src)
    out = np.full(n, np.nan, dtype=np.float64)
    s = 0.0
    for i in range(length):
        s += src[i]
    out[length - 1] = s / length
    for i in range(length, n):
        s += src[i] - src[i - length]
        out[i] = s / length
    return out


@nb.njit(cache=True)
def wma_nb(src, length):
    n = len(src)
    out = np.full(n, np.nan, dtype=np.float64)
    wsum = length * (length + 1) / 2.0
    for i in range(length - 1, n):
        s = 0.0
        for j in range(length):
            s += src[i - length + 1 + j] * (j + 1)
        out[i] = s / wsum
    return out


@nb.njit(cache=True)
def hma_nb(src, length):
    half = max(1, length // 2)
    sqr = max(1, int(np.sqrt(length)))
    wma_half = wma_nb(src, half)
    wma_full = wma_nb(src, length)
    n = len(src)
    diff = np.empty(n, dtype=np.float64)
    for i in range(n):
        diff[i] = 2.0 * wma_half[i] - wma_full[i]
    # fill NaN at start
    first_valid = -1
    for i in range(n):
        if not np.isnan(diff[i]):
            first_valid = i
            break
    if first_valid > 0:
        for i in range(first_valid):
            diff[i] = diff[first_valid]
    return wma_nb(diff, sqr)


def compute_ma(src, ma_type, length):
    if ma_type == "EMA":
        return ema_nb(src, length)
    elif ma_type == "SMA":
        return sma_nb(src, length)
    elif ma_type == "WMA":
        return wma_nb(src, length)
    elif ma_type == "HMA":
        return hma_nb(src, length)
    raise ValueError(f"Unknown MA type: {ma_type}")


@nb.njit(cache=True)
def rsi_nb(close, period):
    n = len(close)
    delta = np.empty(n, dtype=np.float64)
    delta[0] = 0.0
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    gain = np.empty(n, dtype=np.float64)
    loss = np.empty(n, dtype=np.float64)
    for i in range(n):
        if delta[i] > 0:
            gain[i] = delta[i]; loss[i] = 0.0
        else:
            gain[i] = 0.0; loss[i] = -delta[i]
    avg_gain = ema_nb(gain, period)
    avg_loss = ema_nb(loss, period)
    rsi = np.empty(n, dtype=np.float64)
    for i in range(n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        else:
            rsi[i] = 100.0
    return rsi


@nb.njit(cache=True)
def atr_nb(high, low, close, period):
    n = len(high)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))
    return ema_nb(tr, period)


@nb.njit(cache=True)
def adx_nb(high, low, close, period):
    n = len(high)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        if up > dn and up > 0:
            plus_dm[i] = up
        if dn > up and dn > 0:
            minus_dm[i] = dn

    atr_val = atr_nb(high, low, close, period)
    sm_plus = ema_nb(plus_dm, period)
    sm_minus = ema_nb(minus_dm, period)

    dx = np.zeros(n, dtype=np.float64)
    for i in range(n):
        safe_atr = atr_val[i] if atr_val[i] > 1e-10 else 1e-10
        pdi = 100.0 * sm_plus[i] / safe_atr
        mdi = 100.0 * sm_minus[i] / safe_atr
        di_sum = pdi + mdi
        if di_sum > 0:
            dx[i] = 100.0 * abs(pdi - mdi) / di_sum
    return ema_nb(dx, period)


# ────────────────────────────────────────────
# NUMBA-ACCELERATED BACKTEST CORE
# ────────────────────────────────────────────

@nb.njit(cache=True)
def backtest_core(signals, close, high, low, atr, pw,
                  leverage, pos_ratio, sl_mult, tp1_mult,
                  trail_mult, trail_act, min_gap,
                  init_cap, fee_rate, max_sl_pct):
    """
    Core backtest loop compiled with numba for speed.
    Returns: (pf, mdd_pct, return_pct, final_capital, trade_count,
              win_rate, rr_ratio, wins, losses, gross_profit, gross_loss,
              liquidations, weighted_pnl)
    """
    n = len(signals)
    capital = init_cap
    peak_capital = capital
    max_dd = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    losses = 0
    trade_count = 0
    weighted_pnl = 0.0
    liquidations = 0

    in_pos = False
    pos_dir = 0
    entry_p = 0.0
    pos_margin = 0.0
    sl_p = 0.0
    tp1_p = 0.0
    tp1_hit = False
    tp2_done = False
    trail_on = False
    trail_ext = 0.0
    remain = 1.0
    last_bar = -min_gap

    for i in range(n):
        if in_pos:
            bh = high[i]
            bl = low[i]
            bc = close[i]

            # SL check
            sl_hit = False
            if pos_dir == 1 and bl <= sl_p:
                sl_hit = True
            elif pos_dir == -1 and bh >= sl_p:
                sl_hit = True

            if sl_hit:
                if pos_dir == 1:
                    pnl_pct = (sl_p - entry_p) / entry_p
                else:
                    pnl_pct = (entry_p - sl_p) / entry_p
                pnl_usd = pos_margin * leverage * pnl_pct * remain - pos_margin * remain * leverage * fee_rate
                if pnl_pct * leverage <= -0.90:
                    liquidations += 1
                weighted_pnl += pnl_usd * pw[i]
                capital += pnl_usd
                if pnl_usd > 0:
                    gross_profit += pnl_usd
                    wins += 1
                else:
                    gross_loss += abs(pnl_usd)
                    losses += 1
                trade_count += 1
                in_pos = False
                last_bar = i
                if capital > peak_capital:
                    peak_capital = capital
                dd = (peak_capital - capital) / peak_capital * 100.0
                if dd > max_dd:
                    max_dd = dd
                continue

            # TP1: 30% partial
            if not tp1_hit:
                tp1_reached = False
                if pos_dir == 1 and bh >= tp1_p:
                    tp1_reached = True
                elif pos_dir == -1 and bl <= tp1_p:
                    tp1_reached = True
                if tp1_reached:
                    partial = 0.30
                    if pos_dir == 1:
                        pnl_pct = (tp1_p - entry_p) / entry_p
                    else:
                        pnl_pct = (entry_p - tp1_p) / entry_p
                    pnl_usd = pos_margin * leverage * pnl_pct * partial - pos_margin * partial * leverage * fee_rate
                    weighted_pnl += pnl_usd * pw[i]
                    capital += pnl_usd
                    if pnl_usd > 0:
                        gross_profit += pnl_usd
                    remain -= partial
                    tp1_hit = True
                    # Move SL to breakeven on TP1 hit
                    if pos_dir == 1:
                        be_sl = entry_p * 1.001  # tiny profit lock
                        if be_sl > sl_p:
                            sl_p = be_sl
                    else:
                        be_sl = entry_p * 0.999
                        if be_sl < sl_p:
                            sl_p = be_sl

            # TP2: another 30% at 2x distance
            if tp1_hit and not tp2_done and remain > 0.41:
                tp2_dist = abs(tp1_p - entry_p) * 2.0
                if pos_dir == 1:
                    tp2_p = entry_p + tp2_dist
                else:
                    tp2_p = entry_p - tp2_dist
                tp2_reached = False
                if pos_dir == 1 and bh >= tp2_p:
                    tp2_reached = True
                elif pos_dir == -1 and bl <= tp2_p:
                    tp2_reached = True
                if tp2_reached:
                    partial = 0.30
                    if pos_dir == 1:
                        pnl_pct = (tp2_p - entry_p) / entry_p
                    else:
                        pnl_pct = (entry_p - tp2_p) / entry_p
                    pnl_usd = pos_margin * leverage * pnl_pct * partial - pos_margin * partial * leverage * fee_rate
                    weighted_pnl += pnl_usd * pw[i]
                    capital += pnl_usd
                    if pnl_usd > 0:
                        gross_profit += pnl_usd
                    remain -= partial
                    tp2_done = True

            # Trailing stop
            if pos_dir == 1:
                cur_r = (bh - entry_p) / entry_p
                if cur_r >= trail_act:
                    trail_on = True
                if trail_on:
                    if bh > trail_ext:
                        trail_ext = bh
                    a_val = atr[i]
                    if a_val <= 0:
                        a_val = entry_p * 0.01
                    new_sl = trail_ext * (1.0 - a_val * trail_mult / bc)
                    if new_sl > sl_p:
                        sl_p = new_sl
            else:
                cur_r = (entry_p - bl) / entry_p
                if cur_r >= trail_act:
                    trail_on = True
                if trail_on:
                    if bl < trail_ext:
                        trail_ext = bl
                    a_val = atr[i]
                    if a_val <= 0:
                        a_val = entry_p * 0.01
                    new_sl = trail_ext * (1.0 + a_val * trail_mult / bc)
                    if new_sl < sl_p:
                        sl_p = new_sl

            # Opposite signal -> close
            sig = signals[i]
            if sig != 0 and sig != pos_dir:
                if pos_dir == 1:
                    pnl_pct = (bc - entry_p) / entry_p
                else:
                    pnl_pct = (entry_p - bc) / entry_p
                pnl_usd = pos_margin * leverage * pnl_pct * remain - pos_margin * remain * leverage * fee_rate
                weighted_pnl += pnl_usd * pw[i]
                capital += pnl_usd
                if pnl_usd > 0:
                    gross_profit += pnl_usd
                    wins += 1
                else:
                    gross_loss += abs(pnl_usd)
                    losses += 1
                trade_count += 1
                in_pos = False
                last_bar = i
                if capital > peak_capital:
                    peak_capital = capital
                dd = (peak_capital - capital) / peak_capital * 100.0
                if dd > max_dd:
                    max_dd = dd

        # Open new position
        if not in_pos and signals[i] != 0 and (i - last_bar) >= min_gap and capital > 10.0:
            pos_dir = signals[i]
            entry_p = close[i]
            pos_margin = capital * pos_ratio
            cur_atr = atr[i]
            if np.isnan(cur_atr) or cur_atr <= 0:
                cur_atr = entry_p * 0.01
            sl_pct = cur_atr * sl_mult / entry_p
            if sl_pct > max_sl_pct:
                sl_pct = max_sl_pct
            if pos_dir == 1:
                sl_p = entry_p * (1.0 - sl_pct)
            else:
                sl_p = entry_p * (1.0 + sl_pct)
            tp1_dist = cur_atr * tp1_mult
            if pos_dir == 1:
                tp1_p = entry_p + tp1_dist
            else:
                tp1_p = entry_p - tp1_dist
            tp1_hit = False
            tp2_done = False
            trail_on = False
            trail_ext = entry_p
            remain = 1.0
            in_pos = True
            capital -= pos_margin * leverage * fee_rate

    # Close remaining
    if in_pos:
        if pos_dir == 1:
            pnl_pct = (close[n - 1] - entry_p) / entry_p
        else:
            pnl_pct = (entry_p - close[n - 1]) / entry_p
        pnl_usd = pos_margin * leverage * pnl_pct * remain - pos_margin * remain * leverage * fee_rate
        capital += pnl_usd
        if pnl_usd > 0:
            gross_profit += pnl_usd
            wins += 1
        else:
            gross_loss += abs(pnl_usd)
            losses += 1
        trade_count += 1

    if capital > peak_capital:
        peak_capital = capital
    if peak_capital > 0:
        dd = (peak_capital - capital) / peak_capital * 100.0
        if dd > max_dd:
            max_dd = dd

    if gross_loss > 0:
        pf = gross_profit / gross_loss
    elif gross_profit > 0:
        pf = 99.0
    else:
        pf = 0.0

    win_rate = 0.0
    if trade_count > 0:
        win_rate = wins / trade_count * 100.0

    avg_w = gross_profit / wins if wins > 0 else 0.0
    avg_l = gross_loss / losses if losses > 0 else 0.0
    if avg_l > 0:
        rr = avg_w / avg_l
    elif avg_w > 0:
        rr = 99.0
    else:
        rr = 0.0

    ret = (capital - init_cap) / init_cap * 100.0

    return (pf, max_dd, ret, capital, trade_count, win_rate, rr,
            wins, losses, gross_profit, gross_loss, liquidations, weighted_pnl)


# ────────────────────────────────────────────
# SIGNAL GENERATION (vectorized numpy)
# ────────────────────────────────────────────

def generate_signals(cache, params):
    fast_ma = cache[f"fast_{params['fast_ma_type']}_{params['fast_ma_len']}"]
    slow_ma = cache[f"slow_{params['slow_ma_type']}_{params['slow_ma_len']}"]
    adx = cache[f"adx_{params['adx_period']}"]
    rsi = cache["rsi"]

    adx_thresh = params["adx_threshold"]
    rsi_hi = params["rsi_upper"]
    warmup = max(params["slow_ma_len"], 50)

    n = len(fast_ma)
    signals = np.zeros(n, dtype=np.int64)

    above = fast_ma > slow_ma
    prev_above = np.roll(above, 1)
    prev_above[0] = above[0]
    cross_up = (~prev_above) & above
    cross_down = prev_above & (~above)
    trend_long = above & (adx > adx_thresh)
    trend_short = (~above) & (adx > adx_thresh)

    long_cond = (cross_up | trend_long) & (adx > adx_thresh) & (rsi < rsi_hi)
    short_cond = (cross_down | trend_short) & (adx > adx_thresh) & (rsi > (100 - rsi_hi))

    signals[long_cond] = 1
    signals[short_cond] = -1
    signals[:warmup] = 0
    nan_mask = np.isnan(fast_ma) | np.isnan(slow_ma)
    signals[nan_mask] = 0

    return signals


# ────────────────────────────────────────────
# DATA LOADING & RESAMPLING
# ────────────────────────────────────────────

def load_data():
    print("[DATA] Loading 3 CSV files...")
    t0 = time.time()
    parts = []
    for i in range(1, 4):
        fn = DATA_DIR / f"btc_usdt_5m_2020_to_now_part{i}.csv"
        df = pd.read_csv(fn, parse_dates=["timestamp"])
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df.drop_duplicates(subset="timestamp", keep="first", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[DATA] Loaded {len(df):,} rows "
          f"({df['timestamp'].min()} -> {df['timestamp'].max()}) in {time.time()-t0:.1f}s")
    return df


def resample_ohlcv(df5m, tf):
    df = df5m.set_index("timestamp")
    resampled = df.resample(tf).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open"])
    resampled.reset_index(inplace=True)
    return resampled


# ────────────────────────────────────────────
# PRE-COMPUTE INDICATOR CACHE
# ────────────────────────────────────────────

def precompute_indicators(df, tf_label):
    t0 = time.time()
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    hlc3 = (h + l + c) / 3.0

    cache = {"timestamp": df["timestamp"].values, "open": o, "high": h, "low": l, "close": c, "hlc3": hlc3}

    for ma_type in ["WMA", "HMA", "EMA"]:
        for length in [2, 3, 4, 5, 7]:
            cache[f"fast_{ma_type}_{length}"] = compute_ma(hlc3, ma_type, length)

    for ma_type in ["EMA", "SMA"]:
        for length in [100, 150, 200, 250, 300]:
            cache[f"slow_{ma_type}_{length}"] = compute_ma(c, ma_type, length)

    cache["rsi"] = rsi_nb(c, 14)
    cache["atr_14"] = atr_nb(h, l, c, 14)
    cache["atr_20"] = atr_nb(h, l, c, 20)
    cache["adx_14"] = adx_nb(h, l, c, 14)
    cache["adx_20"] = adx_nb(h, l, c, 20)

    years = pd.DatetimeIndex(df["timestamp"].values).year.values
    cache["period_weight"] = np.where(years <= 2022, 0.7, 1.2).astype(np.float64)

    print(f"[INDICATORS] {tf_label}: precomputed in {time.time()-t0:.1f}s")
    return cache


# ────────────────────────────────────────────
# WRAPPER: run one backtest
# ────────────────────────────────────────────

def run_single(cache, params):
    signals = generate_signals(cache, params)
    atr = cache[f"atr_{params['adx_period']}"]
    result = backtest_core(
        signals, cache["close"], cache["high"], cache["low"], atr, cache["period_weight"],
        params["leverage"], params["position_ratio"],
        params["atr_sl_mult"], params["atr_tp1_mult"],
        params["atr_trail_mult"], params["trail_activation"],
        params["min_gap_bars"],
        INIT_CAP, FEE_RATE, MAX_SL_PCT
    )
    pf, mdd, ret, final_cap, tc, wr, rr, w, l, gp, gl, liq, wpnl = result
    return {
        "pf": round(pf, 3), "mdd_pct": round(mdd, 2),
        "return_pct": round(ret, 2), "final_capital": round(final_cap, 2),
        "trade_count": int(tc), "win_rate": round(wr, 2),
        "rr_ratio": round(rr, 3), "wins": int(w), "losses": int(l),
        "gross_profit": round(gp, 2), "gross_loss": round(gl, 2),
        "liquidations": int(liq), "weighted_pnl": round(wpnl, 2),
    }


# ────────────────────────────────────────────
# RANDOM SAMPLING
# ────────────────────────────────────────────

def generate_random_combos(n, seed=SEED):
    rng = random.Random(seed)
    combos = []
    seen = set()
    while len(combos) < n:
        combo = {k: rng.choice(v) for k, v in PARAM_SPACE.items()}
        key = tuple(combo.values())
        if key not in seen:
            seen.add(key)
            combos.append(combo)
    return combos


# ────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  BTC/USDT FUTURES PARAMETER OPTIMIZER (Numba JIT)")
    print(f"  {NUM_COMBOS:,} random combinations")
    print("=" * 70)
    t_start = time.time()

    df5m = load_data()

    print("\n[RESAMPLE] Creating 15m and 30m data...")
    df15m = resample_ohlcv(df5m, "15min")
    df30m = resample_ohlcv(df5m, "30min")
    print(f"  15m: {len(df15m):,} bars | 30m: {len(df30m):,} bars")

    print("\n[INDICATORS] Precomputing all indicator variants...")
    cache_15m = precompute_indicators(df15m, "15m")
    cache_30m = precompute_indicators(df30m, "30m")

    # Warm up numba JIT
    print("\n[JIT] Warming up numba (first run)...")
    t_jit = time.time()
    test_params = {
        'fast_ma_type': 'WMA', 'fast_ma_len': 3,
        'slow_ma_type': 'EMA', 'slow_ma_len': 200,
        'adx_period': 14, 'adx_threshold': 25,
        'rsi_lower': 30, 'rsi_upper': 70,
        'atr_sl_mult': 3.0, 'atr_tp1_mult': 6.0,
        'atr_trail_mult': 2.5, 'trail_activation': 0.06,
        'timeframe': '30m', 'leverage': 7,
        'position_ratio': 0.20, 'min_gap_bars': 24,
    }
    _ = run_single(cache_30m, test_params)
    _ = run_single(cache_15m, test_params)
    print(f"  JIT compiled in {time.time()-t_jit:.1f}s")

    # Benchmark
    t_bench = time.time()
    for _ in range(200):
        run_single(cache_30m, test_params)
    bench_time = time.time() - t_bench
    rate = 200 / bench_time
    print(f"  Benchmark: {rate:.0f} backtests/s (ETA: {NUM_COMBOS/rate:.0f}s = {NUM_COMBOS/rate/60:.1f}min)")

    # Generate combos
    print(f"\n[SAMPLING] Generating {NUM_COMBOS:,} random parameter combinations...")
    combos = generate_random_combos(NUM_COMBOS)
    print(f"  Generated {len(combos):,} unique combinations")

    # Run backtests
    print(f"\n[BACKTEST] Running {len(combos):,} backtests...")
    results = []
    errors = 0
    t_bt = time.time()

    for idx, params in enumerate(combos):
        try:
            cache = cache_15m if params["timeframe"] == "15m" else cache_30m
            metrics = run_single(cache, params)
            metrics.update(params)
            results.append(metrics)
        except Exception as e:
            errors += 1

        done = idx + 1
        if done % 5000 == 0 or done == len(combos):
            elapsed = time.time() - t_bt
            r = done / elapsed if elapsed > 0 else 0
            eta = (len(combos) - done) / r if r > 0 else 0
            # Show best so far
            best_ret = max((x["return_pct"] for x in results[-5000:]), default=0)
            print(f"  Progress: {done:,}/{len(combos):,} ({done/len(combos)*100:.1f}%) "
                  f"| {r:.0f}/s | ETA {eta:.0f}s | Errors: {errors} | BestRet(batch): {best_ret:.1f}%")

    total_time = time.time() - t_bt
    print(f"\n[DONE] {len(results):,} backtests completed in {total_time:.1f}s "
          f"({len(results)/total_time:.0f}/s) | {errors} errors")

    # Filter & Score
    print("\n[SCORING] Filtering and ranking results...")
    df_res = pd.DataFrame(results)

    mask = (df_res["trade_count"] >= 50) & (df_res["mdd_pct"] < 60) & (df_res["liquidations"] == 0)
    df_filtered = df_res[mask].copy()
    print(f"  {len(df_filtered):,} pass strict filters (of {len(df_res):,} total)")

    if len(df_filtered) < 50:
        print("  Relaxing filters to get at least 50 results...")
        mask2 = (df_res["trade_count"] >= 20) & (df_res["mdd_pct"] < 80) & (df_res["liquidations"] == 0)
        df_filtered = df_res[mask2].copy()
        print(f"  Relaxed filters: {len(df_filtered):,} pass")
        if len(df_filtered) < 50:
            mask3 = df_res["trade_count"] >= 10
            df_filtered = df_res[mask3].copy()
            print(f"  Minimal filters: {len(df_filtered):,} pass")
        if len(df_filtered) == 0:
            df_filtered = df_res.copy()

    df_filtered["score"] = (
        df_filtered["pf"].clip(0, 10) * 2.0
        + (100 + df_filtered["return_pct"].clip(-100, 10000)) / 100.0
        - df_filtered["mdd_pct"].clip(0, 100) / 10.0
    )
    df_filtered.sort_values("score", ascending=False, inplace=True)
    top50 = df_filtered.head(50).copy()

    # Save CSV
    csv_cols = [
        "score", "pf", "return_pct", "final_capital", "mdd_pct", "trade_count",
        "win_rate", "rr_ratio", "wins", "losses", "gross_profit", "gross_loss",
        "liquidations", "weighted_pnl",
        "fast_ma_type", "fast_ma_len", "slow_ma_type", "slow_ma_len",
        "adx_period", "adx_threshold", "rsi_lower", "rsi_upper",
        "atr_sl_mult", "atr_tp1_mult", "atr_trail_mult", "trail_activation",
        "timeframe", "leverage", "position_ratio", "min_gap_bars"
    ]
    csv_path = DATA_DIR / "optimization_results_top50.csv"
    top50[csv_cols].to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n[SAVED] Top 50 results -> {csv_path}")

    # Print TOP 10
    print("\n" + "=" * 100)
    print("  TOP 10 PARAMETER COMBINATIONS")
    print("=" * 100)
    for idx, (_, row) in enumerate(top50.head(10).iterrows(), 1):
        print(f"\n  --- RANK #{idx} ---  Score: {row['score']:.3f}")
        print(f"  PF: {row['pf']:.3f} | Return: {row['return_pct']:.1f}% | "
              f"Final: ${row['final_capital']:,.0f} | MDD: {row['mdd_pct']:.1f}%")
        print(f"  Trades: {row['trade_count']:.0f} | WR: {row['win_rate']:.1f}% | "
              f"R:R: {row['rr_ratio']:.3f} | W/L: {row['wins']:.0f}/{row['losses']:.0f}")
        print(f"  Fast: {row['fast_ma_type']}({row['fast_ma_len']:.0f}) | "
              f"Slow: {row['slow_ma_type']}({row['slow_ma_len']:.0f}) | "
              f"TF: {row['timeframe']}")
        print(f"  ADX: {row['adx_period']:.0f}/{row['adx_threshold']:.0f} | "
              f"RSI: {row['rsi_lower']:.0f}-{row['rsi_upper']:.0f}")
        print(f"  ATR SL: {row['atr_sl_mult']:.1f}x | TP1: {row['atr_tp1_mult']:.1f}x | "
              f"Trail: {row['atr_trail_mult']:.1f}x @ {row['trail_activation']:.0%}")
        print(f"  Leverage: {row['leverage']:.0f}x | Size: {row['position_ratio']:.0%} | "
              f"Gap: {row['min_gap_bars']:.0f} bars")

    # Save summary
    summary_path = DATA_DIR / "optimization_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("BTC/USDT FUTURES PARAMETER OPTIMIZATION SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total combinations tested: {len(results):,}\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"Passing strict filters (trades>=50, MDD<60%, no liq): {(mask).sum():,}\n")
        f.write(f"Total backtest time: {total_time:.1f}s\n")
        f.write(f"Rate: {len(results)/total_time:.0f} backtests/sec\n")
        f.write(f"Data: {len(df5m):,} 5m bars | {len(df15m):,} 15m | {len(df30m):,} 30m\n")
        f.write(f"Period: {df5m['timestamp'].min()} -> {df5m['timestamp'].max()}\n")
        f.write(f"Initial capital: ${INIT_CAP:,.0f}\n\n")

        f.write("SCORING FORMULA\n")
        f.write("-" * 40 + "\n")
        f.write("  Score = PF*2 + (100+Return%)/100 - MDD%/10\n")
        f.write("  Higher is better. PF capped at 10, Return capped at 10000%\n\n")

        f.write("DISTRIBUTION OF ALL RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Return%: mean={df_res['return_pct'].mean():.1f}%, "
                f"median={df_res['return_pct'].median():.1f}%, "
                f"std={df_res['return_pct'].std():.1f}%\n")
        f.write(f"  PF: mean={df_res['pf'].mean():.3f}, "
                f"median={df_res['pf'].median():.3f}\n")
        f.write(f"  MDD: mean={df_res['mdd_pct'].mean():.1f}%, "
                f"max={df_res['mdd_pct'].max():.1f}%\n")
        f.write(f"  Trades: mean={df_res['trade_count'].mean():.0f}, "
                f"median={df_res['trade_count'].median():.0f}\n")
        f.write(f"  Win Rate: mean={df_res['win_rate'].mean():.1f}%\n")
        profit_count = (df_res['return_pct'] > 0).sum()
        f.write(f"  Profitable combos: {profit_count:,} "
                f"({profit_count/len(df_res)*100:.1f}%)\n\n")

        for section_name, col_name, vals in [
            ("TIMEFRAME BREAKDOWN", "timeframe", ["15m", "30m"]),
            ("FAST MA TYPE BREAKDOWN", "fast_ma_type", ["WMA", "HMA", "EMA"]),
            ("SLOW MA TYPE BREAKDOWN", "slow_ma_type", ["EMA", "SMA"]),
        ]:
            f.write(f"{section_name}\n")
            f.write("-" * 40 + "\n")
            for v in vals:
                sub = df_res[df_res[col_name] == v]
                prof = (sub['return_pct'] > 0).sum() / len(sub) * 100 if len(sub) > 0 else 0
                f.write(f"  {v}: n={len(sub):,}, "
                        f"avgReturn={sub['return_pct'].mean():.1f}%, "
                        f"avgPF={sub['pf'].mean():.3f}, "
                        f"profitable={prof:.1f}%\n")
            f.write("\n")

        for section_name, col_name, vals in [
            ("LEVERAGE BREAKDOWN", "leverage", [5, 7, 10]),
            ("ADX THRESHOLD BREAKDOWN", "adx_threshold", [25, 30, 35, 40, 45]),
            ("ATR SL MULTIPLIER BREAKDOWN", "atr_sl_mult", [2.0, 2.5, 3.0, 3.5, 4.0]),
            ("ATR TP1 MULTIPLIER BREAKDOWN", "atr_tp1_mult", [4.0, 5.0, 6.0, 8.0]),
            ("ATR TRAIL MULTIPLIER BREAKDOWN", "atr_trail_mult", [2.0, 2.5, 3.0, 3.5]),
            ("TRAILING ACTIVATION BREAKDOWN", "trail_activation", [0.04, 0.06, 0.08, 0.10]),
            ("FAST MA LENGTH BREAKDOWN", "fast_ma_len", [2, 3, 4, 5, 7]),
            ("SLOW MA LENGTH BREAKDOWN", "slow_ma_len", [100, 150, 200, 250, 300]),
            ("MIN GAP BREAKDOWN", "min_gap_bars", [12, 24, 36, 48]),
            ("POSITION RATIO BREAKDOWN", "position_ratio", [0.15, 0.20, 0.25]),
            ("RSI UPPER BREAKDOWN", "rsi_upper", [65, 70, 75]),
            ("RSI LOWER BREAKDOWN", "rsi_lower", [25, 30, 35]),
        ]:
            f.write(f"{section_name}\n")
            f.write("-" * 40 + "\n")
            for v in vals:
                sub = df_res[df_res[col_name] == v]
                f.write(f"  {v}: avgReturn={sub['return_pct'].mean():.1f}%, "
                        f"avgMDD={sub['mdd_pct'].mean():.1f}%, "
                        f"avgPF={sub['pf'].mean():.3f}, "
                        f"avgTrades={sub['trade_count'].mean():.0f}\n")
            f.write("\n")

        f.write("\nTOP 50 PARAMETER COMBINATIONS\n")
        f.write("=" * 70 + "\n\n")
        for idx2, (_, row) in enumerate(top50.iterrows(), 1):
            f.write(f"--- RANK #{idx2} (Score: {row['score']:.3f}) ---\n")
            f.write(f"  PF: {row['pf']:.3f} | Return: {row['return_pct']:.1f}% | "
                    f"Final: ${row['final_capital']:,.0f} | MDD: {row['mdd_pct']:.1f}%\n")
            f.write(f"  Trades: {row['trade_count']:.0f} | WR: {row['win_rate']:.1f}% | "
                    f"R:R: {row['rr_ratio']:.3f} | W/L: {row['wins']:.0f}/{row['losses']:.0f}\n")
            f.write(f"  Fast: {row['fast_ma_type']}({row['fast_ma_len']:.0f}) | "
                    f"Slow: {row['slow_ma_type']}({row['slow_ma_len']:.0f}) | "
                    f"TF: {row['timeframe']}\n")
            f.write(f"  ADX: {row['adx_period']:.0f}/{row['adx_threshold']:.0f} | "
                    f"RSI: {row['rsi_lower']:.0f}-{row['rsi_upper']:.0f}\n")
            f.write(f"  ATR SL: {row['atr_sl_mult']:.1f}x | TP1: {row['atr_tp1_mult']:.1f}x | "
                    f"Trail: {row['atr_trail_mult']:.1f}x @ {row['trail_activation']:.0%}\n")
            f.write(f"  Leverage: {row['leverage']:.0f}x | Size: {row['position_ratio']:.0%} | "
                    f"Gap: {row['min_gap_bars']:.0f} bars\n\n")

    print(f"[SAVED] Summary -> {summary_path}")
    total_elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  TOTAL TIME: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
