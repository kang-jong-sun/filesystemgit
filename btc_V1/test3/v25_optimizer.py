"""
v25 BTC/USDT Futures Backtest - Parameter Optimization (Numba-JIT)
===================================================================
150,000 random parameter combinations.
Incorporates ALL discoveries from prior versions:
  - v16.4: Wilder's ADX smoothing (manual, NOT ewm)
  - v16.6: 5-candle entry delay -> PF 89.85
  - v16.5: HMA(21) on 10m -> PF 25.66
  - v23: HMA dominates top 100 (47%), Trail -2% (56%), ADX(14) (74%)
  - Trailing on 30m CLOSE (not intrabar high/low)
  - REVERSE signal: opposite cross with filters -> close + flip (replaces SL)
  - Same-direction re-entry SKIP
  - DEMA added to fast MA types
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
from numba import njit
import numba

print(f"Numba version: {numba.__version__}")
print(f"Python version: {sys.version}")
print("Compiling JIT functions (first run takes ~30-60s)...\n")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = r"D:\filesystem\futures\btc_V1\test3"
CSV_FILES = [
    os.path.join(DATA_DIR, "btc_usdt_5m_2020_to_now_part1.csv"),
    os.path.join(DATA_DIR, "btc_usdt_5m_2020_to_now_part2.csv"),
    os.path.join(DATA_DIR, "btc_usdt_5m_2020_to_now_part3.csv"),
]
INITIAL_CAPITAL = 3000.0
FEE_RATE = 0.0004  # 0.04% per side
NUM_SAMPLES = 150000
SEED = 42
PROGRESS_INTERVAL = 15000

# ============================================================
# NUMBA JIT INDICATOR FUNCTIONS
# ============================================================

@njit(cache=True)
def nb_ema(data, period):
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    alpha = 2.0 / (period + 1)
    start = -1
    for i in range(n):
        if not np.isnan(data[i]):
            start = i
            break
    if start < 0 or (start + period) > n:
        return result
    s = 0.0
    for j in range(start, start + period):
        s += data[j]
    result[start + period - 1] = s / period
    for i in range(start + period, n):
        if not np.isnan(data[i]):
            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1]
        else:
            result[i] = result[i - 1]
    return result


@njit(cache=True)
def nb_sma(data, period):
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    for i in range(period - 1, n):
        s = 0.0
        valid = True
        for j in range(i - period + 1, i + 1):
            if np.isnan(data[j]):
                valid = False
                break
            s += data[j]
        if valid:
            result[i] = s / period
    return result


@njit(cache=True)
def nb_wma(data, period):
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    w_sum = period * (period + 1) / 2.0
    for i in range(period - 1, n):
        s = 0.0
        valid = True
        for j in range(period):
            idx = i - period + 1 + j
            if np.isnan(data[idx]):
                valid = False
                break
            s += data[idx] * (j + 1)
        if valid:
            result[i] = s / w_sum
    return result


@njit(cache=True)
def nb_hma(data, period):
    half_p = max(period // 2, 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    wma_half = nb_wma(data, half_p)
    wma_full = nb_wma(data, period)
    n = len(data)
    diff = np.empty(n, dtype=np.float64)
    diff[:] = np.nan
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    return nb_wma(diff, sqrt_p)


@njit(cache=True)
def nb_dema(data, period):
    """Double EMA: 2*EMA(data, period) - EMA(EMA(data, period), period)"""
    ema1 = nb_ema(data, period)
    ema2 = nb_ema(ema1, period)
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    for i in range(n):
        if not np.isnan(ema1[i]) and not np.isnan(ema2[i]):
            result[i] = 2.0 * ema1[i] - ema2[i]
    return result


@njit(cache=True)
def nb_rsi(close, period):
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    if n < period + 1:
        return result
    gain_sum = 0.0
    loss_sum = 0.0
    for i in range(1, period + 1):
        d = close[i] - close[i - 1]
        if d > 0:
            gain_sum += d
        else:
            loss_sum -= d
    avg_gain = gain_sum / period
    avg_loss = loss_sum / period
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period + 1, n):
        d = close[i] - close[i - 1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - 100.0 / (1.0 + rs)
    return result


@njit(cache=True)
def wilder_smooth(values, period):
    """Wilder's smoothing - NOT ewm, manual formula per v16.4"""
    n = len(values)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    start = -1
    for i in range(n):
        if not np.isnan(values[i]):
            start = i
            break
    if start < 0 or (start + period) > n:
        return result
    s = 0.0
    for j in range(start, start + period):
        s += values[j]
    result[start + period - 1] = s / period
    for i in range(start + period, n):
        if not np.isnan(values[i]) and not np.isnan(result[i - 1]):
            result[i] = (result[i - 1] * (period - 1) + values[i]) / period
        elif not np.isnan(result[i - 1]):
            result[i] = result[i - 1]
    return result


@njit(cache=True)
def nb_adx(high, low, close, period):
    """ADX with Wilder's smoothing (v16.4 fix)"""
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    plus_dm = np.empty(n, dtype=np.float64)
    minus_dm = np.empty(n, dtype=np.float64)

    tr[0] = high[0] - low[0]
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    for i in range(1, n):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hpc, lpc))
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        else:
            plus_dm[i] = 0.0
        if down > up and down > 0:
            minus_dm[i] = down
        else:
            minus_dm[i] = 0.0

    atr = wilder_smooth(tr, period)
    s_pdm = wilder_smooth(plus_dm, period)
    s_mdm = wilder_smooth(minus_dm, period)

    pdi = np.empty(n, dtype=np.float64)
    mdi = np.empty(n, dtype=np.float64)
    dx = np.empty(n, dtype=np.float64)
    pdi[:] = np.nan
    mdi[:] = np.nan
    dx[:] = np.nan

    for i in range(n):
        if not np.isnan(atr[i]) and atr[i] > 0:
            pdi[i] = 100.0 * s_pdm[i] / atr[i]
            mdi[i] = 100.0 * s_mdm[i] / atr[i]
            di_sum = pdi[i] + mdi[i]
            if di_sum > 0:
                dx[i] = 100.0 * abs(pdi[i] - mdi[i]) / di_sum

    adx = wilder_smooth(dx, period)
    return adx, pdi, mdi


# ============================================================
# NUMBA JIT BACKTEST ENGINE - v25
# ============================================================
# Key differences from v17:
#   - Trailing stop checked on CLOSE price only (not intrabar high/low)
#   - Same-direction re-entry SKIP
#   - SL checked on intrabar high/low
#   - REVERSE: opposite cross with ADX+RSI -> close + flip (after delay)

@njit(cache=True)
def nb_backtest_v25(close, high, low, fast_ma, slow_ma, adx, rsi,
                    adx_thresh, rsi_lo, rsi_hi,
                    sl_pct, trail_act, trail_width,
                    delay, margin_pct, leverage,
                    initial_capital, fee_rate):
    """
    v25 backtest engine.
    Returns: (trades, wins, losses, gross_profit, gross_loss,
              final_balance, max_drawdown, sl_hits, liquidated)
    """
    n = len(close)

    # Pre-compute raw cross signals
    cross_up_raw = np.zeros(n, dtype=numba.boolean)
    cross_down_raw = np.zeros(n, dtype=numba.boolean)

    for i in range(1, n):
        if (not np.isnan(fast_ma[i]) and not np.isnan(fast_ma[i-1]) and
            not np.isnan(slow_ma[i]) and not np.isnan(slow_ma[i-1])):
            if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
                cross_up_raw[i] = True
            if fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
                cross_down_raw[i] = True

    # Apply entry delay (v16.6 discovery)
    cross_up = np.zeros(n, dtype=numba.boolean)
    cross_down = np.zeros(n, dtype=numba.boolean)
    if delay == 0:
        for i in range(n):
            cross_up[i] = cross_up_raw[i]
            cross_down[i] = cross_down_raw[i]
    else:
        for i in range(delay, n):
            cross_up[i] = cross_up_raw[i - delay]
            cross_down[i] = cross_down_raw[i - delay]

    # Pre-compute valid filter conditions (ADX + RSI)
    valid_long = np.zeros(n, dtype=numba.boolean)
    valid_short = np.zeros(n, dtype=numba.boolean)

    for i in range(n):
        if np.isnan(adx[i]) or np.isnan(rsi[i]):
            continue
        if adx[i] >= adx_thresh:
            if rsi_lo <= rsi[i] <= rsi_hi:
                if not np.isnan(fast_ma[i]) and not np.isnan(slow_ma[i]):
                    if fast_ma[i] > slow_ma[i]:
                        valid_long[i] = True
                    if fast_ma[i] < slow_ma[i]:
                        valid_short[i] = True

    # Backtest loop
    capital = initial_capital
    peak_capital = initial_capital
    max_dd = 0.0

    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    pos_size = 0.0
    pos_margin = 0.0
    t_active = False
    t_peak = 0.0
    t_sl = 0.0

    total_trades = 0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    sl_hits = 0
    liquidated = False

    # Find start bar where all indicators valid
    start_bar = 50
    for i in range(n):
        if (not np.isnan(fast_ma[i]) and not np.isnan(slow_ma[i]) and
            not np.isnan(adx[i]) and not np.isnan(rsi[i])):
            start_bar = max(i, 50)
            break

    for i in range(start_bar, n):
        if liquidated:
            break

        price = close[i]
        bh = high[i]
        bl = low[i]

        if position != 0:
            # --- Position management ---
            if position == 1:
                worst_roi = (bl - entry_price) / entry_price * leverage
                close_roi = (price - entry_price) / entry_price * leverage
            else:
                worst_roi = (entry_price - bh) / entry_price * leverage
                close_roi = (entry_price - price) / entry_price * leverage

            close_pos = False
            exit_price = price
            reason = 0  # 1=SL, 2=TSL, 3=LIQ, 4=REV

            # 1) Fixed SL check on intrabar high/low
            if worst_roi <= sl_pct:
                close_pos = True
                reason = 1
                if position == 1:
                    exit_price = entry_price * (1.0 + sl_pct / leverage)
                else:
                    exit_price = entry_price * (1.0 - sl_pct / leverage)
                sl_hits += 1

            # 2) Liquidation check on intrabar
            if not close_pos and worst_roi <= -0.9:
                close_pos = True
                reason = 3
                liquidated = True
                if position == 1:
                    exit_price = bl
                else:
                    exit_price = bh

            # 3) Trailing stop: activate and check on CLOSE price only
            #    (v23 discovery: trailing on close, not intrabar)
            if not close_pos:
                if not t_active and close_roi >= trail_act:
                    t_active = True
                    t_peak = close_roi
                    t_sl = t_peak + trail_width  # trail_width is negative

                if t_active:
                    if close_roi > t_peak:
                        t_peak = close_roi
                        t_sl = t_peak + trail_width
                    if close_roi <= t_sl:
                        close_pos = True
                        reason = 2
                        exit_price = price  # close at bar close price

            # 4) REVERSE signal: opposite cross with ADX+RSI filter
            #    close existing + open new direction
            if not close_pos:
                if position == 1 and cross_down[i] and valid_short[i]:
                    close_pos = True
                    reason = 4
                    exit_price = price
                elif position == -1 and cross_up[i] and valid_long[i]:
                    close_pos = True
                    reason = 4
                    exit_price = price

            if close_pos:
                if position == 1:
                    pnl_pct = (exit_price - entry_price) / entry_price * leverage
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * leverage

                fee = pos_size * exit_price * fee_rate
                pnl_dollar = pos_margin * pnl_pct - fee
                capital += pnl_dollar
                total_trades += 1

                if pnl_dollar > 0:
                    wins += 1
                    gross_profit += pnl_dollar
                else:
                    losses += 1
                    gross_loss += abs(pnl_dollar)

                if capital > peak_capital:
                    peak_capital = capital
                dd = (peak_capital - capital) / peak_capital * 100.0 if peak_capital > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd

                # Reverse entry on reason=4 (flip direction)
                if reason == 4 and capital > 0:
                    old_pos = position
                    position = -old_pos
                    entry_price = price
                    pos_margin = capital * margin_pct
                    pos_size = pos_margin * leverage / price
                    capital -= pos_size * price * fee_rate
                    t_active = False
                    t_peak = 0.0
                    t_sl = 0.0
                else:
                    position = 0

                if capital <= 0:
                    liquidated = True
                    break
                continue

        # --- Entry signals ---
        if position == 0 and capital > 100:
            # Same-direction re-entry SKIP is inherent:
            # after a position closes (non-reverse), we only enter on new cross signal
            if cross_up[i] and valid_long[i]:
                position = 1
                entry_price = price
                pos_margin = capital * margin_pct
                pos_size = pos_margin * leverage / price
                capital -= pos_size * price * fee_rate
                t_active = False
                t_peak = 0.0
                t_sl = 0.0
            elif cross_down[i] and valid_short[i]:
                position = -1
                entry_price = price
                pos_margin = capital * margin_pct
                pos_size = pos_margin * leverage / price
                capital -= pos_size * price * fee_rate
                t_active = False
                t_peak = 0.0
                t_sl = 0.0

    # Close remaining position at last bar
    if position != 0 and not liquidated and n > 0:
        exit_price = close[n - 1]
        if position == 1:
            pnl_pct = (exit_price - entry_price) / entry_price * leverage
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * leverage
        fee = pos_size * exit_price * fee_rate
        pnl_dollar = pos_margin * pnl_pct - fee
        capital += pnl_dollar
        total_trades += 1
        if pnl_dollar > 0:
            wins += 1
            gross_profit += pnl_dollar
        else:
            losses += 1
            gross_loss += abs(pnl_dollar)

    if capital > peak_capital:
        peak_capital = capital
    dd = (peak_capital - capital) / peak_capital * 100.0 if peak_capital > 0 else 0.0
    if dd > max_dd:
        max_dd = dd

    return (total_trades, wins, losses, gross_profit, gross_loss,
            capital, max_dd, sl_hits, liquidated)


# ============================================================
# DATA LOADING & RESAMPLING
# ============================================================
def load_data():
    print("Loading CSV files...")
    t0 = time.time()
    dfs = []
    for f in CSV_FILES:
        df = pd.read_csv(f, parse_dates=['timestamp'])
        dfs.append(df)
        print(f"  {os.path.basename(f)}: {len(df):,} rows")
    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.set_index('timestamp', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(np.float64)
    print(f"Total: {len(df):,} rows, {df.index[0]} to {df.index[-1]}")
    print(f"Loaded in {time.time()-t0:.1f}s")
    return df


def resample_ohlcv(df_5m, tf_str):
    rule_map = {'10m': '10min', '15m': '15min', '30m': '30min', '1h': '1h'}
    resampled = df_5m.resample(rule_map[tf_str]).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()
    return resampled


# ============================================================
# INDICATOR CACHE
# ============================================================
FAST_MA_TYPES = ['WMA', 'HMA', 'EMA', 'DEMA']
FAST_MA_LENS = [2, 3, 4, 5, 7, 10, 14, 21]
SLOW_MA_TYPES = ['EMA', 'SMA']
SLOW_MA_LENS = [100, 150, 200, 250, 300]
ADX_PERIODS = [14, 20]
TIMEFRAMES = ['10m', '15m', '30m', '1h']


def calc_fast_ma(close, ma_type, period):
    if ma_type == 'EMA':
        return nb_ema(close, period)
    elif ma_type == 'WMA':
        return nb_wma(close, period)
    elif ma_type == 'HMA':
        return nb_hma(close, period)
    elif ma_type == 'DEMA':
        return nb_dema(close, period)
    return nb_ema(close, period)


def calc_slow_ma(close, ma_type, period):
    if ma_type == 'EMA':
        return nb_ema(close, period)
    elif ma_type == 'SMA':
        return nb_sma(close, period)
    return nb_ema(close, period)


def precompute_indicators(ohlcv_df, tf_label):
    close = ohlcv_df['close'].values.astype(np.float64)
    high_arr = ohlcv_df['high'].values.astype(np.float64)
    low_arr = ohlcv_df['low'].values.astype(np.float64)

    cache = {
        'close': close,
        'high': high_arr,
        'low': low_arr,
        'fast_ma': {},
        'slow_ma': {},
        'adx': {},
        'rsi': nb_rsi(close, 14),
    }

    print(f"  [{tf_label}] Computing fast MAs ({len(FAST_MA_TYPES)}x{len(FAST_MA_LENS)})...")
    for ma_type in FAST_MA_TYPES:
        for period in FAST_MA_LENS:
            cache['fast_ma'][(ma_type, period)] = calc_fast_ma(close, ma_type, period)

    print(f"  [{tf_label}] Computing slow MAs ({len(SLOW_MA_TYPES)}x{len(SLOW_MA_LENS)})...")
    for ma_type in SLOW_MA_TYPES:
        for period in SLOW_MA_LENS:
            cache['slow_ma'][(ma_type, period)] = calc_slow_ma(close, ma_type, period)

    print(f"  [{tf_label}] Computing ADX (Wilder's) for periods {ADX_PERIODS}...")
    for p in ADX_PERIODS:
        adx_val, pdi, mdi = nb_adx(high_arr, low_arr, close, p)
        cache['adx'][p] = adx_val

    print(f"  [{tf_label}] Done. Bars: {len(close):,}")
    return cache


# ============================================================
# PARAMETER SPACE (v25 expanded)
# ============================================================
PARAM_SPACE = {
    'fast_ma_type': ['WMA', 'HMA', 'EMA', 'DEMA'],       # 4
    'fast_ma_len': [2, 3, 4, 5, 7, 10, 14, 21],           # 8
    'slow_ma_type': ['EMA', 'SMA'],                         # 2
    'slow_ma_len': [100, 150, 200, 250, 300],               # 5
    'timeframe': ['10m', '15m', '30m', '1h'],               # 4
    'adx_period': [14, 20],                                 # 2
    'adx_threshold': [25, 30, 35, 40, 45, 50],             # 6
    'rsi_range': [(25, 65), (30, 65), (30, 70), (35, 65), (35, 70), (40, 75)],  # 6
    'sl_pct': [-5, -6, -7, -8, -10],                       # 5
    'trail_act': [2, 3, 4, 5, 7, 10],                      # 6
    'trail_width': [-2, -3, -4, -5],                        # 4
    'entry_delay': [0, 1, 2, 3, 5, 8],                     # 6
    'margin_pct': [15, 20, 25, 30, 40, 50],                # 6
    'leverage': [3, 5, 7, 10, 15],                          # 5
}


def count_total():
    total = 1
    for v in PARAM_SPACE.values():
        total *= len(v)
    return total


def sample_params(rng, n):
    combos = []
    for _ in range(n):
        p = {}
        for key, values in PARAM_SPACE.items():
            p[key] = values[rng.randint(0, len(values) - 1)]
        combos.append(p)
    return combos


# ============================================================
# SCORING (v25 formula)
# ============================================================
def calc_score(trades, pf, return_pct, max_dd):
    """
    Score = PF*3 + log10(max(return%,1))*5 - MDD%/5 + min(trades,200)/50
    """
    score = pf * 3.0
    score += log10(max(return_pct, 1.0)) * 5.0
    score -= max_dd / 5.0
    score += min(trades, 200) / 50.0
    return score


# ============================================================
# MAIN OPTIMIZATION
# ============================================================
def main():
    total_space = count_total()
    print(f"Total parameter space: {total_space:,}")
    print(f"Sampling {NUM_SAMPLES:,} random combinations")
    print(f"Seed: {SEED}")
    print()

    # Load data
    df_5m = load_data()
    print()

    # Resample to all timeframes and precompute indicators
    tf_caches = {}
    print("Resampling and computing indicators...")
    for tf in TIMEFRAMES:
        t0 = time.time()
        df_tf = resample_ohlcv(df_5m, tf)
        cache = precompute_indicators(df_tf, tf)
        tf_caches[tf] = cache
        print(f"  [{tf}] Total time: {time.time()-t0:.1f}s")
    print()

    # Warm up Numba JIT with a tiny test run
    print("Warming up Numba JIT...")
    t0 = time.time()
    _cache = tf_caches['10m']
    _fm = list(_cache['fast_ma'].values())[0]
    _sm = list(_cache['slow_ma'].values())[0]
    _adx = list(_cache['adx'].values())[0]
    _rsi = _cache['rsi']
    _ = nb_backtest_v25(
        _cache['close'][:1000], _cache['high'][:1000], _cache['low'][:1000],
        _fm[:1000], _sm[:1000], _adx[:1000], _rsi[:1000],
        30.0, 30.0, 70.0,
        -0.05, 0.03, -0.02,
        0, 0.2, 5.0,
        INITIAL_CAPITAL, FEE_RATE
    )
    print(f"JIT compilation done in {time.time()-t0:.1f}s")
    print()

    # Generate random parameter combinations
    rng = random.Random(SEED)
    print(f"Generating {NUM_SAMPLES:,} parameter combinations...")
    combos = sample_params(rng, NUM_SAMPLES)
    print("Done.")
    print()

    # Run backtests
    results = []
    valid_count = 0
    start_time = time.time()

    print(f"{'='*80}")
    print(f"Starting optimization: {NUM_SAMPLES:,} backtests")
    print(f"{'='*80}")
    print()

    for idx, p in enumerate(combos):
        tf = p['timeframe']
        cache = tf_caches[tf]

        fast_ma_key = (p['fast_ma_type'], p['fast_ma_len'])
        slow_ma_key = (p['slow_ma_type'], p['slow_ma_len'])

        fast_ma = cache['fast_ma'][fast_ma_key]
        slow_ma = cache['slow_ma'][slow_ma_key]
        adx_arr = cache['adx'][p['adx_period']]
        rsi_arr = cache['rsi']
        close_arr = cache['close']
        high_arr = cache['high']
        low_arr = cache['low']

        rsi_lo, rsi_hi = p['rsi_range']
        sl = p['sl_pct'] / 100.0
        t_act = p['trail_act'] / 100.0
        t_width = p['trail_width'] / 100.0
        margin = p['margin_pct'] / 100.0
        lev = float(p['leverage'])
        delay_val = p['entry_delay']

        (trades, w, l, gp, gl, final_bal, mdd, sl_h, liq) = nb_backtest_v25(
            close_arr, high_arr, low_arr,
            fast_ma, slow_ma, adx_arr, rsi_arr,
            float(p['adx_threshold']), float(rsi_lo), float(rsi_hi),
            sl, t_act, t_width,
            delay_val, margin, lev,
            INITIAL_CAPITAL, FEE_RATE
        )

        # Filter: trades >= 15, PF > 1.0, no liquidation
        if trades >= 15 and not liq and gl > 0:
            pf = gp / gl if gl > 0 else 999.0
            ret_pct = (final_bal - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0
            win_rate = w / trades * 100.0 if trades > 0 else 0.0
            score = calc_score(trades, pf, max(ret_pct, 0.01), mdd)

            if pf > 1.0:
                results.append({
                    'score': score,
                    'pf': pf,
                    'return_pct': ret_pct,
                    'mdd': mdd,
                    'trades': trades,
                    'wins': w,
                    'losses': l,
                    'win_rate': win_rate,
                    'sl_hits': sl_h,
                    'gross_profit': gp,
                    'gross_loss': gl,
                    'final_balance': final_bal,
                    'fast_ma_type': p['fast_ma_type'],
                    'fast_ma_len': p['fast_ma_len'],
                    'slow_ma_type': p['slow_ma_type'],
                    'slow_ma_len': p['slow_ma_len'],
                    'timeframe': tf,
                    'adx_period': p['adx_period'],
                    'adx_threshold': p['adx_threshold'],
                    'rsi_lo': rsi_lo,
                    'rsi_hi': rsi_hi,
                    'sl_pct': p['sl_pct'],
                    'trail_act': p['trail_act'],
                    'trail_width': p['trail_width'],
                    'entry_delay': delay_val,
                    'margin_pct': p['margin_pct'],
                    'leverage': p['leverage'],
                })
                valid_count += 1

        # Progress
        done = idx + 1
        if done % PROGRESS_INTERVAL == 0 or done == NUM_SAMPLES:
            elapsed = time.time() - start_time
            speed = done / elapsed if elapsed > 0 else 0
            eta = (NUM_SAMPLES - done) / speed if speed > 0 else 0
            # Current best
            best_pf = 0.0
            best_ret = 0.0
            if results:
                sorted_tmp = sorted(results, key=lambda x: x['score'], reverse=True)
                best_pf = sorted_tmp[0]['pf']
                best_ret = sorted_tmp[0]['return_pct']
            print(f"[{done:>7,}/{NUM_SAMPLES:,}] {elapsed:>7.1f}s | "
                  f"{speed:>6.0f}/s | ETA {eta:>5.0f}s | "
                  f"Valid: {valid_count:,} | "
                  f"Best PF: {best_pf:.2f} | Best Ret: {best_ret:,.0f}%")

    elapsed_total = time.time() - start_time
    print()
    print(f"{'='*80}")
    print(f"Optimization complete: {elapsed_total:.1f}s ({NUM_SAMPLES/elapsed_total:.0f} backtests/s)")
    print(f"Valid results: {valid_count:,} / {NUM_SAMPLES:,}")
    print(f"{'='*80}")
    print()

    if not results:
        print("ERROR: No valid results found!")
        return

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    # ========================================
    # SAVE TOP 100 CSV
    # ========================================
    top100 = results[:100]
    csv_path = os.path.join(DATA_DIR, "v25_opt_top100.csv")
    df_top = pd.DataFrame(top100)
    col_order = ['score', 'pf', 'return_pct', 'mdd', 'trades', 'wins', 'losses',
                 'win_rate', 'sl_hits', 'gross_profit', 'gross_loss', 'final_balance',
                 'fast_ma_type', 'fast_ma_len', 'slow_ma_type', 'slow_ma_len',
                 'timeframe', 'adx_period', 'adx_threshold', 'rsi_lo', 'rsi_hi',
                 'sl_pct', 'trail_act', 'trail_width', 'entry_delay',
                 'margin_pct', 'leverage']
    df_top = df_top[col_order]
    df_top.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved: {csv_path}")

    # ========================================
    # SUMMARY REPORT
    # ========================================
    summary_path = os.path.join(DATA_DIR, "v25_opt_summary.txt")
    lines = []
    lines.append("=" * 90)
    lines.append("v25 OPTIMIZATION SUMMARY")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Samples: {NUM_SAMPLES:,} | Valid: {valid_count:,} | Time: {elapsed_total:.0f}s")
    lines.append(f"Speed: {NUM_SAMPLES/elapsed_total:.0f} backtests/s")
    lines.append(f"Initial capital: ${INITIAL_CAPITAL:,.0f} | Fee: {FEE_RATE*100:.2f}%/side")
    lines.append("=" * 90)
    lines.append("")

    # Top 20 by score
    lines.append("-" * 90)
    lines.append("TOP 20 BY COMPOSITE SCORE")
    lines.append("-" * 90)
    lines.append(f"{'#':>3} {'Score':>7} {'PF':>7} {'Return%':>10} {'MDD%':>7} {'Trades':>6} {'WR%':>6} "
                 f"{'FastMA':>10} {'SlowMA':>10} {'TF':>4} {'ADX':>6} {'SL%':>5} "
                 f"{'TrAct%':>6} {'TrW%':>5} {'Dly':>3} {'Mgn%':>4} {'Lev':>3}")
    for i, r in enumerate(top100[:20]):
        lines.append(
            f"{i+1:>3} {r['score']:>7.2f} {r['pf']:>7.2f} {r['return_pct']:>10,.1f} "
            f"{r['mdd']:>7.1f} {r['trades']:>6} {r['win_rate']:>6.1f} "
            f"{r['fast_ma_type']+'('+str(r['fast_ma_len'])+')':>10} "
            f"{r['slow_ma_type']+'('+str(r['slow_ma_len'])+')':>10} "
            f"{r['timeframe']:>4} {str(r['adx_period'])+'/'+str(r['adx_threshold']):>6} "
            f"{r['sl_pct']:>5} {r['trail_act']:>6} {r['trail_width']:>5} "
            f"{r['entry_delay']:>3} {r['margin_pct']:>4} {r['leverage']:>3}"
        )
    lines.append("")

    # Parameter frequency analysis (top 100)
    lines.append("-" * 90)
    lines.append("PARAMETER FREQUENCY ANALYSIS (Top 100)")
    lines.append("-" * 90)

    freq_params = [
        ('fast_ma_type', 'Fast MA Type'),
        ('fast_ma_len', 'Fast MA Length'),
        ('slow_ma_type', 'Slow MA Type'),
        ('slow_ma_len', 'Slow MA Length'),
        ('timeframe', 'Timeframe'),
        ('adx_period', 'ADX Period'),
        ('adx_threshold', 'ADX Threshold'),
        ('sl_pct', 'Stop Loss %'),
        ('trail_act', 'Trail Activation %'),
        ('trail_width', 'Trail Width %'),
        ('entry_delay', 'Entry Delay'),
        ('margin_pct', 'Margin %'),
        ('leverage', 'Leverage'),
    ]

    for param_key, param_label in freq_params:
        counter = defaultdict(int)
        for r in top100:
            val = r[param_key]
            if isinstance(val, tuple):
                val = str(val)
            counter[val] += 1
        sorted_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        lines.append(f"\n  {param_label}:")
        for val, cnt in sorted_freq:
            pct = cnt / len(top100) * 100
            bar = "#" * int(pct / 2)
            lines.append(f"    {str(val):>10}: {cnt:>3} ({pct:>5.1f}%) {bar}")

    # RSI range frequency
    lines.append(f"\n  RSI Range:")
    rsi_counter = defaultdict(int)
    for r in top100:
        key = f"({r['rsi_lo']},{r['rsi_hi']})"
        rsi_counter[key] += 1
    for val, cnt in sorted(rsi_counter.items(), key=lambda x: x[1], reverse=True):
        pct = cnt / len(top100) * 100
        bar = "#" * int(pct / 2)
        lines.append(f"    {val:>10}: {cnt:>3} ({pct:>5.1f}%) {bar}")

    lines.append("")

    # TF breakdown
    lines.append("-" * 90)
    lines.append("TIMEFRAME BREAKDOWN (all valid results)")
    lines.append("-" * 90)
    tf_stats = defaultdict(list)
    for r in results:
        tf_stats[r['timeframe']].append(r)
    for tf in TIMEFRAMES:
        rs = tf_stats.get(tf, [])
        if rs:
            avg_pf = np.mean([r['pf'] for r in rs])
            avg_ret = np.mean([r['return_pct'] for r in rs])
            avg_mdd = np.mean([r['mdd'] for r in rs])
            top_in_100 = sum(1 for r in top100 if r['timeframe'] == tf)
            lines.append(f"  {tf}: {len(rs):>5} valid | Avg PF {avg_pf:.2f} | "
                         f"Avg Ret {avg_ret:,.0f}% | Avg MDD {avg_mdd:.1f}% | "
                         f"In Top100: {top_in_100}")
        else:
            lines.append(f"  {tf}: 0 valid")
    lines.append("")

    # Best by PF (min 30 trades)
    lines.append("-" * 90)
    lines.append("BEST BY PROFIT FACTOR (min 30 trades)")
    lines.append("-" * 90)
    pf_filtered = [r for r in results if r['trades'] >= 30]
    pf_sorted = sorted(pf_filtered, key=lambda x: x['pf'], reverse=True)
    lines.append(f"{'#':>3} {'PF':>7} {'Return%':>10} {'MDD%':>7} {'Trades':>6} {'WR%':>6} "
                 f"{'FastMA':>10} {'SlowMA':>10} {'TF':>4} {'Dly':>3} {'Lev':>3}")
    for i, r in enumerate(pf_sorted[:10]):
        lines.append(
            f"{i+1:>3} {r['pf']:>7.2f} {r['return_pct']:>10,.1f} "
            f"{r['mdd']:>7.1f} {r['trades']:>6} {r['win_rate']:>6.1f} "
            f"{r['fast_ma_type']+'('+str(r['fast_ma_len'])+')':>10} "
            f"{r['slow_ma_type']+'('+str(r['slow_ma_len'])+')':>10} "
            f"{r['timeframe']:>4} {r['entry_delay']:>3} {r['leverage']:>3}"
        )
    lines.append("")

    # Most trades with PF > 2
    lines.append("-" * 90)
    lines.append("MOST TRADES WITH PF > 2")
    lines.append("-" * 90)
    pf2_filtered = [r for r in results if r['pf'] > 2.0]
    trade_sorted = sorted(pf2_filtered, key=lambda x: x['trades'], reverse=True)
    lines.append(f"{'#':>3} {'Trades':>6} {'PF':>7} {'Return%':>10} {'MDD%':>7} {'WR%':>6} "
                 f"{'FastMA':>10} {'SlowMA':>10} {'TF':>4} {'Dly':>3} {'Lev':>3}")
    for i, r in enumerate(trade_sorted[:10]):
        lines.append(
            f"{i+1:>3} {r['trades']:>6} {r['pf']:>7.2f} {r['return_pct']:>10,.1f} "
            f"{r['mdd']:>7.1f} {r['win_rate']:>6.1f} "
            f"{r['fast_ma_type']+'('+str(r['fast_ma_len'])+')':>10} "
            f"{r['slow_ma_type']+'('+str(r['slow_ma_len'])+')':>10} "
            f"{r['timeframe']:>4} {r['entry_delay']:>3} {r['leverage']:>3}"
        )
    lines.append("")

    # Best return with PF > 2
    lines.append("-" * 90)
    lines.append("BEST RETURN WITH PF > 2")
    lines.append("-" * 90)
    ret_sorted = sorted(pf2_filtered, key=lambda x: x['return_pct'], reverse=True)
    lines.append(f"{'#':>3} {'Return%':>12} {'PF':>7} {'MDD%':>7} {'Trades':>6} {'WR%':>6} "
                 f"{'FastMA':>10} {'SlowMA':>10} {'TF':>4} {'Dly':>3} {'Lev':>3} {'Mgn%':>4}")
    for i, r in enumerate(ret_sorted[:10]):
        lines.append(
            f"{i+1:>3} {r['return_pct']:>12,.1f} {r['pf']:>7.2f} "
            f"{r['mdd']:>7.1f} {r['trades']:>6} {r['win_rate']:>6.1f} "
            f"{r['fast_ma_type']+'('+str(r['fast_ma_len'])+')':>10} "
            f"{r['slow_ma_type']+'('+str(r['slow_ma_len'])+')':>10} "
            f"{r['timeframe']:>4} {r['entry_delay']:>3} {r['leverage']:>3} {r['margin_pct']:>4}"
        )
    lines.append("")
    lines.append("=" * 90)

    summary_text = "\n".join(lines)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"Saved: {summary_path}")
    print()

    # Print summary to console
    print(summary_text)
    print()

    # Print top 5
    print("=" * 80)
    print("TOP 5 RESULTS")
    print("=" * 80)
    for i, r in enumerate(top100[:5]):
        print(f"\n#{i+1} | Score: {r['score']:.2f}")
        print(f"  PF: {r['pf']:.2f} | Return: {r['return_pct']:,.1f}% | MDD: {r['mdd']:.1f}%")
        print(f"  Trades: {r['trades']} | Win Rate: {r['win_rate']:.1f}% | SL Hits: {r['sl_hits']}")
        print(f"  Fast: {r['fast_ma_type']}({r['fast_ma_len']}) | "
              f"Slow: {r['slow_ma_type']}({r['slow_ma_len']}) | TF: {r['timeframe']}")
        print(f"  ADX: {r['adx_period']}/{r['adx_threshold']} | "
              f"RSI: ({r['rsi_lo']},{r['rsi_hi']})")
        print(f"  SL: {r['sl_pct']}% | Trail: {r['trail_act']}%/{r['trail_width']}% | "
              f"Delay: {r['entry_delay']}")
        print(f"  Margin: {r['margin_pct']}% | Leverage: {r['leverage']}x")
        print(f"  Final Balance: ${r['final_balance']:,.2f} (from ${INITIAL_CAPITAL:,.0f})")

    print()
    print("Optimization complete.")


if __name__ == "__main__":
    main()
