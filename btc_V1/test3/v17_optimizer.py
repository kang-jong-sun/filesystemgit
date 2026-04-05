"""
v17 BTC/USDT Futures Backtest - Massive Parameter Optimization (Numba-JIT)
==========================================================================
Random sampling of 120,000 parameter combinations.
Correct Wilder's ADX smoothing. Pre-computed indicators.
Numba JIT for all hot loops.
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
from numba import njit, prange
import numba

print(f"Numba version: {numba.__version__}")
print("Compiling JIT functions (first run takes ~30s)...\n")

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
NUM_SAMPLES = 120000
SEED = 42

# ============================================================
# NUMBA JIT INDICATOR FUNCTIONS
# ============================================================

@njit(cache=True)
def nb_ema(data, period):
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    alpha = 2.0 / (period + 1)
    # Find first valid
    start = -1
    for i in range(n):
        if not np.isnan(data[i]):
            start = i
            break
    if start < 0 or (start + period) > n:
        return result
    # Seed with SMA
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
def nb_rsi(close, period):
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    if n < period + 1:
        return result
    # Calculate deltas
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
# NUMBA JIT BACKTEST ENGINE
# ============================================================

@njit(cache=True)
def nb_backtest(close, high, low, fast_ma, slow_ma, adx, rsi,
                adx_thresh, rsi_lo, rsi_hi,
                sl_pct, trail_act, trail_width,
                delay, margin_pct, leverage,
                initial_capital, fee_rate):
    """
    Run a single backtest entirely in numba.
    Returns: (trades, wins, losses, gross_profit, gross_loss,
              final_balance, max_drawdown, sl_hits, liquidated)
    """
    n = len(close)

    # Pre-compute cross signals
    cross_up_raw = np.zeros(n, dtype=numba.boolean)
    cross_down_raw = np.zeros(n, dtype=numba.boolean)

    for i in range(1, n):
        if (not np.isnan(fast_ma[i]) and not np.isnan(fast_ma[i-1]) and
            not np.isnan(slow_ma[i]) and not np.isnan(slow_ma[i-1])):
            if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
                cross_up_raw[i] = True
            if fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
                cross_down_raw[i] = True

    # Apply delay
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

    # Pre-compute valid conditions
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

    # Start after indicators warm up
    # We'll find the first bar where all indicators are valid
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
            # Position management
            if position == 1:
                worst_roi = (bl - entry_price) / entry_price * leverage
                best_roi = (bh - entry_price) / entry_price * leverage
                cur_roi = (price - entry_price) / entry_price * leverage
            else:
                worst_roi = (entry_price - bh) / entry_price * leverage
                best_roi = (entry_price - bl) / entry_price * leverage
                cur_roi = (entry_price - price) / entry_price * leverage

            # Update trailing
            if t_active and best_roi > t_peak:
                t_peak = best_roi
                t_sl = t_peak + trail_width

            close_pos = False
            exit_price = price
            reason = 0  # 1=SL, 2=TSL, 3=LIQ, 4=REV

            # Fixed SL
            if worst_roi <= sl_pct:
                close_pos = True
                reason = 1
                if position == 1:
                    exit_price = entry_price * (1.0 + sl_pct / leverage)
                else:
                    exit_price = entry_price * (1.0 - sl_pct / leverage)
                sl_hits += 1

            # Trailing stop
            if not close_pos and t_active and worst_roi <= t_sl:
                close_pos = True
                reason = 2
                if position == 1:
                    exit_price = entry_price * (1.0 + t_sl / leverage)
                else:
                    exit_price = entry_price * (1.0 - t_sl / leverage)

            # Liquidation check
            if not close_pos and worst_roi <= -0.9:
                close_pos = True
                reason = 3
                liquidated = True
                if position == 1:
                    exit_price = bl
                else:
                    exit_price = bh

            # Activate trailing
            if not t_active and best_roi >= trail_act:
                t_active = True
                t_peak = best_roi
                t_sl = t_peak + trail_width

            # Reverse signal
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

                # Reverse entry
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

        # Entry signals
        if position == 0 and capital > 100:
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

    # Close remaining position
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
FAST_MA_TYPES = ['WMA', 'HMA', 'EMA']
FAST_MA_LENS = [2, 3, 4, 5, 7, 10, 14, 21]
SLOW_MA_TYPES = ['EMA', 'SMA']
SLOW_MA_LENS = [100, 150, 200, 250, 300]
ADX_PERIODS = [14, 20]

def calc_fast_ma(close, ma_type, period):
    if ma_type == 'EMA':
        return nb_ema(close, period)
    elif ma_type == 'WMA':
        return nb_wma(close, period)
    elif ma_type == 'HMA':
        return nb_hma(close, period)

def calc_slow_ma(close, ma_type, period):
    if ma_type == 'EMA':
        return nb_ema(close, period)
    elif ma_type == 'SMA':
        return nb_sma(close, period)

def precompute_indicators(ohlcv_df):
    close = ohlcv_df['close'].values.astype(np.float64)
    high = ohlcv_df['high'].values.astype(np.float64)
    low = ohlcv_df['low'].values.astype(np.float64)

    cache = {
        'close': close,
        'high': high,
        'low': low,
        'fast_ma': {},
        'slow_ma': {},
        'adx': {},
        'rsi': nb_rsi(close, 14),
    }

    for ma_type in FAST_MA_TYPES:
        for period in FAST_MA_LENS:
            cache['fast_ma'][(ma_type, period)] = calc_fast_ma(close, ma_type, period)

    for ma_type in SLOW_MA_TYPES:
        for period in SLOW_MA_LENS:
            cache['slow_ma'][(ma_type, period)] = calc_slow_ma(close, ma_type, period)

    for p in ADX_PERIODS:
        adx, pdi, mdi = nb_adx(high, low, close, p)
        cache['adx'][p] = adx

    return cache

# ============================================================
# PARAMETER GENERATION
# ============================================================
PARAM_SPACE = {
    'fast_ma_type': ['WMA', 'HMA', 'EMA'],
    'fast_ma_len': [2, 3, 4, 5, 7, 10, 14, 21],
    'slow_ma_type': ['EMA', 'SMA'],
    'slow_ma_len': [100, 150, 200, 250, 300],
    'timeframe': ['10m', '15m', '30m'],
    'adx_period': [14, 20],
    'adx_threshold': [30, 35, 40, 45],
    'rsi_range': [(25, 65), (30, 65), (30, 70), (35, 65), (35, 70), (40, 75)],
    'sl_pct': [-5, -6, -7, -8],
    'trail_act': [3, 4, 5, 7, 10],
    'trail_width': [-2, -3, -4, -5],
    'entry_delay': [0, 1, 2, 3, 5],
    'margin_pct': [15, 20, 25, 30, 40, 50],
    'leverage': [5, 7, 10],
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
            p[key] = rng.choice(values)
        combos.append(p)
    return combos

# ============================================================
# SCORING
# ============================================================
def calc_score(trades, pf, return_pct, max_dd):
    pf_c = min(pf, 100.0)
    if return_pct <= 0:
        log_ret = -log10(abs(return_pct) + 1) * 5
    else:
        log_ret = log10(return_pct + 1) * 5
    return pf_c * 3.0 + log_ret - max_dd / 5.0 + trades / 100.0

# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()

    total_space = count_total()
    print(f"Total parameter space: {total_space:,} combinations")
    print(f"Random sampling: {NUM_SAMPLES:,} combinations")
    print(f"Initial capital: ${INITIAL_CAPITAL:,.0f}\n")

    # Load data
    df_5m = load_data()

    # Resample
    print("\nResampling...")
    tf_data = {}
    for tf in ['10m', '15m', '30m']:
        tf_data[tf] = resample_ohlcv(df_5m, tf)
        print(f"  {tf}: {len(tf_data[tf]):,} bars")

    del df_5m

    # Warm up numba by calling each function once with tiny arrays
    print("\nWarming up Numba JIT (one-time compilation)...")
    t_jit = time.time()
    _dummy = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
    _ = nb_ema(_dummy, 3)
    _ = nb_sma(_dummy, 3)
    _ = nb_wma(_dummy, 3)
    _ = nb_hma(_dummy, 3)
    _ = nb_rsi(_dummy, 5)
    _ = wilder_smooth(_dummy, 3)
    _ = nb_adx(_dummy, _dummy * 0.99, _dummy * 0.98, 5)
    _ = nb_backtest(_dummy, _dummy, _dummy * 0.99, _dummy, _dummy,
                    _dummy, _dummy, 30.0, 25.0, 65.0,
                    -0.05, 0.03, -0.02, 0, 0.2, 10.0, 3000.0, 0.0004)
    print(f"JIT compilation done in {time.time()-t_jit:.1f}s\n")

    # Pre-compute indicators
    print("Pre-computing indicators for all timeframes...")
    tf_caches = {}
    for tf in ['10m', '15m', '30m']:
        t0 = time.time()
        tf_caches[tf] = precompute_indicators(tf_data[tf])
        dt = time.time() - t0
        n_fast = len(tf_caches[tf]['fast_ma'])
        n_slow = len(tf_caches[tf]['slow_ma'])
        print(f"  {tf}: {dt:.1f}s ({n_fast} fast MAs, {n_slow} slow MAs, {len(tf_caches[tf]['adx'])} ADX)")

    del tf_data

    # Generate samples
    print(f"\nGenerating {NUM_SAMPLES:,} random parameter samples...")
    rng = random.Random(SEED)
    param_list = sample_params(rng, NUM_SAMPLES)

    # Run backtests
    print(f"\nRunning {NUM_SAMPLES:,} backtests...\n")
    sys.stdout.flush()

    results = []
    t_bt = time.time()
    completed = 0
    filtered = 0

    for idx, params in enumerate(param_list):
        tf = params['timeframe']
        cache = tf_caches[tf]

        fast_ma = cache['fast_ma'][(params['fast_ma_type'], params['fast_ma_len'])]
        slow_ma = cache['slow_ma'][(params['slow_ma_type'], params['slow_ma_len'])]
        adx_arr = cache['adx'][params['adx_period']]
        rsi_arr = cache['rsi']
        close = cache['close']
        high_arr = cache['high']
        low_arr = cache['low']

        rsi_lo, rsi_hi = params['rsi_range']

        res = nb_backtest(
            close, high_arr, low_arr, fast_ma, slow_ma, adx_arr, rsi_arr,
            float(params['adx_threshold']),
            float(rsi_lo), float(rsi_hi),
            params['sl_pct'] / 100.0,
            params['trail_act'] / 100.0,
            params['trail_width'] / 100.0,
            params['entry_delay'],
            params['margin_pct'] / 100.0,
            float(params['leverage']),
            INITIAL_CAPITAL, FEE_RATE
        )

        trades, wins, losses, gp, gl, final_bal, max_dd, sl_h, liq = res
        completed += 1

        if trades >= 20 and not liq and gl > 0:
            pf = gp / gl
            if pf > 1.0:
                ret_pct = (final_bal - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
                wr = wins / trades * 100 if trades > 0 else 0
                score = calc_score(trades, pf, ret_pct, max_dd)

                results.append({
                    'score': score,
                    'pf': pf,
                    'return_pct': ret_pct,
                    'final_balance': final_bal,
                    'max_drawdown': max_dd,
                    'trades': trades,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': wr,
                    'sl_hits': sl_h,
                    'gross_profit': gp,
                    'gross_loss': gl,
                    'fast_ma_type': params['fast_ma_type'],
                    'fast_ma_len': params['fast_ma_len'],
                    'slow_ma_type': params['slow_ma_type'],
                    'slow_ma_len': params['slow_ma_len'],
                    'timeframe': params['timeframe'],
                    'adx_period': params['adx_period'],
                    'adx_threshold': params['adx_threshold'],
                    'rsi_range': f"{rsi_lo}-{rsi_hi}",
                    'sl_pct': params['sl_pct'],
                    'trail_act': params['trail_act'],
                    'trail_width': params['trail_width'],
                    'entry_delay': params['entry_delay'],
                    'margin_pct': params['margin_pct'],
                    'leverage': params['leverage'],
                })
                filtered += 1

        # Progress
        if completed % 10000 == 0:
            elapsed = time.time() - t_bt
            speed = completed / elapsed
            eta = (NUM_SAMPLES - completed) / speed if speed > 0 else 0
            top5 = sorted(results, key=lambda x: x['score'], reverse=True)[:5]

            print(f"--- {completed:,}/{NUM_SAMPLES:,} ({completed/NUM_SAMPLES*100:.1f}%) | "
                  f"{speed:.0f}/s | ETA {eta/60:.1f}min | Filtered: {filtered:,} ---")
            for rank, r in enumerate(top5, 1):
                print(f"  #{rank} Sc={r['score']:.1f} PF={r['pf']:.1f} Ret={r['return_pct']:.0f}% "
                      f"${r['final_balance']:,.0f} MDD={r['max_drawdown']:.0f}% T={r['trades']} "
                      f"WR={r['win_rate']:.0f}% | "
                      f"{r['fast_ma_type']}{r['fast_ma_len']}/{r['slow_ma_type']}{r['slow_ma_len']} "
                      f"{r['timeframe']} ADX{r['adx_period']}>={r['adx_threshold']} "
                      f"D{r['entry_delay']} TSL+{r['trail_act']}/{r['trail_width']}% "
                      f"SL{r['sl_pct']}% M{r['margin_pct']}% L{r['leverage']}x")
            print()
            sys.stdout.flush()

    total_bt_time = time.time() - t_bt
    print(f"\n{'='*90}")
    print(f"DONE: {completed:,} backtests in {total_bt_time:.1f}s ({completed/total_bt_time:.0f}/s)")
    print(f"Passed filter: {filtered:,} / {completed:,} ({filtered/completed*100:.1f}%)")
    print(f"{'='*90}\n")

    # Sort
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)

    # Save top 100 CSV
    top100 = results_sorted[:100]
    if top100:
        csv_path = os.path.join(DATA_DIR, "v17_optimization_top100.csv")
        df_out = pd.DataFrame(top100)
        col_order = ['score', 'pf', 'return_pct', 'final_balance', 'max_drawdown',
                      'trades', 'wins', 'losses', 'win_rate', 'sl_hits',
                      'gross_profit', 'gross_loss',
                      'fast_ma_type', 'fast_ma_len', 'slow_ma_type', 'slow_ma_len',
                      'timeframe', 'adx_period', 'adx_threshold', 'rsi_range',
                      'sl_pct', 'trail_act', 'trail_width', 'entry_delay',
                      'margin_pct', 'leverage']
        cols = [c for c in col_order if c in df_out.columns]
        df_out = df_out[cols]
        df_out.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"Saved: {csv_path}")

    # Save summary
    summary_path = os.path.join(DATA_DIR, "v17_optimization_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("v17 BTC/USDT Futures Backtest - Parameter Optimization Summary\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Total combinations tested: {completed:,}\n")
        f.write(f"Passed filter (trades>=20, PF>1, no liquidation): {filtered:,}\n")
        f.write(f"Backtest runtime: {total_bt_time:.1f}s ({completed/total_bt_time:.0f}/s)\n")
        f.write(f"Total elapsed: {time.time()-t_start:.1f}s\n")
        f.write(f"Initial capital: ${INITIAL_CAPITAL:,.0f}\n\n")

        f.write("=" * 100 + "\n")
        f.write("TOP 20 RESULTS BY SCORE\n")
        f.write("=" * 100 + "\n\n")

        for rank, r in enumerate(results_sorted[:20], 1):
            f.write(f"--- #{rank} (Score: {r['score']:.2f}) ---\n")
            f.write(f"  Strategy: {r['fast_ma_type']}({r['fast_ma_len']}) / {r['slow_ma_type']}({r['slow_ma_len']}) on {r['timeframe']}\n")
            f.write(f"  Filters:  ADX({r['adx_period']}) >= {r['adx_threshold']}, RSI {r['rsi_range']}\n")
            f.write(f"  Entry delay: {r['entry_delay']} candles\n")
            f.write(f"  Risk:     SL {r['sl_pct']}%, Trail +{r['trail_act']}% / {r['trail_width']}%\n")
            f.write(f"  Sizing:   Margin {r['margin_pct']}%, Leverage {r['leverage']}x\n")
            f.write(f"  Results:  PF={r['pf']:.2f}, Return={r['return_pct']:.1f}%, Final=${r['final_balance']:,.0f}\n")
            f.write(f"            Trades={r['trades']}, Win={r['wins']}, Loss={r['losses']}, WR={r['win_rate']:.1f}%\n")
            f.write(f"            MDD={r['max_drawdown']:.1f}%, SL hits={r['sl_hits']}\n\n")

        # Parameter frequency for top 100
        if len(results_sorted) >= 10:
            f.write("=" * 100 + "\n")
            f.write("PARAMETER FREQUENCY ANALYSIS (Top 100)\n")
            f.write("=" * 100 + "\n\n")
            top_set = results_sorted[:min(100, len(results_sorted))]
            for pname in ['fast_ma_type', 'fast_ma_len', 'slow_ma_type', 'slow_ma_len',
                          'timeframe', 'adx_period', 'adx_threshold', 'rsi_range',
                          'sl_pct', 'trail_act', 'trail_width', 'entry_delay',
                          'margin_pct', 'leverage']:
                counts = defaultdict(int)
                for r in top_set:
                    counts[r[pname]] += 1
                sc = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                f.write(f"  {pname}:\n")
                for val, cnt in sc:
                    f.write(f"    {val}: {cnt} ({cnt/len(top_set)*100:.0f}%)\n")
                f.write("\n")

        # TF breakdown
        f.write("=" * 100 + "\n")
        f.write("TIMEFRAME BREAKDOWN\n")
        f.write("=" * 100 + "\n\n")
        for tf in ['10m', '15m', '30m']:
            tf_r = [r for r in results_sorted if r['timeframe'] == tf]
            if tf_r:
                avg_pf = np.mean([r['pf'] for r in tf_r])
                avg_ret = np.mean([r['return_pct'] for r in tf_r])
                avg_t = np.mean([r['trades'] for r in tf_r])
                mx_pf = max(r['pf'] for r in tf_r)
                f.write(f"  {tf}: {len(tf_r)} configs | avg PF={avg_pf:.2f} max PF={mx_pf:.2f} | "
                        f"avg Ret={avg_ret:.1f}% | avg Trades={avg_t:.0f}\n")
        f.write("\n")

        # Best by PF (min 30 trades)
        f.write("=" * 100 + "\n")
        f.write("BEST BY PROFIT FACTOR (min 30 trades)\n")
        f.write("=" * 100 + "\n\n")
        pf_s = sorted([r for r in results_sorted if r['trades'] >= 30],
                       key=lambda x: x['pf'], reverse=True)
        for rank, r in enumerate(pf_s[:10], 1):
            f.write(f"  #{rank}: PF={r['pf']:.2f} | {r['fast_ma_type']}({r['fast_ma_len']})/{r['slow_ma_type']}({r['slow_ma_len']}) "
                    f"{r['timeframe']} ADX{r['adx_period']}>={r['adx_threshold']} D{r['entry_delay']} "
                    f"T={r['trades']} WR={r['win_rate']:.1f}% Ret={r['return_pct']:.1f}% MDD={r['max_drawdown']:.1f}%\n")

        # Most trades (PF > 2)
        f.write("\n" + "=" * 100 + "\n")
        f.write("MOST TRADES (PF > 2)\n")
        f.write("=" * 100 + "\n\n")
        tr_s = sorted([r for r in results_sorted if r['pf'] > 2],
                       key=lambda x: x['trades'], reverse=True)
        for rank, r in enumerate(tr_s[:10], 1):
            f.write(f"  #{rank}: Trades={r['trades']} PF={r['pf']:.2f} | "
                    f"{r['fast_ma_type']}({r['fast_ma_len']})/{r['slow_ma_type']}({r['slow_ma_len']}) "
                    f"{r['timeframe']} ADX{r['adx_period']}>={r['adx_threshold']} D{r['entry_delay']} "
                    f"WR={r['win_rate']:.1f}% Ret={r['return_pct']:.1f}% MDD={r['max_drawdown']:.1f}%\n")

        # Best return (PF > 2)
        f.write("\n" + "=" * 100 + "\n")
        f.write("BEST RETURN (PF > 2)\n")
        f.write("=" * 100 + "\n\n")
        ret_s = sorted([r for r in results_sorted if r['pf'] > 2],
                        key=lambda x: x['return_pct'], reverse=True)
        for rank, r in enumerate(ret_s[:10], 1):
            f.write(f"  #{rank}: Ret={r['return_pct']:.1f}% (${r['final_balance']:,.0f}) PF={r['pf']:.2f} | "
                    f"{r['fast_ma_type']}({r['fast_ma_len']})/{r['slow_ma_type']}({r['slow_ma_len']}) "
                    f"{r['timeframe']} ADX{r['adx_period']}>={r['adx_threshold']} D{r['entry_delay']} "
                    f"T={r['trades']} WR={r['win_rate']:.1f}% MDD={r['max_drawdown']:.1f}%\n")

    print(f"Saved: {summary_path}")

    # Print top 5
    print(f"\n{'='*90}")
    print(f"{'TOP 5 RESULTS':^90}")
    print(f"{'='*90}\n")

    for rank, r in enumerate(results_sorted[:5], 1):
        print(f"#{rank} Score={r['score']:.2f}")
        print(f"   Strategy: {r['fast_ma_type']}({r['fast_ma_len']}) / {r['slow_ma_type']}({r['slow_ma_len']}) on {r['timeframe']}")
        print(f"   Filters:  ADX({r['adx_period']}) >= {r['adx_threshold']}, RSI {r['rsi_range']}")
        print(f"   Entry delay: {r['entry_delay']} candles")
        print(f"   Risk:     SL {r['sl_pct']}%, Trail +{r['trail_act']}% / {r['trail_width']}%")
        print(f"   Sizing:   Margin {r['margin_pct']}%, Leverage {r['leverage']}x")
        print(f"   PF={r['pf']:.2f}  Return={r['return_pct']:.1f}%  Final=${r['final_balance']:,.0f}")
        print(f"   Trades={r['trades']}  Win={r['wins']}  Loss={r['losses']}  WR={r['win_rate']:.1f}%")
        print(f"   MDD={r['max_drawdown']:.1f}%  SL hits={r['sl_hits']}")
        print()

    print(f"Total elapsed: {time.time()-t_start:.1f}s")
    print("DONE.")

if __name__ == '__main__':
    main()
