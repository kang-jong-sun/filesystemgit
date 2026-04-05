#!/usr/bin/env python3
"""
6-Engine Cross-Verification Backtest for Top 15 Strategies
===========================================================
Engines differ ONLY in ADX/RSI calculation method.
MA calculations, SL/Trail/REV logic, position sizing all identical.

Engine 1: Wilder Manual     - manual Wilder smoothing (standard)
Engine 2: Pandas EWM alpha  - ewm(alpha=1/period, adjust=False)
Engine 3: Pandas EWM span   - ewm(span=period, adjust=False)
Engine 4: SMA-seed Wilder   - seed=SMA(first N), then Wilder
Engine 5: No ADX filter     - remove ADX, keep MA cross + RSI
Engine 6: No RSI filter     - remove RSI, keep MA cross + ADX
"""

import numpy as np
import pandas as pd
import time
import os
import sys
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS
# ============================================================
INITIAL_CAPITAL = 5000.0
FEE_RATE = 0.0004  # 0.04%
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Top 15 strategies
# (name, fast_type, fast_len, slow_type, slow_len, tf, adx_th, rsi_lo, rsi_hi, delay, sl, trail_act, trail_w, margin, lev)
STRATEGIES = [
    ("v32.1F_A", "EMA", 100, "EMA", 600, "30m", 30, 35, 75, 0, 0.03, 0.12, 0.09, 0.35, 10),
    ("v32.1F_B", "EMA", 75, "SMA", 750, "30m", 30, 35, 75, 0, 0.03, 0.12, 0.09, 0.35, 10),
    ("v22.7",    "EMA", 7,  "EMA", 250, "15m", 45, 35, 75, 0, 0.08, 0.07, 0.05, 0.50, 15),
    ("v22.4",    "EMA", 7,  "EMA", 250, "15m", 45, 35, 75, 0, 0.08, 0.07, 0.05, 0.40, 15),
    ("v22.0F",   "WMA", 3,  "EMA", 200, "30m", 35, 35, 65, 5, 0.08, 0.03, 0.02, 0.50, 10),
    ("v16.6",    "WMA", 3,  "EMA", 200, "30m", 35, 30, 70, 5, 0.08, 0.03, 0.02, 0.50, 10),
    ("v16.4",    "WMA", 3,  "EMA", 200, "30m", 35, 35, 65, 0, 0.08, 0.04, 0.03, 0.30, 10),
    ("v22.8",    "EMA", 100,"EMA", 600, "30m", 30, 35, 75, 0, 0.08, 0.06, 0.05, 0.35, 10),
    ("v14.4",    "EMA", 3,  "EMA", 200, "30m", 35, 30, 65, 0, 0.07, 0.06, 0.03, 0.25, 10),
    ("v25.1A",   "HMA", 21, "EMA", 250, "10m", 35, 40, 75, 0, 0.06, 0.07, 0.03, 0.50, 10),
    ("v28_T1",   "WMA", 5,  "EMA", 300, "15m", 35, 35, 75, 2, 0.05, 0.04, 0.02, 0.15, 15),
    ("v28_T2",   "HMA", 14, "EMA", 300, "15m", 35, 35, 70, 3, 0.07, 0.06, 0.03, 0.50, 10),
    ("v24.2",    "EMA", 3,  "EMA", 100, "30m", 30, 30, 70, 0, 0.08, 0.06, 0.05, 0.70, 10),
    ("v15.4",    "EMA", 3,  "EMA", 200, "30m", 35, 30, 65, 0, 0.07, 0.06, 0.03, 0.40, 10),
    ("v25.2C",   "EMA", 5,  "EMA", 300, "10m", 35, 40, 75, 0, 0.07, 0.08, 0.03, 0.40, 10),
]

ENGINE_NAMES = [
    "Wilder Manual",
    "Pandas EWM alpha",
    "Pandas EWM span",
    "SMA-seed Wilder",
    "No ADX filter",
    "No RSI filter",
]


# ============================================================
# DATA LOADING
# ============================================================
def load_5m_data():
    """Load and concatenate all 3 CSV parts."""
    parts = []
    for i in range(1, 4):
        fp = os.path.join(DATA_DIR, f"btc_usdt_5m_2020_to_now_part{i}.csv")
        df = pd.read_csv(fp, parse_dates=['timestamp'])
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    return df


def resample_ohlcv(df_5m, tf):
    """Resample 5m data to target timeframe."""
    rule_map = {'5m': '5min', '10m': '10min', '15m': '15min', '30m': '30min', '1h': '60min'}
    rule = rule_map.get(tf, tf)
    ohlcv = df_5m.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    return ohlcv


# ============================================================
# INDICATOR CALCULATIONS
# ============================================================
def calc_ema(series, period):
    """Standard EMA using pandas ewm."""
    return series.ewm(span=period, adjust=False).mean()


def calc_sma(series, period):
    return series.rolling(period).mean()


def calc_wma(series, period):
    weights = np.arange(1, period + 1, dtype=float)
    def _wma(x):
        return np.dot(x, weights) / weights.sum()
    return series.rolling(period).apply(_wma, raw=True)


def calc_hma(series, period):
    """HMA(n) = WMA(sqrt(n), 2*WMA(n/2) - WMA(n))"""
    half_len = max(int(period / 2), 1)
    sqrt_len = max(int(np.sqrt(period)), 1)
    wma_half = calc_wma(series, half_len)
    wma_full = calc_wma(series, period)
    diff = 2 * wma_half - wma_full
    return calc_wma(diff, sqrt_len)


def calc_ma(series, ma_type, period):
    """Dispatch MA calculation."""
    if ma_type == 'EMA':
        return calc_ema(series, period)
    elif ma_type == 'SMA':
        return calc_sma(series, period)
    elif ma_type == 'WMA':
        return calc_wma(series, period)
    elif ma_type == 'HMA':
        return calc_hma(series, period)
    else:
        raise ValueError(f"Unknown MA type: {ma_type}")


# ============================================================
# ADX CALCULATIONS - 6 ENGINE VARIANTS
# ============================================================
def _true_range(high, low, close):
    """Calculate True Range array."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
    return tr


def _dm_plus_minus(high, low):
    """Calculate +DM and -DM arrays."""
    n = len(high)
    dm_p = np.zeros(n)
    dm_m = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        dn = low[i-1] - low[i]
        if up > dn and up > 0:
            dm_p[i] = up
        if dn > up and dn > 0:
            dm_m[i] = dn
    return dm_p, dm_m


def wilder_smooth_manual(values, period, start_idx=None):
    """Engine 1: Manual Wilder smoothing. seed = first valid value."""
    n = len(values)
    result = np.full(n, np.nan)
    if start_idx is None:
        start_idx = period
    # seed = sum of first period values starting from index 1
    seed_vals = values[1:period+1]
    if len(seed_vals) < period or np.any(np.isnan(seed_vals)):
        # fallback
        seed_vals = values[:period]
    result[period] = np.nansum(seed_vals)
    for i in range(period + 1, n):
        result[i] = result[i-1] - result[i-1] / period + values[i]
    return result


def calc_adx_engine1(high, low, close, period=20):
    """Engine 1: Wilder Manual - standard Wilder smoothing."""
    tr = _true_range(high, low, close)
    dm_p, dm_m = _dm_plus_minus(high, low)

    atr = wilder_smooth_manual(tr, period)
    sdm_p = wilder_smooth_manual(dm_p, period)
    sdm_m = wilder_smooth_manual(dm_m, period)

    n = len(high)
    di_p = np.full(n, np.nan)
    di_m = np.full(n, np.nan)
    dx = np.full(n, np.nan)

    for i in range(period, n):
        if atr[i] > 0 and not np.isnan(atr[i]):
            di_p[i] = 100.0 * sdm_p[i] / atr[i]
            di_m[i] = 100.0 * sdm_m[i] / atr[i]
            s = di_p[i] + di_m[i]
            if s > 0:
                dx[i] = 100.0 * abs(di_p[i] - di_m[i]) / s

    adx = wilder_smooth_manual(dx, period)
    # normalize: divide by period for the smoothed sums
    for i in range(period, n):
        if not np.isnan(atr[i]) and atr[i] > 0:
            di_p[i] = 100.0 * sdm_p[i] / atr[i]
            di_m[i] = 100.0 * sdm_m[i] / atr[i]

    # ADX from smoothed DX
    adx_final = np.full(n, np.nan)
    start = 2 * period
    if start < n:
        valid_dx = []
        for i in range(period, n):
            if not np.isnan(dx[i]):
                valid_dx.append((i, dx[i]))
                if len(valid_dx) == period:
                    break
        if len(valid_dx) == period:
            seed_idx = valid_dx[-1][0]
            adx_final[seed_idx] = np.mean([v for _, v in valid_dx])
            for i in range(seed_idx + 1, n):
                if not np.isnan(dx[i]) and not np.isnan(adx_final[i-1]):
                    adx_final[i] = (adx_final[i-1] * (period - 1) + dx[i]) / period
    return adx_final


def calc_adx_engine2(high, low, close, period=20):
    """Engine 2: Pandas EWM alpha=1/period, adjust=False."""
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    dm_p = pd.Series(np.where((h - h.shift(1) > l.shift(1) - l) & (h - h.shift(1) > 0), h - h.shift(1), 0.0))
    dm_m = pd.Series(np.where((l.shift(1) - l > h - h.shift(1)) & (l.shift(1) - l > 0), l.shift(1) - l, 0.0))

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    sdm_p = dm_p.ewm(alpha=alpha, adjust=False).mean()
    sdm_m = dm_m.ewm(alpha=alpha, adjust=False).mean()

    di_p = 100 * sdm_p / atr
    di_m = 100 * sdm_m / atr
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx.values


def calc_adx_engine3(high, low, close, period=20):
    """Engine 3: Pandas EWM span=period, adjust=False."""
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    dm_p = pd.Series(np.where((h - h.shift(1) > l.shift(1) - l) & (h - h.shift(1) > 0), h - h.shift(1), 0.0))
    dm_m = pd.Series(np.where((l.shift(1) - l > h - h.shift(1)) & (l.shift(1) - l > 0), l.shift(1) - l, 0.0))

    atr = tr.ewm(span=period, adjust=False).mean()
    sdm_p = dm_p.ewm(span=period, adjust=False).mean()
    sdm_m = dm_m.ewm(span=period, adjust=False).mean()

    di_p = 100 * sdm_p / atr
    di_m = 100 * sdm_m / atr
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx.values


def calc_adx_engine4(high, low, close, period=20):
    """Engine 4: SMA-seed Wilder. seed = SMA of first N values, then Wilder."""
    tr = _true_range(high, low, close)
    dm_p, dm_m = _dm_plus_minus(high, low)
    n = len(high)

    def sma_seed_wilder(vals, per):
        res = np.full(n, np.nan)
        # seed = SMA of first per values (starting from 1)
        s = 0.0
        cnt = 0
        for j in range(1, min(per + 1, n)):
            if not np.isnan(vals[j]):
                s += vals[j]
                cnt += 1
        if cnt == per:
            res[per] = s  # sum, not average for smoothed TR/DM
            for i in range(per + 1, n):
                res[i] = res[i-1] - res[i-1] / per + vals[i]
        return res

    atr = sma_seed_wilder(tr, period)
    sdm_p = sma_seed_wilder(dm_p, period)
    sdm_m = sma_seed_wilder(dm_m, period)

    di_p = np.full(n, np.nan)
    di_m = np.full(n, np.nan)
    dx = np.full(n, np.nan)

    for i in range(period, n):
        if not np.isnan(atr[i]) and atr[i] > 0:
            di_p[i] = 100.0 * sdm_p[i] / atr[i]
            di_m[i] = 100.0 * sdm_m[i] / atr[i]
            s = di_p[i] + di_m[i]
            if s > 0:
                dx[i] = 100.0 * abs(di_p[i] - di_m[i]) / s

    # ADX: SMA seed then Wilder
    adx = np.full(n, np.nan)
    valid_dx = []
    for i in range(period, n):
        if not np.isnan(dx[i]):
            valid_dx.append((i, dx[i]))
            if len(valid_dx) == period:
                break
    if len(valid_dx) == period:
        seed_idx = valid_dx[-1][0]
        adx[seed_idx] = np.mean([v for _, v in valid_dx])
        for i in range(seed_idx + 1, n):
            if not np.isnan(dx[i]) and not np.isnan(adx[i-1]):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    return adx


# ============================================================
# RSI CALCULATIONS
# ============================================================
def calc_rsi_engine1(close, period=10):
    """Engine 1: Wilder Manual RSI."""
    n = len(close)
    rsi = np.full(n, np.nan)
    delta = np.diff(close, prepend=close[0])

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    if period < n:
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
        for i in range(period, n):
            if not np.isnan(avg_gain[i]) and not np.isnan(avg_loss[i]):
                if avg_loss[i] == 0:
                    rsi[i] = 100.0
                else:
                    rs = avg_gain[i] / avg_loss[i]
                    rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def calc_rsi_engine2(close, period=10):
    """Engine 2: Pandas EWM alpha=1/period."""
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.values


def calc_rsi_engine3(close, period=10):
    """Engine 3: Pandas EWM span=period."""
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.values


def calc_rsi_engine4(close, period=10):
    """Engine 4: SMA-seed Wilder RSI. Identical to engine 1 (SMA seed IS the standard)."""
    return calc_rsi_engine1(close, period)


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(ohlcv_df, strat, engine_id):
    """
    Run a single backtest for one strategy on one engine.
    Returns dict with results.
    """
    name, fast_type, fast_len, slow_type, slow_len, tf, adx_th, rsi_lo, rsi_hi, delay, sl_pct, trail_act, trail_w, margin_pct, leverage = strat

    close = ohlcv_df['close'].values
    high = ohlcv_df['high'].values
    low = ohlcv_df['low'].values
    open_ = ohlcv_df['open'].values
    n = len(close)

    # --- MA ---
    fast_ma = calc_ma(ohlcv_df['close'], fast_type, fast_len).values
    slow_ma = calc_ma(ohlcv_df['close'], slow_type, slow_len).values

    # --- ADX (period=20) ---
    adx_period = 20
    use_adx = (engine_id != 5)
    use_rsi = (engine_id != 6)

    if use_adx:
        if engine_id == 1:
            adx = calc_adx_engine1(high, low, close, adx_period)
        elif engine_id == 2:
            adx = calc_adx_engine2(high, low, close, adx_period)
        elif engine_id == 3:
            adx = calc_adx_engine3(high, low, close, adx_period)
        elif engine_id == 4:
            adx = calc_adx_engine4(high, low, close, adx_period)
        elif engine_id == 6:
            adx = calc_adx_engine1(high, low, close, adx_period)
        else:
            adx = calc_adx_engine1(high, low, close, adx_period)
    else:
        adx = np.full(n, 50.0)  # always pass

    # --- RSI (period=10) ---
    rsi_period = 10
    if use_rsi:
        if engine_id == 1:
            rsi = calc_rsi_engine1(close, rsi_period)
        elif engine_id == 2:
            rsi = calc_rsi_engine2(close, rsi_period)
        elif engine_id == 3:
            rsi = calc_rsi_engine3(close, rsi_period)
        elif engine_id == 4:
            rsi = calc_rsi_engine4(close, rsi_period)
        elif engine_id == 5:
            rsi = calc_rsi_engine1(close, rsi_period)
        else:
            rsi = calc_rsi_engine1(close, rsi_period)
    else:
        rsi = np.full(n, 50.0)  # always pass

    # --- v32.1F extra filters (Engine 2 only for v32.1F strategies) ---
    is_v32 = name.startswith("v32.1F")
    adx_rise_bars = 6 if is_v32 else 0
    ema_gap_filter = 0.002 if is_v32 else 0.0
    monitor_window = 24 if is_v32 else 0

    # --- Warmup ---
    warmup = max(slow_len, fast_len, 2 * adx_period + 10, 100)

    # --- Simulation ---
    balance = INITIAL_CAPITAL
    peak_balance = INITIAL_CAPITAL
    max_dd = 0.0

    pos = 0       # 1=long, -1=short, 0=none
    entry_price = 0.0
    pos_size_usd = 0.0
    pos_qty = 0.0
    trail_high = 0.0
    trail_low = 999999.0
    trail_active = False

    trades = 0
    wins = 0
    sl_hits = 0
    gross_profit = 0.0
    gross_loss = 0.0

    # Cross detection with delay
    cross_signal = 0  # 1=long, -1=short
    cross_bar = -9999
    pending_signal = 0
    pending_bar = -9999

    for i in range(warmup, n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]):
            continue

        # Detect cross
        if i >= 1 and not np.isnan(fast_ma[i-1]) and not np.isnan(slow_ma[i-1]):
            if fast_ma[i-1] <= slow_ma[i-1] and fast_ma[i] > slow_ma[i]:
                cross_signal = 1
                cross_bar = i
            elif fast_ma[i-1] >= slow_ma[i-1] and fast_ma[i] < slow_ma[i]:
                cross_signal = -1
                cross_bar = i

        # Determine current signal (with delay)
        cur_signal = 0
        if cross_signal != 0:
            bars_since = i - cross_bar
            if delay == 0:
                if bars_since == 0:
                    cur_signal = cross_signal
            else:
                if bars_since == delay:
                    cur_signal = cross_signal

            # Monitor window for v32.1F
            if monitor_window > 0 and bars_since > monitor_window:
                cross_signal = 0  # expire

        # Also allow sustained signal (fast above/below slow)
        if cur_signal == 0 and pos == 0:
            if fast_ma[i] > slow_ma[i]:
                cur_signal = 1
            elif fast_ma[i] < slow_ma[i]:
                cur_signal = -1

        # --- Filters ---
        adx_ok = True
        rsi_ok = True

        if use_adx and not np.isnan(adx[i]):
            adx_ok = adx[i] >= adx_th
            if adx_rise_bars > 0 and i >= adx_rise_bars:
                if not np.isnan(adx[i - adx_rise_bars]):
                    adx_ok = adx_ok and (adx[i] > adx[i - adx_rise_bars])
        elif use_adx and np.isnan(adx[i]):
            adx_ok = False

        if use_rsi and not np.isnan(rsi[i]):
            rsi_ok = rsi_lo <= rsi[i] <= rsi_hi
        elif use_rsi and np.isnan(rsi[i]):
            rsi_ok = False

        # EMA gap filter for v32.1F
        if is_v32 and ema_gap_filter > 0:
            if slow_ma[i] != 0:
                gap = abs(fast_ma[i] - slow_ma[i]) / abs(slow_ma[i])
                if gap < ema_gap_filter:
                    adx_ok = False  # block entry

        # --- Position management ---
        if pos != 0:
            # Current PnL
            if pos == 1:
                roi = (close[i] - entry_price) / entry_price * leverage
                # Trail on close
                if close[i] > trail_high:
                    trail_high = close[i]
                # SL check on high/low
                sl_roi = (low[i] - entry_price) / entry_price * leverage
            else:  # short
                roi = (entry_price - close[i]) / entry_price * leverage
                if close[i] < trail_low:
                    trail_low = close[i]
                sl_roi = (entry_price - high[i]) / entry_price * leverage

            # --- SL hit ---
            if sl_roi <= -sl_pct:
                pnl = -sl_pct * pos_size_usd
                balance += pnl
                gross_loss += abs(pnl)
                trades += 1
                sl_hits += 1
                pos = 0
                trail_active = False
                if balance <= 0:
                    balance = 0
                    break
                continue

            # --- Trailing stop ---
            if roi >= trail_act:
                trail_active = True

            if trail_active:
                if pos == 1:
                    trail_roi = (close[i] - trail_high) / trail_high * leverage
                    if trail_roi <= -trail_w:
                        pnl = (close[i] - entry_price) / entry_price * leverage * pos_size_usd
                        balance += pnl
                        if pnl > 0:
                            gross_profit += pnl
                            wins += 1
                        else:
                            gross_loss += abs(pnl)
                        trades += 1
                        pos = 0
                        trail_active = False
                        continue
                else:
                    trail_roi = (trail_low - close[i]) / trail_low * leverage
                    if trail_roi <= -trail_w:
                        pnl = (entry_price - close[i]) / entry_price * leverage * pos_size_usd
                        balance += pnl
                        if pnl > 0:
                            gross_profit += pnl
                            wins += 1
                        else:
                            gross_loss += abs(pnl)
                        trades += 1
                        pos = 0
                        trail_active = False
                        continue

            # --- REV signal (opposite direction) ---
            if cur_signal != 0 and cur_signal != pos and adx_ok and rsi_ok:
                pnl = roi * pos_size_usd
                fee = pos_size_usd * FEE_RATE * 2  # close + open
                balance += pnl - fee
                if pnl > 0:
                    gross_profit += pnl
                    wins += 1
                else:
                    gross_loss += abs(pnl)
                trades += 1
                pos = 0
                trail_active = False

                if balance <= 0:
                    balance = 0
                    break

                # Open new position in opposite direction
                pos = cur_signal
                pos_size_usd = balance * margin_pct
                entry_price = close[i]
                pos_qty = pos_size_usd * leverage / entry_price
                trail_high = close[i]
                trail_low = close[i]
                trail_active = False

                fee_entry = pos_size_usd * FEE_RATE
                balance -= fee_entry

                # MDD
                if balance > peak_balance:
                    peak_balance = balance
                dd = (peak_balance - balance) / peak_balance * 100
                if dd > max_dd:
                    max_dd = dd
                continue

        # --- Entry ---
        if pos == 0 and cur_signal != 0 and adx_ok and rsi_ok:
            pos = cur_signal
            pos_size_usd = balance * margin_pct
            entry_price = close[i]
            pos_qty = pos_size_usd * leverage / entry_price
            trail_high = close[i]
            trail_low = close[i]
            trail_active = False

            fee_entry = pos_size_usd * FEE_RATE
            balance -= fee_entry

        # MDD tracking
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100
        if dd > max_dd:
            max_dd = dd

    # Close any remaining position
    if pos != 0:
        if pos == 1:
            roi = (close[-1] - entry_price) / entry_price * leverage
        else:
            roi = (entry_price - close[-1]) / entry_price * leverage
        pnl = roi * pos_size_usd
        fee = pos_size_usd * FEE_RATE
        balance += pnl - fee
        if pnl > 0:
            gross_profit += pnl
            wins += 1
        else:
            gross_loss += abs(pnl)
        trades += 1

    if balance > peak_balance:
        peak_balance = balance
    dd = (peak_balance - balance) / peak_balance * 100
    if dd > max_dd:
        max_dd = dd

    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    ret_pct = (balance - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return {
        'strategy': name,
        'engine_id': engine_id,
        'engine_name': ENGINE_NAMES[engine_id - 1],
        'final_balance': round(balance, 2),
        'return_pct': round(ret_pct, 2),
        'pf': round(pf, 2),
        'mdd_pct': round(max_dd, 2),
        'trades': trades,
        'wins': wins,
        'sl_hits': sl_hits,
    }


def _pool_init(shared_dict):
    """Initializer for worker processes - receives shared data."""
    global _SHARED_DATA
    _SHARED_DATA = shared_dict


def worker_task(args):
    """Worker function for multiprocessing."""
    strat, engine_id, _ = args
    tf = strat[5]  # timeframe
    ohlcv_df = _SHARED_DATA[tf]
    return run_backtest(ohlcv_df, strat, engine_id)


# Global shared data for multiprocessing
_SHARED_DATA = {}


def main():
    global _SHARED_DATA

    print("=" * 80)
    print("  6-ENGINE CROSS-VERIFICATION BACKTEST")
    print("  15 Strategies x 6 Engines = 90 Backtests")
    print("=" * 80)
    print()

    t_start = time.time()

    # --- Load data ---
    print("[1/4] Loading 5m data...")
    df_5m = load_5m_data()
    print(f"      Loaded {len(df_5m):,} rows of 5m data")

    # --- Resample to needed timeframes ---
    print("[2/4] Resampling to 10m, 15m, 30m...")
    needed_tfs = set(s[5] for s in STRATEGIES)
    print(f"      Needed timeframes: {sorted(needed_tfs)}")

    for tf in needed_tfs:
        _SHARED_DATA[tf] = resample_ohlcv(df_5m, tf)
        print(f"      {tf}: {len(_SHARED_DATA[tf]):,} bars")

    del df_5m  # free memory

    # --- Build task list ---
    print("[3/4] Running 90 backtests...")
    tasks = []
    for strat in STRATEGIES:
        for eng_id in range(1, 7):
            tasks.append((strat, eng_id, None))

    # --- Run with multiprocessing ---
    n_workers = min(cpu_count(), 8)
    print(f"      Using {n_workers} workers")

    results = []
    with Pool(processes=n_workers, initializer=_pool_init, initargs=(_SHARED_DATA,)) as pool:
        for i, res in enumerate(pool.imap_unordered(worker_task, tasks)):
            results.append(res)
            done = i + 1
            if done % 15 == 0 or done == len(tasks):
                elapsed = time.time() - t_start
                print(f"      {done}/{len(tasks)} done ({elapsed:.1f}s)")

    # --- Organize results ---
    print("[4/4] Generating reports...")
    print()

    # Build results DataFrame
    df_res = pd.DataFrame(results)

    # ============================================================
    # OUTPUT 1: Return% Matrix (15 strategies x 6 engines)
    # ============================================================
    print("=" * 120)
    print("  RETURN (%) MATRIX: 15 Strategies x 6 Engines")
    print("=" * 120)

    strat_names = [s[0] for s in STRATEGIES]
    matrix_data = {}
    for sname in strat_names:
        row = {}
        for eid in range(1, 7):
            mask = (df_res['strategy'] == sname) & (df_res['engine_id'] == eid)
            vals = df_res.loc[mask, 'return_pct'].values
            row[ENGINE_NAMES[eid-1]] = vals[0] if len(vals) > 0 else np.nan
        matrix_data[sname] = row

    matrix_df = pd.DataFrame(matrix_data).T
    matrix_df['AVG'] = matrix_df.mean(axis=1)
    matrix_df['STD'] = matrix_df.iloc[:, :6].std(axis=1)

    # Format and print
    header = f"{'Strategy':<12}"
    for ename in ENGINE_NAMES:
        header += f" {ename:>17}"
    header += f" {'AVG':>10} {'STD':>10}"
    print(header)
    print("-" * 120)

    for sname in strat_names:
        line = f"{sname:<12}"
        for ename in ENGINE_NAMES:
            v = matrix_data[sname].get(ename, np.nan)
            if np.isnan(v):
                line += f" {'N/A':>17}"
            else:
                line += f" {v:>16.1f}%"
        avg_v = matrix_df.loc[sname, 'AVG']
        std_v = matrix_df.loc[sname, 'STD']
        line += f" {avg_v:>9.1f}% {std_v:>9.1f}"
        print(line)

    print()

    # ============================================================
    # OUTPUT 2: PF Matrix
    # ============================================================
    print("=" * 120)
    print("  PROFIT FACTOR MATRIX: 15 Strategies x 6 Engines")
    print("=" * 120)

    pf_matrix = {}
    for sname in strat_names:
        row = {}
        for eid in range(1, 7):
            mask = (df_res['strategy'] == sname) & (df_res['engine_id'] == eid)
            vals = df_res.loc[mask, 'pf'].values
            row[ENGINE_NAMES[eid-1]] = vals[0] if len(vals) > 0 else np.nan
        pf_matrix[sname] = row

    header = f"{'Strategy':<12}"
    for ename in ENGINE_NAMES:
        header += f" {ename:>17}"
    print(header)
    print("-" * 120)

    for sname in strat_names:
        line = f"{sname:<12}"
        for ename in ENGINE_NAMES:
            v = pf_matrix[sname].get(ename, np.nan)
            if np.isnan(v):
                line += f" {'N/A':>17}"
            else:
                line += f" {v:>17.2f}"
        print(line)

    print()

    # ============================================================
    # OUTPUT 3: MDD Matrix
    # ============================================================
    print("=" * 120)
    print("  MAX DRAWDOWN (%) MATRIX: 15 Strategies x 6 Engines")
    print("=" * 120)

    mdd_matrix = {}
    for sname in strat_names:
        row = {}
        for eid in range(1, 7):
            mask = (df_res['strategy'] == sname) & (df_res['engine_id'] == eid)
            vals = df_res.loc[mask, 'mdd_pct'].values
            row[ENGINE_NAMES[eid-1]] = vals[0] if len(vals) > 0 else np.nan
        mdd_matrix[sname] = row

    header = f"{'Strategy':<12}"
    for ename in ENGINE_NAMES:
        header += f" {ename:>17}"
    header += f" {'AVG':>10}"
    print(header)
    print("-" * 120)

    for sname in strat_names:
        line = f"{sname:<12}"
        vals_list = []
        for ename in ENGINE_NAMES:
            v = mdd_matrix[sname].get(ename, np.nan)
            if np.isnan(v):
                line += f" {'N/A':>17}"
            else:
                line += f" {v:>16.1f}%"
                vals_list.append(v)
        avg_mdd = np.mean(vals_list) if vals_list else np.nan
        line += f" {avg_mdd:>9.1f}%"
        print(line)

    print()

    # ============================================================
    # OUTPUT 4: Trades Matrix
    # ============================================================
    print("=" * 120)
    print("  TRADES COUNT MATRIX: 15 Strategies x 6 Engines")
    print("=" * 120)

    header = f"{'Strategy':<12}"
    for ename in ENGINE_NAMES:
        header += f" {ename:>17}"
    print(header)
    print("-" * 120)

    for sname in strat_names:
        line = f"{sname:<12}"
        for eid in range(1, 7):
            mask = (df_res['strategy'] == sname) & (df_res['engine_id'] == eid)
            vals = df_res.loc[mask, 'trades'].values
            v = vals[0] if len(vals) > 0 else 0
            line += f" {v:>17d}"
        print(line)

    print()

    # ============================================================
    # RANKINGS
    # ============================================================
    print("=" * 80)
    print("  RANKINGS")
    print("=" * 80)
    print()

    # Calculate average metrics per strategy
    ranking_data = []
    for sname in strat_names:
        sdf = df_res[df_res['strategy'] == sname]
        avg_ret = sdf['return_pct'].mean()
        avg_mdd = sdf['mdd_pct'].mean()
        avg_pf = sdf['pf'].mean()
        avg_trades = sdf['trades'].mean()
        profitable_engines = (sdf['return_pct'] > 0).sum()
        loss_engines = (sdf['return_pct'] <= 0).sum()
        # MDD where profitable
        prof_mask = sdf['return_pct'] > 0
        avg_mdd_prof = sdf.loc[prof_mask, 'mdd_pct'].mean() if prof_mask.any() else 999.0
        std_ret = sdf['return_pct'].std()

        ranking_data.append({
            'strategy': sname,
            'avg_return': avg_ret,
            'avg_mdd': avg_mdd,
            'avg_mdd_profitable': avg_mdd_prof,
            'avg_pf': avg_pf,
            'avg_trades': avg_trades,
            'profitable_engines': profitable_engines,
            'loss_engines': loss_engines,
            'std_return': std_ret,
        })

    rank_df = pd.DataFrame(ranking_data)

    # --- RETURN BEST 10 ---
    print("  [A] RETURN BEST 10 (avg return% across 6 engines)")
    print("  " + "-" * 70)
    ret_sorted = rank_df.sort_values('avg_return', ascending=False).head(10)
    print(f"  {'Rank':<5} {'Strategy':<12} {'Avg Return%':>12} {'Avg PF':>8} {'Avg MDD%':>10} {'Prof/6':>7} {'Trades':>7}")
    print("  " + "-" * 70)
    for rank, (_, row) in enumerate(ret_sorted.iterrows(), 1):
        print(f"  {rank:<5} {row['strategy']:<12} {row['avg_return']:>11.1f}% {row['avg_pf']:>8.2f} {row['avg_mdd']:>9.1f}% {int(row['profitable_engines']):>5}/6 {row['avg_trades']:>7.0f}")
    print()

    # --- STABILITY BEST 10 ---
    print("  [B] STABILITY BEST 10 (lowest avg MDD% across engines where profitable)")
    print("  " + "-" * 70)
    stable_df = rank_df[rank_df['profitable_engines'] >= 1].sort_values('avg_mdd_profitable').head(10)
    print(f"  {'Rank':<5} {'Strategy':<12} {'MDD%(prof)':>12} {'Avg Return%':>12} {'Avg PF':>8} {'Prof/6':>7}")
    print("  " + "-" * 70)
    for rank, (_, row) in enumerate(stable_df.iterrows(), 1):
        print(f"  {rank:<5} {row['strategy']:<12} {row['avg_mdd_profitable']:>11.1f}% {row['avg_return']:>11.1f}% {row['avg_pf']:>8.2f} {int(row['profitable_engines']):>5}/6")
    print()

    # --- DISCARD 10 ---
    print("  [C] DISCARD 10 (worst avg return or majority engines show loss)")
    print("  " + "-" * 70)
    # Sort by: first, majority-loss engines (loss_engines >= 4), then worst avg return
    rank_df['discard_score'] = -rank_df['avg_return'] + (rank_df['loss_engines'] >= 4).astype(int) * 100000
    discard_sorted = rank_df.sort_values('discard_score', ascending=False).head(10)
    print(f"  {'Rank':<5} {'Strategy':<12} {'Avg Return%':>12} {'Avg PF':>8} {'Loss/6':>7} {'Reason':<30}")
    print("  " + "-" * 70)
    for rank, (_, row) in enumerate(discard_sorted.iterrows(), 1):
        reason = "Majority engines loss" if row['loss_engines'] >= 4 else "Worst avg return"
        print(f"  {rank:<5} {row['strategy']:<12} {row['avg_return']:>11.1f}% {row['avg_pf']:>8.2f} {int(row['loss_engines']):>5}/6 {reason:<30}")
    print()

    # ============================================================
    # SAVE FILES
    # ============================================================
    out_dir = DATA_DIR

    # CSV
    csv_path = os.path.join(out_dir, "cross_verify_results.csv")
    df_res.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {csv_path}")

    # Report TXT
    txt_path = os.path.join(out_dir, "cross_verify_report.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  6-ENGINE CROSS-VERIFICATION BACKTEST REPORT\n")
        f.write(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  15 Strategies x 6 Engines = 90 Backtests\n")
        f.write(f"  Initial Capital: ${INITIAL_CAPITAL:,.0f}\n")
        f.write(f"  Fee: {FEE_RATE*100:.2f}%\n")
        f.write("=" * 80 + "\n\n")

        # Engine descriptions
        f.write("ENGINES:\n")
        for i, ename in enumerate(ENGINE_NAMES, 1):
            descs = {
                1: "Manual Wilder smoothing for ADX/RSI (standard)",
                2: "ewm(alpha=1/period, adjust=False) for ADX/RSI",
                3: "ewm(span=period, adjust=False) for ADX/RSI",
                4: "Wilder but seed=SMA of first N, then Wilder",
                5: "Remove ADX filter entirely, keep only MA cross + RSI",
                6: "Remove RSI filter entirely, keep only MA cross + ADX",
            }
            f.write(f"  Engine {i}: {ename} - {descs[i]}\n")
        f.write("\n")

        # Return matrix
        f.write("RETURN (%) MATRIX:\n")
        f.write(f"{'Strategy':<12}")
        for ename in ENGINE_NAMES:
            f.write(f" {ename:>17}")
        f.write(f" {'AVG':>10}\n")
        f.write("-" * 130 + "\n")

        for sname in strat_names:
            f.write(f"{sname:<12}")
            vals_list = []
            for ename in ENGINE_NAMES:
                v = matrix_data[sname].get(ename, np.nan)
                if np.isnan(v):
                    f.write(f" {'N/A':>17}")
                else:
                    f.write(f" {v:>16.1f}%")
                    vals_list.append(v)
            avg_v = np.mean(vals_list) if vals_list else np.nan
            f.write(f" {avg_v:>9.1f}%\n")
        f.write("\n")

        # Rankings
        f.write("RETURN BEST 10:\n")
        for rank, (_, row) in enumerate(ret_sorted.iterrows(), 1):
            f.write(f"  {rank}. {row['strategy']:<12} Avg Return: {row['avg_return']:>10.1f}%  PF: {row['avg_pf']:.2f}  MDD: {row['avg_mdd']:.1f}%  Profitable: {int(row['profitable_engines'])}/6\n")
        f.write("\n")

        f.write("STABILITY BEST 10:\n")
        for rank, (_, row) in enumerate(stable_df.iterrows(), 1):
            f.write(f"  {rank}. {row['strategy']:<12} MDD(prof): {row['avg_mdd_profitable']:>8.1f}%  Avg Return: {row['avg_return']:>10.1f}%  PF: {row['avg_pf']:.2f}\n")
        f.write("\n")

        f.write("DISCARD 10:\n")
        for rank, (_, row) in enumerate(discard_sorted.iterrows(), 1):
            reason = "Majority engines loss" if row['loss_engines'] >= 4 else "Worst avg return"
            f.write(f"  {rank}. {row['strategy']:<12} Avg Return: {row['avg_return']:>10.1f}%  Loss: {int(row['loss_engines'])}/6  Reason: {reason}\n")
        f.write("\n")

        # Detailed results
        f.write("=" * 80 + "\n")
        f.write("DETAILED RESULTS (all 90 backtests):\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Strategy':<12} {'Engine':<20} {'Balance':>12} {'Return%':>10} {'PF':>8} {'MDD%':>8} {'Trades':>7} {'Wins':>6} {'SL':>5}\n")
        f.write("-" * 95 + "\n")

        for sname in strat_names:
            for eid in range(1, 7):
                mask = (df_res['strategy'] == sname) & (df_res['engine_id'] == eid)
                if mask.any():
                    r = df_res[mask].iloc[0]
                    f.write(f"{r['strategy']:<12} {r['engine_name']:<20} ${r['final_balance']:>11,.2f} {r['return_pct']:>9.1f}% {r['pf']:>8.2f} {r['mdd_pct']:>7.1f}% {r['trades']:>7d} {r['wins']:>6d} {r['sl_hits']:>5d}\n")
            f.write("\n")

    print(f"  Saved: {txt_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
