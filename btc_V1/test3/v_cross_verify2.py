#!/usr/bin/env python3
"""
6-Engine Cross-Verification Backtest v2 — 15 Strategies
=========================================================
v32.1F_A/B removed. v32.2 and v32.3 added with full spec from planning docs.

v32.2: EMA(100)/EMA(600) Tight-SL — ADX rise 6, gap>=0.2%, monitor 24, skip same dir,
       TSL activates at +12% (intrabar), SL disabled when TSL active,
       TSL trail 9% from peak, daily loss -20%.
v32.3: EMA(75)/SMA(750) Low-MDD  — same filters, RSI period 11.

Engines differ ONLY in ADX/RSI calculation method.
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

# --------------------------------------------------------------------------
# Strategy tuple layout:
# (name, fast_type, fast_len, slow_type, slow_len, tf,
#  adx_th, rsi_lo, rsi_hi, delay,
#  sl, trail_act, trail_w, margin, lev)
#
# v32.2 and v32.3 are flagged by name prefix "v32." and carry special filters:
#   adx_rise_bars=6, ema_gap=0.2%, monitor_window=24, skip_same_dir=True,
#   daily_loss_limit=-0.20, rsi_period override, TSL-priority-over-SL logic.
# --------------------------------------------------------------------------
STRATEGIES = [
    # --- v32.2: EMA(100)/EMA(600) 30m, RSI(10) 40-80, SL3%, TA12%, TSL9%, margin35%, lev10 ---
    ("v32.2",   "EMA", 100, "EMA", 600, "30m", 30, 40, 80, 0, 0.03, 0.12, 0.09, 0.35, 10),
    # --- v32.3: EMA(75)/SMA(750)  30m, RSI(11) 40-80, SL3%, TA12%, TSL9%, margin35%, lev10 ---
    ("v32.3",   "EMA", 75,  "SMA", 750, "30m", 30, 40, 80, 0, 0.03, 0.12, 0.09, 0.35, 10),
    # --- unchanged from v1 ---
    ("v22.7",   "EMA", 7,   "EMA", 250, "15m", 45, 35, 75, 0, 0.08, 0.07, 0.05, 0.50, 15),
    ("v22.4",   "EMA", 7,   "EMA", 250, "15m", 45, 35, 75, 0, 0.08, 0.07, 0.05, 0.40, 15),
    ("v22.0F",  "WMA", 3,   "EMA", 200, "30m", 35, 35, 65, 5, 0.08, 0.03, 0.02, 0.50, 10),
    ("v16.6",   "WMA", 3,   "EMA", 200, "30m", 35, 30, 70, 5, 0.08, 0.03, 0.02, 0.50, 10),
    ("v16.4",   "WMA", 3,   "EMA", 200, "30m", 35, 35, 65, 0, 0.08, 0.04, 0.03, 0.30, 10),
    ("v22.8",   "EMA", 100, "EMA", 600, "30m", 30, 35, 75, 0, 0.08, 0.06, 0.05, 0.35, 10),
    ("v14.4",   "EMA", 3,   "EMA", 200, "30m", 35, 30, 65, 0, 0.07, 0.06, 0.03, 0.25, 10),
    ("v25.1A",  "HMA", 21,  "EMA", 250, "10m", 35, 40, 75, 0, 0.06, 0.07, 0.03, 0.50, 10),
    ("v28_T1",  "WMA", 5,   "EMA", 300, "15m", 35, 35, 75, 2, 0.05, 0.04, 0.02, 0.15, 15),
    ("v28_T2",  "HMA", 14,  "EMA", 300, "15m", 35, 35, 70, 3, 0.07, 0.06, 0.03, 0.50, 10),
    ("v24.2",   "EMA", 3,   "EMA", 100, "30m", 30, 30, 70, 0, 0.08, 0.06, 0.05, 0.70, 10),
    ("v15.4",   "EMA", 3,   "EMA", 200, "30m", 35, 30, 65, 0, 0.07, 0.06, 0.03, 0.40, 10),
    ("v25.2C",  "EMA", 5,   "EMA", 300, "10m", 35, 40, 75, 0, 0.07, 0.08, 0.03, 0.40, 10),
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
    return series.ewm(span=period, adjust=False).mean()

def calc_sma(series, period):
    return series.rolling(period).mean()

def calc_wma(series, period):
    weights = np.arange(1, period + 1, dtype=float)
    def _wma(x):
        return np.dot(x, weights) / weights.sum()
    return series.rolling(period).apply(_wma, raw=True)

def calc_hma(series, period):
    half_len = max(int(period / 2), 1)
    sqrt_len = max(int(np.sqrt(period)), 1)
    wma_half = calc_wma(series, half_len)
    wma_full = calc_wma(series, period)
    diff = 2 * wma_half - wma_full
    return calc_wma(diff, sqrt_len)

def calc_ma(series, ma_type, period):
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
# ADX CALCULATIONS — 6 ENGINE VARIANTS
# ============================================================
def _true_range(high, low, close):
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
    return tr

def _dm_plus_minus(high, low):
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

def wilder_smooth_manual(values, period):
    n = len(values)
    result = np.full(n, np.nan)
    seed_vals = values[1:period+1]
    if len(seed_vals) < period or np.any(np.isnan(seed_vals)):
        seed_vals = values[:period]
    result[period] = np.nansum(seed_vals)
    for i in range(period + 1, n):
        result[i] = result[i-1] - result[i-1] / period + values[i]
    return result

def calc_adx_engine1(high, low, close, period=20):
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
        if not np.isnan(atr[i]) and atr[i] > 0:
            di_p[i] = 100.0 * sdm_p[i] / atr[i]
            di_m[i] = 100.0 * sdm_m[i] / atr[i]
            s = di_p[i] + di_m[i]
            if s > 0:
                dx[i] = 100.0 * abs(di_p[i] - di_m[i]) / s
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

def calc_adx_engine2(high, low, close, period=20):
    h = pd.Series(high); l = pd.Series(low); c = pd.Series(close)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    dm_p = pd.Series(np.where((h - h.shift(1) > l.shift(1) - l) & (h - h.shift(1) > 0), h - h.shift(1), 0.0))
    dm_m = pd.Series(np.where((l.shift(1) - l > h - h.shift(1)) & (l.shift(1) - l > 0), l.shift(1) - l, 0.0))
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    sdm_p = dm_p.ewm(alpha=alpha, adjust=False).mean()
    sdm_m = dm_m.ewm(alpha=alpha, adjust=False).mean()
    di_p = 100 * sdm_p / atr.replace(0, 1e-10)
    di_m = 100 * sdm_m / atr.replace(0, 1e-10)
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, 1e-10)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx.values

def calc_adx_engine3(high, low, close, period=20):
    h = pd.Series(high); l = pd.Series(low); c = pd.Series(close)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    dm_p = pd.Series(np.where((h - h.shift(1) > l.shift(1) - l) & (h - h.shift(1) > 0), h - h.shift(1), 0.0))
    dm_m = pd.Series(np.where((l.shift(1) - l > h - h.shift(1)) & (l.shift(1) - l > 0), l.shift(1) - l, 0.0))
    atr = tr.ewm(span=period, adjust=False).mean()
    sdm_p = dm_p.ewm(span=period, adjust=False).mean()
    sdm_m = dm_m.ewm(span=period, adjust=False).mean()
    di_p = 100 * sdm_p / atr.replace(0, 1e-10)
    di_m = 100 * sdm_m / atr.replace(0, 1e-10)
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx.values

def calc_adx_engine4(high, low, close, period=20):
    tr = _true_range(high, low, close)
    dm_p, dm_m = _dm_plus_minus(high, low)
    n = len(high)
    def sma_seed_wilder(vals, per):
        res = np.full(n, np.nan)
        s = 0.0; cnt = 0
        for j in range(1, min(per + 1, n)):
            if not np.isnan(vals[j]):
                s += vals[j]; cnt += 1
        if cnt == per:
            res[per] = s
            for i in range(per + 1, n):
                res[i] = res[i-1] - res[i-1] / per + vals[i]
        return res
    atr = sma_seed_wilder(tr, period)
    sdm_p = sma_seed_wilder(dm_p, period)
    sdm_m = sma_seed_wilder(dm_m, period)
    di_p = np.full(n, np.nan); di_m = np.full(n, np.nan); dx = np.full(n, np.nan)
    for i in range(period, n):
        if not np.isnan(atr[i]) and atr[i] > 0:
            di_p[i] = 100.0 * sdm_p[i] / atr[i]
            di_m[i] = 100.0 * sdm_m[i] / atr[i]
            s = di_p[i] + di_m[i]
            if s > 0:
                dx[i] = 100.0 * abs(di_p[i] - di_m[i]) / s
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
                    rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi

def calc_rsi_engine2(close, period=10):
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss_s = (-delta).clip(lower=0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss_s.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.values

def calc_rsi_engine3(close, period=10):
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss_s = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss_s.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.values

def calc_rsi_engine4(close, period=10):
    return calc_rsi_engine1(close, period)


# ============================================================
# BACKTEST ENGINE — v32.x full-spec loop vs legacy loop
# ============================================================
def _run_v32_backtest(close, high, low, fast_ma, slow_ma, adx, rsi,
                      strat, engine_id, use_adx, use_rsi):
    """
    Full v32.2 / v32.3 spec backtest per planning docs.
    Implements: cross-watch-monitor entry, TSL priority over SL,
    ADX rise check, EMA gap, daily loss limit, skip same dir,
    intrabar TA activation, close-based TSL exit.
    """
    name, fast_type, fast_len, slow_type, slow_len, tf, \
        adx_th, rsi_lo, rsi_hi, delay, \
        sl_pct_dec, trail_act_dec, trail_w_dec, margin_pct, leverage = strat

    # Convert decimals to percent for the v32 doc convention
    sl_pct   = sl_pct_dec * 100.0      # 3.0
    ta_pct   = trail_act_dec * 100.0    # 12.0
    tsl_pct  = trail_w_dec * 100.0      # 9.0

    # v32 specific constants
    ADX_RISE_BARS  = 6
    EMA_GAP_MIN    = 0.2      # percent
    MONITOR_WINDOW = 24
    SKIP_SAME_DIR  = True
    DAILY_LOSS_PCT = -0.20
    BARS_PER_DAY   = 1440     # for 30m: 48 bars per day — but doc says i%1440

    n = len(close)

    # RSI period override: v32.3 uses RSI(11)
    # (already handled upstream via engine dispatch with correct period)

    # State
    cap = INITIAL_CAPITAL
    pos = 0          # 1=LONG, -1=SHORT, 0=none
    epx = 0.0        # entry price
    psz = 0.0        # position size (USD amount)
    slp = 0.0        # current SL price
    ton = False       # TSL active
    thi = 0.0         # trail high
    tlo = 999999.0    # trail low
    watching = 0      # 1=watch long, -1=watch short
    ws = 0            # watch start bar
    ld = 0            # last close direction
    le_bar = 0        # last close bar
    pk = cap          # peak balance
    mdd = 0.0         # max drawdown fraction
    ms = cap          # daily start balance

    trades = 0; wins = 0; sl_hits = 0; tsl_hits = 0; rev_hits = 0
    gross_profit = 0.0; gross_loss = 0.0

    warmup = 600

    for i in range(warmup, n):
        px = close[i]
        h_ = high[i]
        l_ = low[i]

        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]):
            continue

        # Daily reset
        if i > warmup and i % BARS_PER_DAY == 0:
            ms = cap

        # ============ STEP A: Position open — exit checks ============
        if pos != 0:
            watching = 0  # reset watch while in position

            # A1: SL check (only when TSL NOT active)
            if not ton:
                sl_hit = False
                if pos == 1 and l_ <= slp:
                    sl_hit = True
                elif pos == -1 and h_ >= slp:
                    sl_hit = True
                if sl_hit:
                    pnl = (slp - epx) / epx * psz * pos - psz * FEE_RATE
                    cap += pnl
                    if pnl > 0:
                        gross_profit += pnl; wins += 1
                    else:
                        gross_loss += abs(pnl)
                    sl_hits += 1; trades += 1
                    ld = pos; le_bar = i; pos = 0
                    pk = max(pk, cap)
                    dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    if cap <= 0: break
                    continue

            # A2: TA activation (intrabar high/low)
            if pos == 1:
                br = (h_ - epx) / epx * 100.0
            else:
                br = (epx - l_) / epx * 100.0
            if br >= ta_pct:
                ton = True

            # A3: TSL check (only when TSL IS active)
            if ton:
                tsl_exit = False
                if pos == 1:
                    if h_ > thi: thi = h_
                    ns = thi * (1.0 - tsl_pct / 100.0)
                    if ns > slp: slp = ns
                    if px <= slp:
                        tsl_exit = True
                else:
                    if l_ < tlo: tlo = l_
                    ns = tlo * (1.0 + tsl_pct / 100.0)
                    if ns < slp: slp = ns
                    if px >= slp:
                        tsl_exit = True
                if tsl_exit:
                    pnl = (px - epx) / epx * psz * pos - psz * FEE_RATE
                    cap += pnl
                    if pnl > 0:
                        gross_profit += pnl; wins += 1
                    else:
                        gross_loss += abs(pnl)
                    tsl_hits += 1; trades += 1
                    ld = pos; le_bar = i; pos = 0
                    pk = max(pk, cap)
                    dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    if cap <= 0: break
                    continue

            # A4: REV check (EMA cross reversal)
            if i > 0 and not np.isnan(fast_ma[i-1]) and not np.isnan(slow_ma[i-1]):
                bull_now = fast_ma[i] > slow_ma[i]
                bull_prev = fast_ma[i-1] > slow_ma[i-1]
                cross_up = bull_now and not bull_prev
                cross_down = (not bull_now) and bull_prev
                if (pos == 1 and cross_down) or (pos == -1 and cross_up):
                    pnl = (px - epx) / epx * psz * pos - psz * FEE_RATE
                    cap += pnl
                    if pnl > 0:
                        gross_profit += pnl; wins += 1
                    else:
                        gross_loss += abs(pnl)
                    rev_hits += 1; trades += 1
                    ld = pos; le_bar = i; pos = 0
                    # do NOT continue — same bar entry is possible

        # ============ STEP B: No position — entry checks ============
        if i < 1:
            continue

        if np.isnan(fast_ma[i-1]) or np.isnan(slow_ma[i-1]):
            # Can't detect cross
            pk = max(pk, cap)
            dd = (pk - cap) / pk if pk > 0 else 0
            if dd > mdd: mdd = dd
            continue

        bull_now  = fast_ma[i] > slow_ma[i]
        bull_prev = fast_ma[i-1] > slow_ma[i-1]
        cross_up   = bull_now and not bull_prev
        cross_down = (not bull_now) and bull_prev

        if pos == 0:
            # B2: new cross -> start watch
            if cross_up:
                watching = 1; ws = i
            elif cross_down:
                watching = -1; ws = i

            if watching != 0 and i > ws:
                # B3: monitor window expired
                if i - ws > MONITOR_WINDOW:
                    watching = 0
                    pk = max(pk, cap)
                    dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    continue

                # B4: opposite cross during watch
                if watching == 1 and cross_down:
                    watching = -1; ws = i
                    pk = max(pk, cap)
                    dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    continue
                elif watching == -1 and cross_up:
                    watching = 1; ws = i
                    pk = max(pk, cap)
                    dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    continue

                # B5: skip same direction as last close
                if SKIP_SAME_DIR and watching == ld:
                    pk = max(pk, cap)
                    dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    continue

                # B6: ADX filter
                if use_adx:
                    if np.isnan(adx[i]) or adx[i] < adx_th:
                        pk = max(pk, cap)
                        dd = (pk - cap) / pk if pk > 0 else 0
                        if dd > mdd: mdd = dd
                        continue
                    # B7: ADX rise
                    if ADX_RISE_BARS > 0 and i >= ADX_RISE_BARS:
                        if not np.isnan(adx[i - ADX_RISE_BARS]):
                            if adx[i] <= adx[i - ADX_RISE_BARS]:
                                pk = max(pk, cap)
                                dd = (pk - cap) / pk if pk > 0 else 0
                                if dd > mdd: mdd = dd
                                continue

                # B8: RSI filter
                if use_rsi:
                    if np.isnan(rsi[i]):
                        pk = max(pk, cap)
                        dd = (pk - cap) / pk if pk > 0 else 0
                        if dd > mdd: mdd = dd
                        continue
                    if rsi[i] < rsi_lo or rsi[i] > rsi_hi:
                        pk = max(pk, cap)
                        dd = (pk - cap) / pk if pk > 0 else 0
                        if dd > mdd: mdd = dd
                        continue

                # B9: EMA gap filter
                if EMA_GAP_MIN > 0 and slow_ma[i] != 0:
                    gap = abs(fast_ma[i] - slow_ma[i]) / abs(slow_ma[i]) * 100.0
                    if gap < EMA_GAP_MIN:
                        pk = max(pk, cap)
                        dd = (pk - cap) / pk if pk > 0 else 0
                        if dd > mdd: mdd = dd
                        continue

                # B10: daily loss limit
                if ms > 0 and (cap - ms) / ms <= DAILY_LOSS_PCT:
                    watching = 0
                    pk = max(pk, cap)
                    dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    continue

                # B11: balance check
                if cap <= 0:
                    pk = max(pk, cap)
                    dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    continue

                # ======== ENTER ========
                mg = cap * margin_pct
                psz = mg * leverage
                cap -= psz * FEE_RATE      # entry fee
                pos = watching
                epx = px
                ton = False
                thi = px
                tlo = px
                if pos == 1:
                    slp = epx * (1.0 - sl_pct / 100.0)
                else:
                    slp = epx * (1.0 + sl_pct / 100.0)
                pk = max(pk, cap)
                watching = 0

        # MDD update
        pk = max(pk, cap)
        dd = (pk - cap) / pk if pk > 0 else 0
        if dd > mdd: mdd = dd
        if cap <= 0: break

    # Force close at end
    if pos != 0 and cap > 0:
        pnl = (close[-1] - epx) / epx * psz * pos - psz * FEE_RATE
        cap += pnl
        if pnl > 0:
            gross_profit += pnl; wins += 1
        else:
            gross_loss += abs(pnl)
        trades += 1

    pk = max(pk, cap)
    dd = (pk - cap) / pk if pk > 0 else 0
    if dd > mdd: mdd = dd

    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    ret_pct = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0

    return {
        'strategy': name,
        'engine_id': engine_id,
        'engine_name': ENGINE_NAMES[engine_id - 1],
        'final_balance': round(cap, 2),
        'return_pct': round(ret_pct, 2),
        'pf': round(pf, 2),
        'mdd_pct': round(mdd * 100.0, 2),
        'trades': trades,
        'wins': wins,
        'sl_hits': sl_hits,
    }


def _run_legacy_backtest(close, high, low, fast_ma, slow_ma, adx, rsi,
                         strat, engine_id, use_adx, use_rsi):
    """
    Legacy backtest loop for non-v32 strategies (same as v_cross_verify.py).
    """
    name, fast_type, fast_len, slow_type, slow_len, tf, \
        adx_th, rsi_lo, rsi_hi, delay, \
        sl_pct, trail_act, trail_w, margin_pct, leverage = strat

    n = len(close)
    warmup = max(slow_len, fast_len, 40, 100)

    balance = INITIAL_CAPITAL
    peak_balance = INITIAL_CAPITAL
    max_dd = 0.0

    pos = 0
    entry_price = 0.0
    pos_size_usd = 0.0
    trail_high = 0.0
    trail_low = 999999.0
    trail_active = False

    trades = 0; wins = 0; sl_hits = 0
    gross_profit = 0.0; gross_loss = 0.0

    cross_signal = 0; cross_bar = -9999

    for i in range(warmup, n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]):
            continue

        # Cross detection
        if i >= 1 and not np.isnan(fast_ma[i-1]) and not np.isnan(slow_ma[i-1]):
            if fast_ma[i-1] <= slow_ma[i-1] and fast_ma[i] > slow_ma[i]:
                cross_signal = 1; cross_bar = i
            elif fast_ma[i-1] >= slow_ma[i-1] and fast_ma[i] < slow_ma[i]:
                cross_signal = -1; cross_bar = i

        cur_signal = 0
        if cross_signal != 0:
            bars_since = i - cross_bar
            if delay == 0:
                if bars_since == 0:
                    cur_signal = cross_signal
            else:
                if bars_since == delay:
                    cur_signal = cross_signal

        # Sustained signal for non-delay strategies
        if cur_signal == 0 and pos == 0:
            if fast_ma[i] > slow_ma[i]:
                cur_signal = 1
            elif fast_ma[i] < slow_ma[i]:
                cur_signal = -1

        # Filters
        adx_ok = True; rsi_ok = True
        if use_adx:
            if not np.isnan(adx[i]):
                adx_ok = adx[i] >= adx_th
            else:
                adx_ok = False
        if use_rsi:
            if not np.isnan(rsi[i]):
                rsi_ok = rsi_lo <= rsi[i] <= rsi_hi
            else:
                rsi_ok = False

        # Position management
        if pos != 0:
            if pos == 1:
                roi = (close[i] - entry_price) / entry_price * leverage
                if close[i] > trail_high: trail_high = close[i]
                sl_roi = (low[i] - entry_price) / entry_price * leverage
            else:
                roi = (entry_price - close[i]) / entry_price * leverage
                if close[i] < trail_low: trail_low = close[i]
                sl_roi = (entry_price - high[i]) / entry_price * leverage

            # SL hit
            if sl_roi <= -sl_pct:
                pnl = -sl_pct * pos_size_usd
                balance += pnl
                gross_loss += abs(pnl)
                trades += 1; sl_hits += 1
                pos = 0; trail_active = False
                if balance <= 0:
                    balance = 0; break
                continue

            # Trailing activation
            if roi >= trail_act:
                trail_active = True

            if trail_active:
                if pos == 1:
                    trail_roi = (close[i] - trail_high) / trail_high * leverage
                else:
                    trail_roi = (trail_low - close[i]) / trail_low * leverage
                if trail_roi <= -trail_w:
                    pnl = (close[i] - entry_price) / entry_price * leverage * pos_size_usd if pos == 1 else \
                          (entry_price - close[i]) / entry_price * leverage * pos_size_usd
                    balance += pnl
                    if pnl > 0: gross_profit += pnl; wins += 1
                    else: gross_loss += abs(pnl)
                    trades += 1; pos = 0; trail_active = False
                    continue

            # REV
            if cur_signal != 0 and cur_signal != pos and adx_ok and rsi_ok:
                pnl = roi * pos_size_usd
                fee = pos_size_usd * FEE_RATE * 2
                balance += pnl - fee
                if pnl > 0: gross_profit += pnl; wins += 1
                else: gross_loss += abs(pnl)
                trades += 1; pos = 0; trail_active = False
                if balance <= 0:
                    balance = 0; break
                # Open new
                pos = cur_signal
                pos_size_usd = balance * margin_pct
                entry_price = close[i]
                trail_high = close[i]; trail_low = close[i]
                trail_active = False
                balance -= pos_size_usd * FEE_RATE
                if balance > peak_balance: peak_balance = balance
                dd = (peak_balance - balance) / peak_balance * 100
                if dd > max_dd: max_dd = dd
                continue

        # Entry
        if pos == 0 and cur_signal != 0 and adx_ok and rsi_ok:
            pos = cur_signal
            pos_size_usd = balance * margin_pct
            entry_price = close[i]
            trail_high = close[i]; trail_low = close[i]
            trail_active = False
            balance -= pos_size_usd * FEE_RATE

        # MDD
        if balance > peak_balance: peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100
        if dd > max_dd: max_dd = dd

    # Close remaining
    if pos != 0:
        if pos == 1:
            roi = (close[-1] - entry_price) / entry_price * leverage
        else:
            roi = (entry_price - close[-1]) / entry_price * leverage
        pnl = roi * pos_size_usd
        fee = pos_size_usd * FEE_RATE
        balance += pnl - fee
        if pnl > 0: gross_profit += pnl; wins += 1
        else: gross_loss += abs(pnl)
        trades += 1

    if balance > peak_balance: peak_balance = balance
    dd = (peak_balance - balance) / peak_balance * 100
    if dd > max_dd: max_dd = dd

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


# ============================================================
# MAIN DISPATCH
# ============================================================
def run_backtest(ohlcv_df, strat, engine_id):
    name = strat[0]
    fast_type, fast_len = strat[1], strat[2]
    slow_type, slow_len = strat[3], strat[4]

    close = ohlcv_df['close'].values
    high  = ohlcv_df['high'].values
    low   = ohlcv_df['low'].values
    n = len(close)

    # MA
    fast_ma = calc_ma(ohlcv_df['close'], fast_type, fast_len).values
    slow_ma = calc_ma(ohlcv_df['close'], slow_type, slow_len).values

    # ADX
    adx_period = 20
    use_adx = (engine_id != 5)
    use_rsi = (engine_id != 6)

    if use_adx:
        if engine_id == 1:   adx = calc_adx_engine1(high, low, close, adx_period)
        elif engine_id == 2: adx = calc_adx_engine2(high, low, close, adx_period)
        elif engine_id == 3: adx = calc_adx_engine3(high, low, close, adx_period)
        elif engine_id == 4: adx = calc_adx_engine4(high, low, close, adx_period)
        elif engine_id == 6: adx = calc_adx_engine1(high, low, close, adx_period)
        else:                adx = calc_adx_engine1(high, low, close, adx_period)
    else:
        adx = np.full(n, 50.0)

    # RSI — use strategy-specific period for v32.x
    is_v32 = name.startswith("v32.")
    if is_v32 and name == "v32.3":
        rsi_period = 11
    elif is_v32 and name == "v32.2":
        rsi_period = 10
    else:
        rsi_period = 10   # default

    if use_rsi:
        if engine_id == 1:   rsi = calc_rsi_engine1(close, rsi_period)
        elif engine_id == 2: rsi = calc_rsi_engine2(close, rsi_period)
        elif engine_id == 3: rsi = calc_rsi_engine3(close, rsi_period)
        elif engine_id == 4: rsi = calc_rsi_engine4(close, rsi_period)
        elif engine_id == 5: rsi = calc_rsi_engine1(close, rsi_period)
        else:                rsi = calc_rsi_engine1(close, rsi_period)
    else:
        rsi = np.full(n, 50.0)

    # Dispatch
    if is_v32:
        return _run_v32_backtest(close, high, low, fast_ma, slow_ma, adx, rsi,
                                 strat, engine_id, use_adx, use_rsi)
    else:
        return _run_legacy_backtest(close, high, low, fast_ma, slow_ma, adx, rsi,
                                    strat, engine_id, use_adx, use_rsi)


# ============================================================
# MULTIPROCESSING
# ============================================================
_SHARED_DATA = {}

def _pool_init(shared_dict):
    global _SHARED_DATA
    _SHARED_DATA = shared_dict

def worker_task(args):
    strat, engine_id, _ = args
    tf = strat[5]
    ohlcv_df = _SHARED_DATA[tf]
    return run_backtest(ohlcv_df, strat, engine_id)


# ============================================================
# MAIN
# ============================================================
def main():
    global _SHARED_DATA

    print("=" * 80)
    print("  6-ENGINE CROSS-VERIFICATION BACKTEST v2")
    print("  15 Strategies x 6 Engines = 90 Backtests")
    print("  v32.2 (EMA100/EMA600) + v32.3 (EMA75/SMA750) with full spec")
    print("=" * 80)
    print()

    t_start = time.time()

    # --- Load data ---
    print("[1/4] Loading 5m data...")
    df_5m = load_5m_data()
    print(f"      Loaded {len(df_5m):,} rows of 5m data")

    # --- Resample ---
    print("[2/4] Resampling to needed timeframes...")
    needed_tfs = set(s[5] for s in STRATEGIES)
    print(f"      Needed: {sorted(needed_tfs)}")

    for tf in needed_tfs:
        _SHARED_DATA[tf] = resample_ohlcv(df_5m, tf)
        print(f"      {tf}: {len(_SHARED_DATA[tf]):,} bars")

    del df_5m

    # --- Build tasks ---
    print("[3/4] Running 90 backtests...")
    tasks = []
    for strat in STRATEGIES:
        for eng_id in range(1, 7):
            tasks.append((strat, eng_id, None))

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

    # --- Reports ---
    print("[4/4] Generating reports...")
    print()

    df_res = pd.DataFrame(results)
    strat_names = [s[0] for s in STRATEGIES]

    # ============================================================
    # RETURN (%) MATRIX
    # ============================================================
    print("=" * 130)
    print("  RETURN (%) MATRIX: 15 Strategies x 6 Engines")
    print("=" * 130)

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

    header = f"{'Strategy':<12}"
    for ename in ENGINE_NAMES:
        header += f" {ename:>17}"
    header += f" {'AVG':>12} {'STD':>10}"
    print(header)
    print("-" * 130)

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
        line += f" {avg_v:>11.1f}% {std_v:>9.1f}"
        print(line)
    print()

    # ============================================================
    # PF MATRIX
    # ============================================================
    print("=" * 130)
    print("  PROFIT FACTOR MATRIX: 15 Strategies x 6 Engines")
    print("=" * 130)

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
    print("-" * 130)
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
    # MDD MATRIX
    # ============================================================
    print("=" * 130)
    print("  MAX DRAWDOWN (%) MATRIX: 15 Strategies x 6 Engines")
    print("=" * 130)

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
    print("-" * 130)
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
    # TRADES MATRIX
    # ============================================================
    print("=" * 130)
    print("  TRADES COUNT MATRIX: 15 Strategies x 6 Engines")
    print("=" * 130)
    header = f"{'Strategy':<12}"
    for ename in ENGINE_NAMES:
        header += f" {ename:>17}"
    print(header)
    print("-" * 130)
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

    ranking_data = []
    for sname in strat_names:
        sdf = df_res[df_res['strategy'] == sname]
        avg_ret = sdf['return_pct'].mean()
        avg_mdd = sdf['mdd_pct'].mean()
        avg_pf = sdf['pf'].mean()
        avg_trades = sdf['trades'].mean()
        profitable_engines = (sdf['return_pct'] > 0).sum()
        loss_engines = (sdf['return_pct'] <= 0).sum()
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
    print("  " + "-" * 78)
    ret_sorted = rank_df.sort_values('avg_return', ascending=False).head(10)
    print(f"  {'Rank':<5} {'Strategy':<12} {'Avg Return%':>14} {'Avg PF':>8} {'Avg MDD%':>10} {'Prof/6':>7} {'Trades':>7}")
    print("  " + "-" * 78)
    for rank, (_, row) in enumerate(ret_sorted.iterrows(), 1):
        print(f"  {rank:<5} {row['strategy']:<12} {row['avg_return']:>13.1f}% {row['avg_pf']:>8.2f} {row['avg_mdd']:>9.1f}% {int(row['profitable_engines']):>5}/6 {row['avg_trades']:>7.0f}")
    print()

    # --- STABILITY BEST 10 ---
    print("  [B] STABILITY BEST 10 (lowest avg MDD% where profitable)")
    print("  " + "-" * 78)
    stable_df = rank_df[rank_df['profitable_engines'] >= 1].sort_values('avg_mdd_profitable').head(10)
    print(f"  {'Rank':<5} {'Strategy':<12} {'MDD%(prof)':>12} {'Avg Return%':>14} {'Avg PF':>8} {'Prof/6':>7}")
    print("  " + "-" * 78)
    for rank, (_, row) in enumerate(stable_df.iterrows(), 1):
        print(f"  {rank:<5} {row['strategy']:<12} {row['avg_mdd_profitable']:>11.1f}% {row['avg_return']:>13.1f}% {row['avg_pf']:>8.2f} {int(row['profitable_engines']):>5}/6")
    print()

    # --- DISCARD 10 ---
    print("  [C] DISCARD 10 (worst performance)")
    print("  " + "-" * 78)
    rank_df['discard_score'] = -rank_df['avg_return'] + (rank_df['loss_engines'] >= 4).astype(int) * 100000
    discard_sorted = rank_df.sort_values('discard_score', ascending=False).head(10)
    print(f"  {'Rank':<5} {'Strategy':<12} {'Avg Return%':>14} {'Avg PF':>8} {'Loss/6':>7} {'Reason':<30}")
    print("  " + "-" * 78)
    for rank, (_, row) in enumerate(discard_sorted.iterrows(), 1):
        reason = "Majority engines loss" if row['loss_engines'] >= 4 else "Worst avg return"
        print(f"  {rank:<5} {row['strategy']:<12} {row['avg_return']:>13.1f}% {row['avg_pf']:>8.2f} {int(row['loss_engines']):>5}/6 {reason:<30}")
    print()

    # ============================================================
    # SAVE FILES
    # ============================================================
    out_dir = DATA_DIR

    # CSV
    csv_path = os.path.join(out_dir, "cross_verify2_results.csv")
    df_res.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {csv_path}")

    # Report TXT
    txt_path = os.path.join(out_dir, "cross_verify2_report.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  6-ENGINE CROSS-VERIFICATION BACKTEST v2 REPORT\n")
        f.write(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  15 Strategies x 6 Engines = 90 Backtests\n")
        f.write(f"  Initial Capital: ${INITIAL_CAPITAL:,.0f}\n")
        f.write(f"  Fee: {FEE_RATE*100:.2f}%\n")
        f.write("=" * 80 + "\n\n")

        f.write("STRATEGIES:\n")
        f.write("  v32.2: EMA(100)/EMA(600) 30m — RSI(10) 40-80, ADX>=30 rise6, gap>=0.2%,\n")
        f.write("         monitor24, skip-same-dir, SL3%, TA12%, TSL9%, margin35%, lev10x\n")
        f.write("  v32.3: EMA(75)/SMA(750) 30m  — RSI(11) 40-80, ADX>=30 rise6, gap>=0.2%,\n")
        f.write("         monitor24, skip-same-dir, SL3%, TA12%, TSL9%, margin35%, lev10x\n")
        f.write("  v22.7 - v25.2C: Legacy strategies (unchanged from v1)\n\n")

        f.write("ENGINES:\n")
        descs = {
            1: "Manual Wilder smoothing for ADX/RSI (standard)",
            2: "ewm(alpha=1/period, adjust=False) for ADX/RSI",
            3: "ewm(span=period, adjust=False) for ADX/RSI",
            4: "Wilder but seed=SMA of first N, then Wilder",
            5: "Remove ADX filter entirely, keep only MA cross + RSI",
            6: "Remove RSI filter entirely, keep only MA cross + ADX",
        }
        for i, ename in enumerate(ENGINE_NAMES, 1):
            f.write(f"  Engine {i}: {ename} - {descs[i]}\n")
        f.write("\n")

        # Return matrix
        f.write("RETURN (%) MATRIX:\n")
        f.write(f"{'Strategy':<12}")
        for ename in ENGINE_NAMES:
            f.write(f" {ename:>17}")
        f.write(f" {'AVG':>12}\n")
        f.write("-" * 140 + "\n")
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
            f.write(f" {avg_v:>11.1f}%\n")
        f.write("\n")

        # MDD matrix
        f.write("MDD (%) MATRIX:\n")
        f.write(f"{'Strategy':<12}")
        for ename in ENGINE_NAMES:
            f.write(f" {ename:>17}")
        f.write(f" {'AVG':>12}\n")
        f.write("-" * 140 + "\n")
        for sname in strat_names:
            f.write(f"{sname:<12}")
            vals_list = []
            for ename in ENGINE_NAMES:
                v = mdd_matrix[sname].get(ename, np.nan)
                if np.isnan(v):
                    f.write(f" {'N/A':>17}")
                else:
                    f.write(f" {v:>16.1f}%")
                    vals_list.append(v)
            avg_v = np.mean(vals_list) if vals_list else np.nan
            f.write(f" {avg_v:>11.1f}%\n")
        f.write("\n")

        # Rankings
        f.write("RETURN BEST 10:\n")
        for rank, (_, row) in enumerate(ret_sorted.iterrows(), 1):
            f.write(f"  {rank}. {row['strategy']:<12} Avg Return: {row['avg_return']:>12.1f}%  PF: {row['avg_pf']:.2f}  MDD: {row['avg_mdd']:.1f}%  Profitable: {int(row['profitable_engines'])}/6\n")
        f.write("\n")

        f.write("STABILITY BEST 10:\n")
        for rank, (_, row) in enumerate(stable_df.iterrows(), 1):
            f.write(f"  {rank}. {row['strategy']:<12} MDD(prof): {row['avg_mdd_profitable']:>8.1f}%  Avg Return: {row['avg_return']:>12.1f}%  PF: {row['avg_pf']:.2f}\n")
        f.write("\n")

        f.write("DISCARD 10:\n")
        for rank, (_, row) in enumerate(discard_sorted.iterrows(), 1):
            reason = "Majority engines loss" if row['loss_engines'] >= 4 else "Worst avg return"
            f.write(f"  {rank}. {row['strategy']:<12} Avg Return: {row['avg_return']:>12.1f}%  Loss: {int(row['loss_engines'])}/6  Reason: {reason}\n")
        f.write("\n")

        # Detailed
        f.write("=" * 100 + "\n")
        f.write("DETAILED RESULTS (all 90 backtests):\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Strategy':<12} {'Engine':<20} {'Balance':>14} {'Return%':>12} {'PF':>8} {'MDD%':>8} {'Trades':>7} {'Wins':>6} {'SL':>5}\n")
        f.write("-" * 100 + "\n")
        for sname in strat_names:
            for eid in range(1, 7):
                mask = (df_res['strategy'] == sname) & (df_res['engine_id'] == eid)
                if mask.any():
                    r = df_res[mask].iloc[0]
                    f.write(f"{r['strategy']:<12} {r['engine_name']:<20} ${r['final_balance']:>13,.2f} {r['return_pct']:>11.1f}% {r['pf']:>8.2f} {r['mdd_pct']:>7.1f}% {r['trades']:>7d} {r['wins']:>6d} {r['sl_hits']:>5d}\n")
            f.write("\n")

    print(f"  Saved: {txt_path}")
    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
