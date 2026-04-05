"""
v28 6-Engine Cross-Validation Backtest
======================================
6개의 독립적 백테스트 엔진으로 전략 일관성(robustness)을 검증합니다.

Engine 1: Numba-Close   - 원본 Numba JIT, SL/TSL를 종가 기준으로 체크
Engine 2: Numba-HL      - Numba JIT, SL를 고가/저가(intrabar)로 체크, TSL은 종가
Engine 3: Pandas-Close   - 순수 pandas/numpy, SL/TSL를 종가 기준
Engine 4: NextBar        - 다음 봉 시가(next bar open)로 진입
Engine 5: Wilder-ADX     - ADX seed=MEAN (v16.4 fix), 나머지 Engine 1과 동일
Engine 6: StdEMA-ADX     - ADX를 표준 EMA로 계산 (Wilder 아님)

결과: 전략 x 엔진 매트릭스 + 일관성 판정
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

# =============================================================================
# Import from existing engine
# =============================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v28_backtest_engine import (
    load_5m_data, resample_ohlcv, build_multitf_data,
    calc_ema, calc_wma, calc_sma, calc_hma, calc_vwma,
    calc_rsi_wilder, calc_adx_wilder, calc_ma,
    backtest_core,
)
from numba import njit

INITIAL_CAPITAL = 3000.0
FEE_RATE = 0.0004  # 0.04% x 2 sides


# =============================================================================
# Strategy definitions
# =============================================================================
strategies = [
    {"name": "v28_T1", "tf": "15m", "fast_type": 1, "fast_p": 5, "slow_type": 4, "slow_p": 300,
     "adx_p": 20, "adx_t": 35, "rsi_lo": 35, "rsi_hi": 75, "sl": -8, "trail_act": 10, "trail_pct": 5,
     "margin": 15, "lev": 15, "delay": 2, "offset": 0.0, "skip": 0},
    {"name": "v28_T2", "tf": "15m", "fast_type": 3, "fast_p": 14, "slow_type": 4, "slow_p": 300,
     "adx_p": 20, "adx_t": 35, "rsi_lo": 35, "rsi_hi": 70, "sl": -7, "trail_act": 10, "trail_pct": 3,
     "margin": 50, "lev": 10, "delay": 3, "offset": -1.5, "skip": 0},
    {"name": "v28_T3", "tf": "15m", "fast_type": 4, "fast_p": 2, "slow_type": 4, "slow_p": 300,
     "adx_p": 14, "adx_t": 45, "rsi_lo": 30, "rsi_hi": 75, "sl": -9, "trail_act": 5, "trail_pct": 5,
     "margin": 40, "lev": 10, "delay": 0, "offset": 0.0, "skip": 0},
    {"name": "v26_TB", "tf": "30m", "fast_type": 1, "fast_p": 3, "slow_type": 0, "slow_p": 200,
     "adx_p": 14, "adx_t": 35, "rsi_lo": 30, "rsi_hi": 70, "sl": -8, "trail_act": 3, "trail_pct": 2,
     "margin": 30, "lev": 10, "delay": 0, "offset": 0.0, "skip": 1},
    {"name": "v22_0F", "tf": "30m", "fast_type": 1, "fast_p": 3, "slow_type": 0, "slow_p": 200,
     "adx_p": 20, "adx_t": 35, "rsi_lo": 30, "rsi_hi": 70, "sl": -8, "trail_act": 3, "trail_pct": 2,
     "margin": 50, "lev": 10, "delay": 5, "offset": 0.0, "skip": 1},
    {"name": "v16_4", "tf": "30m", "fast_type": 1, "fast_p": 3, "slow_type": 0, "slow_p": 200,
     "adx_p": 20, "adx_t": 35, "rsi_lo": 35, "rsi_hi": 65, "sl": -8, "trail_act": 4, "trail_pct": 3,
     "margin": 30, "lev": 10, "delay": 0, "offset": 0.0, "skip": 0},
    {"name": "v32_1A", "tf": "30m", "fast_type": 0, "fast_p": 100, "slow_type": 0, "slow_p": 600,
     "adx_p": 20, "adx_t": 30, "rsi_lo": 35, "rsi_hi": 75, "sl": -3, "trail_act": 12, "trail_pct": 9,
     "margin": 35, "lev": 10, "delay": 0, "offset": 0.0, "skip": 1},
    {"name": "v22_8", "tf": "30m", "fast_type": 0, "fast_p": 100, "slow_type": 0, "slow_p": 600,
     "adx_p": 20, "adx_t": 30, "rsi_lo": 35, "rsi_hi": 75, "sl": -8, "trail_act": 6, "trail_pct": 5,
     "margin": 35, "lev": 10, "delay": 0, "offset": 0.0, "skip": 1},
    {"name": "v15_5", "tf": "30m", "fast_type": 0, "fast_p": 3, "slow_type": 0, "slow_p": 200,
     "adx_p": 14, "adx_t": 35, "rsi_lo": 35, "rsi_hi": 65, "sl": -7, "trail_act": 6, "trail_pct": 5,
     "margin": 35, "lev": 10, "delay": 0, "offset": 0.0, "skip": 0},
    {"name": "v25_0", "tf": "5m", "fast_type": 0, "fast_p": 5, "slow_type": 0, "slow_p": 100,
     "adx_p": 14, "adx_t": 30, "rsi_lo": 40, "rsi_hi": 60, "sl": -4, "trail_act": 5, "trail_pct": 3,
     "margin": 30, "lev": 10, "delay": 0, "offset": 0.0, "skip": 0},
    {"name": "v12_3", "tf": "5m", "fast_type": 0, "fast_p": 7, "slow_type": 0, "slow_p": 100,
     "adx_p": 14, "adx_t": 30, "rsi_lo": 30, "rsi_hi": 58, "sl": -9, "trail_act": 8, "trail_pct": 6,
     "margin": 20, "lev": 10, "delay": 0, "offset": 0.0, "skip": 0},
    {"name": "v23_5", "tf": "10m", "fast_type": 0, "fast_p": 3, "slow_type": 2, "slow_p": 200,
     "adx_p": 14, "adx_t": 35, "rsi_lo": 40, "rsi_hi": 75, "sl": -10, "trail_act": 8, "trail_pct": 4,
     "margin": 25, "lev": 3, "delay": 5, "offset": 0.0, "skip": 0},
    {"name": "v14_4", "tf": "30m", "fast_type": 0, "fast_p": 3, "slow_type": 0, "slow_p": 200,
     "adx_p": 14, "adx_t": 35, "rsi_lo": 30, "rsi_hi": 65, "sl": -7, "trail_act": 6, "trail_pct": 3,
     "margin": 25, "lev": 10, "delay": 0, "offset": 0.0, "skip": 0},
    {"name": "v16_6", "tf": "30m", "fast_type": 1, "fast_p": 3, "slow_type": 0, "slow_p": 200,
     "adx_p": 20, "adx_t": 35, "rsi_lo": 30, "rsi_hi": 70, "sl": -8, "trail_act": 3, "trail_pct": 2,
     "margin": 50, "lev": 10, "delay": 5, "offset": 0.0, "skip": 1},
    {"name": "v22_1B", "tf": "30m", "fast_type": 1, "fast_p": 3, "slow_type": 0, "slow_p": 200,
     "adx_p": 20, "adx_t": 35, "rsi_lo": 35, "rsi_hi": 60, "sl": -8, "trail_act": 4, "trail_pct": 1,
     "margin": 50, "lev": 10, "delay": 0, "offset": 0.0, "skip": 0},
]


# =============================================================================
# ADX with Standard EMA (Engine 6)
# =============================================================================
@njit
def calc_adx_standard_ema(high, low, close, period):
    """ADX using standard EMA (k=2/(p+1)) instead of Wilder's smoothing"""
    n = len(close)
    adx = np.empty(n)
    adx[:] = np.nan
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        h_diff = high[i] - high[i - 1]
        l_diff = low[i - 1] - low[i]
        if h_diff > l_diff and h_diff > 0:
            plus_dm[i] = h_diff
        if l_diff > h_diff and l_diff > 0:
            minus_dm[i] = l_diff
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, max(tr2, tr3))

    if n < 2 * period + 1:
        return adx

    # Standard EMA smoothing: k = 2/(period+1)
    k = 2.0 / (period + 1.0)

    # Seed with SMA
    atr = np.mean(tr[1:period + 1])
    a_plus = np.mean(plus_dm[1:period + 1])
    a_minus = np.mean(minus_dm[1:period + 1])

    plus_di_arr = np.zeros(n)
    minus_di_arr = np.zeros(n)
    dx_arr = np.zeros(n)

    if atr > 0:
        plus_di_arr[period] = 100.0 * a_plus / atr
        minus_di_arr[period] = 100.0 * a_minus / atr
    di_sum = plus_di_arr[period] + minus_di_arr[period]
    if di_sum > 0:
        dx_arr[period] = 100.0 * abs(plus_di_arr[period] - minus_di_arr[period]) / di_sum

    for i in range(period + 1, n):
        # Standard EMA smoothing
        atr = tr[i] * k + atr * (1.0 - k)
        a_plus = plus_dm[i] * k + a_plus * (1.0 - k)
        a_minus = minus_dm[i] * k + a_minus * (1.0 - k)
        if atr > 0:
            plus_di_arr[i] = 100.0 * a_plus / atr
            minus_di_arr[i] = 100.0 * a_minus / atr
        di_sum = plus_di_arr[i] + minus_di_arr[i]
        if di_sum > 0:
            dx_arr[i] = 100.0 * abs(plus_di_arr[i] - minus_di_arr[i]) / di_sum

    # ADX = Standard EMA of DX
    adx_start = 2 * period
    if adx_start >= n:
        return adx
    adx[adx_start] = np.mean(dx_arr[period:adx_start + 1])
    for i in range(adx_start + 1, n):
        adx[i] = dx_arr[i] * k + adx[i - 1] * (1.0 - k)

    return adx


# =============================================================================
# Engine 2: Numba-HL - SL check on high/low (intrabar), TSL on close
# =============================================================================
@njit
def backtest_engine2_hl(
    close, high, low, volume, timestamps_epoch,
    fast_ma, slow_ma, adx, rsi,
    adx_thresh, rsi_lo, rsi_hi,
    sl_pct, trail_act_pct, trail_pct,
    margin_pct, leverage,
    entry_delay_bars, entry_offset_pct,
    fee_rate, skip_same_dir, weight_start_idx
):
    """Engine 2: SL checked on high/low (intrabar), TSL on close"""
    n = len(close)
    balance = 3000.0
    peak_balance = 3000.0
    max_dd = 0.0

    position = 0
    entry_price = 0.0
    position_size = 0.0
    position_margin = 0.0
    highest_roi = 0.0
    trail_active = False
    trail_sl_price = 0.0

    pending_signal = 0
    pending_bar = 0
    pending_price = 0.0
    last_closed_dir = 0

    total_trades = 0
    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0
    sl_count = 0
    tsl_count = 0
    rev_count = 0
    consec_loss = 0
    max_consec_loss = 0

    month_start_balance = 3000.0
    current_month = -1

    yearly_start = np.zeros(7)
    yearly_end = np.zeros(7)
    for i in range(7):
        yearly_start[i] = -1.0

    for i in range(1, n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            continue

        cur_price = close[i]

        approx_month = i // 8640
        if approx_month != current_month:
            current_month = approx_month
            month_start_balance = balance

        if balance < month_start_balance * 0.80:
            if position != 0:
                if position == 1:
                    roi = (cur_price - entry_price) / entry_price
                else:
                    roi = (entry_price - cur_price) / entry_price
                pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
                balance += pnl
                if pnl > 0:
                    total_profit += pnl
                    wins += 1
                else:
                    total_loss += abs(pnl)
                    losses += 1
                total_trades += 1
                position = 0
            continue

        # Position management - KEY DIFFERENCE: use high/low for SL check
        if position != 0:
            # Intrabar SL check using high/low
            if position == 1:
                # For long, worst case is low price
                worst_roi = (low[i] - entry_price) / entry_price
                roi = (cur_price - entry_price) / entry_price
            else:
                # For short, worst case is high price
                worst_roi = (entry_price - high[i]) / entry_price
                roi = (entry_price - cur_price) / entry_price

            # SL check on intrabar extreme (high/low)
            if worst_roi <= sl_pct / 100.0:
                # Use SL level as exit price, not close
                sl_roi = sl_pct / 100.0
                pnl = position_margin * leverage * sl_roi - position_margin * leverage * fee_rate * 2
                balance += pnl
                total_loss += abs(pnl)
                losses += 1
                sl_count += 1
                total_trades += 1
                consec_loss += 1
                if consec_loss > max_consec_loss:
                    max_consec_loss = consec_loss
                last_closed_dir = position
                position = 0
                trail_active = False
                continue

            # Trailing activation and TSL check on CLOSE price
            if roi >= trail_act_pct / 100.0:
                trail_active = True
                if roi > highest_roi:
                    highest_roi = roi

            if trail_active and highest_roi > 0:
                drop = highest_roi - roi
                if drop >= trail_pct / 100.0:
                    pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
                    balance += pnl
                    if pnl > 0:
                        total_profit += pnl
                        wins += 1
                        consec_loss = 0
                    else:
                        total_loss += abs(pnl)
                        losses += 1
                        consec_loss += 1
                        if consec_loss > max_consec_loss:
                            max_consec_loss = consec_loss
                    tsl_count += 1
                    total_trades += 1
                    last_closed_dir = position
                    position = 0
                    trail_active = False
                    continue

            if roi > highest_roi:
                highest_roi = roi

        # Cross detection
        cross_signal = 0
        if fast_ma[i - 1] <= slow_ma[i - 1] and fast_ma[i] > slow_ma[i]:
            cross_signal = 1
        elif fast_ma[i - 1] >= slow_ma[i - 1] and fast_ma[i] < slow_ma[i]:
            cross_signal = -1

        if cross_signal != 0:
            if position != 0 and position != cross_signal:
                if position == 1:
                    roi = (cur_price - entry_price) / entry_price
                else:
                    roi = (entry_price - cur_price) / entry_price
                pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
                balance += pnl
                if pnl > 0:
                    total_profit += pnl
                    wins += 1
                    consec_loss = 0
                else:
                    total_loss += abs(pnl)
                    losses += 1
                    consec_loss += 1
                    if consec_loss > max_consec_loss:
                        max_consec_loss = consec_loss
                rev_count += 1
                total_trades += 1
                last_closed_dir = position
                position = 0
                trail_active = False

            if adx[i] >= adx_thresh and rsi_lo <= rsi[i] <= rsi_hi:
                if skip_same_dir and cross_signal == last_closed_dir:
                    pass
                else:
                    pending_signal = cross_signal
                    pending_bar = i
                    pending_price = cur_price

        # Delayed entry
        if pending_signal != 0 and position == 0:
            bars_elapsed = i - pending_bar
            if bars_elapsed >= entry_delay_bars:
                if pending_signal == 1:
                    price_change = (cur_price - pending_price) / pending_price * 100.0
                    ok = price_change >= entry_offset_pct
                else:
                    price_change = (pending_price - cur_price) / pending_price * 100.0
                    ok = price_change >= entry_offset_pct

                if ok and adx[i] >= adx_thresh and rsi_lo <= rsi[i] <= rsi_hi:
                    entry_price = cur_price
                    position_margin = balance * margin_pct / 100.0
                    position_size = position_margin * leverage / cur_price
                    position = pending_signal
                    highest_roi = 0.0
                    trail_active = False
                    pending_signal = 0
                elif bars_elapsed > entry_delay_bars + 12:
                    pending_signal = 0

        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100.0
        if dd > max_dd:
            max_dd = dd

        year_idx = i // 105120
        if year_idx < 7:
            if yearly_start[year_idx] < 0:
                yearly_start[year_idx] = balance
            yearly_end[year_idx] = balance

    # Close remaining position
    if position != 0:
        if position == 1:
            roi = (close[n - 1] - entry_price) / entry_price
        else:
            roi = (entry_price - close[n - 1]) / entry_price
        pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
        balance += pnl
        if pnl > 0:
            total_profit += pnl
            wins += 1
        else:
            total_loss += abs(pnl)
            losses += 1
        total_trades += 1

    pf = total_profit / total_loss if total_loss > 0 else 999.0

    return (balance, total_trades, wins, losses, pf, max_dd,
            sl_count, tsl_count, rev_count, max_consec_loss,
            total_profit, total_loss,
            yearly_start[0], yearly_end[0],
            yearly_start[1], yearly_end[1],
            yearly_start[2], yearly_end[2],
            yearly_start[3], yearly_end[3],
            yearly_start[4], yearly_end[4],
            yearly_start[5], yearly_end[5],
            yearly_start[6], yearly_end[6])


# =============================================================================
# Engine 3: Pandas-Close - Pure pandas/numpy, no Numba
# =============================================================================
def calc_ma_numpy(close, volume, ma_type, period):
    """MA calculation without Numba (numpy only)"""
    n = len(close)
    out = np.full(n, np.nan)

    if ma_type == 0:  # EMA
        if n < period:
            return out
        k = 2.0 / (period + 1.0)
        out[period - 1] = np.mean(close[:period])
        for i in range(period, n):
            out[i] = close[i] * k + out[i - 1] * (1.0 - k)
    elif ma_type == 1:  # WMA
        w_sum = period * (period + 1) / 2.0
        weights = np.arange(1, period + 1, dtype=np.float64)
        for i in range(period - 1, n):
            out[i] = np.sum(close[i - period + 1:i + 1] * weights) / w_sum
    elif ma_type == 2:  # SMA
        for i in range(period - 1, n):
            out[i] = np.mean(close[i - period + 1:i + 1])
    elif ma_type == 3:  # HMA
        half_p = max(int(period / 2), 1)
        sqrt_p = max(int(np.sqrt(period)), 1)
        wma_half = calc_ma_numpy(close, volume, 1, half_p)
        wma_full = calc_ma_numpy(close, volume, 1, period)
        diff = np.full(n, np.nan)
        valid = ~np.isnan(wma_half) & ~np.isnan(wma_full)
        diff[valid] = 2.0 * wma_half[valid] - wma_full[valid]
        out = calc_ma_numpy(diff, volume, 1, sqrt_p)
    elif ma_type == 4:  # VWMA
        for i in range(period - 1, n):
            sl = slice(i - period + 1, i + 1)
            v_sum = np.sum(volume[sl])
            if v_sum > 0:
                out[i] = np.sum(close[sl] * volume[sl]) / v_sum
    return out


def calc_rsi_numpy(data, period):
    """RSI with Wilder's smoothing (numpy)"""
    n = len(data)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses_arr = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses_arr[:period])
    if avg_loss == 0:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses_arr[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            out[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def calc_adx_numpy(high, low, close, period):
    """ADX with Wilder's smoothing (numpy, no Numba)"""
    n = len(close)
    adx = np.full(n, np.nan)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        h_diff = high[i] - high[i - 1]
        l_diff = low[i - 1] - low[i]
        if h_diff > l_diff and h_diff > 0:
            plus_dm[i] = h_diff
        if l_diff > h_diff and l_diff > 0:
            minus_dm[i] = l_diff
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, max(tr2, tr3))

    if n < 2 * period + 1:
        return adx

    atr = np.mean(tr[1:period + 1])
    a_plus = np.mean(plus_dm[1:period + 1])
    a_minus = np.mean(minus_dm[1:period + 1])

    plus_di_arr = np.zeros(n)
    minus_di_arr = np.zeros(n)
    dx_arr = np.zeros(n)

    if atr > 0:
        plus_di_arr[period] = 100.0 * a_plus / atr
        minus_di_arr[period] = 100.0 * a_minus / atr
    di_sum = plus_di_arr[period] + minus_di_arr[period]
    if di_sum > 0:
        dx_arr[period] = 100.0 * abs(plus_di_arr[period] - minus_di_arr[period]) / di_sum

    for i in range(period + 1, n):
        atr = (atr * (period - 1) + tr[i]) / period
        a_plus = (a_plus * (period - 1) + plus_dm[i]) / period
        a_minus = (a_minus * (period - 1) + minus_dm[i]) / period
        if atr > 0:
            plus_di_arr[i] = 100.0 * a_plus / atr
            minus_di_arr[i] = 100.0 * a_minus / atr
        di_sum = plus_di_arr[i] + minus_di_arr[i]
        if di_sum > 0:
            dx_arr[i] = 100.0 * abs(plus_di_arr[i] - minus_di_arr[i]) / di_sum

    adx_start = 2 * period
    if adx_start >= n:
        return adx
    adx[adx_start] = np.mean(dx_arr[period:adx_start + 1])
    for i in range(adx_start + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx_arr[i]) / period
    return adx


def backtest_engine3_pandas(
    close, high, low, volume,
    fast_ma, slow_ma, adx, rsi,
    adx_thresh, rsi_lo, rsi_hi,
    sl_pct, trail_act_pct, trail_pct,
    margin_pct, leverage,
    entry_delay_bars, entry_offset_pct,
    fee_rate, skip_same_dir
):
    """Engine 3: Pure pandas/numpy backtest, SL/TSL on close"""
    n = len(close)
    balance = 3000.0
    peak_balance = 3000.0
    max_dd = 0.0

    position = 0
    entry_price = 0.0
    position_margin = 0.0
    highest_roi = 0.0
    trail_active = False

    pending_signal = 0
    pending_bar = 0
    pending_price = 0.0
    last_closed_dir = 0

    total_trades = 0
    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0
    sl_count = 0
    tsl_count = 0
    rev_count = 0
    consec_loss = 0
    max_consec_loss = 0

    month_start_balance = 3000.0
    current_month = -1

    for i in range(1, n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            continue

        cur_price = close[i]

        approx_month = i // 8640
        if approx_month != current_month:
            current_month = approx_month
            month_start_balance = balance

        if balance < month_start_balance * 0.80:
            if position != 0:
                roi = (cur_price - entry_price) / entry_price if position == 1 else (entry_price - cur_price) / entry_price
                pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
                balance += pnl
                if pnl > 0:
                    total_profit += pnl
                    wins += 1
                else:
                    total_loss += abs(pnl)
                    losses += 1
                total_trades += 1
                position = 0
            continue

        if position != 0:
            roi = (cur_price - entry_price) / entry_price if position == 1 else (entry_price - cur_price) / entry_price

            if roi <= sl_pct / 100.0:
                pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
                balance += pnl
                total_loss += abs(pnl)
                losses += 1
                sl_count += 1
                total_trades += 1
                consec_loss += 1
                if consec_loss > max_consec_loss:
                    max_consec_loss = consec_loss
                last_closed_dir = position
                position = 0
                trail_active = False
                continue

            if roi >= trail_act_pct / 100.0:
                trail_active = True
                if roi > highest_roi:
                    highest_roi = roi

            if trail_active and highest_roi > 0:
                drop = highest_roi - roi
                if drop >= trail_pct / 100.0:
                    pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
                    balance += pnl
                    if pnl > 0:
                        total_profit += pnl
                        wins += 1
                        consec_loss = 0
                    else:
                        total_loss += abs(pnl)
                        losses += 1
                        consec_loss += 1
                        if consec_loss > max_consec_loss:
                            max_consec_loss = consec_loss
                    tsl_count += 1
                    total_trades += 1
                    last_closed_dir = position
                    position = 0
                    trail_active = False
                    continue

            if roi > highest_roi:
                highest_roi = roi

        cross_signal = 0
        if fast_ma[i - 1] <= slow_ma[i - 1] and fast_ma[i] > slow_ma[i]:
            cross_signal = 1
        elif fast_ma[i - 1] >= slow_ma[i - 1] and fast_ma[i] < slow_ma[i]:
            cross_signal = -1

        if cross_signal != 0:
            if position != 0 and position != cross_signal:
                roi = (cur_price - entry_price) / entry_price if position == 1 else (entry_price - cur_price) / entry_price
                pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
                balance += pnl
                if pnl > 0:
                    total_profit += pnl
                    wins += 1
                    consec_loss = 0
                else:
                    total_loss += abs(pnl)
                    losses += 1
                    consec_loss += 1
                    if consec_loss > max_consec_loss:
                        max_consec_loss = consec_loss
                rev_count += 1
                total_trades += 1
                last_closed_dir = position
                position = 0
                trail_active = False

            if adx[i] >= adx_thresh and rsi_lo <= rsi[i] <= rsi_hi:
                if skip_same_dir and cross_signal == last_closed_dir:
                    pass
                else:
                    pending_signal = cross_signal
                    pending_bar = i
                    pending_price = cur_price

        if pending_signal != 0 and position == 0:
            bars_elapsed = i - pending_bar
            if bars_elapsed >= entry_delay_bars:
                if pending_signal == 1:
                    price_change = (cur_price - pending_price) / pending_price * 100.0
                    ok = price_change >= entry_offset_pct
                else:
                    price_change = (pending_price - cur_price) / pending_price * 100.0
                    ok = price_change >= entry_offset_pct

                if ok and adx[i] >= adx_thresh and rsi_lo <= rsi[i] <= rsi_hi:
                    entry_price = cur_price
                    position_margin = balance * margin_pct / 100.0
                    position = pending_signal
                    highest_roi = 0.0
                    trail_active = False
                    pending_signal = 0
                elif bars_elapsed > entry_delay_bars + 12:
                    pending_signal = 0

        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100.0
        if dd > max_dd:
            max_dd = dd

    if position != 0:
        roi = (close[n - 1] - entry_price) / entry_price if position == 1 else (entry_price - close[n - 1]) / entry_price
        pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
        balance += pnl
        if pnl > 0:
            total_profit += pnl
            wins += 1
        else:
            total_loss += abs(pnl)
            losses += 1
        total_trades += 1

    pf = total_profit / total_loss if total_loss > 0 else 999.0
    return (balance, total_trades, wins, losses, pf, max_dd, sl_count, tsl_count, rev_count)


# =============================================================================
# Engine 4: NextBar - Entry on next bar's open
# =============================================================================
@njit
def backtest_engine4_nextbar(
    close, high, low, open_price, volume, timestamps_epoch,
    fast_ma, slow_ma, adx, rsi,
    adx_thresh, rsi_lo, rsi_hi,
    sl_pct, trail_act_pct, trail_pct,
    margin_pct, leverage,
    entry_delay_bars, entry_offset_pct,
    fee_rate, skip_same_dir, weight_start_idx
):
    """Engine 4: Entry at next bar's open instead of current close"""
    n = len(close)
    balance = 3000.0
    peak_balance = 3000.0
    max_dd = 0.0

    position = 0
    entry_price = 0.0
    position_margin = 0.0
    highest_roi = 0.0
    trail_active = False

    pending_signal = 0
    pending_bar = 0
    pending_price = 0.0
    last_closed_dir = 0

    # Deferred entry: signal confirmed, waiting to enter on next bar open
    deferred_entry = 0  # direction to enter on this bar's open

    total_trades = 0
    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0
    sl_count = 0
    tsl_count = 0
    rev_count = 0
    consec_loss = 0
    max_consec_loss = 0

    month_start_balance = 3000.0
    current_month = -1

    yearly_start = np.zeros(7)
    yearly_end = np.zeros(7)
    for idx in range(7):
        yearly_start[idx] = -1.0

    for i in range(1, n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            continue

        cur_price = close[i]

        approx_month = i // 8640
        if approx_month != current_month:
            current_month = approx_month
            month_start_balance = balance

        if balance < month_start_balance * 0.80:
            if position != 0:
                roi = (cur_price - entry_price) / entry_price if position == 1 else (entry_price - cur_price) / entry_price
                pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
                balance += pnl
                if pnl > 0:
                    total_profit += pnl
                    wins += 1
                else:
                    total_loss += abs(pnl)
                    losses += 1
                total_trades += 1
                position = 0
            deferred_entry = 0
            continue

        # Execute deferred entry at this bar's OPEN
        if deferred_entry != 0 and position == 0:
            entry_price = open_price[i]  # Enter at open of this bar
            position_margin = balance * margin_pct / 100.0
            position = deferred_entry
            highest_roi = 0.0
            trail_active = False
            deferred_entry = 0

        # Position management (same as Engine 1, on close)
        if position != 0:
            roi = (cur_price - entry_price) / entry_price if position == 1 else (entry_price - cur_price) / entry_price

            if roi <= sl_pct / 100.0:
                pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
                balance += pnl
                total_loss += abs(pnl)
                losses += 1
                sl_count += 1
                total_trades += 1
                consec_loss += 1
                if consec_loss > max_consec_loss:
                    max_consec_loss = consec_loss
                last_closed_dir = position
                position = 0
                trail_active = False
                continue

            if roi >= trail_act_pct / 100.0:
                trail_active = True
                if roi > highest_roi:
                    highest_roi = roi

            if trail_active and highest_roi > 0:
                drop = highest_roi - roi
                if drop >= trail_pct / 100.0:
                    pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
                    balance += pnl
                    if pnl > 0:
                        total_profit += pnl
                        wins += 1
                        consec_loss = 0
                    else:
                        total_loss += abs(pnl)
                        losses += 1
                        consec_loss += 1
                        if consec_loss > max_consec_loss:
                            max_consec_loss = consec_loss
                    tsl_count += 1
                    total_trades += 1
                    last_closed_dir = position
                    position = 0
                    trail_active = False
                    continue

            if roi > highest_roi:
                highest_roi = roi

        # Cross detection
        cross_signal = 0
        if fast_ma[i - 1] <= slow_ma[i - 1] and fast_ma[i] > slow_ma[i]:
            cross_signal = 1
        elif fast_ma[i - 1] >= slow_ma[i - 1] and fast_ma[i] < slow_ma[i]:
            cross_signal = -1

        if cross_signal != 0:
            if position != 0 and position != cross_signal:
                roi = (cur_price - entry_price) / entry_price if position == 1 else (entry_price - cur_price) / entry_price
                pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
                balance += pnl
                if pnl > 0:
                    total_profit += pnl
                    wins += 1
                    consec_loss = 0
                else:
                    total_loss += abs(pnl)
                    losses += 1
                    consec_loss += 1
                    if consec_loss > max_consec_loss:
                        max_consec_loss = consec_loss
                rev_count += 1
                total_trades += 1
                last_closed_dir = position
                position = 0
                trail_active = False

            if adx[i] >= adx_thresh and rsi_lo <= rsi[i] <= rsi_hi:
                if skip_same_dir and cross_signal == last_closed_dir:
                    pass
                else:
                    pending_signal = cross_signal
                    pending_bar = i
                    pending_price = cur_price

        # Delayed entry check - but defer actual entry to next bar open
        if pending_signal != 0 and position == 0 and deferred_entry == 0:
            bars_elapsed = i - pending_bar
            if bars_elapsed >= entry_delay_bars:
                if pending_signal == 1:
                    price_change = (cur_price - pending_price) / pending_price * 100.0
                    ok = price_change >= entry_offset_pct
                else:
                    price_change = (pending_price - cur_price) / pending_price * 100.0
                    ok = price_change >= entry_offset_pct

                if ok and adx[i] >= adx_thresh and rsi_lo <= rsi[i] <= rsi_hi:
                    # Defer to next bar's open
                    deferred_entry = pending_signal
                    pending_signal = 0
                elif bars_elapsed > entry_delay_bars + 12:
                    pending_signal = 0

        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100.0
        if dd > max_dd:
            max_dd = dd

        year_idx = i // 105120
        if year_idx < 7:
            if yearly_start[year_idx] < 0:
                yearly_start[year_idx] = balance
            yearly_end[year_idx] = balance

    if position != 0:
        roi = (close[n - 1] - entry_price) / entry_price if position == 1 else (entry_price - close[n - 1]) / entry_price
        pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
        balance += pnl
        if pnl > 0:
            total_profit += pnl
            wins += 1
        else:
            total_loss += abs(pnl)
            losses += 1
        total_trades += 1

    pf = total_profit / total_loss if total_loss > 0 else 999.0

    return (balance, total_trades, wins, losses, pf, max_dd,
            sl_count, tsl_count, rev_count, max_consec_loss,
            total_profit, total_loss,
            yearly_start[0], yearly_end[0],
            yearly_start[1], yearly_end[1],
            yearly_start[2], yearly_end[2],
            yearly_start[3], yearly_end[3],
            yearly_start[4], yearly_end[4],
            yearly_start[5], yearly_end[5],
            yearly_start[6], yearly_end[6])


# =============================================================================
# Runner: run a single strategy on a single engine
# =============================================================================
def run_single(engine_id, strat, df, tf_data_dict):
    """
    Run one strategy on one engine.
    Returns: (balance, trades, wins, losses, pf, mdd, sl_count, tsl_count, rev_count)
    """
    tf = strat["tf"]
    if tf in tf_data_dict:
        data = tf_data_dict[tf]
    else:
        data = df  # fallback

    close = data['close'].values.astype(np.float64)
    high = data['high'].values.astype(np.float64)
    low = data['low'].values.astype(np.float64)
    volume = data['volume'].values.astype(np.float64)
    open_price = data['open'].values.astype(np.float64) if 'open' in data.columns else close.copy()
    n = len(close)
    timestamps_epoch = np.zeros(n, dtype=np.float64)

    # Compute indicators based on engine variant
    if engine_id in (1, 2, 4, 5):
        # Numba-based MA
        fast_ma = calc_ma(close, volume, strat["fast_type"], strat["fast_p"])
        slow_ma = calc_ma(close, volume, strat["slow_type"], strat["slow_p"])
        rsi = calc_rsi_wilder(close, 14)
        adx = calc_adx_wilder(high, low, close, strat["adx_p"])
    elif engine_id == 3:
        # Pure numpy MA
        fast_ma = calc_ma_numpy(close, volume, strat["fast_type"], strat["fast_p"])
        slow_ma = calc_ma_numpy(close, volume, strat["slow_type"], strat["slow_p"])
        rsi = calc_rsi_numpy(close, 14)
        adx = calc_adx_numpy(high, low, close, strat["adx_p"])
    elif engine_id == 6:
        # Numba MA but standard EMA ADX
        fast_ma = calc_ma(close, volume, strat["fast_type"], strat["fast_p"])
        slow_ma = calc_ma(close, volume, strat["slow_type"], strat["slow_p"])
        rsi = calc_rsi_wilder(close, 14)
        adx = calc_adx_standard_ema(high, low, close, strat["adx_p"])

    # Common parameters
    adx_t = float(strat["adx_t"])
    rsi_lo = float(strat["rsi_lo"])
    rsi_hi = float(strat["rsi_hi"])
    sl = float(strat["sl"])
    t_act = float(strat["trail_act"])
    t_pct = float(strat["trail_pct"])
    m_pct = float(strat["margin"])
    lev = float(strat["lev"])
    delay = int(strat["delay"])
    offset = float(strat["offset"])
    skip = int(strat["skip"])
    w_start = n // 3

    if engine_id == 1 or engine_id == 5:
        # Engine 1 & 5: Original Numba backtest_core (Engine 5 differs only in ADX calc, done above)
        result = backtest_core(
            close, high, low, volume, timestamps_epoch,
            fast_ma, slow_ma, adx, rsi,
            adx_t, rsi_lo, rsi_hi,
            sl, t_act, t_pct,
            m_pct, lev,
            delay, offset,
            FEE_RATE, skip, w_start
        )
        return (result[0], result[1], result[2], result[3], result[4], result[5],
                result[6], result[7], result[8])

    elif engine_id == 2:
        # Engine 2: Numba-HL
        result = backtest_engine2_hl(
            close, high, low, volume, timestamps_epoch,
            fast_ma, slow_ma, adx, rsi,
            adx_t, rsi_lo, rsi_hi,
            sl, t_act, t_pct,
            m_pct, lev,
            delay, offset,
            FEE_RATE, skip, w_start
        )
        return (result[0], result[1], result[2], result[3], result[4], result[5],
                result[6], result[7], result[8])

    elif engine_id == 3:
        # Engine 3: Pandas-Close
        result = backtest_engine3_pandas(
            close, high, low, volume,
            fast_ma, slow_ma, adx, rsi,
            adx_t, rsi_lo, rsi_hi,
            sl, t_act, t_pct,
            m_pct, lev,
            delay, offset,
            FEE_RATE, skip
        )
        return result  # already (bal, trades, wins, losses, pf, mdd, sl, tsl, rev)

    elif engine_id == 4:
        # Engine 4: NextBar
        result = backtest_engine4_nextbar(
            close, high, low, open_price, volume, timestamps_epoch,
            fast_ma, slow_ma, adx, rsi,
            adx_t, rsi_lo, rsi_hi,
            sl, t_act, t_pct,
            m_pct, lev,
            delay, offset,
            FEE_RATE, skip, w_start
        )
        return (result[0], result[1], result[2], result[3], result[4], result[5],
                result[6], result[7], result[8])

    elif engine_id == 6:
        # Engine 6: StdEMA-ADX (uses backtest_core with different ADX)
        result = backtest_core(
            close, high, low, volume, timestamps_epoch,
            fast_ma, slow_ma, adx, rsi,
            adx_t, rsi_lo, rsi_hi,
            sl, t_act, t_pct,
            m_pct, lev,
            delay, offset,
            FEE_RATE, skip, w_start
        )
        return (result[0], result[1], result[2], result[3], result[4], result[5],
                result[6], result[7], result[8])


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 90)
    print("  v28 6-Engine Cross-Validation Backtest")
    print("  Initial Capital: $3,000 | Fee: 0.04% x2 | Monthly Loss Limit: -20%")
    print("=" * 90)

    base_path = os.path.dirname(os.path.abspath(__file__))

    # ---- Phase 1: Load data ----
    print("\n[PHASE 1] Loading 5m data...")
    t0 = time.time()
    df_5m = load_5m_data(base_path)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ---- Phase 2: Resample ----
    print("\n[PHASE 2] Resampling to multi-timeframe...")
    tf_data = build_multitf_data(df_5m)
    print(f"  Available TFs: {list(tf_data.keys())}")

    # Add 10m if not present
    if '10m' not in tf_data:
        tf_data['10m'] = resample_ohlcv(df_5m, 10)
        print(f"  [DATA] 10m resampled: {len(tf_data['10m'])} rows")

    # ---- Phase 3: Warm up Numba JIT ----
    print("\n[PHASE 3] Warming up Numba JIT (first call compilation)...")
    t0 = time.time()
    _dummy_c = np.random.randn(200).astype(np.float64)
    _dummy_h = _dummy_c + 0.5
    _dummy_l = _dummy_c - 0.5
    _dummy_v = np.abs(np.random.randn(200)).astype(np.float64)
    _dummy_o = _dummy_c.copy()
    _dummy_ts = np.zeros(200, dtype=np.float64)
    _ = calc_ma(_dummy_c, _dummy_v, 0, 5)
    _ = calc_adx_wilder(_dummy_h, _dummy_l, _dummy_c, 14)
    _ = calc_adx_standard_ema(_dummy_h, _dummy_l, _dummy_c, 14)
    _ = calc_rsi_wilder(_dummy_c, 14)
    _fm = calc_ema(_dummy_c, 5)
    _sm = calc_ema(_dummy_c, 20)
    _adx = calc_adx_wilder(_dummy_h, _dummy_l, _dummy_c, 14)
    _rsi = calc_rsi_wilder(_dummy_c, 14)
    _ = backtest_core(_dummy_c, _dummy_h, _dummy_l, _dummy_v, _dummy_ts,
                      _fm, _sm, _adx, _rsi,
                      30.0, 30.0, 70.0, -8.0, 5.0, 3.0, 30.0, 10.0,
                      0, 0.0, 0.0004, 0, 50)
    _ = backtest_engine2_hl(_dummy_c, _dummy_h, _dummy_l, _dummy_v, _dummy_ts,
                            _fm, _sm, _adx, _rsi,
                            30.0, 30.0, 70.0, -8.0, 5.0, 3.0, 30.0, 10.0,
                            0, 0.0, 0.0004, 0, 50)
    _ = backtest_engine4_nextbar(_dummy_c, _dummy_h, _dummy_l, _dummy_o, _dummy_v, _dummy_ts,
                                 _fm, _sm, _adx, _rsi,
                                 30.0, 30.0, 70.0, -8.0, 5.0, 3.0, 30.0, 10.0,
                                 0, 0.0, 0.0004, 0, 50)
    print(f"  JIT warmup done in {time.time() - t0:.1f}s")

    # ---- Phase 4: Run cross-validation ----
    engine_names = {
        1: "Numba-Close",
        2: "Numba-HL",
        3: "Pandas-Close",
        4: "NextBar",
        5: "Wilder-ADX",
        6: "StdEMA-ADX",
    }

    n_strategies = len(strategies)
    n_engines = 6

    # Results matrix: [strategy_idx][engine_id] = (bal, trades, wins, losses, pf, mdd, sl, tsl, rev)
    results = {}

    print(f"\n[PHASE 4] Running {n_strategies} strategies x {n_engines} engines = {n_strategies * n_engines} backtests...")
    print("-" * 90)

    total_runs = n_strategies * n_engines
    run_count = 0

    for s_idx, strat in enumerate(strategies):
        strat_name = strat["name"]
        print(f"\n  [{s_idx + 1}/{n_strategies}] Strategy: {strat_name} ({strat['tf']})", flush=True)
        results[strat_name] = {}

        for eng_id in range(1, 7):
            t0 = time.time()
            try:
                res = run_single(eng_id, strat, df_5m, tf_data)
                results[strat_name][eng_id] = res
                bal, trades, wins, losses, pf, mdd, sl_c, tsl_c, rev_c = res
                elapsed = time.time() - t0
                run_count += 1
                print(f"    Engine {eng_id} ({engine_names[eng_id]:12s}): "
                      f"${bal:>10,.0f} | PF={pf:>6.2f} | MDD={mdd:>5.1f}% | "
                      f"T={trades:>4d} W={wins:>3d} L={losses:>3d} | "
                      f"SL={sl_c:>3d} TSL={tsl_c:>3d} REV={rev_c:>3d} | "
                      f"{elapsed:.1f}s", flush=True)
            except Exception as e:
                run_count += 1
                print(f"    Engine {eng_id} ({engine_names[eng_id]:12s}): ERROR - {e}", flush=True)
                results[strat_name][eng_id] = (3000.0, 0, 0, 0, 0.0, 0.0, 0, 0, 0)

        print(f"  Progress: {run_count}/{total_runs} ({run_count/total_runs*100:.0f}%)", flush=True)

    # ---- Phase 5: Print results table ----
    print("\n")
    print("=" * 140)
    print("  CROSS-VALIDATION RESULTS MATRIX")
    print("=" * 140)

    # Header
    header = f"{'Strategy':<10}"
    for eng_id in range(1, 7):
        short = engine_names[eng_id][:10]
        header += f" | {'Bal':>9s} {'PF':>5s} {'MDD':>5s} {'T':>4s}"
    print(header)
    print("-" * 140)

    for strat in strategies:
        sn = strat["name"]
        line = f"{sn:<10}"
        for eng_id in range(1, 7):
            if eng_id in results[sn]:
                bal, trades, wins, losses, pf, mdd, sl_c, tsl_c, rev_c = results[sn][eng_id]
                line += f" | {bal:>9,.0f} {pf:>5.2f} {mdd:>5.1f} {trades:>4d}"
            else:
                line += f" | {'ERR':>9s} {'--':>5s} {'--':>5s} {'--':>4s}"
        print(line)

    print("-" * 140)

    # ---- Phase 6: Consistency analysis ----
    print("\n")
    print("=" * 90)
    print("  CONSISTENCY ANALYSIS (< 10% balance deviation = CONSISTENT)")
    print("=" * 90)
    print(f"{'Strategy':<10} | {'Mean Bal':>10} | {'Std Dev':>10} | {'CV%':>6} | {'Min Bal':>10} | {'Max Bal':>10} | {'Status':<14}")
    print("-" * 90)

    consistent = []
    inconsistent = []

    for strat in strategies:
        sn = strat["name"]
        balances = []
        for eng_id in range(1, 7):
            if eng_id in results[sn]:
                bal = results[sn][eng_id][0]
                balances.append(bal)

        if len(balances) < 2:
            print(f"{sn:<10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>6} | {'N/A':>10} | {'N/A':>10} | INSUFFICIENT")
            continue

        mean_bal = np.mean(balances)
        std_bal = np.std(balances)
        cv = (std_bal / mean_bal * 100) if mean_bal != 0 else 999.0
        min_bal = np.min(balances)
        max_bal = np.max(balances)

        # Deviation check: (max - min) / mean < 10%
        deviation = (max_bal - min_bal) / mean_bal * 100 if mean_bal != 0 else 999.0

        if deviation < 10.0:
            status = "CONSISTENT"
            consistent.append(sn)
        else:
            status = "INCONSISTENT"
            inconsistent.append(sn)

        print(f"{sn:<10} | ${mean_bal:>9,.0f} | ${std_bal:>9,.0f} | {cv:>5.1f}% | ${min_bal:>9,.0f} | ${max_bal:>9,.0f} | {status}")

    print("-" * 90)

    # ---- Phase 7: Summary ----
    print(f"\n{'=' * 90}")
    print("  SUMMARY")
    print(f"{'=' * 90}")
    print(f"\n  Total strategies tested: {n_strategies}")
    print(f"  Consistent (< 10% dev): {len(consistent)}")
    print(f"  Inconsistent:           {len(inconsistent)}")

    if consistent:
        print(f"\n  --- CONSISTENT strategies (robust across engine variants) ---")
        for sn in consistent:
            balances = [results[sn][e][0] for e in range(1, 7) if e in results[sn]]
            pfs = [results[sn][e][4] for e in range(1, 7) if e in results[sn]]
            mean_bal = np.mean(balances)
            mean_pf = np.mean(pfs)
            tf = [s["tf"] for s in strategies if s["name"] == sn][0]
            print(f"    {sn:10s} ({tf:>4s}) | Mean Bal: ${mean_bal:>10,.0f} | Mean PF: {mean_pf:.2f}")

    if inconsistent:
        print(f"\n  --- INCONSISTENT strategies (sensitive to implementation) ---")
        for sn in inconsistent:
            balances = [results[sn][e][0] for e in range(1, 7) if e in results[sn]]
            mean_bal = np.mean(balances)
            dev = (max(balances) - min(balances)) / mean_bal * 100 if mean_bal != 0 else 999.0
            tf = [s["tf"] for s in strategies if s["name"] == sn][0]
            print(f"    {sn:10s} ({tf:>4s}) | Deviation: {dev:.1f}% | Range: ${min(balances):,.0f} ~ ${max(balances):,.0f}")

    # ---- Phase 8: Engine comparison ----
    print(f"\n{'=' * 90}")
    print("  ENGINE COMPARISON (average across all strategies)")
    print(f"{'=' * 90}")
    print(f"  {'Engine':<16} | {'Avg Bal':>10} | {'Avg PF':>7} | {'Avg MDD':>7} | {'Avg Trades':>10}")
    print(f"  {'-'*16}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}")

    for eng_id in range(1, 7):
        bals = []
        pfs = []
        mdds = []
        trades_list = []
        for strat in strategies:
            sn = strat["name"]
            if eng_id in results[sn]:
                bal, trades, wins, losses, pf, mdd, sl_c, tsl_c, rev_c = results[sn][eng_id]
                bals.append(bal)
                pfs.append(min(pf, 100.0))  # cap PF for averaging
                mdds.append(mdd)
                trades_list.append(trades)
        if bals:
            print(f"  {engine_names[eng_id]:<16} | ${np.mean(bals):>9,.0f} | {np.mean(pfs):>6.2f} | {np.mean(mdds):>6.1f}% | {np.mean(trades_list):>9.0f}")

    print(f"\n{'=' * 90}")
    print("  Cross-validation complete.")
    print(f"{'=' * 90}")


if __name__ == '__main__':
    main()
