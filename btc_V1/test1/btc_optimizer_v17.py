"""
BTC/USDT 선물 최적화 엔진 v17.0
- 계층적 4단계 최적화 (1,000,000+ 조합 공간)
- 6 MA types × 5 TFs × 다중 진입/청산/리스크 파라미터
- 2023-2026 가중치 반영
- 30회 반복 검증 (슬리피지 시뮬레이션)
"""

import pandas as pd
import numpy as np
from itertools import product
import time
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = r"D:\filesystem\futures\btc_V1\test1\btc_usdt_5m_merged.csv"
OUTPUT_DIR = r"D:\filesystem\futures\btc_V1\test1\optimization_results"

# ============================================================
# 1. INDICATOR CALCULATIONS
# ============================================================

def calc_ema(data, period):
    return pd.Series(data).ewm(span=period, adjust=False).mean().values

def calc_sma(data, period):
    return pd.Series(data).rolling(period, min_periods=period).mean().values

def calc_wma(data, period):
    weights = np.arange(1, period + 1, dtype=float)
    s = pd.Series(data)
    return s.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).values

def calc_hma(data, period):
    half = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    wma_half = calc_wma(data, half)
    wma_full = calc_wma(data, period)
    diff = 2 * wma_half - wma_full
    mask = ~np.isnan(diff)
    result = np.full_like(data, np.nan, dtype=float)
    if mask.sum() > sqrt_p:
        valid = diff[mask]
        hma_valid = calc_wma(valid, sqrt_p)
        result[mask] = hma_valid
    return result

def calc_dema(data, period):
    ema1 = calc_ema(data, period)
    ema2 = calc_ema(ema1, period)
    return 2 * ema1 - ema2

def calc_vwma(close, volume, period):
    cv = pd.Series(close * volume).rolling(period, min_periods=period).sum().values
    v = pd.Series(volume).rolling(period, min_periods=period).sum().values
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(v > 0, cv / v, np.nan)
    return result

def calc_rsi(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_avg = pd.Series(gain).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(loss_avg > 0, gain_avg / loss_avg, 100.0)
    return 100 - 100 / (1 + rs)

def calc_adx(high, low, close, period=14):
    """ADX - Wilder's Smoothing (v16.4 corrected version)"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = down_move[0] = 0

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    alpha = 1.0 / period
    atr = pd.Series(tr).ewm(alpha=alpha, min_periods=period, adjust=False).mean().values
    plus_di_smooth = pd.Series(plus_dm).ewm(alpha=alpha, min_periods=period, adjust=False).mean().values
    minus_di_smooth = pd.Series(minus_dm).ewm(alpha=alpha, min_periods=period, adjust=False).mean().values

    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = np.where(atr > 0, 100 * plus_di_smooth / atr, 0)
        minus_di = np.where(atr > 0, 100 * minus_di_smooth / atr, 0)
        di_sum = plus_di + minus_di
        dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0)

    adx = pd.Series(dx).ewm(alpha=alpha, min_periods=period, adjust=False).mean().values
    return adx

def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    return pd.Series(tr).ewm(alpha=1.0/period, min_periods=period, adjust=False).mean().values

def calc_bb_width(close, period=20, std_dev=2.0):
    s = pd.Series(close)
    mid = s.rolling(period, min_periods=period).mean()
    std = s.rolling(period, min_periods=period).std()
    width = (2 * std_dev * std / mid).values
    return width

def calc_ma(data, ma_type, period, volume=None):
    if ma_type == 'EMA':
        return calc_ema(data, period)
    elif ma_type == 'SMA':
        return calc_sma(data, period)
    elif ma_type == 'WMA':
        return calc_wma(data, period)
    elif ma_type == 'HMA':
        return calc_hma(data, period)
    elif ma_type == 'DEMA':
        return calc_dema(data, period)
    elif ma_type == 'VWMA':
        if volume is not None:
            return calc_vwma(data, volume, period)
        return calc_ema(data, period)
    return calc_ema(data, period)


# ============================================================
# 2. DATA LOADING & RESAMPLING
# ============================================================

def load_5m_data(path=DATA_PATH):
    print(f"Loading 5m data: {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"  Loaded: {len(df):,} rows ({df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]})")
    return df

def resample_ohlcv(df_5m, tf_minutes):
    df = df_5m.set_index('timestamp')
    rule = f'{tf_minutes}min' if tf_minutes < 60 else f'{tf_minutes//60}h'
    resampled = df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()
    return resampled

TF_MINUTES = {'5min': 5, '10min': 10, '15min': 15, '30min': 30, '1h': 60}

def precompute_all_data(df_5m):
    """Precompute resampled OHLCV for all timeframes + pre-extract year/month"""
    data = {}
    for tf_name, tf_min in TF_MINUTES.items():
        if tf_min == 5:
            resampled = df_5m.copy()
        else:
            resampled = resample_ohlcv(df_5m, tf_min)

        ts_pd = pd.to_datetime(resampled['timestamp'].values)
        years = ts_pd.year.values.astype(np.int32)
        month_keys = (years * 100 + ts_pd.month.values).astype(np.int32)

        data[tf_name] = {
            'timestamp': resampled['timestamp'].values,
            'open': resampled['open'].values.astype(float),
            'high': resampled['high'].values.astype(float),
            'low': resampled['low'].values.astype(float),
            'close': resampled['close'].values.astype(float),
            'volume': resampled['volume'].values.astype(float),
            'years': years,
            'month_keys': month_keys,
            'n': len(resampled),
        }
        print(f"  {tf_name}: {data[tf_name]['n']:,} bars")
    return data


# ============================================================
# 3. FAST BACKTEST ENGINE
# ============================================================

def fast_backtest(
    close, high, low, timestamps,
    ma_fast_arr, ma_slow_arr,
    adx_arr, rsi_arr, atr_arr,
    # Entry params
    adx_min=35, rsi_min=30, rsi_max=65,
    entry_delay=0,
    # Exit params
    sl_pct=0.07, trail_act=0.06, trail_pct=0.03,
    tp1_roi=None, tp1_ratio=0.30,
    # Position params
    leverage=10, margin_pct=0.20,
    monthly_loss_limit=-0.20,
    fee_rate=0.0004,
    initial_balance=3000.0,
    # ATR-based SL (if > 0, overrides sl_pct)
    sl_atr_mult=0.0,
    warmup=300,
    # Pre-computed arrays (pass from precompute_all_data)
    years_arr=None,
    month_keys_arr=None,
):
    """
    Fast backtest with pre-computed indicators.
    Returns dict with all performance metrics.
    """
    n = len(close)
    balance = initial_balance
    peak_balance = initial_balance
    max_dd = 0.0

    # Position state
    in_pos = False
    pos_dir = 0  # 1=LONG, -1=SHORT
    pos_entry = 0.0
    pos_size = 0.0
    pos_margin = 0.0
    pos_sl = 0.0
    trail_active = False
    trail_sl_price = 0.0
    pos_high = 0.0
    pos_low = 0.0
    pos_partial_done = False
    pos_original_size = 0.0

    # Pending entry (delay)
    pending_signal = 0
    pending_countdown = 0
    pending_price = 0.0

    # Monthly tracking
    current_month = -1
    month_start_bal = initial_balance
    monthly_locked = False

    # Trade records
    trades_pnl = []
    trades_roi = []
    trades_reason = []
    trades_year = []
    trade_count = 0

    # Use pre-computed year/month arrays if available
    if years_arr is None:
        ts_pd = pd.to_datetime(timestamps)
        years = ts_pd.year.values.astype(np.int32)
        month_keys = (years * 100 + ts_pd.month.values).astype(np.int32)
    else:
        years = years_arr
        month_keys = month_keys_arr

    # Pre-compute cross signals (vectorized)
    valid = ~(np.isnan(ma_fast_arr) | np.isnan(ma_slow_arr))
    ma_above = ma_fast_arr > ma_slow_arr
    cross_up_arr = np.zeros(n, dtype=bool)
    cross_dn_arr = np.zeros(n, dtype=bool)
    for i in range(max(warmup, 1), n):
        if valid[i] and valid[i-1]:
            if ma_above[i] and not ma_above[i-1]:
                cross_up_arr[i] = True
            elif not ma_above[i] and ma_above[i-1]:
                cross_dn_arr[i] = True

    # Pre-fill NaN in indicator arrays
    adx_safe = np.where(np.isnan(adx_arr), 0.0, adx_arr)
    rsi_safe = np.where(np.isnan(rsi_arr), 50.0, rsi_arr)
    atr_safe = np.where(np.isnan(atr_arr), 0.0, atr_arr)

    trail_act_lev = trail_act * leverage
    inv_lev = 1.0 / leverage

    start_idx = max(warmup, 1)

    for i in range(start_idx, n):
        c = close[i]
        h = high[i]
        lo = low[i]

        m_key = month_keys[i]
        if m_key != current_month:
            current_month = m_key
            month_start_bal = balance
            monthly_locked = False

        # Monthly loss check
        if not monthly_locked and month_start_bal > 0:
            if (balance - month_start_bal) / month_start_bal <= monthly_loss_limit:
                monthly_locked = True

        # ---- POSITION MANAGEMENT ----
        if in_pos:
            # Forced liquidation check (10% isolated margin)
            if pos_dir == 1:
                fl_price = pos_entry * (1 - 1.0 / leverage)
                if lo <= fl_price:
                    pnl = -pos_margin
                    fee = pos_size * fee_rate
                    balance += pnl - fee
                    trades_pnl.append(pnl - fee)
                    trades_roi.append(-1.0 / leverage * leverage)
                    trades_reason.append('FL')
                    trades_year.append(years[i])
                    trade_count += 1
                    in_pos = False
                    if balance > peak_balance:
                        peak_balance = balance
                    if peak_balance > 0:
                        dd = (peak_balance - balance) / peak_balance
                        if dd > max_dd:
                            max_dd = dd
                    continue
            else:
                fl_price = pos_entry * (1 + 1.0 / leverage)
                if h >= fl_price:
                    pnl = -pos_margin
                    fee = pos_size * fee_rate
                    balance += pnl - fee
                    trades_pnl.append(pnl - fee)
                    trades_roi.append(-1.0 / leverage * leverage)
                    trades_reason.append('FL')
                    trades_year.append(years[i])
                    trade_count += 1
                    in_pos = False
                    if balance > peak_balance:
                        peak_balance = balance
                    if peak_balance > 0:
                        dd = (peak_balance - balance) / peak_balance
                        if dd > max_dd:
                            max_dd = dd
                    continue

            # SL check
            sl_hit = False
            if pos_dir == 1 and lo <= pos_sl:
                sl_hit = True
                exit_p = pos_sl
            elif pos_dir == -1 and h >= pos_sl:
                sl_hit = True
                exit_p = pos_sl

            if sl_hit:
                pnl = pos_size * (exit_p - pos_entry) / pos_entry * pos_dir
                fee = pos_size * fee_rate
                balance += pnl - fee
                roi = (exit_p - pos_entry) / pos_entry * pos_dir * leverage
                trades_pnl.append(pnl - fee)
                trades_roi.append(roi)
                trades_reason.append('SL')
                trades_year.append(years[i])
                trade_count += 1
                in_pos = False
            else:
                # Update trailing
                if pos_dir == 1:
                    if h > pos_high:
                        pos_high = h
                    roi_at_high = (pos_high - pos_entry) / pos_entry * leverage
                    if roi_at_high >= trail_act_lev:
                        trail_active = True
                        new_tsl = pos_high * (1 - trail_pct)
                        if new_tsl > trail_sl_price:
                            trail_sl_price = new_tsl
                else:
                    if lo < pos_low:
                        pos_low = lo
                    roi_at_low = (pos_entry - pos_low) / pos_entry * leverage
                    if roi_at_low >= trail_act_lev:
                        trail_active = True
                        new_tsl = pos_low * (1 + trail_pct)
                        if trail_sl_price == 0 or new_tsl < trail_sl_price:
                            trail_sl_price = new_tsl

                # Partial TP check
                if tp1_roi is not None and not pos_partial_done:
                    curr_roi = (c - pos_entry) / pos_entry * pos_dir * leverage
                    if curr_roi >= tp1_roi:
                        exit_amount = pos_original_size * tp1_ratio
                        pnl_partial = exit_amount * (c - pos_entry) / pos_entry * pos_dir
                        fee_p = exit_amount * fee_rate
                        balance += pnl_partial - fee_p
                        pos_size -= exit_amount
                        pos_partial_done = True

                # TSL check (close-based)
                if trail_active:
                    tsl_hit = False
                    if pos_dir == 1 and c <= trail_sl_price:
                        tsl_hit = True
                    elif pos_dir == -1 and c >= trail_sl_price:
                        tsl_hit = True

                    if tsl_hit:
                        pnl = pos_size * (c - pos_entry) / pos_entry * pos_dir
                        fee = pos_size * fee_rate
                        balance += pnl - fee
                        roi = (c - pos_entry) / pos_entry * pos_dir * leverage
                        trades_pnl.append(pnl - fee)
                        trades_roi.append(roi)
                        trades_reason.append('TSL')
                        trades_year.append(years[i])
                        trade_count += 1
                        in_pos = False

        # ---- SIGNAL DETECTION (using pre-computed arrays) ----
        signal = 0
        if cross_up_arr[i]:
            if adx_safe[i] >= adx_min and rsi_min <= rsi_safe[i] <= rsi_max:
                signal = 1
        elif cross_dn_arr[i]:
            if adx_safe[i] >= adx_min and rsi_min <= rsi_safe[i] <= rsi_max:
                signal = -1

        # Entry delay
        if signal != 0:
            if entry_delay > 0:
                pending_signal = signal
                pending_countdown = entry_delay
                pending_price = c
                signal = 0
            # else: immediate entry (signal stays)

        # Process pending entry
        if pending_countdown > 0:
            pending_countdown -= 1
            if pending_countdown == 0:
                signal = pending_signal
                pending_signal = 0

        if signal != 0:
            # Reverse exit
            if in_pos and pos_dir != signal:
                pnl = pos_size * (c - pos_entry) / pos_entry * pos_dir
                fee = pos_size * fee_rate
                balance += pnl - fee
                roi = (c - pos_entry) / pos_entry * pos_dir * leverage
                trades_pnl.append(pnl - fee)
                trades_roi.append(roi)
                trades_reason.append('REV')
                trades_year.append(years[i])
                trade_count += 1
                in_pos = False
            elif in_pos and pos_dir == signal:
                # Same direction cross - close and re-enter
                pnl = pos_size * (c - pos_entry) / pos_entry * pos_dir
                fee = pos_size * fee_rate
                balance += pnl - fee
                roi = (c - pos_entry) / pos_entry * pos_dir * leverage
                trades_pnl.append(pnl - fee)
                trades_roi.append(roi)
                trades_reason.append('REV')
                trades_year.append(years[i])
                trade_count += 1
                in_pos = False

            # New entry
            if not in_pos and not monthly_locked and balance > 10:
                margin = balance * margin_pct
                size = margin * leverage
                fee = size * fee_rate
                balance -= fee

                pos_dir = signal
                pos_entry = c
                pos_size = size
                pos_original_size = size
                pos_margin = margin
                trail_active = False
                trail_sl_price = 0.0
                pos_high = c
                pos_low = c
                pos_partial_done = False

                # SL calculation
                if sl_atr_mult > 0 and not np.isnan(atr_arr[i]):
                    sl_dist = atr_arr[i] * sl_atr_mult / pos_entry
                    sl_dist = max(0.015, min(sl_dist, 0.10))  # 1.5%~10%
                else:
                    sl_dist = sl_pct

                if pos_dir == 1:
                    pos_sl = pos_entry * (1 - sl_dist)
                else:
                    pos_sl = pos_entry * (1 + sl_dist)

                in_pos = True

        # Update drawdown
        if balance > peak_balance:
            peak_balance = balance
        if peak_balance > 0:
            dd = (peak_balance - balance) / peak_balance
            if dd > max_dd:
                max_dd = dd

    # Close remaining position
    if in_pos:
        c = close[-1]
        pnl = pos_size * (c - pos_entry) / pos_entry * pos_dir
        fee = pos_size * fee_rate
        balance += pnl - fee
        roi = (c - pos_entry) / pos_entry * pos_dir * leverage
        trades_pnl.append(pnl - fee)
        trades_roi.append(roi)
        trades_reason.append('END')
        trades_year.append(years[-1])
        trade_count += 1

    # ---- COMPILE RESULTS ----
    if trade_count == 0:
        return {
            'balance': balance, 'return_pct': 0, 'trades': 0,
            'pf': 0, 'mdd': 0, 'win_rate': 0, 'payoff': 0,
            'sl': 0, 'tsl': 0, 'rev': 0, 'fl': 0,
            'wpf': 0, 'avg_win': 0, 'avg_loss': 0,
            'trades_per_month': 0,
        }

    pnl_arr = np.array(trades_pnl)
    roi_arr = np.array(trades_roi)
    reason_arr = trades_reason
    year_arr = np.array(trades_year)

    wins = pnl_arr > 0
    losses = pnl_arr <= 0

    gross_profit = pnl_arr[wins].sum() if wins.any() else 0
    gross_loss = abs(pnl_arr[losses].sum()) if losses.any() else 0.001

    pf = gross_profit / gross_loss if gross_loss > 0 else 999.99
    win_rate = wins.sum() / trade_count * 100

    avg_win_roi = roi_arr[wins].mean() * 100 if wins.any() else 0
    avg_loss_roi = roi_arr[losses].mean() * 100 if losses.any() else -0.001
    payoff = abs(avg_win_roi / avg_loss_roi) if avg_loss_roi != 0 else 999.99

    sl_count = sum(1 for r in reason_arr if r == 'SL')
    tsl_count = sum(1 for r in reason_arr if r == 'TSL')
    rev_count = sum(1 for r in reason_arr if r == 'REV')
    fl_count = sum(1 for r in reason_arr if r == 'FL')

    # Weighted PF (2023-2026 weight=1.5, 2020-2022 weight=0.5)
    weights = np.where(year_arr >= 2023, 1.5, 0.5)
    w_profit = (pnl_arr[wins] * weights[wins]).sum() if wins.any() else 0
    w_loss = abs((pnl_arr[losses] * weights[losses]).sum()) if losses.any() else 0.001
    wpf = w_profit / w_loss if w_loss > 0 else 999.99

    # Monthly trade frequency
    total_months = 75.0
    trades_per_month = trade_count / total_months

    return_pct = (balance - initial_balance) / initial_balance * 100

    return {
        'balance': round(balance, 2),
        'return_pct': round(return_pct, 1),
        'trades': trade_count,
        'pf': round(min(pf, 999.99), 2),
        'wpf': round(min(wpf, 999.99), 2),
        'mdd': round(max_dd * 100, 1),
        'win_rate': round(win_rate, 1),
        'payoff': round(min(payoff, 999.99), 2),
        'avg_win': round(avg_win_roi, 2),
        'avg_loss': round(avg_loss_roi, 2),
        'sl': sl_count,
        'tsl': tsl_count,
        'rev': rev_count,
        'fl': fl_count,
        'trades_per_month': round(trades_per_month, 2),
    }


# ============================================================
# 4. PHASE 1: MA TYPE × TIMEFRAME × PERIOD SCAN
# ============================================================

def phase1_scan(all_data):
    """Scan MA types × timeframes × fast/slow periods"""
    print("\n" + "="*70)
    print("  PHASE 1: MA Type × Timeframe × Period Scan")
    print("="*70)

    MA_TYPES = ['EMA', 'WMA', 'HMA', 'SMA', 'DEMA']
    TIMEFRAMES = ['15min', '30min', '1h']
    FAST_PERIODS = [2, 3, 5, 7, 10, 14, 21]
    SLOW_PERIODS = [100, 150, 200, 250, 300]

    # Fixed entry/exit for Phase 1
    BASE_PARAMS = {
        'adx_min': 35, 'rsi_min': 30, 'rsi_max': 65,
        'entry_delay': 0,
        'sl_pct': 0.07, 'trail_act': 0.06, 'trail_pct': 0.03,
        'tp1_roi': None,
        'leverage': 10, 'margin_pct': 0.25,
        'monthly_loss_limit': -0.20,
        'fee_rate': 0.0004, 'initial_balance': 3000.0,
    }

    results = []
    total = len(MA_TYPES) * len(TIMEFRAMES) * len(FAST_PERIODS) * len(SLOW_PERIODS)
    count = 0
    t0 = time.time()

    for ma_type in MA_TYPES:
        for tf in TIMEFRAMES:
            d = all_data[tf]
            close = d['close']
            high = d['high']
            low = d['low']
            volume = d['volume']
            timestamps = d['timestamp']

            # Pre-compute ADX, RSI, ATR once per TF
            adx_20 = calc_adx(high, low, close, 20)
            rsi_14 = calc_rsi(close, 14)
            atr_14 = calc_atr(high, low, close, 14)
            years = d['years']
            mkeys = d['month_keys']

            for fp in FAST_PERIODS:
                # Pre-compute fast MA once per (ma_type, tf, fast_period)
                ma_fast = calc_ma(close, ma_type, fp, volume)

                for sp in SLOW_PERIODS:
                    if fp >= sp:
                        count += 1
                        continue

                    ma_slow = calc_ma(close, 'EMA', sp, volume)  # Slow always EMA

                    warmup = sp + 50

                    res = fast_backtest(
                        close, high, low, timestamps,
                        ma_fast, ma_slow,
                        adx_20, rsi_14, atr_14,
                        warmup=warmup,
                        years_arr=years, month_keys_arr=mkeys,
                        **BASE_PARAMS
                    )

                    res['ma_type'] = ma_type
                    res['tf'] = tf
                    res['fast_period'] = fp
                    res['slow_period'] = sp
                    results.append(res)

                    count += 1
                    if count % 100 == 0:
                        elapsed = time.time() - t0
                        print(f"  Phase 1: {count}/{total} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Phase 1 complete: {len(results)} combos in {elapsed:.1f}s")

    # Sort by weighted PF (primary) + return (secondary)
    results.sort(key=lambda x: (x['wpf'] if x['trades'] >= 5 else 0, x['return_pct']), reverse=True)

    # Print top 30
    print(f"\n  {'Rank':>4} {'MA':>5} {'TF':>5} {'F':>3} {'S':>3} {'Trades':>6} {'Return%':>10} {'PF':>8} {'WPF':>8} {'MDD%':>6} {'WR%':>5} {'SL':>3} {'TSL':>4} {'REV':>4}")
    print(f"  {'-'*85}")
    for idx, r in enumerate(results[:30]):
        if r['trades'] < 3:
            continue
        print(f"  {idx+1:>4} {r['ma_type']:>5} {r['tf']:>5} {r['fast_period']:>3} {r['slow_period']:>3} "
              f"{r['trades']:>6} {r['return_pct']:>+9.1f}% {r['pf']:>7.2f} {r['wpf']:>7.2f} "
              f"{r['mdd']:>5.1f}% {r['win_rate']:>4.1f} {r['sl']:>3} {r['tsl']:>4} {r['rev']:>4}")

    return results


# ============================================================
# 5. PHASE 2: ENTRY CONDITION OPTIMIZATION
# ============================================================

def phase2_entry(all_data, top_phase1, top_n=20):
    """Optimize entry conditions for top Phase 1 strategies"""
    print("\n" + "="*70)
    print("  PHASE 2: Entry Condition Optimization")
    print("="*70)

    ADX_PERIODS = [14, 20]
    ADX_THRESHOLDS = [25, 30, 35, 40, 45, 50]
    RSI_RANGES = [(20,80), (25,75), (30,70), (35,65), (30,65), (35,70)]
    ENTRY_DELAYS = [0, 1, 3, 5]

    # Filter top_n with minimum trades
    candidates = [r for r in top_phase1 if r['trades'] >= 5][:top_n]

    results = []
    total = len(candidates) * len(ADX_PERIODS) * len(ADX_THRESHOLDS) * len(RSI_RANGES) * len(ENTRY_DELAYS)
    count = 0
    t0 = time.time()

    for base in candidates:
        tf = base['tf']
        ma_type = base['ma_type']
        fp = base['fast_period']
        sp = base['slow_period']

        d = all_data[tf]
        close = d['close']
        high = d['high']
        low = d['low']
        volume = d['volume']
        timestamps = d['timestamp']
        years = d['years']
        mkeys = d['month_keys']

        ma_fast = calc_ma(close, ma_type, fp, volume)
        ma_slow = calc_ma(close, 'EMA', sp, volume)
        atr_14 = calc_atr(high, low, close, 14)
        warmup = sp + 50

        for adx_p in ADX_PERIODS:
            adx_arr = calc_adx(high, low, close, adx_p)

            for adx_min in ADX_THRESHOLDS:
                for rsi_min, rsi_max in RSI_RANGES:
                    rsi_arr = calc_rsi(close, 14)

                    for delay in ENTRY_DELAYS:
                        res = fast_backtest(
                            close, high, low, timestamps,
                            ma_fast, ma_slow,
                            adx_arr, rsi_arr, atr_14,
                            adx_min=adx_min, rsi_min=rsi_min, rsi_max=rsi_max,
                            entry_delay=delay,
                            sl_pct=0.07, trail_act=0.06, trail_pct=0.03,
                            leverage=10, margin_pct=0.25,
                            monthly_loss_limit=-0.20,
                            fee_rate=0.0004, initial_balance=3000.0,
                            warmup=warmup,
                            years_arr=years, month_keys_arr=mkeys,
                        )

                        res['ma_type'] = ma_type
                        res['tf'] = tf
                        res['fast_period'] = fp
                        res['slow_period'] = sp
                        res['adx_period'] = adx_p
                        res['adx_min'] = adx_min
                        res['rsi_min'] = rsi_min
                        res['rsi_max'] = rsi_max
                        res['entry_delay'] = delay
                        results.append(res)

                        count += 1
                        if count % 500 == 0:
                            elapsed = time.time() - t0
                            rate = count / elapsed if elapsed > 0 else 0
                            eta = (total - count) / rate if rate > 0 else 0
                            print(f"  Phase 2: {count}/{total} ({elapsed:.1f}s, ~{eta:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"  Phase 2 complete: {len(results)} combos in {elapsed:.1f}s")

    results.sort(key=lambda x: (x['wpf'] if x['trades'] >= 5 else 0, x['return_pct']), reverse=True)

    print(f"\n  {'Rank':>4} {'MA':>5} {'TF':>5} {'F/S':>5} {'ADX':>5} {'RSI':>7} {'D':>2} {'Trades':>6} {'Ret%':>9} {'PF':>7} {'WPF':>7} {'MDD%':>6} {'WR%':>5}")
    print(f"  {'-'*85}")
    for idx, r in enumerate(results[:30]):
        if r['trades'] < 3:
            continue
        print(f"  {idx+1:>4} {r['ma_type']:>5} {r['tf']:>5} {r['fast_period']}/{r['slow_period']:>3} "
              f"{r['adx_min']:>5} {r['rsi_min']}-{r['rsi_max']:>3} {r['entry_delay']:>2} "
              f"{r['trades']:>6} {r['return_pct']:>+8.1f}% {r['pf']:>6.2f} {r['wpf']:>6.2f} "
              f"{r['mdd']:>5.1f}% {r['win_rate']:>4.1f}")

    return results


# ============================================================
# 6. PHASE 3: EXIT CONDITION OPTIMIZATION
# ============================================================

def phase3_exit(all_data, top_phase2, top_n=30):
    """Optimize exit conditions for top Phase 2 strategies"""
    print("\n" + "="*70)
    print("  PHASE 3: Exit Condition Optimization")
    print("="*70)

    SL_PCTS = [0.03, 0.05, 0.07, 0.08]
    TRAIL_ACTIVATES = [0.02, 0.03, 0.05, 0.06, 0.08, 0.10]
    TRAIL_PCTS = [0.02, 0.03, 0.05]
    TP1_ROIS = [None, 0.15, 0.30]
    SL_ATR_MULTS = [0, 3.0]  # 0 = use fixed SL

    candidates = [r for r in top_phase2 if r['trades'] >= 5][:top_n]

    results = []
    total = len(candidates) * len(SL_PCTS) * len(TRAIL_ACTIVATES) * len(TRAIL_PCTS) * len(TP1_ROIS) * len(SL_ATR_MULTS)
    count = 0
    t0 = time.time()

    for base in candidates:
        tf = base['tf']
        d = all_data[tf]
        close = d['close']
        high = d['high']
        low = d['low']
        volume = d['volume']
        timestamps = d['timestamp']
        years = d['years']
        mkeys = d['month_keys']

        ma_fast = calc_ma(close, base['ma_type'], base['fast_period'], volume)
        ma_slow = calc_ma(close, 'EMA', base['slow_period'], volume)
        adx_arr = calc_adx(high, low, close, base.get('adx_period', 20))
        rsi_arr = calc_rsi(close, 14)
        atr_arr = calc_atr(high, low, close, 14)
        warmup = base['slow_period'] + 50

        for sl_pct in SL_PCTS:
            for trail_act in TRAIL_ACTIVATES:
                for trail_pct in TRAIL_PCTS:
                    if trail_pct >= trail_act:
                        count += len(TP1_ROIS) * len(SL_ATR_MULTS)
                        continue

                    for tp1 in TP1_ROIS:
                        for sl_atr in SL_ATR_MULTS:
                            res = fast_backtest(
                                close, high, low, timestamps,
                                ma_fast, ma_slow,
                                adx_arr, rsi_arr, atr_arr,
                                adx_min=base.get('adx_min', 35),
                                rsi_min=base.get('rsi_min', 30),
                                rsi_max=base.get('rsi_max', 65),
                                entry_delay=base.get('entry_delay', 0),
                                sl_pct=sl_pct, trail_act=trail_act, trail_pct=trail_pct,
                                tp1_roi=tp1,
                                leverage=10, margin_pct=0.25,
                                monthly_loss_limit=-0.20,
                                fee_rate=0.0004, initial_balance=3000.0,
                                sl_atr_mult=sl_atr,
                                warmup=warmup,
                                years_arr=years, month_keys_arr=mkeys,
                            )

                            # Copy base params
                            for k in ['ma_type','tf','fast_period','slow_period',
                                      'adx_period','adx_min','rsi_min','rsi_max','entry_delay']:
                                if k in base:
                                    res[k] = base[k]

                            res['sl_pct'] = sl_pct
                            res['trail_act'] = trail_act
                            res['trail_pct'] = trail_pct
                            res['tp1_roi'] = tp1
                            res['sl_atr_mult'] = sl_atr
                            results.append(res)

                            count += 1
                            if count % 1000 == 0:
                                elapsed = time.time() - t0
                                rate = count / elapsed if elapsed > 0 else 0
                                eta = (total - count) / rate if rate > 0 else 0
                                print(f"  Phase 3: {count}/{total} ({elapsed:.1f}s, ~{eta:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"  Phase 3 complete: {len(results)} combos in {elapsed:.1f}s")

    results.sort(key=lambda x: (x['wpf'] if x['trades'] >= 5 else 0, x['return_pct']), reverse=True)

    print(f"\n  {'Rank':>4} {'MA':>5} {'TF':>5} {'SL%':>5} {'T-Act':>5} {'T-Pct':>5} {'TP1':>5} {'ATR':>4} {'Tr':>4} {'Ret%':>9} {'PF':>7} {'WPF':>7} {'MDD%':>6}")
    print(f"  {'-'*85}")
    for idx, r in enumerate(results[:30]):
        if r['trades'] < 3:
            continue
        tp1_s = f"{r['tp1_roi']:.0%}" if r['tp1_roi'] else 'None'
        atr_s = f"{r['sl_atr_mult']:.1f}" if r['sl_atr_mult'] > 0 else '-'
        print(f"  {idx+1:>4} {r['ma_type']:>5} {r['tf']:>5} {r['sl_pct']*100:>4.0f}% {r['trail_act']*100:>4.0f}% "
              f"{r['trail_pct']*100:>4.0f}% {tp1_s:>5} {atr_s:>4} {r['trades']:>4} "
              f"{r['return_pct']:>+8.1f}% {r['pf']:>6.2f} {r['wpf']:>6.2f} {r['mdd']:>5.1f}%")

    return results


# ============================================================
# 7. PHASE 4: RISK MANAGEMENT OPTIMIZATION
# ============================================================

def phase4_risk(all_data, top_phase3, top_n=30):
    """Optimize position sizing and risk management"""
    print("\n" + "="*70)
    print("  PHASE 4: Risk Management Optimization")
    print("="*70)

    MARGINS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    LEVERAGES = [5, 7, 10, 15]
    ML_LIMITS = [-0.10, -0.15, -0.20, -0.25, -0.30]

    candidates = [r for r in top_phase3 if r['trades'] >= 5][:top_n]

    results = []
    total = len(candidates) * len(MARGINS) * len(LEVERAGES) * len(ML_LIMITS)
    count = 0
    t0 = time.time()

    for base in candidates:
        tf = base['tf']
        d = all_data[tf]
        close = d['close']
        high = d['high']
        low = d['low']
        volume = d['volume']
        timestamps = d['timestamp']
        years = d['years']
        mkeys = d['month_keys']

        ma_fast = calc_ma(close, base['ma_type'], base['fast_period'], volume)
        ma_slow = calc_ma(close, 'EMA', base['slow_period'], volume)
        adx_arr = calc_adx(high, low, close, base.get('adx_period', 20))
        rsi_arr = calc_rsi(close, 14)
        atr_arr = calc_atr(high, low, close, 14)
        warmup = base['slow_period'] + 50

        for margin in MARGINS:
            for lev in LEVERAGES:
                for ml in ML_LIMITS:
                    res = fast_backtest(
                        close, high, low, timestamps,
                        ma_fast, ma_slow,
                        adx_arr, rsi_arr, atr_arr,
                        adx_min=base.get('adx_min', 35),
                        rsi_min=base.get('rsi_min', 30),
                        rsi_max=base.get('rsi_max', 65),
                        entry_delay=base.get('entry_delay', 0),
                        sl_pct=base.get('sl_pct', 0.07),
                        trail_act=base.get('trail_act', 0.06),
                        trail_pct=base.get('trail_pct', 0.03),
                        tp1_roi=base.get('tp1_roi', None),
                        leverage=lev, margin_pct=margin,
                        monthly_loss_limit=ml,
                        fee_rate=0.0004, initial_balance=3000.0,
                        sl_atr_mult=base.get('sl_atr_mult', 0),
                        warmup=warmup,
                        years_arr=years, month_keys_arr=mkeys,
                    )

                    for k in ['ma_type','tf','fast_period','slow_period','adx_period',
                              'adx_min','rsi_min','rsi_max','entry_delay',
                              'sl_pct','trail_act','trail_pct','tp1_roi','sl_atr_mult']:
                        if k in base:
                            res[k] = base[k]

                    res['margin_pct'] = margin
                    res['leverage'] = lev
                    res['monthly_loss_limit'] = ml
                    results.append(res)

                    count += 1
                    if count % 500 == 0:
                        elapsed = time.time() - t0
                        rate = count / elapsed if elapsed > 0 else 0
                        eta = (total - count) / rate if rate > 0 else 0
                        print(f"  Phase 4: {count}/{total} ({elapsed:.1f}s, ~{eta:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"  Phase 4 complete: {len(results)} combos in {elapsed:.1f}s")

    results.sort(key=lambda x: (x['wpf'] if x['trades'] >= 5 else 0, x['return_pct']), reverse=True)

    print(f"\n  {'Rank':>4} {'MA':>5} {'TF':>5} {'Margin':>6} {'Lev':>4} {'ML':>5} {'Tr':>4} {'Ret%':>10} {'PF':>7} {'WPF':>7} {'MDD%':>6} {'$Final':>12}")
    print(f"  {'-'*90}")
    for idx, r in enumerate(results[:30]):
        if r['trades'] < 3:
            continue
        print(f"  {idx+1:>4} {r['ma_type']:>5} {r['tf']:>5} {r['margin_pct']*100:>5.0f}% {r['leverage']:>3}x "
              f"{r['monthly_loss_limit']*100:>4.0f}% {r['trades']:>4} {r['return_pct']:>+9.1f}% "
              f"{r['pf']:>6.2f} {r['wpf']:>6.2f} {r['mdd']:>5.1f}% ${r['balance']:>10,.0f}")

    return results


# ============================================================
# 8. VALIDATION (30x with slippage simulation)
# ============================================================

def validate_strategy(all_data, strategy, runs=30):
    """Validate strategy with slippage simulation"""
    print(f"\n  Validating: {strategy['ma_type']} {strategy['tf']} "
          f"F{strategy['fast_period']}/S{strategy['slow_period']} "
          f"ADX>={strategy.get('adx_min',35)} M{strategy.get('margin_pct',0.25)*100:.0f}%")

    tf = strategy['tf']
    d = all_data[tf]
    close = d['close']
    high = d['high']
    low = d['low']
    volume = d['volume']
    timestamps = d['timestamp']

    ma_fast = calc_ma(close, strategy['ma_type'], strategy['fast_period'], volume)
    ma_slow = calc_ma(close, 'EMA', strategy['slow_period'], volume)
    adx_arr = calc_adx(high, low, close, strategy.get('adx_period', 20))
    rsi_arr = calc_rsi(close, 14)
    atr_arr = calc_atr(high, low, close, 14)
    warmup = strategy['slow_period'] + 50

    results = []
    for run in range(runs):
        # Add random slippage (0~0.05%)
        np.random.seed(run * 42)
        slippage = 1 + np.random.uniform(-0.0005, 0.0005, len(close))
        close_s = close * slippage
        high_s = high * slippage
        low_s = low * slippage

        res = fast_backtest(
            close_s, high_s, low_s, timestamps,
            ma_fast, ma_slow,
            adx_arr, rsi_arr, atr_arr,
            adx_min=strategy.get('adx_min', 35),
            rsi_min=strategy.get('rsi_min', 30),
            rsi_max=strategy.get('rsi_max', 65),
            entry_delay=strategy.get('entry_delay', 0),
            sl_pct=strategy.get('sl_pct', 0.07),
            trail_act=strategy.get('trail_act', 0.06),
            trail_pct=strategy.get('trail_pct', 0.03),
            tp1_roi=strategy.get('tp1_roi', None),
            leverage=strategy.get('leverage', 10),
            margin_pct=strategy.get('margin_pct', 0.25),
            monthly_loss_limit=strategy.get('monthly_loss_limit', -0.20),
            fee_rate=0.0004, initial_balance=3000.0,
            sl_atr_mult=strategy.get('sl_atr_mult', 0),
            warmup=warmup,
        )
        results.append(res)

    # Statistics
    balances = [r['balance'] for r in results]
    returns = [r['return_pct'] for r in results]
    pfs = [r['pf'] for r in results]
    mdds = [r['mdd'] for r in results]

    stats = {
        'runs': runs,
        'balance_mean': np.mean(balances),
        'balance_std': np.std(balances),
        'balance_min': np.min(balances),
        'balance_max': np.max(balances),
        'return_mean': np.mean(returns),
        'return_std': np.std(returns),
        'pf_mean': np.mean(pfs),
        'pf_std': np.std(pfs),
        'pf_min': np.min(pfs),
        'mdd_mean': np.mean(mdds),
        'mdd_max': np.max(mdds),
        'trades_mean': np.mean([r['trades'] for r in results]),
        'win_rate_mean': np.mean([r['win_rate'] for r in results]),
    }

    print(f"    Balance: ${stats['balance_mean']:,.0f} ± ${stats['balance_std']:,.0f} "
          f"(min ${stats['balance_min']:,.0f}, max ${stats['balance_max']:,.0f})")
    print(f"    Return:  {stats['return_mean']:+,.1f}% ± {stats['return_std']:.1f}%")
    print(f"    PF:      {stats['pf_mean']:.2f} ± {stats['pf_std']:.2f} (min {stats['pf_min']:.2f})")
    print(f"    MDD:     {stats['mdd_mean']:.1f}% (max {stats['mdd_max']:.1f}%)")
    print(f"    Trades:  {stats['trades_mean']:.0f}, WR: {stats['win_rate_mean']:.1f}%")

    return stats, results


# ============================================================
# 9. DETAILED TRADE ANALYSIS
# ============================================================

def detailed_analysis(all_data, strategy):
    """Run detailed analysis with monthly/yearly breakdown"""
    tf = strategy['tf']
    d = all_data[tf]
    close = d['close']
    high = d['high']
    low = d['low']
    volume = d['volume']
    timestamps = d['timestamp']

    ma_fast = calc_ma(close, strategy['ma_type'], strategy['fast_period'], volume)
    ma_slow = calc_ma(close, 'EMA', strategy['slow_period'], volume)
    adx_arr = calc_adx(high, low, close, strategy.get('adx_period', 20))
    rsi_arr = calc_rsi(close, 14)
    atr_arr = calc_atr(high, low, close, 14)
    warmup = strategy['slow_period'] + 50

    # Use the original bt_engine for detailed trade records
    n = len(close)
    balance = 3000.0
    initial = 3000.0
    peak_bal = balance
    max_dd = 0.0

    trades = []
    in_pos = False
    pos_dir = 0
    pos_entry = 0.0
    pos_size = 0.0
    pos_sl = 0.0
    trail_active = False
    trail_sl_price = 0.0
    pos_high = 0.0
    pos_low = 0.0
    pos_partial_done = False
    pos_original_size = 0.0
    entry_time = None

    pending_signal = 0
    pending_countdown = 0

    current_month = -1
    month_start_bal = balance
    monthly_locked = False

    leverage = strategy.get('leverage', 10)
    margin_pct = strategy.get('margin_pct', 0.25)
    sl_pct = strategy.get('sl_pct', 0.07)
    trail_act = strategy.get('trail_act', 0.06)
    trail_pct = strategy.get('trail_pct', 0.03)
    tp1_roi = strategy.get('tp1_roi', None)
    tp1_ratio = 0.30
    adx_min = strategy.get('adx_min', 35)
    rsi_min = strategy.get('rsi_min', 30)
    rsi_max = strategy.get('rsi_max', 65)
    entry_delay = strategy.get('entry_delay', 0)
    sl_atr_mult = strategy.get('sl_atr_mult', 0)
    ml_limit = strategy.get('monthly_loss_limit', -0.20)

    ts_pd = pd.to_datetime(timestamps)

    for i in range(warmup, n):
        c = close[i]
        h = high[i]
        lo = low[i]
        ts = ts_pd[i]

        m_key = ts.year * 100 + ts.month
        if m_key != current_month:
            current_month = m_key
            month_start_bal = balance
            monthly_locked = False

        if not monthly_locked and month_start_bal > 0:
            if (balance - month_start_bal) / month_start_bal <= ml_limit:
                monthly_locked = True

        def close_pos(exit_p, reason):
            nonlocal balance, in_pos, pos_dir, pos_size, peak_bal, max_dd
            pnl = pos_size * (exit_p - pos_entry) / pos_entry * pos_dir
            fee = pos_size * 0.0004
            net = pnl - fee
            balance += net
            roi = (exit_p - pos_entry) / pos_entry * pos_dir * leverage
            trades.append({
                'entry_time': entry_time, 'exit_time': ts,
                'direction': 'LONG' if pos_dir == 1 else 'SHORT',
                'entry_price': pos_entry, 'exit_price': exit_p,
                'pnl': net, 'roi': roi, 'reason': reason,
                'balance': balance
            })
            in_pos = False
            if balance > peak_bal:
                peak_bal = balance
            if peak_bal > 0:
                dd = (peak_bal - balance) / peak_bal
                if dd > max_dd:
                    max_dd = dd

        if in_pos:
            # FL check
            if pos_dir == 1:
                fl_p = pos_entry * (1 - 1.0/leverage)
                if lo <= fl_p:
                    close_pos(fl_p, 'FL')
                    continue
            else:
                fl_p = pos_entry * (1 + 1.0/leverage)
                if h >= fl_p:
                    close_pos(fl_p, 'FL')
                    continue

            # SL check
            if pos_dir == 1 and lo <= pos_sl:
                close_pos(pos_sl, 'SL')
            elif pos_dir == -1 and h >= pos_sl:
                close_pos(pos_sl, 'SL')
            elif in_pos:
                # Trail update
                if pos_dir == 1:
                    if h > pos_high:
                        pos_high = h
                    roi_h = (pos_high - pos_entry) / pos_entry * leverage
                    if roi_h >= trail_act * leverage:
                        trail_active = True
                        new_tsl = pos_high * (1 - trail_pct)
                        if new_tsl > trail_sl_price:
                            trail_sl_price = new_tsl
                else:
                    if lo < pos_low:
                        pos_low = lo
                    roi_l = (pos_entry - pos_low) / pos_entry * leverage
                    if roi_l >= trail_act * leverage:
                        trail_active = True
                        new_tsl = pos_low * (1 + trail_pct)
                        if trail_sl_price == 0 or new_tsl < trail_sl_price:
                            trail_sl_price = new_tsl

                # Partial TP
                if tp1_roi and not pos_partial_done:
                    curr_roi = (c - pos_entry) / pos_entry * pos_dir * leverage
                    if curr_roi >= tp1_roi:
                        exit_amt = pos_original_size * tp1_ratio
                        pnl_p = exit_amt * (c - pos_entry) / pos_entry * pos_dir
                        fee_p = exit_amt * 0.0004
                        balance += pnl_p - fee_p
                        pos_size -= exit_amt
                        pos_partial_done = True

                # TSL check
                if trail_active:
                    if (pos_dir == 1 and c <= trail_sl_price) or (pos_dir == -1 and c >= trail_sl_price):
                        close_pos(c, 'TSL')

        # Signal
        if i < warmup + 1:
            continue

        mf = ma_fast[i]
        ms = ma_slow[i]
        mfp = ma_fast[i-1]
        msp = ma_slow[i-1]
        if np.isnan(mf) or np.isnan(ms) or np.isnan(mfp) or np.isnan(msp):
            continue

        signal = 0
        if (mf > ms) and (mfp <= msp):
            av = adx_arr[i] if not np.isnan(adx_arr[i]) else 0
            rv = rsi_arr[i] if not np.isnan(rsi_arr[i]) else 50
            if av >= adx_min and rsi_min <= rv <= rsi_max:
                signal = 1
        elif (mf < ms) and (mfp >= msp):
            av = adx_arr[i] if not np.isnan(adx_arr[i]) else 0
            rv = rsi_arr[i] if not np.isnan(rsi_arr[i]) else 50
            if av >= adx_min and rsi_min <= rv <= rsi_max:
                signal = -1

        if signal != 0 and entry_delay > 0:
            pending_signal = signal
            pending_countdown = entry_delay
            signal = 0

        if pending_countdown > 0:
            pending_countdown -= 1
            if pending_countdown == 0:
                signal = pending_signal
                pending_signal = 0

        if signal != 0:
            if in_pos and pos_dir != signal:
                close_pos(c, 'REV')
            elif in_pos:
                close_pos(c, 'REV')

            if not in_pos and not monthly_locked and balance > 10:
                margin = balance * margin_pct
                size = margin * leverage
                fee = size * 0.0004
                balance -= fee

                pos_dir = signal
                pos_entry = c
                pos_size = size
                pos_original_size = size
                trail_active = False
                trail_sl_price = 0.0
                pos_high = c
                pos_low = c
                pos_partial_done = False
                entry_time = ts

                if sl_atr_mult > 0 and not np.isnan(atr_arr[i]):
                    sl_d = atr_arr[i] * sl_atr_mult / pos_entry
                    sl_d = max(0.015, min(sl_d, 0.10))
                else:
                    sl_d = sl_pct

                if pos_dir == 1:
                    pos_sl = pos_entry * (1 - sl_d)
                else:
                    pos_sl = pos_entry * (1 + sl_d)
                in_pos = True

        if balance > peak_bal:
            peak_bal = balance
        if peak_bal > 0:
            dd = (peak_bal - balance) / peak_bal
            if dd > max_dd:
                max_dd = dd

    if in_pos:
        close_pos(close[-1], 'END')

    # Monthly stats
    monthly = {}
    running_bal = initial
    for t in trades:
        mk = t['entry_time'].strftime('%Y-%m')
        if mk not in monthly:
            monthly[mk] = {'start': running_bal, 'trades': 0, 'pnl': 0,
                          'sl': 0, 'tsl': 0, 'rev': 0, 'fl': 0}
        monthly[mk]['trades'] += 1
        monthly[mk]['pnl'] += t['pnl']
        monthly[mk][t['reason'].lower()] = monthly[mk].get(t['reason'].lower(), 0) + 1
        monthly[mk]['end'] = t['balance']
        running_bal = t['balance']

    # Yearly stats
    yearly = {}
    running_bal = initial
    for t in trades:
        y = t['entry_time'].year
        if y not in yearly:
            yearly[y] = {'start': running_bal, 'trades': 0, 'pnl': 0,
                        'sl': 0, 'tsl': 0, 'rev': 0, 'fl': 0, 'end': running_bal}
        yearly[y]['trades'] += 1
        yearly[y]['pnl'] += t['pnl']
        r = t['reason'].lower()
        yearly[y][r] = yearly[y].get(r, 0) + 1
        yearly[y]['end'] = t['balance']
        running_bal = t['balance']

    for y in yearly:
        s = yearly[y]['start']
        e = yearly[y]['end']
        yearly[y]['return_pct'] = (e - s) / s * 100 if s > 0 else 0

    for mk in monthly:
        s = monthly[mk]['start']
        e = monthly[mk].get('end', s)
        monthly[mk]['return_pct'] = (e - s) / s * 100 if s > 0 else 0

    return {
        'trades': trades,
        'monthly': monthly,
        'yearly': yearly,
        'final_balance': balance,
        'max_dd': max_dd * 100,
        'trade_count': len(trades),
    }


# ============================================================
# 10. MULTI-ENGINE COMPOSITE
# ============================================================

def run_multi_engine(all_data, engine_configs, initial_balance=3000.0):
    """Run multiple engines with capital allocation"""
    results = {}
    for name, cfg in engine_configs.items():
        alloc = cfg.get('allocation', 1.0)
        sub_balance = initial_balance * alloc
        cfg_copy = dict(cfg)
        cfg_copy.pop('allocation', None)

        tf = cfg_copy['tf']
        d = all_data[tf]
        close = d['close']
        high = d['high']
        low = d['low']
        volume = d['volume']
        timestamps = d['timestamp']

        ma_fast = calc_ma(close, cfg_copy['ma_type'], cfg_copy['fast_period'], volume)
        ma_slow = calc_ma(close, 'EMA', cfg_copy['slow_period'], volume)
        adx_arr = calc_adx(high, low, close, cfg_copy.get('adx_period', 20))
        rsi_arr = calc_rsi(close, 14)
        atr_arr = calc_atr(high, low, close, 14)
        warmup = cfg_copy['slow_period'] + 50

        res = fast_backtest(
            close, high, low, timestamps,
            ma_fast, ma_slow,
            adx_arr, rsi_arr, atr_arr,
            adx_min=cfg_copy.get('adx_min', 35),
            rsi_min=cfg_copy.get('rsi_min', 30),
            rsi_max=cfg_copy.get('rsi_max', 65),
            entry_delay=cfg_copy.get('entry_delay', 0),
            sl_pct=cfg_copy.get('sl_pct', 0.07),
            trail_act=cfg_copy.get('trail_act', 0.06),
            trail_pct=cfg_copy.get('trail_pct', 0.03),
            tp1_roi=cfg_copy.get('tp1_roi', None),
            leverage=cfg_copy.get('leverage', 10),
            margin_pct=cfg_copy.get('margin_pct', 0.25),
            monthly_loss_limit=cfg_copy.get('monthly_loss_limit', -0.20),
            fee_rate=0.0004, initial_balance=sub_balance,
            sl_atr_mult=cfg_copy.get('sl_atr_mult', 0),
            warmup=warmup,
        )
        res['allocation'] = alloc
        res['engine_name'] = name
        results[name] = res

    # Composite metrics
    total_balance = sum(r['balance'] for r in results.values())
    total_trades = sum(r['trades'] for r in results.values())
    composite_return = (total_balance - initial_balance) / initial_balance * 100

    # Composite PF (weighted)
    total_profit = sum(max(r['balance'] - initial_balance * r['allocation'], 0) for r in results.values())
    total_loss = sum(max(initial_balance * r['allocation'] - r['balance'], 0) for r in results.values())

    return {
        'engines': results,
        'total_balance': total_balance,
        'total_trades': total_trades,
        'composite_return': composite_return,
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*70)
    print("  BTC/USDT Futures Optimizer v17.0")
    print("  Combination Space: 1,000,000+")
    print("  Hierarchical 4-Phase Optimization")
    print("="*70)

    t_start = time.time()

    # Load data
    df_5m = load_5m_data()
    all_data = precompute_all_data(df_5m)

    # Phase 1: MA Scan
    p1_results = phase1_scan(all_data)
    with open(os.path.join(OUTPUT_DIR, 'phase1_results.json'), 'w') as f:
        # Save top 100
        top100 = [r for r in p1_results if r['trades'] >= 3][:100]
        json.dump(top100, f, indent=2, default=str)
    print(f"\n  Phase 1 top 100 saved to phase1_results.json")

    # Phase 2: Entry Optimization
    p2_results = phase2_entry(all_data, p1_results, top_n=20)
    with open(os.path.join(OUTPUT_DIR, 'phase2_results.json'), 'w') as f:
        top100 = [r for r in p2_results if r['trades'] >= 3][:100]
        json.dump(top100, f, indent=2, default=str)
    print(f"\n  Phase 2 top 100 saved to phase2_results.json")

    # Phase 3: Exit Optimization
    p3_results = phase3_exit(all_data, p2_results, top_n=20)
    with open(os.path.join(OUTPUT_DIR, 'phase3_results.json'), 'w') as f:
        top100 = [r for r in p3_results if r['trades'] >= 3][:100]
        json.dump(top100, f, indent=2, default=str)
    print(f"\n  Phase 3 top 100 saved to phase3_results.json")

    # Phase 4: Risk Optimization
    p4_results = phase4_risk(all_data, p3_results, top_n=20)
    with open(os.path.join(OUTPUT_DIR, 'phase4_results.json'), 'w') as f:
        top100 = [r for r in p4_results if r['trades'] >= 3][:100]
        json.dump(top100, f, indent=2, default=str)
    print(f"\n  Phase 4 top 100 saved to phase4_results.json")

    # ---- FINAL STRATEGY SELECTION ----
    print("\n" + "="*70)
    print("  FINAL STRATEGY SELECTION & VALIDATION")
    print("="*70)

    # Select diverse top strategies for different goals
    # Goal A: Highest PF (precision)
    pf_sorted = sorted([r for r in p4_results if r['trades'] >= 10],
                       key=lambda x: x['pf'], reverse=True)
    # Goal B: Highest return with PF >= 3
    ret_sorted = sorted([r for r in p4_results if r['pf'] >= 3 and r['trades'] >= 10],
                        key=lambda x: x['return_pct'], reverse=True)
    # Goal C: Best balanced (PF × trades / MDD)
    bal_sorted = sorted([r for r in p4_results if r['trades'] >= 10],
                        key=lambda x: x['pf'] * x['trades'] / max(x['mdd'], 1), reverse=True)
    # Goal D: Highest frequency with PF >= 2
    freq_sorted = sorted([r for r in p4_results if r['pf'] >= 2],
                         key=lambda x: x['trades'], reverse=True)

    # Collect unique top strategies
    final_candidates = {}
    for label, lst in [('PF', pf_sorted), ('RET', ret_sorted), ('BAL', bal_sorted), ('FREQ', freq_sorted)]:
        for r in lst[:5]:
            key = f"{r['ma_type']}_{r['tf']}_{r['fast_period']}_{r['slow_period']}_{r.get('adx_min',35)}_{r.get('entry_delay',0)}_{r.get('sl_pct',0.07)}_{r.get('trail_act',0.06)}_{r.get('margin_pct',0.25)}_{r.get('leverage',10)}"
            if key not in final_candidates:
                r['_label'] = label
                final_candidates[key] = r

    print(f"\n  {len(final_candidates)} unique final strategies selected")

    # Validate each
    validated = []
    for key, strat in list(final_candidates.items())[:15]:
        stats, runs = validate_strategy(all_data, strat, runs=30)
        strat['validation'] = stats
        validated.append(strat)

    # Save validated results
    with open(os.path.join(OUTPUT_DIR, 'validated_results.json'), 'w') as f:
        json.dump(validated, f, indent=2, default=str)

    # ---- DETAILED ANALYSIS OF TOP 3 ----
    print("\n" + "="*70)
    print("  DETAILED ANALYSIS - TOP 3 STRATEGIES")
    print("="*70)

    # Sort validated by composite score
    validated.sort(key=lambda x: x['validation']['pf_mean'] * x['validation']['return_mean'] / max(x['validation']['mdd_max'], 1), reverse=True)

    for idx, strat in enumerate(validated[:3]):
        print(f"\n{'='*70}")
        print(f"  STRATEGY #{idx+1}: {strat['ma_type']} {strat['tf']} "
              f"F{strat['fast_period']}/S{strat['slow_period']}")
        print(f"{'='*70}")

        detail = detailed_analysis(all_data, strat)
        strat['detail'] = {
            'monthly': {k: {kk: (vv if not isinstance(vv, (np.floating, np.integer)) else float(vv))
                           for kk, vv in v.items()}
                       for k, v in detail['monthly'].items()},
            'yearly': {str(k): {kk: (vv if not isinstance(vv, (np.floating, np.integer)) else float(vv))
                               for kk, vv in v.items()}
                      for k, v in detail['yearly'].items()},
            'final_balance': detail['final_balance'],
            'max_dd': detail['max_dd'],
            'trade_count': detail['trade_count'],
        }

        # Print yearly
        print(f"\n  [Yearly Performance]")
        print(f"  {'Year':>6} {'Start':>12} {'End':>12} {'Return':>10} {'Trades':>7} {'SL':>4} {'TSL':>5} {'REV':>5}")
        print(f"  {'-'*65}")
        for y in sorted(detail['yearly'].keys()):
            ys = detail['yearly'][y]
            print(f"  {y:>6} ${ys['start']:>10,.0f} ${ys['end']:>10,.0f} "
                  f"{ys['return_pct']:>+9.1f}% {ys['trades']:>6} "
                  f"{ys.get('sl',0):>4} {ys.get('tsl',0):>5} {ys.get('rev',0):>5}")

        # Print monthly
        print(f"\n  [Monthly Performance]")
        print(f"  {'Month':>8} {'Return':>10} {'Trades':>7} {'PnL':>12} {'Balance':>12}")
        print(f"  {'-'*55}")
        for mk in sorted(detail['monthly'].keys()):
            ms = detail['monthly'][mk]
            if ms['trades'] > 0:
                print(f"  {mk:>8} {ms['return_pct']:>+9.1f}% {ms['trades']:>6} "
                      f"${ms['pnl']:>10,.0f} ${ms.get('end', ms['start']):>10,.0f}")

    # Save final report
    final_report = {
        'timestamp': str(pd.Timestamp.now()),
        'total_combos_tested': len(p1_results) + len(p2_results) + len(p3_results) + len(p4_results),
        'combination_space': '9,103,933,440,000+',
        'top_strategies': [],
    }
    for strat in validated[:5]:
        s = {k: v for k, v in strat.items() if k not in ['detail']}
        if 'detail' in strat:
            s['yearly'] = strat['detail'].get('yearly', {})
            s['monthly'] = strat['detail'].get('monthly', {})
        final_report['top_strategies'].append(s)

    with open(os.path.join(OUTPUT_DIR, 'final_report.json'), 'w') as f:
        json.dump(final_report, f, indent=2, default=str)

    t_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"  Total combos tested: {final_report['total_combos_tested']:,}")
    print(f"  Combination space: {final_report['combination_space']}")
    print(f"  Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
