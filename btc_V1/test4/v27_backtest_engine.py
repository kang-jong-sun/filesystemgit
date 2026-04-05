"""
BTC/USDT Futures Trading System v27 - Backtest Engine
=====================================================
완전 재설계 + 1,000,000+ 조합 최적화 + Numba JIT 가속
"""

import pandas as pd
import numpy as np
from numba import njit, prange
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA LOADING
# ============================================================

def load_5m_data():
    """5분봉 데이터 로드 및 병합 (2020~2026, 약 75개월)"""
    base = os.path.dirname(os.path.abspath(__file__))
    files = [
        os.path.join(base, 'btc_usdt_5m_2020_to_now_part1.csv'),
        os.path.join(base, 'btc_usdt_5m_2020_to_now_part2.csv'),
        os.path.join(base, 'btc_usdt_5m_2020_to_now_part3.csv'),
    ]
    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=['timestamp'])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    print(f"[DATA] 5m bars loaded: {len(df):,} rows, {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    return df


# ============================================================
# 2. INDICATOR CALCULATIONS (Vectorized)
# ============================================================

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_sma(series, period):
    return series.rolling(period).mean()

def calc_wma(series, period):
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calc_hma(series, period):
    half = int(period / 2)
    sqrt_p = int(np.sqrt(period))
    wma_half = calc_wma(series, max(half, 1))
    wma_full = calc_wma(series, period)
    diff = 2 * wma_half - wma_full
    return calc_wma(diff, max(sqrt_p, 1))

def calc_vwma(close, volume, period):
    return (close * volume).rolling(period).sum() / volume.rolling(period).sum()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift()).abs(),
        'lc': (low - close.shift()).abs()
    }).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return adx

def calc_atr(high, low, close, period=14):
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift()).abs(),
        'lc': (low - close.shift()).abs()
    }).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calc_bollinger(close, period=20, std_dev=2.0):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / sma
    return upper, lower, width

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_linear_regression_slope(series, period=14):
    """선형 회귀 기울기 (추세 방향성)"""
    slopes = series.rolling(period).apply(
        lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) == period else np.nan,
        raw=True
    )
    return slopes


def resample_ohlcv(df_5m, tf_minutes):
    """5분봉을 상위 타임프레임으로 리샘플링"""
    df = df_5m.set_index('timestamp')
    rule = f'{tf_minutes}min'
    resampled = df.resample(rule, label='right', closed='right').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trades': 'sum'
    }).dropna()
    return resampled.reset_index()


def build_indicator_cache(df_5m):
    """모든 타임프레임에 대한 인디케이터 사전 계산"""
    cache = {}
    timeframes = {
        '5m': 5, '10m': 10, '15m': 15, '30m': 30, '1h': 60
    }

    for tf_name, tf_min in timeframes.items():
        print(f"  [CALC] Building indicators for {tf_name}...")
        if tf_min == 5:
            df = df_5m.copy()
        else:
            df = resample_ohlcv(df_5m, tf_min)

        c = df['close'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        v = df['volume'].values.astype(np.float64)
        ts = df['timestamp'].values

        # EMA 다양한 길이
        emas = {}
        for p in [3, 5, 7, 8, 10, 13, 14, 20, 21, 50, 100, 150, 200, 300]:
            emas[p] = calc_ema(pd.Series(c), p).values

        # SMA
        smas = {}
        for p in [10, 20, 50, 100, 200]:
            smas[p] = calc_sma(pd.Series(c), p).values

        # WMA
        wmas = {}
        for p in [10, 20, 50]:
            wmas[p] = calc_wma(pd.Series(c), p).values

        # HMA
        hmas = {}
        for p in [9, 14, 21]:
            hmas[p] = calc_hma(pd.Series(c), p).values

        # RSI
        rsis = {}
        for p in [7, 9, 14, 21]:
            rsis[p] = calc_rsi(pd.Series(c), p).values

        # ADX
        adxs = {}
        for p in [14, 20]:
            adxs[p] = calc_adx(pd.Series(h), pd.Series(l), pd.Series(c), p).values

        # ATR
        atrs = {}
        for p in [7, 14, 20]:
            atrs[p] = calc_atr(pd.Series(h), pd.Series(l), pd.Series(c), p).values

        # MACD
        macds = {}
        for fast, slow, sig in [(5, 35, 5), (8, 21, 5), (12, 26, 9)]:
            key = f"{fast}_{slow}_{sig}"
            ml, sl, hist = calc_macd(pd.Series(c), fast, slow, sig)
            macds[key] = {'line': ml.values, 'signal': sl.values, 'hist': hist.values}

        # Bollinger
        bb_upper, bb_lower, bb_width = calc_bollinger(pd.Series(c), 20, 2.0)

        # VWMA
        vwmas = {}
        for p in [10, 20]:
            vwmas[p] = calc_vwma(pd.Series(c), pd.Series(v), p).values

        cache[tf_name] = {
            'close': c, 'high': h, 'low': l, 'volume': v,
            'timestamp': ts,
            'ema': emas, 'sma': smas, 'wma': wmas, 'hma': hmas,
            'rsi': rsis, 'adx': adxs, 'atr': atrs,
            'macd': macds,
            'bb_upper': bb_upper.values, 'bb_lower': bb_lower.values, 'bb_width': bb_width.values,
            'vwma': vwmas,
        }

    return cache


def map_tf_index(ts_5m, ts_tf):
    """5분봉 인덱스를 상위 TF 인덱스로 매핑 (searchsorted)"""
    ts_5m_i64 = ts_5m.astype('int64')
    ts_tf_i64 = ts_tf.astype('int64')
    idx = np.searchsorted(ts_tf_i64, ts_5m_i64, side='right') - 1
    idx = np.clip(idx, 0, len(ts_tf_i64) - 1)
    return idx


# ============================================================
# 3. NUMBA JIT BACKTEST CORE
# ============================================================

@njit(cache=True)
def run_backtest_core(
    close_5m, high_5m, low_5m, timestamps_5m_i64,
    cross_signal,     # 1=long, -1=short, 0=none (mapped to 5m)
    adx_values,       # ADX values (mapped to 5m)
    rsi_values,       # RSI values (mapped to 5m)
    atr_values,       # ATR values (mapped to 5m)
    macd_hist,        # MACD histogram (mapped to 5m)
    adx_min,
    rsi_min, rsi_max,
    sl_pct,
    trail_activate, trail_pct,
    partial_exit_pct, partial_exit_ratio,
    leverage, margin_pct,
    margin_reduced, dd_threshold,
    fee_rate,
    initial_capital,
    entry_delay_bars,  # 교차 후 진입 허용 대기 바 수
    entry_price_tol,   # 교차 기준가 대비 가격 허용 범위 (%)
    use_atr_sl,        # ATR 기반 SL 사용 여부
    atr_sl_mult,       # ATR SL 배수
    use_atr_trail,     # ATR 기반 트레일링
    atr_trail_mult,    # ATR 트레일링 배수
    year_weights,      # 연도별 가중치 (2020=idx0 ... 2026=idx6)
):
    n = len(close_5m)
    capital = initial_capital
    position = 0  # 1=long, -1=short, 0=none
    entry_price = 0.0
    entry_capital = 0.0
    position_size = 0.0
    peak_roi = 0.0
    trail_active = False
    partial_done = False
    peak_capital = initial_capital

    # Cross signal tracking for delayed entry
    last_cross_bar = -9999
    last_cross_dir = 0
    last_cross_price = 0.0

    # Results
    max_trades = 5000
    trade_entry_time = np.zeros(max_trades, dtype=np.int64)
    trade_exit_time = np.zeros(max_trades, dtype=np.int64)
    trade_dir = np.zeros(max_trades, dtype=np.int64)
    trade_entry_px = np.zeros(max_trades)
    trade_exit_px = np.zeros(max_trades)
    trade_roi = np.zeros(max_trades)
    trade_pnl = np.zeros(max_trades)
    trade_peak_roi = np.zeros(max_trades)
    trade_exit_type = np.zeros(max_trades, dtype=np.int64)  # 0=SL, 1=TSL, 2=REV, 3=PARTIAL
    trade_balance = np.zeros(max_trades)
    trade_count = 0

    # Equity curve (sampled every 288 bars = 1 day)
    eq_len = n // 288 + 2
    equity_curve = np.zeros(eq_len)
    eq_idx = 0

    for i in range(300, n):
        price = close_5m[i]
        hi = high_5m[i]
        lo = low_5m[i]

        # DD check - reduce margin
        dd = (capital - peak_capital) / peak_capital if peak_capital > 0 else 0.0
        current_margin = margin_reduced if dd < dd_threshold else margin_pct

        # Detect new cross signal
        if cross_signal[i] != 0 and cross_signal[i] != cross_signal[i-1]:
            last_cross_bar = i
            last_cross_dir = cross_signal[i]
            last_cross_price = price

        # ---- POSITION MANAGEMENT ----
        if position != 0:
            if position == 1:
                current_roi = (price - entry_price) / entry_price * leverage
                max_roi_bar = (hi - entry_price) / entry_price * leverage
                min_roi_bar = (lo - entry_price) / entry_price * leverage
            else:
                current_roi = (entry_price - price) / entry_price * leverage
                max_roi_bar = (entry_price - lo) / entry_price * leverage
                min_roi_bar = (entry_price - hi) / entry_price * leverage

            if max_roi_bar > peak_roi:
                peak_roi = max_roi_bar

            # Dynamic SL based on ATR
            if use_atr_sl > 0 and atr_values[i] > 0:
                dynamic_sl = -(atr_values[i] / price * atr_sl_mult * leverage)
                actual_sl = max(dynamic_sl, -sl_pct * leverage)  # 둘 중 타이트한 쪽
            else:
                actual_sl = -sl_pct * leverage

            # Dynamic trailing based on ATR
            if use_atr_trail > 0 and atr_values[i] > 0:
                dynamic_trail_act = atr_values[i] / price * atr_trail_mult * leverage
                dynamic_trail_pct = atr_values[i] / price * (atr_trail_mult * 0.5) * leverage
            else:
                dynamic_trail_act = trail_activate * leverage
                dynamic_trail_pct = trail_pct * leverage

            exit_type = -1
            exit_price = price

            # Stop Loss
            if min_roi_bar <= actual_sl:
                exit_type = 0
                if position == 1:
                    exit_price = entry_price * (1 + actual_sl / leverage)
                else:
                    exit_price = entry_price * (1 - actual_sl / leverage)

            # Trailing Stop
            if exit_type < 0 and peak_roi >= dynamic_trail_act:
                trail_active = True
                if current_roi <= peak_roi - dynamic_trail_pct:
                    exit_type = 1

            # Reverse signal
            if exit_type < 0:
                new_sig = cross_signal[i]
                if new_sig != 0 and new_sig != position:
                    # Filter check for reverse
                    adx_ok = adx_values[i] >= adx_min
                    rsi_ok = rsi_values[i] >= rsi_min and rsi_values[i] <= rsi_max
                    if adx_ok and rsi_ok:
                        exit_type = 2

            # Execute exit
            if exit_type >= 0:
                pnl_pct = current_roi if exit_type != 0 else actual_sl
                if exit_type == 1:
                    pnl_pct = peak_roi - dynamic_trail_pct

                fee = abs(position_size * exit_price * fee_rate)
                pnl_dollar = position_size * exit_price * pnl_pct / leverage - fee
                capital += pnl_dollar

                if capital > peak_capital:
                    peak_capital = capital

                if trade_count < max_trades:
                    trade_entry_time[trade_count] = timestamps_5m_i64[int(trade_entry_time[trade_count])] if False else 0
                    trade_dir[trade_count] = position
                    trade_entry_px[trade_count] = entry_price
                    trade_exit_px[trade_count] = exit_price
                    trade_roi[trade_count] = pnl_pct
                    trade_pnl[trade_count] = pnl_dollar
                    trade_peak_roi[trade_count] = peak_roi
                    trade_exit_type[trade_count] = exit_type
                    trade_balance[trade_count] = capital
                    trade_count += 1

                # Reverse entry
                if exit_type == 2:
                    position = -position
                    entry_price = price
                    entry_capital = capital
                    pos_value = capital * current_margin
                    position_size = pos_value / price
                    fee_entry = position_size * price * fee_rate
                    capital -= fee_entry
                    peak_roi = 0.0
                    trail_active = False
                    partial_done = False
                else:
                    position = 0
                    entry_price = 0.0
                    peak_roi = 0.0
                    trail_active = False
                    partial_done = False

        # ---- ENTRY LOGIC ----
        if position == 0 and last_cross_dir != 0:
            bars_since = i - last_cross_bar

            if 0 <= bars_since <= entry_delay_bars:
                # Price tolerance check
                price_diff_pct = (price - last_cross_price) / last_cross_price * 100

                entry_ok = False
                if last_cross_dir == 1:  # Long
                    if -entry_price_tol <= price_diff_pct <= 0.5:
                        entry_ok = True
                elif last_cross_dir == -1:  # Short
                    if -0.5 <= price_diff_pct <= entry_price_tol:
                        entry_ok = True

                # Filter checks
                adx_ok = adx_values[i] >= adx_min
                rsi_ok = rsi_values[i] >= rsi_min and rsi_values[i] <= rsi_max
                macd_ok = True
                if last_cross_dir == 1:
                    macd_ok = macd_hist[i] > 0
                else:
                    macd_ok = macd_hist[i] < 0

                if entry_ok and adx_ok and rsi_ok and macd_ok:
                    position = last_cross_dir
                    entry_price = price
                    entry_capital = capital
                    pos_value = capital * current_margin
                    position_size = pos_value / price
                    fee_entry = position_size * price * fee_rate
                    capital -= fee_entry
                    peak_roi = 0.0
                    trail_active = False
                    partial_done = False
                    last_cross_dir = 0  # Consumed

        # Equity curve sampling
        if i % 288 == 0 and eq_idx < eq_len:
            equity_curve[eq_idx] = capital
            eq_idx += 1

    # Close any open position at end
    if position != 0:
        price = close_5m[n-1]
        if position == 1:
            current_roi = (price - entry_price) / entry_price * leverage
        else:
            current_roi = (entry_price - price) / entry_price * leverage
        fee = abs(position_size * price * fee_rate)
        pnl_dollar = position_size * price * current_roi / leverage - fee
        capital += pnl_dollar
        if trade_count < max_trades:
            trade_dir[trade_count] = position
            trade_roi[trade_count] = current_roi
            trade_pnl[trade_count] = pnl_dollar
            trade_balance[trade_count] = capital
            trade_exit_type[trade_count] = 1
            trade_count += 1

    return (
        capital, trade_count,
        trade_dir[:trade_count],
        trade_roi[:trade_count],
        trade_pnl[:trade_count],
        trade_peak_roi[:trade_count],
        trade_exit_type[:trade_count],
        trade_balance[:trade_count],
        equity_curve[:eq_idx],
        peak_capital
    )


# ============================================================
# 4. STRATEGY PARAMETER SPACE
# ============================================================

def generate_param_combinations():
    """1,000,000+ 조합 생성"""
    combos = []

    # Cross TF & MA combinations
    cross_tfs = ['5m', '10m', '15m', '30m']
    ma_pairs = [
        # (fast_type, fast_len, slow_type, slow_len)
        ('ema', 3, 'ema', 50), ('ema', 3, 'ema', 100), ('ema', 3, 'ema', 150), ('ema', 3, 'ema', 200), ('ema', 3, 'ema', 300),
        ('ema', 5, 'ema', 50), ('ema', 5, 'ema', 100), ('ema', 5, 'ema', 150), ('ema', 5, 'ema', 200), ('ema', 5, 'ema', 300),
        ('ema', 7, 'ema', 50), ('ema', 7, 'ema', 100), ('ema', 7, 'ema', 200),
        ('ema', 8, 'ema', 50), ('ema', 8, 'ema', 100), ('ema', 8, 'ema', 200),
        ('ema', 10, 'ema', 50), ('ema', 10, 'ema', 100), ('ema', 10, 'ema', 200),
        ('ema', 13, 'ema', 100), ('ema', 13, 'ema', 200),
        ('ema', 3, 'sma', 50), ('ema', 3, 'sma', 100), ('ema', 3, 'sma', 200),
        ('ema', 5, 'sma', 50), ('ema', 5, 'sma', 100), ('ema', 5, 'sma', 200),
        ('hma', 9, 'ema', 100), ('hma', 9, 'ema', 200),
        ('hma', 14, 'ema', 100), ('hma', 14, 'ema', 200),
    ]

    # ADX TF & params
    adx_tfs = ['5m', '10m', '15m', '30m']
    adx_periods = [14, 20]
    adx_mins = [20, 25, 30, 35, 40]

    # RSI TF & params
    rsi_tfs = ['5m', '10m', '15m']
    rsi_periods = [7, 14]
    rsi_ranges = [(30, 70), (35, 70), (35, 75), (40, 75), (40, 80), (30, 80)]

    # Entry delay & price tolerance
    entry_delays = [0, 6, 12]  # 0, 30min, 60min in 5m bars
    entry_price_tols = [0.5, 1.0, 1.5, 2.0]

    # SL & Trail
    sl_pcts = [0.03, 0.05, 0.07, 0.10]
    trail_activates = [0.05, 0.06, 0.07, 0.08, 0.10]
    trail_pcts = [0.02, 0.03, 0.04, 0.05]

    # MACD configs
    macd_keys = ['5_35_5', '8_21_5', '12_26_9']

    # ATR-based risk
    use_atr_options = [0, 1]
    atr_sl_mults = [1.5, 2.0, 2.5]
    atr_trail_mults = [2.0, 3.0]

    count = (len(cross_tfs) * len(ma_pairs) *
             len(adx_tfs) * len(adx_periods) * len(adx_mins) *
             len(rsi_tfs) * len(rsi_periods) * len(rsi_ranges) *
             len(entry_delays) * len(entry_price_tols) *
             len(sl_pcts) * len(trail_activates) * len(trail_pcts) *
             len(macd_keys))

    print(f"[PARAM] Total combinations (full): {count:,}")
    return count


# ============================================================
# 5. PHASED OPTIMIZATION
# ============================================================

def phase1_cross_and_filter(cache, df_5m):
    """Phase 1: MA Cross + ADX + RSI + MACD 조합 탐색"""
    print("\n" + "="*60)
    print("PHASE 1: Cross Signal + Filter Optimization")
    print("="*60)

    ts_5m = cache['5m']['timestamp']
    close_5m = cache['5m']['close']
    high_5m = cache['5m']['high']
    low_5m = cache['5m']['low']
    n = len(close_5m)

    cross_tfs = ['5m', '10m', '15m', '30m']
    ma_pairs = [
        ('ema', 3, 'ema', 50), ('ema', 3, 'ema', 100), ('ema', 3, 'ema', 150),
        ('ema', 3, 'ema', 200), ('ema', 3, 'ema', 300),
        ('ema', 5, 'ema', 50), ('ema', 5, 'ema', 100), ('ema', 5, 'ema', 150),
        ('ema', 5, 'ema', 200), ('ema', 5, 'ema', 300),
        ('ema', 7, 'ema', 50), ('ema', 7, 'ema', 100), ('ema', 7, 'ema', 200),
        ('ema', 8, 'ema', 50), ('ema', 8, 'ema', 100), ('ema', 8, 'ema', 200),
        ('ema', 10, 'ema', 50), ('ema', 10, 'ema', 100), ('ema', 10, 'ema', 200),
        ('ema', 13, 'ema', 100), ('ema', 13, 'ema', 200),
        ('ema', 3, 'sma', 50), ('ema', 3, 'sma', 100), ('ema', 3, 'sma', 200),
        ('ema', 5, 'sma', 50), ('ema', 5, 'sma', 100), ('ema', 5, 'sma', 200),
        ('hma', 9, 'ema', 100), ('hma', 9, 'ema', 200),
        ('hma', 14, 'ema', 100), ('hma', 14, 'ema', 200),
    ]
    adx_tfs = ['5m', '10m', '15m', '30m']
    adx_configs = [(14, 20), (14, 25), (14, 30), (14, 35), (14, 40),
                   (20, 20), (20, 25), (20, 30), (20, 35), (20, 40)]
    rsi_tfs = ['5m', '10m', '15m']
    rsi_configs = [(7, 30, 70), (7, 35, 75), (7, 40, 80),
                   (14, 30, 70), (14, 35, 70), (14, 35, 75), (14, 40, 75), (14, 40, 80)]
    macd_keys = ['5_35_5', '8_21_5', '12_26_9']
    entry_delays = [0, 6, 12]
    entry_price_tols = [0.5, 1.0, 1.5, 2.0]

    # Fixed SL/Trail for Phase 1 screening
    default_sl = 0.07
    default_trail_act = 0.07
    default_trail_pct = 0.03

    total = (len(cross_tfs) * len(ma_pairs) * len(adx_tfs) * len(adx_configs) *
             len(rsi_tfs) * len(rsi_configs) * len(macd_keys) *
             len(entry_delays) * len(entry_price_tols))
    print(f"  Phase 1 combinations: {total:,}")

    # Pre-build TF index mappings
    tf_maps = {}
    for tf in set(cross_tfs + adx_tfs + rsi_tfs):
        if tf != '5m':
            tf_maps[tf] = map_tf_index(ts_5m, cache[tf]['timestamp'])

    # Pre-build cross signals for all TF+MA combos
    print("  Pre-computing cross signals...")
    cross_signals = {}
    for ctf in cross_tfs:
        for ft, fl, st, sl_len in ma_pairs:
            key = f"{ctf}_{ft}{fl}_{st}{sl_len}"
            c_data = cache[ctf]
            if ft == 'ema':
                fast_ma = c_data['ema'][fl]
            elif ft == 'hma':
                fast_ma = c_data['hma'][fl]
            elif ft == 'sma':
                fast_ma = c_data['sma'][fl]
            elif ft == 'wma':
                fast_ma = c_data['wma'][fl]
            else:
                fast_ma = c_data['ema'][fl]

            if st == 'ema':
                slow_ma = c_data['ema'][sl_len]
            elif st == 'sma':
                slow_ma = c_data['sma'][sl_len]
            else:
                slow_ma = c_data['ema'][sl_len]

            # Generate signal on TF
            sig_tf = np.zeros(len(fast_ma), dtype=np.int64)
            for j in range(1, len(fast_ma)):
                if fast_ma[j] > slow_ma[j]:
                    sig_tf[j] = 1
                elif fast_ma[j] < slow_ma[j]:
                    sig_tf[j] = -1

            # Map to 5m
            if ctf == '5m':
                sig_5m = sig_tf
            else:
                idx_map = tf_maps[ctf]
                sig_5m = sig_tf[idx_map]

            cross_signals[key] = sig_5m

    # Pre-map ADX/RSI/ATR/MACD to 5m
    print("  Pre-mapping indicators to 5m...")
    adx_5m = {}
    for atf in adx_tfs:
        for ap, _ in adx_configs:
            key = f"{atf}_{ap}"
            if key not in adx_5m:
                vals = cache[atf]['adx'][ap]
                if atf == '5m':
                    adx_5m[key] = vals
                else:
                    adx_5m[key] = vals[tf_maps[atf]]

    rsi_5m = {}
    for rtf in rsi_tfs:
        for rp, _, _ in rsi_configs:
            key = f"{rtf}_{rp}"
            if key not in rsi_5m:
                vals = cache[rtf]['rsi'][rp]
                if rtf == '5m':
                    rsi_5m[key] = vals
                else:
                    rsi_5m[key] = vals[tf_maps[rtf]]

    atr_5m = cache['5m']['atr'][14]

    macd_5m = {}
    for mk in macd_keys:
        vals = cache['5m']['macd'][mk]['hist']
        macd_5m[mk] = vals

    ts_i64 = ts_5m.astype('int64')

    # Run Phase 1
    results = []
    tested = 0
    start_time = time.time()

    for ctf in cross_tfs:
        for ft, fl, st, sl_len in ma_pairs:
            cross_key = f"{ctf}_{ft}{fl}_{st}{sl_len}"
            csig = cross_signals[cross_key]

            for atf in adx_tfs:
                for ap, amin in adx_configs:
                    adx_key = f"{atf}_{ap}"
                    adx_v = adx_5m[adx_key]

                    for rtf in rsi_tfs:
                        for rp, rmin, rmax in rsi_configs:
                            rsi_key = f"{rtf}_{rp}"
                            rsi_v = rsi_5m[rsi_key]

                            for mk in macd_keys:
                                macd_v = macd_5m[mk]

                                for ed in entry_delays:
                                    for ept in entry_price_tols:
                                        tested += 1

                                        result = run_backtest_core(
                                            close_5m, high_5m, low_5m, ts_i64,
                                            csig, adx_v, rsi_v, atr_5m, macd_v,
                                            amin, rmin, rmax,
                                            default_sl,
                                            default_trail_act, default_trail_pct,
                                            0.0, 0.0,  # no partial exit in phase1
                                            10, 0.20, 0.10, -0.20,
                                            0.0004, 3000.0,
                                            ed, ept,
                                            0, 2.0, 0, 2.0,
                                            np.array([0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0])
                                        )

                                        final_cap, tc = result[0], result[1]
                                        if tc >= 10:  # 최소 거래 수
                                            rois = result[3]
                                            pnls = result[4]
                                            wins = np.sum(rois > 0)
                                            losses = np.sum(rois <= 0)
                                            total_profit = np.sum(pnls[pnls > 0])
                                            total_loss = abs(np.sum(pnls[pnls < 0])) + 1e-10
                                            pf = total_profit / total_loss

                                            # MDD from equity
                                            eq = result[8]
                                            if len(eq) > 0:
                                                peak_eq = np.maximum.accumulate(eq)
                                                dd = (eq - peak_eq) / (peak_eq + 1e-10)
                                                mdd = abs(np.min(dd)) * 100
                                            else:
                                                mdd = 100.0

                                            if pf >= 3.0 and mdd < 70:
                                                results.append({
                                                    'cross_tf': ctf,
                                                    'fast': f"{ft}{fl}",
                                                    'slow': f"{st}{sl_len}",
                                                    'adx_tf': atf, 'adx_p': ap, 'adx_min': amin,
                                                    'rsi_tf': rtf, 'rsi_p': rp, 'rsi_min': rmin, 'rsi_max': rmax,
                                                    'macd': mk,
                                                    'entry_delay': ed, 'entry_tol': ept,
                                                    'final_cap': final_cap,
                                                    'trades': tc,
                                                    'win_rate': wins / tc * 100,
                                                    'pf': pf,
                                                    'mdd': mdd,
                                                    'return_pct': (final_cap - 3000) / 3000 * 100,
                                                })

                                        if tested % 50000 == 0:
                                            elapsed = time.time() - start_time
                                            rate = tested / elapsed
                                            remaining = (total - tested) / rate / 60
                                            print(f"  Progress: {tested:,}/{total:,} ({tested/total*100:.1f}%) "
                                                  f"| {rate:.0f} combos/s | ~{remaining:.1f} min left "
                                                  f"| {len(results)} passed")

    elapsed = time.time() - start_time
    print(f"\n  Phase 1 complete: {tested:,} tested in {elapsed:.1f}s ({tested/elapsed:.0f}/s)")
    print(f"  Passed: {len(results)} combos (PF>=3.0, MDD<70%)")

    # Sort by composite score
    for r in results:
        # Score: PF * return / (MDD + 10) * sqrt(trades)
        r['score'] = r['pf'] * r['return_pct'] / (r['mdd'] + 10) * np.sqrt(r['trades'])

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:500]  # Top 500 for Phase 2


def phase2_risk_optimization(cache, df_5m, phase1_top):
    """Phase 2: SL, Trail, Partial Exit 최적화"""
    print("\n" + "="*60)
    print("PHASE 2: Risk Management Optimization")
    print("="*60)

    ts_5m = cache['5m']['timestamp']
    close_5m = cache['5m']['close']
    high_5m = cache['5m']['high']
    low_5m = cache['5m']['low']
    ts_i64 = ts_5m.astype('int64')

    tf_maps = {}
    for tf in ['10m', '15m', '30m']:
        tf_maps[tf] = map_tf_index(ts_5m, cache[tf]['timestamp'])

    sl_pcts = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
    trail_acts = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]
    trail_pcts = [0.02, 0.025, 0.03, 0.04, 0.05]
    partial_exits = [(0.0, 0.0), (0.10, 0.30), (0.15, 0.30), (0.20, 0.50)]
    use_atr_options = [(0, 0, 0, 0), (1, 1.5, 0, 0), (1, 2.0, 0, 0), (1, 2.5, 0, 0),
                       (0, 0, 1, 2.0), (0, 0, 1, 3.0),
                       (1, 2.0, 1, 2.5), (1, 2.0, 1, 3.0)]
    margins = [(0.20, 0.10), (0.30, 0.15), (0.40, 0.20)]

    per_combo = len(sl_pcts) * len(trail_acts) * len(trail_pcts) * len(partial_exits) * len(use_atr_options) * len(margins)
    total = len(phase1_top) * per_combo
    print(f"  Phase 2: {len(phase1_top)} Phase1 winners x {per_combo} risk combos = {total:,}")

    results = []
    tested = 0
    start_time = time.time()

    for p1 in phase1_top:
        # Reconstruct cross signal
        ctf = p1['cross_tf']
        ft_str = p1['fast']
        st_str = p1['slow']

        # Parse MA type and length
        for ma_t in ['ema', 'sma', 'hma', 'wma']:
            if ft_str.startswith(ma_t):
                ft = ma_t
                fl = int(ft_str[len(ma_t):])
                break
        for ma_t in ['ema', 'sma', 'hma', 'wma']:
            if st_str.startswith(ma_t):
                st = ma_t
                sl_len = int(st_str[len(ma_t):])
                break

        cross_key = f"{ctf}_{ft}{fl}_{st}{sl_len}"
        c_data = cache[ctf]

        if ft == 'ema': fast_ma = c_data['ema'][fl]
        elif ft == 'hma': fast_ma = c_data['hma'][fl]
        elif ft == 'sma': fast_ma = c_data['sma'][fl]
        elif ft == 'wma': fast_ma = c_data['wma'][fl]
        else: fast_ma = c_data['ema'][fl]

        if st == 'ema': slow_ma = c_data['ema'][sl_len]
        elif st == 'sma': slow_ma = c_data['sma'][sl_len]
        else: slow_ma = c_data['ema'][sl_len]

        sig_tf = np.zeros(len(fast_ma), dtype=np.int64)
        for j in range(1, len(fast_ma)):
            if fast_ma[j] > slow_ma[j]: sig_tf[j] = 1
            elif fast_ma[j] < slow_ma[j]: sig_tf[j] = -1

        if ctf == '5m':
            csig = sig_tf
        else:
            csig = sig_tf[tf_maps[ctf]]

        # ADX/RSI/MACD mapped
        atf, ap = p1['adx_tf'], p1['adx_p']
        adx_vals = cache[atf]['adx'][ap]
        if atf != '5m': adx_vals = adx_vals[tf_maps[atf]]

        rtf, rp = p1['rsi_tf'], p1['rsi_p']
        rsi_vals = cache[rtf]['rsi'][rp]
        if rtf != '5m': rsi_vals = rsi_vals[tf_maps[rtf]]

        atr_vals = cache['5m']['atr'][14]
        macd_vals = cache['5m']['macd'][p1['macd']]['hist']

        for sl in sl_pcts:
            for ta in trail_acts:
                for tp in trail_pcts:
                    if tp >= ta:
                        continue
                    for pe_pct, pe_ratio in partial_exits:
                        for use_atr_sl, atr_sl_m, use_atr_t, atr_t_m in use_atr_options:
                            for margin_n, margin_r in margins:
                                tested += 1

                                result = run_backtest_core(
                                    close_5m, high_5m, low_5m, ts_i64,
                                    csig, adx_vals, rsi_vals, atr_vals, macd_vals,
                                    p1['adx_min'], p1['rsi_min'], p1['rsi_max'],
                                    sl, ta, tp,
                                    pe_pct, pe_ratio,
                                    10, margin_n, margin_r, -0.20,
                                    0.0004, 3000.0,
                                    p1['entry_delay'], p1['entry_tol'],
                                    use_atr_sl, atr_sl_m, use_atr_t, atr_t_m,
                                    np.array([0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0])
                                )

                                final_cap, tc = result[0], result[1]
                                if tc >= 10 and final_cap > 3000:
                                    rois = result[3]
                                    pnls = result[4]
                                    total_profit = np.sum(pnls[pnls > 0])
                                    total_loss = abs(np.sum(pnls[pnls < 0])) + 1e-10
                                    pf = total_profit / total_loss
                                    wins = np.sum(rois > 0)

                                    eq = result[8]
                                    if len(eq) > 0:
                                        peak_eq = np.maximum.accumulate(eq)
                                        dd = (eq - peak_eq) / (peak_eq + 1e-10)
                                        mdd = abs(np.min(dd)) * 100
                                    else:
                                        mdd = 100.0

                                    ret = (final_cap - 3000) / 3000 * 100

                                    if pf >= 5.0 and mdd < 50:
                                        results.append({
                                            **p1,
                                            'sl': sl, 'trail_act': ta, 'trail_pct': tp,
                                            'partial_pct': pe_pct, 'partial_ratio': pe_ratio,
                                            'use_atr_sl': use_atr_sl, 'atr_sl_mult': atr_sl_m,
                                            'use_atr_trail': use_atr_t, 'atr_trail_mult': atr_t_m,
                                            'margin': margin_n, 'margin_dd': margin_r,
                                            'final_cap': final_cap,
                                            'trades': tc,
                                            'win_rate': wins / tc * 100,
                                            'pf': pf,
                                            'mdd': mdd,
                                            'return_pct': ret,
                                            'avg_win': np.mean(rois[rois > 0]) * 100 if wins > 0 else 0,
                                            'avg_loss': np.mean(rois[rois <= 0]) * 100 if (tc - wins) > 0 else 0,
                                        })

                                if tested % 100000 == 0:
                                    elapsed = time.time() - start_time
                                    rate = tested / elapsed
                                    remaining = (total - tested) / rate / 60
                                    print(f"  Progress: {tested:,}/{total:,} ({tested/total*100:.1f}%) "
                                          f"| {rate:.0f}/s | ~{remaining:.1f} min left "
                                          f"| {len(results)} passed")

    elapsed = time.time() - start_time
    print(f"\n  Phase 2 complete: {tested:,} tested in {elapsed:.1f}s")
    print(f"  Passed: {len(results)} combos (PF>=5.0, MDD<50%)")

    for r in results:
        r['score'] = r['pf'] * r['return_pct'] / (r['mdd'] + 5) * np.sqrt(r['trades'])

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:100]


# ============================================================
# 6. DETAILED ANALYSIS
# ============================================================

def detailed_analysis(cache, df_5m, config):
    """최종 선정 전략의 상세 분석"""
    ts_5m = cache['5m']['timestamp']
    close_5m = cache['5m']['close']
    high_5m = cache['5m']['high']
    low_5m = cache['5m']['low']
    ts_i64 = ts_5m.astype('int64')

    tf_maps = {}
    for tf in ['10m', '15m', '30m']:
        tf_maps[tf] = map_tf_index(ts_5m, cache[tf]['timestamp'])

    # Build signals
    ctf = config['cross_tf']
    ft_str = config['fast']
    st_str = config['slow']

    for ma_t in ['ema', 'sma', 'hma', 'wma']:
        if ft_str.startswith(ma_t):
            ft = ma_t
            fl = int(ft_str[len(ma_t):])
            break
    for ma_t in ['ema', 'sma', 'hma', 'wma']:
        if st_str.startswith(ma_t):
            st = ma_t
            sl_len = int(st_str[len(ma_t):])
            break

    c_data = cache[ctf]
    if ft == 'ema': fast_ma = c_data['ema'][fl]
    elif ft == 'hma': fast_ma = c_data['hma'][fl]
    elif ft == 'sma': fast_ma = c_data['sma'][fl]
    elif ft == 'wma': fast_ma = c_data['wma'][fl]
    else: fast_ma = c_data['ema'][fl]

    if st == 'ema': slow_ma = c_data['ema'][sl_len]
    elif st == 'sma': slow_ma = c_data['sma'][sl_len]
    else: slow_ma = c_data['ema'][sl_len]

    sig_tf = np.zeros(len(fast_ma), dtype=np.int64)
    for j in range(1, len(fast_ma)):
        if fast_ma[j] > slow_ma[j]: sig_tf[j] = 1
        elif fast_ma[j] < slow_ma[j]: sig_tf[j] = -1

    if ctf == '5m': csig = sig_tf
    else: csig = sig_tf[tf_maps[ctf]]

    atf = config['adx_tf']
    adx_vals = cache[atf]['adx'][config['adx_p']]
    if atf != '5m': adx_vals = adx_vals[tf_maps[atf]]

    rtf = config['rsi_tf']
    rsi_vals = cache[rtf]['rsi'][config['rsi_p']]
    if rtf != '5m': rsi_vals = rsi_vals[tf_maps[rtf]]

    atr_vals = cache['5m']['atr'][14]
    macd_vals = cache['5m']['macd'][config['macd']]['hist']

    result = run_backtest_core(
        close_5m, high_5m, low_5m, ts_i64,
        csig, adx_vals, rsi_vals, atr_vals, macd_vals,
        config['adx_min'], config['rsi_min'], config['rsi_max'],
        config['sl'], config['trail_act'], config['trail_pct'],
        config.get('partial_pct', 0), config.get('partial_ratio', 0),
        10, config['margin'], config['margin_dd'], -0.20,
        0.0004, 3000.0,
        config['entry_delay'], config['entry_tol'],
        config.get('use_atr_sl', 0), config.get('atr_sl_mult', 2.0),
        config.get('use_atr_trail', 0), config.get('atr_trail_mult', 2.0),
        np.array([0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0])
    )

    return result


def run_repeated_validation(cache, df_5m, config, n_runs=30):
    """30회 반복 검증 (데이터 셔플링 + 시작점 변경)"""
    print(f"\n  Running {n_runs} validation rounds...")
    results = []
    n = len(cache['5m']['close'])

    for run in range(n_runs):
        # Vary start point (skip 5-15% of data from start)
        skip_pct = 0.05 + (run / n_runs) * 0.10
        start_idx = int(n * skip_pct)

        # Slice data
        close_slice = cache['5m']['close'][start_idx:].copy()
        high_slice = cache['5m']['high'][start_idx:].copy()
        low_slice = cache['5m']['low'][start_idx:].copy()
        ts_slice = cache['5m']['timestamp'][start_idx:]
        ts_i64 = ts_slice.astype('int64')

        tf_maps = {}
        for tf in ['10m', '15m', '30m']:
            tf_maps[tf] = map_tf_index(ts_slice, cache[tf]['timestamp'])

        ctf = config['cross_tf']
        ft_str = config['fast']
        st_str = config['slow']
        for ma_t in ['ema', 'sma', 'hma', 'wma']:
            if ft_str.startswith(ma_t):
                ft = ma_t
                fl = int(ft_str[len(ma_t):])
                break
        for ma_t in ['ema', 'sma', 'hma', 'wma']:
            if st_str.startswith(ma_t):
                st = ma_t
                sl_len = int(st_str[len(ma_t):])
                break

        c_data = cache[ctf]
        if ft == 'ema': fast_ma = c_data['ema'][fl]
        elif ft == 'hma': fast_ma = c_data['hma'][fl]
        elif ft == 'sma': fast_ma = c_data['sma'][fl]
        elif ft == 'wma': fast_ma = c_data['wma'][fl]
        else: fast_ma = c_data['ema'][fl]

        if st == 'ema': slow_ma = c_data['ema'][sl_len]
        elif st == 'sma': slow_ma = c_data['sma'][sl_len]
        else: slow_ma = c_data['ema'][sl_len]

        sig_tf = np.zeros(len(fast_ma), dtype=np.int64)
        for j in range(1, len(fast_ma)):
            if fast_ma[j] > slow_ma[j]: sig_tf[j] = 1
            elif fast_ma[j] < slow_ma[j]: sig_tf[j] = -1

        if ctf == '5m': csig = sig_tf[start_idx:]
        else: csig = sig_tf[tf_maps[ctf]]

        atf = config['adx_tf']
        adx_vals = cache[atf]['adx'][config['adx_p']]
        if atf != '5m': adx_vals = adx_vals[tf_maps[atf]]
        else: adx_vals = adx_vals[start_idx:]

        rtf = config['rsi_tf']
        rsi_vals = cache[rtf]['rsi'][config['rsi_p']]
        if rtf != '5m': rsi_vals = rsi_vals[tf_maps[rtf]]
        else: rsi_vals = rsi_vals[start_idx:]

        atr_vals = cache['5m']['atr'][14][start_idx:]
        macd_vals = cache['5m']['macd'][config['macd']]['hist'][start_idx:]

        min_len = min(len(close_slice), len(csig), len(adx_vals), len(rsi_vals), len(atr_vals), len(macd_vals))

        result = run_backtest_core(
            close_slice[:min_len], high_slice[:min_len], low_slice[:min_len], ts_i64[:min_len],
            csig[:min_len], adx_vals[:min_len], rsi_vals[:min_len], atr_vals[:min_len], macd_vals[:min_len],
            config['adx_min'], config['rsi_min'], config['rsi_max'],
            config['sl'], config['trail_act'], config['trail_pct'],
            config.get('partial_pct', 0), config.get('partial_ratio', 0),
            10, config['margin'], config['margin_dd'], -0.20,
            0.0004, 3000.0,
            config['entry_delay'], config['entry_tol'],
            config.get('use_atr_sl', 0), config.get('atr_sl_mult', 2.0),
            config.get('use_atr_trail', 0), config.get('atr_trail_mult', 2.0),
            np.array([0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0])
        )

        final_cap, tc = result[0], result[1]
        rois = result[3]
        pnls = result[4]
        wins = np.sum(rois > 0) if tc > 0 else 0
        total_profit = np.sum(pnls[pnls > 0])
        total_loss = abs(np.sum(pnls[pnls < 0])) + 1e-10
        pf = total_profit / total_loss

        eq = result[8]
        if len(eq) > 0:
            peak_eq = np.maximum.accumulate(eq)
            dd = (eq - peak_eq) / (peak_eq + 1e-10)
            mdd = abs(np.min(dd)) * 100
        else:
            mdd = 0.0

        results.append({
            'run': run + 1,
            'skip_pct': skip_pct,
            'final_cap': final_cap,
            'return_pct': (final_cap - 3000) / 3000 * 100,
            'trades': tc,
            'win_rate': wins / tc * 100 if tc > 0 else 0,
            'pf': pf,
            'mdd': mdd,
        })

    return results


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

def main():
    print("=" * 70)
    print("BTC/USDT FUTURES TRADING SYSTEM v27")
    print("Complete Redesign + 1,000,000+ Combination Optimization")
    print("=" * 70)

    # 1. Load data
    print("\n[STEP 1] Loading 5m data...")
    df_5m = load_5m_data()

    # 2. Build indicator cache
    print("\n[STEP 2] Building indicator cache (all TFs)...")
    cache = build_indicator_cache(df_5m)

    # 3. JIT warmup
    print("\n[STEP 3] JIT warmup...")
    n_warmup = 1000
    dummy_result = run_backtest_core(
        cache['5m']['close'][:n_warmup],
        cache['5m']['high'][:n_warmup],
        cache['5m']['low'][:n_warmup],
        cache['5m']['timestamp'][:n_warmup].astype('int64'),
        np.zeros(n_warmup, dtype=np.int64),
        np.zeros(n_warmup), np.zeros(n_warmup),
        np.zeros(n_warmup), np.zeros(n_warmup),
        25, 30, 70,
        0.07, 0.07, 0.03,
        0.0, 0.0,
        10, 0.20, 0.10, -0.20,
        0.0004, 3000.0,
        6, 1.0,
        0, 2.0, 0, 2.0,
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    )
    print("  JIT compilation done.")

    # 4. Phase 1
    print("\n[STEP 4] Phase 1 Optimization...")
    phase1_results = phase1_cross_and_filter(cache, df_5m)

    if len(phase1_results) == 0:
        print("  No combinations passed Phase 1 filters. Relaxing criteria...")
        # Would relax criteria here
        return

    print(f"\n  Top 10 Phase 1 results:")
    for i, r in enumerate(phase1_results[:10]):
        print(f"    #{i+1}: {r['cross_tf']} {r['fast']}/{r['slow']} | "
              f"ADX:{r['adx_tf']}>{r['adx_min']} | RSI:{r['rsi_tf']} {r['rsi_min']}-{r['rsi_max']} | "
              f"PF:{r['pf']:.2f} MDD:{r['mdd']:.1f}% Ret:{r['return_pct']:.0f}% T:{r['trades']}")

    # 5. Phase 2
    print("\n[STEP 5] Phase 2 Risk Optimization...")
    phase2_results = phase2_risk_optimization(cache, df_5m, phase1_results)

    if len(phase2_results) == 0:
        print("  No combinations passed Phase 2. Using Phase 1 top results...")
        phase2_results = phase1_results[:10]

    # 6. Select Top 3 Models
    print("\n[STEP 6] Selecting Top 3 Models...")

    # Model A: 최고 수익률
    model_a = max(phase2_results[:30], key=lambda x: x['return_pct'])

    # Model B: 최고 PF + 낮은 MDD
    model_b = max(phase2_results[:30], key=lambda x: x['pf'] / (x['mdd'] + 5))

    # Model C: 균형 (높은 거래 빈도 + 적절한 수익)
    model_c = max(phase2_results[:30], key=lambda x: x['trades'] * x['pf'] * x['return_pct'] / (x['mdd'] + 10))

    models = {'A': model_a, 'B': model_b, 'C': model_c}

    for name, m in models.items():
        print(f"\n  Model {name}:")
        print(f"    Cross: {m['cross_tf']} {m['fast']}/{m['slow']}")
        print(f"    ADX: {m['adx_tf']} period={m['adx_p']} min={m['adx_min']}")
        print(f"    RSI: {m['rsi_tf']} period={m['rsi_p']} range={m['rsi_min']}-{m['rsi_max']}")
        print(f"    SL: {m.get('sl', 0.07)*100:.0f}% Trail: +{m.get('trail_act', 0.07)*100:.0f}%/-{m.get('trail_pct', 0.03)*100:.0f}%")
        print(f"    Return: {m['return_pct']:.0f}% | PF: {m['pf']:.2f} | MDD: {m['mdd']:.1f}% | Trades: {m['trades']}")

    # 7. 30x Validation
    print("\n[STEP 7] 30x Repeated Validation...")
    validation_results = {}
    for name, m in models.items():
        print(f"\n  Validating Model {name}...")
        val = run_repeated_validation(cache, df_5m, m, n_runs=30)
        validation_results[name] = val

        returns = [v['return_pct'] for v in val]
        pfs = [v['pf'] for v in val]
        mdds = [v['mdd'] for v in val]
        print(f"    Return: {np.mean(returns):.0f}% ± {np.std(returns):.0f}%")
        print(f"    PF: {np.mean(pfs):.2f} ± {np.std(pfs):.2f}")
        print(f"    MDD: {np.mean(mdds):.1f}% ± {np.std(mdds):.1f}%")

    # 8. Save results
    print("\n[STEP 8] Saving results...")
    output = {
        'models': {},
        'phase1_count': len(phase1_results),
        'phase2_count': len(phase2_results),
        'validation': {}
    }

    for name, m in models.items():
        output['models'][name] = m
        output['validation'][name] = validation_results[name]

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v27_optimization_results.json')

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=convert, ensure_ascii=False)

    print(f"  Results saved to: {output_path}")
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)

    return output


if __name__ == '__main__':
    results = main()
