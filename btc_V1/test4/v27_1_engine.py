"""
BTC/USDT v27.1 - PF 5+ 목표 최적화 엔진
========================================
핵심 변화:
- v27 문제점: REV 86% → PF 1.23 (거래 과다, 품질 낮음)
- v27.1 해법: 다중 필터 강화로 고품질 진입만 허용 → PF 5+ 목표
- 추가 인디케이터: Stochastic, CCI, OBV, Bollinger, Ichimoku
- REV 조건 강화: 역방향 전환에도 모든 필터 충족 필요
"""

import pandas as pd
import numpy as np
from numba import njit
import time, os, json, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

from v27_backtest_engine import (
    load_5m_data, resample_ohlcv, map_tf_index,
    calc_ema, calc_sma, calc_wma, calc_hma, calc_vwma,
    calc_rsi, calc_adx, calc_atr, calc_macd, calc_bollinger
)

# ============================================================
# 추가 인디케이터
# ============================================================

def calc_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d

def calc_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad + 1e-10)

def calc_obv(close, volume):
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv[i] = obv[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv[i] = obv[i-1] - volume.iloc[i]
        else:
            obv[i] = obv[i-1]
    return pd.Series(obv, index=close.index)

def calc_obv_slope(close, volume, period=20):
    obv = calc_obv(close, volume)
    obv_sma = obv.rolling(period).mean()
    slope = (obv - obv_sma) / (obv_sma.abs() + 1e-10) * 100
    return slope.values

def calc_ichimoku_cloud(high, low, tenkan=9, kijun=26, senkou_b=52):
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_val = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    return tenkan_sen.values, kijun_sen.values, senkou_a.values, senkou_b_val.values

def calc_cmf(high, low, close, volume, period=20):
    mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
    mfv = mfm * volume
    return (mfv.rolling(period).sum() / volume.rolling(period).sum()).values


# ============================================================
# Numba JIT 백테스트 - PF 5+ 버전
# ============================================================

@njit(cache=True)
def run_backtest_v271(
    close_5m, high_5m, low_5m, ts_i64,
    cross_signal,
    adx_values, rsi_values, atr_values, macd_hist,
    stoch_k, stoch_d, cci_values, obv_slope, bb_width,
    ichimoku_above_cloud,  # 1=above, -1=below, 0=in cloud
    # Filter params
    adx_min, rsi_min, rsi_max,
    stoch_oversold, stoch_overbought,  # e.g. 20, 80
    cci_min, cci_max,  # e.g. -100, 100
    obv_threshold,  # OBV slope threshold
    use_ichimoku,   # 0 or 1
    use_bb_filter,  # 0 or 1
    bb_width_min,   # min BB width for volatility
    # Risk params
    sl_pct, trail_activate, trail_pct,
    partial_exit_pct, partial_exit_roi,
    leverage, margin_pct, margin_reduced, dd_threshold,
    fee_rate, initial_capital,
    entry_delay_bars, entry_price_tol,
    # REV strictness: 0=no REV, 1=REV with full filter, 2=REV with relaxed filter
    rev_mode,
    min_bars_between_trades,  # minimum bars between trades to avoid overtrading
):
    n = len(close_5m)
    capital = initial_capital
    position = 0
    entry_price = 0.0
    position_size = 0.0
    peak_roi = 0.0
    trail_active = False
    partial_done = False
    peak_capital = initial_capital
    last_exit_bar = -9999

    last_cross_bar = -9999
    last_cross_dir = 0
    last_cross_price = 0.0

    max_trades = 5000
    trade_roi = np.zeros(max_trades)
    trade_pnl = np.zeros(max_trades)
    trade_peak_roi = np.zeros(max_trades)
    trade_exit_type = np.zeros(max_trades, dtype=np.int64)
    trade_balance = np.zeros(max_trades)
    trade_dir = np.zeros(max_trades, dtype=np.int64)
    trade_count = 0

    eq_len = n // 288 + 2
    equity_curve = np.zeros(eq_len)
    eq_idx = 0

    for i in range(300, n):
        price = close_5m[i]
        hi = high_5m[i]
        lo = low_5m[i]

        dd = (capital - peak_capital) / peak_capital if peak_capital > 0 else 0.0
        current_margin = margin_reduced if dd < dd_threshold else margin_pct

        # Detect cross
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

            actual_sl = -sl_pct * leverage
            trail_act_v = trail_activate * leverage
            trail_pct_v = trail_pct * leverage

            exit_type = -1

            # SL
            if min_roi_bar <= actual_sl:
                exit_type = 0

            # TSL
            if exit_type < 0 and peak_roi >= trail_act_v:
                trail_active = True
                if current_roi <= peak_roi - trail_pct_v:
                    exit_type = 1

            # Partial exit
            if exit_type < 0 and not partial_done and partial_exit_roi > 0:
                if current_roi >= partial_exit_roi * leverage:
                    # Reduce position
                    reduce_pct = partial_exit_pct
                    reduce_size = position_size * reduce_pct
                    fee = abs(reduce_size * price * fee_rate)
                    pnl = reduce_size * price * current_roi / leverage - fee
                    capital += pnl
                    position_size -= reduce_size
                    partial_done = True
                    if capital > peak_capital:
                        peak_capital = capital

            # REV
            if exit_type < 0 and rev_mode > 0:
                new_sig = cross_signal[i]
                if new_sig != 0 and new_sig != position:
                    rev_ok = False
                    if rev_mode == 1:
                        # Full filter check
                        a_ok = adx_values[i] >= adx_min
                        r_ok = rsi_values[i] >= rsi_min and rsi_values[i] <= rsi_max
                        m_ok = (new_sig == 1 and macd_hist[i] > 0) or (new_sig == -1 and macd_hist[i] < 0)

                        # Stochastic
                        s_ok = True
                        if new_sig == 1:
                            s_ok = stoch_k[i] < stoch_overbought
                        else:
                            s_ok = stoch_k[i] > stoch_oversold

                        # CCI
                        c_ok = cci_values[i] >= cci_min and cci_values[i] <= cci_max

                        # Ichimoku
                        ich_ok = True
                        if use_ichimoku > 0:
                            if new_sig == 1:
                                ich_ok = ichimoku_above_cloud[i] >= 0
                            else:
                                ich_ok = ichimoku_above_cloud[i] <= 0

                        rev_ok = a_ok and r_ok and m_ok and s_ok and c_ok and ich_ok
                    elif rev_mode == 2:
                        # Relaxed: only ADX + direction
                        a_ok = adx_values[i] >= adx_min * 0.7
                        m_ok = (new_sig == 1 and macd_hist[i] > 0) or (new_sig == -1 and macd_hist[i] < 0)
                        rev_ok = a_ok and m_ok

                    if rev_ok:
                        exit_type = 2

            # Execute exit
            if exit_type >= 0:
                if exit_type == 0:
                    pnl_pct = actual_sl
                elif exit_type == 1:
                    pnl_pct = peak_roi - trail_pct_v
                else:
                    pnl_pct = current_roi

                fee = abs(position_size * price * fee_rate)
                pnl_dollar = position_size * price * pnl_pct / leverage - fee
                capital += pnl_dollar
                if capital > peak_capital:
                    peak_capital = capital

                if trade_count < max_trades:
                    trade_dir[trade_count] = position
                    trade_roi[trade_count] = pnl_pct / leverage
                    trade_pnl[trade_count] = pnl_dollar
                    trade_peak_roi[trade_count] = peak_roi / leverage
                    trade_exit_type[trade_count] = exit_type
                    trade_balance[trade_count] = capital
                    trade_count += 1

                last_exit_bar = i

                if exit_type == 2:
                    position = -position
                    entry_price = price
                    pos_value = capital * current_margin
                    position_size = pos_value / price
                    fee_e = position_size * price * fee_rate
                    capital -= fee_e
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
            bars_since_exit = i - last_exit_bar

            if 0 <= bars_since <= entry_delay_bars and bars_since_exit >= min_bars_between_trades:
                price_diff_pct = (price - last_cross_price) / last_cross_price * 100

                entry_ok = False
                if last_cross_dir == 1:
                    if -entry_price_tol <= price_diff_pct <= 0.5:
                        entry_ok = True
                elif last_cross_dir == -1:
                    if -0.5 <= price_diff_pct <= entry_price_tol:
                        entry_ok = True

                if entry_ok:
                    # Multi-filter check
                    adx_ok = adx_values[i] >= adx_min
                    rsi_ok = rsi_values[i] >= rsi_min and rsi_values[i] <= rsi_max
                    macd_ok = (last_cross_dir == 1 and macd_hist[i] > 0) or (last_cross_dir == -1 and macd_hist[i] < 0)

                    # Stochastic filter
                    stoch_ok = True
                    if last_cross_dir == 1:
                        stoch_ok = stoch_k[i] < stoch_overbought  # Not overbought for long
                    else:
                        stoch_ok = stoch_k[i] > stoch_oversold  # Not oversold for short

                    # CCI filter
                    cci_ok = cci_values[i] >= cci_min and cci_values[i] <= cci_max

                    # OBV slope filter
                    obv_ok = True
                    if obv_threshold > 0:
                        if last_cross_dir == 1:
                            obv_ok = obv_slope[i] > obv_threshold
                        else:
                            obv_ok = obv_slope[i] < -obv_threshold

                    # Ichimoku cloud filter
                    ich_ok = True
                    if use_ichimoku > 0:
                        if last_cross_dir == 1:
                            ich_ok = ichimoku_above_cloud[i] >= 0  # Above or in cloud for long
                        else:
                            ich_ok = ichimoku_above_cloud[i] <= 0  # Below or in cloud for short

                    # BB width filter (volatility)
                    bb_ok = True
                    if use_bb_filter > 0:
                        bb_ok = bb_width[i] >= bb_width_min

                    if adx_ok and rsi_ok and macd_ok and stoch_ok and cci_ok and obv_ok and ich_ok and bb_ok:
                        position = last_cross_dir
                        entry_price = price
                        pos_value = capital * current_margin
                        position_size = pos_value / price
                        fee_e = position_size * price * fee_rate
                        capital -= fee_e
                        peak_roi = 0.0
                        trail_active = False
                        partial_done = False
                        last_cross_dir = 0

        if i % 288 == 0 and eq_idx < eq_len:
            equity_curve[eq_idx] = capital
            eq_idx += 1

    # Close open position
    if position != 0:
        price = close_5m[n-1]
        if position == 1:
            cr = (price - entry_price) / entry_price * leverage
        else:
            cr = (entry_price - price) / entry_price * leverage
        fee = abs(position_size * price * fee_rate)
        pnl = position_size * price * cr / leverage - fee
        capital += pnl
        if trade_count < max_trades:
            trade_dir[trade_count] = position
            trade_roi[trade_count] = cr / leverage
            trade_pnl[trade_count] = pnl
            trade_balance[trade_count] = capital
            trade_exit_type[trade_count] = 1
            trade_count += 1

    return (
        capital, trade_count,
        trade_dir[:trade_count], trade_roi[:trade_count],
        trade_pnl[:trade_count], trade_peak_roi[:trade_count],
        trade_exit_type[:trade_count], trade_balance[:trade_count],
        equity_curve[:eq_idx], peak_capital
    )


def build_extended_cache(df_5m):
    """확장 인디케이터 캐시 (기존 + Stochastic, CCI, OBV, Ichimoku, CMF)"""
    cache = {}
    timeframes = {'5m': 5, '10m': 10, '15m': 15, '30m': 30, '1h': 60}

    for tf_name, tf_min in timeframes.items():
        print(f"  [CALC] {tf_name}...", flush=True)
        if tf_min == 5:
            df = df_5m.copy()
        else:
            df = resample_ohlcv(df_5m, tf_min)

        c = df['close']; h = df['high']; l = df['low']; v = df['volume']
        ts = df['timestamp'].values

        # EMA
        emas = {}
        for p in [3, 5, 7, 8, 10, 13, 14, 20, 21, 50, 100, 150, 200, 300]:
            emas[p] = calc_ema(c, p).values
        # SMA
        smas = {}
        for p in [10, 20, 50, 100, 200]:
            smas[p] = calc_sma(c, p).values
        # HMA
        hmas = {}
        for p in [9, 14, 21]:
            hmas[p] = calc_hma(c, p).values
        # RSI
        rsis = {}
        for p in [7, 9, 14, 21]:
            rsis[p] = calc_rsi(c, p).values
        # ADX
        adxs = {}
        for p in [14, 20]:
            adxs[p] = calc_adx(h, l, c, p).values
        # ATR
        atrs = {}
        for p in [7, 14, 20]:
            atrs[p] = calc_atr(h, l, c, p).values
        # MACD
        macds = {}
        for fast, slow, sig in [(5, 35, 5), (8, 21, 5), (12, 26, 9)]:
            key = f"{fast}_{slow}_{sig}"
            ml, sl_v, hist = calc_macd(c, fast, slow, sig)
            macds[key] = {'line': ml.values, 'signal': sl_v.values, 'hist': hist.values}
        # Bollinger
        bb_u, bb_l, bb_w = calc_bollinger(c, 20, 2.0)
        # Stochastic
        stoch_k, stoch_d = calc_stochastic(h, l, c, 14, 3)
        # CCI
        cci = calc_cci(h, l, c, 20)
        # OBV slope
        obv_sl = calc_obv_slope(c, v, 20)
        # Ichimoku
        tenkan, kijun, senkou_a, senkou_b = calc_ichimoku_cloud(h, l)
        # Cloud position: 1=above, -1=below, 0=in cloud
        ich_cloud = np.zeros(len(c))
        for j in range(len(c)):
            if np.isnan(senkou_a[j]) or np.isnan(senkou_b[j]):
                ich_cloud[j] = 0
            else:
                cloud_top = max(senkou_a[j], senkou_b[j])
                cloud_bot = min(senkou_a[j], senkou_b[j])
                if c.iloc[j] > cloud_top:
                    ich_cloud[j] = 1
                elif c.iloc[j] < cloud_bot:
                    ich_cloud[j] = -1
                else:
                    ich_cloud[j] = 0

        cache[tf_name] = {
            'close': c.values.astype(np.float64),
            'high': h.values.astype(np.float64),
            'low': l.values.astype(np.float64),
            'volume': v.values.astype(np.float64),
            'timestamp': ts,
            'ema': emas, 'sma': smas, 'hma': hmas,
            'rsi': rsis, 'adx': adxs, 'atr': atrs, 'macd': macds,
            'bb_upper': bb_u.values, 'bb_lower': bb_l.values, 'bb_width': bb_w.values,
            'stoch_k': stoch_k.values.astype(np.float64),
            'stoch_d': stoch_d.values.astype(np.float64),
            'cci': cci.values.astype(np.float64),
            'obv_slope': obv_sl.astype(np.float64),
            'ichimoku_cloud': ich_cloud.astype(np.float64),
        }

    return cache


def run_optimization():
    print("="*70, flush=True)
    print("v27.1 PF 5+ OPTIMIZATION ENGINE", flush=True)
    print("="*70, flush=True)

    df_5m = load_5m_data()
    print("\nBuilding extended indicator cache...", flush=True)
    cache = build_extended_cache(df_5m)

    ts_5m = cache['5m']['timestamp']
    close_5m = cache['5m']['close']
    high_5m = cache['5m']['high']
    low_5m = cache['5m']['low']
    n = len(close_5m)
    ts_i64 = ts_5m.astype('int64')

    tf_maps = {}
    for tf in ['10m', '15m', '30m', '1h']:
        tf_maps[tf] = map_tf_index(ts_5m, cache[tf]['timestamp'])

    # JIT warmup
    print("JIT warmup...", flush=True)
    nw = 1000
    _ = run_backtest_v271(
        close_5m[:nw], high_5m[:nw], low_5m[:nw], ts_i64[:nw],
        np.zeros(nw, dtype=np.int64),
        np.zeros(nw), np.zeros(nw), np.zeros(nw), np.zeros(nw),
        np.zeros(nw), np.zeros(nw), np.zeros(nw), np.zeros(nw), np.zeros(nw),
        np.zeros(nw),
        25, 30, 70, 20, 80, -100, 100, 0, 0, 0, 0.01,
        0.07, 0.07, 0.03, 0.0, 0.0,
        10, 0.40, 0.20, -0.20, 0.0004, 3000.0, 6, 1.0,
        1, 6
    )
    print("  Done.", flush=True)

    # ================================================================
    # PHASE 1: Cross + Multi-Filter 최적화
    # ================================================================
    print("\n[Phase 1] Cross + Multi-Filter Optimization...", flush=True)

    # Cross configs (from v27 top performers + v25.2 proven)
    cross_configs = [
        ('30m', 'ema', 3, 'ema', 200),   # v25.2 Model A
        ('30m', 'ema', 5, 'sma', 50),    # v27 best
        ('30m', 'ema', 7, 'ema', 100),   # v25.2 Model B
        ('30m', 'ema', 5, 'ema', 200),
        ('30m', 'ema', 3, 'ema', 100),
        ('30m', 'ema', 3, 'ema', 150),
        ('30m', 'hma', 9, 'ema', 200),
        ('30m', 'hma', 14, 'ema', 200),
        ('15m', 'ema', 5, 'ema', 200),
        ('15m', 'ema', 7, 'ema', 50),
        ('10m', 'ema', 5, 'ema', 300),   # v25.2 Model C
        ('10m', 'ema', 5, 'ema', 200),
        ('5m', 'ema', 5, 'ema', 150),    # v27 Phase1B best
        ('5m', 'ema', 5, 'ema', 200),
    ]

    # ADX configs
    adx_tfs = ['5m', '10m', '15m', '30m']
    adx_configs = [(14, 25), (14, 30), (14, 35), (14, 40), (20, 25), (20, 30), (20, 35), (20, 40)]

    # RSI configs
    rsi_tfs = ['5m', '10m', '15m']
    rsi_configs = [(14, 30, 70), (14, 35, 70), (14, 35, 75), (14, 40, 75), (14, 40, 80)]

    # MACD
    macd_keys = ['5_35_5', '8_21_5', '12_26_9']

    # Stochastic
    stoch_configs = [(20, 80), (25, 75), (30, 70)]

    # CCI
    cci_configs = [(-200, 200), (-150, 150), (-100, 100)]

    # OBV
    obv_thresholds = [0, 1, 3]

    # Ichimoku
    ich_options = [0, 1]

    # BB width filter
    bb_options = [(0, 0), (1, 0.02), (1, 0.03)]

    # Entry
    entry_delays = [6, 12]
    entry_tols = [0.5, 1.0, 1.5]

    # REV mode
    rev_modes = [0, 1, 2]  # 0=no REV, 1=strict REV, 2=relaxed REV

    # Min bars between trades
    min_bars_opts = [6, 12, 24]

    # SL/Trail
    sl_trail_configs = [
        (0.05, 0.06, 0.03), (0.05, 0.07, 0.03),
        (0.07, 0.07, 0.03), (0.07, 0.08, 0.03),
        (0.07, 0.10, 0.04), (0.10, 0.10, 0.04),
        (0.10, 0.12, 0.05),
    ]

    # Margins
    margins = [(0.20, 0.10), (0.30, 0.15), (0.40, 0.20)]

    # Partial exit
    partial_configs = [(0, 0), (0.30, 0.10), (0.30, 0.15)]

    # Pre-build cross signals
    print("  Pre-computing signals...", flush=True)
    cross_sigs = {}
    for ctf, ft, fl, st, sl_len in cross_configs:
        c_data = cache[ctf]
        fm = c_data[ft][fl] if ft in c_data and isinstance(c_data.get(ft), dict) and fl in c_data[ft] else c_data['ema'][fl]
        sm = c_data[st][sl_len] if st in c_data and isinstance(c_data.get(st), dict) and sl_len in c_data[st] else c_data['ema'][sl_len]

        sig = np.zeros(len(fm), dtype=np.int64)
        for j in range(1, len(fm)):
            if fm[j] > sm[j]: sig[j] = 1
            elif fm[j] < sm[j]: sig[j] = -1

        cs = sig if ctf == '5m' else sig[tf_maps[ctf]]
        cross_sigs[f"{ctf}_{ft}{fl}_{st}{sl_len}"] = cs

    # Pre-map indicators
    adx_m = {}
    for atf in adx_tfs:
        for ap, _ in adx_configs:
            key = f"{atf}_{ap}"
            if key not in adx_m:
                v = cache[atf]['adx'][ap]
                adx_m[key] = v if atf == '5m' else v[tf_maps[atf]]

    rsi_m = {}
    for rtf in rsi_tfs:
        for rp, _, _ in rsi_configs:
            key = f"{rtf}_{rp}"
            if key not in rsi_m:
                v = cache[rtf]['rsi'][rp]
                rsi_m[key] = v if rtf == '5m' else v[tf_maps[rtf]]

    macd_m = {mk: cache['5m']['macd'][mk]['hist'] for mk in macd_keys}
    atr_5m = cache['5m']['atr'][14]
    stoch_k_5m = cache['5m']['stoch_k']
    stoch_d_5m = cache['5m']['stoch_d']
    cci_5m = cache['5m']['cci']
    obv_5m = cache['5m']['obv_slope']
    bb_w_5m = cache['5m']['bb_width']
    ich_5m = cache['5m']['ichimoku_cloud']

    # Calculate total combos
    total = (len(cross_configs) * len(adx_tfs) * len(adx_configs) *
             len(rsi_tfs) * len(rsi_configs) * len(macd_keys) *
             len(stoch_configs) * len(cci_configs) * len(obv_thresholds) *
             len(ich_options) * len(bb_options) * len(entry_delays) * len(entry_tols) *
             len(rev_modes) * len(min_bars_opts) * len(sl_trail_configs) * len(margins) * len(partial_configs))
    print(f"  Full combo count: {total:,} (too large, using staged approach)", flush=True)

    # Staged: First find best cross+ADX+RSI+MACD, then add extra filters
    print("\n  Stage A: Cross + ADX + RSI + MACD + SL/Trail...", flush=True)

    stage_a_results = []
    tested = 0
    t0 = time.time()

    for ck, csig in cross_sigs.items():
        for atf in adx_tfs:
            for ap, amin in adx_configs:
                adx_v = adx_m[f"{atf}_{ap}"]
                for rtf in rsi_tfs:
                    for rp, rmin, rmax in rsi_configs:
                        rsi_v = rsi_m[f"{rtf}_{rp}"]
                        for mk in macd_keys:
                            macd_v = macd_m[mk]
                            for sl, ta, tp in sl_trail_configs:
                                for mn, mr in margins:
                                    for ed in entry_delays:
                                        for et in entry_tols:
                                            for rm in rev_modes:
                                                for mb in min_bars_opts:
                                                    tested += 1

                                                    result = run_backtest_v271(
                                                        close_5m, high_5m, low_5m, ts_i64,
                                                        csig, adx_v, rsi_v, atr_5m, macd_v,
                                                        stoch_k_5m, stoch_d_5m, cci_5m, obv_5m, bb_w_5m, ich_5m,
                                                        amin, rmin, rmax,
                                                        20, 80, -200, 200, 0, 0, 0, 0.01,
                                                        sl, ta, tp, 0, 0,
                                                        10, mn, mr, -0.20, 0.0004, 3000.0,
                                                        ed, et, rm, mb
                                                    )

                                                    fc, tc = result[0], result[1]
                                                    if tc >= 15 and fc > 5000:
                                                        rois = result[3]; pnls = result[4]
                                                        wins = np.sum(rois > 0)
                                                        tpro = np.sum(pnls[pnls > 0])
                                                        tlos = abs(np.sum(pnls[pnls < 0])) + 1e-10
                                                        pf = tpro / tlos
                                                        ret = (fc - 3000) / 3000 * 100

                                                        eq = result[8]; mdd = 0
                                                        if len(eq) > 0:
                                                            peq = np.maximum.accumulate(eq)
                                                            dd = (eq - peq) / (peq + 1e-10)
                                                            mdd = abs(np.min(dd)) * 100

                                                        etypes = result[6]
                                                        sl_c = int(np.sum(etypes == 0))
                                                        tsl_c = int(np.sum(etypes == 1))
                                                        rev_c = int(np.sum(etypes == 2))

                                                        if pf >= 3.0 and mdd < 60:
                                                            stage_a_results.append({
                                                                'cross': ck,
                                                                'adx_tf': atf, 'adx_p': ap, 'adx_min': amin,
                                                                'rsi_tf': rtf, 'rsi_p': rp, 'rsi_min': rmin, 'rsi_max': rmax,
                                                                'macd': mk,
                                                                'sl': sl, 'trail_act': ta, 'trail_pct': tp,
                                                                'margin': mn, 'margin_dd': mr,
                                                                'entry_delay': ed, 'entry_tol': et,
                                                                'rev_mode': rm, 'min_bars': mb,
                                                                'final_cap': float(fc), 'trades': int(tc),
                                                                'win_rate': float(wins/tc*100),
                                                                'pf': float(pf), 'mdd': float(mdd),
                                                                'return_pct': float(ret),
                                                                'avg_win': float(np.mean(rois[rois > 0])*100) if wins > 0 else 0,
                                                                'avg_loss': float(np.mean(rois[rois <= 0])*100) if (tc-wins) > 0 else 0,
                                                                'sl_count': sl_c, 'tsl_count': tsl_c, 'rev_count': rev_c,
                                                            })

                                                    if tested % 100000 == 0:
                                                        elapsed = time.time() - t0
                                                        rate = tested / elapsed
                                                        tot_est = (len(cross_configs) * len(adx_tfs) * len(adx_configs) *
                                                                   len(rsi_tfs) * len(rsi_configs) * len(macd_keys) *
                                                                   len(sl_trail_configs) * len(margins) * len(entry_delays) *
                                                                   len(entry_tols) * len(rev_modes) * len(min_bars_opts))
                                                        rem = (tot_est - tested) / rate / 60 if rate > 0 else 999
                                                        print(f"    {tested:,} tested | {rate:.0f}/s | ~{rem:.0f}min | {len(stage_a_results)} passed (PF>=3)", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Stage A: {tested:,} in {elapsed:.0f}s -> {len(stage_a_results)} passed", flush=True)

    # Sort by PF * sqrt(trades) / MDD
    for r in stage_a_results:
        r['score'] = r['pf'] * np.sqrt(r['trades']) * r['return_pct'] / (r['mdd'] + 5)

    stage_a_results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n  Top 15 Stage A:", flush=True)
    for i, r in enumerate(stage_a_results[:15]):
        print(f"    #{i+1}: {r['cross']} ADX:{r['adx_tf']}>{r['adx_min']} RSI:{r['rsi_tf']}{r['rsi_min']}-{r['rsi_max']} "
              f"SL:{r['sl']*100:.0f}% T+{r['trail_act']*100:.0f}%/-{r['trail_pct']*100:.0f}% M:{r['margin']*100:.0f}% "
              f"REV:{r['rev_mode']} MB:{r['min_bars']} "
              f"| Ret:{r['return_pct']:.0f}% PF:{r['pf']:.2f} MDD:{r['mdd']:.1f}% T:{r['trades']} "
              f"SL:{r['sl_count']} TSL:{r['tsl_count']} REV:{r['rev_count']}", flush=True)

    # ================================================================
    # Stage B: Extra filters on top winners
    # ================================================================
    print("\n  Stage B: Adding Stochastic + CCI + OBV + Ichimoku + BB filters...", flush=True)

    top_a = stage_a_results[:100]
    stage_b_results = []
    tested_b = 0
    t0 = time.time()

    for r in top_a:
        ck = r['cross']
        csig = cross_sigs[ck]
        adx_v = adx_m[f"{r['adx_tf']}_{r['adx_p']}"]
        rsi_v = rsi_m[f"{r['rsi_tf']}_{r['rsi_p']}"]
        macd_v = macd_m[r['macd']]

        for so, sob in stoch_configs:
            for cmin, cmax in cci_configs:
                for obv_t in obv_thresholds:
                    for ich in ich_options:
                        for bb_use, bb_min in bb_options:
                            for pe_ratio, pe_roi in partial_configs:
                                tested_b += 1

                                result = run_backtest_v271(
                                    close_5m, high_5m, low_5m, ts_i64,
                                    csig, adx_v, rsi_v, atr_5m, macd_v,
                                    stoch_k_5m, stoch_d_5m, cci_5m, obv_5m, bb_w_5m, ich_5m,
                                    r['adx_min'], r['rsi_min'], r['rsi_max'],
                                    so, sob, cmin, cmax, obv_t, ich, bb_use, bb_min,
                                    r['sl'], r['trail_act'], r['trail_pct'], pe_ratio, pe_roi,
                                    10, r['margin'], r['margin_dd'], -0.20, 0.0004, 3000.0,
                                    r['entry_delay'], r['entry_tol'], r['rev_mode'], r['min_bars']
                                )

                                fc, tc = result[0], result[1]
                                if tc >= 10 and fc > 5000:
                                    rois = result[3]; pnls = result[4]
                                    wins = np.sum(rois > 0)
                                    tpro = np.sum(pnls[pnls > 0])
                                    tlos = abs(np.sum(pnls[pnls < 0])) + 1e-10
                                    pf = tpro / tlos
                                    ret = (fc - 3000) / 3000 * 100

                                    eq = result[8]; mdd = 0
                                    if len(eq) > 0:
                                        peq = np.maximum.accumulate(eq)
                                        dd = (eq - peq) / (peq + 1e-10)
                                        mdd = abs(np.min(dd)) * 100

                                    etypes = result[6]

                                    if pf >= 4.0:
                                        stage_b_results.append({
                                            **{k:v for k,v in r.items() if k != 'score'},
                                            'stoch_os': so, 'stoch_ob': sob,
                                            'cci_min': cmin, 'cci_max': cmax,
                                            'obv_threshold': obv_t,
                                            'use_ichimoku': ich,
                                            'use_bb': bb_use, 'bb_width_min': bb_min,
                                            'partial_ratio': pe_ratio, 'partial_roi': pe_roi,
                                            'final_cap': float(fc), 'trades': int(tc),
                                            'win_rate': float(wins/tc*100),
                                            'pf': float(pf), 'mdd': float(mdd),
                                            'return_pct': float(ret),
                                            'avg_win': float(np.mean(rois[rois > 0])*100) if wins > 0 else 0,
                                            'avg_loss': float(np.mean(rois[rois <= 0])*100) if (tc-wins) > 0 else 0,
                                            'sl_count': int(np.sum(etypes==0)),
                                            'tsl_count': int(np.sum(etypes==1)),
                                            'rev_count': int(np.sum(etypes==2)),
                                        })

        if (top_a.index(r)+1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"    Config {top_a.index(r)+1}/{len(top_a)} | {tested_b:,} tested | {len(stage_b_results)} passed (PF>=4)", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Stage B: {tested_b:,} in {elapsed:.0f}s -> {len(stage_b_results)} passed", flush=True)

    for r in stage_b_results:
        r['score'] = r['pf'] * np.sqrt(r['trades']) * r['return_pct'] / (r['mdd'] + 5)

    stage_b_results.sort(key=lambda x: x['score'], reverse=True)

    # If PF 5+ found, filter
    pf5_results = [r for r in stage_b_results if r['pf'] >= 5.0]
    print(f"  PF >= 5.0: {len(pf5_results)} found", flush=True)
    pf4_results = [r for r in stage_b_results if r['pf'] >= 4.0]
    print(f"  PF >= 4.0: {len(pf4_results)} found", flush=True)

    # Use best available
    final_pool = pf5_results if len(pf5_results) >= 3 else (pf4_results if len(pf4_results) >= 3 else stage_b_results)
    if len(final_pool) == 0:
        final_pool = stage_a_results

    print(f"\n  Top 10 Final:", flush=True)
    for i, r in enumerate(final_pool[:10]):
        print(f"    #{i+1}: {r['cross']} PF:{r['pf']:.2f} Ret:{r['return_pct']:.0f}% MDD:{r['mdd']:.1f}% "
              f"T:{r['trades']} WR:{r['win_rate']:.0f}% SL:{r['sl_count']} TSL:{r['tsl_count']} REV:{r['rev_count']}", flush=True)

    # ================================================================
    # SELECT TOP 3
    # ================================================================
    print("\n[Model Selection]", flush=True)

    model_a = max(final_pool[:30], key=lambda x: x['return_pct'] * x['pf'])
    model_b = max(final_pool[:30], key=lambda x: x['pf'] / (x['mdd'] + 3))
    model_c = max(final_pool[:30], key=lambda x: x['trades'] * x['pf'] * np.log(max(x['return_pct'],1)+1) / (x['mdd']+5))

    models = {'A': model_a, 'B': model_b, 'C': model_c}

    for name, m in models.items():
        print(f"\n  Model {name}: {m['cross']}", flush=True)
        print(f"    ADX:{m['adx_tf']}>{m['adx_min']} RSI:{m['rsi_tf']}{m['rsi_min']}-{m['rsi_max']} MACD:{m['macd']}", flush=True)
        print(f"    Stoch:{m.get('stoch_os',20)}/{m.get('stoch_ob',80)} CCI:{m.get('cci_min',-200)}/{m.get('cci_max',200)}", flush=True)
        print(f"    OBV:{m.get('obv_threshold',0)} Ichimoku:{m.get('use_ichimoku',0)} BB:{m.get('use_bb',0)}/{m.get('bb_width_min',0)}", flush=True)
        print(f"    SL:{m['sl']*100:.0f}% Trail:+{m['trail_act']*100:.0f}%/-{m['trail_pct']*100:.0f}% M:{m['margin']*100:.0f}%", flush=True)
        print(f"    REV:{m['rev_mode']} MinBars:{m['min_bars']}", flush=True)
        print(f"    Return:{m['return_pct']:.0f}% PF:{m['pf']:.2f} MDD:{m['mdd']:.1f}% T:{m['trades']}", flush=True)
        print(f"    WR:{m['win_rate']:.1f}% AvgW:{m['avg_win']:.1f}% AvgL:{m['avg_loss']:.1f}%", flush=True)
        print(f"    SL:{m['sl_count']} TSL:{m['tsl_count']} REV:{m['rev_count']}", flush=True)
        print(f"    Final: ${m['final_cap']:,.0f}", flush=True)

    # ================================================================
    # 30x VALIDATION
    # ================================================================
    print("\n[30x Validation]", flush=True)

    validation = {}
    for name, m in models.items():
        print(f"\n  Validating Model {name}...", flush=True)
        csig = cross_sigs[m['cross']]
        adx_v = adx_m[f"{m['adx_tf']}_{m['adx_p']}"]
        rsi_v = rsi_m[f"{m['rsi_tf']}_{m['rsi_p']}"]
        macd_v = macd_m[m['macd']]

        val_results = []
        for run in range(30):
            skip = 0.02 + (run/30)*0.13
            start = int(n * skip)
            sl_e = n

            cs_s = csig[start:sl_e]
            adx_s = adx_v[start:sl_e]
            rsi_s = rsi_v[start:sl_e]
            atr_s = atr_5m[start:sl_e]
            macd_s = macd_v[start:sl_e]
            sk_s = stoch_k_5m[start:sl_e]
            sd_s = stoch_d_5m[start:sl_e]
            cci_s = cci_5m[start:sl_e]
            obv_s = obv_5m[start:sl_e]
            bbw_s = bb_w_5m[start:sl_e]
            ich_s = ich_5m[start:sl_e]
            c_s = close_5m[start:sl_e]
            h_s = high_5m[start:sl_e]
            l_s = low_5m[start:sl_e]
            t_s = ts_i64[start:sl_e]

            ml = min(len(c_s), len(cs_s), len(adx_s), len(rsi_s), len(atr_s),
                     len(macd_s), len(sk_s), len(cci_s), len(obv_s), len(bbw_s), len(ich_s))

            result = run_backtest_v271(
                c_s[:ml], h_s[:ml], l_s[:ml], t_s[:ml],
                cs_s[:ml], adx_s[:ml], rsi_s[:ml], atr_s[:ml], macd_s[:ml],
                sk_s[:ml], sd_s[:ml], cci_s[:ml], obv_s[:ml], bbw_s[:ml], ich_s[:ml],
                m['adx_min'], m['rsi_min'], m['rsi_max'],
                m.get('stoch_os', 20), m.get('stoch_ob', 80),
                m.get('cci_min', -200), m.get('cci_max', 200),
                m.get('obv_threshold', 0), m.get('use_ichimoku', 0),
                m.get('use_bb', 0), m.get('bb_width_min', 0.01),
                m['sl'], m['trail_act'], m['trail_pct'],
                m.get('partial_ratio', 0), m.get('partial_roi', 0),
                10, m['margin'], m['margin_dd'], -0.20, 0.0004, 3000.0,
                m['entry_delay'], m['entry_tol'], m['rev_mode'], m['min_bars']
            )

            fc, tc = result[0], result[1]
            rois = result[3]; pnls = result[4]
            wins = np.sum(rois > 0) if tc > 0 else 0
            tpro = np.sum(pnls[pnls > 0])
            tlos = abs(np.sum(pnls[pnls < 0])) + 1e-10
            pf = tpro / tlos
            eq = result[8]; mdd = 0
            if len(eq) > 0:
                peq = np.maximum.accumulate(eq)
                dd = (eq - peq) / (peq + 1e-10)
                mdd = abs(np.min(dd)) * 100

            val_results.append({
                'run': run+1, 'final_cap': float(fc),
                'return_pct': float((fc-3000)/3000*100),
                'trades': int(tc), 'win_rate': float(wins/tc*100) if tc > 0 else 0,
                'pf': float(pf), 'mdd': float(mdd),
            })

        validation[name] = val_results
        rets = [v['return_pct'] for v in val_results]
        pfs = [v['pf'] for v in val_results]
        mdds = [v['mdd'] for v in val_results]
        print(f"    Ret: {np.mean(rets):.0f}% +/- {np.std(rets):.0f}% (min:{np.min(rets):.0f}%)", flush=True)
        print(f"    PF: {np.mean(pfs):.2f} +/- {np.std(pfs):.2f} (min:{np.min(pfs):.2f})", flush=True)
        print(f"    MDD: {np.mean(mdds):.1f}% +/- {np.std(mdds):.1f}%", flush=True)

    # ================================================================
    # SAVE
    # ================================================================
    total_tested = tested + tested_b
    output = {
        'version': 'v27.1',
        'total_tested': total_tested,
        'stage_a_tested': tested, 'stage_a_passed': len(stage_a_results),
        'stage_b_tested': tested_b, 'stage_b_passed': len(stage_b_results),
        'pf5_count': len(pf5_results), 'pf4_count': len(pf4_results),
        'models': {name: {k:v for k,v in m.items() if k != 'score'} for name, m in models.items()},
        'validation': validation,
        'top20': [{k:v for k,v in r.items() if k != 'score'} for r in final_pool[:20]],
    }

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v27_1_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=convert, ensure_ascii=False)

    print(f"\nSaved: {path}", flush=True)
    print(f"Total tested: {total_tested:,}", flush=True)
    print("="*70, flush=True)
    print("v27.1 OPTIMIZATION COMPLETE", flush=True)


if __name__ == '__main__':
    run_optimization()
