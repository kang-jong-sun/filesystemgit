"""
v28 AI 트레이딩 자동매매 백테스트 엔진
- Numba JIT 고속 최적화
- 멀티 타임프레임 지원
- 1,000,000+ 조합 스캔
- 선행+후행 지표 통합
"""

import numpy as np
import pandas as pd
from numba import njit, prange
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 데이터 로드 및 멀티 타임프레임 리샘플링
# ============================================================

def load_5m_data(base_path):
    """5분봉 데이터 3파트 로드 및 병합"""
    parts = []
    for i in range(1, 4):
        f = os.path.join(base_path, f"btc_usdt_5m_2020_to_now_part{i}.csv")
        df = pd.read_csv(f, parse_dates=['timestamp'])
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    print(f"[DATA] 5분봉 로드 완료: {len(df)}행, {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")
    return df


def resample_ohlcv(df_5m, minutes):
    """5분봉을 상위 타임프레임으로 리샘플링"""
    df = df_5m.set_index('timestamp')
    resampled = df.resample(f'{minutes}min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trades': 'sum'
    }).dropna()
    return resampled.reset_index()


def build_multitf_data(df_5m):
    """멀티 타임프레임 데이터 생성"""
    tf_data = {'5m': df_5m}
    for mins, label in [(10, '10m'), (15, '15m'), (30, '30m'), (60, '1h')]:
        tf_data[label] = resample_ohlcv(df_5m, mins)
        print(f"[DATA] {label} 리샘플링 완료: {len(tf_data[label])}행")
    return tf_data


# ============================================================
# 2. 지표 계산 함수 (Numba 최적화)
# ============================================================

@njit
def calc_ema(data, period):
    """EMA 계산 (Wilder's 아닌 표준 EMA)"""
    n = len(data)
    out = np.empty(n)
    out[:] = np.nan
    if n < period:
        return out
    k = 2.0 / (period + 1.0)
    out[period - 1] = np.mean(data[:period])
    for i in range(period, n):
        out[i] = data[i] * k + out[i - 1] * (1.0 - k)
    return out


@njit
def calc_sma(data, period):
    """SMA 계산"""
    n = len(data)
    out = np.empty(n)
    out[:] = np.nan
    for i in range(period - 1, n):
        out[i] = np.mean(data[i - period + 1:i + 1])
    return out


@njit
def calc_wma(data, period):
    """WMA 계산"""
    n = len(data)
    out = np.empty(n)
    out[:] = np.nan
    w_sum = period * (period + 1) / 2.0
    for i in range(period - 1, n):
        s = 0.0
        for j in range(period):
            s += data[i - period + 1 + j] * (j + 1)
        out[i] = s / w_sum
    return out


@njit
def calc_hma(data, period):
    """HMA 계산: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    n = len(data)
    half_p = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    wma_half = calc_wma(data, half_p)
    wma_full = calc_wma(data, period)
    diff = np.empty(n)
    diff[:] = np.nan
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    return calc_wma(diff, sqrt_p)


@njit
def calc_vwma(close, volume, period):
    """VWMA 계산"""
    n = len(close)
    out = np.empty(n)
    out[:] = np.nan
    for i in range(period - 1, n):
        cv_sum = 0.0
        v_sum = 0.0
        for j in range(period):
            cv_sum += close[i - period + 1 + j] * volume[i - period + 1 + j]
            v_sum += volume[i - period + 1 + j]
        if v_sum > 0:
            out[i] = cv_sum / v_sum
    return out


@njit
def calc_rsi_wilder(data, period):
    """RSI (Wilder's Smoothing)"""
    n = len(data)
    out = np.empty(n)
    out[:] = np.nan
    if n < period + 1:
        return out
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        d = data[i] - data[i - 1]
        if d > 0:
            gains[i] = d
        else:
            losses[i] = -d
    avg_gain = np.mean(gains[1:period + 1])
    avg_loss = np.mean(losses[1:period + 1])
    if avg_loss == 0:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i] = 100.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


@njit
def calc_adx_wilder(high, low, close, period):
    """ADX (Wilder's Smoothing - v16.4 수정 버전)"""
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
    # Wilder smoothing 시드: MEAN (v16.4 수정)
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
    # ADX = Wilder smoothing of DX
    adx_start = 2 * period
    if adx_start >= n:
        return adx
    adx[adx_start] = np.mean(dx_arr[period:adx_start + 1])
    for i in range(adx_start + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx_arr[i]) / period
    return adx


@njit
def calc_atr(high, low, close, period):
    """ATR 계산"""
    n = len(close)
    out = np.empty(n)
    out[:] = np.nan
    tr = np.zeros(n)
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, max(tr2, tr3))
    if n < period + 1:
        return out
    out[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


@njit
def calc_stochastic(high, low, close, k_period, d_period):
    """Stochastic Oscillator (%K, %D)"""
    n = len(close)
    k_out = np.empty(n)
    k_out[:] = np.nan
    d_out = np.empty(n)
    d_out[:] = np.nan
    for i in range(k_period - 1, n):
        hh = high[i]
        ll = low[i]
        for j in range(1, k_period):
            if high[i - j] > hh:
                hh = high[i - j]
            if low[i - j] < ll:
                ll = low[i - j]
        if hh - ll > 0:
            k_out[i] = 100.0 * (close[i] - ll) / (hh - ll)
        else:
            k_out[i] = 50.0
    d_out = calc_sma(k_out, d_period)
    return k_out, d_out


@njit
def calc_obv(close, volume):
    """OBV 계산"""
    n = len(close)
    out = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - volume[i]
        else:
            out[i] = out[i - 1]
    return out


@njit
def calc_cci(high, low, close, period):
    """CCI 계산"""
    n = len(close)
    out = np.empty(n)
    out[:] = np.nan
    tp = (high + low + close) / 3.0
    for i in range(period - 1, n):
        mean_tp = np.mean(tp[i - period + 1:i + 1])
        mad = 0.0
        for j in range(period):
            mad += abs(tp[i - period + 1 + j] - mean_tp)
        mad /= period
        if mad > 0:
            out[i] = (tp[i] - mean_tp) / (0.015 * mad)
        else:
            out[i] = 0.0
    return out


@njit
def calc_macd(close, fast, slow, signal):
    """MACD 계산"""
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    n = len(close)
    macd_line = np.empty(n)
    macd_line[:] = np.nan
    for i in range(n):
        if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]):
            macd_line[i] = ema_fast[i] - ema_slow[i]
    signal_line = calc_ema(macd_line, signal)
    histogram = np.empty(n)
    histogram[:] = np.nan
    for i in range(n):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
            histogram[i] = macd_line[i] - signal_line[i]
    return macd_line, signal_line, histogram


@njit
def calc_bollinger(close, period, std_mult):
    """볼린저 밴드"""
    n = len(close)
    mid = calc_sma(close, period)
    upper = np.empty(n)
    lower = np.empty(n)
    upper[:] = np.nan
    lower[:] = np.nan
    for i in range(period - 1, n):
        s = 0.0
        for j in range(period):
            d = close[i - period + 1 + j] - mid[i]
            s += d * d
        std = np.sqrt(s / period)
        upper[i] = mid[i] + std_mult * std
        lower[i] = mid[i] - std_mult * std
    return mid, upper, lower


# ============================================================
# 3. MA 셀렉터
# ============================================================

@njit
def calc_ma(close, volume, ma_type, period):
    """MA 타입별 계산 (0=EMA, 1=WMA, 2=SMA, 3=HMA, 4=VWMA)"""
    if ma_type == 0:
        return calc_ema(close, period)
    elif ma_type == 1:
        return calc_wma(close, period)
    elif ma_type == 2:
        return calc_sma(close, period)
    elif ma_type == 3:
        return calc_hma(close, period)
    elif ma_type == 4:
        return calc_vwma(close, volume, period)
    else:
        return calc_ema(close, period)


# ============================================================
# 4. 핵심 백테스트 엔진 (Numba JIT)
# ============================================================

@njit
def backtest_core(
    close, high, low, volume, timestamps_epoch,
    fast_ma, slow_ma, adx, rsi,
    # 파라미터
    adx_thresh, rsi_lo, rsi_hi,
    sl_pct, trail_act_pct, trail_pct,
    margin_pct, leverage,
    entry_delay_bars, entry_offset_pct,
    fee_rate,
    skip_same_dir,
    # 기간 가중치 (2023이후=1.0, 이전=0.5 가중시 사용)
    weight_start_idx
):
    """
    핵심 백테스트 루프
    Returns: (final_balance, total_trades, wins, losses, pf, mdd, sl_count,
              tsl_count, rev_count, max_consec_loss, total_profit, total_loss,
              yearly_returns_6)  # 2020~2025 연도별
    """
    n = len(close)
    balance = 3000.0
    peak_balance = 3000.0
    max_dd = 0.0

    position = 0  # 0=없음, 1=롱, -1=숏
    entry_price = 0.0
    position_size = 0.0
    position_margin = 0.0
    highest_roi = 0.0
    trail_active = False
    trail_sl_price = 0.0

    # 크로스 감지 상태
    pending_signal = 0  # 1=롱 대기, -1=숏 대기
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

    # 월간 손실 제한
    month_start_balance = 3000.0
    current_month = -1

    # 연도별 수익 추적 (2020~2025, idx 0~5)
    yearly_start = np.zeros(7)  # 2020~2026
    yearly_end = np.zeros(7)
    for i in range(7):
        yearly_start[i] = -1.0

    for i in range(1, n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            continue

        cur_price = close[i]

        # 월간 리셋 (대략 30일=8640 5분봉으로 근사)
        approx_month = i // 8640
        if approx_month != current_month:
            current_month = approx_month
            month_start_balance = balance

        # 월간 손실 -20% 체크
        if balance < month_start_balance * 0.80:
            # 포지션 있으면 청산
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

        # 포지션 관리
        if position != 0:
            if position == 1:
                roi = (cur_price - entry_price) / entry_price
            else:
                roi = (entry_price - cur_price) / entry_price

            # SL 체크
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

            # 트레일링 활성화
            if roi >= trail_act_pct / 100.0:
                trail_active = True
                if roi > highest_roi:
                    highest_roi = roi

            # 트레일링 스톱 체크
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

        # 크로스 감지
        cross_signal = 0
        if fast_ma[i - 1] <= slow_ma[i - 1] and fast_ma[i] > slow_ma[i]:
            cross_signal = 1  # 골든크로스 → 롱
        elif fast_ma[i - 1] >= slow_ma[i - 1] and fast_ma[i] < slow_ma[i]:
            cross_signal = -1  # 데드크로스 → 숏

        if cross_signal != 0:
            # 역신호 청산 (REV)
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

            # ADX/RSI 필터
            if adx[i] >= adx_thresh and rsi_lo <= rsi[i] <= rsi_hi:
                if skip_same_dir and cross_signal == last_closed_dir:
                    pass  # 동일방향 재진입 스킵
                else:
                    pending_signal = cross_signal
                    pending_bar = i
                    pending_price = cur_price

        # 지연 진입 체크
        if pending_signal != 0 and position == 0:
            bars_elapsed = i - pending_bar
            if bars_elapsed >= entry_delay_bars:
                # 가격 오프셋 체크
                if pending_signal == 1:
                    price_change = (cur_price - pending_price) / pending_price * 100.0
                    ok = price_change >= entry_offset_pct  # 양 또는 허용 범위
                else:
                    price_change = (pending_price - cur_price) / pending_price * 100.0
                    ok = price_change >= entry_offset_pct

                # ADX/RSI 재확인
                if ok and adx[i] >= adx_thresh and rsi_lo <= rsi[i] <= rsi_hi:
                    entry_price = cur_price
                    position_margin = balance * margin_pct / 100.0
                    position_size = position_margin * leverage / cur_price
                    position = pending_signal
                    highest_roi = 0.0
                    trail_active = False
                    pending_signal = 0
                elif bars_elapsed > entry_delay_bars + 12:  # 60분(12x5분) 초과시 만료
                    pending_signal = 0

        # MDD 추적
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100.0
        if dd > max_dd:
            max_dd = dd

        # 연도별 추적 (5분봉 기준 대략 105,120/년)
        year_idx = i // 105120  # 대략적 연도 인덱스
        if year_idx < 7:
            if yearly_start[year_idx] < 0:
                yearly_start[year_idx] = balance
            yearly_end[year_idx] = balance

    # 잔여 포지션 청산
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


# ============================================================
# 5. 대규모 스캔 프레임워크
# ============================================================

def precompute_indicators(close, high, low, volume):
    """모든 필요한 지표를 미리 계산"""
    indicators = {}

    # MA 타입별 (0=EMA, 1=WMA, 2=SMA, 3=HMA, 4=VWMA)
    ma_types = [0, 1, 2, 3, 4]
    ma_names = ['EMA', 'WMA', 'SMA', 'HMA', 'VWMA']

    fast_periods = [2, 3, 5, 7, 10, 14, 21]
    slow_periods = [50, 100, 150, 200, 250, 300]

    print("[IND] MA 계산 시작...")
    for mt_idx, mt in enumerate(ma_types):
        for p in fast_periods + slow_periods:
            key = f"ma_{ma_names[mt_idx]}_{p}"
            if key not in indicators:
                indicators[key] = calc_ma(close, volume, mt, p)

    # ADX
    print("[IND] ADX 계산 시작...")
    for p in [14, 20]:
        indicators[f"adx_{p}"] = calc_adx_wilder(high, low, close, p)

    # RSI
    print("[IND] RSI 계산 시작...")
    for p in [14]:
        indicators[f"rsi_{p}"] = calc_rsi_wilder(close, p)

    # ATR
    print("[IND] ATR 계산 시작...")
    for p in [14]:
        indicators[f"atr_{p}"] = calc_atr(high, low, close, p)

    # Stochastic
    print("[IND] Stochastic 계산 시작...")
    k, d = calc_stochastic(high, low, close, 14, 3)
    indicators['stoch_k'] = k
    indicators['stoch_d'] = d

    # OBV
    indicators['obv'] = calc_obv(close, volume)

    # CCI
    indicators['cci_20'] = calc_cci(high, low, close, 20)

    # MACD
    macd_l, macd_s, macd_h = calc_macd(close, 12, 26, 9)
    indicators['macd_line'] = macd_l
    indicators['macd_signal'] = macd_s
    indicators['macd_hist'] = macd_h

    # Bollinger
    bb_mid, bb_up, bb_lo = calc_bollinger(close, 20, 2.0)
    indicators['bb_mid'] = bb_mid
    indicators['bb_upper'] = bb_up
    indicators['bb_lower'] = bb_lo

    print(f"[IND] 총 {len(indicators)}개 지표 계산 완료")
    return indicators


def generate_param_combinations(n_combos=1000000):
    """1,000,000+ 파라미터 조합 랜덤 생성"""
    np.random.seed(42)

    ma_names = ['EMA', 'WMA', 'SMA', 'HMA', 'VWMA']
    fast_periods = [2, 3, 5, 7, 10, 14, 21]
    slow_periods = [50, 100, 150, 200, 250, 300]
    adx_periods = [14, 20]
    adx_thresholds = [25, 30, 35, 40, 45]
    rsi_lows = [25, 30, 35, 40]
    rsi_highs = [60, 65, 70, 75]
    sl_pcts = [-4, -5, -6, -7, -8, -9, -10]
    trail_acts = [3, 4, 5, 6, 7, 8, 10]
    trail_pcts = [1, 2, 3, 4, 5]
    margin_pcts = [15, 20, 25, 30, 35, 40, 50, 60]
    leverages = [5, 7, 10, 15]
    entry_delays = [0, 1, 2, 3, 5, 6]  # 5분봉 bar 수 (0~30분)
    entry_offsets = [-2.5, -1.5, -1.0, -0.5, -0.1, 0.0]
    skip_same = [0, 1]

    combos = []
    for _ in range(n_combos):
        fast_ma_type = np.random.choice(len(ma_names))
        slow_ma_type = np.random.choice(len(ma_names))
        fast_p = np.random.choice(fast_periods)
        slow_p = np.random.choice(slow_periods)
        adx_p = np.random.choice(adx_periods)
        adx_t = np.random.choice(adx_thresholds)
        rsi_lo = np.random.choice(rsi_lows)
        rsi_hi = np.random.choice(rsi_highs)
        sl = np.random.choice(sl_pcts)
        t_act = np.random.choice(trail_acts)
        t_pct = np.random.choice(trail_pcts)
        m_pct = np.random.choice(margin_pcts)
        lev = np.random.choice(leverages)
        delay = np.random.choice(entry_delays)
        offset = np.random.choice(entry_offsets)
        skip = np.random.choice(skip_same)

        combos.append((
            fast_ma_type, slow_ma_type, fast_p, slow_p,
            adx_p, adx_t, rsi_lo, rsi_hi,
            sl, t_act, t_pct, m_pct, lev,
            delay, offset, skip
        ))

    return combos


def run_scan(tf_label, df, indicators, combos, top_n=200):
    """대규모 스캔 실행"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    n = len(close)
    timestamps_epoch = np.zeros(n)  # 플레이스홀더

    ma_names = ['EMA', 'WMA', 'SMA', 'HMA', 'VWMA']
    results = []
    total = len(combos)

    print(f"\n[SCAN] {tf_label} 스캔 시작: {total:,}개 조합")
    start_time = time.time()

    for idx, combo in enumerate(combos):
        (fast_mt, slow_mt, fast_p, slow_p,
         adx_p, adx_t, rsi_lo, rsi_hi,
         sl, t_act, t_pct, m_pct, lev,
         delay, offset, skip) = combo

        fast_key = f"ma_{ma_names[fast_mt]}_{fast_p}"
        slow_key = f"ma_{ma_names[slow_mt]}_{slow_p}"
        adx_key = f"adx_{adx_p}"
        rsi_key = f"rsi_14"

        if fast_key not in indicators or slow_key not in indicators:
            continue

        fast_ma = indicators[fast_key]
        slow_ma = indicators[slow_key]
        adx = indicators[adx_key]
        rsi = indicators[rsi_key]

        result = backtest_core(
            close, high, low, volume, timestamps_epoch,
            fast_ma, slow_ma, adx, rsi,
            float(adx_t), float(rsi_lo), float(rsi_hi),
            float(sl), float(t_act), float(t_pct),
            float(m_pct), float(lev),
            int(delay), float(offset),
            0.0004,  # 수수료 0.04%
            int(skip),
            n // 3  # 2023 시작 대략 인덱스
        )

        bal, trades, w, l, pf, mdd, sl_c, tsl_c, rev_c, mcl, tp, tl = result[:12]

        # 필터: 최소 20거래, 잔액 > 초기
        if trades >= 20 and bal > 3000:
            results.append({
                'combo': combo,
                'balance': bal,
                'trades': trades,
                'wins': w,
                'losses': l,
                'pf': pf,
                'mdd': mdd,
                'sl_count': sl_c,
                'tsl_count': tsl_c,
                'rev_count': rev_c,
                'total_profit': tp,
                'total_loss': tl,
                'win_rate': w / trades * 100 if trades > 0 else 0,
                'return_pct': (bal - 3000) / 3000 * 100
            })

        if (idx + 1) % 50000 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            eta = (total - idx - 1) / speed / 60
            print(f"  [{idx+1:>8,}/{total:,}] {speed:.0f} combo/s, ETA: {eta:.1f}분, 유효: {len(results)}")

    elapsed = time.time() - start_time
    print(f"[SCAN] 완료: {elapsed:.1f}초, {total/elapsed:.0f} combo/s, 유효 조합: {len(results)}")

    # 정렬: PF x Return 복합 점수
    results.sort(key=lambda x: x['pf'] * min(x['return_pct'], 100000), reverse=True)
    return results[:top_n]


def run_detailed_backtest(df, indicators, combo, runs=30):
    """상세 백테스트 + 30회 반복 검증"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    n = len(close)
    timestamps_epoch = np.zeros(n)

    ma_names = ['EMA', 'WMA', 'SMA', 'HMA', 'VWMA']

    (fast_mt, slow_mt, fast_p, slow_p,
     adx_p, adx_t, rsi_lo, rsi_hi,
     sl, t_act, t_pct, m_pct, lev,
     delay, offset, skip) = combo

    fast_ma = indicators[f"ma_{ma_names[fast_mt]}_{fast_p}"]
    slow_ma = indicators[f"ma_{ma_names[slow_mt]}_{slow_p}"]
    adx = indicators[f"adx_{adx_p}"]
    rsi = indicators[f"rsi_14"]

    results = []
    for run in range(runs):
        result = backtest_core(
            close, high, low, volume, timestamps_epoch,
            fast_ma, slow_ma, adx, rsi,
            float(adx_t), float(rsi_lo), float(rsi_hi),
            float(sl), float(t_act), float(t_pct),
            float(m_pct), float(lev),
            int(delay), float(offset),
            0.0004, int(skip), n // 3
        )
        results.append(result)

    # 결정론적 검증
    balances = [r[0] for r in results]
    std = np.std(balances)
    mean_bal = np.mean(balances)

    return {
        'combo': combo,
        'combo_str': f"{ma_names[fast_mt]}({fast_p})/{ma_names[slow_mt]}({slow_p}) ADX({adx_p})>={adx_t} RSI {rsi_lo}-{rsi_hi} SL{sl}% Trail+{t_act}/-{t_pct} M{m_pct}% Lev{lev}x Delay{delay} Offset{offset} Skip{skip}",
        'balance': mean_bal,
        'trades': results[0][1],
        'wins': results[0][2],
        'losses': results[0][3],
        'pf': results[0][4],
        'mdd': results[0][5],
        'sl_count': results[0][6],
        'tsl_count': results[0][7],
        'rev_count': results[0][8],
        'max_consec_loss': results[0][9],
        'total_profit': results[0][10],
        'total_loss': results[0][11],
        'return_pct': (mean_bal - 3000) / 3000 * 100,
        'win_rate': results[0][2] / results[0][1] * 100 if results[0][1] > 0 else 0,
        'runs': runs,
        'std': std,
        'deterministic': std < 0.01,
        'yearly_returns': [
            (results[0][12], results[0][13]),  # 2020
            (results[0][14], results[0][15]),  # 2021
            (results[0][16], results[0][17]),  # 2022
            (results[0][18], results[0][19]),  # 2023
            (results[0][20], results[0][21]),  # 2024
            (results[0][22], results[0][23]),  # 2025
            (results[0][24], results[0][25]),  # 2026
        ]
    }


# ============================================================
# 6. 멀티 타임프레임 스캔
# ============================================================

def run_multitf_scan(tf_data, n_combos=200000):
    """각 타임프레임별 스캔 후 결과 통합"""
    all_results = {}

    for tf_label in ['5m', '10m', '15m', '30m', '1h']:
        df = tf_data[tf_label]
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        print(f"\n{'='*60}")
        print(f"[MTF] {tf_label} 타임프레임 처리 시작 ({len(df)}행)")
        print(f"{'='*60}")

        indicators = precompute_indicators(close, high, low, volume)
        combos = generate_param_combinations(n_combos)
        top_results = run_scan(tf_label, df, indicators, combos, top_n=100)
        all_results[tf_label] = (top_results, df, indicators)

    return all_results


# ============================================================
# 7. 메인 실행
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  v28 AI 트레이딩 백테스트 엔진 시작")
    print("=" * 70)

    base = r"D:\filesystem\futures\btc_V1\test4"

    # 1) 데이터 로드
    print("\n[PHASE 1] 데이터 로드...")
    df_5m = load_5m_data(base)

    # 2) 멀티 타임프레임 생성
    print("\n[PHASE 2] 멀티 타임프레임 생성...")
    tf_data = build_multitf_data(df_5m)

    # 3) 각 TF별 스캔 (각 200,000 조합 = 총 1,000,000)
    print("\n[PHASE 3] 1,000,000 조합 대규모 스캔...")
    all_results = run_multitf_scan(tf_data, n_combos=200000)

    # 4) 전체 TF 통합 Top 50
    print("\n[PHASE 4] 전체 타임프레임 통합 Top 50 선정...")
    unified = []
    for tf_label, (results, df, indicators) in all_results.items():
        for r in results[:20]:  # 각 TF Top 20
            r['tf'] = tf_label
            unified.append(r)

    unified.sort(key=lambda x: x['pf'] * min(x['return_pct'], 100000), reverse=True)
    top50 = unified[:50]

    # 5) Top 50 상세 백테스트 + 30회 검증
    print("\n[PHASE 5] Top 50 상세 백테스트 (30회 반복 검증)...")
    detailed_results = []
    for rank, r in enumerate(top50):
        tf_label = r['tf']
        _, df, indicators = all_results[tf_label]
        detail = run_detailed_backtest(df, indicators, r['combo'], runs=30)
        detail['tf'] = tf_label
        detail['rank'] = rank + 1
        detailed_results.append(detail)
        print(f"  #{rank+1:2d} [{tf_label}] {detail['combo_str'][:60]}... "
              f"PF={detail['pf']:.2f} MDD={detail['mdd']:.1f}% "
              f"Return={detail['return_pct']:.0f}% "
              f"Trades={detail['trades']} "
              f"{'PASS' if detail['deterministic'] else 'FAIL'}")

    # 6) 결과 저장
    print("\n[PHASE 6] 결과 저장...")
    save_path = os.path.join(base, "v28_scan_results.json")
    save_data = []
    for d in detailed_results:
        entry = {k: v for k, v in d.items() if k != 'combo'}
        entry['combo'] = [int(x) if isinstance(x, (np.integer, int)) else float(x) for x in d['combo']]
        # yearly_returns 변환
        yr = []
        for s, e in d['yearly_returns']:
            yr.append({'start': float(s), 'end': float(e)})
        entry['yearly_returns'] = yr
        save_data.append(entry)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {save_path}")

    # 7) 요약 출력
    print("\n" + "=" * 70)
    print("  v28 최종 Top 10 전략")
    print("=" * 70)
    for d in detailed_results[:10]:
        print(f"\n  #{d['rank']} [{d['tf']}] {d['combo_str']}")
        print(f"    잔액: ${d['balance']:,.0f} | 수익률: +{d['return_pct']:,.0f}%")
        print(f"    PF: {d['pf']:.2f} | MDD: {d['mdd']:.1f}% | 거래: {d['trades']}회")
        print(f"    승률: {d['win_rate']:.1f}% | SL: {d['sl_count']}회 | TSL: {d['tsl_count']}회 | REV: {d['rev_count']}회")
        print(f"    30회 검증: {'PASS' if d['deterministic'] else 'FAIL'} (std={d['std']:.4f})")

    print("\n[완료] 전체 프로세스 종료")
