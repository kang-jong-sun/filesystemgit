"""
verify3_engine.py
=================
5 strategies x 6 engines backtest -> Excel "기획서 검증 V3.xlsx"

Strategies:
  v23.4  : EMA(3)/EMA(200) ADX(14)>=35 RSI(14) 30-65  SL-7% TA+6% TSL-3%  M30% ML-20%(daily)
  v32.2  : EMA(100)/EMA(600) ADX(20)>=30 RSI(10) 40-80 SL-3% TA+12% TSL-9% M35% DL-20%(daily)
  v32.3  : EMA(75)/SMA(750) ADX(20)>=30 RSI(11) 40-80  SL-3% TA+12% TSL-9% M35% DL-20%(daily)
  v15.4  : EMA(3)/EMA(200) ADX(14)>=35 RSI(14) 30-65  SL-7% TA+6% TSL-3%  M40% ML-30%(monthly)
  v15.5  : EMA(3)/EMA(200) ADX(14)>=35 RSI(14) 35-65  SL-7% TA+6% TSL-5%  M35% ML-25%(monthly) DD-30%

Engines (ADX/RSI calculation method variants):
  Wilder      : ewm(alpha=1/N) for both ADX and RSI
  EWM_alpha   : ewm(alpha=1/N) for ADX, ewm(span=N) for RSI
  EWM_span    : ewm(span=N) for both ADX and RSI
  SMA_Wilder  : SMA warmup then Wilder recursive for both
  No_ADX      : ADX filter disabled (always passes)
  No_RSI      : RSI filter disabled (always passes)
"""

import pandas as pd
import numpy as np
import time
import warnings
from multiprocessing import Pool, cpu_count
from itertools import product
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
CSV_PATH = r"D:\filesystem\futures\btc_V1\test\btc_usdt_5m_merged.csv"
OUTPUT_DIR = Path(r"D:\filesystem\futures\btc_V1\test3")
EXCEL_PATH = OUTPUT_DIR / "기획서 검증 V3.xlsx"

# ============================================================
# STRATEGY DEFINITIONS
# ============================================================
STRATEGIES = {
    'v23.4': dict(
        fast_ma_type='EMA', fast_ma_period=3,
        slow_ma_type='EMA', slow_ma_period=200,
        adx_period=14, adx_min=35.0, adx_rise_bars=0,
        rsi_period=14, rsi_min=30.0, rsi_max=65.0,
        ema_gap_min=0.0, monitor_window=0,
        sl_pct=7.0, ta_pct=6.0, tsl_pct=3.0,
        margin_pct=0.30, leverage=10,
        skip_same_dir=True,
        protect_type='daily', protect_limit=-0.20,
        dd_threshold=0.0, margin_reduced=0.0,
        warmup=200, initial_capital=5000.0,
    ),
    'v32.2': dict(
        fast_ma_type='EMA', fast_ma_period=100,
        slow_ma_type='EMA', slow_ma_period=600,
        adx_period=20, adx_min=30.0, adx_rise_bars=6,
        rsi_period=10, rsi_min=40.0, rsi_max=80.0,
        ema_gap_min=0.2, monitor_window=24,
        sl_pct=3.0, ta_pct=12.0, tsl_pct=9.0,
        margin_pct=0.35, leverage=10,
        skip_same_dir=True,
        protect_type='daily', protect_limit=-0.20,
        dd_threshold=0.0, margin_reduced=0.0,
        warmup=600, initial_capital=5000.0,
    ),
    'v32.3': dict(
        fast_ma_type='EMA', fast_ma_period=75,
        slow_ma_type='SMA', slow_ma_period=750,
        adx_period=20, adx_min=30.0, adx_rise_bars=6,
        rsi_period=11, rsi_min=40.0, rsi_max=80.0,
        ema_gap_min=0.2, monitor_window=24,
        sl_pct=3.0, ta_pct=12.0, tsl_pct=9.0,
        margin_pct=0.35, leverage=10,
        skip_same_dir=True,
        protect_type='daily', protect_limit=-0.20,
        dd_threshold=0.0, margin_reduced=0.0,
        warmup=600, initial_capital=5000.0,
    ),
    'v15.4': dict(
        fast_ma_type='EMA', fast_ma_period=3,
        slow_ma_type='EMA', slow_ma_period=200,
        adx_period=14, adx_min=35.0, adx_rise_bars=0,
        rsi_period=14, rsi_min=30.0, rsi_max=65.0,
        ema_gap_min=0.0, monitor_window=0,
        sl_pct=7.0, ta_pct=6.0, tsl_pct=3.0,
        margin_pct=0.40, leverage=10,
        skip_same_dir=True,
        protect_type='monthly', protect_limit=-0.30,
        dd_threshold=0.0, margin_reduced=0.0,
        warmup=200, initial_capital=5000.0,
    ),
    'v15.5': dict(
        fast_ma_type='EMA', fast_ma_period=3,
        slow_ma_type='EMA', slow_ma_period=200,
        adx_period=14, adx_min=35.0, adx_rise_bars=0,
        rsi_period=14, rsi_min=35.0, rsi_max=65.0,
        ema_gap_min=0.0, monitor_window=0,
        sl_pct=7.0, ta_pct=6.0, tsl_pct=5.0,
        margin_pct=0.35, leverage=10,
        skip_same_dir=True,
        protect_type='monthly', protect_limit=-0.25,
        dd_threshold=-0.30, margin_reduced=0.175,
        warmup=200, initial_capital=5000.0,
    ),
}

ENGINES = ['Wilder', 'EWM_alpha', 'EWM_span', 'SMA_Wilder', 'No_ADX', 'No_RSI']
FEE_RATE = 0.0004


# ============================================================
# INDICATOR CALCULATION
# ============================================================
def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_sma(series, period):
    return series.rolling(window=period, min_periods=period).mean()


def calc_ma(series, ma_type, period):
    if ma_type == 'EMA':
        return calc_ema(series, period)
    elif ma_type == 'SMA':
        return calc_sma(series, period)
    else:
        raise ValueError(f"Unknown MA type: {ma_type}")


def calc_adx_wilder(high, low, close, period):
    """ADX using ewm(alpha=1/period) -- Wilder smoothing"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr.replace(0, 1e-10)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr.replace(0, 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return adx


def calc_adx_ewm_span(high, low, close, period):
    """ADX using ewm(span=period) -- standard EMA"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr.replace(0, 1e-10)
    minus_di = 100 * minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr.replace(0, 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    return adx


def calc_adx_sma_wilder(high, low, close, period):
    """ADX using SMA warmup then Wilder recursive"""
    n = len(high)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr_arr = np.zeros(n)

    h = high.values
    l = low.values
    c = close.values

    for i in range(1, n):
        pdm = h[i] - h[i-1]
        mdm = l[i-1] - l[i]
        if pdm > mdm and pdm > 0:
            plus_dm[i] = pdm
        if mdm > pdm and mdm > 0:
            minus_dm[i] = mdm
        tr_arr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))

    # SMA warmup for first 'period' bars, then Wilder smoothing
    atr = np.full(n, np.nan)
    sm_plus = np.full(n, np.nan)
    sm_minus = np.full(n, np.nan)

    if n > period:
        atr[period] = np.mean(tr_arr[1:period+1])
        sm_plus[period] = np.mean(plus_dm[1:period+1])
        sm_minus[period] = np.mean(minus_dm[1:period+1])
        for i in range(period+1, n):
            atr[i] = (atr[i-1] * (period - 1) + tr_arr[i]) / period
            sm_plus[i] = (sm_plus[i-1] * (period - 1) + plus_dm[i]) / period
            sm_minus[i] = (sm_minus[i-1] * (period - 1) + minus_dm[i]) / period

    atr_safe = np.where(atr == 0, 1e-10, atr)
    plus_di = 100 * sm_plus / atr_safe
    minus_di = 100 * sm_minus / atr_safe
    denom = np.where((plus_di + minus_di) == 0, 1e-10, plus_di + minus_di)
    dx = 100 * np.abs(plus_di - minus_di) / denom

    adx = np.full(n, np.nan)
    start = 2 * period
    if n > start:
        adx[start] = np.nanmean(dx[period:start+1])
        for i in range(start+1, n):
            if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    return pd.Series(adx, index=high.index)


def calc_rsi_wilder(close, period):
    """RSI using ewm(alpha=1/period) -- Wilder smoothing"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)


def calc_rsi_ewm_span(close, period):
    """RSI using ewm(span=period) -- standard EMA"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)


def calc_rsi_sma_wilder(close, period):
    """RSI using SMA warmup then Wilder recursive"""
    delta = close.diff().values
    n = len(delta)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_g = np.full(n, np.nan)
    avg_l = np.full(n, np.nan)
    if n > period:
        avg_g[period] = np.mean(gain[1:period+1])
        avg_l[period] = np.mean(loss[1:period+1])
        for i in range(period+1, n):
            avg_g[i] = (avg_g[i-1] * (period - 1) + gain[i]) / period
            avg_l[i] = (avg_l[i-1] * (period - 1) + loss[i]) / period

    avg_l_safe = np.where(avg_l == 0, 1e-10, avg_l)
    rs = avg_g / avg_l_safe
    rsi = 100 - 100 / (1 + rs)
    return pd.Series(rsi, index=close.index)


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(df30, params, engine):
    """
    Run backtest on 30m OHLCV data with given strategy params and engine type.
    Returns dict with all metrics + trade list + monthly/yearly data.
    """
    p = params
    cap = p['initial_capital']
    fee = FEE_RATE
    warmup = p['warmup']

    close = df30['close'].values
    high = df30['high'].values
    low = df30['low'].values
    opn = df30['open'].values
    timestamps = df30.index

    n = len(close)

    # --- Moving Averages ---
    fast_ma = calc_ma(df30['close'], p['fast_ma_type'], p['fast_ma_period']).values
    slow_ma = calc_ma(df30['close'], p['slow_ma_type'], p['slow_ma_period']).values

    # --- ADX ---
    adx_period = p['adx_period']
    if engine == 'No_ADX':
        adx = np.full(n, 99.0)  # always passes
    elif engine in ('Wilder', 'EWM_alpha'):
        adx = calc_adx_wilder(df30['high'], df30['low'], df30['close'], adx_period).values
    elif engine == 'EWM_span':
        adx = calc_adx_ewm_span(df30['high'], df30['low'], df30['close'], adx_period).values
    elif engine == 'SMA_Wilder':
        adx = calc_adx_sma_wilder(df30['high'], df30['low'], df30['close'], adx_period).values
    elif engine == 'No_RSI':
        adx = calc_adx_wilder(df30['high'], df30['low'], df30['close'], adx_period).values
    else:
        adx = calc_adx_wilder(df30['high'], df30['low'], df30['close'], adx_period).values

    # --- RSI ---
    rsi_period = p['rsi_period']
    if engine == 'No_RSI':
        rsi = np.full(n, 50.0)  # always passes
    elif engine == 'Wilder':
        rsi = calc_rsi_wilder(df30['close'], rsi_period).values
    elif engine == 'EWM_alpha':
        rsi = calc_rsi_wilder(df30['close'], rsi_period).values  # same alpha=1/N
    elif engine == 'EWM_span':
        rsi = calc_rsi_ewm_span(df30['close'], rsi_period).values
    elif engine == 'SMA_Wilder':
        rsi = calc_rsi_sma_wilder(df30['close'], rsi_period).values
    elif engine == 'No_ADX':
        rsi = calc_rsi_wilder(df30['close'], rsi_period).values
    else:
        rsi = calc_rsi_wilder(df30['close'], rsi_period).values

    # Backtest state
    pos = 0  # 0=none, 1=LONG, -1=SHORT
    epx = 0.0  # entry price
    psz = 0.0  # position size (notional)
    slp = 0.0  # stop loss price
    ton = False  # trailing active
    thi = 0.0  # high tracker
    tlo = 999999.0  # low tracker
    watching = 0  # watch direction
    ws = 0  # watch start bar
    le = 0  # last exit bar
    ld = 0  # last exit direction
    pk = cap  # peak capital
    mdd = 0.0

    # Protection state
    ms = cap  # period start capital
    month_start_cap = cap
    current_month = None
    peak_cap_dd = cap  # for DD threshold

    trades = []
    entry_time = None
    gross_profit = 0.0
    gross_loss = 0.0

    sl_pct = p['sl_pct']
    ta_pct = p['ta_pct']
    tsl_pct = p['tsl_pct']
    margin_pct = p['margin_pct']
    lev = p['leverage']
    adx_min = p['adx_min']
    rsi_min = p['rsi_min']
    rsi_max = p['rsi_max']
    adx_rise_bars = p['adx_rise_bars']
    ema_gap_min = p['ema_gap_min']
    monitor_window = p['monitor_window']
    skip_same = p['skip_same_dir']
    protect_type = p['protect_type']
    protect_limit = p['protect_limit']
    dd_threshold = p['dd_threshold']
    margin_reduced = p['margin_reduced']

    # Daily reset tracking (for daily protection)
    daily_bars = 48  # 30min bars per day

    for i in range(warmup, n):
        px = close[i]
        h_ = high[i]
        l_ = low[i]
        ts = timestamps[i]

        # Protection period resets
        if protect_type == 'daily':
            if i > warmup and (i - warmup) % daily_bars == 0:
                ms = cap
        elif protect_type == 'monthly':
            m = ts.month if hasattr(ts, 'month') else pd.Timestamp(ts).month
            y = ts.year if hasattr(ts, 'year') else pd.Timestamp(ts).year
            ym = y * 100 + m
            if current_month is None:
                current_month = ym
                month_start_cap = cap
            elif ym != current_month:
                current_month = ym
                month_start_cap = cap

        # DD threshold tracking
        if dd_threshold < 0:
            if cap > peak_cap_dd:
                peak_cap_dd = cap

        # ========== STEP A: Position management ==========
        if pos != 0:
            watching = 0

            # A1: SL check (only when TSL not active)
            if not ton:
                sl_hit = False
                if pos == 1 and l_ <= slp:
                    sl_hit = True
                    exit_px = slp
                elif pos == -1 and h_ >= slp:
                    sl_hit = True
                    exit_px = slp

                if sl_hit:
                    pnl = (exit_px - epx) / epx * psz * pos - psz * fee
                    cap += pnl
                    if pnl >= 0:
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                    trades.append(dict(
                        entry_time=entry_time, exit_time=ts,
                        direction='LONG' if pos == 1 else 'SHORT',
                        entry_px=epx, exit_px=exit_px,
                        pnl=pnl, exit_type='SL', balance=cap
                    ))
                    ld = pos
                    le = i
                    pos = 0
                    pk = max(pk, cap)
                    dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd:
                        mdd = dd
                    if cap <= 0:
                        break
                    continue

            # A2: TA activation check (high/low based)
            if pos == 1:
                br = (h_ - epx) / epx * 100
            else:
                br = (epx - l_) / epx * 100
            if br >= ta_pct:
                ton = True

            # A3: TSL check
            if ton:
                tsl_triggered = False
                if pos == 1:
                    if h_ > thi:
                        thi = h_
                    ns = thi * (1 - tsl_pct / 100)
                    if ns > slp:
                        slp = ns
                    if px <= slp:
                        tsl_triggered = True
                        exit_px = px
                else:
                    if l_ < tlo:
                        tlo = l_
                    ns = tlo * (1 + tsl_pct / 100)
                    if ns < slp:
                        slp = ns
                    if px >= slp:
                        tsl_triggered = True
                        exit_px = px

                if tsl_triggered:
                    pnl = (exit_px - epx) / epx * psz * pos - psz * fee
                    cap += pnl
                    if pnl >= 0:
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                    trades.append(dict(
                        entry_time=entry_time, exit_time=ts,
                        direction='LONG' if pos == 1 else 'SHORT',
                        entry_px=epx, exit_px=exit_px,
                        pnl=pnl, exit_type='TSL', balance=cap
                    ))
                    ld = pos
                    le = i
                    pos = 0
                    pk = max(pk, cap)
                    dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd:
                        mdd = dd
                    if cap <= 0:
                        break
                    continue

            # A4: REV check
            if i > 0 and not np.isnan(fast_ma[i]) and not np.isnan(fast_ma[i-1]) \
                    and not np.isnan(slow_ma[i]) and not np.isnan(slow_ma[i-1]):
                bull_now = fast_ma[i] > slow_ma[i]
                bull_prev = fast_ma[i-1] > slow_ma[i-1]
                cross_up = bull_now and not bull_prev
                cross_down = not bull_now and bull_prev
                if (pos == 1 and cross_down) or (pos == -1 and cross_up):
                    exit_px = px
                    pnl = (exit_px - epx) / epx * psz * pos - psz * fee
                    cap += pnl
                    if pnl >= 0:
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                    trades.append(dict(
                        entry_time=entry_time, exit_time=ts,
                        direction='LONG' if pos == 1 else 'SHORT',
                        entry_px=epx, exit_px=exit_px,
                        pnl=pnl, exit_type='REV', balance=cap
                    ))
                    ld = pos
                    le = i
                    pos = 0
                    # Don't continue -- allow same bar entry

        # ========== STEP B: Entry check (no position) ==========
        if i < 1:
            continue

        # Cross detection
        if not np.isnan(fast_ma[i]) and not np.isnan(fast_ma[i-1]) \
                and not np.isnan(slow_ma[i]) and not np.isnan(slow_ma[i-1]):
            bull_now = fast_ma[i] > slow_ma[i]
            bull_prev = fast_ma[i-1] > slow_ma[i-1]
            cross_up = bull_now and not bull_prev
            cross_down = not bull_now and bull_prev
        else:
            cross_up = False
            cross_down = False

        if pos == 0:
            # Monitor window logic (for v32.x strategies)
            if monitor_window > 0:
                if cross_up:
                    watching = 1
                    ws = i
                elif cross_down:
                    watching = -1
                    ws = i

                if watching != 0 and i > ws:
                    if i - ws > monitor_window:
                        watching = 0
                        pk = max(pk, cap)
                        dd = (pk - cap) / pk if pk > 0 else 0
                        if dd > mdd:
                            mdd = dd
                        continue

                    if watching == 1 and cross_down:
                        watching = -1
                        ws = i
                        pk = max(pk, cap)
                        dd = (pk - cap) / pk if pk > 0 else 0
                        if dd > mdd:
                            mdd = dd
                        continue
                    elif watching == -1 and cross_up:
                        watching = 1
                        ws = i
                        pk = max(pk, cap)
                        dd = (pk - cap) / pk if pk > 0 else 0
                        if dd > mdd:
                            mdd = dd
                        continue
            else:
                # Immediate entry (no monitor window)
                if cross_up:
                    watching = 1
                    ws = i
                elif cross_down:
                    watching = -1
                    ws = i

            if watching == 0:
                pk = max(pk, cap)
                dd = (pk - cap) / pk if pk > 0 else 0
                if dd > mdd:
                    mdd = dd
                continue

            # Skip same direction
            if skip_same and watching == ld:
                pk = max(pk, cap)
                dd = (pk - cap) / pk if pk > 0 else 0
                if dd > mdd:
                    mdd = dd
                continue

            # ADX filter
            av = adx[i]
            if np.isnan(av) or av < adx_min:
                pk = max(pk, cap)
                dd = (pk - cap) / pk if pk > 0 else 0
                if dd > mdd:
                    mdd = dd
                continue

            # ADX rise check
            if adx_rise_bars > 0 and i >= adx_rise_bars:
                if not np.isnan(adx[i - adx_rise_bars]):
                    if av <= adx[i - adx_rise_bars]:
                        pk = max(pk, cap)
                        dd = (pk - cap) / pk if pk > 0 else 0
                        if dd > mdd:
                            mdd = dd
                        continue

            # RSI filter
            rv = rsi[i]
            if np.isnan(rv) or rv < rsi_min or rv > rsi_max:
                pk = max(pk, cap)
                dd = (pk - cap) / pk if pk > 0 else 0
                if dd > mdd:
                    mdd = dd
                continue

            # EMA gap filter
            if ema_gap_min > 0 and not np.isnan(slow_ma[i]) and slow_ma[i] != 0:
                gap = abs(fast_ma[i] - slow_ma[i]) / abs(slow_ma[i]) * 100
                if gap < ema_gap_min:
                    pk = max(pk, cap)
                    dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd:
                        mdd = dd
                    continue

            # Protection check
            if protect_type == 'daily':
                if ms > 0 and (cap - ms) / ms <= protect_limit:
                    watching = 0
                    pk = max(pk, cap)
                    dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd:
                        mdd = dd
                    continue
            elif protect_type == 'monthly':
                if month_start_cap > 0 and (cap - month_start_cap) / month_start_cap <= protect_limit:
                    watching = 0
                    pk = max(pk, cap)
                    dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd:
                        mdd = dd
                    continue

            # Balance check
            if cap <= 0:
                break

            # ======== ENTRY ========
            # DD-based margin reduction
            actual_margin = margin_pct
            if dd_threshold < 0 and peak_cap_dd > 0:
                dd_now = (peak_cap_dd - cap) / peak_cap_dd
                if dd_now >= abs(dd_threshold):
                    actual_margin = margin_reduced

            mg = cap * actual_margin
            psz = mg * lev
            cap -= psz * fee  # entry fee
            pos = watching
            epx = px
            entry_time = ts
            ton = False
            thi = px
            tlo = px
            if pos == 1:
                slp = epx * (1 - sl_pct / 100)
            else:
                slp = epx * (1 + sl_pct / 100)
            pk = max(pk, cap)
            watching = 0

        # MDD update
        pk = max(pk, cap)
        dd = (pk - cap) / pk if pk > 0 else 0
        if dd > mdd:
            mdd = dd
        if cap <= 0:
            break

    # Force close open position at end
    if pos != 0 and cap > 0:
        exit_px = close[-1]
        pnl = (exit_px - epx) / epx * psz * pos - psz * fee
        cap += pnl
        if pnl >= 0:
            gross_profit += pnl
        else:
            gross_loss += abs(pnl)
        trades.append(dict(
            entry_time=entry_time, exit_time=timestamps[-1],
            direction='LONG' if pos == 1 else 'SHORT',
            entry_px=epx, exit_px=exit_px,
            pnl=pnl, exit_type='END', balance=cap
        ))

    # Final MDD update
    pk = max(pk, cap)
    dd = (pk - cap) / pk if pk > 0 else 0
    if dd > mdd:
        mdd = dd

    # Compute metrics
    total_trades = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = total_trades - wins
    sl_hits = sum(1 for t in trades if t['exit_type'] == 'SL')
    tsl_hits = sum(1 for t in trades if t['exit_type'] == 'TSL')
    rev_hits = sum(1 for t in trades if t['exit_type'] == 'REV')
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    ret_pct = (cap - p['initial_capital']) / p['initial_capital'] * 100

    # Monthly breakdown
    monthly = {}
    for t in trades:
        et = pd.Timestamp(t['entry_time'])
        ym = f"{et.year}-{et.month:02d}"
        if ym not in monthly:
            monthly[ym] = dict(trades=0, wins=0, losses=0, profit=0.0, loss=0.0)
        monthly[ym]['trades'] += 1
        if t['pnl'] > 0:
            monthly[ym]['wins'] += 1
            monthly[ym]['profit'] += t['pnl']
        else:
            monthly[ym]['losses'] += 1
            monthly[ym]['loss'] += abs(t['pnl'])

    # Yearly breakdown
    yearly = {}
    for t in trades:
        et = pd.Timestamp(t['entry_time'])
        yr = str(et.year)
        if yr not in yearly:
            yearly[yr] = dict(trades=0, wins=0, losses=0, profit=0.0, loss=0.0)
        yearly[yr]['trades'] += 1
        if t['pnl'] > 0:
            yearly[yr]['wins'] += 1
            yearly[yr]['profit'] += t['pnl']
        else:
            yearly[yr]['losses'] += 1
            yearly[yr]['loss'] += abs(t['pnl'])

    return dict(
        final_balance=round(cap, 2),
        return_pct=round(ret_pct, 2),
        pf=round(pf, 4),
        mdd_pct=round(mdd * 100, 2),
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        win_rate=round(win_rate, 2),
        sl_hits=sl_hits,
        tsl_hits=tsl_hits,
        rev_hits=rev_hits,
        gross_profit=round(gross_profit, 2),
        gross_loss=round(gross_loss, 2),
        trades_list=trades,
        monthly=monthly,
        yearly=yearly,
    )


# ============================================================
# WORKER FUNCTION (for multiprocessing)
# ============================================================
def worker(args):
    """Worker for parallel execution: (strategy_name, engine_name, df30_pickle_path)"""
    strat_name, eng_name, df30_path = args
    df30 = pd.read_pickle(df30_path)
    params = STRATEGIES[strat_name]
    result = run_backtest(df30, params, eng_name)
    return (strat_name, eng_name, result)


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("  5 Strategies x 6 Engines Backtest")
    print("  Output: 기획서 검증 V3.xlsx")
    print("=" * 70)

    # Load 5m data
    print("\n[1/5] Loading 5m CSV data...")
    df5 = pd.read_csv(CSV_PATH, parse_dates=['timestamp'], index_col='timestamp')
    print(f"  5m candles: {len(df5):,}")

    # Resample to 30m
    print("[2/5] Resampling to 30m...")
    df30 = df5.resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    print(f"  30m candles: {len(df30):,}")

    # Save df30 as pickle for multiprocessing
    pkl_path = str(OUTPUT_DIR / "_tmp_df30.pkl")
    df30.to_pickle(pkl_path)

    # Build task list: 5 strategies x 6 engines x 2 runs
    tasks_run1 = [(s, e, pkl_path) for s, e in product(STRATEGIES.keys(), ENGINES)]
    tasks_run2 = [(s, e, pkl_path) for s, e in product(STRATEGIES.keys(), ENGINES)]

    # Run with multiprocessing
    ncpu = min(cpu_count(), 8)
    print(f"[3/5] Running {len(tasks_run1)} backtests x2 (verification), {ncpu} workers...")

    with Pool(ncpu) as pool:
        results_run1 = pool.map(worker, tasks_run1)

    print("  Run 1 complete.")

    with Pool(ncpu) as pool:
        results_run2 = pool.map(worker, tasks_run2)

    print("  Run 2 complete.")

    # Verify determinism
    print("[4/5] Verifying determinism (Run1 vs Run2)...")
    mismatches = 0
    for r1, r2 in zip(results_run1, results_run2):
        s1, e1, res1 = r1
        s2, e2, res2 = r2
        if abs(res1['final_balance'] - res2['final_balance']) > 0.01:
            print(f"  MISMATCH: {s1}/{e1} balance {res1['final_balance']} vs {res2['final_balance']}")
            mismatches += 1
    if mismatches == 0:
        print("  ALL MATCH - deterministic results confirmed.")
    else:
        print(f"  WARNING: {mismatches} mismatches found!")

    # Organize results
    print("[5/5] Generating Excel...")

    # Build summary dataframe
    summary_rows = []
    all_monthly = {}
    all_yearly = {}
    all_trades = {}

    for strat_name, eng_name, res in results_run1:
        key = f"{strat_name}_{eng_name}"
        summary_rows.append(dict(
            Strategy=strat_name,
            Engine=eng_name,
            Final_Balance=res['final_balance'],
            Return_pct=res['return_pct'],
            PF=res['pf'],
            MDD_pct=res['mdd_pct'],
            Trades=res['total_trades'],
            Wins=res['wins'],
            Losses=res['losses'],
            Win_Rate=res['win_rate'],
            SL_Hits=res['sl_hits'],
            TSL_Hits=res['tsl_hits'],
            REV_Hits=res['rev_hits'],
            Gross_Profit=res['gross_profit'],
            Gross_Loss=res['gross_loss'],
        ))
        all_monthly[key] = res['monthly']
        all_yearly[key] = res['yearly']
        all_trades[key] = res['trades_list']

    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values('Return_pct', ascending=False).reset_index(drop=True)

    # TOP3 by different criteria
    top3_return = df_summary.nlargest(3, 'Return_pct')
    top3_pf = df_summary[df_summary['PF'] < float('inf')].nlargest(3, 'PF')
    top3_mdd = df_summary.nsmallest(3, 'MDD_pct')
    top3_trades = df_summary.nlargest(3, 'Trades')

    # Build monthly detail sheet
    monthly_rows = []
    for strat_name, eng_name, res in results_run1:
        key = f"{strat_name}_{eng_name}"
        for ym, data in sorted(all_monthly[key].items()):
            pf_m = data['profit'] / data['loss'] if data['loss'] > 0 else float('inf')
            monthly_rows.append(dict(
                Strategy=strat_name,
                Engine=eng_name,
                Month=ym,
                Trades=data['trades'],
                Wins=data['wins'],
                Losses=data['losses'],
                Profit=round(data['profit'], 2),
                Loss=round(data['loss'], 2),
                Net=round(data['profit'] - data['loss'], 2),
                PF=round(pf_m, 2) if pf_m < float('inf') else 'INF',
            ))
    df_monthly = pd.DataFrame(monthly_rows)

    # Build yearly detail sheet
    yearly_rows = []
    for strat_name, eng_name, res in results_run1:
        key = f"{strat_name}_{eng_name}"
        for yr, data in sorted(all_yearly[key].items()):
            pf_y = data['profit'] / data['loss'] if data['loss'] > 0 else float('inf')
            yearly_rows.append(dict(
                Strategy=strat_name,
                Engine=eng_name,
                Year=yr,
                Trades=data['trades'],
                Wins=data['wins'],
                Losses=data['losses'],
                Profit=round(data['profit'], 2),
                Loss=round(data['loss'], 2),
                Net=round(data['profit'] - data['loss'], 2),
                PF=round(pf_y, 2) if pf_y < float('inf') else 'INF',
            ))
    df_yearly = pd.DataFrame(yearly_rows)

    # Build trades detail sheet
    trade_rows = []
    for strat_name, eng_name, res in results_run1:
        key = f"{strat_name}_{eng_name}"
        for t in all_trades[key]:
            trade_rows.append(dict(
                Strategy=strat_name,
                Engine=eng_name,
                Entry_Time=t['entry_time'],
                Exit_Time=t['exit_time'],
                Direction=t['direction'],
                Entry_Px=round(t['entry_px'], 2),
                Exit_Px=round(t['exit_px'], 2),
                PnL=round(t['pnl'], 2),
                Exit_Type=t['exit_type'],
                Balance=round(t['balance'], 2),
            ))
    df_trades = pd.DataFrame(trade_rows)

    # Strategy comparison: per strategy, best engine
    strat_best_rows = []
    for sname in STRATEGIES:
        sub = df_summary[df_summary['Strategy'] == sname].copy()
        if len(sub) > 0:
            best = sub.loc[sub['Return_pct'].idxmax()]
            strat_best_rows.append(dict(
                Strategy=sname,
                Best_Engine=best['Engine'],
                Final_Balance=best['Final_Balance'],
                Return_pct=best['Return_pct'],
                PF=best['PF'],
                MDD_pct=best['MDD_pct'],
                Trades=best['Trades'],
                Wins=best['Wins'],
                Win_Rate=best['Win_Rate'],
                SL_Hits=best['SL_Hits'],
            ))
    df_strat_best = pd.DataFrame(strat_best_rows)

    # Engine comparison: per engine, average across strategies
    eng_avg_rows = []
    for ename in ENGINES:
        sub = df_summary[df_summary['Engine'] == ename].copy()
        if len(sub) > 0:
            sub_finite = sub[sub['PF'] < 1e10]
            avg_pf = sub_finite['PF'].mean() if len(sub_finite) > 0 else float('inf')
            eng_avg_rows.append(dict(
                Engine=ename,
                Avg_Return_pct=round(sub['Return_pct'].mean(), 2),
                Avg_PF=round(avg_pf, 2),
                Avg_MDD_pct=round(sub['MDD_pct'].mean(), 2),
                Avg_Trades=round(sub['Trades'].mean(), 1),
                Avg_Win_Rate=round(sub['Win_Rate'].mean(), 2),
            ))
    df_eng_avg = pd.DataFrame(eng_avg_rows)

    # Write Excel
    with pd.ExcelWriter(str(EXCEL_PATH), engine='openpyxl') as writer:
        # Sheet 1: TOP3
        top3_combined = pd.DataFrame()
        top3_combined = pd.concat([
            pd.DataFrame([{'Category': '--- TOP3 by Return ---'}]),
            top3_return,
            pd.DataFrame([{'Category': '--- TOP3 by PF ---'}]),
            top3_pf,
            pd.DataFrame([{'Category': '--- TOP3 by MDD (lowest) ---'}]),
            top3_mdd,
            pd.DataFrame([{'Category': '--- TOP3 by Trade Count ---'}]),
            top3_trades,
        ], ignore_index=True)
        top3_combined.to_excel(writer, sheet_name='TOP3_Rankings', index=False)

        # Sheet 2: Full Summary (30 rows)
        df_summary.to_excel(writer, sheet_name='Full_Summary', index=False)

        # Sheet 3: Strategy Best Engine
        df_strat_best.to_excel(writer, sheet_name='Strategy_Best', index=False)

        # Sheet 4: Engine Average
        df_eng_avg.to_excel(writer, sheet_name='Engine_Average', index=False)

        # Sheet 5: Yearly breakdown
        df_yearly.to_excel(writer, sheet_name='Yearly_Detail', index=False)

        # Sheet 6: Monthly breakdown
        df_monthly.to_excel(writer, sheet_name='Monthly_Detail', index=False)

        # Sheet 7: All trades
        df_trades.to_excel(writer, sheet_name='All_Trades', index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Excel: {EXCEL_PATH}")
    print(f"  Total combinations: {len(summary_rows)}")
    print(f"  Determinism check: {'PASS' if mismatches == 0 else 'FAIL'}")
    print(f"{'='*70}")

    # Print top results
    print(f"\n  TOP 5 by Return:")
    for _, row in df_summary.head(5).iterrows():
        print(f"    {row['Strategy']:8s} / {row['Engine']:12s}  "
              f"${row['Final_Balance']:>14,.2f}  "
              f"{row['Return_pct']:>10,.1f}%  "
              f"PF={row['PF']:.2f}  MDD={row['MDD_pct']:.1f}%  "
              f"Trades={row['Trades']}")

    # Cleanup
    try:
        import os
        os.remove(pkl_path)
    except Exception:
        pass


if __name__ == '__main__':
    main()
