"""
v16.2 BTC/USDT Futures Backtest Engine v2
- Pre-computed signals on all TFs
- Fixed DD stop timeout
- Proper cross signal propagation
- Optimized iteration
"""

import pandas as pd
import numpy as np
import warnings
import os
import time
import sys
warnings.filterwarnings('ignore')

DATA_DIR = r"D:\filesystem\futures\btc_V1\test3"

# ============================================================
# INDICATOR CALCULATIONS
# ============================================================

def calc_wma(series, period):
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_hma(series, period):
    half = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    wma_half = calc_wma(series, half)
    wma_full = calc_wma(series, period)
    return calc_wma(2 * wma_half - wma_full, sqrt_p)

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm_s = pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, min_periods=period).mean()
    minus_dm_s = pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * plus_dm_s / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / atr.replace(0, np.nan)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    return adx, plus_di, minus_di, atr

def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def calc_bb(series, period=20, std_dev=2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    width = (upper - lower) / mid * 100
    pct_b = (series - lower) / (upper - lower)
    return width, pct_b


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def load_and_prepare():
    print("[1/4] Loading 5-minute data...")
    t0 = time.time()
    files = [
        os.path.join(DATA_DIR, f"btc_usdt_5m_2020_to_now_part{i}.csv") for i in range(1, 4)
    ]
    dfs = [pd.read_csv(f, parse_dates=['timestamp']) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)
    print(f"  {len(df):,} rows | {df.index[0]} ~ {df.index[-1]} | {time.time()-t0:.1f}s")

    print("\n[2/4] Resampling to MTF...")
    t0 = time.time()
    tf_map = {'5min': df}
    for rule in ['15min', '30min', '1h']:
        tf_map[rule] = df.resample(rule).agg({
            'open':'first','high':'max','low':'min','close':'last',
            'volume':'sum','quote_volume':'sum','trades':'sum'
        }).dropna()

    for k, v in tf_map.items():
        print(f"  {k}: {len(v):,} rows")
    print(f"  {time.time()-t0:.1f}s")

    print("\n[3/4] Computing indicators...")
    t0 = time.time()

    # ---- Engine A: 30m WMA(3)/EMA(200), ADX(20)>=38 ----
    d = tf_map['30min']
    d['fast_ma'] = calc_wma(d['close'], 3)
    d['slow_ma'] = calc_ema(d['close'], 200)
    d['adx'], d['pdi'], d['mdi'], d['atr_adx'] = calc_adx(d['high'], d['low'], d['close'], 20)
    d['rsi'] = calc_rsi(d['close'], 14)
    d['atr14'] = calc_atr(d['high'], d['low'], d['close'], 14)
    d['atr_sma50'] = d['atr14'].rolling(50).mean()
    d['macd_l'], d['macd_s'], d['macd_h'] = calc_macd(d['close'], 12, 26, 9)
    d['bb_w'], d['bb_pctb'] = calc_bb(d['close'], 20, 2.0)
    d['vol_ratio'] = d['volume'] / d['volume'].rolling(20).mean()
    d['adx_slope'] = d['adx'] - d['adx'].shift(3)
    d['fast_above'] = (d['fast_ma'] > d['slow_ma']).astype(int)
    d['cross_up'] = (d['fast_above'].diff() == 1)
    d['cross_down'] = (d['fast_above'].diff() == -1)

    # ---- Engine B: 15m WMA(3)/EMA(150), ADX(20)>=30 ----
    d2 = tf_map['15min']
    d2['fast_ma'] = calc_wma(d2['close'], 3)
    d2['slow_ma'] = calc_ema(d2['close'], 150)
    d2['adx'], d2['pdi'], d2['mdi'], d2['atr_adx'] = calc_adx(d2['high'], d2['low'], d2['close'], 20)
    d2['rsi'] = calc_rsi(d2['close'], 14)
    d2['atr14'] = calc_atr(d2['high'], d2['low'], d2['close'], 14)
    d2['atr_sma50'] = d2['atr14'].rolling(50).mean()
    d2['macd_l'], d2['macd_s'], d2['macd_h'] = calc_macd(d2['close'], 12, 26, 9)
    d2['bb_w'], d2['bb_pctb'] = calc_bb(d2['close'], 20, 2.0)
    d2['vol_ratio'] = d2['volume'] / d2['volume'].rolling(20).mean()
    d2['adx_slope'] = d2['adx'] - d2['adx'].shift(3)
    d2['fast_above'] = (d2['fast_ma'] > d2['slow_ma']).astype(int)
    d2['cross_up'] = (d2['fast_above'].diff() == 1)
    d2['cross_down'] = (d2['fast_above'].diff() == -1)

    # ---- Engine C: 5m HMA(5)/EMA(100), ADX(14)>=25 ----
    d3 = tf_map['5min']
    d3['fast_ma'] = calc_hma(d3['close'], 5)
    d3['slow_ma'] = calc_ema(d3['close'], 100)
    d3['adx'], d3['pdi'], d3['mdi'], d3['atr_adx'] = calc_adx(d3['high'], d3['low'], d3['close'], 14)
    d3['rsi'] = calc_rsi(d3['close'], 14)
    d3['atr14'] = calc_atr(d3['high'], d3['low'], d3['close'], 14)
    d3['atr_sma50'] = d3['atr14'].rolling(50).mean()
    d3['macd_l'], d3['macd_s'], d3['macd_h'] = calc_macd(d3['close'], 8, 21, 5)
    d3['bb_w'], d3['bb_pctb'] = calc_bb(d3['close'], 20, 2.0)
    d3['vol_ratio'] = d3['volume'] / d3['volume'].rolling(20).mean()
    d3['adx_slope'] = d3['adx'] - d3['adx'].shift(3)
    d3['fast_above'] = (d3['fast_ma'] > d3['slow_ma']).astype(int)
    d3['cross_up'] = (d3['fast_above'].diff() == 1)
    d3['cross_down'] = (d3['fast_above'].diff() == -1)

    # 1h for MTF confirmation
    d4 = tf_map['1h']
    d4['fast_ma'] = calc_wma(d4['close'], 3)
    d4['slow_ma'] = calc_ema(d4['close'], 200)
    d4['fast_above'] = (d4['fast_ma'] > d4['slow_ma']).astype(int)
    d4['adx'], _, _, _ = calc_adx(d4['high'], d4['low'], d4['close'], 20)

    print(f"  {time.time()-t0:.1f}s")

    # ---- Create signal arrays aligned to 5m for fast iteration ----
    print("\n[4/4] Aligning signals to 5m timeline...")
    t0 = time.time()

    idx_5m = tf_map['5min'].index

    # For each higher TF, create forward-filled alignment to 5m index
    def align_to_5m(htf_df, cols):
        subset = htf_df[cols].reindex(idx_5m, method='ffill')
        return subset

    # 30m signals
    sig_30m = align_to_5m(tf_map['30min'], [
        'fast_above','cross_up','cross_down','adx','rsi','atr14','atr_sma50',
        'macd_h','bb_pctb','vol_ratio','adx_slope','close'
    ])
    sig_30m.columns = ['s30_' + c for c in sig_30m.columns]

    # The cross signals should NOT be forward-filled (they're one-time events)
    # Instead, create a "cross window" column
    cross_30m_up = tf_map['30min']['cross_up'].reindex(idx_5m).fillna(False)
    cross_30m_down = tf_map['30min']['cross_down'].reindex(idx_5m).fillna(False)
    # Create entry window: True for N bars after cross
    window_30m = 6  # 6 x 5min = 30 min entry window for Engine A
    sig_30m['s30_cross_up_window'] = cross_30m_up.rolling(window_30m, min_periods=1).max().fillna(0).astype(bool)
    sig_30m['s30_cross_down_window'] = cross_30m_down.rolling(window_30m, min_periods=1).max().fillna(0).astype(bool)

    # 15m signals
    sig_15m = align_to_5m(tf_map['15min'], [
        'fast_above','cross_up','cross_down','adx','rsi','atr14','atr_sma50',
        'macd_h','bb_pctb','vol_ratio','adx_slope','close'
    ])
    sig_15m.columns = ['s15_' + c for c in sig_15m.columns]

    cross_15m_up = tf_map['15min']['cross_up'].reindex(idx_5m).fillna(False)
    cross_15m_down = tf_map['15min']['cross_down'].reindex(idx_5m).fillna(False)
    window_15m = 9  # 9 x 5min = 45 min for Engine B
    sig_15m['s15_cross_up_window'] = cross_15m_up.rolling(window_15m, min_periods=1).max().fillna(0).astype(bool)
    sig_15m['s15_cross_down_window'] = cross_15m_down.rolling(window_15m, min_periods=1).max().fillna(0).astype(bool)

    # 5m signals (already aligned)
    sig_5m = tf_map['5min'][[
        'fast_above','cross_up','cross_down','adx','rsi','atr14','atr_sma50',
        'macd_h','bb_pctb','vol_ratio','adx_slope'
    ]].copy()
    sig_5m.columns = ['s5_' + c for c in sig_5m.columns]

    cross_5m_up = tf_map['5min']['cross_up'].fillna(False)
    cross_5m_down = tf_map['5min']['cross_down'].fillna(False)
    window_5m = 12  # 12 x 5min = 60 min for Engine C
    sig_5m['s5_cross_up_window'] = cross_5m_up.rolling(window_5m, min_periods=1).max().fillna(0).astype(bool)
    sig_5m['s5_cross_down_window'] = cross_5m_down.rolling(window_5m, min_periods=1).max().fillna(0).astype(bool)

    # 1h alignment
    sig_1h = align_to_5m(tf_map['1h'], ['fast_above', 'adx'])
    sig_1h.columns = ['s1h_fast_above', 's1h_adx']

    # Merge all into one big aligned dataframe
    master = pd.concat([
        tf_map['5min'][['open','high','low','close','volume']],
        sig_5m, sig_15m, sig_30m, sig_1h
    ], axis=1)

    print(f"  Master shape: {master.shape} | {time.time()-t0:.1f}s")
    return master


# ============================================================
# BACKTEST ENGINE
# ============================================================

class Position:
    __slots__ = ['direction','entry_price','entry_time','size_usdt','leverage',
                 'sl_price','tp1_price','tp2_price','engine','score','atr_val',
                 'highest','lowest','peak_roi','tp1_hit','tp2_hit','remaining',
                 'trail_active','trail_sl','partial_profits','sl_pct']

    def __init__(self, direction, entry_price, entry_time, size_usdt, leverage,
                 sl_price, tp1_price, tp2_price, engine, score, atr_val, sl_pct):
        self.direction = direction
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.size_usdt = size_usdt
        self.leverage = leverage
        self.sl_price = sl_price
        self.tp1_price = tp1_price
        self.tp2_price = tp2_price
        self.engine = engine
        self.score = score
        self.atr_val = atr_val
        self.sl_pct = sl_pct
        self.highest = entry_price
        self.lowest = entry_price
        self.peak_roi = 0.0
        self.tp1_hit = False
        self.tp2_hit = False
        self.remaining = 1.0
        self.trail_active = False
        self.trail_sl = None
        self.partial_profits = 0.0


ENGINE_CFG = {
    'A': {'name':'Sniper','adx_min':38,'rsi_lo':35,'rsi_hi':65,'min_score':50,
           'lev':12,'ratio':0.18,'atr_sl':2.0,'atr_tp1':4.5,'atr_tp2':8.0,
           'trail_atr':2.5,'trail_act':0.08},
    'B': {'name':'Core','adx_min':30,'rsi_lo':28,'rsi_hi':70,'min_score':35,
           'lev':8,'ratio':0.22,'atr_sl':2.5,'atr_tp1':4.0,'atr_tp2':7.0,
           'trail_atr':2.5,'trail_act':0.06},
    'C': {'name':'Swing','adx_min':25,'rsi_lo':25,'rsi_hi':75,'min_score':25,
           'lev':5,'ratio':0.15,'atr_sl':3.0,'atr_tp1':3.5,'atr_tp2':6.0,
           'trail_atr':3.0,'trail_act':0.05},
}


def entry_score(adx_slope, vol_ratio, mtf_count, bb_pctb, macd_h, rsi, direction):
    s = 0
    if not np.isnan(adx_slope) and adx_slope > 0:
        s += 10
    if not np.isnan(vol_ratio) and vol_ratio > 1.0:
        s += min(10, int(vol_ratio * 5))
    s += min(20, mtf_count * 6)
    if not np.isnan(bb_pctb):
        if direction == 1 and bb_pctb < 0.75:
            s += 8
        elif direction == -1 and bb_pctb > 0.25:
            s += 8
    if not np.isnan(macd_h):
        if (direction == 1 and macd_h > 0) or (direction == -1 and macd_h < 0):
            s += 10
    if not np.isnan(rsi):
        if direction == 1 and 30 <= rsi <= 60:
            s += 8
        elif direction == -1 and 40 <= rsi <= 70:
            s += 8
    return s


def run_backtest(master):
    FEE = 0.0004
    INIT = 3000.0
    balance = INIT
    peak_bal = INIT
    positions = []
    trades = []
    consec_loss = 0
    cooldown_end = None
    monthly_start_bal = {}

    # numpy arrays for speed
    highs = master['high'].values
    lows = master['low'].values
    closes = master['close'].values
    timestamps = master.index

    # Engine A signals (30m based)
    s30_fa = master['s30_fast_above'].values
    s30_cu = master['s30_cross_up_window'].values
    s30_cd = master['s30_cross_down_window'].values
    s30_adx = master['s30_adx'].values
    s30_rsi = master['s30_rsi'].values
    s30_atr = master['s30_atr14'].values
    s30_atr50 = master['s30_atr_sma50'].values
    s30_macdh = master['s30_macd_h'].values
    s30_bbp = master['s30_bb_pctb'].values
    s30_vr = master['s30_vol_ratio'].values
    s30_adxs = master['s30_adx_slope'].values

    # Engine B signals (15m based)
    s15_fa = master['s15_fast_above'].values
    s15_cu = master['s15_cross_up_window'].values
    s15_cd = master['s15_cross_down_window'].values
    s15_adx = master['s15_adx'].values
    s15_rsi = master['s15_rsi'].values
    s15_atr = master['s15_atr14'].values
    s15_atr50 = master['s15_atr_sma50'].values
    s15_macdh = master['s15_macd_h'].values
    s15_bbp = master['s15_bb_pctb'].values
    s15_vr = master['s15_vol_ratio'].values
    s15_adxs = master['s15_adx_slope'].values

    # Engine C signals (5m based)
    s5_fa = master['s5_fast_above'].values
    s5_cu = master['s5_cross_up_window'].values
    s5_cd = master['s5_cross_down_window'].values
    s5_adx = master['s5_adx'].values
    s5_rsi = master['s5_rsi'].values
    s5_atr = master['s5_atr14'].values
    s5_atr50 = master['s5_atr_sma50'].values
    s5_macdh = master['s5_macd_h'].values
    s5_bbp = master['s5_bb_pctb'].values
    s5_vr = master['s5_vol_ratio'].values
    s5_adxs = master['s5_adx_slope'].values

    # 1h MTF
    s1h_fa = master['s1h_fast_above'].values

    n = len(master)
    warmup = 2500  # skip first ~8.7 days

    last_entry_engine = {}
    MIN_BARS_BETWEEN = {'A': 36, 'B': 18, 'C': 6}  # Minimum bars between entries per engine

    balance_curve = []

    print(f"\n  Running {n:,} bars...")
    t0 = time.time()

    for i in range(warmup, n):
        ts = timestamps[i]
        h, l, c = highs[i], lows[i], closes[i]

        # ---- Check positions ----
        to_remove = []
        for pi, pos in enumerate(positions):
            exit_reason = None
            exit_price = c

            if pos.direction == 'LONG':
                pos.highest = max(pos.highest, h)
                cur_roi = (c - pos.entry_price) / pos.entry_price
                pos.peak_roi = max(pos.peak_roi, (pos.highest - pos.entry_price) / pos.entry_price)

                # SL
                if l <= pos.sl_price:
                    exit_price = pos.sl_price
                    exit_reason = 'SL'
                else:
                    # TP1
                    if not pos.tp1_hit and h >= pos.tp1_price:
                        ps = pos.size_usdt * 0.30
                        pp = ps * (pos.tp1_price - pos.entry_price) / pos.entry_price - ps * FEE
                        pos.partial_profits += pp
                        pos.remaining -= 0.30
                        pos.tp1_hit = True
                        pos.sl_price = pos.entry_price  # BE
                    # TP2
                    if pos.tp1_hit and not pos.tp2_hit and h >= pos.tp2_price:
                        ps = pos.size_usdt * 0.30
                        pp = ps * (pos.tp2_price - pos.entry_price) / pos.entry_price - ps * FEE
                        pos.partial_profits += pp
                        pos.remaining -= 0.30
                        pos.tp2_hit = True
                        pos.sl_price = pos.tp1_price

                    # Trail
                    cfg = ENGINE_CFG[pos.engine]
                    if cur_roi >= cfg['trail_act'] or pos.tp1_hit:
                        pos.trail_active = True
                        td = pos.atr_val * cfg['trail_atr']
                        new_ts = pos.highest - td
                        if pos.trail_sl is None or new_ts > pos.trail_sl:
                            pos.trail_sl = max(new_ts, pos.sl_price)
                    if pos.trail_active and pos.trail_sl and l <= pos.trail_sl:
                        exit_price = pos.trail_sl
                        exit_reason = 'TRAIL'

            else:  # SHORT
                pos.lowest = min(pos.lowest, l)
                cur_roi = (pos.entry_price - c) / pos.entry_price
                pos.peak_roi = max(pos.peak_roi, (pos.entry_price - pos.lowest) / pos.entry_price)

                if h >= pos.sl_price:
                    exit_price = pos.sl_price
                    exit_reason = 'SL'
                else:
                    if not pos.tp1_hit and l <= pos.tp1_price:
                        ps = pos.size_usdt * 0.30
                        pp = ps * (pos.entry_price - pos.tp1_price) / pos.entry_price - ps * FEE
                        pos.partial_profits += pp
                        pos.remaining -= 0.30
                        pos.tp1_hit = True
                        pos.sl_price = pos.entry_price
                    if pos.tp1_hit and not pos.tp2_hit and l <= pos.tp2_price:
                        ps = pos.size_usdt * 0.30
                        pp = ps * (pos.entry_price - pos.tp2_price) / pos.entry_price - ps * FEE
                        pos.partial_profits += pp
                        pos.remaining -= 0.30
                        pos.tp2_hit = True
                        pos.sl_price = pos.tp1_price

                    cfg = ENGINE_CFG[pos.engine]
                    if cur_roi >= cfg['trail_act'] or pos.tp1_hit:
                        pos.trail_active = True
                        td = pos.atr_val * cfg['trail_atr']
                        new_ts = pos.lowest + td
                        if pos.trail_sl is None or new_ts < pos.trail_sl:
                            pos.trail_sl = min(new_ts, pos.sl_price)
                    if pos.trail_active and pos.trail_sl and h >= pos.trail_sl:
                        exit_price = pos.trail_sl
                        exit_reason = 'TRAIL'

            if exit_reason:
                rs = pos.size_usdt * pos.remaining
                if pos.direction == 'LONG':
                    rpnl = rs * (exit_price - pos.entry_price) / pos.entry_price
                else:
                    rpnl = rs * (pos.entry_price - exit_price) / pos.entry_price
                total_pnl = rpnl + pos.partial_profits - rs * FEE
                balance += total_pnl
                peak_bal = max(peak_bal, balance)
                consec_loss = consec_loss + 1 if total_pnl < 0 else 0

                trades.append({
                    'engine': pos.engine,
                    'direction': pos.direction,
                    'entry_time': pos.entry_time,
                    'exit_time': ts,
                    'entry_price': pos.entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'size_usdt': pos.size_usdt,
                    'leverage': pos.leverage,
                    'margin': pos.size_usdt / pos.leverage,
                    'pnl': total_pnl,
                    'roi_pct': total_pnl / (pos.size_usdt / pos.leverage) * 100,
                    'peak_roi': pos.peak_roi * 100,
                    'tp1_hit': pos.tp1_hit,
                    'tp2_hit': pos.tp2_hit,
                    'score': pos.score,
                    'balance_after': balance,
                    'sl_pct': pos.sl_pct * 100,
                    'hold_min': (ts - pos.entry_time).total_seconds() / 60,
                })
                to_remove.append(pi)

        for pi in sorted(to_remove, reverse=True):
            positions.pop(pi)

        # Balance curve (every 2h = 24 bars)
        if i % 24 == 0:
            balance_curve.append((ts, balance))

        # ---- Cooldown check ----
        if cooldown_end and ts < cooldown_end:
            continue
        cooldown_end = None

        # ---- DD check ----
        dd = (balance - peak_bal) / peak_bal if peak_bal > 0 else 0
        if dd < -0.35:
            cooldown_end = ts + pd.Timedelta(hours=48)
            continue

        dd_mult = 0.5 if dd < -0.20 else 1.0
        streak_mult = max(0.3, 1.0 - consec_loss * 0.15)

        # ---- Monthly limit ----
        mk = ts.strftime('%Y-%m')
        if mk not in monthly_start_bal:
            monthly_start_bal[mk] = balance
        mo_pnl = (balance - monthly_start_bal[mk]) / monthly_start_bal[mk] if monthly_start_bal[mk] > 0 else 0
        if mo_pnl < -0.15:
            continue

        # ---- Max positions ----
        if len(positions) >= 3:
            continue

        # ---- Try each engine ----
        def try_engine(engine_key, cross_up_w, cross_down_w, fast_above,
                       adx_val, rsi_val, atr_val, atr_sma50, macd_h,
                       bb_pctb, vol_ratio, adx_slope,
                       mtf_fas, mtf_labels):
            nonlocal balance, positions, cooldown_end

            cfg = ENGINE_CFG[engine_key]

            # Cross window active?
            if not (cross_up_w or cross_down_w):
                return
            direction = 1 if cross_up_w else -1
            dir_str = 'LONG' if direction == 1 else 'SHORT'

            # Already have same-engine same-direction position?
            for p in positions:
                if p.engine == engine_key and p.direction == dir_str:
                    return

            # Min bars between entries
            if engine_key in last_entry_engine:
                if i - last_entry_engine[engine_key] < MIN_BARS_BETWEEN[engine_key]:
                    return

            # Filter checks
            if np.isnan(adx_val) or adx_val < cfg['adx_min']:
                return
            if np.isnan(rsi_val) or not (cfg['rsi_lo'] <= rsi_val <= cfg['rsi_hi']):
                return
            if np.isnan(atr_val) or atr_val <= 0:
                return

            # MTF alignment
            mtf_count = 0
            for fa in mtf_fas:
                if not np.isnan(fa):
                    if (direction == 1 and fa > 0.5) or (direction == -1 and fa < 0.5):
                        mtf_count += 1
            if mtf_count < 1:
                return

            # Score
            sc = entry_score(adx_slope, vol_ratio, mtf_count, bb_pctb, macd_h, rsi_val, direction)
            if sc < cfg['min_score']:
                return

            # Regime multiplier
            vr = atr_val / atr_sma50 if (not np.isnan(atr_sma50) and atr_sma50 > 0) else 1.0
            if vr > 1.3:
                regime_m = 0.7
            elif vr < 0.7:
                regime_m = 1.3
            else:
                regime_m = 1.0

            # Close opposite direction positions for this engine
            opp = []
            for pi2, p in enumerate(positions):
                if p.direction != dir_str and (p.engine == engine_key or engine_key == 'A'):
                    opp.append(pi2)
            for pi2 in sorted(opp, reverse=True):
                p = positions[pi2]
                rs = p.size_usdt * p.remaining
                if p.direction == 'LONG':
                    rpnl = rs * (c - p.entry_price) / p.entry_price
                else:
                    rpnl = rs * (p.entry_price - c) / p.entry_price
                total_pnl = rpnl + p.partial_profits - rs * FEE
                balance += total_pnl
                trades.append({
                    'engine': p.engine, 'direction': p.direction,
                    'entry_time': p.entry_time, 'exit_time': ts,
                    'entry_price': p.entry_price, 'exit_price': c,
                    'exit_reason': 'REVERSE', 'size_usdt': p.size_usdt,
                    'leverage': p.leverage, 'margin': p.size_usdt / p.leverage,
                    'pnl': total_pnl, 'roi_pct': total_pnl / (p.size_usdt / p.leverage) * 100,
                    'peak_roi': p.peak_roi * 100, 'tp1_hit': p.tp1_hit,
                    'tp2_hit': p.tp2_hit, 'score': p.score,
                    'balance_after': balance, 'sl_pct': p.sl_pct * 100,
                    'hold_min': (ts - p.entry_time).total_seconds() / 60,
                })
                positions.pop(pi2)

            # Position sizing
            size = balance * cfg['ratio'] * regime_m * dd_mult * streak_mult
            if size < 5:
                return
            lev = cfg['lev']
            notional = size * lev

            # SL/TP
            sl_pct = max(0.015, min(0.08, atr_val * cfg['atr_sl'] / c))
            sl_dist = c * sl_pct
            tp1_dist = atr_val * cfg['atr_tp1']
            tp2_dist = atr_val * cfg['atr_tp2']

            if direction == 1:
                sl_p = c - sl_dist
                tp1_p = c + tp1_dist
                tp2_p = c + tp2_dist
            else:
                sl_p = c + sl_dist
                tp1_p = c - tp1_dist
                tp2_p = c - tp2_dist

            # Entry fee
            balance -= notional * FEE

            pos = Position(dir_str, c, ts, notional, lev, sl_p, tp1_p, tp2_p,
                          engine_key, sc, atr_val, sl_pct)
            positions.append(pos)
            last_entry_engine[engine_key] = i

        # Engine A: 30m based
        try_engine('A',
                   s30_cu[i], s30_cd[i], s30_fa[i],
                   s30_adx[i], s30_rsi[i], s30_atr[i], s30_atr50[i],
                   s30_macdh[i], s30_bbp[i], s30_vr[i], s30_adxs[i],
                   [s30_fa[i], s15_fa[i], s1h_fa[i]],
                   ['30m','15m','1h'])

        # Engine B: 15m based
        try_engine('B',
                   s15_cu[i], s15_cd[i], s15_fa[i],
                   s15_adx[i], s15_rsi[i], s15_atr[i], s15_atr50[i],
                   s15_macdh[i], s15_bbp[i], s15_vr[i], s15_adxs[i],
                   [s15_fa[i], s30_fa[i], s5_fa[i]],
                   ['15m','30m','5m'])

        # Engine C: 5m based
        try_engine('C',
                   s5_cu[i], s5_cd[i], s5_fa[i],
                   s5_adx[i], s5_rsi[i], s5_atr[i], s5_atr50[i],
                   s5_macdh[i], s5_bbp[i], s5_vr[i], s5_adxs[i],
                   [s5_fa[i], s15_fa[i]],
                   ['5m','15m'])

        # Progress
        if (i - warmup) % 100000 == 0 and i > warmup:
            pct = (i - warmup) / (n - warmup) * 100
            print(f"    {pct:.0f}% | Bar {i:,}/{n:,} | Bal: ${balance:,.0f} | Trades: {len(trades)} | Pos: {len(positions)}")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s | Trades: {len(trades)}")
    return trades, balance, peak_bal, INIT, balance_curve


# ============================================================
# REPORTING
# ============================================================

def report(trades, final_bal, peak_bal, init_bal, balance_curve):
    if not trades:
        print("NO TRADES!")
        return

    df = pd.DataFrame(trades)
    total = len(df)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    wr = len(wins) / total * 100
    gp = wins['pnl'].sum() if len(wins) else 0
    gl = abs(losses['pnl'].sum()) if len(losses) else 0
    pf = gp / gl if gl > 0 else float('inf')
    avg_w = wins['roi_pct'].mean() if len(wins) else 0
    avg_l = losses['roi_pct'].mean() if len(losses) else 0
    rr = abs(avg_w / avg_l) if avg_l != 0 else float('inf')
    total_ret = (final_bal - init_bal) / init_bal * 100

    # MDD
    bals = [init_bal] + [t['balance_after'] for t in trades]
    pk = bals[0]
    mdd = 0
    for b in bals:
        pk = max(pk, b)
        dd = (b - pk) / pk
        mdd = min(mdd, dd)

    # Max consec losses
    mc = 0; cc = 0
    for t in trades:
        if t['pnl'] <= 0: cc += 1; mc = max(mc, cc)
        else: cc = 0

    sl_n = len(df[df['exit_reason']=='SL'])
    tr_n = len(df[df['exit_reason']=='TRAIL'])
    rv_n = len(df[df['exit_reason']=='REVERSE'])
    tp1_n = df['tp1_hit'].sum()
    tp2_n = df['tp2_hit'].sum()

    months_span = max(1, (df['exit_time'].max() - df['entry_time'].min()).days / 30)

    print("\n" + "=" * 110)
    print("  v16.2 BACKTEST RESULTS")
    print("=" * 110)
    print(f"""
  Initial:       ${init_bal:,.0f}
  Final:         ${final_bal:,.2f}
  Total Return:  {total_ret:+,.1f}%
  Profit Factor: {pf:.2f}
  MDD:           {mdd*100:.1f}%
  Trades:        {total}   (over {months_span:.0f} months, avg {total/months_span:.1f}/mo)
  Win Rate:      {wr:.1f}%
  Avg Win ROI:   {avg_w:+.2f}%    Avg Loss ROI: {avg_l:+.2f}%
  R:R Ratio:     {rr:.2f}
  Max Consec L:  {mc}
  Exit: SL={sl_n}({sl_n/total*100:.0f}%) TRAIL={tr_n}({tr_n/total*100:.0f}%) REV={rv_n}({rv_n/total*100:.0f}%)
  TP1 Hits: {tp1_n:.0f}({tp1_n/total*100:.0f}%)   TP2 Hits: {tp2_n:.0f}({tp2_n/total*100:.0f}%)
""")

    # ========== ENGINE BREAKDOWN ==========
    print("=" * 110)
    print("  ENGINE BREAKDOWN")
    print("=" * 110)
    print(f"  {'Engine':<12} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'GrossP':>12} {'GrossL':>12} {'PF':>7} {'AvgW%':>8} {'AvgL%':>8} {'R:R':>6} {'PnL':>12}")
    print("-" * 110)
    for eng in ['A','B','C']:
        et = df[df['engine']==eng]
        if len(et) == 0:
            print(f"  {eng}({ENGINE_CFG[eng]['name']:<7}) {'0':>7}")
            continue
        ew = et[et['pnl']>0]; el = et[et['pnl']<=0]
        ewr = len(ew)/len(et)*100
        egp = ew['pnl'].sum() if len(ew) else 0
        egl = abs(el['pnl'].sum()) if len(el) else 0
        epf = egp/egl if egl>0 else float('inf')
        eaw = ew['roi_pct'].mean() if len(ew) else 0
        eal = el['roi_pct'].mean() if len(el) else 0
        err = abs(eaw/eal) if eal!=0 else float('inf')
        pf_s = f"{epf:.2f}" if epf < 999 else "INF"
        print(f"  {eng}({ENGINE_CFG[eng]['name']:<7}) {len(et):>6} {len(ew):>6} {ewr:>6.1f}% ${egp:>10,.2f} ${egl:>10,.2f} {pf_s:>6} {eaw:>+7.2f}% {eal:>+7.2f}% {err:>5.2f} ${et['pnl'].sum():>+10,.2f}")

    # ========== ENTRY STRUCTURE ==========
    print("\n" + "=" * 110)
    print("  POSITION ENTRY STRUCTURE (진입 구조 상세)")
    print("=" * 110)

    l = df[df['direction']=='LONG']; s = df[df['direction']=='SHORT']
    print(f"\n  [방향별 분포]")
    print(f"    LONG:  {len(l):>5} ({len(l)/total*100:.1f}%) | WR:{len(l[l['pnl']>0])/max(1,len(l))*100:.1f}% | AvgROI:{l['roi_pct'].mean():+.2f}% | PnL:${l['pnl'].sum():+,.2f}")
    print(f"    SHORT: {len(s):>5} ({len(s)/total*100:.1f}%) | WR:{len(s[s['pnl']>0])/max(1,len(s))*100:.1f}% | AvgROI:{s['roi_pct'].mean():+.2f}% | PnL:${s['pnl'].sum():+,.2f}")

    print(f"\n  [진입 스코어 분포]")
    for lo, hi in [(0,25),(25,35),(35,45),(45,55),(55,65),(65,80)]:
        mask = (df['score']>=lo) & (df['score']<hi)
        bt = df[mask]
        if len(bt) > 0:
            bwr = len(bt[bt['pnl']>0])/len(bt)*100
            print(f"    Score {lo:>2}~{hi:>2}: {len(bt):>5} trades | WR:{bwr:.1f}% | AvgROI:{bt['roi_pct'].mean():+.2f}% | PnL:${bt['pnl'].sum():+,.2f}")

    print(f"\n  [보유시간 분석]")
    df['hold_h'] = df['hold_min'] / 60
    for lo, hi, lbl in [(0,0.5,'0~30m'),(0.5,1,'30m~1h'),(1,4,'1~4h'),(4,12,'4~12h'),(12,24,'12~24h'),(24,72,'1~3d'),(72,168,'3~7d'),(168,9999,'7d+')]:
        mask = (df['hold_h']>=lo) & (df['hold_h']<hi)
        ht = df[mask]
        if len(ht) > 0:
            print(f"    {lbl:>8}: {len(ht):>5} trades | WR:{len(ht[ht['pnl']>0])/len(ht)*100:.1f}% | AvgROI:{ht['roi_pct'].mean():+.2f}% | PnL:${ht['pnl'].sum():+,.2f}")

    print(f"\n  [SL 거리 분포 (ATR 기반)]")
    for lo, hi in [(1,2),(2,3),(3,4),(4,5),(5,6),(6,8)]:
        mask = (df['sl_pct']>=lo) & (df['sl_pct']<hi)
        st = df[mask]
        if len(st) > 0:
            print(f"    SL {lo}%~{hi}%: {len(st):>5} trades | WR:{len(st[st['pnl']>0])/len(st)*100:.1f}% | AvgROI:{st['roi_pct'].mean():+.2f}%")

    print(f"\n  [엔진별 청산사유]")
    for eng in ['A','B','C']:
        et = df[df['engine']==eng]
        if len(et)==0: continue
        print(f"    Engine {eng} ({ENGINE_CFG[eng]['name']}):")
        for r in ['SL','TRAIL','REVERSE']:
            rt = et[et['exit_reason']==r]
            if len(rt): print(f"      {r:>8}: {len(rt):>4} ({len(rt)/len(et)*100:>5.1f}%) AvgROI:{rt['roi_pct'].mean():+.2f}%")

    # ========== MONTHLY DATA ==========
    print("\n" + "=" * 110)
    print("  MONTHLY PERFORMANCE (월별 상세)")
    print("=" * 110)

    df['month'] = pd.to_datetime(df['exit_time']).dt.to_period('M')
    mg = df.groupby('month')

    print(f"\n  {'Month':>8} {'Trd':>5} {'W':>4} {'L':>4} {'WR%':>6} {'GrossP':>11} {'GrossL':>11} {'NetPnL':>11} {'PF':>6} {'Balance':>13} {'MoRet%':>9}")
    print("  " + "-" * 105)

    rb = init_bal
    yearly = {}
    loss_months = 0
    total_months = 0

    for mo in sorted(mg.groups.keys()):
        g = mg.get_group(mo)
        nt = len(g); nw = len(g[g['pnl']>0]); nl = nt-nw
        wr2 = nw/nt*100 if nt else 0
        gp2 = g[g['pnl']>0]['pnl'].sum() if nw else 0
        gl2 = abs(g[g['pnl']<=0]['pnl'].sum()) if nl else 0
        net = g['pnl'].sum()
        mpf = gp2/gl2 if gl2>0 else float('inf')
        sbr = rb
        rb += net
        mret = net/sbr*100 if sbr>0 else 0
        total_months += 1
        if net < 0: loss_months += 1

        yr = str(mo)[:4]
        if yr not in yearly:
            yearly[yr] = {'pnl':0,'trades':0,'wins':0,'losses':0,'gp':0,'gl':0,'sb':sbr}
        yearly[yr]['pnl'] += net; yearly[yr]['trades'] += nt
        yearly[yr]['wins'] += nw; yearly[yr]['losses'] += nl
        yearly[yr]['gp'] += gp2; yearly[yr]['gl'] += gl2
        yearly[yr]['eb'] = rb

        pfs = f"{mpf:.2f}" if mpf < 999 else "INF"
        marker = " << LOSS" if net < 0 else ""
        print(f"  {str(mo):>8} {nt:>5} {nw:>4} {nl:>4} {wr2:>5.1f}% ${gp2:>9,.2f} ${gl2:>9,.2f} ${net:>+9,.2f} {pfs:>5} ${rb:>11,.2f} {mret:>+8.2f}%{marker}")

    # ========== YEARLY DATA ==========
    print("\n" + "=" * 110)
    print("  YEARLY PERFORMANCE (연도별)")
    print("=" * 110)
    print(f"\n  {'Year':>6} {'Trd':>6} {'W':>5} {'L':>5} {'WR%':>6} {'GrossP':>13} {'GrossL':>13} {'NetPnL':>13} {'PF':>7} {'YearRet%':>10}")
    print("  " + "-" * 100)
    for yr in sorted(yearly.keys()):
        y = yearly[yr]
        ywr = y['wins']/y['trades']*100 if y['trades'] else 0
        ypf = y['gp']/y['gl'] if y['gl']>0 else float('inf')
        yret = y['pnl']/y['sb']*100 if y['sb']>0 else 0
        pfs = f"{ypf:.2f}" if ypf < 999 else "INF"
        print(f"  {yr:>6} {y['trades']:>5} {y['wins']:>5} {y['losses']:>5} {ywr:>5.1f}% ${y['gp']:>11,.2f} ${y['gl']:>11,.2f} ${y['pnl']:>+11,.2f} {pfs:>6} {yret:>+9.1f}%")

    prof_yrs = sum(1 for y in yearly.values() if y['pnl'] > 0)
    print(f"\n  Loss Months: {loss_months}/{total_months} ({loss_months/max(1,total_months)*100:.0f}%)")
    print(f"  Profitable Years: {prof_yrs}/{len(yearly)}")

    # ========== ENGINE BY YEAR ==========
    print("\n" + "=" * 110)
    print("  ENGINE x YEAR (엔진별 연도 성과)")
    print("=" * 110)
    df['year'] = pd.to_datetime(df['exit_time']).dt.year
    for eng in ['A','B','C']:
        et = df[df['engine']==eng]
        if len(et)==0:
            print(f"\n  Engine {eng} ({ENGINE_CFG[eng]['name']}): No trades")
            continue
        print(f"\n  Engine {eng} ({ENGINE_CFG[eng]['name']}):")
        print(f"    {'Year':>6} {'Trd':>5} {'WR%':>6} {'PnL':>12} {'AvgROI%':>9} {'PF':>7}")
        for yr in sorted(et['year'].unique()):
            yt = et[et['year']==yr]
            yw = yt[yt['pnl']>0]; yl = yt[yt['pnl']<=0]
            ywr = len(yw)/len(yt)*100
            ygp = yw['pnl'].sum() if len(yw) else 0
            ygl = abs(yl['pnl'].sum()) if len(yl) else 0
            ypf = ygp/ygl if ygl>0 else float('inf')
            pfs = f"{ypf:.2f}" if ypf<999 else "INF"
            print(f"    {yr:>6} {len(yt):>4} {ywr:>5.1f}% ${yt['pnl'].sum():>+10,.2f} {yt['roi_pct'].mean():>+8.2f}% {pfs:>6}")

    # ========== TOP/BOTTOM TRADES ==========
    print("\n" + "=" * 110)
    print("  TOP 10 / BOTTOM 10 TRADES")
    print("=" * 110)
    ds = df.sort_values('pnl', ascending=False)
    print(f"\n  Top 10 Winners:")
    print(f"    {'#':>3} {'Eng':>4} {'Dir':>5} {'Entry Time':>20} {'Exit Time':>20} {'Reason':>7} {'ROI%':>8} {'PnL':>11} {'Sc':>4}")
    for idx, (_, r) in enumerate(ds.head(10).iterrows()):
        print(f"    {idx+1:>3} {r['engine']:>4} {r['direction']:>5} {str(r['entry_time'])[:19]:>20} {str(r['exit_time'])[:19]:>20} {r['exit_reason']:>7} {r['roi_pct']:>+7.1f}% ${r['pnl']:>9,.2f} {r['score']:>3.0f}")

    print(f"\n  Bottom 10 Losers:")
    for idx, (_, r) in enumerate(ds.tail(10).iterrows()):
        print(f"    {idx+1:>3} {r['engine']:>4} {r['direction']:>5} {str(r['entry_time'])[:19]:>20} {str(r['exit_time'])[:19]:>20} {r['exit_reason']:>7} {r['roi_pct']:>+7.1f}% ${r['pnl']:>9,.2f} {r['score']:>3.0f}")

    print("\n" + "=" * 110)
    print("  COMPLETE")
    print("=" * 110)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 110)
    print("  v16.2 Triple Engine Backtest - BTC/USDT Futures")
    print("=" * 110)
    master = load_and_prepare()
    trades, final_bal, peak_bal, init_bal, bcurve = run_backtest(master)
    report(trades, final_bal, peak_bal, init_bal, bcurve)
