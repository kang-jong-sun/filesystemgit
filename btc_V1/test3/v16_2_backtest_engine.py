"""
v16.2 BTC/USDT Futures Backtest Engine
- Triple Engine Architecture (Engine A/B/C)
- ATR-based Dynamic SL/TP/Trail
- Multi-Timeframe Confirmation
- Market Regime Detection
- 3-Stage Partial Exit
"""

import pandas as pd
import numpy as np
import warnings
import os
import time
warnings.filterwarnings('ignore')

DATA_DIR = r"D:\filesystem\futures\btc_V1\test3"

# ============================================================
# 1. DATA LOADING & RESAMPLING
# ============================================================

def load_5m_data():
    """Load and concatenate all 5m CSV files"""
    files = [
        os.path.join(DATA_DIR, "btc_usdt_5m_2020_to_now_part1.csv"),
        os.path.join(DATA_DIR, "btc_usdt_5m_2020_to_now_part2.csv"),
        os.path.join(DATA_DIR, "btc_usdt_5m_2020_to_now_part3.csv"),
    ]
    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=['timestamp'])
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data.sort_values('timestamp', inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.set_index('timestamp', inplace=True)
    return data


def resample_ohlcv(df_5m, rule):
    """Resample 5m data to higher timeframe"""
    resampled = df_5m.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trades': 'sum'
    }).dropna()
    return resampled


# ============================================================
# 2. INDICATOR CALCULATIONS
# ============================================================

def calc_wma(series, period):
    """Weighted Moving Average"""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def calc_ema(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calc_hma(series, period):
    """Hull Moving Average"""
    half = int(period / 2)
    sqrt_p = int(np.sqrt(period))
    wma_half = calc_wma(series, max(half, 1))
    wma_full = calc_wma(series, period)
    diff = 2 * wma_half - wma_full
    return calc_wma(diff, max(sqrt_p, 1))


def calc_vwma(series, volume, period):
    """Volume Weighted Moving Average"""
    return (series * volume).rolling(period).sum() / volume.rolling(period).sum()


def calc_rsi(series, period=14):
    """RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_adx(high, low, close, period=14):
    """ADX with DI+/DI-"""
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
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()


def calc_bb(series, period=20, std_dev=2.0):
    """Bollinger Bands"""
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    width = (upper - lower) / mid * 100
    pct_b = (series - lower) / (upper - lower)
    return mid, upper, lower, width, pct_b


def calc_macd(series, fast=12, slow=26, signal=9):
    """MACD"""
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_volume_ratio(volume, period=20):
    """Volume / SMA(Volume)"""
    return volume / volume.rolling(period).mean()


def calc_linreg_r2(series, period=20):
    """Linear Regression R-squared"""
    r2 = series.rolling(period).apply(
        lambda y: np.corrcoef(np.arange(len(y)), y)[0, 1] ** 2 if len(y) == period else np.nan,
        raw=True
    )
    return r2


# ============================================================
# 3. MULTI-TIMEFRAME INDICATOR PRECOMPUTATION
# ============================================================

def compute_indicators(df, fast_ma_type='WMA', fast_len=3, slow_ma_type='EMA', slow_len=200,
                       adx_period=20, rsi_period=14):
    """Compute all indicators on a single timeframe dataframe"""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Fast MA
    if fast_ma_type == 'WMA':
        df['fast_ma'] = calc_wma(close, fast_len)
    elif fast_ma_type == 'HMA':
        df['fast_ma'] = calc_hma(close, fast_len)
    elif fast_ma_type == 'EMA':
        df['fast_ma'] = calc_ema(close, fast_len)
    elif fast_ma_type == 'VWMA':
        df['fast_ma'] = calc_vwma(close, volume, fast_len)
    elif fast_ma_type == 'DEMA':
        e1 = calc_ema(close, fast_len)
        e2 = calc_ema(e1, fast_len)
        df['fast_ma'] = 2 * e1 - e2
    elif fast_ma_type == 'TEMA':
        e1 = calc_ema(close, fast_len)
        e2 = calc_ema(e1, fast_len)
        e3 = calc_ema(e2, fast_len)
        df['fast_ma'] = 3 * e1 - 3 * e2 + e3

    # Slow MA
    if slow_ma_type == 'EMA':
        df['slow_ma'] = calc_ema(close, slow_len)
    elif slow_ma_type == 'SMA':
        df['slow_ma'] = close.rolling(slow_len).mean()
    elif slow_ma_type == 'WMA':
        df['slow_ma'] = calc_wma(close, slow_len)
    elif slow_ma_type == 'HMA':
        df['slow_ma'] = calc_hma(close, slow_len)

    # ADX
    df['adx'], df['plus_di'], df['minus_di'], df['atr'] = calc_adx(high, low, close, adx_period)

    # RSI
    df['rsi'] = calc_rsi(close, rsi_period)

    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = calc_macd(close, 12, 26, 9)

    # Bollinger Bands
    df['bb_mid'], df['bb_upper'], df['bb_lower'], df['bb_width'], df['bb_pctb'] = calc_bb(close, 20, 2.0)

    # Volume Ratio
    df['vol_ratio'] = calc_volume_ratio(volume, 20)

    # ATR for SL/TP
    df['atr14'] = calc_atr(high, low, close, 14)

    # Cross detection
    df['fast_above'] = df['fast_ma'] > df['slow_ma']
    df['cross_up'] = df['fast_above'] & ~df['fast_above'].shift(1).fillna(False)
    df['cross_down'] = ~df['fast_above'] & df['fast_above'].shift(1).fillna(True)

    return df


# ============================================================
# 4. MARKET REGIME DETECTION
# ============================================================

def detect_regime(adx, atr, atr_sma50):
    """Classify market regime"""
    vol_ratio = atr / atr_sma50 if atr_sma50 > 0 else 1.0
    if vol_ratio > 1.3:
        vol_cat = 'HIGH'
    elif vol_ratio < 0.7:
        vol_cat = 'LOW'
    else:
        vol_cat = 'MID'

    if adx > 35:
        adx_cat = 'STRONG'
    elif adx > 20:
        adx_cat = 'WEAK'
    else:
        adx_cat = 'NONE'

    regime_map = {
        ('STRONG', 'HIGH'): ('STRONG_TREND', 1.2),
        ('STRONG', 'MID'):  ('STABLE_TREND', 1.1),
        ('STRONG', 'LOW'):  ('LOW_VOL_TREND', 1.0),
        ('WEAK', 'HIGH'):   ('CHOPPY_TREND', 0.8),
        ('WEAK', 'MID'):    ('WEAK_TREND', 0.7),
        ('WEAK', 'LOW'):    ('TRANSITION', 0.6),
        ('NONE', 'HIGH'):   ('VOLATILE_RANGE', 0.4),
        ('NONE', 'MID'):    ('RANGING', 0.3),
        ('NONE', 'LOW'):    ('SQUEEZE', 0.5),
    }
    regime, multiplier = regime_map.get((adx_cat, vol_cat), ('UNKNOWN', 0.5))
    return regime, multiplier


# ============================================================
# 5. ENTRY SCORING SYSTEM
# ============================================================

def compute_entry_score(row, direction, mtf_aligned=3):
    """
    Compute entry quality score (0~80)
    direction: 'LONG' or 'SHORT'
    """
    score = 0

    # ADX slope positive (+10)
    if row.get('adx_slope', 0) > 0:
        score += 10

    # Volume ratio > 1.0 (+10)
    vr = row.get('vol_ratio', 1.0)
    if not np.isnan(vr) and vr > 1.0:
        score += min(10, int(vr * 5))

    # MTF alignment (+20 max)
    score += min(20, mtf_aligned * 5)

    # BB position (+10)
    bb_pctb = row.get('bb_pctb', 0.5)
    if not np.isnan(bb_pctb):
        if direction == 'LONG' and bb_pctb < 0.7:
            score += 10
        elif direction == 'SHORT' and bb_pctb > 0.3:
            score += 10

    # MACD direction (+10)
    macd_hist = row.get('macd_hist', 0)
    if not np.isnan(macd_hist):
        if direction == 'LONG' and macd_hist > 0:
            score += 10
        elif direction == 'SHORT' and macd_hist < 0:
            score += 10

    # RSI quality (+10)
    rsi = row.get('rsi', 50)
    if not np.isnan(rsi):
        if direction == 'LONG' and 35 <= rsi <= 60:
            score += 10
        elif direction == 'SHORT' and 40 <= rsi <= 65:
            score += 10

    return score


# ============================================================
# 6. TRIPLE ENGINE BACKTEST
# ============================================================

class Position:
    def __init__(self, direction, entry_price, entry_time, size_usdt, leverage,
                 sl_price, tp1_price, tp2_price, engine, entry_score, atr_val):
        self.direction = direction
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.size_usdt = size_usdt
        self.leverage = leverage
        self.sl_price = sl_price
        self.tp1_price = tp1_price
        self.tp2_price = tp2_price
        self.engine = engine
        self.entry_score = entry_score
        self.atr_val = atr_val

        self.highest = entry_price
        self.lowest = entry_price
        self.peak_roi = 0
        self.tp1_hit = False
        self.tp2_hit = False
        self.remaining_pct = 1.0
        self.trail_active = False
        self.trail_sl = None
        self.partial_profits = 0.0


class TripleEngineBacktester:
    def __init__(self, initial_capital=3000, fee_rate=0.0004):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.balance = initial_capital
        self.peak_balance = initial_capital

        self.positions = []
        self.closed_trades = []
        self.balance_history = []
        self.monthly_data = {}

        # Engine configs
        self.engine_configs = {
            'A': {
                'name': 'Sniper', 'adx_min': 42, 'rsi_low': 35, 'rsi_high': 65,
                'min_score': 55, 'base_leverage': 12, 'base_ratio': 0.15,
                'atr_sl_mult': 2.0, 'atr_tp1_mult': 4.0, 'atr_tp2_mult': 7.0,
                'trail_atr_mult': 2.5, 'trail_activation_pct': 0.08,
                'tf': '30min', 'mtf_confirm': '1h', 'mtf_entry': '15min',
            },
            'B': {
                'name': 'Core', 'adx_min': 33, 'rsi_low': 30, 'rsi_high': 68,
                'min_score': 40, 'base_leverage': 8, 'base_ratio': 0.20,
                'atr_sl_mult': 2.5, 'atr_tp1_mult': 4.5, 'atr_tp2_mult': 7.0,
                'trail_atr_mult': 2.5, 'trail_activation_pct': 0.06,
                'tf': '15min', 'mtf_confirm': '30min', 'mtf_entry': '5min',
            },
            'C': {
                'name': 'Swing', 'adx_min': 27, 'rsi_low': 28, 'rsi_high': 72,
                'min_score': 30, 'base_leverage': 5, 'base_ratio': 0.15,
                'atr_sl_mult': 3.0, 'atr_tp1_mult': 3.5, 'atr_tp2_mult': 6.0,
                'trail_atr_mult': 3.0, 'trail_activation_pct': 0.05,
                'tf': '5min', 'mtf_confirm': '15min', 'mtf_entry': '5min',
            },
        }

        # Global risk
        self.max_positions = 2
        self.max_margin_pct = 0.60
        self.daily_loss_limit = -0.05
        self.monthly_loss_limit = -0.15
        self.dd_reduce_threshold = -0.20
        self.dd_stop_threshold = -0.35
        self.consecutive_losses = 0
        self.cooldown_until = None
        self.daily_start_balance = initial_capital
        self.monthly_start_balance = initial_capital
        self.current_month = None

    def get_streak_multiplier(self):
        if self.consecutive_losses >= 5:
            return 0.3
        elif self.consecutive_losses >= 4:
            return 0.5
        elif self.consecutive_losses >= 3:
            return 0.5
        elif self.consecutive_losses >= 2:
            return 0.7
        return 1.0

    def get_dd_multiplier(self):
        dd = (self.balance - self.peak_balance) / self.peak_balance if self.peak_balance > 0 else 0
        if dd <= self.dd_stop_threshold:
            return 0.0
        elif dd <= self.dd_reduce_threshold:
            return 0.5
        return 1.0

    def can_enter(self, timestamp, engine_key):
        if self.cooldown_until and timestamp < self.cooldown_until:
            return False
        if len(self.positions) >= self.max_positions:
            return False
        total_margin = sum(p.size_usdt / p.leverage for p in self.positions)
        if total_margin / self.balance > self.max_margin_pct:
            return False
        dd_mult = self.get_dd_multiplier()
        if dd_mult == 0.0:
            return False
        # Check daily/monthly limits
        month_key = timestamp.strftime('%Y-%m')
        if month_key != self.current_month:
            self.current_month = month_key
            self.monthly_start_balance = self.balance
        monthly_pnl = (self.balance - self.monthly_start_balance) / self.monthly_start_balance
        if monthly_pnl <= self.monthly_loss_limit:
            return False
        return True

    def compute_position_size(self, engine_key, regime_mult, atr_val, close_price):
        cfg = self.engine_configs[engine_key]
        base = self.balance * cfg['base_ratio']
        streak_m = self.get_streak_multiplier()
        dd_m = self.get_dd_multiplier()
        size = base * regime_mult * streak_m * dd_m
        leverage = cfg['base_leverage']
        notional = size * leverage
        return size, leverage, notional

    def open_position(self, direction, entry_price, entry_time, engine_key,
                      regime_mult, atr_val, entry_score):
        cfg = self.engine_configs[engine_key]
        size, leverage, notional = self.compute_position_size(engine_key, regime_mult, atr_val, entry_price)

        if size < 1:
            return None

        # ATR-based SL/TP
        sl_dist = atr_val * cfg['atr_sl_mult']
        tp1_dist = atr_val * cfg['atr_tp1_mult']
        tp2_dist = atr_val * cfg['atr_tp2_mult']

        # Clamp SL between 1% and 8%
        sl_pct = sl_dist / entry_price
        sl_pct = max(0.01, min(0.08, sl_pct))
        sl_dist = entry_price * sl_pct

        if direction == 'LONG':
            sl_price = entry_price - sl_dist
            tp1_price = entry_price + tp1_dist
            tp2_price = entry_price + tp2_dist
        else:
            sl_price = entry_price + sl_dist
            tp1_price = entry_price - tp1_dist
            tp2_price = entry_price - tp2_dist

        # Entry fee
        fee = notional * self.fee_rate
        self.balance -= fee

        pos = Position(direction, entry_price, entry_time, size * leverage, leverage,
                       sl_price, tp1_price, tp2_price, engine_key, entry_score, atr_val)

        self.positions.append(pos)
        return pos

    def check_and_close_positions(self, high, low, close, timestamp):
        """Check SL/TP/Trail for all positions"""
        to_remove = []

        for i, pos in enumerate(self.positions):
            pnl = 0
            exit_reason = None
            exit_price = close

            if pos.direction == 'LONG':
                pos.highest = max(pos.highest, high)
                pos.lowest = min(pos.lowest, low)
                current_roi = (close - pos.entry_price) / pos.entry_price
                pos.peak_roi = max(pos.peak_roi, (pos.highest - pos.entry_price) / pos.entry_price)

                # SL check
                if low <= pos.sl_price:
                    exit_price = pos.sl_price
                    exit_reason = 'SL'
                # TP1 partial (30%)
                elif not pos.tp1_hit and high >= pos.tp1_price:
                    partial_size = pos.size_usdt * 0.30
                    partial_pnl = partial_size * (pos.tp1_price - pos.entry_price) / pos.entry_price
                    fee = partial_size * self.fee_rate
                    pos.partial_profits += partial_pnl - fee
                    pos.remaining_pct -= 0.30
                    pos.tp1_hit = True
                    pos.sl_price = pos.entry_price  # Move SL to breakeven
                # TP2 partial (30%)
                elif pos.tp1_hit and not pos.tp2_hit and high >= pos.tp2_price:
                    partial_size = pos.size_usdt * 0.30
                    partial_pnl = partial_size * (pos.tp2_price - pos.entry_price) / pos.entry_price
                    fee = partial_size * self.fee_rate
                    pos.partial_profits += partial_pnl - fee
                    pos.remaining_pct -= 0.30
                    pos.tp2_hit = True
                    pos.sl_price = pos.tp1_price  # Move SL to TP1

                # Trailing stop
                cfg = self.engine_configs[pos.engine]
                if current_roi >= cfg['trail_activation_pct'] or pos.tp1_hit:
                    pos.trail_active = True
                    trail_dist = pos.atr_val * cfg['trail_atr_mult']
                    new_trail = pos.highest - trail_dist
                    if pos.trail_sl is None or new_trail > pos.trail_sl:
                        pos.trail_sl = new_trail

                if pos.trail_active and pos.trail_sl and low <= pos.trail_sl:
                    exit_price = pos.trail_sl
                    exit_reason = 'TRAIL'

            else:  # SHORT
                pos.highest = max(pos.highest, high)
                pos.lowest = min(pos.lowest, low)
                current_roi = (pos.entry_price - close) / pos.entry_price
                pos.peak_roi = max(pos.peak_roi, (pos.entry_price - pos.lowest) / pos.entry_price)

                if high >= pos.sl_price:
                    exit_price = pos.sl_price
                    exit_reason = 'SL'
                elif not pos.tp1_hit and low <= pos.tp1_price:
                    partial_size = pos.size_usdt * 0.30
                    partial_pnl = partial_size * (pos.entry_price - pos.tp1_price) / pos.entry_price
                    fee = partial_size * self.fee_rate
                    pos.partial_profits += partial_pnl - fee
                    pos.remaining_pct -= 0.30
                    pos.tp1_hit = True
                    pos.sl_price = pos.entry_price
                elif pos.tp1_hit and not pos.tp2_hit and low <= pos.tp2_price:
                    partial_size = pos.size_usdt * 0.30
                    partial_pnl = partial_size * (pos.entry_price - pos.tp2_price) / pos.entry_price
                    fee = partial_size * self.fee_rate
                    pos.partial_profits += partial_pnl - fee
                    pos.remaining_pct -= 0.30
                    pos.tp2_hit = True
                    pos.sl_price = pos.tp1_price

                cfg = self.engine_configs[pos.engine]
                if current_roi >= cfg['trail_activation_pct'] or pos.tp1_hit:
                    pos.trail_active = True
                    trail_dist = pos.atr_val * cfg['trail_atr_mult']
                    new_trail = pos.lowest + trail_dist
                    if pos.trail_sl is None or new_trail < pos.trail_sl:
                        pos.trail_sl = new_trail

                if pos.trail_active and pos.trail_sl and high >= pos.trail_sl:
                    exit_price = pos.trail_sl
                    exit_reason = 'TRAIL'

            if exit_reason:
                remaining_size = pos.size_usdt * pos.remaining_pct
                if pos.direction == 'LONG':
                    remaining_pnl = remaining_size * (exit_price - pos.entry_price) / pos.entry_price
                else:
                    remaining_pnl = remaining_size * (pos.entry_price - exit_price) / pos.entry_price

                exit_fee = remaining_size * self.fee_rate
                total_pnl = remaining_pnl + pos.partial_profits - exit_fee
                self.balance += total_pnl

                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance

                if total_pnl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0

                roi_pct = total_pnl / (pos.size_usdt / pos.leverage) * 100

                trade_record = {
                    'engine': pos.engine,
                    'direction': pos.direction,
                    'entry_time': pos.entry_time,
                    'exit_time': timestamp,
                    'entry_price': pos.entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'size_usdt': pos.size_usdt,
                    'leverage': pos.leverage,
                    'margin': pos.size_usdt / pos.leverage,
                    'pnl': total_pnl,
                    'roi_pct': roi_pct,
                    'peak_roi': pos.peak_roi * 100,
                    'tp1_hit': pos.tp1_hit,
                    'tp2_hit': pos.tp2_hit,
                    'entry_score': pos.entry_score,
                    'balance_after': self.balance,
                    'atr': pos.atr_val,
                    'sl_pct': abs(pos.entry_price - pos.sl_price) / pos.entry_price * 100,
                    'hold_minutes': (timestamp - pos.entry_time).total_seconds() / 60,
                }
                self.closed_trades.append(trade_record)

                month_key = timestamp.strftime('%Y-%m')
                if month_key not in self.monthly_data:
                    self.monthly_data[month_key] = {
                        'trades': 0, 'wins': 0, 'losses': 0,
                        'gross_profit': 0, 'gross_loss': 0,
                        'pnl': 0, 'start_balance': self.balance - total_pnl,
                    }
                md = self.monthly_data[month_key]
                md['trades'] += 1
                md['pnl'] += total_pnl
                if total_pnl > 0:
                    md['wins'] += 1
                    md['gross_profit'] += total_pnl
                else:
                    md['losses'] += 1
                    md['gross_loss'] += abs(total_pnl)
                md['end_balance'] = self.balance

                to_remove.append(i)

        for i in sorted(to_remove, reverse=True):
            self.positions.pop(i)

        return len(to_remove) > 0

    def close_position_for_reverse(self, close, timestamp):
        """Close all positions for reverse signal"""
        for pos in list(self.positions):
            remaining_size = pos.size_usdt * pos.remaining_pct
            if pos.direction == 'LONG':
                remaining_pnl = remaining_size * (close - pos.entry_price) / pos.entry_price
            else:
                remaining_pnl = remaining_size * (pos.entry_price - close) / pos.entry_price

            exit_fee = remaining_size * self.fee_rate
            total_pnl = remaining_pnl + pos.partial_profits - exit_fee
            self.balance += total_pnl
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance

            if total_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

            roi_pct = total_pnl / (pos.size_usdt / pos.leverage) * 100

            trade_record = {
                'engine': pos.engine,
                'direction': pos.direction,
                'entry_time': pos.entry_time,
                'exit_time': timestamp,
                'entry_price': pos.entry_price,
                'exit_price': close,
                'exit_reason': 'REVERSE',
                'size_usdt': pos.size_usdt,
                'leverage': pos.leverage,
                'margin': pos.size_usdt / pos.leverage,
                'pnl': total_pnl,
                'roi_pct': roi_pct,
                'peak_roi': pos.peak_roi * 100,
                'tp1_hit': pos.tp1_hit,
                'tp2_hit': pos.tp2_hit,
                'entry_score': pos.entry_score,
                'balance_after': self.balance,
                'atr': pos.atr_val,
                'sl_pct': abs(pos.entry_price - pos.sl_price) / pos.entry_price * 100,
                'hold_minutes': (timestamp - pos.entry_time).total_seconds() / 60,
            }
            self.closed_trades.append(trade_record)

            month_key = timestamp.strftime('%Y-%m')
            if month_key not in self.monthly_data:
                self.monthly_data[month_key] = {
                    'trades': 0, 'wins': 0, 'losses': 0,
                    'gross_profit': 0, 'gross_loss': 0,
                    'pnl': 0, 'start_balance': self.balance - total_pnl,
                }
            md = self.monthly_data[month_key]
            md['trades'] += 1
            md['pnl'] += total_pnl
            if total_pnl > 0:
                md['wins'] += 1
                md['gross_profit'] += total_pnl
            else:
                md['losses'] += 1
                md['gross_loss'] += abs(total_pnl)
            md['end_balance'] = self.balance

        self.positions.clear()


def run_backtest():
    print("=" * 80)
    print("  v16.2 Triple Engine Backtest - BTC/USDT Futures")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading 5-minute data...")
    t0 = time.time()
    df_5m = load_5m_data()
    print(f"  Loaded {len(df_5m):,} rows ({df_5m.index[0]} ~ {df_5m.index[-1]})")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Resample
    print("\n[2/5] Resampling to multiple timeframes...")
    t0 = time.time()
    tf_data = {
        '5min': df_5m.copy(),
        '15min': resample_ohlcv(df_5m, '15min'),
        '30min': resample_ohlcv(df_5m, '30min'),
        '1h': resample_ohlcv(df_5m, '1h'),
    }
    for k, v in tf_data.items():
        print(f"  {k}: {len(v):,} rows")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Compute indicators
    print("\n[3/5] Computing indicators on all timeframes...")
    t0 = time.time()

    # Engine A: 30m main, WMA(3)/EMA(200)
    tf_data['30min'] = compute_indicators(tf_data['30min'], 'WMA', 3, 'EMA', 200, adx_period=20)
    # Engine B: 15m main, WMA(3)/EMA(150)
    tf_data['15min'] = compute_indicators(tf_data['15min'], 'WMA', 3, 'EMA', 150, adx_period=20)
    # Engine C: 5m main, HMA(5)/EMA(100)
    tf_data['5min'] = compute_indicators(tf_data['5min'], 'HMA', 5, 'EMA', 100, adx_period=14)
    # 1h for confirmation
    tf_data['1h'] = compute_indicators(tf_data['1h'], 'WMA', 3, 'EMA', 200, adx_period=20)

    # ADX slope
    for k in tf_data:
        tf_data[k]['adx_slope'] = tf_data[k]['adx'] - tf_data[k]['adx'].shift(3)

    # ATR SMA50 for regime detection
    for k in tf_data:
        tf_data[k]['atr_sma50'] = tf_data[k]['atr14'].rolling(50).mean()

    print(f"  Time: {time.time()-t0:.1f}s")

    # Run backtest
    print("\n[4/5] Running Triple Engine Backtest...")
    t0 = time.time()

    bt = TripleEngineBacktester(initial_capital=3000, fee_rate=0.0004)

    # Use 5m as the iteration timeframe for high resolution
    df_iter = tf_data['5min']
    df_30m = tf_data['30min']
    df_15m = tf_data['15min']
    df_1h = tf_data['1h']

    # Pre-align: for each 5m bar, find the corresponding higher TF bar
    last_signal = {'A': None, 'B': None, 'C': None}
    last_cross_time = {'A': None, 'B': None, 'C': None}
    entry_window_active = {'A': False, 'B': False, 'C': False}
    bar_count = 0
    total_bars = len(df_iter)

    for ts, row_5m in df_iter.iterrows():
        bar_count += 1

        # Check existing positions
        bt.check_and_close_positions(row_5m['high'], row_5m['low'], row_5m['close'], ts)

        # Record balance
        if bar_count % 60 == 0:  # every 5 hours
            bt.balance_history.append({'timestamp': ts, 'balance': bt.balance})

        # Get higher TF data for this timestamp
        try:
            row_30m = df_30m.loc[:ts].iloc[-1] if ts >= df_30m.index[0] else None
            row_15m = df_15m.loc[:ts].iloc[-1] if ts >= df_15m.index[0] else None
            row_1h = df_1h.loc[:ts].iloc[-1] if ts >= df_1h.index[0] else None
        except:
            continue

        if row_30m is None or row_15m is None or row_1h is None:
            continue

        # Skip warmup (need 200+ bars for slow MA)
        if bar_count < 2400:  # 200 bars × 12 (1h = 12×5m)
            continue

        # ====== ENGINE A: 30m Sniper ======
        engine_key = 'A'
        cfg = bt.engine_configs[engine_key]

        # Check for cross on 30m
        if hasattr(row_30m, 'cross_up') and row_30m['cross_up']:
            last_signal[engine_key] = 'LONG'
            last_cross_time[engine_key] = ts
            entry_window_active[engine_key] = True
        elif hasattr(row_30m, 'cross_down') and row_30m['cross_down']:
            last_signal[engine_key] = 'SHORT'
            last_cross_time[engine_key] = ts
            entry_window_active[engine_key] = True

        # Entry window check (0~30 min for Engine A)
        if entry_window_active[engine_key] and last_cross_time[engine_key]:
            elapsed = (ts - last_cross_time[engine_key]).total_seconds() / 60
            if elapsed > 30:
                entry_window_active[engine_key] = False

            if entry_window_active[engine_key] and bt.can_enter(ts, engine_key):
                direction = last_signal[engine_key]
                adx_val = row_30m['adx'] if not np.isnan(row_30m['adx']) else 0
                rsi_val = row_30m['rsi'] if not np.isnan(row_30m['rsi']) else 50
                atr_val = row_30m['atr14'] if not np.isnan(row_30m['atr14']) else 0

                if (adx_val >= cfg['adx_min'] and
                    cfg['rsi_low'] <= rsi_val <= cfg['rsi_high'] and
                    atr_val > 0):

                    # MTF: check 1h alignment
                    mtf_count = 0
                    if row_30m['fast_above'] and direction == 'LONG':
                        mtf_count += 1
                    elif not row_30m['fast_above'] and direction == 'SHORT':
                        mtf_count += 1
                    if row_1h['fast_above'] and direction == 'LONG':
                        mtf_count += 1
                    elif not row_1h['fast_above'] and direction == 'SHORT':
                        mtf_count += 1
                    if row_15m['fast_above'] and direction == 'LONG':
                        mtf_count += 1
                    elif not row_15m['fast_above'] and direction == 'SHORT':
                        mtf_count += 1

                    # Need at least 2 TF aligned
                    if mtf_count >= 2:
                        score = compute_entry_score(row_30m, direction, mtf_count)
                        if score >= cfg['min_score']:
                            regime, regime_mult = detect_regime(
                                adx_val, atr_val,
                                row_30m['atr_sma50'] if not np.isnan(row_30m['atr_sma50']) else atr_val
                            )
                            # Close reverse positions
                            for p in bt.positions:
                                if p.direction != direction:
                                    bt.close_position_for_reverse(row_5m['close'], ts)
                                    break

                            if bt.can_enter(ts, engine_key):
                                bt.open_position(direction, row_5m['close'], ts, engine_key,
                                               regime_mult, atr_val, score)
                                entry_window_active[engine_key] = False

        # ====== ENGINE B: 15m Core ======
        engine_key = 'B'
        cfg = bt.engine_configs[engine_key]

        if hasattr(row_15m, 'cross_up') and row_15m['cross_up']:
            last_signal[engine_key] = 'LONG'
            last_cross_time[engine_key] = ts
            entry_window_active[engine_key] = True
        elif hasattr(row_15m, 'cross_down') and row_15m['cross_down']:
            last_signal[engine_key] = 'SHORT'
            last_cross_time[engine_key] = ts
            entry_window_active[engine_key] = True

        if entry_window_active[engine_key] and last_cross_time[engine_key]:
            elapsed = (ts - last_cross_time[engine_key]).total_seconds() / 60
            if elapsed > 45:
                entry_window_active[engine_key] = False

            if entry_window_active[engine_key] and bt.can_enter(ts, engine_key):
                direction = last_signal[engine_key]
                adx_val = row_15m['adx'] if not np.isnan(row_15m['adx']) else 0
                rsi_val = row_15m['rsi'] if not np.isnan(row_15m['rsi']) else 50
                atr_val = row_15m['atr14'] if not np.isnan(row_15m['atr14']) else 0

                if (adx_val >= cfg['adx_min'] and
                    cfg['rsi_low'] <= rsi_val <= cfg['rsi_high'] and
                    atr_val > 0):

                    mtf_count = 0
                    if row_15m['fast_above'] and direction == 'LONG':
                        mtf_count += 1
                    elif not row_15m['fast_above'] and direction == 'SHORT':
                        mtf_count += 1
                    if row_30m['fast_above'] and direction == 'LONG':
                        mtf_count += 1
                    elif not row_30m['fast_above'] and direction == 'SHORT':
                        mtf_count += 1
                    if row_5m.get('fast_above', False) and direction == 'LONG':
                        mtf_count += 1
                    elif not row_5m.get('fast_above', True) and direction == 'SHORT':
                        mtf_count += 1

                    if mtf_count >= 2:
                        score = compute_entry_score(row_15m, direction, mtf_count)
                        if score >= cfg['min_score']:
                            regime, regime_mult = detect_regime(
                                adx_val, atr_val,
                                row_15m['atr_sma50'] if not np.isnan(row_15m['atr_sma50']) else atr_val
                            )
                            for p in bt.positions:
                                if p.direction != direction:
                                    bt.close_position_for_reverse(row_5m['close'], ts)
                                    break
                            if bt.can_enter(ts, engine_key):
                                bt.open_position(direction, row_5m['close'], ts, engine_key,
                                               regime_mult, atr_val, score)
                                entry_window_active[engine_key] = False

        # ====== ENGINE C: 5m Swing ======
        engine_key = 'C'
        cfg = bt.engine_configs[engine_key]

        cross_up_5 = row_5m.get('cross_up', False)
        cross_down_5 = row_5m.get('cross_down', False)

        if cross_up_5:
            last_signal[engine_key] = 'LONG'
            last_cross_time[engine_key] = ts
            entry_window_active[engine_key] = True
        elif cross_down_5:
            last_signal[engine_key] = 'SHORT'
            last_cross_time[engine_key] = ts
            entry_window_active[engine_key] = True

        if entry_window_active[engine_key] and last_cross_time[engine_key]:
            elapsed = (ts - last_cross_time[engine_key]).total_seconds() / 60
            if elapsed > 60:
                entry_window_active[engine_key] = False

            if entry_window_active[engine_key] and bt.can_enter(ts, engine_key):
                direction = last_signal[engine_key]
                adx_val = row_5m.get('adx', 0)
                adx_val = adx_val if not np.isnan(adx_val) else 0
                rsi_val = row_5m.get('rsi', 50)
                rsi_val = rsi_val if not np.isnan(rsi_val) else 50
                atr_val = row_5m.get('atr14', 0)
                atr_val = atr_val if not np.isnan(atr_val) else 0

                if (adx_val >= cfg['adx_min'] and
                    cfg['rsi_low'] <= rsi_val <= cfg['rsi_high'] and
                    atr_val > 0):

                    mtf_count = 0
                    fast_above_5 = row_5m.get('fast_above', None)
                    if fast_above_5 is not None:
                        if fast_above_5 and direction == 'LONG':
                            mtf_count += 1
                        elif not fast_above_5 and direction == 'SHORT':
                            mtf_count += 1
                    if row_15m['fast_above'] and direction == 'LONG':
                        mtf_count += 1
                    elif not row_15m['fast_above'] and direction == 'SHORT':
                        mtf_count += 1

                    if mtf_count >= 1:
                        score = compute_entry_score(row_5m, direction, mtf_count)
                        if score >= cfg['min_score']:
                            regime, regime_mult = detect_regime(
                                adx_val, atr_val,
                                row_5m.get('atr_sma50', atr_val)
                                if not np.isnan(row_5m.get('atr_sma50', atr_val)) else atr_val
                            )
                            for p in bt.positions:
                                if p.direction != direction and p.engine == 'C':
                                    bt.close_position_for_reverse(row_5m['close'], ts)
                                    break
                            if bt.can_enter(ts, engine_key):
                                bt.open_position(direction, row_5m['close'], ts, engine_key,
                                               regime_mult, atr_val, score)
                                entry_window_active[engine_key] = False

        # Progress
        if bar_count % 100000 == 0:
            pct = bar_count / total_bars * 100
            print(f"  Progress: {pct:.1f}% ({bar_count:,}/{total_bars:,}) | Balance: ${bt.balance:,.2f} | Trades: {len(bt.closed_trades)}")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s | Total trades: {len(bt.closed_trades)}")

    # ====== RESULTS ======
    print("\n[5/5] Generating Results...")
    print_results(bt)
    return bt


def print_results(bt):
    trades = bt.closed_trades
    if not trades:
        print("  No trades executed!")
        return

    df_trades = pd.DataFrame(trades)

    print("\n" + "=" * 100)
    print("  BACKTEST RESULTS SUMMARY")
    print("=" * 100)

    # Overall stats
    total_pnl = df_trades['pnl'].sum()
    total_trades = len(df_trades)
    wins = df_trades[df_trades['pnl'] > 0]
    losses = df_trades[df_trades['pnl'] <= 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

    gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = wins['roi_pct'].mean() if len(wins) > 0 else 0
    avg_loss = losses['roi_pct'].mean() if len(losses) > 0 else 0
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    # MDD calculation
    bal_series = [bt.initial_capital]
    for t in trades:
        bal_series.append(t['balance_after'])
    peak = bal_series[0]
    mdd = 0
    for b in bal_series:
        if b > peak:
            peak = b
        dd = (b - peak) / peak
        if dd < mdd:
            mdd = dd

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    for t in trades:
        if t['pnl'] <= 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    sl_count = len(df_trades[df_trades['exit_reason'] == 'SL'])
    trail_count = len(df_trades[df_trades['exit_reason'] == 'TRAIL'])
    rev_count = len(df_trades[df_trades['exit_reason'] == 'REVERSE'])
    tp1_hits = df_trades['tp1_hit'].sum()
    tp2_hits = df_trades['tp2_hit'].sum()

    total_return = (bt.balance - bt.initial_capital) / bt.initial_capital * 100
    months_total = max(1, (df_trades['exit_time'].max() - df_trades['entry_time'].min()).days / 30)
    monthly_avg_return = total_return / months_total

    print(f"""
┌────────────────────────────────────────────────────────────────┐
│                    OVERALL PERFORMANCE                         │
├────────────────────────────────────────────────────────────────┤
│  Initial Capital:     $3,000.00                                │
│  Final Balance:       ${bt.balance:>12,.2f}                    │
│  Total Return:        {total_return:>+10,.1f}%                 │
│  Profit Factor:       {pf:>10.2f}                              │
│  MDD:                 {mdd*100:>10.1f}%                        │
│  Total Trades:        {total_trades:>10,d}                     │
│  Win Rate:            {win_rate:>10.1f}%                       │
│  Avg Win ROI:         {avg_win:>+10.2f}%                       │
│  Avg Loss ROI:        {avg_loss:>+10.2f}%                      │
│  Risk/Reward Ratio:   {rr_ratio:>10.2f}                        │
│  Max Consec Losses:   {max_consec:>10d}                        │
│  Monthly Avg Return:  {monthly_avg_return:>+10.2f}%            │
├────────────────────────────────────────────────────────────────┤
│  Exit Reasons:                                                 │
│    SL:       {sl_count:>6d}  ({sl_count/total_trades*100:>5.1f}%)                          │
│    Trail:    {trail_count:>6d}  ({trail_count/total_trades*100:>5.1f}%)                     │
│    Reverse:  {rev_count:>6d}  ({rev_count/total_trades*100:>5.1f}%)                        │
│  TP1 Hits:   {tp1_hits:>6.0f}  ({tp1_hits/total_trades*100:>5.1f}%)                       │
│  TP2 Hits:   {tp2_hits:>6.0f}  ({tp2_hits/total_trades*100:>5.1f}%)                       │
└────────────────────────────────────────────────────────────────┘
""")

    # Engine breakdown
    print("=" * 100)
    print("  ENGINE BREAKDOWN")
    print("=" * 100)
    print(f"{'Engine':>8} {'Trades':>8} {'Wins':>6} {'WR%':>7} {'GrossP':>12} {'GrossL':>12} {'PF':>8} {'AvgWin%':>9} {'AvgLoss%':>10} {'R:R':>7}")
    print("-" * 100)

    for eng in ['A', 'B', 'C']:
        eng_trades = df_trades[df_trades['engine'] == eng]
        if len(eng_trades) == 0:
            print(f"{'Engine '+eng:>8} {'0':>8}")
            continue
        eng_wins = eng_trades[eng_trades['pnl'] > 0]
        eng_losses = eng_trades[eng_trades['pnl'] <= 0]
        eng_wr = len(eng_wins) / len(eng_trades) * 100
        eng_gp = eng_wins['pnl'].sum() if len(eng_wins) > 0 else 0
        eng_gl = abs(eng_losses['pnl'].sum()) if len(eng_losses) > 0 else 0
        eng_pf = eng_gp / eng_gl if eng_gl > 0 else float('inf')
        eng_avg_w = eng_wins['roi_pct'].mean() if len(eng_wins) > 0 else 0
        eng_avg_l = eng_losses['roi_pct'].mean() if len(eng_losses) > 0 else 0
        eng_rr = abs(eng_avg_w / eng_avg_l) if eng_avg_l != 0 else float('inf')
        eng_name = bt.engine_configs[eng]['name']
        print(f"  {eng}({eng_name:>7}) {len(eng_trades):>6} {len(eng_wins):>6} {eng_wr:>6.1f}% ${eng_gp:>10,.2f} ${eng_gl:>10,.2f} {eng_pf:>7.2f} {eng_avg_w:>+8.2f}% {eng_avg_l:>+9.2f}% {eng_rr:>6.2f}")

    # ====== ENTRY STRUCTURE DETAIL ======
    print("\n" + "=" * 100)
    print("  POSITION ENTRY STRUCTURE (진입 구조 상세)")
    print("=" * 100)

    # Direction breakdown
    longs = df_trades[df_trades['direction'] == 'LONG']
    shorts = df_trades[df_trades['direction'] == 'SHORT']

    print(f"\n  ■ Direction Distribution:")
    print(f"    LONG:  {len(longs):>5} trades ({len(longs)/total_trades*100:.1f}%) | "
          f"WR: {len(longs[longs['pnl']>0])/max(1,len(longs))*100:.1f}% | "
          f"Avg ROI: {longs['roi_pct'].mean():+.2f}% | PnL: ${longs['pnl'].sum():+,.2f}")
    print(f"    SHORT: {len(shorts):>5} trades ({len(shorts)/total_trades*100:.1f}%) | "
          f"WR: {len(shorts[shorts['pnl']>0])/max(1,len(shorts))*100:.1f}% | "
          f"Avg ROI: {shorts['roi_pct'].mean():+.2f}% | PnL: ${shorts['pnl'].sum():+,.2f}")

    # Entry score distribution
    print(f"\n  ■ Entry Score Distribution:")
    for bucket in [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80)]:
        mask = (df_trades['entry_score'] >= bucket[0]) & (df_trades['entry_score'] < bucket[1])
        bucket_trades = df_trades[mask]
        if len(bucket_trades) > 0:
            bw = bucket_trades[bucket_trades['pnl'] > 0]
            bwr = len(bw) / len(bucket_trades) * 100
            bavg = bucket_trades['roi_pct'].mean()
            print(f"    Score {bucket[0]:>2}~{bucket[1]:>2}: {len(bucket_trades):>5} trades | WR: {bwr:.1f}% | Avg ROI: {bavg:+.2f}%")

    # Hold time analysis
    print(f"\n  ■ Hold Time Analysis:")
    df_trades['hold_hours'] = df_trades['hold_minutes'] / 60
    for bucket in [(0, 1), (1, 4), (4, 12), (12, 24), (24, 72), (72, 168), (168, 99999)]:
        mask = (df_trades['hold_hours'] >= bucket[0]) & (df_trades['hold_hours'] < bucket[1])
        ht = df_trades[mask]
        if len(ht) > 0:
            label = f"{bucket[0]}h~{bucket[1]}h" if bucket[1] < 99999 else f"{bucket[0]}h+"
            hw = ht[ht['pnl'] > 0]
            hwr = len(hw) / len(ht) * 100
            print(f"    {label:>10}: {len(ht):>5} trades | WR: {hwr:.1f}% | Avg ROI: {ht['roi_pct'].mean():+.2f}%")

    # SL distance analysis
    print(f"\n  ■ SL Distance (ATR-based):")
    for bucket in [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8)]:
        mask = (df_trades['sl_pct'] >= bucket[0]) & (df_trades['sl_pct'] < bucket[1])
        st = df_trades[mask]
        if len(st) > 0:
            sw = st[st['pnl'] > 0]
            swr = len(sw) / len(st) * 100
            print(f"    SL {bucket[0]}%~{bucket[1]}%: {len(st):>5} trades | WR: {swr:.1f}% | Avg ROI: {st['roi_pct'].mean():+.2f}%")

    # Exit reason x Engine
    print(f"\n  ■ Exit Reason by Engine:")
    for eng in ['A', 'B', 'C']:
        eng_t = df_trades[df_trades['engine'] == eng]
        if len(eng_t) == 0:
            continue
        print(f"    Engine {eng}:")
        for reason in ['SL', 'TRAIL', 'REVERSE']:
            rt = eng_t[eng_t['exit_reason'] == reason]
            if len(rt) > 0:
                print(f"      {reason:>8}: {len(rt):>4} ({len(rt)/len(eng_t)*100:>5.1f}%) | Avg ROI: {rt['roi_pct'].mean():+.2f}%")

    # ====== MONTHLY DETAILED DATA ======
    print("\n" + "=" * 100)
    print("  MONTHLY PERFORMANCE DATA (월별 상세 데이터)")
    print("=" * 100)

    # Build monthly summary from trades
    df_trades['month'] = pd.to_datetime(df_trades['exit_time']).dt.to_period('M')
    monthly_groups = df_trades.groupby('month')

    print(f"\n{'Month':>10} {'Trades':>7} {'Wins':>5} {'Losses':>7} {'WR%':>7} {'GrossP':>12} {'GrossL':>12} {'NetPnL':>12} {'PF':>8} {'Balance':>14} {'MoReturn%':>11}")
    print("-" * 130)

    running_balance = bt.initial_capital
    yearly_data = {}
    loss_months = 0
    total_months = 0

    all_months = sorted(monthly_groups.groups.keys())
    for month in all_months:
        grp = monthly_groups.get_group(month)
        n_trades = len(grp)
        w = grp[grp['pnl'] > 0]
        l = grp[grp['pnl'] <= 0]
        n_wins = len(w)
        n_losses = len(l)
        wr = n_wins / n_trades * 100 if n_trades > 0 else 0
        gp = w['pnl'].sum() if len(w) > 0 else 0
        gl = abs(l['pnl'].sum()) if len(l) > 0 else 0
        net = grp['pnl'].sum()
        m_pf = gp / gl if gl > 0 else float('inf')

        start_bal = running_balance
        running_balance += net
        mo_ret = net / start_bal * 100 if start_bal > 0 else 0

        total_months += 1
        if net < 0:
            loss_months += 1

        year = str(month)[:4]
        if year not in yearly_data:
            yearly_data[year] = {'pnl': 0, 'trades': 0, 'wins': 0, 'losses': 0,
                                 'gross_profit': 0, 'gross_loss': 0, 'start_balance': start_bal}
        yearly_data[year]['pnl'] += net
        yearly_data[year]['trades'] += n_trades
        yearly_data[year]['wins'] += n_wins
        yearly_data[year]['losses'] += n_losses
        yearly_data[year]['gross_profit'] += gp
        yearly_data[year]['gross_loss'] += gl
        yearly_data[year]['end_balance'] = running_balance

        pf_str = f"{m_pf:.2f}" if m_pf < 999 else "INF"
        net_sign = "+" if net >= 0 else ""
        ret_sign = "+" if mo_ret >= 0 else ""
        loss_marker = " ◀ LOSS" if net < 0 else ""

        print(f"  {str(month):>8} {n_trades:>6} {n_wins:>5} {n_losses:>6} {wr:>6.1f}% ${gp:>10,.2f} ${gl:>10,.2f} {net_sign}${abs(net):>10,.2f} {pf_str:>7} ${running_balance:>12,.2f} {ret_sign}{mo_ret:>9.2f}%{loss_marker}")

    # Yearly summary
    print("\n" + "=" * 100)
    print("  YEARLY PERFORMANCE (연도별 요약)")
    print("=" * 100)
    print(f"\n{'Year':>6} {'Trades':>8} {'Wins':>6} {'Losses':>7} {'WR%':>7} {'GrossP':>14} {'GrossL':>14} {'NetPnL':>14} {'PF':>8} {'YearReturn%':>12}")
    print("-" * 110)

    for year in sorted(yearly_data.keys()):
        yd = yearly_data[year]
        wr = yd['wins'] / yd['trades'] * 100 if yd['trades'] > 0 else 0
        ypf = yd['gross_profit'] / yd['gross_loss'] if yd['gross_loss'] > 0 else float('inf')
        yr = yd['pnl'] / yd['start_balance'] * 100 if yd['start_balance'] > 0 else 0
        pf_str = f"{ypf:.2f}" if ypf < 999 else "INF"
        sign = "+" if yd['pnl'] >= 0 else ""
        yr_sign = "+" if yr >= 0 else ""
        print(f"  {year:>5} {yd['trades']:>7} {yd['wins']:>6} {yd['losses']:>6} {wr:>6.1f}% ${yd['gross_profit']:>12,.2f} ${yd['gross_loss']:>12,.2f} {sign}${abs(yd['pnl']):>12,.2f} {pf_str:>7} {yr_sign}{yr:>10.1f}%")

    print(f"\n  ━━━ Summary ━━━")
    print(f"  Loss Months: {loss_months}/{total_months} ({loss_months/max(1,total_months)*100:.1f}%)")
    print(f"  Profitable Years: {sum(1 for y in yearly_data.values() if y['pnl'] > 0)}/{len(yearly_data)}")

    # Top/Bottom trades
    print("\n" + "=" * 100)
    print("  TOP 10 & BOTTOM 10 TRADES")
    print("=" * 100)

    df_sorted = df_trades.sort_values('pnl', ascending=False)
    print(f"\n  ■ Top 10 Winning Trades:")
    print(f"    {'#':>3} {'Engine':>7} {'Dir':>6} {'Entry':>20} {'Exit':>20} {'Reason':>8} {'ROI%':>8} {'PnL':>12} {'Score':>6}")
    for i, (_, row) in enumerate(df_sorted.head(10).iterrows()):
        print(f"    {i+1:>3} {row['engine']:>7} {row['direction']:>6} {str(row['entry_time'])[:19]:>20} {str(row['exit_time'])[:19]:>20} {row['exit_reason']:>8} {row['roi_pct']:>+7.2f}% ${row['pnl']:>10,.2f} {row['entry_score']:>5.0f}")

    print(f"\n  ■ Bottom 10 Losing Trades:")
    for i, (_, row) in enumerate(df_sorted.tail(10).iterrows()):
        print(f"    {i+1:>3} {row['engine']:>7} {row['direction']:>6} {str(row['entry_time'])[:19]:>20} {str(row['exit_time'])[:19]:>20} {row['exit_reason']:>8} {row['roi_pct']:>+7.2f}% ${row['pnl']:>10,.2f} {row['entry_score']:>5.0f}")

    # Engine by year
    print("\n" + "=" * 100)
    print("  ENGINE PERFORMANCE BY YEAR (엔진별 연도 성과)")
    print("=" * 100)

    df_trades['year'] = pd.to_datetime(df_trades['exit_time']).dt.year
    for eng in ['A', 'B', 'C']:
        eng_name = bt.engine_configs[eng]['name']
        eng_t = df_trades[df_trades['engine'] == eng]
        if len(eng_t) == 0:
            continue
        print(f"\n  Engine {eng} ({eng_name}):")
        print(f"    {'Year':>6} {'Trades':>7} {'WR%':>7} {'PnL':>12} {'AvgROI%':>9} {'PF':>8}")
        for yr in sorted(eng_t['year'].unique()):
            yr_t = eng_t[eng_t['year'] == yr]
            yr_w = yr_t[yr_t['pnl'] > 0]
            yr_l = yr_t[yr_t['pnl'] <= 0]
            yr_wr = len(yr_w) / len(yr_t) * 100
            yr_gp = yr_w['pnl'].sum() if len(yr_w) > 0 else 0
            yr_gl = abs(yr_l['pnl'].sum()) if len(yr_l) > 0 else 0
            yr_pf = yr_gp / yr_gl if yr_gl > 0 else float('inf')
            pf_str = f"{yr_pf:.2f}" if yr_pf < 999 else "INF"
            sign = "+" if yr_t['pnl'].sum() >= 0 else ""
            print(f"    {yr:>6} {len(yr_t):>6} {yr_wr:>6.1f}% {sign}${abs(yr_t['pnl'].sum()):>10,.2f} {yr_t['roi_pct'].mean():>+8.2f}% {pf_str:>7}")

    print("\n" + "=" * 100)
    print("  BACKTEST COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    bt = run_backtest()
