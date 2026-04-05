#!/usr/bin/env python3
"""
v25.1 FINAL Backtest - Dual Model Architecture
Based on v22.0 proven approach + v25 optimization findings
BTC/USDT Futures | 15m timeframe | 2020-01 to 2026-03

Proper isolated margin mechanics:
- margin_used = balance * margin_pct
- notional = margin_used * leverage
- qty = notional / entry_price
- PnL = qty * price_diff  (LONG: exit-entry, SHORT: entry-exit)
- Max loss per trade = margin_used (isolated margin liquidation)
- Fee = notional * fee_rate on each side (entry + exit)
"""

import numpy as np
import pandas as pd
import json
import time
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

BASE_DIR = Path(r"D:\filesystem\futures\btc_V1\test3")
DATA_FILES = [
    BASE_DIR / "btc_usdt_5m_2020_to_now_part1.csv",
    BASE_DIR / "btc_usdt_5m_2020_to_now_part2.csv",
    BASE_DIR / "btc_usdt_5m_2020_to_now_part3.csv",
]

INITIAL_CAPITAL = 3000.0
FEE_RATE = 0.0004  # 0.04% taker fee (one-side)

# ============================================================
# INDICATOR FUNCTIONS
# ============================================================

def wilder(arr, p):
    """Wilder's smoothing (NOT ewm). Manual implementation."""
    out = np.full(len(arr), np.nan)
    s = 0
    while s < len(arr) and np.isnan(arr[s]):
        s += 1
    if s + p > len(arr):
        return out
    out[s + p - 1] = np.nanmean(arr[s:s + p])
    for i in range(s + p, len(arr)):
        if not np.isnan(arr[i]) and not np.isnan(out[i - 1]):
            out[i] = (out[i - 1] * (p - 1) + arr[i]) / p
    return out


def calc_adx_wilders(high, low, close, period=14):
    """ADX using Wilder's smoothing method."""
    n = len(high)
    tr = np.full(n, np.nan)
    plus_dm = np.full(n, np.nan)
    minus_dm = np.full(n, np.nan)

    for i in range(1, n):
        h_l = high[i] - low[i]
        h_pc = abs(high[i] - close[i - 1])
        l_pc = abs(low[i] - close[i - 1])
        tr[i] = max(h_l, h_pc, l_pc)

        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0

    atr = wilder(tr, period)
    smooth_plus = wilder(plus_dm, period)
    smooth_minus = wilder(minus_dm, period)

    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)

    for i in range(n):
        if not np.isnan(smooth_plus[i]) and not np.isnan(atr[i]) and atr[i] > 0:
            plus_di[i] = 100.0 * smooth_plus[i] / atr[i]
        if not np.isnan(smooth_minus[i]) and not np.isnan(atr[i]) and atr[i] > 0:
            minus_di[i] = 100.0 * smooth_minus[i] / atr[i]
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            s = plus_di[i] + minus_di[i]
            if s > 0:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / s

    adx = wilder(dx, period)
    return adx, plus_di, minus_di


def calc_rsi(close, period=14):
    """Standard RSI calculation."""
    n = len(close)
    rsi = np.full(n, np.nan)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    if len(gain) < period:
        return rsi

    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period, len(gain)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def calc_wma(values, period):
    """Weighted Moving Average."""
    n = len(values)
    wma = np.full(n, np.nan)
    weights = np.arange(1, period + 1, dtype=float)
    w_sum = weights.sum()
    for i in range(period - 1, n):
        window = values[i - period + 1: i + 1]
        if not np.any(np.isnan(window)):
            wma[i] = np.dot(window, weights) / w_sum
    return wma


def calc_hma(close, period):
    """Hull Moving Average: HMA(n) = WMA(sqrt(n), 2*WMA(n/2) - WMA(n))"""
    half_p = max(int(round(period / 2)), 1)
    sqrt_p = max(int(round(np.sqrt(period))), 1)

    wma_half = calc_wma(close, half_p)
    wma_full = calc_wma(close, period)

    diff = 2.0 * wma_half - wma_full
    hma = calc_wma(diff, sqrt_p)
    return hma


def calc_ema(close, period):
    """Exponential Moving Average."""
    n = len(close)
    ema = np.full(n, np.nan)
    k = 2.0 / (period + 1)

    start = 0
    while start < n and np.isnan(close[start]):
        start += 1
    if start + period > n:
        return ema

    ema[start + period - 1] = np.nanmean(close[start: start + period])
    for i in range(start + period, n):
        if not np.isnan(close[i]) and not np.isnan(ema[i - 1]):
            ema[i] = close[i] * k + ema[i - 1] * (1 - k)
    return ema


def calc_sma(close, period):
    """Simple Moving Average."""
    n = len(close)
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1: i + 1])
    return sma


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load 5m CSV files, concatenate, resample to 15m."""
    print("Loading 5m data...")
    dfs = []
    for f in DATA_FILES:
        df = pd.read_csv(f, parse_dates=['timestamp'])
        dfs.append(df)
        print(f"  {f.name}: {len(df)} rows, {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")

    df = pd.concat(dfs, ignore_index=True)
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df.set_index('timestamp', inplace=True)

    # Filter to 2020-01-01 onward
    df = df[df.index >= '2020-01-01']
    print(f"  5m data after filter: {len(df)} rows, {df.index[0]} ~ {df.index[-1]}")

    # Resample to 15m
    df_15m = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()

    print(f"  15m data: {len(df_15m)} rows, {df_15m.index[0]} ~ {df_15m.index[-1]}")
    return df_15m


# ============================================================
# MODEL CONFIGURATION
# ============================================================

class ModelConfig:
    def __init__(self, name, fast_ma_type, fast_ma_period, slow_ma_type, slow_ma_period,
                 adx_threshold, rsi_low, rsi_high, entry_delay,
                 sl_pct, trail_activate_pct, trail_drop_pct,
                 margin_pct, leverage, capital_pct):
        self.name = name
        self.fast_ma_type = fast_ma_type
        self.fast_ma_period = fast_ma_period
        self.slow_ma_type = slow_ma_type
        self.slow_ma_period = slow_ma_period
        self.adx_threshold = adx_threshold
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.entry_delay = entry_delay  # candles
        self.sl_pct = sl_pct
        self.trail_activate_pct = trail_activate_pct
        self.trail_drop_pct = trail_drop_pct
        self.margin_pct = margin_pct
        self.leverage = leverage
        self.capital_pct = capital_pct


MODEL_A = ModelConfig(
    name="Model_A_PF_Maximizer",
    fast_ma_type="HMA", fast_ma_period=3,
    slow_ma_type="EMA", slow_ma_period=250,
    adx_threshold=45, rsi_low=35, rsi_high=70,
    entry_delay=3,  # 3 candles = 45 min
    sl_pct=0.08, trail_activate_pct=0.03, trail_drop_pct=0.02,
    margin_pct=0.40, leverage=5,
    capital_pct=0.50,
)

MODEL_B = ModelConfig(
    name="Model_B_Compounder",
    fast_ma_type="HMA", fast_ma_period=4,
    slow_ma_type="SMA", slow_ma_period=150,
    adx_threshold=35, rsi_low=35, rsi_high=65,
    entry_delay=2,  # 2 candles = 30 min
    sl_pct=0.07, trail_activate_pct=0.03, trail_drop_pct=0.02,
    margin_pct=0.30, leverage=10,
    capital_pct=0.50,
)


# ============================================================
# INDICATOR PRECOMPUTATION
# ============================================================

def precompute_indicators(df, config):
    """Precompute all indicators for a model config."""
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)

    if config.fast_ma_type == "HMA":
        fast_ma = calc_hma(close, config.fast_ma_period)
    elif config.fast_ma_type == "EMA":
        fast_ma = calc_ema(close, config.fast_ma_period)
    else:
        fast_ma = calc_sma(close, config.fast_ma_period)

    if config.slow_ma_type == "EMA":
        slow_ma = calc_ema(close, config.slow_ma_period)
    elif config.slow_ma_type == "SMA":
        slow_ma = calc_sma(close, config.slow_ma_period)
    else:
        slow_ma = calc_wma(close, config.slow_ma_period)

    adx, plus_di, minus_di = calc_adx_wilders(high, low, close, 14)
    rsi = calc_rsi(close, 14)

    return fast_ma, slow_ma, adx, rsi


# ============================================================
# BACKTEST ENGINE (Single Model) - Proper Isolated Margin
# ============================================================

def close_position(position, exit_price, exit_reason, bar_idx, timestamps, config):
    """
    Close a position and calculate PnL with proper isolated margin.
    Returns: (pnl_usd, trade_record)

    Isolated margin rules:
    - margin_used is locked at entry
    - PnL = qty * (exit_price - entry_price) for LONG
    - PnL = qty * (entry_price - exit_price) for SHORT
    - Total fees = notional * fee_rate * 2 (entry + exit)
    - Max loss = margin_used (liquidation)
    """
    side = position['side']
    entry_price = position['entry_price']
    qty = position['qty']
    margin_used = position['margin_used']

    # Raw PnL
    if side == 'LONG':
        raw_pnl = qty * (exit_price - entry_price)
    else:
        raw_pnl = qty * (entry_price - exit_price)

    # Fees: entry notional * fee_rate + exit notional * fee_rate
    entry_notional = qty * entry_price
    exit_notional = qty * exit_price
    total_fee = (entry_notional + exit_notional) * FEE_RATE

    # Net PnL
    net_pnl = raw_pnl - total_fee

    # Isolated margin: max loss is margin_used
    if net_pnl < -margin_used:
        net_pnl = -margin_used

    # Leveraged ROI for reporting
    roi_pct = (net_pnl / margin_used * 100) if margin_used > 0 else 0

    trade = {
        'model': config.name,
        'entry_time': pd.Timestamp(timestamps[position['entry_idx']]),
        'exit_time': pd.Timestamp(timestamps[bar_idx]),
        'side': side,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'pnl_pct': round(roi_pct, 3),
        'pnl_usd': round(net_pnl, 4),
        'exit_reason': exit_reason,
        'hold_bars': bar_idx - position['entry_idx'],
        'margin_used': round(margin_used, 2),
    }

    return net_pnl, trade


def run_single_model(df, config, initial_balance, run_id=0):
    """
    Run backtest for a single model with proper isolated margin.
    """
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    timestamps = df.index.values
    n = len(close)

    fast_ma, slow_ma, adx, rsi = precompute_indicators(df, config)

    balance = initial_balance
    position = None
    trades = []
    balance_history = []

    # Signal tracking for entry delay
    signal_type = None
    signal_count = 0

    for i in range(1, n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            balance_history.append((timestamps[i], balance))
            continue

        bar_close = close[i]
        bar_high = high[i]
        bar_low = low[i]

        # --- Determine raw crossover signal ---
        raw_signal = None
        if fast_ma[i] > slow_ma[i] and fast_ma[i - 1] <= slow_ma[i - 1]:
            raw_signal = 'LONG'
        elif fast_ma[i] < slow_ma[i] and fast_ma[i - 1] >= slow_ma[i - 1]:
            raw_signal = 'SHORT'

        # --- Entry delay logic ---
        if raw_signal is not None:
            if raw_signal == signal_type:
                signal_count += 1
            else:
                signal_type = raw_signal
                signal_count = 1
        else:
            if signal_type == 'LONG' and fast_ma[i] > slow_ma[i]:
                signal_count += 1
            elif signal_type == 'SHORT' and fast_ma[i] < slow_ma[i]:
                signal_count += 1
            else:
                signal_type = None
                signal_count = 0

        # Confirmed signal after delay + filters
        confirmed_signal = None
        if signal_count >= config.entry_delay and signal_type is not None:
            if adx[i] >= config.adx_threshold and config.rsi_low <= rsi[i] <= config.rsi_high:
                confirmed_signal = signal_type

        # --- Position management (priority: SL > TRAIL > REVERSE) ---
        if position is not None:
            side = position['side']
            entry_price = position['entry_price']

            # 1) SL check on bar high/low (worst-case intra-bar)
            if side == 'LONG':
                sl_price = entry_price * (1 - config.sl_pct)
                if bar_low <= sl_price:
                    pnl, trade = close_position(position, sl_price, 'SL', i, timestamps, config)
                    balance += pnl
                    if balance < 0:
                        balance = 0
                    trade['balance_after'] = round(balance, 2)
                    trades.append(trade)
                    position = None
                    balance_history.append((timestamps[i], balance))
                    continue
            else:
                sl_price = entry_price * (1 + config.sl_pct)
                if bar_high >= sl_price:
                    pnl, trade = close_position(position, sl_price, 'SL', i, timestamps, config)
                    balance += pnl
                    if balance < 0:
                        balance = 0
                    trade['balance_after'] = round(balance, 2)
                    trades.append(trade)
                    position = None
                    balance_history.append((timestamps[i], balance))
                    continue

            # 2) Trail check on bar CLOSE only
            if side == 'LONG':
                current_pnl_pct = (bar_close - entry_price) / entry_price
            else:
                current_pnl_pct = (entry_price - bar_close) / entry_price

            if current_pnl_pct >= config.trail_activate_pct:
                if not position['trail_active']:
                    position['trail_active'] = True
                    position['trail_peak'] = current_pnl_pct
                else:
                    if current_pnl_pct > position['trail_peak']:
                        position['trail_peak'] = current_pnl_pct

            if position['trail_active']:
                drop = position['trail_peak'] - current_pnl_pct
                if drop >= config.trail_drop_pct:
                    pnl, trade = close_position(position, bar_close, 'TRAIL', i, timestamps, config)
                    balance += pnl
                    if balance < 0:
                        balance = 0
                    trade['balance_after'] = round(balance, 2)
                    trades.append(trade)
                    position = None
                    balance_history.append((timestamps[i], balance))
                    continue

            # 3) REVERSE signal: opposite direction -> close + open new
            if confirmed_signal is not None and confirmed_signal != side:
                pnl, trade = close_position(position, bar_close, 'REVERSE', i, timestamps, config)
                balance += pnl
                if balance < 0:
                    balance = 0
                trade['balance_after'] = round(balance, 2)
                trades.append(trade)
                position = None
                # Fall through to open new position

            # 4) Same-direction signal: skip
            if position is not None and confirmed_signal is not None and confirmed_signal == side:
                confirmed_signal = None

        # --- Open new position ---
        if position is None and confirmed_signal is not None and balance > 10:
            margin_used = balance * config.margin_pct
            notional = margin_used * config.leverage
            qty = notional / bar_close

            # Deduct entry fee from available balance
            entry_fee = notional * FEE_RATE
            balance -= entry_fee

            position = {
                'side': confirmed_signal,
                'entry_price': bar_close,
                'qty': qty,
                'margin_used': margin_used,
                'entry_idx': i,
                'trail_active': False,
                'trail_peak': 0,
            }
            # Reset signal counter
            signal_type = None
            signal_count = 0

        balance_history.append((timestamps[i], balance))

    # Close remaining position at last bar
    if position is not None:
        pnl, trade = close_position(position, close[-1], 'EOD', n - 1, timestamps, config)
        balance += pnl
        if balance < 0:
            balance = 0
        trade['balance_after'] = round(balance, 2)
        trades.append(trade)

    return balance, trades, balance_history


# ============================================================
# METRICS CALCULATION
# ============================================================

def calc_metrics(trades, initial_balance):
    """Calculate comprehensive metrics from trades list."""
    if not trades:
        return {
            'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
            'total_return_pct': 0, 'max_drawdown_pct': 0,
            'avg_pnl_pct': 0, 'avg_win_pct': 0, 'avg_loss_pct': 0,
            'final_balance': initial_balance, 'wins': 0, 'losses': 0,
            'gross_profit': 0, 'gross_loss': 0,
        }

    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]

    gross_profit = sum(t['pnl_usd'] for t in wins)
    gross_loss = abs(sum(t['pnl_usd'] for t in losses))

    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    wr = len(wins) / len(trades) * 100 if trades else 0

    # Max drawdown from balance curve
    balances = [initial_balance]
    for t in trades:
        balances.append(t['balance_after'])
    peak = initial_balance
    mdd = 0
    for b in balances:
        if b > peak:
            peak = b
        dd = (peak - b) / peak * 100 if peak > 0 else 0
        if dd > mdd:
            mdd = dd

    final = trades[-1]['balance_after'] if trades else initial_balance
    total_ret = (final / initial_balance - 1) * 100

    avg_pnl = np.mean([t['pnl_pct'] for t in trades])
    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0

    return {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(wr, 2),
        'profit_factor': round(pf, 3),
        'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2),
        'total_return_pct': round(total_ret, 2),
        'max_drawdown_pct': round(mdd, 2),
        'avg_pnl_pct': round(avg_pnl, 3),
        'avg_win_pct': round(avg_win, 3),
        'avg_loss_pct': round(avg_loss, 3),
        'final_balance': round(final, 2),
    }


def calc_monthly_table(trades, initial_balance, model_name=None):
    """Generate monthly breakdown with running balance."""
    filtered = [t for t in trades if model_name is None or t['model'] == model_name]

    # Group by month of exit_time
    monthly = {}
    for t in filtered:
        key = t['exit_time'].strftime('%Y-%m')
        if key not in monthly:
            monthly[key] = []
        monthly[key].append(t)

    running_balance = initial_balance
    start = pd.Timestamp('2020-01-01')
    end = pd.Timestamp('2026-03-31')
    all_periods = pd.period_range(start, end, freq='M')

    rows = []
    for period in all_periods:
        key = str(period)
        month_trades = monthly.get(key, [])
        month_pnl = sum(t['pnl_usd'] for t in month_trades)
        month_trades_count = len(month_trades)
        month_wins = len([t for t in month_trades if t['pnl_usd'] > 0])
        month_wr = (month_wins / month_trades_count * 100) if month_trades_count > 0 else 0

        start_bal = running_balance
        running_balance += month_pnl
        if running_balance < 0:
            running_balance = 0
        ret_pct = (month_pnl / start_bal * 100) if start_bal > 0 else 0

        rows.append({
            'Month': key,
            'Trades': month_trades_count,
            'Wins': month_wins,
            'WR%': round(month_wr, 1),
            'PnL_USD': round(month_pnl, 2),
            'Return%': round(ret_pct, 2),
            'Balance': round(running_balance, 2),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_full_backtest(df, run_id=0):
    """Run both models and return combined results."""
    cap_a = INITIAL_CAPITAL * MODEL_A.capital_pct
    cap_b = INITIAL_CAPITAL * MODEL_B.capital_pct

    bal_a, trades_a, hist_a = run_single_model(df, MODEL_A, cap_a, run_id)
    bal_b, trades_b, hist_b = run_single_model(df, MODEL_B, cap_b, run_id)

    final_balance = bal_a + bal_b
    all_trades = trades_a + trades_b
    all_trades.sort(key=lambda x: x['exit_time'])

    return {
        'final_balance': final_balance,
        'balance_a': bal_a,
        'balance_b': bal_b,
        'trades_a': trades_a,
        'trades_b': trades_b,
        'all_trades': all_trades,
        'hist_a': hist_a,
        'hist_b': hist_b,
    }


def main():
    t_start = time.time()
    print("=" * 80)
    print("  v25.1 FINAL BACKTEST - Dual Model Architecture")
    print("  BTC/USDT Futures | 15m Timeframe | 2020-01 ~ 2026-03")
    print("  Proper Isolated Margin Mechanics")
    print("=" * 80)
    print()

    # Load data
    df = load_data()
    print()

    # ===== PRIMARY RUN =====
    print("Running primary backtest...")
    result = run_full_backtest(df, run_id=0)
    elapsed = time.time() - t_start
    print(f"Primary run complete in {elapsed:.1f}s")
    print()

    all_trades = result['all_trades']
    trades_a = result['trades_a']
    trades_b = result['trades_b']

    cap_a = INITIAL_CAPITAL * MODEL_A.capital_pct
    cap_b = INITIAL_CAPITAL * MODEL_B.capital_pct

    metrics_all = calc_metrics(all_trades, INITIAL_CAPITAL)
    metrics_a = calc_metrics(trades_a, cap_a)
    metrics_b = calc_metrics(trades_b, cap_b)

    # ===== BUILD REPORT =====
    report_lines = []

    def p(s=""):
        print(s)
        report_lines.append(s)

    p("=" * 80)
    p("  v25.1 FINAL BACKTEST RESULTS")
    p("  Dual Model | BTC/USDT Futures | 15m | 2020-01 ~ 2026-03")
    p("=" * 80)
    p()

    # --- Portfolio Summary ---
    p("=" * 60)
    p(" PORTFOLIO SUMMARY")
    p("=" * 60)
    p(f"  Initial Capital:    ${INITIAL_CAPITAL:,.2f}")
    p(f"  Final Balance:      ${result['final_balance']:,.2f}")
    p(f"  Total Return:       {(result['final_balance']/INITIAL_CAPITAL - 1)*100:+.2f}%")
    p(f"  Total Trades:       {metrics_all['total_trades']}")
    p(f"  Win Rate:           {metrics_all['win_rate']:.2f}%")
    p(f"  Profit Factor:      {metrics_all['profit_factor']:.3f}")
    p(f"  Max Drawdown:       {metrics_all['max_drawdown_pct']:.2f}%")
    p(f"  Gross Profit:       ${metrics_all['gross_profit']:,.2f}")
    p(f"  Gross Loss:         ${metrics_all['gross_loss']:,.2f}")
    p(f"  Avg PnL/trade:      {metrics_all['avg_pnl_pct']:.3f}%")
    p(f"  Avg Win:            {metrics_all['avg_win_pct']:.3f}%")
    p(f"  Avg Loss:           {metrics_all['avg_loss_pct']:.3f}%")
    p()

    # --- Model Breakdown ---
    p("=" * 60)
    p(" MODEL BREAKDOWN")
    p("=" * 60)
    for label, m, cap, bal, cfg in [
        ("Model A (PF Maximizer)", metrics_a, cap_a, result['balance_a'], MODEL_A),
        ("Model B (Compounder)", metrics_b, cap_b, result['balance_b'], MODEL_B),
    ]:
        p(f"\n  {label}")
        p(f"    Config:      {cfg.fast_ma_type}({cfg.fast_ma_period}) / {cfg.slow_ma_type}({cfg.slow_ma_period})")
        p(f"    ADX>={cfg.adx_threshold}, RSI {cfg.rsi_low}-{cfg.rsi_high}, Delay={cfg.entry_delay}")
        p(f"    SL={cfg.sl_pct*100}%, Trail=+{cfg.trail_activate_pct*100}%/-{cfg.trail_drop_pct*100}%")
        p(f"    Margin={cfg.margin_pct*100}%, Leverage={cfg.leverage}x")
        p(f"    Initial:     ${cap:,.2f}")
        p(f"    Final:       ${bal:,.2f}")
        p(f"    Return:      {(bal/cap-1)*100:+.2f}%")
        p(f"    Trades:      {m['total_trades']}")
        p(f"    Win Rate:    {m['win_rate']:.2f}%")
        p(f"    PF:          {m['profit_factor']:.3f}")
        p(f"    MDD:         {m['max_drawdown_pct']:.2f}%")
        p(f"    Avg Win:     {m['avg_win_pct']:.3f}%")
        p(f"    Avg Loss:    {m['avg_loss_pct']:.3f}%")
    p()

    # --- Monthly Table (Portfolio) ---
    p("=" * 60)
    p(" FULL MONTHLY TABLE (Portfolio)")
    p("=" * 60)
    monthly_df = calc_monthly_table(all_trades, INITIAL_CAPITAL)
    if not monthly_df.empty:
        # Only show months that have trades or are near trade months
        active = monthly_df[monthly_df['Trades'] > 0]
        if len(active) > 0:
            p(f"\n  Active months ({len(active)} of {len(monthly_df)}):\n")
        p(monthly_df.to_string(index=False))
    p()

    # --- Monthly per model ---
    for cfg, cap_m in [(MODEL_A, cap_a), (MODEL_B, cap_b)]:
        p(f"--- Monthly: {cfg.name} ---")
        mdf = calc_monthly_table(all_trades, cap_m, cfg.name)
        if not mdf.empty:
            active = mdf[mdf['Trades'] > 0]
            if len(active) > 0:
                p(f"  (Showing only active months)\n")
                p(active.to_string(index=False))
            else:
                p("  No trades")
        p()

    # --- Yearly Summary ---
    p("=" * 60)
    p(" YEARLY SUMMARY")
    p("=" * 60)
    yearly = {}
    for t in all_trades:
        y = t['exit_time'].year
        if y not in yearly:
            yearly[y] = {'pnl': 0, 'trades': 0, 'wins': 0}
        yearly[y]['pnl'] += t['pnl_usd']
        yearly[y]['trades'] += 1
        if t['pnl_usd'] > 0:
            yearly[y]['wins'] += 1

    p(f"  {'Year':<6} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'PnL USD':>12} {'Return%':>10}")
    p(f"  {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*12} {'-'*10}")
    running = INITIAL_CAPITAL
    for y in sorted(yearly.keys()):
        d = yearly[y]
        wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
        ret = d['pnl'] / running * 100 if running > 0 else 0
        p(f"  {y:<6} {d['trades']:>7} {d['wins']:>6} {wr:>6.1f}% {d['pnl']:>+12,.2f} {ret:>+9.2f}%")
        running += d['pnl']
    p()

    # --- Model x Year Breakdown ---
    p("=" * 60)
    p(" MODEL x YEAR BREAKDOWN")
    p("=" * 60)
    for cfg_name, trades_list, cap_m in [
        (MODEL_A.name, trades_a, cap_a),
        (MODEL_B.name, trades_b, cap_b),
    ]:
        p(f"\n  {cfg_name}:")
        yearly_m = {}
        for t in trades_list:
            y = t['exit_time'].year
            if y not in yearly_m:
                yearly_m[y] = {'pnl': 0, 'trades': 0, 'wins': 0}
            yearly_m[y]['pnl'] += t['pnl_usd']
            yearly_m[y]['trades'] += 1
            if t['pnl_usd'] > 0:
                yearly_m[y]['wins'] += 1

        p(f"    {'Year':<6} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'PnL USD':>12}")
        p(f"    {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*12}")
        for y in sorted(yearly_m.keys()):
            d = yearly_m[y]
            wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
            p(f"    {y:<6} {d['trades']:>7} {d['wins']:>6} {wr:>6.1f}% {d['pnl']:>+12,.2f}")
    p()

    # --- Direction Analysis ---
    p("=" * 60)
    p(" DIRECTION ANALYSIS (LONG vs SHORT)")
    p("=" * 60)
    for direction in ['LONG', 'SHORT']:
        dt = [t for t in all_trades if t['side'] == direction]
        if not dt:
            p(f"  {direction}: No trades")
            continue
        wins_d = len([t for t in dt if t['pnl_usd'] > 0])
        total_pnl = sum(t['pnl_usd'] for t in dt)
        wr = wins_d / len(dt) * 100
        avg_pnl = np.mean([t['pnl_pct'] for t in dt])
        gp = sum(t['pnl_usd'] for t in dt if t['pnl_usd'] > 0)
        gl = abs(sum(t['pnl_usd'] for t in dt if t['pnl_usd'] <= 0))
        pf = gp / gl if gl > 0 else float('inf')
        p(f"  {direction}:")
        p(f"    Trades: {len(dt)}, Wins: {wins_d}, WR: {wr:.1f}%")
        p(f"    PnL: ${total_pnl:+,.2f}, PF: {pf:.3f}, Avg ROI: {avg_pnl:+.3f}%")
    p()

    # --- Hold Time Analysis ---
    p("=" * 60)
    p(" HOLD TIME ANALYSIS (in 15m bars)")
    p("=" * 60)
    hold_bars = [t['hold_bars'] for t in all_trades]
    if hold_bars:
        hold_hours = [b * 15 / 60 for b in hold_bars]
        p(f"  Min:     {min(hold_bars)} bars ({min(hold_hours):.1f} hours)")
        p(f"  Max:     {max(hold_bars)} bars ({max(hold_hours):.1f} hours)")
        p(f"  Mean:    {np.mean(hold_bars):.1f} bars ({np.mean(hold_hours):.1f} hours)")
        p(f"  Median:  {np.median(hold_bars):.0f} bars ({np.median(hold_hours):.1f} hours)")

        win_holds = [t['hold_bars'] for t in all_trades if t['pnl_usd'] > 0]
        loss_holds = [t['hold_bars'] for t in all_trades if t['pnl_usd'] <= 0]
        if win_holds:
            p(f"  Avg Win Hold:  {np.mean(win_holds):.1f} bars ({np.mean(win_holds)*15/60:.1f} hours)")
        if loss_holds:
            p(f"  Avg Loss Hold: {np.mean(loss_holds):.1f} bars ({np.mean(loss_holds)*15/60:.1f} hours)")
    p()

    # --- Exit Reason Breakdown ---
    p("=" * 60)
    p(" EXIT REASON BREAKDOWN")
    p("=" * 60)
    reasons = {}
    for t in all_trades:
        r = t['exit_reason']
        if r not in reasons:
            reasons[r] = {'count': 0, 'pnl': 0, 'wins': 0}
        reasons[r]['count'] += 1
        reasons[r]['pnl'] += t['pnl_usd']
        if t['pnl_usd'] > 0:
            reasons[r]['wins'] += 1

    p(f"  {'Reason':<10} {'Count':>7} {'Wins':>6} {'WR%':>7} {'PnL USD':>12}")
    p(f"  {'-'*10} {'-'*7} {'-'*6} {'-'*7} {'-'*12}")
    for r in sorted(reasons.keys()):
        d = reasons[r]
        wr = d['wins'] / d['count'] * 100
        p(f"  {r:<10} {d['count']:>7} {d['wins']:>6} {wr:>6.1f}% {d['pnl']:>+12,.2f}")
    p()

    # --- Top 10 / Bottom 10 Trades ---
    p("=" * 60)
    p(" TOP 10 BEST TRADES")
    p("=" * 60)
    sorted_trades = sorted(all_trades, key=lambda x: x['pnl_usd'], reverse=True)
    p(f"  {'#':<4} {'Model':<25} {'Side':<6} {'ROI%':>8} {'PnL USD':>12} {'Exit':>8} {'Date'}")
    p(f"  {'-'*4} {'-'*25} {'-'*6} {'-'*8} {'-'*12} {'-'*8} {'-'*19}")
    for idx, t in enumerate(sorted_trades[:10]):
        p(f"  {idx+1:<4} {t['model']:<25} {t['side']:<6} {t['pnl_pct']:>+7.2f}% {t['pnl_usd']:>+12,.2f} {t['exit_reason']:>8} {t['exit_time'].strftime('%Y-%m-%d %H:%M')}")
    p()

    p("=" * 60)
    p(" BOTTOM 10 WORST TRADES")
    p("=" * 60)
    p(f"  {'#':<4} {'Model':<25} {'Side':<6} {'ROI%':>8} {'PnL USD':>12} {'Exit':>8} {'Date'}")
    p(f"  {'-'*4} {'-'*25} {'-'*6} {'-'*8} {'-'*12} {'-'*8} {'-'*19}")
    for idx, t in enumerate(sorted_trades[-10:]):
        p(f"  {idx+1:<4} {t['model']:<25} {t['side']:<6} {t['pnl_pct']:>+7.2f}% {t['pnl_usd']:>+12,.2f} {t['exit_reason']:>8} {t['exit_time'].strftime('%Y-%m-%d %H:%M')}")
    p()

    # ===== 30-RUN VERIFICATION =====
    p("=" * 60)
    p(" 30-RUN VERIFICATION")
    p("=" * 60)
    print("\nRunning 30 verification iterations...")
    run_results = []
    for r in range(30):
        res = run_full_backtest(df, run_id=r)
        m = calc_metrics(res['all_trades'], INITIAL_CAPITAL)
        run_results.append({
            'run': r + 1,
            'final_balance': round(res['final_balance'], 2),
            'total_return_pct': m['total_return_pct'],
            'trades': m['total_trades'],
            'win_rate': m['win_rate'],
            'profit_factor': m['profit_factor'],
            'max_drawdown': m['max_drawdown_pct'],
        })
        if (r + 1) % 10 == 0:
            print(f"  Run {r+1}/30 complete...")

    run_df = pd.DataFrame(run_results)
    p(f"\n  {'Metric':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    p(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for col in ['final_balance', 'total_return_pct', 'trades', 'win_rate', 'profit_factor', 'max_drawdown']:
        mean_v = run_df[col].mean()
        std_v = run_df[col].std()
        min_v = run_df[col].min()
        max_v = run_df[col].max()
        p(f"  {col:<20} {mean_v:>12.2f} {std_v:>12.4f} {min_v:>12.2f} {max_v:>12.2f}")
    p()
    p("  NOTE: Deterministic backtest - all 30 runs produce identical results.")
    p()

    # ===== SAVE FILES =====
    print("\nSaving output files...")

    # 1. Report
    report_path = BASE_DIR / "v25_FINAL_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"  Saved: {report_path}")

    # 2. Trades CSV
    trades_path = BASE_DIR / "v25_FINAL_trades.csv"
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv(trades_path, index=False, encoding='utf-8')
    print(f"  Saved: {trades_path}")

    # 3. Monthly CSV
    monthly_path = BASE_DIR / "v25_FINAL_monthly.csv"
    monthly_df.to_csv(monthly_path, index=False, encoding='utf-8')
    print(f"  Saved: {monthly_path}")

    # 4. Metrics JSON
    metrics_path = BASE_DIR / "v25_FINAL_metrics.json"
    metrics_json = {
        'version': 'v25.1_FINAL',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'initial_capital': INITIAL_CAPITAL,
        'fee_rate': FEE_RATE,
        'portfolio': metrics_all,
        'model_a': metrics_a,
        'model_b': metrics_b,
        'config_a': {
            'name': MODEL_A.name,
            'fast_ma': f"{MODEL_A.fast_ma_type}({MODEL_A.fast_ma_period})",
            'slow_ma': f"{MODEL_A.slow_ma_type}({MODEL_A.slow_ma_period})",
            'adx_threshold': MODEL_A.adx_threshold,
            'rsi_range': f"{MODEL_A.rsi_low}-{MODEL_A.rsi_high}",
            'entry_delay_candles': MODEL_A.entry_delay,
            'sl_pct': MODEL_A.sl_pct,
            'trail_activate_pct': MODEL_A.trail_activate_pct,
            'trail_drop_pct': MODEL_A.trail_drop_pct,
            'margin_pct': MODEL_A.margin_pct,
            'leverage': MODEL_A.leverage,
            'capital_allocation': MODEL_A.capital_pct,
        },
        'config_b': {
            'name': MODEL_B.name,
            'fast_ma': f"{MODEL_B.fast_ma_type}({MODEL_B.fast_ma_period})",
            'slow_ma': f"{MODEL_B.slow_ma_type}({MODEL_B.slow_ma_period})",
            'adx_threshold': MODEL_B.adx_threshold,
            'rsi_range': f"{MODEL_B.rsi_low}-{MODEL_B.rsi_high}",
            'entry_delay_candles': MODEL_B.entry_delay,
            'sl_pct': MODEL_B.sl_pct,
            'trail_activate_pct': MODEL_B.trail_activate_pct,
            'trail_drop_pct': MODEL_B.trail_drop_pct,
            'margin_pct': MODEL_B.margin_pct,
            'leverage': MODEL_B.leverage,
            'capital_allocation': MODEL_B.capital_pct,
        },
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {metrics_path}")

    # 5. 30-run CSV
    run30_path = BASE_DIR / "v25_FINAL_30run.csv"
    run_df.to_csv(run30_path, index=False, encoding='utf-8')
    print(f"  Saved: {run30_path}")

    total_time = time.time() - t_start
    p(f"\nTotal execution time: {total_time:.1f}s")
    print(f"\n{'='*80}")
    print(f"  ALL DONE. Total time: {total_time:.1f}s")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
