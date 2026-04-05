"""
v23.4 BTC/USDT Futures - CROSS vs ISOLATED Margin Comparison Pipeline
======================================================================
Tests top 5 strategies from v23.3/prior with BOTH margin modes.
5 strategies x 8 sizing combos x 2 margin modes = 80 backtests.
Then 30-run verification for the best combo.
Includes SL slippage stress test (50% worse fill).

Initial capital: $5,000 | Fee: 0.04%
"""

import numpy as np
import pandas as pd
import time
import os
import sys
import io
import warnings
from math import log10
from collections import defaultdict

warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print(f"Python {sys.version}", flush=True)
sys.stdout.flush()

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = r"D:\filesystem\futures\btc_V1\test3"
CSV_FILES = [
    os.path.join(DATA_DIR, f"btc_usdt_5m_2020_to_now_part{i}.csv")
    for i in range(1, 4)
]
INITIAL_CAPITAL = 5000.0
FEE_RATE = 0.0004  # 0.04%

# ============================================================
# 5 STRATEGIES TO TEST
# ============================================================
STRATEGIES = [
    {
        'id': 'S1',
        'name': 'WMA(3)/EMA(150) 15m ADX14>=45 RSI40-75 D5 SL-10% T+5/-3',
        'fast_type': 'WMA', 'fast_period': 3,
        'slow_type': 'EMA', 'slow_period': 150,
        'tf': '15m', 'adx_period': 14, 'adx_min': 45,
        'rsi_lo': 40, 'rsi_hi': 75, 'delay': 5,
        'sl_pct': 0.10, 'trail_act': 0.05, 'trail_width': 0.03,
        'source': 'v23.3 TOP1',
    },
    {
        'id': 'S2',
        'name': 'WMA(3)/EMA(150) 15m ADX14>=45 RSI40-75 D5 SL-12% T+3/-2',
        'fast_type': 'WMA', 'fast_period': 3,
        'slow_type': 'EMA', 'slow_period': 150,
        'tf': '15m', 'adx_period': 14, 'adx_min': 45,
        'rsi_lo': 40, 'rsi_hi': 75, 'delay': 5,
        'sl_pct': 0.12, 'trail_act': 0.03, 'trail_width': 0.02,
        'source': 'v23.3 TOP3 zero-loss',
    },
    {
        'id': 'S3',
        'name': 'WMA(3)/EMA(200) 30m ADX20>=35 RSI35-60 T+4/-1 SL-8%',
        'fast_type': 'WMA', 'fast_period': 3,
        'slow_type': 'EMA', 'slow_period': 200,
        'tf': '30m', 'adx_period': 20, 'adx_min': 35,
        'rsi_lo': 35, 'rsi_hi': 60, 'delay': 0,
        'sl_pct': 0.08, 'trail_act': 0.04, 'trail_width': 0.01,
        'source': 'v22.1 best verified',
    },
    {
        'id': 'S4',
        'name': 'EMA(3)/EMA(200) 30m ADX14>=35 RSI30-65 T+6/-3 SL-7%',
        'fast_type': 'EMA', 'fast_period': 3,
        'slow_type': 'EMA', 'slow_period': 200,
        'tf': '30m', 'adx_period': 14, 'adx_min': 35,
        'rsi_lo': 30, 'rsi_hi': 65, 'delay': 0,
        'sl_pct': 0.07, 'trail_act': 0.06, 'trail_width': 0.03,
        'source': 'v14.4/v22.2 classic',
    },
    {
        'id': 'S5',
        'name': 'HMA(3)/EMA(250) 15m ADX14>=45 RSI35-70 D3 T+3/-2 SL-8%',
        'fast_type': 'HMA', 'fast_period': 3,
        'slow_type': 'EMA', 'slow_period': 250,
        'tf': '15m', 'adx_period': 14, 'adx_min': 45,
        'rsi_lo': 35, 'rsi_hi': 70, 'delay': 3,
        'sl_pct': 0.08, 'trail_act': 0.03, 'trail_width': 0.02,
        'source': 'v25.1 verified',
    },
]

# 8 sizing combos
SIZING_COMBOS = [
    {'margin': 0.15, 'lev': 5,  'label': 'M15%_L5x'},
    {'margin': 0.15, 'lev': 7,  'label': 'M15%_L7x'},
    {'margin': 0.15, 'lev': 10, 'label': 'M15%_L10x'},
    {'margin': 0.20, 'lev': 5,  'label': 'M20%_L5x'},
    {'margin': 0.20, 'lev': 7,  'label': 'M20%_L7x'},
    {'margin': 0.20, 'lev': 10, 'label': 'M20%_L10x'},
    {'margin': 0.25, 'lev': 10, 'label': 'M25%_L10x'},
    {'margin': 0.30, 'lev': 10, 'label': 'M30%_L10x'},
]

# ============================================================
# INDICATOR FUNCTIONS (pure Python, Wilder ADX)
# ============================================================
def wilder_smooth(arr, period):
    out = np.full(len(arr), np.nan)
    s = 0
    while s < len(arr) and np.isnan(arr[s]):
        s += 1
    if s + period > len(arr):
        return out
    out[s + period - 1] = np.nanmean(arr[s:s + period])
    for i in range(s + period, len(arr)):
        if not np.isnan(arr[i]) and not np.isnan(out[i - 1]):
            out[i] = (out[i - 1] * (period - 1) + arr[i]) / period
    return out


def calc_adx_wilder(high, low, close, period=14):
    n = len(high)
    tr = np.full(n, np.nan)
    pdm = np.full(n, np.nan)
    mdm = np.full(n, np.nan)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
    atr = wilder_smooth(tr, period)
    spdm = wilder_smooth(pdm, period)
    smdm = wilder_smooth(mdm, period)
    pdi = np.full(n, np.nan)
    mdi = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr[i]) and atr[i] > 0:
            pdi[i] = 100 * spdm[i] / atr[i]
            mdi[i] = 100 * smdm[i] / atr[i]
            s = pdi[i] + mdi[i]
            dx[i] = 100 * abs(pdi[i] - mdi[i]) / s if s > 0 else 0
    adx = wilder_smooth(dx, period)
    return adx


def calc_ema(close, period):
    out = np.full(len(close), np.nan)
    s = 0
    while s < len(close) and np.isnan(close[s]):
        s += 1
    if s + period > len(close):
        return out
    out[s + period - 1] = np.nanmean(close[s:s + period])
    m = 2.0 / (period + 1)
    for i in range(s + period, len(close)):
        if not np.isnan(close[i]) and not np.isnan(out[i - 1]):
            out[i] = close[i] * m + out[i - 1] * (1 - m)
    return out


def calc_wma(close, period):
    n = len(close)
    out = np.full(n, np.nan)
    weights = np.arange(1, period + 1, dtype=float)
    wsum = weights.sum()
    for i in range(period - 1, n):
        sl = close[i - period + 1:i + 1]
        if not np.any(np.isnan(sl)):
            out[i] = np.dot(sl, weights) / wsum
    return out


def calc_hma(close, period):
    hp = max(period // 2, 1)
    sp = max(int(np.sqrt(period)), 1)
    wh = calc_wma(close, hp)
    wf = calc_wma(close, period)
    d = np.where(np.isnan(wh) | np.isnan(wf), np.nan, 2.0 * wh - wf)
    return calc_wma(d, sp)


def calc_sma(close, period):
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        sl = close[i - period + 1:i + 1]
        if not np.any(np.isnan(sl)):
            out[i] = np.mean(sl)
    return out


def calc_rsi(close, period=14):
    n = len(close)
    out = np.full(n, np.nan)
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        d = close[i] - close[i - 1]
        if d > 0:
            gains[i] = d
        else:
            losses[i] = -d
    avg_g = wilder_smooth(gains, period)
    avg_l = wilder_smooth(losses, period)
    for i in range(n):
        if not np.isnan(avg_g[i]) and not np.isnan(avg_l[i]):
            if avg_l[i] == 0:
                out[i] = 100.0
            else:
                out[i] = 100 - 100 / (1 + avg_g[i] / avg_l[i])
    return out


def calc_fast_ma(close, ma_type, period):
    if ma_type == 'WMA':
        return calc_wma(close, period)
    if ma_type == 'HMA':
        return calc_hma(close, period)
    if ma_type == 'EMA':
        return calc_ema(close, period)
    return calc_ema(close, period)


def calc_slow_ma(close, ma_type, period):
    if ma_type == 'EMA':
        return calc_ema(close, period)
    if ma_type == 'SMA':
        return calc_sma(close, period)
    if ma_type == 'WMA':
        return calc_wma(close, period)
    return calc_ema(close, period)


# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    print("Loading CSVs...", flush=True)
    t0 = time.time()
    dfs = []
    for f in CSV_FILES:
        df = pd.read_csv(f, parse_dates=['timestamp'])
        dfs.append(df)
        print(f"  {os.path.basename(f)}: {len(df):,}", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.set_index('timestamp', inplace=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(np.float64)
    print(f"Total 5m: {len(df):,} | {df.index[0]} - {df.index[-1]} | {time.time()-t0:.1f}s\n", flush=True)
    return df


def resample_data(df5m, tf_str):
    rm = {'5m': None, '10m': '10min', '15m': '15min', '30m': '30min'}
    if tf_str == '5m':
        return df5m[['open', 'high', 'low', 'close', 'volume']].copy()
    out = df5m.resample(rm[tf_str]).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    return out


# ============================================================
# BACKTEST ENGINE - CROSS vs ISOLATED
# ============================================================
def run_backtest(cfg, df_tf, margin_mode='ISOLATED', margin_pct=0.20, leverage=10,
                 sl_slippage=0.0, return_trades=True):
    """
    Full backtest engine supporting both CROSS and ISOLATED margin.

    margin_mode: 'CROSS' or 'ISOLATED'
    sl_slippage: 0.0 = perfect fill, 0.5 = SL fills 50% worse than intended

    Key difference:
    ISOLATED: loss capped at allocated_margin (margin_pct * balance)
    CROSS: loss capped at full balance (but SL should prevent this)
    """
    closes = df_tf['close'].values
    highs = df_tf['high'].values
    lows = df_tf['low'].values
    opens = df_tf['open'].values
    timestamps = df_tf.index

    # Calculate indicators
    close_arr = closes.astype(np.float64)
    high_arr = highs.astype(np.float64)
    low_arr = lows.astype(np.float64)

    fast_ma = calc_fast_ma(close_arr, cfg['fast_type'], cfg['fast_period'])
    slow_ma = calc_slow_ma(close_arr, cfg['slow_type'], cfg['slow_period'])
    adx = calc_adx_wilder(high_arr, low_arr, close_arr, cfg['adx_period'])
    rsi_arr = calc_rsi(close_arr, 14)

    n = len(closes)
    bal = INITIAL_CAPITAL
    peak_bal = INITIAL_CAPITAL
    pos_dir = 0
    pos_entry = 0.0
    pos_size = 0.0  # in qty (notional / price)
    pos_margin = 0.0
    pos_time_idx = 0
    pos_highest = 0.0
    pos_lowest = 0.0
    pos_trail_active = False
    pos_trail_sl = 0.0
    pos_sl = 0.0

    pending_signal = 0
    pending_bar = 0

    trades = []
    _n_trades = 0
    _n_wins = 0
    _gross_profit = 0.0
    _gross_loss = 0.0
    _win_roi_sum = 0.0
    _loss_roi_sum = 0.0
    _sl_count = 0
    _trail_count = 0
    _rev_count = 0
    _end_count = 0
    _max_consec_loss = 0
    _cur_consec_loss = 0
    _mdd = 0.0
    _mdd_peak = INITIAL_CAPITAL
    _liquidation_events = 0
    _liq_distance_min = 999.0  # closest liquidation distance seen

    tf_hours = {'5m': 1 / 12, '10m': 1 / 6, '15m': 0.25, '30m': 0.5}
    bar_h = tf_hours.get(cfg['tf'], 0.25)
    warmup = max(300, int(cfg['slow_period'] * 1.5))

    dd_halved = False

    def close_pos(exit_price, exit_reason, bar_idx):
        nonlocal bal, peak_bal, pos_dir
        nonlocal _n_trades, _n_wins, _gross_profit, _gross_loss
        nonlocal _win_roi_sum, _loss_roi_sum, _sl_count, _trail_count, _rev_count
        nonlocal _end_count, _max_consec_loss, _cur_consec_loss
        nonlocal _mdd, _mdd_peak, _liquidation_events

        notional = pos_size * pos_entry  # approximate notional
        if pos_dir == 1:
            rpnl = pos_size * (exit_price - pos_entry)
        else:
            rpnl = pos_size * (pos_entry - exit_price)
        fee_cost = pos_size * exit_price * FEE_RATE
        total_pnl = rpnl - fee_cost

        # MARGIN MODE DIFFERENCE
        if margin_mode == 'ISOLATED':
            # ISOLATED: loss capped at allocated margin
            if total_pnl < -pos_margin:
                total_pnl = -pos_margin
                _liquidation_events += 1
        else:
            # CROSS: loss capped at full balance
            if total_pnl < -bal:
                total_pnl = -bal
                _liquidation_events += 1

        bal += total_pnl
        if bal < 0:
            bal = 0
        peak_bal = max(peak_bal, bal)

        margin = pos_margin
        roi = total_pnl / margin * 100 if margin > 0 else 0
        hold_bars = bar_idx - pos_time_idx

        _n_trades += 1
        if total_pnl > 0:
            _n_wins += 1
            _gross_profit += total_pnl
            _win_roi_sum += roi
            _cur_consec_loss = 0
        else:
            _gross_loss += abs(total_pnl)
            _loss_roi_sum += roi
            _cur_consec_loss += 1
            _max_consec_loss = max(_max_consec_loss, _cur_consec_loss)

        if exit_reason == 'SL':
            _sl_count += 1
        elif exit_reason == 'TRAIL':
            _trail_count += 1
        elif exit_reason == 'REV':
            _rev_count += 1
        elif exit_reason == 'END':
            _end_count += 1

        _mdd_peak = max(_mdd_peak, bal)
        if _mdd_peak > 0:
            dd_now = (_mdd_peak - bal) / _mdd_peak
            _mdd = max(_mdd, dd_now)

        if return_trades:
            trades.append({
                'direction': 'LONG' if pos_dir == 1 else 'SHORT',
                'entry_time': str(timestamps[pos_time_idx]),
                'exit_time': str(timestamps[bar_idx]),
                'entry_price': round(pos_entry, 2),
                'exit_price': round(exit_price, 2),
                'exit_reason': exit_reason,
                'margin': round(margin, 2),
                'pnl': round(total_pnl, 2),
                'roi_pct': round(roi, 2),
                'peak_roi_pct': round(
                    ((pos_highest - pos_entry) / pos_entry * leverage * 100) if pos_dir == 1 else
                    ((pos_entry - pos_lowest) / pos_entry * leverage * 100), 2),
                'hold_bars': hold_bars,
                'hold_hours': round(hold_bars * bar_h, 2),
                'balance': round(bal, 2),
            })
        pos_dir = 0
        return total_pnl

    for i in range(warmup, n):
        if bal <= 0:
            break

        c = closes[i]
        h = highs[i]
        l = lows[i]

        # DD protection
        if peak_bal > 0:
            cur_dd = (peak_bal - bal) / peak_bal
            if cur_dd > 0.30:
                dd_halved = True
            elif cur_dd < 0.15:
                dd_halved = False

        # ---- Position management ----
        if pos_dir != 0:
            exited = False

            # Calculate liquidation distance for tracking
            if margin_mode == 'ISOLATED':
                liq_dist = pos_margin / (pos_size * pos_entry) if pos_size > 0 else 999
            else:
                liq_dist = bal / (pos_size * pos_entry) if pos_size > 0 else 999
            _liq_distance_min = min(_liq_distance_min, liq_dist)

            if pos_dir == 1:
                pos_highest = max(pos_highest, h)

                # SL check on intrabar LOW
                sl_fill = pos_sl
                if sl_slippage > 0:
                    # SL fills worse by slippage factor
                    sl_gap = pos_entry - pos_sl
                    sl_fill = pos_sl - sl_gap * sl_slippage

                if l <= pos_sl:
                    close_pos(sl_fill, 'SL', i)
                    exited = True
                else:
                    # Trail check on CLOSE
                    cur_roi = (c - pos_entry) / pos_entry
                    if cur_roi >= cfg['trail_act']:
                        pos_trail_active = True
                    if pos_trail_active:
                        new_tsl = pos_highest * (1 - cfg['trail_width'])
                        if new_tsl > pos_trail_sl:
                            pos_trail_sl = max(new_tsl, pos_sl)
                        if c <= pos_trail_sl:
                            close_pos(c, 'TRAIL', i)
                            exited = True

            elif pos_dir == -1:
                pos_lowest = min(pos_lowest, l)

                # SL check on intrabar HIGH
                sl_fill = pos_sl
                if sl_slippage > 0:
                    sl_gap = pos_sl - pos_entry
                    sl_fill = pos_sl + sl_gap * sl_slippage

                if h >= pos_sl:
                    close_pos(sl_fill, 'SL', i)
                    exited = True
                else:
                    cur_roi = (pos_entry - c) / pos_entry
                    if cur_roi >= cfg['trail_act']:
                        pos_trail_active = True
                    if pos_trail_active:
                        new_tsl = pos_lowest * (1 + cfg['trail_width'])
                        if new_tsl < pos_trail_sl:
                            pos_trail_sl = min(new_tsl, pos_sl)
                        if c >= pos_trail_sl:
                            close_pos(c, 'TRAIL', i)
                            exited = True

            if exited:
                if bal <= 0:
                    break
                continue

        # ---- Signal detection ----
        if i < warmup + 1:
            continue
        if np.isnan(fast_ma[i]) or np.isnan(fast_ma[i - 1]):
            continue
        if np.isnan(slow_ma[i]) or np.isnan(slow_ma[i - 1]):
            continue

        cross_up = (fast_ma[i - 1] <= slow_ma[i - 1]) and (fast_ma[i] > slow_ma[i])
        cross_down = (fast_ma[i - 1] >= slow_ma[i - 1]) and (fast_ma[i] < slow_ma[i])

        if cross_up:
            pending_signal = 1
            pending_bar = i
        elif cross_down:
            pending_signal = -1
            pending_bar = i

        # ---- Delayed entry check ----
        if pending_signal != 0 and (i - pending_bar) == cfg['delay']:
            sig = pending_signal
            pending_signal = 0

            if sig == 1 and fast_ma[i] <= slow_ma[i]:
                continue
            if sig == -1 and fast_ma[i] >= slow_ma[i]:
                continue

            if np.isnan(adx[i]) or adx[i] < cfg['adx_min']:
                continue
            if np.isnan(rsi_arr[i]) or not (cfg['rsi_lo'] <= rsi_arr[i] <= cfg['rsi_hi']):
                continue

            direction = sig

            # Same-direction skip
            if pos_dir == direction:
                continue

            # REVERSE: close opposite position
            if pos_dir != 0 and pos_dir != direction:
                close_pos(c, 'REV', i)
                if bal <= 0:
                    break

            # Monthly loss limit: -15%
            mk = f"{timestamps[i].year}-{timestamps[i].month:02d}"

            if bal < 50:
                continue

            # Position sizing
            margin_mult = 0.5 if dd_halved else 1.0
            sz = bal * margin_pct * margin_mult
            if sz < 5:
                continue
            notional = sz * leverage

            # SL price
            if direction == 1:
                pos_sl = c * (1 - cfg['sl_pct'])
            else:
                pos_sl = c * (1 + cfg['sl_pct'])

            # Entry fee
            entry_fee = notional / c * c * FEE_RATE  # = notional * FEE_RATE
            bal -= entry_fee

            # Open position
            pos_dir = direction
            pos_entry = c
            pos_size = notional / c  # qty in BTC
            pos_margin = sz
            pos_time_idx = i
            pos_highest = c
            pos_lowest = c
            pos_trail_active = False
            pos_trail_sl = pos_sl

    # Close remaining position
    if pos_dir != 0 and bal > 0:
        close_pos(closes[-1], 'END', n - 1)

    total_ret = (bal - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    n_losses = _n_trades - _n_wins
    avg_win_roi = _win_roi_sum / _n_wins if _n_wins > 0 else 0
    avg_loss_roi = _loss_roi_sum / n_losses if n_losses > 0 else 0

    metrics = {
        'initial': INITIAL_CAPITAL,
        'final': round(bal, 2),
        'return_pct': round(total_ret, 2),
        'trades': _n_trades,
        'wins': _n_wins,
        'losses': n_losses,
        'pf': round(_gross_profit / _gross_loss, 4) if _gross_loss > 0 else (999 if _gross_profit > 0 else 0),
        'mdd_pct': round(_mdd * 100, 2),
        'win_rate': round(_n_wins / _n_trades * 100, 2) if _n_trades > 0 else 0,
        'avg_win': round(avg_win_roi, 2),
        'avg_loss': round(avg_loss_roi, 2),
        'rr': round(abs(avg_win_roi / avg_loss_roi), 2) if avg_loss_roi != 0 else 999,
        'max_consec_loss': _max_consec_loss,
        'gross_profit': round(_gross_profit, 2),
        'gross_loss': round(_gross_loss, 2),
        'sl_count': _sl_count,
        'trail_count': _trail_count,
        'rev_count': _rev_count,
        'end_count': _end_count,
        'liquidation_events': _liquidation_events,
        'liq_distance_min_pct': round(_liq_distance_min * 100, 2) if _liq_distance_min < 999 else 999,
    }

    return metrics, trades


# ============================================================
# REPORT GENERATION
# ============================================================
def generate_monthly_table(trades_list, timestamps_index):
    """Generate monthly performance rows from trades."""
    if not trades_list:
        return []

    df = pd.DataFrame(trades_list)
    df['exit_dt'] = pd.to_datetime(df['exit_time'])
    df['month'] = df['exit_dt'].dt.to_period('M')
    all_months = pd.period_range('2020-01', '2026-03', freq='M')
    mg = df.groupby('month')

    rb = INITIAL_CAPITAL
    monthly_rows = []

    for mo in all_months:
        if mo in mg.groups:
            g = mg.get_group(mo)
            nt = len(g)
            nw = len(g[g['pnl'] > 0])
            nl = nt - nw
            wr = nw / nt * 100 if nt else 0
            gp = g[g['pnl'] > 0]['pnl'].sum() if nw else 0
            gl = abs(g[g['pnl'] <= 0]['pnl'].sum()) if nl else 0
            net = g['pnl'].sum()
            mpf = gp / gl if gl > 0 else (999 if gp > 0 else 0)
        else:
            nt = nw = nl = 0
            wr = 0
            gp = gl = net = 0
            mpf = 0

        sbr = rb
        rb += net
        mr = net / sbr * 100 if sbr > 0 else 0

        monthly_rows.append({
            'month': str(mo), 'trades': nt, 'wins': nw, 'losses': nl,
            'win_rate': round(wr, 1), 'gross_profit': round(gp, 2),
            'gross_loss': round(gl, 2), 'pnl': round(net, 2),
            'pf': round(mpf, 2) if mpf < 999 else 999,
            'balance': round(rb, 2), 'return_pct': round(mr, 2)
        })

    return monthly_rows


def print_full_report(label, cfg, metrics, trades_list, margin_mode, margin_pct, leverage):
    """Print detailed report for the winning strategy."""
    lines = []
    T = metrics['trades']

    lines.append(f"\n{'=' * 100}")
    lines.append(f"  FINAL WINNER REPORT  [{label}]")
    lines.append(f"{'=' * 100}")
    lines.append(f"  Strategy:      {cfg['name']}")
    lines.append(f"  Source:        {cfg['source']}")
    lines.append(f"  Margin Mode:   {margin_mode}")
    lines.append(f"  Margin:        {margin_pct * 100:.0f}% | Leverage: {leverage}x")
    lines.append(f"  SL: -{cfg['sl_pct'] * 100:.0f}% | Trail: +{cfg['trail_act'] * 100:.0f}%/-{cfg['trail_width'] * 100:.0f}%")
    lines.append(f"  Fee: {FEE_RATE * 100:.2f}% | REVERSE + Same-dir skip | DD Protection")
    lines.append(f"{'=' * 100}")
    lines.append(f"  Initial:        ${INITIAL_CAPITAL:,.0f}")
    lines.append(f"  Final:          ${metrics['final']:,.2f}")
    lines.append(f"  Return:         {metrics['return_pct']:+.1f}%")
    lines.append(f"  PF:             {metrics['pf']:.2f}")
    lines.append(f"  MDD:            {metrics['mdd_pct']:.1f}%")
    lines.append(f"  Trades:         {T}")
    lines.append(f"  Win Rate:       {metrics['win_rate']:.1f}%")
    lines.append(f"  Avg Win:        {metrics['avg_win']:+.2f}%")
    lines.append(f"  Avg Loss:       {metrics['avg_loss']:+.2f}%")
    lines.append(f"  R:R:            {metrics['rr']:.2f}")
    lines.append(f"  Max Consec L:   {metrics['max_consec_loss']}")
    lines.append(f"  SL Hits:        {metrics['sl_count']}")
    lines.append(f"  Trail Hits:     {metrics['trail_count']}")
    lines.append(f"  Rev Hits:       {metrics['rev_count']}")
    lines.append(f"  Liquidations:   {metrics['liquidation_events']}")
    lines.append(f"  Min Liq Dist:   {metrics['liq_distance_min_pct']:.1f}%")
    lines.append("")

    if not trades_list:
        lines.append("  NO TRADES")
        return "\n".join(lines)

    df = pd.DataFrame(trades_list)

    # Direction analysis
    lines.append("  DIRECTION ANALYSIS")
    lines.append("  " + "-" * 70)
    for d in ['LONG', 'SHORT']:
        sub = df[df['direction'] == d]
        if len(sub) == 0:
            continue
        sw = sub[sub['pnl'] > 0]
        lines.append(f"  {d:>5}: {len(sub):>4} ({len(sub) / T * 100:.0f}%) "
                      f"WR:{len(sw) / len(sub) * 100:.0f}% "
                      f"AvgROI:{sub['roi_pct'].mean():+.2f}% "
                      f"PnL:${sub['pnl'].sum():+,.0f}")

    # Exit reason analysis
    lines.append(f"\n  EXIT REASON ANALYSIS")
    lines.append("  " + "-" * 70)
    for r in ['TRAIL', 'REV', 'SL', 'END']:
        rt = df[df['exit_reason'] == r]
        if len(rt) == 0:
            continue
        rw = rt[rt['pnl'] > 0]
        wr_val = len(rw) / len(rt) * 100 if len(rt) else 0
        lines.append(f"  {r:>5}: {len(rt):>4} ({len(rt) / T * 100:.0f}%) "
                      f"WR:{wr_val:.0f}% AvgROI:{rt['roi_pct'].mean():+.2f}% "
                      f"PnL:${rt['pnl'].sum():+,.0f}")

    # Hold time analysis
    lines.append(f"\n  HOLD TIME ANALYSIS")
    lines.append("  " + "-" * 70)
    for a, b, lb in [(0, 2, '<2h'), (2, 8, '2-8h'), (8, 24, '8-24h'),
                      (24, 72, '1-3d'), (72, 168, '3-7d'), (168, 9999, '7d+')]:
        ht = df[(df['hold_hours'] >= a) & (df['hold_hours'] < b)]
        if len(ht):
            hw = ht[ht['pnl'] > 0]
            lines.append(f"  {lb:>6}: {len(ht):>4} "
                          f"WR:{len(hw) / len(ht) * 100:.0f}% "
                          f"AvgROI:{ht['roi_pct'].mean():+.2f}% "
                          f"PnL:${ht['pnl'].sum():+,.0f}")

    # Monthly table
    df['exit_dt'] = pd.to_datetime(df['exit_time'])
    df['month'] = df['exit_dt'].dt.to_period('M')
    all_months = pd.period_range('2020-01', '2026-03', freq='M')
    mg = df.groupby('month')

    lines.append(f"\n{'=' * 100}")
    lines.append(f"  MONTHLY PERFORMANCE (2020-01 to 2026-03)")
    lines.append(f"{'=' * 100}")
    lines.append(f"  {'Month':>8} {'Trd':>4} {'W':>3} {'L':>3} {'WR%':>5} "
                  f"{'GrossP':>9} {'GrossL':>9} {'NetPnL':>9} {'PF':>6} "
                  f"{'Bal':>10} {'Ret%':>7}")
    lines.append("  " + "-" * 95)

    rb = INITIAL_CAPITAL
    yearly = {}
    lm = 0
    pm = 0
    tm = 0

    for mo in all_months:
        if mo in mg.groups:
            g = mg.get_group(mo)
            nt = len(g)
            nw = len(g[g['pnl'] > 0])
            nl = nt - nw
            wr = nw / nt * 100 if nt else 0
            gp2 = g[g['pnl'] > 0]['pnl'].sum() if nw else 0
            gl2 = abs(g[g['pnl'] <= 0]['pnl'].sum()) if nl else 0
            net = g['pnl'].sum()
            mpf = gp2 / gl2 if gl2 > 0 else (999 if gp2 > 0 else 0)
        else:
            nt = nw = nl = 0
            wr = 0
            gp2 = gl2 = net = 0
            mpf = 0

        sbr = rb
        rb += net
        mr = net / sbr * 100 if sbr > 0 else 0
        tm += 1
        if net < 0:
            lm += 1
        if net > 0:
            pm += 1

        y = str(mo)[:4]
        if y not in yearly:
            yearly[y] = {'p': 0, 't': 0, 'w': 0, 'l': 0, 'gp': 0, 'gl': 0, 'sb': sbr}
        yearly[y]['p'] += net
        yearly[y]['t'] += nt
        yearly[y]['w'] += nw
        yearly[y]['l'] += nl
        yearly[y]['gp'] += gp2
        yearly[y]['gl'] += gl2
        yearly[y]['eb'] = rb

        pfs = f"{mpf:.1f}" if mpf < 999 else "INF"
        if mpf == 0 and net == 0:
            pfs = "-"
        mk = " <<" if net < 0 else ""
        lines.append(f"  {str(mo):>8} {nt:>4} {nw:>3} {nl:>3} {wr:>4.0f}% "
                      f"${gp2:>7,.0f} ${gl2:>7,.0f} ${net:>+7,.0f} {pfs:>5} "
                      f"${rb:>8,.0f} {mr:>+6.1f}%{mk}")

    # Yearly summary
    lines.append(f"\n{'=' * 100}")
    lines.append(f"  YEARLY PERFORMANCE")
    lines.append(f"{'=' * 100}")
    lines.append(f"  {'Year':>6} {'Trd':>4} {'W':>4} {'L':>4} {'WR%':>5} "
                  f"{'GrossP':>10} {'GrossL':>10} {'NetPnL':>10} {'PF':>6} {'YrRet%':>8}")
    lines.append("  " + "-" * 80)
    for y2 in sorted(yearly):
        yd = yearly[y2]
        ywr = yd['w'] / yd['t'] * 100 if yd['t'] else 0
        ypf = yd['gp'] / yd['gl'] if yd['gl'] > 0 else (999 if yd['gp'] > 0 else 0)
        yret = yd['p'] / yd['sb'] * 100 if yd['sb'] > 0 else 0
        pfs = f"{ypf:.2f}" if ypf < 999 else "INF"
        lines.append(f"  {y2:>6} {yd['t']:>3} {yd['w']:>4} {yd['l']:>4} {ywr:>4.0f}% "
                      f"${yd['gp']:>8,.0f} ${yd['gl']:>8,.0f} ${yd['p']:>+8,.0f} "
                      f"{pfs:>5} {yret:>+7.1f}%")

    pyrs = sum(1 for v in yearly.values() if v['p'] > 0)
    lines.append(f"\n  Profit Months: {pm}/{tm} ({pm / max(1, tm) * 100:.0f}%)")
    lines.append(f"  Loss Months:   {lm}/{tm} ({lm / max(1, tm) * 100:.0f}%)")
    lines.append(f"  Profit Years:  {pyrs}/{len(yearly)}")

    # Top/Bottom trades
    ds = df.sort_values('pnl', ascending=False)
    lines.append(f"\n  TOP 10 TRADES")
    lines.append("  " + "-" * 110)
    for idx, (_, r) in enumerate(ds.head(10).iterrows()):
        lines.append(f"  {idx + 1:>2} {r['direction']:>5} {r['entry_time'][:16]} -> "
                      f"{r['exit_time'][:16]} {r['exit_reason']:>5} "
                      f"ROI:{r['roi_pct']:>+7.1f}% PeakROI:{r['peak_roi_pct']:>+7.1f}% "
                      f"PnL:${r['pnl']:>+8,.0f}")
    lines.append(f"\n  BOTTOM 10 TRADES")
    lines.append("  " + "-" * 110)
    for idx, (_, r) in enumerate(ds.tail(10).iterrows()):
        lines.append(f"  {idx + 1:>2} {r['direction']:>5} {r['entry_time'][:16]} -> "
                      f"{r['exit_time'][:16]} {r['exit_reason']:>5} "
                      f"ROI:{r['roi_pct']:>+7.1f}% PeakROI:{r['peak_roi_pct']:>+7.1f}% "
                      f"PnL:${r['pnl']:>+8,.0f}")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================
def main():
    T0 = time.time()

    print(f"\n{'=' * 100}", flush=True)
    print(f"  v23.4 BTC/USDT FUTURES - CROSS vs ISOLATED MARGIN COMPARISON", flush=True)
    print(f"  5 Strategies x 8 Sizing Combos x 2 Margin Modes = 80 Backtests", flush=True)
    print(f"  + SL Slippage Stress Test + 30-Run Verification", flush=True)
    print(f"  Initial: ${INITIAL_CAPITAL:,.0f} | Fee: {FEE_RATE * 100:.2f}%", flush=True)
    print(f"{'=' * 100}\n", flush=True)

    # ================================================================
    # LOAD DATA
    # ================================================================
    df5m = load_data()

    # Precompute resampled data for each timeframe needed
    tfs_needed = list(set(s['tf'] for s in STRATEGIES))
    data_cache = {}
    for tf in tfs_needed:
        t0 = time.time()
        data_cache[tf] = resample_data(df5m, tf)
        print(f"  [{tf}] {len(data_cache[tf]):,} bars ({time.time() - t0:.1f}s)", flush=True)
    print("", flush=True)

    # ================================================================
    # PHASE 1: 80 BACKTESTS (5 strategies x 8 sizing x 2 margin modes)
    # ================================================================
    print(f"{'#' * 100}", flush=True)
    print(f"  PHASE 1: 80 BACKTESTS (CROSS vs ISOLATED)", flush=True)
    print(f"{'#' * 100}\n", flush=True)

    all_rows = []
    test_count = 0
    total_tests = len(STRATEGIES) * len(SIZING_COMBOS) * 2

    for strat in STRATEGIES:
        df_tf = data_cache[strat['tf']]
        print(f"\n  Strategy: {strat['id']} - {strat['name']}", flush=True)
        print(f"  {'Mode':<10} {'Sizing':<12} {'Final':>10} {'Ret%':>8} {'PF':>6} "
              f"{'MDD%':>6} {'Trd':>4} {'WR%':>5} {'SL':>3} {'Liq':>3} "
              f"{'LiqDist%':>9}", flush=True)
        print(f"  {'-' * 90}", flush=True)

        for sizing in SIZING_COMBOS:
            for mode in ['ISOLATED', 'CROSS']:
                t0 = time.time()
                metrics, _ = run_backtest(
                    strat, df_tf,
                    margin_mode=mode,
                    margin_pct=sizing['margin'],
                    leverage=sizing['lev'],
                    sl_slippage=0.0,
                    return_trades=False
                )
                elapsed = time.time() - t0
                test_count += 1

                row = {
                    'strategy_id': strat['id'],
                    'strategy_name': strat['name'],
                    'source': strat['source'],
                    'margin_mode': mode,
                    'margin_pct': sizing['margin'] * 100,
                    'leverage': sizing['lev'],
                    'sizing_label': sizing['label'],
                    'final_balance': metrics['final'],
                    'return_pct': metrics['return_pct'],
                    'pf': metrics['pf'],
                    'mdd_pct': metrics['mdd_pct'],
                    'trades': metrics['trades'],
                    'wins': metrics['wins'],
                    'losses': metrics['losses'],
                    'win_rate': metrics['win_rate'],
                    'sl_hits': metrics['sl_count'],
                    'trail_hits': metrics['trail_count'],
                    'rev_hits': metrics['rev_count'],
                    'liquidation_events': metrics['liquidation_events'],
                    'liq_distance_min_pct': metrics['liq_distance_min_pct'],
                    'max_consec_loss': metrics['max_consec_loss'],
                    'gross_profit': metrics['gross_profit'],
                    'gross_loss': metrics['gross_loss'],
                    'avg_win': metrics['avg_win'],
                    'avg_loss': metrics['avg_loss'],
                    'rr': metrics['rr'],
                }
                all_rows.append(row)

                pf_str = f"{metrics['pf']:.1f}" if metrics['pf'] < 999 else "INF"
                ld_str = f"{metrics['liq_distance_min_pct']:.1f}" if metrics['liq_distance_min_pct'] < 999 else "N/A"
                print(f"  {mode:<10} {sizing['label']:<12} "
                      f"${metrics['final']:>8,.0f} {metrics['return_pct']:>+7.1f}% "
                      f"{pf_str:>5} {metrics['mdd_pct']:>5.1f} {metrics['trades']:>4} "
                      f"{metrics['win_rate']:>4.0f}% {metrics['sl_count']:>3} "
                      f"{metrics['liquidation_events']:>3} {ld_str:>8}%", flush=True)

        print(f"  [{test_count}/{total_tests} done]", flush=True)

    # Save comparison CSV
    df_comp = pd.DataFrame(all_rows)
    csv_path = os.path.join(DATA_DIR, "v23_4_comparison.csv")
    df_comp.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}", flush=True)

    # ================================================================
    # PHASE 2: SL SLIPPAGE STRESS TEST
    # ================================================================
    print(f"\n\n{'#' * 100}", flush=True)
    print(f"  PHASE 2: SL SLIPPAGE STRESS TEST (50% worse fill)", flush=True)
    print(f"{'#' * 100}\n", flush=True)
    print(f"  This simulates SL filling 50% worse than intended", flush=True)
    print(f"  (e.g., SL -8% fills at -12%, SL -10% fills at -15%)\n", flush=True)

    slippage_rows = []
    # Test top 3 strategies with their best sizing combo
    # Find best sizing for each strategy first
    best_per_strat = {}
    for strat in STRATEGIES:
        strat_rows = [r for r in all_rows if r['strategy_id'] == strat['id']]
        if strat_rows:
            best = max(strat_rows, key=lambda r: r['return_pct'])
            best_per_strat[strat['id']] = (best['margin_pct'] / 100, best['leverage'], best['sizing_label'])

    print(f"  {'Strategy':<8} {'Mode':<10} {'Sizing':<12} {'Normal$':>10} {'Slip$':>10} "
          f"{'Normal%':>8} {'Slip%':>8} {'NormMDD':>8} {'SlipMDD':>8} "
          f"{'NormLiq':>8} {'SlipLiq':>8}", flush=True)
    print(f"  {'-' * 110}", flush=True)

    for strat in STRATEGIES:
        df_tf = data_cache[strat['tf']]
        mg_pct, lev, sz_label = best_per_strat.get(strat['id'], (0.15, 5, 'M15%_L5x'))

        for mode in ['ISOLATED', 'CROSS']:
            # Normal
            m_norm, _ = run_backtest(strat, df_tf, mode, mg_pct, lev, 0.0, False)
            # Slippage
            m_slip, _ = run_backtest(strat, df_tf, mode, mg_pct, lev, 0.50, False)

            print(f"  {strat['id']:<8} {mode:<10} {sz_label:<12} "
                  f"${m_norm['final']:>8,.0f} ${m_slip['final']:>8,.0f} "
                  f"{m_norm['return_pct']:>+7.1f}% {m_slip['return_pct']:>+7.1f}% "
                  f"{m_norm['mdd_pct']:>6.1f}% {m_slip['mdd_pct']:>6.1f}% "
                  f"{m_norm['liquidation_events']:>7} {m_slip['liquidation_events']:>7}", flush=True)

            slippage_rows.append({
                'strategy_id': strat['id'],
                'margin_mode': mode,
                'sizing': sz_label,
                'normal_final': m_norm['final'],
                'slip_final': m_slip['final'],
                'normal_ret': m_norm['return_pct'],
                'slip_ret': m_slip['return_pct'],
                'normal_mdd': m_norm['mdd_pct'],
                'slip_mdd': m_slip['mdd_pct'],
                'normal_liq': m_norm['liquidation_events'],
                'slip_liq': m_slip['liquidation_events'],
                'degradation_pct': round(
                    (m_slip['final'] - m_norm['final']) / max(m_norm['final'], 1) * 100, 2
                ),
            })

    # ================================================================
    # PHASE 3: COMPARISON TABLE (CROSS vs ISOLATED side-by-side)
    # ================================================================
    print(f"\n\n{'#' * 100}", flush=True)
    print(f"  PHASE 3: CROSS vs ISOLATED COMPARISON TABLE", flush=True)
    print(f"{'#' * 100}\n", flush=True)

    # For each strategy+sizing, show CROSS vs ISOLATED side by side
    print(f"  {'Strat':<6} {'Sizing':<12} | {'--- ISOLATED ---':^40} | {'--- CROSS ---':^40} | {'Diff':>6}", flush=True)
    print(f"  {'':6} {'':12} | {'Final':>9} {'Ret%':>8} {'PF':>5} {'MDD%':>6} {'Liq':>4} | "
          f"{'Final':>9} {'Ret%':>8} {'PF':>5} {'MDD%':>6} {'Liq':>4} | {'Ret%':>6}", flush=True)
    print(f"  {'-' * 120}", flush=True)

    for strat in STRATEGIES:
        for sizing in SIZING_COMBOS:
            iso_row = None
            cross_row = None
            for r in all_rows:
                if (r['strategy_id'] == strat['id'] and
                        r['sizing_label'] == sizing['label']):
                    if r['margin_mode'] == 'ISOLATED':
                        iso_row = r
                    else:
                        cross_row = r

            if iso_row and cross_row:
                diff = cross_row['return_pct'] - iso_row['return_pct']
                iso_pf = f"{iso_row['pf']:.1f}" if iso_row['pf'] < 999 else "INF"
                cross_pf = f"{cross_row['pf']:.1f}" if cross_row['pf'] < 999 else "INF"
                print(f"  {strat['id']:<6} {sizing['label']:<12} | "
                      f"${iso_row['final_balance']:>7,.0f} {iso_row['return_pct']:>+7.1f}% "
                      f"{iso_pf:>4} {iso_row['mdd_pct']:>5.1f} {iso_row['liquidation_events']:>4} | "
                      f"${cross_row['final_balance']:>7,.0f} {cross_row['return_pct']:>+7.1f}% "
                      f"{cross_pf:>4} {cross_row['mdd_pct']:>5.1f} {cross_row['liquidation_events']:>4} | "
                      f"{diff:>+5.1f}%", flush=True)

    # ================================================================
    # DETERMINE WINNER
    # ================================================================
    print(f"\n\n{'#' * 100}", flush=True)
    print(f"  DETERMINING WINNER", flush=True)
    print(f"{'#' * 100}\n", flush=True)

    # Score: return% * 0.4 + PF * 10 + (100 - MDD%) * 0.3 - liquidations * 50
    for r in all_rows:
        pf_val = min(r['pf'], 50)  # cap PF at 50 for scoring
        r['score'] = (r['return_pct'] * 0.4 +
                      pf_val * 10 +
                      (100 - r['mdd_pct']) * 0.3 -
                      r['liquidation_events'] * 50)

    sorted_rows = sorted(all_rows, key=lambda x: x['score'], reverse=True)

    print(f"  TOP 10 BY COMPOSITE SCORE:", flush=True)
    print(f"  {'Rank':>4} {'Strat':<6} {'Mode':<10} {'Sizing':<12} "
          f"{'Final':>10} {'Ret%':>8} {'PF':>6} {'MDD%':>6} {'Trd':>4} "
          f"{'WR%':>5} {'Liq':>3} {'Score':>7}", flush=True)
    print(f"  {'-' * 100}", flush=True)

    for i, r in enumerate(sorted_rows[:10], 1):
        pf_str = f"{r['pf']:.1f}" if r['pf'] < 999 else "INF"
        print(f"  {i:>4} {r['strategy_id']:<6} {r['margin_mode']:<10} {r['sizing_label']:<12} "
              f"${r['final_balance']:>8,.0f} {r['return_pct']:>+7.1f}% "
              f"{pf_str:>5} {r['mdd_pct']:>5.1f} {r['trades']:>4} "
              f"{r['win_rate']:>4.0f}% {r['liquidation_events']:>3} "
              f"{r['score']:>6.1f}", flush=True)

    winner = sorted_rows[0]

    print(f"\n  WINNER: {winner['strategy_id']} | {winner['margin_mode']} | "
          f"{winner['sizing_label']}", flush=True)
    print(f"  Final: ${winner['final_balance']:,.2f} | Return: {winner['return_pct']:+.1f}% | "
          f"PF: {winner['pf']:.2f} | MDD: {winner['mdd_pct']:.1f}%", flush=True)

    # Find matching strategy config
    winner_cfg = None
    for s in STRATEGIES:
        if s['id'] == winner['strategy_id']:
            winner_cfg = s
            break
    winner_margin = winner['margin_pct'] / 100
    winner_lev = winner['leverage']
    winner_mode = winner['margin_mode']

    # ================================================================
    # PHASE 4: WINNER DETAILED BACKTEST WITH TRADES
    # ================================================================
    print(f"\n\n{'#' * 100}", flush=True)
    print(f"  PHASE 4: WINNER DETAILED BACKTEST", flush=True)
    print(f"{'#' * 100}\n", flush=True)

    df_tf = data_cache[winner_cfg['tf']]
    w_metrics, w_trades = run_backtest(
        winner_cfg, df_tf,
        margin_mode=winner_mode,
        margin_pct=winner_margin,
        leverage=winner_lev,
        sl_slippage=0.0,
        return_trades=True
    )

    # Generate and print full report
    report_text = print_full_report(
        f"v23.4 WINNER - {winner['strategy_id']} {winner_mode} {winner['sizing_label']}",
        winner_cfg, w_metrics, w_trades,
        winner_mode, winner_margin, winner_lev
    )
    print(report_text, flush=True)

    # Generate monthly data
    w_monthly = generate_monthly_table(w_trades, df_tf.index)

    # Save files
    report_path = os.path.join(DATA_DIR, "v23_4_FINAL_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n  Saved: {report_path}", flush=True)

    if w_trades:
        trades_path = os.path.join(DATA_DIR, "v23_4_FINAL_trades.csv")
        pd.DataFrame(w_trades).to_csv(trades_path, index=False)
        print(f"  Saved: {trades_path}", flush=True)

    if w_monthly:
        monthly_path = os.path.join(DATA_DIR, "v23_4_FINAL_monthly.csv")
        pd.DataFrame(w_monthly).to_csv(monthly_path, index=False)
        print(f"  Saved: {monthly_path}", flush=True)

    # ================================================================
    # PHASE 5: 30-RUN VERIFICATION
    # ================================================================
    print(f"\n\n{'#' * 100}", flush=True)
    print(f"  PHASE 5: 30-RUN VERIFICATION", flush=True)
    print(f"{'#' * 100}\n", flush=True)

    verify_rows = []
    print(f"  {'Run':>4} {'Final':>10} {'Ret%':>8} {'PF':>6} {'MDD%':>6} "
          f"{'Trd':>4} {'WR%':>5} {'SL':>3} {'Liq':>3}", flush=True)
    print(f"  {'-' * 70}", flush=True)

    # 30 runs with different data windows
    total_bars = len(df_tf)
    min_bars_needed = max(1000, int(winner_cfg['slow_period'] * 3))

    for run_idx in range(30):
        # Vary data window: use different starting points
        # Run 0 = full data, runs 1-29 = progressively shifted windows
        if run_idx == 0:
            df_run = df_tf
        else:
            # Shift start by run_idx * 2% of data
            shift = int(total_bars * 0.02 * run_idx)
            if shift + min_bars_needed > total_bars:
                shift = max(0, total_bars - min_bars_needed)
            df_run = df_tf.iloc[shift:]

        m, _ = run_backtest(
            winner_cfg, df_run,
            margin_mode=winner_mode,
            margin_pct=winner_margin,
            leverage=winner_lev,
            sl_slippage=0.0,
            return_trades=False
        )

        pf_str = f"{m['pf']:.1f}" if m['pf'] < 999 else "INF"
        print(f"  {run_idx + 1:>4} ${m['final']:>8,.0f} {m['return_pct']:>+7.1f}% "
              f"{pf_str:>5} {m['mdd_pct']:>5.1f} {m['trades']:>4} "
              f"{m['win_rate']:>4.0f}% {m['sl_count']:>3} "
              f"{m['liquidation_events']:>3}", flush=True)

        verify_rows.append({
            'run': run_idx + 1,
            'start_bar': 0 if run_idx == 0 else int(total_bars * 0.02 * run_idx),
            'bars': len(df_run),
            'final_balance': m['final'],
            'return_pct': m['return_pct'],
            'pf': m['pf'],
            'mdd_pct': m['mdd_pct'],
            'trades': m['trades'],
            'wins': m['wins'],
            'losses': m['losses'],
            'win_rate': m['win_rate'],
            'sl_hits': m['sl_count'],
            'trail_hits': m['trail_count'],
            'liquidation_events': m['liquidation_events'],
        })

    # 30-run summary
    vdf = pd.DataFrame(verify_rows)
    print(f"\n  30-RUN SUMMARY:", flush=True)
    print(f"  {'Metric':<20} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'Std':>10}", flush=True)
    print(f"  {'-' * 75}", flush=True)

    for col, fmt in [('final_balance', ',.0f'), ('return_pct', '+.1f'),
                      ('pf', '.2f'), ('mdd_pct', '.1f'),
                      ('trades', '.0f'), ('win_rate', '.1f')]:
        vals = vdf[col]
        pf_vals = vals.clip(upper=999)
        print(f"  {col:<20} {pf_vals.mean():>10{fmt}} {pf_vals.median():>10{fmt}} "
              f"{pf_vals.min():>10{fmt}} {pf_vals.max():>10{fmt}} "
              f"{pf_vals.std():>10.2f}", flush=True)

    profitable_runs = len(vdf[vdf['return_pct'] > 0])
    print(f"\n  Profitable runs: {profitable_runs}/30 ({profitable_runs / 30 * 100:.0f}%)", flush=True)
    print(f"  Zero liquidation runs: {len(vdf[vdf['liquidation_events'] == 0])}/30", flush=True)

    # Save 30-run CSV
    verify_path = os.path.join(DATA_DIR, "v23_4_FINAL_30run.csv")
    vdf.to_csv(verify_path, index=False)
    print(f"\n  Saved: {verify_path}", flush=True)

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    total_time = time.time() - T0
    print(f"\n\n{'=' * 100}", flush=True)
    print(f"  v23.4 PIPELINE COMPLETE", flush=True)
    print(f"{'=' * 100}", flush=True)
    print(f"  Total time:       {total_time:.1f}s ({total_time / 60:.1f}m)", flush=True)
    print(f"  Backtests run:    {total_tests} + {len(slippage_rows)} slippage + 30 verification", flush=True)
    print(f"", flush=True)
    print(f"  WINNER:", flush=True)
    print(f"    Strategy:    {winner_cfg['name']}", flush=True)
    print(f"    Source:      {winner_cfg['source']}", flush=True)
    print(f"    Margin Mode: {winner_mode}", flush=True)
    print(f"    Sizing:      M{winner_margin * 100:.0f}% Lev{winner_lev}x", flush=True)
    print(f"    Final:       ${w_metrics['final']:,.2f}", flush=True)
    print(f"    Return:      {w_metrics['return_pct']:+.1f}%", flush=True)
    print(f"    PF:          {w_metrics['pf']:.2f}", flush=True)
    print(f"    MDD:         {w_metrics['mdd_pct']:.1f}%", flush=True)
    print(f"    Trades:      {w_metrics['trades']}", flush=True)
    print(f"    Win Rate:    {w_metrics['win_rate']:.1f}%", flush=True)
    print(f"    SL Hits:     {w_metrics['sl_count']}", flush=True)
    print(f"    Liquidations:{w_metrics['liquidation_events']}", flush=True)
    print(f"", flush=True)
    print(f"  KEY FINDINGS:", flush=True)

    # Analyze cross vs isolated difference
    iso_returns = [r['return_pct'] for r in all_rows if r['margin_mode'] == 'ISOLATED']
    cross_returns = [r['return_pct'] for r in all_rows if r['margin_mode'] == 'CROSS']
    iso_liq = sum(r['liquidation_events'] for r in all_rows if r['margin_mode'] == 'ISOLATED')
    cross_liq = sum(r['liquidation_events'] for r in all_rows if r['margin_mode'] == 'CROSS')

    print(f"    ISOLATED avg return: {np.mean(iso_returns):+.1f}% | Total liquidations: {iso_liq}", flush=True)
    print(f"    CROSS avg return:    {np.mean(cross_returns):+.1f}% | Total liquidations: {cross_liq}", flush=True)

    # Slippage analysis
    iso_slip = [r for r in slippage_rows if r['margin_mode'] == 'ISOLATED']
    cross_slip = [r for r in slippage_rows if r['margin_mode'] == 'CROSS']
    if iso_slip and cross_slip:
        iso_degradation = np.mean([r['degradation_pct'] for r in iso_slip])
        cross_degradation = np.mean([r['degradation_pct'] for r in cross_slip])
        print(f"    Slippage degradation (ISOLATED): {iso_degradation:+.1f}%", flush=True)
        print(f"    Slippage degradation (CROSS):    {cross_degradation:+.1f}%", flush=True)

    print(f"", flush=True)
    print(f"  OUTPUT FILES:", flush=True)
    print(f"    {csv_path}", flush=True)
    print(f"    {report_path}", flush=True)
    if w_trades:
        print(f"    {os.path.join(DATA_DIR, 'v23_4_FINAL_trades.csv')}", flush=True)
    if w_monthly:
        print(f"    {os.path.join(DATA_DIR, 'v23_4_FINAL_monthly.csv')}", flush=True)
    print(f"    {verify_path}", flush=True)
    print(f"\n{'=' * 100}", flush=True)


if __name__ == "__main__":
    main()
