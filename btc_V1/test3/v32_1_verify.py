"""
v32.1 Backtest + Verification
Plan A: EMA(100)/EMA(600), 30m
Plan B: EMA(75)/SMA(750), 30m

Common: ADX(20)>=30 rising, RSI(14) 35-75, EMA gap>=0.2%,
        monitor window 24 bars, daily loss -20%,
        Cross margin, Leverage 10x, Margin 35%
        SL -3% (intrabar H/L), TSL +12% act / -10% trail (close), REV exit
"""

import numpy as np
import pandas as pd
import time
import os
import sys
import io
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

DATA_DIR = r"D:\filesystem\futures\btc_V1\test3"
CSV_FILES = [
    os.path.join(DATA_DIR, f"btc_usdt_5m_2020_to_now_part{i}.csv")
    for i in range(1, 4)
]
INITIAL_CAPITAL = 5000.0
FEE_RATE = 0.0004  # 0.04% per side

PLANS = {
    'A': {
        'name': 'Plan A: EMA(100)/EMA(600) 30m',
        'fast_type': 'EMA', 'fast_period': 100,
        'slow_type': 'EMA', 'slow_period': 600,
    },
    'B': {
        'name': 'Plan B: EMA(75)/SMA(750) 30m',
        'fast_type': 'EMA', 'fast_period': 75,
        'slow_type': 'SMA', 'slow_period': 750,
    },
}

# Common parameters
ADX_PERIOD = 20
ADX_MIN = 30
ADX_RISING_LOOKBACK = 6
RSI_LO = 35
RSI_HI = 75
EMA_GAP_MIN = 0.002  # 0.2%
MONITOR_WINDOW = 24   # bars after cross
DAILY_LOSS_LIMIT = -0.20
SL_PCT = 0.03         # -3%
TSL_ACTIVATE = 0.12   # +12% ROI
TSL_TRAIL = 0.10      # -10% from peak
MARGIN_PCT = 0.35
LEVERAGE = 10


# ─── Indicator functions ───

def wilder(arr, p):
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


def ema_np(c, p):
    out = np.full(len(c), np.nan)
    s = 0
    while s < len(c) and np.isnan(c[s]):
        s += 1
    if s >= len(c):
        return out
    out[s] = c[s]
    m = 2.0 / (p + 1)
    for i in range(s + 1, len(c)):
        if not np.isnan(c[i]) and not np.isnan(out[i - 1]):
            out[i] = c[i] * m + out[i - 1] * (1 - m)
    return out


def sma_np(c, p):
    out = np.full(len(c), np.nan)
    cs = np.nancumsum(c)
    for i in range(p - 1, len(c)):
        if i >= p:
            out[i] = (cs[i] - cs[i - p]) / p
        else:
            out[i] = cs[i] / (i + 1)
    return out


def calc_adx_wilder(high, low, close, period=20):
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
    atr = wilder(tr, period)
    spdm = wilder(pdm, period)
    smdm = wilder(mdm, period)
    pdi = np.full(n, np.nan)
    mdi = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr[i]) and atr[i] > 0:
            pdi[i] = 100 * spdm[i] / atr[i]
            mdi[i] = 100 * smdm[i] / atr[i]
            s_ = pdi[i] + mdi[i]
            dx[i] = 100 * abs(pdi[i] - mdi[i]) / s_ if s_ > 0 else 0
    return wilder(dx, period)


def calc_rsi(close, period=14):
    n = len(close)
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        d = close[i] - close[i - 1]
        if d > 0:
            gains[i] = d
        else:
            losses[i] = -d
    avg_g = wilder(gains, period)
    avg_l = wilder(losses, period)
    out = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(avg_g[i]) and not np.isnan(avg_l[i]):
            if avg_l[i] == 0:
                out[i] = 100.0
            else:
                out[i] = 100 - 100 / (1 + avg_g[i] / avg_l[i])
    return out


# ─── Data loading ───

def load_data():
    frames = []
    for f in CSV_FILES:
        df = pd.read_csv(f, parse_dates=['timestamp'])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df.set_index('timestamp', inplace=True)
    # Resample 5m -> 30m
    df30 = df.resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['open'])
    df30.reset_index(inplace=True)
    return df30


# ─── Backtest engine ───

def run_backtest(df30, plan_key, record_trades=True):
    """
    Returns dict with metrics + optional trade list.
    Fully deterministic -- no randomness.
    """
    cfg = PLANS[plan_key]
    closes = df30['close'].values.astype(np.float64)
    highs = df30['high'].values.astype(np.float64)
    lows = df30['low'].values.astype(np.float64)
    opens = df30['open'].values.astype(np.float64)
    timestamps = df30['timestamp'].values
    n = len(closes)

    # Compute indicators
    fast_ma = ema_np(closes, cfg['fast_period'])
    if cfg['slow_type'] == 'EMA':
        slow_ma = ema_np(closes, cfg['slow_period'])
    else:
        slow_ma = sma_np(closes, cfg['slow_period'])

    adx = calc_adx_wilder(highs, lows, closes, ADX_PERIOD)
    rsi = calc_rsi(closes, 14)

    # State
    bal = INITIAL_CAPITAL
    peak_bal = INITIAL_CAPITAL
    pos_dir = 0        # 0=flat, 1=long, -1=short
    pos_entry = 0.0
    pos_size = 0.0     # number of contracts (BTC qty)
    pos_margin = 0.0
    pos_highest = 0.0
    pos_lowest = 999999999.0
    trail_active = False
    trail_sl = 0.0
    pos_sl = 0.0
    pos_bar_idx = 0    # bar index of entry
    pos_alignment = 0  # MA alignment at entry time (+1 or -1)

    # Cross tracking
    last_cross_bar = -9999
    last_cross_dir = 0  # +1 golden, -1 death

    # Daily loss tracking
    day_start_bal = INITIAL_CAPITAL
    current_day = None

    # Stats
    trades_list = []
    n_trades = 0
    n_wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    sl_count = 0
    tsl_count = 0
    rev_count = 0
    max_consec_loss = 0
    cur_consec_loss = 0
    mdd = 0.0
    mdd_peak = INITIAL_CAPITAL
    liq_events = 0
    daily_locked = False
    daily_locked_date = None

    # Balance curve for MDD
    bal_curve = []

    warmup = max(800, int(cfg['slow_period'] * 1.5))

    def close_position(exit_price, exit_bar, exit_reason):
        nonlocal bal, pos_dir, pos_entry, pos_size, pos_margin
        nonlocal pos_highest, pos_lowest, trail_active, trail_sl, pos_sl
        nonlocal n_trades, n_wins, gross_profit, gross_loss
        nonlocal sl_count, tsl_count, rev_count
        nonlocal max_consec_loss, cur_consec_loss
        nonlocal mdd, mdd_peak, liq_events, peak_bal

        if pos_dir == 1:
            rpnl = pos_size * (exit_price - pos_entry)
        else:
            rpnl = pos_size * (pos_entry - exit_price)

        fee_exit = pos_size * exit_price * FEE_RATE
        pnl = rpnl - fee_exit

        # Cross margin: loss limited to full balance
        if pnl < -bal:
            pnl = -bal
            liq_events += 1

        roi_pct = rpnl / (pos_margin) * 100 if pos_margin > 0 else 0

        bal += pnl
        n_trades += 1

        if exit_reason == 'SL':
            sl_count += 1
        elif exit_reason == 'TSL':
            tsl_count += 1
        elif exit_reason == 'REV':
            rev_count += 1

        if pnl > 0:
            n_wins += 1
            gross_profit += pnl
            cur_consec_loss = 0
        else:
            gross_loss += abs(pnl)
            cur_consec_loss += 1
            max_consec_loss = max(max_consec_loss, cur_consec_loss)

        peak_bal = max(peak_bal, bal)
        mdd_peak = max(mdd_peak, bal)
        if mdd_peak > 0:
            mdd = max(mdd, (mdd_peak - bal) / mdd_peak)

        if record_trades:
            trades_list.append({
                'entry_time': str(timestamps[pos_bar_idx])[:19],
                'exit_time': str(timestamps[exit_bar])[:19],
                'direction': 'LONG' if pos_dir == 1 else 'SHORT',
                'entry_price': round(pos_entry, 2),
                'exit_price': round(exit_price, 2),
                'size_usd': round(pos_size * pos_entry, 2),
                'pnl': round(pnl, 2),
                'roi_pct': round(roi_pct, 2),
                'balance': round(bal, 2),
                'exit_reason': exit_reason,
                'hold_bars': exit_bar - pos_bar_idx,
            })

        pos_dir = 0
        pos_entry = 0.0
        pos_size = 0.0
        pos_margin = 0.0
        pos_highest = 0.0
        pos_lowest = 999999999.0
        trail_active = False
        trail_sl = 0.0
        pos_sl = 0.0

    for i in range(warmup, n):
        if bal <= 0:
            break

        c = closes[i]
        h = highs[i]
        l = lows[i]

        # Daily loss limit tracking
        ts = pd.Timestamp(timestamps[i])
        day_str = str(ts.date())
        if current_day is None or day_str != current_day:
            current_day = day_str
            day_start_bal = bal
            daily_locked = False
            daily_locked_date = None

        if not daily_locked and day_start_bal > 0:
            day_loss = (bal - day_start_bal) / day_start_bal
            if day_loss <= DAILY_LOSS_LIMIT:
                daily_locked = True
                daily_locked_date = day_str

        # Track balance curve
        if record_trades and i % 10 == 0:
            bal_curve.append({'bar': i, 'timestamp': str(timestamps[i])[:19], 'balance': round(bal, 2)})

        # ─── Position management ───
        if pos_dir != 0:
            exited = False

            # Priority 1: SL on intrabar high/low
            if pos_dir == 1:
                pos_highest = max(pos_highest, h)
                if l <= pos_sl:
                    close_position(pos_sl, i, 'SL')
                    exited = True
            elif pos_dir == -1:
                pos_lowest = min(pos_lowest, l)
                if h >= pos_sl:
                    close_position(pos_sl, i, 'SL')
                    exited = True

            # Priority 2: TSL on close
            if not exited and pos_dir != 0:
                if pos_dir == 1:
                    cur_roi = (c - pos_entry) / pos_entry
                    if cur_roi >= TSL_ACTIVATE:
                        trail_active = True
                    if trail_active:
                        # Trail from highest close
                        # Use pos_highest for trail reference (on close we compare)
                        new_tsl = pos_highest * (1 - TSL_TRAIL)
                        if new_tsl > trail_sl:
                            trail_sl = new_tsl
                        if c <= trail_sl:
                            close_position(c, i, 'TSL')
                            exited = True
                elif pos_dir == -1:
                    cur_roi = (pos_entry - c) / pos_entry
                    if cur_roi >= TSL_ACTIVATE:
                        trail_active = True
                    if trail_active:
                        new_tsl = pos_lowest * (1 + TSL_TRAIL)
                        if new_tsl < trail_sl:
                            trail_sl = new_tsl
                        if c >= trail_sl:
                            close_position(c, i, 'TSL')
                            exited = True

            # Priority 3: REV exit -- MA alignment reversal
            if not exited and pos_dir != 0:
                if not np.isnan(fast_ma[i]) and not np.isnan(slow_ma[i]):
                    if pos_dir == 1 and fast_ma[i] < slow_ma[i]:
                        # Was long, now fast below slow -> reversal
                        close_position(c, i, 'REV')
                        exited = True
                    elif pos_dir == -1 and fast_ma[i] > slow_ma[i]:
                        # Was short, now fast above slow -> reversal
                        close_position(c, i, 'REV')
                        exited = True

            if exited:
                if bal <= 0:
                    break
                continue

        # ─── Signal detection ───
        if i < warmup + 1:
            continue
        if np.isnan(fast_ma[i]) or np.isnan(fast_ma[i - 1]):
            continue
        if np.isnan(slow_ma[i]) or np.isnan(slow_ma[i - 1]):
            continue

        # Detect crosses
        cross_up = (fast_ma[i - 1] <= slow_ma[i - 1]) and (fast_ma[i] > slow_ma[i])
        cross_down = (fast_ma[i - 1] >= slow_ma[i - 1]) and (fast_ma[i] < slow_ma[i])

        if cross_up:
            last_cross_bar = i
            last_cross_dir = 1
        elif cross_down:
            last_cross_bar = i
            last_cross_dir = -1

        # Already in position -> skip entry
        if pos_dir != 0:
            continue

        # Daily locked -> skip new entries
        if daily_locked:
            continue

        # Check monitor window
        bars_since_cross = i - last_cross_bar
        if bars_since_cross > MONITOR_WINDOW or last_cross_dir == 0:
            continue

        sig = last_cross_dir

        # Same-direction skip: don't re-enter same direction after same cross
        # (handled by cross tracking -- once cross is consumed by entry or expires, done)

        # Filter: ADX >= 30
        if np.isnan(adx[i]) or adx[i] < ADX_MIN:
            continue

        # Filter: ADX rising -- adx[i] > adx[i-6]
        if i < ADX_RISING_LOOKBACK or np.isnan(adx[i - ADX_RISING_LOOKBACK]):
            continue
        if adx[i] <= adx[i - ADX_RISING_LOOKBACK]:
            continue

        # Filter: RSI 35-75
        if np.isnan(rsi[i]) or rsi[i] < RSI_LO or rsi[i] > RSI_HI:
            continue

        # Filter: EMA gap >= 0.2%
        if slow_ma[i] == 0:
            continue
        ema_gap = abs(fast_ma[i] - slow_ma[i]) / slow_ma[i]
        if ema_gap < EMA_GAP_MIN:
            continue

        # ─── ENTER ───
        entry_price = c
        margin = bal * MARGIN_PCT
        notional = margin * LEVERAGE  # 3.5x balance exposure
        size = notional / entry_price
        fee_entry = size * entry_price * FEE_RATE

        if margin <= 0 or size <= 0:
            continue

        bal -= fee_entry  # entry fee
        pos_dir = sig
        pos_entry = entry_price
        pos_size = size
        pos_margin = margin
        pos_bar_idx = i
        pos_highest = h
        pos_lowest = l
        trail_active = False
        trail_sl = 999999999.0 if sig == -1 else 0.0
        pos_alignment = sig

        if sig == 1:
            pos_sl = entry_price * (1 - SL_PCT)
        else:
            pos_sl = entry_price * (1 + SL_PCT)

    # Close any open position at end
    if pos_dir != 0:
        close_position(closes[-1], n - 1, 'END')

    # Compute metrics
    total_return = (bal - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    years = max((n - warmup) * 30 / 525600, 0.01)  # 30m bars
    cagr = ((bal / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if bal > 0 else -100
    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    avg_pnl = (bal - INITIAL_CAPITAL) / n_trades if n_trades > 0 else 0

    metrics = {
        'plan': plan_key,
        'plan_name': cfg['name'],
        'final_balance': round(bal, 2),
        'total_return_pct': round(total_return, 2),
        'cagr_pct': round(cagr, 2),
        'total_trades': n_trades,
        'win_rate_pct': round(win_rate, 2),
        'profit_factor': round(profit_factor, 4),
        'max_drawdown_pct': round(mdd * 100, 2),
        'max_consec_loss': max_consec_loss,
        'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2),
        'avg_pnl': round(avg_pnl, 2),
        'sl_exits': sl_count,
        'tsl_exits': tsl_count,
        'rev_exits': rev_count,
        'liquidations': liq_events,
    }

    return metrics, trades_list, bal_curve


# ─── Monthly / Yearly tables ───

def compute_monthly_yearly(trades_list, plan_key):
    if not trades_list:
        return pd.DataFrame(), pd.DataFrame()

    tdf = pd.DataFrame(trades_list)
    tdf['exit_dt'] = pd.to_datetime(tdf['exit_time'])
    tdf['month'] = tdf['exit_dt'].dt.to_period('M')
    tdf['year'] = tdf['exit_dt'].dt.year

    monthly = tdf.groupby('month').agg(
        trades=('pnl', 'count'),
        gross_pnl=('pnl', 'sum'),
        wins=('pnl', lambda x: (x > 0).sum()),
        best=('pnl', 'max'),
        worst=('pnl', 'min'),
    ).reset_index()
    monthly['win_rate'] = (monthly['wins'] / monthly['trades'] * 100).round(1)
    monthly['plan'] = plan_key
    monthly['month'] = monthly['month'].astype(str)

    yearly = tdf.groupby('year').agg(
        trades=('pnl', 'count'),
        gross_pnl=('pnl', 'sum'),
        wins=('pnl', lambda x: (x > 0).sum()),
        best=('pnl', 'max'),
        worst=('pnl', 'min'),
    ).reset_index()
    yearly['win_rate'] = (yearly['wins'] / yearly['trades'] * 100).round(1)
    yearly['plan'] = plan_key

    return monthly, yearly


# ─── Report formatting ───

def format_report(metrics_a, trades_a, metrics_b, trades_b, monthly_a, yearly_a, monthly_b, yearly_b):
    lines = []
    sep = "=" * 90

    lines.append(sep)
    lines.append("  v32.1 BACKTEST REPORT")
    lines.append("  Strategy: EMA Cross + ADX(20)>=30 rising + RSI(14) 35-75 + EMA Gap 0.2%")
    lines.append("  Cross Margin | Leverage 10x | Margin 35% | SL -3% | TSL +12%/-10% | REV exit")
    lines.append("  Monitor Window: 24 bars (12h) | Daily Loss Limit: -20%")
    lines.append("  Data: BTC/USDT 5m -> 30m resample | Initial: $5,000")
    lines.append(sep)
    lines.append("")

    for label, m, trades, monthly, yearly in [
        ("PLAN A: EMA(100)/EMA(600)", metrics_a, trades_a, monthly_a, yearly_a),
        ("PLAN B: EMA(75)/SMA(750)", metrics_b, trades_b, monthly_b, yearly_b),
    ]:
        lines.append(f"  >>> {label}")
        lines.append("-" * 90)
        lines.append(f"  Final Balance    : ${m['final_balance']:>12,.2f}")
        lines.append(f"  Total Return     : {m['total_return_pct']:>10.2f}%")
        lines.append(f"  CAGR             : {m['cagr_pct']:>10.2f}%")
        lines.append(f"  Total Trades     : {m['total_trades']:>10d}")
        lines.append(f"  Win Rate         : {m['win_rate_pct']:>10.2f}%")
        lines.append(f"  Profit Factor    : {m['profit_factor']:>10.4f}")
        lines.append(f"  Max Drawdown     : {m['max_drawdown_pct']:>10.2f}%")
        lines.append(f"  Max Consec Loss  : {m['max_consec_loss']:>10d}")
        lines.append(f"  Gross Profit     : ${m['gross_profit']:>12,.2f}")
        lines.append(f"  Gross Loss       : ${m['gross_loss']:>12,.2f}")
        lines.append(f"  Avg PnL/Trade    : ${m['avg_pnl']:>12,.2f}")
        lines.append(f"  SL Exits         : {m['sl_exits']:>10d}")
        lines.append(f"  TSL Exits        : {m['tsl_exits']:>10d}")
        lines.append(f"  REV Exits        : {m['rev_exits']:>10d}")
        lines.append(f"  Liquidations     : {m['liquidations']:>10d}")
        lines.append("")

        # Exit analysis
        total_ex = m['sl_exits'] + m['tsl_exits'] + m['rev_exits']
        if total_ex > 0:
            lines.append("  Exit Analysis:")
            lines.append(f"    SL  : {m['sl_exits']:>5d}  ({m['sl_exits']/total_ex*100:5.1f}%)")
            lines.append(f"    TSL : {m['tsl_exits']:>5d}  ({m['tsl_exits']/total_ex*100:5.1f}%)")
            lines.append(f"    REV : {m['rev_exits']:>5d}  ({m['rev_exits']/total_ex*100:5.1f}%)")
            lines.append("")

        # Yearly table
        if not yearly.empty:
            lines.append("  Yearly Performance:")
            lines.append(f"  {'Year':<6} {'Trades':>7} {'PnL':>12} {'WinRate':>8} {'Best':>10} {'Worst':>10}")
            lines.append("  " + "-" * 55)
            for _, row in yearly.iterrows():
                lines.append(f"  {int(row['year']):<6} {int(row['trades']):>7} ${row['gross_pnl']:>10,.2f} {row['win_rate']:>7.1f}% ${row['best']:>8,.2f} ${row['worst']:>8,.2f}")
            lines.append("")

        # Monthly table
        if not monthly.empty:
            lines.append("  Monthly Performance:")
            lines.append(f"  {'Month':<8} {'Trades':>7} {'PnL':>12} {'WinRate':>8} {'Best':>10} {'Worst':>10}")
            lines.append("  " + "-" * 57)
            for _, row in monthly.iterrows():
                lines.append(f"  {row['month']:<8} {int(row['trades']):>7} ${row['gross_pnl']:>10,.2f} {row['win_rate']:>7.1f}% ${row['best']:>8,.2f} ${row['worst']:>8,.2f}")
            lines.append("")

        # Trade list (first 30 + last 10)
        if trades:
            lines.append(f"  Trade List (total {len(trades)}):")
            lines.append(f"  {'#':>4} {'Entry Time':<20} {'Exit Time':<20} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'PnL':>10} {'ROI%':>8} {'Exit':>5} {'Bars':>5}")
            lines.append("  " + "-" * 110)
            show_trades = trades[:30]
            for idx, t in enumerate(show_trades):
                lines.append(f"  {idx+1:>4} {t['entry_time']:<20} {t['exit_time']:<20} {t['direction']:<6} {t['entry_price']:>10.2f} {t['exit_price']:>10.2f} ${t['pnl']:>9.2f} {t['roi_pct']:>7.2f}% {t['exit_reason']:>5} {t['hold_bars']:>5}")
            if len(trades) > 40:
                lines.append(f"  {'...':>4} {'...':^20} {'...':^20}")
                for idx, t in enumerate(trades[-10:]):
                    real_idx = len(trades) - 10 + idx + 1
                    lines.append(f"  {real_idx:>4} {t['entry_time']:<20} {t['exit_time']:<20} {t['direction']:<6} {t['entry_price']:>10.2f} {t['exit_price']:>10.2f} ${t['pnl']:>9.2f} {t['roi_pct']:>7.2f}% {t['exit_reason']:>5} {t['hold_bars']:>5}")
            elif len(trades) > 30:
                for idx, t in enumerate(trades[30:]):
                    real_idx = 30 + idx + 1
                    lines.append(f"  {real_idx:>4} {t['entry_time']:<20} {t['exit_time']:<20} {t['direction']:<6} {t['entry_price']:>10.2f} {t['exit_price']:>10.2f} ${t['pnl']:>9.2f} {t['roi_pct']:>7.2f}% {t['exit_reason']:>5} {t['hold_bars']:>5}")

        lines.append("")
        lines.append(sep)
        lines.append("")

    return "\n".join(lines)


# ─── Multi-run verification ───

def run_multi(df30, plan_key, num_runs=10):
    results = []
    for r in range(num_runs):
        m, _, _ = run_backtest(df30, plan_key, record_trades=False)
        results.append(m)
    return results


# ─── Main ───

def main():
    t0 = time.time()

    print("=" * 70)
    print("  v32.1 Backtest + 30-Run Verification")
    print("=" * 70)
    print()

    # Load data
    print("[1/6] Loading and resampling data (5m -> 30m)...")
    df30 = load_data()
    print(f"       30m bars: {len(df30):,}")
    print(f"       Range: {df30['timestamp'].iloc[0]} ~ {df30['timestamp'].iloc[-1]}")
    print()

    all_run_results = []   # for 30run CSV
    all_monthly = []       # for monthly CSV

    for plan_key in ['A', 'B']:
        cfg = PLANS[plan_key]
        print(f"[{2 if plan_key=='A' else 3}/6] Running Plan {plan_key}: {cfg['name']}...")

        # Main backtest with trades
        t1 = time.time()
        metrics, trades, bal_curve = run_backtest(df30, plan_key, record_trades=True)
        bt_time = time.time() - t1
        print(f"       Backtest done in {bt_time:.1f}s | Trades: {metrics['total_trades']} | Final: ${metrics['final_balance']:,.2f}")

        # Monthly/Yearly
        monthly, yearly = compute_monthly_yearly(trades, plan_key)

        # 10 backtest runs
        print(f"       Running 10 backtest runs...")
        bt_results = run_multi(df30, plan_key, 10)
        bt_finals = [r['final_balance'] for r in bt_results]
        bt_std = np.std(bt_finals)
        print(f"       BT 10 runs: mean=${np.mean(bt_finals):,.2f} std=${bt_std:.6f}")

        # 10 verification runs
        print(f"       Running 10 verification runs...")
        vr_results = run_multi(df30, plan_key, 10)
        vr_finals = [r['final_balance'] for r in vr_results]
        vr_std = np.std(vr_finals)
        print(f"       VR 10 runs: mean=${np.mean(vr_finals):,.2f} std=${vr_std:.6f}")

        if bt_std == 0 and vr_std == 0:
            print(f"       DETERMINISTIC CONFIRMED (std=0)")
        else:
            print(f"       WARNING: Non-zero std detected!")

        # Collect run results
        for idx, r in enumerate(bt_results):
            row = {'plan': plan_key, 'run_type': 'backtest', 'run_num': idx + 1}
            row.update(r)
            all_run_results.append(row)
        for idx, r in enumerate(vr_results):
            row = {'plan': plan_key, 'run_type': 'verify', 'run_num': idx + 1}
            row.update(r)
            all_run_results.append(row)

        if not monthly.empty:
            all_monthly.append(monthly)

        # Save trades CSV
        if trades:
            tdf = pd.DataFrame(trades)
            tpath = os.path.join(DATA_DIR, f"v32_1_trades_{plan_key}.csv")
            tdf.to_csv(tpath, index=False, encoding='utf-8-sig')
            print(f"       Saved: {tpath}")

        # Store for report
        if plan_key == 'A':
            metrics_a, trades_a, monthly_a, yearly_a = metrics, trades, monthly, yearly
        else:
            metrics_b, trades_b, monthly_b, yearly_b = metrics, trades, monthly, yearly

        print()

    # Generate report
    print("[4/6] Generating report...")
    report_text = format_report(metrics_a, trades_a, metrics_b, trades_b,
                                 monthly_a, yearly_a, monthly_b, yearly_b)

    # Add verification summary to report
    report_text += "\n"
    report_text += "=" * 90 + "\n"
    report_text += "  30-RUN VERIFICATION SUMMARY\n"
    report_text += "=" * 90 + "\n"
    for plan_key in ['A', 'B']:
        plan_runs = [r for r in all_run_results if r['plan'] == plan_key]
        bt_runs = [r for r in plan_runs if r['run_type'] == 'backtest']
        vr_runs = [r for r in plan_runs if r['run_type'] == 'verify']
        bt_f = [r['final_balance'] for r in bt_runs]
        vr_f = [r['final_balance'] for r in vr_runs]
        report_text += f"\n  Plan {plan_key}: {PLANS[plan_key]['name']}\n"
        report_text += f"    Backtest  10 runs: mean=${np.mean(bt_f):>10,.2f}  std=${np.std(bt_f):.6f}  min=${min(bt_f):>10,.2f}  max=${max(bt_f):>10,.2f}\n"
        report_text += f"    Verify    10 runs: mean=${np.mean(vr_f):>10,.2f}  std=${np.std(vr_f):.6f}  min=${min(vr_f):>10,.2f}  max=${max(vr_f):>10,.2f}\n"
        report_text += f"    ALL 20 runs: DETERMINISTIC={'YES' if np.std(bt_f)==0 and np.std(vr_f)==0 else 'NO'}\n"

    report_text += "\n" + "=" * 90 + "\n"

    print(report_text)

    # Save files
    print("[5/6] Saving output files...")

    # Report
    rpath = os.path.join(DATA_DIR, "v32_1_report.txt")
    with open(rpath, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"       Saved: {rpath}")

    # 30-run CSV
    run_df = pd.DataFrame(all_run_results)
    run_path = os.path.join(DATA_DIR, "v32_1_30run.csv")
    run_df.to_csv(run_path, index=False, encoding='utf-8-sig')
    print(f"       Saved: {run_path}")

    # Monthly CSV
    if all_monthly:
        mdf = pd.concat(all_monthly, ignore_index=True)
        mpath = os.path.join(DATA_DIR, "v32_1_monthly.csv")
        mdf.to_csv(mpath, index=False, encoding='utf-8-sig')
        print(f"       Saved: {mpath}")

    elapsed = time.time() - t0
    print(f"\n[6/6] Complete in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
