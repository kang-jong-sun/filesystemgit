"""
v32.1 FINAL Backtest + Verification
Plan A: EMA(100)/EMA(600), 30m
Plan B: EMA(75)/SMA(750), 30m

CRITICAL DIFFERENCES from prior v32.1:
- ADX uses pandas ewm(alpha=1/20, adjust=False) NOT manual Wilder smoothing
- RSI uses ewm(alpha=1/14, adjust=False) NOT Wilder
- EMA uses ewm(span=period, adjust=False) -- standard EMA
- TSL width is 9% (not 10%)
- SL check ONLY when TSL is NOT active
- TA activation checks on HIGH/LOW (not close)
- TSL check on CLOSE
- Warmup: 600 bars (fixed)
- REV does NOT continue -- same bar can have REV exit + new entry
- Daily loss reset every 1440 bars (i % 1440 == 0 when i > 600)
"""

import numpy as np
import pandas as pd
import time
import os
import sys
import io
import warnings

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

DATA_DIR = r"D:\filesystem\futures\btc_V1\test3"
CSV_FILES = [
    os.path.join(DATA_DIR, f"btc_usdt_5m_2020_to_now_part{i}.csv")
    for i in range(1, 4)
]

# ─── EXACT PARAMETERS ───
INITIAL_CAPITAL = 5000.0
FEE_RATE = 0.0004       # 0.04% per side
WARMUP = 600             # fixed 600 bars
MARGIN_PCT = 0.35        # 35% of balance
LEVERAGE = 10
SL_PCT = 0.03            # 3%
TA_PCT = 0.12            # 12% activation threshold
TSL_PCT = 0.09           # 9% trailing width (NOT 10%)
ADX_PERIOD = 20
ADX_MIN = 30.0
ADX_RISE_BARS = 6
RSI_PERIOD = 14
RSI_MIN = 35.0
RSI_MAX = 75.0
EMA_GAP_MIN = 0.002     # 0.2%
MONITOR_WINDOW = 24      # bars after cross
SKIP_SAME_DIR = True
DAILY_LOSS_LIMIT = -0.20

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


# ─── Indicator calculation using pandas ewm ───

def calc_indicators(df30, plan_cfg):
    """Calculate all indicators using pandas ewm methods as specified."""
    close = df30['close'].astype(float)
    high = df30['high'].astype(float)
    low = df30['low'].astype(float)

    # --- EMA / SMA for plan ---
    fast_ma = close.ewm(span=plan_cfg['fast_period'], adjust=False).mean()
    if plan_cfg['slow_type'] == 'EMA':
        slow_ma = close.ewm(span=plan_cfg['slow_period'], adjust=False).mean()
    else:
        slow_ma = close.rolling(plan_cfg['slow_period']).mean()

    # --- ADX using ewm(alpha=1/20) ---
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    alpha_adx = 1.0 / ADX_PERIOD
    atr = tr.ewm(alpha=alpha_adx, min_periods=ADX_PERIOD, adjust=False).mean()
    smooth_pdm = plus_dm.ewm(alpha=alpha_adx, min_periods=ADX_PERIOD, adjust=False).mean()
    smooth_mdm = minus_dm.ewm(alpha=alpha_adx, min_periods=ADX_PERIOD, adjust=False).mean()

    plus_di = 100.0 * smooth_pdm / atr.replace(0, 1e-10)
    minus_di = 100.0 * smooth_mdm / atr.replace(0, 1e-10)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    adx = dx.ewm(alpha=alpha_adx, min_periods=ADX_PERIOD, adjust=False).mean()

    # --- RSI using ewm(alpha=1/14) ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    alpha_rsi = 1.0 / RSI_PERIOD
    avg_gain = gain.ewm(alpha=alpha_rsi, min_periods=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha_rsi, min_periods=RSI_PERIOD, adjust=False).mean()
    rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss.replace(0, 1e-10))

    return (fast_ma.values, slow_ma.values, adx.values, rsi.values)


# ─── Backtest engine ───

def run_backtest(df30, plan_key, record_trades=True):
    """
    EXACT v32.1 FINAL backtest.
    Fully deterministic -- no randomness.
    """
    cfg = PLANS[plan_key]
    closes = df30['close'].values.astype(np.float64)
    highs = df30['high'].values.astype(np.float64)
    lows = df30['low'].values.astype(np.float64)
    timestamps = df30['timestamp'].values
    n = len(closes)

    # Compute indicators using pandas ewm
    fast_ma, slow_ma, adx, rsi = calc_indicators(df30, cfg)

    # ─── State ───
    bal = INITIAL_CAPITAL
    peak_bal = INITIAL_CAPITAL
    pos_dir = 0        # 0=flat, 1=long, -1=short
    pos_entry = 0.0
    pos_size = 0.0     # BTC qty
    pos_margin = 0.0
    pos_highest = 0.0
    pos_lowest = 999999999.0
    trail_active = False
    trail_sl = 0.0
    pos_sl = 0.0
    pos_bar_idx = 0

    # Cross tracking
    last_cross_bar = -9999
    last_cross_dir = 0  # +1 golden, -1 death
    last_entry_cross_bar = -9999  # for SKIP_SAME_DIR

    # Daily loss tracking (every 1440 bars)
    day_start_bal = INITIAL_CAPITAL
    daily_locked = False

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

        roi_pct = rpnl / pos_margin * 100 if pos_margin > 0 else 0

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

        saved_dir = pos_dir
        pos_dir = 0
        pos_entry = 0.0
        pos_size = 0.0
        pos_margin = 0.0
        pos_highest = 0.0
        pos_lowest = 999999999.0
        trail_active = False
        trail_sl = 0.0
        pos_sl = 0.0
        return saved_dir

    def try_enter(sig, bar_idx):
        """Attempt entry. Returns True if entered."""
        nonlocal bal, pos_dir, pos_entry, pos_size, pos_margin
        nonlocal pos_bar_idx, pos_highest, pos_lowest
        nonlocal trail_active, trail_sl, pos_sl
        nonlocal last_entry_cross_bar

        c = closes[bar_idx]
        h = highs[bar_idx]
        l = lows[bar_idx]

        entry_price = c
        margin = bal * MARGIN_PCT
        notional = margin * LEVERAGE
        size = notional / entry_price
        fee_entry = size * entry_price * FEE_RATE

        if margin <= 0 or size <= 0:
            return False

        bal -= fee_entry
        pos_dir = sig
        pos_entry = entry_price
        pos_size = size
        pos_margin = margin
        pos_bar_idx = bar_idx
        pos_highest = h
        pos_lowest = l
        trail_active = False
        trail_sl = 999999999.0 if sig == -1 else 0.0
        last_entry_cross_bar = last_cross_bar

        if sig == 1:
            pos_sl = entry_price * (1 - SL_PCT)
        else:
            pos_sl = entry_price * (1 + SL_PCT)

        return True

    # ─── Main loop ───
    for i in range(WARMUP, n):
        if bal <= 0:
            break

        c = closes[i]
        h = highs[i]
        l = lows[i]

        # Daily loss limit reset every 1440 bars
        if i > WARMUP and (i % 1440 == 0):
            day_start_bal = bal
            daily_locked = False

        if not daily_locked and day_start_bal > 0:
            day_loss = (bal - day_start_bal) / day_start_bal
            if day_loss <= DAILY_LOSS_LIMIT:
                daily_locked = True

        # ─── Position management ───
        rev_exited = False
        if pos_dir != 0:
            exited = False

            # Update tracking prices
            if pos_dir == 1:
                pos_highest = max(pos_highest, h)
            elif pos_dir == -1:
                pos_lowest = min(pos_lowest, l)

            # Priority 1: SL on intrabar H/L -- ONLY when TSL is NOT active
            if not trail_active:
                if pos_dir == 1:
                    if l <= pos_sl:
                        close_position(pos_sl, i, 'SL')
                        exited = True
                elif pos_dir == -1:
                    if h >= pos_sl:
                        close_position(pos_sl, i, 'SL')
                        exited = True

            # Priority 2: TA activation on HIGH/LOW (not close)
            if not exited and pos_dir != 0:
                if pos_dir == 1:
                    # Check activation on HIGH
                    ta_roi = (h - pos_entry) / pos_entry
                    if ta_roi >= TA_PCT:
                        trail_active = True
                    # TSL check on CLOSE
                    if trail_active:
                        new_tsl = pos_highest * (1 - TSL_PCT)
                        if new_tsl > trail_sl:
                            trail_sl = new_tsl
                        if c <= trail_sl:
                            close_position(c, i, 'TSL')
                            exited = True
                elif pos_dir == -1:
                    # Check activation on LOW
                    ta_roi = (pos_entry - l) / pos_entry
                    if ta_roi >= TA_PCT:
                        trail_active = True
                    # TSL check on CLOSE
                    if trail_active:
                        new_tsl = pos_lowest * (1 + TSL_PCT)
                        if new_tsl < trail_sl:
                            trail_sl = new_tsl
                        if c >= trail_sl:
                            close_position(c, i, 'TSL')
                            exited = True

            # Priority 3: REV exit -- MA alignment reversal
            # REV does NOT continue to next bar -- same bar can have REV exit + new entry
            if not exited and pos_dir != 0:
                if not np.isnan(fast_ma[i]) and not np.isnan(slow_ma[i]):
                    if pos_dir == 1 and fast_ma[i] < slow_ma[i]:
                        close_position(c, i, 'REV')
                        exited = True
                        rev_exited = True  # allow new entry on same bar
                    elif pos_dir == -1 and fast_ma[i] > slow_ma[i]:
                        close_position(c, i, 'REV')
                        exited = True
                        rev_exited = True  # allow new entry on same bar

            if exited and not rev_exited:
                if bal <= 0:
                    break
                continue  # SL/TSL exit -> next bar

            if not exited:
                continue  # still in position, no exit -> next bar (skip entry)

            # If rev_exited, fall through to entry logic
            if bal <= 0:
                break

        # ─── Signal detection ───
        if i < WARMUP + 1:
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

        # Already in position -> skip entry (shouldn't happen after above logic, but safety)
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

        # SKIP_SAME_DIR: don't re-enter from same cross event
        if SKIP_SAME_DIR and last_cross_bar == last_entry_cross_bar:
            continue

        # Filter: ADX >= 30
        if np.isnan(adx[i]) or adx[i] < ADX_MIN:
            continue

        # Filter: ADX rising -- adx[i] > adx[i - ADX_RISE_BARS]
        if i < ADX_RISE_BARS or np.isnan(adx[i - ADX_RISE_BARS]):
            continue
        if adx[i] <= adx[i - ADX_RISE_BARS]:
            continue

        # Filter: RSI 35-75
        if np.isnan(rsi[i]) or rsi[i] < RSI_MIN or rsi[i] > RSI_MAX:
            continue

        # Filter: EMA gap >= 0.2%
        if slow_ma[i] == 0 or np.isnan(slow_ma[i]):
            continue
        ema_gap = abs(fast_ma[i] - slow_ma[i]) / slow_ma[i]
        if ema_gap < EMA_GAP_MIN:
            continue

        # ─── ENTER ───
        try_enter(sig, i)

    # Close any open position at end
    if pos_dir != 0:
        close_position(closes[-1], n - 1, 'END')

    # Compute metrics
    total_return = (bal - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    years = max((n - WARMUP) * 30 / 525600, 0.01)  # 30m bars -> minutes -> years
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

    return metrics, trades_list


# ─── Monthly / Yearly tables ───

def compute_yearly(trades_list, plan_key):
    if not trades_list:
        return pd.DataFrame()

    tdf = pd.DataFrame(trades_list)
    tdf['exit_dt'] = pd.to_datetime(tdf['exit_time'])
    tdf['year'] = tdf['exit_dt'].dt.year

    yearly = tdf.groupby('year').agg(
        trades=('pnl', 'count'),
        gross_pnl=('pnl', 'sum'),
        wins=('pnl', lambda x: (x > 0).sum()),
        best=('pnl', 'max'),
        worst=('pnl', 'min'),
    ).reset_index()
    yearly['win_rate'] = (yearly['wins'] / yearly['trades'] * 100).round(1)
    yearly['plan'] = plan_key

    return yearly


# ─── Report formatting ───

def format_report(metrics_a, trades_a, yearly_a, metrics_b, trades_b, yearly_b):
    lines = []
    sep = "=" * 100

    lines.append(sep)
    lines.append("  v32.1 FINAL BACKTEST REPORT")
    lines.append("  Strategy: EMA Cross + ADX(20)>=30 rising + RSI(14) 35-75 + EMA Gap 0.2%")
    lines.append("  CRITICAL: ADX/RSI use pandas ewm(alpha=1/period) NOT Wilder smoothing")
    lines.append("  Cross Margin | Leverage 10x | Margin 35% | SL -3% | TSL +12% act / -9% trail | REV exit")
    lines.append("  SL only when TSL not active | TA activation on H/L | TSL check on close")
    lines.append("  Monitor Window: 24 bars (12h) | Daily Loss Limit: -20% (reset every 1440 bars)")
    lines.append("  Warmup: 600 bars | Same-bar REV exit + entry allowed")
    lines.append("  Data: BTC/USDT 5m -> 30m resample | Initial: $5,000")
    lines.append(sep)
    lines.append("")

    for label, m, trades, yearly in [
        ("PLAN A: EMA(100)/EMA(600)", metrics_a, trades_a, yearly_a),
        ("PLAN B: EMA(75)/SMA(750)", metrics_b, trades_b, yearly_b),
    ]:
        lines.append(f"  >>> {label}")
        lines.append("-" * 100)
        lines.append(f"  Final Balance    : ${m['final_balance']:>15,.2f}")
        lines.append(f"  Total Return     : {m['total_return_pct']:>12.2f}%")
        lines.append(f"  CAGR             : {m['cagr_pct']:>12.2f}%")
        lines.append(f"  Total Trades     : {m['total_trades']:>12d}")
        lines.append(f"  Win Rate         : {m['win_rate_pct']:>12.2f}%")
        lines.append(f"  Profit Factor    : {m['profit_factor']:>12.4f}")
        lines.append(f"  Max Drawdown     : {m['max_drawdown_pct']:>12.2f}%")
        lines.append(f"  Max Consec Loss  : {m['max_consec_loss']:>12d}")
        lines.append(f"  Gross Profit     : ${m['gross_profit']:>15,.2f}")
        lines.append(f"  Gross Loss       : ${m['gross_loss']:>15,.2f}")
        lines.append(f"  Avg PnL/Trade    : ${m['avg_pnl']:>15,.2f}")
        lines.append(f"  SL Exits         : {m['sl_exits']:>12d}")
        lines.append(f"  TSL Exits        : {m['tsl_exits']:>12d}")
        lines.append(f"  REV Exits        : {m['rev_exits']:>12d}")
        lines.append(f"  Liquidations     : {m['liquidations']:>12d}")
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
            lines.append(f"  {'Year':<6} {'Trades':>7} {'PnL':>15} {'WinRate':>8} {'Best':>12} {'Worst':>12}")
            lines.append("  " + "-" * 65)
            for _, row in yearly.iterrows():
                lines.append(
                    f"  {int(row['year']):<6} {int(row['trades']):>7} "
                    f"${row['gross_pnl']:>13,.2f} {row['win_rate']:>7.1f}% "
                    f"${row['best']:>10,.2f} ${row['worst']:>10,.2f}"
                )
            lines.append("")

        # Trade list (first 30 + last 10)
        if trades:
            lines.append(f"  Trade List (total {len(trades)}):")
            lines.append(
                f"  {'#':>4} {'Entry Time':<20} {'Exit Time':<20} {'Dir':<6} "
                f"{'Entry':>10} {'Exit':>10} {'PnL':>12} {'ROI%':>8} {'Exit':>5} {'Bars':>5}"
            )
            lines.append("  " + "-" * 110)
            show_trades = trades[:30]
            for idx, t in enumerate(show_trades):
                lines.append(
                    f"  {idx+1:>4} {t['entry_time']:<20} {t['exit_time']:<20} {t['direction']:<6} "
                    f"{t['entry_price']:>10.2f} {t['exit_price']:>10.2f} "
                    f"${t['pnl']:>10.2f} {t['roi_pct']:>7.2f}% {t['exit_reason']:>5} {t['hold_bars']:>5}"
                )
            if len(trades) > 40:
                lines.append(f"  {'...':>4} {'...':^20} {'...':^20}")
                for idx, t in enumerate(trades[-10:]):
                    real_idx = len(trades) - 10 + idx + 1
                    lines.append(
                        f"  {real_idx:>4} {t['entry_time']:<20} {t['exit_time']:<20} {t['direction']:<6} "
                        f"{t['entry_price']:>10.2f} {t['exit_price']:>10.2f} "
                        f"${t['pnl']:>10.2f} {t['roi_pct']:>7.2f}% {t['exit_reason']:>5} {t['hold_bars']:>5}"
                    )
            elif len(trades) > 30:
                for idx, t in enumerate(trades[30:]):
                    real_idx = 30 + idx + 1
                    lines.append(
                        f"  {real_idx:>4} {t['entry_time']:<20} {t['exit_time']:<20} {t['direction']:<6} "
                        f"{t['entry_price']:>10.2f} {t['exit_price']:>10.2f} "
                        f"${t['pnl']:>10.2f} {t['roi_pct']:>7.2f}% {t['exit_reason']:>5} {t['hold_bars']:>5}"
                    )

        lines.append("")
        lines.append(sep)
        lines.append("")

    return "\n".join(lines)


# ─── Multi-run verification ───

def run_multi(df30, plan_key, num_runs=10):
    results = []
    for r in range(num_runs):
        m, _ = run_backtest(df30, plan_key, record_trades=False)
        results.append(m)
    return results


# ─── Main ───

def main():
    t0 = time.time()

    print("=" * 80)
    print("  v32.1 FINAL Backtest + 30-Run Verification")
    print("  ADX/RSI: pandas ewm(alpha=1/period) | TSL: 9% | SL: only when TSL inactive")
    print("  TA activation: H/L | TSL check: close | Warmup: 600 | REV: same-bar entry OK")
    print("=" * 80)
    print()

    # Load data
    print("[1/6] Loading and resampling data (5m -> 30m)...")
    df30 = load_data()
    print(f"       30m bars: {len(df30):,}")
    print(f"       Range: {df30['timestamp'].iloc[0]} ~ {df30['timestamp'].iloc[-1]}")
    print()

    all_run_results = []

    results_store = {}

    for plan_key in ['A', 'B']:
        cfg = PLANS[plan_key]
        step = 2 if plan_key == 'A' else 3
        print(f"[{step}/6] Running Plan {plan_key}: {cfg['name']}...")

        # Main backtest with trades
        t1 = time.time()
        metrics, trades = run_backtest(df30, plan_key, record_trades=True)
        bt_time = time.time() - t1
        print(f"       Backtest done in {bt_time:.1f}s")
        print(f"       Trades: {metrics['total_trades']} | Final: ${metrics['final_balance']:,.2f} | PF: {metrics['profit_factor']:.4f}")
        print(f"       WR: {metrics['win_rate_pct']:.1f}% | MDD: {metrics['max_drawdown_pct']:.1f}% | CAGR: {metrics['cagr_pct']:.1f}%")

        # Yearly
        yearly = compute_yearly(trades, plan_key)

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

        if bt_std < 1e-6 and vr_std < 1e-6:
            print(f"       DETERMINISTIC CONFIRMED (std~0)")
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

        # Save trades CSV
        if trades:
            tdf = pd.DataFrame(trades)
            tpath = os.path.join(DATA_DIR, f"v32_1f_trades_{plan_key}.csv")
            tdf.to_csv(tpath, index=False, encoding='utf-8-sig')
            print(f"       Saved: {tpath}")

        results_store[plan_key] = {
            'metrics': metrics,
            'trades': trades,
            'yearly': yearly,
        }

        print()

    # Generate report
    print("[4/6] Generating report...")
    report_text = format_report(
        results_store['A']['metrics'], results_store['A']['trades'], results_store['A']['yearly'],
        results_store['B']['metrics'], results_store['B']['trades'], results_store['B']['yearly'],
    )

    # Add verification summary
    report_text += "\n"
    report_text += "=" * 100 + "\n"
    report_text += "  30-RUN VERIFICATION SUMMARY\n"
    report_text += "=" * 100 + "\n"
    for plan_key in ['A', 'B']:
        plan_runs = [r for r in all_run_results if r['plan'] == plan_key]
        bt_runs = [r for r in plan_runs if r['run_type'] == 'backtest']
        vr_runs = [r for r in plan_runs if r['run_type'] == 'verify']
        bt_f = [r['final_balance'] for r in bt_runs]
        vr_f = [r['final_balance'] for r in vr_runs]
        report_text += f"\n  Plan {plan_key}: {PLANS[plan_key]['name']}\n"
        report_text += f"    Backtest  10 runs: mean=${np.mean(bt_f):>15,.2f}  std=${np.std(bt_f):.6f}  min=${min(bt_f):>15,.2f}  max=${max(bt_f):>15,.2f}\n"
        report_text += f"    Verify    10 runs: mean=${np.mean(vr_f):>15,.2f}  std=${np.std(vr_f):.6f}  min=${min(vr_f):>15,.2f}  max=${max(vr_f):>15,.2f}\n"
        report_text += f"    ALL 20 runs: DETERMINISTIC={'YES' if np.std(bt_f)<1e-6 and np.std(vr_f)<1e-6 else 'NO'}\n"

        # Trade count comparison
        m = results_store[plan_key]['metrics']
        report_text += f"    Trades: {m['total_trades']} | Final: ${m['final_balance']:,.2f} | PF: {m['profit_factor']:.4f}\n"

    report_text += "\n" + "=" * 100 + "\n"

    # Add parameter summary
    report_text += "\n  PARAMETER SUMMARY:\n"
    report_text += f"    Initial Capital : ${INITIAL_CAPITAL:,.0f}\n"
    report_text += f"    Fee Rate        : {FEE_RATE*100:.2f}%\n"
    report_text += f"    Warmup          : {WARMUP} bars\n"
    report_text += f"    Margin          : {MARGIN_PCT*100:.0f}%\n"
    report_text += f"    Leverage        : {LEVERAGE}x\n"
    report_text += f"    SL              : {SL_PCT*100:.1f}%\n"
    report_text += f"    TA Activation   : {TA_PCT*100:.1f}% (checked on H/L)\n"
    report_text += f"    TSL Width       : {TSL_PCT*100:.1f}% (checked on close)\n"
    report_text += f"    ADX Period      : {ADX_PERIOD} (ewm alpha=1/{ADX_PERIOD})\n"
    report_text += f"    ADX Min         : {ADX_MIN}\n"
    report_text += f"    ADX Rise Bars   : {ADX_RISE_BARS}\n"
    report_text += f"    RSI Period      : {RSI_PERIOD} (ewm alpha=1/{RSI_PERIOD})\n"
    report_text += f"    RSI Range       : {RSI_MIN}-{RSI_MAX}\n"
    report_text += f"    EMA Gap Min     : {EMA_GAP_MIN*100:.1f}%\n"
    report_text += f"    Monitor Window  : {MONITOR_WINDOW} bars\n"
    report_text += f"    Skip Same Dir   : {SKIP_SAME_DIR}\n"
    report_text += f"    Daily Loss Limit: {DAILY_LOSS_LIMIT*100:.0f}% (reset every 1440 bars)\n"
    report_text += "\n" + "=" * 100 + "\n"

    print(report_text)

    # Save files
    print("[5/6] Saving output files...")

    # Report
    rpath = os.path.join(DATA_DIR, "v32_1f_report.txt")
    with open(rpath, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"       Saved: {rpath}")

    # 30-run CSV
    run_df = pd.DataFrame(all_run_results)
    run_path = os.path.join(DATA_DIR, "v32_1f_30run.csv")
    run_df.to_csv(run_path, index=False, encoding='utf-8-sig')
    print(f"       Saved: {run_path}")

    elapsed = time.time() - t0
    print(f"\n[6/6] Complete in {elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
