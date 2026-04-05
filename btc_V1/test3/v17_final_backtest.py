#!/usr/bin/env python3
"""
v17.0 3-Tier BTC Futures Backtest
=================================
Three independent engines on the same 30m resampled data.
Tier 1 (Sniper):     WMA(3)/EMA(200), ADX>=45, RSI 25-65,  delay=5, SL-8%, Trail+3/-2, M50% L7x
Tier 2 (Core):       HMA(3)/EMA(250), ADX>=45, RSI 35-70,  delay=3, SL-8%, Trail+3/-2, M40% L5x
Tier 3 (Compounder): EMA(3)/SMA(250), ADX>=35, RSI 25-65,  delay=0, SL-7%, Trail+3/-2, M30% L10x
"""

import numpy as np
import pandas as pd
import json, time, sys, os
from datetime import datetime
from io import StringIO

# ─── Indicator helpers ───────────────────────────────────────────────

def wilder_smooth(values, period):
    result = np.full(len(values), np.nan)
    start = 0
    while start < len(values) and np.isnan(values[start]):
        start += 1
    if start + period > len(values):
        return result
    result[start + period - 1] = np.mean(values[start:start + period])
    for i in range(start + period, len(values)):
        if not np.isnan(values[i]) and not np.isnan(result[i - 1]):
            result[i] = (result[i - 1] * (period - 1) + values[i]) / period
    return result

def calc_adx(high, low, close, period=14):
    n = len(close)
    tr = np.full(n, np.nan)
    plus_dm = np.full(n, np.nan)
    minus_dm = np.full(n, np.nan)
    for i in range(1, n):
        h, l, pc = high[i], low[i], close[i - 1]
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))
        up = h - high[i - 1]
        dn = low[i - 1] - l
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
    atr = wilder_smooth(tr, period)
    sdm_plus = wilder_smooth(plus_dm, period)
    sdm_minus = wilder_smooth(minus_dm, period)
    di_plus = np.where(atr > 0, 100.0 * sdm_plus / atr, 0.0)
    di_minus = np.where(atr > 0, 100.0 * sdm_minus / atr, 0.0)
    dx = np.where((di_plus + di_minus) > 0, 100.0 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0.0)
    adx = wilder_smooth(dx, period)
    return adx

def calc_rsi(close, period=14):
    n = len(close)
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain[0] = np.nan
    loss[0] = np.nan
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[np.isnan(avg_gain) | np.isnan(avg_loss)] = np.nan
    return rsi

def calc_ema(values, period):
    result = np.full(len(values), np.nan)
    start = 0
    while start < len(values) and np.isnan(values[start]):
        start += 1
    if start + period > len(values):
        return result
    result[start + period - 1] = np.mean(values[start:start + period])
    k = 2.0 / (period + 1)
    for i in range(start + period, len(values)):
        if not np.isnan(values[i]) and not np.isnan(result[i - 1]):
            result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result

def calc_sma(values, period):
    result = np.full(len(values), np.nan)
    for i in range(period - 1, len(values)):
        result[i] = np.mean(values[i - period + 1:i + 1])
    return result

def calc_wma(values, period):
    result = np.full(len(values), np.nan)
    weights = np.arange(1, period + 1, dtype=float)
    wsum = weights.sum()
    for i in range(period - 1, len(values)):
        seg = values[i - period + 1:i + 1]
        if not np.any(np.isnan(seg)):
            result[i] = np.dot(seg, weights) / wsum
    return result

def calc_hma(values, period):
    half = max(int(period / 2), 1)
    wma_half = calc_wma(values, half)
    wma_full = calc_wma(values, period)
    diff = 2.0 * wma_half - wma_full
    sqrt_p = max(int(np.sqrt(period)), 1)
    return calc_wma(diff, sqrt_p)


# ─── Data loading ────────────────────────────────────────────────────

def load_data(data_dir):
    parts = []
    for i in range(1, 4):
        fp = os.path.join(data_dir, f"btc_usdt_5m_2020_to_now_part{i}.csv")
        df = pd.read_csv(fp, parse_dates=["timestamp"])
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset="timestamp", keep="first", inplace=True)
    df.set_index("timestamp", inplace=True)

    ohlcv = df[["open", "high", "low", "close", "volume"]].resample("30min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    return ohlcv


# ─── Backtest engine ─────────────────────────────────────────────────

FEE_RATE = 0.0004  # 0.04% per side

def run_tier(ohlcv_np, timestamps, tier_cfg, return_trades=True):
    """
    ohlcv_np: (N,5) array  [open, high, low, close, volume]
    tier_cfg: dict with all tier params
    """
    close = ohlcv_np[:, 3]
    high = ohlcv_np[:, 1]
    low = ohlcv_np[:, 2]
    n = len(close)

    # --- indicators ---
    fast_type = tier_cfg["fast_type"]
    fast_period = tier_cfg["fast_period"]
    slow_type = tier_cfg["slow_type"]
    slow_period = tier_cfg["slow_period"]
    adx_thresh = tier_cfg["adx_thresh"]
    rsi_lo = tier_cfg["rsi_lo"]
    rsi_hi = tier_cfg["rsi_hi"]
    entry_delay = tier_cfg["entry_delay"]
    sl_pct = tier_cfg["sl_pct"]
    trail_act = tier_cfg["trail_act"]
    trail_width = tier_cfg["trail_width"]
    margin_pct = tier_cfg["margin_pct"]
    leverage = tier_cfg["leverage"]
    init_balance = tier_cfg["init_balance"]

    # fast MA
    if fast_type == "WMA":
        fast_ma = calc_wma(close, fast_period)
    elif fast_type == "HMA":
        fast_ma = calc_hma(close, fast_period)
    elif fast_type == "EMA":
        fast_ma = calc_ema(close, fast_period)
    else:
        fast_ma = calc_sma(close, fast_period)

    # slow MA
    if slow_type == "EMA":
        slow_ma = calc_ema(close, slow_period)
    elif slow_type == "SMA":
        slow_ma = calc_sma(close, slow_period)
    else:
        slow_ma = calc_ema(close, slow_period)

    adx = calc_adx(high, low, close, 14)
    rsi = calc_rsi(close, 14)

    # --- cross detection ---
    cross_bull = np.full(n, False)
    cross_bear = np.full(n, False)
    for i in range(1, n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(fast_ma[i-1]) or np.isnan(slow_ma[i-1]):
            continue
        if fast_ma[i-1] <= slow_ma[i-1] and fast_ma[i] > slow_ma[i]:
            cross_bull[i] = True
        if fast_ma[i-1] >= slow_ma[i-1] and fast_ma[i] < slow_ma[i]:
            cross_bear[i] = True

    # --- simulation ---
    balance = init_balance
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    qty = 0.0
    peak_roi = 0.0
    trailing_active = False
    peak_close = 0.0

    pending_signal = 0  # 1=long pending, -1=short pending
    pending_countdown = 0

    trades = []
    equity_curve = [balance]

    for i in range(1, n):
        c = close[i]
        h = high[i]
        l = low[i]
        a = adx[i] if not np.isnan(adx[i]) else 0.0
        r = rsi[i] if not np.isnan(rsi[i]) else 50.0

        # --- detect new signals ---
        new_bull = cross_bull[i] and a >= adx_thresh and rsi_lo <= r <= rsi_hi
        new_bear = cross_bear[i] and a >= adx_thresh and rsi_lo <= r <= rsi_hi

        # --- REVERSE signal handling (highest priority) ---
        closed_by_reverse = False
        if position == 1 and new_bear:
            # close long at this bar's close
            pnl_pct = (c - entry_price) / entry_price * leverage
            notional = qty * entry_price
            fee = notional * FEE_RATE
            raw_pnl = qty * (c - entry_price) * (1 if position == 1 else -1)
            pnl = raw_pnl - fee
            balance += pnl
            if return_trades:
                trades.append({
                    "entry_time": entry_ts, "exit_time": timestamps[i],
                    "direction": "LONG", "entry_price": entry_price, "exit_price": c,
                    "qty": qty, "pnl": pnl, "pnl_pct": pnl_pct * 100,
                    "exit_reason": "REVERSE", "balance": balance,
                    "hold_bars": i - entry_bar
                })
            position = 0
            closed_by_reverse = True
            trailing_active = False
            # queue short entry
            if entry_delay > 0:
                pending_signal = -1
                pending_countdown = entry_delay
            else:
                pending_signal = -1
                pending_countdown = 0

        elif position == -1 and new_bull:
            pnl_pct = (entry_price - c) / entry_price * leverage
            notional = qty * entry_price
            fee = notional * FEE_RATE
            raw_pnl = qty * (entry_price - c)
            pnl = raw_pnl - fee
            balance += pnl
            if return_trades:
                trades.append({
                    "entry_time": entry_ts, "exit_time": timestamps[i],
                    "direction": "SHORT", "entry_price": entry_price, "exit_price": c,
                    "qty": qty, "pnl": pnl, "pnl_pct": pnl_pct * 100,
                    "exit_reason": "REVERSE", "balance": balance,
                    "hold_bars": i - entry_bar
                })
            position = 0
            closed_by_reverse = True
            trailing_active = False
            if entry_delay > 0:
                pending_signal = 1
                pending_countdown = entry_delay
            else:
                pending_signal = 1
                pending_countdown = 0

        # --- Same direction re-entry SKIP ---
        if not closed_by_reverse:
            if position == 1 and new_bull:
                pass  # skip
            elif position == -1 and new_bear:
                pass  # skip
            elif position == 0 and not closed_by_reverse:
                if new_bull:
                    if entry_delay > 0:
                        pending_signal = 1
                        pending_countdown = entry_delay
                    else:
                        pending_signal = 1
                        pending_countdown = 0
                elif new_bear:
                    if entry_delay > 0:
                        pending_signal = -1
                        pending_countdown = entry_delay
                    else:
                        pending_signal = -1
                        pending_countdown = 0

        # --- Trailing stop check on CLOSE (before SL to let trail have priority if both) ---
        if position != 0 and trailing_active:
            if position == 1:
                cur_roi = (c - entry_price) / entry_price * leverage
                if c > peak_close:
                    peak_close = c
                    peak_roi = cur_roi
                drawdown_from_peak = (peak_close - c) / peak_close * leverage
                if drawdown_from_peak >= trail_width:
                    pnl_pct = cur_roi
                    notional = qty * entry_price
                    fee = notional * FEE_RATE
                    raw_pnl = qty * (c - entry_price)
                    pnl = raw_pnl - fee
                    balance += pnl
                    if return_trades:
                        trades.append({
                            "entry_time": entry_ts, "exit_time": timestamps[i],
                            "direction": "LONG", "entry_price": entry_price, "exit_price": c,
                            "qty": qty, "pnl": pnl, "pnl_pct": pnl_pct * 100,
                            "exit_reason": "TRAIL", "balance": balance,
                            "hold_bars": i - entry_bar
                        })
                    position = 0
                    trailing_active = False
            elif position == -1:
                cur_roi = (entry_price - c) / entry_price * leverage
                if c < peak_close:
                    peak_close = c
                    peak_roi = cur_roi
                drawdown_from_peak = (c - peak_close) / peak_close * leverage
                if drawdown_from_peak >= trail_width:
                    pnl_pct = cur_roi
                    notional = qty * entry_price
                    fee = notional * FEE_RATE
                    raw_pnl = qty * (entry_price - c)
                    pnl = raw_pnl - fee
                    balance += pnl
                    if return_trades:
                        trades.append({
                            "entry_time": entry_ts, "exit_time": timestamps[i],
                            "direction": "SHORT", "entry_price": entry_price, "exit_price": c,
                            "qty": qty, "pnl": pnl, "pnl_pct": pnl_pct * 100,
                            "exit_reason": "TRAIL", "balance": balance,
                            "hold_bars": i - entry_bar
                        })
                    position = 0
                    trailing_active = False

        # --- SL check on high/low (lowest priority) ---
        if position == 1:
            roi_worst = (l - entry_price) / entry_price * leverage
            if roi_worst <= sl_pct:
                sl_price = entry_price * (1 + sl_pct / leverage)
                exit_p = max(sl_price, l)
                pnl_pct_val = (exit_p - entry_price) / entry_price * leverage
                notional = qty * entry_price
                fee = notional * FEE_RATE
                raw_pnl = qty * (exit_p - entry_price)
                pnl = raw_pnl - fee
                balance += pnl
                if return_trades:
                    trades.append({
                        "entry_time": entry_ts, "exit_time": timestamps[i],
                        "direction": "LONG", "entry_price": entry_price, "exit_price": exit_p,
                        "qty": qty, "pnl": pnl, "pnl_pct": pnl_pct_val * 100,
                        "exit_reason": "SL", "balance": balance,
                        "hold_bars": i - entry_bar
                    })
                position = 0
                trailing_active = False

        elif position == -1:
            roi_worst = (entry_price - h) / entry_price * leverage
            if roi_worst <= sl_pct:
                sl_price = entry_price * (1 - sl_pct / leverage)
                exit_p = min(sl_price, h)
                pnl_pct_val = (entry_price - exit_p) / entry_price * leverage
                notional = qty * entry_price
                fee = notional * FEE_RATE
                raw_pnl = qty * (entry_price - exit_p)
                pnl = raw_pnl - fee
                balance += pnl
                if return_trades:
                    trades.append({
                        "entry_time": entry_ts, "exit_time": timestamps[i],
                        "direction": "SHORT", "entry_price": entry_price, "exit_price": exit_p,
                        "qty": qty, "pnl": pnl, "pnl_pct": pnl_pct_val * 100,
                        "exit_reason": "SL", "balance": balance,
                        "hold_bars": i - entry_bar
                    })
                position = 0
                trailing_active = False

        # --- Trailing activation check ---
        if position == 1 and not trailing_active:
            cur_roi = (c - entry_price) / entry_price * leverage
            if cur_roi >= trail_act:
                trailing_active = True
                peak_close = c
                peak_roi = cur_roi
        elif position == -1 and not trailing_active:
            cur_roi = (entry_price - c) / entry_price * leverage
            if cur_roi >= trail_act:
                trailing_active = True
                peak_close = c
                peak_roi = cur_roi

        # --- Update peak for trailing ---
        if position == 1 and trailing_active:
            if c > peak_close:
                peak_close = c
                peak_roi = (c - entry_price) / entry_price * leverage
        elif position == -1 and trailing_active:
            if c < peak_close:
                peak_close = c
                peak_roi = (entry_price - c) / entry_price * leverage

        # --- Pending entry countdown ---
        if pending_signal != 0 and position == 0:
            if pending_countdown > 0:
                pending_countdown -= 1
            else:
                # execute entry
                if balance > 0:
                    margin = balance * margin_pct
                    notional = margin * leverage
                    qty = notional / c
                    fee_entry = notional * FEE_RATE
                    balance -= fee_entry
                    entry_price = c
                    position = pending_signal
                    entry_bar = i
                    entry_ts = timestamps[i]
                    trailing_active = False
                    peak_roi = 0.0
                    peak_close = c
                pending_signal = 0

        equity_curve.append(balance + (qty * (c - entry_price) * position if position != 0 else 0))

    # close any open position at end
    if position != 0:
        c = close[-1]
        if position == 1:
            raw_pnl = qty * (c - entry_price)
        else:
            raw_pnl = qty * (entry_price - c)
        fee = qty * entry_price * FEE_RATE
        pnl = raw_pnl - fee
        pnl_pct = ((c - entry_price) / entry_price * leverage * position) * 100
        balance += pnl
        if return_trades:
            trades.append({
                "entry_time": entry_ts, "exit_time": timestamps[-1],
                "direction": "LONG" if position == 1 else "SHORT",
                "entry_price": entry_price, "exit_price": c,
                "qty": qty, "pnl": pnl, "pnl_pct": pnl_pct,
                "exit_reason": "END", "balance": balance,
                "hold_bars": len(close) - 1 - entry_bar
            })

    return balance, trades, equity_curve


# ─── Metrics ─────────────────────────────────────────────────────────

def calc_metrics(trades_list, init_balance, tier_name):
    if not trades_list:
        return {"tier": tier_name, "trades": 0}
    df = pd.DataFrame(trades_list)
    total = len(df)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    wr = len(wins) / total * 100 if total > 0 else 0
    total_pnl = df["pnl"].sum()
    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0.001
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
    final_bal = df["balance"].iloc[-1]
    roi = (final_bal - init_balance) / init_balance * 100
    # max drawdown
    cum = df["pnl"].cumsum() + init_balance
    peak = cum.cummax()
    dd = (cum - peak) / peak * 100
    max_dd = dd.min()
    # avg hold
    avg_hold = df["hold_bars"].mean()
    # exit reasons
    reasons = df["exit_reason"].value_counts().to_dict()
    # direction breakdown
    longs = df[df["direction"] == "LONG"]
    shorts = df[df["direction"] == "SHORT"]
    return {
        "tier": tier_name,
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(wr, 2),
        "total_pnl": round(total_pnl, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(pf, 3),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "final_balance": round(final_bal, 2),
        "roi_pct": round(roi, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_hold_bars": round(avg_hold, 1),
        "exit_reasons": reasons,
        "long_trades": len(longs),
        "short_trades": len(shorts),
        "long_pnl": round(longs["pnl"].sum(), 2) if len(longs) > 0 else 0,
        "short_pnl": round(shorts["pnl"].sum(), 2) if len(shorts) > 0 else 0,
    }


# ─── Tier configs ────────────────────────────────────────────────────

TOTAL_CAPITAL = 3000.0

TIERS = {
    "Tier1_Sniper": {
        "fast_type": "WMA", "fast_period": 3,
        "slow_type": "EMA", "slow_period": 200,
        "adx_thresh": 45, "rsi_lo": 25, "rsi_hi": 65,
        "entry_delay": 5,
        "sl_pct": -0.08, "trail_act": 0.03, "trail_width": 0.02,
        "margin_pct": 0.50, "leverage": 7,
        "init_balance": TOTAL_CAPITAL * 0.30,  # 900
    },
    "Tier2_Core": {
        "fast_type": "HMA", "fast_period": 3,
        "slow_type": "EMA", "slow_period": 250,
        "adx_thresh": 45, "rsi_lo": 35, "rsi_hi": 70,
        "entry_delay": 3,
        "sl_pct": -0.08, "trail_act": 0.03, "trail_width": 0.02,
        "margin_pct": 0.40, "leverage": 5,
        "init_balance": TOTAL_CAPITAL * 0.40,  # 1200
    },
    "Tier3_Compounder": {
        "fast_type": "EMA", "fast_period": 3,
        "slow_type": "SMA", "slow_period": 250,
        "adx_thresh": 35, "rsi_lo": 25, "rsi_hi": 65,
        "entry_delay": 0,
        "sl_pct": -0.07, "trail_act": 0.03, "trail_width": 0.02,
        "margin_pct": 0.30, "leverage": 10,
        "init_balance": TOTAL_CAPITAL * 0.30,  # 900
    },
}


# ─── Main ────────────────────────────────────────────────────────────

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 80)
    print("  v17.0 3-Tier BTC Futures Backtest")
    print("=" * 80)

    print("\n[1/5] Loading and resampling data...")
    t0 = time.time()
    ohlcv = load_data(data_dir)
    print(f"  Loaded {len(ohlcv)} bars of 30m data  ({ohlcv.index[0]} ~ {ohlcv.index[-1]})")
    print(f"  Time: {time.time()-t0:.1f}s")

    ohlcv_np = ohlcv[["open", "high", "low", "close", "volume"]].values.astype(np.float64)
    timestamps = ohlcv.index.values

    # ─── Single detailed run ──────────────────────────────────────
    print("\n[2/5] Running detailed backtest (1 run with trades)...")
    t0 = time.time()
    all_trades = []
    tier_metrics = {}
    tier_balances = {}

    for tname, tcfg in TIERS.items():
        bal, trades, eq = run_tier(ohlcv_np, timestamps, tcfg, return_trades=True)
        tier_balances[tname] = bal
        tier_metrics[tname] = calc_metrics(trades, tcfg["init_balance"], tname)
        for t in trades:
            t["tier"] = tname
        all_trades.extend(trades)
        print(f"  {tname}: {len(trades)} trades, Final=${bal:.2f}, PnL=${bal - tcfg['init_balance']:.2f}")

    print(f"  Time: {time.time()-t0:.1f}s")

    portfolio_final = sum(tier_balances.values())
    portfolio_pnl = portfolio_final - TOTAL_CAPITAL
    portfolio_roi = portfolio_pnl / TOTAL_CAPITAL * 100

    # ─── 30-run verification ──────────────────────────────────────
    print("\n[3/5] Running 30-run verification...")
    t0 = time.time()
    run_results = []
    for run_i in range(30):
        run_total = 0.0
        run_detail = {}
        for tname, tcfg in TIERS.items():
            bal, _, _ = run_tier(ohlcv_np, timestamps, tcfg, return_trades=False)
            run_total += bal
            run_detail[tname] = bal
        run_results.append({
            "run": run_i + 1,
            "portfolio_final": round(run_total, 2),
            "portfolio_pnl": round(run_total - TOTAL_CAPITAL, 2),
            "portfolio_roi_pct": round((run_total - TOTAL_CAPITAL) / TOTAL_CAPITAL * 100, 2),
            **{f"{k}_bal": round(v, 2) for k, v in run_detail.items()}
        })
    run_df = pd.DataFrame(run_results)
    print(f"  30 runs complete. Time: {time.time()-t0:.1f}s")
    print(f"  Consistency check: Unique finals = {run_df['portfolio_final'].nunique()}")

    # ─── Build report ─────────────────────────────────────────────
    print("\n[4/5] Building report...")

    trades_df = pd.DataFrame(all_trades)
    if len(trades_df) > 0:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
        trades_df.sort_values("exit_time", inplace=True)

    report = StringIO()

    def w(s=""):
        report.write(s + "\n")

    w("=" * 80)
    w("  v17.0 3-Tier BTC Futures Backtest -- FINAL REPORT")
    w(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"  Data: {ohlcv.index[0]} ~ {ohlcv.index[-1]}  ({len(ohlcv)} x 30m bars)")
    w("=" * 80)

    w("\n" + "=" * 80)
    w("  PORTFOLIO SUMMARY")
    w("=" * 80)
    w(f"  Initial Capital:   ${TOTAL_CAPITAL:>12,.2f}")
    w(f"  Final Balance:     ${portfolio_final:>12,.2f}")
    w(f"  Total PnL:         ${portfolio_pnl:>12,.2f}")
    w(f"  ROI:               {portfolio_roi:>12.2f}%")
    w(f"  Total Trades:      {len(all_trades):>12d}")

    if len(trades_df) > 0:
        wins_all = trades_df[trades_df["pnl"] > 0]
        losses_all = trades_df[trades_df["pnl"] <= 0]
        wr_all = len(wins_all) / len(trades_df) * 100
        gp = wins_all["pnl"].sum() if len(wins_all) > 0 else 0
        gl = abs(losses_all["pnl"].sum()) if len(losses_all) > 0 else 0.001
        pf_all = gp / gl if gl > 0 else float("inf")
        # Portfolio max DD
        sorted_trades = trades_df.sort_values("exit_time")
        cum_pnl = sorted_trades["pnl"].cumsum() + TOTAL_CAPITAL
        peak_pnl = cum_pnl.cummax()
        dd_pnl = (cum_pnl - peak_pnl) / peak_pnl * 100
        max_dd_all = dd_pnl.min()

        w(f"  Win Rate:          {wr_all:>12.2f}%")
        w(f"  Profit Factor:     {pf_all:>12.3f}")
        w(f"  Max Drawdown:      {max_dd_all:>12.2f}%")
        w(f"  Avg Win:           ${wins_all['pnl'].mean():>12.2f}" if len(wins_all) > 0 else "  Avg Win:                   N/A")
        w(f"  Avg Loss:          ${losses_all['pnl'].mean():>12.2f}" if len(losses_all) > 0 else "  Avg Loss:                  N/A")

    w("\n" + "=" * 80)
    w("  PER-TIER BREAKDOWN")
    w("=" * 80)
    for tname in TIERS:
        m = tier_metrics[tname]
        cfg = TIERS[tname]
        w(f"\n  --- {tname} ---")
        w(f"  Fast: {cfg['fast_type']}({cfg['fast_period']})  Slow: {cfg['slow_type']}({cfg['slow_period']})")
        w(f"  ADX>={cfg['adx_thresh']}  RSI [{cfg['rsi_lo']}-{cfg['rsi_hi']}]  Delay={cfg['entry_delay']}  SL={cfg['sl_pct']*100:.0f}%  Trail={cfg['trail_act']*100:.0f}%/{cfg['trail_width']*100:.0f}%")
        w(f"  Margin={cfg['margin_pct']*100:.0f}%  Leverage={cfg['leverage']}x  Init=${cfg['init_balance']:.0f}")
        w(f"  Trades: {m['trades']}  Wins: {m.get('wins',0)}  Losses: {m.get('losses',0)}  WR: {m.get('win_rate',0):.2f}%")
        w(f"  PnL: ${m.get('total_pnl',0):.2f}  PF: {m.get('profit_factor',0):.3f}  MaxDD: {m.get('max_drawdown_pct',0):.2f}%")
        w(f"  Final: ${m.get('final_balance',0):.2f}  ROI: {m.get('roi_pct',0):.2f}%")
        w(f"  Avg Hold: {m.get('avg_hold_bars',0):.1f} bars")
        w(f"  Exit Reasons: {m.get('exit_reasons', {})}")
        w(f"  Long: {m.get('long_trades',0)} trades  PnL=${m.get('long_pnl',0):.2f}")
        w(f"  Short: {m.get('short_trades',0)} trades  PnL=${m.get('short_pnl',0):.2f}")

    # ─── Monthly performance ─────────────────────────────────────
    w("\n" + "=" * 80)
    w("  MONTHLY PERFORMANCE")
    w("=" * 80)
    if len(trades_df) > 0:
        trades_df["month"] = trades_df["exit_time"].dt.to_period("M")
        monthly = trades_df.groupby("month").agg(
            trades=("pnl", "count"),
            pnl=("pnl", "sum"),
            wins=("pnl", lambda x: (x > 0).sum()),
        )
        monthly["wr"] = (monthly["wins"] / monthly["trades"] * 100).round(1)
        monthly["cum_pnl"] = monthly["pnl"].cumsum()
        w(f"  {'Month':<10} {'Trades':>7} {'PnL':>12} {'WR%':>7} {'Cum PnL':>12}")
        w(f"  {'-'*10} {'-'*7} {'-'*12} {'-'*7} {'-'*12}")
        for idx, row in monthly.iterrows():
            w(f"  {str(idx):<10} {int(row['trades']):>7} ${row['pnl']:>11,.2f} {row['wr']:>6.1f}% ${row['cum_pnl']:>11,.2f}")
        monthly_csv = monthly.copy()
        monthly_csv.index = monthly_csv.index.astype(str)

    # ─── Yearly performance ──────────────────────────────────────
    w("\n" + "=" * 80)
    w("  YEARLY PERFORMANCE")
    w("=" * 80)
    if len(trades_df) > 0:
        trades_df["year"] = trades_df["exit_time"].dt.year
        yearly = trades_df.groupby("year").agg(
            trades=("pnl", "count"),
            pnl=("pnl", "sum"),
            wins=("pnl", lambda x: (x > 0).sum()),
        )
        yearly["wr"] = (yearly["wins"] / yearly["trades"] * 100).round(1)
        w(f"  {'Year':<6} {'Trades':>7} {'PnL':>12} {'WR%':>7}")
        w(f"  {'-'*6} {'-'*7} {'-'*12} {'-'*7}")
        for idx, row in yearly.iterrows():
            w(f"  {idx:<6} {int(row['trades']):>7} ${row['pnl']:>11,.2f} {row['wr']:>6.1f}%")

    # ─── Entry structure analysis ─────────────────────────────────
    w("\n" + "=" * 80)
    w("  ENTRY STRUCTURE ANALYSIS")
    w("=" * 80)
    if len(trades_df) > 0:
        w(f"\n  Direction Distribution:")
        dir_counts = trades_df["direction"].value_counts()
        for d, cnt in dir_counts.items():
            pnl_d = trades_df[trades_df["direction"] == d]["pnl"].sum()
            w(f"    {d}: {cnt} trades, PnL=${pnl_d:,.2f}")

        w(f"\n  Exit Reason Distribution:")
        reason_counts = trades_df["exit_reason"].value_counts()
        for r, cnt in reason_counts.items():
            pnl_r = trades_df[trades_df["exit_reason"] == r]["pnl"].sum()
            avg_r = trades_df[trades_df["exit_reason"] == r]["pnl"].mean()
            w(f"    {r}: {cnt} trades, Total PnL=${pnl_r:,.2f}, Avg=${avg_r:,.2f}")

        w(f"\n  Hold Time Distribution (bars):")
        w(f"    Min: {trades_df['hold_bars'].min()}")
        w(f"    Mean: {trades_df['hold_bars'].mean():.1f}")
        w(f"    Median: {trades_df['hold_bars'].median():.0f}")
        w(f"    Max: {trades_df['hold_bars'].max()}")

        w(f"\n  Per-Tier Direction Breakdown:")
        for tname in TIERS:
            tdf = trades_df[trades_df["tier"] == tname]
            if len(tdf) > 0:
                longs = tdf[tdf["direction"] == "LONG"]
                shorts = tdf[tdf["direction"] == "SHORT"]
                w(f"    {tname}: L={len(longs)} S={len(shorts)}  L_PnL=${longs['pnl'].sum():.2f}  S_PnL=${shorts['pnl'].sum():.2f}")

    # ─── Top/Bottom 10 ───────────────────────────────────────────
    w("\n" + "=" * 80)
    w("  TOP 10 TRADES (by PnL)")
    w("=" * 80)
    if len(trades_df) > 0:
        top10 = trades_df.nlargest(10, "pnl")
        w(f"  {'Tier':<20} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'PnL':>10} {'PnL%':>8} {'Reason':<8} {'Bars':>5}")
        for _, r in top10.iterrows():
            w(f"  {r['tier']:<20} {r['direction']:<6} {r['entry_price']:>10.1f} {r['exit_price']:>10.1f} ${r['pnl']:>9.2f} {r['pnl_pct']:>7.2f}% {r['exit_reason']:<8} {r['hold_bars']:>5}")

    w("\n" + "=" * 80)
    w("  BOTTOM 10 TRADES (by PnL)")
    w("=" * 80)
    if len(trades_df) > 0:
        bot10 = trades_df.nsmallest(10, "pnl")
        w(f"  {'Tier':<20} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'PnL':>10} {'PnL%':>8} {'Reason':<8} {'Bars':>5}")
        for _, r in bot10.iterrows():
            w(f"  {r['tier']:<20} {r['direction']:<6} {r['entry_price']:>10.1f} {r['exit_price']:>10.1f} ${r['pnl']:>9.2f} {r['pnl_pct']:>7.2f}% {r['exit_reason']:<8} {r['hold_bars']:>5}")

    # ─── 30-run verification ─────────────────────────────────────
    w("\n" + "=" * 80)
    w("  30-RUN VERIFICATION")
    w("=" * 80)
    w(f"  Unique results: {run_df['portfolio_final'].nunique()}")
    w(f"  Mean final:  ${run_df['portfolio_final'].mean():,.2f}")
    w(f"  Std final:   ${run_df['portfolio_final'].std():,.4f}")
    w(f"  Min final:   ${run_df['portfolio_final'].min():,.2f}")
    w(f"  Max final:   ${run_df['portfolio_final'].max():,.2f}")

    report_text = report.getvalue()
    print(report_text)

    # ─── Save files ───────────────────────────────────────────────
    print("\n[5/5] Saving files...")

    with open(os.path.join(data_dir, "v17_FINAL_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Saved v17_FINAL_report.txt")

    if len(trades_df) > 0:
        trades_df.to_csv(os.path.join(data_dir, "v17_FINAL_trades.csv"), index=False)
        print(f"  Saved v17_FINAL_trades.csv ({len(trades_df)} trades)")

        monthly_csv.to_csv(os.path.join(data_dir, "v17_FINAL_monthly.csv"))
        print(f"  Saved v17_FINAL_monthly.csv")

    run_df.to_csv(os.path.join(data_dir, "v17_FINAL_30run.csv"), index=False)
    print(f"  Saved v17_FINAL_30run.csv")

    # metrics JSON
    metrics_out = {
        "portfolio": {
            "initial": TOTAL_CAPITAL,
            "final": round(portfolio_final, 2),
            "pnl": round(portfolio_pnl, 2),
            "roi_pct": round(portfolio_roi, 2),
            "total_trades": len(all_trades),
        },
        "tiers": {k: v for k, v in tier_metrics.items()},
        "verification_30run": {
            "unique_results": int(run_df["portfolio_final"].nunique()),
            "mean": round(run_df["portfolio_final"].mean(), 2),
            "std": round(run_df["portfolio_final"].std(), 4),
        }
    }
    with open(os.path.join(data_dir, "v17_FINAL_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"  Saved v17_FINAL_metrics.json")

    print("\n" + "=" * 80)
    print("  ALL DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
