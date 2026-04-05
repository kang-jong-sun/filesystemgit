"""
v23.4 - 30-Run Verification for Winner
S4: EMA(3)/EMA(200) 30m ADX14>=35 RSI30-65 T+6/-3 SL-7% ISOLATED M30% L10x
Uses 30m data (smaller dataset = faster)
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
INITIAL_CAPITAL = 5000.0
FEE_RATE = 0.0004

# Winner config
WINNER = {
    'id': 'S4', 'name': 'EMA(3)/EMA(200) 30m ADX14>=35 RSI30-65 T+6/-3 SL-7%',
    'fast_type': 'EMA', 'fast_period': 3,
    'slow_type': 'EMA', 'slow_period': 200,
    'tf': '30m', 'adx_period': 14, 'adx_min': 35,
    'rsi_lo': 30, 'rsi_hi': 65, 'delay': 0,
    'sl_pct': 0.07, 'trail_act': 0.06, 'trail_width': 0.03,
    'source': 'v14.4/v22.2 classic',
}
MARGIN_MODE = 'ISOLATED'
MARGIN_PCT = 0.30
LEVERAGE = 10


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
            s_ = pdi[i] + mdi[i]
            dx[i] = 100 * abs(pdi[i] - mdi[i]) / s_ if s_ > 0 else 0
    return wilder_smooth(dx, period)


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


def run_backtest_fast(df_tf, margin_mode='ISOLATED', margin_pct=0.30, leverage=10):
    """Optimized single-strategy backtest."""
    closes = df_tf['close'].values.astype(np.float64)
    highs = df_tf['high'].values.astype(np.float64)
    lows = df_tf['low'].values.astype(np.float64)

    cfg = WINNER
    fast_ma = calc_ema(closes, cfg['fast_period'])
    slow_ma = calc_ema(closes, cfg['slow_period'])
    adx = calc_adx_wilder(highs, lows, closes, cfg['adx_period'])
    rsi_arr = calc_rsi(closes, 14)

    n = len(closes)
    bal = INITIAL_CAPITAL
    peak_bal = INITIAL_CAPITAL
    pos_dir = 0
    pos_entry = 0.0
    pos_size = 0.0
    pos_margin = 0.0
    pos_time_idx = 0
    pos_highest = 0.0
    pos_lowest = 0.0
    pos_trail_active = False
    pos_trail_sl = 0.0
    pos_sl = 0.0

    _n_trades = 0; _n_wins = 0
    _gross_profit = 0.0; _gross_loss = 0.0
    _sl_count = 0; _trail_count = 0; _rev_count = 0
    _max_consec_loss = 0; _cur_consec_loss = 0
    _mdd = 0.0; _mdd_peak = INITIAL_CAPITAL
    _liquidation_events = 0

    dd_halved = False
    warmup = max(300, int(cfg['slow_period'] * 1.5))

    for i in range(warmup, n):
        if bal <= 0:
            break
        c = closes[i]; h = highs[i]; l = lows[i]

        if peak_bal > 0:
            cur_dd = (peak_bal - bal) / peak_bal
            dd_halved = cur_dd > 0.30

        if pos_dir != 0:
            exited = False

            if pos_dir == 1:
                pos_highest = max(pos_highest, h)
                if l <= pos_sl:
                    # Close at SL
                    rpnl = pos_size * (pos_sl - pos_entry)
                    fee = pos_size * pos_sl * FEE_RATE
                    pnl = rpnl - fee
                    if margin_mode == 'ISOLATED':
                        if pnl < -pos_margin: pnl = -pos_margin; _liquidation_events += 1
                    else:
                        if pnl < -bal: pnl = -bal; _liquidation_events += 1
                    bal += pnl; _n_trades += 1; _sl_count += 1
                    if pnl > 0: _n_wins += 1; _gross_profit += pnl; _cur_consec_loss = 0
                    else: _gross_loss += abs(pnl); _cur_consec_loss += 1; _max_consec_loss = max(_max_consec_loss, _cur_consec_loss)
                    peak_bal = max(peak_bal, bal); _mdd_peak = max(_mdd_peak, bal)
                    if _mdd_peak > 0: _mdd = max(_mdd, (_mdd_peak - bal) / _mdd_peak)
                    pos_dir = 0; exited = True
                else:
                    cur_roi = (c - pos_entry) / pos_entry
                    if cur_roi >= cfg['trail_act']: pos_trail_active = True
                    if pos_trail_active:
                        new_tsl = pos_highest * (1 - cfg['trail_width'])
                        if new_tsl > pos_trail_sl: pos_trail_sl = max(new_tsl, pos_sl)
                        if c <= pos_trail_sl:
                            rpnl = pos_size * (c - pos_entry)
                            fee = pos_size * c * FEE_RATE
                            pnl = rpnl - fee
                            if margin_mode == 'ISOLATED':
                                if pnl < -pos_margin: pnl = -pos_margin; _liquidation_events += 1
                            else:
                                if pnl < -bal: pnl = -bal; _liquidation_events += 1
                            bal += pnl; _n_trades += 1; _trail_count += 1
                            if pnl > 0: _n_wins += 1; _gross_profit += pnl; _cur_consec_loss = 0
                            else: _gross_loss += abs(pnl); _cur_consec_loss += 1; _max_consec_loss = max(_max_consec_loss, _cur_consec_loss)
                            peak_bal = max(peak_bal, bal); _mdd_peak = max(_mdd_peak, bal)
                            if _mdd_peak > 0: _mdd = max(_mdd, (_mdd_peak - bal) / _mdd_peak)
                            pos_dir = 0; exited = True

            elif pos_dir == -1:
                pos_lowest = min(pos_lowest, l)
                if h >= pos_sl:
                    rpnl = pos_size * (pos_entry - pos_sl)
                    fee = pos_size * pos_sl * FEE_RATE
                    pnl = rpnl - fee
                    if margin_mode == 'ISOLATED':
                        if pnl < -pos_margin: pnl = -pos_margin; _liquidation_events += 1
                    else:
                        if pnl < -bal: pnl = -bal; _liquidation_events += 1
                    bal += pnl; _n_trades += 1; _sl_count += 1
                    if pnl > 0: _n_wins += 1; _gross_profit += pnl; _cur_consec_loss = 0
                    else: _gross_loss += abs(pnl); _cur_consec_loss += 1; _max_consec_loss = max(_max_consec_loss, _cur_consec_loss)
                    peak_bal = max(peak_bal, bal); _mdd_peak = max(_mdd_peak, bal)
                    if _mdd_peak > 0: _mdd = max(_mdd, (_mdd_peak - bal) / _mdd_peak)
                    pos_dir = 0; exited = True
                else:
                    cur_roi = (pos_entry - c) / pos_entry
                    if cur_roi >= cfg['trail_act']: pos_trail_active = True
                    if pos_trail_active:
                        new_tsl = pos_lowest * (1 + cfg['trail_width'])
                        if new_tsl < pos_trail_sl: pos_trail_sl = min(new_tsl, pos_sl)
                        if c >= pos_trail_sl:
                            rpnl = pos_size * (pos_entry - c)
                            fee = pos_size * c * FEE_RATE
                            pnl = rpnl - fee
                            if margin_mode == 'ISOLATED':
                                if pnl < -pos_margin: pnl = -pos_margin; _liquidation_events += 1
                            else:
                                if pnl < -bal: pnl = -bal; _liquidation_events += 1
                            bal += pnl; _n_trades += 1; _trail_count += 1
                            if pnl > 0: _n_wins += 1; _gross_profit += pnl; _cur_consec_loss = 0
                            else: _gross_loss += abs(pnl); _cur_consec_loss += 1; _max_consec_loss = max(_max_consec_loss, _cur_consec_loss)
                            peak_bal = max(peak_bal, bal); _mdd_peak = max(_mdd_peak, bal)
                            if _mdd_peak > 0: _mdd = max(_mdd, (_mdd_peak - bal) / _mdd_peak)
                            pos_dir = 0; exited = True

            if exited:
                if bal <= 0: break
                continue

        # Signal detection
        if i < warmup + 1: continue
        if np.isnan(fast_ma[i]) or np.isnan(fast_ma[i-1]): continue
        if np.isnan(slow_ma[i]) or np.isnan(slow_ma[i-1]): continue

        cross_up = (fast_ma[i-1] <= slow_ma[i-1]) and (fast_ma[i] > slow_ma[i])
        cross_down = (fast_ma[i-1] >= slow_ma[i-1]) and (fast_ma[i] < slow_ma[i])

        if not cross_up and not cross_down: continue

        sig = 1 if cross_up else -1

        if sig == 1 and fast_ma[i] <= slow_ma[i]: continue
        if sig == -1 and fast_ma[i] >= slow_ma[i]: continue
        if np.isnan(adx[i]) or adx[i] < cfg['adx_min']: continue
        if np.isnan(rsi_arr[i]) or not (cfg['rsi_lo'] <= rsi_arr[i] <= cfg['rsi_hi']): continue

        direction = sig
        if pos_dir == direction: continue

        # Reverse close
        if pos_dir != 0 and pos_dir != direction:
            if pos_dir == 1:
                rpnl = pos_size * (c - pos_entry)
            else:
                rpnl = pos_size * (pos_entry - c)
            fee = pos_size * c * FEE_RATE
            pnl = rpnl - fee
            if margin_mode == 'ISOLATED':
                if pnl < -pos_margin: pnl = -pos_margin; _liquidation_events += 1
            else:
                if pnl < -bal: pnl = -bal; _liquidation_events += 1
            bal += pnl; _n_trades += 1; _rev_count += 1
            if pnl > 0: _n_wins += 1; _gross_profit += pnl; _cur_consec_loss = 0
            else: _gross_loss += abs(pnl); _cur_consec_loss += 1; _max_consec_loss = max(_max_consec_loss, _cur_consec_loss)
            peak_bal = max(peak_bal, bal); _mdd_peak = max(_mdd_peak, bal)
            if _mdd_peak > 0: _mdd = max(_mdd, (_mdd_peak - bal) / _mdd_peak)
            pos_dir = 0
            if bal <= 0: break

        if bal < 50: continue

        margin_mult = 0.5 if dd_halved else 1.0
        sz = bal * margin_pct * margin_mult
        if sz < 5: continue
        notional = sz * leverage

        if direction == 1:
            pos_sl = c * (1 - cfg['sl_pct'])
        else:
            pos_sl = c * (1 + cfg['sl_pct'])

        bal -= notional * FEE_RATE
        pos_dir = direction
        pos_entry = c
        pos_size = notional / c
        pos_margin = sz
        pos_time_idx = i
        pos_highest = c
        pos_lowest = c
        pos_trail_active = False
        pos_trail_sl = pos_sl

    # Close remaining
    if pos_dir != 0 and bal > 0:
        if pos_dir == 1:
            rpnl = pos_size * (closes[-1] - pos_entry)
        else:
            rpnl = pos_size * (pos_entry - closes[-1])
        fee = pos_size * closes[-1] * FEE_RATE
        pnl = rpnl - fee
        if margin_mode == 'ISOLATED':
            if pnl < -pos_margin: pnl = -pos_margin
        else:
            if pnl < -bal: pnl = -bal
        bal += pnl; _n_trades += 1
        if pnl > 0: _n_wins += 1; _gross_profit += pnl
        else: _gross_loss += abs(pnl)

    peak_bal = max(peak_bal, bal)
    _mdd_peak = max(_mdd_peak, bal)
    if _mdd_peak > 0: _mdd = max(_mdd, (_mdd_peak - bal) / _mdd_peak)

    ret = (bal - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    pf = _gross_profit / _gross_loss if _gross_loss > 0 else (999 if _gross_profit > 0 else 0)
    wr = _n_wins / _n_trades * 100 if _n_trades > 0 else 0

    return {
        'final': round(bal, 2), 'return_pct': round(ret, 2),
        'pf': round(pf, 4), 'mdd_pct': round(_mdd * 100, 2),
        'trades': _n_trades, 'wins': _n_wins, 'losses': _n_trades - _n_wins,
        'win_rate': round(wr, 2), 'sl_count': _sl_count,
        'trail_count': _trail_count, 'rev_count': _rev_count,
        'liquidation_events': _liquidation_events,
        'max_consec_loss': _max_consec_loss,
        'gross_profit': round(_gross_profit, 2),
        'gross_loss': round(_gross_loss, 2),
    }


def main():
    T0 = time.time()
    print(f"{'=' * 100}", flush=True)
    print(f"  v23.4 - 30-RUN VERIFICATION", flush=True)
    print(f"  Winner: S4 EMA(3)/EMA(200) 30m | ISOLATED M30% L10x", flush=True)
    print(f"{'=' * 100}\n", flush=True)

    # Load data
    print("Loading data...", flush=True)
    t0 = time.time()
    dfs = []
    for f in CSV_FILES:
        df = pd.read_csv(f, parse_dates=['timestamp']); dfs.append(df)
    df5m = pd.concat(dfs, ignore_index=True)
    df5m.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df5m.sort_values('timestamp', inplace=True)
    df5m.set_index('timestamp', inplace=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df5m[c] = df5m[c].astype(np.float64)

    df30m = df5m.resample('30min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    print(f"  30m bars: {len(df30m):,} ({time.time()-t0:.1f}s)\n", flush=True)

    total_bars = len(df30m)
    min_bars = max(1000, int(WINNER['slow_period'] * 3))

    print(f"  {'Run':>4} {'Offset':>7} {'Bars':>7} {'Final':>12} {'Ret%':>10} "
          f"{'PF':>7} {'MDD%':>7} {'Trd':>4} {'WR%':>6} {'SL':>3} "
          f"{'Trail':>5} {'Rev':>4} {'Liq':>3} {'MCL':>4}", flush=True)
    print(f"  {'-' * 100}", flush=True)

    verify_rows = []
    for run_idx in range(30):
        t0 = time.time()
        if run_idx == 0:
            df_run = df30m
            offset = 0
        else:
            offset = int(total_bars * 0.02 * run_idx)
            if offset + min_bars > total_bars:
                offset = max(0, total_bars - min_bars)
            df_run = df30m.iloc[offset:]

        m = run_backtest_fast(df_run, MARGIN_MODE, MARGIN_PCT, LEVERAGE)
        elapsed = time.time() - t0

        pf_str = f"{m['pf']:.2f}" if m['pf'] < 999 else "INF"
        print(f"  {run_idx+1:>4} {offset:>7} {len(df_run):>7} "
              f"${m['final']:>10,.0f} {m['return_pct']:>+9.1f}% "
              f"{pf_str:>6} {m['mdd_pct']:>6.1f} {m['trades']:>4} "
              f"{m['win_rate']:>5.1f} {m['sl_count']:>3} "
              f"{m['trail_count']:>5} {m['rev_count']:>4} "
              f"{m['liquidation_events']:>3} {m['max_consec_loss']:>4} "
              f"({elapsed:.1f}s)", flush=True)

        verify_rows.append({
            'run': run_idx + 1,
            'start_offset': offset,
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
            'rev_hits': m['rev_count'],
            'liquidation_events': m['liquidation_events'],
            'max_consec_loss': m['max_consec_loss'],
            'gross_profit': m['gross_profit'],
            'gross_loss': m['gross_loss'],
        })

    # Summary
    vdf = pd.DataFrame(verify_rows)

    print(f"\n{'=' * 100}", flush=True)
    print(f"  30-RUN SUMMARY", flush=True)
    print(f"{'=' * 100}", flush=True)
    print(f"  {'Metric':<20} {'Mean':>12} {'Median':>12} {'Min':>12} {'Max':>12} {'Std':>12}", flush=True)
    print(f"  {'-' * 82}", flush=True)

    stats_data = [
        ('final_balance', vdf['final_balance']),
        ('return_pct', vdf['return_pct']),
        ('pf', vdf['pf'].clip(upper=999)),
        ('mdd_pct', vdf['mdd_pct']),
        ('trades', vdf['trades'].astype(float)),
        ('win_rate', vdf['win_rate']),
        ('sl_hits', vdf['sl_hits'].astype(float)),
        ('trail_hits', vdf['trail_hits'].astype(float)),
        ('max_consec_loss', vdf['max_consec_loss'].astype(float)),
    ]
    for col_name, vals in stats_data:
        mn = vals.mean()
        md = vals.median()
        mi = vals.min()
        mx = vals.max()
        sd = vals.std()
        if col_name == 'final_balance':
            print(f"  {col_name:<20} ${mn:>11,.0f} ${md:>11,.0f} ${mi:>11,.0f} ${mx:>11,.0f} {sd:>12,.0f}", flush=True)
        elif col_name == 'return_pct':
            print(f"  {col_name:<20} {mn:>11,.1f}% {md:>11,.1f}% {mi:>11,.1f}% {mx:>11,.1f}% {sd:>12,.1f}", flush=True)
        elif col_name == 'pf':
            print(f"  {col_name:<20} {mn:>12.2f} {md:>12.2f} {mi:>12.2f} {mx:>12.2f} {sd:>12.2f}", flush=True)
        elif col_name == 'mdd_pct':
            print(f"  {col_name:<20} {mn:>11.1f}% {md:>11.1f}% {mi:>11.1f}% {mx:>11.1f}% {sd:>12.1f}", flush=True)
        else:
            print(f"  {col_name:<20} {mn:>12.1f} {md:>12.1f} {mi:>12.1f} {mx:>12.1f} {sd:>12.2f}", flush=True)

    profitable = len(vdf[vdf['return_pct'] > 0])
    zero_liq = len(vdf[vdf['liquidation_events'] == 0])
    print(f"\n  Profitable runs:      {profitable}/30 ({profitable/30*100:.0f}%)", flush=True)
    print(f"  Zero liquidation:     {zero_liq}/30 ({zero_liq/30*100:.0f}%)", flush=True)
    print(f"  Avg return:           {vdf['return_pct'].mean():+,.1f}%", flush=True)
    print(f"  Min return:           {vdf['return_pct'].min():+,.1f}%", flush=True)
    print(f"  Max return:           {vdf['return_pct'].max():+,.1f}%", flush=True)
    print(f"  Avg PF:               {vdf['pf'].clip(upper=999).mean():.2f}", flush=True)

    # Save
    csv_path = os.path.join(DATA_DIR, "v23_4_FINAL_30run.csv")
    vdf.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}", flush=True)

    total_time = time.time() - T0
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f}m)", flush=True)
    print(f"{'=' * 100}", flush=True)


if __name__ == "__main__":
    main()
