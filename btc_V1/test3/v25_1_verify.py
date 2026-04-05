"""
v25.1 Final Strategy Backtest + 30-Run Verification
Single Model: HMA(3)/EMA(250), 15m, ADX(14)>=45, Delay 3, Trail +3/-2%, M40% Lev5x, SL -8%
Wilder ADX, REVERSE, Same-dir skip
"""
import pandas as pd, numpy as np, os, time, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
DIR = r"D:\filesystem\futures\btc_V1\test3"
FEE = 0.0004
INIT = 3000.0

# ---- Indicators ----
def wilder(arr, p):
    out = np.full(len(arr), np.nan)
    s = 0
    while s < len(arr) and np.isnan(arr[s]): s += 1
    if s + p > len(arr): return out
    out[s+p-1] = np.nanmean(arr[s:s+p])
    for i in range(s+p, len(arr)):
        if not np.isnan(arr[i]) and not np.isnan(out[i-1]):
            out[i] = (out[i-1]*(p-1) + arr[i]) / p
    return out

def calc_adx_wilder(high, low, close, period=14):
    n = len(high)
    tr = np.full(n, np.nan)
    pdm = np.full(n, np.nan)
    mdm = np.full(n, np.nan)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
        up = high[i] - high[i-1]
        dn = low[i-1] - low[i]
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
            s = pdi[i] + mdi[i]
            dx[i] = 100 * abs(pdi[i] - mdi[i]) / s if s > 0 else 0
    adx = wilder(dx, period)
    return adx, atr

def calc_hma(close, period):
    n = len(close)
    half = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)

    def _wma(src, p):
        out = np.full(len(src), np.nan)
        w = np.arange(1, p+1, dtype=float)
        ws = w.sum()
        for i in range(p-1, len(src)):
            sl = src[i-p+1:i+1]
            if np.any(np.isnan(sl)): continue
            out[i] = np.dot(sl, w) / ws
        return out

    wma_half = _wma(close, half)
    wma_full = _wma(close, period)
    diff = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    return _wma(diff, sqrt_p)

def calc_ema(close, period):
    out = np.full(len(close), np.nan)
    s = 0
    while s < len(close) and np.isnan(close[s]): s += 1
    if s >= len(close): return out
    out[s] = close[s]
    m = 2.0 / (period + 1)
    for i in range(s+1, len(close)):
        if not np.isnan(close[i]) and not np.isnan(out[i-1]):
            out[i] = close[i] * m + out[i-1] * (1 - m)
    return out

def calc_rsi(close, period=14):
    n = len(close)
    out = np.full(n, np.nan)
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        d = close[i] - close[i-1]
        if d > 0: gains[i] = d
        else: losses[i] = -d
    avg_g = wilder(gains, period)
    avg_l = wilder(losses, period)
    for i in range(n):
        if not np.isnan(avg_g[i]) and not np.isnan(avg_l[i]):
            if avg_l[i] == 0:
                out[i] = 100.0
            else:
                out[i] = 100 - 100 / (1 + avg_g[i] / avg_l[i])
    return out

# ---- Data ----
def load_15m():
    t0 = time.time()
    fs = [os.path.join(DIR, f"btc_usdt_5m_2020_to_now_part{i}.csv") for i in range(1,4)]
    df = pd.concat([pd.read_csv(f, parse_dates=['timestamp']) for f in fs], ignore_index=True)
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    d15 = df.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    print(f"[Data] 5m:{len(df):,} -> 15m:{len(d15):,} | {time.time()-t0:.1f}s")
    return d15

# ---- Config ----
CFG = {
    'adx_min': 45, 'rsi_lo': 35, 'rsi_hi': 70,
    'delay': 3, 'sl_pct': 0.08,
    'trail_act': 0.03, 'trail_width': 0.02,
    'margin': 0.40, 'lev': 5,
}

# ---- Backtest ----
def run_backtest(opens, highs, lows, closes, adx, rsi_arr, fast_ma, slow_ma, timestamps, return_trades=True):
    n = len(closes)
    bal = INIT
    peak_bal = INIT
    pos_dir = 0  # 0=none, 1=long, -1=short
    pos_entry = 0.0
    pos_size = 0.0
    pos_time_idx = 0
    pos_highest = 0.0
    pos_lowest = 0.0
    pos_trail_active = False
    pos_trail_sl = 0.0
    pos_sl = 0.0
    pos_pp = 0.0
    pos_rem = 1.0
    pos_tp1 = False

    pending_signal = 0  # 1=long, -1=short
    pending_bar = 0

    trades = []
    bal_hist = []
    consec_loss = 0
    cooldown_until = -1
    monthly_start = {}

    warmup = 300  # need 250+ for EMA(250)

    def close_pos(exit_price, exit_reason, bar_idx):
        nonlocal bal, peak_bal, pos_dir, consec_loss
        rs = pos_size * pos_rem
        if pos_dir == 1:
            rpnl = rs * (exit_price - pos_entry) / pos_entry
        else:
            rpnl = rs * (pos_entry - exit_price) / pos_entry
        total_pnl = rpnl + pos_pp - rs * FEE
        bal += total_pnl
        peak_bal = max(peak_bal, bal)
        consec_loss = consec_loss + 1 if total_pnl < 0 else 0

        margin = pos_size / CFG['lev']
        roi = total_pnl / margin * 100 if margin > 0 else 0
        hold_bars = bar_idx - pos_time_idx

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
                    ((pos_highest - pos_entry) / pos_entry * 100) if pos_dir == 1 else
                    ((pos_entry - pos_lowest) / pos_entry * 100), 2),
                'tp1_hit': pos_tp1,
                'hold_bars': hold_bars,
                'hold_hours': round(hold_bars * 0.25, 2),
                'balance': round(bal, 2),
                'sl_pct': round(CFG['sl_pct'] * 100, 1),
            })
        old_dir = pos_dir
        pos_dir = 0
        return total_pnl

    for i in range(warmup, n):
        h = highs[i]; l = lows[i]; c = closes[i]

        # ---- Position management ----
        if pos_dir != 0:
            exited = False

            if pos_dir == 1:
                pos_highest = max(pos_highest, h)
                # SL on intrabar low
                if l <= pos_sl:
                    close_pos(pos_sl, 'SL', i)
                    exited = True
                else:
                    # Trail on close
                    cur_roi = (c - pos_entry) / pos_entry
                    if cur_roi >= CFG['trail_act'] or pos_tp1:
                        pos_trail_active = True
                        new_tsl = pos_highest * (1 - CFG['trail_width'])
                        if new_tsl > pos_trail_sl:
                            pos_trail_sl = max(new_tsl, pos_sl)
                    if pos_trail_active and c <= pos_trail_sl:
                        close_pos(c, 'TRAIL', i)
                        exited = True

            elif pos_dir == -1:
                pos_lowest = min(pos_lowest, l)
                if h >= pos_sl:
                    close_pos(pos_sl, 'SL', i)
                    exited = True
                else:
                    cur_roi = (pos_entry - c) / pos_entry
                    if cur_roi >= CFG['trail_act'] or pos_tp1:
                        pos_trail_active = True
                        new_tsl = pos_lowest * (1 + CFG['trail_width'])
                        if new_tsl < pos_trail_sl:
                            pos_trail_sl = min(new_tsl, pos_sl)
                    if pos_trail_active and c >= pos_trail_sl:
                        close_pos(c, 'TRAIL', i)
                        exited = True

        # Balance history
        if i % 4 == 0:
            bal_hist.append({'ts': str(timestamps[i]), 'bal': round(bal, 2)})

        # ---- Signal detection ----
        if i < warmup + 1: continue
        if np.isnan(fast_ma[i]) or np.isnan(fast_ma[i-1]): continue
        if np.isnan(slow_ma[i]) or np.isnan(slow_ma[i-1]): continue

        cross_up = (fast_ma[i-1] <= slow_ma[i-1]) and (fast_ma[i] > slow_ma[i])
        cross_down = (fast_ma[i-1] >= slow_ma[i-1]) and (fast_ma[i] < slow_ma[i])

        if cross_up:
            pending_signal = 1
            pending_bar = i
        elif cross_down:
            pending_signal = -1
            pending_bar = i

        # ---- Delayed entry check ----
        if pending_signal != 0 and (i - pending_bar) == CFG['delay']:
            sig = pending_signal
            pending_signal = 0

            # Verify MA still aligned after delay
            if sig == 1 and fast_ma[i] <= slow_ma[i]: continue
            if sig == -1 and fast_ma[i] >= slow_ma[i]: continue

            # ADX filter
            if np.isnan(adx[i]) or adx[i] < CFG['adx_min']: continue
            # RSI filter
            if np.isnan(rsi_arr[i]) or not (CFG['rsi_lo'] <= rsi_arr[i] <= CFG['rsi_hi']): continue

            direction = sig  # 1=long, -1=short
            dir_str = 'LONG' if direction == 1 else 'SHORT'

            # Same-direction skip
            if pos_dir == direction:
                continue

            # REVERSE: close opposite position
            if pos_dir != 0 and pos_dir != direction:
                close_pos(c, 'REV', i)

            # Risk checks
            if cooldown_until > i: continue
            dd = (bal - peak_bal) / peak_bal if peak_bal > 0 else 0
            if dd < -0.45:
                cooldown_until = i + 192  # 48h in 15m bars
                continue

            mk = f"{timestamps[i].year}-{timestamps[i].month:02d}"
            if mk not in monthly_start: monthly_start[mk] = bal
            if monthly_start[mk] > 0 and (bal - monthly_start[mk]) / monthly_start[mk] < -0.20:
                continue

            # Streak adjustment
            streak_m = max(0.4, 1.0 - consec_loss * 0.10)

            # Position sizing
            sz = bal * CFG['margin'] * streak_m
            if sz < 5: continue
            notional = sz * CFG['lev']

            # SL price
            if direction == 1:
                pos_sl = c * (1 - CFG['sl_pct'])
            else:
                pos_sl = c * (1 + CFG['sl_pct'])

            # Entry fee
            bal -= notional * FEE

            # Open position
            pos_dir = direction
            pos_entry = c
            pos_size = notional
            pos_time_idx = i
            pos_highest = c
            pos_lowest = c
            pos_trail_active = False
            pos_trail_sl = pos_sl
            pos_pp = 0.0
            pos_rem = 1.0
            pos_tp1 = False

    # ---- Metrics ----
    total_ret = (bal - INIT) / INIT * 100
    metrics = {'initial': INIT, 'final': round(bal, 2), 'return_pct': round(total_ret, 2), 'trades': len(trades)}

    if trades:
        df_t = pd.DataFrame(trades)
        wins = df_t[df_t['pnl'] > 0]; losses = df_t[df_t['pnl'] <= 0]
        gp = wins['pnl'].sum() if len(wins) else 0
        gl = abs(losses['pnl'].sum()) if len(losses) else 0

        bals = [INIT] + [t['balance'] for t in trades]
        pk = bals[0]; mdd = 0
        for b in bals:
            pk = max(pk, b); dd = (b - pk) / pk; mdd = min(mdd, dd)

        mc = cc = 0
        for t in trades:
            if t['pnl'] <= 0: cc += 1; mc = max(mc, cc)
            else: cc = 0

        metrics.update({
            'pf': round(gp / gl, 4) if gl > 0 else 999,
            'mdd': round(mdd * 100, 2),
            'win_rate': round(len(wins) / len(df_t) * 100, 2),
            'avg_win': round(wins['roi_pct'].mean(), 2) if len(wins) else 0,
            'avg_loss': round(losses['roi_pct'].mean(), 2) if len(losses) else 0,
            'rr': round(abs(wins['roi_pct'].mean() / losses['roi_pct'].mean()), 2) if len(losses) and losses['roi_pct'].mean() != 0 else 999,
            'max_consec_loss': mc,
            'gross_profit': round(gp, 2),
            'gross_loss': round(gl, 2),
            'sl_count': len(df_t[df_t['exit_reason'] == 'SL']),
            'trail_count': len(df_t[df_t['exit_reason'] == 'TRAIL']),
            'rev_count': len(df_t[df_t['exit_reason'] == 'REV']),
        })

    return metrics, trades, bal_hist


def main():
    print("=" * 100)
    print("  v25.1 FINAL BACKTEST + 30-RUN VERIFICATION")
    print("  HMA(3)/EMA(250) | 15m | ADX(14)>=45 | Delay 3 | Trail +3/-2 | M40% Lev5x")
    print("=" * 100)

    d15 = load_15m()
    print("\n[Indicators] Computing...")
    t0 = time.time()

    opens = d15['open'].values
    highs = d15['high'].values
    lows = d15['low'].values
    closes = d15['close'].values
    timestamps = d15.index

    fast_ma = calc_hma(closes, 3)
    slow_ma = calc_ema(closes, 250)
    adx_arr, _ = calc_adx_wilder(highs, lows, closes, 14)
    rsi_arr = calc_rsi(closes, 14)
    print(f"  Done | {time.time()-t0:.1f}s")

    # ---- Main backtest ----
    print("\n[Backtest] Running...")
    t0 = time.time()
    metrics, trades, bal_hist = run_backtest(opens, highs, lows, closes, adx_arr, rsi_arr, fast_ma, slow_ma, timestamps, True)
    print(f"  {time.time()-t0:.1f}s | Trades: {len(trades)} | Final: ${metrics['final']:,.2f}")

    # ---- Print report ----
    T = len(trades)
    print(f"""
{'='*100}
  PORTFOLIO SUMMARY
{'='*100}
  Initial:       ${INIT:,.0f}
  Final:         ${metrics['final']:,.2f}
  Return:        {metrics['return_pct']:+.1f}%
  PF:            {metrics.get('pf', 0):.2f}
  MDD:           {metrics.get('mdd', 0):.1f}%
  Trades:        {T}
  Win Rate:      {metrics.get('win_rate', 0):.1f}%
  Avg Win:       {metrics.get('avg_win', 0):+.2f}%
  Avg Loss:      {metrics.get('avg_loss', 0):+.2f}%
  R:R:           {metrics.get('rr', 0):.2f}
  Max Consec L:  {metrics.get('max_consec_loss', 0)}
  Gross Profit:  ${metrics.get('gross_profit', 0):,.2f}
  Gross Loss:    ${metrics.get('gross_loss', 0):,.2f}
  SL:            {metrics.get('sl_count', 0)}
  TRAIL:         {metrics.get('trail_count', 0)}
  REV:           {metrics.get('rev_count', 0)}
""")

    if trades:
        df = pd.DataFrame(trades)

        # Direction
        print("  DIRECTION ANALYSIS")
        print("  " + "-" * 60)
        for d in ['LONG', 'SHORT']:
            sub = df[df['direction'] == d]
            if len(sub) == 0: continue
            sw = sub[sub['pnl'] > 0]
            print(f"  {d:>5}: {len(sub):>4} ({len(sub)/T*100:.0f}%) WR:{len(sw)/len(sub)*100:.0f}% AvgROI:{sub['roi_pct'].mean():+.2f}% PnL:${sub['pnl'].sum():+,.0f}")

        # Exit reason
        print(f"\n  EXIT REASON ANALYSIS")
        print("  " + "-" * 60)
        for r in ['TRAIL', 'REV', 'SL']:
            rt = df[df['exit_reason'] == r]
            if len(rt) == 0: continue
            print(f"  {r:>5}: {len(rt):>4} ({len(rt)/T*100:.0f}%) AvgROI:{rt['roi_pct'].mean():+.2f}% PnL:${rt['pnl'].sum():+,.0f}")

        # Hold time
        print(f"\n  HOLD TIME ANALYSIS")
        print("  " + "-" * 60)
        for a, b, lb in [(0,2,'<2h'),(2,8,'2-8h'),(8,24,'8-24h'),(24,72,'1-3d'),(72,168,'3-7d'),(168,9999,'7d+')]:
            ht = df[(df['hold_hours'] >= a) & (df['hold_hours'] < b)]
            if len(ht):
                hw = ht[ht['pnl'] > 0]
                print(f"  {lb:>6}: {len(ht):>4} WR:{len(hw)/len(ht)*100:.0f}% AvgROI:{ht['roi_pct'].mean():+.2f}% PnL:${ht['pnl'].sum():+,.0f}")

        # Monthly
        df['exit_dt'] = pd.to_datetime(df['exit_time'])
        df['month'] = df['exit_dt'].dt.to_period('M')
        mg = df.groupby('month')

        print(f"\n{'='*100}")
        print(f"  MONTHLY PERFORMANCE")
        print(f"{'='*100}")
        print(f"  {'Month':>8} {'Trd':>4} {'W':>3} {'L':>3} {'WR%':>5} {'GrossP':>9} {'GrossL':>9} {'NetPnL':>9} {'PF':>6} {'Bal':>10} {'Ret%':>7}")
        print("  " + "-" * 95)

        rb = INIT; yearly = {}; lm = 0; tm = 0
        for mo in sorted(mg.groups.keys()):
            g = mg.get_group(mo); nt = len(g); nw = len(g[g['pnl'] > 0]); nl = nt - nw
            wr = nw/nt*100 if nt else 0
            gp2 = g[g['pnl'] > 0]['pnl'].sum() if nw else 0
            gl2 = abs(g[g['pnl'] <= 0]['pnl'].sum()) if nl else 0
            net = g['pnl'].sum(); mpf = gp2/gl2 if gl2 > 0 else 999
            sbr = rb; rb += net; mr = net/sbr*100 if sbr > 0 else 0
            tm += 1
            if net < 0: lm += 1
            y = str(mo)[:4]
            if y not in yearly: yearly[y] = {'p':0,'t':0,'w':0,'l':0,'gp':0,'gl':0,'sb':sbr}
            yearly[y]['p'] += net; yearly[y]['t'] += nt; yearly[y]['w'] += nw; yearly[y]['l'] += nl
            yearly[y]['gp'] += gp2; yearly[y]['gl'] += gl2; yearly[y]['eb'] = rb
            pfs = f"{mpf:.1f}" if mpf < 999 else "INF"
            mk = " <<" if net < 0 else ""
            print(f"  {str(mo):>8} {nt:>4} {nw:>3} {nl:>3} {wr:>4.0f}% ${gp2:>7,.0f} ${gl2:>7,.0f} ${net:>+7,.0f} {pfs:>5} ${rb:>8,.0f} {mr:>+6.1f}%{mk}")

        # Yearly
        print(f"\n{'='*100}")
        print(f"  YEARLY PERFORMANCE")
        print(f"{'='*100}")
        print(f"  {'Year':>6} {'Trd':>4} {'W':>4} {'L':>4} {'WR%':>5} {'GrossP':>10} {'GrossL':>10} {'NetPnL':>10} {'PF':>6} {'YrRet%':>8}")
        print("  " + "-" * 80)
        for y2 in sorted(yearly):
            yd = yearly[y2]; ywr = yd['w']/yd['t']*100 if yd['t'] else 0
            ypf = yd['gp']/yd['gl'] if yd['gl'] > 0 else 999
            yret = yd['p']/yd['sb']*100 if yd['sb'] > 0 else 0
            pfs = f"{ypf:.2f}" if ypf < 999 else "INF"
            print(f"  {y2:>6} {yd['t']:>3} {yd['w']:>4} {yd['l']:>4} {ywr:>4.0f}% ${yd['gp']:>8,.0f} ${yd['gl']:>8,.0f} ${yd['p']:>+8,.0f} {pfs:>5} {yret:>+7.1f}%")
        pyrs = sum(1 for v in yearly.values() if v['p'] > 0)
        print(f"\n  Loss Months: {lm}/{tm} ({lm/max(1,tm)*100:.0f}%)")
        print(f"  Profit Years: {pyrs}/{len(yearly)}")

        # Top/Bottom
        ds = df.sort_values('pnl', ascending=False)
        print(f"\n  TOP 10 TRADES")
        print("  " + "-" * 90)
        for idx, (_, r) in enumerate(ds.head(10).iterrows()):
            print(f"  {idx+1:>2} {r['direction']:>5} {r['entry_time'][:16]} -> {r['exit_time'][:16]} {r['exit_reason']:>5} ROI:{r['roi_pct']:>+7.1f}% PnL:${r['pnl']:>+8,.0f}")
        print(f"\n  BOTTOM 10 TRADES")
        print("  " + "-" * 90)
        for idx, (_, r) in enumerate(ds.tail(10).iterrows()):
            print(f"  {idx+1:>2} {r['direction']:>5} {r['entry_time'][:16]} -> {r['exit_time'][:16]} {r['exit_reason']:>5} ROI:{r['roi_pct']:>+7.1f}% PnL:${r['pnl']:>+8,.0f}")

    # ---- Save files ----
    rpt_path = os.path.join(DIR, "v25_1_FINAL_report.txt")
    trd_path = os.path.join(DIR, "v25_1_FINAL_trades.csv")
    met_path = os.path.join(DIR, "v25_1_FINAL_metrics.json")
    mon_path = os.path.join(DIR, "v25_1_FINAL_monthly.csv")
    ver_path = os.path.join(DIR, "v25_1_FINAL_30run.csv")

    if trades:
        pd.DataFrame(trades).to_csv(trd_path, index=False, encoding='utf-8-sig')

    with open(met_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Monthly CSV
    if trades:
        df['exit_dt'] = pd.to_datetime(df['exit_time'])
        df['month'] = df['exit_dt'].dt.to_period('M')
        monthly_data = []
        rb2 = INIT
        for mo in sorted(df.groupby('month').groups.keys()):
            g = df[df['month'] == mo]
            net = g['pnl'].sum()
            sbr = rb2; rb2 += net
            monthly_data.append({'month': str(mo), 'trades': len(g), 'wins': len(g[g['pnl']>0]),
                                 'pnl': round(net, 2), 'balance': round(rb2, 2),
                                 'return_pct': round(net/sbr*100, 2) if sbr > 0 else 0})
        pd.DataFrame(monthly_data).to_csv(mon_path, index=False, encoding='utf-8-sig')

    # ---- 30-run verification ----
    print(f"\n{'='*100}")
    print(f"  30-RUN VERIFICATION")
    print(f"{'='*100}")
    t0 = time.time()
    vresults = []
    for run in range(1, 31):
        m, _, _ = run_backtest(opens, highs, lows, closes, adx_arr, rsi_arr, fast_ma, slow_ma, timestamps, False)
        vresults.append({'run': run, 'final': m['final'], 'return': m['return_pct'], 'trades': m['trades'],
                         'pf': m.get('pf', 0), 'mdd': m.get('mdd', 0), 'wr': m.get('win_rate', 0)})
        if run % 10 == 0:
            print(f"  Run {run}/30 | Bal:${m['final']:,.2f} | PF:{m.get('pf',0):.2f} | MDD:{m.get('mdd',0):.1f}%")

    vdf = pd.DataFrame(vresults)
    vdf.to_csv(ver_path, index=False)
    print(f"\n  Completed in {time.time()-t0:.1f}s")
    print(f"  Balance: mean=${vdf['final'].mean():,.2f}  std=${vdf['final'].std():.4f}")
    print(f"  Return:  mean={vdf['return'].mean():+.2f}%  std={vdf['return'].std():.4f}%")
    print(f"  PF:      mean={vdf['pf'].mean():.4f}  std={vdf['pf'].std():.6f}")
    print(f"  MDD:     mean={vdf['mdd'].mean():.2f}%  std={vdf['mdd'].std():.6f}%")
    print(f"  Trades:  mean={vdf['trades'].mean():.1f}  std={vdf['trades'].std():.4f}")
    det = vdf['final'].std() < 0.01
    print(f"  Deterministic: {'YES' if det else 'NO'}")

    print(f"\n  FILES SAVED:")
    for p in [trd_path, met_path, mon_path, ver_path]:
        print(f"    {p}")

    print(f"\n{'='*100}")
    print(f"  COMPLETE")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()
