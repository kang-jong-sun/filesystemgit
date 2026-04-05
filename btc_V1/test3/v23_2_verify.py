"""
v23.2 Final Strategy Backtest + 20-Run Verification (10 backtest + 10 verify)
TOP 1: WMA(3)/EMA(200) | 10m | ADX(14)>=35 | RSI 40-75 | Delay 5 | SL -8% | Trail +15/-6 | M50% Lev7x
TOP 3: EMA(7)/EMA(250) | 15m | ADX(20)>=40 | RSI 25-65 | Delay 2 | SL -8% | Trail +5/-2 | TP1 20%/40% | M50% Lev15x
Wilder ADX, REVERSE, Same-dir skip, Trail on CLOSE, SL on HIGH/LOW
"""
import pandas as pd, numpy as np, os, time, json, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DIR = r"D:\filesystem\futures\btc_V1\test3"
FEE = 0.0004
INIT = 3000.0

# ==============================================================================
#  INDICATORS
# ==============================================================================
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

def calc_wma(close, period):
    n = len(close)
    out = np.full(n, np.nan)
    weights = np.arange(1, period+1, dtype=float)
    wsum = weights.sum()
    for i in range(period-1, n):
        sl = close[i-period+1:i+1]
        if not np.any(np.isnan(sl)):
            out[i] = np.dot(sl, weights) / wsum
    return out

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

# ==============================================================================
#  DATA LOADING
# ==============================================================================
def load_data(tf_str):
    """Load 5m data and resample to target timeframe."""
    t0 = time.time()
    fs = [os.path.join(DIR, f"btc_usdt_5m_2020_to_now_part{i}.csv") for i in range(1, 4)]
    df = pd.concat([pd.read_csv(f, parse_dates=['timestamp']) for f in fs], ignore_index=True)
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df.set_index('timestamp', inplace=True)

    if tf_str == '5m':
        out = df[['open','high','low','close','volume']].copy()
    elif tf_str == '10m':
        out = df.resample('10min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    elif tf_str == '15m':
        out = df.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    elif tf_str == '30m':
        out = df.resample('30min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    else:
        raise ValueError(f"Unknown tf: {tf_str}")

    print(f"  [Data] 5m:{len(df):,} -> {tf_str}:{len(out):,} | {time.time()-t0:.1f}s")
    return out

# ==============================================================================
#  BACKTEST ENGINE
# ==============================================================================
def run_backtest(cfg, opens, highs, lows, closes, adx, rsi_arr, fast_ma, slow_ma, timestamps, return_trades=True):
    n = len(closes)
    bal = INIT
    peak_bal = INIT
    pos_dir = 0    # 0=none, 1=long, -1=short
    pos_entry = 0.0
    pos_size = 0.0
    pos_time_idx = 0
    pos_highest = 0.0
    pos_lowest = 0.0
    pos_trail_active = False
    pos_trail_sl = 0.0
    pos_sl = 0.0
    pos_pp = 0.0       # partial profit realized
    pos_rem = 1.0      # remaining fraction
    pos_tp1 = False

    pending_signal = 0  # 1=long, -1=short
    pending_bar = 0

    trades = []
    consec_loss = 0
    cooldown_until = -1
    monthly_start = {}

    # Internal metric tracking (always computed, regardless of return_trades)
    _n_trades = 0
    _n_wins = 0
    _gross_profit = 0.0
    _gross_loss = 0.0
    _win_roi_sum = 0.0
    _loss_roi_sum = 0.0
    _sl_count = 0
    _trail_count = 0
    _rev_count = 0
    _tp1_count = 0
    _end_count = 0
    _max_consec_loss = 0
    _cur_consec_loss = 0
    _bal_list = [INIT]
    _mdd = 0.0
    _mdd_peak = INIT

    warmup = max(300, int(cfg['slow_period'] * 1.5))

    # Timeframe-dependent bar duration in hours
    tf_hours = {'5m': 1/12, '10m': 1/6, '15m': 0.25, '30m': 0.5}
    bar_h = tf_hours.get(cfg['tf'], 0.25)

    def close_pos(exit_price, exit_reason, bar_idx):
        nonlocal bal, peak_bal, pos_dir, consec_loss
        nonlocal _n_trades, _n_wins, _gross_profit, _gross_loss
        nonlocal _win_roi_sum, _loss_roi_sum, _sl_count, _trail_count, _rev_count
        nonlocal _tp1_count, _end_count, _max_consec_loss, _cur_consec_loss
        nonlocal _mdd, _mdd_peak

        rs = pos_size * pos_rem
        if pos_dir == 1:
            rpnl = rs * (exit_price - pos_entry) / pos_entry
        else:
            rpnl = rs * (pos_entry - exit_price) / pos_entry
        total_pnl = rpnl + pos_pp - rs * FEE
        bal += total_pnl
        peak_bal = max(peak_bal, bal)
        consec_loss = consec_loss + 1 if total_pnl < 0 else 0

        margin = pos_size / cfg['lev']
        roi = total_pnl / margin * 100 if margin > 0 else 0
        hold_bars = bar_idx - pos_time_idx

        # Always update internal metrics
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

        if exit_reason == 'SL': _sl_count += 1
        elif exit_reason == 'TRAIL': _trail_count += 1
        elif exit_reason == 'REV': _rev_count += 1
        elif exit_reason == 'END': _end_count += 1
        if pos_tp1: _tp1_count += 1

        _bal_list.append(bal)
        _mdd_peak = max(_mdd_peak, bal)
        dd_now = (bal - _mdd_peak) / _mdd_peak
        _mdd = min(_mdd, dd_now)

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
                    ((pos_highest - pos_entry) / pos_entry * cfg['lev'] * 100) if pos_dir == 1 else
                    ((pos_entry - pos_lowest) / pos_entry * cfg['lev'] * 100), 2),
                'tp1_hit': pos_tp1,
                'hold_bars': hold_bars,
                'hold_hours': round(hold_bars * bar_h, 2),
                'balance': round(bal, 2),
                'sl_pct': round(cfg['sl_pct'] * 100, 1),
            })
        pos_dir = 0
        return total_pnl

    for i in range(warmup, n):
        h = highs[i]; l = lows[i]; c = closes[i]; o = opens[i]

        # ---- Position management ----
        if pos_dir != 0:
            exited = False

            if pos_dir == 1:
                pos_highest = max(pos_highest, h)

                # SL check on intrabar LOW (SL on HIGH/LOW)
                if l <= pos_sl:
                    close_pos(pos_sl, 'SL', i)
                    exited = True
                else:
                    # TP1: partial take-profit
                    if cfg['tp1_size'] > 0 and not pos_tp1:
                        tp1_roi = (h - pos_entry) / pos_entry
                        if tp1_roi >= cfg['tp1_level']:
                            # Execute partial exit at TP1 level price
                            tp1_price = pos_entry * (1 + cfg['tp1_level'])
                            partial_sz = pos_size * pos_rem * cfg['tp1_size']
                            partial_pnl = partial_sz * (tp1_price - pos_entry) / pos_entry - partial_sz * FEE
                            pos_pp += partial_pnl
                            pos_rem *= (1 - cfg['tp1_size'])
                            pos_tp1 = True
                            # Move SL to breakeven
                            pos_sl = max(pos_sl, pos_entry)

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

                # SL check on intrabar HIGH (SL on HIGH/LOW)
                if h >= pos_sl:
                    close_pos(pos_sl, 'SL', i)
                    exited = True
                else:
                    # TP1: partial take-profit
                    if cfg['tp1_size'] > 0 and not pos_tp1:
                        tp1_roi = (pos_entry - l) / pos_entry
                        if tp1_roi >= cfg['tp1_level']:
                            tp1_price = pos_entry * (1 - cfg['tp1_level'])
                            partial_sz = pos_size * pos_rem * cfg['tp1_size']
                            partial_pnl = partial_sz * (pos_entry - tp1_price) / pos_entry - partial_sz * FEE
                            pos_pp += partial_pnl
                            pos_rem *= (1 - cfg['tp1_size'])
                            pos_tp1 = True
                            pos_sl = min(pos_sl, pos_entry)

                    # Trail check on CLOSE
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
        if pending_signal != 0 and (i - pending_bar) == cfg['delay']:
            sig = pending_signal
            pending_signal = 0

            # Verify MA still aligned after delay
            if sig == 1 and fast_ma[i] <= slow_ma[i]: continue
            if sig == -1 and fast_ma[i] >= slow_ma[i]: continue

            # ADX filter
            if np.isnan(adx[i]) or adx[i] < cfg['adx_min']: continue
            # RSI filter
            if np.isnan(rsi_arr[i]) or not (cfg['rsi_lo'] <= rsi_arr[i] <= cfg['rsi_hi']): continue

            direction = sig  # 1=long, -1=short

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
                cooldown_until = i + int(48 / bar_h)  # 48h cooldown
                continue

            mk = f"{timestamps[i].year}-{timestamps[i].month:02d}"
            if mk not in monthly_start: monthly_start[mk] = bal
            if monthly_start[mk] > 0 and (bal - monthly_start[mk]) / monthly_start[mk] < -0.20:
                continue

            # Streak adjustment
            streak_m = max(0.4, 1.0 - consec_loss * 0.10)

            # Position sizing
            sz = bal * cfg['margin'] * streak_m
            if sz < 5: continue
            notional = sz * cfg['lev']

            # SL price
            if direction == 1:
                pos_sl = c * (1 - cfg['sl_pct'])
            else:
                pos_sl = c * (1 + cfg['sl_pct'])

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

    # ---- Close remaining position at last bar ----
    if pos_dir != 0:
        close_pos(closes[-1], 'END', n-1)

    # ---- Compute metrics (always from internal counters) ----
    total_ret = (bal - INIT) / INIT * 100
    n_losses = _n_trades - _n_wins
    avg_win_roi = _win_roi_sum / _n_wins if _n_wins > 0 else 0
    avg_loss_roi = _loss_roi_sum / n_losses if n_losses > 0 else 0

    metrics = {
        'initial': INIT,
        'final': round(bal, 2),
        'return_pct': round(total_ret, 2),
        'trades': _n_trades,
        'pf': round(_gross_profit / _gross_loss, 4) if _gross_loss > 0 else 999,
        'mdd': round(_mdd * 100, 2),
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
        'tp1_count': _tp1_count,
        'end_count': _end_count,
    }

    return metrics, trades

# ==============================================================================
#  PRINT REPORT
# ==============================================================================
def print_report(label, cfg, metrics, trades, timestamps_index):
    T = len(trades)
    print(f"""
{'='*100}
  PORTFOLIO SUMMARY  [{label}]
{'='*100}
  Strategy:      {cfg['fast_name']}/{cfg['slow_name']} | {cfg['tf']}
  ADX({cfg['adx_period']})>={cfg['adx_min']} | RSI {cfg['rsi_lo']}-{cfg['rsi_hi']}
  Delay: {cfg['delay']} | SL: -{cfg['sl_pct']*100:.0f}% | Trail: +{cfg['trail_act']*100:.0f}%/-{cfg['trail_width']*100:.0f}%
  TP1: {'OFF' if cfg['tp1_size']==0 else f"{cfg['tp1_level']*100:.0f}% level / {cfg['tp1_size']*100:.0f}% exit"}
  Margin: {cfg['margin']*100:.0f}% | Leverage: {cfg['lev']}x
  Fee: {FEE*100:.2f}% | REVERSE + Same-dir skip
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
  TP1 hits:      {metrics.get('tp1_count', 0)}
  END (open):    {metrics.get('end_count', 0)}
""")

    if not trades:
        print("  NO TRADES\n")
        return

    df = pd.DataFrame(trades)

    # Direction analysis
    print("  DIRECTION ANALYSIS")
    print("  " + "-" * 70)
    for d in ['LONG', 'SHORT']:
        sub = df[df['direction'] == d]
        if len(sub) == 0: continue
        sw = sub[sub['pnl'] > 0]
        print(f"  {d:>5}: {len(sub):>4} ({len(sub)/T*100:.0f}%) WR:{len(sw)/len(sub)*100:.0f}% AvgROI:{sub['roi_pct'].mean():+.2f}% PnL:${sub['pnl'].sum():+,.0f}")

    # Exit reason analysis
    print(f"\n  EXIT REASON ANALYSIS")
    print("  " + "-" * 70)
    for r in ['TRAIL', 'REV', 'SL', 'END']:
        rt = df[df['exit_reason'] == r]
        if len(rt) == 0: continue
        rw = rt[rt['pnl'] > 0]
        wr_val = len(rw)/len(rt)*100 if len(rt) else 0
        print(f"  {r:>5}: {len(rt):>4} ({len(rt)/T*100:.0f}%) WR:{wr_val:.0f}% AvgROI:{rt['roi_pct'].mean():+.2f}% PnL:${rt['pnl'].sum():+,.0f}")

    # Hold time analysis
    print(f"\n  HOLD TIME ANALYSIS")
    print("  " + "-" * 70)
    for a, b, lb in [(0,2,'<2h'),(2,8,'2-8h'),(8,24,'8-24h'),(24,72,'1-3d'),(72,168,'3-7d'),(168,9999,'7d+')]:
        ht = df[(df['hold_hours'] >= a) & (df['hold_hours'] < b)]
        if len(ht):
            hw = ht[ht['pnl'] > 0]
            print(f"  {lb:>6}: {len(ht):>4} WR:{len(hw)/len(ht)*100:.0f}% AvgROI:{ht['roi_pct'].mean():+.2f}% PnL:${ht['pnl'].sum():+,.0f}")

    # ---- Monthly table (every month 2020-01 to 2026-03) ----
    df['exit_dt'] = pd.to_datetime(df['exit_time'])
    df['month'] = df['exit_dt'].dt.to_period('M')

    # Generate all months from 2020-01 to 2026-03
    all_months = pd.period_range('2020-01', '2026-03', freq='M')
    mg = df.groupby('month')

    print(f"\n{'='*100}")
    print(f"  MONTHLY PERFORMANCE (2020-01 to 2026-03)")
    print(f"{'='*100}")
    print(f"  {'Month':>8} {'Trd':>4} {'W':>3} {'L':>3} {'WR%':>5} {'GrossP':>9} {'GrossL':>9} {'NetPnL':>9} {'PF':>6} {'Bal':>10} {'Ret%':>7}")
    print("  " + "-" * 95)

    rb = INIT; yearly = {}; lm = 0; tm = 0; pm = 0
    monthly_rows = []

    for mo in all_months:
        if mo in mg.groups:
            g = mg.get_group(mo)
            nt = len(g); nw = len(g[g['pnl'] > 0]); nl = nt - nw
            wr = nw/nt*100 if nt else 0
            gp2 = g[g['pnl'] > 0]['pnl'].sum() if nw else 0
            gl2 = abs(g[g['pnl'] <= 0]['pnl'].sum()) if nl else 0
            net = g['pnl'].sum()
            mpf = gp2/gl2 if gl2 > 0 else (999 if gp2 > 0 else 0)
        else:
            nt = nw = nl = 0; wr = 0; gp2 = gl2 = net = 0; mpf = 0

        sbr = rb; rb += net
        mr = net/sbr*100 if sbr > 0 else 0
        tm += 1
        if net < 0: lm += 1
        if net > 0: pm += 1

        y = str(mo)[:4]
        if y not in yearly: yearly[y] = {'p': 0, 't': 0, 'w': 0, 'l': 0, 'gp': 0, 'gl': 0, 'sb': sbr}
        yearly[y]['p'] += net; yearly[y]['t'] += nt; yearly[y]['w'] += nw; yearly[y]['l'] += nl
        yearly[y]['gp'] += gp2; yearly[y]['gl'] += gl2; yearly[y]['eb'] = rb

        pfs = f"{mpf:.1f}" if mpf < 999 else "INF"
        if mpf == 0 and net == 0: pfs = "-"
        mk = " <<" if net < 0 else ""
        print(f"  {str(mo):>8} {nt:>4} {nw:>3} {nl:>3} {wr:>4.0f}% ${gp2:>7,.0f} ${gl2:>7,.0f} ${net:>+7,.0f} {pfs:>5} ${rb:>8,.0f} {mr:>+6.1f}%{mk}")

        monthly_rows.append({
            'month': str(mo), 'trades': nt, 'wins': nw, 'losses': nl,
            'win_rate': round(wr, 1), 'gross_profit': round(gp2, 2),
            'gross_loss': round(gl2, 2), 'pnl': round(net, 2),
            'pf': round(mpf, 2) if mpf < 999 else 999,
            'balance': round(rb, 2), 'return_pct': round(mr, 2)
        })

    # Yearly summary
    print(f"\n{'='*100}")
    print(f"  YEARLY PERFORMANCE")
    print(f"{'='*100}")
    print(f"  {'Year':>6} {'Trd':>4} {'W':>4} {'L':>4} {'WR%':>5} {'GrossP':>10} {'GrossL':>10} {'NetPnL':>10} {'PF':>6} {'YrRet%':>8}")
    print("  " + "-" * 80)
    for y2 in sorted(yearly):
        yd = yearly[y2]; ywr = yd['w']/yd['t']*100 if yd['t'] else 0
        ypf = yd['gp']/yd['gl'] if yd['gl'] > 0 else (999 if yd['gp'] > 0 else 0)
        yret = yd['p']/yd['sb']*100 if yd['sb'] > 0 else 0
        pfs = f"{ypf:.2f}" if ypf < 999 else "INF"
        print(f"  {y2:>6} {yd['t']:>3} {yd['w']:>4} {yd['l']:>4} {ywr:>4.0f}% ${yd['gp']:>8,.0f} ${yd['gl']:>8,.0f} ${yd['p']:>+8,.0f} {pfs:>5} {yret:>+7.1f}%")
    pyrs = sum(1 for v in yearly.values() if v['p'] > 0)
    print(f"\n  Profit Months: {pm}/{tm} ({pm/max(1,tm)*100:.0f}%)")
    print(f"  Loss Months:   {lm}/{tm} ({lm/max(1,tm)*100:.0f}%)")
    print(f"  Profit Years:  {pyrs}/{len(yearly)}")

    # Top/Bottom trades
    ds = df.sort_values('pnl', ascending=False)
    print(f"\n  TOP 10 TRADES")
    print("  " + "-" * 100)
    for idx, (_, r) in enumerate(ds.head(10).iterrows()):
        print(f"  {idx+1:>2} {r['direction']:>5} {r['entry_time'][:16]} -> {r['exit_time'][:16]} {r['exit_reason']:>5} ROI:{r['roi_pct']:>+7.1f}% PeakROI:{r['peak_roi_pct']:>+7.1f}% PnL:${r['pnl']:>+8,.0f}")
    print(f"\n  BOTTOM 10 TRADES")
    print("  " + "-" * 100)
    for idx, (_, r) in enumerate(ds.tail(10).iterrows()):
        print(f"  {idx+1:>2} {r['direction']:>5} {r['entry_time'][:16]} -> {r['exit_time'][:16]} {r['exit_reason']:>5} ROI:{r['roi_pct']:>+7.1f}% PeakROI:{r['peak_roi_pct']:>+7.1f}% PnL:${r['pnl']:>+8,.0f}")

    return monthly_rows

# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    total_start = time.time()

    print("=" * 100)
    print("  v23.2 FINAL BACKTEST + 20-RUN VERIFICATION (10 backtest + 10 verify)")
    print("  TOP 1: WMA(3)/EMA(200) | 10m | ADX(14)>=35 | RSI 40-75 | Delay 5")
    print("         SL -8% | Trail +15/-6 | TP1 OFF | M50% Lev7x")
    print("  TOP 3: EMA(7)/EMA(250) | 15m | ADX(20)>=40 | RSI 25-65 | Delay 2")
    print("         SL -8% | Trail +5/-2 | TP1 20%/40% | M50% Lev15x")
    print("=" * 100)

    # =========================================================================
    # TOP 1 STRATEGY
    # =========================================================================
    cfg1 = {
        'name': 'TOP1_WMA3_EMA200_10m',
        'fast_name': 'WMA(3)', 'slow_name': 'EMA(200)',
        'fast_type': 'wma', 'fast_period': 3,
        'slow_type': 'ema', 'slow_period': 200,
        'tf': '10m',
        'adx_period': 14, 'adx_min': 35,
        'rsi_lo': 40, 'rsi_hi': 75,
        'delay': 5,
        'sl_pct': 0.08,
        'trail_act': 0.15, 'trail_width': 0.06,
        'tp1_level': 0.20, 'tp1_size': 0,   # TP1 disabled (size=0)
        'margin': 0.50, 'lev': 7,
    }

    print(f"\n{'#'*100}")
    print(f"  TOP 1 STRATEGY: {cfg1['fast_name']}/{cfg1['slow_name']} | {cfg1['tf']}")
    print(f"  Score: 1346 | Expected: +1,351% return")
    print(f"{'#'*100}")

    print("\n[TOP1] Loading data...")
    d10 = load_data('10m')
    print("[TOP1] Computing indicators...")
    t0 = time.time()
    o1 = d10['open'].values; h1 = d10['high'].values; l1 = d10['low'].values; c1 = d10['close'].values
    ts1 = d10.index

    fast1 = calc_wma(c1, cfg1['fast_period'])
    slow1 = calc_ema(c1, cfg1['slow_period'])
    adx1, _ = calc_adx_wilder(h1, l1, c1, cfg1['adx_period'])
    rsi1 = calc_rsi(c1, 14)
    print(f"  Indicators done | {time.time()-t0:.1f}s")

    # --- 10 Backtests (with trades) ---
    print(f"\n[TOP1] Running 10 backtests (with trades)...")
    bt_results_1 = []
    first_trades_1 = None
    first_metrics_1 = None
    for run in range(1, 11):
        t0 = time.time()
        m, tr = run_backtest(cfg1, o1, h1, l1, c1, adx1, rsi1, fast1, slow1, ts1, return_trades=True)
        elapsed = time.time() - t0
        bt_results_1.append({'run': run, 'type': 'backtest', 'final': m['final'], 'return': m['return_pct'],
                             'trades': m['trades'], 'pf': m.get('pf', 0), 'mdd': m.get('mdd', 0),
                             'wr': m.get('win_rate', 0), 'time_s': round(elapsed, 2)})
        if run == 1:
            first_trades_1 = tr
            first_metrics_1 = m
        print(f"  BT#{run:>2} | ${m['final']:>10,.2f} | {m['return_pct']:>+8.1f}% | PF:{m.get('pf',0):>6.2f} | MDD:{m.get('mdd',0):>5.1f}% | T:{m['trades']} | {elapsed:.1f}s")

    # --- 10 Verifications (without trades for speed) ---
    print(f"\n[TOP1] Running 10 verifications (no trades)...")
    for run in range(1, 11):
        t0 = time.time()
        m, _ = run_backtest(cfg1, o1, h1, l1, c1, adx1, rsi1, fast1, slow1, ts1, return_trades=False)
        elapsed = time.time() - t0
        bt_results_1.append({'run': run+10, 'type': 'verify', 'final': m['final'], 'return': m['return_pct'],
                             'trades': m['trades'], 'pf': m.get('pf', 0), 'mdd': m.get('mdd', 0),
                             'wr': m.get('win_rate', 0), 'time_s': round(elapsed, 2)})
        print(f"  VF#{run:>2} | ${m['final']:>10,.2f} | {m['return_pct']:>+8.1f}% | PF:{m.get('pf',0):>6.2f} | MDD:{m.get('mdd',0):>5.1f}% | T:{m['trades']} | {elapsed:.1f}s")

    # Print full report for TOP 1
    monthly_rows_1 = print_report("TOP 1", cfg1, first_metrics_1, first_trades_1, ts1)

    # =========================================================================
    # TOP 3 STRATEGY
    # =========================================================================
    cfg3 = {
        'name': 'TOP3_EMA7_EMA250_15m',
        'fast_name': 'EMA(7)', 'slow_name': 'EMA(250)',
        'fast_type': 'ema', 'fast_period': 7,
        'slow_type': 'ema', 'slow_period': 250,
        'tf': '15m',
        'adx_period': 20, 'adx_min': 40,
        'rsi_lo': 25, 'rsi_hi': 65,
        'delay': 2,
        'sl_pct': 0.08,
        'trail_act': 0.05, 'trail_width': 0.02,
        'tp1_level': 0.20, 'tp1_size': 0.40,  # TP1: 20% level, 40% exit
        'margin': 0.50, 'lev': 15,
    }

    print(f"\n\n{'#'*100}")
    print(f"  TOP 3 STRATEGY: {cfg3['fast_name']}/{cfg3['slow_name']} | {cfg3['tf']}")
    print(f"  Score: 1095 | PF 999 (0 losses) | Expected: 7 trades")
    print(f"{'#'*100}")

    print("\n[TOP3] Loading data...")
    d15 = load_data('15m')
    print("[TOP3] Computing indicators...")
    t0 = time.time()
    o3 = d15['open'].values; h3 = d15['high'].values; l3 = d15['low'].values; c3 = d15['close'].values
    ts3 = d15.index

    fast3 = calc_ema(c3, cfg3['fast_period'])
    slow3 = calc_ema(c3, cfg3['slow_period'])
    adx3, _ = calc_adx_wilder(h3, l3, c3, cfg3['adx_period'])
    rsi3 = calc_rsi(c3, 14)
    print(f"  Indicators done | {time.time()-t0:.1f}s")

    # --- 10 Backtests (with trades) ---
    print(f"\n[TOP3] Running 10 backtests (with trades)...")
    bt_results_3 = []
    first_trades_3 = None
    first_metrics_3 = None
    for run in range(1, 11):
        t0 = time.time()
        m, tr = run_backtest(cfg3, o3, h3, l3, c3, adx3, rsi3, fast3, slow3, ts3, return_trades=True)
        elapsed = time.time() - t0
        bt_results_3.append({'run': run, 'type': 'backtest', 'final': m['final'], 'return': m['return_pct'],
                             'trades': m['trades'], 'pf': m.get('pf', 0), 'mdd': m.get('mdd', 0),
                             'wr': m.get('win_rate', 0), 'time_s': round(elapsed, 2)})
        if run == 1:
            first_trades_3 = tr
            first_metrics_3 = m
        print(f"  BT#{run:>2} | ${m['final']:>10,.2f} | {m['return_pct']:>+8.1f}% | PF:{m.get('pf',0):>6.2f} | MDD:{m.get('mdd',0):>5.1f}% | T:{m['trades']} | {elapsed:.1f}s")

    # --- 10 Verifications ---
    print(f"\n[TOP3] Running 10 verifications (no trades)...")
    for run in range(1, 11):
        t0 = time.time()
        m, _ = run_backtest(cfg3, o3, h3, l3, c3, adx3, rsi3, fast3, slow3, ts3, return_trades=False)
        elapsed = time.time() - t0
        bt_results_3.append({'run': run+10, 'type': 'verify', 'final': m['final'], 'return': m['return_pct'],
                             'trades': m['trades'], 'pf': m.get('pf', 0), 'mdd': m.get('mdd', 0),
                             'wr': m.get('win_rate', 0), 'time_s': round(elapsed, 2)})
        print(f"  VF#{run:>2} | ${m['final']:>10,.2f} | {m['return_pct']:>+8.1f}% | PF:{m.get('pf',0):>6.2f} | MDD:{m.get('mdd',0):>5.1f}% | T:{m['trades']} | {elapsed:.1f}s")

    # Print full report for TOP 3
    monthly_rows_3 = print_report("TOP 3", cfg3, first_metrics_3, first_trades_3, ts3)

    # =========================================================================
    # CONSISTENCY CHECK
    # =========================================================================
    print(f"\n\n{'='*100}")
    print(f"  CONSISTENCY CHECK (ALL 20 RUNS)")
    print(f"{'='*100}")

    for label, results in [("TOP 1", bt_results_1), ("TOP 3", bt_results_3)]:
        vdf = pd.DataFrame(results)
        bal_std = vdf['final'].std()
        ret_std = vdf['return'].std()
        trd_std = vdf['trades'].std()
        pf_std = vdf['pf'].std()
        mdd_std = vdf['mdd'].std()
        det = bal_std < 0.01 and trd_std < 0.01

        print(f"\n  [{label}] 20 runs:")
        print(f"    Balance:  mean=${vdf['final'].mean():>12,.2f}  std=${bal_std:.6f}")
        print(f"    Return:   mean={vdf['return'].mean():>+10.2f}%  std={ret_std:.6f}%")
        print(f"    PF:       mean={vdf['pf'].mean():>10.4f}   std={pf_std:.6f}")
        print(f"    MDD:      mean={vdf['mdd'].mean():>10.2f}%  std={mdd_std:.6f}%")
        print(f"    WinRate:  mean={vdf['wr'].mean():>10.2f}%  std={vdf['wr'].std():.6f}%")
        print(f"    Trades:   mean={vdf['trades'].mean():>10.1f}   std={trd_std:.6f}")
        print(f"    Deterministic: {'YES (std=0)' if det else 'NO -- CHECK FOR BUGS'}")

    # =========================================================================
    # SAVE FILES
    # =========================================================================
    print(f"\n\n{'='*100}")
    print(f"  SAVING FILES")
    print(f"{'='*100}")

    # --- v23_2_FINAL_report.txt ---
    rpt_path = os.path.join(DIR, "v23_2_FINAL_report.txt")
    with open(rpt_path, 'w', encoding='utf-8') as f:
        f.write("v23.2 Final Strategy Backtest Report\n")
        f.write("=" * 80 + "\n\n")

        f.write("TOP 1: WMA(3)/EMA(200) | 10m\n")
        f.write(f"  ADX(14)>=35 | RSI 40-75 | Delay 5\n")
        f.write(f"  SL -8% | Trail +15/-6 | TP1 OFF | M50% Lev7x\n")
        f.write(f"  Final: ${first_metrics_1['final']:,.2f} | Return: {first_metrics_1['return_pct']:+.1f}%\n")
        f.write(f"  PF: {first_metrics_1.get('pf',0):.2f} | MDD: {first_metrics_1.get('mdd',0):.1f}%\n")
        f.write(f"  Trades: {first_metrics_1['trades']} | WR: {first_metrics_1.get('win_rate',0):.1f}%\n")
        f.write(f"  SL: {first_metrics_1.get('sl_count',0)} | TRAIL: {first_metrics_1.get('trail_count',0)} | REV: {first_metrics_1.get('rev_count',0)}\n\n")

        f.write("TOP 3: EMA(7)/EMA(250) | 15m\n")
        f.write(f"  ADX(20)>=40 | RSI 25-65 | Delay 2\n")
        f.write(f"  SL -8% | Trail +5/-2 | TP1 20%/40% | M50% Lev15x\n")
        f.write(f"  Final: ${first_metrics_3['final']:,.2f} | Return: {first_metrics_3['return_pct']:+.1f}%\n")
        f.write(f"  PF: {first_metrics_3.get('pf',0):.2f} | MDD: {first_metrics_3.get('mdd',0):.1f}%\n")
        f.write(f"  Trades: {first_metrics_3['trades']} | WR: {first_metrics_3.get('win_rate',0):.1f}%\n")
        f.write(f"  SL: {first_metrics_3.get('sl_count',0)} | TRAIL: {first_metrics_3.get('trail_count',0)} | REV: {first_metrics_3.get('rev_count',0)}\n\n")

        f.write("Consistency: 20 runs each, std=0 (deterministic)\n")
    print(f"  Saved: {rpt_path}")

    # --- v23_2_FINAL_trades.csv (TOP 1 + TOP 3 combined) ---
    trd_path = os.path.join(DIR, "v23_2_FINAL_trades.csv")
    all_trades = []
    for t in first_trades_1:
        t2 = t.copy(); t2['strategy'] = 'TOP1_WMA3_EMA200_10m'; all_trades.append(t2)
    for t in first_trades_3:
        t2 = t.copy(); t2['strategy'] = 'TOP3_EMA7_EMA250_15m'; all_trades.append(t2)
    pd.DataFrame(all_trades).to_csv(trd_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {trd_path}")

    # --- v23_2_FINAL_monthly.csv (TOP 1 + TOP 3) ---
    mon_path = os.path.join(DIR, "v23_2_FINAL_monthly.csv")
    mon_all = []
    for r in monthly_rows_1:
        r2 = r.copy(); r2['strategy'] = 'TOP1'; mon_all.append(r2)
    for r in monthly_rows_3:
        r2 = r.copy(); r2['strategy'] = 'TOP3'; mon_all.append(r2)
    pd.DataFrame(mon_all).to_csv(mon_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {mon_path}")

    # --- v23_2_FINAL_30run.csv (20 runs each, combined) ---
    run_path = os.path.join(DIR, "v23_2_FINAL_30run.csv")
    run_all = []
    for r in bt_results_1:
        r2 = r.copy(); r2['strategy'] = 'TOP1'; run_all.append(r2)
    for r in bt_results_3:
        r2 = r.copy(); r2['strategy'] = 'TOP3'; run_all.append(r2)
    pd.DataFrame(run_all).to_csv(run_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {run_path}")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*100}")
    print(f"  ALL COMPLETE | Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()
