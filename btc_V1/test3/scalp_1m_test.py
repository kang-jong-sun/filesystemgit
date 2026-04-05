"""
1-Minute Scalping Feasibility Test
- 24 months of 1m BTC data (1,036,800 bars)
- Test multiple fast scalping strategies
- Key question: can 1m generate positive expectancy after fees?
"""
import pandas as pd, numpy as np, os, time, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
DIR = r"D:\filesystem\futures\btc_V1\test3"
FEE = 0.0004  # 0.04% per side
INIT = 3000.0

def load_1m():
    t0 = time.time()
    fs = [os.path.join(DIR, f"btc_usdt_1m_24months_part{i}.csv") for i in range(1,5)]
    df = pd.concat([pd.read_csv(f, parse_dates=['timestamp']) for f in fs], ignore_index=True)
    df.sort_values('timestamp', inplace=True); df.set_index('timestamp', inplace=True)
    print(f"[1m Data] {len(df):,} bars | {df.index[0]} ~ {df.index[-1]} | {time.time()-t0:.1f}s")
    return df

def ema_np(c, p):
    out = np.empty_like(c); out[0] = c[0]; m = 2.0/(p+1)
    for i in range(1, len(c)): out[i] = c[i]*m + out[i-1]*(1-m)
    return out

def rsi_np(c, p=14):
    n = len(c); out = np.full(n, 50.0)
    g = np.zeros(n); l = np.zeros(n)
    for i in range(1, n):
        d = c[i]-c[i-1]
        if d > 0: g[i] = d
        else: l[i] = -d
    ag = np.zeros(n); al = np.zeros(n)
    ag[p] = np.mean(g[1:p+1]); al[p] = np.mean(l[1:p+1])
    for i in range(p+1, n):
        ag[i] = (ag[i-1]*(p-1)+g[i])/p; al[i] = (al[i-1]*(p-1)+l[i])/p
    for i in range(p, n):
        if al[i] == 0: out[i] = 100
        else: out[i] = 100-100/(1+ag[i]/al[i])
    return out

def scalp_backtest(highs, lows, closes, fast_ma, slow_ma, rsi_arr, params):
    """Fast scalping backtest on 1m bars"""
    n = len(closes)
    bal = INIT
    peak_bal = INIT
    pos = 0  # 0=none, 1=long, -1=short
    entry_p = 0.0
    entry_i = 0
    trades = 0; wins = 0; gross_p = 0.0; gross_l = 0.0; sl_hits = 0
    max_dd = 0.0

    sl_pct = params['sl']
    tp_pct = params['tp']
    trail_act = params['trail_act']
    trail_w = params['trail_w']
    adx_min = params.get('adx_min', 0)
    rsi_lo = params['rsi_lo']
    rsi_hi = params['rsi_hi']
    margin = params['margin']
    lev = params['lev']
    max_hold = params.get('max_hold', 999999)  # max bars to hold

    trail_active = False
    trail_sl = 0.0
    hi_since = 0.0
    lo_since = 0.0

    warmup = max(200, params.get('warmup', 200))

    for i in range(warmup, n):
        h = highs[i]; l = lows[i]; c = closes[i]

        # Position management
        if pos != 0:
            hold = i - entry_i
            if pos == 1:
                hi_since = max(hi_since, h)
                roi = (c - entry_p) / entry_p

                # SL
                if l <= entry_p * (1 - sl_pct):
                    pnl_p = entry_p * (1 - sl_pct)
                    pnl = (pnl_p - entry_p) / entry_p * margin * lev * bal - margin * lev * bal * FEE
                    bal += pnl; gross_l += abs(pnl); trades += 1; sl_hits += 1; pos = 0
                # TP
                elif tp_pct > 0 and h >= entry_p * (1 + tp_pct):
                    pnl_p = entry_p * (1 + tp_pct)
                    pnl = (pnl_p - entry_p) / entry_p * margin * lev * bal - margin * lev * bal * FEE
                    bal += pnl; gross_p += pnl; trades += 1; wins += 1; pos = 0
                # Trail
                elif roi >= trail_act:
                    trail_active = True
                    new_tsl = hi_since * (1 - trail_w)
                    trail_sl = max(trail_sl, new_tsl)
                    if c <= trail_sl:
                        pnl = (c - entry_p) / entry_p * margin * lev * bal - margin * lev * bal * FEE
                        bal += pnl
                        if pnl > 0: gross_p += pnl; wins += 1
                        else: gross_l += abs(pnl)
                        trades += 1; pos = 0
                # Max hold
                elif hold >= max_hold:
                    pnl = (c - entry_p) / entry_p * margin * lev * bal - margin * lev * bal * FEE
                    bal += pnl
                    if pnl > 0: gross_p += pnl; wins += 1
                    else: gross_l += abs(pnl)
                    trades += 1; pos = 0

            elif pos == -1:
                lo_since = min(lo_since, l)
                roi = (entry_p - c) / entry_p

                if h >= entry_p * (1 + sl_pct):
                    pnl_p = entry_p * (1 + sl_pct)
                    pnl = (entry_p - pnl_p) / entry_p * margin * lev * bal - margin * lev * bal * FEE
                    bal += pnl; gross_l += abs(pnl); trades += 1; sl_hits += 1; pos = 0
                elif tp_pct > 0 and l <= entry_p * (1 - tp_pct):
                    pnl_p = entry_p * (1 - tp_pct)
                    pnl = (entry_p - pnl_p) / entry_p * margin * lev * bal - margin * lev * bal * FEE
                    bal += pnl; gross_p += pnl; trades += 1; wins += 1; pos = 0
                elif roi >= trail_act:
                    trail_active = True
                    new_tsl = lo_since * (1 + trail_w)
                    trail_sl = min(trail_sl, new_tsl)
                    if c >= trail_sl:
                        pnl = (entry_p - c) / entry_p * margin * lev * bal - margin * lev * bal * FEE
                        bal += pnl
                        if pnl > 0: gross_p += pnl; wins += 1
                        else: gross_l += abs(pnl)
                        trades += 1; pos = 0
                elif hold >= max_hold:
                    pnl = (entry_p - c) / entry_p * margin * lev * bal - margin * lev * bal * FEE
                    bal += pnl
                    if pnl > 0: gross_p += pnl; wins += 1
                    else: gross_l += abs(pnl)
                    trades += 1; pos = 0

            peak_bal = max(peak_bal, bal)
            dd = (bal - peak_bal) / peak_bal if peak_bal > 0 else 0
            max_dd = min(max_dd, dd)
            if bal <= 0: break
            continue  # skip entry if already in position

        # Entry
        if pos == 0 and i > warmup + 1:
            if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]): continue
            if np.isnan(rsi_arr[i]): continue
            if not (rsi_lo <= rsi_arr[i] <= rsi_hi): continue

            # Cross detection
            cross_up = fast_ma[i-1] <= slow_ma[i-1] and fast_ma[i] > slow_ma[i]
            cross_down = fast_ma[i-1] >= slow_ma[i-1] and fast_ma[i] < slow_ma[i]

            if cross_up:
                pos = 1; entry_p = c; entry_i = i
                hi_since = c; lo_since = c; trail_active = False
                trail_sl = c * (1 - sl_pct)
                bal -= margin * lev * bal * FEE  # entry fee

            elif cross_down:
                pos = -1; entry_p = c; entry_i = i
                hi_since = c; lo_since = c; trail_active = False
                trail_sl = c * (1 + sl_pct)
                bal -= margin * lev * bal * FEE

    pf = gross_p / gross_l if gross_l > 0 else 999
    wr = wins / trades * 100 if trades > 0 else 0
    ret = (bal - INIT) / INIT * 100

    return {
        'final': round(bal, 2), 'return': round(ret, 2), 'pf': round(pf, 4),
        'mdd': round(max_dd * 100, 2), 'trades': trades, 'wins': wins, 'wr': round(wr, 1),
        'sl': sl_hits, 'gp': round(gross_p, 2), 'gl': round(gross_l, 2),
        'avg_trades_day': round(trades / (len(closes) / 1440), 1),
    }


def main():
    print("=" * 110)
    print("  1-MINUTE SCALPING FEASIBILITY TEST")
    print("  BTC/USDT | 24 months | 1,036,800 bars | Fee 0.04%/side")
    print("=" * 110)

    df = load_1m()
    c = df['close'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)

    # Pre-compute MAs and RSI
    print("\n[Indicators]...")
    t0 = time.time()
    ema3 = ema_np(c, 3); ema5 = ema_np(c, 5); ema7 = ema_np(c, 7)
    ema10 = ema_np(c, 10); ema20 = ema_np(c, 20); ema50 = ema_np(c, 50)
    ema100 = ema_np(c, 100); ema200 = ema_np(c, 200)
    rsi14 = rsi_np(c, 14); rsi7 = rsi_np(c, 7)
    print(f"  Done | {time.time()-t0:.1f}s")

    # Strategy definitions
    strategies = [
        # Ultra-fast scalp
        {'name': 'S1: EMA3/EMA20 TP0.3% SL0.5%', 'fast': ema3, 'slow': ema20, 'rsi': rsi14,
         'sl': 0.005, 'tp': 0.003, 'trail_act': 99, 'trail_w': 99, 'rsi_lo': 30, 'rsi_hi': 70,
         'margin': 0.20, 'lev': 10, 'max_hold': 60},  # max 1h

        {'name': 'S2: EMA5/EMA50 TP0.5% SL0.8%', 'fast': ema5, 'slow': ema50, 'rsi': rsi14,
         'sl': 0.008, 'tp': 0.005, 'trail_act': 99, 'trail_w': 99, 'rsi_lo': 35, 'rsi_hi': 65,
         'margin': 0.20, 'lev': 10, 'max_hold': 120},

        {'name': 'S3: EMA7/EMA100 TP1% SL1%', 'fast': ema7, 'slow': ema100, 'rsi': rsi14,
         'sl': 0.01, 'tp': 0.01, 'trail_act': 99, 'trail_w': 99, 'rsi_lo': 30, 'rsi_hi': 70,
         'margin': 0.20, 'lev': 10, 'max_hold': 240},

        # Trail-based scalp (no TP, trail captures)
        {'name': 'S4: EMA3/EMA20 Trail+0.3/-0.2 SL0.5%', 'fast': ema3, 'slow': ema20, 'rsi': rsi14,
         'sl': 0.005, 'tp': 0, 'trail_act': 0.003, 'trail_w': 0.002, 'rsi_lo': 30, 'rsi_hi': 70,
         'margin': 0.20, 'lev': 10, 'max_hold': 60},

        {'name': 'S5: EMA5/EMA50 Trail+0.5/-0.3 SL0.8%', 'fast': ema5, 'slow': ema50, 'rsi': rsi14,
         'sl': 0.008, 'tp': 0, 'trail_act': 0.005, 'trail_w': 0.003, 'rsi_lo': 35, 'rsi_hi': 65,
         'margin': 0.20, 'lev': 10, 'max_hold': 120},

        {'name': 'S6: EMA7/EMA100 Trail+1/-0.5 SL1.5%', 'fast': ema7, 'slow': ema100, 'rsi': rsi14,
         'sl': 0.015, 'tp': 0, 'trail_act': 0.01, 'trail_w': 0.005, 'rsi_lo': 30, 'rsi_hi': 70,
         'margin': 0.20, 'lev': 10, 'max_hold': 360},

        # Wider scalp (more like micro-swing)
        {'name': 'S7: EMA5/EMA100 TP2% SL1% M30%', 'fast': ema5, 'slow': ema100, 'rsi': rsi14,
         'sl': 0.01, 'tp': 0.02, 'trail_act': 99, 'trail_w': 99, 'rsi_lo': 35, 'rsi_hi': 65,
         'margin': 0.30, 'lev': 10, 'max_hold': 480},

        {'name': 'S8: EMA10/EMA200 Trail+2/-1 SL2%', 'fast': ema10, 'slow': ema200, 'rsi': rsi14,
         'sl': 0.02, 'tp': 0, 'trail_act': 0.02, 'trail_w': 0.01, 'rsi_lo': 30, 'rsi_hi': 70,
         'margin': 0.30, 'lev': 10, 'max_hold': 720},

        # RSI-tight scalp
        {'name': 'S9: EMA3/EMA50 TP0.5% SL0.5% RSI40-60', 'fast': ema3, 'slow': ema50, 'rsi': rsi14,
         'sl': 0.005, 'tp': 0.005, 'trail_act': 99, 'trail_w': 99, 'rsi_lo': 40, 'rsi_hi': 60,
         'margin': 0.20, 'lev': 10, 'max_hold': 60},

        # High leverage micro
        {'name': 'S10: EMA3/EMA20 TP0.2% SL0.3% Lev20x', 'fast': ema3, 'slow': ema20, 'rsi': rsi7,
         'sl': 0.003, 'tp': 0.002, 'trail_act': 99, 'trail_w': 99, 'rsi_lo': 30, 'rsi_hi': 70,
         'margin': 0.15, 'lev': 20, 'max_hold': 30},

        # v25.0 style on 1m (adapted)
        {'name': 'S11: EMA5/EMA100 SL-4% Trail+5/-3 (v25)', 'fast': ema5, 'slow': ema100, 'rsi': rsi14,
         'sl': 0.04, 'tp': 0, 'trail_act': 0.05, 'trail_w': 0.03, 'rsi_lo': 40, 'rsi_hi': 60,
         'margin': 0.30, 'lev': 10, 'max_hold': 1440},

        # Aggressive micro
        {'name': 'S12: EMA3/EMA10 TP0.15% SL0.2% Lev20x', 'fast': ema3, 'slow': ema10, 'rsi': rsi7,
         'sl': 0.002, 'tp': 0.0015, 'trail_act': 99, 'trail_w': 99, 'rsi_lo': 25, 'rsi_hi': 75,
         'margin': 0.10, 'lev': 20, 'max_hold': 15},
    ]

    # Run all
    print(f"\n{'='*110}")
    print(f"  RESULTS")
    print(f"{'='*110}")
    print(f"  {'Strategy':<45} {'Final$':>9} {'Ret%':>8} {'PF':>6} {'MDD%':>7} {'Trades':>7} {'WR%':>5} {'SL':>4} {'T/day':>5}")
    print("  " + "-" * 105)

    results = []
    for s in strategies:
        t0 = time.time()
        r = scalp_backtest(h, l, c, s['fast'], s['slow'], s['rsi'], s)
        elapsed = time.time() - t0

        pf_str = f"{r['pf']:.2f}" if r['pf'] < 999 else "INF"
        marker = " ***" if r['return'] > 0 and r['pf'] > 1.0 else ""
        print(f"  {s['name']:<45} ${r['final']:>7,.0f} {r['return']:>+7.1f}% {pf_str:>5} {r['mdd']:>6.1f}% {r['trades']:>6} {r['wr']:>4.0f}% {r['sl']:>4} {r['avg_trades_day']:>4.1f}{marker}")
        results.append({'strategy': s['name'], **r})

    # Summary
    profitable = [r for r in results if r['return'] > 0]
    print(f"\n{'='*110}")
    print(f"  SUMMARY")
    print(f"{'='*110}")
    print(f"  Total strategies tested: {len(results)}")
    print(f"  Profitable: {len(profitable)} / {len(results)}")

    if profitable:
        best = max(profitable, key=lambda x: x['return'])
        print(f"\n  BEST STRATEGY: {best['strategy']}")
        print(f"    Return: {best['return']:+.1f}% | PF: {best['pf']:.2f} | MDD: {best['mdd']:.1f}%")
        print(f"    Trades: {best['trades']} | WR: {best['wr']:.0f}% | Trades/day: {best['avg_trades_day']:.1f}")

    # Fee impact analysis
    print(f"\n{'='*110}")
    print(f"  FEE IMPACT ANALYSIS")
    print(f"{'='*110}")
    print(f"  Fee per round-trip: 0.08% of notional")
    print(f"  At M20% Lev10x: fee = 0.08% x 10 = 0.8% of margin per trade")
    print(f"  At M20% Lev20x: fee = 0.08% x 20 = 1.6% of margin per trade")
    print(f"")
    print(f"  For TP 0.3% with Lev10x: gross profit = 3.0% of margin")
    print(f"    Net after fee: 3.0% - 0.8% = 2.2% (fee is 27% of gross)")
    print(f"  For TP 0.15% with Lev20x: gross profit = 3.0% of margin")
    print(f"    Net after fee: 3.0% - 1.6% = 1.4% (fee is 53% of gross)")
    print(f"")
    print(f"  CONCLUSION: Fee consumes 27-53% of gross profit on micro-scalps")
    print(f"  Minimum viable TP: ~0.5% at 10x leverage (fee = 16% of gross)")

    # Breakeven analysis
    print(f"\n{'='*110}")
    print(f"  BREAKEVEN WIN RATE ANALYSIS")
    print(f"{'='*110}")
    for tp, sl, lev2 in [(0.003, 0.005, 10), (0.005, 0.008, 10), (0.01, 0.01, 10),
                          (0.002, 0.003, 20), (0.02, 0.01, 10), (0.05, 0.04, 10)]:
        fee_impact = 2 * FEE * lev2  # round trip fee as % of margin
        net_tp = tp * lev2 - fee_impact
        net_sl = sl * lev2 + fee_impact
        if net_tp > 0:
            be_wr = net_sl / (net_tp + net_sl) * 100
            print(f"  TP {tp*100:.1f}% SL {sl*100:.1f}% Lev{lev2}x: NetTP={net_tp*100:.1f}% NetSL={net_sl*100:.1f}% -> Breakeven WR: {be_wr:.0f}%")
        else:
            print(f"  TP {tp*100:.1f}% SL {sl*100:.1f}% Lev{lev2}x: NetTP NEGATIVE (fee > profit)")

    print(f"\n{'='*110}")
    print(f"  SCALPING FEASIBILITY VERDICT")
    print(f"{'='*110}")
    if profitable:
        print(f"  POSSIBLE - {len(profitable)} strategies were profitable")
        print(f"  Best: {max(profitable, key=lambda x: x['return'])['strategy']}")
        print(f"         {max(profitable, key=lambda x: x['return'])['return']:+.1f}% return over 24 months")
    else:
        print(f"  NOT VIABLE with tested parameters")
        print(f"  Fee drag is too heavy for micro-scalping on 1m BTC")

    print(f"\n{'='*110}")
    print(f"  COMPLETE")
    print(f"{'='*110}")

if __name__ == "__main__":
    main()
