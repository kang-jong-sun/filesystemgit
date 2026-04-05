"""
Verification 3: v26.0 Dual-Track Strategy
Track A: EMA(3)/SMA(300), 30m, ADX(14)>=40, TS+4/-3, M50%, Lev10x (60%)
Track B: WMA(3)/EMA(200), 30m, ADX(14)>=35, TS+3/-2, M50%, Lev10x (40%)
10 Backtests + 10 Verifications + Full Report
"""
import pandas as pd, numpy as np, os, time, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
DIR = r"D:\filesystem\futures\btc_V1\test"
FEE = 0.0004; INIT = 3000.0

# ==== Wilder Smoothing ====
def wilder(arr, p):
    out = np.full(len(arr), np.nan)
    s = 0
    while s < len(arr) and np.isnan(arr[s]): s += 1
    if s+p > len(arr): return out
    out[s+p-1] = np.nanmean(arr[s:s+p])
    for i in range(s+p, len(arr)):
        if not np.isnan(arr[i]) and not np.isnan(out[i-1]):
            out[i] = (out[i-1]*(p-1) + arr[i]) / p
    return out

def adx_wilder(high, low, close, period=14):
    n = len(high)
    tr = np.full(n, np.nan); pdm = np.full(n, np.nan); mdm = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        up = high[i]-high[i-1]; dn = low[i-1]-low[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
    atr = wilder(tr, period); spdm = wilder(pdm, period); smdm = wilder(mdm, period)
    pdi = np.full(n, np.nan); mdi = np.full(n, np.nan); dx = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr[i]) and atr[i] > 0:
            pdi[i] = 100*spdm[i]/atr[i]; mdi[i] = 100*smdm[i]/atr[i]
            s = pdi[i]+mdi[i]; dx[i] = 100*abs(pdi[i]-mdi[i])/s if s > 0 else 0
    return wilder(dx, period), atr

def calc_ema(c, p):
    out = np.full(len(c), np.nan)
    s = 0
    while s < len(c) and np.isnan(c[s]): s += 1
    if s >= len(c): return out
    out[s] = c[s]; m = 2.0/(p+1)
    for i in range(s+1, len(c)):
        if not np.isnan(c[i]) and not np.isnan(out[i-1]):
            out[i] = c[i]*m + out[i-1]*(1-m)
    return out

def calc_sma(c, p):
    out = np.full(len(c), np.nan)
    for i in range(p-1, len(c)):
        sl = c[i-p+1:i+1]
        if not np.any(np.isnan(sl)): out[i] = np.mean(sl)
    return out

def calc_wma(c, p):
    out = np.full(len(c), np.nan)
    w = np.arange(1, p+1, dtype=float); ws = w.sum()
    for i in range(p-1, len(c)):
        sl = c[i-p+1:i+1]
        if not np.any(np.isnan(sl)): out[i] = np.dot(sl, w)/ws
    return out

def calc_rsi(c, p=14):
    n = len(c); gains = np.zeros(n); losses = np.zeros(n)
    for i in range(1, n):
        d = c[i]-c[i-1]
        if d > 0: gains[i] = d
        else: losses[i] = -d
    ag = wilder(gains, p); al = wilder(losses, p)
    out = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(ag[i]) and not np.isnan(al[i]):
            out[i] = 100.0 if al[i] == 0 else 100-100/(1+ag[i]/al[i])
    return out

# ==== Data ====
def load_30m():
    t0 = time.time()
    fs = [os.path.join(DIR, f"btc_usdt_5m_2020_to_now_part{i}.csv") for i in range(1,4)]
    df = pd.concat([pd.read_csv(f, parse_dates=['timestamp']) for f in fs], ignore_index=True)
    df.sort_values('timestamp', inplace=True); df.set_index('timestamp', inplace=True)
    d30 = df.resample('30min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    print(f"[Data] 5m:{len(df):,} -> 30m:{len(d30):,} | {time.time()-t0:.1f}s")
    return d30

# ==== Track Config ====
TRACKS = {
    'A': {'name':'Sniper','fast_type':'EMA','fast_len':3,'slow_type':'SMA','slow_len':300,
           'adx_min':40,'rsi_lo':30,'rsi_hi':70,'sl':0.08,'trail_act':0.04,'trail_w':0.03,
           'margin':0.50,'lev':10,'cap_pct':0.60},
    'B': {'name':'Compounder','fast_type':'WMA','fast_len':3,'slow_type':'EMA','slow_len':200,
           'adx_min':35,'rsi_lo':30,'rsi_hi':70,'sl':0.08,'trail_act':0.03,'trail_w':0.02,
           'margin':0.50,'lev':10,'cap_pct':0.40},
}

# ==== Single Backtest ====
def run_backtest(highs, lows, closes, timestamps, adx_arr, rsi_arr, fast_ma_a, slow_ma_a, fast_ma_b, slow_ma_b, return_trades=True):
    n = len(closes)
    results = {}
    all_trades = []

    for tk, cfg in TRACKS.items():
        bal = INIT * cfg['cap_pct']
        peak_bal = bal
        fast = fast_ma_a if tk == 'A' else fast_ma_b
        slow = slow_ma_a if tk == 'A' else slow_ma_b

        pos_dir = 0; pos_entry = 0; pos_size = 0; pos_idx = 0
        pos_hi = 0; pos_lo = 0; pos_sl = 0; pos_trail_active = False; pos_trail_sl = 0
        trades = []; warmup = 350

        def close_p(xp, xr, bi):
            nonlocal bal, peak_bal, pos_dir
            rs = pos_size
            rpnl = rs*((xp-pos_entry)/pos_entry if pos_dir==1 else (pos_entry-xp)/pos_entry)
            tpnl = rpnl - rs*FEE
            bal += tpnl; peak_bal = max(peak_bal, bal)
            margin = rs/cfg['lev']
            roi = tpnl/margin*100 if margin > 0 else 0
            if return_trades:
                trades.append({
                    'track':tk,'direction':'LONG' if pos_dir==1 else 'SHORT',
                    'entry_time':str(timestamps[pos_idx]),'exit_time':str(timestamps[bi]),
                    'entry_price':round(pos_entry,2),'exit_price':round(xp,2),
                    'exit_reason':xr,'pnl':round(tpnl,2),'roi_pct':round(roi,2),
                    'hold_hours':round((bi-pos_idx)*0.5,1),'balance':round(bal,2)
                })
            pos_dir = 0
            return tpnl

        for i in range(warmup, n):
            h = highs[i]; l = lows[i]; c = closes[i]

            # Position mgmt
            if pos_dir != 0:
                if pos_dir == 1:
                    pos_hi = max(pos_hi, h)
                    if l <= pos_sl: close_p(pos_sl, 'SL', i)
                    else:
                        roi = (c-pos_entry)/pos_entry
                        if roi >= cfg['trail_act']: pos_trail_active = True
                        if pos_trail_active:
                            new_tsl = pos_hi*(1-cfg['trail_w'])
                            if new_tsl > pos_trail_sl: pos_trail_sl = max(new_tsl, pos_sl)
                        if pos_trail_active and c <= pos_trail_sl: close_p(c, 'TSL', i)
                else:
                    pos_lo = min(pos_lo, l)
                    if h >= pos_sl: close_p(pos_sl, 'SL', i)
                    else:
                        roi = (pos_entry-c)/pos_entry
                        if roi >= cfg['trail_act']: pos_trail_active = True
                        if pos_trail_active:
                            new_tsl = pos_lo*(1+cfg['trail_w'])
                            if new_tsl < pos_trail_sl: pos_trail_sl = min(new_tsl, pos_sl)
                        if pos_trail_active and c >= pos_trail_sl: close_p(c, 'TSL', i)

            # Signal
            if i < warmup+1: continue
            if np.isnan(fast[i]) or np.isnan(fast[i-1]) or np.isnan(slow[i]) or np.isnan(slow[i-1]): continue
            cross_up = fast[i-1] <= slow[i-1] and fast[i] > slow[i]
            cross_down = fast[i-1] >= slow[i-1] and fast[i] < slow[i]
            if not cross_up and not cross_down: continue

            sig = 1 if cross_up else -1
            if np.isnan(adx_arr[i]) or adx_arr[i] < cfg['adx_min']: continue
            if np.isnan(rsi_arr[i]) or not (cfg['rsi_lo'] <= rsi_arr[i] <= cfg['rsi_hi']): continue

            # Same-dir skip
            if pos_dir == sig: continue
            # REV
            if pos_dir != 0 and pos_dir != sig: close_p(c, 'REV', i)

            # DD check
            dd = (bal-peak_bal)/peak_bal if peak_bal > 0 else 0
            if dd < -0.40: continue
            dd_m = 0.5 if dd < -0.25 else 1.0

            sz = bal * cfg['margin'] * dd_m
            if sz < 5: continue
            notional = sz * cfg['lev']
            bal -= notional * FEE

            pos_dir = sig; pos_entry = c; pos_size = notional; pos_idx = i
            pos_hi = c; pos_lo = c; pos_trail_active = False
            if sig == 1: pos_sl = c*(1-cfg['sl']); pos_trail_sl = pos_sl
            else: pos_sl = c*(1+cfg['sl']); pos_trail_sl = pos_sl

        # Metrics
        gp = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gl = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
        wins = [t for t in trades if t['pnl'] > 0]
        losses_t = [t for t in trades if t['pnl'] <= 0]

        bals = [INIT*cfg['cap_pct']] + [t['balance'] for t in trades]
        pk = bals[0]; mdd = 0
        for b in bals: pk = max(pk, b); mdd = min(mdd, (b-pk)/pk)

        results[tk] = {
            'final': round(bal, 2), 'return': round((bal-INIT*cfg['cap_pct'])/INIT*cfg['cap_pct']*100, 2),
            'trades': len(trades), 'wins': len(wins), 'wr': round(len(wins)/max(1,len(trades))*100, 1),
            'pf': round(gp/gl, 2) if gl > 0 else 999,
            'mdd': round(mdd*100, 2), 'gp': round(gp, 2), 'gl': round(gl, 2),
            'sl': sum(1 for t in trades if t['exit_reason']=='SL'),
            'tsl': sum(1 for t in trades if t['exit_reason']=='TSL'),
            'rev': sum(1 for t in trades if t['exit_reason']=='REV'),
        }
        all_trades.extend(trades)

    total_bal = results['A']['final'] + results['B']['final']
    total_trades = results['A']['trades'] + results['B']['trades']
    total_gp = results['A']['gp'] + results['B']['gp']
    total_gl = results['A']['gl'] + results['B']['gl']

    portfolio = {
        'final': round(total_bal, 2),
        'return': round((total_bal-INIT)/INIT*100, 2),
        'trades': total_trades,
        'pf': round(total_gp/total_gl, 2) if total_gl > 0 else 999,
        'tracks': results,
    }

    # Portfolio MDD from combined balance curve
    if all_trades:
        all_trades.sort(key=lambda x: x['exit_time'])
        combined_bals = [INIT]
        for t in all_trades: combined_bals.append(results['A']['final'] + results['B']['final'])
        # Approximate from track-level
        portfolio['mdd'] = min(results['A']['mdd'], results['B']['mdd'])

    return portfolio, all_trades


def main():
    print("="*100)
    print("  VERIFICATION 3 (v26.0 Dual-Track)")
    print("  Track A: EMA(3)/SMA(300) 30m ADX>=40 | Track B: WMA(3)/EMA(200) 30m ADX>=35")
    print("="*100)

    d30 = load_30m()
    print("\n[Indicators]...")
    t0 = time.time()
    h = d30['high'].values; l = d30['low'].values; c = d30['close'].values; ts = d30.index
    adx_arr, _ = adx_wilder(h, l, c, 14)
    rsi_arr = calc_rsi(c, 14)
    fa_a = calc_ema(c, 3); sa_a = calc_sma(c, 300)
    fa_b = calc_wma(c, 3); sa_b = calc_ema(c, 200)
    print(f"  Done | {time.time()-t0:.1f}s")

    # ==== 10 Backtests ====
    print(f"\n{'='*100}")
    print(f"  PHASE 1: 10 BACKTESTS")
    print(f"{'='*100}")
    bt_results = []
    for r in range(1, 11):
        t0 = time.time()
        port, trades = run_backtest(h, l, c, ts, adx_arr, rsi_arr, fa_a, sa_a, fa_b, sa_b, r == 1)
        elapsed = time.time() - t0
        bt_results.append(port)
        ta = port['tracks']['A']; tb = port['tracks']['B']
        print(f"  BT#{r:>2} | {elapsed:.1f}s | Total:${port['final']:>10,.2f} ({port['return']:>+8.1f}%) | "
              f"PF:{port['pf']:>5.2f} | Trd:{port['trades']:>3} | "
              f"A:${ta['final']:>9,.2f}(PF{ta['pf']:>5.1f}) B:${tb['final']:>8,.2f}(PF{tb['pf']:>5.1f})")
        if r == 1:
            first_trades = trades

    # ==== 10 Verifications ====
    print(f"\n{'='*100}")
    print(f"  PHASE 2: 10 VERIFICATIONS")
    print(f"{'='*100}")
    vr_results = []
    for r in range(1, 11):
        port, _ = run_backtest(h, l, c, ts, adx_arr, rsi_arr, fa_a, sa_a, fa_b, sa_b, False)
        vr_results.append(port)
        print(f"  VR#{r:>2} | Total:${port['final']:>10,.2f} ({port['return']:>+8.1f}%) | PF:{port['pf']:>5.2f} | Trd:{port['trades']:>3}")

    # ==== Consistency Check ====
    bt_finals = [r['final'] for r in bt_results]
    vr_finals = [r['final'] for r in vr_results]
    all_finals = bt_finals + vr_finals
    std_all = np.std(all_finals)
    det = std_all < 0.01

    print(f"\n{'='*100}")
    print(f"  CONSISTENCY CHECK")
    print(f"{'='*100}")
    print(f"  BT mean:${np.mean(bt_finals):,.2f} std:${np.std(bt_finals):.4f}")
    print(f"  VR mean:${np.mean(vr_finals):,.2f} std:${np.std(vr_finals):.4f}")
    print(f"  All 20 runs: mean=${np.mean(all_finals):,.2f} std=${std_all:.4f}")
    print(f"  DETERMINISTIC: {'YES' if det else 'NO'}")

    # ==== Detailed Report from first run ====
    port = bt_results[0]; ta = port['tracks']['A']; tb = port['tracks']['B']

    print(f"\n{'='*100}")
    print(f"  DETAILED RESULTS (Run #1)")
    print(f"{'='*100}")
    print(f"""
  PORTFOLIO:
    Initial: ${INIT:,.0f}  Final: ${port['final']:,.2f}  Return: {port['return']:+.1f}%
    PF: {port['pf']:.2f}  Trades: {port['trades']}

  TRACK A (Sniper, 60%):
    Final: ${ta['final']:,.2f}  Return: {ta['return']:+.1f}%
    Trades: {ta['trades']}  WR: {ta['wr']}%  PF: {ta['pf']}
    MDD: {ta['mdd']:.1f}%  SL:{ta['sl']} TSL:{ta['tsl']} REV:{ta['rev']}

  TRACK B (Compounder, 40%):
    Final: ${tb['final']:,.2f}  Return: {tb['return']:+.1f}%
    Trades: {tb['trades']}  WR: {tb['wr']}%  PF: {tb['pf']}
    MDD: {tb['mdd']:.1f}%  SL:{tb['sl']} TSL:{tb['tsl']} REV:{tb['rev']}
""")

    # Monthly & Yearly from trades
    if first_trades:
        df = pd.DataFrame(first_trades)
        df['exit_dt'] = pd.to_datetime(df['exit_time'])
        df['month'] = df['exit_dt'].dt.to_period('M')

        # Direction
        print("  DIRECTION:")
        for d in ['LONG','SHORT']:
            sub = df[df['direction']==d]
            if len(sub)==0: continue
            sw = sub[sub['pnl']>0]
            print(f"    {d:>5}: {len(sub):>3} ({len(sub)/len(df)*100:.0f}%) WR:{len(sw)/len(sub)*100:.0f}% PnL:${sub['pnl'].sum():+,.0f}")

        print(f"\n  EXIT REASONS:")
        for r in ['TSL','REV','SL']:
            rt = df[df['exit_reason']==r]
            if len(rt)==0: continue
            print(f"    {r:>3}: {len(rt):>3} ({len(rt)/len(df)*100:.0f}%) AvgROI:{rt['roi_pct'].mean():+.1f}% PnL:${rt['pnl'].sum():+,.0f}")

        print(f"\n  HOLD TIME:")
        for a,b,lb in [(0,4,'<4h'),(4,12,'4-12h'),(12,48,'12h-2d'),(48,168,'2-7d'),(168,9999,'7d+')]:
            ht = df[(df['hold_hours']>=a)&(df['hold_hours']<b)]
            if len(ht):
                hw = ht[ht['pnl']>0]
                print(f"    {lb:>6}: {len(ht):>3} WR:{len(hw)/len(ht)*100:.0f}% PnL:${ht['pnl'].sum():+,.0f}")

        # Monthly
        mg = df.groupby('month')
        print(f"\n{'='*100}")
        print(f"  MONTHLY PERFORMANCE")
        print(f"{'='*100}")
        print(f"  {'Month':>8} {'#':>3} {'W':>2} {'L':>2} {'WR%':>5} {'PnL':>9} {'PF':>6} {'Bal':>10} {'Ret%':>7} {'A':>2} {'B':>2}")
        print("  "+"-"*80)

        rb = INIT; yearly = {}; lm = 0; tm = 0
        for mo in sorted(mg.groups.keys()):
            g = mg.get_group(mo); nt = len(g); nw = len(g[g['pnl']>0]); nl = nt-nw
            wr = nw/nt*100 if nt else 0
            gp2 = g[g['pnl']>0]['pnl'].sum() if nw else 0
            gl2 = abs(g[g['pnl']<=0]['pnl'].sum()) if nl else 0
            net = g['pnl'].sum(); mpf = gp2/gl2 if gl2 > 0 else 999
            sbr = rb; rb += net; mr = net/sbr*100 if sbr > 0 else 0; tm += 1
            if net < 0: lm += 1
            ea = len(g[g['track']=='A']); eb = len(g[g['track']=='B'])
            y = str(mo)[:4]
            if y not in yearly: yearly[y] = {'p':0,'t':0,'w':0,'l':0,'gp':0,'gl':0,'sb':sbr}
            yearly[y]['p']+=net; yearly[y]['t']+=nt; yearly[y]['w']+=nw; yearly[y]['l']+=nl
            yearly[y]['gp']+=gp2; yearly[y]['gl']+=gl2; yearly[y]['eb']=rb
            pfs = f"{mpf:.1f}" if mpf < 999 else "INF"
            mk = " <<" if net < 0 else ""
            print(f"  {str(mo):>8} {nt:>3} {nw:>2} {nl:>2} {wr:>4.0f}% ${net:>+7,.0f} {pfs:>5} ${rb:>8,.0f} {mr:>+6.1f}% {ea:>2} {eb:>2}{mk}")

        print(f"\n  YEARLY:")
        print(f"  {'Year':>6} {'#':>3} {'W':>2} {'L':>2} {'WR%':>5} {'NetPnL':>10} {'PF':>6} {'Ret%':>8}")
        print("  "+"-"*55)
        for y2 in sorted(yearly):
            yd=yearly[y2]; ywr=yd['w']/yd['t']*100 if yd['t'] else 0
            ypf=yd['gp']/yd['gl'] if yd['gl']>0 else 999; yret=yd['p']/yd['sb']*100 if yd['sb']>0 else 0
            pfs=f"{ypf:.2f}" if ypf<999 else "INF"
            print(f"  {y2:>6} {yd['t']:>3} {yd['w']:>2} {yd['l']:>2} {ywr:>4.0f}% ${yd['p']:>+8,.0f} {pfs:>5} {yret:>+7.1f}%")
        pyrs=sum(1 for v in yearly.values() if v['p']>0)
        print(f"\n  Loss Months: {lm}/{tm} ({lm/max(1,tm)*100:.0f}%)")
        print(f"  Profit Years: {pyrs}/{len(yearly)}")

        # Top/Bottom
        ds = df.sort_values('pnl', ascending=False)
        print(f"\n  TOP 5:")
        for i2,(_,r) in enumerate(ds.head(5).iterrows()):
            print(f"    {i2+1} {r['track']} {r['direction']:>5} {r['entry_time'][:16]}->{r['exit_time'][:16]} {r['exit_reason']:>3} ROI:{r['roi_pct']:>+6.1f}% PnL:${r['pnl']:>+8,.0f}")
        print(f"\n  BOTTOM 5:")
        for i2,(_,r) in enumerate(ds.tail(5).iterrows()):
            print(f"    {i2+1} {r['track']} {r['direction']:>5} {r['entry_time'][:16]}->{r['exit_time'][:16]} {r['exit_reason']:>3} ROI:{r['roi_pct']:>+6.1f}% PnL:${r['pnl']:>+8,.0f}")

    # ==== Save Report ====
    # Save trades
    if first_trades:
        pd.DataFrame(first_trades).to_csv(os.path.join(DIR, "verify3_trades.csv"), index=False, encoding='utf-8-sig')

    # Save metrics
    save_metrics = {
        'portfolio': port,
        'bt_results': [{'run':i+1,'final':r['final'],'return':r['return'],'pf':r['pf'],'trades':r['trades']} for i,r in enumerate(bt_results)],
        'vr_results': [{'run':i+1,'final':r['final'],'return':r['return'],'pf':r['pf'],'trades':r['trades']} for i,r in enumerate(vr_results)],
        'consistency': {'std': round(std_all, 6), 'deterministic': det},
    }
    with open(os.path.join(DIR, "verify3_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(save_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*100}")
    print(f"  FILES: verify3_trades.csv, verify3_metrics.json")
    print(f"  VERIFICATION 3 COMPLETE")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()
