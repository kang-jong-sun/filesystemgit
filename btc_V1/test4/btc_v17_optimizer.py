"""
v17.0 Mega Optimizer - incorporating ALL findings from v13.5~v16.6
Key innovations to test:
1. HMA(21)/EMA(250) on 10m (v16.5 PF 25.66)
2. 5-candle delayed entry (v16.6 PF 89.85)
3. Multiple timeframes (5m, 10m, 15m, 30m, 1h)
4. WMA(3)/EMA(200) baseline (v16.0/v16.4)
5. Higher trade frequency for compounding
"""
import pandas as pd, numpy as np, time, sys, itertools
from pathlib import Path
from btc_v164_backtest import (load_5m_data, resample, calc_wma, calc_ema, calc_sma,
                                calc_adx, calc_rsi, calc_atr, add_indicators, FEE_RATE)

DATA_DIR = Path(r'D:\filesystem\futures\btc_V1\test4')
CAPITAL = 3000

def pr(*a, **k): print(*a, **k, flush=True)

def calc_hma(s, period):
    h = max(int(period/2),1); sq = max(int(np.sqrt(period)),1)
    wh = calc_wma(s,h); wf = calc_wma(s,period)
    return calc_wma(2*wh - wf, sq)

def get_ma(close_s, ma_type, period):
    if ma_type == 'wma': return calc_wma(close_s, period).values
    elif ma_type == 'ema': return calc_ema(close_s, period).values
    elif ma_type == 'sma': return calc_sma(close_s, period).values
    elif ma_type == 'hma': return calc_hma(close_s, period).values
    elif ma_type == 'dema':
        e1 = calc_ema(close_s, period); return (2*e1 - calc_ema(e1, period)).values
    return calc_ema(close_s, period).values

def fast_bt(C, H, L, fma, sma, adx, rsi, adx_min, rsi_min, rsi_max,
            sl_pct, ts_act, ts_trail, margin, lev, delay=0, skip_same_dir=False,
            fee=FEE_RATE, capital=CAPITAL):
    n = len(C); bal = float(capital)
    pos_dir = 0; pos_epx = 0.; pos_sz = 0.; pos_pk = 0.
    pos_hi = 0.; pos_lo = 0.; pf = np.nan; ps = np.nan
    peak_bal = bal; mdd = 0.; tp = 0.; tl = 0.
    nt = 0; nw = 0; nsl = 0; ntsl = 0; nrev = 0; fl = 0
    # Delayed entry state
    pending_sig = 0; pending_since = -9999
    last_exit_dir = 0  # for same-dir skip

    for i in range(1, n):
        f = fma[i]; s = sma[i]; a = adx[i]; r = rsi[i]
        if np.isnan(f) or np.isnan(s) or np.isnan(a) or np.isnan(r):
            pf = f; ps = s; continue
        if np.isnan(pf) or np.isnan(ps):
            pf = f; ps = s; continue
        cpx = C[i]; hpx = H[i]; lpx = L[i]
        gc = pf <= ps and f > s; dc = pf >= ps and f < s

        # Position management
        if pos_dir != 0:
            if pos_dir == 1:
                pos_hi = max(pos_hi, hpx); pos_lo = min(pos_lo, lpx)
                roi = cpx/pos_epx - 1.; best = pos_hi/pos_epx - 1.
            else:
                pos_hi = max(pos_hi, hpx); pos_lo = min(pos_lo, lpx)
                roi = 1. - cpx/pos_epx; best = 1. - pos_lo/pos_epx
            pos_pk = max(pos_pk, best)
            ex = 0; ex_px = cpx
            if pos_dir == 1:
                slp = pos_epx*(1.-sl_pct)
                if lpx <= slp: ex = 1; ex_px = slp
            else:
                slp = pos_epx*(1.+sl_pct)
                if hpx >= slp: ex = 1; ex_px = slp
            if ex == 0 and pos_pk >= ts_act and pos_pk - roi >= ts_trail:
                ex = 2; ex_px = cpx
            if ex == 0:
                if pos_dir==1 and dc and a>=adx_min and rsi_min<=r<=rsi_max and roi<0.20:
                    ex = 3; ex_px = cpx
                elif pos_dir==-1 and gc and a>=adx_min and rsi_min<=r<=rsi_max and roi<0.20:
                    ex = 3; ex_px = cpx
            if ex > 0:
                tr = (ex_px/pos_epx-1.) if pos_dir==1 else (1.-ex_px/pos_epx)
                pnl = pos_sz*lev*tr - pos_sz*lev*fee*2; bal += pnl; nt += 1
                if pnl > 0: nw += 1; tp += pnl
                else: tl += abs(pnl)
                if ex==1: nsl+=1
                elif ex==2: ntsl+=1
                else: nrev+=1
                if bal <= 0: fl += 1; bal = 0.01
                last_exit_dir = pos_dir
                nd = -pos_dir if ex==3 else 0; pos_dir = 0
                if nd != 0 and bal > 0:
                    if skip_same_dir and nd == last_exit_dir:
                        pass  # skip same direction re-entry
                    else:
                        pos_dir=nd; pos_epx=cpx; pos_sz=bal*margin; pos_pk=0.; pos_hi=cpx; pos_lo=cpx
                        pending_sig = 0

        # Detect new cross -> set pending
        if pos_dir == 0:
            if gc:
                if pending_sig != 1: pending_sig = 1; pending_since = i
            elif dc:
                if pending_sig != -1: pending_sig = -1; pending_since = i

        # Entry with delay
        if pos_dir == 0 and bal > 0 and pending_sig != 0:
            candles_waited = i - pending_since
            if candles_waited >= delay:
                sig = pending_sig
                if a < adx_min: sig = 0
                if sig and not (rsi_min <= r <= rsi_max): sig = 0
                if sig and skip_same_dir and sig == last_exit_dir: sig = 0
                if sig:
                    pos_dir=sig; pos_epx=cpx; pos_sz=bal*margin; pos_pk=0.; pos_hi=cpx; pos_lo=cpx
                    pending_sig = 0
            elif candles_waited > delay + 12:  # expire after 12 extra candles
                pending_sig = 0

        pf = f; ps = s
        if bal > peak_bal: peak_bal = bal
        dd = (peak_bal-bal)/peak_bal if peak_bal > 0 else 0
        if dd > mdd: mdd = dd

    if pos_dir != 0:
        tr = (C[-1]/pos_epx-1.) if pos_dir==1 else (1.-C[-1]/pos_epx)
        pnl = pos_sz*lev*tr - pos_sz*lev*fee*2; bal += pnl; nt += 1
        if pnl > 0: nw+=1; tp+=pnl
        else: tl+=abs(pnl)
    pf_val = tp/tl if tl > 0 else (999. if tp > 0 else 0.)
    return {'bal':bal, 'ret':(bal-capital)/capital*100, 'pf':pf_val, 'mdd':mdd*100,
            'trades':nt, 'wr':nw/nt*100 if nt else 0, 'sl':nsl, 'tsl':ntsl, 'rev':nrev, 'fl':fl}

def main():
    pr("="*90)
    pr("  v17.0 MEGA OPTIMIZER - All Versions Combined")
    pr("  Incorporating v13.5~v16.6 discoveries")
    pr("="*90)
    t0 = time.time()

    pr("\n[DATA]")
    df_5m = load_5m_data()

    # Multiple timeframes
    tfs = {}
    for rule, label in [('10min','10m'),('15min','15m'),('30min','30m'),('60min','1h')]:
        tfs[label] = resample(df_5m, rule)
        tfs[label] = add_indicators(tfs[label])
        pr(f"  {label}: {len(tfs[label]):,} candles")
    pr(f"  Data ready: {time.time()-t0:.1f}s")

    # Pre-compute MAs for each TF
    pr("\n[MA CACHE]")
    ma_cache = {}
    for tf_label, df in tfs.items():
        cs = df['close']
        for mt in ['wma','ema','hma','sma','dema']:
            for p in [2,3,4,5,7,10,14,21]:
                key = f"{tf_label}_{mt}_{p}"
                try: ma_cache[key] = get_ma(cs, mt, p)
                except: pass
        for mt in ['ema','sma']:
            for p in [100,150,200,250,300]:
                key = f"{tf_label}_{mt}_{p}"
                ma_cache[key] = get_ma(cs, mt, p)
    # ADX cache
    adx_cache = {}
    for tf_label, df in tfs.items():
        for ap in [14, 20]:
            adx_cache[f"{tf_label}_{ap}"] = calc_adx(df['high'],df['low'],df['close'],ap).values
    pr(f"  {len(ma_cache)} MAs, {len(adx_cache)} ADX cached")

    # ====================================================================
    # STAGE 1: MEGA SCAN - All TF x MA x Filter x Delay combos
    # ====================================================================
    pr("\n" + "="*90)
    pr("  STAGE 1: MEGA SCAN (TF x MA x Filter x Delay)")
    pr("="*90)

    combos = []
    # Phase A: Fast scan on 30m + 1h (fast candle counts)
    for tf in ['30m','1h']:
        for ft,fl in [('hma',21),('hma',14),('hma',7),('hma',5),
                       ('wma',3),('wma',4),('wma',5),('wma',7),
                       ('ema',3),('ema',5),('dema',3),('sma',3)]:
            for st,sl_ in [('ema',200),('ema',250),('ema',300),('ema',150),('sma',250),('sma',300)]:
                for ap in [20]:
                    for am in [30,35,40]:
                        for rmin,rmax in [(35,65),(30,70),(40,75)]:
                            for delay in [0,3,5]:
                                fk = f"{tf}_{ft}_{fl}"; sk = f"{tf}_{st}_{sl_}"
                                ak = f"{tf}_{ap}"
                                if fk in ma_cache and sk in ma_cache and ak in adx_cache:
                                    combos.append((tf,ft,fl,st,sl_,ap,am,rmin,rmax,delay,fk,sk,ak))
    # Phase B: Targeted 10m/15m scan (only proven best MAs)
    for tf in ['10m','15m']:
        for ft,fl in [('hma',21),('hma',14),('wma',3),('ema',3)]:
            for st,sl_ in [('ema',200),('ema',250)]:
                for am in [30,35]:
                    for rmin,rmax in [(35,65),(30,70)]:
                        for delay in [0,3,5]:
                            fk = f"{tf}_{ft}_{fl}"; sk = f"{tf}_{st}_{sl_}"
                            ak = f"{tf}_20"
                            if fk in ma_cache and sk in ma_cache and ak in adx_cache:
                                combos.append((tf,ft,fl,st,sl_,20,am,rmin,rmax,delay,fk,sk,ak))

    pr(f"  Total combinations: {len(combos):,}")
    SL=0.08; TA=0.04; TT=0.03; M=0.30; LEV=10

    results1 = []; done = 0; t1 = time.time()
    for tf,ft,fl,st,sl_,ap,am,rmin,rmax,delay,fk,sk,ak in combos:
        df = tfs[tf]
        r = fast_bt(df['close'].values, df['high'].values, df['low'].values,
                    ma_cache[fk], ma_cache[sk], adx_cache[ak], df['rsi14'].values,
                    am, rmin, rmax, SL, TA, TT, M, LEV, delay=delay, skip_same_dir=(delay>0))
        r['tf']=tf; r['fast']=f"{ft}({fl})"; r['slow']=f"{st}({sl_})"
        r['adx_p']=ap; r['adx_min']=am; r['rsi_r']=f"{rmin}-{rmax}"; r['delay']=delay
        results1.append(r)
        done += 1
        if done % 5000 == 0:
            elapsed = time.time()-t1
            pr(f"    {done:,}/{len(combos):,} ({done/len(combos)*100:.0f}%) {elapsed:.0f}s")

    elapsed = time.time()-t1
    pr(f"  Done: {len(combos):,} in {elapsed:.0f}s ({len(combos)/max(elapsed,1):.0f}/s)")

    valid1 = [r for r in results1 if r['pf']>=2.0 and r['trades']>=10 and r['fl']==0]
    valid1.sort(key=lambda x: x['pf']*(1-x['mdd']/100)*np.log(x['trades']+1), reverse=True)
    pr(f"  Valid: {len(valid1)} (PF>=2, trades>=10, FL=0)")

    pr(f"\n  [Top 30 by Composite Score]")
    pr(f"  {'#':>2} | {'TF':>3} | {'Fast':>8} | {'Slow':>8} | {'ADX':>5} | {'RSI':>6} | {'D':>1} | {'PF':>6} | {'MDD':>6} | {'Ret%':>10} | {'$':>9} | {'Tr':>3} | {'SL':>2} | {'WR':>5} | {'Sc':>5}")
    sep = f"  {'-'*2}-+-{'-'*3}-+-{'-'*8}-+-{'-'*8}-+-{'-'*5}-+-{'-'*6}-+-{'-'*1}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*9}-+-{'-'*3}-+-{'-'*2}-+-{'-'*5}-+-{'-'*5}"
    pr(sep)
    for i,r in enumerate(valid1[:30]):
        sc = r['pf']*(1-r['mdd']/100)*np.log(r['trades']+1)
        pr(f"  {i+1:>2} | {r['tf']:>3} | {r['fast']:>8} | {r['slow']:>8} | {r['adx_p']}>{r['adx_min']:>2} | {r['rsi_r']:>6} | {r['delay']:>1} | {r['pf']:>5.1f} | {r['mdd']:>5.1f}% | {r['ret']:>+9.1f}% | ${r['bal']:>8,.0f} | {r['trades']:>3} | {r['sl']:>2} | {r['wr']:>4.1f}% | {sc:>4.1f}")

    # Top by PF (trades >= 20)
    pf_top = [r for r in valid1 if r['trades']>=20]
    pf_top.sort(key=lambda x: x['pf'], reverse=True)
    pr(f"\n  [Top 15 by PF (trades>=20)]")
    pr(sep)
    for i,r in enumerate(pf_top[:15]):
        sc = r['pf']*(1-r['mdd']/100)*np.log(r['trades']+1)
        pr(f"  {i+1:>2} | {r['tf']:>3} | {r['fast']:>8} | {r['slow']:>8} | {r['adx_p']}>{r['adx_min']:>2} | {r['rsi_r']:>6} | {r['delay']:>1} | {r['pf']:>5.1f} | {r['mdd']:>5.1f}% | {r['ret']:>+9.1f}% | ${r['bal']:>8,.0f} | {r['trades']:>3} | {r['sl']:>2} | {r['wr']:>4.1f}% | {sc:>4.1f}")

    # Top by trades (for compounding, PF >= 5)
    tr_top = [r for r in valid1 if r['pf']>=5.0]
    tr_top.sort(key=lambda x: x['trades'], reverse=True)
    pr(f"\n  [Top 15 by Trade Count (PF>=5)]")
    pr(sep)
    for i,r in enumerate(tr_top[:15]):
        sc = r['pf']*(1-r['mdd']/100)*np.log(r['trades']+1)
        pr(f"  {i+1:>2} | {r['tf']:>3} | {r['fast']:>8} | {r['slow']:>8} | {r['adx_p']}>{r['adx_min']:>2} | {r['rsi_r']:>6} | {r['delay']:>1} | {r['pf']:>5.1f} | {r['mdd']:>5.1f}% | {r['ret']:>+9.1f}% | ${r['bal']:>8,.0f} | {r['trades']:>3} | {r['sl']:>2} | {r['wr']:>4.1f}% | {sc:>4.1f}")

    # Top by absolute return
    ret_top = sorted(valid1, key=lambda x: x['bal'], reverse=True)
    pr(f"\n  [Top 15 by Absolute Return]")
    pr(sep)
    for i,r in enumerate(ret_top[:15]):
        sc = r['pf']*(1-r['mdd']/100)*np.log(r['trades']+1)
        pr(f"  {i+1:>2} | {r['tf']:>3} | {r['fast']:>8} | {r['slow']:>8} | {r['adx_p']}>{r['adx_min']:>2} | {r['rsi_r']:>6} | {r['delay']:>1} | {r['pf']:>5.1f} | {r['mdd']:>5.1f}% | {r['ret']:>+9.1f}% | ${r['bal']:>8,.0f} | {r['trades']:>3} | {r['sl']:>2} | {r['wr']:>4.1f}% | {sc:>4.1f}")

    # ====================================================================
    # STAGE 2: EXIT + MARGIN OPTIMIZATION (Top 20 x exit x margin)
    # ====================================================================
    pr("\n" + "="*90)
    pr("  STAGE 2: Exit + Margin Optimization")
    pr("="*90)

    top20 = valid1[:20]
    exit_combos = list(itertools.product(
        [0.05,0.06,0.07,0.08,0.10],          # SL
        [0.03,0.04,0.05,0.06,0.08,0.10],     # TS act
        [0.02,0.03,0.04,0.05],                # TS trail
        [0.20,0.30,0.40,0.50],                # margin
    ))
    total2 = len(top20)*len(exit_combos)
    pr(f"  Top entries: {len(top20)}, Exit+Margin combos: {len(exit_combos)}, Total: {total2:,}")

    results2 = []; done = 0; t2 = time.time()
    for entry in top20:
        tf = entry['tf']; df = tfs[tf]
        ft=entry['fast'].split('(')[0]; fl=int(entry['fast'].split('(')[1].rstrip(')'))
        st=entry['slow'].split('(')[0]; sl_=int(entry['slow'].split('(')[1].rstrip(')'))
        fk=f"{tf}_{ft}_{fl}"; sk=f"{tf}_{st}_{sl_}"; ak=f"{tf}_{entry['adx_p']}"
        am=entry['adx_min']; rmin=int(entry['rsi_r'].split('-')[0]); rmax=int(entry['rsi_r'].split('-')[1])
        dl=entry['delay']
        for sl,ta,tt,m in exit_combos:
            if sl*m*10 > 0.95: continue
            r = fast_bt(df['close'].values,df['high'].values,df['low'].values,
                        ma_cache[fk],ma_cache[sk],adx_cache[ak],df['rsi14'].values,
                        am,rmin,rmax,sl,ta,tt,m,10,delay=dl,skip_same_dir=(dl>0))
            r.update({'tf':tf,'fast':entry['fast'],'slow':entry['slow'],
                      'adx_p':entry['adx_p'],'adx_min':am,'rsi_r':entry['rsi_r'],
                      'delay':dl,'sl_pct':sl,'ts_act':ta,'ts_trail':tt,'margin':m})
            r['score'] = r['pf']*(1-r['mdd']/100)*np.log(r['trades']+1)
            results2.append(r)
            done += 1
        if done % 3000 == 0:
            pr(f"    {done:,}/{total2:,} ({done/total2*100:.0f}%) {time.time()-t2:.0f}s")

    pr(f"  Done: {len(results2):,} in {time.time()-t2:.0f}s")

    valid2 = [r for r in results2 if r['fl']==0 and r['trades']>=10]
    valid2.sort(key=lambda x: x['score'], reverse=True)

    pr(f"\n  {'='*90}")
    pr(f"  FINAL RESULTS")
    pr(f"  {'='*90}")

    hdr = f"  {'#':>2} | {'TF':>3} | {'Fast':>8} | {'Slow':>8} | {'ADX':>5} | {'D':>1} | {'SL':>4} | {'TS':>8} | {'M':>4} | {'PF':>6} | {'MDD':>6} | {'Ret%':>10} | {'$':>10} | {'Tr':>3} | {'WR':>5} | {'Sc':>5}"
    sep2 = f"  {'-'*2}-+-{'-'*3}-+-{'-'*8}-+-{'-'*8}-+-{'-'*5}-+-{'-'*1}-+-{'-'*4}-+-{'-'*8}-+-{'-'*4}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*3}-+-{'-'*5}-+-{'-'*5}"

    def prow(i, r):
        pr(f"  {i:>2} | {r['tf']:>3} | {r['fast']:>8} | {r['slow']:>8} | {r['adx_p']}>{r['adx_min']:>2} | {r['delay']:>1} | {r['sl_pct']:.0%} | +{r['ts_act']:.0%}/-{r['ts_trail']:.0%} | {r['margin']:.0%} | {r['pf']:>5.1f} | {r['mdd']:>5.1f}% | {r['ret']:>+9.1f}% | ${r['bal']:>9,.0f} | {r['trades']:>3} | {r['wr']:>4.1f}% | {r['score']:>4.1f}")

    pr(f"\n  [TOP 30 by Score]")
    pr(hdr); pr(sep2)
    for i,r in enumerate(valid2[:30]): prow(i+1, r)

    pf_top2 = [r for r in valid2 if r['trades']>=20]
    pf_top2.sort(key=lambda x: x['pf'], reverse=True)
    pr(f"\n  [TOP 15 by PF (trades>=20)]")
    pr(hdr); pr(sep2)
    for i,r in enumerate(pf_top2[:15]): prow(i+1, r)

    ret_top2 = sorted(valid2, key=lambda x: x['bal'], reverse=True)
    pr(f"\n  [TOP 15 by Return]")
    pr(hdr); pr(sep2)
    for i,r in enumerate(ret_top2[:15]): prow(i+1, r)

    tr_top2 = [r for r in valid2 if r['pf']>=5]
    tr_top2.sort(key=lambda x: x['trades'], reverse=True)
    pr(f"\n  [TOP 15 by Trades (PF>=5)]")
    pr(hdr); pr(sep2)
    for i,r in enumerate(tr_top2[:15]): prow(i+1, r)

    mdd_top2 = [r for r in valid2 if r['ret']>500]
    mdd_top2.sort(key=lambda x: x['mdd'])
    pr(f"\n  [TOP 15 by Lowest MDD (ret>500%)]")
    pr(hdr); pr(sep2)
    for i,r in enumerate(mdd_top2[:15]): prow(i+1, r)

    # Save
    pd.DataFrame(results2).to_csv(DATA_DIR/'v17_opt_results.csv', index=False)
    total_combos = len(combos) + len(results2)
    pr(f"\n  Total combos tested: {total_combos:,}")
    pr(f"  Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")
    pr(f"  Saved: v17_opt_results.csv")

if __name__ == '__main__':
    import traceback
    try: main()
    except Exception as e: traceback.print_exc(); pr(f"ERROR: {e}")
