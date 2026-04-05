"""
v16.4 Focused 3-Stage Optimizer
Targeted search based on v16.0 findings (WMA best, 30m best)
Total ~800 combos = ~10 min execution
"""
import pandas as pd, numpy as np, time, sys, itertools
from pathlib import Path
from btc_v164_backtest import (load_5m_data, resample, calc_wma, calc_ema, calc_sma,
                                calc_adx, calc_rsi, calc_atr, add_indicators, FEE_RATE)

DATA_DIR = Path(r'D:\filesystem\futures\btc_V1\test4')
CAPITAL = 3000

def pr(*args, **kw):
    kw['flush'] = True
    __builtins__['print'](*args, **kw) if isinstance(__builtins__, dict) else print(*args, **kw)

# ====================================================================
# FAST BACKTEST ENGINE
# ====================================================================
def fast_bt(closes, highs, lows, fast_ma, slow_ma, adx_vals, rsi_vals,
            adx_min, rsi_min, rsi_max, sl_pct, ts_act, ts_trail, margin, leverage,
            fee=FEE_RATE, capital=CAPITAL):
    n = len(closes)
    bal = float(capital); pos_dir = 0; pos_epx = 0.; pos_sz = 0.; pos_pk = 0.
    pos_hi = 0.; pos_lo = 0.; prev_f = np.nan; prev_s = np.nan
    peak_bal = bal; mdd = 0.; tp = 0.; tl = 0.
    nt = 0; nw = 0; nsl = 0; ntsl = 0; nrev = 0; fl = 0

    for i in range(1, n):
        f = fast_ma[i]; s = slow_ma[i]; a = adx_vals[i]; r = rsi_vals[i]
        if np.isnan(f) or np.isnan(s) or np.isnan(a) or np.isnan(r):
            prev_f = f; prev_s = s; continue
        if np.isnan(prev_f) or np.isnan(prev_s):
            prev_f = f; prev_s = s; continue
        cpx = closes[i]; hpx = highs[i]; lpx = lows[i]
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
                gc = prev_f <= prev_s and f > s; dc = prev_f >= prev_s and f < s
                if pos_dir==1 and dc and a>=adx_min and rsi_min<=r<=rsi_max and roi<0.20:
                    ex = 3; ex_px = cpx
                elif pos_dir==-1 and gc and a>=adx_min and rsi_min<=r<=rsi_max and roi<0.20:
                    ex = 3; ex_px = cpx
            if ex > 0:
                tr = (ex_px/pos_epx-1.) if pos_dir==1 else (1.-ex_px/pos_epx)
                pnl = pos_sz*leverage*tr - pos_sz*leverage*fee*2; bal += pnl; nt += 1
                if pnl > 0: nw += 1; tp += pnl
                else: tl += abs(pnl)
                if ex==1: nsl+=1
                elif ex==2: ntsl+=1
                else: nrev+=1
                if bal <= 0: fl += 1; bal = 0.01
                nd = -pos_dir if ex==3 else 0; pos_dir = 0
                if nd != 0 and bal > 0:
                    pos_dir=nd; pos_epx=cpx; pos_sz=bal*margin; pos_pk=0.; pos_hi=cpx; pos_lo=cpx
        if pos_dir == 0 and bal > 0:
            gc = prev_f<=prev_s and f>s; dc = prev_f>=prev_s and f<s
            sig = 1 if gc else (-1 if dc else 0)
            if sig and a < adx_min: sig = 0
            if sig and not (rsi_min <= r <= rsi_max): sig = 0
            if sig:
                pos_dir=sig; pos_epx=cpx; pos_sz=bal*margin; pos_pk=0.; pos_hi=cpx; pos_lo=cpx
        prev_f = f; prev_s = s
        if bal > peak_bal: peak_bal = bal
        dd = (peak_bal-bal)/peak_bal if peak_bal > 0 else 0
        if dd > mdd: mdd = dd
    if pos_dir != 0:
        tr = (closes[-1]/pos_epx-1.) if pos_dir==1 else (1.-closes[-1]/pos_epx)
        pnl = pos_sz*leverage*tr - pos_sz*leverage*fee*2; bal += pnl; nt += 1
        if pnl > 0: nw+=1; tp+=pnl
        else: tl+=abs(pnl)
    pf = tp/tl if tl > 0 else (999. if tp > 0 else 0.)
    return {'bal':bal, 'ret':(bal-capital)/capital*100, 'pf':pf, 'mdd':mdd*100,
            'trades':nt, 'wr':nw/nt*100 if nt else 0, 'sl':nsl, 'tsl':ntsl, 'rev':nrev, 'fl':fl}

def get_ma(close_arr, ma_type, period):
    s = pd.Series(close_arr)
    if ma_type == 'wma': return calc_wma(s, period).values
    elif ma_type == 'ema': return calc_ema(s, period).values
    elif ma_type == 'sma': return calc_sma(s, period).values
    elif ma_type == 'dema':
        e1 = calc_ema(s, period); return (2*e1 - calc_ema(e1, period)).values
    elif ma_type == 'hma':
        h = max(int(period/2),1); sq = max(int(np.sqrt(period)),1)
        return calc_wma(2*calc_wma(s,h)-calc_wma(s,period), sq).values
    return calc_ema(s, period).values

# ====================================================================
def main():
    print("="*80, flush=True)
    print("  v16.4 Focused 3-Stage Optimizer", flush=True)
    print("="*80, flush=True)

    t0 = time.time()
    print("\n[DATA] Loading...", flush=True)
    df_5m = load_5m_data()
    df_30m = resample(df_5m, '30min')
    df_30m = add_indicators(df_30m)
    C = df_30m['close'].values; H = df_30m['high'].values; L = df_30m['low'].values
    rsi14 = df_30m['rsi14'].values
    adx14 = calc_adx(df_30m['high'], df_30m['low'], df_30m['close'], 14).values
    adx20 = calc_adx(df_30m['high'], df_30m['low'], df_30m['close'], 20).values
    print(f"  Ready: {len(df_30m):,} candles, {time.time()-t0:.1f}s", flush=True)

    # Pre-compute all MAs
    print("\n[MA CACHE] Pre-computing...", flush=True)
    ma_cache = {}
    for mt in ['wma','ema','dema','hma','sma']:
        for p in [2,3,4,5,7,10]:
            ma_cache[f"{mt}_{p}"] = get_ma(C, mt, p)
    for mt in ['ema','sma','wma']:
        for p in [100,150,200,250,300]:
            ma_cache[f"{mt}_{p}"] = get_ma(C, mt, p)
    adx_d = {14: adx14, 20: adx20}
    print(f"  {len(ma_cache)} MAs cached", flush=True)

    # ======================
    # STAGE 1: Entry search
    # ======================
    print("\n" + "="*80, flush=True)
    print("  STAGE 1: MA + Filter Search", flush=True)
    print("="*80, flush=True)

    fast_list = [(t,p) for t in ['wma'] for p in [2,3,4,5,7]] + \
                [(t,3) for t in ['ema','dema','hma','sma']]
    slow_list = [('ema',p) for p in [150,200,250,300]]
    adx_list = [(14,30),(14,35),(14,40),(20,30),(20,35),(20,40)]
    rsi_list = [(35,65),(30,70)]
    SL=0.08; TA=0.04; TT=0.03; M=0.25; LEV=10

    combos = list(itertools.product(fast_list, slow_list, adx_list, rsi_list))
    print(f"  Combinations: {len(combos)}", flush=True)
    results1 = []
    for (ft,fl),(st,sl),(ap,am),(rmin,rmax) in combos:
        fma = ma_cache[f"{ft}_{fl}"]; sma = ma_cache[f"{st}_{sl}"]
        adx = adx_d[ap]
        r = fast_bt(C,H,L,fma,sma,adx,rsi14, am,rmin,rmax, SL,TA,TT,M,LEV)
        r['fast']=f"{ft}({fl})"; r['slow']=f"{st}({sl})"
        r['adx_p']=ap; r['adx_min']=am; r['rsi_min']=rmin; r['rsi_max']=rmax
        results1.append(r)

    valid1 = [r for r in results1 if r['pf']>=1.5 and r['trades']>=10 and r['fl']==0]
    valid1.sort(key=lambda x: x['pf'], reverse=True)
    print(f"\n  Stage 1 done: {len(combos)} combos, {len(valid1)} valid, {time.time()-t0:.1f}s", flush=True)

    print(f"\n  [Stage 1 Top 25]", flush=True)
    print(f"  {'#':>2} | {'Fast':>7} | {'Slow':>8} | {'ADX':>9} | {'RSI':>7} | {'PF':>6} | {'MDD':>6} | {'Ret%':>10} | {'$':>9} | {'Tr':>3} | {'SL':>2} | {'WR':>5}", flush=True)
    print(f"  {'-'*2}-+-{'-'*7}-+-{'-'*8}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*9}-+-{'-'*3}-+-{'-'*2}-+-{'-'*5}", flush=True)
    for i,r in enumerate(valid1[:25]):
        print(f"  {i+1:>2} | {r['fast']:>7} | {r['slow']:>8} | {r['adx_p']}>{r['adx_min']:>2} | {r['rsi_min']}-{r['rsi_max']} | {r['pf']:>5.2f} | {r['mdd']:>5.1f}% | {r['ret']:>+9.1f}% | ${r['bal']:>8,.0f} | {r['trades']:>3} | {r['sl']:>2} | {r['wr']:>4.1f}%", flush=True)

    # ======================
    # STAGE 2: Exit optimization (top 15 entries x 36 exit combos)
    # ======================
    print("\n" + "="*80, flush=True)
    print("  STAGE 2: Exit Optimization", flush=True)
    print("="*80, flush=True)

    top15 = valid1[:15]
    sl_list = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
    ta_list = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    tt_list = [0.02, 0.03, 0.04, 0.05]
    exit_combos = list(itertools.product(sl_list, ta_list, tt_list))
    total2 = len(top15) * len(exit_combos)
    print(f"  Top entries: {len(top15)}, Exit combos: {len(exit_combos)}, Total: {total2}", flush=True)

    results2 = []
    done = 0
    for entry in top15:
        ft=entry['fast'].split('(')[0]; fl=int(entry['fast'].split('(')[1].rstrip(')'))
        st=entry['slow'].split('(')[0]; sl_=int(entry['slow'].split('(')[1].rstrip(')'))
        fma = ma_cache[f"{ft}_{fl}"]; sma = ma_cache[f"{st}_{sl_}"]
        adx = adx_d[entry['adx_p']]; am=entry['adx_min']
        rmin=entry['rsi_min']; rmax=entry['rsi_max']
        for s,ta,tt in exit_combos:
            r = fast_bt(C,H,L,fma,sma,adx,rsi14, am,rmin,rmax, s,ta,tt,M,LEV)
            r.update({'fast':entry['fast'],'slow':entry['slow'],
                      'adx_p':entry['adx_p'],'adx_min':am,'rsi_min':rmin,'rsi_max':rmax,
                      'sl_pct':s,'ts_act':ta,'ts_trail':tt})
            results2.append(r)
            done += 1
        if done % 500 == 0:
            print(f"    {done}/{total2} ({done/total2*100:.0f}%)", flush=True)

    valid2 = [r for r in results2 if r['pf']>=2.0 and r['trades']>=10 and r['fl']==0]
    valid2.sort(key=lambda x: x['pf']*(1-x['mdd']/100)*np.log(x['trades']+1), reverse=True)
    print(f"\n  Stage 2 done: {total2} combos, {len(valid2)} valid, {time.time()-t0:.1f}s", flush=True)

    print(f"\n  [Stage 2 Top 20]", flush=True)
    print(f"  {'#':>2} | {'Fast':>7} | {'Slow':>8} | {'ADX':>5} | {'SL':>4} | {'TS':>8} | {'PF':>6} | {'MDD':>6} | {'Ret%':>10} | {'$':>9} | {'Tr':>3} | {'Sc':>5}", flush=True)
    sep = f"  {'-'*2}-+-{'-'*7}-+-{'-'*8}-+-{'-'*5}-+-{'-'*4}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*9}-+-{'-'*3}-+-{'-'*5}"
    print(sep, flush=True)
    for i,r in enumerate(valid2[:20]):
        sc = r['pf']*(1-r['mdd']/100)*np.log(r['trades']+1)
        print(f"  {i+1:>2} | {r['fast']:>7} | {r['slow']:>8} | {r['adx_p']}>{r['adx_min']:>2} | {r['sl_pct']:.0%} | +{r['ts_act']:.0%}/-{r['ts_trail']:.0%} | {r['pf']:>5.2f} | {r['mdd']:>5.1f}% | {r['ret']:>+9.1f}% | ${r['bal']:>8,.0f} | {r['trades']:>3} | {sc:>4.1f}", flush=True)

    # ======================
    # STAGE 3: Margin optimization (top 30 x 7 margins)
    # ======================
    print("\n" + "="*80, flush=True)
    print("  STAGE 3: Margin Optimization + Final", flush=True)
    print("="*80, flush=True)

    top30 = valid2[:30]
    margins = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    results3 = []
    for entry in top30:
        ft=entry['fast'].split('(')[0]; fl=int(entry['fast'].split('(')[1].rstrip(')'))
        st=entry['slow'].split('(')[0]; sl_=int(entry['slow'].split('(')[1].rstrip(')'))
        fma=ma_cache[f"{ft}_{fl}"]; sma=ma_cache[f"{st}_{sl_}"]
        adx=adx_d[entry['adx_p']]; am=entry['adx_min']
        s=entry['sl_pct']; ta=entry['ts_act']; tt=entry['ts_trail']
        rmin=entry['rsi_min']; rmax=entry['rsi_max']
        for m in margins:
            if s*m*10 > 0.95: continue
            r = fast_bt(C,H,L,fma,sma,adx,rsi14, am,rmin,rmax, s,ta,tt,m,10)
            r.update({'fast':entry['fast'],'slow':entry['slow'],
                      'adx_p':entry['adx_p'],'adx_min':am,'rsi_min':rmin,'rsi_max':rmax,
                      'sl_pct':s,'ts_act':ta,'ts_trail':tt,'margin':m})
            r['score'] = r['pf']*(1-r['mdd']/100)*np.log(r['trades']+1)
            results3.append(r)

    results3.sort(key=lambda x: x['score'], reverse=True)
    valid3 = [r for r in results3 if r['fl']==0 and r['trades']>=10]

    print(f"  Stage 3 done: {len(results3)} combos, {time.time()-t0:.1f}s", flush=True)

    # FINAL REPORTS
    hdr = f"  {'#':>2} | {'Fast':>7} | {'Slow':>8} | {'ADX':>5} | {'SL':>4} | {'TS':>8} | {'M':>4} | {'PF':>6} | {'MDD':>6} | {'Ret%':>10} | {'$':>10} | {'Tr':>3} | {'WR':>5} | {'Sc':>5}"
    sep = f"  {'-'*2}-+-{'-'*7}-+-{'-'*8}-+-{'-'*5}-+-{'-'*4}-+-{'-'*8}-+-{'-'*4}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*3}-+-{'-'*5}-+-{'-'*5}"

    def prow(i, r):
        print(f"  {i:>2} | {r['fast']:>7} | {r['slow']:>8} | {r['adx_p']}>{r['adx_min']:>2} | {r['sl_pct']:.0%} | +{r['ts_act']:.0%}/-{r['ts_trail']:.0%} | {r['margin']:.0%} | {r['pf']:>5.2f} | {r['mdd']:>5.1f}% | {r['ret']:>+9.1f}% | ${r['bal']:>9,.0f} | {r['trades']:>3} | {r['wr']:>4.1f}% | {r['score']:>4.1f}", flush=True)

    print(f"\n  {'='*80}", flush=True)
    print(f"  FINAL TOP 30 (by composite score: PF*(1-MDD)*ln(trades))", flush=True)
    print(f"  {'='*80}", flush=True)
    print(hdr, flush=True); print(sep, flush=True)
    for i,r in enumerate(valid3[:30]): prow(i+1, r)

    # Top by PF (trades >= 20)
    pf_top = [r for r in valid3 if r['trades']>=20]
    pf_top.sort(key=lambda x: x['pf'], reverse=True)
    print(f"\n  [TOP 10 by PF (trades>=20)]", flush=True)
    print(hdr, flush=True); print(sep, flush=True)
    for i,r in enumerate(pf_top[:10]): prow(i+1, r)

    # Top by absolute return
    ret_top = sorted(valid3, key=lambda x: x['bal'], reverse=True)
    print(f"\n  [TOP 10 by Absolute Return]", flush=True)
    print(hdr, flush=True); print(sep, flush=True)
    for i,r in enumerate(ret_top[:10]): prow(i+1, r)

    # Top by lowest MDD (ret > 1000%)
    mdd_top = [r for r in valid3 if r['ret'] > 1000]
    mdd_top.sort(key=lambda x: x['mdd'])
    print(f"\n  [TOP 10 by Lowest MDD (ret>1000%)]", flush=True)
    print(hdr, flush=True); print(sep, flush=True)
    for i,r in enumerate(mdd_top[:10]): prow(i+1, r)

    # v16.0 baseline comparison
    print(f"\n  {'='*80}", flush=True)
    print(f"  v16.0 BASELINE vs BEST FOUND", flush=True)
    print(f"  {'='*80}", flush=True)
    v16 = fast_bt(C,H,L,ma_cache['wma_3'],ma_cache['ema_200'],adx20,rsi14, 35,35,65, 0.08,0.04,0.03,0.50,10)
    print(f"  v16.0 (WMA3/EMA200 ADX20>=35 SL8% TS+4/-3 M50%): PF {v16['pf']:.2f} | MDD {v16['mdd']:.1f}% | ${v16['bal']:,.0f} ({v16['ret']:+,.1f}%) | {v16['trades']}tr | SL:{v16['sl']}", flush=True)
    if valid3:
        b = valid3[0]
        print(f"  BEST  ({b['fast']}/{b['slow']} ADX{b['adx_p']}>{b['adx_min']} SL{b['sl_pct']:.0%} TS+{b['ts_act']:.0%}/-{b['ts_trail']:.0%} M{b['margin']:.0%}): PF {b['pf']:.2f} | MDD {b['mdd']:.1f}% | ${b['bal']:,.0f} ({b['ret']:+,.1f}%) | {b['trades']}tr | SL:{b['sl']}", flush=True)
    if pf_top:
        b = pf_top[0]
        print(f"  PF-BT ({b['fast']}/{b['slow']} ADX{b['adx_p']}>{b['adx_min']} SL{b['sl_pct']:.0%} TS+{b['ts_act']:.0%}/-{b['ts_trail']:.0%} M{b['margin']:.0%}): PF {b['pf']:.2f} | MDD {b['mdd']:.1f}% | ${b['bal']:,.0f} ({b['ret']:+,.1f}%) | {b['trades']}tr | SL:{b['sl']}", flush=True)
    if ret_top:
        b = ret_top[0]
        print(f"  RET-BT({b['fast']}/{b['slow']} ADX{b['adx_p']}>{b['adx_min']} SL{b['sl_pct']:.0%} TS+{b['ts_act']:.0%}/-{b['ts_trail']:.0%} M{b['margin']:.0%}): PF {b['pf']:.2f} | MDD {b['mdd']:.1f}% | ${b['bal']:,.0f} ({b['ret']:+,.1f}%) | {b['trades']}tr | SL:{b['sl']}", flush=True)

    # Save
    pd.DataFrame(results3).to_csv(DATA_DIR/'v164_opt_results.csv', index=False)
    print(f"\n  Saved: v164_opt_results.csv ({len(results3)} rows)", flush=True)
    print(f"  Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)", flush=True)

if __name__ == '__main__':
    import traceback
    try: main()
    except Exception as e: traceback.print_exc(); print(f"ERROR: {e}", flush=True)
