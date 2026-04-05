"""v25.2 Optimizer: Multi-TF indicators + Partial Exit + 1M+ combos"""
import numpy as np, pandas as pd, json, time, itertools
from bt_fast import (load_5m_data, build_mtf, calc_ema, calc_rsi, calc_adx,
                     calc_hma, calc_wma, calc_ma, calc_dema, calc_atr, IndicatorCache)

print("=== v25.2 Multi-TF Optimizer ===")
t0 = time.time()

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)

# Pre-compute ALL indicators for ALL timeframes
print("\nPre-computing indicators...")
ind = {}
for tf in ['5m','10m','15m','30m']:
    df = mtf[tf]
    c = df['close']; h = df['high']; l = df['low']; v = df['volume']
    ind[tf] = {
        'close': c.values, 'high': h.values, 'low': l.values,
        'time': df['time'].values, 'n': len(df),
        'months': (df['time'].dt.year*100 + df['time'].dt.month).values,
    }
    # MAs
    for ma_t in ['ema','hma']:
        for p in [3,5,7,10,14,21]:
            ind[tf][f'{ma_t}_{p}'] = calc_ma(c, ma_t, p).values
        for p in [100,150,200,250,300]:
            ind[tf][f'{ma_t}_{p}'] = calc_ma(c, ma_t, p).values
    for p in [100,150,200,250,300]:
        ind[tf][f'ema_{p}'] = calc_ema(c, p).values
    # RSI
    for p in [14]:
        ind[tf][f'rsi_{p}'] = calc_rsi(c, p).values
    # ADX
    for p in [14,20]:
        ind[tf][f'adx_{p}'] = calc_adx(h, l, c, p).values

# Time mapping between TFs (align by timestamp)
tf_map = {}
for cross_tf in ['5m','10m','15m','30m']:
    for ind_tf in ['5m','10m','15m']:
        key = f'{cross_tf}_to_{ind_tf}'
        ct = ind[cross_tf]['time'].astype('int64')
        it = ind[ind_tf]['time'].astype('int64')
        idx = np.searchsorted(it, ct, side='right') - 1
        tf_map[key] = np.clip(idx, 0, len(it)-1)

print(f"Indicators ready in {time.time()-t0:.1f}s")

# Backtest function with multi-TF + partial exit
def run_bt(cross_tf, fast_t, fast_p, slow_t, slow_p,
           adx_tf, adx_p, adx_min, rsi_tf, rsi_p, rsi_min, rsi_max,
           sl_pct, trail_act, trail_pct, partial_pct, partial_at):

    d = ind[cross_tf]
    n = d['n']
    closes = d['close']; highs = d['high']; lows = d['low']
    months = d['months']

    maf = ind[cross_tf].get(f'{fast_t}_{fast_p}')
    mas = ind[cross_tf].get(f'{slow_t}_{slow_p}')
    if maf is None or mas is None: return None

    # ADX/RSI from different TF
    adx_key = f'{cross_tf}_to_{adx_tf}'
    rsi_key = f'{cross_tf}_to_{rsi_tf}'
    adx_arr = ind[adx_tf].get(f'adx_{adx_p}')
    rsi_arr = ind[rsi_tf].get(f'rsi_{rsi_p}')
    if adx_arr is None or rsi_arr is None: return None
    adx_map = tf_map.get(adx_key)
    rsi_map = tf_map.get(rsi_key)
    if adx_map is None or rsi_map is None: return None

    bal=3000.0; peak_bal=bal; pos=0; ep=0.0; su=0.0; ppnl=0.0; trail=False; rem=1.0
    msb=bal; cm=0; mp=False
    tot=0; wc=0; lc=0; gp=0.0; gl=0.0; slc=0; tslc=0; revc=0; flc=0
    rpeak=bal; mdd=0.0; lm=0; partial_done=False
    lev=10; mg_n=0.40; mg_r=0.20; dd_th=0.20; ml=-0.20; fee=0.0004; liq_d=0.1

    # Year tracking
    yrk=[]; yrs_v=[]; yre_v=[]

    for i in range(1, n):
        cp=closes[i]; hp=highs[i]; lp=lows[i]
        if np.isnan(maf[i]) or np.isnan(mas[i]): continue

        ai = adx_map[i]; ri = rsi_map[i]
        if ai >= len(adx_arr) or ri >= len(rsi_arr): continue
        av = adx_arr[ai]; rv = rsi_arr[ri]
        if np.isnan(av) or np.isnan(rv): continue

        mk = months[i]
        if mk != cm:
            if cm != 0 and bal < msb: lm += 1
            cm=mk; msb=bal; mp=False

        yk = mk // 100
        if not yrk or yrk[-1] != yk:
            yrk.append(yk); yrs_v.append(bal); yre_v.append(bal)
        yre_v[-1] = bal

        if pos != 0:
            if pos==1: pnl=(cp-ep)/ep; pkc=(hp-ep)/ep; lwc=(lp-ep)/ep
            else: pnl=(ep-cp)/ep; pkc=(ep-lp)/ep; lwc=(ep-hp)/ep
            if pkc > ppnl: ppnl = pkc

            # Liquidation
            if lwc <= -liq_d:
                pu=su*rem*(-liq_d)-su*rem*fee; bal+=pu
                if bal<0: bal=0
                tot+=1; lc+=1; gl+=abs(pu); flc+=1; pos=0
                if bal>rpeak: rpeak=bal
                dd=(rpeak-bal)/rpeak if rpeak>0 else 0
                if dd>mdd: mdd=dd
                continue
            # SL
            if lwc <= -sl_pct:
                pu=su*rem*(-sl_pct)-su*rem*fee; bal+=pu
                if bal<0: bal=0
                tot+=1; lc+=1; gl+=abs(pu); slc+=1; pos=0
                if bal>rpeak: rpeak=bal
                dd=(rpeak-bal)/rpeak if rpeak>0 else 0
                if dd>mdd: mdd=dd
                continue

            # Partial exit
            if partial_pct > 0 and not partial_done and ppnl >= partial_at:
                part_su = su * partial_pct
                pu = part_su * rem * pnl - part_su * rem * fee
                bal += pu
                if pnl > 0: gp += pu
                rem *= (1.0 - partial_pct)
                partial_done = True

            # Trail
            if ppnl >= trail_act: trail = True
            if trail:
                tl = ppnl - trail_pct
                if pnl <= tl:
                    pu=su*rem*tl-su*rem*fee; bal+=pu; tot+=1; tslc+=1
                    if tl>0: wc+=1; gp+=pu
                    else: lc+=1; gl+=abs(pu)
                    pos=0
                    if bal>rpeak: rpeak=bal
                    dd=(rpeak-bal)/rpeak if rpeak>0 else 0
                    if dd>mdd: mdd=dd
                    continue

            # Reversal
            cu=maf[i]>mas[i] and maf[i-1]<=mas[i-1]
            cd=maf[i]<mas[i] and maf[i-1]>=mas[i-1]
            ao=av>=adx_min; ro=rsi_min<=rv<=rsi_max
            rvs=False; nd=0
            if pos==1 and cd and ao and ro: rvs=True; nd=-1
            elif pos==-1 and cu and ao and ro: rvs=True; nd=1
            if rvs:
                pu=su*rem*pnl-su*rem*fee; bal+=pu; tot+=1; revc+=1
                if pnl>0: wc+=1; gp+=pu
                else: lc+=1; gl+=abs(pu)
                pos=0
                if bal>rpeak: rpeak=bal
                dd=(rpeak-bal)/rpeak if rpeak>0 else 0
                if dd>mdd: mdd=dd
                if bal>10:
                    mg=mg_n
                    if peak_bal>0:
                        dn=(peak_bal-bal)/peak_bal
                        if dn>dd_th: mg=mg_r
                    mu=bal*mg; s2=mu*lev; bal-=s2*fee
                    pos=nd; ep=cp; su=s2; ppnl=0.0; trail=False; rem=1.0; partial_done=False
                    if bal>peak_bal: peak_bal=bal
                continue

        # Entry
        if pos==0 and bal>10:
            if ml<0 and msb>0:
                if (bal-msb)/msb<ml: mp=True
            if mp: continue
            cu=maf[i]>mas[i] and maf[i-1]<=mas[i-1]
            cd=maf[i]<mas[i] and maf[i-1]>=mas[i-1]
            ao=av>=adx_min; ro=rsi_min<=rv<=rsi_max
            sig=0
            if cu and ao and ro: sig=1
            elif cd and ao and ro: sig=-1
            if sig!=0:
                mg=mg_n
                if peak_bal>0:
                    dn=(peak_bal-bal)/peak_bal
                    if dn>dd_th: mg=mg_r
                mu=bal*mg; s2=mu*lev; bal-=s2*fee
                pos=sig; ep=cp; su=s2; ppnl=0.0; trail=False; rem=1.0; partial_done=False
                if bal>peak_bal: peak_bal=bal

        if bal>peak_bal: peak_bal=bal
        if bal>rpeak: rpeak=bal

    # Close
    if pos!=0 and n>0:
        if pos==1: pf=(closes[n-1]-ep)/ep
        else: pf=(ep-closes[n-1])/ep
        pu=su*rem*pf-su*rem*fee; bal+=pu; tot+=1
        if pf>0: wc+=1; gp+=pu
        else: lc+=1; gl+=abs(pu)

    if yrk: yre_v[-1]=bal
    if cm!=0 and bal<msb: lm+=1
    if tot<5: return None

    pf_v=gp/gl if gl>0 else gp
    wr=wc/tot*100 if tot>0 else 0
    yearly={}
    for j in range(len(yrk)):
        if yrs_v[j]>0: yearly[str(yrk[j])]=round((yre_v[j]-yrs_v[j])/yrs_v[j]*100,1)

    # Recent weight
    rec=[yearly.get(y,0) for y in ['2023','2024','2025','2026'] if y in yearly]
    rec_avg = np.mean(rec) if rec else 0

    return {'bal':round(bal),'pf':round(pf_v,2),'mdd':round(mdd*100,1),
            'tr':tot,'wr':round(wr,1),'sl':slc,'tsl':tslc,'rev':revc,'fl':flc,
            'yr':yearly,'lm':lm,'rec':round(rec_avg,1),
            'cfg':f'{cross_tf} {fast_t}({fast_p})/{slow_t}({slow_p}) A{adx_p}@{adx_tf}>={adx_min} R{rsi_p}@{rsi_tf}:{rsi_min}-{rsi_max} SL{sl_pct:.2f} TA{trail_act:.2f}/{trail_pct:.2f} P{partial_pct:.0%}@{partial_at:.0%}'}

# Parameter space
cross_tfs = ['5m','10m','15m','30m']
fast_mas = [('ema',3),('ema',5),('ema',7),('hma',3),('hma',5),('hma',7)]
slow_mas = [('ema',100),('ema',150),('ema',200),('ema',250),('ema',300)]
adx_tfs = ['5m','10m','15m']
adx_ps = [20]
adx_mins = [25,30,35]
rsi_tfs = ['5m','10m','15m']
rsi_ps = [14]
rsi_ranges = [(35,65),(40,60),(40,75),(30,70)]
sl_pcts = [0.03,0.04,0.05,0.06,0.07]
trail_acts = [0.04,0.05,0.06,0.07,0.08]
trail_pcts = [0.02,0.03,0.04]
partials = [(0,0),(0.5,0.03),(0.5,0.05)]  # (pct, at)

total = (len(cross_tfs)*len(fast_mas)*len(slow_mas)*len(adx_tfs)*len(adx_ps)*len(adx_mins)*
        len(rsi_tfs)*len(rsi_ps)*len(rsi_ranges)*len(sl_pcts)*len(trail_acts)*len(trail_pcts)*len(partials))
print(f"\nTotal combinations: {total:,}")

# Phase 1: Screen with default SL/Trail, find best MA/TF/Filter combos
print("\n=== Phase 1: MA x TF x Filter screening ===")
p1_total = len(cross_tfs)*len(fast_mas)*len(slow_mas)*len(adx_tfs)*len(adx_mins)*len(rsi_tfs)*len(rsi_ranges)
print(f"Phase 1 combos: {p1_total:,}")

p1_results = []
cnt = 0
t1 = time.time()
for ctf in cross_tfs:
    for ft, fp in fast_mas:
        for st, sp in slow_mas:
            for atf in adx_tfs:
                for amin in adx_mins:
                    for rtf in rsi_tfs:
                        for rmin, rmax in rsi_ranges:
                            cnt += 1
                            if cnt % 2000 == 0:
                                elapsed = time.time()-t1
                                rate = cnt/elapsed if elapsed>0 else 0
                                eta = (p1_total-cnt)/rate if rate>0 else 0
                                print(f"  P1: {cnt:,}/{p1_total:,} ({cnt/p1_total*100:.1f}%) {rate:.0f}/s ETA:{eta:.0f}s | found:{len(p1_results)}", flush=True)

                            r = run_bt(ctf, ft, fp, st, sp, atf, 20, amin, rtf, 14, rmin, rmax,
                                      0.06, 0.07, 0.03, 0, 0)
                            if r and r['fl']==0 and r['bal']>5000:
                                p1_results.append(r)

print(f"\nPhase 1 done: {len(p1_results)} valid strategies in {time.time()-t1:.1f}s")

# Sort by weighted score
def score(r):
    pf=r['pf']; bal=r['bal']; mdd=r['mdd']; rec=r['rec']
    if bal<=3000 or pf<=0: return 0
    base = pf * np.log1p((bal-3000)/3000)
    mp = 0.3 if mdd>70 else (0.5 if mdd>55 else (0.8 if mdd>40 else 1.0))
    rp = 1.5 if rec>80 else (1.3 if rec>40 else (1.0 if rec>0 else 0.7))
    return base * mp * rp

p1_results.sort(key=score, reverse=True)
top200 = p1_results[:200]

print(f"\nTop 10 Phase 1:")
for i, r in enumerate(top200[:10]):
    print(f"  #{i+1} ${r['bal']:>11,} PF:{r['pf']:>6.2f} MDD:{r['mdd']:>5.1f}% TR:{r['tr']:>3} WR:{r['wr']:>4.1f}% | {r['cfg']}")

# Phase 2: SL/Trail/Partial optimization on top 200
print(f"\n=== Phase 2: SL x Trail x Partial on top {len(top200)} ===")
p2_total = len(top200) * len(sl_pcts) * len(trail_acts) * len(trail_pcts) * len(partials)
print(f"Phase 2 combos: {p2_total:,}")

p2_results = []
cnt2 = 0
t2 = time.time()
for r in top200:
    cfg = r['cfg']
    # Parse config
    parts = cfg.split()
    ctf = parts[0]
    ma_part = parts[1]  # e.g. "ema(3)/ema(100)"
    ma_info = ma_part.split('/')
    ft = ma_info[0].split('(')[0]
    fp = int(ma_info[0].split('(')[1].rstrip(')'))
    st = ma_info[1].split('(')[0]
    sp = int(ma_info[1].split('(')[1].rstrip(')'))

    adx_part = parts[2]  # e.g. "A20@10m>=35"
    ap = int(adx_part.split('@')[0][1:])
    atf_amin = adx_part.split('@')[1]
    atf = atf_amin.split('>=')[0]
    amin = int(atf_amin.split('>=')[1])

    rsi_part = parts[3]  # e.g. "R14@5m:40-75"
    rp = int(rsi_part.split('@')[0][1:])
    rtf_range = rsi_part.split('@')[1]
    rtf = rtf_range.split(':')[0]
    rng = rtf_range.split(':')[1]
    rmin = int(rng.split('-')[0])
    rmax = int(rng.split('-')[1])

    for sl in sl_pcts:
        for ta in trail_acts:
            for tp in trail_pcts:
                if tp >= ta: continue  # trail pct must be < activate
                for pp, pat in partials:
                    cnt2 += 1
                    if cnt2 % 5000 == 0:
                        elapsed = time.time()-t2
                        rate = cnt2/elapsed if elapsed>0 else 0
                        eta = (p2_total-cnt2)/rate if rate>0 else 0
                        print(f"  P2: {cnt2:,}/{p2_total:,} ({cnt2/p2_total*100:.1f}%) {rate:.0f}/s ETA:{eta:.0f}s | found:{len(p2_results)}", flush=True)

                    r2 = run_bt(ctf, ft, fp, st, sp, atf, ap, amin, rtf, rp, rmin, rmax,
                               sl, ta, tp, pp, pat)
                    if r2 and r2['fl']==0 and r2['bal']>10000:
                        p2_results.append(r2)

print(f"\nPhase 2 done: {len(p2_results)} strategies in {time.time()-t2:.1f}s")

p2_results.sort(key=score, reverse=True)

# Print top 30
print(f"\n{'='*100}")
print(f"TOP 30 FINAL RESULTS")
print(f"{'='*100}")
for i, r in enumerate(p2_results[:30]):
    print(f"#{i+1:>2} ${r['bal']:>12,} PF:{r['pf']:>7.2f} MDD:{r['mdd']:>5.1f}% TR:{r['tr']:>3} WR:{r['wr']:>4.1f}% SL:{r['sl']} TSL:{r['tsl']} REV:{r['rev']} FL:{r['fl']}")
    print(f"     {r['cfg']}")
    yr = r.get('yr',{})
    print(f"     {' | '.join(f'{k}:{v:+.0f}%' for k,v in sorted(yr.items()))}")

# Save
total_tested = p1_total + p2_total
save = {
    'total_combos': total_tested,
    'total_possible': total,
    'phase1': p1_total,
    'phase2': p2_total,
    'found': len(p2_results),
    'top30': [r for r in p2_results[:30]],
    'elapsed': round(time.time()-t0, 1),
}
with open('v252_results.json','w',encoding='utf-8') as f:
    json.dump(save, f, indent=2, ensure_ascii=False)

print(f"\nTotal time: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f}min)")
print(f"Tested: {total_tested:,} (of {total:,} possible)")
print(f"Saved: v252_results.json")
