"""v25.2 Fast Optimizer: Multi-TF indicators + Partial Exit + 1M+ combos
   5m에서는 EMA만, 10m+에서 HMA 추가 → 메모리/속도 최적화"""
import numpy as np, pandas as pd, json, time
from bt_fast import (load_5m_data, build_mtf, calc_ema, calc_rsi, calc_adx,
                     calc_hma, calc_ma)

print("=== v25.2 Fast Optimizer ===")
t0 = time.time()

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)

# Pre-compute indicators (lightweight)
print("Pre-computing indicators...", flush=True)
ind = {}
for tf in ['5m','10m','15m','30m']:
    df = mtf[tf]
    c = df['close']; h = df['high']; l = df['low']
    d = {
        'close': c.values.astype(np.float64),
        'high': h.values.astype(np.float64),
        'low': l.values.astype(np.float64),
        'n': len(df),
        'months': (df['time'].dt.year*100 + df['time'].dt.month).values,
        'time': df['time'].values,
    }
    # Fast MAs (EMA only for 5m, EMA+HMA for others)
    for p in [3,5,7]:
        d[f'ema_{p}'] = calc_ema(c, p).values.astype(np.float64)
    # Slow MAs
    for p in [100,150,200,250,300]:
        d[f'ema_{p}'] = calc_ema(c, p).values.astype(np.float64)
    # HMA only for 10m+ (5m too slow)
    if tf != '5m':
        for p in [3,5,7]:
            d[f'hma_{p}'] = calc_hma(c, p).values.astype(np.float64)
    # RSI, ADX
    d['rsi_14'] = calc_rsi(c, 14).values.astype(np.float64)
    d['adx_20'] = calc_adx(h, l, c, 20).values.astype(np.float64)
    ind[tf] = d
    print(f"  {tf}: {len(df):,} candles done", flush=True)

# TF mapping
tf_map = {}
for ctf in ['5m','10m','15m','30m']:
    for itf in ['5m','10m','15m']:
        key = f'{ctf}_{itf}'
        ct = ind[ctf]['time'].astype('int64')
        it = ind[itf]['time'].astype('int64')
        idx = np.searchsorted(it, ct, side='right') - 1
        tf_map[key] = np.clip(idx, 0, ind[itf]['n']-1)

print(f"Indicators ready in {time.time()-t0:.1f}s", flush=True)

# Backtest
def run_bt(ctf, ft, fp, st, sp, atf, amin, rtf, rmin, rmax, sl, ta, tp, pp, pat):
    d = ind[ctf]
    n = d['n']
    maf = d.get(f'{ft}_{fp}')
    mas = d.get(f'{st}_{sp}')
    if maf is None or mas is None: return None

    closes=d['close']; highs=d['high']; lows=d['low']; months=d['months']
    adx_arr=ind[atf]['adx_20']; rsi_arr=ind[rtf]['rsi_14']
    amap=tf_map[f'{ctf}_{atf}']; rmap=tf_map[f'{ctf}_{rtf}']

    bal=3000.0; pkb=bal; pos=0; ep=0.0; su=0.0; ppnl=0.0; trail=False; rem=1.0
    msb=bal; cm=0; mp=False; pdone=False
    tot=0; wc=0; lc=0; gp=0.0; gl=0.0; slc=0; tslc=0; revc=0; flc=0
    rpk=bal; mdd=0.0; lm=0
    lev=10; mgn=0.40; mgr=0.20; ddt=0.20; mll=-0.20; fee=0.0004; liqd=0.1
    yrk=[]; yrs=[]; yre=[]

    for i in range(1, n):
        cp=closes[i]; hp=highs[i]; lp=lows[i]
        if np.isnan(maf[i]) or np.isnan(mas[i]): continue
        ai=amap[i]; ri=rmap[i]
        if np.isnan(adx_arr[ai]) or np.isnan(rsi_arr[ri]): continue
        av=adx_arr[ai]; rv=rsi_arr[ri]

        mk=months[i]
        if mk!=cm:
            if cm!=0 and bal<msb: lm+=1
            cm=mk; msb=bal; mp=False
        yk=mk//100
        if not yrk or yrk[-1]!=yk: yrk.append(yk); yrs.append(bal); yre.append(bal)
        yre[-1]=bal

        if pos!=0:
            if pos==1: pnl=(cp-ep)/ep; pkc=(hp-ep)/ep; lwc=(lp-ep)/ep
            else: pnl=(ep-cp)/ep; pkc=(ep-lp)/ep; lwc=(ep-hp)/ep
            if pkc>ppnl: ppnl=pkc
            if lwc<=-liqd:
                pu=su*rem*(-liqd)-su*rem*fee; bal+=pu; bal=max(bal,0)
                tot+=1; lc+=1; gl+=abs(pu); flc+=1; pos=0
                rpk=max(rpk,bal); dd=(rpk-bal)/rpk if rpk>0 else 0; mdd=max(mdd,dd)
                continue
            if lwc<=-sl:
                pu=su*rem*(-sl)-su*rem*fee; bal+=pu; bal=max(bal,0)
                tot+=1; lc+=1; gl+=abs(pu); slc+=1; pos=0
                rpk=max(rpk,bal); dd=(rpk-bal)/rpk if rpk>0 else 0; mdd=max(mdd,dd)
                continue
            if pp>0 and not pdone and ppnl>=pat:
                psu=su*pp; pu=psu*rem*pnl-psu*rem*fee; bal+=pu
                if pnl>0: gp+=pu
                rem*=(1.0-pp); pdone=True
            if ppnl>=ta: trail=True
            if trail:
                tl=ppnl-tp
                if pnl<=tl:
                    pu=su*rem*tl-su*rem*fee; bal+=pu; tot+=1; tslc+=1
                    if tl>0: wc+=1; gp+=pu
                    else: lc+=1; gl+=abs(pu)
                    pos=0; rpk=max(rpk,bal); dd=(rpk-bal)/rpk if rpk>0 else 0; mdd=max(mdd,dd)
                    continue
            cu=maf[i]>mas[i] and maf[i-1]<=mas[i-1]
            cd=maf[i]<mas[i] and maf[i-1]>=mas[i-1]
            ao=av>=amin; ro=rmin<=rv<=rmax
            rvs=False; nd=0
            if pos==1 and cd and ao and ro: rvs=True; nd=-1
            elif pos==-1 and cu and ao and ro: rvs=True; nd=1
            if rvs:
                pu=su*rem*pnl-su*rem*fee; bal+=pu; tot+=1; revc+=1
                if pnl>0: wc+=1; gp+=pu
                else: lc+=1; gl+=abs(pu)
                pos=0; rpk=max(rpk,bal); dd=(rpk-bal)/rpk if rpk>0 else 0; mdd=max(mdd,dd)
                if bal>10:
                    mg=mgn
                    if pkb>0 and (pkb-bal)/pkb>ddt: mg=mgr
                    mu=bal*mg; s2=mu*lev; bal-=s2*fee
                    pos=nd; ep=cp; su=s2; ppnl=0.0; trail=False; rem=1.0; pdone=False
                    pkb=max(pkb,bal)
                continue
        if pos==0 and bal>10:
            if mll<0 and msb>0 and (bal-msb)/msb<mll: mp=True
            if mp: continue
            cu=maf[i]>mas[i] and maf[i-1]<=mas[i-1]
            cd=maf[i]<mas[i] and maf[i-1]>=mas[i-1]
            ao=av>=amin; ro=rmin<=rv<=rmax
            sig=0
            if cu and ao and ro: sig=1
            elif cd and ao and ro: sig=-1
            if sig!=0:
                mg=mgn
                if pkb>0 and (pkb-bal)/pkb>ddt: mg=mgr
                mu=bal*mg; s2=mu*lev; bal-=s2*fee
                pos=sig; ep=cp; su=s2; ppnl=0.0; trail=False; rem=1.0; pdone=False
                pkb=max(pkb,bal)
        pkb=max(pkb,bal); rpk=max(rpk,bal)

    if pos!=0 and n>0:
        if pos==1: pf=(closes[n-1]-ep)/ep
        else: pf=(ep-closes[n-1])/ep
        pu=su*rem*pf-su*rem*fee; bal+=pu; tot+=1
        if pf>0: wc+=1; gp+=pu
        else: lc+=1; gl+=abs(pu)
    if yrk: yre[-1]=bal
    if cm!=0 and bal<msb: lm+=1
    if tot<5: return None

    pfv=gp/gl if gl>0 else gp
    wr=wc/tot*100 if tot>0 else 0
    yearly={}
    for j in range(len(yrk)):
        if yrs[j]>0: yearly[str(yrk[j])]=round((yre[j]-yrs[j])/yrs[j]*100,1)
    rec=[yearly.get(y,0) for y in ['2023','2024','2025','2026'] if y in yearly]
    wps = gp/wc if wc>0 else 0
    lps = gl/lc if lc>0 else 0
    avg_w = wps/(bal/tot)*100 if tot>0 and wc>0 else 0
    avg_l = lps/(bal/tot)*100 if tot>0 and lc>0 else 0

    return {'bal':round(bal),'pf':round(pfv,2),'mdd':round(mdd*100,1),
            'tr':tot,'wr':round(wr,1),'sl':slc,'tsl':tslc,'rev':revc,'fl':flc,
            'yr':yearly,'lm':lm,'rec':round(np.mean(rec) if rec else 0,1),
            'cfg':f'{ctf} {ft}({fp})/{st}({sp}) A20@{atf}>={amin} R14@{rtf}:{rmin}-{rmax} SL{sl:.2f} TA{ta:.2f}/{tp:.2f} P{pp:.0%}@{pat:.0%}'}

def score(r):
    pf=r['pf']; bal=r['bal']; mdd=r['mdd']; rec=r['rec']
    if bal<=3000 or pf<=0: return 0
    base=pf*np.log1p((bal-3000)/3000)
    mp=0.3 if mdd>70 else(0.5 if mdd>55 else(0.8 if mdd>40 else 1.0))
    rp=1.5 if rec>80 else(1.3 if rec>40 else(1.0 if rec>0 else 0.7))
    return base*mp*rp

# === Phase 1: MA x TF x Filter ===
cross_tfs = ['5m','10m','15m','30m']
fast_mas_by_tf = {
    '5m':  [('ema',3),('ema',5),('ema',7)],
    '10m': [('ema',3),('ema',5),('ema',7),('hma',3),('hma',5),('hma',7)],
    '15m': [('ema',3),('ema',5),('ema',7),('hma',3),('hma',5),('hma',7)],
    '30m': [('ema',3),('ema',5),('ema',7),('hma',3),('hma',5),('hma',7)],
}
slow_mas = [('ema',100),('ema',150),('ema',200),('ema',250),('ema',300)]
adx_tfs = ['5m','10m','15m']
adx_mins = [25,30,35]
rsi_tfs = ['5m','10m','15m']
rsi_ranges = [(35,65),(40,60),(40,75),(30,70)]

p1_total = sum(len(fast_mas_by_tf[ctf]) for ctf in cross_tfs) * len(slow_mas) * len(adx_tfs) * len(adx_mins) * len(rsi_tfs) * len(rsi_ranges)
print(f"\n=== Phase 1: {p1_total:,} combos ===", flush=True)

p1_results = []
cnt = 0
t1 = time.time()
for ctf in cross_tfs:
    for ft, fp in fast_mas_by_tf[ctf]:
        for st, sp in slow_mas:
            for atf in adx_tfs:
                for amin in adx_mins:
                    for rtf in rsi_tfs:
                        for rmin, rmax in rsi_ranges:
                            cnt += 1
                            if cnt % 1000 == 0:
                                el=time.time()-t1; rate=cnt/el if el>0 else 0
                                eta=(p1_total-cnt)/rate if rate>0 else 0
                                print(f"  P1: {cnt:,}/{p1_total:,} ({cnt/p1_total*100:.1f}%) {rate:.0f}/s ETA:{eta:.0f}s found:{len(p1_results)}", flush=True)
                            r = run_bt(ctf, ft, fp, st, sp, atf, amin, rtf, rmin, rmax, 0.06, 0.07, 0.03, 0, 0)
                            if r and r['fl']==0 and r['bal']>5000:
                                p1_results.append(r)

print(f"\nPhase 1: {len(p1_results)} found in {time.time()-t1:.1f}s", flush=True)
p1_results.sort(key=score, reverse=True)
top200 = p1_results[:200]
print(f"\nTop 10 Phase 1:")
for i,r in enumerate(top200[:10]):
    print(f"  #{i+1} ${r['bal']:>11,} PF:{r['pf']:>6.2f} MDD:{r['mdd']:>5.1f}% TR:{r['tr']:>3} | {r['cfg']}")

# === Phase 2: SL x Trail x Partial ===
sl_pcts = [0.03,0.04,0.05,0.06,0.07]
trail_acts = [0.04,0.05,0.06,0.07,0.08]
trail_pcts = [0.02,0.03,0.04]
partials = [(0,0),(0.5,0.03),(0.5,0.05),(0.3,0.04)]

p2_total = len(top200)*len(sl_pcts)*len(trail_acts)*len(trail_pcts)*len(partials)
print(f"\n=== Phase 2: {p2_total:,} combos on top {len(top200)} ===", flush=True)

p2_results = []
cnt2 = 0
t2 = time.time()
for r in top200:
    c = r['cfg']
    ps = c.split()
    ctf = ps[0]
    mi = ps[1].split('/')
    ft=mi[0].split('(')[0]; fp=int(mi[0].split('(')[1].rstrip(')'))
    st=mi[1].split('(')[0]; sp=int(mi[1].split('(')[1].rstrip(')'))
    ap = ps[2]; atf=ap.split('@')[1].split('>=')[0]; amin=int(ap.split('>=')[1])
    rp = ps[3]; rtf=rp.split('@')[1].split(':')[0]
    rng=rp.split(':')[1]; rmin=int(rng.split('-')[0]); rmax=int(rng.split('-')[1])

    for sl in sl_pcts:
        for ta in trail_acts:
            for tp in trail_pcts:
                if tp>=ta: continue
                for pp,pat in partials:
                    cnt2+=1
                    if cnt2%5000==0:
                        el=time.time()-t2; rate=cnt2/el if el>0 else 0
                        eta=(p2_total-cnt2)/rate if rate>0 else 0
                        print(f"  P2: {cnt2:,}/{p2_total:,} ({cnt2/p2_total*100:.1f}%) {rate:.0f}/s ETA:{eta:.0f}s found:{len(p2_results)}", flush=True)
                    r2 = run_bt(ctf, ft, fp, st, sp, atf, amin, rtf, rmin, rmax, sl, ta, tp, pp, pat)
                    if r2 and r2['fl']==0 and r2['bal']>10000:
                        p2_results.append(r2)

print(f"\nPhase 2: {len(p2_results)} found in {time.time()-t2:.1f}s", flush=True)
p2_results.sort(key=score, reverse=True)

print(f"\n{'='*110}")
print(f"TOP 30 v25.2 RESULTS")
print(f"{'='*110}")
for i,r in enumerate(p2_results[:30]):
    print(f"#{i+1:>2} ${r['bal']:>12,} PF:{r['pf']:>7.2f} MDD:{r['mdd']:>5.1f}% TR:{r['tr']:>4} WR:{r['wr']:>4.1f}% SL:{r['sl']} TSL:{r['tsl']} REV:{r['rev']} FL:{r['fl']}")
    print(f"     {r['cfg']}")
    print(f"     {' | '.join(f'{k}:{v:+.0f}%' for k,v in sorted(r['yr'].items()))}")

total_tested = p1_total + p2_total
save = {'total_combos':total_tested,'phase1':p1_total,'phase2':p2_total,
        'p1_found':len(p1_results),'p2_found':len(p2_results),
        'top30':p2_results[:30],'elapsed':round(time.time()-t0,1)}
with open('v252_results.json','w',encoding='utf-8') as f:
    json.dump(save, f, indent=2, ensure_ascii=False)

print(f"\nTotal: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f}min)")
print(f"Tested: {total_tested:,}")
print(f"Saved: v252_results.json")
