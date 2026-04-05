"""v25.2 JIT Optimizer: bt_fast Numba 엔진 활용 + 멀티TF ADX/RSI 매핑"""
import numpy as np, pandas as pd, json, time
from bt_fast import (load_5m_data, build_mtf, calc_ema, calc_rsi, calc_adx,
                     calc_hma, calc_ma, IndicatorCache, run_backtest)

print("=== v25.2 JIT Optimizer (Numba) ===")
t0 = time.time()

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)

# Extended cache: 각 TF별 ADX/RSI를 다른 TF에서 가져와 매핑
print("Building extended indicator cache...", flush=True)

class ExtCache(IndicatorCache):
    """기본 IndicatorCache + 크로스TF ADX/RSI 매핑"""
    def __init__(self, mtf):
        super().__init__(mtf)
        self.tf_maps = {}
        # 각 TF 간 시간 매핑 (cross_tf → indicator_tf)
        for ctf in ['5m','10m','15m','30m']:
            for itf in ['5m','10m','15m']:
                key = f'{ctf}_{itf}'
                ct = mtf[ctf]['time'].values.astype('int64')
                it = mtf[itf]['time'].values.astype('int64')
                idx = np.searchsorted(it, ct, side='right') - 1
                self.tf_maps[key] = np.clip(idx, 0, len(mtf[itf])-1)

    def get_mapped_rsi(self, cross_tf, ind_tf, period):
        """cross_tf 시간축에서 ind_tf의 RSI를 매핑"""
        k = (cross_tf, ind_tf, f'mapped_rsi_{period}')
        if k not in self.cache:
            rsi_orig = self.get_rsi(ind_tf, period)
            mapping = self.tf_maps[f'{cross_tf}_{ind_tf}']
            self.cache[k] = rsi_orig[mapping]
        return self.cache[k]

    def get_mapped_adx(self, cross_tf, ind_tf, period):
        """cross_tf 시간축에서 ind_tf의 ADX를 매핑"""
        k = (cross_tf, ind_tf, f'mapped_adx_{period}')
        if k not in self.cache:
            adx_orig = self.get_adx(ind_tf, period)
            mapping = self.tf_maps[f'{cross_tf}_{ind_tf}']
            self.cache[k] = adx_orig[mapping]
        return self.cache[k]

cache = ExtCache(mtf)

# Pre-warm all needed indicators
tfs = ['5m','10m','15m','30m']
ind_tfs = ['5m','10m','15m']
ma_types = ['ema','hma']
fast_periods = [3,5,7]
slow_periods = [100,150,200,250,300]

print("Pre-warming MA cache...", flush=True)
for tf in tfs:
    for mt in ma_types:
        if mt == 'hma' and tf == '5m':
            continue  # 5m HMA too slow
        for p in fast_periods + slow_periods:
            try:
                cache.get_ma(tf, mt, p)
            except:
                pass
    for itf in ind_tfs:
        cache.get_mapped_rsi(tf, itf, 14)
        cache.get_mapped_adx(tf, itf, 20)

print(f"Cache ready in {time.time()-t0:.1f}s", flush=True)

# bt_fast의 run_backtest를 활용하되, ADX/RSI를 매핑된 것으로 교체
def run_mapped(cross_tf, fast_type, fast_p, slow_type, slow_p,
               adx_tf, adx_min, rsi_tf, rsi_min, rsi_max,
               sl, trail_act, trail_pct, partial_pct=0, partial_at=0):
    """Numba _bt_core를 사용하되, ADX/RSI는 다른 TF에서 매핑"""
    base = cache.get_base(cross_tf)
    if base['n'] < 200: return None

    ma_fast = cache.get_ma(cross_tf, fast_type, fast_p)
    ma_slow = cache.get_ma(cross_tf, slow_type, slow_p)

    # ADX/RSI: 다른 TF에서 매핑
    if adx_tf == cross_tf:
        adx = cache.get_adx(cross_tf, 20)
    else:
        adx = cache.get_mapped_adx(cross_tf, adx_tf, 20)

    if rsi_tf == cross_tf:
        rsi = cache.get_rsi(cross_tf, 14)
    else:
        rsi = cache.get_mapped_rsi(cross_tf, rsi_tf, 14)

    atr = cache.get_atr(cross_tf, 14)
    times = cache.get_times(cross_tf)

    from bt_fast import _bt_core
    r = _bt_core(base['close'], base['high'], base['low'], times,
                 ma_fast, ma_slow, rsi, adx, atr,
                 adx_min, rsi_min, rsi_max,
                 sl, trail_act, trail_pct,
                 False, 2.0, 0.02, 0.12, False, 1.5,  # ATR options
                 10, 0.40, 0.20,  # leverage, margin
                 -0.20, 0, 0,  # monthly loss, consec pause
                 -0.20, 0.0004, 3000.0,  # dd thresh, fee, capital
                 False, 6, -0.001, -0.025)  # delayed entry

    (bal,tot,wc,lc,gp,gl,slc,tslc,revc,flc,mdd_v,yrk,yrs,yre,lm,wps,lps,mwp,mlp) = r
    if tot < 5: return None
    pf = gp/gl if gl > 0 else gp
    wr = wc/tot*100 if tot > 0 else 0
    yearly = {}
    for j in range(len(yrk)):
        if yrs[j] > 0: yearly[str(int(yrk[j]))] = round((yre[j]-yrs[j])/yrs[j]*100, 1)
    rec = [yearly.get(y,0) for y in ['2023','2024','2025','2026'] if y in yearly]

    return {'bal':round(bal),'pf':round(pf,2),'mdd':round(mdd_v*100,1),
            'tr':int(tot),'wr':round(wr,1),'sl':int(slc),'tsl':int(tslc),
            'rev':int(revc),'fl':int(flc),'yr':yearly,'lm':int(lm),
            'rec':round(np.mean(rec) if rec else 0, 1),
            'avg_win':round(wps/wc*100,2) if wc>0 else 0,
            'avg_loss':round(-lps/lc*100,2) if lc>0 else 0,
            'cfg':f'{cross_tf} {fast_type}({fast_p})/{slow_type}({slow_p}) A20@{adx_tf}>={adx_min:.0f} R14@{rsi_tf}:{rsi_min:.0f}-{rsi_max:.0f} SL{sl:.2f} TA{trail_act:.2f}/{trail_pct:.2f}'}

def score(r):
    pf=r['pf']; bal=r['bal']; mdd=r['mdd']; rec=r['rec']
    if bal<=3000 or pf<=0: return 0
    base=pf*np.log1p((bal-3000)/3000)
    mp=0.3 if mdd>70 else(0.5 if mdd>55 else(0.8 if mdd>40 else 1.0))
    rp=1.5 if rec>80 else(1.3 if rec>40 else(1.0 if rec>0 else 0.7))
    return base*mp*rp

# === PHASE 1: MA × TF × Filter screening ===
cross_tfs = ['5m','10m','15m','30m']
fast_combos_by_tf = {
    '5m':  [('ema',3),('ema',5),('ema',7)],
    '10m': [('ema',3),('ema',5),('ema',7),('hma',3),('hma',5),('hma',7)],
    '15m': [('ema',3),('ema',5),('ema',7),('hma',3),('hma',5),('hma',7)],
    '30m': [('ema',3),('ema',5),('ema',7),('hma',3),('hma',5),('hma',7)],
}
slow_combos = [('ema',100),('ema',150),('ema',200),('ema',250),('ema',300)]
adx_tfs_list = ['5m','10m','15m']
adx_mins_list = [25,30,35]
rsi_tfs_list = ['5m','10m','15m']
rsi_ranges_list = [(35,65),(40,60),(40,75),(30,70)]

p1_total = sum(len(fast_combos_by_tf[c]) for c in cross_tfs) * len(slow_combos) * len(adx_tfs_list) * len(adx_mins_list) * len(rsi_tfs_list) * len(rsi_ranges_list)
print(f"\n=== Phase 1: {p1_total:,} combos ===", flush=True)

p1_results = []
cnt = 0
t1 = time.time()

for ctf in cross_tfs:
    for ft, fp in fast_combos_by_tf[ctf]:
        for st, sp in slow_combos:
            for atf in adx_tfs_list:
                for amin in adx_mins_list:
                    for rtf in rsi_tfs_list:
                        for rmin, rmax in rsi_ranges_list:
                            cnt += 1
                            if cnt % 500 == 0:
                                el=time.time()-t1; rate=cnt/el if el>0 else 0
                                eta=(p1_total-cnt)/rate if rate>0 else 0
                                print(f"  P1: {cnt:,}/{p1_total:,} ({cnt/p1_total*100:.1f}%) {rate:.0f}/s ETA:{eta:.0f}s found:{len(p1_results)}", flush=True)
                            r = run_mapped(ctf, ft, fp, st, sp, atf, amin, rtf, rmin, rmax, 0.06, 0.07, 0.03)
                            if r and r['fl']==0 and r['bal']>5000:
                                p1_results.append(r)

print(f"\nPhase 1 done: {len(p1_results)} found in {time.time()-t1:.1f}s", flush=True)
p1_results.sort(key=score, reverse=True)
top200 = p1_results[:200]

print("\nTop 10 Phase 1:")
for i,r in enumerate(top200[:10]):
    print(f"  #{i+1} ${r['bal']:>11,} PF:{r['pf']:>6.2f} MDD:{r['mdd']:>5.1f}% TR:{r['tr']:>4} WR:{r['wr']:>4.1f}% | {r['cfg']}")

# === PHASE 2: SL × Trail × Partial on top 200 ===
sl_pcts = [0.03,0.04,0.05,0.06,0.07]
trail_acts = [0.04,0.05,0.06,0.07,0.08]
trail_pcts = [0.02,0.03,0.04]

p2_total = len(top200) * len(sl_pcts) * len(trail_acts) * len(trail_pcts)
print(f"\n=== Phase 2: {p2_total:,} combos on top {len(top200)} ===", flush=True)

p2_results = []
cnt2 = 0
t2 = time.time()

for r in top200:
    c = r['cfg']; ps = c.split()
    ctf = ps[0]
    mi = ps[1].split('/'); ft=mi[0].split('(')[0]; fp=int(mi[0].split('(')[1].rstrip(')'))
    st=mi[1].split('(')[0]; sp=int(mi[1].split('(')[1].rstrip(')'))
    ap=ps[2]; atf=ap.split('@')[1].split('>=')[0]; amin=int(ap.split('>=')[1])
    rp=ps[3]; rtf=rp.split('@')[1].split(':')[0]
    rng=rp.split(':')[1]; rmin=int(rng.split('-')[0]); rmax=int(rng.split('-')[1])

    for sl in sl_pcts:
        for ta in trail_acts:
            for tp in trail_pcts:
                if tp >= ta: continue
                cnt2 += 1
                if cnt2 % 3000 == 0:
                    el=time.time()-t2; rate=cnt2/el if el>0 else 0
                    eta=(p2_total-cnt2)/rate if rate>0 else 0
                    print(f"  P2: {cnt2:,}/{p2_total:,} ({cnt2/p2_total*100:.1f}%) {rate:.0f}/s ETA:{eta:.0f}s found:{len(p2_results)}", flush=True)
                r2 = run_mapped(ctf, ft, fp, st, sp, atf, amin, rtf, rmin, rmax, sl, ta, tp)
                if r2 and r2['fl']==0 and r2['bal']>10000:
                    p2_results.append(r2)

print(f"\nPhase 2 done: {len(p2_results)} found in {time.time()-t2:.1f}s", flush=True)
p2_results.sort(key=score, reverse=True)

# Results
print(f"\n{'='*120}")
print(f"TOP 30 v25.2 FINAL RESULTS")
print(f"{'='*120}")
for i,r in enumerate(p2_results[:30]):
    print(f"#{i+1:>2} ${r['bal']:>12,} PF:{r['pf']:>7.2f} MDD:{r['mdd']:>5.1f}% TR:{r['tr']:>4} WR:{r['wr']:>4.1f}% SL:{r['sl']} TSL:{r['tsl']} REV:{r['rev']} FL:{r['fl']} avgW:{r['avg_win']:.1f}% avgL:{r['avg_loss']:.1f}%")
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
