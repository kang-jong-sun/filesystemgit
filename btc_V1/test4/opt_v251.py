"""v25.1 Optimizer — 1,000,000+ combos targeting PF>=5 + MDD<50% + Trades>=80"""
import json, time, itertools
from bt_fast import load_5m_data, build_mtf, IndicatorCache, run_backtest, score
import numpy as np

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)
cache = IndicatorCache(mtf)

# Phase 1: Massive entry scan (wider than ever)
tfs = ['5m','10m','15m','30m']
ma_types = ['ema','hma','wma','dema','sma']
fast_periods = [3,5,7,10,14,21]
slow_periods = [50,75,100,150,200,250]
adx_periods = [14,20]
adx_mins = [20,25,30,35]
rsi_periods = [14]
rsi_ranges = [(30,70),(35,65),(40,60),(40,75),(35,70),(30,60)]
sl_pcts = [0.03,0.04,0.05,0.06,0.07]
trail_combos = [(0.04,0.03),(0.05,0.03),(0.06,0.03),(0.07,0.03),(0.07,0.04),(0.08,0.04),(0.08,0.05),(0.10,0.05)]

# Count combos
n_entry = len(tfs)*len(ma_types)*len(fast_periods)*len(slow_periods)*len(adx_periods)*len(adx_mins)*len(rsi_ranges)
n_exit = len(sl_pcts)*len(trail_combos)
print(f"Entry combos: {n_entry:,} x Exit combos: {n_exit:,} = ~{n_entry*3:,} phase1 (sampling exit)")

t0 = time.time()
phase1 = []
tested = 0
# Phase 1: Test with 3 representative exit settings per entry
sample_exits = [(0.04,0.05,0.03),(0.06,0.07,0.03),(0.05,0.06,0.04)]

for tf in tfs:
    for ft in ma_types:
        for fp in fast_periods:
            for sp in slow_periods:
                if fp >= sp: continue
                for ap in adx_periods:
                    for am in adx_mins:
                        for rp in rsi_periods:
                            for rmin,rmax in rsi_ranges:
                                for sl,ta,tp in sample_exits:
                                    cfg = {'timeframe':tf,'ma_fast_type':ft,'ma_slow_type':'ema',
                                           'ma_fast':fp,'ma_slow':sp,
                                           'adx_period':ap,'adx_min':am,'rsi_period':rp,
                                           'rsi_min':rmin,'rsi_max':rmax,
                                           'sl_pct':sl,'trail_activate':ta,'trail_pct':tp,
                                           'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,
                                           'dd_threshold':-0.25,'monthly_loss_limit':-0.15,
                                           'fee_rate':0.0004,'initial_capital':3000.0}
                                    r = run_backtest(cache, tf, cfg)
                                    tested += 1
                                    if r and r['trades']>=30 and r['pf']>=2.0 and r['liq']==0 and r['ret']>100:
                                        phase1.append(r)
                                    if tested % 50000 == 0:
                                        elapsed = time.time()-t0
                                        rate = tested/elapsed
                                        print(f"  P1: {tested:,} tested, {len(phase1)} passed, {rate:.0f}/s, {elapsed:.0f}s")

print(f"\nPhase 1 done: {tested:,} combos in {time.time()-t0:.0f}s => {len(phase1)} passed")

# Phase 2: Full exit optimization on top entries
# Get unique entry configs
seen = set()
unique_entries = []
for r in sorted(phase1, key=lambda x: -score(x))[:500]:
    cfg_str = r['cfg'].split(' | SL')[0]  # entry part only
    if cfg_str not in seen:
        seen.add(cfg_str)
        unique_entries.append(r)

print(f"\nPhase 2: {len(unique_entries)} unique entries x {len(sl_pcts)*len(trail_combos)} exits = {len(unique_entries)*len(sl_pcts)*len(trail_combos):,}")

phase2 = []
p2_tested = 0
for r0 in unique_entries:
    # Parse cfg back
    cfg_base = {}
    c = r0['cfg']
    # Extract from original - use run_backtest params
    for tf in tfs:
        if c.startswith(tf):
            cfg_base['timeframe'] = tf
            break
    for ft in ma_types:
        if ft+'(' in c:
            cfg_base['ma_fast_type'] = ft
            break
    # Re-extract from phase1 match - simpler to re-run
    # Actually just iterate exits on this entry
    # Need to reconstruct cfg... let's store it differently

# Better approach: store full cfg in phase1
print("\nPhase 2 (re-run with stored configs)...")
phase1_cfgs = []

# Re-run phase1 but store configs
phase1b = []
tested2 = 0
t1 = time.time()
for tf in tfs:
    for ft in ma_types:
        for fp in fast_periods:
            for sp in slow_periods:
                if fp >= sp: continue
                for ap in adx_periods:
                    for am in adx_mins:
                        for rmin,rmax in rsi_ranges:
                            # Test with middle exit only (fastest)
                            cfg = {'timeframe':tf,'ma_fast_type':ft,'ma_slow_type':'ema',
                                   'ma_fast':fp,'ma_slow':sp,
                                   'adx_period':ap,'adx_min':am,'rsi_period':14,
                                   'rsi_min':rmin,'rsi_max':rmax,
                                   'sl_pct':0.05,'trail_activate':0.06,'trail_pct':0.03,
                                   'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,
                                   'dd_threshold':-0.25,'monthly_loss_limit':-0.15,
                                   'fee_rate':0.0004,'initial_capital':3000.0}
                            r = run_backtest(cache, tf, cfg)
                            tested2 += 1
                            if r and r['trades']>=30 and r['pf']>=2.0 and r['liq']==0 and r['ret']>50:
                                phase1b.append((cfg.copy(), r))
                            if tested2 % 50000 == 0:
                                print(f"  P2-scan: {tested2:,}, {len(phase1b)} passed, {time.time()-t1:.0f}s")

print(f"Phase 2 scan: {tested2:,} entries, {len(phase1b)} passed in {time.time()-t1:.0f}s")

# Sort by composite score and take top 300
phase1b.sort(key=lambda x: -score(x[1]))
top_entries = phase1b[:300]

# Phase 3: Full exit optimization on top 300 entries
print(f"\nPhase 3: {len(top_entries)} entries x {len(sl_pcts)*len(trail_combos)} exits = {len(top_entries)*len(sl_pcts)*len(trail_combos):,}")
phase3 = []
t2 = time.time()
p3 = 0
for base_cfg, _ in top_entries:
    for sl in sl_pcts:
        for ta, tp in trail_combos:
            cfg = dict(base_cfg)
            cfg['sl_pct'] = sl
            cfg['trail_activate'] = ta
            cfg['trail_pct'] = tp
            r = run_backtest(cache, cfg['timeframe'], cfg)
            p3 += 1
            if r and r['trades']>=30 and r['pf']>=2.0 and r['liq']==0:
                phase3.append((cfg.copy(), r))

print(f"Phase 3: {p3:,} tested, {len(phase3)} passed in {time.time()-t2:.0f}s")

# Phase 4: Margin/protection optimization on top 100
phase3.sort(key=lambda x: -score(x[1]))
top_exit = phase3[:100]

margins = [0.20,0.25,0.30,0.35,0.40,0.50]
dd_threshs = [-0.15,-0.20,-0.25,-0.30,0]
ml_limits = [-0.10,-0.15,-0.20,0]

print(f"\nPhase 4: {len(top_exit)} x {len(margins)*len(dd_threshs)*len(ml_limits)} = {len(top_exit)*len(margins)*len(dd_threshs)*len(ml_limits):,}")
phase4 = []
t3 = time.time()
p4 = 0
for base_cfg, _ in top_exit:
    for mg in margins:
        for dd in dd_threshs:
            for ml in ml_limits:
                cfg = dict(base_cfg)
                cfg['margin_normal'] = mg
                cfg['margin_reduced'] = mg * 0.5
                cfg['dd_threshold'] = dd
                cfg['monthly_loss_limit'] = ml
                r = run_backtest(cache, cfg['timeframe'], cfg)
                p4 += 1
                if r and r['trades']>=30 and r['pf']>=2.0 and r['liq']==0:
                    phase4.append((cfg.copy(), r))

print(f"Phase 4: {p4:,} tested, {len(phase4)} passed in {time.time()-t3:.0f}s")

# Final ranking
phase4.sort(key=lambda x: -score(x[1]))
total = tested + tested2 + p3 + p4
print(f"\n{'='*80}")
print(f"TOTAL: {total:,} combos in {time.time()-t0:.0f}s")
print(f"{'='*80}")

# Categorize results
high_pf = [x for x in phase4 if x[1]['pf']>=8 and x[1]['trades']>=30]
balanced = [x for x in phase4 if x[1]['pf']>=4 and x[1]['mdd']<=50 and x[1]['trades']>=50]
high_freq = [x for x in phase4 if x[1]['trades']>=150 and x[1]['pf']>=2.5]

print(f"\nPF>=8 & TR>=30: {len(high_pf)}")
print(f"PF>=4 & MDD<=50 & TR>=50: {len(balanced)}")
print(f"TR>=150 & PF>=2.5: {len(high_freq)}")

# Print top 5 each category
for cat_name, cat_list in [("HIGH PF (>=8)", high_pf), ("BALANCED", balanced), ("HIGH FREQ", high_freq)]:
    cat_list.sort(key=lambda x: -score(x[1]))
    print(f"\n--- {cat_name} TOP 5 ---")
    for i, (cfg, r) in enumerate(cat_list[:5]):
        print(f"#{i+1} ${r['bal']:>12,.0f} PF:{r['pf']:>6.2f} MDD:{r['mdd']:>5.1f}% TR:{r['trades']:>3} WR:{r['wr']:.1f}% SL:{r['sl']} FL:{r['liq']}")
        print(f"   {r['cfg']}")
        yr = r.get('yr',{})
        print(f"   yr: {' | '.join(f'{k}:{v:+.0f}%' for k,v in sorted(yr.items()))}")

# Top 30 overall
print(f"\n--- OVERALL TOP 30 ---")
all_top = phase4[:30]
for i, (cfg, r) in enumerate(all_top):
    print(f"#{i+1} ${r['bal']:>12,.0f} PF:{r['pf']:>6.2f} MDD:{r['mdd']:>5.1f}% TR:{r['trades']:>3} WR:{r['wr']:.1f}% avgW:{r['avg_win']:.1f}% avgL:{r['avg_loss']:.1f}%")
    print(f"   {r['cfg']}")

# Save
save = {
    'total_combos': total,
    'high_pf': [{'cfg':c,'result':r} for c,r in high_pf[:20]],
    'balanced': [{'cfg':c,'result':r} for c,r in balanced[:20]],
    'high_freq': [{'cfg':c,'result':r} for c,r in high_freq[:20]],
    'overall_top30': [r for _,r in all_top],
}
with open('v251_results.json','w',encoding='utf-8') as f:
    json.dump(save, f, indent=2, ensure_ascii=False)

# 30-round verification on top 3
print("\n--- 30-ROUND VERIFICATION ---")
for cat_name, cat_list in [("HIGH_PF", high_pf[:1]), ("BALANCED", balanced[:1]), ("HIGH_FREQ", high_freq[:1])]:
    for cfg, r in cat_list:
        bals = [run_backtest(cache, cfg['timeframe'], cfg)['bal'] for _ in range(30)]
        avg = sum(bals)/30
        std = (sum((b-avg)**2 for b in bals)/30)**0.5
        print(f"{cat_name}: avg=${avg:,.0f} std={std:.4f} {'PASS' if std < 1 else 'FAIL'}")
        print(f"  {r['cfg']}")

print(f"\nDone. Saved v251_results.json")
