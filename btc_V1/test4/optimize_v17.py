"""v17.0 BTC 최적화 - 거래수 활성화 + 고PF + 저MDD
핵심 변경: ADX 20~35, 짧은 MA, 짧은 TF로 거래빈도 대폭 증가
목표: 월 2~5회 거래, PF>=5, MDD<=40%
"""
import itertools, json, time, sys
from bt_fast import load_5m_data, build_mtf, IndicatorCache, run_backtest
import numpy as np

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)
cache = IndicatorCache(mtf)

# === Phase 1: 거래수 활성화 탐색 (핵심 변경) ===
# 기존 v16.5: ADX>=35, slow=250 → 45거래
# v17: ADX 20~30, slow 50~150, fast 3~21 → 100~500+ 거래 기대

tfs = ['5m','10m','15m','30m']
ma_types = ['ema','hma','wma','dema','sma']
fast_ps = [3,5,7,10,14,21]
slow_ps = [30,50,75,100,150,200]
adx_ps = [14,20]
adx_mins = [20,25,30]
rsi_ps = [14]
rsi_ranges = [(30,70),(35,65),(35,75),(40,60),(25,75)]
sl_pcts = [0.03,0.04,0.05,0.06,0.07]
trail_combos = [(0.04,0.02),(0.05,0.03),(0.06,0.03),(0.07,0.04),(0.08,0.05),(0.10,0.05)]

total = len(tfs)*len(ma_types)*len(fast_ps)*len(slow_ps)*len(adx_ps)*len(adx_mins)*len(rsi_ranges)*len(sl_pcts)*len(trail_combos)
print(f"Phase 1: {total:,} combinations")

results = []
cnt = 0
t0 = time.time()

for tf in tfs:
    for mat in ma_types:
        for fp in fast_ps:
            for sp in slow_ps:
                if fp >= sp: continue
                for ap in adx_ps:
                    for am in adx_mins:
                        for rmin,rmax in rsi_ranges:
                            for slp in sl_pcts:
                                for ta,tp in trail_combos:
                                    if ta <= slp: continue  # trail must activate above SL
                                    cnt += 1
                                    if cnt % 10000 == 0:
                                        el = time.time()-t0
                                        rate = cnt/el if el>0 else 0
                                        eta = (total-cnt)/rate/60 if rate>0 else 0
                                        print(f"  {cnt:,}/{total:,} ({cnt/total*100:.1f}%) {rate:.0f}/s ETA:{eta:.0f}m | top={len(results)}", flush=True)

                                    cfg = {
                                        'timeframe':tf,'ma_fast_type':mat,'ma_slow_type':'ema',
                                        'ma_fast':fp,'ma_slow':sp,
                                        'adx_period':ap,'adx_min':am,
                                        'rsi_period':14,'rsi_min':rmin,'rsi_max':rmax,
                                        'sl_pct':slp,'trail_activate':ta,'trail_pct':tp,
                                        'leverage':10,'margin_normal':0.35,'margin_reduced':0.15,
                                        'dd_threshold':-0.25,'monthly_loss_limit':-0.15,
                                        'fee_rate':0.0004,'initial_capital':3000.0
                                    }
                                    r = run_backtest(cache, tf, cfg)
                                    if r is None: continue
                                    tr = r['trades']
                                    pf = r['pf']
                                    mdd = r['mdd']
                                    fl = r['liq']

                                    # v17 핵심 필터: 거래수 50+, PF>=2, MDD<=60%, FL=0
                                    if tr >= 50 and pf >= 2.0 and mdd <= 60 and fl == 0:
                                        # 복리 점수: 거래수 x PF / MDD
                                        compound_score = tr * pf / max(mdd, 1)
                                        results.append({**r, 'compound_score': compound_score})

print(f"\nPhase 1 done: {cnt:,} tested, {len(results)} passed filter")

# Sort by compound score (거래수 x PF / MDD)
results.sort(key=lambda x: x['compound_score'], reverse=True)

# Print top 30
print(f"\n{'='*120}")
print(f"TOP 30 by Compound Score (trades x PF / MDD)")
print(f"{'='*120}")
for i, r in enumerate(results[:30]):
    yr = r.get('yr',{})
    rec = ' '.join(f"{k}:{v:+.0f}%" for k,v in sorted(yr.items()))
    print(f"#{i+1:>2} ${r['bal']:>10,.0f} ({r['ret']:>+8.1f}%) PF:{r['pf']:>5.2f} MDD:{r['mdd']:>5.1f}% TR:{r['trades']:>3} "
          f"SL:{r['sl']:>2} WR:{r['wr']:>4.1f}% CS:{r['compound_score']:>6.1f}")
    print(f"     {r['cfg']}")
    print(f"     {rec}")

# === Phase 2: 상위 전략 마진/보호 변형 ===
print(f"\n{'='*120}")
print(f"Phase 2: Top 10 margin optimization")
print(f"{'='*120}")

margin_opts = [
    (0.20, 0.10, -0.20, -0.15, 'M20_safe'),
    (0.25, 0.12, -0.20, -0.15, 'M25_mod'),
    (0.30, 0.15, -0.25, -0.15, 'M30_bal'),
    (0.35, 0.15, -0.25, -0.15, 'M35_agg'),
    (0.40, 0.20, -0.25, -0.20, 'M40_max'),
]

phase2 = []
for r in results[:10]:
    cfg_str = r['cfg']
    # Parse back config from cfg string (need to rebuild)
    for mn, mr, dd, ml, label in margin_opts:
        # Rebuild cfg from the result
        cfg = {
            'timeframe': cfg_str.split(' |')[0].strip(),
            'margin_normal': mn, 'margin_reduced': mr,
            'dd_threshold': dd, 'monthly_loss_limit': ml,
            'leverage': 10, 'fee_rate': 0.0004, 'initial_capital': 3000.0
        }
        # Extract MA info from cfg_str
        # Just re-run with same params but different margin
        # Need to parse... easier to store raw cfg
        pass

# Actually, let me store raw configs
results2 = []
cnt2 = 0
for i, r in enumerate(results[:20]):
    # re-parse cfg_str to extract params
    cs = r['cfg']
    parts = cs.split(' | ')
    tf = parts[0].strip()

    # Parse MA: e.g., "hma(21/250)"
    ma_part = parts[1].strip()
    ma_type = ma_part.split('(')[0]
    ma_nums = ma_part.split('(')[1].rstrip(')').split('/')
    fp = int(ma_nums[0])
    sp = int(ma_nums[1])

    # Parse ADX
    adx_part = parts[2].strip()  # e.g., "A20>=25"
    ap = int(adx_part[1:adx_part.index('>')])
    am = int(adx_part.split('>=')[1])

    # Parse RSI
    rsi_part = parts[3].strip()  # e.g., "R14:35-75"
    rp = int(rsi_part[1:rsi_part.index(':')])
    rsi_range = rsi_part.split(':')[1].split('-')
    rmin = int(rsi_range[0])
    rmax = int(rsi_range[1])

    # Parse SL
    sl_part = parts[4].strip()  # e.g., "SL0.05"
    slp = float(sl_part[2:])

    # Parse Trail
    trail_part = parts[5].strip()  # e.g., "TA0.06/0.03"
    ta_tp = trail_part[2:].split('/')
    ta = float(ta_tp[0])
    tp = float(ta_tp[1])

    for mn, mr, dd, ml, label in margin_opts:
        cfg = {
            'timeframe':tf,'ma_fast_type':ma_type,'ma_slow_type':'ema',
            'ma_fast':fp,'ma_slow':sp,
            'adx_period':ap,'adx_min':am,
            'rsi_period':rp,'rsi_min':rmin,'rsi_max':rmax,
            'sl_pct':slp,'trail_activate':ta,'trail_pct':tp,
            'leverage':10,'margin_normal':mn,'margin_reduced':mr,
            'dd_threshold':dd,'monthly_loss_limit':ml,
            'fee_rate':0.0004,'initial_capital':3000.0
        }
        r2 = run_backtest(cache, tf, cfg)
        if r2 and r2['liq']==0 and r2['trades']>=50:
            cs2 = r2['trades'] * r2['pf'] / max(r2['mdd'],1)
            results2.append({**r2, 'compound_score':cs2, 'margin_label':label})
            cnt2 += 1

results2.sort(key=lambda x: x['compound_score'], reverse=True)
print(f"Phase 2: {cnt2} variants tested")
for i, r in enumerate(results2[:20]):
    yr = r.get('yr',{})
    rec = ' '.join(f"{k}:{v:+.0f}%" for k,v in sorted(yr.items()))
    print(f"#{i+1:>2} ${r['bal']:>10,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>5.1f}% TR:{r['trades']:>3} "
          f"WR:{r['wr']:>4.1f}% CS:{r['compound_score']:>6.1f} [{r['margin_label']}]")
    print(f"     {r['cfg']}")
    print(f"     {rec}")

# === Phase 3: 30회 검증 ===
print(f"\n{'='*120}")
print(f"Phase 3: 30-run verification on top 5")
print(f"{'='*120}")

final_top = results2[:5] if results2 else results[:5]
verified = []
for i, r in enumerate(final_top):
    cs = r['cfg']
    parts = cs.split(' | ')
    tf = parts[0].strip()

    ma_part = parts[1].strip()
    ma_type = ma_part.split('(')[0]
    ma_nums = ma_part.split('(')[1].rstrip(')').split('/')
    fp = int(ma_nums[0])
    sp = int(ma_nums[1])

    adx_part = parts[2].strip()
    ap = int(adx_part[1:adx_part.index('>')])
    am = int(adx_part.split('>=')[1])

    rsi_part = parts[3].strip()
    rp = int(rsi_part[1:rsi_part.index(':')])
    rsi_range = rsi_part.split(':')[1].split('-')
    rmin = int(rsi_range[0])
    rmax = int(rsi_range[1])

    sl_part = parts[4].strip()
    slp = float(sl_part[2:])

    trail_part = parts[5].strip()
    ta_tp = trail_part[2:].split('/')
    ta = float(ta_tp[0])
    tp = float(ta_tp[1])

    # Get margin from cfg string
    mg_part = [p for p in parts if 'M' in p and '%' in p]
    mn = r.get('margin_normal', 0.35) if 'margin_label' not in r else float(r['margin_label'].split('M')[1].split('_')[0])/100

    cfg = {
        'timeframe':tf,'ma_fast_type':ma_type,'ma_slow_type':'ema',
        'ma_fast':fp,'ma_slow':sp,
        'adx_period':ap,'adx_min':am,
        'rsi_period':rp,'rsi_min':rmin,'rsi_max':rmax,
        'sl_pct':slp,'trail_activate':ta,'trail_pct':tp,
        'leverage':10,'margin_normal':mn,'margin_reduced':mn*0.5,
        'dd_threshold':-0.25,'monthly_loss_limit':-0.15,
        'fee_rate':0.0004,'initial_capital':3000.0
    }

    bals = []
    for run in range(30):
        r2 = run_backtest(cache, tf, cfg)
        if r2: bals.append(r2['bal'])

    avg = np.mean(bals) if bals else 0
    std = np.std(bals) if bals else 0
    status = "PASS" if std < 1 else "FAIL"
    print(f"  #{i+1} avg=${avg:,.0f} std=${std:.4f} {status} | {r['cfg'][:80]}")
    verified.append({**r, 'verify_avg':avg, 'verify_std':std, 'verify_status':status})

# Save
save_data = {
    'phase1_count': len(results),
    'phase1_total_tested': cnt,
    'phase2_count': len(results2),
    'top30_phase1': [{k:v for k,v in r.items() if k not in ['trades_detail']} for r in results[:30]],
    'top20_phase2': [{k:v for k,v in r.items() if k not in ['trades_detail']} for r in results2[:20]],
    'verified': [{k:v for k,v in r.items() if k not in ['trades_detail']} for r in verified],
}
with open('v17_results.json','w',encoding='utf-8') as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False)

print(f"\nSaved: v17_results.json")
print(f"Total time: {(time.time()-t0)/60:.1f} min")
