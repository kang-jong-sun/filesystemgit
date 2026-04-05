"""v17.0 BTC 빠른 최적화 - 거래수 활성화 집중
Phase 1: ADX 20-30 (거래수 핵심), 짧은 MA → 빠른 스캔
Phase 2: 상위 전략 마진/보호 세분화
Phase 3: 30회 검증
"""
import json, time, numpy as np
from bt_fast import load_5m_data, build_mtf, IndicatorCache, run_backtest

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)
cache = IndicatorCache(mtf)

# === Phase 1: 핵심 조합만 스캔 ===
combos = []
for tf in ['5m','10m','15m','30m']:
    for mat in ['ema','hma','wma','dema','sma']:
        for fp in [3,5,7,10,14,21]:
            for sp in [30,50,75,100,150,200,250]:
                if fp >= sp: continue
                for ap in [14,20]:
                    for am in [20,25,30,35]:
                        for rmin,rmax in [(30,70),(35,65),(35,75),(40,60),(25,75)]:
                            for slp in [0.03,0.04,0.05,0.06,0.07]:
                                for ta,tp in [(0.04,0.02),(0.05,0.03),(0.06,0.03),(0.07,0.04),(0.08,0.05),(0.10,0.05)]:
                                    if ta <= slp: continue
                                    combos.append((tf,mat,fp,sp,ap,am,rmin,rmax,slp,ta,tp))

print(f"Total combos: {len(combos):,}")

results = []
t0 = time.time()

for idx, (tf,mat,fp,sp,ap,am,rmin,rmax,slp,ta,tp) in enumerate(combos):
    if idx % 20000 == 0 and idx > 0:
        el = time.time()-t0
        rate = idx/el if el > 0 else 1
        eta = (len(combos)-idx)/rate/60
        print(f"  {idx:,}/{len(combos):,} ({idx/len(combos)*100:.1f}%) {rate:.0f}/s ETA:{eta:.0f}m | found={len(results)}", flush=True)

    cfg = {
        'timeframe':tf,'ma_fast_type':mat,'ma_slow_type':'ema',
        'ma_fast':fp,'ma_slow':sp,'adx_period':ap,'adx_min':am,
        'rsi_period':14,'rsi_min':rmin,'rsi_max':rmax,
        'sl_pct':slp,'trail_activate':ta,'trail_pct':tp,
        'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,
        'dd_threshold':-0.25,'monthly_loss_limit':-0.15,
        'fee_rate':0.0004,'initial_capital':3000.0
    }
    r = run_backtest(cache, tf, cfg)
    if r is None: continue
    tr = r['trades']; pf = r['pf']; mdd = r['mdd']; fl = r['liq']

    # 거래수 50+, PF>=2, FL=0
    if tr >= 50 and pf >= 2.0 and fl == 0 and r['ret'] > 0:
        cs = tr * pf / max(mdd, 1) * np.log1p(r['ret']/100)
        results.append({**r, 'cs': round(cs, 2)})

print(f"\nPhase 1: {len(combos):,} tested, {len(results)} passed")

# === Sort and display ===
results.sort(key=lambda x: x['cs'], reverse=True)

print(f"\n{'='*130}")
print(f"TOP 30 (거래수 x PF / MDD x log(수익))")
print(f"{'='*130}")
for i, r in enumerate(results[:30]):
    yr = r.get('yr',{})
    loss_yrs = sum(1 for v in yr.values() if v < 0)
    rec = ' '.join(f"{k}:{v:+.0f}%" for k,v in sorted(yr.items()))
    print(f"#{i+1:>2} ${r['bal']:>10,.0f} ({r['ret']:>+8.1f}%) PF:{r['pf']:>5.2f} MDD:{r['mdd']:>5.1f}% TR:{r['trades']:>3} "
          f"SL:{r['sl']:>2} WR:{r['wr']:>4.1f}% LY:{loss_yrs} CS:{r['cs']:>6.1f}")
    print(f"     {r['cfg']}")
    print(f"     {rec}")

# === Phase 2: 상위 20개 마진 변형 ===
print(f"\n{'='*130}")
print(f"Phase 2: Top 20 margin/protection variants")
print(f"{'='*130}")

margin_sets = [
    {'margin_normal':0.20,'margin_reduced':0.10,'dd_threshold':-0.20,'monthly_loss_limit':-0.10, '_label':'M20safe'},
    {'margin_normal':0.25,'margin_reduced':0.12,'dd_threshold':-0.20,'monthly_loss_limit':-0.15, '_label':'M25mod'},
    {'margin_normal':0.30,'margin_reduced':0.15,'dd_threshold':-0.25,'monthly_loss_limit':-0.15, '_label':'M30bal'},
    {'margin_normal':0.35,'margin_reduced':0.17,'dd_threshold':-0.25,'monthly_loss_limit':-0.20, '_label':'M35agg'},
    {'margin_normal':0.40,'margin_reduced':0.20,'dd_threshold':-0.25,'monthly_loss_limit':-0.20, '_label':'M40max'},
    {'margin_normal':0.50,'margin_reduced':0.25,'dd_threshold':-0.25,'monthly_loss_limit':-0.20, '_label':'M50ult'},
]

phase2 = []
for r in results[:20]:
    cs = r['cfg']
    parts = cs.split(' | ')
    tf = parts[0].strip()
    ma_part = parts[1].strip()
    ma_type = ma_part.split('(')[0]
    ma_nums = ma_part.split('(')[1].rstrip(')').split('/')
    fp = int(ma_nums[0]); sp = int(ma_nums[1])
    adx_part = parts[2].strip()
    ap = int(adx_part[1:adx_part.index('>')])
    am = int(adx_part.split('>=')[1])
    rsi_part = parts[3].strip()
    rsi_range = rsi_part.split(':')[1].split('-')
    rmin = int(rsi_range[0]); rmax = int(rsi_range[1])
    sl_part = parts[4].strip()
    slp = float(sl_part[2:])
    trail_part = parts[5].strip()
    ta_tp = trail_part[2:].split('/')
    ta = float(ta_tp[0]); tp = float(ta_tp[1])

    for ms in margin_sets:
        cfg = {
            'timeframe':tf,'ma_fast_type':ma_type,'ma_slow_type':'ema',
            'ma_fast':fp,'ma_slow':sp,'adx_period':ap,'adx_min':am,
            'rsi_period':14,'rsi_min':rmin,'rsi_max':rmax,
            'sl_pct':slp,'trail_activate':ta,'trail_pct':tp,
            'leverage':10,'fee_rate':0.0004,'initial_capital':3000.0,
            **{k:v for k,v in ms.items() if k != '_label'}
        }
        r2 = run_backtest(cache, tf, cfg)
        if r2 and r2['liq']==0 and r2['trades']>=30:
            cs2 = r2['trades'] * r2['pf'] / max(r2['mdd'],1) * np.log1p(r2['ret']/100)
            phase2.append({**r2, 'cs':round(cs2,2), 'mlabel':ms['_label']})

phase2.sort(key=lambda x: x['cs'], reverse=True)
print(f"Phase 2: {len(phase2)} variants")
for i, r in enumerate(phase2[:30]):
    yr = r.get('yr',{})
    rec = ' '.join(f"{k}:{v:+.0f}%" for k,v in sorted(yr.items()))
    print(f"#{i+1:>2} ${r['bal']:>10,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>5.1f}% TR:{r['trades']:>3} "
          f"WR:{r['wr']:>4.1f}% [{r['mlabel']}] CS:{r['cs']:>6.1f}")
    print(f"     {r['cfg']}")
    print(f"     {rec}")

# === Phase 3: 30회 검증 ===
print(f"\n{'='*130}")
print(f"Phase 3: 30-run verification top 10")
print(f"{'='*130}")

final = phase2[:10] if phase2 else results[:10]
for i, r in enumerate(final):
    cs = r['cfg']
    parts = cs.split(' | ')
    tf = parts[0].strip()
    ma_part = parts[1].strip()
    ma_type = ma_part.split('(')[0]
    ma_nums = ma_part.split('(')[1].rstrip(')').split('/')
    fp = int(ma_nums[0]); sp = int(ma_nums[1])
    adx_part = parts[2].strip()
    ap = int(adx_part[1:adx_part.index('>')])
    am = int(adx_part.split('>=')[1])
    rsi_part = parts[3].strip()
    rsi_range = rsi_part.split(':')[1].split('-')
    rmin = int(rsi_range[0]); rmax = int(rsi_range[1])
    sl_part = parts[4].strip()
    slp = float(sl_part[2:])
    trail_part = parts[5].strip()
    ta_tp = trail_part[2:].split('/')
    ta = float(ta_tp[0]); tp = float(ta_tp[1])

    # margin from cfg string
    mg_parts = [p.strip() for p in parts if '%' in p and 'M' in p]
    mn = float(r.get('mlabel','M30bal').replace('M','').replace('safe','').replace('mod','').replace('bal','').replace('agg','').replace('max','').replace('ult',''))/100 if 'mlabel' in r else 0.30

    cfg = {
        'timeframe':tf,'ma_fast_type':ma_type,'ma_slow_type':'ema',
        'ma_fast':fp,'ma_slow':sp,'adx_period':ap,'adx_min':am,
        'rsi_period':14,'rsi_min':rmin,'rsi_max':rmax,
        'sl_pct':slp,'trail_activate':ta,'trail_pct':tp,
        'leverage':10,'margin_normal':mn,'margin_reduced':mn*0.5,
        'dd_threshold':-0.25,'monthly_loss_limit':-0.15,
        'fee_rate':0.0004,'initial_capital':3000.0
    }
    bals = [run_backtest(cache,tf,cfg)['bal'] for _ in range(30)]
    avg = np.mean(bals); std = np.std(bals)
    status = "PASS" if std < 1 else "FAIL"
    print(f"#{i+1} ${avg:,.0f} std={std:.4f} {status} TR:{r['trades']} PF:{r['pf']:.2f} MDD:{r['mdd']:.1f}% [{r.get('mlabel','')}]")

# Save
with open('v17_results.json','w',encoding='utf-8') as f:
    json.dump({
        'total_combos': len(combos),
        'phase1_passed': len(results),
        'phase2_passed': len(phase2),
        'top30': [{k:v for k,v in r.items()} for r in results[:30]],
        'phase2_top30': [{k:v for k,v in r.items()} for r in phase2[:30]],
    }, f, indent=2, ensure_ascii=False)

print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
print("Saved: v17_results.json")
