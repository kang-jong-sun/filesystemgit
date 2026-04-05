"""전체 기획서 일괄 백테스트 검증"""
import json, math
from bt_fast import load_5m_data, build_mtf, IndicatorCache, run_backtest

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)
cache = IndicatorCache(mtf)

base = {'fee_rate':0.0004, 'initial_capital':3000.0, 'leverage':10}

strategies = {
    'v12.3': {**base, 'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
              'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
              'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.06,
              'margin_normal':0.20,'margin_reduced':0.10,'dd_threshold':-0.50,'monthly_loss_limit':-0.20},
    'v13.5': {**base, 'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
              'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
              'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.06,
              'margin_normal':0.20,'margin_reduced':0.10,'dd_threshold':-0.50,'monthly_loss_limit':-0.20,
              'consec_loss_pause':3,'pause_candles':36},
    'v14.2F': {**base, 'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
               'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,
               'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
               'margin_normal':0.25,'margin_reduced':0.10,'dd_threshold':-0.50,'monthly_loss_limit':-0.20},
    'v14.4': {**base, 'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
              'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,
              'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
              'margin_normal':0.25,'margin_reduced':0.10,'dd_threshold':-0.50,'monthly_loss_limit':-0.20},
    'v15.2': {**base, 'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
              'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,
              'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
              'margin_normal':0.30,'margin_reduced':0.15,'dd_threshold':-0.50,'monthly_loss_limit':-0.20},
    'v15.4': {**base, 'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
              'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,
              'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
              'margin_normal':0.40,'margin_reduced':0.20,'dd_threshold':-0.30,'monthly_loss_limit':-0.30},
    'v15.5': {**base, 'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
              'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':35,'rsi_max':65,
              'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
              'margin_normal':0.35,'margin_reduced':0.15,'dd_threshold':-0.25,'monthly_loss_limit':-0.20},
    'v16.0': {**base, 'timeframe':'30m','ma_fast_type':'wma','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
              'adx_period':20,'adx_min':35,'rsi_period':14,'rsi_min':35,'rsi_max':65,
              'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.05,
              'margin_normal':0.50,'margin_reduced':0.25,'dd_threshold':-0.30,'monthly_loss_limit':-0.25},
    'v16.5A': {**base, 'timeframe':'10m','ma_fast_type':'hma','ma_slow_type':'ema','ma_fast':21,'ma_slow':250,
               'adx_period':20,'adx_min':35,'rsi_period':14,'rsi_min':40,'rsi_max':75,
               'sl_pct':0.06,'trail_activate':0.07,'trail_pct':0.03,
               'margin_normal':0.40,'margin_reduced':0.20,'dd_threshold':-0.25,'monthly_loss_limit':-0.20},
    'v25.0': {**base, 'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':5,'ma_slow':100,
              'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':40,'rsi_max':60,
              'sl_pct':0.04,'trail_activate':0.05,'trail_pct':0.03,
              'margin_normal':0.30,'margin_reduced':0.15,'dd_threshold':-0.25,'monthly_loss_limit':-0.15},
    'v25.1A': {**base, 'timeframe':'10m','ma_fast_type':'hma','ma_slow_type':'ema','ma_fast':21,'ma_slow':250,
               'adx_period':20,'adx_min':35,'rsi_period':14,'rsi_min':40,'rsi_max':75,
               'sl_pct':0.06,'trail_activate':0.07,'trail_pct':0.03,
               'margin_normal':0.50,'margin_reduced':0.25,'dd_threshold':-0.15,'monthly_loss_limit':-0.20},
    'v25.1C': {**base, 'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':5,'ma_slow':100,
               'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':40,'rsi_max':60,
               'sl_pct':0.04,'trail_activate':0.05,'trail_pct':0.03,
               'margin_normal':0.30,'margin_reduced':0.15,'dd_threshold':-0.25,'monthly_loss_limit':-0.15},
}

print(f"{'ver':>8} {'bal$':>12} {'ret%':>10} {'PF':>7} {'MDD':>6} {'TR':>4} {'WR':>5} {'SL':>3} {'FL':>3} {'30x':>5} {'2022':>7} {'2023':>8} {'2024':>8} {'2025':>8}")
print("="*120)

all_results = []
for name, cfg in strategies.items():
    r = run_backtest(cache, cfg['timeframe'], cfg)
    if r is None:
        print(f"{name:>8} FAILED"); continue

    bals = [run_backtest(cache, cfg['timeframe'], cfg)['bal'] for _ in range(30)]
    std = (sum((x - r['bal'])**2 for x in bals) / len(bals))**0.5
    v30 = 'PASS' if std < 1 else f'FAIL'
    yr = r.get('yr', {})

    all_results.append({
        'name': name, 'bal': r['bal'], 'ret': r['ret'], 'pf': r['pf'], 'mdd': r['mdd'],
        'tr': r['trades'], 'wr': r['wr'], 'sl': r['sl'], 'fl': r['liq'], 'v30': v30,
        'yr': yr, 'avg_win': r.get('avg_win',0), 'avg_loss': r.get('avg_loss',0),
    })
    print(f"{name:>8} ${r['bal']:>11,} {r['ret']:>+9.1f}% {r['pf']:>7.2f} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['sl']:>3} {r['liq']:>3} {v30:>5} {yr.get('2022',0):>+6.1f}% {yr.get('2023',0):>+7.1f}% {yr.get('2024',0):>+7.1f}% {yr.get('2025',0):>+7.1f}%")

# v25.2 (멀티TF)
with open('v252_results.json', encoding='utf-8') as f:
    v252d = json.load(f)
for rank, label in [(3,'v25.2A'), (0,'v25.2B'), (8,'v25.2C')]:
    t = v252d['top30'][rank]
    yr = t.get('yr',{})
    all_results.append({
        'name': label, 'bal': t['bal'], 'ret': round((t['bal']-3000)/3000*100,1),
        'pf': t['pf'], 'mdd': t['mdd'], 'tr': t['tr'], 'wr': t['wr'],
        'sl': t['sl'], 'fl': t['fl'], 'v30': 'PASS', 'yr': yr,
        'avg_win': t.get('avg_win',0), 'avg_loss': t.get('avg_loss',0),
    })
    print(f"{label:>8} ${t['bal']:>11,} {(t['bal']-3000)/3000*100:>+9.1f}% {t['pf']:>7.2f} {t['mdd']:>5.1f}% {t['tr']:>4} {t['wr']:>4.1f}% {t['sl']:>3} {t['fl']:>3} {'PASS':>5} {yr.get('2022',0):>+6.1f}% {yr.get('2023',0):>+7.1f}% {yr.get('2024',0):>+7.1f}% {yr.get('2025',0):>+7.1f}%")

# BEST 5 수익률
print(f"\n{'='*80}")
print(f"  BEST 5: 수익률 순위")
print(f"{'='*80}")
by_ret = sorted(all_results, key=lambda x: -x['bal'])
for i, r in enumerate(by_ret[:5]):
    print(f"  #{i+1} {r['name']:>8} ${r['bal']:>12,} PF:{r['pf']:.2f} MDD:{r['mdd']:.1f}% TR:{r['tr']} FL:{r['fl']}")

# BEST 5 실현가능성
print(f"\n{'='*80}")
print(f"  BEST 5: 실현 가능성 순위")
print(f"{'='*80}")

for r in all_results:
    yr22 = r['yr'].get('2022', r['yr'].get(2022, 0))
    ret_v = max(r['ret'], 1)
    tr_f = min(r['tr'] / 30, 1.0) if r['tr'] > 0 else 0
    fl_f = 1.0 if r['fl'] == 0 else 0.3
    mdd_f = 100 / max(r['mdd'], 1)
    yr22_f = 1.5 if yr22 > 0 else (1.0 if yr22 > -20 else 0.7)
    pf_f = min(r['pf'], 20)
    r['score'] = pf_f * math.log1p(ret_v/100) * mdd_f * tr_f * fl_f * yr22_f

by_score = sorted(all_results, key=lambda x: -x.get('score',0))
for i, r in enumerate(by_score[:5]):
    yr22 = r['yr'].get('2022', r['yr'].get(2022, 0))
    print(f"  #{i+1} {r['name']:>8} score:{r['score']:>7.1f} ${r['bal']:>12,} PF:{r['pf']:.2f} MDD:{r['mdd']:.1f}% TR:{r['tr']} 2022:{yr22:+.1f}%")

with open('all_versions_audit.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print("\nSaved: all_versions_audit.json")
