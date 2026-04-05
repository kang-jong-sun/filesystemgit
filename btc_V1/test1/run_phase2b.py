"""Phase 2B-4B: High-frequency focused optimization"""
import sys, time, json, os
sys.stdout.reconfigure(line_buffering=True)
from btc_optimizer_v17 import *

print('Loading data...')
df_5m = load_5m_data()
all_data = precompute_all_data(df_5m)

with open('optimization_results/phase1_results.json') as f:
    p1 = json.load(f)

high_freq = [r for r in p1 if r['trades'] >= 20 and r['pf'] >= 1.5]
print('High-freq candidates (>=20 trades, PF>=1.5): %d' % len(high_freq))

# ===== Phase 2B =====
print('\n' + '='*70)
print('  PHASE 2B: Entry Optimization (High-Frequency)')
print('='*70)

ADX_PERIODS = [14, 20]
ADX_THRESHOLDS = [25, 30, 35, 40, 45]
RSI_RANGES = [(20,80), (25,75), (30,70), (35,65), (30,65)]
ENTRY_DELAYS = [0, 1, 3, 5]

candidates = high_freq[:15]
results_2b = []
count = 0
t0 = time.time()

for base in candidates:
    tf = base['tf']
    d = all_data[tf]
    close, high_arr, low_arr = d['close'], d['high'], d['low']
    volume, timestamps = d['volume'], d['timestamp']
    years, mkeys = d['years'], d['month_keys']
    ma_fast = calc_ma(close, base['ma_type'], base['fast_period'], volume)
    ma_slow = calc_ma(close, 'EMA', base['slow_period'], volume)
    atr_14 = calc_atr(high_arr, low_arr, close, 14)
    warmup = base['slow_period'] + 50

    for adx_p in ADX_PERIODS:
        adx_arr = calc_adx(high_arr, low_arr, close, adx_p)
        for adx_min in ADX_THRESHOLDS:
            for rsi_min, rsi_max in RSI_RANGES:
                rsi_arr = calc_rsi(close, 14)
                for delay in ENTRY_DELAYS:
                    res = fast_backtest(
                        close, high_arr, low_arr, timestamps, ma_fast, ma_slow,
                        adx_arr, rsi_arr, atr_14,
                        adx_min=adx_min, rsi_min=rsi_min, rsi_max=rsi_max,
                        entry_delay=delay, sl_pct=0.07, trail_act=0.06, trail_pct=0.03,
                        leverage=10, margin_pct=0.25, monthly_loss_limit=-0.20,
                        fee_rate=0.0004, initial_balance=3000.0, warmup=warmup,
                        years_arr=years, month_keys_arr=mkeys)
                    for k in ['ma_type','tf','fast_period','slow_period']:
                        res[k] = base[k]
                    res.update({'adx_period':adx_p,'adx_min':adx_min,
                               'rsi_min':rsi_min,'rsi_max':rsi_max,'entry_delay':delay})
                    results_2b.append(res)
                    count += 1
                    if count % 500 == 0:
                        print('  Phase 2B: %d (%ds)' % (count, time.time()-t0))

print('  Phase 2B: %d combos in %ds' % (len(results_2b), time.time()-t0))

for r in results_2b:
    r['score'] = r['pf'] * (r['trades']**0.5) / max(r['mdd'], 5) if r['trades'] >= 10 else 0
results_2b.sort(key=lambda x: x['score'], reverse=True)

print('\n  Top 30 (Score = PF * sqrt(Trades) / MDD):')
print('  %4s %5s %5s %5s %5s %7s %2s %4s %9s %7s %6s %5s %7s' %
      ('Rank','MA','TF','F/S','ADX','RSI','D','Tr','Ret%','PF','MDD%','WR%','Score'))
print('  ' + '-'*85)
for idx, r in enumerate(results_2b[:30]):
    print('  %4d %5s %5s %d/%3d %5d %d-%3d %2d %4d %+8.1f%% %6.2f %5.1f%% %4.1f %6.1f' %
          (idx+1, r['ma_type'], r['tf'], r['fast_period'], r['slow_period'],
           r['adx_min'], r['rsi_min'], r['rsi_max'], r['entry_delay'],
           r['trades'], r['return_pct'], r['pf'], r['mdd'], r['win_rate'], r['score']))

with open('optimization_results/phase2b_results.json', 'w') as f:
    json.dump(results_2b[:100], f, indent=2, default=str)

# ===== Phase 3B =====
print('\n' + '='*70)
print('  PHASE 3B: Exit Optimization (High-Frequency)')
print('='*70)

SL_PCTS = [0.03, 0.05, 0.07, 0.08]
TRAIL_ACTS = [0.02, 0.03, 0.05, 0.06, 0.08, 0.10]
TRAIL_PCTS = [0.02, 0.03, 0.05]
TP1_ROIS = [None, 0.15, 0.30]
SL_ATR_MULTS = [0, 3.0]

top_2b = [r for r in results_2b if r['trades'] >= 15 and r['score'] > 0][:20]
results_3b = []
count = 0
t0 = time.time()

for base in top_2b:
    tf = base['tf']
    d = all_data[tf]
    close, high_arr, low_arr = d['close'], d['high'], d['low']
    volume, timestamps = d['volume'], d['timestamp']
    years, mkeys = d['years'], d['month_keys']
    ma_fast = calc_ma(close, base['ma_type'], base['fast_period'], volume)
    ma_slow = calc_ma(close, 'EMA', base['slow_period'], volume)
    adx_arr = calc_adx(high_arr, low_arr, close, base.get('adx_period', 20))
    rsi_arr = calc_rsi(close, 14)
    atr_arr = calc_atr(high_arr, low_arr, close, 14)
    warmup = base['slow_period'] + 50

    for sl in SL_PCTS:
        for ta in TRAIL_ACTS:
            for tp in TRAIL_PCTS:
                if tp >= ta:
                    continue
                for tp1 in TP1_ROIS:
                    for sa in SL_ATR_MULTS:
                        res = fast_backtest(
                            close, high_arr, low_arr, timestamps, ma_fast, ma_slow,
                            adx_arr, rsi_arr, atr_arr,
                            adx_min=base.get('adx_min',35), rsi_min=base.get('rsi_min',30),
                            rsi_max=base.get('rsi_max',65), entry_delay=base.get('entry_delay',0),
                            sl_pct=sl, trail_act=ta, trail_pct=tp, tp1_roi=tp1,
                            leverage=10, margin_pct=0.25, monthly_loss_limit=-0.20,
                            fee_rate=0.0004, initial_balance=3000.0, sl_atr_mult=sa,
                            warmup=warmup, years_arr=years, month_keys_arr=mkeys)
                        for k in ['ma_type','tf','fast_period','slow_period','adx_period',
                                  'adx_min','rsi_min','rsi_max','entry_delay']:
                            if k in base: res[k] = base[k]
                        res.update({'sl_pct':sl,'trail_act':ta,'trail_pct':tp,
                                   'tp1_roi':tp1,'sl_atr_mult':sa})
                        results_3b.append(res)
                        count += 1
                        if count % 1000 == 0:
                            print('  Phase 3B: %d (%ds)' % (count, time.time()-t0))

print('  Phase 3B: %d combos in %ds' % (len(results_3b), time.time()-t0))

for r in results_3b:
    r['score'] = r['pf'] * (r['trades']**0.5) / max(r['mdd'], 5) if r['trades'] >= 10 else 0
results_3b.sort(key=lambda x: x['score'], reverse=True)

print('\n  Top 30:')
print('  %4s %5s %5s %5s %5s %5s %5s %4s %4s %9s %7s %6s %5s %7s' %
      ('Rank','MA','TF','SL%','TAct','TPct','TP1','ATR','Tr','Ret%','PF','MDD%','WR%','Score'))
print('  ' + '-'*90)
for idx, r in enumerate(results_3b[:30]):
    tp1s = '%.0f%%' % (r['tp1_roi']*100) if r['tp1_roi'] else 'None'
    atrs = '%.1f' % r['sl_atr_mult'] if r['sl_atr_mult'] > 0 else '-'
    print('  %4d %5s %5s %4.0f%% %4.0f%% %4.0f%% %5s %4s %4d %+8.1f%% %6.2f %5.1f%% %4.1f %6.1f' %
          (idx+1, r['ma_type'], r['tf'], r['sl_pct']*100, r['trail_act']*100,
           r['trail_pct']*100, tp1s, atrs, r['trades'], r['return_pct'],
           r['pf'], r['mdd'], r['win_rate'], r['score']))

with open('optimization_results/phase3b_results.json', 'w') as f:
    json.dump(results_3b[:100], f, indent=2, default=str)

# ===== Phase 4B =====
print('\n' + '='*70)
print('  PHASE 4B: Risk Management Optimization')
print('='*70)

MARGINS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
LEVERAGES = [5, 7, 10, 15]
ML_LIMITS = [-0.15, -0.20, -0.25, -0.30]

top_3b = [r for r in results_3b if r['trades'] >= 15 and r['score'] > 0][:20]
results_4b = []
count = 0
t0 = time.time()

for base in top_3b:
    tf = base['tf']
    d = all_data[tf]
    close, high_arr, low_arr = d['close'], d['high'], d['low']
    volume, timestamps = d['volume'], d['timestamp']
    years, mkeys = d['years'], d['month_keys']
    ma_fast = calc_ma(close, base['ma_type'], base['fast_period'], volume)
    ma_slow = calc_ma(close, 'EMA', base['slow_period'], volume)
    adx_arr = calc_adx(high_arr, low_arr, close, base.get('adx_period', 20))
    rsi_arr = calc_rsi(close, 14)
    atr_arr = calc_atr(high_arr, low_arr, close, 14)
    warmup = base['slow_period'] + 50

    for margin in MARGINS:
        for lev in LEVERAGES:
            for ml in ML_LIMITS:
                res = fast_backtest(
                    close, high_arr, low_arr, timestamps, ma_fast, ma_slow,
                    adx_arr, rsi_arr, atr_arr,
                    adx_min=base.get('adx_min',35), rsi_min=base.get('rsi_min',30),
                    rsi_max=base.get('rsi_max',65), entry_delay=base.get('entry_delay',0),
                    sl_pct=base.get('sl_pct',0.07), trail_act=base.get('trail_act',0.06),
                    trail_pct=base.get('trail_pct',0.03), tp1_roi=base.get('tp1_roi',None),
                    leverage=lev, margin_pct=margin, monthly_loss_limit=ml,
                    fee_rate=0.0004, initial_balance=3000.0,
                    sl_atr_mult=base.get('sl_atr_mult',0),
                    warmup=warmup, years_arr=years, month_keys_arr=mkeys)
                for k in ['ma_type','tf','fast_period','slow_period','adx_period','adx_min',
                          'rsi_min','rsi_max','entry_delay','sl_pct','trail_act','trail_pct',
                          'tp1_roi','sl_atr_mult']:
                    if k in base: res[k] = base[k]
                res.update({'margin_pct':margin,'leverage':lev,'monthly_loss_limit':ml})
                results_4b.append(res)
                count += 1
                if count % 500 == 0:
                    print('  Phase 4B: %d (%ds)' % (count, time.time()-t0))

print('  Phase 4B: %d combos in %ds' % (len(results_4b), time.time()-t0))

for r in results_4b:
    r['score'] = r['pf'] * (r['trades']**0.5) / max(r['mdd'], 5) if r['trades'] >= 10 else 0
results_4b.sort(key=lambda x: x['score'], reverse=True)

print('\n  Top 30:')
print('  %4s %5s %5s %5s %4s %5s %4s %10s %7s %6s %12s %7s' %
      ('Rank','MA','TF','M%','Lev','ML','Tr','Ret%','PF','MDD%','$Final','Score'))
print('  ' + '-'*90)
for idx, r in enumerate(results_4b[:30]):
    print('  %4d %5s %5s %4.0f%% %3dx %4.0f%% %4d %+9.1f%% %6.2f %5.1f%% $%10s %6.1f' %
          (idx+1, r['ma_type'], r['tf'], r['margin_pct']*100, r['leverage'],
           r['monthly_loss_limit']*100, r['trades'], r['return_pct'],
           r['pf'], r['mdd'], '{:,.0f}'.format(r['balance']), r['score']))

with open('optimization_results/phase4b_results.json', 'w') as f:
    json.dump(results_4b[:100], f, indent=2, default=str)

# ===== VALIDATION =====
print('\n' + '='*70)
print('  VALIDATION (30x with slippage)')
print('='*70)

pf_top = sorted([r for r in results_4b if r['trades'] >= 20], key=lambda x: x['pf'], reverse=True)
ret_top = sorted([r for r in results_4b if r['pf'] >= 2 and r['trades'] >= 15], key=lambda x: x['return_pct'], reverse=True)
bal_top = sorted([r for r in results_4b if r['trades'] >= 15], key=lambda x: x['score'], reverse=True)
freq_top = sorted([r for r in results_4b if r['pf'] >= 1.5], key=lambda x: x['trades'], reverse=True)

final = {}
for label, lst in [('PF', pf_top), ('RET', ret_top), ('BAL', bal_top), ('FREQ', freq_top)]:
    for r in lst[:3]:
        key = '%s_%s_%d_%d_%d_%d_%.3f_%.3f_%.2f_%d' % (
            r['ma_type'], r['tf'], r['fast_period'], r['slow_period'],
            r.get('adx_min',35), r.get('entry_delay',0),
            r.get('sl_pct',0.07), r.get('trail_act',0.06),
            r.get('margin_pct',0.25), r.get('leverage',10))
        if key not in final:
            r['_label'] = label
            final[key] = r

print('  %d unique strategies for validation' % len(final))

validated = []
for key, strat in list(final.items())[:12]:
    stats, runs = validate_strategy(all_data, strat, runs=30)
    strat['validation'] = stats
    validated.append(strat)

with open('optimization_results/validated_hf_results.json', 'w') as f:
    json.dump(validated, f, indent=2, default=str)

# ===== DETAILED TOP 5 =====
print('\n' + '='*70)
print('  DETAILED ANALYSIS - TOP 5')
print('='*70)

validated.sort(key=lambda x: x['validation']['pf_mean'] * x['validation']['return_mean'] / max(x['validation']['mdd_max'], 1), reverse=True)

for idx, strat in enumerate(validated[:5]):
    print('\n' + '='*70)
    lbl = 'STRATEGY #%d: %s %s F%d/S%d ADX>=%d D=%d SL=%.0f%% Trail=%.0f%%/%.0f%% M=%.0f%% L=%dx' % (
        idx+1, strat['ma_type'], strat['tf'], strat['fast_period'], strat['slow_period'],
        strat.get('adx_min',35), strat.get('entry_delay',0),
        strat.get('sl_pct',0.07)*100, strat.get('trail_act',0.06)*100,
        strat.get('trail_pct',0.03)*100, strat.get('margin_pct',0.25)*100,
        strat.get('leverage',10))
    print('  ' + lbl)
    print('='*70)

    detail = detailed_analysis(all_data, strat)

    print('\n  [Yearly Performance]')
    print('  %6s %12s %12s %10s %7s %4s %5s %5s' % ('Year','Start','End','Return','Trades','SL','TSL','REV'))
    print('  ' + '-'*65)
    for y in sorted(detail['yearly'].keys()):
        ys = detail['yearly'][y]
        print('  %6d $%10s $%10s %+9.1f%% %6d %4d %5d %5d' % (
            y, '{:,.0f}'.format(ys['start']), '{:,.0f}'.format(ys['end']),
            ys['return_pct'], ys['trades'], ys.get('sl',0), ys.get('tsl',0), ys.get('rev',0)))

    print('\n  [Monthly Performance]')
    print('  %8s %10s %7s %12s %12s' % ('Month','Return','Trades','PnL','Balance'))
    print('  ' + '-'*55)
    for mk in sorted(detail['monthly'].keys()):
        ms = detail['monthly'][mk]
        if ms['trades'] > 0:
            print('  %8s %+9.1f%% %6d $%10s $%10s' % (
                mk, ms['return_pct'], ms['trades'],
                '{:,.0f}'.format(ms['pnl']),
                '{:,.0f}'.format(ms.get('end', ms['start']))))

# Save final
total = 525 + len(results_2b) + len(results_3b) + len(results_4b) + 14845
report = {
    'total_combos': total,
    'phase1': 525, 'phase2_sniper': 5760, 'phase2b_hf': len(results_2b),
    'phase3_sniper': 8640, 'phase3b_hf': len(results_3b),
    'phase4_sniper': 2800, 'phase4b_hf': len(results_4b),
    'validated_count': len(validated),
    'validated': validated[:5],
}
with open('optimization_results/final_hf_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

print('\n' + '='*70)
print('  TOTAL COMBOS: %s (Sniper: 14,845 + HF: %s)' % (
    '{:,}'.format(total), '{:,}'.format(total - 14845)))
print('='*70)
