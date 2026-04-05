"""v21.5 Re-optimization: MDD <= 30%, PF relaxed, maximize return"""
import sys, time, json, os
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
from btc_optimizer_v21 import *

print('Loading data...')
df_5m = load_data()
all_data = precompute(df_5m)

# Core strategies to sweep
BASES = [
    {'name':'WMA3_D5','ma_type':'WMA','fast_period':3,'slow_period':200,'adx_min':35,'rsi_min':30,'rsi_max':70,'entry_delay':5,'trail_act':0.03,'trail_pct':0.02},
    {'name':'WMA3_D0','ma_type':'WMA','fast_period':3,'slow_period':200,'adx_min':35,'rsi_min':35,'rsi_max':65,'entry_delay':0,'trail_act':0.06,'trail_pct':0.03},
    {'name':'SMA14','ma_type':'SMA','fast_period':14,'slow_period':200,'adx_min':35,'rsi_min':35,'rsi_max':65,'entry_delay':3,'trail_act':0.06,'trail_pct':0.05},
    {'name':'EMA2','ma_type':'EMA','fast_period':2,'slow_period':200,'adx_min':35,'rsi_min':30,'rsi_max':65,'entry_delay':0,'trail_act':0.06,'trail_pct':0.03},
    {'name':'WMA3_T5','ma_type':'WMA','fast_period':3,'slow_period':200,'adx_min':35,'rsi_min':30,'rsi_max':70,'entry_delay':5,'trail_act':0.06,'trail_pct':0.05},
    {'name':'WMA3_T2','ma_type':'WMA','fast_period':3,'slow_period':200,'adx_min':35,'rsi_min':30,'rsi_max':70,'entry_delay':5,'trail_act':0.03,'trail_pct':0.02,'sl_pct_override':0.07},
]

LEVS = [7, 10, 15]
MARGINS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
SL_PCTS = [0.03, 0.04, 0.05, 0.06, 0.07]

d = all_data['30min']
cl,hi,lo,vol = d['close'],d['high'],d['low'],d['volume']
yr,mk = d['years'],d['month_keys']

print('\n' + '='*70)
print('  FULL SWEEP: MDD <= 30%%, maximize return')
print('='*70)

all_results = []
cnt = 0; t0 = time.time()

for base in BASES:
    mf = calc_ma(cl, base['ma_type'], base['fast_period'], vol)
    ms = calc_ma(cl, 'EMA', base['slow_period'], vol)
    ax = calc_adx(hi, lo, cl, 20)
    rs = calc_rsi(cl, 14)
    at = calc_atr(hi, lo, cl, 14)
    wu = base['slow_period'] + 50

    for lev in LEVS:
        for mpct in MARGINS:
            for sl_raw in SL_PCTS:
                sl = min(sl_raw, 0.9/lev)
                r = fast_bt(cl,hi,lo,mf,ms,ax,rs,at,yr,mk,
                           adx_min=base['adx_min'],rsi_min=base['rsi_min'],rsi_max=base['rsi_max'],
                           entry_delay=base['entry_delay'],sl_pct=sl,
                           trail_act=base['trail_act'],trail_pct=base['trail_pct'],
                           leverage=lev,margin_pct=mpct,ml_limit=-0.25,
                           init_bal=5000.0,warmup=wu)
                r.update({'name':base['name'],'ma_type':base['ma_type'],
                         'fast_period':base['fast_period'],'slow_period':base['slow_period'],
                         'adx_min':base['adx_min'],'rsi_min':base['rsi_min'],'rsi_max':base['rsi_max'],
                         'entry_delay':base['entry_delay'],'trail_act':base['trail_act'],
                         'trail_pct':base['trail_pct'],'leverage':lev,'margin_pct':mpct,'sl_pct':sl})
                all_results.append(r)
                cnt += 1
                if cnt % 200 == 0:
                    print('  Sweep: %d (%ds)' % (cnt, time.time()-t0))

print('  Sweep: %d combos in %ds' % (cnt, time.time()-t0))

# Filter MDD <= 30% and sort by return
mdd30 = [r for r in all_results if r['mdd'] <= 30.0 and r['trades'] >= 10]
mdd30.sort(key=lambda x: x['return_pct'], reverse=True)

print('\n' + '='*70)
print('  TOP 40: MDD <= 30%%, sorted by RETURN')
print('='*70)
print('  %3s %10s %3s %4s %4s %5s %5s %5s %4s %10s %7s %6s %5s %4s %4s' %
      ('R','Strategy','L','M%','SL%','TAct','TPct','Tr/m','Tr','Return','PF','MDD%','WR%','SL','FL'))
print('  '+'-'*105)
for i, r in enumerate(mdd30[:40]):
    print('  %3d %10s %2dx %3.0f%% %3.0f%% %4.0f%% %4.0f%% %4.2f %4d %+9.1f%% %6.2f %5.1f%% %4.1f %4d %4d' %
          (i+1, r['name'], r['leverage'], r['margin_pct']*100, r['sl_pct']*100,
           r['trail_act']*100, r['trail_pct']*100, r['tpm'], r['trades'],
           r['return_pct'], r['pf'], r['mdd'], r['win_rate'], r['sl'], r['fl']))

# Also show PF >= 5 subset
pf5 = [r for r in mdd30 if r['pf'] >= 5]
pf5.sort(key=lambda x: x['return_pct'], reverse=True)
print('\n' + '='*70)
print('  TOP 20: MDD <= 30%% AND PF >= 5')
print('='*70)
for i, r in enumerate(pf5[:20]):
    print('  %3d %10s %2dx %3.0f%% SL%3.0f%% T%2.0f/%2.0f Tr=%d Ret=%+.1f%% PF=%.2f MDD=%.1f%% $%s' %
          (i+1, r['name'], r['leverage'], r['margin_pct']*100, r['sl_pct']*100,
           r['trail_act']*100, r['trail_pct']*100, r['trades'],
           r['return_pct'], r['pf'], r['mdd'], '{:,.0f}'.format(r['balance'])))

# Validate top 5 + detailed analysis
print('\n' + '='*70)
print('  VALIDATION + DETAIL: Top 5')
print('='*70)

seen = set()
top5 = []
for r in mdd30:
    key = '%s_%d_%.2f_%.3f' % (r['name'], r['leverage'], r['margin_pct'], r['sl_pct'])
    if key not in seen:
        seen.add(key)
        top5.append(r)
    if len(top5) >= 5:
        break

for idx, strat in enumerate(top5):
    name = strat['name']
    print('\n' + '='*70)
    print('  #%d: %s L%dx M%.0f%% SL%.0f%% Trail+%.0f%%/-%.0f%%' % (
        idx+1, name, strat['leverage'], strat['margin_pct']*100, strat['sl_pct']*100,
        strat['trail_act']*100, strat['trail_pct']*100))
    print('='*70)

    # 30x validation
    base = [b for b in BASES if b['name'] == name][0]
    mf2 = calc_ma(cl, base['ma_type'], base['fast_period'], vol)
    ms2 = calc_ma(cl, 'EMA', base['slow_period'], vol)
    ax2 = calc_adx(hi, lo, cl, 20)
    rs2 = calc_rsi(cl, 14); at2 = calc_atr(hi, lo, cl, 14)

    runs = []
    for run in range(30):
        np.random.seed(run * 42)
        slip = 1 + np.random.uniform(-0.0005, 0.0005, len(cl))
        r = fast_bt(cl*slip,hi*slip,lo*slip,mf2,ms2,ax2,rs2,at2,yr,mk,
                   adx_min=base['adx_min'],rsi_min=base['rsi_min'],rsi_max=base['rsi_max'],
                   entry_delay=base['entry_delay'],sl_pct=strat['sl_pct'],
                   trail_act=base['trail_act'],trail_pct=base['trail_pct'],
                   leverage=strat['leverage'],margin_pct=strat['margin_pct'],
                   ml_limit=-0.25,init_bal=5000.0,warmup=250)
        runs.append(r)

    bals = [r['balance'] for r in runs]
    pfs_v = [r['pf'] for r in runs]
    mdds_v = [r['mdd'] for r in runs]
    rets_v = [r['return_pct'] for r in runs]

    print('  30x Validation:')
    print('    Balance: $%.0f +/- $%.0f (min $%.0f, max $%.0f)' % (np.mean(bals),np.std(bals),np.min(bals),np.max(bals)))
    print('    Return: %.1f%% +/- %.1f%%' % (np.mean(rets_v), np.std(rets_v)))
    print('    PF: %.2f +/- %.2f (min %.2f)' % (np.mean(pfs_v), np.std(pfs_v), np.min(pfs_v)))
    print('    MDD: %.1f%% (max %.1f%%)' % (np.mean(mdds_v), np.max(mdds_v)))

    # Detailed
    p_det = {
        'tf':'30min','ma_type':base['ma_type'],'fast_period':base['fast_period'],
        'slow_period':base['slow_period'],'adx_period':20,'adx_min':base['adx_min'],
        'rsi_min':base['rsi_min'],'rsi_max':base['rsi_max'],'entry_delay':base['entry_delay'],
        'sl_pct':strat['sl_pct'],'trail_act':base['trail_act'],'trail_pct':base['trail_pct'],
        'tp1_roi':None,'leverage':strat['leverage'],'margin_pct':strat['margin_pct'],
        'ml_limit':-0.25,'init_bal':5000.0,'cross_margin':False,
    }
    detail = detail_bt(all_data, p_det)

    print('\n  [Yearly]')
    print('  %6s %12s %12s %10s %7s %4s %5s %5s %4s' % ('Year','Start','End','Return','Trades','SL','TSL','REV','FL'))
    print('  '+'-'*70)
    for y in sorted(detail['yearly'].keys()):
        ys = detail['yearly'][y]
        print('  %6d $%10s $%10s %+9.1f%% %6d %4d %5d %5d %4d' % (
            y, '{:,.0f}'.format(ys['start']), '{:,.0f}'.format(ys.get('end',ys['start'])),
            ys.get('ret',0), ys['trades'], ys.get('sl',0), ys.get('tsl',0), ys.get('rev',0), ys.get('fl',0)))

    print('\n  [Monthly]')
    print('  %8s %10s %7s %12s %12s' % ('Month','Return','Trades','PnL','Balance'))
    print('  '+'-'*55)
    for mk3 in sorted(detail['monthly'].keys()):
        ms3 = detail['monthly'][mk3]
        if ms3['trades'] > 0:
            print('  %8s %+9.1f%% %6d $%10s $%10s' % (
                mk3, ms3.get('ret',0), ms3['trades'],
                '{:,.0f}'.format(ms3['pnl']), '{:,.0f}'.format(ms3.get('end',ms3['start']))))

with open(os.path.join(OUTPUT_DIR, 'mdd30_results.json'), 'w') as f:
    json.dump(mdd30[:50], f, indent=2, default=str)

print('\n' + '='*70)
print('  DONE: %d combos, %d passed MDD<=30%%' % (cnt, len(mdd30)))
print('='*70)
