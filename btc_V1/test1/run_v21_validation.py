"""v21.5 Validation + Detailed Analysis (fix format error)"""
import sys, time, json, os
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
from btc_optimizer_v21 import *

print('Loading data...')
df_5m = load_data()
all_data = precompute(df_5m)

# Best strategies from Phase 3
STRATEGIES = [
    {  # #1: Best Score - SMA14 L15x M25% Trail+6%/-5%
        'name': 'A_Best', 'ma_type':'SMA','tf':'30min','fast_period':14,'slow_period':200,
        'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'entry_delay':0,
        'sl_pct':0.05,'trail_act':0.06,'trail_pct':0.05,'tp1_roi':None,
        'leverage':15,'margin_pct':0.25,'ml_limit':-0.20,'cross_margin':False,'init_bal':5000.0,
    },
    {  # #2: SMA14 L15x M25% Trail+6%/-2% (tighter trail)
        'name': 'B_TightTrail', 'ma_type':'SMA','tf':'30min','fast_period':14,'slow_period':200,
        'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'entry_delay':0,
        'sl_pct':0.05,'trail_act':0.06,'trail_pct':0.02,'tp1_roi':None,
        'leverage':15,'margin_pct':0.25,'ml_limit':-0.20,'cross_margin':False,'init_bal':5000.0,
    },
    {  # #3: SMA14 L15x M20% (lower MDD)
        'name': 'C_LowMDD', 'ma_type':'SMA','tf':'30min','fast_period':14,'slow_period':200,
        'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'entry_delay':0,
        'sl_pct':0.05,'trail_act':0.06,'trail_pct':0.05,'tp1_roi':None,
        'leverage':15,'margin_pct':0.20,'ml_limit':-0.20,'cross_margin':False,'init_bal':5000.0,
    },
    {  # #4: WMA3 L15x M25% (higher frequency)
        'name': 'D_Freq', 'ma_type':'WMA','tf':'30min','fast_period':3,'slow_period':200,
        'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'entry_delay':0,
        'sl_pct':0.05,'trail_act':0.06,'trail_pct':0.03,'tp1_roi':None,
        'leverage':15,'margin_pct':0.25,'ml_limit':-0.20,'cross_margin':False,'init_bal':5000.0,
    },
    {  # #5: WMA3 D5 Trail+3%/-2% (v16.6 style, max return)
        'name': 'E_v166_Style', 'ma_type':'WMA','tf':'30min','fast_period':3,'slow_period':200,
        'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':70,'entry_delay':5,
        'sl_pct':0.05,'trail_act':0.03,'trail_pct':0.02,'tp1_roi':None,
        'leverage':15,'margin_pct':0.25,'ml_limit':-0.20,'cross_margin':False,'init_bal':5000.0,
    },
    {  # #6: EMA2 L10x M25% (max frequency)
        'name': 'F_MaxFreq', 'ma_type':'EMA','tf':'30min','fast_period':2,'slow_period':200,
        'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':65,'entry_delay':0,
        'sl_pct':0.05,'trail_act':0.06,'trail_pct':0.03,'tp1_roi':None,
        'leverage':10,'margin_pct':0.25,'ml_limit':-0.20,'cross_margin':False,'init_bal':5000.0,
    },
]

# Also test margin sweep for best strategy
print('\n' + '='*70)
print('  MARGIN/LEVERAGE SWEEP for SMA14/EMA200')
print('='*70)

d = all_data['30min']
cl,hi,lo,vol = d['close'],d['high'],d['low'],d['volume']
yr,mk = d['years'],d['month_keys']
mf = calc_ma(cl,'SMA',14,vol); ms = calc_ma(cl,'EMA',200,vol)
ax = calc_adx(hi,lo,cl,20); rs = calc_rsi(cl,14); at = calc_atr(hi,lo,cl,14)

print('\n  %3s %4s %10s %7s %6s %4s %4s %4s' % ('L','M%','Return','PF','MDD%','Tr','SL','FL'))
print('  '+'-'*50)
for lev in [5, 7, 10, 15]:
    for mpct in [0.10, 0.15, 0.20, 0.25]:
        sl = min(0.05, 0.9/lev)
        r = fast_bt(cl,hi,lo,mf,ms,ax,rs,at,yr,mk,
                   adx_min=35,rsi_min=35,rsi_max=65,
                   sl_pct=sl,trail_act=0.06,trail_pct=0.05,
                   leverage=lev,margin_pct=mpct,ml_limit=-0.20,
                   init_bal=5000.0,warmup=250)
        print('  %2dx %3.0f%% %+9.1f%% %6.2f %5.1f%% %4d %4d %4d' % (
            lev, mpct*100, r['return_pct'], r['pf'], r['mdd'], r['trades'], r['sl'], r['fl']))

# Validate + detail each strategy
all_results = []
for strat in STRATEGIES:
    name = strat['name']
    print('\n' + '='*70)
    print('  %s' % name)
    print('='*70)

    # 30x Validation
    tf = strat['tf']; d = all_data[tf]
    cl2,hi2,lo2,vol2 = d['close'],d['high'],d['low'],d['volume']
    yr2,mk2 = d['years'],d['month_keys']
    mf2 = calc_ma(cl2, strat['ma_type'], strat['fast_period'], vol2)
    ms2 = calc_ma(cl2, 'EMA', strat['slow_period'], vol2)
    ax2 = calc_adx(hi2, lo2, cl2, strat.get('adx_period', 20))
    rs2 = calc_rsi(cl2, 14); at2 = calc_atr(hi2, lo2, cl2, 14)
    wu = strat['slow_period'] + 50

    runs = []
    for run in range(30):
        np.random.seed(run * 42)
        slip = 1 + np.random.uniform(-0.0005, 0.0005, len(cl2))
        r = fast_bt(cl2*slip,hi2*slip,lo2*slip,mf2,ms2,ax2,rs2,at2,yr2,mk2,
                   adx_min=strat.get('adx_min',35),rsi_min=strat.get('rsi_min',30),
                   rsi_max=strat.get('rsi_max',65),entry_delay=strat.get('entry_delay',0),
                   sl_pct=strat.get('sl_pct',0.05),trail_act=strat.get('trail_act',0.06),
                   trail_pct=strat.get('trail_pct',0.03),tp1_roi=strat.get('tp1_roi'),
                   leverage=strat['leverage'],margin_pct=strat['margin_pct'],
                   ml_limit=strat.get('ml_limit',-0.20),init_bal=5000.0,warmup=wu,
                   cross_margin=strat.get('cross_margin',False))
        runs.append(r)

    bals = [r['balance'] for r in runs]
    pfs = [r['pf'] for r in runs]
    mdds = [r['mdd'] for r in runs]
    rets = [r['return_pct'] for r in runs]

    print('  30x Validation:')
    print('    Balance: $%.0f +/- $%.0f (min $%.0f, max $%.0f)' % (np.mean(bals),np.std(bals),np.min(bals),np.max(bals)))
    print('    Return: %.1f%% +/- %.1f%%' % (np.mean(rets), np.std(rets)))
    print('    PF: %.2f +/- %.2f (min %.2f)' % (np.mean(pfs), np.std(pfs), np.min(pfs)))
    print('    MDD: %.1f%% (max %.1f%%)' % (np.mean(mdds), np.max(mdds)))
    print('    Trades: %.0f, WR: %.1f%%' % (np.mean([r['trades'] for r in runs]), np.mean([r['win_rate'] for r in runs])))

    # Detailed analysis
    detail = detail_bt(all_data, strat)

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

    strat['validation'] = {
        'bal_mean': float(np.mean(bals)), 'bal_std': float(np.std(bals)),
        'bal_min': float(np.min(bals)), 'ret_mean': float(np.mean(rets)),
        'pf_mean': float(np.mean(pfs)), 'pf_min': float(np.min(pfs)),
        'mdd_mean': float(np.mean(mdds)), 'mdd_max': float(np.max(mdds)),
    }
    strat['yearly'] = {str(k): v for k, v in detail['yearly'].items()}
    strat['monthly'] = detail['monthly']
    all_results.append(strat)

with open(os.path.join(OUTPUT_DIR, 'final_validated.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print('\nAll validation complete!')
