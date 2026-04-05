"""Final analysis: 3 engines with detailed monthly/yearly data"""
import sys, time, json
sys.stdout.reconfigure(line_buffering=True)
from btc_optimizer_v17 import *

print('Loading data...')
df_5m = load_5m_data()
all_data = precompute_all_data(df_5m)

ENGINES = {
    'A_Sniper': {
        'ma_type': 'SMA', 'tf': '30min', 'fast_period': 14, 'slow_period': 200,
        'adx_period': 20, 'adx_min': 35, 'rsi_min': 35, 'rsi_max': 65,
        'entry_delay': 3, 'sl_pct': 0.07, 'trail_act': 0.06, 'trail_pct': 0.05,
        'tp1_roi': None, 'sl_atr_mult': 0, 'margin_pct': 0.35, 'leverage': 10,
        'monthly_loss_limit': -0.20,
    },
    'B_Core': {
        'ma_type': 'WMA', 'tf': '30min', 'fast_period': 3, 'slow_period': 200,
        'adx_period': 20, 'adx_min': 35, 'rsi_min': 35, 'rsi_max': 65,
        'entry_delay': 0, 'sl_pct': 0.07, 'trail_act': 0.06, 'trail_pct': 0.03,
        'tp1_roi': None, 'sl_atr_mult': 0, 'margin_pct': 0.35, 'leverage': 10,
        'monthly_loss_limit': -0.20,
    },
    'C_Frequency': {
        'ma_type': 'EMA', 'tf': '30min', 'fast_period': 2, 'slow_period': 200,
        'adx_period': 20, 'adx_min': 35, 'rsi_min': 30, 'rsi_max': 65,
        'entry_delay': 0, 'sl_pct': 0.07, 'trail_act': 0.06, 'trail_pct': 0.03,
        'tp1_roi': None, 'sl_atr_mult': 0, 'margin_pct': 0.35, 'leverage': 10,
        'monthly_loss_limit': -0.20,
    },
}

# Also test margin variants for each engine
MARGIN_VARIANTS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

all_results = {}

for eng_name, params in ENGINES.items():
    print('\n' + '='*70)
    print('  ENGINE: %s' % eng_name)
    print('='*70)

    # Margin sweep
    print('\n  [Margin Sweep]')
    print('  %5s %10s %7s %6s %6s %4s %12s' % ('M%','Return','PF','MDD%','WR%','Tr','$Final'))
    print('  ' + '-'*55)
    for m in MARGIN_VARIANTS:
        p = dict(params)
        p['margin_pct'] = m
        tf = p['tf']
        d = all_data[tf]
        close, high_arr, low_arr = d['close'], d['high'], d['low']
        volume, timestamps = d['volume'], d['timestamp']
        years, mkeys = d['years'], d['month_keys']
        ma_fast = calc_ma(close, p['ma_type'], p['fast_period'], volume)
        ma_slow = calc_ma(close, 'EMA', p['slow_period'], volume)
        adx_arr = calc_adx(high_arr, low_arr, close, p.get('adx_period', 20))
        rsi_arr = calc_rsi(close, 14)
        atr_arr = calc_atr(high_arr, low_arr, close, 14)
        warmup = p['slow_period'] + 50

        res = fast_backtest(
            close, high_arr, low_arr, timestamps, ma_fast, ma_slow,
            adx_arr, rsi_arr, atr_arr,
            adx_min=p['adx_min'], rsi_min=p['rsi_min'], rsi_max=p['rsi_max'],
            entry_delay=p['entry_delay'], sl_pct=p['sl_pct'],
            trail_act=p['trail_act'], trail_pct=p['trail_pct'],
            tp1_roi=p.get('tp1_roi'), leverage=p['leverage'],
            margin_pct=m, monthly_loss_limit=p['monthly_loss_limit'],
            fee_rate=0.0004, initial_balance=3000.0,
            sl_atr_mult=p.get('sl_atr_mult', 0),
            warmup=warmup, years_arr=years, month_keys_arr=mkeys)
        print('  %4.0f%% %+9.1f%% %6.2f %5.1f%% %5.1f %4d $%10s' % (
            m*100, res['return_pct'], res['pf'], res['mdd'],
            res['win_rate'], res['trades'], '{:,.0f}'.format(res['balance'])))

    # Detailed analysis with default params
    print('\n  [Detailed Analysis - Default M%.0f%%]' % (params['margin_pct']*100))
    detail = detailed_analysis(all_data, params)

    print('\n  [Yearly Performance]')
    print('  %6s %12s %12s %10s %7s %4s %5s %5s %4s' % (
        'Year','Start','End','Return','Trades','SL','TSL','REV','FL'))
    print('  ' + '-'*70)
    for y in sorted(detail['yearly'].keys()):
        ys = detail['yearly'][y]
        print('  %6d $%10s $%10s %+9.1f%% %6d %4d %5d %5d %4d' % (
            y, '{:,.0f}'.format(ys['start']), '{:,.0f}'.format(ys['end']),
            ys['return_pct'], ys['trades'], ys.get('sl',0), ys.get('tsl',0),
            ys.get('rev',0), ys.get('fl',0)))

    print('\n  [Monthly Performance]')
    print('  %8s %10s %7s %12s %12s' % ('Month','Return','Trades','PnL','Balance'))
    print('  ' + '-'*55)
    monthly_list = []
    for mk in sorted(detail['monthly'].keys()):
        ms = detail['monthly'][mk]
        if ms['trades'] > 0:
            print('  %8s %+9.1f%% %6d $%10s $%10s' % (
                mk, ms['return_pct'], ms['trades'],
                '{:,.0f}'.format(ms['pnl']),
                '{:,.0f}'.format(ms.get('end', ms['start']))))
            monthly_list.append({'month': mk, **ms})

    # 30x Validation
    print('\n  [30x Validation]')
    stats, _ = validate_strategy(all_data, params, runs=30)

    all_results[eng_name] = {
        'params': params,
        'yearly': {str(k): {kk: float(vv) if hasattr(vv, '__float__') else vv
                            for kk, vv in v.items()}
                  for k, v in detail['yearly'].items()},
        'monthly': monthly_list,
        'validation': stats,
        'detail': {
            'final_balance': detail['final_balance'],
            'max_dd': detail['max_dd'],
            'trade_count': detail['trade_count'],
        }
    }

# Also test Engine B with WMA(3) D5 (v16.6 replica) and Trail +3%/-2%
print('\n' + '='*70)
print('  ENGINE B_v166: WMA(3)/EMA(200) D5 Trail+3%/-2% (v16.6 replica)')
print('='*70)

params_v166 = {
    'ma_type': 'WMA', 'tf': '30min', 'fast_period': 3, 'slow_period': 200,
    'adx_period': 20, 'adx_min': 35, 'rsi_min': 30, 'rsi_max': 70,
    'entry_delay': 5, 'sl_pct': 0.08, 'trail_act': 0.03, 'trail_pct': 0.02,
    'tp1_roi': None, 'sl_atr_mult': 0, 'margin_pct': 0.50, 'leverage': 10,
    'monthly_loss_limit': -0.20,
}
detail_v166 = detailed_analysis(all_data, params_v166)
print('\n  [Yearly]')
print('  %6s %12s %12s %10s %7s %4s %5s %5s' % ('Year','Start','End','Return','Trades','SL','TSL','REV'))
print('  ' + '-'*65)
for y in sorted(detail_v166['yearly'].keys()):
    ys = detail_v166['yearly'][y]
    print('  %6d $%10s $%10s %+9.1f%% %6d %4d %5d %5d' % (
        y, '{:,.0f}'.format(ys['start']), '{:,.0f}'.format(ys['end']),
        ys['return_pct'], ys['trades'], ys.get('sl',0), ys.get('tsl',0), ys.get('rev',0)))

print('\n  [Monthly]')
for mk in sorted(detail_v166['monthly'].keys()):
    ms = detail_v166['monthly'][mk]
    if ms['trades'] > 0:
        print('  %8s %+9.1f%% %6d $%10s $%10s' % (
            mk, ms['return_pct'], ms['trades'],
            '{:,.0f}'.format(ms['pnl']),
            '{:,.0f}'.format(ms.get('end', ms['start']))))

stats_v166, _ = validate_strategy(all_data, params_v166, runs=30)

# Save everything
with open('optimization_results/final_engines.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print('\nAll engine analysis complete!')
