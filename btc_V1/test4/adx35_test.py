"""ADX 45->35 변경 시 거래건수/성과 변화 테스트"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from bt_fast import *
import numpy as np

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)
cache = IndicatorCache(mtf)

for p in [2,3,5,7,14,21]:
    for tf in ['5m','10m','15m','30m','1h']:
        try: cache.cache[(tf,f'ma_ema_{p}')] = calc_ema(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_hma_{p}')] = calc_hma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_wma_{p}')] = calc_wma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_sma_{p}')] = calc_sma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
for p in [50,75,100,150,200,250,300,600,750]:
    for tf in ['5m','10m','15m','30m','1h']:
        try: cache.cache[(tf,f'ma_ema_{p}')] = calc_ema(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_sma_{p}')] = calc_sma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass

D = {'adx_period':14,'rsi_period':14,'atr_period':14,'fee_rate':0.0004,
     'use_atr_sl':False,'use_atr_trail':False,'atr_sl_mult':2,'atr_sl_min':0.02,'atr_sl_max':0.12,'atr_trail_mult':1.5,
     'consec_loss_pause':0,'pause_candles':0,'delayed_entry':False,'delay_max_candles':6,
     'delay_price_min':-0.001,'delay_price_max':-0.025}

targets = [
    ('v23.3',45,'15m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':150,'adx_min':45,'rsi_min':40,'rsi_max':75,'sl_pct':0.10,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.25,'margin_reduced':0.12,'initial_capital':5000,'delayed_entry':True,'delay_max_candles':5,'monthly_loss_limit':0,'dd_threshold':0}),
    ('v25.2_2',35,'30m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':100,'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'initial_capital':3000,'monthly_loss_limit':-0.20,'dd_threshold':-0.20}),
    ('v15.6_1',45,'15m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':150,'adx_min':45,'rsi_min':35,'rsi_max':70,'sl_pct':0.04,'trail_activate':0.10,'trail_pct':0.03,'leverage':15,'margin_normal':0.40,'margin_reduced':0.20,'initial_capital':3000,'monthly_loss_limit':0,'dd_threshold':0}),
    ('v17.0_1',35,'30m',{**D,'ma_fast_type':'sma','ma_fast':14,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.05,'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':3,'monthly_loss_limit':-0.20,'dd_threshold':0}),
    ('v22.0_1',45,'15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':70,'sl_pct':0.05,'trail_activate':0.08,'trail_pct':0.04,'leverage':15,'margin_normal':0.50,'margin_reduced':0.25,'initial_capital':3000,'monthly_loss_limit':0,'dd_threshold':0}),
    ('v28.0_1',45,'15m',{**D,'ma_fast_type':'wma','ma_fast':5,'ma_slow_type':'ema','ma_slow':300,'adx_min':45,'rsi_min':35,'rsi_max':70,'sl_pct':0.05,'trail_activate':0.07,'trail_pct':0.03,'leverage':15,'margin_normal':0.15,'margin_reduced':0.075,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':2,'monthly_loss_limit':0,'dd_threshold':0}),
    ('v22.0F',45,'15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.07,'trail_pct':0.05,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'initial_capital':3000,'monthly_loss_limit':0,'dd_threshold':0}),
    ('v22.1_1',45,'15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.07,'trail_pct':0.05,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'initial_capital':3000,'monthly_loss_limit':0,'dd_threshold':0}),
    ('v22.4',45,'15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.07,'trail_pct':0.05,'leverage':15,'margin_normal':0.40,'margin_reduced':0.20,'initial_capital':5000,'monthly_loss_limit':0,'dd_threshold':0}),
    ('v22.5',45,'15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.07,'trail_pct':0.05,'leverage':15,'margin_normal':0.50,'margin_reduced':0.25,'initial_capital':5000,'monthly_loss_limit':0,'dd_threshold':0}),
    ('v22.6_1',45,'15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'initial_capital':5000,'monthly_loss_limit':0,'dd_threshold':0}),
    ('v22.7',45,'15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.07,'trail_pct':0.05,'leverage':15,'margin_normal':0.50,'margin_reduced':0.25,'initial_capital':5000,'monthly_loss_limit':0,'dd_threshold':0}),
    ('v23.0',45,'30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':45,'rsi_min':25,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,'leverage':7,'margin_normal':0.50,'margin_reduced':0.25,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':5,'monthly_loss_limit':0,'dd_threshold':0}),
    ('v16.2_1',45,'30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':45,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,'initial_capital':3000,'monthly_loss_limit':0,'dd_threshold':0}),
]

print()
print('='*145)
print('  ADX -> 35 변경 시 결과 비교 (14개 기획서)')
print('='*145)
hdr = f'| # | {"버전":^10} | {"원ADX":^5} |  {"원거래":^4} | {"원수익률":^10} | {"원PF":^6} | {"원MDD":^5} ||  {"35거래":^4} | {"35수익률":^12} | {"35PF":^6} | {"35MDD":^5} | {"35SL":^3} | {"30+?":^4} |'
print(hdr)
print(f'|---|{"-"*12}|{"-"*7}|{"-"*7}|{"-"*12}|{"-"*8}|{"-"*7}||{"-"*7}|{"-"*14}|{"-"*8}|{"-"*7}|{"-"*5}|{"-"*6}|')

for i, (name, orig_adx, tf, cfg) in enumerate(targets):
    r1 = run_backtest(cache, tf, cfg)
    t1 = r1['trades'] if r1 else 0
    ret1 = r1['ret'] if r1 else -100
    pf1 = r1['pf'] if r1 else 0
    mdd1 = r1['mdd'] if r1 else 0

    cfg35 = {**cfg, 'adx_min': 35}
    r2 = run_backtest(cache, tf, cfg35)
    t2 = r2['trades'] if r2 else 0
    ret2 = r2['ret'] if r2 else -100
    pf2 = r2['pf'] if r2 else 0
    mdd2 = r2['mdd'] if r2 else 0
    sl2 = r2['sl'] if r2 else 0

    mark = 'O' if t2 > 30 else 'X'
    print(f'| {i+1:>1} | {name:^10} | {orig_adx:>4} | {t1:>5} | {ret1:>+9,.0f}% | {pf1:>5.1f} | {mdd1:>4.1f}% || {t2:>5} | {ret2:>+11,.0f}% | {pf2:>5.1f} | {mdd2:>4.1f}% | {sl2:>3} | {mark:^4} |')

over30 = sum(1 for name,adx,tf,cfg in targets if (run_backtest(cache,tf,{**cfg,'adx_min':35}) or {}).get('trades',0) > 30)
print(f'\n  ADX 35 적용 후 30건 초과: {over30}/{len(targets)}')
print('DONE.')
