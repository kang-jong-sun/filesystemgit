"""v15.6/v25.2/v17.0/v22.1/v26.0 각 안별 거래수 확인"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from bt_fast import *
import numpy as np

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)
cache = IndicatorCache(mtf)

for p in [2,3,5,7,14,21]:
    for tf in ['5m','10m','15m','30m']:
        try: cache.cache[(tf,f'ma_ema_{p}')] = calc_ema(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_hma_{p}')] = calc_hma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_wma_{p}')] = calc_wma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_sma_{p}')] = calc_sma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
for p in [100,150,200,250,300]:
    for tf in ['5m','10m','15m','30m']:
        cache.cache[(tf,f'ma_ema_{p}')] = calc_ema(mtf[tf]['close'],p).values.astype(np.float64)
        cache.cache[(tf,f'ma_sma_{p}')] = calc_sma(mtf[tf]['close'],p).values.astype(np.float64)

D = {'adx_period':14,'rsi_period':14,'atr_period':14,'fee_rate':0.0004,
     'use_atr_sl':False,'use_atr_trail':False,'atr_sl_mult':2,'atr_sl_min':0.02,'atr_sl_max':0.12,'atr_trail_mult':1.5,
     'consec_loss_pause':0,'pause_candles':0,'delayed_entry':False,'delay_max_candles':6,
     'delay_price_min':-0.001,'delay_price_max':-0.025}

models = [
    ('v15.6', 'Model A', '15m', {**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':150,
     'adx_min':45,'rsi_min':35,'rsi_max':70,'sl_pct':0.04,'trail_activate':0.10,'trail_pct':0.03,
     'leverage':15,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v15.6', 'Model B', '30m', {**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.05,
     'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),

    ('v25.2', 'Model A', '30m', {**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.07,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),
    ('v25.2', 'Model B', '30m', {**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':100,
     'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),
    ('v25.2', 'Model C', '10m', {**D,'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'ema','ma_slow':300,
     'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),

    ('v17.0', 'Engine A (Sniper)', '30m', {**D,'ma_fast_type':'sma','ma_fast':14,'ma_slow_type':'ema','ma_slow':200,
     'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.05,
     'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000,
     'delayed_entry':True,'delay_max_candles':3}),
    ('v17.0', 'Engine B (Core)', '30m', {**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000}),
    ('v17.0', 'Engine C (Freq)', '30m', {**D,'ma_fast_type':'ema','ma_fast':2,'ma_slow_type':'ema','ma_slow':200,
     'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000}),
    ('v17.0', 'Engine D (v16.6+)', '30m', {**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,
     'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000,
     'delayed_entry':True,'delay_max_candles':5}),

    ('v22.1', 'Engine A', '15m', {**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,
     'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.07,'trail_pct':0.05,
     'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v22.1', 'Engine B', '30m', {**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':60,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.01,
     'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),

    ('v26.0', 'Track A (Sniper)', '30m', {**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'sma','ma_slow':300,
     'adx_min':40,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v26.0', 'Track B (Compounder)', '30m', {**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_min':35,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,
     'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
]

print()
cur = ''
for ver, name, tf, cfg in models:
    if ver != cur:
        print(f'\n  [{ver}] ({sum(1 for v,_,_,_ in models if v==ver)}개 안)')
        print(f'  {"─"*115}')
        cur = ver
    r = run_backtest(cache, tf, cfg)
    if r:
        flag = ' <<< 30건 미만' if r['trades'] < 30 else ''
        print(f'    {name:25} | {tf:4} | 거래: {r["trades"]:>4}건 | ${r["bal"]:>12,} | +{r["ret"]:>10,.1f}% | PF:{r["pf"]:>6.2f} | MDD:{r["mdd"]:>5.1f}% | SL:{r["sl"]:>2} TSL:{r["tsl"]:>2} REV:{r["sig"]:>2} LIQ:{r["liq"]}{flag}')
    else:
        print(f'    {name:25} | {tf:4} | NO TRADES                                                    <<< 30건 미만')

print('\nDONE.')
