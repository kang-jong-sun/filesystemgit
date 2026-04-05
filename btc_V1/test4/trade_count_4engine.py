"""4가지 엔진으로 전체 기획서 거래 건수 측정"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from bt_fast import *
from v26_wilder_bt import calc_adx_wilder
import numpy as np, json

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)
cache = IndicatorCache(mtf)

# Pre-cache
for p in [2,3,4,5,7,9,14,21]:
    for tf in ['5m','10m','15m','30m','1h']:
        try: cache.cache[(tf,f'ma_ema_{p}')] = calc_ema(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_hma_{p}')] = calc_hma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_wma_{p}')] = calc_wma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_sma_{p}')] = calc_sma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
for p in [50,75,100,150,200,250,300,400,500,600,750]:
    for tf in ['5m','10m','15m','30m','1h']:
        try: cache.cache[(tf,f'ma_ema_{p}')] = calc_ema(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_sma_{p}')] = calc_sma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass

# Wilder ADX caches per TF
wilder_cache = {}
for tf in ['5m','10m','15m','30m','1h']:
    for p in [14, 20]:
        key = f'{tf}_{p}'
        wilder_cache[key] = calc_adx_wilder(mtf[tf]['high'], mtf[tf]['low'], mtf[tf]['close'], p)

D = {'adx_period':14,'rsi_period':14,'atr_period':14,'fee_rate':0.0004,
     'use_atr_sl':False,'use_atr_trail':False,'atr_sl_mult':2,'atr_sl_min':0.02,'atr_sl_max':0.12,'atr_trail_mult':1.5,
     'consec_loss_pause':0,'pause_candles':0,'delayed_entry':False,'delay_max_candles':6,
     'delay_price_min':-0.001,'delay_price_max':-0.025}

versions = [
    ('v12.3','5m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':100,'adx_min':30,'rsi_min':30,'rsi_max':58,'sl_pct':0.09,'trail_activate':0.08,'trail_pct':0.06,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v13.5','5m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':100,'adx_min':30,'rsi_min':30,'rsi_max':58,'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.06,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,'monthly_loss_limit':-0.20,'dd_threshold':-0.50,'initial_capital':3000}),
    ('v14.2F','30m',{**D,'ma_fast_type':'hma','ma_fast':7,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':25,'rsi_min':25,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.10,'trail_pct':0.01,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':-0.15,'dd_threshold':-0.40,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':3}),
    ('v14.4','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.25,'margin_reduced':0.12,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000}),
    ('v15.2','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.05,'trail_activate':0.06,'trail_pct':0.05,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':-0.15,'dd_threshold':0,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':6}),
    ('v15.4','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.30,'dd_threshold':0,'initial_capital':3000}),
    ('v15.5','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.05,'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':-0.25,'dd_threshold':-0.30,'initial_capital':3000}),
    ('v15.6_1','15m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':150,'adx_min':45,'rsi_min':35,'rsi_max':70,'sl_pct':0.04,'trail_activate':0.10,'trail_pct':0.03,'leverage':15,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v15.6_2','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.05,'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v16.0','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v16.2_1','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':45,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v16.2_2','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v16.2_3','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':30,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.25,'margin_reduced':0.12,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v16.2F_1','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'sma','ma_slow':300,'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v16.2F_2','30m',{**D,'ma_fast_type':'hma','ma_fast':5,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,'leverage':10,'margin_normal':0.15,'margin_reduced':0.075,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v16.4','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v16.5','10m',{**D,'ma_fast_type':'hma','ma_fast':21,'ma_slow_type':'ema','ma_slow':250,'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.06,'trail_activate':0.07,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':-0.25,'initial_capital':3000}),
    ('v16.6','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':5}),
    ('v17.0_1','30m',{**D,'ma_fast_type':'sma','ma_fast':14,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.05,'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':3}),
    ('v17.0_2','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000}),
    ('v17.0_3','30m',{**D,'ma_fast_type':'ema','ma_fast':2,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000}),
    ('v17.0_4','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':5}),
    ('v22.0_1','15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':70,'sl_pct':0.05,'trail_activate':0.08,'trail_pct':0.04,'leverage':15,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v22.0_2','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':60,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v22.0F','15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.07,'trail_pct':0.05,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v22.1_1','15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.07,'trail_pct':0.05,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v22.1_2','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':60,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.01,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v22.2','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.60,'margin_reduced':0.30,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000}),
    ('v22.3','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'wma','ma_slow':250,'adx_period':20,'adx_min':25,'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.05,'trail_pct':0.04,'leverage':10,'margin_normal':0.60,'margin_reduced':0.30,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000}),
    ('v22.4','15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.07,'trail_pct':0.05,'leverage':15,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v22.5','15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.07,'trail_pct':0.05,'leverage':15,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v22.6_1','15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v22.6_2','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':60,'sl_pct':0.07,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v22.6_3','1h',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':100,'adx_period':20,'adx_min':30,'rsi_min':30,'rsi_max':70,'sl_pct':0.07,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v22.7','15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.07,'trail_pct':0.05,'leverage':15,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v22.8','30m',{**D,'ma_fast_type':'ema','ma_fast':100,'ma_slow_type':'ema','ma_slow':600,'adx_period':20,'adx_min':30,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.06,'trail_pct':0.05,'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v23.0','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':45,'rsi_min':25,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,'leverage':7,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':5}),
    ('v23.2','10m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.15,'trail_pct':0.06,'leverage':7,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':5}),
    ('v23.3','15m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':150,'adx_min':45,'rsi_min':40,'rsi_max':75,'sl_pct':0.10,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.25,'margin_reduced':0.12,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000,'delayed_entry':True,'delay_max_candles':5}),
    ('v23.4','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v23.5','10m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'sma','ma_slow':200,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.10,'trail_activate':0.08,'trail_pct':0.04,'leverage':3,'margin_normal':0.25,'margin_reduced':0.12,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000,'delayed_entry':True,'delay_max_candles':5}),
    ('v23.5b','30m',{**D,'ma_fast_type':'hma','ma_fast':5,'ma_slow_type':'ema','ma_slow':150,'adx_period':20,'adx_min':25,'rsi_min':30,'rsi_max':65,'sl_pct':0.10,'trail_activate':0.10,'trail_pct':0.01,'leverage':10,'margin_normal':0.25,'margin_reduced':0.12,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000,'delayed_entry':True,'delay_max_candles':3}),
    ('v24.2','1h',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':100,'adx_period':20,'adx_min':30,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.06,'trail_pct':0.05,'leverage':10,'margin_normal':0.70,'margin_reduced':0.35,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v25.0','5m',{**D,'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'ema','ma_slow':100,'adx_min':30,'rsi_min':40,'rsi_max':60,'sl_pct':0.04,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':-0.15,'dd_threshold':-0.25,'initial_capital':3000}),
    ('v25.1','15m',{**D,'ma_fast_type':'hma','ma_fast':3,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':3}),
    ('v25.1A','10m',{**D,'ma_fast_type':'hma','ma_fast':21,'ma_slow_type':'ema','ma_slow':250,'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.06,'trail_activate':0.07,'trail_pct':0.03,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':-0.20,'dd_threshold':-0.15,'initial_capital':3000}),
    ('v25.1C','5m',{**D,'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'ema','ma_slow':100,'adx_min':30,'rsi_min':40,'rsi_max':60,'sl_pct':0.04,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':-0.15,'dd_threshold':-0.25,'initial_capital':3000}),
    ('v25.2_1','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.07,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),
    ('v25.2_2','30m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':100,'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),
    ('v25.2_3','10m',{**D,'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'ema','ma_slow':300,'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),
    ('v26.0_1','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'sma','ma_slow':300,'adx_min':40,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v26.0_2','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':35,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v27_1','30m',{**D,'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'sma','ma_slow':50,'adx_min':35,'rsi_min':40,'rsi_max':80,'sl_pct':0.12,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),
    ('v27_2','30m',{**D,'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'sma','ma_slow':50,'adx_min':35,'rsi_min':40,'rsi_max':80,'sl_pct':0.10,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),
    ('v28.0_1','15m',{**D,'ma_fast_type':'wma','ma_fast':5,'ma_slow_type':'ema','ma_slow':300,'adx_min':45,'rsi_min':35,'rsi_max':70,'sl_pct':0.05,'trail_activate':0.07,'trail_pct':0.03,'leverage':15,'margin_normal':0.15,'margin_reduced':0.075,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':2}),
    ('v28.0_2','15m',{**D,'ma_fast_type':'hma','ma_fast':14,'ma_slow_type':'ema','ma_slow':300,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.03,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':3}),
    ('v28.0_3','15m',{**D,'ma_fast_type':'ema','ma_fast':2,'ma_slow_type':'ema','ma_slow':300,'adx_min':45,'rsi_min':35,'rsi_max':70,'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v32.2','30m',{**D,'ma_fast_type':'ema','ma_fast':100,'ma_slow_type':'ema','ma_slow':600,'adx_period':20,'adx_min':30,'rsi_min':35,'rsi_max':75,'sl_pct':0.03,'trail_activate':0.12,'trail_pct':0.10,'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v32.3','30m',{**D,'ma_fast_type':'ema','ma_fast':75,'ma_slow_type':'sma','ma_slow':750,'adx_period':20,'adx_min':30,'rsi_min':35,'rsi_max':75,'sl_pct':0.03,'trail_activate':0.12,'trail_pct':0.10,'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
]

print(f'\nTotal: {len(versions)} versions', flush=True)

# ═══ 4 ENGINES ═══
# Engine 1: bt_fast standard
# Engine 2: ADX min -5 (more permissive)
# Engine 3: ADX min +5 (more strict)
# Engine 4: RSI range widened (20-80)

results = []

for name, tf, cfg in versions:
    row = {'version': name, 'tf': tf}

    # Engine 1: Standard
    r1 = run_backtest(cache, tf, cfg)
    row['E1_trades'] = r1['trades'] if r1 else 0
    row['E1_ret'] = r1['ret'] if r1 else -100
    row['E1_pf'] = r1['pf'] if r1 else 0

    # Engine 2: ADX -5
    cfg2 = {**cfg, 'adx_min': max(cfg['adx_min'] - 5, 15)}
    r2 = run_backtest(cache, tf, cfg2)
    row['E2_trades'] = r2['trades'] if r2 else 0

    # Engine 3: ADX +5
    cfg3 = {**cfg, 'adx_min': cfg['adx_min'] + 5}
    r3 = run_backtest(cache, tf, cfg3)
    row['E3_trades'] = r3['trades'] if r3 else 0

    # Engine 4: RSI 20-80 (widened)
    cfg4 = {**cfg, 'rsi_min': max(cfg['rsi_min'] - 10, 15), 'rsi_max': min(cfg['rsi_max'] + 10, 85)}
    r4 = run_backtest(cache, tf, cfg4)
    row['E4_trades'] = r4['trades'] if r4 else 0

    row['min_trades'] = min(row['E1_trades'], row['E2_trades'], row['E3_trades'], row['E4_trades'])
    row['max_trades'] = max(row['E1_trades'], row['E2_trades'], row['E3_trades'], row['E4_trades'])
    row['avg_trades'] = round((row['E1_trades'] + row['E2_trades'] + row['E3_trades'] + row['E4_trades']) / 4, 1)
    row['all_under30'] = all(t <= 30 for t in [row['E1_trades'], row['E2_trades'], row['E3_trades'], row['E4_trades']])

    results.append(row)

# Print results
under30 = [r for r in results if r['all_under30']]
under30.sort(key=lambda x: x['avg_trades'])

print(f'\n{"="*120}')
print(f'  4엔진 모두 30건 이하: {len(under30)}개 / {len(results)}개')
print(f'{"="*120}')
print(f'| {"#":>2} | {"버전":^12} | {"TF":^4} | {"E1":^4} | {"E2(-5)":^6} | {"E3(+5)":^6} | {"E4(RSI)":^7} | {"최소":^4} | {"최대":^4} | {"평균":^5} | {"수익률":^10} | {"PF":^6} |')
print(f'|{"-"*4}|{"-"*14}|{"-"*6}|{"-"*6}|{"-"*8}|{"-"*8}|{"-"*9}|{"-"*6}|{"-"*6}|{"-"*7}|{"-"*12}|{"-"*8}|')

for i, r in enumerate(under30):
    print(f'| {i+1:>2} | {r["version"]:^12} | {r["tf"]:^4} | {r["E1_trades"]:>4} | {r["E2_trades"]:>6} | {r["E3_trades"]:>6} | {r["E4_trades"]:>7} | {r["min_trades"]:>4} | {r["max_trades"]:>4} | {r["avg_trades"]:>5} | {r["E1_ret"]:>+9,.0f}% | {r["E1_pf"]:>5.1f} |')

# Also show borderline (some under 30)
some_under = [r for r in results if not r['all_under30'] and r['min_trades'] <= 30]
if some_under:
    print(f'\n  경계선 (일부 엔진만 30건 이하): {len(some_under)}개')
    for r in sorted(some_under, key=lambda x: x['min_trades']):
        print(f'    {r["version"]:12} E1:{r["E1_trades"]:>3} E2:{r["E2_trades"]:>3} E3:{r["E3_trades"]:>3} E4:{r["E4_trades"]:>3} min:{r["min_trades"]} max:{r["max_trades"]}')

# Save
with open(r'D:\filesystem\futures\btc_V1\test4\trade_count_4engine.json', 'w') as f:
    json.dump({'under30_all4': [r['version'] for r in under30],
               'borderline': [r['version'] for r in some_under] if some_under else [],
               'all_results': results}, f, indent=2, default=str)

print(f'\nDONE.', flush=True)
