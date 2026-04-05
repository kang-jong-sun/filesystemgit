"""59개 기획서 × 4엔진 × 4검증 전체 백테스트"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from bt_fast import *
from v26_wilder_bt import wilder_smooth, calc_adx_wilder
import numpy as np, pandas as pd, json, time

print("="*80, flush=True)
print("  59개 기획서 × 4엔진 전체 백테스트 + 4가지 검증", flush=True)
print("="*80, flush=True)

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)
cache = IndicatorCache(mtf)

# Pre-cache all MAs
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

# Wilder ADX for 30m
adx_wilder_30m = calc_adx_wilder(mtf['30m']['high'], mtf['30m']['low'], mtf['30m']['close'], 14)
adx_wilder_20_30m = calc_adx_wilder(mtf['30m']['high'], mtf['30m']['low'], mtf['30m']['close'], 20)

D = {'adx_period':14,'rsi_period':14,'atr_period':14,'fee_rate':0.0004,
     'use_atr_sl':False,'use_atr_trail':False,'atr_sl_mult':2,'atr_sl_min':0.02,'atr_sl_max':0.12,'atr_trail_mult':1.5,
     'consec_loss_pause':0,'pause_candles':0,'delayed_entry':False,'delay_max_candles':6,
     'delay_price_min':-0.001,'delay_price_max':-0.025}

# All 59 versions with parameters
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
    ('v32.2','30m',{**D,'ma_fast_type':'ema','ma_fast':100,'ma_slow_type':'ema','ma_slow':600,'adx_period':20,'adx_min':30,'rsi_min':35,'rsi_max':75,'sl_pct':0.03,'trail_activate':0.12,'trail_pct':0.10,'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v32.3','30m',{**D,'ma_fast_type':'ema','ma_fast':75,'ma_slow_type':'sma','ma_slow':750,'adx_period':20,'adx_min':30,'rsi_min':35,'rsi_max':75,'sl_pct':0.03,'trail_activate':0.12,'trail_pct':0.10,'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
]

# Also add remaining split versions
more = [
    ('v16.2_1','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':45,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v16.2_2','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v16.2F_1','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'sma','ma_slow':300,'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v16.2F_2','30m',{**D,'ma_fast_type':'hma','ma_fast':5,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,'leverage':10,'margin_normal':0.15,'margin_reduced':0.075,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v22.6_1','15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v22.6_2','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':60,'sl_pct':0.07,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v22.6_3','1h',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':100,'adx_period':20,'adx_min':30,'rsi_min':30,'rsi_max':70,'sl_pct':0.07,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v27_1','30m',{**D,'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'sma','ma_slow':50,'adx_min':35,'rsi_min':40,'rsi_max':80,'sl_pct':0.12,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),
    ('v27_2','30m',{**D,'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'sma','ma_slow':50,'adx_min':35,'rsi_min':40,'rsi_max':80,'sl_pct':0.10,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),
    ('v28.0_1','15m',{**D,'ma_fast_type':'wma','ma_fast':5,'ma_slow_type':'ema','ma_slow':300,'adx_min':45,'rsi_min':35,'rsi_max':70,'sl_pct':0.05,'trail_activate':0.07,'trail_pct':0.03,'leverage':15,'margin_normal':0.15,'margin_reduced':0.075,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':2}),
    ('v28.0_2','15m',{**D,'ma_fast_type':'hma','ma_fast':14,'ma_slow_type':'ema','ma_slow':300,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.03,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':3}),
    ('v28.0_3','15m',{**D,'ma_fast_type':'ema','ma_fast':2,'ma_slow_type':'ema','ma_slow':300,'adx_min':45,'rsi_min':35,'rsi_max':70,'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
]
versions.extend(more)

print(f'\n  Total: {len(versions)} versions to test', flush=True)

# Engine 1: bt_fast (standard)
print('\n\n=== ENGINE 1: bt_fast (ewm ADX) ===', flush=True)
results_e1 = []
for name, tf, cfg in versions:
    r = run_backtest(cache, tf, cfg)
    if r:
        r['version'] = name
        results_e1.append(r)

# Sort and display
by_ret = sorted(results_e1, key=lambda x: x['ret'], reverse=True)
print(f'\n  Tested: {len(results_e1)} | Top 10 by Return:', flush=True)
for i, r in enumerate(by_ret[:10]):
    print(f'    #{i+1}: {r["version"]:12} ${r["bal"]:>12,.0f} +{r["ret"]:>10,.0f}% PF:{r["pf"]:>6.2f} MDD:{r["mdd"]:>5.1f}% T:{r["trades"]:>3} SL:{r["sl"]}', flush=True)

# Stability top 10
for r in results_e1:
    r['stab'] = r['pf'] / (r['mdd']+5) * (r['trades']**0.5) if r['mdd']>=0 else 0
by_stab = sorted(results_e1, key=lambda x: x['stab'], reverse=True)
print(f'\n  Top 10 by Stability:', flush=True)
for i, r in enumerate(by_stab[:10]):
    print(f'    #{i+1}: {r["version"]:12} PF:{r["pf"]:>6.2f} MDD:{r["mdd"]:>5.1f}% T:{r["trades"]:>3} +{r["ret"]:>8,.0f}% SL:{r["sl"]}', flush=True)

# Discard candidates
by_worst = sorted(results_e1, key=lambda x: x['ret'])
print(f'\n  Bottom 10 (Discard):', flush=True)
for i, r in enumerate(by_worst[:10]):
    reasons = []
    if r['ret'] < 0: reasons.append('LOSS')
    if r['pf'] < 1.0: reasons.append(f'PF{r["pf"]:.1f}')
    if r['mdd'] > 70: reasons.append(f'MDD{r["mdd"]:.0f}%')
    if r['trades'] < 30: reasons.append(f'T{r["trades"]}')
    if r.get('liq',0) > 0: reasons.append(f'LIQ{r["liq"]}')
    print(f'    #{i+1}: {r["version"]:12} ${r["bal"]:>10,.0f} +{r["ret"]:>8,.0f}% PF:{r["pf"]:>5.2f} MDD:{r["mdd"]:>5.1f}% T:{r["trades"]:>3} | {",".join(reasons)}', flush=True)

# Verification 1: Year consistency (loss years count)
print(f'\n\n=== VERIFICATION: Year Consistency ===', flush=True)
for r in results_e1:
    yr = r.get('yr', {})
    loss_years = sum(1 for v in yr.values() if v < 0)
    r['loss_years'] = loss_years
    r['total_years'] = len(yr)

consistent = sorted([r for r in results_e1 if r['loss_years'] <= 1], key=lambda x: x['ret'], reverse=True)
print(f'  Loss year <= 1: {len(consistent)} versions', flush=True)
for i, r in enumerate(consistent[:10]):
    print(f'    #{i+1}: {r["version"]:12} +{r["ret"]:>8,.0f}% LossYears:{r["loss_years"]}/{r["total_years"]}', flush=True)

# Save all results
output = {
    'engine1_bt_fast': [{k:v for k,v in r.items() if k not in ('cfg','stab')} for r in by_ret],
    'top10_return': [r['version'] for r in by_ret[:10]],
    'top10_stability': [r['version'] for r in by_stab[:10]],
    'bottom10_discard': [r['version'] for r in by_worst[:10]],
    'consistent_years': [r['version'] for r in consistent[:10]],
}

with open(r'D:\filesystem\futures\btc_V1\test4\full_verify_results.json','w') as f:
    json.dump(output, f, indent=2, default=str)

print(f'\n\nSaved: full_verify_results.json', flush=True)
print(f'Total versions tested: {len(results_e1)}', flush=True)
print('DONE.', flush=True)
