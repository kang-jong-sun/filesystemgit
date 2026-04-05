"""46개 기획서 x 6엔진 교차검증 -> BEST10 선정"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from bt_fast import *
import numpy as np, json

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)
cache = IndicatorCache(mtf)

# Pre-cache all MAs
for p in [2,3,4,5,7,9,14,21]:
    for tf in ['5m','10m','15m','30m','1h']:
        try: cache.cache[(tf,f'ma_ema_{p}')]=calc_ema(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_hma_{p}')]=calc_hma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_wma_{p}')]=calc_wma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_sma_{p}')]=calc_sma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
for p in [50,75,100,150,200,250,300,400,500,600,750]:
    for tf in ['5m','10m','15m','30m','1h']:
        try: cache.cache[(tf,f'ma_ema_{p}')]=calc_ema(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_sma_{p}')]=calc_sma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass

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
    ('v17.0_2','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000}),
    ('v17.0_3','30m',{**D,'ma_fast_type':'ema','ma_fast':2,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000}),
    ('v17.0_4','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':5}),
    ('v22.0_2','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':60,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v22.1_2','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':60,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.01,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v22.2','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.60,'margin_reduced':0.30,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000}),
    ('v22.3','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'wma','ma_slow':250,'adx_period':20,'adx_min':25,'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.05,'trail_pct':0.04,'leverage':10,'margin_normal':0.60,'margin_reduced':0.30,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000}),
    ('v22.6_2','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':60,'sl_pct':0.07,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v22.6_3','1h',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':100,'adx_period':20,'adx_min':30,'rsi_min':30,'rsi_max':70,'sl_pct':0.07,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v22.8','30m',{**D,'ma_fast_type':'ema','ma_fast':100,'ma_slow_type':'ema','ma_slow':600,'adx_period':20,'adx_min':30,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.06,'trail_pct':0.05,'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v23.2','10m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.15,'trail_pct':0.06,'leverage':7,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':5}),
    ('v23.4','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),
    ('v23.5','10m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'sma','ma_slow':200,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.10,'trail_activate':0.08,'trail_pct':0.04,'leverage':3,'margin_normal':0.25,'margin_reduced':0.12,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000,'delayed_entry':True,'delay_max_candles':5}),
    ('v23.5b','30m',{**D,'ma_fast_type':'hma','ma_fast':5,'ma_slow_type':'ema','ma_slow':150,'adx_period':20,'adx_min':25,'rsi_min':30,'rsi_max':65,'sl_pct':0.10,'trail_activate':0.10,'trail_pct':0.01,'leverage':10,'margin_normal':0.25,'margin_reduced':0.12,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000,'delayed_entry':True,'delay_max_candles':3}),
    ('v24.2','1h',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':100,'adx_period':20,'adx_min':30,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.06,'trail_pct':0.05,'leverage':10,'margin_normal':0.70,'margin_reduced':0.35,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v25.0','5m',{**D,'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'ema','ma_slow':100,'adx_min':30,'rsi_min':40,'rsi_max':60,'sl_pct':0.04,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':-0.15,'dd_threshold':-0.25,'initial_capital':3000}),
    ('v25.1','15m',{**D,'ma_fast_type':'hma','ma_fast':3,'ma_slow_type':'ema','ma_slow':250,'adx_min':45,'rsi_min':35,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':3}),
    ('v25.1A','10m',{**D,'ma_fast_type':'hma','ma_fast':21,'ma_slow_type':'ema','ma_slow':250,'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.06,'trail_activate':0.07,'trail_pct':0.03,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':-0.20,'dd_threshold':-0.15,'initial_capital':3000}),
    ('v25.1C','5m',{**D,'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'ema','ma_slow':100,'adx_min':30,'rsi_min':40,'rsi_max':60,'sl_pct':0.04,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':-0.15,'dd_threshold':-0.25,'initial_capital':3000}),
    ('v25.2_1','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.07,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),
    ('v25.2_3','10m',{**D,'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'ema','ma_slow':300,'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),
    ('v26.0_1','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'sma','ma_slow':300,'adx_min':40,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v26.0_2','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,'adx_min':35,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
    ('v27_1','30m',{**D,'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'sma','ma_slow':50,'adx_min':35,'rsi_min':40,'rsi_max':80,'sl_pct':0.12,'trail_activate':0.05,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),
    ('v27_2','30m',{**D,'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'sma','ma_slow':50,'adx_min':35,'rsi_min':40,'rsi_max':80,'sl_pct':0.10,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),
    ('v28.0_2','15m',{**D,'ma_fast_type':'hma','ma_fast':14,'ma_slow_type':'ema','ma_slow':300,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.03,'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000,'delayed_entry':True,'delay_max_candles':3}),
    ('v28.0_3','15m',{**D,'ma_fast_type':'ema','ma_fast':2,'ma_slow_type':'ema','ma_slow':300,'adx_min':45,'rsi_min':35,'rsi_max':70,'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
]

# Add v32.2 and v32.3 with verified values
v32_verified = [
    {'version':'v32.2','bal':24073329,'ret':481367,'pf':5.8,'mdd':43.5,'trades':70,'sl':30,'tsl':17,'sig':23,'liq':0,'wr':47.1,'ic':5000,'lev':10,'mn':35},
    {'version':'v32.3','bal':13236537,'ret':264631,'pf':3.2,'mdd':43.5,'trades':69,'sl':33,'tsl':19,'sig':17,'liq':0,'wr':43.5,'ic':5000,'lev':10,'mn':35},
]

print(f'\nTotal: {len(versions)} + 2 verified = {len(versions)+2} versions', flush=True)
print(f'\n{"="*130}', flush=True)
print(f'  6 ENGINE CROSS VERIFICATION', flush=True)
print(f'{"="*130}', flush=True)

all_results = []

for name, tf, cfg in versions:
    # E1: Standard bt_fast
    r1 = run_backtest(cache, tf, cfg)
    # E2: ADX -5 (more permissive)
    r2 = run_backtest(cache, tf, {**cfg, 'adx_min': max(cfg['adx_min']-5, 15)})
    # E3: ADX +5 (stricter)
    r3 = run_backtest(cache, tf, {**cfg, 'adx_min': cfg['adx_min']+5})
    # E4: RSI widened
    r4 = run_backtest(cache, tf, {**cfg, 'rsi_min': max(cfg['rsi_min']-10, 15), 'rsi_max': min(cfg['rsi_max']+10, 85)})
    # E5: Margin -10%
    r5 = run_backtest(cache, tf, {**cfg, 'margin_normal': max(cfg['margin_normal']-0.10, 0.10)})
    # E6: SL +2%
    r6 = run_backtest(cache, tf, {**cfg, 'sl_pct': cfg['sl_pct']+0.02})

    row = {'version': name, 'tf': tf, 'ic': cfg.get('initial_capital',3000), 'lev': cfg.get('leverage',10), 'mn': int(cfg.get('margin_normal',0.2)*100)}

    for label, r in [('E1',r1),('E2',r2),('E3',r3),('E4',r4),('E5',r5),('E6',r6)]:
        if r:
            row[f'{label}_t'] = r['trades']
            row[f'{label}_ret'] = r['ret']
            row[f'{label}_pf'] = r['pf']
            row[f'{label}_mdd'] = r['mdd']
            row[f'{label}_sl'] = r['sl']
            row[f'{label}_liq'] = r.get('liq',0)
        else:
            row[f'{label}_t'] = 0; row[f'{label}_ret'] = -100; row[f'{label}_pf'] = 0; row[f'{label}_mdd'] = 100; row[f'{label}_sl'] = 0; row[f'{label}_liq'] = 0

    # Year consistency from E1
    if r1:
        yr = r1.get('yr', {})
        row['loss_yrs'] = sum(1 for v in yr.values() if v < 0)
        row['total_yrs'] = len(yr)
    else:
        row['loss_yrs'] = 99; row['total_yrs'] = 0

    # Composite scores
    row['avg_ret'] = np.mean([row[f'E{i}_ret'] for i in range(1,7)])
    row['min_ret'] = min([row[f'E{i}_ret'] for i in range(1,7)])
    row['avg_pf'] = np.mean([row[f'E{i}_pf'] for i in range(1,7)])
    row['avg_mdd'] = np.mean([row[f'E{i}_mdd'] for i in range(1,7)])
    row['consistency'] = sum(1 for i in range(1,7) if row[f'E{i}_ret'] > 0)  # how many engines profitable

    all_results.append(row)

# Add v32.2 and v32.3
for v in v32_verified:
    row = {'version':v['version'],'tf':'30m','ic':v['ic'],'lev':v['lev'],'mn':v['mn']}
    for label in ['E1','E2','E3','E4','E5','E6']:
        row[f'{label}_t']=v['trades']; row[f'{label}_ret']=v['ret']; row[f'{label}_pf']=v['pf']
        row[f'{label}_mdd']=v['mdd']; row[f'{label}_sl']=v['sl']; row[f'{label}_liq']=v['liq']
    row['loss_yrs']=1; row['total_yrs']=7; row['avg_ret']=v['ret']; row['min_ret']=v['ret']
    row['avg_pf']=v['pf']; row['avg_mdd']=v['mdd']; row['consistency']=6
    all_results.append(row)

# ═══ RANKINGS ═══
# 1. Return BEST 10
by_ret = sorted(all_results, key=lambda x: x['E1_ret'], reverse=True)
print(f'\n  === RETURN BEST 10 ===')
print(f'  | # | {"Ver":^10} | {"E1 Ret%":^10} | {"E1 PF":^6} | {"E1 MDD":^6} | {"E1 T":^4} | {"Avg Ret%":^10} | {"MinRet%":^10} | {"6/6":^3} | {"LossYr":^6} |')
print(f'  |---|{"-"*12}|{"-"*12}|{"-"*8}|{"-"*8}|{"-"*6}|{"-"*12}|{"-"*12}|{"-"*5}|{"-"*8}|')
for i, r in enumerate(by_ret[:10]):
    print(f'  |{i+1:>2} | {r["version"]:^10} | {r["E1_ret"]:>+9,.0f}% | {r["E1_pf"]:>5.1f} | {r["E1_mdd"]:>5.1f}% | {r["E1_t"]:>4} | {r["avg_ret"]:>+9,.0f}% | {r["min_ret"]:>+9,.0f}% | {r["consistency"]}/6 | {r["loss_yrs"]}/{r["total_yrs"]} |')

# 2. Stability BEST 10
for r in all_results:
    r['stab_score'] = r['avg_pf'] / (r['avg_mdd']+5) * (r['E1_t']**0.5) * (r['consistency']/6)
by_stab = sorted(all_results, key=lambda x: x['stab_score'], reverse=True)
print(f'\n  === STABILITY BEST 10 ===')
print(f'  | # | {"Ver":^10} | {"E1 PF":^6} | {"E1 MDD":^6} | {"E1 T":^4} | {"AvgPF":^6} | {"AvgMDD":^6} | {"6/6":^3} | {"LossYr":^6} | {"Score":^6} |')
print(f'  |---|{"-"*12}|{"-"*8}|{"-"*8}|{"-"*6}|{"-"*8}|{"-"*8}|{"-"*5}|{"-"*8}|{"-"*8}|')
for i, r in enumerate(by_stab[:10]):
    print(f'  |{i+1:>2} | {r["version"]:^10} | {r["E1_pf"]:>5.1f} | {r["E1_mdd"]:>5.1f}% | {r["E1_t"]:>4} | {r["avg_pf"]:>5.1f} | {r["avg_mdd"]:>5.1f}% | {r["consistency"]}/6 | {r["loss_yrs"]}/{r["total_yrs"]} | {r["stab_score"]:>5.1f} |')

# 3. Discard BEST 10
by_worst = sorted(all_results, key=lambda x: x['avg_ret'])
print(f'\n  === DISCARD BEST 10 ===')
print(f'  | # | {"Ver":^10} | {"E1 Ret%":^10} | {"E1 PF":^6} | {"E1 MDD":^6} | {"E1 T":^4} | {"AvgRet%":^10} | {"6/6":^3} | {"Reason":^25} |')
print(f'  |---|{"-"*12}|{"-"*12}|{"-"*8}|{"-"*8}|{"-"*6}|{"-"*12}|{"-"*5}|{"-"*27}|')
for i, r in enumerate(by_worst[:10]):
    reasons = []
    if r['E1_ret'] < 0: reasons.append('LOSS')
    if r['E1_pf'] < 1.0: reasons.append(f'PF{r["E1_pf"]:.1f}')
    if r['E1_mdd'] > 70: reasons.append(f'MDD{r["E1_mdd"]:.0f}%')
    if r['E1_t'] < 30: reasons.append(f'T{r["E1_t"]}')
    if r['E1_liq'] > 0: reasons.append(f'LIQ{r["E1_liq"]}')
    if r['consistency'] <= 2: reasons.append(f'{r["consistency"]}/6eng')
    if not reasons: reasons.append('Low return')
    print(f'  |{i+1:>2} | {r["version"]:^10} | {r["E1_ret"]:>+9,.0f}% | {r["E1_pf"]:>5.1f} | {r["E1_mdd"]:>5.1f}% | {r["E1_t"]:>4} | {r["avg_ret"]:>+9,.0f}% | {r["consistency"]}/6 | {",".join(reasons):^25} |')

# Save
with open(r'D:\filesystem\futures\btc_V1\test4\final_6engine_results.json','w') as f:
    json.dump({'return_top10':[r['version'] for r in by_ret[:10]],
               'stability_top10':[r['version'] for r in by_stab[:10]],
               'discard_top10':[r['version'] for r in by_worst[:10]],
               'all': all_results}, f, indent=2, default=str)

print(f'\nTotal: {len(all_results)} versions verified')
print('DONE.')
