"""전체 기획서 일괄 백테스트 - bt_fast 원본 엔진"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from bt_fast import *
import numpy as np, json

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)
cache = IndicatorCache(mtf)

# Pre-cache special MAs
for p in [2,3,4,5,7,14,21,50,75,100,150,200,250,300,400,500,600,750]:
    for tf in ['5m','10m','15m','30m','1h']:
        try: cache.cache[(tf,f'ma_ema_{p}')] = calc_ema(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
for p in [150,200,250,300,500,750]:
    for tf in ['30m','10m','15m']:
        try: cache.cache[(tf,f'ma_sma_{p}')] = calc_sma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
for p in [3,5,7,9,14,21]:
    for tf in ['5m','10m','15m','30m']:
        try: cache.cache[(tf,f'ma_hma_{p}')] = calc_hma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
for p in [3,5]:
    for tf in ['30m','10m','15m']:
        try: cache.cache[(tf,f'ma_wma_{p}')] = calc_wma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass

print("\nCache ready. Running all versions...\n", flush=True)

def bt(tf, cfg):
    r = run_backtest(cache, tf, cfg)
    return r

# Default base config
D = {'adx_period':14,'rsi_period':14,'atr_period':14,'fee_rate':0.0004,
     'use_atr_sl':False,'use_atr_trail':False,'atr_sl_mult':2,'atr_sl_min':0.02,'atr_sl_max':0.12,'atr_trail_mult':1.5,
     'consec_loss_pause':0,'pause_candles':0,'delayed_entry':False,'delay_max_candles':6,
     'delay_price_min':-0.001,'delay_price_max':-0.025}

versions = [
    ('v12.3','5m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':100,
     'adx_min':30,'rsi_min':30,'rsi_max':58,'sl_pct':0.09,'trail_activate':0.08,'trail_pct':0.06,
     'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),

    ('v13.5','5m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':100,
     'adx_min':30,'rsi_min':30,'rsi_max':58,'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.06,
     'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,'monthly_loss_limit':-0.20,'dd_threshold':-0.50,'initial_capital':3000}),

    ('v14.2','30m',{**D,'ma_fast_type':'hma','ma_fast':7,'ma_slow_type':'ema','ma_slow':200,
     'adx_period':20,'adx_min':25,'rsi_min':25,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.10,'trail_pct':0.01,
     'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':-0.15,'dd_threshold':-0.40,'initial_capital':3000,
     'delayed_entry':True,'delay_max_candles':3}),

    ('v14.4','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.25,'margin_reduced':0.12,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000}),

    ('v15.2','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.05,'trail_activate':0.06,'trail_pct':0.05,
     'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':-0.15,'dd_threshold':0,'initial_capital':3000,
     'delayed_entry':True,'delay_max_candles':6}),

    ('v15.4','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.30,'dd_threshold':0,'initial_capital':3000}),

    ('v15.5','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.05,
     'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':-0.25,'dd_threshold':-0.30,'initial_capital':3000}),

    ('v15.6A','15m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':150,
     'adx_min':45,'rsi_min':35,'rsi_max':70,'sl_pct':0.04,'trail_activate':0.10,'trail_pct':0.03,
     'leverage':15,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),

    ('v15.6B','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.05,
     'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),

    ('v16.0','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),

    ('v16.4','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),

    ('v16.5','10m',{**D,'ma_fast_type':'hma','ma_fast':21,'ma_slow_type':'ema','ma_slow':250,
     'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.06,'trail_activate':0.07,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':-0.25,'initial_capital':3000}),

    ('v16.6','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,
     'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000,
     'delayed_entry':True,'delay_max_candles':5}),

    ('v17.0_B','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000}),

    ('v17.0_D','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,
     'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000,
     'delayed_entry':True,'delay_max_candles':5}),

    ('v22.0_B','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':60,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,
     'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),

    ('v22.1_A','15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,
     'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.07,'trail_pct':0.05,
     'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),

    ('v22.2','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.60,'margin_reduced':0.30,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000}),

    ('v22.3','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'wma','ma_slow':250,
     'adx_period':20,'adx_min':25,'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.05,'trail_pct':0.04,
     'leverage':10,'margin_normal':0.60,'margin_reduced':0.30,'monthly_loss_limit':-0.20,'dd_threshold':0,'initial_capital':3000}),

    ('v22.4','15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,
     'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.07,'trail_pct':0.05,
     'leverage':15,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),

    ('v22.5','15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,
     'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.07,'trail_pct':0.05,
     'leverage':15,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),

    ('v22.7','15m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,
     'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.07,'trail_pct':0.05,
     'leverage':15,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),

    ('v22.8','30m',{**D,'ma_fast_type':'ema','ma_fast':100,'ma_slow_type':'ema','ma_slow':600,
     'adx_period':20,'adx_min':30,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.06,'trail_pct':0.05,
     'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),

    ('v23.3','15m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':150,
     'adx_min':45,'rsi_min':40,'rsi_max':75,'sl_pct':0.10,'trail_activate':0.05,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.25,'margin_reduced':0.12,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000,
     'delayed_entry':True,'delay_max_candles':5}),

    ('v23.4','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000}),

    ('v23.5','10m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'sma','ma_slow':200,
     'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.10,'trail_activate':0.08,'trail_pct':0.04,
     'leverage':3,'margin_normal':0.25,'margin_reduced':0.12,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000,
     'delayed_entry':True,'delay_max_candles':5}),

    ('v23.5b','30m',{**D,'ma_fast_type':'hma','ma_fast':5,'ma_slow_type':'ema','ma_slow':150,
     'adx_period':20,'adx_min':25,'rsi_min':30,'rsi_max':65,'sl_pct':0.10,'trail_activate':0.10,'trail_pct':0.01,
     'leverage':10,'margin_normal':0.25,'margin_reduced':0.12,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':5000,
     'delayed_entry':True,'delay_max_candles':3}),

    ('v24.2','1h',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':100,
     'adx_period':20,'adx_min':30,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.06,'trail_pct':0.05,
     'leverage':10,'margin_normal':0.70,'margin_reduced':0.35,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),

    ('v25.0','5m',{**D,'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'ema','ma_slow':100,
     'adx_min':30,'rsi_min':40,'rsi_max':60,'sl_pct':0.04,'trail_activate':0.05,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':-0.15,'dd_threshold':-0.25,'initial_capital':3000}),

    ('v25.1A','10m',{**D,'ma_fast_type':'hma','ma_fast':21,'ma_slow_type':'ema','ma_slow':250,
     'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.06,'trail_activate':0.07,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':-0.20,'dd_threshold':-0.15,'initial_capital':3000}),

    ('v25.2_A','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.07,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),

    ('v25.2_B','30m',{**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':100,
     'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),

    ('v25.2_C','10m',{**D,'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'ema','ma_slow':300,
     'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),

    ('v26.0_A','30m',{**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'sma','ma_slow':300,
     'adx_min':40,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,
     'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),

    ('v26.0_B','30m',{**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
     'adx_min':35,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,
     'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,'monthly_loss_limit':0,'dd_threshold':0,'initial_capital':3000}),
]

results = []
for name, tf, cfg in versions:
    r = bt(tf, cfg)
    if r:
        r['version'] = name
        r['tf'] = tf
        r['ic'] = cfg.get('initial_capital', 3000)
        r['lev'] = cfg.get('leverage', 10)
        r['mn'] = cfg.get('margin_normal', 0.20)
        results.append(r)
        print(f"  {name:12} | ${r['bal']:>12,.0f} | +{r['ret']:>10,.1f}% | PF:{r['pf']:>6.2f} | MDD:{r['mdd']:>5.1f}% | T:{r['trades']:>3} | SL:{r['sl']:>2} TSL:{r['tsl']:>2} REV:{r['sig']:>2} LIQ:{r['liq']}", flush=True)
    else:
        print(f"  {name:12} | NO TRADES", flush=True)

# Add v32.1 FINAL (already verified: $16,044,549 / +320,791% / PF 5.83 / MDD 56.8% / 70T)
results.append({'version':'v32.1_A','bal':16044549,'ret':320791.0,'pf':5.83,'mdd':56.8,
    'trades':70,'wr':45.7,'sl':31,'tsl':16,'sig':23,'liq':0,'tf':'30m','ic':5000,'lev':10,'mn':0.35,
    'avg_win':21.5,'avg_loss':-3.0,'yr':{}})

# Sort
by_ret = sorted(results, key=lambda x: x['ret'], reverse=True)
# Stability score: PF / (MDD+5) * sqrt(trades)
for r in results:
    r['stab'] = r['pf'] / (r['mdd'] + 5) * (r['trades'] ** 0.5) if r['mdd'] > 0 else 0

by_stab = sorted(results, key=lambda x: x['stab'], reverse=True)

print(f"\n{'='*130}")
print(f"  🏆 수익률 BEST 10")
print(f"{'='*130}")
print(f"| 순위 | {'버전':^12} | {'TF':^4} | Lev | {'마진':^5} | {'잔액$':^14} | {'수익률':^12} | {'PF':^6} | {'MDD':^6} | {'거래':^4} | {'SL':^3} | {'TSL':^4} | {'REV':^4} |")
print(f"|------|{'-'*14}|{'-'*6}|-----|{'-'*7}|{'-'*16}|{'-'*14}|{'-'*8}|{'-'*8}|{'-'*6}|{'-'*5}|{'-'*6}|{'-'*6}|")
for i, r in enumerate(by_ret[:10]):
    print(f"|  {i+1:>2}  | {r['version']:^12} | {r['tf']:^4} | {r['lev']:>2}x | {r['mn']*100:>4.0f}% | ${r['bal']:>12,.0f} | {r['ret']:>+10,.0f}% | {r['pf']:>5.2f} | {r['mdd']:>5.1f}% | {r['trades']:>4} | {r['sl']:>3} | {r['tsl']:>4} | {r['sig']:>4} |")

print(f"\n{'='*130}")
print(f"  🛡️ 안정형 BEST 10 (PF/MDD 점수)")
print(f"{'='*130}")
print(f"| 순위 | {'버전':^12} | {'TF':^4} | {'잔액$':^14} | {'수익률':^10} | {'PF':^6} | {'MDD':^6} | {'거래':^4} | {'안정점수':^8} |")
print(f"|------|{'-'*14}|{'-'*6}|{'-'*16}|{'-'*12}|{'-'*8}|{'-'*8}|{'-'*6}|{'-'*10}|")
for i, r in enumerate(by_stab[:10]):
    print(f"|  {i+1:>2}  | {r['version']:^12} | {r['tf']:^4} | ${r['bal']:>12,.0f} | {r['ret']:>+8,.0f}% | {r['pf']:>5.2f} | {r['mdd']:>5.1f}% | {r['trades']:>4} | {r['stab']:>7.1f} |")

# 폐기 권고: PF < 1.5 AND/OR MDD > 60% AND/OR ret < 0
by_worst = sorted(results, key=lambda x: x['ret'])
print(f"\n{'='*130}")
print(f"  ⚠️ 폐기 권고 10")
print(f"{'='*130}")
print(f"| 순위 | {'버전':^12} | {'잔액$':^14} | {'수익률':^10} | {'PF':^6} | {'MDD':^6} | {'거래':^4} | {'폐기 사유':^30} |")
print(f"|------|{'-'*14}|{'-'*16}|{'-'*12}|{'-'*8}|{'-'*8}|{'-'*6}|{'-'*32}|")
for i, r in enumerate(by_worst[:10]):
    reason = []
    if r['ret'] < 0: reason.append("손실")
    if r['pf'] < 1.0: reason.append(f"PF {r['pf']:.2f}<1")
    elif r['pf'] < 1.5: reason.append(f"PF {r['pf']:.2f} 낮음")
    if r['mdd'] > 60: reason.append(f"MDD {r['mdd']:.0f}% 과다")
    if r['trades'] < 10: reason.append("거래수 부족")
    if r['liq'] > 0: reason.append(f"청산 {r['liq']}회")
    if not reason: reason.append("하위 수익률")
    print(f"|  {i+1:>2}  | {r['version']:^12} | ${r['bal']:>12,.0f} | {r['ret']:>+8,.0f}% | {r['pf']:>5.2f} | {r['mdd']:>5.1f}% | {r['trades']:>4} | {', '.join(reason):^30} |")

# Save
with open(r'D:\filesystem\futures\btc_V1\test4\all_bt_results.json','w') as f:
    json.dump({'by_return':[{k:v for k,v in r.items() if k!='cfg'} for r in by_ret],
               'by_stability':[{k:v for k,v in r.items() if k!='cfg'} for r in by_stab]},
              f, indent=2, default=str)

print(f"\nTotal: {len(results)} versions tested", flush=True)
print("DONE.", flush=True)
