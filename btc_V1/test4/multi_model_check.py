"""
다중 안 기획서 동시 운영 가능 여부 백테스트 검증
================================================
1. 각 안의 포지션 시계열 생성 (매 30m봉마다 pos = 1/0/-1)
2. 두 안이 동시에 반대 포지션인 시점 = 충돌
3. 충돌 빈도, 충돌 기간 분석
4. 합산 자본배분 시뮬레이션
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from bt_fast import *
import numpy as np, pandas as pd

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)
cache = IndicatorCache(mtf)

# Pre-cache all needed MAs
for p in [2,3,4,5,7,14,21]:
    for tf in ['5m','10m','15m','30m','1h']:
        try: cache.cache[(tf,f'ma_ema_{p}')] = calc_ema(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_hma_{p}')] = calc_hma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_wma_{p}')] = calc_wma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
        try: cache.cache[(tf,f'ma_sma_{p}')] = calc_sma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass
for p in [100,150,200,250,300]:
    for tf in ['5m','10m','15m','30m','1h']:
        cache.cache[(tf,f'ma_ema_{p}')] = calc_ema(mtf[tf]['close'],p).values.astype(np.float64)
        try: cache.cache[(tf,f'ma_sma_{p}')] = calc_sma(mtf[tf]['close'],p).values.astype(np.float64)
        except: pass

D = {'adx_period':14,'rsi_period':14,'atr_period':14,'fee_rate':0.0004,
     'use_atr_sl':False,'use_atr_trail':False,'atr_sl_mult':2,'atr_sl_min':0.02,'atr_sl_max':0.12,'atr_trail_mult':1.5,
     'consec_loss_pause':0,'pause_candles':0,'delayed_entry':False,'delay_max_candles':6,
     'delay_price_min':-0.001,'delay_price_max':-0.025}


def get_position_series(tf, cfg):
    """각 봉마다 포지션 상태 (1=LONG, -1=SHORT, 0=없음) 추출"""
    base = cache.get_base(tf)
    n = base['n']
    closes = base['close']; highs = base['high']; lows = base['low']
    ma_fast = cache.get_ma(tf, cfg.get('ma_fast_type','ema'), cfg.get('ma_fast',7))
    ma_slow = cache.get_ma(tf, cfg.get('ma_slow_type','ema'), cfg.get('ma_slow',100))
    rsi = cache.get_rsi(tf, cfg.get('rsi_period',14))
    adx = cache.get_adx(tf, cfg.get('adx_period',14))

    pos_series = np.zeros(n, dtype=np.int64)
    pos = 0; ep = 0.0; su = 0.0; ppnl = 0.0; trail = False
    bal = cfg.get('initial_capital', 3000.0)
    peak_bal = bal; lev = cfg.get('leverage', 10)

    for i in range(1, n):
        if np.isnan(ma_fast[i]) or np.isnan(ma_slow[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            pos_series[i] = pos; continue

        cp = closes[i]; hp = highs[i]; lp = lows[i]

        if pos != 0:
            if pos == 1: pnl = (cp-ep)/ep; pkc = (hp-ep)/ep; lwc = (lp-ep)/ep
            else: pnl = (ep-cp)/ep; pkc = (ep-lp)/ep; lwc = (ep-hp)/ep
            if pkc > ppnl: ppnl = pkc

            # SL
            sth = cfg.get('sl_pct', 0.07)
            if lwc <= -sth:
                pu = su * (-sth) - su * 0.0004; bal += pu
                pos = 0; pos_series[i] = 0; continue

            # TSL
            if ppnl >= cfg.get('trail_activate', 0.08): trail = True
            if trail:
                tw = cfg.get('trail_pct', 0.06)
                tl = ppnl - tw
                if pnl <= tl:
                    pu = su * tl - su * 0.0004; bal += pu
                    pos = 0; pos_series[i] = 0; continue

            # REV
            cu = ma_fast[i] > ma_slow[i] and ma_fast[i-1] <= ma_slow[i-1]
            cd = ma_fast[i] < ma_slow[i] and ma_fast[i-1] >= ma_slow[i-1]
            ao = adx[i] >= cfg.get('adx_min', 30)
            ro = cfg.get('rsi_min', 30) <= rsi[i] <= cfg.get('rsi_max', 70)
            if (pos == 1 and cd and ao and ro) or (pos == -1 and cu and ao and ro):
                pu = su * pnl - su * 0.0004; bal += pu
                nd = 1 if cu else -1
                pos = nd; ep = cp
                mg = bal * cfg.get('margin_normal', 0.35)
                su = mg * lev; bal -= su * 0.0004
                ppnl = 0; trail = False
                pos_series[i] = pos; continue

        # Entry
        if pos == 0 and bal > 10:
            cu = ma_fast[i] > ma_slow[i] and ma_fast[i-1] <= ma_slow[i-1]
            cd = ma_fast[i] < ma_slow[i] and ma_fast[i-1] >= ma_slow[i-1]
            ao = adx[i] >= cfg.get('adx_min', 30)
            ro = cfg.get('rsi_min', 30) <= rsi[i] <= cfg.get('rsi_max', 70)
            sig = 0
            if cu and ao and ro: sig = 1
            elif cd and ao and ro: sig = -1
            if sig != 0:
                mg = bal * cfg.get('margin_normal', 0.35)
                su = mg * lev; bal -= su * 0.0004
                pos = sig; ep = cp; ppnl = 0; trail = False

        pos_series[i] = pos

    return pos_series


# 다중 안 기획서 목록
multi_docs = [
    {
        'ver': 'v15.6',
        'models': [
            ('Model A (15m)', '15m', {**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':150,
             'adx_min':45,'rsi_min':35,'rsi_max':70,'sl_pct':0.04,'trail_activate':0.10,'trail_pct':0.03,
             'leverage':15,'margin_normal':0.40,'initial_capital':3000}),
            ('Model B (30m)', '30m', {**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
             'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.05,
             'leverage':10,'margin_normal':0.35,'initial_capital':3000}),
        ]
    },
    {
        'ver': 'v17.0',
        'models': [
            ('Engine B (Core)', '30m', {**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
             'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
             'leverage':10,'margin_normal':0.35,'monthly_loss_limit':-0.20,'initial_capital':3000}),
            ('Engine C (Freq)', '30m', {**D,'ma_fast_type':'ema','ma_fast':2,'ma_slow_type':'ema','ma_slow':200,
             'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
             'leverage':10,'margin_normal':0.35,'monthly_loss_limit':-0.20,'initial_capital':3000}),
            ('Engine D (v16.6+)', '30m', {**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
             'adx_period':20,'adx_min':35,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,
             'leverage':10,'margin_normal':0.50,'monthly_loss_limit':-0.20,'initial_capital':3000,
             'delayed_entry':True,'delay_max_candles':5}),
        ]
    },
    {
        'ver': 'v22.1',
        'models': [
            ('Engine A (15m)', '15m', {**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':250,
             'adx_min':45,'rsi_min':35,'rsi_max':75,'sl_pct':0.08,'trail_activate':0.07,'trail_pct':0.05,
             'leverage':10,'margin_normal':0.50,'initial_capital':3000}),
            ('Engine B (30m)', '30m', {**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
             'adx_period':20,'adx_min':35,'rsi_min':35,'rsi_max':60,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.01,
             'leverage':10,'margin_normal':0.50,'initial_capital':3000}),
        ]
    },
    {
        'ver': 'v25.2',
        'models': [
            ('Model A (30m)', '30m', {**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
             'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.07,'trail_pct':0.03,
             'leverage':10,'margin_normal':0.40,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),
            ('Model B (30m)', '30m', {**D,'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':100,
             'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
             'leverage':10,'margin_normal':0.40,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),
            ('Model C (10m)', '10m', {**D,'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'ema','ma_slow':300,
             'adx_period':20,'adx_min':35,'rsi_min':40,'rsi_max':75,'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.03,
             'leverage':10,'margin_normal':0.40,'monthly_loss_limit':-0.20,'dd_threshold':-0.20,'initial_capital':3000}),
        ]
    },
    {
        'ver': 'v26.0',
        'models': [
            ('Track A (Sniper)', '30m', {**D,'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'sma','ma_slow':300,
             'adx_min':40,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,
             'leverage':10,'margin_normal':0.50,'initial_capital':3000}),
            ('Track B (Compounder)', '30m', {**D,'ma_fast_type':'wma','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
             'adx_min':35,'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.03,'trail_pct':0.02,
             'leverage':10,'margin_normal':0.30,'initial_capital':3000}),
        ]
    },
    {
        'ver': 'v32.1',
        'models': [
            ('A안 (EMA100/600)', '30m', {**D,'ma_fast_type':'ema','ma_fast':100,'ma_slow_type':'ema','ma_slow':600,
             'adx_period':20,'adx_min':30,'rsi_min':35,'rsi_max':75,'sl_pct':0.03,'trail_activate':0.12,'trail_pct':0.10,
             'leverage':10,'margin_normal':0.35,'initial_capital':5000}),
            ('B안 (EMA75/SMA750)', '30m', {**D,'ma_fast_type':'ema','ma_fast':75,'ma_slow_type':'sma','ma_slow':750,
             'adx_period':20,'adx_min':30,'rsi_min':35,'rsi_max':75,'sl_pct':0.03,'trail_activate':0.12,'trail_pct':0.10,
             'leverage':10,'margin_normal':0.35,'initial_capital':5000}),
        ]
    },
]

# EMA(2), EMA(75), EMA(100), EMA(600), SMA(300), SMA(750) 추가 캐시
cache.cache[('30m','ma_ema_2')] = calc_ema(mtf['30m']['close'],2).values.astype(np.float64)
cache.cache[('30m','ma_ema_75')] = calc_ema(mtf['30m']['close'],75).values.astype(np.float64)
cache.cache[('30m','ma_ema_100')] = calc_ema(mtf['30m']['close'],100).values.astype(np.float64)
cache.cache[('30m','ma_ema_600')] = calc_ema(mtf['30m']['close'],600).values.astype(np.float64)
cache.cache[('30m','ma_sma_300')] = calc_sma(mtf['30m']['close'],300).values.astype(np.float64)
cache.cache[('30m','ma_sma_750')] = calc_sma(mtf['30m']['close'],750).values.astype(np.float64)

print()
print('='*120)
print('  다중 안 기획서 동시 운영 백테스트 검증')
print('  - 각 안의 포지션 시계열 생성 -> 충돌(반대 포지션) 분석')
print('='*120)

for doc in multi_docs:
    ver = doc['ver']
    models = doc['models']
    print(f'\n{"="*120}')
    print(f'  [{ver}] - {len(models)}개 안')
    print(f'{"="*120}')

    # 각 안의 포지션 시계열 생성
    pos_data = {}
    for name, tf, cfg in models:
        ps = get_position_series(tf, cfg)
        # 30m 기준으로 통일 (다른 TF는 30m에 매핑)
        if tf != '30m':
            ts_tf = mtf[tf]['time'].values.astype('int64')
            ts_30 = mtf['30m']['time'].values.astype('int64')
            idx = np.searchsorted(ts_tf, ts_30, side='right') - 1
            idx = np.clip(idx, 0, len(ps)-1)
            ps_30 = ps[idx]
        else:
            ps_30 = ps
        pos_data[name] = ps_30

        # 포지션 통계
        long_bars = np.sum(ps_30 == 1)
        short_bars = np.sum(ps_30 == -1)
        flat_bars = np.sum(ps_30 == 0)
        total = len(ps_30)
        print(f'  {name:25} | LONG:{long_bars:>6}봉({long_bars/total*100:.1f}%) | SHORT:{short_bars:>6}봉({short_bars/total*100:.1f}%) | 무포지션:{flat_bars:>6}봉({flat_bars/total*100:.1f}%)')

    # 모든 쌍 조합에 대해 충돌 분석
    names = list(pos_data.keys())
    n30 = len(mtf['30m'])

    print(f'\n  --- 충돌 분석 (반대 포지션 동시 보유) ---')
    has_conflict = False

    for i in range(len(names)):
        for j in range(i+1, len(names)):
            pa = pos_data[names[i]]
            pb = pos_data[names[j]]
            min_len = min(len(pa), len(pb))

            # 동시 보유
            both_active = (pa[:min_len] != 0) & (pb[:min_len] != 0)
            same_dir = both_active & (pa[:min_len] == pb[:min_len])
            opposite = both_active & (pa[:min_len] != pb[:min_len]) & both_active

            total_bars = min_len
            both_count = np.sum(both_active)
            same_count = np.sum(same_dir)
            opp_count = np.sum(opposite)

            print(f'\n    {names[i]} vs {names[j]}:')
            print(f'      동시 포지션 보유: {both_count:>6}봉 ({both_count/total_bars*100:.1f}%)')
            print(f'        같은 방향:     {same_count:>6}봉 ({same_count/total_bars*100:.1f}%)')
            print(f'        반대 방향:     {opp_count:>6}봉 ({opp_count/total_bars*100:.1f}%) {"*** 충돌!" if opp_count > 0 else ""}')

            if opp_count > 0:
                has_conflict = True

    # 동시 운영 판정
    if has_conflict:
        print(f'\n  >>> [{ver}] 판정: *** 1계정 동시 운영 불가 (반대 포지션 충돌 발생) ***')
        print(f'      해결책: 서브계정 분리 또는 택 1 운영')
    else:
        all_same_tf = len(set(tf for _, tf, _ in models)) == 1
        if all_same_tf:
            print(f'\n  >>> [{ver}] 판정: 충돌 없음 (같은 방향만 동시 보유)')
            print(f'      그러나 같은 심볼+TF → 바이낸스에서 별도 포지션 불가 → 마진 합산 필요')
        else:
            print(f'\n  >>> [{ver}] 판정: 충돌 없음 + 다른 TF 사용')
            print(f'      바이낸스에서는 여전히 같은 심볼 1포지션 제약 있음')

print('\n\nDONE.')
