"""v26.0 Track A/B 개별 운영 vs 동시 운영 월별 상세 비교"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from bt_fast import *
import numpy as np, pandas as pd

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)
cache = IndicatorCache(mtf)

# Cache
for p in [3,5]:
    cache.cache[('30m',f'ma_ema_{p}')] = calc_ema(mtf['30m']['close'],p).values.astype(np.float64)
    cache.cache[('30m',f'ma_wma_{p}')] = calc_wma(mtf['30m']['close'],p).values.astype(np.float64)
for p in [200,300]:
    cache.cache[('30m',f'ma_ema_{p}')] = calc_ema(mtf['30m']['close'],p).values.astype(np.float64)
    cache.cache[('30m',f'ma_sma_{p}')] = calc_sma(mtf['30m']['close'],p).values.astype(np.float64)


def run_detail_30m(cfg, label):
    """30m 봉 기준 건별 거래 추출"""
    c = mtf['30m']['close'].values.astype(np.float64)
    h = mtf['30m']['high'].values.astype(np.float64)
    l = mtf['30m']['low'].values.astype(np.float64)
    t = mtf['30m']['time'].values
    n = len(c)

    fast_ma = cache.get_ma('30m', cfg['ft'], cfg['fl'])
    slow_ma = cache.get_ma('30m', cfg['st'], cfg['sl'])
    adx = cache.get_adx('30m', cfg.get('adx_p', 14))
    rsi = cache.get_rsi('30m', 14)

    lev = cfg['lev']; fee = 0.0004; cap = cfg['cap']
    mn = cfg['mn']; pos = 0; ep = 0.0; su = 0.0; ppnl = 0.0; trail = False
    peak_cap = cap; entry_bar = 0
    trades = []

    for i in range(1, n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            continue
        px = c[i]; hi = h[i]; lo = l[i]

        if pos != 0:
            if pos == 1: pnl=(px-ep)/ep; pkc=(hi-ep)/ep; lwc=(lo-ep)/ep
            else: pnl=(ep-px)/ep; pkc=(ep-lo)/ep; lwc=(ep-hi)/ep
            if pkc > ppnl: ppnl = pkc
            et = None; ep_out = 0

            if lwc <= -cfg['sl_pct']:
                et = 'SL'; ep_out = -cfg['sl_pct']
            if et is None and ppnl >= cfg['ta']:
                trail = True
            if trail and et is None:
                tl = ppnl - cfg['tp']
                if pnl <= tl:
                    et = 'TSL'; ep_out = tl
            if et is None:
                cu = fast_ma[i]>slow_ma[i] and fast_ma[i-1]<=slow_ma[i-1]
                cd = fast_ma[i]<slow_ma[i] and fast_ma[i-1]>=slow_ma[i-1]
                ao = adx[i] >= cfg['amin']; ro = cfg['rmin'] <= rsi[i] <= cfg['rmax']
                if (pos==1 and cd and ao and ro) or (pos==-1 and cu and ao and ro):
                    et = 'REV'; ep_out = pnl

            if et:
                dollar = su * ep_out - su * fee
                cap += dollar
                if cap < 0: cap = 0
                if cap > peak_cap: peak_cap = cap
                ts_exit = pd.Timestamp(t[i])
                trades.append({
                    'entry_time': pd.Timestamp(t[entry_bar]),
                    'exit_time': ts_exit,
                    'dir': 'L' if pos==1 else 'S',
                    'entry_px': round(ep,0),
                    'exit_px': round(px,0),
                    'roi_pct': round(ep_out*100,2),
                    'pnl': round(dollar,0),
                    'type': et,
                    'balance': round(cap,0),
                    'year': ts_exit.year,
                    'month': ts_exit.month,
                })

                nd = 0
                if et == 'REV':
                    if fast_ma[i]>slow_ma[i] and fast_ma[i-1]<=slow_ma[i-1]: nd=1
                    elif fast_ma[i]<slow_ma[i] and fast_ma[i-1]>=slow_ma[i-1]: nd=-1
                if et == 'REV' and nd != 0 and cap > 10:
                    pos=nd; ep=px; entry_bar=i
                    su=cap*mn*lev; cap-=su*fee; su=cap*mn*lev
                    ppnl=0; trail=False
                else:
                    pos=0; ep=0; su=0; ppnl=0; trail=False

        if pos == 0 and cap > 10:
            cu = fast_ma[i]>slow_ma[i] and fast_ma[i-1]<=slow_ma[i-1]
            cd = fast_ma[i]<slow_ma[i] and fast_ma[i-1]>=slow_ma[i-1]
            ao = adx[i] >= cfg['amin']; ro = cfg['rmin'] <= rsi[i] <= cfg['rmax']
            sig = 0
            if cu and ao and ro: sig = 1
            elif cd and ao and ro: sig = -1
            if sig != 0:
                pos=sig; ep=px; entry_bar=i
                su=cap*mn*lev; cap-=su*fee; su=cap*mn*lev
                ppnl=0; trail=False

    if pos != 0 and cap > 10:
        px = c[n-1]
        if pos==1: pnl_f=(px-ep)/ep
        else: pnl_f=(ep-px)/ep
        dollar = su*pnl_f - su*fee; cap += dollar
        trades.append({
            'entry_time':pd.Timestamp(t[entry_bar]),'exit_time':pd.Timestamp(t[n-1]),
            'dir':'L' if pos==1 else 'S','entry_px':round(ep,0),'exit_px':round(px,0),
            'roi_pct':round(pnl_f*100,2),'pnl':round(dollar,0),'type':'END',
            'balance':round(cap,0),'year':2026,'month':3,
        })
    return pd.DataFrame(trades), cap


def print_monthly(df, label, init_cap):
    if len(df) == 0:
        print(f'  {label}: NO TRADES'); return

    all_months = pd.period_range('2020-01','2026-03',freq='M')
    df['ym'] = df['exit_time'].dt.to_period('M')

    print(f'\n{"="*130}')
    print(f'  {label}')
    print(f'{"="*130}')
    print(f'| {"월":^8} | {"거래":^3} | {"승":^2} | {"패":^2} | {"총이익$":^12} | {"총손실$":^12} | {"순손익$":^12} | {"누적$":^12} | {"PF":^6} | {"SL":^2} | {"TSL":^3} | {"REV":^3} |')
    print(f'|{"-"*10}|{"-"*5}|{"-"*4}|{"-"*4}|{"-"*14}|{"-"*14}|{"-"*14}|{"-"*14}|{"-"*8}|{"-"*4}|{"-"*5}|{"-"*5}|')

    cum = 0
    for m in all_months:
        mt = df[df['ym']==m]; tc = len(mt)
        if tc == 0:
            print(f'| {str(m):^8} | {0:^3} | {"-":^2} | {"-":^2} | {"-":^12} | {"-":^12} | {"-":^12} | {cum:>+11,} | {"-":^6} | {"-":^2} | {"-":^3} | {"-":^3} |')
        else:
            w=(mt['pnl']>0).sum(); lo=tc-w
            tp=mt[mt['pnl']>0]['pnl'].sum(); tl=mt[mt['pnl']<=0]['pnl'].sum()
            net=tp+tl; cum+=net
            pf=tp/abs(tl) if tl!=0 else 999
            pfs=f'{pf:.1f}' if pf<100 else 'INF'
            sl_c=(mt['type']=='SL').sum(); tsl_c=(mt['type']=='TSL').sum(); rev_c=(mt['type']=='REV').sum()
            print(f'| {str(m):^8} | {tc:^3} | {w:^2} | {lo:^2} | {tp:>+11,} | {tl:>+11,} | {net:>+11,} | {cum:>+11,} | {pfs:^6} | {sl_c:^2} | {tsl_c:^3} | {rev_c:^3} |')

    # Yearly
    print(f'\n  연도별:')
    for yr in sorted(df['year'].unique()):
        yt=df[df['year']==yr]; tc=len(yt); w=(yt['pnl']>0).sum()
        tp=yt[yt['pnl']>0]['pnl'].sum(); tl=yt[yt['pnl']<=0]['pnl'].sum()
        pf=tp/abs(tl) if tl!=0 else 999; pfs=f'{pf:.2f}' if pf<100 else 'INF'
        print(f'    {yr}: {tc}거래 {w}W/{tc-w}L PF:{pfs} 순손익:${tp+tl:>+,}')

    tt=len(df); tw=(df['pnl']>0).sum()
    ttp=df[df['pnl']>0]['pnl'].sum(); ttl=df[df['pnl']<=0]['pnl'].sum()
    tpf=ttp/abs(ttl) if ttl!=0 else 999
    print(f'\n  ** 최종: ${df.iloc[-1]["balance"]:,} | +{(df.iloc[-1]["balance"]-init_cap)/init_cap*100:,.0f}% | PF:{tpf:.2f} | {tt}거래 **')


def run_combined(df_a, df_b, cap_total, ratio_a, ratio_b):
    """동시 운영 시뮬레이션: 자본 분배 후 각 안의 PnL을 비율적으로 합산"""
    cap_a = cap_total * ratio_a
    cap_b = cap_total * ratio_b

    # 각 거래의 PnL을 초기자본 대비 비율로 변환하여 합산
    events = []
    for _, r in df_a.iterrows():
        events.append({'time': r['exit_time'], 'source': 'A', 'dir': r['dir'],
                       'pnl_ratio': r['pnl'] / 3000,  # 원래 3000 기준 비율
                       'type': r['type'], 'year': r['year'], 'month': r['month']})
    for _, r in df_b.iterrows():
        events.append({'time': r['exit_time'], 'source': 'B', 'dir': r['dir'],
                       'pnl_ratio': r['pnl'] / 3000,
                       'type': r['type'], 'year': r['year'], 'month': r['month']})

    events.sort(key=lambda x: x['time'])

    combined_trades = []
    combined_cap = cap_total
    for e in events:
        if e['source'] == 'A':
            pnl_dollar = combined_cap * ratio_a * e['pnl_ratio']
        else:
            pnl_dollar = combined_cap * ratio_b * e['pnl_ratio']
        combined_cap += pnl_dollar
        if combined_cap < 0: combined_cap = 0

        combined_trades.append({
            'exit_time': e['time'],
            'dir': f"{e['source']}:{e['dir']}",
            'pnl': round(pnl_dollar, 0),
            'type': e['type'],
            'balance': round(combined_cap, 0),
            'year': e['year'],
            'month': e['month'],
        })

    return pd.DataFrame(combined_trades), combined_cap


# ═══ 실행 ═══
cfg_a = {'ft':'ema','fl':3,'st':'sma','sl':300,'adx_p':14,
         'amin':40,'rmin':30,'rmax':70,'sl_pct':0.08,'ta':0.04,'tp':0.03,
         'lev':10,'mn':0.50,'cap':3000.0}

cfg_b = {'ft':'wma','fl':3,'st':'ema','sl':200,'adx_p':14,
         'amin':35,'rmin':30,'rmax':70,'sl_pct':0.08,'ta':0.03,'tp':0.02,
         'lev':10,'mn':0.30,'cap':3000.0}

print('\n\n1) Track A 개별 운영 ($3,000)', flush=True)
df_a, cap_a = run_detail_30m(cfg_a, 'Track A')
print_monthly(df_a, 'v26.0 Track A (Sniper): EMA3/SMA300, ADX>=40, M50%', 3000)

print('\n\n2) Track B 개별 운영 ($3,000)', flush=True)
df_b, cap_b = run_detail_30m(cfg_b, 'Track B')
print_monthly(df_b, 'v26.0 Track B (Compounder): WMA3/EMA200, ADX>=35, M30%', 3000)

print('\n\n3) 동시 운영 ($3,000, A:60% B:40%)', flush=True)
df_c, cap_c = run_combined(df_a, df_b, 3000, 0.60, 0.40)
print_monthly(df_c, 'v26.0 동시운영 (A:60% + B:40%, 자본 $3,000)', 3000)

# 충돌 경고
print('\n\n4) 포지션 충돌 경고', flush=True)
if len(df_a) > 0 and len(df_b) > 0:
    # 시간 겹침 찾기
    conflicts = 0
    for _, ra in df_a.iterrows():
        for _, rb in df_b.iterrows():
            # 두 거래가 시간적으로 겹치고 방향이 반대
            if ra['entry_time'] < rb['exit_time'] and rb['entry_time'] < ra['exit_time']:
                if ra['dir'] != rb['dir']:
                    conflicts += 1
                    if conflicts <= 5:
                        print(f'  충돌 #{conflicts}: A={ra["dir"]} ({str(ra["entry_time"])[:10]}~{str(ra["exit_time"])[:10]}) vs B={rb["dir"]} ({str(rb["entry_time"])[:10]}~{str(rb["exit_time"])[:10]})')

    print(f'\n  총 반대방향 충돌: {conflicts}건')
    if conflicts > 0:
        print(f'  >>> 바이낸스 1계정 동시 운영 불가 (Hedge 모드 필요 또는 서브계정 분리)')
    else:
        print(f'  >>> 충돌 없음 - 1계정 동시 운영 가능')

print('\n\nDONE.', flush=True)
