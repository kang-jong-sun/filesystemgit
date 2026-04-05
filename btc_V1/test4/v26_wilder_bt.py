"""v26.0 Wilder ADX 정밀 재현 백테스트"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from bt_fast import load_5m_data, build_mtf, calc_ema, calc_sma, calc_wma, calc_rsi

def wilder_smooth(values, period):
    """Wilder's Smoothing (v16.4/v26.0 정확 공식)"""
    result = np.full(len(values), np.nan)
    # 시드: 처음 N개의 평균
    valid = values[~np.isnan(values)]
    if len(valid) < period:
        return result
    first_valid = np.where(~np.isnan(values))[0]
    if len(first_valid) < period:
        return result
    start = first_valid[period - 1]
    result[start] = np.nanmean(values[first_valid[0]:first_valid[0]+period])
    for i in range(start + 1, len(values)):
        if np.isnan(values[i]):
            result[i] = result[i-1]
        else:
            result[i] = (result[i-1] * (period - 1) + values[i]) / period
    return result

def calc_adx_wilder(high, low, close, period=14):
    """ADX Wilder's 방식 (v26.0 기획서 정확 구현)"""
    h = high.values if hasattr(high, 'values') else high
    l = low.values if hasattr(low, 'values') else low
    c = close.values if hasattr(close, 'values') else close
    n = len(h)

    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        tr1 = h[i] - l[i]
        tr2 = abs(h[i] - c[i-1])
        tr3 = abs(l[i] - c[i-1])
        tr[i] = max(tr1, tr2, tr3)

        up = h[i] - h[i-1]
        dn = l[i-1] - l[i]
        if up > dn and up > 0:
            plus_dm[i] = up
        if dn > up and dn > 0:
            minus_dm[i] = dn

    atr = wilder_smooth(tr, period)
    smooth_pdm = wilder_smooth(plus_dm, period)
    smooth_mdm = wilder_smooth(minus_dm, period)

    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)

    for i in range(n):
        if not np.isnan(atr[i]) and atr[i] > 0:
            plus_di[i] = 100 * smooth_pdm[i] / atr[i]
            minus_di[i] = 100 * smooth_mdm[i] / atr[i]
            s = plus_di[i] + minus_di[i]
            if s > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / s

    adx = wilder_smooth(dx, period)
    return adx


def run_v26_track(c, h, l, t, fast_ma, slow_ma, adx, rsi, cfg, label):
    """v26.0 Track 백테스트 (v16.4 엔진 로직)"""
    n = len(c)
    cap = cfg['cap']; lev = cfg['lev']; fee = 0.0004
    mn = cfg['mn']
    pos = 0; ep = 0.0; su = 0.0; ppnl = 0.0; trail = False
    peak_cap = cap
    trades = []

    for i in range(1, n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            continue
        px = c[i]; hi = h[i]; lo = l[i]

        if pos != 0:
            if pos == 1:
                pnl = (px-ep)/ep; pkc = (hi-ep)/ep; lwc = (lo-ep)/ep
            else:
                pnl = (ep-px)/ep; pkc = (ep-lo)/ep; lwc = (ep-hi)/ep
            if pkc > ppnl: ppnl = pkc

            et = None; ep_val = 0

            # SL
            if lwc <= -cfg['sl']:
                et = 'SL'; ep_val = -cfg['sl']

            # TSL
            if et is None and ppnl >= cfg['ta']:
                trail = True
            if trail and et is None:
                tl = ppnl - cfg['tp']
                if pnl <= tl:
                    et = 'TSL'; ep_val = tl

            # REV
            if et is None:
                cu = fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]
                cd = fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]
                ao = adx[i] >= cfg['amin']
                ro = cfg['rmin'] <= rsi[i] <= cfg['rmax']
                if (pos == 1 and cd and ao and ro) or (pos == -1 and cu and ao and ro):
                    et = 'REV'; ep_val = pnl

            if et:
                dollar = su * ep_val - su * fee
                cap += dollar
                if cap < 0: cap = 0
                if cap > peak_cap: peak_cap = cap

                ts_exit = pd.Timestamp(t[i])
                trades.append({
                    'entry_time': pd.Timestamp(t[trades[-1]['_entry_bar'] if trades and '_entry_bar' in trades[-1] else 0]) if False else '',
                    'exit_time': ts_exit,
                    'dir': 'L' if pos == 1 else 'S',
                    'entry_px': round(ep, 0),
                    'exit_px': round(px, 0),
                    'roi_pct': round(ep_val * 100, 2),
                    'pnl': round(dollar, 0),
                    'type': et,
                    'balance': round(cap, 0),
                    'year': ts_exit.year,
                    'month': ts_exit.month,
                })

                # REV -> re-entry
                nd = 0
                if et == 'REV':
                    if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]: nd = 1
                    elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]: nd = -1
                if et == 'REV' and nd != 0 and cap > 10:
                    pos = nd; ep = px
                    su = cap * mn * lev; cap -= su * fee; su = cap * mn * lev
                    ppnl = 0; trail = False
                else:
                    pos = 0; ep = 0; su = 0; ppnl = 0; trail = False

        if pos == 0 and cap > 10:
            cu = fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]
            cd = fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]
            ao = adx[i] >= cfg['amin']
            ro = cfg['rmin'] <= rsi[i] <= cfg['rmax']
            sig = 0
            if cu and ao and ro: sig = 1
            elif cd and ao and ro: sig = -1
            if sig != 0:
                pos = sig; ep = px
                su = cap * mn * lev; cap -= su * fee; su = cap * mn * lev
                ppnl = 0; trail = False

    # Close open
    if pos != 0 and cap > 10:
        px = c[n-1]
        if pos == 1: pf = (px-ep)/ep
        else: pf = (ep-px)/ep
        dollar = su * pf - su * fee; cap += dollar
        trades.append({
            'exit_time': pd.Timestamp(t[n-1]), 'dir': 'L' if pos == 1 else 'S',
            'entry_px': round(ep,0), 'exit_px': round(px,0),
            'roi_pct': round(pf*100,2), 'pnl': round(dollar,0), 'type': 'END',
            'balance': round(cap,0), 'year': pd.Timestamp(t[n-1]).year, 'month': pd.Timestamp(t[n-1]).month,
        })

    return pd.DataFrame(trades), cap


def print_results(df, label, init_cap):
    if len(df) == 0:
        print(f'  {label}: NO TRADES'); return

    print(f'\n{"="*120}')
    print(f'  {label}')
    print(f'{"="*120}')

    # Trade list
    print(f'\n  건별 ({len(df)}건)')
    print(f'| # | {"시간":^16} | {"방향":^2} | {"진입가":>8} | {"청산가":>8} | {"ROI%":>7} | {"손익$":>12} | {"사유":^3} | {"잔액$":>12} |')
    print(f'|---|{"-"*18}|{"-"*4}|{"-"*10}|{"-"*10}|{"-"*9}|{"-"*14}|{"-"*5}|{"-"*14}|')
    for i, r in df.iterrows():
        print(f'| {i+1:>1} | {str(r["exit_time"])[:16]:^16} | {r["dir"]:^2} | {r["entry_px"]:>8,.0f} | {r["exit_px"]:>8,.0f} | {r["roi_pct"]:>+6.1f}% | {r["pnl"]:>+11,.0f} | {r["type"]:^3} | {r["balance"]:>11,.0f} |')

    # Yearly
    print(f'\n  연도별:')
    for yr in sorted(df['year'].unique()):
        yt = df[df['year'] == yr]; tc = len(yt)
        w = (yt['pnl'] > 0).sum()
        tp = yt[yt['pnl'] > 0]['pnl'].sum()
        tl = yt[yt['pnl'] <= 0]['pnl'].sum()
        pf = tp / abs(tl) if tl != 0 else 999
        pfs = f'{pf:.1f}' if pf < 100 else 'INF'
        sl_c = (yt['type'] == 'SL').sum()
        tsl_c = (yt['type'] == 'TSL').sum()
        rev_c = (yt['type'] == 'REV').sum()
        print(f'    {yr}: {tc}거래 {w}W/{tc-w}L PF:{pfs} ${tp+tl:>+,.0f} SL:{sl_c} TSL:{tsl_c} REV:{rev_c}')

    tt = len(df); tw = (df['pnl'] > 0).sum()
    ttp = df[df['pnl'] > 0]['pnl'].sum(); ttl = df[df['pnl'] <= 0]['pnl'].sum()
    tpf = ttp / abs(ttl) if ttl != 0 else 999

    # MDD
    bals = [init_cap] + list(df['balance'])
    peak = init_cap; max_dd = 0
    for b in bals:
        if b > peak: peak = b
        dd = (peak - b) / peak * 100
        if dd > max_dd: max_dd = dd

    print(f'\n  ** 최종: ${df.iloc[-1]["balance"]:,.0f} | +{(df.iloc[-1]["balance"]-init_cap)/init_cap*100:,.0f}% | PF:{tpf:.1f} | MDD:{max_dd:.1f}% | {tt}거래 ({tw}W/{tt-tw}L) | SL:{(df["type"]=="SL").sum()} **')
    print(f'  ** 기획서: Track A=PF27.9/MDD9.0%/18T, Track B=PF9.4/MDD8.8%/51T **')


def main():
    print("="*80, flush=True)
    print("  v26.0 Wilder ADX 정밀 재현 백테스트", flush=True)
    print("="*80, flush=True)

    df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
    mtf = build_mtf(df5)
    df30 = mtf['30m']

    c = df30['close'].values.astype(np.float64)
    h = df30['high'].values.astype(np.float64)
    l = df30['low'].values.astype(np.float64)
    t = df30['time'].values

    # Wilder ADX(14)
    print('\nCalculating Wilder ADX(14)...', flush=True)
    adx = calc_adx_wilder(df30['high'], df30['low'], df30['close'], 14)
    print(f'  ADX range: {np.nanmin(adx):.1f} ~ {np.nanmax(adx):.1f}', flush=True)
    print(f'  ADX mean: {np.nanmean(adx):.1f}', flush=True)
    print(f'  ADX>=35 bars: {np.sum(adx>=35):,}', flush=True)
    print(f'  ADX>=40 bars: {np.sum(adx>=40):,}', flush=True)

    # Compare with ewm ADX
    from bt_fast import calc_adx as calc_adx_ewm
    adx_ewm = calc_adx_ewm(df30['high'], df30['low'], df30['close'], 14).values
    diff = adx - adx_ewm
    print(f'  vs ewm ADX diff: mean={np.nanmean(diff):.2f}, max={np.nanmax(np.abs(diff)):.2f}', flush=True)

    # RSI(14)
    rsi = calc_rsi(df30['close'], 14).values

    # MAs
    ema3 = calc_ema(df30['close'], 3).values
    sma300 = calc_sma(df30['close'], 300).values
    wma3 = calc_wma(df30['close'], 3).values
    ema200 = calc_ema(df30['close'], 200).values

    # Track A: EMA3/SMA300 ADX>=40 RSI30-70 TS+4/-3 M50%
    cfg_a = {'sl': 0.08, 'ta': 0.04, 'tp': 0.03, 'lev': 10, 'mn': 0.50,
             'amin': 40, 'rmin': 30, 'rmax': 70, 'cap': 3000.0}

    # Track B: WMA3/EMA200 ADX>=35 RSI30-70 TS+3/-2 M30%
    cfg_b = {'sl': 0.08, 'ta': 0.03, 'tp': 0.02, 'lev': 10, 'mn': 0.30,
             'amin': 35, 'rmin': 30, 'rmax': 70, 'cap': 3000.0}

    print('\n\n1) Track A (Sniper): EMA3/SMA300, ADX>=40, M50%', flush=True)
    df_a, cap_a = run_v26_track(c, h, l, t, ema3, sma300, adx, rsi, cfg_a, 'Track A')
    print_results(df_a, 'Track A (Sniper)', 3000)

    print('\n\n2) Track B (Compounder): WMA3/EMA200, ADX>=35, M30%', flush=True)
    df_b, cap_b = run_v26_track(c, h, l, t, wma3, ema200, adx, rsi, cfg_b, 'Track B')
    print_results(df_b, 'Track B (Compounder)', 3000)

    # 충돌 분석
    print('\n\n3) 동시 운영 충돌 분석', flush=True)
    if len(df_a) > 0 and len(df_b) > 0:
        conflicts = 0
        for _, ra in df_a.iterrows():
            for _, rb in df_b.iterrows():
                if hasattr(ra.get('entry_time',''), 'year') and hasattr(rb.get('entry_time',''), 'year'):
                    pass  # entry_time not tracked in this version
        print(f'  (진입시간 미추적 - 포지션 시계열로 분석)')

        # Simple: check if both tracks have trades in same month with different directions
        for yr in range(2020, 2027):
            for mo in range(1, 13):
                ta = df_a[(df_a['year'] == yr) & (df_a['month'] == mo)]
                tb = df_b[(df_b['year'] == yr) & (df_b['month'] == mo)]
                if len(ta) > 0 and len(tb) > 0:
                    dirs_a = set(ta['dir'])
                    dirs_b = set(tb['dir'])
                    if dirs_a & dirs_b != dirs_a | dirs_b:  # different directions exist
                        pass  # would need exact time overlap

    print('\n\nDONE.', flush=True)


if __name__ == '__main__':
    main()
