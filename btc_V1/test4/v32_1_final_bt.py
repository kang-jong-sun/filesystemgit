"""v32.1 FINAL 정밀 재현 백테스트 - 의사코드 100% 구현"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from bt_fast import load_5m_data, build_mtf, calc_ema, calc_sma, calc_adx, calc_rsi

def run_v321_final(c, h, l, fast_ma, slow_ma, av, rv, cfg):
    """v32.1 FINAL 의사코드 정확 구현"""
    n = len(c)
    cap = cfg['cap']; pos = 0; epx = 0.0; psz = 0.0; slp = 0.0
    ton = False; thi = 0.0; tlo = 999999.0
    watching = 0; ws = 0; le = 0; ld = 0
    pk = cap; mdd = 0.0; ms = cap
    trades = []

    for i in range(600, n):
        px = c[i]; h_ = h[i]; l_ = l[i]
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(av[i]) or np.isnan(rv[i]):
            continue

        # 일일 리셋 (30분봉 48봉 = 1일)
        if i > 600 and i % 48 == 0:
            ms = cap

        # ═══ STEP A: 포지션 보유 중 ═══
        if pos != 0:
            watching = 0

            # A1: SL (TSL 미활성 시에만)
            if not ton:
                sl_hit = (pos == 1 and l_ <= slp) or (pos == -1 and h_ >= slp)
                if sl_hit:
                    pnl_v = (slp - epx) / epx * psz * pos - psz * 0.0004
                    cap += pnl_v
                    trades.append(_trade(i, epx, slp, pos, pnl_v, 'SL', thi if pos==1 else tlo, cap, cfg))
                    ld = pos; le = i; pos = 0
                    pk = max(pk, cap); dd = (pk-cap)/pk if pk>0 else 0
                    if dd > mdd: mdd = dd
                    continue

            # A2: TA 활성화 (고가/저가 기준)
            if pos == 1:
                br = (h_ - epx) / epx * 100
            else:
                br = (epx - l_) / epx * 100
            if br >= cfg['ta_pct']:
                ton = True

            # A3: TSL (TSL 활성 시)
            if ton:
                if pos == 1:
                    if h_ > thi: thi = h_
                    ns = thi * (1 - cfg['tsl_pct'] / 100)
                    if ns > slp: slp = ns
                    if px <= slp:
                        pnl_v = (px - epx) / epx * psz * pos - psz * 0.0004
                        cap += pnl_v
                        trades.append(_trade(i, epx, px, pos, pnl_v, 'TSL', thi, cap, cfg))
                        ld = pos; le = i; pos = 0
                        pk = max(pk, cap); dd = (pk-cap)/pk if pk>0 else 0
                        if dd > mdd: mdd = dd
                        continue
                else:  # SHORT
                    if l_ < tlo: tlo = l_
                    ns = tlo * (1 + cfg['tsl_pct'] / 100)
                    if ns < slp: slp = ns
                    if px >= slp:
                        pnl_v = (px - epx) / epx * psz * pos - psz * 0.0004
                        cap += pnl_v
                        trades.append(_trade(i, epx, px, pos, pnl_v, 'TSL', tlo, cap, cfg))
                        ld = pos; le = i; pos = 0
                        pk = max(pk, cap); dd = (pk-cap)/pk if pk>0 else 0
                        if dd > mdd: mdd = dd
                        continue

            # A4: REV
            if i > 0:
                bull_now = fast_ma[i] > slow_ma[i]
                bull_prev = fast_ma[i-1] > slow_ma[i-1]
                cross_up = bull_now and not bull_prev
                cross_down = not bull_now and bull_prev
                if (pos == 1 and cross_down) or (pos == -1 and cross_up):
                    pnl_v = (px - epx) / epx * psz * pos - psz * 0.0004
                    cap += pnl_v
                    trades.append(_trade(i, epx, px, pos, pnl_v, 'REV', thi if pos==1 else tlo, cap, cfg))
                    ld = pos; le = i; pos = 0
                    pk = max(pk, cap); dd = (pk-cap)/pk if pk>0 else 0
                    if dd > mdd: mdd = dd
                    # CONTINUE 하지 않음 — 같은 봉에서 진입 가능

        # ═══ STEP B: 진입 체크 ═══
        if i < 1: continue

        bull_now = fast_ma[i] > slow_ma[i]
        bull_prev = fast_ma[i-1] > slow_ma[i-1]
        cross_up = bull_now and not bull_prev
        cross_down = not bull_now and bull_prev

        if pos == 0:
            # B2: 새 크로스 → 감시 시작
            if cross_up: watching = 1; ws = i
            elif cross_down: watching = -1; ws = i

            if watching != 0 and i > ws:
                # B3: 24봉 초과
                if i - ws > 24: watching = 0; continue
                # B4: 반대 크로스
                if watching == 1 and cross_down: watching = -1; ws = i; continue
                elif watching == -1 and cross_up: watching = 1; ws = i; continue
                # B5: 동일방향 스킵
                if cfg.get('skip_same', True) and watching == ld: continue
                # B6: ADX
                if av[i] < cfg['adx_min']: continue
                # B7: ADX 상승
                if cfg.get('adx_rise', 6) > 0 and i >= 6:
                    if av[i] <= av[i - cfg.get('adx_rise', 6)]: continue
                # B8: RSI
                if rv[i] < cfg['rsi_min'] or rv[i] > cfg['rsi_max']: continue
                # B9: EMA 갭
                if cfg.get('gap_min', 0.2) > 0:
                    gap = abs(fast_ma[i] - slow_ma[i]) / slow_ma[i] * 100 if slow_ma[i] > 0 else 0
                    if gap < cfg.get('gap_min', 0.2): continue
                # B10: 일일 손실
                if ms > 0 and (cap - ms) / ms <= -0.20: watching = 0; continue
                # B11: 잔액
                if cap <= 0: continue

                # ═══ 진입! ═══
                mg = cap * cfg['margin']
                psz = mg * cfg['lev']
                cap -= psz * 0.0004  # 진입 수수료
                pos = watching
                epx = px
                ton = False
                thi = px; tlo = px
                if pos == 1:
                    slp = epx * (1 - cfg['sl_pct'] / 100)
                else:
                    slp = epx * (1 + cfg['sl_pct'] / 100)
                pk = max(pk, cap)
                watching = 0

        pk = max(pk, cap)
        dd = (pk - cap) / pk if pk > 0 else 0
        if dd > mdd: mdd = dd
        if cap <= 0: break

    # 미청산
    if pos != 0 and cap > 0:
        px = c[n-1]
        pnl_v = (px - epx) / epx * psz * pos - psz * 0.0004
        cap += pnl_v
        trades.append(_trade(n-1, epx, px, pos, pnl_v, 'END', thi if pos==1 else tlo, cap, cfg))

    return trades, cap, mdd


def _trade(i, epx, exit_px, pos, pnl, etype, peak, bal, cfg):
    return {
        'bar': i, 'entry_price': round(epx, 1), 'exit_price': round(exit_px, 1),
        'direction': 'LONG' if pos == 1 else 'SHORT',
        'change_pct': round((exit_px - epx) / epx * 100 * pos, 2),
        'lev_roi_pct': round((exit_px - epx) / epx * 100 * pos * cfg['lev'], 1),
        'pnl': round(pnl, 0), 'exit_type': etype,
        'peak_pct': round(abs(peak - epx) / epx * 100, 2) if epx > 0 else 0,
        'balance': round(bal, 0),
    }


def main():
    print("="*80, flush=True)
    print("  v32.1 FINAL 정밀 재현 백테스트", flush=True)
    print("  TSL활성→SL비활성, 감시메커니즘, TSL 9%, 종가기준TSL", flush=True)
    print("="*80, flush=True)

    df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
    mtf = build_mtf(df5)
    df30 = mtf['30m']
    c = df30['close'].values.astype(np.float64)
    h = df30['high'].values.astype(np.float64)
    l = df30['low'].values.astype(np.float64)
    t = df30['time'].values

    adx = calc_adx(df30['high'], df30['low'], df30['close'], 20).values
    rsi = calc_rsi(df30['close'], 14).values

    configs = [
        {'name': 'A안 원본', 'desc': 'EMA100/EMA600',
         'fl': 100, 'sl_len': 600, 'sl_type': 'ema',
         'sl_pct': 3.0, 'ta_pct': 12.0, 'tsl_pct': 9.0,
         'lev': 10, 'margin': 0.35, 'cap': 5000.0,
         'adx_min': 30, 'adx_rise': 6, 'rsi_min': 35, 'rsi_max': 75,
         'gap_min': 0.2, 'skip_same': True},
        {'name': 'B안 원본', 'desc': 'EMA75/SMA750',
         'fl': 75, 'sl_len': 750, 'sl_type': 'sma',
         'sl_pct': 3.0, 'ta_pct': 12.0, 'tsl_pct': 9.0,
         'lev': 10, 'margin': 0.35, 'cap': 5000.0,
         'adx_min': 30, 'adx_rise': 6, 'rsi_min': 35, 'rsi_max': 75,
         'gap_min': 0.2, 'skip_same': True},
    ]

    all_sheets = {}

    for cfg in configs:
        print(f"\n{'='*80}", flush=True)
        print(f"  {cfg['name']}: {cfg['desc']}", flush=True)
        print(f"  SL:{cfg['sl_pct']}% TA:{cfg['ta_pct']}% TSL:{cfg['tsl_pct']}% Lev:{cfg['lev']}x M:{cfg['margin']*100:.0f}%", flush=True)
        print(f"{'='*80}", flush=True)

        fast_ma = calc_ema(df30['close'], cfg['fl']).values
        if cfg['sl_type'] == 'sma':
            slow_ma = df30['close'].rolling(cfg['sl_len']).mean().values
        else:
            slow_ma = calc_ema(df30['close'], cfg['sl_len']).values

        trades, final_cap, mdd_v = run_v321_final(c, h, l, fast_ma, slow_ma, adx, rsi, cfg)
        df_t = pd.DataFrame(trades)

        if len(df_t) == 0:
            print("  NO TRADES", flush=True); continue

        # Add timestamps
        df_t['청산시간'] = [pd.Timestamp(t[b]) for b in df_t['bar']]
        df_t['연도'] = df_t['청산시간'].dt.year

        # Print trades
        print(f"\n  건별 상세 ({len(df_t)}건)", flush=True)
        print(f"| # | {'시간':^16} | {'방향':^5} | {'진입가':>10} | {'청산가':>10} | {'변동%':>7} | {'ROI%':>8} | {'손익$':>14} | {'사유':^3} | {'잔액$':>14} |", flush=True)
        print(f"|---|{'-'*18}|{'-'*7}|{'-'*12}|{'-'*12}|{'-'*9}|{'-'*10}|{'-'*16}|{'-'*5}|{'-'*16}|", flush=True)
        for i, r in df_t.iterrows():
            print(f"| {i+1:>1} | {str(r['청산시간'])[:16]:^16} | {r['direction']:^5} | {r['entry_price']:>10,.0f} | {r['exit_price']:>10,.0f} | {r['change_pct']:>+6.1f}% | {r['lev_roi_pct']:>+7.1f}% | {r['pnl']:>+13,.0f} | {r['exit_type']:^3} | {r['balance']:>13,.0f} |", flush=True)

        # Yearly
        print(f"\n  연도별 요약", flush=True)
        for yr in sorted(df_t['연도'].unique()):
            yt = df_t[df_t['연도'] == yr]
            w = (yt['pnl'] > 0).sum(); tc = len(yt)
            tp = yt[yt['pnl'] > 0]['pnl'].sum(); tl = yt[yt['pnl'] <= 0]['pnl'].sum()
            pf = tp / abs(tl) if tl != 0 else 999
            sl_c = (yt['exit_type'] == 'SL').sum(); tsl_c = (yt['exit_type'] == 'TSL').sum()
            rev_c = (yt['exit_type'] == 'REV').sum()
            print(f"  {yr}: {tc}거래 {w}W/{tc-w}L PF:{pf:.2f} 순손익:${tp+tl:+,.0f} SL:{sl_c} TSL:{tsl_c} REV:{rev_c}", flush=True)

        tw = (df_t['pnl'] > 0).sum(); tt = len(df_t)
        ttp = df_t[df_t['pnl'] > 0]['pnl'].sum(); ttl = df_t[df_t['pnl'] <= 0]['pnl'].sum()
        tpf = ttp / abs(ttl) if ttl != 0 else 999

        print(f"\n  ★ 최종: ${final_cap:,.0f} | +{(final_cap-cfg['cap'])/cfg['cap']*100:,.0f}% | PF:{tpf:.2f} | MDD:{mdd_v*100:.1f}% | {tt}거래 ({tw}W/{tt-tw}L)", flush=True)
        print(f"  ★ 기획서: $16,044,549 | +320,791% | PF:5.8 | MDD:56.8% | 70거래", flush=True)

        # Save to sheets
        df_t_out = df_t[['청산시간','direction','entry_price','exit_price','change_pct','lev_roi_pct','pnl','exit_type','peak_pct','balance']].copy()
        df_t_out.columns = ['청산시간','방향','진입가','청산가','가격변동%','레버ROI%','손익$','청산사유','최고수익%','잔액$']
        all_sheets[cfg['name']] = df_t_out

    # Excel
    path = r'D:\filesystem\futures\btc_V1\test4\v32_1_FINAL_backtest.xlsx'
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for sn, df in all_sheets.items():
            df.to_excel(writer, sheet_name=sn[:31], index=False)
    print(f"\nExcel: {path}", flush=True)
    print("DONE.", flush=True)

if __name__ == '__main__':
    main()
