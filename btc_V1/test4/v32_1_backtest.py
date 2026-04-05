"""v32.1 정밀 백테스트 - 5분봉 기반, 특수 조건 완전 구현"""
import pandas as pd, numpy as np, time, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

from v27_backtest_engine import load_5m_data, resample_ohlcv, calc_ema, calc_sma, calc_adx, calc_rsi

def main():
    print("="*70, flush=True)
    print("v32.1 PRECISE BACKTEST - 5m data, all special conditions", flush=True)
    print("="*70, flush=True)

    df_5m = load_5m_data()
    df_30m = resample_ohlcv(df_5m, 30)
    print(f"30m bars: {len(df_30m):,}", flush=True)

    c30 = df_30m['close']; h30 = df_30m['high']; l30 = df_30m['low']

    configs = [
        {'name':'A안 EMA100/EMA600', 'fast':('ema',100), 'slow':('ema',600)},
        {'name':'B안 EMA75/SMA750',  'fast':('ema',75),  'slow':('sma',750)},
    ]

    for cfg in configs:
        print(f"\n{'='*100}", flush=True)
        print(f"  {cfg['name']}", flush=True)
        print(f"{'='*100}", flush=True)

        # Calc MAs
        ft, fl = cfg['fast']
        st, sl_len = cfg['slow']
        fast_ma = c30.ewm(span=fl, adjust=False).mean() if ft=='ema' else c30.rolling(fl).mean()
        slow_ma = c30.ewm(span=sl_len, adjust=False).mean() if st=='ema' else c30.rolling(sl_len).mean()

        # ADX(20) on 30m
        adx = calc_adx(h30, l30, c30, 20)
        # RSI(14) on 30m
        rsi = calc_rsi(c30, 14)

        # Run backtest on 30m bars
        n = len(df_30m)
        capital = 5000.0; lev = 10; margin_pct = 0.35; fee_rate = 0.0004
        position = 0; entry_price = 0.0; margin_used = 0.0
        peak_price = 0.0; tsl_active = False; last_dir = 0
        peak_capital = capital; daily_pnl = 0.0; last_day = None

        trades = []

        for i in range(max(sl_len+10, 700), n):
            price = c30.iloc[i]; hi = h30.iloc[i]; lo = l30.iloc[i]
            ts = df_30m['timestamp'].iloc[i]
            day = ts.date() if hasattr(ts, 'date') else pd.Timestamp(ts).date()

            if last_day is not None and day != last_day:
                daily_pnl = 0.0
            last_day = day

            fast_v = fast_ma.iloc[i]; slow_v = slow_ma.iloc[i]
            adx_v = adx.iloc[i]; rsi_v = rsi.iloc[i]

            if pd.isna(fast_v) or pd.isna(slow_v) or pd.isna(adx_v): continue

            # Cross signal
            cross_now = 1 if fast_v > slow_v else -1 if fast_v < slow_v else 0

            # EMA gap filter: |fast - slow| / slow >= 0.2%
            ema_gap = abs(fast_v - slow_v) / slow_v if slow_v > 0 else 0

            # ADX rise: ADX now > ADX 6 bars ago
            adx_6ago = adx.iloc[i-6] if i >= 6 else 0
            adx_rising = adx_v > adx_6ago if not pd.isna(adx_6ago) else False

            # ---- POSITION MANAGEMENT ----
            if position != 0:
                if position == 1:
                    price_change = (price - entry_price) / entry_price
                    hi_change = (hi - entry_price) / entry_price
                    lo_change = (lo - entry_price) / entry_price
                else:
                    price_change = (entry_price - price) / entry_price
                    hi_change = (entry_price - lo) / entry_price
                    lo_change = (entry_price - hi) / entry_price

                if hi_change > peak_price: peak_price = hi_change

                exit_type = None

                # 1. SL -3%
                if lo_change <= -0.03:
                    exit_type = 'SL'
                    actual_change = -0.03

                # 2. TSL +12%/-10%
                if exit_type is None:
                    if peak_price >= 0.12:
                        tsl_active = True
                    if tsl_active and price_change <= peak_price - 0.10:
                        exit_type = 'TSL'
                        actual_change = peak_price - 0.10

                # 3. REV
                if exit_type is None:
                    if cross_now != 0 and cross_now != position:
                        exit_type = 'REV'
                        actual_change = price_change

                if exit_type:
                    pnl = margin_used * lev * actual_change
                    fee = margin_used * lev * fee_rate
                    capital += pnl - fee
                    if capital < 0: capital = 0
                    daily_pnl += pnl - fee
                    if capital > peak_capital: peak_capital = capital

                    trades.append({
                        'time': pd.Timestamp(ts),
                        'direction': 'LONG' if position == 1 else 'SHORT',
                        'entry_price': entry_price, 'exit_price': price,
                        'change_pct': actual_change * 100,
                        'lev_roi_pct': actual_change * lev * 100,
                        'pnl': pnl - fee, 'exit_type': exit_type,
                        'peak_pct': peak_price * 100, 'balance': capital,
                        'year': pd.Timestamp(ts).year, 'month': pd.Timestamp(ts).month,
                    })

                    last_dir = position
                    position = 0; entry_price = 0; margin_used = 0
                    peak_price = 0; tsl_active = False

            # ---- ENTRY ----
            if position == 0 and capital > 0 and cross_now != 0:
                # 동일방향 스킵
                if cross_now == last_dir:
                    continue

                # 일일 손실 제한
                if daily_pnl / (capital + abs(daily_pnl)) < -0.20:
                    continue

                # Filter conditions
                adx_ok = adx_v >= 30
                adx_rise_ok = adx_rising
                rsi_ok = 35 <= rsi_v <= 75
                gap_ok = ema_gap >= 0.002

                if adx_ok and adx_rise_ok and rsi_ok and gap_ok:
                    position = cross_now
                    entry_price = price
                    margin_used = capital * margin_pct
                    fee_entry = margin_used * lev * fee_rate
                    capital -= fee_entry
                    margin_used = capital * margin_pct
                    peak_price = 0; tsl_active = False

        # Close open
        if position != 0 and capital > 0:
            price = c30.iloc[n-1]
            if position == 1: pc = (price - entry_price) / entry_price
            else: pc = (entry_price - price) / entry_price
            pnl = margin_used * lev * pc
            fee = margin_used * lev * fee_rate
            capital += pnl - fee
            trades.append({'time':pd.Timestamp(df_30m['timestamp'].iloc[n-1]),
                'direction':'LONG' if position==1 else 'SHORT',
                'entry_price':entry_price,'exit_price':price,
                'change_pct':pc*100,'lev_roi_pct':pc*lev*100,
                'pnl':pnl-fee,'exit_type':'END','peak_pct':peak_price*100,
                'balance':capital,'year':2026,'month':3})

        # Print trades
        df = pd.DataFrame(trades)
        if len(df) == 0:
            print("  NO TRADES", flush=True); continue

        print(f"\n  전체 거래 ({len(df)}건)", flush=True)
        print(f"| {'#':>3} | {'시간':^16} | {'방향':^5} | {'진입가':>10} | {'청산가':>10} | {'변동%':>7} | {'ROI%':>8} | {'손익$':>14} | {'사유':^3} | {'잔액$':>14} |", flush=True)
        print(f"|{'-'*5}|{'-'*18}|{'-'*7}|{'-'*12}|{'-'*12}|{'-'*9}|{'-'*10}|{'-'*16}|{'-'*5}|{'-'*16}|", flush=True)
        for i, t in df.iterrows():
            print(f"| {i+1:>3} | {str(t['time'])[:16]:^16} | {t['direction']:^5} | {t['entry_price']:>10,.0f} | {t['exit_price']:>10,.0f} | {t['change_pct']:>+6.1f}% | {t['lev_roi_pct']:>+7.1f}% | {t['pnl']:>+13,.0f} | {t['exit_type']:^3} | {t['balance']:>13,.0f} |", flush=True)

        # Yearly summary
        yearly = {}
        for _, t in df.iterrows():
            yr = t['year']
            if yr not in yearly: yearly[yr] = {'t':0,'w':0,'p':0,'l':0,'sl':0,'tsl':0,'rev':0}
            yearly[yr]['t'] += 1
            if t['pnl'] > 0: yearly[yr]['w'] += 1; yearly[yr]['p'] += t['pnl']
            else: yearly[yr]['l'] += t['pnl']
            if t['exit_type'] == 'SL': yearly[yr]['sl'] += 1
            elif t['exit_type'] == 'TSL': yearly[yr]['tsl'] += 1
            elif t['exit_type'] == 'REV': yearly[yr]['rev'] += 1

        print(f"\n  연도별 요약", flush=True)
        print(f"| {'연도':^6} | {'거래':^4} | {'승':^3} | {'패':^3} | {'총이익$':>14} | {'총손실$':>14} | {'순손익$':>14} | {'PF':>6} | {'SL':^3} | {'TSL':^4} | {'REV':^4} |", flush=True)
        print(f"|{'-'*8}|{'-'*6}|{'-'*5}|{'-'*5}|{'-'*16}|{'-'*16}|{'-'*16}|{'-'*8}|{'-'*5}|{'-'*6}|{'-'*6}|", flush=True)
        for yr in sorted(yearly.keys()):
            y = yearly[yr]; tc = y['t']; w = y['w']; l = tc - w
            net = y['p'] + y['l']; pf = y['p'] / abs(y['l']) if y['l'] != 0 else 999
            pfs = f"{pf:.2f}" if pf < 100 else "INF"
            print(f"| {yr:^6} | {tc:^4} | {w:^3} | {l:^3} | {y['p']:>+13,.0f} | {y['l']:>+13,.0f} | {net:>+13,.0f} | {pfs:>6} | {y['sl']:^3} | {y['tsl']:^4} | {y['rev']:^4} |", flush=True)

        tt = len(df); tw = len(df[df['pnl']>0])
        ttp = df[df['pnl']>0]['pnl'].sum(); ttl = df[df['pnl']<=0]['pnl'].sum()
        tpf = ttp/abs(ttl) if ttl!=0 else 999
        print(f"|{'─'*8}|{'─'*6}|{'─'*5}|{'─'*5}|{'─'*16}|{'─'*16}|{'─'*16}|{'─'*8}|{'─'*5}|{'─'*6}|{'─'*6}|", flush=True)
        print(f"| {'합계':^5} | {tt:^4} | {tw:^3} | {tt-tw:^3} | {ttp:>+13,.0f} | {ttl:>+13,.0f} | {ttp+ttl:>+13,.0f} | {tpf:.2f} | | | |", flush=True)

        mdd = (capital - peak_capital) / peak_capital * 100 if peak_capital > 0 else 0
        # Calc true MDD from equity
        bal_series = [5000] + [t['balance'] for _, t in df.iterrows()]
        peak = 5000; max_dd = 0
        for b in bal_series:
            if b > peak: peak = b
            dd = (b - peak) / peak * 100
            if dd < max_dd: max_dd = dd

        print(f"\n  최종잔액: ${capital:,.0f} | 수익률: +{(capital-5000)/5000*100:,.0f}% | PF: {tpf:.2f} | MDD: {abs(max_dd):.1f}% | 거래 {tt}회", flush=True)

    print("\nDONE.", flush=True)

if __name__ == '__main__':
    main()
