"""v32.1 건별 상세 엑셀 생성"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from bt_fast import load_5m_data, build_mtf, calc_ema, calc_sma, calc_adx, calc_rsi
import numpy as np, pandas as pd

def run_detail(c30, h30, l30, t30, cfg):
    n = len(c30)
    fast_ma = calc_ema(pd.Series(c30), cfg['fl']).values
    if cfg['st'] == 'ema':
        slow_ma = calc_ema(pd.Series(c30), cfg['sl_len']).values
    else:
        slow_ma = pd.Series(c30).rolling(cfg['sl_len']).mean().values
    adx = calc_adx(pd.Series(h30), pd.Series(l30), pd.Series(c30), 20).values
    rsi = calc_rsi(pd.Series(c30), 14).values

    lev = cfg['lev']; fee = 0.0004; capital = float(cfg['cap'])
    mn = cfg['mn']; mr = mn * 0.5
    position = 0; ep = 0.0; margin_used = 0.0; peak_pnl = 0.0; trail = False
    peak_cap = capital; entry_bar = 0
    trades = []

    start = max(cfg['sl_len'] + 10, 700)
    for i in range(start, n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            continue
        price = c30[i]; hi = h30[i]; lo = l30[i]

        if position != 0:
            if position == 1:
                pnl = (price - ep) / ep; pk = (hi - ep) / ep; lw = (lo - ep) / ep
            else:
                pnl = (ep - price) / ep; pk = (ep - lo) / ep; lw = (ep - hi) / ep
            if pk > peak_pnl:
                peak_pnl = pk

            exit_type = None; exit_pnl = 0

            if lw <= -cfg['sl']:
                exit_type = 'SL'; exit_pnl = -cfg['sl']

            if exit_type is None and peak_pnl >= cfg['ta']:
                trail = True
            if trail and exit_type is None:
                tl = peak_pnl - cfg['tp']
                if pnl <= tl:
                    exit_type = 'TSL'; exit_pnl = tl

            if exit_type is None:
                cu = fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]
                cd = fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]
                ao = adx[i] >= cfg['amin']
                ro = cfg['rmin'] <= rsi[i] <= cfg['rmax']
                if (position == 1 and cd and ao and ro) or (position == -1 and cu and ao and ro):
                    exit_type = 'REV'; exit_pnl = pnl

            if exit_type:
                dollar_pnl = margin_used * exit_pnl - margin_used * fee
                capital += dollar_pnl
                if capital < 0:
                    capital = 0
                if capital > peak_cap:
                    peak_cap = capital
                dd = (peak_cap - capital) / peak_cap * 100 if peak_cap > 0 else 0

                if position == 1:
                    exit_price = ep * (1 + exit_pnl)
                else:
                    exit_price = ep * (1 - exit_pnl)

                trades.append({
                    '번호': len(trades) + 1,
                    '진입시간': pd.Timestamp(t30[entry_bar]),
                    '청산시간': pd.Timestamp(t30[i]),
                    '보유기간': str(pd.Timestamp(t30[i]) - pd.Timestamp(t30[entry_bar])),
                    '방향': 'LONG' if position == 1 else 'SHORT',
                    '진입가($)': round(ep, 1),
                    '청산가($)': round(exit_price, 1),
                    '가격변동(%)': round(exit_pnl * 100, 2),
                    '레버리지ROI(%)': round(exit_pnl * lev * 100, 1),
                    '투입마진($)': round(margin_used / lev, 0),
                    '포지션크기($)': round(margin_used, 0),
                    '손익($)': round(dollar_pnl, 0),
                    '청산사유': exit_type,
                    '최고수익(%)': round(peak_pnl * 100, 2),
                    '잔액($)': round(capital, 0),
                    'DD(%)': round(dd, 1),
                    '누적수익률(%)': round((capital - cfg['cap']) / cfg['cap'] * 100, 1),
                })

                new_dir = 0
                if exit_type == 'REV':
                    if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
                        new_dir = 1
                    elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
                        new_dir = -1

                if exit_type == 'REV' and new_dir != 0 and capital > 10:
                    mg = mn if (peak_cap - capital) / peak_cap < 0.25 else mr
                    position = new_dir; ep = price; entry_bar = i
                    margin_used = capital * mg * lev
                    capital -= margin_used * fee
                    margin_used = capital * mg * lev
                    peak_pnl = 0; trail = False
                else:
                    position = 0; ep = 0; margin_used = 0; peak_pnl = 0; trail = False

        if position == 0 and capital > 10:
            cu = fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]
            cd = fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]
            ao = adx[i] >= cfg['amin']
            ro = cfg['rmin'] <= rsi[i] <= cfg['rmax']
            sig = 0
            if cu and ao and ro: sig = 1
            elif cd and ao and ro: sig = -1
            if sig != 0:
                mg = mn if (peak_cap - capital) / peak_cap < 0.25 else mr
                position = sig; ep = price; entry_bar = i
                margin_used = capital * mg * lev
                capital -= margin_used * fee
                margin_used = capital * mg * lev
                peak_pnl = 0; trail = False

    if position != 0 and capital > 10:
        price = c30[n - 1]
        if position == 1: pnl_f = (price - ep) / ep
        else: pnl_f = (ep - price) / ep
        dollar_pnl = margin_used * pnl_f - margin_used * fee
        capital += dollar_pnl
        trades.append({
            '번호': len(trades) + 1, '진입시간': pd.Timestamp(t30[entry_bar]),
            '청산시간': pd.Timestamp(t30[n-1]), '보유기간': str(pd.Timestamp(t30[n-1]) - pd.Timestamp(t30[entry_bar])),
            '방향': 'LONG' if position == 1 else 'SHORT',
            '진입가($)': round(ep, 1), '청산가($)': round(price, 1),
            '가격변동(%)': round(pnl_f * 100, 2), '레버리지ROI(%)': round(pnl_f * lev * 100, 1),
            '투입마진($)': round(margin_used / lev, 0), '포지션크기($)': round(margin_used, 0),
            '손익($)': round(dollar_pnl, 0), '청산사유': 'END',
            '최고수익(%)': round(peak_pnl * 100, 2), '잔액($)': round(capital, 0),
            'DD(%)': 0, '누적수익률(%)': round((capital - cfg['cap']) / cfg['cap'] * 100, 1),
        })

    return pd.DataFrame(trades), capital


def main():
    df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
    mtf = build_mtf(df5)
    df30 = mtf['30m']
    c30 = df30['close'].values.astype(np.float64)
    h30 = df30['high'].values.astype(np.float64)
    l30 = df30['low'].values.astype(np.float64)
    t30 = df30['time'].values

    configs = [
        {'name': 'A안_원본', 'desc': 'EMA100/EMA600 SL3% TA12/TP10 10x M35%',
         'ft': 'ema', 'fl': 100, 'st': 'ema', 'sl_len': 600,
         'sl': 0.03, 'ta': 0.12, 'tp': 0.10, 'lev': 10, 'mn': 0.35,
         'amin': 30, 'rmin': 35, 'rmax': 75, 'cap': 5000},
        {'name': 'B안_원본', 'desc': 'EMA75/SMA750 SL3% TA12/TP10 10x M35%',
         'ft': 'ema', 'fl': 75, 'st': 'sma', 'sl_len': 750,
         'sl': 0.03, 'ta': 0.12, 'tp': 0.10, 'lev': 10, 'mn': 0.35,
         'amin': 30, 'rmin': 35, 'rmax': 75, 'cap': 5000},
        {'name': 'A안_SL7%', 'desc': 'EMA100/EMA600 SL7% TA15/TP10 10x M40%',
         'ft': 'ema', 'fl': 100, 'st': 'ema', 'sl_len': 600,
         'sl': 0.07, 'ta': 0.15, 'tp': 0.10, 'lev': 10, 'mn': 0.40,
         'amin': 25, 'rmin': 40, 'rmax': 75, 'cap': 5000},
        {'name': 'A안_SL7%_15x', 'desc': 'EMA100/EMA600 SL7% TA15/TP10 15x M30%',
         'ft': 'ema', 'fl': 100, 'st': 'ema', 'sl_len': 600,
         'sl': 0.07, 'ta': 0.15, 'tp': 0.10, 'lev': 15, 'mn': 0.30,
         'amin': 25, 'rmin': 40, 'rmax': 75, 'cap': 5000},
        {'name': '컨셉최적_EMA3_200', 'desc': 'EMA3/EMA200 SL7% TA6/TP3 15x M35%',
         'ft': 'ema', 'fl': 3, 'st': 'ema', 'sl_len': 200,
         'sl': 0.07, 'ta': 0.06, 'tp': 0.03, 'lev': 15, 'mn': 0.35,
         'amin': 30, 'rmin': 30, 'rmax': 70, 'cap': 5000},
    ]

    all_sheets = {}

    for cfg in configs:
        print(f"\nProcessing: {cfg['name']} ({cfg['desc']})...", flush=True)
        df_trades, final_cap = run_detail(c30, h30, l30, t30, cfg)
        print(f"  {len(df_trades)}건 | ${final_cap:,.0f} | +{(final_cap-cfg['cap'])/cfg['cap']*100:,.0f}%", flush=True)

        all_sheets[cfg['name']] = df_trades

        if len(df_trades) > 0:
            # Monthly
            df_trades['연월'] = df_trades['청산시간'].dt.to_period('M').astype(str)
            monthly = df_trades.groupby('연월').agg(
                거래수=('번호', 'count'),
                승=('손익($)', lambda x: (x > 0).sum()),
                패=('손익($)', lambda x: (x <= 0).sum()),
                총이익=('손익($)', lambda x: x[x > 0].sum()),
                총손실=('손익($)', lambda x: x[x <= 0].sum()),
                순손익=('손익($)', 'sum'),
                최종잔액=('잔액($)', 'last'),
            ).reset_index()
            monthly['PF'] = monthly.apply(
                lambda r: round(r['총이익'] / abs(r['총손실']), 2) if r['총손실'] != 0 else 999, axis=1)
            monthly['승률(%)'] = round(monthly['승'] / monthly['거래수'] * 100, 1)
            all_sheets[cfg['name'][:20] + '_월별'] = monthly

            # Yearly
            yearly = df_trades.groupby(df_trades['청산시간'].dt.year).agg(
                거래수=('번호', 'count'),
                승=('손익($)', lambda x: (x > 0).sum()),
                패=('손익($)', lambda x: (x <= 0).sum()),
                총이익=('손익($)', lambda x: x[x > 0].sum()),
                총손실=('손익($)', lambda x: x[x <= 0].sum()),
                순손익=('손익($)', 'sum'),
                SL=('청산사유', lambda x: (x == 'SL').sum()),
                TSL=('청산사유', lambda x: (x == 'TSL').sum()),
                REV=('청산사유', lambda x: (x == 'REV').sum()),
                최종잔액=('잔액($)', 'last'),
            ).reset_index()
            yearly.columns = ['연도', '거래수', '승', '패', '총이익($)', '총손실($)', '순손익($)',
                              'SL', 'TSL', 'REV', '최종잔액($)']
            yearly['PF'] = yearly.apply(
                lambda r: round(r['총이익($)'] / abs(r['총손실($)']), 2) if r['총손실($)'] != 0 else 999, axis=1)
            yearly['승률(%)'] = round(yearly['승'] / yearly['거래수'] * 100, 1)
            all_sheets[cfg['name'][:20] + '_연도별'] = yearly

    # Summary sheet
    summary_rows = []
    for cfg in configs:
        df_t = all_sheets.get(cfg['name'])
        if df_t is not None and len(df_t) > 0:
            fc = df_t.iloc[-1]['잔액($)']
            wins = (df_t['손익($)'] > 0).sum()
            tc = len(df_t)
            tp_sum = df_t[df_t['손익($)'] > 0]['손익($)'].sum()
            tl_sum = df_t[df_t['손익($)'] <= 0]['손익($)'].sum()
            pf = round(tp_sum / abs(tl_sum), 2) if tl_sum != 0 else 999
            summary_rows.append({
                '모델': cfg['name'],
                '설명': cfg['desc'],
                '최종잔액($)': fc,
                '수익률(%)': round((fc - cfg['cap']) / cfg['cap'] * 100, 1),
                'PF': pf,
                '거래수': tc,
                '승률(%)': round(wins / tc * 100, 1),
                'SL': (df_t['청산사유'] == 'SL').sum(),
                'TSL': (df_t['청산사유'] == 'TSL').sum(),
                'REV': (df_t['청산사유'] == 'REV').sum(),
            })
    summary_df = pd.DataFrame(summary_rows)
    all_sheets = {'요약': summary_df, **all_sheets}

    # Save
    path = r'D:\filesystem\futures\btc_V1\test4\v32_1_detailed.xlsx'
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for sname, df in all_sheets.items():
            df.to_excel(writer, sheet_name=sname[:31], index=False)

    print(f"\nSaved: {path}", flush=True)
    print("DONE.", flush=True)


if __name__ == '__main__':
    main()
