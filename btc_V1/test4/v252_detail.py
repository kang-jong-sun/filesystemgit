"""v25.2 Multi-TF Trade Detail Extractor
   크로스TF, ADX TF, RSI TF가 각각 다른 멀티 타임프레임 백테스트 상세 추출"""
import numpy as np, pandas as pd, csv, json, sys
from bt_fast import (load_5m_data, build_mtf, calc_ema, calc_rsi, calc_adx, calc_hma, calc_ma)

def run_detail_mtf(mtf, cross_tf, adx_tf, rsi_tf, cfg):
    """멀티TF 지표를 사용하는 거래 상세 추출"""
    df_cross = mtf[cross_tf]
    c = df_cross['close']; h = df_cross['high']; l = df_cross['low']

    # Cross TF에서 MA 계산
    ft = cfg.get('ma_fast_type','ema')
    fp = cfg.get('ma_fast',3)
    st = cfg.get('ma_slow_type','ema')
    sp = cfg.get('ma_slow',200)

    ma_fast = calc_ma(c, ft, fp, df_cross['volume']).values.astype(np.float64)
    ma_slow = calc_ma(c, st, sp, df_cross['volume']).values.astype(np.float64)

    # ADX TF에서 ADX 계산
    df_adx = mtf[adx_tf]
    adx_arr = calc_adx(df_adx['high'], df_adx['low'], df_adx['close'],
                       cfg.get('adx_period',20)).values.astype(np.float64)

    # RSI TF에서 RSI 계산
    df_rsi = mtf[rsi_tf]
    rsi_arr = calc_rsi(df_rsi['close'], cfg.get('rsi_period',14)).values.astype(np.float64)

    # TF 매핑: cross_tf 시간 -> adx_tf/rsi_tf 인덱스
    ct = df_cross['time'].values.astype('int64')
    at = df_adx['time'].values.astype('int64')
    rt = df_rsi['time'].values.astype('int64')
    amap = np.clip(np.searchsorted(at, ct, side='right') - 1, 0, len(df_adx)-1)
    rmap = np.clip(np.searchsorted(rt, ct, side='right') - 1, 0, len(df_rsi)-1)

    closes = c.values.astype(np.float64)
    highs = h.values.astype(np.float64)
    lows = l.values.astype(np.float64)
    n = len(df_cross)
    months = (df_cross['time'].dt.year*100 + df_cross['time'].dt.month).values
    times = df_cross['time'].values

    adx_min = cfg.get('adx_min', 35.0)
    rsi_min = cfg.get('rsi_min', 40.0)
    rsi_max = cfg.get('rsi_max', 75.0)
    sl_pct = cfg.get('sl_pct', 0.07)
    trail_act = cfg.get('trail_activate', 0.07)
    trail_pct_v = cfg.get('trail_pct', 0.03)
    leverage = cfg.get('leverage', 10)
    margin_normal = cfg.get('margin_normal', 0.50)
    margin_reduced = cfg.get('margin_reduced', 0.25)
    dd_thresh = cfg.get('dd_threshold', -0.25)
    ml_limit = cfg.get('monthly_loss_limit', -0.20)
    fee = cfg.get('fee_rate', 0.0004)
    init_cap = cfg.get('initial_capital', 3000.0)
    liq_d = 1.0 / leverage

    bal = init_cap; peak_bal = bal; pos = 0; ep = 0.0; ei = 0
    su = 0.0; ppnl = 0.0; trail = False; rem = 1.0
    msb = bal; cm = 0; mp = False; cl = 0; pu_idx = 0
    rpeak = init_cap; mdd = 0.0

    trades = []

    def ts(idx):
        return str(pd.Timestamp(times[idx]))[:19]

    def close_trade(exit_i, exit_px, roi, pnl_usd, reason, peak_r):
        trades.append({
            'entry_time': ts(ei), 'exit_time': ts(exit_i),
            'dir': 'long' if pos == 1 else 'short',
            'entry_px': round(ep, 1), 'exit_px': round(exit_px, 1),
            'roi_pct': round(roi * 100, 2),
            'pnl_usd': round(pnl_usd, 0),
            'reason': reason,
            'peak_roi_pct': round(peak_r * 100, 2),
            'bal_after': round(bal, 0),
            'margin_used': round(su / leverage, 0),
        })

    for i in range(1, n):
        cp = closes[i]; hp = highs[i]; lp = lows[i]
        if np.isnan(ma_fast[i]) or np.isnan(ma_slow[i]): continue
        ai = amap[i]; ri = rmap[i]
        if np.isnan(adx_arr[ai]) or np.isnan(rsi_arr[ri]): continue
        av = adx_arr[ai]; rv = rsi_arr[ri]

        mk = months[i]
        if mk != cm:
            if cm != 0 and bal < msb: pass
            cm = mk; msb = bal; mp = False

        if pos != 0:
            if pos == 1:
                pnl = (cp - ep) / ep; pkc = (hp - ep) / ep; lwc = (lp - ep) / ep
            else:
                pnl = (ep - cp) / ep; pkc = (ep - lp) / ep; lwc = (ep - hp) / ep
            if pkc > ppnl: ppnl = pkc

            # Liquidation
            if lwc <= -liq_d:
                pu2 = su * rem * (-liq_d) - su * rem * fee; bal += pu2
                if bal < 0: bal = 0
                close_trade(i, cp, -liq_d, pu2, 'LIQ', ppnl)
                pos = 0; cl += 1
                if bal > rpeak: rpeak = bal
                dd = (rpeak - bal) / rpeak if rpeak > 0 else 0
                if dd > mdd: mdd = dd
                continue

            # SL
            if lwc <= -sl_pct:
                pu2 = su * rem * (-sl_pct) - su * rem * fee; bal += pu2
                if bal < 0: bal = 0
                close_trade(i, cp, -sl_pct, pu2, 'SL', ppnl)
                pos = 0; cl += 1
                if bal > rpeak: rpeak = bal
                dd = (rpeak - bal) / rpeak if rpeak > 0 else 0
                if dd > mdd: mdd = dd
                continue

            # Trail
            if ppnl >= trail_act: trail = True
            if trail:
                tl = ppnl - trail_pct_v
                if pnl <= tl:
                    pu2 = su * rem * tl - su * rem * fee; bal += pu2
                    close_trade(i, cp, tl, pu2, 'TSL', ppnl)
                    pos = 0
                    if tl > 0: cl = 0
                    else: cl += 1
                    if bal > rpeak: rpeak = bal
                    dd = (rpeak - bal) / rpeak if rpeak > 0 else 0
                    if dd > mdd: mdd = dd
                    continue

            # Reversal
            cu = ma_fast[i] > ma_slow[i] and ma_fast[i-1] <= ma_slow[i-1]
            cd = ma_fast[i] < ma_slow[i] and ma_fast[i-1] >= ma_slow[i-1]
            ao = av >= adx_min
            ro = rsi_min <= rv <= rsi_max
            rv_flag = False; nd = 0
            if pos == 1 and cd and ao and ro: rv_flag = True; nd = -1
            elif pos == -1 and cu and ao and ro: rv_flag = True; nd = 1
            if rv_flag:
                pu2 = su * rem * pnl - su * rem * fee; bal += pu2
                close_trade(i, cp, pnl, pu2, 'REV', ppnl)
                pos = 0
                if pnl > 0: cl = 0
                else: cl += 1
                if bal > rpeak: rpeak = bal
                dd = (rpeak - bal) / rpeak if rpeak > 0 else 0
                if dd > mdd: mdd = dd
                # Immediate re-entry
                can = not mp
                if bal > 10 and can:
                    mg = margin_normal
                    if peak_bal > 0 and dd_thresh < 0:
                        dn = (peak_bal - bal) / peak_bal
                        if dn > abs(dd_thresh): mg = margin_reduced
                    mu = bal * mg; s2 = mu * leverage
                    bal -= s2 * fee; pos = nd; ep = cp; ei = i; su = s2; ppnl = 0.0; trail = False; rem = 1.0
                    if bal > peak_bal: peak_bal = bal
                continue

        # Entry
        if pos == 0 and bal > 10:
            if ml_limit < 0 and msb > 0:
                mr = (bal - msb) / msb
                if mr < ml_limit: mp = True
            can = not mp
            if not can: continue
            cu = ma_fast[i] > ma_slow[i] and ma_fast[i-1] <= ma_slow[i-1]
            cd = ma_fast[i] < ma_slow[i] and ma_fast[i-1] >= ma_slow[i-1]
            ao = av >= adx_min
            ro = rsi_min <= rv <= rsi_max
            sig = 0
            if cu and ao and ro: sig = 1
            elif cd and ao and ro: sig = -1
            if sig != 0:
                mg = margin_normal
                if peak_bal > 0 and dd_thresh < 0:
                    dn = (peak_bal - bal) / peak_bal
                    if dn > abs(dd_thresh): mg = margin_reduced
                mu = bal * mg; s2 = mu * leverage
                bal -= s2 * fee; pos = sig; ep = cp; ei = i; su = s2; ppnl = 0.0; trail = False; rem = 1.0
                if bal > peak_bal: peak_bal = bal

        if bal > peak_bal: peak_bal = bal
        if bal > rpeak: rpeak = bal

    # Close open position
    if pos != 0 and n > 0:
        pf_v = 0.0
        if pos == 1: pf_v = (closes[n-1] - ep) / ep
        else: pf_v = (ep - closes[n-1]) / ep
        pu2 = su * rem * pf_v - su * rem * fee; bal += pu2
        close_trade(n-1, closes[n-1], pf_v, pu2, 'END', ppnl)

    return trades, round(bal, 0)


def monthly_summary(trades):
    """거래 목록에서 월별 요약 생성"""
    monthly = {}
    for t in trades:
        ym = t['entry_time'][:7]
        if ym not in monthly:
            monthly[ym] = {'cnt':0,'wins':0,'losses':0,'pnl':0,'gross_win':0,'gross_loss':0}
        monthly[ym]['cnt'] += 1
        monthly[ym]['pnl'] += t['pnl_usd']
        if t['pnl_usd'] > 0:
            monthly[ym]['wins'] += 1
            monthly[ym]['gross_win'] += t['pnl_usd']
        else:
            monthly[ym]['losses'] += 1
            monthly[ym]['gross_loss'] += t['pnl_usd']
    return monthly


if __name__ == '__main__':
    print("=== v25.2 Multi-TF Trade Detail Extractor ===")
    df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
    mtf = build_mtf(df5)

    models = {
        'A': {
            'name': 'Model A (수익극대형)',
            'cross_tf': '30m', 'adx_tf': '15m', 'rsi_tf': '5m',
            'cfg': {
                'ma_fast_type':'ema','ma_fast':3,'ma_slow_type':'ema','ma_slow':200,
                'adx_period':20,'adx_min':35,'rsi_period':14,'rsi_min':40,'rsi_max':75,
                'sl_pct':0.07,'trail_activate':0.07,'trail_pct':0.03,
                'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,
                'dd_threshold':-0.20,'monthly_loss_limit':-0.20,
                'fee_rate':0.0004,'initial_capital':3000.0
            }
        },
        'B': {
            'name': 'Model B (안정형)',
            'cross_tf': '30m', 'adx_tf': '15m', 'rsi_tf': '10m',
            'cfg': {
                'ma_fast_type':'ema','ma_fast':7,'ma_slow_type':'ema','ma_slow':100,
                'adx_period':20,'adx_min':35,'rsi_period':14,'rsi_min':40,'rsi_max':75,
                'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
                'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,
                'dd_threshold':-0.20,'monthly_loss_limit':-0.20,
                'fee_rate':0.0004,'initial_capital':3000.0
            }
        },
        'C': {
            'name': 'Model C (균형형)',
            'cross_tf': '10m', 'adx_tf': '15m', 'rsi_tf': '5m',
            'cfg': {
                'ma_fast_type':'ema','ma_fast':5,'ma_slow_type':'ema','ma_slow':300,
                'adx_period':20,'adx_min':35,'rsi_period':14,'rsi_min':40,'rsi_max':75,
                'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.03,
                'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,
                'dd_threshold':-0.20,'monthly_loss_limit':-0.20,
                'fee_rate':0.0004,'initial_capital':3000.0
            }
        },
    }

    all_results = {}

    for key, model in models.items():
        print(f"\n{'='*80}")
        print(f"  {model['name']}")
        print(f"  Cross: {model['cross_tf']} | ADX: {model['adx_tf']} | RSI: {model['rsi_tf']}")
        print(f"{'='*80}")

        trades, final_bal = run_detail_mtf(mtf, model['cross_tf'], model['adx_tf'], model['rsi_tf'], model['cfg'])
        all_results[key] = {'trades': trades, 'final_bal': final_bal, 'name': model['name']}

        print(f"  거래수: {len(trades)} | 최종잔액: ${final_bal:,.0f}")
        print(f"\n  {'#':>3} {'진입시간':>20} {'청산시간':>20} {'방향':>5} {'진입가':>11} {'청산가':>11} {'ROI%':>7} {'손익$':>10} {'사유':>3} {'최고ROI%':>8} {'잔액$':>11}")
        print(f"  {'-'*118}")
        for i, t in enumerate(trades):
            print(f"  {i+1:>3} {t['entry_time']:>20} {t['exit_time']:>20} {t['dir']:>5} {t['entry_px']:>11,.1f} {t['exit_px']:>11,.1f} {t['roi_pct']:>+7.2f} {t['pnl_usd']:>+10,.0f} {t['reason']:>3} {t['peak_roi_pct']:>+8.2f} ${t['bal_after']:>10,.0f}")

        # Monthly Summary
        ms = monthly_summary(trades)
        print(f"\n  월별 요약:")
        print(f"  {'월':>7} {'거래':>4} {'승':>3} {'패':>3} {'총익':>10} {'총손':>10} {'순손익':>10} {'누적':>12}")
        print(f"  {'-'*70}")
        cum = 0
        for ym in sorted(ms.keys()):
            m = ms[ym]; cum += m['pnl']
            print(f"  {ym:>7} {m['cnt']:>4} {m['wins']:>3} {m['losses']:>3} {m['gross_win']:>+10,.0f} {m['gross_loss']:>+10,.0f} {m['pnl']:>+10,.0f} {cum:>+12,.0f}")

    # Save all trades CSV
    with open('v252_all_trades.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model','#','entry_time','exit_time','dir','entry_px','exit_px','roi_pct','pnl_usd','reason','peak_roi_pct','bal_after','margin_used'])
        for key in ['A','B','C']:
            for i, t in enumerate(all_results[key]['trades']):
                w.writerow([key, i+1, t['entry_time'], t['exit_time'], t['dir'],
                           t['entry_px'], t['exit_px'], t['roi_pct'], t['pnl_usd'],
                           t['reason'], t['peak_roi_pct'], t['bal_after'], t['margin_used']])

    # Save monthly CSV
    with open('v252_monthly_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model','month','trades','wins','losses','gross_win','gross_loss','net_pnl','cum_pnl'])
        for key in ['A','B','C']:
            ms = monthly_summary(all_results[key]['trades'])
            cum = 0
            for ym in sorted(ms.keys()):
                m = ms[ym]; cum += m['pnl']
                w.writerow([key, ym, m['cnt'], m['wins'], m['losses'],
                           round(m['gross_win'],0), round(m['gross_loss'],0),
                           round(m['pnl'],0), round(cum,0)])

    # Save JSON
    save = {}
    for key in ['A','B','C']:
        r = all_results[key]
        wins = [t for t in r['trades'] if t['pnl_usd'] > 0]
        losses = [t for t in r['trades'] if t['pnl_usd'] <= 0]
        save[key] = {
            'name': r['name'],
            'final_bal': r['final_bal'],
            'trades': len(r['trades']),
            'wins': len(wins),
            'losses': len(losses),
            'wr': round(len(wins)/len(r['trades'])*100, 1) if r['trades'] else 0,
            'total_profit': round(sum(t['pnl_usd'] for t in wins), 0),
            'total_loss': round(sum(t['pnl_usd'] for t in losses), 0),
            'avg_win_roi': round(np.mean([t['roi_pct'] for t in wins]), 2) if wins else 0,
            'avg_loss_roi': round(np.mean([t['roi_pct'] for t in losses]), 2) if losses else 0,
            'best_trade': max(r['trades'], key=lambda x: x['roi_pct']) if r['trades'] else None,
            'worst_trade': min(r['trades'], key=lambda x: x['roi_pct']) if r['trades'] else None,
            'trade_list': r['trades'],
            'monthly': monthly_summary(r['trades']),
        }

    with open('v252_detail_results.json', 'w', encoding='utf-8') as f:
        json.dump(save, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("Saved: v252_all_trades.csv, v252_monthly_summary.csv, v252_detail_results.json")
    print(f"{'='*80}")
