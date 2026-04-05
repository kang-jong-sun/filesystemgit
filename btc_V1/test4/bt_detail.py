"""v16.5 거래 상세 추출기 - bt_fast.py _bt_core 로직을 Python으로 1:1 재현"""
import numpy as np, pandas as pd, csv, json
from bt_fast import load_5m_data, build_mtf, IndicatorCache

def run_detail(cache, tf, cfg):
    """run_backtest와 동일 로직이지만 거래 상세를 반환"""
    base = cache.get_base(tf)
    df = cache.mtf[tf]
    n = base['n']
    closes = base['close']
    highs = base['high']
    lows = base['low']
    times = df['time'].values  # datetime
    months = cache.get_times(tf)
    ma_fast = cache.get_ma(tf, cfg.get('ma_fast_type','ema'), cfg.get('ma_fast',7))
    ma_slow = cache.get_ma(tf, cfg.get('ma_slow_type','ema'), cfg.get('ma_slow',100))
    rsi_arr = cache.get_rsi(tf, cfg.get('rsi_period',14))
    adx_arr = cache.get_adx(tf, cfg.get('adx_period',14))
    atr_arr = cache.get_atr(tf, cfg.get('atr_period',14))

    adx_min = cfg.get('adx_min',30.0)
    rsi_min = cfg.get('rsi_min',30.0)
    rsi_max = cfg.get('rsi_max',58.0)
    sl_pct = cfg.get('sl_pct',0.07)
    trail_act = cfg.get('trail_activate',0.08)
    trail_pct_v = cfg.get('trail_pct',0.06)
    use_atr_sl = cfg.get('use_atr_sl',False)
    atr_sl_mult = cfg.get('atr_sl_mult',2.0)
    atr_sl_min = cfg.get('atr_sl_min',0.02)
    atr_sl_max = cfg.get('atr_sl_max',0.12)
    use_atr_trail = cfg.get('use_atr_trail',False)
    atr_trail_mult = cfg.get('atr_trail_mult',1.5)
    leverage = cfg.get('leverage',10)
    margin_normal = cfg.get('margin_normal',0.20)
    margin_reduced = cfg.get('margin_reduced',0.10)
    ml_limit = cfg.get('monthly_loss_limit',0.0)
    cp_pause = cfg.get('consec_loss_pause',0)
    cp_candles = cfg.get('pause_candles',0)
    dd_thresh = cfg.get('dd_threshold',0.0)
    fee = cfg.get('fee_rate',0.0004)
    init_cap = cfg.get('initial_capital',3000.0)
    delayed = cfg.get('delayed_entry',False)
    delay_max = cfg.get('delay_max_candles',6)
    delay_pmin = cfg.get('delay_price_min',-0.001)
    delay_pmax = cfg.get('delay_price_max',-0.025)

    bal = init_cap; peak_bal = bal; pos = 0; ep = 0.0; ei = 0
    su = 0.0; ppnl = 0.0; trail = False; rem = 1.0
    msb = bal; cm = 0; mp = False; cl = 0; pu_idx = 0
    psig = 0; sprice = 0.0; sidx = 0
    liq_d = 1.0 / leverage
    rpeak = init_cap; mdd = 0.0

    trades = []

    def ts(idx):
        return str(pd.Timestamp(times[idx]))[:19]

    def close_trade(exit_i, exit_px, roi, pnl_usd, reason, peak_r):
        nonlocal bal, pos, cl, pu_idx, rpeak, mdd
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
        if np.isnan(ma_fast[i]) or np.isnan(ma_slow[i]) or np.isnan(rsi_arr[i]) or np.isnan(adx_arr[i]):
            continue
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
                if cp_pause > 0 and cl >= cp_pause: pu_idx = i + cp_candles
                if bal > rpeak: rpeak = bal
                dd = (rpeak - bal) / rpeak if rpeak > 0 else 0
                if dd > mdd: mdd = dd
                continue

            # SL
            sth = sl_pct
            if use_atr_sl and not np.isnan(atr_arr[i]):
                a = (atr_arr[i] * atr_sl_mult) / ep
                if a < atr_sl_min: a = atr_sl_min
                if a > atr_sl_max: a = atr_sl_max
                sth = a
            if lwc <= -sth:
                pu2 = su * rem * (-sth) - su * rem * fee; bal += pu2
                if bal < 0: bal = 0
                close_trade(i, cp, -sth, pu2, 'SL', ppnl)
                pos = 0; cl += 1
                if cp_pause > 0 and cl >= cp_pause: pu_idx = i + cp_candles
                if bal > rpeak: rpeak = bal
                dd = (rpeak - bal) / rpeak if rpeak > 0 else 0
                if dd > mdd: mdd = dd
                continue

            # Trail
            if ppnl >= trail_act: trail = True
            if trail:
                tw = trail_pct_v
                if use_atr_trail and not np.isnan(atr_arr[i]):
                    at = (atr_arr[i] * atr_trail_mult) / ep
                    if at > tw: tw = at
                    if tw < 0.02: tw = 0.02
                tl = ppnl - tw
                if pnl <= tl:
                    pu2 = su * rem * tl - su * rem * fee; bal += pu2
                    close_trade(i, cp, tl, pu2, 'TSL', ppnl)
                    pos = 0
                    if tl > 0: cl = 0
                    else: cl += 1
                    if cl > 0 and cp_pause > 0 and cl >= cp_pause: pu_idx = i + cp_candles
                    if bal > rpeak: rpeak = bal
                    dd = (rpeak - bal) / rpeak if rpeak > 0 else 0
                    if dd > mdd: mdd = dd
                    continue

            # Reversal
            cu = ma_fast[i] > ma_slow[i] and ma_fast[i-1] <= ma_slow[i-1]
            cd = ma_fast[i] < ma_slow[i] and ma_fast[i-1] >= ma_slow[i-1]
            ao = adx_arr[i] >= adx_min
            ro = rsi_min <= rsi_arr[i] <= rsi_max
            rv = False; nd = 0
            if pos == 1 and cd and ao and ro: rv = True; nd = -1
            elif pos == -1 and cu and ao and ro: rv = True; nd = 1
            if rv:
                pu2 = su * rem * pnl - su * rem * fee; bal += pu2
                close_trade(i, cp, pnl, pu2, 'REV', ppnl)
                pos = 0
                if pnl > 0: cl = 0
                else: cl += 1
                if cl > 0 and cp_pause > 0 and cl >= cp_pause: pu_idx = i + cp_candles
                if bal > rpeak: rpeak = bal
                dd = (rpeak - bal) / rpeak if rpeak > 0 else 0
                if dd > mdd: mdd = dd
                # Immediate re-entry
                can = not mp and i >= pu_idx
                if bal > 10 and can:
                    mg = margin_normal
                    if peak_bal > 0 and dd_thresh < 0:
                        dn = (peak_bal - bal) / peak_bal
                        if dn > abs(dd_thresh): mg = margin_reduced
                    mu = bal * mg; s2 = mu * leverage
                    bal -= s2 * fee; pos = nd; ep = cp; ei = i; su = s2; ppnl = 0.0; trail = False; rem = 1.0
                    if bal > peak_bal: peak_bal = bal
                continue

        # Entry logic
        if pos == 0 and bal > 10:
            if ml_limit < 0 and msb > 0:
                mr = (bal - msb) / msb
                if mr < ml_limit: mp = True
            can = not mp and i >= pu_idx
            if not can: psig = 0; continue
            cu = ma_fast[i] > ma_slow[i] and ma_fast[i-1] <= ma_slow[i-1]
            cd = ma_fast[i] < ma_slow[i] and ma_fast[i-1] >= ma_slow[i-1]
            ao = adx_arr[i] >= adx_min
            ro = rsi_min <= rsi_arr[i] <= rsi_max
            sig = 0
            if cu and ao and ro: sig = 1
            elif cd and ao and ro: sig = -1
            do = False; ed = 0
            if sig != 0:
                if delayed: psig = sig; sprice = cp; sidx = i
                else: do = True; ed = sig
            elif psig != 0 and delayed:
                el = i - sidx
                if el > delay_max: psig = 0
                else:
                    pc = (cp - sprice) / sprice
                    if psig == 1:
                        if delay_pmax <= pc <= delay_pmin: do = True; ed = 1; psig = 0
                    elif psig == -1:
                        iv = -pc
                        if delay_pmax <= iv <= delay_pmin: do = True; ed = -1; psig = 0
            if do:
                mg = margin_normal
                if peak_bal > 0 and dd_thresh < 0:
                    dn = (peak_bal - bal) / peak_bal
                    if dn > abs(dd_thresh): mg = margin_reduced
                mu = bal * mg; s2 = mu * leverage
                bal -= s2 * fee; pos = ed; ep = cp; ei = i; su = s2; ppnl = 0.0; trail = False; rem = 1.0
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


if __name__ == '__main__':
    df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
    mtf = build_mtf(df5)
    cache = IndicatorCache(mtf)

    modes = {
        'A_balance': {'margin_normal':0.40,'margin_reduced':0.20,'dd_threshold':-0.25,'monthly_loss_limit':-0.20},
        'B_safe':    {'margin_normal':0.25,'margin_reduced':0.15,'dd_threshold':-0.15,'monthly_loss_limit':-0.10},
        'D_aggr':    {'margin_normal':0.50,'margin_reduced':0.25,'dd_threshold':-0.25,'monthly_loss_limit':-0.20},
        'E_ultra':   {'margin_normal':0.50,'margin_reduced':0.50,'dd_threshold':0,'monthly_loss_limit':0},
    }

    base_cfg = {'timeframe':'10m','ma_fast_type':'hma','ma_slow_type':'ema','ma_fast':21,'ma_slow':250,
                'adx_period':20,'adx_min':35,'rsi_period':14,'rsi_min':40,'rsi_max':75,
                'sl_pct':0.06,'trail_activate':0.07,'trail_pct':0.03,
                'leverage':10,'fee_rate':0.0004,'initial_capital':3000.0}

    all_trades = {}
    for mode, overrides in modes.items():
        cfg = {**base_cfg, **overrides}
        trades, final_bal = run_detail(cache, '10m', cfg)
        all_trades[mode] = trades
        print(f'\n=== {mode} | {len(trades)} trades | ${final_bal:,.0f} ===')
        print(f'{"#":>3} {"entry_time":>20} {"exit_time":>20} {"dir":>5} {"entry_px":>11} {"exit_px":>11} {"ROI%":>7} {"PnL$":>10} {"rsn":>3} {"peak%":>6} {"bal$":>11}')
        print('-'*120)
        for i, t in enumerate(trades):
            print(f'{i+1:>3} {t["entry_time"]:>20} {t["exit_time"]:>20} {t["dir"]:>5} {t["entry_px"]:>11,.1f} {t["exit_px"]:>11,.1f} {t["roi_pct"]:>+7.2f} {t["pnl_usd"]:>+10,.0f} {t["reason"]:>3} {t["peak_roi_pct"]:>+6.2f} ${t["bal_after"]:>10,.0f}')

    # Save CSV
    with open('v165_all_trades_detail.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['mode','#','entry_time','exit_time','dir','entry_px','exit_px','roi_pct','pnl_usd','reason','peak_roi_pct','bal_after','margin_used'])
        for mode, trades in all_trades.items():
            for i, t in enumerate(trades):
                w.writerow([mode, i+1, t['entry_time'], t['exit_time'], t['dir'],
                           t['entry_px'], t['exit_px'], t['roi_pct'], t['pnl_usd'],
                           t['reason'], t['peak_roi_pct'], t['bal_after'], t['margin_used']])

    # Monthly summary (mode A)
    trades_a = all_trades['A_balance']
    monthly = {}
    for t in trades_a:
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

    with open('v165_monthly_detail.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['month','trades','wins','losses','gross_win','gross_loss','net_pnl','cum_pnl'])
        cum = 0
        for ym in sorted(monthly.keys()):
            m = monthly[ym]
            cum += m['pnl']
            w.writerow([ym, m['cnt'], m['wins'], m['losses'],
                       round(m['gross_win'],0), round(m['gross_loss'],0),
                       round(m['pnl'],0), round(cum,0)])

    print(f'\nSaved: v165_all_trades_detail.csv, v165_monthly_detail.csv')
