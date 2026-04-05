"""v15.5, v15.6A, v15.6B 월별 상세 거래 내역"""
import numpy as np, pandas as pd, json
from bt_fast import load_5m_data, build_mtf, IndicatorCache, _bt_core, calc_ema, calc_rsi, calc_adx, calc_ma

BASE = r'D:\filesystem\futures\btc_V1\test4'

# ============================================================
# 월별 상세 추적 백테스트 (bt_fast 기반 Python 재구현)
# ============================================================
def bt_monthly(cache, tf, cfg):
    """bt_fast 로직 + 개별 거래 기록 + 월별 집계"""
    base = cache.get_base(tf)
    n = base['n']
    closes = base['close']; highs = base['high']; lows = base['low']
    times = cache.get_times(tf)
    ma_f = cache.get_ma(tf, cfg.get('ma_fast_type','ema'), cfg.get('ma_fast',3))
    ma_s = cache.get_ma(tf, cfg.get('ma_slow_type','ema'), cfg.get('ma_slow',200))
    rsi_v = cache.get_rsi(tf, cfg.get('rsi_period',14))
    adx_v = cache.get_adx(tf, cfg.get('adx_period',14))

    adx_min = cfg.get('adx_min',35); rsi_min = cfg.get('rsi_min',30); rsi_max = cfg.get('rsi_max',65)
    sl_pct = cfg.get('sl_pct',0.07); trail_act = cfg.get('trail_activate',0.06); trail_pct = cfg.get('trail_pct',0.03)
    lev = cfg.get('leverage',10); mg_n = cfg.get('margin_normal',0.25); mg_r = cfg.get('margin_reduced',mg_n/2)
    fee = cfg.get('fee_rate',0.0004); ml_limit = cfg.get('monthly_loss_limit',0.0)
    cp_pause = cfg.get('consec_loss_pause',0); cp_cndl = cfg.get('pause_candles',0)
    dd_thr = cfg.get('dd_threshold',0.0)
    delayed = cfg.get('delayed_entry',False)
    dly_max = cfg.get('delay_max_candles',6)
    dly_pmin = cfg.get('delay_price_min',-0.001); dly_pmax = cfg.get('delay_price_max',-0.025)
    liq_d = 1.0 / lev

    bal = 3000.0; peak_bal = bal; pos = 0; ep = 0.0; su = 0.0; ppnl = 0.0; trail = False; rem = 1.0
    msb = bal; cm = 0; mp = False; cl = 0; pu_idx = 0; psig = 0; sprice = 0.0; sidx = 0
    rpeak = 3000.0; mdd = 0.0

    trades = []  # 개별 거래 리스트
    monthly = {}  # 월별 집계

    def month_str(mk): return f"{mk//100}-{mk%100:02d}"

    def ensure_month(mk):
        ms = month_str(mk)
        if ms not in monthly:
            monthly[ms] = {'sb': bal, 'eb': bal, 'trades': 0, 'wins': 0, 'losses': 0,
                           'sl': 0, 'tsl': 0, 'rev': 0, 'fl': 0, 'pnl': 0.0, 'gross_win': 0.0, 'gross_loss': 0.0}

    def record(i, reason, pnl_val, side):
        nonlocal bal, cl, pu_idx, rpeak, mdd
        mk = times[i]; ms = month_str(mk)
        ensure_month(mk)
        bal += pnl_val
        if bal < 0: bal = 0
        monthly[ms]['trades'] += 1
        monthly[ms]['eb'] = bal
        if pnl_val > 0:
            monthly[ms]['wins'] += 1
            monthly[ms]['gross_win'] += pnl_val
        else:
            monthly[ms]['losses'] += 1
            monthly[ms]['gross_loss'] += abs(pnl_val)
        monthly[ms][reason.lower()] = monthly[ms].get(reason.lower(), 0) + 1
        monthly[ms]['pnl'] += pnl_val

        pnl_pct = (pnl_val / su * 100) if su > 0 else 0
        trades.append({
            'month': ms,
            'side': 'L' if side == 1 else 'S',
            'entry': round(ep, 1),
            'exit': round(closes[i], 1),
            'reason': reason,
            'pnl$': round(pnl_val, 2),
            'pnl%': round(pnl_pct, 2),
            'bal': round(bal, 2),
        })

        if pnl_val < 0:
            cl += 1
            if cp_pause > 0 and cl >= cp_pause: pu_idx = i + cp_cndl
        else:
            cl = 0
        if bal > rpeak: rpeak = bal
        dd = (rpeak - bal) / rpeak if rpeak > 0 else 0
        if dd > mdd: mdd = dd

    for i in range(1, n):
        if np.isnan(ma_f[i]) or np.isnan(ma_s[i]) or np.isnan(rsi_v[i]) or np.isnan(adx_v[i]): continue
        mk = times[i]
        if mk != cm:
            if cm != 0:
                ensure_month(cm)
                monthly[month_str(cm)]['eb'] = bal
            cm = mk; msb = bal; mp = False
            ensure_month(mk)
            monthly[month_str(mk)]['sb'] = bal

        if pos != 0:
            cp_ = closes[i]; hp = highs[i]; lp = lows[i]
            if pos == 1:
                pnl = (cp_-ep)/ep; pkc = (hp-ep)/ep; lwc = (lp-ep)/ep
            else:
                pnl = (ep-cp_)/ep; pkc = (ep-lp)/ep; lwc = (ep-hp)/ep
            if pkc > ppnl: ppnl = pkc

            if lwc <= -liq_d:
                pu2 = su*rem*(-liq_d) - su*rem*fee
                record(i, 'FL', pu2, pos); pos = 0; continue
            sth = sl_pct
            if lwc <= -sth:
                pu2 = su*rem*(-sth) - su*rem*fee
                record(i, 'SL', pu2, pos); pos = 0; continue
            if ppnl >= trail_act: trail = True
            if trail:
                tl = ppnl - trail_pct
                if pnl <= tl:
                    pu2 = su*rem*tl - su*rem*fee
                    record(i, 'TSL', pu2, pos); pos = 0; continue

            cu = ma_f[i]>ma_s[i] and ma_f[i-1]<=ma_s[i-1]
            cd = ma_f[i]<ma_s[i] and ma_f[i-1]>=ma_s[i-1]
            ao = adx_v[i]>=adx_min; ro = rsi_min<=rsi_v[i]<=rsi_max
            rv = False; nd = 0
            if pos==1 and cd and ao and ro: rv=True; nd=-1
            elif pos==-1 and cu and ao and ro: rv=True; nd=1
            if rv:
                pu2 = su*rem*pnl - su*rem*fee
                record(i, 'REV', pu2, pos); pos = 0
                can = (not mp) and (i >= pu_idx)
                if bal > 10 and can:
                    mg = mg_n
                    if peak_bal > 0 and dd_thr < 0:
                        dn = (peak_bal-bal)/peak_bal
                        if dn > abs(dd_thr): mg = mg_r
                    mu = bal*mg; s2 = mu*lev; bal -= s2*fee; pos = nd; ep = closes[i]; su = s2; ppnl = 0.0; trail = False; rem = 1.0
                    if bal > peak_bal: peak_bal = bal
                continue

        if pos == 0 and bal > 10:
            if ml_limit < 0 and msb > 0:
                mr = (bal-msb)/msb
                if mr < ml_limit: mp = True
            can = (not mp) and (i >= pu_idx)
            if not can: psig = 0; continue

            cu = ma_f[i]>ma_s[i] and ma_f[i-1]<=ma_s[i-1]
            cd = ma_f[i]<ma_s[i] and ma_f[i-1]>=ma_s[i-1]
            ao = adx_v[i]>=adx_min; ro = rsi_min<=rsi_v[i]<=rsi_max
            sig = 0
            if cu and ao and ro: sig = 1
            elif cd and ao and ro: sig = -1

            do = False; ed = 0
            if sig != 0:
                if delayed: psig = sig; sprice = closes[i]; sidx = i
                else: do = True; ed = sig
            elif psig != 0 and delayed:
                el = i - sidx
                if el > dly_max: psig = 0
                else:
                    pc = (closes[i]-sprice)/sprice
                    if psig == 1:
                        if dly_pmax <= pc <= dly_pmin: do=True; ed=1; psig=0
                    elif psig == -1:
                        iv = -pc
                        if dly_pmax <= iv <= dly_pmin: do=True; ed=-1; psig=0

            if do:
                mg = mg_n
                if peak_bal > 0 and dd_thr < 0:
                    dn = (peak_bal-bal)/peak_bal
                    if dn > abs(dd_thr): mg = mg_r
                mu = bal*mg; s2 = mu*lev; bal -= s2*fee; pos = ed; ep = closes[i]; su = s2; ppnl = 0.0; trail = False; rem = 1.0
                if bal > peak_bal: peak_bal = bal

        if bal > peak_bal: peak_bal = bal
        if bal > rpeak: rpeak = bal

    if cm != 0:
        ensure_month(cm)
        monthly[month_str(cm)]['eb'] = bal

    if pos != 0 and n > 0:
        if pos == 1: pf_ = (closes[n-1]-ep)/ep
        else: pf_ = (ep-closes[n-1])/ep
        pu2 = su*rem*pf_ - su*rem*fee
        record(n-1, 'END', pu2, pos); pos = 0

    wins = [t for t in trades if t['pnl$'] > 0]
    losses = [t for t in trades if t['pnl$'] <= 0]
    gw = sum(t['pnl$'] for t in wins); gl = abs(sum(t['pnl$'] for t in losses))
    pf = gw / gl if gl > 0 else float('inf')

    return {'bal': round(bal, 0), 'pf': round(pf, 2), 'mdd': round(mdd*100, 1),
            'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
            'fl': sum(1 for t in trades if t['reason']=='FL'),
            'monthly': monthly, 'trade_list': trades}


def print_monthly(name, result):
    """월별 상세 출력"""
    m = result['monthly']
    tl = result['trade_list']
    print(f"\n{'='*100}")
    print(f"  {name}")
    print(f"  잔액: ${result['bal']:,.0f} | PF: {result['pf']} | MDD: {result['mdd']}% | 거래: {result['trades']} | FL: {result['fl']}")
    print(f"{'='*100}")

    print(f"\n  {'월':>8} {'손익금':>12} {'손익률':>8} {'잔액':>14} {'거래':>4} {'승':>3}{'패':>3} {'SL':>3}{'TSL':>4}{'REV':>4}{'FL':>3} 비고")
    print(f"  {'-'*90}")

    prev_year = None
    year_pnl = 0.0
    year_sb = 0.0

    for ms in sorted(m.keys()):
        v = m[ms]
        year = ms[:4]
        if year != prev_year:
            if prev_year is not None and year_sb > 0:
                yr_ret = (year_pnl / year_sb) * 100
                print(f"  {'['+prev_year+' 합계]':>8} ${year_pnl:>10,.0f} {yr_ret:>+7.1f}%")
                print(f"  {'-'*90}")
            prev_year = year
            year_pnl = 0.0
            year_sb = v['sb']

        pnl = v['pnl']
        pct = (pnl / v['sb'] * 100) if v['sb'] > 0 else 0
        year_pnl += pnl

        note = ''
        if v.get('fl', 0) > 0: note += 'FL! '
        if pct <= -20: note += '큰손실 '
        if pct >= 30: note += '대형수익 '
        if v['trades'] == 0 and abs(pnl) < 1: note += ''

        if v['trades'] > 0 or abs(pnl) > 1:
            print(f"  {ms:>8} ${pnl:>10,.0f} {pct:>+7.1f}% ${v['eb']:>12,.0f} {v['trades']:>4} {v['wins']:>3}{v['losses']:>3} {v.get('sl',0):>3}{v.get('tsl',0):>4}{v.get('rev',0):>4}{v.get('fl',0):>3} {note}")

    # 마지막 연도 합계
    if prev_year is not None and year_sb > 0:
        yr_ret = (year_pnl / year_sb) * 100
        print(f"  {'['+prev_year+' 합계]':>8} ${year_pnl:>10,.0f} {yr_ret:>+7.1f}%")

    # 개별 거래 리스트
    print(f"\n  --- 개별 거래 내역 ({len(tl)}건) ---")
    print(f"  {'#':>3} {'월':>8} {'방향':>4} {'진입가':>12} {'청산가':>12} {'사유':>4} {'손익$':>12} {'손익%':>8} {'잔액':>14}")
    print(f"  {'-'*90}")
    for i, t in enumerate(tl):
        print(f"  {i+1:>3} {t['month']:>8} {t['side']:>4} ${t['entry']:>10,.1f} ${t['exit']:>10,.1f} {t['reason']:>4} ${t['pnl$']:>10,.2f} {t['pnl%']:>+7.2f}% ${t['bal']:>12,.2f}")


def main():
    print("=" * 100)
    print("  v15.5 / v15.6A / v15.6B 월별 상세 거래 내역")
    print("=" * 100)

    df5 = load_5m_data(BASE)
    mtf = build_mtf(df5)
    cache = IndicatorCache(mtf)

    # v15.5 (30m EMA(3/200) ADX>=35 RSI35-65 SL7% Trail+6/-5 10x M35% ML-25% DD-30%)
    cfg_v155 = {
        'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
        'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':35,'rsi_max':65,
        'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.05,
        'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,
        'monthly_loss_limit':-0.25,'dd_threshold':-0.30,
        'fee_rate':0.0004,'initial_capital':3000.0,
    }

    # v15.6A (15m EMA(3/150) ADX>=45 RSI35-70 SL4% Trail+10/-3 15x M40%)
    cfg_v156a = {
        'timeframe':'15m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':150,
        'adx_period':14,'adx_min':45,'rsi_period':14,'rsi_min':35,'rsi_max':70,
        'sl_pct':0.04,'trail_activate':0.10,'trail_pct':0.03,
        'leverage':15,'margin_normal':0.40,'margin_reduced':0.20,
        'fee_rate':0.0004,'initial_capital':3000.0,
    }

    # v15.6B (30m EMA(3/200) ADX>=35 RSI35-65 SL7% Trail+6/-5 10x M35% ML-25% DD-30%) = v15.5와 동일
    cfg_v156b = dict(cfg_v155)

    for name, cfg in [('v15.5', cfg_v155), ('v15.6 모델A (PF형)', cfg_v156a), ('v15.6 모델B (수익형)', cfg_v156b)]:
        r = bt_monthly(cache, cfg['timeframe'], cfg)
        print_monthly(name, r)

    print(f"\n{'='*100}")
    print("  완료!")

if __name__ == '__main__':
    main()
