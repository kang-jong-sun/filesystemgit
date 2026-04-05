"""v16.0 월별 상세 거래 내역"""
import numpy as np
from bt_fast import load_5m_data, build_mtf, IndicatorCache

BASE = r'D:\filesystem\futures\btc_V1\test4'

def bt_monthly_v16(cache):
    """v16.0 bt_fast 로직 + 개별 거래 기록"""
    tf = '30m'
    base = cache.get_base(tf)
    n = base['n']
    closes = base['close']; highs = base['high']; lows = base['low']
    times = cache.get_times(tf)
    ma_f = cache.get_ma(tf, 'wma', 3)
    ma_s = cache.get_ma(tf, 'ema', 200)
    rsi_v = cache.get_rsi(tf, 14)
    adx_v = cache.get_adx(tf, 20)

    adx_min=35; rsi_min=35; rsi_max=65
    sl_pct=0.08; trail_act=0.04; trail_pct=0.03
    lev=10; mg=0.50; fee=0.0004; liq_d=1.0/lev

    bal=3000.0; peak_bal=bal; pos=0; ep=0.0; su=0.0; ppnl=0.0; trail=False; rem=1.0
    msb=bal; cm=0; mp=False; cl=0; rpeak=3000.0; mdd=0.0

    trades = []
    monthly = {}

    def ms(mk): return f"{mk//100}-{mk%100:02d}"

    def ensure(mk):
        s = ms(mk)
        if s not in monthly:
            monthly[s] = {'sb':bal,'eb':bal,'tr':0,'w':0,'l':0,'sl':0,'tsl':0,'rev':0,'fl':0,'pnl':0.0}

    def rec(i, reason, pnl_val, side):
        nonlocal bal, cl, rpeak, mdd
        mk = times[i]; s = ms(mk); ensure(mk)
        bal += pnl_val
        if bal < 0: bal = 0
        monthly[s]['tr'] += 1; monthly[s]['eb'] = bal; monthly[s]['pnl'] += pnl_val
        monthly[s][reason.lower()] = monthly[s].get(reason.lower(), 0) + 1
        if pnl_val > 0: monthly[s]['w'] += 1
        else: monthly[s]['l'] += 1

        pnl_pct = (pnl_val / su * 100) if su > 0 else 0
        trades.append({
            'month': s, 'side': 'L' if side==1 else 'S',
            'entry': round(ep, 1), 'exit': round(closes[i], 1),
            'reason': reason, 'pnl$': round(pnl_val, 2),
            'pnl%': round(pnl_pct, 2), 'bal': round(bal, 2),
        })
        if pnl_val < 0: cl += 1
        else: cl = 0
        if bal > rpeak: rpeak = bal
        dd = (rpeak-bal)/rpeak if rpeak > 0 else 0
        if dd > mdd: mdd = dd

    for i in range(1, n):
        if np.isnan(ma_f[i]) or np.isnan(ma_s[i]) or np.isnan(rsi_v[i]) or np.isnan(adx_v[i]): continue
        mk = times[i]
        if mk != cm:
            if cm != 0: ensure(cm); monthly[ms(cm)]['eb'] = bal
            cm = mk; msb = bal; mp = False; ensure(mk); monthly[ms(mk)]['sb'] = bal

        if pos != 0:
            cp_ = closes[i]; hp = highs[i]; lp = lows[i]
            if pos == 1: pnl=(cp_-ep)/ep; pkc=(hp-ep)/ep; lwc=(lp-ep)/ep
            else: pnl=(ep-cp_)/ep; pkc=(ep-lp)/ep; lwc=(ep-hp)/ep
            if pkc > ppnl: ppnl = pkc

            if lwc <= -liq_d:
                rec(i, 'FL', su*rem*(-liq_d)-su*rem*fee, pos); pos=0; continue
            if lwc <= -sl_pct:
                rec(i, 'SL', su*rem*(-sl_pct)-su*rem*fee, pos); pos=0; continue
            if ppnl >= trail_act: trail = True
            if trail:
                tl = ppnl - trail_pct
                if pnl <= tl:
                    rec(i, 'TSL', su*rem*tl-su*rem*fee, pos); pos=0; continue

            cu = ma_f[i]>ma_s[i] and ma_f[i-1]<=ma_s[i-1]
            cd = ma_f[i]<ma_s[i] and ma_f[i-1]>=ma_s[i-1]
            ao = adx_v[i]>=adx_min; ro = rsi_min<=rsi_v[i]<=rsi_max
            rv=False; nd=0
            if pos==1 and cd and ao and ro: rv=True; nd=-1
            elif pos==-1 and cu and ao and ro: rv=True; nd=1
            if rv:
                rec(i, 'REV', su*rem*pnl-su*rem*fee, pos); pos=0
                if bal > 10:
                    mu=bal*mg; s2=mu*lev; bal-=s2*fee; pos=nd; ep=closes[i]; su=s2; ppnl=0.0; trail=False; rem=1.0
                    if bal>peak_bal: peak_bal=bal
                continue

        if pos==0 and bal>10:
            cu = ma_f[i]>ma_s[i] and ma_f[i-1]<=ma_s[i-1]
            cd = ma_f[i]<ma_s[i] and ma_f[i-1]>=ma_s[i-1]
            ao = adx_v[i]>=adx_min; ro = rsi_min<=rsi_v[i]<=rsi_max
            sig=0
            if cu and ao and ro: sig=1
            elif cd and ao and ro: sig=-1
            if sig!=0:
                mu=bal*mg; s2=mu*lev; bal-=s2*fee; pos=sig; ep=closes[i]; su=s2; ppnl=0.0; trail=False; rem=1.0
                if bal>peak_bal: peak_bal=bal

        if bal>peak_bal: peak_bal=bal
        if bal>rpeak: rpeak=bal

    if cm!=0: ensure(cm); monthly[ms(cm)]['eb']=bal
    if pos!=0 and n>0:
        if pos==1: pf_=(closes[n-1]-ep)/ep
        else: pf_=(ep-closes[n-1])/ep
        rec(n-1, 'END', su*rem*pf_-su*rem*fee, pos); pos=0

    wins=[t for t in trades if t['pnl$']>0]
    losses=[t for t in trades if t['pnl$']<=0]
    gw=sum(t['pnl$'] for t in wins); gl=abs(sum(t['pnl$'] for t in losses))
    pf=gw/gl if gl>0 else float('inf')

    return {'bal':round(bal,0),'pf':round(pf,2),'mdd':round(mdd*100,1),
            'trades':len(trades),'wins':len(wins),'losses':len(losses),
            'monthly':monthly,'trade_list':trades}


def main():
    print("="*110)
    print("  v16.0 월별 상세 거래 내역")
    print("  30m WMA(3)/EMA(200) + ADX(20)>=35 + RSI 35-65 | SL-8% Trail+4/-3 | 10x M50%")
    print("="*110)

    df5 = load_5m_data(BASE)
    mtf = build_mtf(df5)
    cache = IndicatorCache(mtf)

    r = bt_monthly_v16(cache)
    m = r['monthly']
    tl = r['trade_list']

    print(f"\n  잔액: ${r['bal']:,.0f} | PF: {r['pf']} | MDD: {r['mdd']}% | 거래: {r['trades']} ({r['wins']}승/{r['losses']}패) | FL: 0")

    # 월별 테이블
    print(f"\n  {'월':>8} {'손익금':>14} {'손익률':>8} {'잔액':>16} {'거래':>4} {'승':>3}{'패':>3} {'SL':>3}{'TSL':>4}{'REV':>4}{'FL':>3} 비고")
    print(f"  {'-'*95}")

    prev_yr = None; yr_pnl = 0.0; yr_sb = 0.0

    for s in sorted(m.keys()):
        v = m[s]; yr = s[:4]
        if yr != prev_yr:
            if prev_yr is not None and yr_sb > 0:
                print(f"  {'['+prev_yr+']':>8} ${yr_pnl:>12,.0f} {yr_pnl/yr_sb*100:>+7.1f}%")
                print(f"  {'-'*95}")
            prev_yr = yr; yr_pnl = 0.0; yr_sb = v['sb']

        pnl = v['pnl']; pct = (pnl/v['sb']*100) if v['sb']>0 else 0; yr_pnl += pnl
        note = ''
        if pct <= -20: note += '큰손실 '
        if pct >= 30: note += '대형수익 '

        if v['tr'] > 0 or abs(pnl) > 1:
            print(f"  {s:>8} ${pnl:>12,.0f} {pct:>+7.1f}% ${v['eb']:>14,.0f} {v['tr']:>4} {v['w']:>3}{v['l']:>3} {v.get('sl',0):>3}{v.get('tsl',0):>4}{v.get('rev',0):>4}{v.get('fl',0):>3} {note}")

    if prev_yr and yr_sb > 0:
        print(f"  {'['+prev_yr+']':>8} ${yr_pnl:>12,.0f} {yr_pnl/yr_sb*100:>+7.1f}%")

    # 개별 거래
    print(f"\n  {'='*110}")
    print(f"  개별 거래 내역 ({len(tl)}건)")
    print(f"  {'='*110}")
    print(f"  {'#':>3} {'월':>8} {'방향':>4} {'진입가':>12} {'청산가':>12} {'사유':>4} {'손익$':>14} {'손익%':>8} {'잔액':>16}")
    print(f"  {'-'*95}")
    for i, t in enumerate(tl):
        mark = ''
        if t['pnl%'] >= 5: mark = ' **'
        elif t['pnl%'] <= -3: mark = ' !!'
        print(f"  {i+1:>3} {t['month']:>8} {t['side']:>4} ${t['entry']:>10,.1f} ${t['exit']:>10,.1f} {t['reason']:>4} ${t['pnl$']:>12,.2f} {t['pnl%']:>+7.2f}% ${t['bal']:>14,.2f}{mark}")

    print(f"\n  완료!")


if __name__ == '__main__':
    main()
