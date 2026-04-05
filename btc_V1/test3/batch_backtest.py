# -*- coding: utf-8 -*-
"""
46개 기획서 일괄 백테스트 - 고정 1000 USDT, 레버리지 10x
5분봉 CSV → 각 기획서 타임프레임으로 리샘플링 → 백테스트
"""
import pandas as pd
import numpy as np
import json, time, os, sys

# ═══════════════════════════════════════════════════════════
# 데이터 로딩
# ═══════════════════════════════════════════════════════════
def load_5m():
    files = [
        'D:/filesystem/futures/btc_V1/test3/btc_usdt_5m_2020_to_now_part1.csv',
        'D:/filesystem/futures/btc_V1/test3/btc_usdt_5m_2020_to_now_part2.csv',
        'D:/filesystem/futures/btc_V1/test3/btc_usdt_5m_2020_to_now_part3.csv',
    ]
    dfs = [pd.read_csv(f, parse_dates=['timestamp']) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset='timestamp', keep='last', inplace=True)
    df.set_index('timestamp', inplace=True)
    return df

def resample(df5m, tf):
    if tf == '5m':
        return df5m[['open','high','low','close','volume']].copy()
    elif tf == '15m':
        rule = '15min'
    else:
        rule = '30min'
    return df5m.resample(rule).agg({
        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
    }).dropna()

# ═══════════════════════════════════════════════════════════
# 지표 계산
# ═══════════════════════════════════════════════════════════
def calc_ema(s, span):
    return s.ewm(span=span, adjust=False).mean().values.astype(np.float64)

def calc_sma(s, span):
    return s.rolling(span).mean().values.astype(np.float64)

def calc_hma(s, span):
    half = max(int(span/2), 1)
    sqrt_n = max(int(np.sqrt(span)), 1)
    wma1 = s.rolling(half).apply(lambda x: np.average(x, weights=range(1,len(x)+1)), raw=True)
    wma2 = s.rolling(span).apply(lambda x: np.average(x, weights=range(1,len(x)+1)), raw=True)
    diff = 2*wma1 - wma2
    hma = diff.rolling(sqrt_n).apply(lambda x: np.average(x, weights=range(1,len(x)+1)), raw=True)
    return hma.values.astype(np.float64)

def calc_wma(s, span):
    return s.rolling(span).apply(lambda x: np.average(x, weights=range(1,len(x)+1)), raw=True).values.astype(np.float64)

def calc_ma(s, ma_type, period):
    if ma_type == 'SMA': return calc_sma(s, period)
    elif ma_type == 'HMA': return calc_hma(s, period)
    elif ma_type == 'WMA': return calc_wma(s, period)
    elif ma_type == 'DEMA':
        e1 = s.ewm(span=period, adjust=False).mean()
        e2 = e1.ewm(span=period, adjust=False).mean()
        return (2*e1 - e2).values.astype(np.float64)
    else:  # EMA
        return calc_ema(s, period)

def calc_adx(h, l, c, period):
    n = len(c)
    pdm = np.zeros(n); mdm = np.zeros(n); tr = np.zeros(n)
    for i in range(1, n):
        hd = h[i]-h[i-1]; ld = l[i-1]-l[i]
        pdm[i] = hd if (hd>ld and hd>0) else 0
        mdm[i] = ld if (ld>hd and ld>0) else 0
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    a = 1.0/period
    atr = pd.Series(tr).ewm(alpha=a, min_periods=period, adjust=False).mean()
    sp = pd.Series(pdm).ewm(alpha=a, min_periods=period, adjust=False).mean()
    sn = pd.Series(mdm).ewm(alpha=a, min_periods=period, adjust=False).mean()
    pdi = 100*sp/atr.replace(0,1e-10)
    mdi = 100*sn/atr.replace(0,1e-10)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,1e-10)
    return dx.ewm(alpha=a, min_periods=period, adjust=False).mean().values.astype(np.float64)

def calc_rsi(c, period):
    d = np.diff(c, prepend=c[0])
    g = np.where(d>0,d,0); lo = np.where(d<0,-d,0)
    a = 1.0/period
    ag = pd.Series(g).ewm(alpha=a, min_periods=period, adjust=False).mean()
    al = pd.Series(lo).ewm(alpha=a, min_periods=period, adjust=False).mean()
    return (100-100/(1+ag/al.replace(0,1e-10))).values.astype(np.float64)

# ═══════════════════════════════════════════════════════════
# 범용 백테스트 엔진 (1000 USDT 고정)
# ═══════════════════════════════════════════════════════════
def backtest(df, p):
    """
    p: parameter dict from extract_params
    Fixed: margin=1000 USDT, leverage=10, position=10,000 USDT
    """
    c = df['close'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    n = len(c)

    fm = calc_ma(df['close'], p['fast_type'], p['fast_period'])
    sm = calc_ma(df['close'], p['slow_type'], p['slow_period'])
    av = calc_adx(h, l, c, p['adx_period'])

    rsi_p = p.get('rsi_period', 14)
    rv = calc_rsi(c, rsi_p) if rsi_p > 0 else np.full(n, 50.0)

    FEE = p.get('fee', 0.0004)
    SL = p['sl_pct']
    TA = p.get('ta_pct', 0)
    TSL = p.get('tsl_pct', 0)
    ADX_MIN = p['adx_min']
    ADX_RISE = p.get('adx_rise', 0)
    RSI_MIN = p.get('rsi_min', 0)
    RSI_MAX = p.get('rsi_max', 100)
    GAP_MIN = p.get('ema_gap', 0)
    MONITOR = p.get('monitor', 0)
    SKIP = p.get('skip_same', False)
    REV_NC = p.get('rev_no_continue', False)
    DAILY_LOSS = p.get('daily_loss', 0)

    FIXED_MARGIN = 1000.0
    LEVERAGE = 10
    PSZ = FIXED_MARGIN * LEVERAGE  # 10,000 USDT

    warmup = p.get('warmup') or max(p['slow_period'], ADX_RISE+1)

    cap = 5000.0
    pos = 0; epx = 0; slp = 0
    ton = False; thi = 0; tlo = 999999
    watching = 0; ws = 0; ld = 0
    pk = cap; mdd = 0; ms = cap

    trades = []
    sl_c = tsl_c = rev_c = wn = ln = 0
    gp = gl = 0.0

    for i in range(warmup, n):
        px = c[i]; h_ = h[i]; l_ = l[i]

        # Daily reset (bar-index based)
        if i > warmup and i % 1440 == 0:
            ms = cap

        if pos != 0:
            watching = 0

            # SL (only if TSL not active)
            if not ton and SL > 0:
                if (pos==1 and l_<=slp) or (pos==-1 and h_>=slp):
                    pnl = (slp-epx)/epx*PSZ*pos - PSZ*FEE
                    cap += pnl; sl_c += 1
                    if pnl>0: wn+=1; gp+=pnl
                    else: ln+=1; gl+=abs(pnl)
                    trades.append({'t':'SL','pnl':pnl})
                    ld=pos; pos=0
                    pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0
                    if dd>mdd: mdd=dd
                    if cap<=0: break
                    continue

            # TA activation
            if TA > 0:
                if pos==1: br=(h_-epx)/epx*100
                else: br=(epx-l_)/epx*100
                if br>=TA and not ton: ton=True

            # TSL
            if ton and TSL > 0:
                if pos==1:
                    if h_>thi: thi=h_
                    ns=thi*(1-TSL/100)
                    if ns>slp: slp=ns
                    if px<=slp:
                        pnl=(px-epx)/epx*PSZ*pos-PSZ*FEE
                        cap+=pnl; tsl_c+=1
                        if pnl>0: wn+=1; gp+=pnl
                        else: ln+=1; gl+=abs(pnl)
                        trades.append({'t':'TSL','pnl':pnl})
                        ld=pos; pos=0
                        pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0
                        if dd>mdd: mdd=dd
                        if cap<=0: break
                        continue
                else:
                    if l_<tlo: tlo=l_
                    ns=tlo*(1+TSL/100)
                    if ns<slp: slp=ns
                    if px>=slp:
                        pnl=(px-epx)/epx*PSZ*pos-PSZ*FEE
                        cap+=pnl; tsl_c+=1
                        if pnl>0: wn+=1; gp+=pnl
                        else: ln+=1; gl+=abs(pnl)
                        trades.append({'t':'TSL','pnl':pnl})
                        ld=pos; pos=0
                        pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0
                        if dd>mdd: mdd=dd
                        if cap<=0: break
                        continue

            # REV
            if i > 0:
                bn=fm[i]>sm[i]; bp=fm[i-1]>sm[i-1]
                cu=bn and not bp; cd=not bn and bp
                if (pos==1 and cd) or (pos==-1 and cu):
                    pnl=(px-epx)/epx*PSZ*pos-PSZ*FEE
                    cap+=pnl; rev_c+=1
                    if pnl>0: wn+=1; gp+=pnl
                    else: ln+=1; gl+=abs(pnl)
                    trades.append({'t':'REV','pnl':pnl})
                    ld=pos; pos=0
                    if not REV_NC:
                        pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0
                        if dd>mdd: mdd=dd
                        if cap<=0: break
                        continue
                    # REV no continue: fall through

        # Entry check
        if i < 1: continue
        bn=fm[i]>sm[i]; bp=fm[i-1]>sm[i-1]
        cu=bn and not bp; cd=not bn and bp

        if pos == 0:
            if cu: watching=1; ws=i
            elif cd: watching=-1; ws=i

            if watching != 0 and i > ws:
                if MONITOR > 0 and i-ws > MONITOR: watching=0; pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd); continue
                if watching==1 and cd: watching=-1; ws=i; pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd); continue
                elif watching==-1 and cu: watching=1; ws=i; pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd); continue
                if SKIP and watching==ld: pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd); continue
                if av[i]<ADX_MIN: pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd); continue
                if ADX_RISE>0 and i>=ADX_RISE and av[i]<=av[i-ADX_RISE]: pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd); continue
                if (RSI_MIN>0 or RSI_MAX<100) and (rv[i]<RSI_MIN or rv[i]>RSI_MAX): pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd); continue
                if GAP_MIN>0:
                    gap=abs(fm[i]-sm[i])/sm[i]*100 if sm[i]>0 else 0
                    if gap<GAP_MIN: pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd); continue
                if DAILY_LOSS>0 and ms>0 and (cap-ms)/ms<=-DAILY_LOSS: watching=0; pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd); continue
                if cap < FIXED_MARGIN: pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd); continue

                # ENTRY (fixed 1000 USDT)
                cap -= PSZ * FEE  # entry fee
                pos = watching; epx = px
                ton = False; thi = px; tlo = px
                if pos==1: slp=px*(1-SL/100) if SL>0 else 0
                else: slp=px*(1+SL/100) if SL>0 else 999999
                pk=max(pk,cap); watching=0

        pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0
        if dd>mdd: mdd=dd
        if cap<=0: break

    # Force close
    if pos!=0 and cap>0:
        pnl=(c[-1]-epx)/epx*PSZ*pos-PSZ*FEE
        cap+=pnl
        if pnl>0: wn+=1; gp+=pnl
        else: ln+=1; gl+=abs(pnl)
        trades.append({'t':'CLOSE','pnl':pnl})

    total = sl_c+tsl_c+rev_c
    pf = gp/gl if gl>0 else float('inf')
    wr = wn/(wn+ln)*100 if (wn+ln)>0 else 0
    net = gp - gl

    return {
        'cap': cap, 'net': net, 'total': total,
        'sl': sl_c, 'tsl': tsl_c, 'rev': rev_c,
        'wn': wn, 'ln': ln, 'wr': wr,
        'pf': pf, 'gp': gp, 'gl': gl,
        'pk': pk, 'mdd': mdd*100,
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("="*90)
    print("  46 STRATEGIES BATCH BACKTEST (Fixed 1000 USDT, 10x Leverage)")
    print("="*90)

    print("\n[1] Loading 5m data...")
    df5m = load_5m()
    print(f"  5m bars: {len(df5m)}")

    # Pre-resample
    print("[2] Resampling...")
    dfs = {}
    for tf in ['5m', '15m', '30m']:
        dfs[tf] = resample(df5m, tf)
        print(f"  {tf}: {len(dfs[tf])} bars")

    print("[3] Loading parameters...")
    with open('D:/filesystem/futures/btc_V1/test3/all_params.json', 'r', encoding='utf-8') as f:
        all_params = json.load(f)
    print(f"  {len(all_params)} strategies")

    print("\n[4] Running backtests...")
    results = []
    for idx, p in enumerate(all_params):
        ver = p['version']
        tf = p['tf']
        df = dfs.get(tf, dfs['30m'])
        t0 = time.time()
        try:
            r = backtest(df, p)
            el = time.time()-t0
            r['version'] = ver
            r['tf'] = tf
            r['fast'] = f"{p['fast_type']}{p['fast_period']}"
            r['slow'] = f"{p['slow_type']}{p['slow_period']}"
            r['sl_pct'] = p['sl_pct']
            r['ta_pct'] = p.get('ta_pct',0)
            r['tsl_pct'] = p.get('tsl_pct',0)
            results.append(r)
            print(f"  [{idx+1:>2}/46] v{ver:<12} ${r['cap']:>10,.0f} net=${r['net']:>+10,.0f} | {r['total']:>3}t PF={r['pf']:>5.1f} MDD={r['mdd']:>5.1f}% | {el:.1f}s")
        except Exception as e:
            print(f"  [{idx+1:>2}/46] v{ver:<12} ERROR: {e}")
            import traceback; traceback.print_exc()

    # Sort by net profit
    results.sort(key=lambda x: x['net'], reverse=True)

    print(f"\n{'='*90}")
    print(f"  RANKING BY NET PROFIT (Fixed 1000 USDT per trade)")
    print(f"{'='*90}")
    print(f"  {'Rank':>4} {'Version':<14} {'TF':<4} {'Fast':<8} {'Slow':<8} {'Net$':>10} {'Trades':>6} {'WR%':>5} {'PF':>5} {'MDD%':>5} {'SL%':>4} {'TA%':>4} {'TSL%':>4}")
    print(f"  {'-'*86}")
    for i, r in enumerate(results, 1):
        print(f"  {i:>4} v{r['version']:<13} {r['tf']:<4} {r['fast']:<8} {r['slow']:<8} ${r['net']:>+9,.0f} {r['total']:>6} {r['wr']:>4.0f}% {r['pf']:>5.1f} {r['mdd']:>5.1f} {r['sl_pct']:>4.0f} {r['ta_pct']:>4.0f} {r['tsl_pct']:>4.0f}")

    # Save CSV
    csv_path = 'D:/filesystem/futures/btc_V1/test3/batch_backtest_results.csv'
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['version','tf','fast','slow','cap','net','total','sl','tsl','rev','wn','ln','wr','pf','gp','gl','mdd','sl_pct','ta_pct','tsl_pct'])
        w.writeheader()
        for r in results:
            w.writerow({k:v for k,v in r.items() if k in w.fieldnames})
    print(f"\nResults saved: {csv_path}")

    # Top 10 / Bottom 10
    print(f"\n{'='*90}")
    print(f"  TOP 10")
    print(f"{'='*90}")
    for i, r in enumerate(results[:10], 1):
        print(f"  {i}. v{r['version']:<12} Net ${r['net']:>+10,.0f} | {r['total']}t WR={r['wr']:.0f}% PF={r['pf']:.1f} MDD={r['mdd']:.1f}%")

    print(f"\n  BOTTOM 10")
    print(f"  {'-'*60}")
    for i, r in enumerate(results[-10:], len(results)-9):
        print(f"  {i}. v{r['version']:<12} Net ${r['net']:>+10,.0f} | {r['total']}t WR={r['wr']:.0f}% PF={r['pf']:.1f} MDD={r['mdd']:.1f}%")


if __name__ == '__main__':
    main()
