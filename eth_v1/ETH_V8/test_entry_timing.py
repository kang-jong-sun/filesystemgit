# ETH V8.16: 1봉 지연 진입 vs 즉시 진입 비교
# 6엔진 + 30회 교차검증
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8'); warnings.filterwarnings('ignore')
def ema_pd(s, p): return s.ewm(span=p, adjust=False).mean()

def bt(c, h, l, fm, sm, n, WU, IMMEDIATE_ENTRY):
    """IMMEDIATE_ENTRY=True: REV 후 같은 봉 즉시 재진입"""
    cap=5000.0; pos=0; epx=0.0; psz=0.0; slp=0.0; ton=0; thi=0.0; tlo=999999.0
    w=0; ws=0; ld=0; pk=cap; mdd=0.0; FEE=0.0005
    SL=2.0; TA=54.0; TSL=8.0; MON=18; MPCT=0.20
    sc=tc=rc=wn=ln_=0; gp=gl=0.0; trades=[]

    for i in range(WU, n):
        px=c[i]; hi=h[i]; lo_=l[i]

        if pos != 0:
            w=0; et=0; ex=0.0
            # SL (저가/고가, SL가 청산)
            if ton==0:
                if (pos==1 and lo_<=slp) or (pos==-1 and hi>=slp): et=1; ex=slp
            # TA (고가/저가)
            br=((hi-epx)/epx*100) if pos==1 else ((epx-lo_)/epx*100)
            if br>=TA and ton==0: ton=1
            # TSL (종가)
            if et==0 and ton==1:
                if pos==1:
                    if hi>thi: thi=hi
                    ns=thi*(1-TSL/100)
                    if ns>slp: slp=ns
                    if px<=slp: et=2; ex=px
                else:
                    if lo_<tlo: tlo=lo_
                    ns=tlo*(1+TSL/100)
                    if ns<slp: slp=ns
                    if px>=slp: et=2; ex=px
            # REV (종가)
            if et==0 and i>0:
                bn=fm[i]>sm[i]; bp=fm[i-1]>sm[i-1]
                cu=bn and not bp; cd=not bn and bp
                if (pos==1 and cd) or (pos==-1 and cu): et=3; ex=px
            if et>0:
                pnl=(ex-epx)/epx*psz*pos - psz*FEE; cap+=pnl
                if et==1: sc+=1
                elif et==2: tc+=1
                elif et==3: rc+=1
                if pnl>0: wn+=1; gp+=pnl
                else: ln_+=1; gl+=abs(pnl)
                trades.append({'i': i, 'pnl': pnl})
                ld=pos; pos=0
                if cap>pk: pk=cap
                dd=(pk-cap)/pk if pk>0 else 0
                if dd>mdd: mdd=dd
                if et != 3:  # SL/TSL은 CONTINUE
                    continue
                # REV는 CONTINUE 안 함 — 아래 진입 로직으로

        # 진입
        if pos==0 and i>0:
            bn=fm[i]>sm[i]; bp=fm[i-1]>sm[i-1]
            cu=bn and not bp; cd=not bn and bp
            if cu: w=1; ws=i
            elif cd: w=-1; ws=i
            if w!=0:
                if IMMEDIATE_ENTRY:
                    # 즉시 진입: 크로스 봉에서 바로 진입
                    entry_ok = (i >= ws)
                else:
                    # 1봉 지연: 다음 봉부터 진입
                    entry_ok = (i > ws)

                if entry_ok:
                    if MON>0 and i-ws>MON: w=0; continue
                    if w==ld: continue  # skip same dir
                    if cap<500: continue
                    margin=cap*MPCT; psz=margin*10
                    cap-=psz*FEE; pos=w; epx=px; ton=0; thi=px; tlo=px
                    if pos==1: slp=px*(1-SL/100)
                    else: slp=px*(1+SL/100)
                    if cap>pk: pk=cap
                    w=0

        if cap>pk: pk=cap
        dd=(pk-cap)/pk if pk>0 else 0
        if dd>mdd: mdd=dd
        if cap<=0: break

    tot=sc+tc+rc; pf=gp/gl if gl>0 else 999; wr=wn/(wn+ln_)*100 if (wn+ln_)>0 else 0
    aw=gp/wn if wn>0 else 0; al=gl/ln_ if ln_>0 else 0; wlr=aw/al if al>0 else 999
    return cap, (cap-5000)/50, tot, sc, tc, rc, wn, ln_, pf, mdd*100, wr, wlr, trades

# Load ETH data
df5 = pd.read_csv('eth_usdt_5m_2020_TO_NOW_merged.csv', parse_dates=['timestamp'])
df5 = df5.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
agg = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
df10 = df5.set_index('timestamp').resample('10min').agg(agg).dropna().reset_index()
cs = df10['close']; c = cs.values.astype(np.float64)
h = df10['high'].values.astype(np.float64); l = df10['low'].values.astype(np.float64)
n = len(c)
fm = ema_pd(cs, 250).values.astype(np.float64); sm = ema_pd(cs, 1575).values.astype(np.float64)
years = pd.DatetimeIndex(df10['timestamp']).year.values

print(f"ETH 10m: {n} bars")
print()

for label, immediate in [("1봉 지연 진입 (현재 코드)", False), ("즉시 진입 (REV 같은 봉)", True)]:
    print("=" * 80)
    print(f"  ETH V8.16 — {label}")
    print("=" * 80)

    # 6-Engine
    caps = []
    for _ in range(6):
        r = bt(c, h, l, fm, sm, n, 1575, immediate)
        caps.append(r[0])
    diff6 = max(caps) - min(caps)

    # 30x
    caps30 = []
    for _ in range(30):
        r = bt(c, h, l, fm, sm, n, 1575, immediate)
        caps30.append(r[0])
    sigma = np.std(caps30)

    r = bt(c, h, l, fm, sm, n, 1575, immediate)

    print(f"  6-Engine: {'ALL MATCH' if diff6 < 0.01 else 'MISMATCH'} (diff=${diff6:.6f})")
    print(f"  30x: sigma=${sigma:.6f} {'DETERMINISTIC' if sigma < 0.01 else ''}")
    print(f"  Final:  ${r[0]:,.0f}")
    print(f"  Return: +{r[1]:,.0f}%")
    print(f"  PF:     {r[8]:.2f}")
    print(f"  MDD:    {r[9]:.1f}%")
    print(f"  Trades: {r[2]} (SL:{r[3]} TSL:{r[4]} REV:{r[5]})")
    print(f"  W/L:    {r[6]}W/{r[7]}L ({r[10]:.1f}%)")
    print(f"  WLR:    {r[11]:.1f}:1")

    # Yearly
    yearly = {}
    for t in r[12]:
        yr = str(years[t['i']])
        if yr not in yearly: yearly[yr] = {'t': 0, 'w': 0, 'l': 0, 'pnl': 0.0}
        yearly[yr]['t'] += 1; yearly[yr]['pnl'] += t['pnl']
        if t['pnl'] >= 0: yearly[yr]['w'] += 1
        else: yearly[yr]['l'] += 1

    cum = 5000.0; ayp = True
    for yr in sorted(yearly.keys()):
        yd = yearly[yr]; cum += yd['pnl']
        if yd['pnl'] < 0: ayp = False
        mark = ' X' if yd['pnl'] < 0 else ''
        print(f"    {yr}: {yd['t']}T {yd['w']}W/{yd['l']}L ${yd['pnl']:+,.0f} -> ${cum:,.0f}{mark}")
    print(f"  AllYears+: {'YES' if ayp else 'NO'}")
    print()
