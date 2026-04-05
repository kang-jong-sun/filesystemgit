# BTC with ETH V8.16 identical parameters test
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8'); warnings.filterwarnings('ignore')

def ema_pd(s, p): return s.ewm(span=p, adjust=False).mean()

def bt(c, h, l, fm, sm, n, SL, TA, TSL, MON, SKIP, WU, MPCT):
    cap=5000.0; pos=0; epx=0.0; psz=0.0; slp=0.0; ton=0; thi=0.0; tlo=999999.0
    w=0; ws=0; ld=0; pk=cap; mdd=0.0; FEE=0.0004
    sc=tc=rc=wn=ln_=0; gp=gl=0.0
    for i in range(WU, n):
        px=c[i]; hi=h[i]; lo_=l[i]
        if pos != 0:
            w=0; et=0; ex=0.0
            if ton==0 and SL>0:
                if (pos==1 and lo_<=slp) or (pos==-1 and hi>=slp): et=1; ex=slp
            br=((hi-epx)/epx*100) if pos==1 else ((epx-lo_)/epx*100)
            if br>=TA and ton==0: ton=1
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
                ld=pos; pos=0
                if cap>pk: pk=cap
                dd=(pk-cap)/pk if pk>0 else 0
                if dd>mdd: mdd=dd
                continue
        if pos==0 and i>0:
            bn=fm[i]>sm[i]; bp=fm[i-1]>sm[i-1]
            cu=bn and not bp; cd=not bn and bp
            if cu: w=1; ws=i
            elif cd: w=-1; ws=i
            if w!=0 and i>ws:
                if MON>0 and i-ws>MON: w=0; continue
                if SKIP and w==ld: continue
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
    return cap, (cap-5000)/50, tot, sc, tc, rc, wn, ln_, pf, mdd*100, wr, wlr

# Load
df5 = pd.read_csv('btc_usdt_5m_merged.csv', parse_dates=['timestamp'])
df5 = df5.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
agg = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
df10 = df5.set_index('timestamp').resample('10min').agg(agg).dropna().reset_index()
df30 = df5.set_index('timestamp').resample('30min').agg(agg).dropna().reset_index()
print(f"BTC 10m: {len(df10)} bars | 30m: {len(df30)} bars")

cs10=df10['close']; c10=cs10.values.astype(np.float64)
h10=df10['high'].values.astype(np.float64); l10=df10['low'].values.astype(np.float64)
cs30=df30['close']; c30=cs30.values.astype(np.float64)
h30=df30['high'].values.astype(np.float64); l30=df30['low'].values.astype(np.float64)

# MAs
fm250_10 = ema_pd(cs10, 250).values.astype(np.float64)
sm1575_10 = ema_pd(cs10, 1575).values.astype(np.float64)
fm83_30 = ema_pd(cs30, 83).values.astype(np.float64)
sm525_30 = ema_pd(cs30, 525).values.astype(np.float64)
fm100_30 = ema_pd(cs30, 100).values.astype(np.float64)
sm600_30 = ema_pd(cs30, 600).values.astype(np.float64)
fm100_10 = ema_pd(cs10, 100).values.astype(np.float64)
sm500_10 = ema_pd(cs10, 500).values.astype(np.float64)

print()
print("=" * 90)
print("  BTC with ETH V8.16 Strategy — Filter ALL OFF, 20% Margin, 10x")
print("=" * 90)
print(f"  {'Strategy':>45} {'Final':>12} {'Ret%':>8} {'PF':>6} {'MDD':>6} {'T':>4} {'SL':>4} {'TSL':>4} {'WLR':>6}")
print("  " + "-" * 90)

tests = [
    ("10m EMA(250)/EMA(1575) Skip=T [ETH동일]", c10, h10, l10, fm250_10, sm1575_10, len(c10), 2.0, 54.0, 8.0, 18, True, 1575, 0.20),
    ("10m EMA(250)/EMA(1575) Skip=F", c10, h10, l10, fm250_10, sm1575_10, len(c10), 2.0, 54.0, 8.0, 18, False, 1575, 0.20),
    ("10m EMA(100)/EMA(500) Skip=T", c10, h10, l10, fm100_10, sm500_10, len(c10), 2.0, 54.0, 8.0, 18, True, 500, 0.20),
    ("30m EMA(83)/EMA(525) Skip=T [10m등가]", c30, h30, l30, fm83_30, sm525_30, len(c30), 2.0, 54.0, 8.0, 6, True, 525, 0.20),
    ("30m EMA(83)/EMA(525) Skip=F", c30, h30, l30, fm83_30, sm525_30, len(c30), 2.0, 54.0, 8.0, 6, False, 525, 0.20),
    ("30m EMA(100)/EMA(600) Skip=T [v32.2 MA]", c30, h30, l30, fm100_30, sm600_30, len(c30), 2.0, 54.0, 8.0, 24, True, 600, 0.20),
    ("30m EMA(100)/EMA(600) Skip=F", c30, h30, l30, fm100_30, sm600_30, len(c30), 2.0, 54.0, 8.0, 24, False, 600, 0.20),
]

for label, cc, hh, ll, ff, ss, nn, sl, ta, tsl, mon, skip, wu, mpct in tests:
    r = bt(cc, hh, ll, ff, ss, nn, sl, ta, tsl, mon, skip, wu, mpct)
    print(f"  {label:>45} ${r[0]:>11,.0f} {r[1]:>+7,.0f}% {r[8]:>6.2f} {r[9]:>5.1f}% {r[2]:>4} {r[3]:>4} {r[4]:>4} {r[11]:>5.1f}:1")

print()
print("  * 모든 전략: Filter ALL OFF, SL 2%, TA 54%, TSL 8%, 마진 20%, 10x")
print("=" * 90)
