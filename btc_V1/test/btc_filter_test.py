# BTC Filter ON vs OFF Test
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8'); warnings.filterwarnings('ignore')

def ema_pd(s, p): return s.ewm(span=p, adjust=False).mean()
def calc_adx(h, l, c, period=20):
    n=len(c); pdm=np.zeros(n); mdm=np.zeros(n); tr=np.zeros(n)
    for i in range(1,n):
        hd=h[i]-h[i-1]; ld=l[i-1]-l[i]
        pdm[i]=hd if(hd>ld and hd>0)else 0; mdm[i]=ld if(ld>hd and ld>0)else 0
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    a=1.0/period
    atr=pd.Series(tr).ewm(alpha=a,min_periods=period,adjust=False).mean()
    sp=pd.Series(pdm).ewm(alpha=a,min_periods=period,adjust=False).mean()
    sn=pd.Series(mdm).ewm(alpha=a,min_periods=period,adjust=False).mean()
    pdi=100*sp/atr.replace(0,1e-10); mdi=100*sn/atr.replace(0,1e-10)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,1e-10)
    return dx.ewm(alpha=a,min_periods=period,adjust=False).mean().values.astype(np.float64)
def calc_rsi(c, period=10):
    d=np.diff(c,prepend=c[0]); g=np.where(d>0,d,0); lo=np.where(d<0,-d,0)
    a=1.0/period
    ag=pd.Series(g).ewm(alpha=a,min_periods=period,adjust=False).mean()
    al=pd.Series(lo).ewm(alpha=a,min_periods=period,adjust=False).mean()
    return (100-100/(1+ag/al.replace(0,1e-10))).values.astype(np.float64)

def bt_no_filter(c, h, l, fm, sm, n, SL, TA, TSL, MON, SKIP, WU, MPCT):
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

def bt_filtered(c, h, l, fm, sm, adx, rsi, n, SL, TA, TSL, MON, SKIP, WU, MPCT):
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
                if adx[i]<30: continue
                if i>=6 and adx[i]<=adx[i-6]: continue
                if rsi[i]<40 or rsi[i]>80: continue
                if sm[i]>0 and abs(fm[i]-sm[i])/sm[i]*100<0.2: continue
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
df30 = df5.set_index('timestamp').resample('30min').agg(agg).dropna().reset_index()
df10 = df5.set_index('timestamp').resample('10min').agg(agg).dropna().reset_index()
print(f"BTC 30m: {len(df30)} bars | 10m: {len(df10)} bars")

cs30=df30['close']; c30=cs30.values.astype(np.float64)
h30=df30['high'].values.astype(np.float64); l30=df30['low'].values.astype(np.float64)
fm100=ema_pd(cs30,100).values.astype(np.float64); sm600=ema_pd(cs30,600).values.astype(np.float64)
adx30=calc_adx(h30,l30,c30,20); rsi30=calc_rsi(c30,10)

cs10=df10['close']; c10=cs10.values.astype(np.float64)
h10=df10['high'].values.astype(np.float64); l10=df10['low'].values.astype(np.float64)
fm250=ema_pd(cs10,250).values.astype(np.float64); sm1575=ema_pd(cs10,1575).values.astype(np.float64)

print()
print("="*90)
print("  BTC Filter ON vs OFF (20% Margin, 10x)")
print("="*90)
header = f"  {'Strategy':>35} {'Final':>12} {'Ret%':>8} {'PF':>6} {'MDD':>6} {'T':>4} {'SL':>4} {'TSL':>4} {'WLR':>6}"
print(header)
print("  " + "-"*85)

tests = [
    ("v32.2 Filter ON (current)", bt_filtered(c30,h30,l30,fm100,sm600,adx30,rsi30,len(c30),3.0,12.0,9.0,24,True,600,0.20)),
    ("30m Filter OFF SL3/TA12/TSL9", bt_no_filter(c30,h30,l30,fm100,sm600,len(c30),3.0,12.0,9.0,24,True,600,0.20)),
    ("30m Filter OFF SL2/TA54/TSL8", bt_no_filter(c30,h30,l30,fm100,sm600,len(c30),2.0,54.0,8.0,24,True,600,0.20)),
    ("30m Filter OFF SL2/TA30/TSL9", bt_no_filter(c30,h30,l30,fm100,sm600,len(c30),2.0,30.0,9.0,24,False,600,0.20)),
    ("10m V8.16 EMA250/1575 OFF", bt_no_filter(c10,h10,l10,fm250,sm1575,len(c10),2.0,54.0,8.0,18,True,1575,0.20)),
]

for label, r in tests:
    print(f"  {label:>35} ${r[0]:>11,.0f} {r[1]:>+7,.0f}% {r[8]:>6.2f} {r[9]:>5.1f}% {r[2]:>4} {r[3]:>4} {r[4]:>4} {r[11]:>5.1f}:1")

print()
print("="*90)
