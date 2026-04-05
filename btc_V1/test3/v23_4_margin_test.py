"""Quick margin impact test - same strategy, vary margin 20-80%"""
import pandas as pd, numpy as np, os, time, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
DIR = r"D:\filesystem\futures\btc_V1\test3"
FEE=0.0004;INIT=5000.0

def wilder(a,p):
    o=np.full(len(a),np.nan);s=0
    while s<len(a) and np.isnan(a[s]):s+=1
    if s+p>len(a):return o
    o[s+p-1]=np.nanmean(a[s:s+p])
    for i in range(s+p,len(a)):
        if not np.isnan(a[i]) and not np.isnan(o[i-1]):o[i]=(o[i-1]*(p-1)+a[i])/p
    return o
def adx_w(h,l,c,p=14):
    n=len(h);tr=np.full(n,np.nan);pd2=np.full(n,np.nan);md=np.full(n,np.nan)
    for i in range(1,n):
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
        u=h[i]-h[i-1];d=l[i-1]-l[i]
        pd2[i]=u if(u>d and u>0)else 0;md[i]=d if(d>u and d>0)else 0
    at=wilder(tr,p);sp=wilder(pd2,p);sm=wilder(md,p)
    pi2=np.full(n,np.nan);mi=np.full(n,np.nan);dx=np.full(n,np.nan)
    for i in range(n):
        if not np.isnan(at[i]) and at[i]>0:
            pi2[i]=100*sp[i]/at[i];mi[i]=100*sm[i]/at[i]
            s2=pi2[i]+mi[i];dx[i]=100*abs(pi2[i]-mi[i])/s2 if s2>0 else 0
    return wilder(dx,p)
def ema_np(c,p):
    o=np.full(len(c),np.nan);s=0
    while s<len(c) and np.isnan(c[s]):s+=1
    if s>=len(c):return o
    o[s]=c[s];m=2.0/(p+1)
    for i in range(s+1,len(c)):
        if not np.isnan(c[i]) and not np.isnan(o[i-1]):o[i]=c[i]*m+o[i-1]*(1-m)
    return o
def rsi_np(c,p=14):
    n=len(c);g=np.zeros(n);l2=np.zeros(n)
    for i in range(1,n):
        d=c[i]-c[i-1]
        if d>0:g[i]=d
        else:l2[i]=-d
    ag=wilder(g,p);al=wilder(l2,p);o=np.full(n,50.0)
    for i in range(n):
        if not np.isnan(ag[i]) and not np.isnan(al[i]):
            o[i]=100.0 if al[i]==0 else 100-100/(1+ag[i]/al[i])
    return o

def bt(hi,lo,cl,fast,slow,adx,rsi,M,L,SL,TA,TW,ADX_TH=35,RSI_LO=30,RSI_HI=65):
    n=len(cl);bal=INIT;pkb=INIT;pos=0;ep=0;hs=0;ls=0;tsl=0;ta2=False
    trd=0;wins=0;gp=0;gl=0;slh=0;mdd=0;be=INIT;w=250
    for i in range(w,n):
        h=hi[i];l=lo[i];c=cl[i]
        if pos!=0:
            if pos==1:
                hs=max(hs,h)
                if l<=ep*(1-SL):
                    xp=ep*(1-SL);nt=M*L*be;pn=nt*(xp-ep)/ep-nt*FEE
                    bal+=pn;pkb=max(pkb,bal);mdd=min(mdd,(bal-pkb)/pkb if pkb>0 else 0)
                    if pn>0:gp+=pn;wins+=1
                    else:gl+=abs(pn)
                    trd+=1;slh+=1;pos=0;continue
                r=(c-ep)/ep
                if r>=TA:ta2=True;nt2=hs*(1-TW);tsl=max(tsl,nt2)
                if ta2 and c<=tsl:
                    nt=M*L*be;pn=nt*(c-ep)/ep-nt*FEE
                    bal+=pn;pkb=max(pkb,bal);mdd=min(mdd,(bal-pkb)/pkb if pkb>0 else 0)
                    if pn>0:gp+=pn;wins+=1
                    else:gl+=abs(pn)
                    trd+=1;pos=0;continue
            else:
                ls=min(ls,l)
                if h>=ep*(1+SL):
                    xp=ep*(1+SL);nt=M*L*be;pn=nt*(ep-xp)/ep-nt*FEE
                    bal+=pn;pkb=max(pkb,bal);mdd=min(mdd,(bal-pkb)/pkb if pkb>0 else 0)
                    if pn>0:gp+=pn;wins+=1
                    else:gl+=abs(pn)
                    trd+=1;slh+=1;pos=0;continue
                r=(ep-c)/ep
                if r>=TA:ta2=True;nt2=ls*(1+TW);tsl=min(tsl,nt2)
                if ta2 and c>=tsl:
                    nt=M*L*be;pn=nt*(ep-c)/ep-nt*FEE
                    bal+=pn;pkb=max(pkb,bal);mdd=min(mdd,(bal-pkb)/pkb if pkb>0 else 0)
                    if pn>0:gp+=pn;wins+=1
                    else:gl+=abs(pn)
                    trd+=1;pos=0;continue
        if pos==0 and i>w+1:
            if np.isnan(fast[i]) or np.isnan(slow[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):continue
            cu=fast[i-1]<=slow[i-1] and fast[i]>slow[i];cd=fast[i-1]>=slow[i-1] and fast[i]<slow[i]
            if not cu and not cd:continue
            if adx[i]<ADX_TH:continue
            if not(RSI_LO<=rsi[i]<=RSI_HI):continue
            sig=1 if cu else -1
            if pos==sig:continue
            if pos!=0 and pos!=sig:
                nt=M*L*be
                pn=nt*((c-ep)/ep if pos==1 else(ep-c)/ep)-nt*FEE
                bal+=pn;pkb=max(pkb,bal);mdd=min(mdd,(bal-pkb)/pkb if pkb>0 else 0)
                if pn>0:gp+=pn;wins+=1
                else:gl+=abs(pn)
                trd+=1;pos=0
            nt=M*L*bal
            if nt<10:continue
            bal-=nt*FEE;be=bal;pos=sig;ep=c;hs=c;ls=c;ta2=False
            tsl=c*(1-SL) if sig==1 else c*(1+SL)
        if bal<=0:break
    pf=gp/gl if gl>0 else 999;ret=(bal-INIT)/INIT*100
    return bal,ret,pf,mdd*100,trd,wins,slh

# Load
t0=time.time()
fs=[os.path.join(DIR,f"btc_usdt_5m_2020_to_now_part{i}.csv") for i in range(1,4)]
df=pd.concat([pd.read_csv(f,parse_dates=['timestamp']) for f in fs],ignore_index=True)
df.sort_values('timestamp',inplace=True);df.set_index('timestamp',inplace=True)
d30=df.resample('30min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
hi=d30['high'].values;lo=d30['low'].values;cl=d30['close'].values
fast=ema_np(cl,3);slow=ema_np(cl,200);adx=adx_w(hi,lo,cl,14);rsi=rsi_np(cl,14)
print(f"Data loaded: {len(d30):,} bars | {time.time()-t0:.1f}s\n")

print("="*120)
print("  MARGIN IMPACT ON RETURN - EMA(3)/EMA(200) 30m ADX>=35 RSI30-65")
print("="*120)

# Test 1: Margin sweep with fixed SL-7% Trail+6/-3 Lev10x
print(f"\n  [TEST 1] SL-7% Trail+6/-3 Lev10x")
print(f"  {'Margin':>7} {'Final$':>14} {'Return%':>12} {'PF':>6} {'MDD%':>7} {'Trd':>4} {'WR%':>5} {'SL':>3} {'SL1Loss':>8}")
print("  "+"-"*80)
for m in [0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.80]:
    b,r,p,d,t,w2,s = bt(hi,lo,cl,fast,slow,adx,rsi,m,10,0.07,0.06,0.03)
    pfs=f"{p:.2f}" if p<999 else "INF"
    print(f"  {m*100:>5.0f}%  ${b:>12,.0f} {r:>+11,.0f}% {pfs:>5} {d:>6.1f}% {t:>3} {w2/max(t,1)*100:>4.0f}% {s:>3} {m*10*0.07*100:>6.0f}%")

# Test 2: SL-8% Trail+6/-3 Lev10x
print(f"\n  [TEST 2] SL-8% Trail+6/-3 Lev10x")
print(f"  {'Margin':>7} {'Final$':>14} {'Return%':>12} {'PF':>6} {'MDD%':>7} {'SL1Loss':>8}")
print("  "+"-"*65)
for m in [0.30,0.40,0.50,0.60,0.70,0.80]:
    b,r,p,d,t,w2,s = bt(hi,lo,cl,fast,slow,adx,rsi,m,10,0.08,0.06,0.03)
    pfs=f"{p:.2f}" if p<999 else "INF"
    print(f"  {m*100:>5.0f}%  ${b:>12,.0f} {r:>+11,.0f}% {pfs:>5} {d:>6.1f}% {m*10*0.08*100:>6.0f}%")

# Test 3: Higher leverage
print(f"\n  [TEST 3] M50% SL-7% Trail+6/-3")
print(f"  {'Lev':>5} {'Final$':>14} {'Return%':>12} {'PF':>6} {'MDD%':>7} {'SL1Loss':>8}")
print("  "+"-"*55)
for lv in [5,7,10,12,15,20]:
    b,r,p,d,t,w2,s = bt(hi,lo,cl,fast,slow,adx,rsi,0.50,lv,0.07,0.06,0.03)
    pfs=f"{p:.2f}" if p<999 else "INF"
    print(f"  {lv:>3}x  ${b:>12,.0f} {r:>+11,.0f}% {pfs:>5} {d:>6.1f}% {0.50*lv*0.07*100:>6.0f}%")

# Test 4: Trail variants with M60% Lev10x SL-7%
print(f"\n  [TEST 4] M60% Lev10x SL-7%")
print(f"  {'Trail':>10} {'Final$':>14} {'Return%':>12} {'PF':>6} {'MDD%':>7}")
print("  "+"-"*55)
for ta,tw in [(0.03,0.01),(0.03,0.02),(0.04,0.02),(0.04,0.03),(0.05,0.03),(0.06,0.03),(0.08,0.03),(0.08,0.05),(0.10,0.05)]:
    b,r,p,d,t,w2,s = bt(hi,lo,cl,fast,slow,adx,rsi,0.60,10,0.07,ta,tw)
    pfs=f"{p:.2f}" if p<999 else "INF"
    print(f"  +{ta*100:.0f}%/-{tw*100:.0f}%  ${b:>12,.0f} {r:>+11,.0f}% {pfs:>5} {d:>6.1f}%")

# Test 5: ADX threshold impact with M60% Lev10x
print(f"\n  [TEST 5] ADX threshold | M60% Lev10x SL-7% Trail+6/-3")
print(f"  {'ADX>=':>7} {'Final$':>14} {'Return%':>12} {'PF':>6} {'MDD%':>7} {'Trd':>4}")
print("  "+"-"*55)
for ath in [25,30,35,40,45]:
    b,r,p,d,t,w2,s = bt(hi,lo,cl,fast,slow,adx,rsi,0.60,10,0.07,0.06,0.03,ADX_TH=ath)
    pfs=f"{p:.2f}" if p<999 else "INF"
    print(f"  >={ath:>3}  ${b:>12,.0f} {r:>+11,.0f}% {pfs:>5} {d:>6.1f}% {t:>3}")

print(f"\n{'='*120}")
print("DONE")
print("="*120)
