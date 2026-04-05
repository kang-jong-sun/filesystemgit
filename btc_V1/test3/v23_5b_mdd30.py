"""v23.5b: MDD<=30%, PF>=2, M<=25%, L<=15x, $5,000"""
import pandas as pd,numpy as np,os,time,sys,io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
DIR=r"D:\filesystem\futures\btc_V1\test3"
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
def sma_np(c,p):
    o=np.full(len(c),np.nan)
    for i in range(p-1,len(c)):o[i]=np.mean(c[i-p+1:i+1])
    return o
def wma_np(c,p):
    o=np.full(len(c),np.nan);w=np.arange(1,p+1,dtype=float);ws=w.sum()
    for i in range(p-1,len(c)):
        sl=c[i-p+1:i+1]
        if not np.any(np.isnan(sl)):o[i]=np.dot(sl,w)/ws
    return o
def hma_np(c,p):
    h2=max(int(p/2),1);sq=max(int(np.sqrt(p)),1)
    wh=wma_np(c,h2);wf=wma_np(c,p)
    d=np.full(len(c),np.nan)
    for i in range(len(c)):
        if not np.isnan(wh[i]) and not np.isnan(wf[i]):d[i]=2*wh[i]-wf[i]
    return wma_np(d,sq)
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

def bt(hi,lo,cl,fast,slow,adx,rsi,M,L,SL,TA,TW,ADX_TH,RSI_LO,RSI_HI,DELAY):
    n=len(cl);bal=INIT;pkb=INIT;pos=0;ep=0;hs=0;ls=0;tsl=0;ta2=False;be=INIT
    trd=0;wins=0;gp=0;gl=0;slh=0;mdd=0;w=max(300,DELAY+10)
    pending=0;pbar=0
    for i in range(w,n):
        h=hi[i];l=lo[i];c=cl[i]
        if pos!=0:
            if pos==1:
                hs=max(hs,h)
                if l<=ep*(1-SL):
                    nt=M*L*be;pn=nt*(ep*(1-SL)-ep)/ep-nt*FEE
                    bal+=pn;pkb=max(pkb,bal);dd=(bal-pkb)/pkb if pkb>0 else 0;mdd=min(mdd,dd)
                    gl+=abs(pn);trd+=1;slh+=1;pos=0;continue
                r=(c-ep)/ep
                if r>=TA:ta2=True;nt2=hs*(1-TW);tsl=max(tsl,nt2)
                if ta2 and c<=tsl:
                    nt=M*L*be;pn=nt*(c-ep)/ep-nt*FEE
                    bal+=pn;pkb=max(pkb,bal);dd=(bal-pkb)/pkb if pkb>0 else 0;mdd=min(mdd,dd)
                    if pn>0:gp+=pn;wins+=1
                    else:gl+=abs(pn)
                    trd+=1;pos=0;continue
            else:
                ls=min(ls,l)
                if h>=ep*(1+SL):
                    nt=M*L*be;pn=nt*(ep-ep*(1+SL))/ep-nt*FEE
                    bal+=pn;pkb=max(pkb,bal);dd=(bal-pkb)/pkb if pkb>0 else 0;mdd=min(mdd,dd)
                    gl+=abs(pn);trd+=1;slh+=1;pos=0;continue
                r=(ep-c)/ep
                if r>=TA:ta2=True;nt2=ls*(1+TW);tsl=min(tsl,nt2)
                if ta2 and c>=tsl:
                    nt=M*L*be;pn=nt*(ep-c)/ep-nt*FEE
                    bal+=pn;pkb=max(pkb,bal);dd=(bal-pkb)/pkb if pkb>0 else 0;mdd=min(mdd,dd)
                    if pn>0:gp+=pn;wins+=1
                    else:gl+=abs(pn)
                    trd+=1;pos=0;continue
        # Signal detection
        if i>w+1 and not np.isnan(fast[i]) and not np.isnan(fast[i-1]) and not np.isnan(slow[i]) and not np.isnan(slow[i-1]):
            cu=fast[i-1]<=slow[i-1] and fast[i]>slow[i]
            cd=fast[i-1]>=slow[i-1] and fast[i]<slow[i]
            if cu:pending=1;pbar=i
            elif cd:pending=-1;pbar=i
        # Delayed entry
        if pending!=0 and i-pbar==DELAY:
            sig=pending;pending=0
            if np.isnan(fast[i]) or np.isnan(slow[i]):continue
            if sig==1 and fast[i]<=slow[i]:continue
            if sig==-1 and fast[i]>=slow[i]:continue
            if np.isnan(adx[i]) or adx[i]<ADX_TH:continue
            if np.isnan(rsi[i]) or not(RSI_LO<=rsi[i]<=RSI_HI):continue
            if pos==sig:continue
            if pos!=0 and pos!=sig:
                nt=M*L*be;pn=nt*((c-ep)/ep if pos==1 else(ep-c)/ep)-nt*FEE
                bal+=pn;pkb=max(pkb,bal);dd=(bal-pkb)/pkb if pkb>0 else 0;mdd=min(mdd,dd)
                if pn>0:gp+=pn;wins+=1
                else:gl+=abs(pn)
                trd+=1;pos=0
            nt=M*L*bal
            if nt<10 or bal<=0:continue
            bal-=nt*FEE;be=bal;pos=sig;ep=c;hs=c;ls=c;ta2=False
            tsl=c*(1-SL) if sig==1 else c*(1+SL)
        if bal<=0:break
    pf=gp/gl if gl>0 else 999;ret=(bal-INIT)/INIT*100
    return bal,ret,pf,mdd*100,trd,wins,slh

# Load data
print("="*110)
print("  v23.5b: MDD<=30%, M<=25%, L<=15x, $5,000")
print("="*110)
t0=time.time()
fs=[os.path.join(DIR,f"btc_usdt_5m_2020_to_now_part{i}.csv") for i in range(1,4)]
df=pd.concat([pd.read_csv(f,parse_dates=['timestamp']) for f in fs],ignore_index=True)
df.sort_values('timestamp',inplace=True);df.set_index('timestamp',inplace=True)

tfs={}
for rule,key in[('10min','10m'),('15min','15m'),('30min','30m'),('1h','1h')]:
    d=df.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    tfs[key]=d
print(f"Data: {' | '.join(f'{k}:{len(v):,}' for k,v in tfs.items())} | {time.time()-t0:.1f}s")

# Pre-compute indicators
print("[Indicators]...")
t0=time.time()
ind={}
for tf_key,d in tfs.items():
    hi=d['high'].values;lo=d['low'].values;cl=d['close'].values
    ind[tf_key]={'hi':hi,'lo':lo,'cl':cl,
        'ema3':ema_np(cl,3),'ema5':ema_np(cl,5),'ema7':ema_np(cl,7),'ema10':ema_np(cl,10),
        'wma3':wma_np(cl,3),'hma3':hma_np(cl,3),'hma5':hma_np(cl,5),
        'ema100':ema_np(cl,100),'ema150':ema_np(cl,150),'ema200':ema_np(cl,200),'ema250':ema_np(cl,250),
        'sma200':sma_np(cl,200),'sma300':sma_np(cl,300),
        'adx14':adx_w(hi,lo,cl,14),'adx20':adx_w(hi,lo,cl,20),
        'rsi':rsi_np(cl,14)}
print(f"Done | {time.time()-t0:.1f}s")

# Random search
import random
random.seed(42)

fast_opts=[('ema3','ema3'),('ema5','ema5'),('ema7','ema7'),('ema10','ema10'),('wma3','wma3'),('hma3','hma3'),('hma5','hma5')]
slow_opts=[('ema100','ema100'),('ema150','ema150'),('ema200','ema200'),('ema250','ema250'),('sma200','sma200'),('sma300','sma300')]
tf_opts=['10m','15m','30m','1h']
adx_p_opts=['adx14','adx20']
adx_th_opts=[25,30,35,40,45]
rsi_opts=[(25,65),(30,65),(30,70),(35,65),(35,70),(40,75)]
delay_opts=[0,1,2,3,5]
sl_opts=[0.03,0.04,0.05,0.06,0.07,0.08,0.10]
ta_opts=[0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.15]
tw_opts=[0.01,0.02,0.03,0.04,0.05]
m_opts=[0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25]
l_opts=[3,5,7,10,12,15]

N=1500000
results=[]
print(f"\n[Optimization] {N:,} random combos...")
t0=time.time()

for idx in range(N):
    fk,fn=random.choice(fast_opts)
    sk,sn=random.choice(slow_opts)
    tf=random.choice(tf_opts)
    ap=random.choice(adx_p_opts)
    ath=random.choice(adx_th_opts)
    rlo,rhi=random.choice(rsi_opts)
    dly=random.choice(delay_opts)
    sl=random.choice(sl_opts)
    ta=random.choice(ta_opts)
    tw=random.choice(tw_opts)
    mg=random.choice(m_opts)
    lv=random.choice(l_opts)

    I=ind[tf]
    b,r,p,d,t,w2,s=bt(I['hi'],I['lo'],I['cl'],I[fk],I[sk],I[ap],I['rsi'],mg,lv,sl,ta,tw,ath,rlo,rhi,dly)

    if t>=5 and b>0 and p>1.0 and d>-35:
        results.append({'fast':fn,'slow':sn,'tf':tf,'adx_p':ap,'adx_th':ath,
            'rsi':f"{rlo}-{rhi}",'delay':dly,'sl':sl,'ta':ta,'tw':tw,'margin':mg,'lev':lv,
            'final':b,'return':r,'pf':p,'mdd':d,'trades':t,'wins':w2,'wr':round(w2/t*100,1),'sl_hits':s})

    if (idx+1)%250000==0:
        elapsed=time.time()-t0
        valid=len(results)
        print(f"  {idx+1:>10,}/{N:,} | Valid:{valid:,} | {elapsed:.0f}s | {(idx+1)/elapsed:.0f}/s")

elapsed=time.time()-t0
print(f"  Done: {len(results):,} valid | {elapsed:.0f}s")

# Filter and sort
mdd30=[r for r in results if r['mdd']>=-30]
mdd30.sort(key=lambda x:-x['return'])

print(f"\n{'='*110}")
print(f"  MDD <= 30% RESULTS: {len(mdd30):,} combos")
print(f"{'='*110}")

print(f"\n  TOP 30 BY RETURN (MDD<=30%)")
print(f"  {'#':>3} {'Final$':>10} {'Ret%':>8} {'PF':>5} {'MDD%':>6} {'Trd':>5} {'WR%':>5} {'SL':>3} {'Fast':>6} {'Slow':>7} {'TF':>4} {'ADX':>6} {'D':>2} {'SL%':>4} {'TA%':>4} {'TW%':>4} {'M%':>4} {'Lv':>3}")
print("  "+"-"*115)
for i,r in enumerate(mdd30[:30]):
    pfs=f"{r['pf']:.1f}" if r['pf']<999 else "INF"
    print(f"  {i+1:>3} ${r['final']:>8,.0f} {r['return']:>+7.0f}% {pfs:>4} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.0f}% {r['sl_hits']:>3} {r['fast']:>6} {r['slow']:>7} {r['tf']:>4} {r['adx_p'][-2:]}/{r['adx_th']:>2} {r['delay']:>2} {r['sl']*100:>3.0f}% {r['ta']*100:>3.0f}% {r['tw']*100:>3.0f}% {r['margin']*100:>3.0f}% {r['lev']:>2}x")

# PF>=2 and MDD<=30%
pf2=[r for r in mdd30 if r['pf']>=2]
pf2.sort(key=lambda x:-x['return'])
print(f"\n  TOP 20 WITH PF>=2 AND MDD<=30% ({len(pf2):,} combos)")
print(f"  {'#':>3} {'Final$':>10} {'Ret%':>8} {'PF':>5} {'MDD%':>6} {'Trd':>5} {'WR%':>5} {'Fast':>6} {'Slow':>7} {'TF':>4} {'ADX':>6} {'D':>2} {'SL%':>4} {'TA%':>4} {'TW%':>4} {'M%':>4} {'Lv':>3}")
print("  "+"-"*115)
for i,r in enumerate(pf2[:20]):
    pfs=f"{r['pf']:.1f}" if r['pf']<999 else "INF"
    print(f"  {i+1:>3} ${r['final']:>8,.0f} {r['return']:>+7.0f}% {pfs:>4} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.0f}% {r['fast']:>6} {r['slow']:>7} {r['tf']:>4} {r['adx_p'][-2:]}/{r['adx_th']:>2} {r['delay']:>2} {r['sl']*100:>3.0f}% {r['ta']*100:>3.0f}% {r['tw']*100:>3.0f}% {r['margin']*100:>3.0f}% {r['lev']:>2}x")

# MDD<=20% best
mdd20=[r for r in results if r['mdd']>=-20]
mdd20.sort(key=lambda x:-x['return'])
print(f"\n  TOP 10 WITH MDD<=20% ({len(mdd20):,} combos)")
print(f"  {'#':>3} {'Final$':>10} {'Ret%':>8} {'PF':>5} {'MDD%':>6} {'Trd':>5}")
print("  "+"-"*50)
for i,r in enumerate(mdd20[:10]):
    pfs=f"{r['pf']:.1f}" if r['pf']<999 else "INF"
    print(f"  {i+1:>3} ${r['final']:>8,.0f} {r['return']:>+7.0f}% {pfs:>4} {r['mdd']:>5.1f}% {r['trades']:>4}")

# Parameter frequency top 100
print(f"\n  PARAMETER FREQUENCY (Top 100 by return, MDD<=30%)")
from collections import Counter
top100=mdd30[:100]
for key in['fast','slow','tf','adx_th','delay','sl','ta','tw','margin','lev']:
    vals=[r[key] for r in top100]
    if key in['sl','ta','tw','margin']:vals=[f"{v*100:.0f}%" for v in vals]
    elif key=='lev':vals=[f"{v}x" for v in vals]
    ctr=Counter(vals).most_common(5)
    items=" | ".join(f"{v}:{c}" for v,c in ctr)
    print(f"  {key:>8}: {items}")

# Save top 100
pd.DataFrame(mdd30[:100]).to_csv(os.path.join(DIR,"v23_5b_top100.csv"),index=False)

print(f"\n{'='*110}")
print("COMPLETE")
print("="*110)
