"""
v23.4 Maximum Return Search
Same winning strategy (EMA3/EMA200 30m ADX14>=35), push margin/leverage to max
"""
import pandas as pd, numpy as np, os, time, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
DIR = r"D:\filesystem\futures\btc_V1\test3"
FEE = 0.0004; INIT = 5000.0

def wilder(arr, p):
    out = np.full(len(arr), np.nan); s = 0
    while s < len(arr) and np.isnan(arr[s]): s += 1
    if s+p > len(arr): return out
    out[s+p-1] = np.nanmean(arr[s:s+p])
    for i in range(s+p, len(arr)):
        if not np.isnan(arr[i]) and not np.isnan(out[i-1]):
            out[i] = (out[i-1]*(p-1) + arr[i]) / p
    return out

def adx_w(h, l, c, p=14):
    n=len(h); tr=np.full(n,np.nan); pdm=np.full(n,np.nan); mdm=np.full(n,np.nan)
    for i in range(1,n):
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
        up=h[i]-h[i-1];dn=l[i-1]-l[i]
        pdm[i]=up if(up>dn and up>0)else 0;mdm[i]=dn if(dn>up and dn>0)else 0
    atr=wilder(tr,p);sp=wilder(pdm,p);sm=wilder(mdm,p)
    pdi=np.full(n,np.nan);mdi=np.full(n,np.nan);dx=np.full(n,np.nan)
    for i in range(n):
        if not np.isnan(atr[i]) and atr[i]>0:
            pdi[i]=100*sp[i]/atr[i];mdi[i]=100*sm[i]/atr[i]
            s2=pdi[i]+mdi[i];dx[i]=100*abs(pdi[i]-mdi[i])/s2 if s2>0 else 0
    return wilder(dx,p)

def ema_np(c, p):
    out=np.full(len(c),np.nan);s=0
    while s<len(c) and np.isnan(c[s]):s+=1
    if s>=len(c):return out
    out[s]=c[s];m=2.0/(p+1)
    for i in range(s+1,len(c)):
        if not np.isnan(c[i]) and not np.isnan(out[i-1]):
            out[i]=c[i]*m+out[i-1]*(1-m)
    return out

def rsi_np(c, p=14):
    n=len(c);g=np.zeros(n);l2=np.zeros(n)
    for i in range(1,n):
        d=c[i]-c[i-1]
        if d>0:g[i]=d
        else:l2[i]=-d
    ag=wilder(g,p);al=wilder(l2,p)
    out=np.full(n,50.0)
    for i in range(n):
        if not np.isnan(ag[i]) and not np.isnan(al[i]):
            out[i]=100.0 if al[i]==0 else 100-100/(1+ag[i]/al[i])
    return out

def run(hi,lo,cl,fast,slow,adx,rsi,margin,lev,sl,tact,tw):
    n=len(cl);bal=INIT;pkb=INIT;pos=0;ep=0;ei=0;hs=0;ls=0;tsl=0;ta=False
    trades=0;wins=0;gp=0;gl=0;slh=0;mdd=0;warmup=250
    for i in range(warmup,n):
        h=hi[i];l=lo[i];c=cl[i]
        if pos!=0:
            if pos==1:
                hs=max(hs,h)
                if l<=ep*(1-sl):
                    xp=ep*(1-sl);ntl=margin*lev*bal_e
                    pnl=ntl*(xp-ep)/ep-ntl*FEE
                    bal+=pnl;pkb=max(pkb,bal);mdd=min(mdd,(bal-pkb)/pkb if pkb>0 else 0)
                    if pnl>0:gp+=pnl;wins+=1
                    else:gl+=abs(pnl)
                    trades+=1;slh+=1;pos=0;continue
                roi=(c-ep)/ep
                if roi>=tact:ta=True;nt=hs*(1-tw);tsl=max(tsl,nt)
                if ta and c<=tsl:
                    ntl=margin*lev*bal_e;pnl=ntl*(c-ep)/ep-ntl*FEE
                    bal+=pnl;pkb=max(pkb,bal);mdd=min(mdd,(bal-pkb)/pkb if pkb>0 else 0)
                    if pnl>0:gp+=pnl;wins+=1
                    else:gl+=abs(pnl)
                    trades+=1;pos=0;continue
            else:
                ls=min(ls,l)
                if h>=ep*(1+sl):
                    xp=ep*(1+sl);ntl=margin*lev*bal_e
                    pnl=ntl*(ep-xp)/ep-ntl*FEE
                    bal+=pnl;pkb=max(pkb,bal);mdd=min(mdd,(bal-pkb)/pkb if pkb>0 else 0)
                    if pnl>0:gp+=pnl;wins+=1
                    else:gl+=abs(pnl)
                    trades+=1;slh+=1;pos=0;continue
                roi=(ep-c)/ep
                if roi>=tact:ta=True;nt=ls*(1+tw);tsl=min(tsl,nt)
                if ta and c>=tsl:
                    ntl=margin*lev*bal_e;pnl=ntl*(ep-c)/ep-ntl*FEE
                    bal+=pnl;pkb=max(pkb,bal);mdd=min(mdd,(bal-pkb)/pkb if pkb>0 else 0)
                    if pnl>0:gp+=pnl;wins+=1
                    else:gl+=abs(pnl)
                    trades+=1;pos=0;continue
        if pos==0 and i>warmup+1:
            if np.isnan(fast[i]) or np.isnan(slow[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):continue
            cu=fast[i-1]<=slow[i-1] and fast[i]>slow[i]
            cd=fast[i-1]>=slow[i-1] and fast[i]<slow[i]
            if not cu and not cd:continue
            if adx[i]<35:continue
            if not(30<=rsi[i]<=65):continue
            sig=1 if cu else -1
            if pos==sig:continue
            if pos!=0 and pos!=sig:
                ntl=margin*lev*bal_e
                if pos==1:pnl=ntl*(c-ep)/ep-ntl*FEE
                else:pnl=ntl*(ep-c)/ep-ntl*FEE
                bal+=pnl;pkb=max(pkb,bal);mdd=min(mdd,(bal-pkb)/pkb if pkb>0 else 0)
                if pnl>0:gp+=pnl;wins+=1
                else:gl+=abs(pnl)
                trades+=1;pos=0
            ntl=margin*lev*bal
            if ntl<10:continue
            bal-=ntl*FEE;bal_e=bal
            pos=sig;ep=c;ei=i;hs=c;ls=c;ta=False
            tsl=c*(1-sl) if sig==1 else c*(1+sl)
        if bal<=0:break
    pf=gp/gl if gl>0 else 999
    ret=(bal-INIT)/INIT*100
    return bal,ret,pf,mdd*100,trades,wins,slh

print("="*120)
print("  v23.4 MAXIMUM RETURN SEARCH - EMA(3)/EMA(200) 30m ADX>=35")
print("="*120)

t0=time.time()
fs=[os.path.join(DIR,f"btc_usdt_5m_2020_to_now_part{i}.csv") for i in range(1,4)]
df=pd.concat([pd.read_csv(f,parse_dates=['timestamp']) for f in fs],ignore_index=True)
df.sort_values('timestamp',inplace=True);df.set_index('timestamp',inplace=True)
d30=df.resample('30min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
hi=d30['high'].values;lo=d30['low'].values;cl=d30['close'].values
fast=ema_np(cl,3);slow=ema_np(cl,200);adx=adx_w(hi,lo,cl,14);rsi=rsi_np(cl,14)
print(f"Data: {len(d30):,} 30m bars | {time.time()-t0:.1f}s\n")

# Test ALL margin/leverage/SL/trail combos
configs = []
for m in [0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.80]:
    for lv in [5,7,10,12,15,20]:
        for sl in [0.05,0.06,0.07,0.08,0.09,0.10]:
            for ta in [0.03,0.04,0.05,0.06,0.08,0.10]:
                for tw in [0.01,0.02,0.03,0.04,0.05]:
                    configs.append((m,lv,sl,ta,tw))

print(f"Testing {len(configs):,} combinations...")
results = []
for i,(m,lv,sl,ta,tw) in enumerate(configs):
    bal,ret,pf,mdd,trd,wins,slh = run(hi,lo,cl,fast,slow,adx,rsi,m,lv,sl,ta,tw)
    if trd >= 5 and bal > 0:
        results.append({'margin':m,'lev':lv,'sl':sl,'trail_act':ta,'trail_w':tw,
                       'final':bal,'return':ret,'pf':pf,'mdd':mdd,'trades':trd,'wins':wins,'sl_hits':slh,
                       'wr':wins/trd*100,'sl_loss':m*lv*sl*100})
    if (i+1)%10000==0:
        print(f"  {i+1:,}/{len(configs):,}...")

print(f"\nValid: {len(results):,} | Time: {time.time()-t0:.1f}s")

# Sort by return
results.sort(key=lambda x:-x['return'])

print(f"\n{'='*120}")
print(f"  TOP 30 BY RETURN (Initial $5,000)")
print(f"{'='*120}")
print(f"  {'#':>3} {'Final$':>12} {'Return%':>10} {'PF':>6} {'MDD%':>7} {'Trd':>4} {'WR%':>5} {'SL':>3} {'M%':>4} {'Lev':>4} {'SL%':>5} {'T.Act':>5} {'T.W':>4} {'SL1Hit%':>8}")
print("  "+"-"*115)

for i,r in enumerate(results[:30]):
    pfs=f"{r['pf']:.2f}" if r['pf']<999 else "INF"
    print(f"  {i+1:>3} ${r['final']:>10,.0f} {r['return']:>+9,.0f}% {pfs:>5} {r['mdd']:>6.1f}% {r['trades']:>3} {r['wr']:>4.0f}% {r['sl_hits']:>3} {r['margin']*100:>3.0f}% {r['lev']:>3}x {r['sl']*100:>4.0f}% {r['trail_act']*100:>4.0f}% {r['trail_w']*100:>3.0f}% {r['sl_loss']:>7.0f}%")

# Best PF with high return
print(f"\n{'='*120}")
print(f"  BEST PF (return > 10,000%)")
print(f"{'='*120}")
high_ret = [r for r in results if r['return']>10000]
if high_ret:
    high_ret.sort(key=lambda x:-x['pf'])
    for i,r in enumerate(high_ret[:10]):
        pfs=f"{r['pf']:.2f}" if r['pf']<999 else "INF"
        print(f"  {i+1:>3} ${r['final']:>10,.0f} {r['return']:>+9,.0f}% PF:{pfs:>5} MDD:{r['mdd']:>5.1f}% Trd:{r['trades']:>3} M:{r['margin']*100:.0f}% L:{r['lev']}x SL:{r['sl']*100:.0f}% T:{r['trail_act']*100:.0f}/{r['trail_w']*100:.0f}")

# Best balance of return and MDD
print(f"\n{'='*120}")
print(f"  BEST RETURN/MDD RATIO (return > 5,000%)")
print(f"{'='*120}")
good = [r for r in results if r['return']>5000 and r['mdd']<0]
if good:
    good.sort(key=lambda x: x['return']/abs(x['mdd']), reverse=True)
    for i,r in enumerate(good[:10]):
        ratio = r['return']/abs(r['mdd'])
        pfs=f"{r['pf']:.2f}" if r['pf']<999 else "INF"
        print(f"  {i+1:>3} Ratio:{ratio:>5.0f} Ret:{r['return']:>+8,.0f}% MDD:{r['mdd']:>5.1f}% PF:{pfs} Trd:{r['trades']:>3} M:{r['margin']*100:.0f}% L:{r['lev']}x SL:{r['sl']*100:.0f}% T:{r['trail_act']*100:.0f}/{r['trail_w']*100:.0f}")

print(f"\n{'='*120}")
print(f"  MARGIN IMPACT (SL-7%, Trail+6/-3, Lev10x)")
print(f"{'='*120}")
print(f"  {'Margin':>7} {'Final$':>12} {'Return%':>10} {'PF':>6} {'MDD%':>7} {'SL1HitLoss':>11}")
print("  "+"-"*60)
for m in [0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.80]:
    bal,ret,pf,mdd,trd,wins,slh = run(hi,lo,cl,fast,slow,adx,rsi,m,10,0.07,0.06,0.03)
    sl1 = m*10*0.07*100
    pfs=f"{pf:.2f}" if pf<999 else "INF"
    marker = " <-- v23.4" if abs(m-0.30)<0.01 else ""
    print(f"  {m*100:>5.0f}%  ${bal:>10,.0f} {ret:>+9,.0f}% {pfs:>5} {mdd:>6.1f}% {sl1:>9.0f}%{marker}")

print(f"\n{'='*120}")
print("COMPLETE")
print("="*120)
