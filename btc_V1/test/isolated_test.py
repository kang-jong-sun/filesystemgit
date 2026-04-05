import sys;sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd,numpy as np
from numba import njit
np.seterr(all='ignore')
BASE="D:/filesystem/futures/btc_V1/test"

@njit
def bt_isolated(cl,hi,lo,ts,fv,sv,av,rv,am,ar,mn,rl,rh,sp,tp,wp,mp,lv,sk,ic,gp,st):
    """격리마진: 강제청산 시 마진만 손실"""
    n=len(cl);c=ic;p=0;ex=0.0;ps=0.0;sl=0.0;mg=0.0
    tn=False;th=0.0;tl=999999.0;pk=c;md=0.0;ms=c
    sc=0;tc=0;rc=0;fc=0;w=0;lo_=0;gpr=0.0;gls=0.0
    wa=0;ws=0;le=0;ld=0
    for i in range(st,n):
        px=cl[i];h_=hi[i];l_=lo[i]
        if i>st and i%1440==0:ms=c
        if p!=0:
            wa=0
            # 격리마진 강제청산 체크
            liq_pct=(1.0/lv)*0.98
            if p==1:
                liq_p=ex*(1-liq_pct)
                if l_<=liq_p:
                    pnl=-mg-ps*0.0004;c+=pnl;fc+=1;lo_+=1;gls+=abs(pnl)
                    ld=p;le=i;p=0;continue
            else:
                liq_p=ex*(1+liq_pct)
                if h_>=liq_p:
                    pnl=-mg-ps*0.0004;c+=pnl;fc+=1;lo_+=1;gls+=abs(pnl)
                    ld=p;le=i;p=0;continue
            if not tn:
                if(p==1 and l_<=sl)or(p==-1 and h_>=sl):
                    pnl=(sl-ex)/ex*ps*p-ps*0.0004;c+=pnl;sc+=1
                    if pnl>0:w+=1;gpr+=pnl
                    else:lo_+=1;gls+=abs(pnl)
                    ld=p;le=i;p=0;continue
            br=(h_-ex)/ex*100 if p==1 else(ex-l_)/ex*100
            if br>=tp:tn=True
            if tn:
                if p==1:
                    if h_>th:th=h_
                    ns=th*(1-wp/100)
                    if ns>sl:sl=ns
                    if px<=sl:
                        pnl=(px-ex)/ex*ps*p-ps*0.0004;c+=pnl;tc+=1
                        if pnl>0:w+=1;gpr+=pnl
                        else:lo_+=1;gls+=abs(pnl)
                        ld=p;le=i;p=0;continue
                else:
                    if l_<tl:tl=l_
                    ns=tl*(1+wp/100)
                    if ns<sl:sl=ns
                    if px>=sl:
                        pnl=(px-ex)/ex*ps*p-ps*0.0004;c+=pnl;tc+=1
                        if pnl>0:w+=1;gpr+=pnl
                        else:lo_+=1;gls+=abs(pnl)
                        ld=p;le=i;p=0;continue
            if i>0:
                bn=fv[i]>sv[i];bp=fv[i-1]>sv[i-1]
                cu=bn and not bp;cd=not bn and bp
                if(p==1 and cd)or(p==-1 and cu):
                    pnl=(px-ex)/ex*ps*p-ps*0.0004;c+=pnl;rc+=1
                    if pnl>0:w+=1;gpr+=pnl
                    else:lo_+=1;gls+=abs(pnl)
                    ld=p;le=i;p=0
        if i<1:continue
        bn=fv[i]>sv[i];bp=fv[i-1]>sv[i-1]
        cu=bn and not bp;cd=not bn and bp
        if p==0:
            if cu:wa=1;ws=i
            elif cd:wa=-1;ws=i
            if wa!=0 and(mn==0 or i>ws):
                if mn>0 and i-ws>mn:wa=0;continue
                if wa==1 and cd:wa=-1;ws=i;continue
                elif wa==-1 and cu:wa=1;ws=i;continue
                if sk and wa==ld:continue
                aok=av[i]>=am;arr=True
                if ar>0 and i>=ar:arr=av[i]>av[i-ar]
                rok=(rv[i]>=rl)and(rv[i]<=rh)
                gok=True
                if gp>0:
                    g=abs(fv[i]-sv[i])/sv[i]*100
                    if g<gp:gok=False
                if aok and arr and rok and gok and c>0:
                    if ms>0 and(c-ms)/ms<=-0.20:wa=0;continue
                    mg=c*mp;ps=mg*lv;c-=ps*0.0004
                    p=wa;ex=px;tn=False;th=px;tl=px
                    if p==1:sl=ex*(1-sp/100)
                    else:sl=ex*(1+sp/100)
                    pk=max(pk,c);wa=0
        pk=max(pk,c);dd=(pk-c)/pk if pk>0 else 0
        if dd>md:md=dd
        if c<=0:break
    if p!=0 and c>0:
        pnl=(cl[n-1]-ex)/ex*ps*p-ps*0.0004;c+=pnl
        if pnl>0:w+=1;gpr+=pnl
        else:lo_+=1;gls+=abs(pnl)
    if gls<0.001:gls=0.001
    return c,gpr/gls,md*100,w+lo_,w,lo_,sc,tc,rc,fc

@njit
def bt_cross(cl,hi,lo,ts,fv,sv,av,rv,am,ar,mn,rl,rh,sp,tp,wp,mp,lv,sk,ic,gp,st):
    """교차마진: 기존 로직 그대로"""
    n=len(cl);c=ic;p=0;ex=0.0;ps=0.0;sl=0.0
    tn=False;th=0.0;tl=999999.0;pk=c;md=0.0;ms=c
    sc=0;tc=0;rc=0;w=0;lo_=0;gpr=0.0;gls=0.0
    wa=0;ws=0;le=0;ld=0
    for i in range(st,n):
        px=cl[i];h_=hi[i];l_=lo[i]
        if i>st and i%1440==0:ms=c
        if p!=0:
            wa=0
            if not tn:
                if(p==1 and l_<=sl)or(p==-1 and h_>=sl):
                    pnl=(sl-ex)/ex*ps*p-ps*0.0004;c+=pnl;sc+=1
                    if pnl>0:w+=1;gpr+=pnl
                    else:lo_+=1;gls+=abs(pnl)
                    ld=p;le=i;p=0;continue
            br=(h_-ex)/ex*100 if p==1 else(ex-l_)/ex*100
            if br>=tp:tn=True
            if tn:
                if p==1:
                    if h_>th:th=h_
                    ns=th*(1-wp/100)
                    if ns>sl:sl=ns
                    if px<=sl:
                        pnl=(px-ex)/ex*ps*p-ps*0.0004;c+=pnl;tc+=1
                        if pnl>0:w+=1;gpr+=pnl
                        else:lo_+=1;gls+=abs(pnl)
                        ld=p;le=i;p=0;continue
                else:
                    if l_<tl:tl=l_
                    ns=tl*(1+wp/100)
                    if ns<sl:sl=ns
                    if px>=sl:
                        pnl=(px-ex)/ex*ps*p-ps*0.0004;c+=pnl;tc+=1
                        if pnl>0:w+=1;gpr+=pnl
                        else:lo_+=1;gls+=abs(pnl)
                        ld=p;le=i;p=0;continue
            if i>0:
                bn=fv[i]>sv[i];bp=fv[i-1]>sv[i-1]
                cu=bn and not bp;cd=not bn and bp
                if(p==1 and cd)or(p==-1 and cu):
                    pnl=(px-ex)/ex*ps*p-ps*0.0004;c+=pnl;rc+=1
                    if pnl>0:w+=1;gpr+=pnl
                    else:lo_+=1;gls+=abs(pnl)
                    ld=p;le=i;p=0
        if i<1:continue
        bn=fv[i]>sv[i];bp=fv[i-1]>sv[i-1]
        cu=bn and not bp;cd=not bn and bp
        if p==0:
            if cu:wa=1;ws=i
            elif cd:wa=-1;ws=i
            if wa!=0 and(mn==0 or i>ws):
                if mn>0 and i-ws>mn:wa=0;continue
                if wa==1 and cd:wa=-1;ws=i;continue
                elif wa==-1 and cu:wa=1;ws=i;continue
                if sk and wa==ld:continue
                aok=av[i]>=am;arr=True
                if ar>0 and i>=ar:arr=av[i]>av[i-ar]
                rok=(rv[i]>=rl)and(rv[i]<=rh)
                gok=True
                if gp>0:
                    g=abs(fv[i]-sv[i])/sv[i]*100
                    if g<gp:gok=False
                if aok and arr and rok and gok and c>0:
                    if ms>0 and(c-ms)/ms<=-0.20:wa=0;continue
                    mg=c*mp;ps=mg*lv;c-=ps*0.0004
                    p=wa;ex=px;tn=False;th=px;tl=px
                    if p==1:sl=ex*(1-sp/100)
                    else:sl=ex*(1+sp/100)
                    pk=max(pk,c);wa=0
        pk=max(pk,c);dd=(pk-c)/pk if pk>0 else 0
        if dd>md:md=dd
        if c<=0:break
    if p!=0 and c>0:
        pnl=(cl[n-1]-ex)/ex*ps*p-ps*0.0004;c+=pnl
        if pnl>0:w+=1;gpr+=pnl
        else:lo_+=1;gls+=abs(pnl)
    if gls<0.001:gls=0.001
    return c,gpr/gls,md*100,w+lo_,w,lo_,sc,tc,rc,0

def ema_c(s,p):return s.ewm(span=p,adjust=False).mean()
def adx_c(h,l,c,p):
    pdm=h.diff();mdm=-l.diff();pdm=pdm.where((pdm>mdm)&(pdm>0),0.0);mdm=mdm.where((mdm>pdm)&(mdm>0),0.0)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/p,min_periods=p,adjust=False).mean()
    pdi=100*(pdm.ewm(alpha=1/p,min_periods=p,adjust=False).mean()/atr)
    mdi=100*(mdm.ewm(alpha=1/p,min_periods=p,adjust=False).mean()/atr)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,1e-10)
    return dx.ewm(alpha=1/p,min_periods=p,adjust=False).mean()
def rsi_c(s,p):
    d=s.diff();g=d.where(d>0,0.0);l=(-d).where(d<0,0.0)
    return 100-100/(1+g.ewm(alpha=1/p,min_periods=p,adjust=False).mean()/l.ewm(alpha=1/p,min_periods=p,adjust=False).mean().replace(0,1e-10))

print("데이터 로딩...")
df=pd.read_csv(f"{BASE}/btc_usdt_5m_merged.csv",parse_dates=['timestamp'])
df.sort_values('timestamp',inplace=True);df.drop_duplicates('timestamp',keep='first',inplace=True)
df.set_index('timestamp',inplace=True);df=df[['open','high','low','close','volume']].astype(float)
d30=df.resample('30min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
cl=d30['close'].values;hi=d30['high'].values;lo=d30['low'].values
ts_ep=d30.index.astype(np.int64).values.astype(np.float64)/1e9
ind={}
for p in[10,11]:ind[f'r{p}']=rsi_c(d30['close'],p).values
ind['a20']=adx_c(d30['high'],d30['low'],d30['close'],20).values
ind['e100']=ema_c(d30['close'],100).values;ind['e600']=ema_c(d30['close'],600).values
ind['e75']=ema_c(d30['close'],75).values;ind['s750']=np.nan_to_num(d30['close'].rolling(750).mean().values,nan=0)

# JIT
bt_cross(cl[:2000],hi[:2000],lo[:2000],ts_ep[:2000],ind['e100'][:2000],ind['e600'][:2000],
         ind['a20'][:2000],ind['r10'][:2000],30.0,6,24,40.0,80.0,3.0,12.0,9.0,0.35,10,True,5000.0,0.2,600)
bt_isolated(cl[:2000],hi[:2000],lo[:2000],ts_ep[:2000],ind['e100'][:2000],ind['e600'][:2000],
            ind['a20'][:2000],ind['r10'][:2000],30.0,6,24,40.0,80.0,3.0,12.0,9.0,0.35,10,True,5000.0,0.2,600)

print("\n" + "=" * 140)
print("  교차마진 vs 격리마진 비교")
print("=" * 140)

for name,fk,sk,rk in[("v32.2","e100","e600","r10"),("v32.3","e75","s750","r11")]:
    fv=ind[fk];sv=ind[sk];rv=ind[rk]
    args=(cl,hi,lo,ts_ep,fv,sv,ind['a20'],rv,30.0,6,24,40.0,80.0,3.0,12.0,9.0,0.35,10,True,5000.0,0.2,600)
    rc=bt_cross(*args)
    ri=bt_isolated(*args)
    diff=ri[0]-rc[0]
    print(f"\n  [{name}]")
    print(f"  {'모드':>8} | {'잔액':>16} | {'PF':>5} | {'MDD':>6} | {'N':>3} | {'SL':>3} {'TSL':>3} {'REV':>3} {'FC':>3}")
    print(f"  {'-'*70}")
    print(f"  {'교차':>8} | ${rc[0]:>14,.2f} | {min(rc[1],999):>5.2f} | {rc[2]:>5.2f}% | {rc[3]:>3} | {rc[6]:>3} {rc[7]:>3} {rc[8]:>3} {rc[9]:>3}")
    print(f"  {'격리':>8} | ${ri[0]:>14,.2f} | {min(ri[1],999):>5.2f} | {ri[2]:>5.2f}% | {ri[3]:>3} | {ri[6]:>3} {ri[7]:>3} {ri[8]:>3} {ri[9]:>3}")
    print(f"  {'차이':>8} | ${diff:>+14,.2f} ({diff/rc[0]*100:+.4f}%) | FC={ri[9]}건")
