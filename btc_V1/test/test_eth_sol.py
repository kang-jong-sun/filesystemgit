import sys;sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd,numpy as np
from numba import njit
np.seterr(all='ignore')
BASE="D:/filesystem/futures/btc_V1/test"

@njit
def bt(cl,hi,lo,ts,fv,sv,av,rv,am,ar,mn,rl,rh,sp,tp,wp,mp,lv,sk,ic,gp,st):
    n=len(cl);c=ic;p=0;ex=0.0;ps=0.0;sl=0.0;mg=0.0
    tn=False;th=0.0;tl=999999.0;pk=c;md=0.0;ms=c
    sc=0;tc=0;rc=0;fc=0;w=0;lo_=0;gpr=0.0;gls=0.0
    wa=0;ws=0;le=0;ld=0
    YP=np.zeros(7);YW=np.zeros(7,dtype=np.int32);YL=np.zeros(7,dtype=np.int32);YM=np.zeros(7);YK=np.zeros(7)
    for j in range(7):YK[j]=ic
    for i in range(st,n):
        px=cl[i];h_=hi[i];l_=lo[i]
        if i>st and i%1440==0:ms=c
        yr=2020+int((ts[i]-1577836800)/(365.25*86400))
        yi=yr-2020 if yr>=2020 and yr<=2026 else -1
        if p!=0:
            wa=0
            # 격리마진 강제청산
            liq_pct=(1.0/lv)*0.98
            if p==1 and l_<=ex*(1-liq_pct):
                pnl=-mg-ps*0.0004;c+=pnl;fc+=1;lo_+=1;gls+=abs(pnl)
                if yi>=0:YP[yi]+=pnl;YL[yi]+=1
                ld=p;le=i;p=0;continue
            if p==-1 and h_>=ex*(1+liq_pct):
                pnl=-mg-ps*0.0004;c+=pnl;fc+=1;lo_+=1;gls+=abs(pnl)
                if yi>=0:YP[yi]+=pnl;YL[yi]+=1
                ld=p;le=i;p=0;continue
            if not tn:
                if(p==1 and l_<=sl)or(p==-1 and h_>=sl):
                    pnl=(sl-ex)/ex*ps*p-ps*0.0004;c+=pnl;sc+=1
                    if pnl>0:w+=1;gpr+=pnl
                    else:lo_+=1;gls+=abs(pnl)
                    if yi>=0:YP[yi]+=pnl;(YW if pnl>0 else YL)[yi]+=1
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
                        if yi>=0:YP[yi]+=pnl;(YW if pnl>0 else YL)[yi]+=1
                        ld=p;le=i;p=0;continue
                else:
                    if l_<tl:tl=l_
                    ns=tl*(1+wp/100)
                    if ns<sl:sl=ns
                    if px>=sl:
                        pnl=(px-ex)/ex*ps*p-ps*0.0004;c+=pnl;tc+=1
                        if pnl>0:w+=1;gpr+=pnl
                        else:lo_+=1;gls+=abs(pnl)
                        if yi>=0:YP[yi]+=pnl;(YW if pnl>0 else YL)[yi]+=1
                        ld=p;le=i;p=0;continue
            if i>0:
                bn=fv[i]>sv[i];bp=fv[i-1]>sv[i-1]
                cu=bn and not bp;cd=not bn and bp
                if(p==1 and cd)or(p==-1 and cu):
                    pnl=(px-ex)/ex*ps*p-ps*0.0004;c+=pnl;rc+=1
                    if pnl>0:w+=1;gpr+=pnl
                    else:lo_+=1;gls+=abs(pnl)
                    if yi>=0:YP[yi]+=pnl;(YW if pnl>0 else YL)[yi]+=1
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
        if yi>=0:
            YK[yi]=max(YK[yi],c)
            yd=(YK[yi]-c)/YK[yi] if YK[yi]>0 else 0
            if yd>YM[yi]:YM[yi]=yd
        if c<=0:break
    if p!=0 and c>0:
        pnl=(cl[n-1]-ex)/ex*ps*p-ps*0.0004;c+=pnl
        if pnl>0:w+=1;gpr+=pnl
        else:lo_+=1;gls+=abs(pnl)
    if gls<0.001:gls=0.001
    return c,gpr/gls,md*100,w+lo_,w,lo_,sc,tc,rc,fc,YP,YW,YL,YM

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

def load_and_test(symbol,files):
    print(f"\n{'='*140}")
    print(f"  {symbol}/USDT — v32.2 전략 적용 테스트 (격리마진)")
    print(f"{'='*140}")
    dfs=[pd.read_csv(f"{BASE}/{f}",parse_dates=['timestamp']) for f in files]
    df=pd.concat(dfs).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp',inplace=True)
    df=df[['open','high','low','close','volume']].astype(float)
    d30=df.resample('30min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    print(f"  5분봉: {len(df)}캔들 ({df.index[0]} ~ {df.index[-1]})")
    print(f"  30분봉: {len(d30)}캔들")

    cl=d30['close'].values;hi=d30['high'].values;lo=d30['low'].values
    ts_ep=d30.index.astype(np.int64).values.astype(np.float64)/1e9
    fv=ema_c(d30['close'],100).values;sv=ema_c(d30['close'],600).values
    av=adx_c(d30['high'],d30['low'],d30['close'],20).values
    rv=rsi_c(d30['close'],10).values

    # v32.2 파라미터 그대로
    args=(cl,hi,lo,ts_ep,fv,sv,av,rv,30.0,6,24,40.0,80.0,3.0,12.0,9.0,0.35,10,True,5000.0,0.2,600)
    r=bt(*args)

    years=[2020,2021,2022,2023,2024,2025,2026]
    print(f"\n  잔액: ${r[0]:>14,.2f} | 수익률: {(r[0]-5000)/5000*100:+,.1f}%")
    print(f"  PF: {min(r[1],999):.2f} | MDD: {r[2]:.2f}% | N: {r[3]} | W{r[4]}/L{r[5]}")
    print(f"  SL: {r[6]} | TSL: {r[7]} | REV: {r[8]} | FC: {r[9]}")
    print(f"\n  연도별:")
    print(f"  {'연도':>6} | {'PnL':>12} | {'W/L':>8} | {'MDD':>6}")
    print(f"  {'-'*45}")
    for i,yr in enumerate(years):
        t=int(r[11][i])+int(r[12][i])
        if t>0:
            print(f"  {yr:>6} | ${r[10][i]:>+11,.0f} | {int(r[11][i]):>3}/{int(r[12][i]):>3} | {r[13][i]*100:>5.1f}%")
    return r

# BTC (기준)
print("BTC 로딩...")
btc_dfs=[pd.read_csv(f"{BASE}/btc_usdt_5m_merged.csv",parse_dates=['timestamp'])]
btc_df=btc_dfs[0].sort_values('timestamp').drop_duplicates('timestamp',keep='first')
btc_df.set_index('timestamp',inplace=True)
btc_df=btc_df[['open','high','low','close','volume']].astype(float)
btc30=btc_df.resample('30min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
cl_b=btc30['close'].values;hi_b=btc30['high'].values;lo_b=btc30['low'].values
ts_b=btc30.index.astype(np.int64).values.astype(np.float64)/1e9
fv_b=ema_c(btc30['close'],100).values;sv_b=ema_c(btc30['close'],600).values
av_b=adx_c(btc30['high'],btc30['low'],btc30['close'],20).values;rv_b=rsi_c(btc30['close'],10).values

# JIT
bt(cl_b[:2000],hi_b[:2000],lo_b[:2000],ts_b[:2000],fv_b[:2000],sv_b[:2000],
   av_b[:2000],rv_b[:2000],30.0,6,24,40.0,80.0,3.0,12.0,9.0,0.35,10,True,5000.0,0.2,600)

r_btc=bt(cl_b,hi_b,lo_b,ts_b,fv_b,sv_b,av_b,rv_b,30.0,6,24,40.0,80.0,3.0,12.0,9.0,0.35,10,True,5000.0,0.2,600)
print(f"\n{'='*140}")
print(f"  BTC/USDT — v32.2 (기준)")
print(f"{'='*140}")
print(f"  잔액: ${r_btc[0]:>14,.2f} | 수익률: {(r_btc[0]-5000)/5000*100:+,.1f}%")
print(f"  PF: {min(r_btc[1],999):.2f} | MDD: {r_btc[2]:.2f}% | N: {r_btc[3]} | FC: {r_btc[9]}")

# ETH
r_eth=load_and_test("ETH",[f"eth_usdt_5m_2020_to_now_part{i}.csv" for i in[1,2,3]])

# SOL
r_sol=load_and_test("SOL",[f"sol_usdt_5m_2020_to_now_part{i}.csv" for i in[1,2,3]])

# ═══ 비교 ═══
print(f"\n\n{'='*140}")
print(f"  v32.2 전략 — BTC vs ETH vs SOL 비교")
print(f"{'='*140}")
print(f"  {'심볼':>6} | {'잔액':>16} | {'수익률':>12} | {'PF':>6} | {'MDD':>6} | {'N':>3} | {'SL':>3} {'TSL':>3} {'REV':>3} {'FC':>3} | 평가")
print(f"  {'-'*100}")
for name,r in[("BTC",r_btc),("ETH",r_eth),("SOL",r_sol)]:
    ret=(r[0]-5000)/5000*100
    grade=""
    if ret>10000 and r[2]<60:grade="우수"
    elif ret>1000:grade="양호"
    elif ret>0:grade="미미"
    else:grade="손실"
    print(f"  {name:>6} | ${r[0]:>14,.2f} | {ret:>+11,.1f}% | {min(r[1],999):>5.2f} | {r[2]:>5.2f}% | {r[3]:>3} | {r[6]:>3} {r[7]:>3} {r[8]:>3} {r[9]:>3} | {grade}")
