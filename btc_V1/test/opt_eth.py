import sys;sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd,numpy as np
from numba import njit
np.seterr(all='ignore')
BASE="D:/filesystem/futures/btc_V1/test"

@njit
def bt(cl,hi,lo,fv,sv,av,rv,am,ar,mn,rl,rh,sp,tp,wp,mp,lv,sk,ic,gp,st):
    n=len(cl);c=ic;p=0;ex=0.0;ps=0.0;sl=0.0;mg=0.0
    tn=False;th=0.0;tl=999999.0;pk=c;md=0.0;ms=c
    sc=0;tc=0;rc=0;fc=0;w=0;lo_=0;gpr=0.0;gls=0.0
    wa=0;ws=0;le=0;ld=0
    for i in range(st,n):
        px=cl[i];h_=hi[i];l_=lo[i]
        if i>st and i%1440==0:ms=c
        if p!=0:
            wa=0
            liq=(1.0/lv)*0.98
            if p==1 and l_<=ex*(1-liq):
                pnl=-mg-ps*0.0004;c+=pnl;fc+=1;lo_+=1;gls+=abs(pnl);ld=p;le=i;p=0;continue
            if p==-1 and h_>=ex*(1+liq):
                pnl=-mg-ps*0.0004;c+=pnl;fc+=1;lo_+=1;gls+=abs(pnl);ld=p;le=i;p=0;continue
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

print("ETH 데이터 로딩...")
dfs=[pd.read_csv(f"{BASE}/eth_usdt_5m_2020_to_now_part{i}.csv",parse_dates=['timestamp']) for i in[1,2,3]]
df=pd.concat(dfs).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
df.set_index('timestamp',inplace=True);df=df[['open','high','low','close','volume']].astype(float)
d30=df.resample('30min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
cl=d30['close'].values;hi=d30['high'].values;lo=d30['low'].values
print(f"  30분봉: {len(d30)}캔들 ({d30.index[0]} ~ {d30.index[-1]})")

# 지표 캐시
print("지표 계산...")
ind={}
for p in[50,75,100,120,150]:ind[f'e{p}']=ema_c(d30['close'],p).values
for p in[300,400,500,600,700,800]:ind[f'e{p}']=ema_c(d30['close'],p).values
for p in[10,11,14,20]:ind[f'r{p}']=rsi_c(d30['close'],p).values
for p in[14,20,25]:ind[f'a{p}']=adx_c(d30['high'],d30['low'],d30['close'],p).values

# JIT
bt(cl[:2000],hi[:2000],lo[:2000],ind['e100'][:2000],ind['e600'][:2000],
   ind['a20'][:2000],ind['r10'][:2000],30.0,6,24,40.0,80.0,3.0,12.0,9.0,0.35,10,True,5000.0,0.2,600)

IC=5000.0

# ═══ 1단계: MA 조합 탐색 (BTC 최적 SL3 TA12/TSL9 유지) ═══
print(f"\n{'='*140}")
print("  [1] ETH — MA Fast/Slow 조합 (SL3 TA12/TSL9 ADX20>=30 +6rise 갭0.2% M35% 10x)")
print(f"{'='*140}")
print(f"  {'Fast/Slow':>10} | {'잔액':>14} {'수익률':>12} {'PF':>5} {'MDD':>5} {'N':>3} {'SL':>2} {'TSL':>3} {'REV':>3} {'FC':>2}")
print(f"  {'-'*90}")
ma_results=[]
for fast in[50,75,100,120,150]:
    for slow in[300,400,500,600,700,800]:
        if fast>=slow:continue
        fv=ind[f'e{fast}'];sv=ind[f'e{slow}']
        r=bt(cl,hi,lo,fv,sv,ind['a20'],ind['r10'],30.0,6,24,40.0,80.0,3.0,12.0,9.0,0.35,10,True,IC,0.2,600)
        ret=(r[0]-IC)/IC*100
        ma_results.append((fast,slow,r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],ret))
        mark=""
        if r[0]>=100000:mark=" **"
        elif r[0]>=50000:mark=" *"
        if r[3]>=5:
            print(f"  {fast:>3}/{slow:>4} | ${r[0]:>12,.0f} {ret:>+11.1f}% {min(r[1],999):>5.1f} {r[2]:>5.1f}% {r[3]:>3} {r[6]:>2} {r[7]:>3} {r[8]:>3} {r[9]:>2}{mark}")

ma_results.sort(key=lambda x:-x[2])
best_ma=ma_results[0]
print(f"\n  최적 MA: EMA({best_ma[0]})/EMA({best_ma[1]}) ${best_ma[2]:,.0f}")

# ═══ 2단계: 최적MA로 SL/TA/TSL 탐색 ═══
bf,bs=best_ma[0],best_ma[1]
fv=ind[f'e{bf}'];sv=ind[f'e{bs}']
print(f"\n{'='*140}")
print(f"  [2] ETH — SL × TA/TSL 그리드 (EMA({bf})/EMA({bs}))")
print(f"{'='*140}")
sl_results=[]
for sl in[3,4,5,6,7,8]:
    for ta in[6,8,10,12,15,20]:
        for tw in[3,5,7,9,10,12]:
            if tw>=ta:continue
            r=bt(cl,hi,lo,fv,sv,ind['a20'],ind['r10'],30.0,6,24,40.0,80.0,
                 float(sl),float(ta),float(tw),0.35,10,True,IC,0.2,600)
            if r[3]>=10 and r[0]>IC:
                sl_results.append((sl,ta,tw,r[0],min(r[1],999),r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9]))
sl_results.sort(key=lambda x:-x[3])
print(f"  {'SL':>3} {'TA':>3}/{'TW':>3} | {'잔액':>14} {'PF':>5} {'MDD':>5} {'N':>3} {'SL':>2} {'TSL':>3} {'REV':>3}")
print(f"  {'-'*70}")
for i,r in enumerate(sl_results[:20]):
    print(f"  {r[0]:>3} {r[1]:>3}/{r[2]:>3} | ${r[3]:>12,.0f} {r[4]:>5.1f} {r[5]:>5.1f}% {r[6]:>3} {r[9]:>2} {r[10]:>3} {r[11]:>3}")

if sl_results:
    best_sl=sl_results[0]
    bsl,bta,btw=best_sl[0],best_sl[1],best_sl[2]
else:
    bsl,bta,btw=3,12,9

# ═══ 3단계: ADX 길이/임계값 탐색 ═══
print(f"\n{'='*140}")
print(f"  [3] ETH — ADX 길이 × 임계값 (EMA({bf}/{bs}) SL{bsl} TA{bta}/TSL{btw})")
print(f"{'='*140}")
adx_results=[]
for ap in[14,20,25]:
    av=ind[f'a{ap}']
    for amin in[20,25,30,35,40]:
        for arise in[0,3,6,8]:
            r=bt(cl,hi,lo,fv,sv,av,ind['r10'],float(amin),arise,24,40.0,80.0,
                 float(bsl),float(bta),float(btw),0.35,10,True,IC,0.2,600)
            if r[3]>=10:
                adx_results.append((ap,amin,arise,r[0],min(r[1],999),r[2],r[3]))
adx_results.sort(key=lambda x:-x[3])
print(f"  {'ADX':>4} {'>= ':>4} {'+':>3} | {'잔액':>14} {'PF':>5} {'MDD':>5} {'N':>3}")
print(f"  {'-'*55}")
for r in adx_results[:15]:
    tag=" <- BTC" if r[0]==20 and r[1]==30 and r[2]==6 else ""
    print(f"  p{r[0]:>2} >={r[1]:<3} +{r[2]:>2} | ${r[3]:>12,.0f} {r[4]:>5.1f} {r[5]:>5.1f}% {r[6]:>3}{tag}")

if adx_results:
    ba=adx_results[0]
    bap,bam,bar=ba[0],ba[1],ba[2]
else:
    bap,bam,bar=20,30,6

# ═══ 4단계: RSI 탐색 ═══
print(f"\n{'='*140}")
print(f"  [4] ETH — RSI 길이 × 범위")
print(f"{'='*140}")
rsi_results=[]
for rp in[10,11,14,20]:
    rv=ind[f'r{rp}']
    for rlo in[0,30,35,40,45]:
        for rhi in[60,65,70,75,80,85,100]:
            if rlo>=rhi:continue
            r=bt(cl,hi,lo,fv,sv,ind[f'a{bap}'],rv,float(bam),bar,24,float(rlo),float(rhi),
                 float(bsl),float(bta),float(btw),0.35,10,True,IC,0.2,600)
            if r[3]>=10 and r[0]>IC:
                rsi_results.append((rp,rlo,rhi,r[0],min(r[1],999),r[2],r[3]))
rsi_results.sort(key=lambda x:-x[3])
print(f"  {'RSI':>4} {'Lo':>3}~{'Hi':>3} | {'잔액':>14} {'PF':>5} {'MDD':>5} {'N':>3}")
print(f"  {'-'*55}")
for r in rsi_results[:15]:
    print(f"  p{r[0]:>2} {r[1]:>3}~{r[2]:>3} | ${r[3]:>12,.0f} {r[4]:>5.1f} {r[5]:>5.1f}% {r[6]:>3}")

if rsi_results:
    br=rsi_results[0]
    brp,brl,brh=br[0],br[1],br[2]
else:
    brp,brl,brh=10,40,80

# ═══ 5단계: 갭 필터 탐색 ═══
print(f"\n{'='*140}")
print(f"  [5] ETH — EMA 갭 필터")
print(f"{'='*140}")
for gap in[0,0.1,0.2,0.3,0.5,1.0]:
    r=bt(cl,hi,lo,fv,sv,ind[f'a{bap}'],ind[f'r{brp}'],float(bam),bar,24,float(brl),float(brh),
         float(bsl),float(bta),float(btw),0.35,10,True,IC,gap,600)
    tag=" <- BTC" if gap==0.2 else ""
    print(f"  갭>={gap:.1f}% | ${r[0]:>12,.0f} PF{min(r[1],999):.1f} MDD{r[2]:.1f}% N{r[3]}{tag}")

# ═══ 최종 결과 ═══
print(f"\n{'='*140}")
print(f"  ETH 최적 파라미터 요약")
print(f"{'='*140}")
print(f"  MA:  EMA({bf})/EMA({bs})")
print(f"  SL:  {bsl}%  TA: {bta}%  TSL: {btw}%")
print(f"  ADX: p{bap} >={bam} +{bar}봉")
print(f"  RSI: p{brp} {brl}~{brh}")

# 최적으로 최종 실행
r_final=bt(cl,hi,lo,fv,sv,ind[f'a{bap}'],ind[f'r{brp}'],float(bam),bar,24,float(brl),float(brh),
           float(bsl),float(bta),float(btw),0.35,10,True,IC,0.2,600)
print(f"\n  최종: ${r_final[0]:>14,.2f} ({(r_final[0]-IC)/IC*100:+,.1f}%)")
print(f"  PF{min(r_final[1],999):.2f} MDD{r_final[2]:.2f}% N{r_final[3]} SL{r_final[6]} TSL{r_final[7]} REV{r_final[8]} FC{r_final[9]}")
