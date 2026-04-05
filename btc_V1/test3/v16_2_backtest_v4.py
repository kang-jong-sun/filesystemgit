"""
v16.2 BTC/USDT Futures Backtest Engine v4
- Multi-trigger entry: Cross + Pullback + ADX Surge + BB Squeeze
- Correct R:R: tight SL, wide TP
- Full 75-month coverage guaranteed
- Triple Engine
"""

import pandas as pd
import numpy as np
import warnings, os, time
warnings.filterwarnings('ignore')

DATA_DIR = r"D:\filesystem\futures\btc_V1\test3"

# ============================================================
# INDICATORS
# ============================================================
def wma(s,p):
    w=np.arange(1,p+1,dtype=float); return s.rolling(p).apply(lambda x:np.dot(x,w)/w.sum(),raw=True)
def ema(s,p): return s.ewm(span=p,adjust=False).mean()
def hma(s,p):
    h=max(int(p/2),1);sq=max(int(np.sqrt(p)),1)
    return wma(2*wma(s,h)-wma(s,p),sq)
def rsi(s,p=14):
    d=s.diff();g=d.where(d>0,0.0);l=(-d).where(d<0,0.0)
    return 100-(100/(1+g.ewm(alpha=1/p,min_periods=p).mean()/l.ewm(alpha=1/p,min_periods=p).mean().replace(0,np.nan)))
def adx_calc(h,l,c,p=14):
    tr=pd.concat([h-l,abs(h-c.shift(1)),abs(l-c.shift(1))],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/p,min_periods=p).mean()
    um=h-h.shift(1);dm=l.shift(1)-l
    pdm=pd.Series(np.where((um>dm)&(um>0),um,0.0),index=h.index).ewm(alpha=1/p,min_periods=p).mean()
    mdm=pd.Series(np.where((dm>um)&(dm>0),dm,0.0),index=h.index).ewm(alpha=1/p,min_periods=p).mean()
    pdi=100*pdm/atr.replace(0,np.nan);mdi=100*mdm/atr.replace(0,np.nan)
    dx=100*abs(pdi-mdi)/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/p,min_periods=p).mean(),pdi,mdi,atr
def atr_calc(h,l,c,p=14):
    tr=pd.concat([h-l,abs(h-c.shift(1)),abs(l-c.shift(1))],axis=1).max(axis=1)
    return tr.ewm(alpha=1/p,min_periods=p).mean()
def macd(s,f=12,sl=26,sg=9):
    ml=ema(s,f)-ema(s,sl);return ml,ema(ml,sg),ml-ema(ml,sg)
def bbpctb(s,p=20,sd=2.0):
    m=s.rolling(p).mean();st=s.rolling(p).std()
    return (s-(m-sd*st))/((m+sd*st)-(m-sd*st))

# ============================================================
# DATA
# ============================================================
def load():
    t0=time.time()
    fs=[os.path.join(DATA_DIR,f"btc_usdt_5m_2020_to_now_part{i}.csv") for i in range(1,4)]
    df=pd.concat([pd.read_csv(f,parse_dates=['timestamp']) for f in fs],ignore_index=True)
    df.sort_values('timestamp',inplace=True);df.set_index('timestamp',inplace=True)
    print(f"[Data] {len(df):,} rows | {df.index[0]}~{df.index[-1]} | {time.time()-t0:.1f}s")
    return df

def build(df5):
    t0=time.time()
    def rs(df,rule):
        return df.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum','quote_volume':'sum','trades':'sum'}).dropna()
    tfs={'5m':df5,'15m':rs(df5,'15min'),'30m':rs(df5,'30min'),'1h':rs(df5,'1h')}

    cfgs={'30m':('WMA',3,'EMA',200,20),'15m':('WMA',3,'EMA',150,20),'5m':('HMA',5,'EMA',100,14),'1h':('WMA',3,'EMA',200,20)}
    for tf,(ft,fl,st,sl,ap) in cfgs.items():
        d=tfs[tf];c=d['close'];h=d['high'];l=d['low'];v=d['volume']
        if ft=='WMA': d['fma']=wma(c,fl)
        elif ft=='HMA': d['fma']=hma(c,fl)
        else: d['fma']=ema(c,fl)
        d['sma']=ema(c,sl)
        d['adx'],d['pdi'],d['mdi'],_=adx_calc(h,l,c,ap)
        d['rsi']=rsi(c,14); d['atr']=atr_calc(h,l,c,14)
        d['atr50']=d['atr'].rolling(50).mean()
        _,_,d['mhist']=macd(c,12,26,9)
        d['bbp']=bbpctb(c,20,2.0)
        d['vr']=v/v.rolling(20).mean()
        d['adxs']=d['adx']-d['adx'].shift(3)
        d['trend']=(d['fma']>d['sma']).astype(int)

        # Pullback detection: price touches fast MA from above(long) or below(short)
        # Long pullback: low touches fma within 0.3%
        d['pb_long'] = ((d['low'] - d['fma']).abs() / d['fma'] < 0.003) & (d['trend']==1)
        d['pb_short'] = ((d['high'] - d['fma']).abs() / d['fma'] < 0.003) & (d['trend']==0)

        # ADX surge: ADX crossed above threshold recently
        d['adx_cross_30'] = (d['adx'] >= 30) & (d['adx'].shift(1) < 30)
        d['adx_cross_25'] = (d['adx'] >= 25) & (d['adx'].shift(1) < 25)

        # BB Squeeze release
        bbw = d['close'].rolling(20).std() / d['close'].rolling(20).mean() * 100
        d['bb_squeeze'] = bbw < bbw.rolling(50).quantile(0.2)
        d['squeeze_release'] = (~d['bb_squeeze']) & (d['bb_squeeze'].shift(1).fillna(False))

    idx5=df5.index
    def align(htf,cols):
        return htf[cols].reindex(idx5,method='ffill')

    all_cols=['trend','adx','rsi','atr','atr50','mhist','bbp','vr','adxs',
              'pb_long','pb_short','adx_cross_30','adx_cross_25','squeeze_release','fma','sma','close']

    a={}
    for tf in ['5m','15m','30m','1h']:
        d=tfs[tf]
        use_cols=[c for c in all_cols if c in d.columns]
        if tf=='5m': sub=d[use_cols].copy()
        else: sub=align(d,use_cols)
        sub.columns=[f'{tf}_{c}' for c in use_cols]
        a[tf]=sub

    master=pd.concat([df5[['open','high','low','close','volume']]]+list(a.values()),axis=1)
    print(f"[Build] {master.shape} | {time.time()-t0:.1f}s")
    return master

# ============================================================
# ENGINE CONFIG - Corrected R:R (tight SL, wide TP)
# ============================================================
ENG={
    'A':{'nm':'Sniper','tf':'30m','adx_mn':32,'rsi_lo':28,'rsi_hi':72,'min_sc':35,
         'lev':10,'ratio':0.20,'atr_sl':1.5,'atr_tp1':5.0,'atr_tp2':10.0,
         'trail_atr':2.0,'trail_act':0.06,'min_gap':30},
    'B':{'nm':'Core','tf':'15m','adx_mn':26,'rsi_lo':25,'rsi_hi':75,'min_sc':25,
         'lev':7,'ratio':0.16,'atr_sl':1.8,'atr_tp1':4.5,'atr_tp2':8.0,
         'trail_atr':2.0,'trail_act':0.05,'min_gap':15},
    'C':{'nm':'Swing','tf':'5m','adx_mn':22,'rsi_lo':20,'rsi_hi':80,'min_sc':15,
         'lev':5,'ratio':0.10,'atr_sl':2.0,'atr_tp1':4.0,'atr_tp2':7.0,
         'trail_atr':2.5,'trail_act':0.04,'min_gap':8},
}

# ============================================================
# POSITION & BACKTEST
# ============================================================
class P:
    __slots__=['d','ep','et','sz','lv','sl','t1p','t2p','eng','sc','atr','slp',
               'hi','lo','pkr','t1','t2','rem','ta','tsl','pp']
    def __init__(s,d,ep,et,sz,lv,sl,t1p,t2p,eng,sc,atr,slp):
        s.d=d;s.ep=ep;s.et=et;s.sz=sz;s.lv=lv;s.sl=sl;s.t1p=t1p;s.t2p=t2p
        s.eng=eng;s.sc=sc;s.atr=atr;s.slp=slp
        s.hi=ep;s.lo=ep;s.pkr=0;s.t1=False;s.t2=False;s.rem=1.0;s.ta=False;s.tsl=None;s.pp=0

def calc_score(adxs,vr,mtf,bbp,mh,rsi_v,d):
    s=0
    if not np.isnan(adxs) and adxs>0:s+=10
    if not np.isnan(vr) and vr>1.0:s+=min(10,int(vr*5))
    s+=min(20,mtf*7)
    if not np.isnan(bbp):
        if d==1 and bbp<0.75:s+=8
        elif d==-1 and bbp>0.25:s+=8
    if not np.isnan(mh):
        if(d==1 and mh>0)or(d==-1 and mh<0):s+=10
    if not np.isnan(rsi_v):
        if d==1 and 30<=rsi_v<=62:s+=8
        elif d==-1 and 38<=rsi_v<=70:s+=8
    return s

def run():
    FEE=0.0004;INIT=3000.0
    df5=load();master=build(df5)

    # Extract arrays
    hi=master['high'].values;lo=master['low'].values;cl=master['close'].values
    ts=master.index;n=len(master)

    sig={}
    for tf in['5m','15m','30m','1h']:
        sig[tf]={}
        for c in['trend','adx','rsi','atr','atr50','mhist','bbp','vr','adxs']:
            col=f'{tf}_{c}'
            sig[tf][c]=master[col].values if col in master.columns else np.full(n,np.nan)
        for c in['pb_long','pb_short','adx_cross_30','adx_cross_25','squeeze_release']:
            col=f'{tf}_{c}'
            sig[tf][c]=master[col].values.astype(float) if col in master.columns else np.zeros(n)

    bal=INIT;pkb=INIT;pos=[];trades=[];closs=0;cd_end=None;mo_s={};leb={'A':-999,'B':-999,'C':-999}
    warmup=3000

    print(f"\n[Run] {n:,} bars (warmup={warmup})...")
    t0=time.time()

    for i in range(warmup,n):
        t=ts[i];h=hi[i];l=lo[i];c=cl[i]

        # Check positions
        rm=[]
        for pi,p in enumerate(pos):
            xr=None;xp=c
            if p.d=='LONG':
                p.hi=max(p.hi,h);cr=(c-p.ep)/p.ep;p.pkr=max(p.pkr,(p.hi-p.ep)/p.ep)
                if l<=p.sl:xp=p.sl;xr='SL'
                else:
                    if not p.t1 and h>=p.t1p:
                        ps=p.sz*0.30;p.pp+=ps*(p.t1p-p.ep)/p.ep-ps*FEE;p.rem-=0.30;p.t1=True;p.sl=p.ep
                    if p.t1 and not p.t2 and h>=p.t2p:
                        ps=p.sz*0.30;p.pp+=ps*(p.t2p-p.ep)/p.ep-ps*FEE;p.rem-=0.30;p.t2=True;p.sl=p.t1p
                    cfg=ENG[p.eng]
                    if cr>=cfg['trail_act'] or p.t1:
                        p.ta=True;nt=p.hi-p.atr*cfg['trail_atr']
                        if p.tsl is None or nt>p.tsl:p.tsl=max(nt,p.sl)
                    if p.ta and p.tsl and l<=p.tsl:xp=p.tsl;xr='TRAIL'
            else:
                p.lo=min(p.lo,l);cr=(p.ep-c)/p.ep;p.pkr=max(p.pkr,(p.ep-p.lo)/p.ep)
                if h>=p.sl:xp=p.sl;xr='SL'
                else:
                    if not p.t1 and l<=p.t1p:
                        ps=p.sz*0.30;p.pp+=ps*(p.ep-p.t1p)/p.ep-ps*FEE;p.rem-=0.30;p.t1=True;p.sl=p.ep
                    if p.t1 and not p.t2 and l<=p.t2p:
                        ps=p.sz*0.30;p.pp+=ps*(p.ep-p.t2p)/p.ep-ps*FEE;p.rem-=0.30;p.t2=True;p.sl=p.t1p
                    cfg=ENG[p.eng]
                    if cr>=cfg['trail_act'] or p.t1:
                        p.ta=True;nt=p.lo+p.atr*cfg['trail_atr']
                        if p.tsl is None or nt<p.tsl:p.tsl=min(nt,p.sl)
                    if p.ta and p.tsl and h>=p.tsl:xp=p.tsl;xr='TRAIL'

            if xr:
                rs=p.sz*p.rem;rpnl=rs*((xp-p.ep)/p.ep if p.d=='LONG' else(p.ep-xp)/p.ep)
                tpnl=rpnl+p.pp-rs*FEE;bal+=tpnl;pkb=max(pkb,bal)
                closs=closs+1 if tpnl<0 else 0
                trades.append({'eng':p.eng,'dir':p.d,'et':p.et,'xt':t,'ep':p.ep,'xp':xp,
                    'xr':xr,'sz':p.sz,'lev':p.lv,'margin':p.sz/p.lv,'pnl':tpnl,
                    'roi':tpnl/(p.sz/p.lv)*100,'pkroi':p.pkr*100,'t1':p.t1,'t2':p.t2,
                    'sc':p.sc,'bal':bal,'slp':p.slp*100,'hold':(t-p.et).total_seconds()/60})
                rm.append(pi)
        for pi in sorted(rm,reverse=True):pos.pop(pi)

        # Risk
        if cd_end:
            if t<cd_end:continue
            cd_end=None
        dd=(bal-pkb)/pkb if pkb>0 else 0
        if dd<-0.40:cd_end=t+pd.Timedelta(hours=72);continue
        dd_m=0.5 if dd<-0.25 else(0.75 if dd<-0.15 else 1.0)
        st_m=max(0.4,1.0-closs*0.10)
        mk=f"{t.year}-{t.month:02d}"
        if mk not in mo_s:mo_s[mk]=bal
        if mo_s[mk]>0 and(bal-mo_s[mk])/mo_s[mk]<-0.18:continue
        if len(pos)>=3:continue

        # Try engines
        for ek,cfg in ENG.items():
            tf=cfg['tf'];s=sig[tf]
            if i-leb[ek]<cfg['min_gap']:continue
            if any(p.eng==ek for p in pos):continue

            trend=s['trend'][i]
            if np.isnan(trend):continue
            d=1 if trend>0.5 else -1
            ds='LONG' if d==1 else 'SHORT'

            adx_v=s['adx'][i]
            if np.isnan(adx_v) or adx_v<cfg['adx_mn']:continue
            rsi_v=s['rsi'][i]
            if np.isnan(rsi_v) or not(cfg['rsi_lo']<=rsi_v<=cfg['rsi_hi']):continue
            atr_v=s['atr'][i]
            if np.isnan(atr_v) or atr_v<=0:continue

            # ENTRY TRIGGERS (need at least 1)
            trigger = False
            # T1: Pullback to fast MA
            if d==1 and s['pb_long'][i]>0.5: trigger=True
            elif d==-1 and s['pb_short'][i]>0.5: trigger=True
            # T2: ADX surge
            if adx_v>=30 and s['adx_cross_30'][i]>0.5: trigger=True
            elif adx_v>=25 and s['adx_cross_25'][i]>0.5: trigger=True
            # T3: BB squeeze release
            if s['squeeze_release'][i]>0.5: trigger=True
            # T4: Strong trend momentum (ADX > 35 and rising)
            adxs=s['adxs'][i]
            if not np.isnan(adxs) and adx_v>=35 and adxs>2: trigger=True

            if not trigger:continue

            # MTF alignment
            mtf_n=0
            for mtf_tf in(['30m','15m','1h'] if ek=='A' else(['15m','30m','5m'] if ek=='B' else['5m','15m'])):
                mt=sig[mtf_tf]['trend'][i]
                if not np.isnan(mt):
                    if(d==1 and mt>0.5)or(d==-1 and mt<0.5):mtf_n+=1
            if mtf_n<1:continue

            sc=calc_score(adxs,s['vr'][i],mtf_n,s['bbp'][i],s['mhist'][i],rsi_v,d)
            if sc<cfg['min_sc']:continue

            # Regime
            a50=s['atr50'][i]
            vrat=atr_v/a50 if(not np.isnan(a50) and a50>0)else 1.0
            reg_m=0.7 if vrat>1.3 else(1.2 if vrat<0.7 else 1.0)

            # Close opposite
            for pi in sorted([pi for pi,p in enumerate(pos) if p.eng==ek and p.d!=ds],reverse=True):
                p=pos[pi];rs=p.sz*p.rem
                rpnl=rs*((c-p.ep)/p.ep if p.d=='LONG' else(p.ep-c)/p.ep)
                tpnl=rpnl+p.pp-rs*FEE;bal+=tpnl;pkb=max(pkb,bal)
                closs=closs+1 if tpnl<0 else 0
                trades.append({'eng':p.eng,'dir':p.d,'et':p.et,'xt':t,'ep':p.ep,'xp':c,
                    'xr':'REV','sz':p.sz,'lev':p.lv,'margin':p.sz/p.lv,'pnl':tpnl,
                    'roi':tpnl/(p.sz/p.lv)*100,'pkroi':p.pkr*100,'t1':p.t1,'t2':p.t2,
                    'sc':p.sc,'bal':bal,'slp':p.slp*100,'hold':(t-p.et).total_seconds()/60})
                pos.pop(pi)

            sz=bal*cfg['ratio']*reg_m*dd_m*st_m
            if sz<5:continue
            lv=cfg['lev'];ntl=sz*lv

            sl_pct=max(0.012,min(0.06,atr_v*cfg['atr_sl']/c))
            sd=c*sl_pct;t1d=atr_v*cfg['atr_tp1'];t2d=atr_v*cfg['atr_tp2']
            if d==1:slp=c-sd;t1p=c+t1d;t2p=c+t2d
            else:slp=c+sd;t1p=c-t1d;t2p=c-t2d

            bal-=ntl*FEE
            pos.append(P(ds,c,t,ntl,lv,slp,t1p,t2p,ek,sc,atr_v,sl_pct))
            leb[ek]=i

        if(i-warmup)%130000==0 and i>warmup:
            pct=(i-warmup)/(n-warmup)*100
            print(f"  {pct:.0f}% | {t.strftime('%Y-%m-%d')} | Bal:${bal:,.0f} | Trades:{len(trades)} | Pos:{len(pos)}")

    elapsed=time.time()-t0
    print(f"  100% | {elapsed:.1f}s | Trades:{len(trades)}")
    return trades,bal,pkb,INIT

# ============================================================
# REPORT
# ============================================================
def report(trades,fbal,pkbal,init):
    if not trades:print("\n  NO TRADES!");return
    df=pd.DataFrame(trades);T=len(df)
    W=df[df['pnl']>0];L=df[df['pnl']<=0]
    wr=len(W)/T*100;gp=W['pnl'].sum() if len(W) else 0;gl=abs(L['pnl'].sum()) if len(L) else 0
    pf=gp/gl if gl>0 else float('inf');aw=W['roi'].mean() if len(W) else 0
    al=L['roi'].mean() if len(L) else 0;rr=abs(aw/al) if al!=0 else float('inf')
    tret=(fbal-init)/init*100

    bals=[init]+[t['bal'] for t in trades];pk=bals[0];mdd=0
    for b in bals:pk=max(pk,b);dd=(b-pk)/pk;mdd=min(mdd,dd)
    mc=0;cc=0
    for t in trades:
        if t['pnl']<=0:cc+=1;mc=max(mc,cc)
        else:cc=0
    sl_n=len(df[df['xr']=='SL']);tr_n=len(df[df['xr']=='TRAIL']);rv_n=len(df[df['xr']=='REV'])
    t1_n=df['t1'].sum();t2_n=df['t2'].sum()
    mo_span=max(1,(df['xt'].max()-df['et'].min()).days/30)

    print("\n"+"="*120)
    print("  v16.2 TRIPLE ENGINE BACKTEST - FULL 75-MONTH RESULTS")
    print("="*120)
    print(f"""
  Initial:       ${init:,.0f}
  Final:         ${fbal:,.2f}
  Total Return:  {tret:+,.1f}%
  Profit Factor: {pf:.2f}
  MDD:           {mdd*100:.1f}%
  Trades:        {T}  ({mo_span:.0f} months, avg {T/mo_span:.1f}/mo)
  Win Rate:      {wr:.1f}%
  Avg Win:       {aw:+.2f}%     Avg Loss: {al:+.2f}%
  R:R Ratio:     {rr:.2f}
  Max Consec L:  {mc}
  Exits: SL={sl_n}({sl_n/T*100:.0f}%) TRAIL={tr_n}({tr_n/T*100:.0f}%) REV={rv_n}({rv_n/T*100:.0f}%)
  TP1: {t1_n:.0f}({t1_n/T*100:.0f}%)  TP2: {t2_n:.0f}({t2_n/T*100:.0f}%)
""")

    # ENGINE
    print("="*120);print("  ENGINE BREAKDOWN");print("="*120)
    print(f"  {'Eng':<12} {'#':>5} {'W':>4} {'WR%':>6} {'GrP':>10} {'GrL':>10} {'PF':>6} {'AvW%':>7} {'AvL%':>7} {'R:R':>5} {'PnL':>11}")
    print("  "+"-"*98)
    for ek in['A','B','C']:
        et=df[df['eng']==ek]
        if len(et)==0:print(f"  {ek}({ENG[ek]['nm']:<7}) {'0':>5}");continue
        ew=et[et['pnl']>0];el=et[et['pnl']<=0];ewr=len(ew)/len(et)*100
        egp=ew['pnl'].sum() if len(ew) else 0;egl=abs(el['pnl'].sum()) if len(el) else 0
        epf=egp/egl if egl>0 else float('inf');eaw=ew['roi'].mean() if len(ew) else 0
        eal=el['roi'].mean() if len(el) else 0;err=abs(eaw/eal) if eal!=0 else float('inf')
        pfs=f"{epf:.2f}" if epf<999 else "INF"
        print(f"  {ek}({ENG[ek]['nm']:<7}) {len(et):>4} {len(ew):>4} {ewr:>5.1f}% ${egp:>8,.1f} ${egl:>8,.1f} {pfs:>5} {eaw:>+6.1f}% {eal:>+6.1f}% {err:>4.2f} ${et['pnl'].sum():>+9,.1f}")

    # ENTRY STRUCTURE
    print("\n"+"="*120);print("  POSITION ENTRY STRUCTURE (진입 구조 상세)");print("="*120)
    lo2=df[df['dir']=='LONG'];sh2=df[df['dir']=='SHORT']
    print(f"\n  [방향]")
    for lb,sub in[('LONG',lo2),('SHORT',sh2)]:
        if len(sub)==0:continue
        sw=sub[sub['pnl']>0]
        print(f"    {lb:>5}: {len(sub):>5} ({len(sub)/T*100:.1f}%) WR:{len(sw)/len(sub)*100:.1f}% AvgROI:{sub['roi'].mean():+.2f}% PnL:${sub['pnl'].sum():+,.1f}")

    print(f"\n  [스코어별]")
    for lo_s,hi_s in[(0,20),(20,30),(30,40),(40,50),(50,60),(60,80)]:
        bt=df[(df['sc']>=lo_s)&(df['sc']<hi_s)]
        if len(bt):
            bw=bt[bt['pnl']>0]
            print(f"    {lo_s:>2}~{hi_s:>2}: {len(bt):>5} WR:{len(bw)/len(bt)*100:.1f}% AvgROI:{bt['roi'].mean():+.2f}% PnL:${bt['pnl'].sum():+,.1f}")

    print(f"\n  [보유시간]")
    df['hh']=df['hold']/60
    for lo_h,hi_h,lb in[(0,0.5,'<30m'),(0.5,2,'30m~2h'),(2,8,'2~8h'),(8,24,'8~24h'),(24,72,'1~3d'),(72,168,'3~7d'),(168,99999,'7d+')]:
        ht=df[(df['hh']>=lo_h)&(df['hh']<hi_h)]
        if len(ht):hw=ht[ht['pnl']>0];print(f"    {lb:>8}: {len(ht):>5} WR:{len(hw)/len(ht)*100:.1f}% AvgROI:{ht['roi'].mean():+.2f}% PnL:${ht['pnl'].sum():+,.1f}")

    print(f"\n  [SL 거리]")
    for lo_s,hi_s in[(1,2),(2,3),(3,4),(4,6)]:
        st=df[(df['slp']>=lo_s)&(df['slp']<hi_s)]
        if len(st):sw=st[st['pnl']>0];print(f"    SL{lo_s}~{hi_s}%: {len(st):>5} WR:{len(sw)/len(st)*100:.1f}% AvgROI:{st['roi'].mean():+.2f}%")

    print(f"\n  [엔진×청산사유]")
    for ek in['A','B','C']:
        et=df[df['eng']==ek]
        if len(et)==0:continue
        print(f"    Eng {ek}({ENG[ek]['nm']}): ",end="")
        parts=[]
        for r in['SL','TRAIL','REV']:
            rt=et[et['xr']==r]
            if len(rt):parts.append(f"{r}={len(rt)}({len(rt)/len(et)*100:.0f}%,{rt['roi'].mean():+.1f}%)")
        print(" | ".join(parts))

    # MONTHLY
    print("\n"+"="*120);print("  MONTHLY PERFORMANCE (월별 상세)");print("="*120)
    df['mo']=pd.to_datetime(df['xt']).dt.to_period('M')
    mg=df.groupby('mo')
    print(f"\n  {'Mo':>8} {'#':>4} {'W':>3} {'L':>3} {'WR%':>5} {'GrP':>9} {'GrL':>9} {'Net':>9} {'PF':>5} {'Bal':>11} {'Ret%':>7} {'A':>3} {'B':>3} {'C':>3}")
    print("  "+"-"*108)

    rb=init;yr={};lm=0;tm=0
    for mo in sorted(mg.groups.keys()):
        g=mg.get_group(mo);nt=len(g);nw=len(g[g['pnl']>0]);nl=nt-nw
        wr2=nw/nt*100 if nt else 0
        gp2=g[g['pnl']>0]['pnl'].sum() if nw else 0
        gl2=abs(g[g['pnl']<=0]['pnl'].sum()) if nl else 0
        net=g['pnl'].sum();mpf=gp2/gl2 if gl2>0 else float('inf')
        sbr=rb;rb+=net;mret=net/sbr*100 if sbr>0 else 0;tm+=1
        if net<0:lm+=1
        ea=len(g[g['eng']=='A']);eb=len(g[g['eng']=='B']);ec=len(g[g['eng']=='C'])
        y=str(mo)[:4]
        if y not in yr:yr[y]={'pnl':0,'t':0,'w':0,'l':0,'gp':0,'gl':0,'sb':sbr}
        yr[y]['pnl']+=net;yr[y]['t']+=nt;yr[y]['w']+=nw;yr[y]['l']+=nl;yr[y]['gp']+=gp2;yr[y]['gl']+=gl2;yr[y]['eb']=rb
        pfs=f"{mpf:.1f}" if mpf<999 else "INF"
        mk=" <<" if net<0 else ""
        print(f"  {str(mo):>8} {nt:>4} {nw:>3} {nl:>3} {wr2:>4.0f}% ${gp2:>7,.0f} ${gl2:>7,.0f} ${net:>+7,.0f} {pfs:>4} ${rb:>9,.0f} {mret:>+6.1f}% {ea:>3} {eb:>3} {ec:>3}{mk}")

    # YEARLY
    print("\n"+"="*120);print("  YEARLY PERFORMANCE (연도별)");print("="*120)
    print(f"\n  {'Year':>6} {'#':>5} {'W':>4} {'L':>4} {'WR%':>5} {'GrossP':>11} {'GrossL':>11} {'NetPnL':>11} {'PF':>6} {'YrRet%':>8}")
    print("  "+"-"*85)
    for y2 in sorted(yr.keys()):
        yd=yr[y2];ywr=yd['w']/yd['t']*100 if yd['t'] else 0
        ypf=yd['gp']/yd['gl'] if yd['gl']>0 else float('inf')
        yret=yd['pnl']/yd['sb']*100 if yd['sb']>0 else 0;pfs=f"{ypf:.2f}" if ypf<999 else "INF"
        print(f"  {y2:>6} {yd['t']:>4} {yd['w']:>4} {yd['l']:>4} {ywr:>4.0f}% ${yd['gp']:>9,.0f} ${yd['gl']:>9,.0f} ${yd['pnl']:>+9,.0f} {pfs:>5} {yret:>+7.1f}%")
    pyrs=sum(1 for v in yr.values() if v['pnl']>0)
    print(f"\n  Loss Months: {lm}/{tm} ({lm/max(1,tm)*100:.0f}%)")
    print(f"  Profitable Years: {pyrs}/{len(yr)}")

    # ENGINE×YEAR
    print("\n"+"="*120);print("  ENGINE×YEAR");print("="*120)
    df['year']=pd.to_datetime(df['xt']).dt.year
    for ek in['A','B','C']:
        et=df[df['eng']==ek]
        if len(et)==0:print(f"\n  {ek}({ENG[ek]['nm']}): No trades");continue
        print(f"\n  {ek}({ENG[ek]['nm']}):")
        print(f"    {'Yr':>6} {'#':>4} {'WR%':>5} {'PnL':>10} {'AvgROI':>7} {'PF':>5}")
        for y3 in sorted(et['year'].unique()):
            yt=et[et['year']==y3];yw=yt[yt['pnl']>0];yl=yt[yt['pnl']<=0]
            ywr=len(yw)/len(yt)*100;ygp=yw['pnl'].sum() if len(yw) else 0
            ygl=abs(yl['pnl'].sum()) if len(yl) else 0;ypf=ygp/ygl if ygl>0 else float('inf')
            pfs=f"{ypf:.2f}" if ypf<999 else "INF"
            print(f"    {y3:>6} {len(yt):>3} {ywr:>4.0f}% ${yt['pnl'].sum():>+8,.0f} {yt['roi'].mean():>+6.1f}% {pfs:>4}")

    # TOP/BOTTOM
    print("\n"+"="*120);print("  TOP 10 / BOTTOM 10");print("="*120)
    ds=df.sort_values('pnl',ascending=False)
    print(f"\n  Top 10:")
    print(f"    {'#':>2} {'E':>2} {'D':>5} {'Entry':>20} {'Exit':>20} {'R':>5} {'ROI%':>7} {'PnL':>9} {'S':>2}")
    for i2,(_,r) in enumerate(ds.head(10).iterrows()):
        print(f"    {i2+1:>2} {r['eng']:>2} {r['dir']:>5} {str(r['et'])[:19]:>20} {str(r['xt'])[:19]:>20} {r['xr']:>5} {r['roi']:>+6.1f}% ${r['pnl']:>7,.0f} {r['sc']:>2.0f}")
    print(f"\n  Bottom 10:")
    for i2,(_,r) in enumerate(ds.tail(10).iterrows()):
        print(f"    {i2+1:>2} {r['eng']:>2} {r['dir']:>5} {str(r['et'])[:19]:>20} {str(r['xt'])[:19]:>20} {r['xr']:>5} {r['roi']:>+6.1f}% ${r['pnl']:>7,.0f} {r['sc']:>2.0f}")

    print("\n"+"="*120);print("  COMPLETE");print("="*120)

if __name__=="__main__":
    print("="*120);print("  v16.2 Triple Engine Backtest v4 - Multi-Trigger Entry");print("="*120)
    trades,fbal,pkbal,init=run()
    report(trades,fbal,pkbal,init)
