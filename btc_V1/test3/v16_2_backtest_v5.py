"""
v16.2 BTC/USDT Futures Backtest Engine v5
- Trend-following: enter when trend + ADX + RSI align, min gap prevents overtrading
- No event-based triggers (fix for disappearing signals in trending markets)
- Tight SL (ATR*1.5), Wide TP (ATR*5/10) for correct R:R
- Full 75 months
"""
import pandas as pd, numpy as np, os, time, warnings, sys
warnings.filterwarnings('ignore')
DIR = r"D:\filesystem\futures\btc_V1\test3"

def w(s,p):
    wt=np.arange(1,p+1,dtype=float);return s.rolling(p).apply(lambda x:np.dot(x,wt)/wt.sum(),raw=True)
def e(s,p): return s.ewm(span=p,adjust=False).mean()
def hm(s,p):
    h=max(int(p/2),1);sq=max(int(np.sqrt(p)),1);return w(2*w(s,h)-w(s,p),sq)
def rs(s,p=14):
    d=s.diff();g=d.where(d>0,0);l=(-d).where(d<0,0)
    return 100-(100/(1+g.ewm(alpha=1/p,min_periods=p).mean()/l.ewm(alpha=1/p,min_periods=p).mean().replace(0,np.nan)))
def ax(h,l,c,p=14):
    tr=pd.concat([h-l,abs(h-c.shift(1)),abs(l-c.shift(1))],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/p,min_periods=p).mean()
    um=h-h.shift(1);dm=l.shift(1)-l
    pd_=pd.Series(np.where((um>dm)&(um>0),um,0),index=h.index).ewm(alpha=1/p,min_periods=p).mean()
    md_=pd.Series(np.where((dm>um)&(dm>0),dm,0),index=h.index).ewm(alpha=1/p,min_periods=p).mean()
    pdi=100*pd_/atr.replace(0,np.nan);mdi=100*md_/atr.replace(0,np.nan)
    dx=100*abs(pdi-mdi)/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/p,min_periods=p).mean(),atr

def load():
    t0=time.time()
    fs=[os.path.join(DIR,f"btc_usdt_5m_2020_to_now_part{i}.csv") for i in range(1,4)]
    df=pd.concat([pd.read_csv(f,parse_dates=['timestamp']) for f in fs],ignore_index=True)
    df.sort_values('timestamp',inplace=True);df.set_index('timestamp',inplace=True)
    print(f"[Data] {len(df):,} rows | {df.index[0]}~{df.index[-1]} | {time.time()-t0:.1f}s")
    return df

def build(df5):
    t0=time.time()
    def resamp(df,rule):
        return df.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    tfs={'5m':df5.copy(),'15m':resamp(df5,'15min'),'30m':resamp(df5,'30min'),'1h':resamp(df5,'1h')}

    cfgs={'30m':('W',3,'E',200,20),'15m':('W',3,'E',150,20),'5m':('H',5,'E',100,14),'1h':('W',3,'E',200,20)}
    for tf,(ft,fl,_,sl,ap) in cfgs.items():
        d=tfs[tf];c=d['close'];H=d['high'];L=d['low']
        d['fma']={'W':w,'H':hm,'E':e}[ft](c,fl); d['sma']=e(c,sl)
        d['adx'],d['atr']=ax(H,L,c,ap); d['rsi']=rs(c,14)
        d['atr50']=d['atr'].rolling(50).mean()
        d['mh']=(e(c,12)-e(c,26))-e(e(c,12)-e(c,26),9)
        d['vr']=d['volume']/d['volume'].rolling(20).mean()
        d['adxs']=d['adx']-d['adx'].shift(3)
        d['trend']=(d['fma']>d['sma']).astype(int)

    idx=df5.index
    def align(htf,cols):return htf[cols].reindex(idx,method='ffill')

    cols=['trend','adx','rsi','atr','atr50','mh','vr','adxs']
    parts=[df5[['open','high','low','close','volume']]]
    for tf in['5m','15m','30m','1h']:
        d=tfs[tf]
        if tf=='5m':sub=d[cols].copy()
        else:sub=align(d,cols)
        sub.columns=[f'{tf}_{c}' for c in cols]
        parts.append(sub)
    M=pd.concat(parts,axis=1)
    print(f"[Build] {M.shape} | {time.time()-t0:.1f}s")
    return M

# ---- Engine Config ----
# Key fix: tight SL, wide TP -> R:R > 2
EC={
    'A':{'nm':'Sniper','tf':'30m','adx':30,'rlo':25,'rhi':75,'lev':8,'rat':0.15,
         'asl':1.5,'at1':5.0,'at2':10.0,'tatr':2.0,'tact':0.06,'gap':36,'sl_max':0.035},
    'B':{'nm':'Core','tf':'15m','adx':25,'rlo':22,'rhi':78,'lev':6,'rat':0.14,
         'asl':1.8,'at1':4.5,'at2':8.0,'tatr':2.0,'tact':0.05,'gap':18,'sl_max':0.04},
    'C':{'nm':'Swing','tf':'5m','adx':20,'rlo':18,'rhi':82,'lev':5,'rat':0.10,
         'asl':2.0,'at1':3.5,'at2':7.0,'tatr':2.5,'tact':0.04,'gap':12,'sl_max':0.05},
}

class P:
    __slots__=['d','ep','et','sz','lv','sl','t1p','t2p','eng','atr','slp',
               'hi','lo','pkr','t1','t2','rem','ta','tsl','pp']
    def __init__(s,d,ep,et,sz,lv,sl,t1p,t2p,eng,atr,slp):
        s.d=d;s.ep=ep;s.et=et;s.sz=sz;s.lv=lv;s.sl=sl;s.t1p=t1p;s.t2p=t2p
        s.eng=eng;s.atr=atr;s.slp=slp
        s.hi=ep;s.lo=ep;s.pkr=0;s.t1=0;s.t2=0;s.rem=1.0;s.ta=0;s.tsl=None;s.pp=0

def run():
    FEE=0.0004;INIT=3000.0
    df5=load();M=build(df5)
    hi=M['high'].values;lo=M['low'].values;cl=M['close'].values;ts=M.index;n=len(M)

    sig={}
    for tf in['5m','15m','30m','1h']:
        sig[tf]={c:M[f'{tf}_{c}'].values for c in['trend','adx','rsi','atr','atr50','mh','vr','adxs']}

    bal=INIT;pkb=INIT;pos=[];trades=[];clo=0;cd=None;mos={};leb={'A':-999,'B':-999,'C':-999}
    WU=3000;bal_curve=[]

    print(f"\n[Run] {n:,} bars...")
    t0=time.time()

    for i in range(WU,n):
        t=ts[i];h=hi[i];l=lo[i];c=cl[i]

        # Position management
        rm=[]
        for pi,p in enumerate(pos):
            xr=None;xp=c
            if p.d=='L':
                p.hi=max(p.hi,h);cr=(c-p.ep)/p.ep;p.pkr=max(p.pkr,(p.hi-p.ep)/p.ep)
                if l<=p.sl:xp=p.sl;xr='SL'
                else:
                    if not p.t1 and h>=p.t1p:
                        ps=p.sz*0.30;p.pp+=ps*(p.t1p-p.ep)/p.ep-ps*FEE;p.rem-=0.30;p.t1=1;p.sl=p.ep
                    if p.t1 and not p.t2 and h>=p.t2p:
                        ps=p.sz*0.30;p.pp+=ps*(p.t2p-p.ep)/p.ep-ps*FEE;p.rem-=0.30;p.t2=1;p.sl=p.t1p
                    cf=EC[p.eng]
                    if cr>=cf['tact'] or p.t1:
                        p.ta=1;nt=p.hi-p.atr*cf['tatr']
                        if p.tsl is None or nt>p.tsl:p.tsl=max(nt,p.sl)
                    if p.ta and p.tsl and l<=p.tsl:xp=p.tsl;xr='TSL'
            else:
                p.lo=min(p.lo,l);cr=(p.ep-c)/p.ep;p.pkr=max(p.pkr,(p.ep-p.lo)/p.ep)
                if h>=p.sl:xp=p.sl;xr='SL'
                else:
                    if not p.t1 and l<=p.t1p:
                        ps=p.sz*0.30;p.pp+=ps*(p.ep-p.t1p)/p.ep-ps*FEE;p.rem-=0.30;p.t1=1;p.sl=p.ep
                    if p.t1 and not p.t2 and l<=p.t2p:
                        ps=p.sz*0.30;p.pp+=ps*(p.ep-p.t2p)/p.ep-ps*FEE;p.rem-=0.30;p.t2=1;p.sl=p.t1p
                    cf=EC[p.eng]
                    if cr>=cf['tact'] or p.t1:
                        p.ta=1;nt=p.lo+p.atr*cf['tatr']
                        if p.tsl is None or nt<p.tsl:p.tsl=min(nt,p.sl)
                    if p.ta and p.tsl and h>=p.tsl:xp=p.tsl;xr='TSL'

            if xr:
                rs2=p.sz*p.rem
                rpnl=rs2*((xp-p.ep)/p.ep if p.d=='L' else(p.ep-xp)/p.ep)
                tp=rpnl+p.pp-rs2*FEE;bal+=tp;pkb=max(pkb,bal)
                clo=clo+1 if tp<0 else 0
                trades.append({'eng':p.eng,'dir':p.d,'et':p.et,'xt':t,'ep':p.ep,'xp':xp,
                    'xr':xr,'sz':p.sz,'lev':p.lv,'m':p.sz/p.lv,'pnl':tp,
                    'roi':tp/(p.sz/p.lv)*100,'pkr':p.pkr*100,'t1':p.t1,'t2':p.t2,
                    'bal':bal,'slp':p.slp*100,'hold':(t-p.et).total_seconds()/60})
                rm.append(pi)
        for pi in sorted(rm,reverse=True):pos.pop(pi)

        if i%24==0:bal_curve.append((t,bal))

        # Risk
        if cd:
            if t<cd:continue
            cd=None;pkb=bal  # Reset peak after cooldown to prevent infinite loop
        dd=(bal-pkb)/pkb if pkb>0 else 0
        if dd<-0.45:cd=t+pd.Timedelta(hours=48);continue
        dd_m=0.4 if dd<-0.30 else(0.6 if dd<-0.20 else(0.8 if dd<-0.10 else 1.0))
        st_m=max(0.4,1.0-clo*0.10)
        mk=f"{t.year}-{t.month:02d}"
        if mk not in mos:mos[mk]=bal
        if mos[mk]>0 and(bal-mos[mk])/mos[mk]<-0.18:continue
        if len(pos)>=3:continue

        # Entry: simple trend-following with min gap
        for ek,cf in EC.items():
            if i-leb[ek]<cf['gap']:continue
            if any(p.eng==ek for p in pos):continue

            tf=cf['tf'];s=sig[tf]
            tr=s['trend'][i]
            if np.isnan(tr):continue
            d=1 if tr>0.5 else -1;ds='L' if d==1 else 'S'

            av=s['adx'][i]
            if np.isnan(av) or av<cf['adx']:continue
            rv=s['rsi'][i]
            if np.isnan(rv) or not(cf['rlo']<=rv<=cf['rhi']):continue
            at=s['atr'][i]
            if np.isnan(at) or at<=0:continue

            # MTF check (need >=1 confirmation)
            mtfs={'A':['30m','1h'],'B':['15m','30m'],'C':['5m','15m']}
            mn=sum(1 for mtf in mtfs[ek]
                   if not np.isnan(sig[mtf]['trend'][i]) and
                   ((d==1 and sig[mtf]['trend'][i]>0.5)or(d==-1 and sig[mtf]['trend'][i]<0.5)))
            if mn<1:continue

            # Close opposite
            for pi in sorted([pi for pi,p in enumerate(pos) if p.eng==ek and p.d!=ds],reverse=True):
                p=pos[pi];rs2=p.sz*p.rem
                rpnl=rs2*((c-p.ep)/p.ep if p.d=='L' else(p.ep-c)/p.ep)
                tp2=rpnl+p.pp-rs2*FEE;bal+=tp2;pkb=max(pkb,bal)
                clo=clo+1 if tp2<0 else 0
                trades.append({'eng':p.eng,'dir':p.d,'et':p.et,'xt':t,'ep':p.ep,'xp':c,
                    'xr':'REV','sz':p.sz,'lev':p.lv,'m':p.sz/p.lv,'pnl':tp2,
                    'roi':tp2/(p.sz/p.lv)*100,'pkr':p.pkr*100,'t1':p.t1,'t2':p.t2,
                    'bal':bal,'slp':p.slp*100,'hold':(t-p.et).total_seconds()/60})
                pos.pop(pi)

            # Regime
            a50=s['atr50'][i]
            vrat=at/a50 if(not np.isnan(a50) and a50>0)else 1.0
            rm2=0.7 if vrat>1.3 else(1.2 if vrat<0.7 else 1.0)

            sz=bal*cf['rat']*rm2*dd_m*st_m
            if sz<5:continue
            lv=cf['lev'];ntl=sz*lv
            sl_pct=max(0.012,min(cf['sl_max'],at*cf['asl']/c))
            sd=c*sl_pct;t1d=at*cf['at1'];t2d=at*cf['at2']
            if d==1:slp=c-sd;t1p=c+t1d;t2p=c+t2d
            else:slp=c+sd;t1p=c-t1d;t2p=c-t2d
            bal-=ntl*FEE
            pos.append(P(ds,c,t,ntl,lv,slp,t1p,t2p,ek,at,sl_pct))
            leb[ek]=i

        if(i-WU)%130000==0 and i>WU:
            pct=(i-WU)/(n-WU)*100
            print(f"  {pct:.0f}% {t.strftime('%Y-%m')} Bal:${bal:,.0f} Trd:{len(trades)} Pos:{len(pos)}")

    print(f"  100% | {time.time()-t0:.1f}s | Trades:{len(trades)}")
    return trades,bal,pkb,INIT,bal_curve

def rpt(trades,fbal,pkb,init,bc):
    if not trades:print("NO TRADES");return
    df=pd.DataFrame(trades);T=len(df)
    # Fix direction labels for display
    df['dir_display']=df['dir'].map({'L':'LONG','S':'SHORT'})
    W=df[df['pnl']>0];L2=df[df['pnl']<=0]
    wr=len(W)/T*100;gp=W['pnl'].sum() if len(W) else 0;gl=abs(L2['pnl'].sum()) if len(L2) else 0
    pf=gp/gl if gl>0 else 999;aw=W['roi'].mean() if len(W) else 0;al=L2['roi'].mean() if len(L2) else 0
    rr=abs(aw/al) if al else 999;tret=(fbal-init)/init*100

    bals=[init]+[t['bal'] for t in trades];pk=bals[0];mdd=0
    for b in bals:pk=max(pk,b);dd=(b-pk)/pk;mdd=min(mdd,dd)
    mc=cc=0
    for t in trades:
        if t['pnl']<=0:cc+=1;mc=max(mc,cc)
        else:cc=0
    sln=len(df[df['xr']=='SL']);tn=len(df[df['xr']=='TSL']);rn=len(df[df['xr']=='REV'])
    t1n=df['t1'].sum();t2n=df['t2'].sum()
    ms=max(1,(df['xt'].max()-df['et'].min()).days/30)

    print("\n"+"="*120)
    print("  v16.2 TRIPLE ENGINE - FULL 75-MONTH RESULTS")
    print("="*120)
    print(f"""
  Initial:     ${init:,.0f}        Final: ${fbal:,.2f}        Return: {tret:+,.1f}%
  PF: {pf:.2f}     MDD: {mdd*100:.1f}%     Trades: {T} ({ms:.0f}mo, {T/ms:.1f}/mo)
  WinRate: {wr:.1f}%   AvgWin: {aw:+.2f}%   AvgLoss: {al:+.2f}%   R:R: {rr:.2f}
  MaxConsecLoss: {mc}
  SL:{sln}({sln/T*100:.0f}%) TSL:{tn}({tn/T*100:.0f}%) REV:{rn}({rn/T*100:.0f}%)  TP1:{t1n:.0f}({t1n/T*100:.0f}%) TP2:{t2n:.0f}({t2n/T*100:.0f}%)
""")

    print("="*120);print("  ENGINE BREAKDOWN");print("="*120)
    print(f"  {'Eng':<11}{'#':>5}{'W':>5}{'WR':>6}{'GrP':>10}{'GrL':>10}{'PF':>6}{'AvW':>7}{'AvL':>7}{'R:R':>5}{'PnL':>11}")
    print("  "+"-"*90)
    for ek in'ABC':
        et=df[df['eng']==ek]
        if not len(et):print(f"  {ek}({EC[ek]['nm']:<6}) 0");continue
        ew=et[et['pnl']>0];el=et[et['pnl']<=0];ewr=len(ew)/len(et)*100
        egp=ew['pnl'].sum() if len(ew) else 0;egl=abs(el['pnl'].sum()) if len(el) else 0
        epf=egp/egl if egl else 999;eaw=ew['roi'].mean() if len(ew) else 0
        eal=el['roi'].mean() if len(el) else 0;err=abs(eaw/eal) if eal else 999
        pfs=f"{epf:.2f}" if epf<999 else "INF"
        print(f"  {ek}({EC[ek]['nm']:<6}){len(et):>4}{len(ew):>5}{ewr:>5.0f}%${egp:>8,.0f}${egl:>8,.0f}{pfs:>5}{eaw:>+6.1f}%{eal:>+6.1f}%{err:>4.1f}${et['pnl'].sum():>+9,.0f}")

    # ENTRY STRUCTURE
    print("\n"+"="*120);print("  ENTRY STRUCTURE");print("="*120)
    for lb,code in[('LONG','L'),('SHORT','S')]:
        sub=df[df['dir']==code]
        if not len(sub):continue
        sw=sub[sub['pnl']>0]
        print(f"  {lb:>5}: {len(sub):>5}({len(sub)/T*100:.0f}%) WR:{len(sw)/len(sub)*100:.0f}% AvgROI:{sub['roi'].mean():+.2f}% PnL:${sub['pnl'].sum():+,.0f}")

    print(f"\n  [Hold Time]")
    df['hh']=df['hold']/60
    for a,b,lb in[(0,0.5,'<30m'),(0.5,2,'30m-2h'),(2,8,'2-8h'),(8,24,'8-24h'),(24,72,'1-3d'),(72,168,'3-7d'),(168,99999,'7d+')]:
        ht=df[(df['hh']>=a)&(df['hh']<b)]
        if len(ht):hw=ht[ht['pnl']>0];print(f"    {lb:>7}: {len(ht):>5} WR:{len(hw)/len(ht)*100:.0f}% AvgROI:{ht['roi'].mean():+.2f}% PnL:${ht['pnl'].sum():+,.0f}")

    print(f"\n  [Engine x Exit]")
    for ek in'ABC':
        et=df[df['eng']==ek]
        if not len(et):continue
        parts=[]
        for r in['SL','TSL','REV']:
            rt=et[et['xr']==r]
            if len(rt):parts.append(f"{r}:{len(rt)}({len(rt)/len(et)*100:.0f}%,avg{rt['roi'].mean():+.1f}%)")
        print(f"    {ek}({EC[ek]['nm']}): {' | '.join(parts)}")

    # MONTHLY
    print("\n"+"="*120);print("  MONTHLY PERFORMANCE");print("="*120)
    df['mo']=pd.to_datetime(df['xt']).dt.to_period('M')
    mg=df.groupby('mo')
    print(f"\n  {'Mo':>7}{'#':>5}{'W':>4}{'L':>4}{'WR':>5}{'GrP':>9}{'GrL':>9}{'Net':>9}{'PF':>5}{'Bal':>10}{'Ret':>7} A B C")
    print("  "+"-"*100)

    rb=init;yr={};lm=0;tm=0
    for mo in sorted(mg.groups.keys()):
        g=mg.get_group(mo);nt=len(g);nw=len(g[g['pnl']>0]);nl=nt-nw
        wr2=nw/nt*100 if nt else 0
        gp2=g[g['pnl']>0]['pnl'].sum() if nw else 0;gl2=abs(g[g['pnl']<=0]['pnl'].sum()) if nl else 0
        net=g['pnl'].sum();mpf=gp2/gl2 if gl2 else 999
        sbr=rb;rb+=net;mr=net/sbr*100 if sbr else 0;tm+=1
        if net<0:lm+=1
        ea=len(g[g['eng']=='A']);eb=len(g[g['eng']=='B']);ec=len(g[g['eng']=='C'])
        y=str(mo)[:4]
        if y not in yr:yr[y]={'p':0,'t':0,'w':0,'l':0,'gp':0,'gl':0,'sb':sbr}
        yr[y]['p']+=net;yr[y]['t']+=nt;yr[y]['w']+=nw;yr[y]['l']+=nl;yr[y]['gp']+=gp2;yr[y]['gl']+=gl2;yr[y]['eb']=rb
        pfs=f"{mpf:.1f}" if mpf<999 else "INF"
        mk=" <<" if net<0 else ""
        print(f"  {str(mo):>7}{nt:>5}{nw:>4}{nl:>4}{wr2:>4.0f}%${gp2:>7,.0f}${gl2:>7,.0f}${net:>+7,.0f}{pfs:>4}${rb:>8,.0f}{mr:>+6.1f}% {ea:>1} {eb:>1} {ec:>1}{mk}")

    # YEARLY
    print("\n"+"="*120);print("  YEARLY PERFORMANCE");print("="*120)
    print(f"\n  {'Yr':>6}{'#':>5}{'W':>5}{'L':>5}{'WR':>5}{'GrossP':>11}{'GrossL':>11}{'NetPnL':>11}{'PF':>6}{'Ret':>8}")
    print("  "+"-"*82)
    for y2 in sorted(yr):
        yd=yr[y2];ywr=yd['w']/yd['t']*100 if yd['t'] else 0
        ypf=yd['gp']/yd['gl'] if yd['gl'] else 999;yret=yd['p']/yd['sb']*100 if yd['sb'] else 0
        pfs=f"{ypf:.2f}" if ypf<999 else "INF"
        print(f"  {y2:>6}{yd['t']:>4}{yd['w']:>5}{yd['l']:>5}{ywr:>4.0f}%${yd['gp']:>9,.0f}${yd['gl']:>9,.0f}${yd['p']:>+9,.0f}{pfs:>5}{yret:>+7.1f}%")
    pyrs=sum(1 for v in yr.values() if v['p']>0)
    print(f"\n  LossMonths:{lm}/{tm}({lm/max(1,tm)*100:.0f}%) ProfitYears:{pyrs}/{len(yr)}")

    # ENGINE x YEAR
    print("\n"+"="*120);print("  ENGINE x YEAR");print("="*120)
    df['year']=pd.to_datetime(df['xt']).dt.year
    for ek in'ABC':
        et=df[df['eng']==ek]
        if not len(et):print(f"  {ek}({EC[ek]['nm']}): none");continue
        print(f"\n  {ek}({EC[ek]['nm']}):")
        for y3 in sorted(et['year'].unique()):
            yt=et[et['year']==y3];yw=yt[yt['pnl']>0];yl=yt[yt['pnl']<=0]
            ywr=len(yw)/len(yt)*100;ygp=yw['pnl'].sum() if len(yw) else 0
            ygl=abs(yl['pnl'].sum()) if len(yl) else 0;ypf=ygp/ygl if ygl else 999
            pfs=f"{ypf:.2f}" if ypf<999 else "INF"
            print(f"    {y3} {len(yt):>4} WR:{ywr:.0f}% PnL:${yt['pnl'].sum():>+8,.0f} AvgROI:{yt['roi'].mean():>+5.1f}% PF:{pfs}")

    # TOP/BOTTOM
    print("\n"+"="*120);print("  TOP 10 / BOTTOM 10");print("="*120)
    ds=df.sort_values('pnl',ascending=False)
    print("\n  Top 10:")
    for i2,(_,r) in enumerate(ds.head(10).iterrows()):
        print(f"    {i2+1:>2} {r['eng']} {r['dir_display']:>5} {str(r['et'])[:16]} {str(r['xt'])[:16]} {r['xr']:>3} ROI:{r['roi']:>+6.1f}% PnL:${r['pnl']:>8,.0f}")
    print("\n  Bottom 10:")
    for i2,(_,r) in enumerate(ds.tail(10).iterrows()):
        print(f"    {i2+1:>2} {r['eng']} {r['dir_display']:>5} {str(r['et'])[:16]} {str(r['xt'])[:16]} {r['xr']:>3} ROI:{r['roi']:>+6.1f}% PnL:${r['pnl']:>8,.0f}")

    print("\n"+"="*120);print("  COMPLETE");print("="*120)

if __name__=="__main__":
    trades,fbal,pkb,init,bc=run()
    rpt(trades,fbal,pkb,init,bc)
