"""BTC/USDT 고속 백테스트 엔진 (Numba JIT)"""
import numpy as np, pandas as pd, json, os, warnings
from numba import njit
warnings.filterwarnings('ignore')

def load_5m_data(d):
    parts=[]
    for f in sorted(os.listdir(d)):
        if f.startswith('btc_usdt_5m') and f.endswith('.csv'):
            parts.append(pd.read_csv(os.path.join(d,f),parse_dates=['timestamp']))
    if not parts: raise FileNotFoundError("5m data not found")
    df=pd.concat(parts,ignore_index=True).drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    df=df.rename(columns={'timestamp':'time'})
    for c in ['open','high','low','close','volume']: df[c]=df[c].astype(float)
    print(f"5m: {len(df):,} ({df['time'].iloc[0]}~{df['time'].iloc[-1]})")
    return df

def resample_tf(df,m):
    d=df.set_index('time')
    return d.resample(f'{m}min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()

def build_mtf(df):
    mtf={'5m':df.copy()}
    for m,n in [(10,'10m'),(15,'15m'),(30,'30m'),(60,'1h')]:
        mtf[n]=resample_tf(df,m); print(f"  {n}: {len(mtf[n]):,}")
    return mtf

def calc_ema(s,p): return s.ewm(span=p,adjust=False).mean()
def calc_sma(s,p): return s.rolling(p).mean()
def calc_wma(s,p):
    w=np.arange(1,p+1,dtype=float); return s.rolling(p).apply(lambda x:np.dot(x,w)/w.sum(),raw=True)
def calc_hma(s,p):
    h=max(int(p/2),1); sq=max(int(np.sqrt(p)),1)
    return calc_wma(2*calc_wma(s,h)-calc_wma(s,p),sq)
def calc_vwma(c,v,p): return (c*v).rolling(p).sum()/v.rolling(p).sum()
def calc_dema(s,p): e=calc_ema(s,p); return 2*e-calc_ema(e,p)
def calc_ma(c,t,p,v=None):
    if t=='ema': return calc_ema(c,p)
    if t=='sma': return calc_sma(c,p)
    if t=='wma': return calc_wma(c,p)
    if t=='hma': return calc_hma(c,p)
    if t=='dema': return calc_dema(c,p)
    if t=='vwma' and v is not None: return calc_vwma(c,v,p)
    return calc_ema(c,p)
def calc_rsi(c,p=14):
    d=c.diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    return 100-100/(1+g.ewm(alpha=1/p,min_periods=p,adjust=False).mean()/l.ewm(alpha=1/p,min_periods=p,adjust=False).mean().replace(0,np.nan))
def calc_adx(h,l,c,p=14):
    tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    up=h-h.shift(1); dn=l.shift(1)-l
    pdm=np.where((up>dn)&(up>0),up,0.0); mdm=np.where((dn>up)&(dn>0),dn,0.0)
    atr=pd.Series(tr,index=c.index).ewm(alpha=1/p,min_periods=p,adjust=False).mean()
    pdi=100*pd.Series(pdm,index=c.index).ewm(alpha=1/p,min_periods=p,adjust=False).mean()/atr
    mdi=100*pd.Series(mdm,index=c.index).ewm(alpha=1/p,min_periods=p,adjust=False).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/p,min_periods=p,adjust=False).mean()
def calc_atr(h,l,c,p=14):
    tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/p,min_periods=p,adjust=False).mean()

class IndicatorCache:
    def __init__(s,mtf): s.mtf=mtf; s.cache={}
    def get_base(s,tf):
        k=(tf,'base')
        if k not in s.cache:
            df=s.mtf[tf]
            s.cache[k]={'close':df['close'].values.astype(np.float64),'high':df['high'].values.astype(np.float64),
                'low':df['low'].values.astype(np.float64),'open':df['open'].values.astype(np.float64),
                'volume':df['volume'].values.astype(np.float64),'n':len(df)}
        return s.cache[k]
    def get_ma(s,tf,t,p):
        k=(tf,f'ma_{t}_{p}')
        if k not in s.cache: df=s.mtf[tf]; s.cache[k]=calc_ma(df['close'],t,p,df['volume']).values.astype(np.float64)
        return s.cache[k]
    def get_rsi(s,tf,p):
        k=(tf,f'rsi_{p}')
        if k not in s.cache: s.cache[k]=calc_rsi(s.mtf[tf]['close'],p).values.astype(np.float64)
        return s.cache[k]
    def get_adx(s,tf,p):
        k=(tf,f'adx_{p}')
        if k not in s.cache: df=s.mtf[tf]; s.cache[k]=calc_adx(df['high'],df['low'],df['close'],p).values.astype(np.float64)
        return s.cache[k]
    def get_atr(s,tf,p):
        k=(tf,f'atr_{p}')
        if k not in s.cache: df=s.mtf[tf]; s.cache[k]=calc_atr(df['high'],df['low'],df['close'],p).values.astype(np.float64)
        return s.cache[k]
    def get_times(s,tf):
        k=(tf,'times')
        if k not in s.cache: t=s.mtf[tf]['time']; s.cache[k]=(t.dt.year*100+t.dt.month).values.astype(np.int64)
        return s.cache[k]

@njit(cache=True)
def _bt_core(closes,highs,lows,months,ma_fast,ma_slow,rsi,adx,atr,
             adx_min,rsi_min,rsi_max,sl_pct,trail_act,trail_pct,
             use_atr_sl,atr_sl_mult,atr_sl_min,atr_sl_max,use_atr_trail,atr_trail_mult,
             leverage,margin_normal,margin_reduced,ml_limit,cp_pause,cp_candles,dd_thresh,fee,init_cap,
             delayed,delay_max,delay_pmin,delay_pmax):
    n=len(closes);bal=init_cap;peak_bal=bal;pos=0;ep=0.0;ei=0;su=0.0;ppnl=0.0;trail=False;rem=1.0
    msb=bal;cm=0;mp=False;cl=0;pu_idx=0;psig=0;sprice=0.0;sidx=0
    tot=0;wc=0;lc=0;gp=0.0;gl=0.0;slc=0;tslc=0;revc=0;flc=0;rpeak=init_cap;mdd=0.0
    yrs=np.zeros(10,dtype=np.float64);yre=np.zeros(10,dtype=np.float64);yrk=np.zeros(10,dtype=np.int64);yrc=0;lm=0
    wps=0.0;lps=0.0;mwp=0.0;mlp=0.0;liq_d=1.0/leverage
    for i in range(1,n):
        cp=closes[i];hp=highs[i];lp=lows[i]
        if np.isnan(ma_fast[i]) or np.isnan(ma_slow[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]): continue
        mk=months[i]
        if mk!=cm:
            if cm!=0 and bal<msb: lm+=1
            cm=mk;msb=bal;mp=False
        yk=mk//100
        if yrc==0 or yrk[yrc-1]!=yk:
            if yrc<10: yrk[yrc]=yk;yrs[yrc]=bal;yre[yrc]=bal;yrc+=1
        if yrc>0: yre[yrc-1]=bal
        if pos!=0:
            if pos==1: pnl=(cp-ep)/ep;pkc=(hp-ep)/ep;lwc=(lp-ep)/ep
            else: pnl=(ep-cp)/ep;pkc=(ep-lp)/ep;lwc=(ep-hp)/ep
            if pkc>ppnl: ppnl=pkc
            if lwc<=-liq_d:
                pu2=su*rem*(-liq_d)-su*rem*fee;bal+=pu2
                if bal<0:bal=0
                tot+=1;lc+=1;gl+=abs(pu2);flc+=1;lps+=liq_d;pos=0;cl+=1
                if cp_pause>0 and cl>=cp_pause:pu_idx=i+cp_candles
                if bal>rpeak:rpeak=bal
                dd=(rpeak-bal)/rpeak if rpeak>0 else 0
                if dd>mdd:mdd=dd
                continue
            sth=sl_pct
            if use_atr_sl and not np.isnan(atr[i]):
                a=(atr[i]*atr_sl_mult)/ep
                if a<atr_sl_min:a=atr_sl_min
                if a>atr_sl_max:a=atr_sl_max
                sth=a
            if lwc<=-sth:
                pu2=su*rem*(-sth)-su*rem*fee;bal+=pu2
                if bal<0:bal=0
                tot+=1;lc+=1;gl+=abs(pu2);slc+=1;lps+=sth;pos=0;cl+=1
                if cp_pause>0 and cl>=cp_pause:pu_idx=i+cp_candles
                if bal>rpeak:rpeak=bal
                dd=(rpeak-bal)/rpeak if rpeak>0 else 0
                if dd>mdd:mdd=dd
                continue
            if ppnl>=trail_act:trail=True
            if trail:
                tw=trail_pct
                if use_atr_trail and not np.isnan(atr[i]):
                    at=(atr[i]*atr_trail_mult)/ep
                    if at>tw:tw=at
                    if tw<0.02:tw=0.02
                tl=ppnl-tw
                if pnl<=tl:
                    pu2=su*rem*tl-su*rem*fee;bal+=pu2;tot+=1;tslc+=1
                    if tl>0:wc+=1;gp+=pu2;wps+=tl;cl=0
                    else:lc+=1;gl+=abs(pu2);lps+=abs(tl);cl+=1
                    if cl>0 and cp_pause>0 and cl>=cp_pause:pu_idx=i+cp_candles
                    pos=0
                    if bal>rpeak:rpeak=bal
                    dd=(rpeak-bal)/rpeak if rpeak>0 else 0
                    if dd>mdd:mdd=dd
                    continue
            cu=ma_fast[i]>ma_slow[i] and ma_fast[i-1]<=ma_slow[i-1]
            cd=ma_fast[i]<ma_slow[i] and ma_fast[i-1]>=ma_slow[i-1]
            ao=adx[i]>=adx_min;ro=rsi_min<=rsi[i]<=rsi_max
            rv=False;nd=0
            if pos==1 and cd and ao and ro:rv=True;nd=-1
            elif pos==-1 and cu and ao and ro:rv=True;nd=1
            if rv:
                pu2=su*rem*pnl-su*rem*fee;bal+=pu2;tot+=1;revc+=1
                if pnl>0:wc+=1;gp+=pu2;wps+=pnl;cl=0
                else:lc+=1;gl+=abs(pu2);lps+=abs(pnl);cl+=1
                if cl>0 and cp_pause>0 and cl>=cp_pause:pu_idx=i+cp_candles
                pos=0
                if bal>rpeak:rpeak=bal
                dd=(rpeak-bal)/rpeak if rpeak>0 else 0
                if dd>mdd:mdd=dd
                can=not mp and i>=pu_idx
                if bal>10 and can:
                    mg=margin_normal
                    if peak_bal>0 and dd_thresh<0:
                        dn=(peak_bal-bal)/peak_bal
                        if dn>abs(dd_thresh):mg=margin_reduced
                    mu=bal*mg;s2=mu*leverage;bal-=s2*fee;pos=nd;ep=cp;ei=i;su=s2;ppnl=0.0;trail=False;rem=1.0
                    if bal>peak_bal:peak_bal=bal
                continue
        if pos==0 and bal>10:
            if ml_limit<0 and msb>0:
                mr=(bal-msb)/msb
                if mr<ml_limit:mp=True
            can=not mp and i>=pu_idx
            if not can:psig=0;continue
            cu=ma_fast[i]>ma_slow[i] and ma_fast[i-1]<=ma_slow[i-1]
            cd=ma_fast[i]<ma_slow[i] and ma_fast[i-1]>=ma_slow[i-1]
            ao=adx[i]>=adx_min;ro=rsi_min<=rsi[i]<=rsi_max
            sig=0
            if cu and ao and ro:sig=1
            elif cd and ao and ro:sig=-1
            do=False;ed=0
            if sig!=0:
                if delayed:psig=sig;sprice=cp;sidx=i
                else:do=True;ed=sig
            elif psig!=0 and delayed:
                el=i-sidx
                if el>delay_max:psig=0
                else:
                    pc=(cp-sprice)/sprice
                    if psig==1:
                        if delay_pmax<=pc<=delay_pmin:do=True;ed=1;psig=0
                    elif psig==-1:
                        iv=-pc
                        if delay_pmax<=iv<=delay_pmin:do=True;ed=-1;psig=0
            if do:
                mg=margin_normal
                if peak_bal>0 and dd_thresh<0:
                    dn=(peak_bal-bal)/peak_bal
                    if dn>abs(dd_thresh):mg=margin_reduced
                mu=bal*mg;s2=mu*leverage;bal-=s2*fee;pos=ed;ep=cp;ei=i;su=s2;ppnl=0.0;trail=False;rem=1.0
                if bal>peak_bal:peak_bal=bal
        if bal>peak_bal:peak_bal=bal
        if bal>rpeak:rpeak=bal
    if cm!=0 and bal<msb:lm+=1
    if pos!=0 and n>0:
        pf=0.0
        if pos==1:pf=(closes[n-1]-ep)/ep
        else:pf=(ep-closes[n-1])/ep
        pu2=su*rem*pf-su*rem*fee;bal+=pu2;tot+=1
        if pf>0:wc+=1;gp+=pu2;wps+=pf
        else:lc+=1;gl+=abs(pu2);lps+=abs(pf)
    if yrc>0:yre[yrc-1]=bal
    return (bal,tot,wc,lc,gp,gl,slc,tslc,revc,flc,mdd,yrk[:yrc],yrs[:yrc],yre[:yrc],lm,wps,lps,mwp,mlp)

def run_backtest(cache,tf,cfg):
    base=cache.get_base(tf)
    if base['n']<200:return None
    r=_bt_core(base['close'],base['high'],base['low'],cache.get_times(tf),
        cache.get_ma(tf,cfg.get('ma_fast_type','ema'),cfg.get('ma_fast',7)),
        cache.get_ma(tf,cfg.get('ma_slow_type','ema'),cfg.get('ma_slow',100)),
        cache.get_rsi(tf,cfg.get('rsi_period',14)),cache.get_adx(tf,cfg.get('adx_period',14)),
        cache.get_atr(tf,cfg.get('atr_period',14)),
        cfg.get('adx_min',30.0),cfg.get('rsi_min',30.0),cfg.get('rsi_max',58.0),
        cfg.get('sl_pct',0.07),cfg.get('trail_activate',0.08),cfg.get('trail_pct',0.06),
        cfg.get('use_atr_sl',False),cfg.get('atr_sl_mult',2.0),cfg.get('atr_sl_min',0.02),cfg.get('atr_sl_max',0.12),
        cfg.get('use_atr_trail',False),cfg.get('atr_trail_mult',1.5),
        cfg.get('leverage',10),cfg.get('margin_normal',0.20),cfg.get('margin_reduced',0.10),
        cfg.get('monthly_loss_limit',0.0),cfg.get('consec_loss_pause',0),cfg.get('pause_candles',0),
        cfg.get('dd_threshold',0.0),cfg.get('fee_rate',0.0004),cfg.get('initial_capital',3000.0),
        cfg.get('delayed_entry',False),cfg.get('delay_max_candles',6),
        cfg.get('delay_price_min',-0.001),cfg.get('delay_price_max',-0.025))
    (bal,tot,wc,lc,gp,gl,slc,tslc,revc,flc,mdd_v,yrk,yrs,yre,lm,wps,lps,mwp,mlp)=r
    if tot<1:return None
    ic=cfg.get('initial_capital',3000.0);pf=gp/gl if gl>0 else gp;wr=wc/tot*100 if tot>0 else 0
    yearly={}
    for j in range(len(yrk)):
        if yrs[j]>0:yearly[str(int(yrk[j]))]=round((yre[j]-yrs[j])/yrs[j]*100,1)
    return {'bal':round(bal,0),'ret':round((bal-ic)/ic*100,1),'trades':int(tot),'wr':round(wr,1),
        'pf':round(pf,2),'mdd':round(mdd_v*100,1),'sl':int(slc),'tsl':int(tslc),'sig':int(revc),
        'liq':int(flc),'cfg':_cs(cfg),'yr':yearly,'lm':int(lm),
        'avg_win':round(wps/wc*100,2) if wc>0 else 0,'avg_loss':round(-lps/lc*100,2) if lc>0 else 0,
        'max_win':round(mwp*100,2),'max_loss':round(-mlp*100,2)}

def _cs(c):
    p=[f"{c.get('timeframe','5m')}",f"{c.get('ma_fast_type','ema')}({c.get('ma_fast',7)}/{c.get('ma_slow',100)})",
       f"A{c.get('adx_period',14)}>={c.get('adx_min',30):.0f}",f"R{c.get('rsi_period',14)}:{c.get('rsi_min',30):.0f}-{c.get('rsi_max',58):.0f}",
       f"SL{c.get('sl_pct',0.07):.2f}"]
    if c.get('use_atr_sl',False):p.append(f"ATR_SL{c.get('atr_sl_mult',2.0):.1f}")
    p.append(f"TA{c.get('trail_activate',0.08):.2f}/{c.get('trail_pct',0.06):.2f}")
    if c.get('use_atr_trail',False):p.append(f"ATR_TS{c.get('atr_trail_mult',1.5):.1f}")
    if c.get('delayed_entry',False):p.append(f"DE{c.get('delay_max_candles',6)}c")
    p.append(f"L{c.get('leverage',10)}x M{c.get('margin_normal',0.20):.0%}")
    pr=[]
    if c.get('monthly_loss_limit',0)<0:pr.append(f"ML{c.get('monthly_loss_limit',0):.0%}")
    if c.get('consec_loss_pause',0)>0:pr.append(f"CP{c.get('pause_candles',0)}")
    if c.get('dd_threshold',0)<0:pr.append(f"DD{c.get('dd_threshold',0):.0%}")
    p.append(' '.join(pr) if pr else '')
    return ' | '.join(p)

def print_result(r,rank=0):
    pf=f"#{rank} " if rank else ""
    print(f"{pf}${r['bal']:,.0f} ({r['ret']:+,.1f}%) PF:{r['pf']:.2f} MDD:{r['mdd']:.1f}% TR:{r['trades']} SL:{r['sl']} TSL:{r['tsl']} REV:{r['sig']} FL:{r['liq']}")
    print(f"  {r['cfg']}")
    if r.get('yr'):print(f"  "+' | '.join(f"{k}:{v:+.1f}%" for k,v in sorted(r['yr'].items())))

def save_results(results,fn):
    with open(fn,'w',encoding='utf-8') as f:json.dump(results,f,indent=2,ensure_ascii=False)

def score(r):
    pf=r.get('pf',0);ret=r.get('ret',0);mdd=r.get('mdd',100);fl=r.get('liq',0)
    if ret<=0 or pf<=0:return 0
    base=pf*np.log1p(ret/100)
    pb=1.5 if pf>=7 else(1.2 if pf>=5 else 1.0)
    mp=0.3 if mdd>80 else(0.5 if mdd>60 else(0.8 if mdd>40 else 1.0))
    fp=max(0,1-fl*0.2)
    yr=r.get('yr',{});rec=[yr.get(y,0) for y in['2023','2024','2025','2026'] if y in yr]
    rb=1.0
    if rec:
        a=np.mean(rec);rb=1.5 if a>80 else(1.3 if a>40 else(1.1 if a>10 else(0.6 if a<-10 else 1.0)))
    return base*pb*mp*fp*rb
