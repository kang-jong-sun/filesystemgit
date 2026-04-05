"""v26.0 상세 월간 데이터: A단독 / B단독 / 병합"""
import sys, time, json
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = r"D:\filesystem\futures\btc_V1\test1\btc_usdt_5m_merged.csv"

def ema(d,p): return pd.Series(d).ewm(span=p,adjust=False).mean().values
def sma(d,p): return pd.Series(d).rolling(p,min_periods=p).mean().values
def wma(d,p):
    w=np.arange(1,p+1,dtype=float)
    return pd.Series(d).rolling(p).apply(lambda x:np.dot(x,w)/w.sum(),raw=True).values
def calc_adx(hi,lo,cl,p=20):
    tr1=hi-lo;tr2=np.abs(hi-np.roll(cl,1));tr3=np.abs(lo-np.roll(cl,1))
    tr1[0]=tr2[0]=tr3[0]=0;tr=np.maximum(np.maximum(tr1,tr2),tr3)
    up=hi-np.roll(hi,1);dn=np.roll(lo,1)-lo;up[0]=dn[0]=0
    pdm=np.where((up>dn)&(up>0),up,0.0);mdm=np.where((dn>up)&(dn>0),dn,0.0)
    a=1.0/p
    atr_v=pd.Series(tr).ewm(alpha=a,min_periods=p,adjust=False).mean().values
    pds=pd.Series(pdm).ewm(alpha=a,min_periods=p,adjust=False).mean().values
    mds=pd.Series(mdm).ewm(alpha=a,min_periods=p,adjust=False).mean().values
    with np.errstate(divide='ignore',invalid='ignore'):
        pdi=np.where(atr_v>0,100*pds/atr_v,0);mdi=np.where(atr_v>0,100*mds/atr_v,0)
        ds=pdi+mdi;dx=np.where(ds>0,100*np.abs(pdi-mdi)/ds,0)
    return pd.Series(dx).ewm(alpha=a,min_periods=p,adjust=False).mean().values
def calc_rsi(cl,p=14):
    d2=np.diff(cl,prepend=cl[0])
    g=np.where(d2>0,d2,0.0);l=np.where(d2<0,-d2,0.0)
    ga=pd.Series(g).ewm(alpha=1/p,min_periods=p,adjust=False).mean().values
    la=pd.Series(l).ewm(alpha=1/p,min_periods=p,adjust=False).mean().values
    with np.errstate(divide='ignore',invalid='ignore'):
        rs=np.where(la>0,ga/la,100.0)
    return 100-100/(1+rs)

def gen_signals(cl, hi, lo, vo, mf_type, mf_p, ms_type, ms_p, adx_min, rsi_min, rsi_max, delay, adx_p=20):
    n = len(cl)
    if mf_type=='WMA': mf=wma(cl,mf_p)
    elif mf_type=='SMA': mf=sma(cl,mf_p)
    else: mf=ema(cl,mf_p)
    if ms_type=='WMA': ms=wma(cl,ms_p)
    elif ms_type=='SMA': ms=sma(cl,ms_p)
    else: ms=ema(cl,ms_p)
    adx_v=calc_adx(hi,lo,cl,adx_p); rsi_v=calc_rsi(cl,14)
    warmup=max(ms_p+50,250)
    valid=~(np.isnan(mf)|np.isnan(ms)); above=mf>ms
    raw=np.zeros(n,dtype=int)
    for i in range(warmup,n):
        if valid[i] and valid[i-1]:
            ao=adx_v[i]>=adx_min if not np.isnan(adx_v[i]) else False
            ro=rsi_min<=rsi_v[i]<=rsi_max if not np.isnan(rsi_v[i]) else False
            if above[i] and not above[i-1] and ao and ro: raw[i]=1
            elif not above[i] and above[i-1] and ao and ro: raw[i]=-1
    sig=np.zeros(n,dtype=int); pend=0;pcnt=0
    for i in range(n):
        if raw[i]!=0:
            if delay>0: pend=raw[i];pcnt=delay
            else: sig[i]=raw[i]
        if pcnt>0:
            pcnt-=1
            if pcnt==0:
                if pend==1 and valid[i] and mf[i]>ms[i]: sig[i]=pend
                elif pend==-1 and valid[i] and mf[i]<ms[i]: sig[i]=pend
                pend=0
    return sig

def run_bt(cl, hi, lo, yr, mk, ts_pd, signals, sl_pct, trail_act, trail_pct,
           leverage, margin_pct, init_bal, label=''):
    n=len(cl);bal=init_bal;peak=init_bal;mdd=0.0
    trades=[];in_pos=False;p_dir=0;p_entry=0.0;p_size=0.0
    p_sl=0.0;p_high=0.0;p_low=0.0;t_active=False;t_sl=0.0
    cur_m=-1;m_start=bal;m_locked=False;fee=0.0004;inv_lev=1.0/leverage

    for i in range(1,n):
        c=cl[i];h=hi[i];l=lo[i]
        m=mk[i]
        if m!=cur_m: cur_m=m;m_start=bal;m_locked=False
        if not m_locked and m_start>0:
            if (bal-m_start)/m_start<=-0.20: m_locked=True
        if in_pos:
            if p_dir==1 and l<=p_sl:
                pnl=p_size*(p_sl-p_entry)/p_entry-p_size*fee;bal+=pnl
                trades.append({'time':ts_pd[i],'dir':'L','pnl':pnl,'reason':'SL','bal':bal,'src':label});in_pos=False
            elif p_dir==-1 and h>=p_sl:
                pnl=p_size*(p_entry-p_sl)/p_entry-p_size*fee;bal+=pnl
                trades.append({'time':ts_pd[i],'dir':'S','pnl':pnl,'reason':'SL','bal':bal,'src':label});in_pos=False
            elif in_pos:
                if p_dir==1:
                    if h>p_high:p_high=h
                    if (p_high-p_entry)/p_entry*leverage>=trail_act*leverage:
                        t_active=True;ns=p_high*(1-trail_pct)
                        if ns>t_sl:t_sl=ns
                    if t_active and c<=t_sl:
                        pnl=p_size*(c-p_entry)/p_entry-p_size*fee;bal+=pnl
                        trades.append({'time':ts_pd[i],'dir':'L','pnl':pnl,'reason':'TSL','bal':bal,'src':label});in_pos=False
                else:
                    if l<p_low:p_low=l
                    if (p_entry-p_low)/p_entry*leverage>=trail_act*leverage:
                        t_active=True;ns=p_low*(1+trail_pct)
                        if t_sl==0 or ns<t_sl:t_sl=ns
                    if t_active and c>=t_sl:
                        pnl=p_size*(p_entry-c)/p_entry-p_size*fee;bal+=pnl
                        trades.append({'time':ts_pd[i],'dir':'S','pnl':pnl,'reason':'TSL','bal':bal,'src':label});in_pos=False
        sig=signals[i]
        if sig!=0:
            if in_pos and p_dir!=sig:
                pnl=p_size*(c-p_entry)/p_entry*p_dir-p_size*fee;bal+=pnl
                trades.append({'time':ts_pd[i],'dir':'L' if p_dir==1 else 'S','pnl':pnl,'reason':'REV','bal':bal,'src':label});in_pos=False
            if not in_pos and not m_locked and bal>10:
                mg=bal*margin_pct;sz=mg*leverage;bal-=sz*fee
                p_dir=sig;p_entry=c;p_size=sz;p_sl=c*(1-sl_pct) if sig==1 else c*(1+sl_pct)
                p_high=c;p_low=c;t_active=False;t_sl=0;in_pos=True
        if bal>peak:peak=bal
        if peak>0:
            dd=(peak-bal)/peak
            if dd>mdd:mdd=dd
    if in_pos:
        c=cl[-1];pnl=p_size*(c-p_entry)/p_entry*p_dir-p_size*fee;bal+=pnl
        trades.append({'time':ts_pd[-1],'dir':'L' if p_dir==1 else 'S','pnl':pnl,'reason':'END','bal':bal,'src':label})
    return trades, bal, mdd

def merged_bt(cl, hi, lo, yr, mk, ts_pd, sig_a, sig_b, sl_pct, trail_act, trail_pct,
              leverage, margin_pct, init_bal):
    n=len(cl);bal=init_bal;peak=init_bal;mdd=0.0
    trades=[];in_pos=False;p_dir=0;p_entry=0.0;p_size=0.0
    p_sl=0.0;p_high=0.0;p_low=0.0;t_active=False;t_sl=0.0
    cur_m=-1;m_start=bal;m_locked=False;fee=0.0004

    for i in range(1,n):
        c=cl[i];h=hi[i];l=lo[i]
        m=mk[i]
        if m!=cur_m:cur_m=m;m_start=bal;m_locked=False
        if not m_locked and m_start>0:
            if (bal-m_start)/m_start<=-0.20:m_locked=True
        if in_pos:
            if p_dir==1 and l<=p_sl:
                pnl=p_size*(p_sl-p_entry)/p_entry-p_size*fee;bal+=pnl
                trades.append({'time':ts_pd[i],'dir':'L','pnl':pnl,'reason':'SL','bal':bal,'src':'POS'});in_pos=False
            elif p_dir==-1 and h>=p_sl:
                pnl=p_size*(p_entry-p_sl)/p_entry-p_size*fee;bal+=pnl
                trades.append({'time':ts_pd[i],'dir':'S','pnl':pnl,'reason':'SL','bal':bal,'src':'POS'});in_pos=False
            elif in_pos:
                if p_dir==1:
                    if h>p_high:p_high=h
                    if (p_high-p_entry)/p_entry*leverage>=trail_act*leverage:
                        t_active=True;ns=p_high*(1-trail_pct)
                        if ns>t_sl:t_sl=ns
                    if t_active and c<=t_sl:
                        pnl=p_size*(c-p_entry)/p_entry-p_size*fee;bal+=pnl
                        trades.append({'time':ts_pd[i],'dir':'L','pnl':pnl,'reason':'TSL','bal':bal,'src':'POS'});in_pos=False
                else:
                    if l<p_low:p_low=l
                    if (p_entry-p_low)/p_entry*leverage>=trail_act*leverage:
                        t_active=True;ns=p_low*(1+trail_pct)
                        if t_sl==0 or ns<t_sl:t_sl=ns
                    if t_active and c>=t_sl:
                        pnl=p_size*(p_entry-c)/p_entry-p_size*fee;bal+=pnl
                        trades.append({'time':ts_pd[i],'dir':'S','pnl':pnl,'reason':'TSL','bal':bal,'src':'POS'});in_pos=False
        # Merge signals
        sa=sig_a[i];sb=sig_b[i]
        sig=0;src=''
        if sa!=0 and sb!=0:
            if sa==sb: sig=sa;src='A+B'
            else: sig=sa;src='A(conflict)'
        elif sa!=0: sig=sa;src='A'
        elif sb!=0: sig=sb;src='B'

        if sig!=0:
            if in_pos and p_dir!=sig:
                pnl=p_size*(c-p_entry)/p_entry*p_dir-p_size*fee;bal+=pnl
                trades.append({'time':ts_pd[i],'dir':'L' if p_dir==1 else 'S','pnl':pnl,'reason':'REV','bal':bal,'src':src});in_pos=False
            if not in_pos and not m_locked and bal>10:
                mg=bal*margin_pct;sz=mg*leverage;bal-=sz*fee
                p_dir=sig;p_entry=c;p_size=sz;p_sl=c*(1-sl_pct) if sig==1 else c*(1+sl_pct)
                p_high=c;p_low=c;t_active=False;t_sl=0;in_pos=True
        if bal>peak:peak=bal
        if peak>0:
            dd=(peak-bal)/peak
            if dd>mdd:mdd=dd
    if in_pos:
        c=cl[-1];pnl=p_size*(c-p_entry)/p_entry*p_dir-p_size*fee;bal+=pnl
        trades.append({'time':ts_pd[-1],'dir':'L' if p_dir==1 else 'S','pnl':pnl,'reason':'END','bal':bal,'src':'END'})
    return trades, bal, mdd

def print_detail(trades, bal, init, mdd, title):
    tc=len(trades)
    if tc==0:
        print('  (no trades)')
        return
    pnls=np.array([t['pnl'] for t in trades])
    w=pnls>0;l=pnls<=0
    gp=pnls[w].sum() if w.any() else 0
    gl=abs(pnls[l].sum()) if l.any() else 0.001
    pf=min(gp/gl,999.99)
    print('  $%.0f -> $%.0f (%+.1f%%) | Tr=%d PF=%.2f MDD=%.1f%% WR=%.1f%%' % (
        init,bal,(bal-init)/init*100,tc,pf,mdd*100,w.sum()/tc*100))
    print('  SL=%d TSL=%d REV=%d' % (
        sum(1 for t in trades if t['reason']=='SL'),
        sum(1 for t in trades if t['reason']=='TSL'),
        sum(1 for t in trades if t['reason']=='REV')))

    # Yearly
    print('\n  [Yearly]')
    print('  %6s %12s %12s %10s %5s %4s %4s %4s %8s' % ('Year','Start','End','Return','Tr','SL','TSL','REV','Source'))
    print('  '+'-'*75)
    yearly={};rb=init
    for t in trades:
        y=t['time'].year
        if y not in yearly: yearly[y]={'start':rb,'trades':0,'pnl':0,'sl':0,'tsl':0,'rev':0,'srcs':[]}
        yearly[y]['trades']+=1;yearly[y]['pnl']+=t['pnl']
        if t['reason']=='SL':yearly[y]['sl']+=1
        elif t['reason']=='TSL':yearly[y]['tsl']+=1
        elif t['reason']=='REV':yearly[y]['rev']+=1
        yearly[y]['end']=t['bal'];yearly[y]['srcs'].append(t.get('src',''))
        rb=t['bal']
    for y in sorted(yearly.keys()):
        ys=yearly[y]
        ret=(ys['end']-ys['start'])/ys['start']*100 if ys['start']>0 else 0
        srcs=set(s for s in ys['srcs'] if s and s not in ('POS','END'))
        print('  %6d $%10s $%10s %+9.1f%% %5d %4d %4d %4d %8s' % (
            y,'{:,.0f}'.format(ys['start']),'{:,.0f}'.format(ys['end']),
            ret,ys['trades'],ys['sl'],ys['tsl'],ys['rev'],
            ','.join(srcs) if srcs else '-'))

    # Monthly
    print('\n  [Monthly]')
    print('  %8s %10s %5s %12s %12s %8s' % ('Month','Return','Tr','PnL','Balance','Source'))
    print('  '+'-'*65)
    monthly={};rb=init
    for t in trades:
        mk=t['time'].strftime('%Y-%m')
        if mk not in monthly: monthly[mk]={'start':rb,'trades':0,'pnl':0,'srcs':[]}
        monthly[mk]['trades']+=1;monthly[mk]['pnl']+=t['pnl']
        monthly[mk]['end']=t['bal'];monthly[mk]['srcs'].append(t.get('src',''))
        rb=t['bal']
    for mk in sorted(monthly.keys()):
        ms=monthly[mk]
        if ms['trades']>0:
            ret=(ms['end']-ms['start'])/ms['start']*100 if ms['start']>0 else 0
            srcs=set(s for s in ms['srcs'] if s and s not in ('POS','END'))
            print('  %8s %+9.1f%% %5d $%10s $%10s %8s' % (
                mk,ret,ms['trades'],
                '{:,.0f}'.format(ms['pnl']),'{:,.0f}'.format(ms['end']),
                ','.join(srcs) if srcs else '-'))

def main():
    print('Loading data...')
    df=pd.read_csv(DATA_PATH,parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    d=df.set_index('timestamp')
    r30=d.resample('30min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
    ts_pd=pd.to_datetime(r30['timestamp'].values)
    cl=r30['close'].values.astype(float);hi=r30['high'].values.astype(float)
    lo=r30['low'].values.astype(float);vo=r30['volume'].values.astype(float)
    yr=ts_pd.year.values.astype(np.int32);mk=(yr*100+ts_pd.month.values).astype(np.int32)
    print('  30min: %d bars' % len(cl))

    SL=0.08;TA=0.03;TP=0.02;LEV=10;MARGIN=0.50;INIT=5000.0

    # v26.0 Track A: EMA(3)/SMA(300) ADX>=40 D5
    sig_a=gen_signals(cl,hi,lo,vo,'EMA',3,'SMA',300,40,30,70,5,20)
    # v26.0 Track B: WMA(3)/EMA(200) ADX>=35 D5
    sig_b=gen_signals(cl,hi,lo,vo,'WMA',3,'EMA',200,35,30,70,5,20)

    print('  Track A signals: %d' % np.sum(sig_a!=0))
    print('  Track B signals: %d' % np.sum(sig_b!=0))

    # ===== A alone =====
    print('\n' + '='*70)
    print('  v26.0 Track A ALONE (EMA3/SMA300 ADX>=40 D5)')
    print('='*70)
    tr_a, bal_a, mdd_a = run_bt(cl,hi,lo,yr,mk,ts_pd,sig_a,SL,TA,TP,LEV,MARGIN,INIT,'A')
    print_detail(tr_a, bal_a, INIT, mdd_a, 'A alone')

    # ===== B alone =====
    print('\n' + '='*70)
    print('  v26.0 Track B ALONE (WMA3/EMA200 ADX>=35 D5)')
    print('='*70)
    tr_b, bal_b, mdd_b = run_bt(cl,hi,lo,yr,mk,ts_pd,sig_b,SL,TA,TP,LEV,MARGIN,INIT,'B')
    print_detail(tr_b, bal_b, INIT, mdd_b, 'B alone')

    # ===== A+B merged =====
    print('\n' + '='*70)
    print('  v26.0 MERGED (A+B on 1 account)')
    print('='*70)
    tr_m, bal_m, mdd_m = merged_bt(cl,hi,lo,yr,mk,ts_pd,sig_a,sig_b,SL,TA,TP,LEV,MARGIN,INIT)
    print_detail(tr_m, bal_m, INIT, mdd_m, 'Merged')

    # Trade list
    print('\n  [Trade List - Merged]')
    print('  %4s %19s %4s %10s %8s %5s %8s' % ('#','Time','Dir','PnL','Reason','Src','Balance'))
    print('  '+'-'*65)
    for i,t in enumerate(tr_m):
        print('  %4d %19s %4s $%9s %8s %5s $%8s' % (
            i+1, str(t['time'])[:19], t['dir'],
            '{:,.0f}'.format(t['pnl']), t['reason'], t.get('src','-'),
            '{:,.0f}'.format(t['bal'])))

    print('\n' + '='*70)
    print('  SUMMARY')
    print('='*70)
    print('  %15s %10s %5s %7s %6s' % ('Mode','Return','Tr','PF','MDD%'))
    print('  '+'-'*50)
    for name,tr,bal,mdd_v in [('A alone',tr_a,bal_a,mdd_a),('B alone',tr_b,bal_b,mdd_b),('A+B Merged',tr_m,bal_m,mdd_m)]:
        tc=len(tr);pnls=np.array([t['pnl'] for t in tr]) if tc>0 else np.array([0])
        w=pnls>0;gp=pnls[w].sum() if w.any() else 0;gl=abs(pnls[pnls<=0].sum()) if (pnls<=0).any() else 0.001
        pf=min(gp/gl,999.99)
        print('  %15s %+9.1f%% %5d %6.2f %5.1f%%' % (name,(bal-INIT)/INIT*100,tc,pf,mdd_v*100))

if __name__=='__main__':
    main()
