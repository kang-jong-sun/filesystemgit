"""
6-Engine Cross Verification System
Engine 1: Wilder ADX + close-cross + SL on high/low (Standard)
Engine 2: EWM ADX (pandas default)
Engine 3: Strict entry (2 consecutive closes required)
Engine 4: With slippage ±0.05%
Engine 5: Higher fees (0.06%)
Engine 6: Same-direction re-entry (close + re-enter)

20 unique strategies × 6 engines = 120 backtests
"""
import sys, time, json, os
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = r"D:\filesystem\futures\btc_V1\test1\btc_usdt_5m_merged.csv"
OUTPUT_DIR = r"D:\filesystem\futures\btc_V1\test1\cross_verify"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ INDICATORS ============
def ema(d,p): return pd.Series(d).ewm(span=p,adjust=False).mean().values
def sma_f(d,p): return pd.Series(d).rolling(p,min_periods=p).mean().values
def wma_f(d,p):
    w=np.arange(1,p+1,dtype=float)
    return pd.Series(d).rolling(p).apply(lambda x:np.dot(x,w)/w.sum(),raw=True).values
def hma_f(d,p):
    h=max(int(p/2),1);sq=max(int(np.sqrt(p)),1)
    wh=wma_f(d,h);wf=wma_f(d,p);diff=2*wh-wf
    m=~np.isnan(diff);r=np.full_like(d,np.nan,dtype=float)
    if m.sum()>sq: r[m]=wma_f(diff[m],sq)
    return r
def vwma_f(cl,vo,p):
    cv=pd.Series(cl*vo).rolling(p,min_periods=p).sum().values
    v=pd.Series(vo).rolling(p,min_periods=p).sum().values
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.where(v>0,cv/v,np.nan)

def calc_ma(cl,vo,ma_type,p):
    if ma_type=='WMA': return wma_f(cl,p)
    elif ma_type=='SMA': return sma_f(cl,p)
    elif ma_type=='HMA': return hma_f(cl,p)
    elif ma_type=='VWMA': return vwma_f(cl,vo,p)
    return ema(cl,p)

def adx_wilder(hi,lo,cl,p):
    """Proper Wilder smoothing ADX"""
    n=len(cl)
    tr=np.zeros(n);pdm=np.zeros(n);mdm=np.zeros(n)
    for i in range(1,n):
        tr1=hi[i]-lo[i];tr2=abs(hi[i]-cl[i-1]);tr3=abs(lo[i]-cl[i-1])
        tr[i]=max(tr1,tr2,tr3)
        up=hi[i]-hi[i-1];dn=lo[i-1]-lo[i]
        pdm[i]=up if up>dn and up>0 else 0
        mdm[i]=dn if dn>up and dn>0 else 0
    # Wilder smoothing: first value = mean, then alpha=1/p
    atr=np.zeros(n);pdi_s=np.zeros(n);mdi_s=np.zeros(n)
    if n>p:
        atr[p]=np.mean(tr[1:p+1])
        pdi_s[p]=np.mean(pdm[1:p+1])
        mdi_s[p]=np.mean(mdm[1:p+1])
        for i in range(p+1,n):
            atr[i]=atr[i-1]*(1-1/p)+tr[i]/p
            pdi_s[i]=pdi_s[i-1]*(1-1/p)+pdm[i]/p
            mdi_s[i]=mdi_s[i-1]*(1-1/p)+mdm[i]/p
    dx=np.zeros(n)
    for i in range(p,n):
        if atr[i]>0:
            pdi=100*pdi_s[i]/atr[i];mdi=100*mdi_s[i]/atr[i]
            s=pdi+mdi
            dx[i]=100*abs(pdi-mdi)/s if s>0 else 0
    adx_arr=np.zeros(n)
    start=2*p
    if n>start:
        adx_arr[start]=np.mean(dx[p:start+1])
        for i in range(start+1,n):
            adx_arr[i]=adx_arr[i-1]*(1-1/p)+dx[i]/p
    return adx_arr

def adx_ewm(hi,lo,cl,p):
    """EWM-based ADX (pandas default, less accurate)"""
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

def calc_atr(hi,lo,cl,p=14):
    n=len(cl);tr=np.zeros(n)
    for i in range(1,n):
        tr[i]=max(hi[i]-lo[i],abs(hi[i]-cl[i-1]),abs(lo[i]-cl[i-1]))
    atr=np.zeros(n)
    if n>p:
        atr[p]=np.mean(tr[1:p+1])
        for i in range(p+1,n):
            atr[i]=atr[i-1]*(1-1/p)+tr[i]/p
    return atr


# ============ UNIFIED BACKTEST ============
def backtest(cl, hi, lo, yr, mk, ma_fast, ma_slow, adx_arr,
             adx_min, delay, sl_pct, trail_act, trail_pct,
             leverage, margin_pct, init_bal,
             fee_rate=0.0004, skip_same_dir=True, strict_entry=False,
             warmup=300):
    n=len(cl);bal=init_bal;peak=init_bal;mdd=0.0
    trades=[];in_pos=False;p_dir=0;p_entry=0.0;p_size=0.0
    p_sl=0.0;p_high=0.0;p_low=0.0;t_active=False;t_sl=0.0
    cur_m=-1;m_start=bal;m_locked=False
    inv_lev=1.0/leverage;pend=0;pcnt=0

    valid=~(np.isnan(ma_fast)|np.isnan(ma_slow))
    above=np.where(valid,ma_fast>ma_slow,False)

    for i in range(warmup,n):
        c=cl[i];h=hi[i];l=lo[i]
        m=mk[i]
        if m!=cur_m:cur_m=m;m_start=bal;m_locked=False
        if not m_locked and m_start>0:
            if (bal-m_start)/m_start<=-0.20:m_locked=True

        if in_pos:
            # FL check
            if p_dir==1 and l<=p_entry*(1-inv_lev):
                bal-=p_size/leverage;trades.append({'pnl':-p_size/leverage,'r':'FL','y':yr[i]});in_pos=False
                if bal>peak:peak=bal
                dd=(peak-bal)/peak if peak>0 else 0
                if dd>mdd:mdd=dd;continue
            if p_dir==-1 and h>=p_entry*(1+inv_lev):
                bal-=p_size/leverage;trades.append({'pnl':-p_size/leverage,'r':'FL','y':yr[i]});in_pos=False
                if bal>peak:peak=bal
                dd=(peak-bal)/peak if peak>0 else 0
                if dd>mdd:mdd=dd;continue

            # SL on candle high/low
            if p_dir==1 and l<=p_sl:
                pnl=p_size*(p_sl-p_entry)/p_entry-p_size*fee_rate;bal+=pnl
                trades.append({'pnl':pnl,'r':'SL','y':yr[i]});in_pos=False
            elif p_dir==-1 and h>=p_sl:
                pnl=p_size*(p_entry-p_sl)/p_entry-p_size*fee_rate;bal+=pnl
                trades.append({'pnl':pnl,'r':'SL','y':yr[i]});in_pos=False
            elif in_pos:
                # Trail
                if p_dir==1:
                    if h>p_high:p_high=h
                    if (p_high-p_entry)/p_entry*leverage>=trail_act*leverage:
                        t_active=True;ns=p_high*(1-trail_pct)
                        if ns>t_sl:t_sl=ns
                    if t_active and c<=t_sl:
                        pnl=p_size*(c-p_entry)/p_entry-p_size*fee_rate;bal+=pnl
                        trades.append({'pnl':pnl,'r':'TSL','y':yr[i]});in_pos=False
                else:
                    if l<p_low:p_low=l
                    if (p_entry-p_low)/p_entry*leverage>=trail_act*leverage:
                        t_active=True;ns=p_low*(1+trail_pct)
                        if t_sl==0 or ns<t_sl:t_sl=ns
                    if t_active and c>=t_sl:
                        pnl=p_size*(p_entry-c)/p_entry-p_size*fee_rate;bal+=pnl
                        trades.append({'pnl':pnl,'r':'TSL','y':yr[i]});in_pos=False

        # Cross detection
        sig=0
        if i>=warmup+1 and valid[i] and valid[i-1]:
            adx_ok=adx_arr[i]>=adx_min if not np.isnan(adx_arr[i]) else False
            if above[i] and not above[i-1] and adx_ok:
                if strict_entry:
                    if i>=2 and above[i] and valid[i-1] and ma_fast[i-1]<=ma_slow[i-1]:
                        sig=1
                else:
                    sig=1
            elif not above[i] and above[i-1] and adx_ok:
                if strict_entry:
                    if i>=2 and not above[i] and valid[i-1] and ma_fast[i-1]>=ma_slow[i-1]:
                        sig=-1
                else:
                    sig=-1

        # Delay
        if sig!=0 and delay>0:
            pend=sig;pcnt=delay;sig=0
        if pcnt>0:
            pcnt-=1
            if pcnt==0:
                if pend==1 and valid[i] and ma_fast[i]>ma_slow[i]:sig=pend
                elif pend==-1 and valid[i] and ma_fast[i]<ma_slow[i]:sig=pend
                pend=0

        if sig!=0:
            if in_pos:
                if p_dir!=sig:
                    pnl=p_size*(c-p_entry)/p_entry*p_dir-p_size*fee_rate;bal+=pnl
                    trades.append({'pnl':pnl,'r':'REV','y':yr[i]});in_pos=False
                elif not skip_same_dir:
                    pnl=p_size*(c-p_entry)/p_entry*p_dir-p_size*fee_rate;bal+=pnl
                    trades.append({'pnl':pnl,'r':'REV','y':yr[i]});in_pos=False

            if not in_pos and not m_locked and bal>10:
                mg=bal*margin_pct;sz=mg*leverage;bal-=sz*fee_rate
                p_dir=sig;p_entry=c;p_size=sz
                p_sl=c*(1-sl_pct) if sig==1 else c*(1+sl_pct)
                p_high=c;p_low=c;t_active=False;t_sl=0;in_pos=True

        if bal>peak:peak=bal
        if peak>0:
            dd=(peak-bal)/peak
            if dd>mdd:mdd=dd

    if in_pos:
        c=cl[-1];pnl=p_size*(c-p_entry)/p_entry*p_dir-p_size*fee_rate;bal+=pnl
        trades.append({'pnl':pnl,'r':'END','y':yr[-1]})

    tc=len(trades)
    if tc==0:
        return {'bal':bal,'ret':0,'tr':0,'pf':0,'mdd':0,'wr':0,'sl':0,'tsl':0,'rev':0,'fl':0}
    pnls=np.array([t['pnl'] for t in trades])
    w=pnls>0;lo2=pnls<=0
    gp=pnls[w].sum() if w.any() else 0;gl=abs(pnls[lo2].sum()) if lo2.any() else 0.001
    return {
        'bal':round(bal,0),'ret':round((bal-init_bal)/init_bal*100,1),
        'tr':tc,'pf':round(min(gp/gl,999.99),2),'mdd':round(mdd*100,1),
        'wr':round(w.sum()/tc*100,1),
        'sl':sum(1 for t in trades if t['r']=='SL'),
        'tsl':sum(1 for t in trades if t['r']=='TSL'),
        'rev':sum(1 for t in trades if t['r']=='REV'),
        'fl':sum(1 for t in trades if t['r']=='FL'),
    }


# ============ DATA ============
def load():
    print('Loading...')
    df=pd.read_csv(DATA_PATH,parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    data={}
    for name,mins in [('5min',5),('10min',10),('15min',15),('30min',30),('1h',60)]:
        d=df.set_index('timestamp')
        rule='%dmin'%mins if mins<60 else '%dh'%(mins//60)
        if mins==5: r=df.copy()
        else: r=d.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
        ts=pd.to_datetime(r['timestamp'].values)
        data[name]={
            'cl':r['close'].values.astype(float),'hi':r['high'].values.astype(float),
            'lo':r['low'].values.astype(float),'vo':r['volume'].values.astype(float),
            'yr':ts.year.values.astype(np.int32),
            'mk':(ts.year.values*100+ts.month.values).astype(np.int32),'n':len(r)}
        print('  %s: %d bars'%(name,data[name]['n']))
    return data


# ============ STRATEGIES ============
STRATEGIES = [
    {'id':'S01','name':'WMA3/EMA200 30m D5 T3/2 M50','tf':'30min','mf':'WMA','fp':3,'ms':'EMA','sp':200,'adx':35,'adxp':20,'d':5,'sl':0.08,'ta':0.03,'tp':0.02,'l':10,'m':0.50},
    {'id':'S02','name':'WMA3/EMA200 30m D0 T6/3 M35','tf':'30min','mf':'WMA','fp':3,'ms':'EMA','sp':200,'adx':35,'adxp':20,'d':0,'sl':0.07,'ta':0.06,'tp':0.03,'l':10,'m':0.35},
    {'id':'S03','name':'SMA14/EMA200 30m D3 T6/5 M35','tf':'30min','mf':'SMA','fp':14,'ms':'EMA','sp':200,'adx':35,'adxp':20,'d':3,'sl':0.07,'ta':0.06,'tp':0.05,'l':10,'m':0.35},
    {'id':'S04','name':'EMA2/EMA200 30m D0 T6/3 M35','tf':'30min','mf':'EMA','fp':2,'ms':'EMA','sp':200,'adx':35,'adxp':20,'d':0,'sl':0.07,'ta':0.06,'tp':0.03,'l':10,'m':0.35},
    {'id':'S05','name':'EMA3/EMA200 30m D0 T6/5 M35','tf':'30min','mf':'EMA','fp':3,'ms':'EMA','sp':200,'adx':35,'adxp':20,'d':0,'sl':0.07,'ta':0.06,'tp':0.05,'l':10,'m':0.35},
    {'id':'S06','name':'EMA3/EMA200 30m D0 T7/3 M40','tf':'30min','mf':'EMA','fp':3,'ms':'EMA','sp':200,'adx':35,'adxp':20,'d':0,'sl':0.07,'ta':0.07,'tp':0.03,'l':10,'m':0.40},
    {'id':'S07','name':'EMA100/EMA600 30m D0 SL3 T12/9','tf':'30min','mf':'EMA','fp':100,'ms':'EMA','sp':600,'adx':30,'adxp':20,'d':0,'sl':0.03,'ta':0.12,'tp':0.09,'l':10,'m':0.35},
    {'id':'S08','name':'EMA75/SMA750 30m D0 SL3 T12/9','tf':'30min','mf':'EMA','fp':75,'ms':'SMA','sp':750,'adx':30,'adxp':20,'d':0,'sl':0.03,'ta':0.12,'tp':0.09,'l':10,'m':0.35},
    {'id':'S09','name':'EMA5/SMA50 30m D0 SL12 T5/3 M40','tf':'30min','mf':'EMA','fp':5,'ms':'SMA','sp':50,'adx':35,'adxp':14,'d':0,'sl':0.12,'ta':0.05,'tp':0.03,'l':10,'m':0.40},
    {'id':'S10','name':'HMA14/VWMA300 15m D3 T10/3 M50','tf':'15min','mf':'HMA','fp':14,'ms':'VWMA','sp':300,'adx':35,'adxp':20,'d':3,'sl':0.07,'ta':0.10,'tp':0.03,'l':10,'m':0.50},
    {'id':'S11','name':'VWMA2/VWMA300 15m D0 SL9 T5/5 M40','tf':'15min','mf':'VWMA','fp':2,'ms':'VWMA','sp':300,'adx':45,'adxp':20,'d':0,'sl':0.09,'ta':0.05,'tp':0.05,'l':10,'m':0.40},
    {'id':'S12','name':'EMA3/SMA300 30m D5 T4/3 M50','tf':'30min','mf':'EMA','fp':3,'ms':'SMA','sp':300,'adx':40,'adxp':20,'d':5,'sl':0.08,'ta':0.04,'tp':0.03,'l':10,'m':0.50},
    {'id':'S13','name':'EMA7/EMA250 15m D5 ADX45 M50','tf':'15min','mf':'EMA','fp':7,'ms':'EMA','sp':250,'adx':45,'adxp':20,'d':5,'sl':0.08,'ta':0.03,'tp':0.02,'l':10,'m':0.50},
    {'id':'S14','name':'EMA3/EMA100 1h D0 ADX30 M40','tf':'1h','mf':'EMA','fp':3,'ms':'EMA','sp':100,'adx':30,'adxp':20,'d':0,'sl':0.07,'ta':0.06,'tp':0.03,'l':10,'m':0.40},
    {'id':'S15','name':'WMA3/SMA300 30m D12 ADX45 M20','tf':'30min','mf':'WMA','fp':3,'ms':'SMA','sp':300,'adx':45,'adxp':20,'d':12,'sl':0.07,'ta':0.06,'tp':0.03,'l':10,'m':0.20},
    {'id':'S16','name':'HMA5/EMA200 30m D24 ADX45 M15','tf':'30min','mf':'HMA','fp':5,'ms':'EMA','sp':200,'adx':45,'adxp':20,'d':24,'sl':0.07,'ta':0.08,'tp':0.03,'l':10,'m':0.15},
    {'id':'S17','name':'EMA5/EMA300 10m D0 T8/3 M40','tf':'10min','mf':'EMA','fp':5,'ms':'EMA','sp':300,'adx':35,'adxp':20,'d':0,'sl':0.07,'ta':0.08,'tp':0.03,'l':10,'m':0.40},
    {'id':'S18','name':'EMA7/EMA100 5m D0 M25','tf':'5min','mf':'EMA','fp':7,'ms':'EMA','sp':100,'adx':35,'adxp':14,'d':0,'sl':0.07,'ta':0.06,'tp':0.03,'l':10,'m':0.25},
    {'id':'S19','name':'HMA21/EMA250 10m D0 M40','tf':'10min','mf':'HMA','fp':21,'ms':'EMA','sp':250,'adx':35,'adxp':20,'d':0,'sl':0.07,'ta':0.06,'tp':0.03,'l':10,'m':0.40},
    {'id':'S20','name':'WMA3/EMA200 30m D5 T3/2 L15 M25','tf':'30min','mf':'WMA','fp':3,'ms':'EMA','sp':200,'adx':35,'adxp':20,'d':5,'sl':0.05,'ta':0.03,'tp':0.02,'l':15,'m':0.25},
]

# ============ MAIN ============
def main():
    t0=time.time()
    print('='*70)
    print('  6-ENGINE CROSS VERIFICATION')
    print('  %d strategies x 6 engines = %d backtests' % (len(STRATEGIES), len(STRATEGIES)*6))
    print('='*70)

    data=load()
    init=5000.0

    # Engine definitions
    ENGINES = [
        {'id':'E1','name':'Wilder ADX','adx_fn':'wilder','fee':0.0004,'slip':0,'strict':False,'skip_same':True},
        {'id':'E2','name':'EWM ADX','adx_fn':'ewm','fee':0.0004,'slip':0,'strict':False,'skip_same':True},
        {'id':'E3','name':'Strict Entry','adx_fn':'wilder','fee':0.0004,'slip':0,'strict':True,'skip_same':True},
        {'id':'E4','name':'Slippage 0.05%','adx_fn':'wilder','fee':0.0004,'slip':0.0005,'strict':False,'skip_same':True},
        {'id':'E5','name':'High Fee 0.06%','adx_fn':'wilder','fee':0.0006,'slip':0,'strict':False,'skip_same':True},
        {'id':'E6','name':'Same-Dir Reentry','adx_fn':'wilder','fee':0.0004,'slip':0,'strict':False,'skip_same':False},
    ]

    all_results = []
    total = len(STRATEGIES) * len(ENGINES)
    cnt = 0

    for s in STRATEGIES:
        tf = s['tf']; d = data[tf]
        cl=d['cl'];hi=d['hi'];lo=d['lo'];vo=d['vo'];yr=d['yr'];mk=d['mk']
        mf = calc_ma(cl, vo, s['mf'], s['fp'])
        ms = calc_ma(cl, vo, s['ms'], s['sp'])
        wu = max(s['sp']+50, 300)

        row = {'id':s['id'],'name':s['name']}

        for eng in ENGINES:
            # ADX
            if eng['adx_fn']=='wilder':
                adx_arr = adx_wilder(hi, lo, cl, s['adxp'])
            else:
                adx_arr = adx_ewm(hi, lo, cl, s['adxp'])

            # Slippage
            if eng['slip'] > 0:
                np.random.seed(42)
                slip = 1 + np.random.uniform(-eng['slip'], eng['slip'], len(cl))
                cl_s = cl * slip; hi_s = hi * slip; lo_s = lo * slip
            else:
                cl_s = cl; hi_s = hi; lo_s = lo

            r = backtest(cl_s, hi_s, lo_s, yr, mk, mf, ms, adx_arr,
                        s['adx'], s['d'], s['sl'], s['ta'], s['tp'],
                        s['l'], s['m'], init,
                        fee_rate=eng['fee'], skip_same_dir=eng['skip_same'],
                        strict_entry=eng['strict'], warmup=wu)

            row[eng['id']+'_ret'] = r['ret']
            row[eng['id']+'_pf'] = r['pf']
            row[eng['id']+'_mdd'] = r['mdd']
            row[eng['id']+'_tr'] = r['tr']
            row[eng['id']+'_sl'] = r['sl']
            row[eng['id']+'_fl'] = r['fl']

            cnt += 1
            if cnt % 20 == 0:
                print('  %d/%d (%.0fs)' % (cnt, total, time.time()-t0))

        # Compute consistency metrics
        rets = [row.get(e['id']+'_ret',0) for e in ENGINES]
        pfs = [row.get(e['id']+'_pf',0) for e in ENGINES]
        mdds = [row.get(e['id']+'_mdd',0) for e in ENGINES]
        trs = [row.get(e['id']+'_tr',0) for e in ENGINES]

        row['avg_ret'] = round(np.mean(rets),1)
        row['min_ret'] = round(np.min(rets),1)
        row['max_ret'] = round(np.max(rets),1)
        row['std_ret'] = round(np.std(rets),1)
        row['avg_pf'] = round(np.mean(pfs),2)
        row['min_pf'] = round(np.min(pfs),2)
        row['avg_mdd'] = round(np.mean(mdds),1)
        row['max_mdd'] = round(np.max(mdds),1)
        row['avg_tr'] = round(np.mean(trs),0)
        row['consistency'] = round(np.min(rets) / max(np.max(rets),1) * 100, 1) if np.max(rets) > 0 else 0

        all_results.append(row)

    # ===== PRINT RESULTS =====
    print('\n' + '='*70)
    print('  CROSS VERIFICATION RESULTS')
    print('='*70)

    # Header
    print('\n%4s %-38s' % ('ID','Strategy'), end='')
    for e in ENGINES:
        print(' %12s' % e['name'], end='')
    print(' %8s %8s %8s %6s' % ('AvgRet','MinRet','AvgPF','Cons%'))
    print('-'*160)

    for row in all_results:
        print('%4s %-38s' % (row['id'], row['name'][:38]), end='')
        for e in ENGINES:
            ret = row.get(e['id']+'_ret',0)
            tr = row.get(e['id']+'_tr',0)
            print(' %+7.0f%%/%2d' % (ret, tr), end='')
        print(' %+7.0f%% %+7.0f%% %7.1f %5.1f%%' % (row['avg_ret'], row['min_ret'], row['avg_pf'], row['consistency']))

    # ===== RANKINGS =====
    # Return BEST 10 (by avg return, min_ret > 0)
    profitable = [r for r in all_results if r['min_ret'] > 0]
    profitable.sort(key=lambda x: x['avg_ret'], reverse=True)

    print('\n' + '='*70)
    print('  RETURN BEST 10 (all 6 engines profitable)')
    print('='*70)
    print('%4s %-38s %10s %10s %8s %8s %8s %6s' % ('R','Strategy','AvgRet','MinRet','AvgPF','AvgMDD','AvgTr','Cons%'))
    print('-'*100)
    for i, r in enumerate(profitable[:10]):
        print('%4d %-38s %+9.1f%% %+9.1f%% %7.2f %7.1f%% %7.0f %5.1f%%' % (
            i+1, r['name'][:38], r['avg_ret'], r['min_ret'], r['avg_pf'], r['avg_mdd'], r['avg_tr'], r['consistency']))

    # Stability BEST 10 (by lowest max_mdd, all profitable)
    stable = [r for r in all_results if r['min_ret'] > 0]
    stable.sort(key=lambda x: x['max_mdd'])

    print('\n' + '='*70)
    print('  STABILITY BEST 10 (lowest MDD across all engines)')
    print('='*70)
    print('%4s %-38s %10s %8s %8s %8s %6s' % ('R','Strategy','AvgRet','AvgPF','MaxMDD','AvgTr','Cons%'))
    print('-'*90)
    for i, r in enumerate(stable[:10]):
        print('%4d %-38s %+9.1f%% %7.2f %7.1f%% %7.0f %5.1f%%' % (
            i+1, r['name'][:38], r['avg_ret'], r['avg_pf'], r['max_mdd'], r['avg_tr'], r['consistency']))

    # Discard 10 (worst: negative in any engine, or MDD>50%, or PF<1.5)
    discard = [r for r in all_results if r['min_ret'] <= 0 or r['max_mdd'] > 50 or r['min_pf'] < 1.0]
    discard.sort(key=lambda x: x['min_ret'])

    print('\n' + '='*70)
    print('  DISCARD BEST 10 (loss in any engine, MDD>50%%, or PF<1)')
    print('='*70)
    print('%4s %-38s %10s %10s %8s %8s %8s %s' % ('R','Strategy','AvgRet','MinRet','MinPF','MaxMDD','AvgTr','Reason'))
    print('-'*110)
    for i, r in enumerate(discard[:10]):
        reasons = []
        if r['min_ret'] <= 0: reasons.append('LOSS')
        if r['max_mdd'] > 50: reasons.append('MDD>50')
        if r['min_pf'] < 1.0: reasons.append('PF<1')
        print('%4d %-38s %+9.1f%% %+9.1f%% %7.2f %7.1f%% %7.0f %s' % (
            i+1, r['name'][:38], r['avg_ret'], r['min_ret'], r['min_pf'], r['max_mdd'], r['avg_tr'], '+'.join(reasons)))

    # Save
    with open(os.path.join(OUTPUT_DIR, 'cross_verify.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print('\n' + '='*70)
    print('  COMPLETE: %d backtests in %.1f min' % (cnt, (time.time()-t0)/60))
    print('='*70)

if __name__=='__main__':
    main()
