"""
멀티엔진 기획서 1계정 동시운영 검증
- 바이낸스 제약: 동일 심볼 1포지션만 (헤지모드 제외)
- 2개 엔진이 동시에 신호 발생시 처리 규칙 검증
- 개별 운영 vs 동시 운영 성능 비교
"""
import sys, time, json, os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = r"D:\filesystem\futures\btc_V1\test1\btc_usdt_5m_merged.csv"
OUTPUT_DIR = r"D:\filesystem\futures\btc_V1\test1\multi_verify"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ INDICATORS ============
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


# ============ DATA ============
def load_all():
    print('Loading data...')
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    data = {}
    for name, mins in [('15min',15),('30min',30)]:
        d2 = df.set_index('timestamp')
        rule = '%dmin' % mins
        r = d2.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
        ts = pd.to_datetime(r['timestamp'].values)
        cl = r['close'].values.astype(float)
        hi = r['high'].values.astype(float)
        lo = r['low'].values.astype(float)
        data[name] = {
            'ts': r['timestamp'].values, 'cl': cl, 'hi': hi, 'lo': lo,
            'vo': r['volume'].values.astype(float),
            'yr': ts.year.values.astype(np.int32),
            'mk': (ts.year.values*100+ts.month.values).astype(np.int32),
            'n': len(cl),
        }
        print('  %s: %d bars' % (name, len(cl)))
    return data


def generate_signals(d, ma_fast_type, ma_fast_p, ma_slow_type, ma_slow_p,
                     adx_min, rsi_min, rsi_max, delay, adx_p=20):
    """Generate signal array: +1=LONG, -1=SHORT, 0=none"""
    cl = d['cl']; hi = d['hi']; lo = d['lo']; vo = d['vo']
    n = d['n']

    # Compute MAs
    if ma_fast_type == 'WMA': mf = wma(cl, ma_fast_p)
    elif ma_fast_type == 'EMA': mf = ema(cl, ma_fast_p)
    elif ma_fast_type == 'SMA': mf = sma(cl, ma_fast_p)
    else: mf = ema(cl, ma_fast_p)

    if ma_slow_type == 'WMA': ms = wma(cl, ma_slow_p)
    elif ma_slow_type == 'EMA': ms = ema(cl, ma_slow_p)
    elif ma_slow_type == 'SMA': ms = sma(cl, ma_slow_p)
    else: ms = ema(cl, ma_slow_p)

    adx_v = calc_adx(hi, lo, cl, adx_p)
    rsi_v = calc_rsi(cl, 14)

    warmup = max(ma_slow_p + 50, 250)
    valid = ~(np.isnan(mf) | np.isnan(ms))
    above = mf > ms

    # Raw crosses
    raw_sig = np.zeros(n, dtype=int)
    for i in range(warmup, n):
        if valid[i] and valid[i-1]:
            adx_ok = adx_v[i] >= adx_min if not np.isnan(adx_v[i]) else False
            rsi_ok = rsi_min <= rsi_v[i] <= rsi_max if not np.isnan(rsi_v[i]) else False
            if above[i] and not above[i-1] and adx_ok and rsi_ok:
                raw_sig[i] = 1
            elif not above[i] and above[i-1] and adx_ok and rsi_ok:
                raw_sig[i] = -1

    # Apply delay
    signals = np.zeros(n, dtype=int)
    pend = 0; pcnt = 0
    for i in range(n):
        if raw_sig[i] != 0:
            if delay > 0:
                pend = raw_sig[i]; pcnt = delay
            else:
                signals[i] = raw_sig[i]
        if pcnt > 0:
            pcnt -= 1
            if pcnt == 0:
                # Verify cross still valid
                if pend == 1 and valid[i] and mf[i] > ms[i]:
                    signals[i] = pend
                elif pend == -1 and valid[i] and mf[i] < ms[i]:
                    signals[i] = pend
                pend = 0

    return signals


def single_engine_bt(d, signals, sl_pct, trail_act, trail_pct,
                     leverage, margin_pct, init_bal=5000.0):
    """Standard single-engine backtest"""
    cl=d['cl'];hi=d['hi'];lo=d['lo'];yr=d['yr'];mk=d['mk']
    n=d['n']; bal=init_bal; peak=init_bal; mdd=0.0
    trades=[]; in_pos=False; p_dir=0; p_entry=0.0; p_size=0.0
    p_sl=0.0; p_high=0.0; p_low=0.0; t_active=False; t_sl=0.0
    cur_m=-1;m_start=bal;m_locked=False
    fee=0.0004; inv_lev=1.0/leverage

    for i in range(1,n):
        c=cl[i];h=hi[i];l=lo[i]
        m=mk[i]
        if m!=cur_m: cur_m=m;m_start=bal;m_locked=False
        if not m_locked and m_start>0:
            if (bal-m_start)/m_start<=-0.20: m_locked=True

        if in_pos:
            # FL
            if p_dir==1 and l<=p_entry*(1-inv_lev):
                bal-=p_size*margin_pct;trades.append({'pnl':-p_size*margin_pct,'reason':'FL','yr':yr[i],'bal':bal});in_pos=False;continue
            if p_dir==-1 and h>=p_entry*(1+inv_lev):
                bal-=p_size*margin_pct;trades.append({'pnl':-p_size*margin_pct,'reason':'FL','yr':yr[i],'bal':bal});in_pos=False;continue
            # SL
            if p_dir==1 and l<=p_sl:
                pnl=p_size*(p_sl-p_entry)/p_entry-p_size*fee
                bal+=pnl;trades.append({'pnl':pnl,'reason':'SL','yr':yr[i],'bal':bal});in_pos=False
            elif p_dir==-1 and h>=p_sl:
                pnl=p_size*(p_entry-p_sl)/p_entry-p_size*fee
                bal+=pnl;trades.append({'pnl':pnl,'reason':'SL','yr':yr[i],'bal':bal});in_pos=False
            elif in_pos:
                if p_dir==1:
                    if h>p_high:p_high=h
                    if (p_high-p_entry)/p_entry*leverage>=trail_act*leverage:
                        t_active=True;ns=p_high*(1-trail_pct)
                        if ns>t_sl:t_sl=ns
                    if t_active and c<=t_sl:
                        pnl=p_size*(c-p_entry)/p_entry-p_size*fee
                        bal+=pnl;trades.append({'pnl':pnl,'reason':'TSL','yr':yr[i],'bal':bal});in_pos=False
                else:
                    if l<p_low:p_low=l
                    if (p_entry-p_low)/p_entry*leverage>=trail_act*leverage:
                        t_active=True;ns=p_low*(1+trail_pct)
                        if t_sl==0 or ns<t_sl:t_sl=ns
                    if t_active and c>=t_sl:
                        pnl=p_size*(p_entry-c)/p_entry-p_size*fee
                        bal+=pnl;trades.append({'pnl':pnl,'reason':'TSL','yr':yr[i],'bal':bal});in_pos=False

        sig = signals[i]
        if sig!=0:
            if in_pos and p_dir!=sig:
                pnl=p_size*(c-p_entry)/p_entry*p_dir-p_size*fee
                bal+=pnl;trades.append({'pnl':pnl,'reason':'REV','yr':yr[i],'bal':bal});in_pos=False
            elif in_pos and p_dir==sig:
                pass  # Same direction, skip

            if not in_pos and not m_locked and bal>10:
                mg=bal*margin_pct;sz=mg*leverage
                bal-=sz*fee
                p_dir=sig;p_entry=c;p_size=sz
                p_sl=c*(1-sl_pct) if sig==1 else c*(1+sl_pct)
                p_high=c;p_low=c;t_active=False;t_sl=0
                in_pos=True

        if bal>peak:peak=bal
        if peak>0:
            dd=(peak-bal)/peak
            if dd>mdd:mdd=dd

    if in_pos:
        c=cl[-1];pnl=p_size*(c-p_entry)/p_entry*p_dir-p_size*fee
        bal+=pnl;trades.append({'pnl':pnl,'reason':'END','yr':yr[-1],'bal':bal})

    return compile_result(trades, bal, init_bal, mdd)


def dual_engine_bt(d_a, signals_a, d_b, signals_b,
                   sl_pct, trail_act, trail_pct,
                   leverage, margin_pct, init_bal=5000.0,
                   priority='A'):
    """
    2엔진 1계정 시뮬레이터
    - 동일 심볼 1포지션 제약
    - 신호 충돌 처리: priority 엔진 우선
    - 동일 방향: 기존 포지션 유지 (무시)
    - 반대 방향: 청산 후 우선 엔진 방향 진입
    """
    # Merge signals onto 30min timeline (common base)
    # If Engine A is 15min, map to 30min indices
    ts_a = d_a['ts']; ts_b = d_b['ts']
    cl_b = d_b['cl']; hi_b = d_b['hi']; lo_b = d_b['lo']
    yr_b = d_b['yr']; mk_b = d_b['mk']
    n = d_b['n']

    # Map 15min signals to 30min bar indices
    # Each 30min bar contains 2x 15min bars. Take any signal within.
    mapped_a = np.zeros(n, dtype=int)
    if d_a['n'] != d_b['n']:  # Different TFs
        a_idx = 0
        for i in range(n):
            t_start = ts_b[i]
            # Find all 15min bars within this 30min window
            while a_idx < len(ts_a) and ts_a[a_idx] < t_start:
                a_idx += 1
            # Check signals in this 30min window
            for j in range(a_idx, min(a_idx+2, len(ts_a))):
                if ts_a[j] >= t_start and (i+1 >= n or ts_a[j] < ts_b[min(i+1,n-1)]):
                    if signals_a[j] != 0:
                        mapped_a[i] = signals_a[j]
    else:
        mapped_a = signals_a.copy()

    sig_b = signals_b

    # Now simulate on 30min bars with merged signals
    bal = init_bal; peak = init_bal; mdd = 0.0
    trades = []; in_pos = False; p_dir = 0; p_entry = 0.0; p_size = 0.0
    p_sl = 0.0; p_high = 0.0; p_low = 0.0; t_active = False; t_sl = 0.0
    cur_m = -1; m_start = bal; m_locked = False
    fee = 0.0004; inv_lev = 1.0/leverage
    source_count = {'A':0, 'B':0, 'conflict':0, 'same_dir':0}

    for i in range(1, n):
        c=cl_b[i];h=hi_b[i];l=lo_b[i]
        m=mk_b[i]
        if m!=cur_m: cur_m=m;m_start=bal;m_locked=False
        if not m_locked and m_start>0:
            if (bal-m_start)/m_start<=-0.20: m_locked=True

        # Position management (same as single)
        if in_pos:
            if p_dir==1 and l<=p_entry*(1-inv_lev):
                bal-=p_size*(bal*margin_pct/p_size if p_size>0 else 0)
                trades.append({'pnl':-(bal*margin_pct),'reason':'FL','yr':yr_b[i],'bal':bal,'src':'POS'});in_pos=False;continue
            if p_dir==-1 and h>=p_entry*(1+inv_lev):
                trades.append({'pnl':0,'reason':'FL','yr':yr_b[i],'bal':bal,'src':'POS'});in_pos=False;continue

            if p_dir==1 and l<=p_sl:
                pnl=p_size*(p_sl-p_entry)/p_entry-p_size*fee
                bal+=pnl;trades.append({'pnl':pnl,'reason':'SL','yr':yr_b[i],'bal':bal,'src':'POS'});in_pos=False
            elif p_dir==-1 and h>=p_sl:
                pnl=p_size*(p_entry-p_sl)/p_entry-p_size*fee
                bal+=pnl;trades.append({'pnl':pnl,'reason':'SL','yr':yr_b[i],'bal':bal,'src':'POS'});in_pos=False
            elif in_pos:
                if p_dir==1:
                    if h>p_high:p_high=h
                    if (p_high-p_entry)/p_entry*leverage>=trail_act*leverage:
                        t_active=True;ns=p_high*(1-trail_pct)
                        if ns>t_sl:t_sl=ns
                    if t_active and c<=t_sl:
                        pnl=p_size*(c-p_entry)/p_entry-p_size*fee
                        bal+=pnl;trades.append({'pnl':pnl,'reason':'TSL','yr':yr_b[i],'bal':bal,'src':'POS'});in_pos=False
                else:
                    if l<p_low:p_low=l
                    if (p_entry-p_low)/p_entry*leverage>=trail_act*leverage:
                        t_active=True;ns=p_low*(1+trail_pct)
                        if t_sl==0 or ns<t_sl:t_sl=ns
                    if t_active and c>=t_sl:
                        pnl=p_size*(p_entry-c)/p_entry-p_size*fee
                        bal+=pnl;trades.append({'pnl':pnl,'reason':'TSL','yr':yr_b[i],'bal':bal,'src':'POS'});in_pos=False

        # Signal merging: both engines on same bar
        sa = mapped_a[i]; sb = sig_b[i]
        final_sig = 0; src = ''

        if sa != 0 and sb != 0:
            if sa == sb:
                final_sig = sa; src = 'BOTH'
                source_count['same_dir'] += 1
            else:
                # Conflict! Priority engine wins
                if priority == 'A':
                    final_sig = sa; src = 'A(conflict)'
                else:
                    final_sig = sb; src = 'B(conflict)'
                source_count['conflict'] += 1
        elif sa != 0:
            final_sig = sa; src = 'A'
            source_count['A'] += 1
        elif sb != 0:
            final_sig = sb; src = 'B'
            source_count['B'] += 1

        if final_sig != 0:
            if in_pos and p_dir != final_sig:
                pnl = p_size*(c-p_entry)/p_entry*p_dir - p_size*fee
                bal += pnl
                trades.append({'pnl':pnl,'reason':'REV','yr':yr_b[i],'bal':bal,'src':src})
                in_pos = False
            elif in_pos and p_dir == final_sig:
                pass  # Already in same direction

            if not in_pos and not m_locked and bal > 10:
                mg = bal*margin_pct; sz = mg*leverage
                bal -= sz*fee
                p_dir = final_sig; p_entry = c; p_size = sz
                p_sl = c*(1-sl_pct) if final_sig==1 else c*(1+sl_pct)
                p_high = c; p_low = c; t_active = False; t_sl = 0
                in_pos = True

        if bal>peak: peak=bal
        if peak>0:
            dd=(peak-bal)/peak
            if dd>mdd: mdd=dd

    if in_pos:
        c=cl_b[-1];pnl=p_size*(c-p_entry)/p_entry*p_dir-p_size*fee
        bal+=pnl;trades.append({'pnl':pnl,'reason':'END','yr':yr_b[-1],'bal':bal,'src':'END'})

    result = compile_result(trades, bal, init_bal, mdd)
    result['source_count'] = source_count
    return result


def compile_result(trades, bal, init, mdd):
    tc=len(trades)
    if tc==0:
        return {'bal':bal,'ret':0,'trades':0,'pf':0,'mdd':0,'wr':0,'sl':0,'tsl':0,'rev':0,'fl':0}
    pnls=np.array([t['pnl'] for t in trades])
    w=pnls>0;l=pnls<=0
    gp=pnls[w].sum() if w.any() else 0
    gl=abs(pnls[l].sum()) if l.any() else 0.001
    pf=min(gp/gl,999.99)
    return {
        'bal':round(bal,2),'ret':round((bal-init)/init*100,1),
        'trades':tc,'pf':round(pf,2),'mdd':round(mdd*100,1),
        'wr':round(w.sum()/tc*100,1),
        'sl':sum(1 for t in trades if t['reason']=='SL'),
        'tsl':sum(1 for t in trades if t['reason']=='TSL'),
        'rev':sum(1 for t in trades if t['reason']=='REV'),
        'fl':sum(1 for t in trades if t['reason']=='FL'),
    }


# ============ MAIN ============
def main():
    t0 = time.time()
    print('='*70)
    print('  MULTI-ENGINE 1-ACCOUNT VERIFICATION')
    print('  Binance constraint: 1 symbol = 1 position')
    print('='*70)

    data = load_all()
    d15 = data['15min']; d30 = data['30min']

    # ===== Define multi-engine documents =====
    MULTI_DOCS = {
        'v22.0': {
            'A': {'tf':'15min','mf_type':'EMA','mf_p':7,'ms_type':'WMA','ms_p':300,'adx_min':45,'rsi_min':30,'rsi_max':70,'delay':5},
            'B': {'tf':'30min','mf_type':'WMA','mf_p':3,'ms_type':'EMA','ms_p':200,'adx_min':35,'rsi_min':30,'rsi_max':70,'delay':5},
            'sl':0.08,'trail_act':0.03,'trail_pct':0.02,'lev':10,'margin':0.50,
        },
        'v22.1': {
            'A': {'tf':'15min','mf_type':'EMA','mf_p':7,'ms_type':'EMA','ms_p':250,'adx_min':45,'rsi_min':35,'rsi_max':75,'delay':5},
            'B': {'tf':'30min','mf_type':'WMA','mf_p':3,'ms_type':'EMA','ms_p':200,'adx_min':35,'rsi_min':35,'rsi_max':60,'delay':5},
            'sl':0.08,'trail_act':0.03,'trail_pct':0.02,'lev':10,'margin':0.50,
        },
        'v26.0': {
            'A': {'tf':'30min','mf_type':'EMA','mf_p':3,'ms_type':'SMA','ms_p':300,'adx_min':40,'rsi_min':30,'rsi_max':70,'delay':5},
            'B': {'tf':'30min','mf_type':'WMA','mf_p':3,'ms_type':'EMA','ms_p':200,'adx_min':35,'rsi_min':30,'rsi_max':70,'delay':5},
            'sl':0.08,'trail_act':0.03,'trail_pct':0.02,'lev':10,'margin':0.50,
        },
        'v15.6': {
            'A': {'tf':'15min','mf_type':'EMA','mf_p':3,'ms_type':'EMA','ms_p':150,'adx_min':45,'rsi_min':35,'rsi_max':70,'delay':0},
            'B': {'tf':'30min','mf_type':'EMA','mf_p':3,'ms_type':'EMA','ms_p':200,'adx_min':35,'rsi_min':35,'rsi_max':65,'delay':0},
            'sl':0.07,'trail_act':0.06,'trail_pct':0.03,'lev':10,'margin':0.35,
        },
    }

    for doc_name, cfg in MULTI_DOCS.items():
        print('\n' + '='*70)
        print('  %s: Engine A + Engine B' % doc_name)
        print('='*70)

        eng_a = cfg['A']; eng_b = cfg['B']
        d_a = data[eng_a['tf']]; d_b = data[eng_b['tf']]

        # Generate signals
        sig_a = generate_signals(d_a, eng_a['mf_type'],eng_a['mf_p'],eng_a['ms_type'],eng_a['ms_p'],
                                 eng_a['adx_min'],eng_a['rsi_min'],eng_a['rsi_max'],eng_a['delay'])
        sig_b = generate_signals(d_b, eng_b['mf_type'],eng_b['mf_p'],eng_b['ms_type'],eng_b['ms_p'],
                                 eng_b['adx_min'],eng_b['rsi_min'],eng_b['rsi_max'],eng_b['delay'])

        sig_a_count = np.sum(sig_a != 0)
        sig_b_count = np.sum(sig_b != 0)
        print('  Engine A signals: %d (%s %s(%d)/%s(%d) ADX>=%d D%d)' % (
            sig_a_count, eng_a['mf_type'],eng_a['mf_type'],eng_a['mf_p'],eng_a['ms_type'],eng_a['ms_p'],eng_a['adx_min'],eng_a['delay']))
        print('  Engine B signals: %d (%s %s(%d)/%s(%d) ADX>=%d D%d)' % (
            sig_b_count, eng_b['mf_type'],eng_b['mf_type'],eng_b['mf_p'],eng_b['ms_type'],eng_b['ms_p'],eng_b['adx_min'],eng_b['delay']))

        sl = cfg['sl']; ta = cfg['trail_act']; tp = cfg['trail_pct']
        lev = cfg['lev']; mpct = cfg['margin']

        # 1. Engine A alone (full capital)
        r_a = single_engine_bt(d_a, sig_a, sl, ta, tp, lev, mpct, 5000.0)
        print('\n  [A alone] $%.0f (%+.1f%%) Tr=%d PF=%.2f MDD=%.1f%% SL=%d FL=%d' % (
            r_a['bal'],r_a['ret'],r_a['trades'],r_a['pf'],r_a['mdd'],r_a['sl'],r_a['fl']))

        # 2. Engine B alone (full capital)
        r_b = single_engine_bt(d_b, sig_b, sl, ta, tp, lev, mpct, 5000.0)
        print('  [B alone] $%.0f (%+.1f%%) Tr=%d PF=%.2f MDD=%.1f%% SL=%d FL=%d' % (
            r_b['bal'],r_b['ret'],r_b['trades'],r_b['pf'],r_b['mdd'],r_b['sl'],r_b['fl']))

        # 3. A+B split capital (70/30)
        r_a70 = single_engine_bt(d_a, sig_a, sl, ta, tp, lev, mpct, 3500.0)
        r_b30 = single_engine_bt(d_b, sig_b, sl, ta, tp, lev, mpct, 1500.0)
        split_bal = r_a70['bal'] + r_b30['bal']
        split_ret = (split_bal - 5000) / 5000 * 100
        print('  [A70%%+B30%% split] $%.0f (%+.1f%%) = A:$%.0f + B:$%.0f' % (
            split_bal, split_ret, r_a70['bal'], r_b30['bal']))

        # 4. A+B merged on 1 account (A priority)
        r_merged_a = dual_engine_bt(d_a, sig_a, d_b, sig_b, sl, ta, tp, lev, mpct, 5000.0, priority='A')
        print('  [A+B merged, A우선] $%.0f (%+.1f%%) Tr=%d PF=%.2f MDD=%.1f%% SL=%d FL=%d' % (
            r_merged_a['bal'],r_merged_a['ret'],r_merged_a['trades'],r_merged_a['pf'],r_merged_a['mdd'],r_merged_a['sl'],r_merged_a['fl']))
        sc = r_merged_a.get('source_count',{})
        print('    Signal sources: A=%d B=%d Both=%d Conflict=%d' % (
            sc.get('A',0),sc.get('B',0),sc.get('same_dir',0),sc.get('conflict',0)))

        # 5. A+B merged on 1 account (B priority)
        r_merged_b = dual_engine_bt(d_a, sig_a, d_b, sig_b, sl, ta, tp, lev, mpct, 5000.0, priority='B')
        print('  [A+B merged, B우선] $%.0f (%+.1f%%) Tr=%d PF=%.2f MDD=%.1f%% SL=%d FL=%d' % (
            r_merged_b['bal'],r_merged_b['ret'],r_merged_b['trades'],r_merged_b['pf'],r_merged_b['mdd'],r_merged_b['sl'],r_merged_b['fl']))

        # Comparison
        print('\n  === COMPARISON ===')
        print('  %20s %10s %6s %7s %6s' % ('Mode','Return','Tr','PF','MDD%'))
        print('  '+'-'*55)
        print('  %20s %+9.1f%% %6d %6.2f %5.1f%%' % ('A alone (full)', r_a['ret'], r_a['trades'], r_a['pf'], r_a['mdd']))
        print('  %20s %+9.1f%% %6d %6.2f %5.1f%%' % ('B alone (full)', r_b['ret'], r_b['trades'], r_b['pf'], r_b['mdd']))
        print('  %20s %9s %6s %6s %5s' % ('A70+B30 (2account)', '%+.1f%%' % split_ret, '%d+%d' % (r_a70['trades'],r_b30['trades']), '-', '-'))
        print('  %20s %+9.1f%% %6d %6.2f %5.1f%%' % ('Merged A-priority', r_merged_a['ret'], r_merged_a['trades'], r_merged_a['pf'], r_merged_a['mdd']))
        print('  %20s %+9.1f%% %6d %6.2f %5.1f%%' % ('Merged B-priority', r_merged_b['ret'], r_merged_b['trades'], r_merged_b['pf'], r_merged_b['mdd']))

        # Verdict
        best_mode = 'A alone'
        best_ret = r_a['ret']
        if r_b['ret'] > best_ret: best_mode = 'B alone'; best_ret = r_b['ret']
        if split_ret > best_ret: best_mode = 'Split (2 accounts)'; best_ret = split_ret
        if r_merged_a['ret'] > best_ret: best_mode = 'Merged A-priority'; best_ret = r_merged_a['ret']
        if r_merged_b['ret'] > best_ret: best_mode = 'Merged B-priority'; best_ret = r_merged_b['ret']

        print('\n  >>> BEST: %s (%+.1f%%)' % (best_mode, best_ret))

        can_simultaneous = r_merged_a['ret'] > max(r_a['ret'], r_b['ret']) * 0.9
        print('  >>> 1계정 동시운영 %s' % ('가능 (병합이 개별보다 우수 또는 유사)' if can_simultaneous else '비추천 (개별 운영이 우수)'))

    print('\n' + '='*70)
    print('  COMPLETE in %.1f min' % ((time.time()-t0)/60))
    print('='*70)

if __name__ == '__main__':
    main()
