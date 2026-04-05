# -*- coding: utf-8 -*-
"""5 strategies x 10x/20x with fixed $1,000 margin per trade"""
import sys, os, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def load_5m():
    parts = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.startswith('btc_usdt_5m') and f.endswith('.csv'):
            parts.append(pd.read_csv(os.path.join(DATA_DIR, f), parse_dates=['timestamp']))
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    df = df.rename(columns={'timestamp': 'time'})
    for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
    return df

def resample(df, mins):
    return df.set_index('time').resample(f'{mins}min').agg(
        {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()

def ema(s,p): return s.ewm(span=p, adjust=False).mean()
def wma(s,p):
    w = np.arange(1,p+1,dtype=float)
    return s.rolling(p).apply(lambda x: np.dot(x,w)/w.sum(), raw=True)
def hma(s,p):
    h=max(int(p/2),1); sq=max(int(np.sqrt(p)),1)
    return wma(2*wma(s,h)-wma(s,p), sq)
def calc_ma(s,t,p):
    if t=='hma': return hma(s,p)
    return ema(s,p)
def adx_w(h,l,c,p=14):
    tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    up=h-h.shift(1); dn=l.shift(1)-l
    pdm=np.where((up>dn)&(up>0),up,0.0); mdm=np.where((dn>up)&(dn>0),dn,0.0)
    atr=pd.Series(tr,index=c.index).ewm(alpha=1/p,min_periods=p,adjust=False).mean()
    pdi=100*pd.Series(pdm,index=c.index).ewm(alpha=1/p,min_periods=p,adjust=False).mean()/atr
    mdi=100*pd.Series(mdm,index=c.index).ewm(alpha=1/p,min_periods=p,adjust=False).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/p,min_periods=p,adjust=False).mean()
def rsi_w(c,p=14):
    d=c.diff(); g=d.clip(lower=0); lo=(-d).clip(lower=0)
    ag=g.ewm(alpha=1/p,min_periods=p,adjust=False).mean()
    al=lo.ewm(alpha=1/p,min_periods=p,adjust=False).mean()
    return 100-100/(1+ag/al.replace(0,np.nan))


def backtest_fixed(df_tf, cfg, lev, fixed_margin=1000):
    n = len(df_tf)
    times=df_tf['time'].values; closes=df_tf['close'].values
    highs=df_tf['high'].values; lows=df_tf['low'].values
    mf=df_tf['ma_fast'].values; ms=df_tf['ma_slow'].values
    rsi_v=df_tf['rsi'].values; adx_v=df_tf['adx'].values
    fee=0.0004; balance=3000.0
    pos=0; entry_p=0.0; sz=0.0; peak_price=0.0; trail_on=False
    cur_m=''; m_start=balance; monthly={}; m_tr=0
    c_loss=0; pause_til=0; m_paused=False; peak_bal=balance
    pending=0; sig_idx=0
    liq_dist=1.0/lev
    delay=cfg.get('delay',0)
    ml=cfg.get('ml',0); cp=cfg.get('cp',0); cp_dur=cfg.get('cp_dur',0)

    for i in range(1, n):
        if np.isnan(mf[i]) or np.isnan(ms[i]) or np.isnan(rsi_v[i]) or np.isnan(adx_v[i]):
            continue
        t=str(times[i])[:19]; cc=closes[i]; hh=highs[i]; ll=lows[i]
        mk=t[:7]
        if mk != cur_m:
            if cur_m:
                monthly[cur_m] = {'pnl':balance-m_start,
                    'pct':(balance-m_start)/m_start*100 if m_start>0 else 0,
                    'bal':round(balance,2), 'tr':m_tr}
            cur_m=mk; m_start=balance; m_tr=0; m_paused=False

        if pos != 0:
            w = ll if pos==1 else hh
            # FL
            if pos*(w-entry_p)/entry_p <= -liq_dist:
                fl_p = entry_p*(1-pos*liq_dist)
                pp = pos*(fl_p-entry_p)/entry_p
                balance += sz*pp - sz*fee
                balance = max(balance, 0)
                m_tr += 1; pos = 0; peak_bal = max(peak_bal, balance)
                c_loss += 1
                if cp>0 and c_loss>=cp: pause_til = i+cp_dur
                continue
            # SL
            if pos*(w-entry_p)/entry_p <= -cfg['sl']:
                sl_p = entry_p*(1-pos*cfg['sl'])
                pp = pos*(sl_p-entry_p)/entry_p
                balance += sz*pp - sz*fee
                m_tr += 1; pos = 0; peak_bal = max(peak_bal, balance)
                c_loss += 1
                if cp>0 and c_loss>=cp: pause_til = i+cp_dur
                continue
            # Trail
            if pos==1: peak_price = max(peak_price, hh)
            else: peak_price = min(peak_price, ll)
            ppnl = pos*(peak_price-entry_p)/entry_p
            if ppnl >= cfg['ta']: trail_on = True
            if trail_on:
                if pos==1:
                    tsl = peak_price*(1-cfg['tw'])
                    if cc <= tsl:
                        pp = pos*(tsl-entry_p)/entry_p
                        balance += sz*pp - sz*fee
                        m_tr += 1; pos = 0; c_loss = 0
                        peak_bal = max(peak_bal, balance); continue
                else:
                    tsl = peak_price*(1+cfg['tw'])
                    if cc >= tsl:
                        pp = pos*(tsl-entry_p)/entry_p
                        balance += sz*pp - sz*fee
                        m_tr += 1; pos = 0; c_loss = 0
                        peak_bal = max(peak_bal, balance); continue
            # REV
            cu = mf[i]>ms[i] and mf[i-1]<=ms[i-1]
            cd = mf[i]<ms[i] and mf[i-1]>=ms[i-1]
            ao = adx_v[i]>=cfg['adx_min']; ro = cfg['rsi_min']<=rsi_v[i]<=cfg['rsi_max']
            nd = 0
            if pos==1 and cd and ao and ro: nd = -1
            elif pos==-1 and cu and ao and ro: nd = 1
            if nd != 0:
                pp = pos*(cc-entry_p)/entry_p
                balance += sz*pp - sz*fee; m_tr += 1
                if pp>0: c_loss = 0
                else:
                    c_loss += 1
                    if cp>0 and c_loss>=cp: pause_til = i+cp_dur
                pos = 0; peak_bal = max(peak_bal, balance)
                if balance > 100 and not m_paused and i >= pause_til:
                    mg = min(fixed_margin, balance*0.9)
                    pos = nd; entry_p = cc; sz = mg*lev
                    balance -= sz*fee; peak_price = cc; trail_on = False
                continue
            if ml<0 and m_start>0:
                ur = sz*pos*(cc-entry_p)/entry_p
                if (balance+ur-m_start)/m_start < ml: m_paused = True

        if pos==0 and balance>100:
            if ml<0 and m_start>0:
                if (balance-m_start)/m_start < ml: m_paused = True
            if m_paused or i<pause_til:
                pending = 0; continue
            cu = mf[i]>ms[i] and mf[i-1]<=ms[i-1]
            cd = mf[i]<ms[i] and mf[i-1]>=ms[i-1]
            ao = adx_v[i]>=cfg['adx_min']; ro = cfg['rsi_min']<=rsi_v[i]<=cfg['rsi_max']
            sig = 0
            if cu and ao and ro: sig = 1
            elif cd and ao and ro: sig = -1
            if delay > 0:
                if sig != 0:
                    pending = sig; sig_idx = i
                elif pending != 0:
                    if i - sig_idx >= delay:
                        mg = min(fixed_margin, balance*0.9)
                        pos = pending; entry_p = cc; sz = mg*lev
                        balance -= sz*fee; peak_price = cc; trail_on = False; pending = 0
                    elif i - sig_idx > delay + 12:
                        pending = 0
            else:
                if sig != 0:
                    mg = min(fixed_margin, balance*0.9)
                    pos = sig; entry_p = cc; sz = mg*lev
                    balance -= sz*fee; peak_price = cc; trail_on = False
        peak_bal = max(peak_bal, balance)

    if pos != 0:
        pp = pos*(closes[-1]-entry_p)/entry_p
        balance += sz*pp - sz*fee; m_tr += 1
    if cur_m:
        monthly[cur_m] = {'pnl':balance-m_start,
            'pct':(balance-m_start)/m_start*100 if m_start>0 else 0,
            'bal':round(balance,2), 'tr':m_tr}

    pk = 3000; mdd = 0
    for m in sorted(monthly.keys()):
        b = monthly[m]['bal']; pk = max(pk, b)
        d = (pk-b)/pk if pk>0 else 0; mdd = max(mdd, d)
    gw = sum(monthly[m]['pnl'] for m in monthly if monthly[m]['pnl']>0)
    gl = abs(sum(monthly[m]['pnl'] for m in monthly if monthly[m]['pnl']<0))
    pf = gw/gl if gl>0 else gw
    total_tr = sum(monthly[m]['tr'] for m in monthly)
    return {'bal':round(balance), 'ret':round((balance-3000)/3000*100,1),
            'pf':round(pf,2), 'mdd':round(mdd*100,1), 'trades':total_tr, 'monthly':monthly}


STRATS = [
    {'name':'v14.2F','tf':'30m','mf_t':'hma','mf_p':7,'ms_t':'ema','ms_p':200,
     'adx_p':20,'adx_min':25,'rsi_min':25,'rsi_max':65,
     'sl':0.07,'ta':0.10,'tw':0.01,'delay':3,'ml':-0.15,'cp':0,'cp_dur':0,'dd':-0.40},
    {'name':'v14.4','tf':'30m','mf_t':'ema','mf_p':3,'ms_t':'ema','ms_p':200,
     'adx_p':14,'adx_min':35,'rsi_min':30,'rsi_max':65,
     'sl':0.07,'ta':0.06,'tw':0.03,'delay':0,'ml':-0.20,'cp':0,'cp_dur':0,'dd':0},
    {'name':'v15.2','tf':'30m','mf_t':'ema','mf_p':3,'ms_t':'ema','ms_p':200,
     'adx_p':14,'adx_min':35,'rsi_min':30,'rsi_max':65,
     'sl':0.05,'ta':0.06,'tw':0.05,'delay':6,'ml':-0.15,'cp':0,'cp_dur':0,'dd':0},
    {'name':'v15.4','tf':'30m','mf_t':'ema','mf_p':3,'ms_t':'ema','ms_p':200,
     'adx_p':14,'adx_min':35,'rsi_min':30,'rsi_max':65,
     'sl':0.07,'ta':0.06,'tw':0.03,'delay':0,'ml':-0.30,'cp':0,'cp_dur':0,'dd':0},
    {'name':'v13.5','tf':'5m','mf_t':'ema','mf_p':7,'ms_t':'ema','ms_p':100,
     'adx_p':14,'adx_min':30,'rsi_min':30,'rsi_max':58,
     'sl':0.07,'ta':0.08,'tw':0.06,'delay':0,'ml':-0.20,'cp':3,'cp_dur':288,'dd':-0.50},
]

def main():
    print("Loading data...")
    sys.stdout.flush()
    df_5m = load_5m()
    mtf = {'5m': df_5m, '15m': resample(df_5m, 15), '30m': resample(df_5m, 30)}
    print(f"5m:{len(mtf['5m']):,} 15m:{len(mtf['15m']):,} 30m:{len(mtf['30m']):,}")
    sys.stdout.flush()

    summary = []

    for strat in STRATS:
        tf = strat['tf']
        df_tf = mtf[tf].copy()
        df_tf['ma_fast'] = calc_ma(df_tf['close'], strat['mf_t'], strat['mf_p'])
        df_tf['ma_slow'] = calc_ma(df_tf['close'], strat['ms_t'], strat['ms_p'])
        df_tf['adx'] = adx_w(df_tf['high'], df_tf['low'], df_tf['close'], strat['adx_p'])
        df_tf['rsi'] = rsi_w(df_tf['close'], 14)

        for lev in [10, 20]:
            r = backtest_fixed(df_tf, strat, lev, 1000)
            monthly = r['monthly']
            label = f"{strat['name']} | {tf} | $1,000 x {lev}x = ${1000*lev:,}"

            print(f"\n{'='*80}")
            print(f"  {label}")
            print(f"  Bal:${r['bal']:,} | Ret:{r['ret']:+,.1f}% | PF:{r['pf']} | MDD:{r['mdd']}% | TR:{r['trades']}")
            print(f"{'='*80}")
            print(f"| 월 | 손익금 | 손익률 | 누적잔액 | 거래 |")
            print(f"|------|--------|--------|----------|------|")
            sys.stdout.flush()

            yr_key=''; yr_pnl=0; yr_start=3000; yr_tr=0
            for m in sorted(monthly.keys()):
                d = monthly[m]; y = m[:4]
                if y != yr_key and yr_key != '':
                    yr_ret = yr_pnl/yr_start*100 if yr_start>0 else 0
                    print(f"| **{yr_key}** | **${yr_pnl:+,.0f}** | **{yr_ret:+.1f}%** | | **{yr_tr}** |")
                    yr_start = d['bal']-d['pnl']; yr_pnl=0; yr_tr=0
                if yr_key=='': yr_start=3000
                yr_key=y; yr_pnl+=d['pnl']; yr_tr+=d['tr']
                print(f"| {m} | ${d['pnl']:+,.0f} | {d['pct']:+.1f}% | ${d['bal']:,.0f} | {d['tr']} |")
            if yr_key:
                yr_ret = yr_pnl/yr_start*100 if yr_start>0 else 0
                print(f"| **{yr_key}** | **${yr_pnl:+,.0f}** | **{yr_ret:+.1f}%** | | **{yr_tr}** |")
            sys.stdout.flush()

            summary.append({'name':strat['name'], 'lev':lev, 'bal':r['bal'],
                           'ret':r['ret'], 'pf':r['pf'], 'mdd':r['mdd'], 'trades':r['trades']})

    # Final comparison
    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON: Fixed $1,000 margin")
    print(f"{'='*80}")
    print(f"| 전략 | Lev | Position | 잔액 | 수익률 | PF | MDD | 거래 |")
    print(f"|------|-----|----------|------|--------|-----|------|------|")
    for s in summary:
        print(f"| {s['name']} | {s['lev']}x | ${s['lev']*1000:,} | ${s['bal']:,} | {s['ret']:+,.1f}% | {s['pf']} | {s['mdd']}% | {s['trades']} |")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
