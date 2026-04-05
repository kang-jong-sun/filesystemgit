"""
5개 전략 x 2개 레버리지(10x/20x) = 10개 조합
포지션 고정 $1,000 USDT (잔액 비율 아님)
"""
import sys,os,numpy as np,pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from bt_fast import load_5m_data,build_mtf,calc_ma,calc_adx,calc_rsi

DATA_DIR=os.path.dirname(os.path.abspath(__file__))

STRATEGIES={
    'v14.4':{'tf':'30m','mft':'ema','mf':3,'mst':'ema','ms':200,'ap':14,'am':35,'rmin':30,'rmax':65,
             'sl':0.07,'ta':0.06,'tp':0.03,'ml':-0.20,'cp':0,'dur':0},
    'v15.4':{'tf':'30m','mft':'ema','mf':3,'mst':'ema','ms':200,'ap':14,'am':35,'rmin':30,'rmax':65,
             'sl':0.07,'ta':0.06,'tp':0.03,'ml':-0.30,'cp':0,'dur':0},
    'v15.2':{'tf':'30m','mft':'ema','mf':3,'mst':'ema','ms':200,'ap':14,'am':35,'rmin':30,'rmax':65,
             'sl':0.05,'ta':0.06,'tp':0.05,'ml':-0.15,'cp':0,'dur':0},
    'v13.5':{'tf':'5m','mft':'ema','mf':7,'mst':'ema','ms':100,'ap':14,'am':30,'rmin':30,'rmax':58,
             'sl':0.07,'ta':0.08,'tp':0.06,'ml':-0.20,'cp':3,'dur':288},
    'v14.2F':{'tf':'30m','mft':'hma','mf':7,'mst':'ema','ms':200,'ap':20,'am':25,'rmin':25,'rmax':65,
              'sl':0.07,'ta':0.10,'tp':0.01,'ml':-0.15,'cp':0,'dur':0},
}

FEE=0.0004; INIT=3000.0; FIXED_POS=1000.0  # 고정 포지션 크기

def run_fixed(df, cfg, lev, label):
    c,h,l,v=df['close'],df['high'],df['low'],df['volume']
    df2=df.copy()
    df2['maf']=calc_ma(c,cfg['mft'],cfg['mf'],v)
    df2['mas']=calc_ma(c,cfg['mst'],cfg['ms'],v)
    df2['adx']=calc_adx(h,l,c,cfg['ap'])
    df2['rsi']=calc_rsi(c,14)

    times=df2['time'].values; closes=df2['close'].values.astype(np.float64)
    highs=df2['high'].values.astype(np.float64); lows=df2['low'].values.astype(np.float64)
    maf=df2['maf'].values.astype(np.float64); mas=df2['mas'].values.astype(np.float64)
    rsi=df2['rsi'].values.astype(np.float64); adx=df2['adx'].values.astype(np.float64)
    n=len(df2); LIQ=1.0/lev
    SL=cfg['sl']; TA=cfg['ta']; TP=cfg['tp']; ML=cfg['ml']
    CP=cfg['cp']; DUR=cfg['dur']

    bal=INIT; pos=0; ep=0.0; su=0.0; ppnl=0.0; trail=False
    msb=bal; cm=''; mp=False; cl=0; pu_idx=0
    monthly={}; gpeak=INIT; gmdd=0.0

    def mk(i): return pd.Timestamp(times[i]).strftime('%Y-%m')
    def em(m):
        if m not in monthly:
            monthly[m]={'sb':bal,'eb':bal,'tr':0,'w':0,'l':0,'sl':0,'tsl':0,'rev':0,'fl':0,'gp':0.0,'gl':0.0}
    def rec(m,pu,et):
        em(m); d=monthly[m]; d['tr']+=1
        if pu>0: d['w']+=1; d['gp']+=pu
        else: d['l']+=1; d['gl']+=abs(pu)
        if et in d: d[et]+=1

    for i in range(1,n):
        if np.isnan(maf[i]) or np.isnan(mas[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]): continue
        m=mk(i)
        if m!=cm:
            if cm and cm in monthly: monthly[cm]['eb']=bal
            cm=m; em(m); monthly[m]['sb']=bal; msb=bal; mp=False
        if bal>gpeak: gpeak=bal
        dd=(gpeak-bal)/gpeak if gpeak>0 else 0
        if dd>gmdd: gmdd=dd

        if pos!=0:
            if pos==1: pnl_p=(closes[i]-ep)/ep; pkc=(highs[i]-ep)/ep; lwc=(lows[i]-ep)/ep
            else: pnl_p=(ep-closes[i])/ep; pkc=(ep-lows[i])/ep; lwc=(ep-highs[i])/ep
            if pkc>ppnl: ppnl=pkc

            # 강제청산: 증거금 전액 손실
            margin_used = su / lev
            if lwc <= -LIQ:
                pu = -margin_used - su*FEE  # 증거금 전액 손실
                bal += pu; bal=max(bal,0)
                rec(m,pu,'fl'); pos=0; cl+=1
                if CP>0 and cl>=CP: pu_idx=i+DUR
                continue
            if lwc<=-SL:
                pu = su*(-SL) - su*FEE
                bal += pu; bal=max(bal,0)
                rec(m,pu,'sl'); pos=0; cl+=1
                if CP>0 and cl>=CP: pu_idx=i+DUR
                continue
            if ppnl>=TA: trail=True
            if trail:
                tl=ppnl-TP
                if pnl_p<=tl:
                    pu=su*tl-su*FEE; bal+=pu
                    rec(m,pu,'tsl'); pos=0
                    if tl>0: cl=0
                    else: cl+=1; (pu_idx:=i+DUR) if CP>0 and cl>=CP else None
                    continue

            cu=maf[i]>mas[i] and maf[i-1]<=mas[i-1]
            cd=maf[i]<mas[i] and maf[i-1]>=mas[i-1]
            ao=adx[i]>=cfg['am']; ro=cfg['rmin']<=rsi[i]<=cfg['rmax']
            rv=False; nd=0
            if pos==1 and cd and ao and ro: rv=True; nd=-1
            elif pos==-1 and cu and ao and ro: rv=True; nd=1
            if rv:
                pu=su*pnl_p-su*FEE; bal+=pu
                rec(m,pu,'rev'); pos=0
                if pnl_p>0: cl=0
                else: cl+=1; (pu_idx:=i+DUR) if CP>0 and cl>=CP else None
                # 역신호 즉시 반대 진입
                if bal>FIXED_POS/lev+10 and not mp and i>=pu_idx:
                    su=FIXED_POS*lev  # 고정 $1000 x 레버리지
                    bal-=su*FEE; pos=nd; ep=closes[i]; ppnl=0.0; trail=False
                continue

        if pos==0 and bal>FIXED_POS/lev+10:
            if ML<0 and msb>0 and (bal-msb)/msb<ML: mp=True
            if mp or i<pu_idx: continue
            cu=maf[i]>mas[i] and maf[i-1]<=mas[i-1]
            cd=maf[i]<mas[i] and maf[i-1]>=mas[i-1]
            ao=adx[i]>=cfg['am']; ro=cfg['rmin']<=rsi[i]<=cfg['rmax']
            sig=0
            if cu and ao and ro: sig=1
            elif cd and ao and ro: sig=-1
            if sig!=0:
                su=FIXED_POS*lev  # 고정 $1000 x 레버리지
                bal-=su*FEE; pos=sig; ep=closes[i]; ppnl=0.0; trail=False

        if cm in monthly: monthly[cm]['eb']=bal

    if pos!=0:
        m=mk(n-1); pnl_p=(closes[-1]-ep)/ep if pos==1 else (ep-closes[-1])/ep
        pu=su*pnl_p-su*FEE; bal+=pu
        em(m); monthly[m]['tr']+=1
        if pu>0: monthly[m]['w']+=1; monthly[m]['gp']+=pu
        else: monthly[m]['l']+=1; monthly[m]['gl']+=abs(pu)
    if cm in monthly: monthly[cm]['eb']=bal

    return bal, monthly, gmdd

def print_result(label, bal, monthly, gmdd):
    sm=sorted(monthly.keys())
    ttr=sum(d['tr'] for d in monthly.values())
    tw=sum(d['w'] for d in monthly.values())
    tsl=sum(d['sl'] for d in monthly.values())
    ttsl=sum(d['tsl'] for d in monthly.values())
    trev=sum(d['rev'] for d in monthly.values())
    tfl=sum(d['fl'] for d in monthly.values())
    tgp=sum(d['gp'] for d in monthly.values())
    tgl=sum(d['gl'] for d in monthly.values())
    tpf=tgp/tgl if tgl>0 else 0

    print(f"\n{'='*110}")
    print(f"  {label} | 고정 포지션 $1,000 | 초기 $3,000")
    print(f"  최종: ${bal:,.0f} ({(bal-INIT)/INIT*100:+,.1f}%) | PF:{tpf:.2f} | MDD:{gmdd*100:.1f}% | "
          f"거래:{ttr} (SL:{tsl} TSL:{ttsl} REV:{trev} FL:{tfl}) | 승률:{tw/ttr*100:.1f}%" if ttr>0 else "")
    print(f"{'='*110}")

    print(f"  {'월':>8} | {'손익률':>8} | {'손익금':>10} | {'잔금':>12} | {'거래':>4} | {'SL':>3} | {'TSL':>3} | {'REV':>3} | {'FL':>3} | 비고")
    print(f"  {'-'*95}")

    yearly={}
    for mk2 in sm:
        d=monthly[mk2]; s=d['sb']; e=d['eb']; pa=e-s; pp=(pa/s*100) if s>0 else 0
        notes=[]
        if pp>20: notes.append("수익")
        elif pp<-15: notes.append("손실")
        if d['fl']>0: notes.append(f"FL{d['fl']}")

        print(f"  {mk2:>8} | {pp:>+7.1f}% | ${pa:>+8,.0f} | ${e:>10,.0f} | {d['tr']:>4} | {d['sl']:>3} | {d['tsl']:>3} | {d['rev']:>3} | {d['fl']:>3} | {', '.join(notes)}")

        yr=mk2[:4]
        if yr not in yearly: yearly[yr]={'start':s,'end':e,'tr':0}
        yearly[yr]['end']=e; yearly[yr]['tr']+=d['tr']

        ni=sm.index(mk2)+1
        if ni<len(sm) and sm[ni][:4]!=yr:
            y=yearly[yr]; yp=y['end']-y['start']; ypc=(yp/y['start']*100) if y['start']>0 else 0
            print(f"  {'-'*95}")
            print(f"  {yr+'합':>8} | {ypc:>+7.1f}% | ${yp:>+8,.0f} | ${y['end']:>10,.0f} | {y['tr']:>4} |")
            print(f"  {'='*95}")

    yr=sm[-1][:4]; y=yearly[yr]; yp=y['end']-y['start']; ypc=(yp/y['start']*100) if y['start']>0 else 0
    print(f"  {'-'*95}")
    print(f"  {yr+'합':>8} | {ypc:>+7.1f}% | ${yp:>+8,.0f} | ${y['end']:>10,.0f} | {y['tr']:>4} |")

    return yearly

# 실행
df_5m=load_5m_data(DATA_DIR)
mtf=build_mtf(df_5m)

all_summary=[]

for lev in [10, 20]:
    print(f"\n\n{'#'*110}")
    print(f"  레버리지 {lev}x | 고정 포지션 $1,000 (실제 포지션: ${1000*lev:,})")
    print(f"{'#'*110}")

    for name, cfg in STRATEGIES.items():
        tf=cfg['tf']; df=mtf[tf]
        label=f"{name} {lev}x"
        bal, monthly, gmdd = run_fixed(df, cfg, lev, label)
        yr_data = print_result(label, bal, monthly, gmdd)

        ttr=sum(d['tr'] for d in monthly.values())
        tw=sum(d['w'] for d in monthly.values())
        tgp=sum(d['gp'] for d in monthly.values())
        tgl=sum(d['gl'] for d in monthly.values())
        tfl=sum(d['fl'] for d in monthly.values())
        all_summary.append({
            'name':name,'lev':lev,'bal':bal,'ret':(bal-INIT)/INIT*100,
            'pf':tgp/tgl if tgl>0 else 0,'mdd':gmdd*100,
            'tr':ttr,'wr':tw/ttr*100 if ttr>0 else 0,'fl':tfl
        })

# 최종 비교
print(f"\n\n{'='*110}")
print(f"  최종 비교: 고정 $1,000 포지션 | 5개 전략 x 2레버리지")
print(f"{'='*110}\n")
print(f"  {'전략':>8} {'Lev':>4} | {'잔액':>10} | {'수익률':>8} | {'PF':>5} | {'MDD':>6} | {'FL':>3} | {'거래':>4} | {'승률':>5}")
print(f"  {'-'*70}")
for s in sorted(all_summary, key=lambda x: x['bal'], reverse=True):
    print(f"  {s['name']:>8} {s['lev']:>3}x | ${s['bal']:>8,.0f} | {s['ret']:>+7.1f}% | {s['pf']:>4.2f} | {s['mdd']:>5.1f}% | {s['fl']:>3} | {s['tr']:>4} | {s['wr']:>4.1f}%")
