"""
전체 기획서 통합 백테스트 - 38개 버전 일괄 실행
레버리지 수정 엔진(v25_5_fixed.py) 사용
"""
import pandas as pd, numpy as np, time, os, json, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

from v27_backtest_engine import load_5m_data, map_tf_index, calc_ema, calc_sma, calc_wma, calc_hma
from v27_1_engine import build_extended_cache
from v25_5_fixed import run_backtest_fixed

# All versions config
ALL_VERSIONS = [
    {'ver':'v12.3','ctf':'5m','ft':'ema','fl':7,'st':'ema','sl':100,'atf':'5m','ap':14,'amin':30,'rtf':'5m','rp':14,'rmin':30,'rmax':58,'sl_v':0.09,'ta':0.08,'tp':0.06,'lev':10,'mn':0.20,'mr':0.10,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v13.5','ctf':'5m','ft':'ema','fl':7,'st':'ema','sl':100,'atf':'5m','ap':14,'amin':30,'rtf':'5m','rp':14,'rmin':30,'rmax':58,'sl_v':0.07,'ta':0.08,'tp':0.06,'lev':10,'mn':0.20,'mr':0.10,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v14.2','ctf':'30m','ft':'hma','fl':9,'st':'ema','sl':200,'atf':'30m','ap':20,'amin':25,'rtf':'5m','rp':14,'rmin':25,'rmax':65,'sl_v':0.07,'ta':0.10,'tp':0.01,'lev':10,'mn':0.30,'mr':0.15,'ich':0,'ed':6,'et':1.0,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v14.4','ctf':'30m','ft':'ema','fl':3,'st':'ema','sl':200,'atf':'30m','ap':14,'amin':35,'rtf':'5m','rp':14,'rmin':30,'rmax':65,'sl_v':0.07,'ta':0.06,'tp':0.03,'lev':10,'mn':0.25,'mr':0.12,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v15.2','ctf':'30m','ft':'ema','fl':3,'st':'ema','sl':200,'atf':'30m','ap':14,'amin':35,'rtf':'5m','rp':14,'rmin':30,'rmax':65,'sl_v':0.05,'ta':0.06,'tp':0.05,'lev':10,'mn':0.30,'mr':0.15,'ich':0,'ed':12,'et':1.0,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v15.4','ctf':'30m','ft':'ema','fl':3,'st':'ema','sl':200,'atf':'30m','ap':14,'amin':35,'rtf':'5m','rp':14,'rmin':30,'rmax':65,'sl_v':0.07,'ta':0.06,'tp':0.03,'lev':10,'mn':0.40,'mr':0.20,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v15.5','ctf':'30m','ft':'ema','fl':3,'st':'ema','sl':200,'atf':'30m','ap':14,'amin':35,'rtf':'5m','rp':14,'rmin':35,'rmax':65,'sl_v':0.07,'ta':0.06,'tp':0.05,'lev':10,'mn':0.35,'mr':0.17,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v15.6A','ctf':'15m','ft':'ema','fl':3,'st':'ema','sl':150,'atf':'15m','ap':14,'amin':45,'rtf':'5m','rp':14,'rmin':35,'rmax':70,'sl_v':0.04,'ta':0.10,'tp':0.03,'lev':15,'mn':0.40,'mr':0.20,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v15.6B','ctf':'30m','ft':'ema','fl':3,'st':'ema','sl':200,'atf':'30m','ap':14,'amin':35,'rtf':'5m','rp':14,'rmin':35,'rmax':65,'sl_v':0.07,'ta':0.06,'tp':0.05,'lev':10,'mn':0.35,'mr':0.17,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v16.0','ctf':'30m','ft':'wma','fl':10,'st':'ema','sl':200,'atf':'30m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':35,'rmax':65,'sl_v':0.08,'ta':0.04,'tp':0.03,'lev':10,'mn':0.50,'mr':0.25,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v16.4','ctf':'30m','ft':'wma','fl':10,'st':'ema','sl':200,'atf':'30m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':35,'rmax':65,'sl_v':0.08,'ta':0.04,'tp':0.03,'lev':10,'mn':0.30,'mr':0.15,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v16.5','ctf':'10m','ft':'hma','fl':21,'st':'ema','sl':200,'atf':'10m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':40,'rmax':75,'sl_v':0.06,'ta':0.07,'tp':0.03,'lev':10,'mn':0.40,'mr':0.20,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v16.6','ctf':'30m','ft':'wma','fl':10,'st':'ema','sl':200,'atf':'30m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':30,'rmax':70,'sl_v':0.08,'ta':0.03,'tp':0.02,'lev':10,'mn':0.50,'mr':0.25,'ich':0,'ed':12,'et':1.0,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v17.0_EngB','ctf':'30m','ft':'wma','fl':10,'st':'ema','sl':200,'atf':'30m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':35,'rmax':65,'sl_v':0.07,'ta':0.06,'tp':0.03,'lev':10,'mn':0.35,'mr':0.17,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v17.0_EngD','ctf':'30m','ft':'wma','fl':10,'st':'ema','sl':200,'atf':'30m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':30,'rmax':70,'sl_v':0.08,'ta':0.03,'tp':0.02,'lev':10,'mn':0.50,'mr':0.25,'ich':0,'ed':12,'et':1.0,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v22.0A','ctf':'15m','ft':'ema','fl':7,'st':'ema','sl':200,'atf':'15m','ap':14,'amin':45,'rtf':'5m','rp':14,'rmin':35,'rmax':70,'sl_v':0.05,'ta':0.08,'tp':0.04,'lev':15,'mn':0.50,'mr':0.25,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v22.2','ctf':'30m','ft':'ema','fl':3,'st':'ema','sl':200,'atf':'30m','ap':14,'amin':35,'rtf':'5m','rp':14,'rmin':30,'rmax':65,'sl_v':0.07,'ta':0.06,'tp':0.03,'lev':10,'mn':0.60,'mr':0.30,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v22.3','ctf':'30m','ft':'ema','fl':3,'st':'ema','sl':200,'atf':'30m','ap':20,'amin':25,'rtf':'5m','rp':14,'rmin':35,'rmax':65,'sl_v':0.08,'ta':0.05,'tp':0.04,'lev':10,'mn':0.60,'mr':0.30,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v22.4','ctf':'15m','ft':'ema','fl':7,'st':'ema','sl':200,'atf':'15m','ap':14,'amin':45,'rtf':'5m','rp':14,'rmin':35,'rmax':75,'sl_v':0.08,'ta':0.07,'tp':0.05,'lev':15,'mn':0.40,'mr':0.20,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':5000},
    {'ver':'v22.5','ctf':'15m','ft':'ema','fl':7,'st':'ema','sl':200,'atf':'15m','ap':14,'amin':45,'rtf':'5m','rp':14,'rmin':35,'rmax':75,'sl_v':0.08,'ta':0.07,'tp':0.05,'lev':15,'mn':0.50,'mr':0.25,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':5000},
    {'ver':'v22.7','ctf':'15m','ft':'ema','fl':7,'st':'ema','sl':200,'atf':'15m','ap':14,'amin':45,'rtf':'5m','rp':14,'rmin':35,'rmax':75,'sl_v':0.08,'ta':0.07,'tp':0.05,'lev':15,'mn':0.50,'mr':0.25,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':5000},
    {'ver':'v23.4','ctf':'30m','ft':'ema','fl':3,'st':'ema','sl':200,'atf':'30m','ap':14,'amin':35,'rtf':'5m','rp':14,'rmin':30,'rmax':65,'sl_v':0.07,'ta':0.06,'tp':0.03,'lev':10,'mn':0.30,'mr':0.15,'ich':0,'ed':0,'et':0.5,'rev':1,'mb':1,'macd':'12_26_9','cap':5000},
    {'ver':'v23.5','ctf':'10m','ft':'ema','fl':3,'st':'sma','sl':200,'atf':'10m','ap':14,'amin':35,'rtf':'5m','rp':14,'rmin':40,'rmax':75,'sl_v':0.10,'ta':0.08,'tp':0.04,'lev':3,'mn':0.25,'mr':0.12,'ich':0,'ed':12,'et':1.0,'rev':0,'mb':1,'macd':'12_26_9','cap':5000},
    {'ver':'v24.2','ctf':'1h','ft':'ema','fl':3,'st':'ema','sl':100,'atf':'30m','ap':20,'amin':30,'rtf':'5m','rp':14,'rmin':30,'rmax':70,'sl_v':0.08,'ta':0.06,'tp':0.05,'lev':10,'mn':0.70,'mr':0.35,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v25.0','ctf':'5m','ft':'ema','fl':5,'st':'ema','sl':100,'atf':'5m','ap':14,'amin':30,'rtf':'5m','rp':14,'rmin':40,'rmax':60,'sl_v':0.04,'ta':0.05,'tp':0.03,'lev':10,'mn':0.30,'mr':0.15,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v25.1A','ctf':'10m','ft':'hma','fl':21,'st':'ema','sl':200,'atf':'10m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':40,'rmax':75,'sl_v':0.06,'ta':0.07,'tp':0.03,'lev':10,'mn':0.50,'mr':0.25,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v25.2A','ctf':'30m','ft':'ema','fl':3,'st':'ema','sl':200,'atf':'15m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':40,'rmax':75,'sl_v':0.07,'ta':0.07,'tp':0.03,'lev':10,'mn':0.40,'mr':0.20,'ich':0,'ed':0,'et':0.5,'rev':1,'mb':1,'macd':'5_35_5','cap':3000},
    {'ver':'v25.2B','ctf':'30m','ft':'ema','fl':7,'st':'ema','sl':100,'atf':'15m','ap':20,'amin':35,'rtf':'10m','rp':14,'rmin':40,'rmax':75,'sl_v':0.07,'ta':0.06,'tp':0.03,'lev':10,'mn':0.40,'mr':0.20,'ich':0,'ed':0,'et':0.5,'rev':1,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v25.2C','ctf':'10m','ft':'ema','fl':5,'st':'ema','sl':300,'atf':'15m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':40,'rmax':75,'sl_v':0.07,'ta':0.08,'tp':0.03,'lev':10,'mn':0.40,'mr':0.20,'ich':0,'ed':0,'et':0.5,'rev':1,'mb':1,'macd':'5_35_5','cap':3000},
    {'ver':'v26.0A','ctf':'30m','ft':'ema','fl':3,'st':'sma','sl':200,'atf':'30m','ap':14,'amin':40,'rtf':'5m','rp':14,'rmin':30,'rmax':70,'sl_v':0.08,'ta':0.04,'tp':0.03,'lev':10,'mn':0.50,'mr':0.25,'ich':0,'ed':0,'et':0.5,'rev':0,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v27A','ctf':'30m','ft':'ema','fl':5,'st':'sma','sl':50,'atf':'5m','ap':14,'amin':35,'rtf':'5m','rp':14,'rmin':40,'rmax':80,'sl_v':0.07,'ta':0.07,'tp':0.03,'lev':10,'mn':0.20,'mr':0.10,'ich':0,'ed':12,'et':0.5,'rev':1,'mb':1,'macd':'12_26_9','cap':3000},
    {'ver':'v25.5_A','ctf':'15m','ft':'ema','fl':3,'st':'ema','sl':200,'atf':'30m','ap':20,'amin':30,'rtf':'5m','rp':14,'rmin':30,'rmax':70,'sl_v':0.03,'ta':0.04,'tp':0.02,'lev':15,'mn':0.30,'mr':0.15,'ich':0,'ed':6,'et':0.5,'rev':1,'mb':24,'macd':'8_21_5','cap':5000},
]


def main():
    print("="*70, flush=True)
    print("ALL VERSIONS BACKTEST (fixed leverage engine)", flush=True)
    print("="*70, flush=True)

    df_5m = load_5m_data()
    print("Building cache...", flush=True)
    cache = build_extended_cache(df_5m)

    ts_5m=cache['5m']['timestamp']; close=cache['5m']['close']; high=cache['5m']['high']; low=cache['5m']['low']
    n=len(close); ts_i64=ts_5m.astype('int64')

    tf_maps={}
    for tf in ['10m','15m','30m','1h']:
        tf_maps[tf]=map_tf_index(ts_5m, cache[tf]['timestamp'])

    ich=cache['5m']['ichimoku_cloud']; atr=cache['5m']['atr'][14]
    macd_m={mk:cache['5m']['macd'][mk]['hist'] for mk in ['5_35_5','8_21_5','12_26_9']}

    # JIT warmup
    nw=1000
    _=run_backtest_fixed(close[:nw],high[:nw],low[:nw],ts_i64[:nw],
        np.zeros(nw,dtype=np.int64),np.zeros(nw),np.zeros(nw),np.zeros(nw),np.zeros(nw),np.zeros(nw),
        25,30,70,0,0.05,0.05,0.03,10,0.30,0.15,-0.20,0.0004,3000.0,6,1.0,0,6)

    results=[]
    for v in ALL_VERSIONS:
        ctf=v['ctf']; c_data=cache[ctf]
        ft=v['ft']; fl=v['fl']; st=v['st']; sl_len=v['sl']

        fm=c_data.get(ft,c_data['ema']).get(fl) if isinstance(c_data.get(ft),dict) else c_data['ema'].get(fl)
        sm=c_data.get(st,c_data['ema']).get(sl_len) if isinstance(c_data.get(st),dict) else c_data['ema'].get(sl_len)
        if fm is None or sm is None:
            print(f"  {v['ver']}: MA not found ({ft}{fl}/{st}{sl_len}), SKIP", flush=True)
            continue

        sig=np.zeros(len(fm),dtype=np.int64)
        for j in range(1,len(fm)):
            if fm[j]>sm[j]: sig[j]=1
            elif fm[j]<sm[j]: sig[j]=-1
        csig=sig if ctf=='5m' else sig[tf_maps[ctf]]

        atf=v['atf']; adx_v=cache[atf]['adx'][v['ap']]
        if atf!='5m': adx_v=adx_v[tf_maps[atf]]
        rtf=v.get('rtf','5m'); rsi_v=cache[rtf]['rsi'][v['rp']]
        if rtf!='5m': rsi_v=rsi_v[tf_maps[rtf]]
        mv=macd_m[v.get('macd','12_26_9')]

        result=run_backtest_fixed(
            close,high,low,ts_i64,
            csig,adx_v,rsi_v,atr,mv,ich,
            v['amin'],v['rmin'],v['rmax'],v.get('ich',0),
            v['sl_v'],v['ta'],v['tp'],
            v['lev'],v['mn'],v['mr'],-0.20,0.0004,float(v.get('cap',3000)),
            v.get('ed',0),v.get('et',0.5),v.get('rev',0),v.get('mb',1)
        )

        fc,tc=result[0],result[1]; pnls=result[4]
        if tc>0:
            wins=np.sum(pnls>0)
            tpro=np.sum(pnls[pnls>0]); tlos=abs(np.sum(pnls[pnls<0]))+1e-10
            pf=tpro/tlos; ret=(fc-v.get('cap',3000))/v.get('cap',3000)*100
            eq=result[8]; mdd=0
            if len(eq)>0: peq=np.maximum.accumulate(eq); dd=(eq-peq)/(peq+1e-10); mdd=abs(np.min(dd))*100
            etypes=result[6]
            results.append({
                'ver':v['ver'],'lev':v['lev'],'margin':v['mn'],
                'cross':f"{ctf} {ft.upper()}{fl}/{st.upper()}{sl_len}",
                'final_cap':float(fc),'cap':v.get('cap',3000),
                'return_pct':float(ret),'pf':float(pf),'mdd':float(mdd),
                'trades':int(tc),'win_rate':float(wins/tc*100),
                'sl_c':int(np.sum(etypes==0)),'tsl_c':int(np.sum(etypes==1)),'rev_c':int(np.sum(etypes==2)),
            })
            print(f"  {v['ver']:>12} | {v['lev']:>2}x M{v['mn']*100:.0f}% | {ctf} {ft.upper()}{fl}/{st.upper()}{sl_len} | "
                  f"${fc:>14,.0f} Ret:{ret:>10,.0f}% PF:{pf:>6.2f} MDD:{mdd:>5.1f}% T:{tc:>4} WR:{wins/tc*100:>5.1f}%", flush=True)
        else:
            print(f"  {v['ver']:>12} | NO TRADES", flush=True)

    # Sort & display
    by_ret=sorted(results, key=lambda x: x['return_pct'], reverse=True)
    by_score=sorted(results, key=lambda x: x['pf']*x['return_pct']*np.sqrt(x['trades'])/(x['mdd']+5), reverse=True)

    print(f"\n{'='*100}")
    print(f"  수익률 BEST 10")
    print(f"{'='*100}")
    print(f"| {'순위':^4} | {'버전':^12} | {'레버':^4} | {'마진':^5} | {'Cross':^25} | {'수익률':^12} | {'최종잔액':^14} | {'PF':^6} | {'MDD':^6} | {'거래':^4} |")
    print(f"|{'-'*6}|{'-'*14}|{'-'*6}|{'-'*7}|{'-'*27}|{'-'*14}|{'-'*16}|{'-'*8}|{'-'*8}|{'-'*6}|")
    for i,r in enumerate(by_ret[:10]):
        print(f"| {i+1:^4} | {r['ver']:^12} | {r['lev']:>2}x  | {r['margin']*100:>4.0f}% | {r['cross']:^25} | {r['return_pct']:>+10,.0f}% | ${r['final_cap']:>12,.0f} | {r['pf']:>5.2f} | {r['mdd']:>5.1f}% | {r['trades']:>4} |")

    print(f"\n{'='*100}")
    print(f"  전체 종합 BEST 10 (수익률 × PF × √거래수 / MDD)")
    print(f"{'='*100}")
    print(f"| {'순위':^4} | {'버전':^12} | {'레버':^4} | {'마진':^5} | {'Cross':^25} | {'수익률':^12} | {'최종잔액':^14} | {'PF':^6} | {'MDD':^6} | {'거래':^4} |")
    print(f"|{'-'*6}|{'-'*14}|{'-'*6}|{'-'*7}|{'-'*27}|{'-'*14}|{'-'*16}|{'-'*8}|{'-'*8}|{'-'*6}|")
    for i,r in enumerate(by_score[:10]):
        print(f"| {i+1:^4} | {r['ver']:^12} | {r['lev']:>2}x  | {r['margin']*100:>4.0f}% | {r['cross']:^25} | {r['return_pct']:>+10,.0f}% | ${r['final_cap']:>12,.0f} | {r['pf']:>5.2f} | {r['mdd']:>5.1f}% | {r['trades']:>4} |")

    # Save
    output={'results':results,'best10_return':[r for r in by_ret[:10]],'best10_score':[r for r in by_score[:10]]}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'v_all_results.json'),'w',encoding='utf-8') as f:
        json.dump(output,f,indent=2,ensure_ascii=False,default=lambda o:int(o) if isinstance(o,(np.integer,)) else float(o) if isinstance(o,(np.floating,)) else o)
    print("\nSaved: v_all_results.json\nDONE.", flush=True)

if __name__=='__main__':
    main()
