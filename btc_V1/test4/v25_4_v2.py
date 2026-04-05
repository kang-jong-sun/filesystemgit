"""
v25.4 v2 - 수익률 극대화 재구성
================================
문제: PF 5+ 고집 → 26거래 → 복리 효과 없음 → 수익 80%
해법: PF 3+ 허용 + 40% 마진 + REV 활성화 → 복리 폭발
참고: v25.2 Model A = PF 10.15, 63거래, +166,670% (40% 마진 복리)

접근:
1. 40% 마진 고정 (복리 효과 극대화)
2. REV 활성화 (거래 빈도 증가)
3. 이치모쿠 선택적 (너무 엄격하면 OFF)
4. PF 3+ & MDD <50% 타겟 (수익률 우선)
5. 초기자본 $5,000
"""
import pandas as pd
import numpy as np
import time, os, json, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

from v27_backtest_engine import load_5m_data, map_tf_index
from v27_1_engine import build_extended_cache, run_backtest_v271


def main():
    print("="*70, flush=True)
    print("v25.4 v2 - RETURN MAXIMIZATION", flush=True)
    print("$5,000 | 40% margin | REV ON | PF 3+ target", flush=True)
    print("="*70, flush=True)

    df_5m = load_5m_data()
    print("\nBuilding cache...", flush=True)
    cache = build_extended_cache(df_5m)

    ts_5m = cache['5m']['timestamp']
    close_5m = cache['5m']['close']
    high_5m = cache['5m']['high']
    low_5m = cache['5m']['low']
    n = len(close_5m)
    ts_i64 = ts_5m.astype('int64')

    tf_maps = {}
    for tf in ['10m','15m','30m','1h']:
        tf_maps[tf] = map_tf_index(ts_5m, cache[tf]['timestamp'])

    # JIT
    nw = 1000
    _ = run_backtest_v271(
        close_5m[:nw],high_5m[:nw],low_5m[:nw],ts_i64[:nw],
        np.zeros(nw,dtype=np.int64),
        np.zeros(nw),np.zeros(nw),np.zeros(nw),np.zeros(nw),
        np.zeros(nw),np.zeros(nw),np.zeros(nw),np.zeros(nw),np.zeros(nw),np.zeros(nw),
        25,30,70,20,80,-200,200,0,0,0,0.01,
        0.07,0.07,0.03,0,0,10,0.40,0.20,-0.20,0.0004,5000.0,6,1.0,1,6
    )

    sk=cache['5m']['stoch_k']; sd=cache['5m']['stoch_d']
    cci_v=cache['5m']['cci']; obv_v=cache['5m']['obv_slope']
    bbw=cache['5m']['bb_width']; ich=cache['5m']['ichimoku_cloud']
    atr_v=cache['5m']['atr'][14]

    # v25.2 검증 구조 기반 cross configs
    cross_configs = [
        # (ctf, ft, fl, st, sl, adx_tf, adx_p, rsi_tf, rsi_p)
        ('30m','ema',3,'ema',200, '15m',20, '5m',14),   # v25.2 A (best)
        ('30m','ema',3,'ema',150, '15m',20, '5m',14),
        ('30m','ema',3,'ema',100, '15m',20, '5m',14),
        ('30m','ema',5,'ema',200, '15m',20, '5m',14),
        ('30m','ema',5,'ema',100, '15m',20, '5m',14),
        ('30m','ema',7,'ema',100, '15m',20, '10m',14),  # v25.2 B
        ('30m','ema',7,'ema',200, '15m',20, '5m',14),
        ('30m','hma',9,'ema',200, '15m',20, '5m',14),
        ('30m','hma',14,'ema',200,'15m',20, '5m',14),
        ('10m','ema',5,'ema',300, '15m',20, '5m',14),   # v25.2 C
        ('10m','ema',5,'ema',200, '15m',20, '5m',14),
        ('15m','ema',3,'ema',200, '30m',20, '5m',14),
        ('15m','ema',5,'ema',200, '15m',20, '5m',14),
    ]

    cross_sigs={}; adx_m={}; rsi_m={}
    for ctf,ft,fl,st,sl_len,atf,ap,rtf,rp in cross_configs:
        label=f"{ctf}_{ft}{fl}_{st}{sl_len}"
        c_data=cache[ctf]
        fm=c_data.get(ft,c_data['ema']).get(fl) if isinstance(c_data.get(ft),dict) else c_data['ema'].get(fl)
        sm=c_data.get(st,c_data['ema']).get(sl_len) if isinstance(c_data.get(st),dict) else c_data['ema'].get(sl_len)
        if fm is None or sm is None: continue
        sig=np.zeros(len(fm),dtype=np.int64)
        for j in range(1,len(fm)):
            if fm[j]>sm[j]: sig[j]=1
            elif fm[j]<sm[j]: sig[j]=-1
        cross_sigs[label]=sig if ctf=='5m' else sig[tf_maps[ctf]]
        ak=f"{atf}_{ap}"
        if ak not in adx_m:
            v=cache[atf]['adx'][ap]; adx_m[ak]=v if atf=='5m' else v[tf_maps[atf]]
        rk=f"{rtf}_{rp}"
        if rk not in rsi_m:
            v=cache[rtf]['rsi'][rp]; rsi_m[rk]=v if rtf=='5m' else v[tf_maps[rtf]]

    macd_m={mk:cache['5m']['macd'][mk]['hist'] for mk in ['5_35_5','8_21_5','12_26_9']}

    # 수익률 극대화 그리드
    adx_mins = [25, 30, 35, 40]
    rsi_ranges = [(30,70), (35,70), (35,75), (40,75)]
    macd_keys = ['5_35_5', '8_21_5', '12_26_9']
    ich_opts = [0, 1]

    # SL/Trail - v25.2 검증 범위 + 확장
    sl_trails = [
        (0.05,0.06,0.03), (0.05,0.07,0.03),
        (0.07,0.07,0.03), (0.07,0.08,0.03), (0.07,0.10,0.04),
        (0.10,0.10,0.04), (0.10,0.12,0.05),
    ]
    # 40% 마진 고정 (DD시 20%)
    margins = [(0.40,0.20)]
    entry_delays = [0, 6, 12]
    entry_tols = [0.5, 1.0, 1.5]
    rev_modes = [0, 1, 2]  # REV 탐색 핵심
    min_bars = [1, 6, 12, 24]

    stoch_cfgs=[(0,100)]; cci_cfgs=[(-300,300)]; bb_opts=[(0,0)]
    partial_cfgs=[(0,0)]

    per=(len(adx_mins)*len(rsi_ranges)*len(macd_keys)*len(ich_opts)*
         len(sl_trails)*len(margins)*len(entry_delays)*len(entry_tols)*
         len(rev_modes)*len(min_bars))
    total=len(cross_configs)*per
    print(f"\n  {len(cross_configs)} crosses x {per:,}/cross = {total:,} total", flush=True)

    all_results=[]
    tested=0; t0=time.time()

    for ctf,ft,fl,st,sl_len,atf,ap,rtf,rp in cross_configs:
        label=f"{ctf}_{ft}{fl}_{st}{sl_len}"
        if label not in cross_sigs: continue
        csig=cross_sigs[label]
        adx_val=adx_m[f"{atf}_{ap}"]
        rsi_val=rsi_m[f"{rtf}_{rp}"]

        for amin in adx_mins:
            for rmin,rmax in rsi_ranges:
                for mk in macd_keys:
                    mv=macd_m[mk]
                    for ich_use in ich_opts:
                        for sl,ta,tp in sl_trails:
                            for mn,mr in margins:
                                for ed in entry_delays:
                                    for et in entry_tols:
                                        for rm in rev_modes:
                                            for mb in min_bars:
                                                tested+=1
                                                result=run_backtest_v271(
                                                    close_5m,high_5m,low_5m,ts_i64,
                                                    csig,adx_val,rsi_val,atr_v,mv,
                                                    sk,sd,cci_v,obv_v,bbw,ich,
                                                    amin,rmin,rmax,0,100,-300,300,
                                                    0,ich_use,0,0,
                                                    sl,ta,tp,0,0,
                                                    10,mn,mr,-0.20,0.0004,5000.0,
                                                    ed,et,rm,mb
                                                )
                                                fc,tc=result[0],result[1]
                                                if tc>=15 and fc>10000:
                                                    rois=result[3];pnls=result[4]
                                                    wins=np.sum(rois>0)
                                                    tpro=np.sum(pnls[pnls>0])
                                                    tlos=abs(np.sum(pnls[pnls<0]))+1e-10
                                                    pf=tpro/tlos
                                                    ret=(fc-5000)/5000*100
                                                    eq=result[8];mdd=0
                                                    if len(eq)>0:
                                                        peq=np.maximum.accumulate(eq)
                                                        dd=(eq-peq)/(peq+1e-10)
                                                        mdd=abs(np.min(dd))*100
                                                    etypes=result[6]
                                                    if pf>=2.0 and mdd<65:
                                                        all_results.append({
                                                            'cross':label,'atf':atf,'ap':ap,'rtf':rtf,'rp':rp,
                                                            'adx_min':amin,'rsi_min':rmin,'rsi_max':rmax,
                                                            'macd':mk,'ichimoku':ich_use,
                                                            'sl':sl,'trail_act':ta,'trail_pct':tp,
                                                            'margin':mn,'margin_dd':mr,
                                                            'entry_delay':ed,'entry_tol':et,
                                                            'rev_mode':rm,'min_bars':mb,
                                                            'final_cap':float(fc),'trades':int(tc),
                                                            'win_rate':float(wins/tc*100),
                                                            'pf':float(pf),'mdd':float(mdd),
                                                            'return_pct':float(ret),
                                                            'avg_win':float(np.mean(rois[rois>0])*100) if wins>0 else 0,
                                                            'avg_loss':float(np.mean(rois[rois<=0])*100) if (tc-wins)>0 else 0,
                                                            'sl_c':int(np.sum(etypes==0)),
                                                            'tsl_c':int(np.sum(etypes==1)),
                                                            'rev_c':int(np.sum(etypes==2)),
                                                        })
                                                if tested%100000==0:
                                                    elapsed=time.time()-t0
                                                    rate=tested/elapsed
                                                    rem=(total-tested)/rate/60 if rate>0 else 999
                                                    # Top return so far
                                                    top_ret=max([r['return_pct'] for r in all_results]) if all_results else 0
                                                    top_pf=max([r['pf'] for r in all_results]) if all_results else 0
                                                    print(f"  {tested:,}/{total:,} ({tested/total*100:.1f}%) | {rate:.0f}/s | ~{rem:.0f}min | {len(all_results)} passed | TopRet:{top_ret:.0f}% TopPF:{top_pf:.1f}", flush=True)

    elapsed=time.time()-t0
    print(f"\n  Done: {tested:,} in {elapsed:.0f}s | {len(all_results)} passed (PF>=2, MDD<65%)", flush=True)

    # Score by RETURN primarily (PF is secondary)
    for r in all_results:
        r['score'] = r['return_pct'] * r['pf'] * np.sqrt(r['trades']) / (r['mdd']+10)
    all_results.sort(key=lambda x: x['score'], reverse=True)

    pf5=[r for r in all_results if r['pf']>=5]
    pf3=[r for r in all_results if r['pf']>=3]
    print(f"  PF>=2:{len(all_results)} PF>=3:{len(pf3)} PF>=5:{len(pf5)}", flush=True)

    # Show top by RETURN
    by_ret=sorted(all_results, key=lambda x: x['return_pct'], reverse=True)
    print(f"\n  === Top 10 by RETURN ===", flush=True)
    for i,r in enumerate(by_ret[:10]):
        print(f"  #{i+1}: {r['cross']} ADX>{r['adx_min']} RSI:{r['rsi_min']}-{r['rsi_max']} ICH:{r['ichimoku']} "
              f"SL:{r['sl']*100:.0f}% T+{r['trail_act']*100:.0f}%/-{r['trail_pct']*100:.0f}% "
              f"REV:{r['rev_mode']} MB:{r['min_bars']} ED:{r['entry_delay']} "
              f"| Ret:{r['return_pct']:,.0f}% PF:{r['pf']:.2f} MDD:{r['mdd']:.1f}% T:{r['trades']}", flush=True)

    # Show top by PF (with decent return)
    good_ret=[r for r in all_results if r['return_pct']>=500]
    if good_ret:
        by_pf=sorted(good_ret, key=lambda x: x['pf'], reverse=True)
        print(f"\n  === Top 10 by PF (Ret>=500%) ===", flush=True)
        for i,r in enumerate(by_pf[:10]):
            print(f"  #{i+1}: {r['cross']} | Ret:{r['return_pct']:,.0f}% PF:{r['pf']:.2f} MDD:{r['mdd']:.1f}% T:{r['trades']} REV:{r['rev_mode']}", flush=True)

    # Show top by composite score
    print(f"\n  === Top 10 by Score ===", flush=True)
    for i,r in enumerate(all_results[:10]):
        print(f"  #{i+1}: {r['cross']} | Ret:{r['return_pct']:,.0f}% PF:{r['pf']:.2f} MDD:{r['mdd']:.1f}% T:{r['trades']} REV:{r['rev_mode']}", flush=True)

    # Select Models
    model_a = by_ret[0]  # 최고 수익률
    model_b = max(all_results[:50], key=lambda x: x['pf']/(x['mdd']+3))  # 안정형
    # 균형형: 높은 수익 + PF 3+
    balanced=[r for r in all_results if r['pf']>=3.0]
    model_c = max(balanced[:50], key=lambda x: x['return_pct']*x['pf']/(x['mdd']+5)) if balanced else all_results[0]

    models={'A':model_a,'B':model_b,'C':model_c}
    for name,m in models.items():
        print(f"\n  Model {name}: {m['cross']}", flush=True)
        print(f"    ADX:{m['atf']}>{m['adx_min']} RSI:{m['rtf']}{m['rsi_min']}-{m['rsi_max']} MACD:{m['macd']} ICH:{m['ichimoku']}", flush=True)
        print(f"    SL:{m['sl']*100:.0f}% Trail:+{m['trail_act']*100:.0f}%/-{m['trail_pct']*100:.0f}% M:{m['margin']*100:.0f}% REV:{m['rev_mode']} MB:{m['min_bars']}", flush=True)
        print(f"    Ret:{m['return_pct']:,.0f}% PF:{m['pf']:.2f} MDD:{m['mdd']:.1f}% T:{m['trades']} WR:{m['win_rate']:.1f}%", flush=True)
        print(f"    AvgW:{m['avg_win']:.1f}% AvgL:{m['avg_loss']:.1f}% Final:${m['final_cap']:,.0f}", flush=True)
        print(f"    SL:{m['sl_c']} TSL:{m['tsl_c']} REV:{m['rev_c']}", flush=True)

    # 30x validation
    print("\n[30x Validation]", flush=True)
    validation={}
    for name,m in models.items():
        csig=cross_sigs[m['cross']]
        adx_val=adx_m[f"{m['atf']}_{m['ap']}"]
        rsi_val=rsi_m[f"{m['rtf']}_{m['rp']}"]
        mv=macd_m[m['macd']]
        vals=[]
        for run in range(30):
            s=int(n*(0.02+(run/30)*0.13))
            ml=min(n-s,len(csig)-s,len(adx_val)-s,len(rsi_val)-s)
            r=run_backtest_v271(
                close_5m[s:s+ml],high_5m[s:s+ml],low_5m[s:s+ml],ts_i64[s:s+ml],
                csig[s:s+ml],adx_val[s:s+ml],rsi_val[s:s+ml],atr_v[s:s+ml],mv[s:s+ml],
                sk[s:s+ml],sd[s:s+ml],cci_v[s:s+ml],obv_v[s:s+ml],bbw[s:s+ml],ich[s:s+ml],
                m['adx_min'],m['rsi_min'],m['rsi_max'],0,100,-300,300,
                0,m['ichimoku'],0,0,
                m['sl'],m['trail_act'],m['trail_pct'],0,0,
                10,m['margin'],m['margin_dd'],-0.20,0.0004,5000.0,
                m['entry_delay'],m['entry_tol'],m['rev_mode'],m['min_bars']
            )
            fc,tc=r[0],r[1];rois=r[3];pnls=r[4]
            w=np.sum(rois>0) if tc>0 else 0
            pf=np.sum(pnls[pnls>0])/(abs(np.sum(pnls[pnls<0]))+1e-10)
            eq=r[8];mdd=0
            if len(eq)>0:peq=np.maximum.accumulate(eq);dd=(eq-peq)/(peq+1e-10);mdd=abs(np.min(dd))*100
            vals.append({'run':run+1,'fc':float(fc),'ret':float((fc-5000)/5000*100),
                        'trades':int(tc),'wr':float(w/tc*100) if tc>0 else 0,
                        'pf':float(pf),'mdd':float(mdd)})
        validation[name]=vals
        rets=[v['ret'] for v in vals];pfs=[v['pf'] for v in vals];mdds=[v['mdd'] for v in vals]
        print(f"  {name}: Ret={np.mean(rets):,.0f}%+/-{np.std(rets):,.0f}% PF={np.mean(pfs):.2f}+/-{np.std(pfs):.2f} MDD={np.mean(mdds):.1f}% T={np.mean([v['trades'] for v in vals]):.0f}", flush=True)
        print(f"      Min Ret={np.min(rets):,.0f}% Max Ret={np.max(rets):,.0f}% Min PF={np.min(pfs):.2f}", flush=True)

    # Save
    output={'version':'v25.4_v2','initial_capital':5000,'margin_mode':'isolated',
            'total_tested':tested,
            'models':{n:{k:v for k,v in m.items() if k!='score'} for n,m in models.items()},
            'validation':validation,
            'top20_by_return':[{k:v for k,v in r.items() if k!='score'} for r in by_ret[:20]],
            'top20_by_score':[{k:v for k,v in r.items() if k!='score'} for r in all_results[:20]]}
    def conv(o):
        if isinstance(o,(np.integer,)):return int(o)
        if isinstance(o,(np.floating,)):return float(o)
        if isinstance(o,np.ndarray):return o.tolist()
        return o
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'v25_4_v2_results.json')
    with open(path,'w',encoding='utf-8') as f:
        json.dump(output,f,indent=2,default=conv,ensure_ascii=False)
    print(f"\nSaved: {path}\nTotal: {tested:,}\n{'='*70}\nv25.4 v2 COMPLETE", flush=True)

if __name__=='__main__':
    main()
