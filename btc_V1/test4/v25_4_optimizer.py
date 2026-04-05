"""
v25.4 Optimizer - PF 5+ & 거래빈도 증가 & 초기자본 $5,000 & 격리마진
=====================================================================
v27.1 검증 모델 기반 + 거래빈도 개선:
- v27.1: PF 5.05, 26회/75개월 → 너무 적음
- v25.4 목표: PF 5+, 50~200회/75개월 → 복리 효과 가속
전략: REV 모드 탐색 + 필터 완화 + 다양한 cross 조합
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
    print("v25.4 OPTIMIZER - PF 5+ & Trade Frequency Up", flush=True)
    print("Initial Capital: $5,000 | Isolated Margin", flush=True)
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
        close_5m[:nw], high_5m[:nw], low_5m[:nw], ts_i64[:nw],
        np.zeros(nw, dtype=np.int64),
        np.zeros(nw),np.zeros(nw),np.zeros(nw),np.zeros(nw),
        np.zeros(nw),np.zeros(nw),np.zeros(nw),np.zeros(nw),np.zeros(nw),np.zeros(nw),
        25,30,70,20,80,-200,200,0,0,0,0.01,
        0.07,0.07,0.03,0,0,10,0.40,0.20,-0.20,0.0004,5000.0,6,1.0,1,6
    )

    sk=cache['5m']['stoch_k']; sd=cache['5m']['stoch_d']
    cci=cache['5m']['cci']; obv=cache['5m']['obv_slope']
    bbw=cache['5m']['bb_width']; ich=cache['5m']['ichimoku_cloud']
    atr=cache['5m']['atr'][14]

    # Broader set of crosses (shorter MAs = more signals = more trades)
    cross_configs = [
        # v25.2 proven (long MAs, few trades)
        ('30m','ema',3,'ema',200, '15m',20, '5m',14),
        ('30m','ema',7,'ema',100, '15m',20, '10m',14),
        ('10m','ema',5,'ema',300, '15m',20, '5m',14),
        # Shorter slow MAs (more crosses = more trades)
        ('30m','ema',3,'ema',100, '15m',20, '5m',14),
        ('30m','ema',3,'ema',50,  '15m',20, '5m',14),
        ('30m','ema',5,'ema',100, '15m',20, '5m',14),
        ('30m','ema',5,'ema',50,  '15m',20, '5m',14),
        ('15m','ema',3,'ema',200, '30m',20, '5m',14),
        ('15m','ema',3,'ema',100, '30m',20, '5m',14),
        ('15m','ema',5,'ema',100, '15m',20, '5m',14),
        ('15m','ema',5,'ema',50,  '30m',20, '5m',14),
        ('10m','ema',3,'ema',200, '15m',20, '5m',14),
        ('10m','ema',5,'ema',100, '15m',20, '5m',14),
        ('10m','ema',5,'ema',50,  '15m',20, '5m',14),
        # HMA crosses
        ('30m','hma',9,'ema',200, '15m',20, '5m',14),
        ('30m','hma',14,'ema',200,'15m',20, '5m',14),
        ('15m','hma',9,'ema',100, '15m',20, '5m',14),
    ]

    # Build cross signals
    cross_sigs = {}
    adx_m = {}; rsi_m = {}
    for ctf,ft,fl,st,sl_len,atf,ap,rtf,rp in cross_configs:
        label = f"{ctf}_{ft}{fl}_{st}{sl_len}"
        c_data = cache[ctf]
        fm = c_data.get(ft, c_data['ema']).get(fl) if isinstance(c_data.get(ft), dict) else c_data['ema'].get(fl)
        sm = c_data.get(st, c_data['ema']).get(sl_len) if isinstance(c_data.get(st), dict) else c_data['ema'].get(sl_len)
        if fm is None or sm is None: continue
        sig = np.zeros(len(fm), dtype=np.int64)
        for j in range(1,len(fm)):
            if fm[j]>sm[j]: sig[j]=1
            elif fm[j]<sm[j]: sig[j]=-1
        cross_sigs[label] = sig if ctf=='5m' else sig[tf_maps[ctf]]

        ak=f"{atf}_{ap}"
        if ak not in adx_m:
            v=cache[atf]['adx'][ap]
            adx_m[ak] = v if atf=='5m' else v[tf_maps[atf]]
        rk=f"{rtf}_{rp}"
        if rk not in rsi_m:
            v=cache[rtf]['rsi'][rp]
            rsi_m[rk] = v if rtf=='5m' else v[tf_maps[rtf]]

    macd_m = {mk: cache['5m']['macd'][mk]['hist'] for mk in ['5_35_5','8_21_5','12_26_9']}

    # Stage 1: Quick screen with fixed extra filters, vary core params
    adx_mins = [20, 25, 30, 35]
    rsi_ranges = [(30, 70), (35, 75), (40, 75)]
    macd_keys = ['5_35_5', '12_26_9']
    ich_opts = [0, 1]

    sl_trails = [
        (0.05,0.06,0.03), (0.07,0.07,0.03), (0.07,0.08,0.03), (0.07,0.10,0.04),
    ]
    margins = [(0.15,0.08), (0.20,0.10), (0.30,0.15)]
    entry_delays = [0, 6, 12]
    entry_tols = [0.5, 1.0]
    rev_modes = [0, 1, 2]
    min_bars = [1, 6, 12, 24]
    partial_cfgs = [(0,0)]

    # Fixed for Stage 1
    stoch_cfgs = [(0,100)]  # disabled
    cci_cfgs = [(-300,300)]  # disabled
    bb_opts = [(0,0)]  # disabled

    per = (len(adx_mins)*len(rsi_ranges)*len(macd_keys)*
           len(stoch_cfgs)*len(cci_cfgs)*len(ich_opts)*len(bb_opts)*
           len(sl_trails)*len(margins)*len(entry_delays)*len(entry_tols)*
           len(rev_modes)*len(min_bars)*len(partial_cfgs))
    total = len(cross_configs) * per
    print(f"\n  {len(cross_configs)} crosses x {per:,}/cross = {total:,} total", flush=True)
    print(f"  Est time at ~500/s: {total/500/60:.0f} min", flush=True)

    all_results = []
    tested = 0
    t0 = time.time()

    for ctf,ft,fl,st,sl_len,atf,ap,rtf,rp in cross_configs:
        label = f"{ctf}_{ft}{fl}_{st}{sl_len}"
        if label not in cross_sigs: continue
        csig = cross_sigs[label]
        adx_v = adx_m[f"{atf}_{ap}"]
        rsi_v = rsi_m[f"{rtf}_{rp}"]

        for amin in adx_mins:
            for rmin,rmax in rsi_ranges:
                for mk in macd_keys:
                    macd_v = macd_m[mk]
                    for so,sob in stoch_cfgs:
                        for cmin,cmax in cci_cfgs:
                            for ich_use in ich_opts:
                                for bb_use,bb_min in bb_opts:
                                    for sl_v,ta,tp in sl_trails:
                                        for mn,mr in margins:
                                            for ed in entry_delays:
                                                for et in entry_tols:
                                                    for rm in rev_modes:
                                                        for mb in min_bars:
                                                            for pe_r,pe_roi in partial_cfgs:
                                                                tested += 1
                                                                result = run_backtest_v271(
                                                                    close_5m,high_5m,low_5m,ts_i64,
                                                                    csig,adx_v,rsi_v,atr,macd_v,
                                                                    sk,sd,cci,obv,bbw,ich,
                                                                    amin,rmin,rmax,so,sob,cmin,cmax,
                                                                    0,ich_use,bb_use,bb_min,
                                                                    sl_v,ta,tp,pe_r,pe_roi,
                                                                    10,mn,mr,-0.20,0.0004,5000.0,
                                                                    ed,et,rm,mb
                                                                )
                                                                fc,tc = result[0],result[1]
                                                                if tc >= 15 and fc > 7000:
                                                                    rois=result[3]; pnls=result[4]
                                                                    wins=np.sum(rois>0)
                                                                    tpro=np.sum(pnls[pnls>0])
                                                                    tlos=abs(np.sum(pnls[pnls<0]))+1e-10
                                                                    pf=tpro/tlos
                                                                    ret=(fc-5000)/5000*100
                                                                    eq=result[8]; mdd=0
                                                                    if len(eq)>0:
                                                                        peq=np.maximum.accumulate(eq)
                                                                        dd=(eq-peq)/(peq+1e-10)
                                                                        mdd=abs(np.min(dd))*100
                                                                    etypes=result[6]
                                                                    if pf >= 3.0:
                                                                        all_results.append({
                                                                            'cross':label,'atf':atf,'ap':ap,'rtf':rtf,'rp':rp,
                                                                            'adx_min':amin,'rsi_min':rmin,'rsi_max':rmax,
                                                                            'macd':mk,'stoch_os':so,'stoch_ob':sob,
                                                                            'cci_min':cmin,'cci_max':cmax,
                                                                            'ichimoku':ich_use,'bb_use':bb_use,'bb_min':bb_min,
                                                                            'sl':sl_v,'trail_act':ta,'trail_pct':tp,
                                                                            'margin':mn,'margin_dd':mr,
                                                                            'entry_delay':ed,'entry_tol':et,
                                                                            'rev_mode':rm,'min_bars':mb,
                                                                            'partial_ratio':pe_r,'partial_roi':pe_roi,
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
                                                                if tested % 200000 == 0:
                                                                    elapsed=time.time()-t0
                                                                    rate=tested/elapsed
                                                                    rem=(total-tested)/rate/60 if rate>0 else 999
                                                                    pf5=len([r for r in all_results if r['pf']>=5.0])
                                                                    pf4=len([r for r in all_results if r['pf']>=4.0])
                                                                    print(f"  {tested:,}/{total:,} ({tested/total*100:.1f}%) | {rate:.0f}/s | ~{rem:.0f}min | PF>=3:{len(all_results)} PF>=4:{pf4} PF>=5:{pf5}", flush=True)

    elapsed = time.time()-t0
    pf5=[r for r in all_results if r['pf']>=5.0]
    pf4=[r for r in all_results if r['pf']>=4.0]
    print(f"\n  Done: {tested:,} in {elapsed:.0f}s | PF>=3:{len(all_results)} PF>=4:{len(pf4)} PF>=5:{len(pf5)}", flush=True)

    for r in all_results:
        # Score: PF * sqrt(trades) * return / (MDD+5) — rewards both PF AND trade count
        r['score'] = r['pf'] * np.sqrt(r['trades']) * r['return_pct'] / (r['mdd']+5)
    all_results.sort(key=lambda x: x['score'], reverse=True)

    pool = pf5 if len(pf5)>=3 else (pf4 if len(pf4)>=3 else all_results)
    for r in pool:
        if 'score' not in r:
            r['score'] = r['pf'] * np.sqrt(r['trades']) * r['return_pct'] / (r['mdd']+5)
    pool.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n  Top 20:", flush=True)
    for i,r in enumerate(pool[:20]):
        print(f"  #{i+1}: {r['cross']} ADX>{r['adx_min']} RSI:{r['rsi_min']}-{r['rsi_max']} M:{r['macd']} "
              f"ICH:{r['ichimoku']} SL:{r['sl']*100:.0f}% T+{r['trail_act']*100:.0f}%/-{r['trail_pct']*100:.0f}% "
              f"M:{r['margin']*100:.0f}% REV:{r['rev_mode']} MB:{r['min_bars']} "
              f"| Ret:{r['return_pct']:.0f}% PF:{r['pf']:.2f} MDD:{r['mdd']:.1f}% T:{r['trades']} "
              f"WR:{r['win_rate']:.0f}% SL:{r['sl_c']} TSL:{r['tsl_c']} REV:{r['rev_c']}", flush=True)

    # Select 3 models
    # A: 최고 복합 점수 (수익 x PF x 빈도)
    model_a = pool[0] if pool else all_results[0]
    # B: PF 최대 / MDD 최소 (안정형)
    model_b = max(pool[:50], key=lambda x: x['pf']/(x['mdd']+3))
    # C: 거래빈도 최대 with PF>=4
    freq_pool = [r for r in pool if r['pf']>=4.0] or pool
    model_c = max(freq_pool[:50], key=lambda x: x['trades']*x['pf']*np.log(max(x['return_pct'],1)+1)/(x['mdd']+5))

    models = {'A':model_a, 'B':model_b, 'C':model_c}
    for name,m in models.items():
        print(f"\n  Model {name}: {m['cross']} | Ret:{m['return_pct']:.0f}% PF:{m['pf']:.2f} MDD:{m['mdd']:.1f}% T:{m['trades']}", flush=True)
        print(f"    WR:{m['win_rate']:.1f}% AvgW:{m['avg_win']:.1f}% AvgL:{m['avg_loss']:.1f}% Final:${m['final_cap']:,.0f}", flush=True)
        print(f"    SL:{m['sl_c']} TSL:{m['tsl_c']} REV:{m['rev_c']}", flush=True)

    # 30x validation
    print("\n[30x Validation]", flush=True)
    validation = {}
    for name,m in models.items():
        csig = cross_sigs[m['cross']]
        adx_v = adx_m[f"{m['atf']}_{m['ap']}"]
        rsi_v = rsi_m[f"{m['rtf']}_{m['rp']}"]
        macd_v = macd_m[m['macd']]
        vals = []
        for run in range(30):
            s=int(n*(0.02+(run/30)*0.13))
            ml=min(n-s,len(csig)-s,len(adx_v)-s,len(rsi_v)-s)
            r=run_backtest_v271(
                close_5m[s:s+ml],high_5m[s:s+ml],low_5m[s:s+ml],ts_i64[s:s+ml],
                csig[s:s+ml],adx_v[s:s+ml],rsi_v[s:s+ml],atr[s:s+ml],macd_v[s:s+ml],
                sk[s:s+ml],sd[s:s+ml],cci[s:s+ml],obv[s:s+ml],bbw[s:s+ml],ich[s:s+ml],
                m['adx_min'],m['rsi_min'],m['rsi_max'],
                m['stoch_os'],m['stoch_ob'],m['cci_min'],m['cci_max'],
                0,m['ichimoku'],m['bb_use'],m['bb_min'],
                m['sl'],m['trail_act'],m['trail_pct'],m['partial_ratio'],m['partial_roi'],
                10,m['margin'],m['margin_dd'],-0.20,0.0004,5000.0,
                m['entry_delay'],m['entry_tol'],m['rev_mode'],m['min_bars']
            )
            fc,tc=r[0],r[1]; rois=r[3]; pnls=r[4]
            w=np.sum(rois>0) if tc>0 else 0
            pf=np.sum(pnls[pnls>0])/(abs(np.sum(pnls[pnls<0]))+1e-10)
            eq=r[8]; mdd=0
            if len(eq)>0: peq=np.maximum.accumulate(eq); dd=(eq-peq)/(peq+1e-10); mdd=abs(np.min(dd))*100
            vals.append({'run':run+1,'fc':float(fc),'ret':float((fc-5000)/5000*100),
                        'trades':int(tc),'wr':float(w/tc*100) if tc>0 else 0,
                        'pf':float(pf),'mdd':float(mdd)})
        validation[name]=vals
        rets=[v['ret'] for v in vals]; pfs=[v['pf'] for v in vals]; mdds=[v['mdd'] for v in vals]
        print(f"  {name}: Ret={np.mean(rets):.0f}%+/-{np.std(rets):.0f}% PF={np.mean(pfs):.2f}+/-{np.std(pfs):.2f} MDD={np.mean(mdds):.1f}%+/-{np.std(mdds):.1f}% T={np.mean([v['trades'] for v in vals]):.0f}", flush=True)

    # Save
    output = {'version':'v25.4','initial_capital':5000,'margin_mode':'isolated',
              'total_tested':tested,
              'pf3':len(all_results),'pf4':len(pf4),'pf5':len(pf5),
              'models':{n:{k:v for k,v in m.items() if k!='score'} for n,m in models.items()},
              'validation':validation,
              'top20':[{k:v for k,v in r.items() if k!='score'} for r in pool[:20]]}
    def conv(o):
        if isinstance(o,(np.integer,)):return int(o)
        if isinstance(o,(np.floating,)):return float(o)
        if isinstance(o,np.ndarray):return o.tolist()
        return o
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'v25_4_results.json')
    with open(path,'w',encoding='utf-8') as f:
        json.dump(output,f,indent=2,default=conv,ensure_ascii=False)
    print(f"\nSaved: {path}\nTotal: {tested:,}\n{'='*70}\nv25.4 COMPLETE", flush=True)

if __name__=='__main__':
    main()
