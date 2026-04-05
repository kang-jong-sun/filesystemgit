"""
v27.1 Targeted - v25.2 최적 파라미터 고정 + 추가 필터만 최적화
=============================================================
v25.2에서 검증된 ADX/RSI/SL/Trail을 고정하고,
추가 필터(Stochastic, CCI, OBV, Ichimoku, BB) + REV모드 + 진입지연만 최적화
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
    print("v27.1 TARGETED OPTIMIZATION", flush=True)
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
        0.07,0.07,0.03,0,0,10,0.40,0.20,-0.20,0.0004,3000.0,6,1.0,1,6
    )

    stoch_k = cache['5m']['stoch_k']
    stoch_d = cache['5m']['stoch_d']
    cci_5m = cache['5m']['cci']
    obv_5m = cache['5m']['obv_slope']
    bb_w_5m = cache['5m']['bb_width']
    ich_5m = cache['5m']['ichimoku_cloud']
    atr_5m = cache['5m']['atr'][14]

    # v25.2 검증 모델 (ADX/RSI/SL/Trail 고정)
    fixed_models = [
        {'label':'v25.2_A', 'ctf':'30m','ft':'ema','fl':3,'st':'ema','sl':200,
         'atf':'15m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':40,'rmax':75,
         'macd':'5_35_5','sl_v':0.07,'ta':0.07,'tp':0.03,'mn':0.40,'mr':0.20},
        {'label':'v25.2_B', 'ctf':'30m','ft':'ema','fl':7,'st':'ema','sl':100,
         'atf':'15m','ap':20,'amin':35,'rtf':'10m','rp':14,'rmin':40,'rmax':75,
         'macd':'12_26_9','sl_v':0.07,'ta':0.06,'tp':0.03,'mn':0.40,'mr':0.20},
        {'label':'v25.2_C', 'ctf':'10m','ft':'ema','fl':5,'st':'ema','sl':300,
         'atf':'15m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':40,'rmax':75,
         'macd':'5_35_5','sl_v':0.07,'ta':0.08,'tp':0.03,'mn':0.40,'mr':0.20},
        # Margin variants
        {'label':'v25.2_A_m30', 'ctf':'30m','ft':'ema','fl':3,'st':'ema','sl':200,
         'atf':'15m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':40,'rmax':75,
         'macd':'5_35_5','sl_v':0.07,'ta':0.07,'tp':0.03,'mn':0.30,'mr':0.15},
        {'label':'v25.2_A_m20', 'ctf':'30m','ft':'ema','fl':3,'st':'ema','sl':200,
         'atf':'15m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':40,'rmax':75,
         'macd':'5_35_5','sl_v':0.07,'ta':0.07,'tp':0.03,'mn':0.20,'mr':0.10},
        # SL/Trail variants on best
        {'label':'v25.2_A_sl5', 'ctf':'30m','ft':'ema','fl':3,'st':'ema','sl':200,
         'atf':'15m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':40,'rmax':75,
         'macd':'5_35_5','sl_v':0.05,'ta':0.06,'tp':0.03,'mn':0.40,'mr':0.20},
        {'label':'v25.2_A_t10', 'ctf':'30m','ft':'ema','fl':3,'st':'ema','sl':200,
         'atf':'15m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':40,'rmax':75,
         'macd':'5_35_5','sl_v':0.07,'ta':0.10,'tp':0.04,'mn':0.40,'mr':0.20},
        {'label':'v25.2_C_t10', 'ctf':'10m','ft':'ema','fl':5,'st':'ema','sl':300,
         'atf':'15m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':40,'rmax':75,
         'macd':'5_35_5','sl_v':0.07,'ta':0.10,'tp':0.04,'mn':0.40,'mr':0.20},
    ]

    # Build cross signals
    cross_sigs = {}
    adx_m = {}; rsi_m = {}
    for fm in fixed_models:
        ctf = fm['ctf']
        c_data = cache[ctf]
        ft = fm['ft']; fl = fm['fl']; st_t = fm['st']; sl_l = fm['sl']
        fa = c_data.get(ft, c_data['ema']).get(fl) if isinstance(c_data.get(ft), dict) else c_data['ema'].get(fl)
        sa = c_data.get(st_t, c_data['ema']).get(sl_l) if isinstance(c_data.get(st_t), dict) else c_data['ema'].get(sl_l)
        if fa is None or sa is None: continue
        sig = np.zeros(len(fa), dtype=np.int64)
        for j in range(1, len(fa)):
            if fa[j] > sa[j]: sig[j] = 1
            elif fa[j] < sa[j]: sig[j] = -1
        cross_sigs[fm['label']] = sig if ctf == '5m' else sig[tf_maps[ctf]]

        ak = f"{fm['atf']}_{fm['ap']}"
        if ak not in adx_m:
            v = cache[fm['atf']]['adx'][fm['ap']]
            adx_m[ak] = v if fm['atf'] == '5m' else v[tf_maps[fm['atf']]]
        rk = f"{fm['rtf']}_{fm['rp']}"
        if rk not in rsi_m:
            v = cache[fm['rtf']]['rsi'][fm['rp']]
            rsi_m[rk] = v if fm['rtf'] == '5m' else v[tf_maps[fm['rtf']]]

    macd_m = {mk: cache['5m']['macd'][mk]['hist'] for mk in ['5_35_5','8_21_5','12_26_9']}

    # Only optimize: extra filters + REV + entry + min_bars
    stoch_cfgs = [(0, 100), (20, 80), (25, 75), (30, 70)]
    cci_cfgs = [(-300, 300), (-200, 200), (-150, 150), (-100, 100)]
    obv_ts = [0, 1, 3, 5]
    ich_opts = [0, 1]
    bb_opts = [(0, 0), (1, 0.02), (1, 0.03), (1, 0.04)]
    entry_delays = [0, 3, 6, 12]
    entry_tols = [0.5, 1.0, 1.5, 2.0]
    rev_modes = [0, 1, 2]
    min_bars_opts = [1, 6, 12, 24, 48]
    partial_cfgs = [(0, 0), (0.30, 0.10), (0.30, 0.15), (0.50, 0.10)]

    per = (len(stoch_cfgs)*len(cci_cfgs)*len(obv_ts)*len(ich_opts)*len(bb_opts)*
           len(entry_delays)*len(entry_tols)*len(rev_modes)*len(min_bars_opts)*len(partial_cfgs))
    total = len(fixed_models) * per
    print(f"\n  {len(fixed_models)} models x {per:,} filter combos = {total:,}", flush=True)
    print(f"  Est time at ~350/s: {total/350/60:.0f} min", flush=True)

    all_results = []
    tested = 0
    t0 = time.time()

    for fm in fixed_models:
        if fm['label'] not in cross_sigs: continue
        csig = cross_sigs[fm['label']]
        adx_v = adx_m[f"{fm['atf']}_{fm['ap']}"]
        rsi_v = rsi_m[f"{fm['rtf']}_{fm['rp']}"]
        macd_v = macd_m[fm['macd']]

        for so, sob in stoch_cfgs:
            for cmin, cmax in cci_cfgs:
                for obv_t in obv_ts:
                    for ich in ich_opts:
                        for bb_use, bb_min in bb_opts:
                            for ed in entry_delays:
                                for et in entry_tols:
                                    for rm in rev_modes:
                                        for mb in min_bars_opts:
                                            for pe_r, pe_roi in partial_cfgs:
                                                tested += 1
                                                result = run_backtest_v271(
                                                    close_5m, high_5m, low_5m, ts_i64,
                                                    csig, adx_v, rsi_v, atr_5m, macd_v,
                                                    stoch_k, stoch_d, cci_5m, obv_5m, bb_w_5m, ich_5m,
                                                    fm['amin'], fm['rmin'], fm['rmax'],
                                                    so, sob, cmin, cmax, obv_t, ich, bb_use, bb_min,
                                                    fm['sl_v'], fm['ta'], fm['tp'], pe_r, pe_roi,
                                                    10, fm['mn'], fm['mr'], -0.20, 0.0004, 3000.0,
                                                    ed, et, rm, mb
                                                )
                                                fc, tc = result[0], result[1]
                                                if tc >= 8 and fc > 4000:
                                                    rois = result[3]; pnls = result[4]
                                                    wins = np.sum(rois > 0)
                                                    tpro = np.sum(pnls[pnls > 0])
                                                    tlos = abs(np.sum(pnls[pnls < 0])) + 1e-10
                                                    pf = tpro / tlos
                                                    ret = (fc - 3000) / 3000 * 100
                                                    eq = result[8]; mdd = 0
                                                    if len(eq) > 0:
                                                        peq = np.maximum.accumulate(eq)
                                                        dd = (eq - peq) / (peq + 1e-10)
                                                        mdd = abs(np.min(dd)) * 100
                                                    etypes = result[6]
                                                    if pf >= 2.5:
                                                        all_results.append({
                                                            'base': fm['label'],
                                                            'stoch_os':so,'stoch_ob':sob,'cci_min':cmin,'cci_max':cmax,
                                                            'obv_t':obv_t,'ichimoku':ich,'bb_use':bb_use,'bb_min':bb_min,
                                                            'entry_delay':ed,'entry_tol':et,'rev_mode':rm,'min_bars':mb,
                                                            'partial_ratio':pe_r,'partial_roi':pe_roi,
                                                            'sl':fm['sl_v'],'trail_act':fm['ta'],'trail_pct':fm['tp'],
                                                            'margin':fm['mn'],'margin_dd':fm['mr'],
                                                            'adx_min':fm['amin'],'rsi_min':fm['rmin'],'rsi_max':fm['rmax'],
                                                            'final_cap':float(fc),'trades':int(tc),
                                                            'win_rate':float(wins/tc*100),'pf':float(pf),
                                                            'mdd':float(mdd),'return_pct':float(ret),
                                                            'avg_win':float(np.mean(rois[rois>0])*100) if wins>0 else 0,
                                                            'avg_loss':float(np.mean(rois[rois<=0])*100) if (tc-wins)>0 else 0,
                                                            'sl_c':int(np.sum(etypes==0)),'tsl_c':int(np.sum(etypes==1)),'rev_c':int(np.sum(etypes==2)),
                                                        })
                                                if tested % 100000 == 0:
                                                    elapsed = time.time() - t0
                                                    rate = tested / elapsed
                                                    rem = (total - tested) / rate / 60 if rate > 0 else 999
                                                    pf5 = len([r for r in all_results if r['pf'] >= 5.0])
                                                    print(f"  {tested:,}/{total:,} ({tested/total*100:.1f}%) | {rate:.0f}/s | ~{rem:.0f}min | PF>=2.5:{len(all_results)} PF>=5:{pf5}", flush=True)

    elapsed = time.time() - t0
    pf5 = [r for r in all_results if r['pf'] >= 5.0]
    pf4 = [r for r in all_results if r['pf'] >= 4.0]
    print(f"\n  Done: {tested:,} in {elapsed:.0f}s | PF>=2.5:{len(all_results)} PF>=4:{len(pf4)} PF>=5:{len(pf5)}", flush=True)

    for r in all_results:
        r['score'] = r['pf'] * np.sqrt(r['trades']) * r['return_pct'] / (r['mdd'] + 5)
    all_results.sort(key=lambda x: x['score'], reverse=True)
    pool = pf5 if len(pf5) >= 3 else (pf4 if len(pf4) >= 3 else all_results)
    for r in pool:
        if 'score' not in r:
            r['score'] = r['pf'] * np.sqrt(r['trades']) * r['return_pct'] / (r['mdd'] + 5)
    pool.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n  Top 20:", flush=True)
    for i, r in enumerate(pool[:20]):
        print(f"  #{i+1}: {r['base']} Stoch:{r['stoch_os']}/{r['stoch_ob']} CCI:{r['cci_min']}/{r['cci_max']} "
              f"OBV:{r['obv_t']} ICH:{r['ichimoku']} BB:{r['bb_use']} "
              f"ED:{r['entry_delay']} ET:{r['entry_tol']} REV:{r['rev_mode']} MB:{r['min_bars']} "
              f"PE:{r['partial_ratio']}/{r['partial_roi']} "
              f"| Ret:{r['return_pct']:.0f}% PF:{r['pf']:.2f} MDD:{r['mdd']:.1f}% T:{r['trades']} "
              f"SL:{r['sl_c']} TSL:{r['tsl_c']} REV:{r['rev_c']}", flush=True)

    # Models
    model_a = max(pool[:50], key=lambda x: x['return_pct'] * x['pf'])
    model_b = max(pool[:50], key=lambda x: x['pf'] / (x['mdd'] + 3))
    model_c = max(pool[:50], key=lambda x: x['trades'] * x['pf'] * np.log(max(x['return_pct'],1)+1) / (x['mdd']+5))
    models = {'A': model_a, 'B': model_b, 'C': model_c}

    for name, m in models.items():
        print(f"\n  Model {name}: {m['base']} | Ret:{m['return_pct']:.0f}% PF:{m['pf']:.2f} MDD:{m['mdd']:.1f}% T:{m['trades']}", flush=True)
        print(f"    WR:{m['win_rate']:.1f}% AvgW:{m['avg_win']:.1f}% AvgL:{m['avg_loss']:.1f}% Final:${m['final_cap']:,.0f}", flush=True)

    # 30x validation
    print("\n[30x Validation]", flush=True)
    validation = {}
    for name, m in models.items():
        csig = cross_sigs[m['base']]
        bc = [f for f in fixed_models if f['label']==m['base']][0]
        adx_v = adx_m[f"{bc['atf']}_{bc['ap']}"]
        rsi_v = rsi_m[f"{bc['rtf']}_{bc['rp']}"]
        macd_v = macd_m[bc['macd']]
        vals = []
        for run in range(30):
            s = int(n * (0.02 + (run/30)*0.13))
            ml = min(n-s, len(csig)-s, len(adx_v)-s, len(rsi_v)-s)
            r = run_backtest_v271(
                close_5m[s:s+ml], high_5m[s:s+ml], low_5m[s:s+ml], ts_i64[s:s+ml],
                csig[s:s+ml], adx_v[s:s+ml], rsi_v[s:s+ml], atr_5m[s:s+ml], macd_v[s:s+ml],
                stoch_k[s:s+ml], stoch_d[s:s+ml], cci_5m[s:s+ml], obv_5m[s:s+ml],
                bb_w_5m[s:s+ml], ich_5m[s:s+ml],
                m['adx_min'],m['rsi_min'],m['rsi_max'],
                m['stoch_os'],m['stoch_ob'],m['cci_min'],m['cci_max'],
                m['obv_t'],m['ichimoku'],m['bb_use'],m['bb_min'],
                m['sl'],m['trail_act'],m['trail_pct'],m['partial_ratio'],m['partial_roi'],
                10,m['margin'],m['margin_dd'],-0.20,0.0004,3000.0,
                m['entry_delay'],m['entry_tol'],m['rev_mode'],m['min_bars']
            )
            fc,tc = r[0],r[1]; rois=r[3]; pnls=r[4]
            w = np.sum(rois>0) if tc>0 else 0
            pf = np.sum(pnls[pnls>0])/(abs(np.sum(pnls[pnls<0]))+1e-10)
            eq=r[8]; mdd=0
            if len(eq)>0: peq=np.maximum.accumulate(eq); dd=(eq-peq)/(peq+1e-10); mdd=abs(np.min(dd))*100
            vals.append({'run':run+1,'fc':float(fc),'ret':float((fc-3000)/3000*100),'trades':int(tc),
                         'wr':float(w/tc*100) if tc>0 else 0,'pf':float(pf),'mdd':float(mdd)})
        validation[name] = vals
        rets=[v['ret'] for v in vals]; pfs=[v['pf'] for v in vals]; mdds=[v['mdd'] for v in vals]
        print(f"  {name}: Ret={np.mean(rets):.0f}%+/-{np.std(rets):.0f}% PF={np.mean(pfs):.2f}+/-{np.std(pfs):.2f} MDD={np.mean(mdds):.1f}%+/-{np.std(mdds):.1f}%", flush=True)

    # Save
    output = {'version':'v27.1','total_tested':tested,
              'pf25':len(all_results),'pf4':len(pf4),'pf5':len(pf5),
              'models':{n:{k:v for k,v in m.items() if k!='score'} for n,m in models.items()},
              'validation':validation,
              'top20':[{k:v for k,v in r.items() if k!='score'} for r in pool[:20]]}
    def conv(o):
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,(np.floating,)): return float(o)
        if isinstance(o,np.ndarray): return o.tolist()
        return o
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v27_1_results.json')
    with open(path,'w',encoding='utf-8') as f:
        json.dump(output,f,indent=2,default=conv,ensure_ascii=False)
    print(f"\nSaved: {path}\nTotal: {tested:,}\n{'='*70}\nv27.1 COMPLETE", flush=True)

if __name__ == '__main__':
    main()
