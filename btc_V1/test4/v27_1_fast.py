"""
v27.1 Fast - v25.2 검증 모델 기반 + 추가 필터 최적화
=====================================================
전략: v25.2의 PF 10+ 모델들을 기반으로
- 추가 인디케이터(Stochastic, CCI, OBV, Ichimoku, BB) 적용
- REV 모드 / 진입 지연 / 최소 거래 간격 최적화
- 30회 검증 + 월별 상세 데이터
"""
import pandas as pd
import numpy as np
from numba import njit
import time, os, json, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

from v27_backtest_engine import (
    load_5m_data, resample_ohlcv, map_tf_index,
    calc_ema, calc_sma, calc_hma,
    calc_rsi, calc_adx, calc_atr, calc_macd, calc_bollinger
)
from v27_1_engine import (
    calc_stochastic, calc_cci, calc_obv_slope, calc_ichimoku_cloud,
    build_extended_cache, run_backtest_v271
)


def main():
    print("="*70, flush=True)
    print("v27.1 FAST - Proven Models + Extra Filters", flush=True)
    print("="*70, flush=True)

    df_5m = load_5m_data()
    print("\nBuilding extended cache...", flush=True)
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
        np.zeros(nw), np.zeros(nw), np.zeros(nw), np.zeros(nw),
        np.zeros(nw), np.zeros(nw), np.zeros(nw), np.zeros(nw), np.zeros(nw), np.zeros(nw),
        25, 30, 70, 20, 80, -200, 200, 0, 0, 0, 0.01,
        0.07, 0.07, 0.03, 0, 0,
        10, 0.40, 0.20, -0.20, 0.0004, 3000.0, 6, 1.0, 1, 6
    )

    # Pre-map all indicators to 5m
    stoch_k = cache['5m']['stoch_k']
    stoch_d = cache['5m']['stoch_d']
    cci_5m = cache['5m']['cci']
    obv_5m = cache['5m']['obv_slope']
    bb_w_5m = cache['5m']['bb_width']
    ich_5m = cache['5m']['ichimoku_cloud']
    atr_5m = cache['5m']['atr'][14]

    # Base configs from v25.2 proven + v27 top
    base_configs = [
        # v25.2 Model A: 30m EMA3/EMA200, ADX 15m>35, RSI 5m 40-75
        {'label': 'v25.2_A', 'cross_tf':'30m','ft':'ema','fl':3,'st':'ema','sl':200,
         'adx_tf':'15m','adx_p':20,'rsi_tf':'5m','rsi_p':14, 'macd':'5_35_5'},
        # v25.2 Model B: 30m EMA7/EMA100, ADX 15m>35, RSI 10m 40-75
        {'label': 'v25.2_B', 'cross_tf':'30m','ft':'ema','fl':7,'st':'ema','sl':100,
         'adx_tf':'15m','adx_p':20,'rsi_tf':'10m','rsi_p':14, 'macd':'12_26_9'},
        # v25.2 Model C: 10m EMA5/EMA300, ADX 15m>35, RSI 5m 40-75
        {'label': 'v25.2_C', 'cross_tf':'10m','ft':'ema','fl':5,'st':'ema','sl':300,
         'adx_tf':'15m','adx_p':20,'rsi_tf':'5m','rsi_p':14, 'macd':'5_35_5'},
        # v27 top: 30m EMA5/SMA50, ADX 5m>35, RSI 5m 40-80
        {'label': 'v27_A', 'cross_tf':'30m','ft':'ema','fl':5,'st':'sma','sl':50,
         'adx_tf':'5m','adx_p':14,'rsi_tf':'5m','rsi_p':14, 'macd':'12_26_9'},
        # Additional promising crosses
        {'label': 'ema3_150', 'cross_tf':'30m','ft':'ema','fl':3,'st':'ema','sl':150,
         'adx_tf':'15m','adx_p':20,'rsi_tf':'5m','rsi_p':14, 'macd':'5_35_5'},
        {'label': 'ema5_200', 'cross_tf':'30m','ft':'ema','fl':5,'st':'ema','sl':200,
         'adx_tf':'15m','adx_p':20,'rsi_tf':'5m','rsi_p':14, 'macd':'8_21_5'},
        {'label': 'hma9_200', 'cross_tf':'30m','ft':'hma','fl':9,'st':'ema','sl':200,
         'adx_tf':'15m','adx_p':20,'rsi_tf':'5m','rsi_p':14, 'macd':'12_26_9'},
        {'label': 'ema3_200_15m', 'cross_tf':'15m','ft':'ema','fl':3,'st':'ema','sl':200,
         'adx_tf':'30m','adx_p':20,'rsi_tf':'5m','rsi_p':14, 'macd':'5_35_5'},
    ]

    # Build cross signals
    cross_sigs = {}
    for bc in base_configs:
        ctf = bc['cross_tf']
        c_data = cache[ctf]
        ft = bc['ft']; fl = bc['fl']; st = bc['st']; sl_len = bc['sl']

        fm = c_data.get(ft, c_data['ema']).get(fl) if isinstance(c_data.get(ft), dict) else c_data['ema'].get(fl)
        sm = c_data.get(st, c_data['ema']).get(sl_len) if isinstance(c_data.get(st), dict) else c_data['ema'].get(sl_len)

        if fm is None or sm is None:
            print(f"  WARNING: {bc['label']} - MA not found, skipping", flush=True)
            continue

        sig = np.zeros(len(fm), dtype=np.int64)
        for j in range(1, len(fm)):
            if fm[j] > sm[j]: sig[j] = 1
            elif fm[j] < sm[j]: sig[j] = -1
        cross_sigs[bc['label']] = sig if ctf == '5m' else sig[tf_maps[ctf]]

    # Pre-map ADX/RSI
    adx_m = {}
    rsi_m = {}
    for bc in base_configs:
        ak = f"{bc['adx_tf']}_{bc['adx_p']}"
        if ak not in adx_m:
            v = cache[bc['adx_tf']]['adx'][bc['adx_p']]
            adx_m[ak] = v if bc['adx_tf'] == '5m' else v[tf_maps[bc['adx_tf']]]
        rk = f"{bc['rsi_tf']}_{bc['rsi_p']}"
        if rk not in rsi_m:
            v = cache[bc['rsi_tf']]['rsi'][bc['rsi_p']]
            rsi_m[rk] = v if bc['rsi_tf'] == '5m' else v[tf_maps[bc['rsi_tf']]]

    macd_m = {mk: cache['5m']['macd'][mk]['hist'] for mk in ['5_35_5','8_21_5','12_26_9']}

    # Optimization grid
    adx_mins = [25, 30, 35, 40]
    rsi_ranges = [(30, 70), (35, 70), (35, 75), (40, 75), (40, 80)]
    stoch_cfgs = [(0, 100), (20, 80), (25, 75), (30, 70)]  # (0,100) = disabled
    cci_cfgs = [(-300, 300), (-200, 200), (-150, 150), (-100, 100)]
    obv_ts = [0, 1, 3, 5]
    ich_opts = [0, 1]
    bb_opts = [(0, 0), (1, 0.02), (1, 0.03)]

    sl_trails = [
        (0.05, 0.05, 0.03), (0.05, 0.06, 0.03), (0.05, 0.07, 0.03),
        (0.07, 0.06, 0.03), (0.07, 0.07, 0.03), (0.07, 0.08, 0.03),
        (0.07, 0.07, 0.04), (0.07, 0.08, 0.04),
        (0.07, 0.10, 0.05), (0.10, 0.10, 0.04), (0.10, 0.12, 0.05),
    ]
    margins = [(0.20, 0.10), (0.30, 0.15), (0.40, 0.20)]
    entry_delays = [0, 6, 12]
    entry_tols = [0.5, 1.0, 2.0]
    rev_modes = [0, 1]
    min_bars = [6, 12, 24, 48]
    partial_cfgs = [(0, 0), (0.30, 0.10), (0.30, 0.15)]

    per_base = (len(adx_mins) * len(rsi_ranges) * len(stoch_cfgs) * len(cci_cfgs) *
                len(obv_ts) * len(ich_opts) * len(bb_opts) * len(sl_trails) *
                len(margins) * len(entry_delays) * len(entry_tols) *
                len(rev_modes) * len(min_bars) * len(partial_cfgs))
    total = len(base_configs) * per_base
    print(f"\n  {len(base_configs)} base x {per_base:,} combos = {total:,} total", flush=True)

    all_results = []
    tested = 0
    t0 = time.time()

    for bc in base_configs:
        if bc['label'] not in cross_sigs:
            continue
        csig = cross_sigs[bc['label']]
        ak = f"{bc['adx_tf']}_{bc['adx_p']}"
        rk = f"{bc['rsi_tf']}_{bc['rsi_p']}"
        adx_v = adx_m[ak]
        rsi_v = rsi_m[rk]
        macd_v = macd_m[bc['macd']]

        print(f"\n  Testing {bc['label']}...", flush=True)

        for amin in adx_mins:
            for rmin, rmax in rsi_ranges:
                for so, sob in stoch_cfgs:
                    for cmin, cmax in cci_cfgs:
                        for obv_t in obv_ts:
                            for ich in ich_opts:
                                for bb_use, bb_min in bb_opts:
                                    for sl_v, ta, tp in sl_trails:
                                        for mn, mr in margins:
                                            for ed in entry_delays:
                                                for et in entry_tols:
                                                    for rm in rev_modes:
                                                        for mb in min_bars:
                                                            for pe_r, pe_roi in partial_cfgs:
                                                                tested += 1

                                                                result = run_backtest_v271(
                                                                    close_5m, high_5m, low_5m, ts_i64,
                                                                    csig, adx_v, rsi_v, atr_5m, macd_v,
                                                                    stoch_k, stoch_d, cci_5m, obv_5m, bb_w_5m, ich_5m,
                                                                    amin, rmin, rmax, so, sob, cmin, cmax,
                                                                    obv_t, ich, bb_use, bb_min,
                                                                    sl_v, ta, tp, pe_r, pe_roi,
                                                                    10, mn, mr, -0.20, 0.0004, 3000.0,
                                                                    ed, et, rm, mb
                                                                )

                                                                fc, tc = result[0], result[1]
                                                                if tc >= 10 and fc > 5000:
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

                                                                    if pf >= 3.0:
                                                                        all_results.append({
                                                                            'base': bc['label'],
                                                                            'adx_min': amin, 'rsi_min': rmin, 'rsi_max': rmax,
                                                                            'stoch_os': so, 'stoch_ob': sob,
                                                                            'cci_min': cmin, 'cci_max': cmax,
                                                                            'obv_t': obv_t, 'ichimoku': ich,
                                                                            'bb_use': bb_use, 'bb_min': bb_min,
                                                                            'sl': sl_v, 'trail_act': ta, 'trail_pct': tp,
                                                                            'margin': mn, 'margin_dd': mr,
                                                                            'entry_delay': ed, 'entry_tol': et,
                                                                            'rev_mode': rm, 'min_bars': mb,
                                                                            'partial_ratio': pe_r, 'partial_roi': pe_roi,
                                                                            'final_cap': float(fc), 'trades': int(tc),
                                                                            'win_rate': float(wins/tc*100),
                                                                            'pf': float(pf), 'mdd': float(mdd),
                                                                            'return_pct': float(ret),
                                                                            'avg_win': float(np.mean(rois[rois>0])*100) if wins>0 else 0,
                                                                            'avg_loss': float(np.mean(rois[rois<=0])*100) if (tc-wins)>0 else 0,
                                                                            'sl_c': int(np.sum(etypes==0)),
                                                                            'tsl_c': int(np.sum(etypes==1)),
                                                                            'rev_c': int(np.sum(etypes==2)),
                                                                        })

                                                                if tested % 200000 == 0:
                                                                    elapsed = time.time() - t0
                                                                    rate = tested / elapsed
                                                                    rem = (total - tested) / rate / 60 if rate > 0 else 999
                                                                    pf5 = len([r for r in all_results if r['pf'] >= 5.0])
                                                                    print(f"    {tested:,}/{total:,} ({tested/total*100:.1f}%) "
                                                                          f"| {rate:.0f}/s | ~{rem:.0f}min "
                                                                          f"| PF>=3:{len(all_results)} PF>=5:{pf5}", flush=True)

        elapsed = time.time() - t0
        pf5 = len([r for r in all_results if r['pf'] >= 5.0])
        print(f"    {bc['label']} done | Total tested: {tested:,} | PF>=3:{len(all_results)} PF>=5:{pf5}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Complete: {tested:,} in {elapsed:.0f}s ({tested/elapsed:.0f}/s)", flush=True)

    pf5_results = [r for r in all_results if r['pf'] >= 5.0]
    pf4_results = [r for r in all_results if r['pf'] >= 4.0]
    print(f"  PF>=3: {len(all_results)} | PF>=4: {len(pf4_results)} | PF>=5: {len(pf5_results)}", flush=True)

    for r in all_results:
        r['score'] = r['pf'] * np.sqrt(r['trades']) * r['return_pct'] / (r['mdd'] + 5)
    all_results.sort(key=lambda x: x['score'], reverse=True)

    pool = pf5_results if len(pf5_results) >= 3 else (pf4_results if len(pf4_results) >= 3 else all_results)
    for r in pool:
        r['score'] = r['pf'] * np.sqrt(r['trades']) * r['return_pct'] / (r['mdd'] + 5)
    pool.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n  Top 15:", flush=True)
    for i, r in enumerate(pool[:15]):
        print(f"    #{i+1}: {r['base']} ADX>{r['adx_min']} RSI:{r['rsi_min']}-{r['rsi_max']} "
              f"Stoch:{r['stoch_os']}/{r['stoch_ob']} CCI:{r['cci_min']}/{r['cci_max']} "
              f"OBV:{r['obv_t']} ICH:{r['ichimoku']} BB:{r['bb_use']}/{r['bb_min']} "
              f"SL:{r['sl']*100:.0f}% T+{r['trail_act']*100:.0f}%/-{r['trail_pct']*100:.0f}% M:{r['margin']*100:.0f}% "
              f"REV:{r['rev_mode']} MB:{r['min_bars']} "
              f"| Ret:{r['return_pct']:.0f}% PF:{r['pf']:.2f} MDD:{r['mdd']:.1f}% T:{r['trades']} "
              f"SL:{r['sl_c']} TSL:{r['tsl_c']} REV:{r['rev_c']}", flush=True)

    # Select Models
    model_a = max(pool[:50], key=lambda x: x['return_pct'] * x['pf'])
    model_b = max(pool[:50], key=lambda x: x['pf'] / (x['mdd'] + 3))
    model_c = max(pool[:50], key=lambda x: x['trades'] * x['pf'] * np.log(max(x['return_pct'],1)+1) / (x['mdd']+5))

    models = {'A': model_a, 'B': model_b, 'C': model_c}
    for name, m in models.items():
        print(f"\n  Model {name}: {m['base']} | Ret:{m['return_pct']:.0f}% PF:{m['pf']:.2f} MDD:{m['mdd']:.1f}% T:{m['trades']}", flush=True)

    # 30x validation
    print("\n[30x Validation]", flush=True)
    validation = {}
    for name, m in models.items():
        csig = cross_sigs[m['base']]
        bc = [b for b in base_configs if b['label'] == m['base']][0]
        adx_v = adx_m[f"{bc['adx_tf']}_{bc['adx_p']}"]
        rsi_v = rsi_m[f"{bc['rsi_tf']}_{bc['rsi_p']}"]
        macd_v = macd_m[bc['macd']]

        vals = []
        for run in range(30):
            skip = 0.02 + (run/30)*0.13
            s = int(n * skip)
            ml = min(n-s, len(csig)-s, len(adx_v)-s, len(rsi_v)-s)
            r = run_backtest_v271(
                close_5m[s:s+ml], high_5m[s:s+ml], low_5m[s:s+ml], ts_i64[s:s+ml],
                csig[s:s+ml], adx_v[s:s+ml], rsi_v[s:s+ml], atr_5m[s:s+ml], macd_v[s:s+ml],
                stoch_k[s:s+ml], stoch_d[s:s+ml], cci_5m[s:s+ml], obv_5m[s:s+ml],
                bb_w_5m[s:s+ml], ich_5m[s:s+ml],
                m['adx_min'], m['rsi_min'], m['rsi_max'],
                m['stoch_os'], m['stoch_ob'], m['cci_min'], m['cci_max'],
                m['obv_t'], m['ichimoku'], m['bb_use'], m['bb_min'],
                m['sl'], m['trail_act'], m['trail_pct'],
                m['partial_ratio'], m['partial_roi'],
                10, m['margin'], m['margin_dd'], -0.20, 0.0004, 3000.0,
                m['entry_delay'], m['entry_tol'], m['rev_mode'], m['min_bars']
            )
            fc, tc = r[0], r[1]
            rois = r[3]; pnls = r[4]
            w = np.sum(rois > 0) if tc > 0 else 0
            tp_s = np.sum(pnls[pnls > 0]); tl_s = abs(np.sum(pnls[pnls < 0])) + 1e-10
            pf = tp_s / tl_s
            eq = r[8]; mdd = 0
            if len(eq) > 0:
                peq = np.maximum.accumulate(eq)
                dd = (eq - peq) / (peq + 1e-10)
                mdd = abs(np.min(dd)) * 100
            vals.append({'run':run+1, 'fc':float(fc), 'ret':float((fc-3000)/3000*100),
                         'trades':int(tc), 'wr':float(w/tc*100) if tc>0 else 0,
                         'pf':float(pf), 'mdd':float(mdd)})
        validation[name] = vals
        rets = [v['ret'] for v in vals]
        pfs = [v['pf'] for v in vals]
        mdds = [v['mdd'] for v in vals]
        print(f"  Model {name}: Ret={np.mean(rets):.0f}%+/-{np.std(rets):.0f}% PF={np.mean(pfs):.2f}+/-{np.std(pfs):.2f} MDD={np.mean(mdds):.1f}%", flush=True)

    # Save
    output = {
        'version': 'v27.1', 'total_tested': tested,
        'pf3_count': len(all_results), 'pf4_count': len(pf4_results), 'pf5_count': len(pf5_results),
        'models': {n: {k:v for k,v in m.items() if k != 'score'} for n,m in models.items()},
        'validation': validation,
        'top20': [{k:v for k,v in r.items() if k != 'score'} for r in pool[:20]],
    }
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v27_1_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=conv, ensure_ascii=False)
    print(f"\nSaved: {path}", flush=True)
    print("="*70, flush=True)
    print("v27.1 COMPLETE", flush=True)

if __name__ == '__main__':
    main()
