"""
v27 Phase 2 Focused: Phase 1B Top 결과에 대한 집중 리스크 최적화 + 30회 검증
"""
import pandas as pd
import numpy as np
import time, os, json, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

from v27_backtest_engine import (
    load_5m_data, build_indicator_cache, map_tf_index, run_backtest_core
)

def main():
    print("="*70, flush=True)
    print("v27 Phase 2 Focused: Risk Optimization + 30x Validation", flush=True)
    print("="*70, flush=True)

    # Load
    df_5m = load_5m_data()
    cache = build_indicator_cache(df_5m)

    ts_5m = cache['5m']['timestamp']
    close_5m = cache['5m']['close']
    high_5m = cache['5m']['high']
    low_5m = cache['5m']['low']
    n = len(close_5m)
    ts_i64 = ts_5m.astype('int64')

    tf_maps = {}
    for tf in ['10m', '15m', '30m', '1h']:
        tf_maps[tf] = map_tf_index(ts_5m, cache[tf]['timestamp'])

    # JIT warmup
    nw = 1000
    _ = run_backtest_core(
        close_5m[:nw], high_5m[:nw], low_5m[:nw], ts_i64[:nw],
        np.zeros(nw, dtype=np.int64), np.zeros(nw), np.zeros(nw),
        np.zeros(nw), np.zeros(nw),
        25, 30, 70, 0.07, 0.07, 0.03, 0.0, 0.0,
        10, 0.20, 0.10, -0.20, 0.0004, 3000.0, 6, 1.0,
        0, 2.0, 0, 2.0, np.array([1.]*7)
    )

    # Phase 1B Top Winners (from previous run)
    top_configs = [
        {'cross_tf':'5m','fast':'ema5','slow':'ema150','adx_tf':'30m','adx_p':20,'adx_min':40,
         'rsi_tf':'10m','rsi_p':14,'rsi_min':30,'rsi_max':70,'macd':'5_35_5','entry_delay':6,'entry_tol':1.5},
        {'cross_tf':'5m','fast':'ema5','slow':'ema150','adx_tf':'30m','adx_p':20,'adx_min':40,
         'rsi_tf':'5m','rsi_p':14,'rsi_min':40,'rsi_max':80,'macd':'8_21_5','entry_delay':6,'entry_tol':1.5},
        {'cross_tf':'30m','fast':'ema5','slow':'sma50','adx_tf':'5m','adx_p':14,'adx_min':35,
         'rsi_tf':'5m','rsi_p':14,'rsi_min':40,'rsi_max':80,'macd':'12_26_9','entry_delay':12,'entry_tol':0.5},
        {'cross_tf':'5m','fast':'ema5','slow':'ema150','adx_tf':'30m','adx_p':20,'adx_min':40,
         'rsi_tf':'10m','rsi_p':7,'rsi_min':35,'rsi_max':75,'macd':'5_35_5','entry_delay':6,'entry_tol':1.5},
        {'cross_tf':'15m','fast':'ema7','slow':'ema50','adx_tf':'30m','adx_p':20,'adx_min':40,
         'rsi_tf':'10m','rsi_p':14,'rsi_min':30,'rsi_max':70,'macd':'5_35_5','entry_delay':6,'entry_tol':1.5},
        {'cross_tf':'30m','fast':'ema3','slow':'ema200','adx_tf':'15m','adx_p':20,'adx_min':35,
         'rsi_tf':'5m','rsi_p':14,'rsi_min':40,'rsi_max':75,'macd':'5_35_5','entry_delay':6,'entry_tol':1.0},
        {'cross_tf':'10m','fast':'ema5','slow':'ema200','adx_tf':'30m','adx_p':20,'adx_min':35,
         'rsi_tf':'10m','rsi_p':14,'rsi_min':30,'rsi_max':70,'macd':'8_21_5','entry_delay':6,'entry_tol':1.5},
        {'cross_tf':'30m','fast':'hma14','slow':'ema200','adx_tf':'15m','adx_p':20,'adx_min':35,
         'rsi_tf':'5m','rsi_p':14,'rsi_min':40,'rsi_max':75,'macd':'12_26_9','entry_delay':6,'entry_tol':1.0},
    ]

    def build_signals(cfg):
        ctf = cfg['cross_tf']
        ft_str = cfg['fast']; st_str = cfg['slow']
        for ma_t in ['ema','sma','hma','wma']:
            if ft_str.startswith(ma_t): ft=ma_t; fl=int(ft_str[len(ma_t):]); break
        for ma_t in ['ema','sma','hma','wma']:
            if st_str.startswith(ma_t): st=ma_t; sl=int(st_str[len(ma_t):]); break

        c_data = cache[ctf]
        fm = c_data[ft][fl] if ft in c_data and fl in c_data[ft] else c_data['ema'][fl]
        sm = c_data[st][sl] if st in c_data and sl in c_data[st] else c_data['ema'][sl]

        sig = np.zeros(len(fm), dtype=np.int64)
        for j in range(1, len(fm)):
            if fm[j] > sm[j]: sig[j] = 1
            elif fm[j] < sm[j]: sig[j] = -1

        csig = sig if ctf == '5m' else sig[tf_maps[ctf]]

        atf = cfg['adx_tf']
        adx_v = cache[atf]['adx'][cfg['adx_p']]
        if atf != '5m': adx_v = adx_v[tf_maps[atf]]

        rtf = cfg['rsi_tf']
        rsi_v = cache[rtf]['rsi'][cfg['rsi_p']]
        if rtf != '5m': rsi_v = rsi_v[tf_maps[rtf]]

        macd_v = cache['5m']['macd'][cfg['macd']]['hist']
        atr_v = cache['5m']['atr'][14]

        return csig, adx_v, rsi_v, atr_v, macd_v

    # ================================================================
    # PHASE 2: Risk Optimization on top configs
    # ================================================================
    print("\n[Phase 2] Risk Management Optimization...", flush=True)

    sl_pcts = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]
    trail_acts = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]
    trail_pcts = [0.02, 0.025, 0.03, 0.04, 0.05]
    margins = [(0.20, 0.10), (0.30, 0.15), (0.40, 0.20)]

    per_cfg = len(sl_pcts) * len(trail_acts) * len(trail_pcts) * len(margins)
    total = len(top_configs) * per_cfg
    print(f"  {len(top_configs)} configs x {per_cfg} risk combos = {total:,}", flush=True)

    all_results = []
    tested = 0
    t0 = time.time()

    for ci, cfg in enumerate(top_configs):
        csig, adx_v, rsi_v, atr_v, macd_v = build_signals(cfg)
        print(f"\n  Config {ci+1}/{len(top_configs)}: {cfg['cross_tf']} {cfg['fast']}/{cfg['slow']}...", flush=True)

        for sl in sl_pcts:
            for ta in trail_acts:
                for tp in trail_pcts:
                    if tp >= ta: continue
                    for mn, mr in margins:
                        tested += 1

                        result = run_backtest_core(
                            close_5m, high_5m, low_5m, ts_i64,
                            csig, adx_v, rsi_v, atr_v, macd_v,
                            cfg['adx_min'], cfg['rsi_min'], cfg['rsi_max'],
                            sl, ta, tp, 0.0, 0.0,
                            10, mn, mr, -0.20,
                            0.0004, 3000.0,
                            cfg['entry_delay'], cfg['entry_tol'],
                            0, 2.0, 0, 2.0,
                            np.array([0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0])
                        )

                        fc, tc = result[0], result[1]
                        if tc >= 8 and fc > 5000:
                            rois = result[3]; pnls = result[4]
                            wins = np.sum(rois > 0)
                            tp_sum = np.sum(pnls[pnls > 0])
                            tl_sum = abs(np.sum(pnls[pnls < 0])) + 1e-10
                            pf = tp_sum / tl_sum
                            ret = (fc - 3000) / 3000 * 100

                            eq = result[8]; mdd = 0
                            if len(eq) > 0:
                                peq = np.maximum.accumulate(eq)
                                dd = (eq - peq) / (peq + 1e-10)
                                mdd = abs(np.min(dd)) * 100

                            exit_types = result[6]
                            sl_count = int(np.sum(exit_types == 0))
                            tsl_count = int(np.sum(exit_types == 1))
                            rev_count = int(np.sum(exit_types == 2))

                            score = pf * ret / (mdd + 5) * np.sqrt(tc)

                            all_results.append({
                                **cfg,
                                'sl': sl, 'trail_act': ta, 'trail_pct': tp,
                                'margin': mn, 'margin_dd': mr,
                                'final_cap': float(fc), 'trades': int(tc),
                                'win_rate': float(wins/tc*100),
                                'pf': float(pf), 'mdd': float(mdd), 'return_pct': float(ret),
                                'avg_win': float(np.mean(rois[rois > 0]) * 100) if wins > 0 else 0,
                                'avg_loss': float(np.mean(rois[rois <= 0]) * 100) if (tc-wins) > 0 else 0,
                                'sl_count': sl_count, 'tsl_count': tsl_count, 'rev_count': rev_count,
                                'score': float(score),
                            })

        elapsed = time.time() - t0
        print(f"    Tested: {tested:,} | {elapsed:.0f}s | {len(all_results)} valid results", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Phase 2 complete: {tested:,} tested in {elapsed:.0f}s -> {len(all_results)} valid", flush=True)

    all_results.sort(key=lambda x: x['score'], reverse=True)

    # ================================================================
    # SELECT TOP 3 MODELS
    # ================================================================
    print("\n[Model Selection]", flush=True)

    # Model A: 수익극대형 (최고 수익률 + 거래빈도)
    model_a = max(all_results[:100], key=lambda x: x['return_pct'] * np.sqrt(x['trades']))
    # Model B: 안정형 (최고 PF / MDD)
    model_b = max(all_results[:100], key=lambda x: x['pf'] / (x['mdd'] + 3))
    # Model C: 균형형 (거래빈도 x PF x 수익)
    model_c = max(all_results[:100], key=lambda x: x['trades'] * x['pf'] * np.log(max(x['return_pct'],1)+1) / (x['mdd']+5))

    models = {'A': model_a, 'B': model_b, 'C': model_c}

    for name, m in models.items():
        print(f"\n  Model {name}: {m['cross_tf']} {m['fast']}/{m['slow']}", flush=True)
        print(f"    ADX: {m['adx_tf']} min={m['adx_min']} | RSI: {m['rsi_tf']} {m['rsi_min']}-{m['rsi_max']}", flush=True)
        print(f"    SL: {m['sl']*100:.0f}% | Trail: +{m['trail_act']*100:.0f}%/-{m['trail_pct']*100:.0f}%", flush=True)
        print(f"    Margin: {m['margin']*100:.0f}% | DD Margin: {m['margin_dd']*100:.0f}%", flush=True)
        print(f"    Return: {m['return_pct']:.0f}% | PF: {m['pf']:.2f} | MDD: {m['mdd']:.1f}%", flush=True)
        print(f"    Trades: {m['trades']} | WR: {m['win_rate']:.1f}% | SL:{m['sl_count']} TSL:{m['tsl_count']} REV:{m['rev_count']}", flush=True)
        print(f"    Avg Win: {m['avg_win']:.1f}% | Avg Loss: {m['avg_loss']:.1f}%", flush=True)
        print(f"    Final Cap: ${m['final_cap']:,.0f}", flush=True)

    # ================================================================
    # 30x VALIDATION
    # ================================================================
    print("\n\n[30x Validation]", flush=True)

    validation = {}
    for name, m in models.items():
        print(f"\n  Validating Model {name}...", flush=True)
        csig, adx_v, rsi_v, atr_v, macd_v = build_signals(m)

        val_results = []
        for run in range(30):
            skip = 0.02 + (run / 30) * 0.13
            start = int(n * skip)

            cs_s = csig[start:]
            adx_s = adx_v[start:] if len(adx_v) > start else adx_v
            rsi_s = rsi_v[start:] if len(rsi_v) > start else rsi_v
            atr_s = atr_v[start:]
            macd_s = macd_v[start:]
            close_s = close_5m[start:]
            high_s = high_5m[start:]
            low_s = low_5m[start:]
            ts_s = ts_i64[start:]

            ml = min(len(close_s), len(cs_s), len(adx_s), len(rsi_s), len(atr_s), len(macd_s))

            result = run_backtest_core(
                close_s[:ml], high_s[:ml], low_s[:ml], ts_s[:ml],
                cs_s[:ml], adx_s[:ml], rsi_s[:ml], atr_s[:ml], macd_s[:ml],
                m['adx_min'], m['rsi_min'], m['rsi_max'],
                m['sl'], m['trail_act'], m['trail_pct'], 0.0, 0.0,
                10, m['margin'], m['margin_dd'], -0.20,
                0.0004, 3000.0, m['entry_delay'], m['entry_tol'],
                0, 2.0, 0, 2.0,
                np.array([0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0])
            )

            fc, tc = result[0], result[1]
            rois = result[3]; pnls = result[4]
            wins = np.sum(rois > 0) if tc > 0 else 0
            tp_sum = np.sum(pnls[pnls > 0])
            tl_sum = abs(np.sum(pnls[pnls < 0])) + 1e-10
            pf = tp_sum / tl_sum
            eq = result[8]; mdd = 0
            if len(eq) > 0:
                peq = np.maximum.accumulate(eq)
                dd = (eq - peq) / (peq + 1e-10)
                mdd = abs(np.min(dd)) * 100

            val_results.append({
                'run': run+1, 'skip': f"{skip*100:.1f}%",
                'final_cap': float(fc), 'return_pct': float((fc-3000)/3000*100),
                'trades': int(tc), 'win_rate': float(wins/tc*100) if tc > 0 else 0,
                'pf': float(pf), 'mdd': float(mdd),
            })

        validation[name] = val_results
        rets = [v['return_pct'] for v in val_results]
        pfs = [v['pf'] for v in val_results]
        mdds = [v['mdd'] for v in val_results]
        print(f"    Return: {np.mean(rets):.0f}% ± {np.std(rets):.0f}% (min:{np.min(rets):.0f}% max:{np.max(rets):.0f}%)", flush=True)
        print(f"    PF: {np.mean(pfs):.2f} ± {np.std(pfs):.2f} (min:{np.min(pfs):.2f})", flush=True)
        print(f"    MDD: {np.mean(mdds):.1f}% ± {np.std(mdds):.1f}%", flush=True)
        print(f"    Trades: {np.mean([v['trades'] for v in val_results]):.0f} avg", flush=True)

    # ================================================================
    # SAVE
    # ================================================================
    total_combos = 124 + 2304000 + tested  # Phase1A + Phase1B + Phase2
    output = {
        'version': 'v27',
        'total_combinations_tested': total_combos,
        'phase1a_cross_tested': 124,
        'phase1b_filter_tested': 2304000,
        'phase1b_passed': 112246,
        'phase2_risk_tested': tested,
        'phase2_valid': len(all_results),
        'models': {},
        'validation': validation,
        'top20_results': all_results[:20],
    }
    for name, m in models.items():
        output['models'][name] = m

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v27_optimization_results.json')
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=convert, ensure_ascii=False)

    print(f"\nResults saved: {path}", flush=True)
    print(f"Total combos tested: {total_combos:,}", flush=True)
    print("="*70, flush=True)
    print("COMPLETE", flush=True)


if __name__ == '__main__':
    main()
