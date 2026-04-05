"""
BTC/USDT v27 - Fast Multi-Phase Optimizer
==========================================
Phase 1A: Cross Signal 선별 (cross_tf x MA pairs = ~124 combos)
Phase 1B: Filter 조합 (상위 cross x ADX x RSI x MACD x Entry = ~대폭 축소)
Phase 2: Risk 최적화 (SL/Trail/ATR/Margin)
Phase 3: 30회 반복 검증
"""

import pandas as pd
import numpy as np
from numba import njit
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(line_buffering=True)

from v27_backtest_engine import (
    load_5m_data, build_indicator_cache, map_tf_index,
    run_backtest_core, calc_ema, calc_sma, calc_wma, calc_hma
)


def run_optimization():
    print("=" * 70, flush=True)
    print("BTC/USDT v27 FAST MULTI-PHASE OPTIMIZER", flush=True)
    print("=" * 70, flush=True)

    # 1. Load & Cache
    print("\n[1/8] Loading data...", flush=True)
    df_5m = load_5m_data()

    print("\n[2/8] Building indicator cache...", flush=True)
    cache = build_indicator_cache(df_5m)

    ts_5m = cache['5m']['timestamp']
    close_5m = cache['5m']['close']
    high_5m = cache['5m']['high']
    low_5m = cache['5m']['low']
    n = len(close_5m)
    ts_i64 = ts_5m.astype('int64')

    # TF mappings
    tf_maps = {}
    for tf in ['10m', '15m', '30m', '1h']:
        tf_maps[tf] = map_tf_index(ts_5m, cache[tf]['timestamp'])

    # JIT warmup
    print("\n[3/8] JIT warmup...", flush=True)
    nw = 1000
    _ = run_backtest_core(
        close_5m[:nw], high_5m[:nw], low_5m[:nw], ts_i64[:nw],
        np.zeros(nw, dtype=np.int64), np.zeros(nw), np.zeros(nw),
        np.zeros(nw), np.zeros(nw),
        25, 30, 70, 0.07, 0.07, 0.03, 0.0, 0.0,
        10, 0.20, 0.10, -0.20, 0.0004, 3000.0, 6, 1.0,
        0, 2.0, 0, 2.0, np.array([1.]*7)
    )
    print("  Done.", flush=True)

    # ================================================================
    # PHASE 1A: Cross Signal 선별
    # ================================================================
    print("\n[4/8] Phase 1A: Cross Signal Screening...", flush=True)

    cross_tfs = ['5m', '10m', '15m', '30m']
    ma_pairs = [
        ('ema', 3, 'ema', 50), ('ema', 3, 'ema', 100), ('ema', 3, 'ema', 150),
        ('ema', 3, 'ema', 200), ('ema', 3, 'ema', 300),
        ('ema', 5, 'ema', 50), ('ema', 5, 'ema', 100), ('ema', 5, 'ema', 150),
        ('ema', 5, 'ema', 200), ('ema', 5, 'ema', 300),
        ('ema', 7, 'ema', 50), ('ema', 7, 'ema', 100), ('ema', 7, 'ema', 200),
        ('ema', 8, 'ema', 50), ('ema', 8, 'ema', 100), ('ema', 8, 'ema', 200),
        ('ema', 10, 'ema', 50), ('ema', 10, 'ema', 100), ('ema', 10, 'ema', 200),
        ('ema', 13, 'ema', 100), ('ema', 13, 'ema', 200),
        ('ema', 3, 'sma', 50), ('ema', 3, 'sma', 100), ('ema', 3, 'sma', 200),
        ('ema', 5, 'sma', 50), ('ema', 5, 'sma', 100), ('ema', 5, 'sma', 200),
        ('hma', 9, 'ema', 100), ('hma', 9, 'ema', 200),
        ('hma', 14, 'ema', 100), ('hma', 14, 'ema', 200),
    ]

    # Default filter for screening
    default_adx = cache['15m']['adx'][20]
    default_adx_5m = default_adx[tf_maps['15m']]
    default_rsi = cache['5m']['rsi'][14]
    default_atr = cache['5m']['atr'][14]
    default_macd = cache['5m']['macd']['12_26_9']['hist']

    # Pre-build all cross signals
    cross_results = []
    total_cross = len(cross_tfs) * len(ma_pairs)
    tested = 0

    for ctf in cross_tfs:
        c_data = cache[ctf]
        for ft, fl, st, sl_len in ma_pairs:
            tested += 1

            # Get MAs
            if ft == 'ema': fast_ma = c_data['ema'].get(fl)
            elif ft == 'hma': fast_ma = c_data['hma'].get(fl)
            elif ft == 'sma': fast_ma = c_data['sma'].get(fl)
            elif ft == 'wma': fast_ma = c_data['wma'].get(fl)
            else: fast_ma = c_data['ema'].get(fl)

            if st == 'ema': slow_ma = c_data['ema'].get(sl_len)
            elif st == 'sma': slow_ma = c_data['sma'].get(sl_len)
            else: slow_ma = c_data['ema'].get(sl_len)

            if fast_ma is None or slow_ma is None:
                continue

            # Generate signal
            sig_tf = np.zeros(len(fast_ma), dtype=np.int64)
            for j in range(1, len(fast_ma)):
                if fast_ma[j] > slow_ma[j]: sig_tf[j] = 1
                elif fast_ma[j] < slow_ma[j]: sig_tf[j] = -1

            if ctf == '5m': csig = sig_tf
            else: csig = sig_tf[tf_maps[ctf]]

            # Quick backtest with default params (40% margin like v25.2)
            result = run_backtest_core(
                close_5m, high_5m, low_5m, ts_i64,
                csig, default_adx_5m, default_rsi, default_atr, default_macd,
                25, 30, 75,
                0.07, 0.07, 0.03, 0.0, 0.0,
                10, 0.40, 0.20, -0.20, 0.0004, 3000.0,
                6, 1.0, 0, 2.0, 0, 2.0,
                np.array([0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0])
            )

            final_cap, tc = result[0], result[1]
            if tc >= 5:
                rois = result[3]
                pnls = result[4]
                total_profit = np.sum(pnls[pnls > 0])
                total_loss = abs(np.sum(pnls[pnls < 0])) + 1e-10
                pf = total_profit / total_loss
                wins = np.sum(rois > 0)
                ret = (final_cap - 3000) / 3000 * 100

                cross_results.append({
                    'cross_tf': ctf,
                    'fast_type': ft, 'fast_len': fl,
                    'slow_type': st, 'slow_len': sl_len,
                    'signal': csig,  # Keep for reuse
                    'final_cap': final_cap,
                    'trades': tc,
                    'pf': pf,
                    'return_pct': ret,
                    'win_rate': wins / tc * 100,
                })

            if tested % 20 == 0:
                print(f"  {tested}/{total_cross} cross combos tested, {len(cross_results)} valid", flush=True)

    # Sort by return and keep top 40
    cross_results.sort(key=lambda x: x['return_pct'], reverse=True)
    top_crosses = cross_results[:40]

    print(f"\n  Phase 1A complete: {tested} tested -> {len(cross_results)} valid -> Top 40 selected", flush=True)
    for i, cr in enumerate(top_crosses[:10]):
        print(f"    #{i+1}: {cr['cross_tf']} {cr['fast_type']}{cr['fast_len']}/{cr['slow_type']}{cr['slow_len']} "
              f"| Ret:{cr['return_pct']:.0f}% PF:{cr['pf']:.2f} T:{cr['trades']} WR:{cr['win_rate']:.0f}%", flush=True)

    # ================================================================
    # PHASE 1B: Filter Optimization
    # ================================================================
    print("\n[5/8] Phase 1B: Filter Optimization (ADX x RSI x MACD x Entry)...", flush=True)

    adx_tfs = ['5m', '10m', '15m', '30m']
    adx_configs = [(14, 20), (14, 25), (14, 30), (14, 35), (14, 40),
                   (20, 20), (20, 25), (20, 30), (20, 35), (20, 40)]
    rsi_tfs = ['5m', '10m', '15m']
    rsi_configs = [(7, 30, 70), (7, 35, 75), (7, 40, 80),
                   (14, 30, 70), (14, 35, 70), (14, 35, 75), (14, 40, 75), (14, 40, 80)]
    macd_keys = ['5_35_5', '8_21_5', '12_26_9']
    entry_delays = [0, 3, 6, 12]  # bars (0, 15, 30, 60 min)
    entry_price_tols = [0.5, 1.0, 1.5, 2.0, 2.5]

    # Pre-map all ADX/RSI
    adx_mapped = {}
    for atf in adx_tfs:
        for ap, _ in adx_configs:
            key = f"{atf}_{ap}"
            if key not in adx_mapped:
                vals = cache[atf]['adx'][ap]
                adx_mapped[key] = vals if atf == '5m' else vals[tf_maps[atf]]

    rsi_mapped = {}
    for rtf in rsi_tfs:
        for rp, _, _ in rsi_configs:
            key = f"{rtf}_{rp}"
            if key not in rsi_mapped:
                vals = cache[rtf]['rsi'][rp]
                rsi_mapped[key] = vals if rtf == '5m' else vals[tf_maps[rtf]]

    macd_mapped = {}
    for mk in macd_keys:
        macd_mapped[mk] = cache['5m']['macd'][mk]['hist']

    atr_5m = cache['5m']['atr'][14]

    per_cross = len(adx_tfs) * len(adx_configs) * len(rsi_tfs) * len(rsi_configs) * len(macd_keys) * len(entry_delays) * len(entry_price_tols)
    total_1b = len(top_crosses) * per_cross
    print(f"  {len(top_crosses)} crosses x {per_cross} filter combos = {total_1b:,} total", flush=True)

    phase1b_results = []
    tested = 0
    t0 = time.time()

    for cr in top_crosses:
        csig = cr['signal']

        for atf in adx_tfs:
            for ap, amin in adx_configs:
                adx_v = adx_mapped[f"{atf}_{ap}"]

                for rtf in rsi_tfs:
                    for rp, rmin, rmax in rsi_configs:
                        rsi_v = rsi_mapped[f"{rtf}_{rp}"]

                        for mk in macd_keys:
                            macd_v = macd_mapped[mk]

                            for ed in entry_delays:
                                for ept in entry_price_tols:
                                    tested += 1

                                    result = run_backtest_core(
                                        close_5m, high_5m, low_5m, ts_i64,
                                        csig, adx_v, rsi_v, atr_5m, macd_v,
                                        amin, rmin, rmax,
                                        0.07, 0.07, 0.03,
                                        0.0, 0.0,
                                        10, 0.40, 0.20, -0.20,
                                        0.0004, 3000.0,
                                        ed, ept,
                                        0, 2.0, 0, 2.0,
                                        np.array([0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0])
                                    )

                                    final_cap, tc = result[0], result[1]
                                    if tc >= 8:
                                        rois = result[3]
                                        pnls = result[4]
                                        wins = np.sum(rois > 0)
                                        total_profit = np.sum(pnls[pnls > 0])
                                        total_loss = abs(np.sum(pnls[pnls < 0])) + 1e-10
                                        pf = total_profit / total_loss
                                        ret = (final_cap - 3000) / 3000 * 100

                                        eq = result[8]
                                        mdd = 0.0
                                        if len(eq) > 0:
                                            peak_eq = np.maximum.accumulate(eq)
                                            dd = (eq - peak_eq) / (peak_eq + 1e-10)
                                            mdd = abs(np.min(dd)) * 100

                                        if pf >= 1.5 and ret > 50:
                                            phase1b_results.append({
                                                'cross_tf': cr['cross_tf'],
                                                'fast': f"{cr['fast_type']}{cr['fast_len']}",
                                                'slow': f"{cr['slow_type']}{cr['slow_len']}",
                                                'adx_tf': atf, 'adx_p': ap, 'adx_min': amin,
                                                'rsi_tf': rtf, 'rsi_p': rp, 'rsi_min': rmin, 'rsi_max': rmax,
                                                'macd': mk,
                                                'entry_delay': ed, 'entry_tol': ept,
                                                'final_cap': final_cap,
                                                'trades': tc,
                                                'win_rate': wins / tc * 100,
                                                'pf': pf,
                                                'mdd': mdd,
                                                'return_pct': ret,
                                                'avg_win': float(np.mean(rois[rois > 0]) * 100) if wins > 0 else 0,
                                                'avg_loss': float(np.mean(rois[rois <= 0]) * 100) if (tc - wins) > 0 else 0,
                                            })

                                    if tested % 100000 == 0:
                                        elapsed = time.time() - t0
                                        rate = tested / elapsed
                                        remaining = (total_1b - tested) / rate / 60
                                        print(f"  Progress: {tested:,}/{total_1b:,} ({tested/total_1b*100:.1f}%) "
                                              f"| {rate:.0f}/s | ~{remaining:.1f}min left "
                                              f"| {len(phase1b_results)} passed", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Phase 1B: {tested:,} tested in {elapsed:.0f}s ({tested/elapsed:.0f}/s) -> {len(phase1b_results)} passed", flush=True)

    # Score and sort
    for r in phase1b_results:
        r['score'] = r['pf'] * r['return_pct'] / (r['mdd'] + 10) * np.sqrt(r['trades'])

    phase1b_results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n  Top 15 Phase 1B results:", flush=True)
    for i, r in enumerate(phase1b_results[:15]):
        print(f"    #{i+1}: {r['cross_tf']} {r['fast']}/{r['slow']} "
              f"ADX:{r['adx_tf']}>{r['adx_min']} RSI:{r['rsi_tf']}{r['rsi_min']}-{r['rsi_max']} "
              f"MACD:{r['macd']} ED:{r['entry_delay']} ET:{r['entry_tol']} "
              f"| Ret:{r['return_pct']:.0f}% PF:{r['pf']:.2f} MDD:{r['mdd']:.1f}% T:{r['trades']}", flush=True)

    # ================================================================
    # PHASE 2: Risk Management Optimization
    # ================================================================
    print("\n[6/8] Phase 2: Risk Management Optimization...", flush=True)

    top_1b = phase1b_results[:200]  # Top 200

    sl_pcts = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
    trail_acts = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]
    trail_pcts = [0.02, 0.025, 0.03, 0.04, 0.05]
    partial_exits = [(0.0, 0.0), (0.10, 0.30), (0.15, 0.30), (0.20, 0.50)]
    atr_options = [(0, 0, 0, 0), (1, 2.0, 0, 0), (1, 2.5, 0, 0),
                   (0, 0, 1, 2.5), (1, 2.0, 1, 2.5)]
    margins = [(0.20, 0.10), (0.30, 0.15), (0.40, 0.20)]

    per_risk = len(sl_pcts) * len(trail_acts) * len(trail_pcts) * len(partial_exits) * len(atr_options) * len(margins)
    total_2 = len(top_1b) * per_risk
    print(f"  {len(top_1b)} Phase1B winners x {per_risk} risk combos = {total_2:,}", flush=True)

    phase2_results = []
    tested = 0
    t0 = time.time()

    for p1 in top_1b:
        # Rebuild cross signal
        ctf = p1['cross_tf']
        ft_str = p1['fast']
        st_str = p1['slow']

        for ma_t in ['ema', 'sma', 'hma', 'wma']:
            if ft_str.startswith(ma_t):
                ft = ma_t; fl = int(ft_str[len(ma_t):]); break
        for ma_t in ['ema', 'sma', 'hma', 'wma']:
            if st_str.startswith(ma_t):
                st = ma_t; sl_len = int(st_str[len(ma_t):]); break

        c_data = cache[ctf]
        fast_ma = c_data.get(ft, c_data['ema']).get(fl) if isinstance(c_data.get(ft), dict) else c_data['ema'].get(fl)
        slow_ma = c_data.get(st, c_data['ema']).get(sl_len) if isinstance(c_data.get(st), dict) else c_data['ema'].get(sl_len)

        if fast_ma is None or slow_ma is None:
            continue

        sig_tf = np.zeros(len(fast_ma), dtype=np.int64)
        for j in range(1, len(fast_ma)):
            if fast_ma[j] > slow_ma[j]: sig_tf[j] = 1
            elif fast_ma[j] < slow_ma[j]: sig_tf[j] = -1

        csig = sig_tf if ctf == '5m' else sig_tf[tf_maps[ctf]]
        adx_v = adx_mapped[f"{p1['adx_tf']}_{p1['adx_p']}"]
        rsi_v = rsi_mapped[f"{p1['rsi_tf']}_{p1['rsi_p']}"]
        macd_v = macd_mapped[p1['macd']]

        for sl in sl_pcts:
            for ta in trail_acts:
                for tp in trail_pcts:
                    if tp >= ta: continue
                    for pe_pct, pe_ratio in partial_exits:
                        for use_atr_sl, atr_sl_m, use_atr_t, atr_t_m in atr_options:
                            for margin_n, margin_r in margins:
                                tested += 1

                                result = run_backtest_core(
                                    close_5m, high_5m, low_5m, ts_i64,
                                    csig, adx_v, rsi_v, atr_5m, macd_v,
                                    p1['adx_min'], p1['rsi_min'], p1['rsi_max'],
                                    sl, ta, tp, pe_pct, pe_ratio,
                                    10, margin_n, margin_r, -0.20,
                                    0.0004, 3000.0,
                                    p1['entry_delay'], p1['entry_tol'],
                                    use_atr_sl, atr_sl_m, use_atr_t, atr_t_m,
                                    np.array([0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0])
                                )

                                final_cap, tc = result[0], result[1]
                                if tc >= 10 and final_cap > 3000:
                                    rois = result[3]
                                    pnls = result[4]
                                    wins = np.sum(rois > 0)
                                    total_profit = np.sum(pnls[pnls > 0])
                                    total_loss = abs(np.sum(pnls[pnls < 0])) + 1e-10
                                    pf = total_profit / total_loss
                                    ret = (final_cap - 3000) / 3000 * 100

                                    eq = result[8]
                                    mdd = 0.0
                                    if len(eq) > 0:
                                        peak_eq = np.maximum.accumulate(eq)
                                        dd = (eq - peak_eq) / (peak_eq + 1e-10)
                                        mdd = abs(np.min(dd)) * 100

                                    if pf >= 5.0 and mdd < 55 and ret > 500:
                                        phase2_results.append({
                                            **{k: v for k, v in p1.items() if k != 'score'},
                                            'sl': sl, 'trail_act': ta, 'trail_pct': tp,
                                            'partial_pct': pe_pct, 'partial_ratio': pe_ratio,
                                            'use_atr_sl': use_atr_sl, 'atr_sl_mult': atr_sl_m,
                                            'use_atr_trail': use_atr_t, 'atr_trail_mult': atr_t_m,
                                            'margin': margin_n, 'margin_dd': margin_r,
                                            'final_cap': final_cap,
                                            'trades': tc,
                                            'win_rate': wins / tc * 100,
                                            'pf': pf, 'mdd': mdd,
                                            'return_pct': ret,
                                            'avg_win': float(np.mean(rois[rois > 0]) * 100) if wins > 0 else 0,
                                            'avg_loss': float(np.mean(rois[rois <= 0]) * 100) if (tc - wins) > 0 else 0,
                                        })

                                if tested % 200000 == 0:
                                    elapsed = time.time() - t0
                                    rate = tested / elapsed
                                    remaining = (total_2 - tested) / rate / 60 if rate > 0 else 999
                                    print(f"  Progress: {tested:,}/{total_2:,} ({tested/total_2*100:.1f}%) "
                                          f"| {rate:.0f}/s | ~{remaining:.1f}min left "
                                          f"| {len(phase2_results)} passed", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Phase 2: {tested:,} tested in {elapsed:.0f}s -> {len(phase2_results)} passed", flush=True)

    # Score
    for r in phase2_results:
        r['score'] = r['pf'] * r['return_pct'] / (r['mdd'] + 5) * np.sqrt(r['trades'])

    phase2_results.sort(key=lambda x: x['score'], reverse=True)

    # If not enough passed, relax criteria
    if len(phase2_results) < 3:
        print("  Relaxing Phase 2 criteria (PF>=3, MDD<65)...", flush=True)
        phase2_results = []
        for p1 in top_1b[:50]:
            # Use best default risk params
            for sl in [0.05, 0.07]:
                for ta in [0.06, 0.07, 0.08]:
                    for tp in [0.03, 0.04]:
                        for margin_n in [0.20, 0.30, 0.40]:
                            ctf = p1['cross_tf']
                            ft_str = p1['fast']; st_str = p1['slow']
                            for ma_t in ['ema', 'sma', 'hma', 'wma']:
                                if ft_str.startswith(ma_t): ft = ma_t; fl = int(ft_str[len(ma_t):]); break
                            for ma_t in ['ema', 'sma', 'hma', 'wma']:
                                if st_str.startswith(ma_t): st = ma_t; sl_len_v = int(st_str[len(ma_t):]); break

                            c_data = cache[ctf]
                            fm = c_data.get(ft, c_data['ema']).get(fl) if isinstance(c_data.get(ft), dict) else c_data['ema'].get(fl)
                            sm = c_data.get(st, c_data['ema']).get(sl_len_v) if isinstance(c_data.get(st), dict) else c_data['ema'].get(sl_len_v)
                            if fm is None or sm is None: continue

                            sig = np.zeros(len(fm), dtype=np.int64)
                            for j in range(1, len(fm)):
                                if fm[j] > sm[j]: sig[j] = 1
                                elif fm[j] < sm[j]: sig[j] = -1
                            cs = sig if ctf == '5m' else sig[tf_maps[ctf]]

                            result = run_backtest_core(
                                close_5m, high_5m, low_5m, ts_i64,
                                cs, adx_mapped[f"{p1['adx_tf']}_{p1['adx_p']}"],
                                rsi_mapped[f"{p1['rsi_tf']}_{p1['rsi_p']}"],
                                atr_5m, macd_mapped[p1['macd']],
                                p1['adx_min'], p1['rsi_min'], p1['rsi_max'],
                                sl, ta, tp, 0.0, 0.0,
                                10, margin_n, margin_n*0.5, -0.20,
                                0.0004, 3000.0, p1['entry_delay'], p1['entry_tol'],
                                0, 2.0, 0, 2.0,
                                np.array([0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0])
                            )
                            fc, tc = result[0], result[1]
                            if tc >= 8 and fc > 3000:
                                rois = result[3]; pnls = result[4]
                                wins = np.sum(rois > 0)
                                tp_sum = np.sum(pnls[pnls > 0])
                                tl_sum = abs(np.sum(pnls[pnls < 0])) + 1e-10
                                pf = tp_sum / tl_sum
                                ret = (fc - 3000) / 3000 * 100
                                eq = result[8]; mdd = 0.0
                                if len(eq) > 0:
                                    peq = np.maximum.accumulate(eq)
                                    dd = (eq - peq) / (peq + 1e-10)
                                    mdd = abs(np.min(dd)) * 100
                                if pf >= 3.0 and mdd < 65:
                                    phase2_results.append({
                                        **{k: v for k, v in p1.items() if k != 'score'},
                                        'sl': sl, 'trail_act': ta, 'trail_pct': tp,
                                        'partial_pct': 0, 'partial_ratio': 0,
                                        'use_atr_sl': 0, 'atr_sl_mult': 0,
                                        'use_atr_trail': 0, 'atr_trail_mult': 0,
                                        'margin': margin_n, 'margin_dd': margin_n*0.5,
                                        'final_cap': fc, 'trades': tc,
                                        'win_rate': wins/tc*100, 'pf': pf, 'mdd': mdd,
                                        'return_pct': ret,
                                        'avg_win': float(np.mean(rois[rois > 0]) * 100) if wins > 0 else 0,
                                        'avg_loss': float(np.mean(rois[rois <= 0]) * 100) if (tc - wins) > 0 else 0,
                                    })

        for r in phase2_results:
            r['score'] = r['pf'] * r['return_pct'] / (r['mdd'] + 5) * np.sqrt(max(r['trades'], 1))
        phase2_results.sort(key=lambda x: x['score'], reverse=True)
        print(f"  Relaxed: {len(phase2_results)} passed", flush=True)

    # ================================================================
    # SELECT TOP 3 MODELS
    # ================================================================
    print("\n[7/8] Selecting Top 3 Models...", flush=True)

    if len(phase2_results) >= 3:
        # Model A: 최고 수익률 (high trade frequency preferred)
        candidates_a = sorted(phase2_results[:50], key=lambda x: x['return_pct'] * np.sqrt(x['trades']), reverse=True)
        model_a = candidates_a[0]

        # Model B: 최고 안정성 (PF / MDD)
        candidates_b = sorted(phase2_results[:50], key=lambda x: x['pf'] / (x['mdd'] + 5) * 100, reverse=True)
        model_b = candidates_b[0]

        # Model C: 균형 (거래빈도 x 손익비 x 수익률)
        candidates_c = sorted(phase2_results[:50], key=lambda x: x['trades'] * x['pf'] * np.log(max(x['return_pct'], 1) + 1) / (x['mdd'] + 10), reverse=True)
        model_c = candidates_c[0]
    elif len(phase2_results) >= 1:
        model_a = phase2_results[0]
        model_b = phase2_results[min(1, len(phase2_results)-1)]
        model_c = phase2_results[min(2, len(phase2_results)-1)]
    else:
        print("  ERROR: No results passed! Using Phase 1B top results.", flush=True)
        model_a = phase1b_results[0] if phase1b_results else None
        model_b = phase1b_results[1] if len(phase1b_results) > 1 else model_a
        model_c = phase1b_results[2] if len(phase1b_results) > 2 else model_a

    models = {'A': model_a, 'B': model_b, 'C': model_c}

    for name, m in models.items():
        if m is None: continue
        print(f"\n  Model {name}: {m['cross_tf']} {m['fast']}/{m['slow']}", flush=True)
        print(f"    ADX: {m['adx_tf']} p={m['adx_p']} min={m['adx_min']}", flush=True)
        print(f"    RSI: {m['rsi_tf']} p={m['rsi_p']} {m['rsi_min']}-{m['rsi_max']}", flush=True)
        print(f"    SL: {m.get('sl',0.07)*100:.0f}% Trail: +{m.get('trail_act',0.07)*100:.0f}%/-{m.get('trail_pct',0.03)*100:.0f}%", flush=True)
        print(f"    Margin: {m.get('margin',0.20)*100:.0f}%/{m.get('margin_dd',0.10)*100:.0f}%", flush=True)
        print(f"    Return: {m['return_pct']:.0f}% | PF: {m['pf']:.2f} | MDD: {m['mdd']:.1f}% | Trades: {m['trades']}", flush=True)
        print(f"    Win Rate: {m['win_rate']:.1f}% | Avg Win: {m.get('avg_win',0):.1f}% | Avg Loss: {m.get('avg_loss',0):.1f}%", flush=True)

    # ================================================================
    # PHASE 3: 30x Validation
    # ================================================================
    print("\n[8/8] 30x Repeated Validation...", flush=True)

    validation = {}
    for name, m in models.items():
        if m is None: continue
        print(f"\n  Validating Model {name}...", flush=True)

        val_results = []
        for run in range(30):
            skip = 0.03 + (run / 30) * 0.12
            start = int(n * skip)

            # Rebuild for sliced data
            ctf = m['cross_tf']
            ft_str = m['fast']; st_str = m['slow']
            for ma_t in ['ema', 'sma', 'hma', 'wma']:
                if ft_str.startswith(ma_t): ft = ma_t; fl = int(ft_str[len(ma_t):]); break
            for ma_t in ['ema', 'sma', 'hma', 'wma']:
                if st_str.startswith(ma_t): stt = ma_t; sll = int(st_str[len(ma_t):]); break

            c_data = cache[ctf]
            fm = c_data.get(ft, c_data['ema']).get(fl) if isinstance(c_data.get(ft), dict) else c_data['ema'].get(fl)
            sm = c_data.get(stt, c_data['ema']).get(sll) if isinstance(c_data.get(stt), dict) else c_data['ema'].get(sll)
            if fm is None or sm is None: continue

            sig = np.zeros(len(fm), dtype=np.int64)
            for j in range(1, len(fm)):
                if fm[j] > sm[j]: sig[j] = 1
                elif fm[j] < sm[j]: sig[j] = -1
            cs = sig if ctf == '5m' else sig[tf_maps[ctf]]

            atf = m['adx_tf']
            adx_v = adx_mapped[f"{atf}_{m['adx_p']}"]
            rtf = m['rsi_tf']
            rsi_v = rsi_mapped[f"{rtf}_{m['rsi_p']}"]
            macd_v = macd_mapped[m['macd']]

            # Slice all
            sl_end = n
            cs_s = cs[start:sl_end]
            adx_s = adx_v[start:sl_end] if len(adx_v) >= sl_end else adx_v[start:]
            rsi_s = rsi_v[start:sl_end] if len(rsi_v) >= sl_end else rsi_v[start:]
            atr_s = atr_5m[start:sl_end]
            macd_s = macd_v[start:sl_end]
            close_s = close_5m[start:sl_end]
            high_s = high_5m[start:sl_end]
            low_s = low_5m[start:sl_end]
            ts_s = ts_i64[start:sl_end]

            ml = min(len(close_s), len(cs_s), len(adx_s), len(rsi_s), len(atr_s), len(macd_s))

            result = run_backtest_core(
                close_s[:ml], high_s[:ml], low_s[:ml], ts_s[:ml],
                cs_s[:ml], adx_s[:ml], rsi_s[:ml], atr_s[:ml], macd_s[:ml],
                m['adx_min'], m['rsi_min'], m['rsi_max'],
                m.get('sl', 0.07), m.get('trail_act', 0.07), m.get('trail_pct', 0.03),
                m.get('partial_pct', 0), m.get('partial_ratio', 0),
                10, m.get('margin', 0.20), m.get('margin_dd', 0.10), -0.20,
                0.0004, 3000.0,
                m.get('entry_delay', 6), m.get('entry_tol', 1.0),
                m.get('use_atr_sl', 0), m.get('atr_sl_mult', 2.0),
                m.get('use_atr_trail', 0), m.get('atr_trail_mult', 2.0),
                np.array([0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0])
            )

            fc, tc = result[0], result[1]
            rois = result[3]; pnls = result[4]
            wins = np.sum(rois > 0) if tc > 0 else 0
            tp_sum = np.sum(pnls[pnls > 0])
            tl_sum = abs(np.sum(pnls[pnls < 0])) + 1e-10
            pf = tp_sum / tl_sum
            eq = result[8]; mdd = 0.0
            if len(eq) > 0:
                peq = np.maximum.accumulate(eq)
                dd = (eq - peq) / (peq + 1e-10)
                mdd = abs(np.min(dd)) * 100

            val_results.append({
                'run': run + 1, 'skip': f"{skip*100:.1f}%",
                'final_cap': fc, 'return_pct': (fc - 3000) / 3000 * 100,
                'trades': tc, 'win_rate': wins / tc * 100 if tc > 0 else 0,
                'pf': pf, 'mdd': mdd,
            })

        validation[name] = val_results
        rets = [v['return_pct'] for v in val_results]
        pfs = [v['pf'] for v in val_results]
        mdds = [v['mdd'] for v in val_results]
        print(f"    Return: {np.mean(rets):.0f}% ± {np.std(rets):.0f}% (min:{np.min(rets):.0f}% max:{np.max(rets):.0f}%)", flush=True)
        print(f"    PF: {np.mean(pfs):.2f} ± {np.std(pfs):.2f}", flush=True)
        print(f"    MDD: {np.mean(mdds):.1f}% ± {np.std(mdds):.1f}%", flush=True)

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    print("\n\nSaving results...", flush=True)

    total_combos_tested = total_cross + total_1b + total_2
    output = {
        'version': 'v27',
        'total_combinations_tested': total_combos_tested,
        'phase1a_cross_tested': total_cross,
        'phase1a_valid': len(cross_results),
        'phase1b_filter_tested': total_1b,
        'phase1b_passed': len(phase1b_results),
        'phase2_risk_tested': total_2,
        'phase2_passed': len(phase2_results),
        'models': {},
        'validation': {},
        'phase1b_top20': [{k: v for k, v in r.items() if k != 'signal'} for r in phase1b_results[:20]],
        'phase2_top20': phase2_results[:20],
    }

    for name, m in models.items():
        if m:
            output['models'][name] = {k: v for k, v in m.items() if k not in ('signal', 'score')}
    for name, v in validation.items():
        output['validation'][name] = v

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v27_optimization_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=convert, ensure_ascii=False)

    print(f"\nResults saved: {path}", flush=True)
    print(f"\nTotal combinations tested: {total_combos_tested:,}", flush=True)
    print("=" * 70, flush=True)
    print("OPTIMIZATION COMPLETE", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    run_optimization()
