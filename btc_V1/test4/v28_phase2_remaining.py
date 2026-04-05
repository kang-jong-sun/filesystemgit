"""
v28 Phase 2: 잔여 TF 스캔 (1h + 5m) + Top50 검증
- 5m은 메모리 최적화: HMA/VWMA 제외, 주요 조합만
- 1h 정상 실행
- 3개 완료 TF 결과 + 합산 → Top50 → 30회 검증
"""

import numpy as np
import pandas as pd
import time
import os
import json
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v28_backtest_engine import *

BASE = r"D:\filesystem\futures\btc_V1\test4"


def run_lightweight_5m_scan(df, n_combos=150000):
    """5m 경량 스캔: HMA/VWMA 제외, EMA/WMA/SMA만"""
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    n = len(close)

    # 경량 지표만 계산 (HMA, VWMA 제외 → 메모리 절약)
    print("[5m-lite] Calculating indicators (EMA/WMA/SMA only)...")
    t0 = time.time()
    indicators = {}

    fast_periods = [2, 3, 5, 7, 10, 14]
    slow_periods = [50, 100, 150, 200]
    all_periods = list(set(fast_periods + slow_periods))

    for mt_idx, mt_name in enumerate(['EMA', 'WMA', 'SMA']):
        mt = mt_idx  # 0=EMA, 1=WMA, 2=SMA
        for p in all_periods:
            key = f"ma_{mt_name}_{p}"
            indicators[key] = calc_ma(close, volume, mt, p)

    for p in [14, 20]:
        indicators[f"adx_{p}"] = calc_adx_wilder(high, low, close, p)
    indicators["rsi_14"] = calc_rsi_wilder(close, 14)
    print(f"[5m-lite] Indicators done: {time.time()-t0:.1f}s, {len(indicators)} items")

    # 조합 생성 (EMA/WMA/SMA만 = type 0,1,2)
    np.random.seed(555)
    ma_types_allowed = [0, 1, 2]  # EMA, WMA, SMA only
    fast_ps = [2, 3, 5, 7, 10, 14]
    slow_ps = [50, 100, 150, 200]
    adx_ps = [14, 20]
    adx_ts = [25, 30, 35, 40, 45]
    rsi_los = [25, 30, 35, 40]
    rsi_his = [60, 65, 70, 75]
    sls = [-4, -5, -6, -7, -8, -9, -10]
    t_acts = [3, 4, 5, 6, 7, 8, 10]
    t_pcts = [1, 2, 3, 4, 5]
    m_pcts = [15, 20, 25, 30, 35, 40, 50, 60]
    levs = [5, 7, 10, 15]
    delays = [0, 1, 2, 3, 5, 6]
    offsets = [-2.5, -1.5, -1.0, -0.5, -0.1, 0.0]
    skips = [0, 1]

    combos = []
    for _ in range(n_combos):
        combos.append((
            np.random.choice(ma_types_allowed),
            np.random.choice(ma_types_allowed),
            np.random.choice(fast_ps),
            np.random.choice(slow_ps),
            np.random.choice(adx_ps),
            np.random.choice(adx_ts),
            np.random.choice(rsi_los),
            np.random.choice(rsi_his),
            np.random.choice(sls),
            np.random.choice(t_acts),
            np.random.choice(t_pcts),
            np.random.choice(m_pcts),
            np.random.choice(levs),
            np.random.choice(delays),
            np.random.choice(offsets),
            np.random.choice(skips)
        ))

    ma_names = ['EMA', 'WMA', 'SMA', 'HMA', 'VWMA']
    timestamps_epoch = np.zeros(n, dtype=np.float64)
    results = []
    t0 = time.time()

    for idx, combo in enumerate(combos):
        (fast_mt, slow_mt, fast_p, slow_p,
         adx_p, adx_t, rsi_lo, rsi_hi,
         sl, t_act, t_pct, m_pct, lev,
         delay, offset, skip) = combo

        fast_key = f"ma_{ma_names[fast_mt]}_{fast_p}"
        slow_key = f"ma_{ma_names[slow_mt]}_{slow_p}"
        adx_key = f"adx_{adx_p}"

        if fast_key not in indicators or slow_key not in indicators:
            continue

        result = backtest_core(
            close, high, low, volume, timestamps_epoch,
            indicators[fast_key], indicators[slow_key],
            indicators[adx_key], indicators["rsi_14"],
            float(adx_t), float(rsi_lo), float(rsi_hi),
            float(sl), float(t_act), float(t_pct),
            float(m_pct), float(lev),
            int(delay), float(offset),
            0.0004, int(skip), n // 3
        )

        bal, trades, w, l, pf, mdd = result[:6]
        sl_c, tsl_c, rev_c, mcl, tp, tl = result[6:12]

        if trades >= 15 and bal > 3000 and pf > 1.0:
            ret = max((bal - 3000) / 3000 * 100, 1)
            pf_s = min(pf, 100)
            mdd_p = mdd / 100.0
            trade_b = min(trades / 50.0, 2.0)
            score = pf_s * np.log10(ret) * trade_b * (1 - mdd_p * 0.5)

            results.append({
                'combo': combo, 'balance': float(bal),
                'trades': int(trades), 'wins': int(w), 'losses': int(l),
                'pf': float(pf), 'mdd': float(mdd),
                'sl_count': int(sl_c), 'tsl_count': int(tsl_c), 'rev_count': int(rev_c),
                'max_consec_loss': int(mcl),
                'total_profit': float(tp), 'total_loss': float(tl),
                'win_rate': float(w / trades * 100) if trades > 0 else 0.0,
                'return_pct': float((bal - 3000) / 3000 * 100),
                'score': score,
                'yearly_data': [float(result[12+i]) for i in range(14)]
            })

        if (idx + 1) % 10000 == 0:
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed
            eta = (n_combos - idx - 1) / speed
            print(f"  [5m] {idx+1:>8,}/{n_combos:,} | {speed:.0f}/s | ETA {eta:.0f}s | valid {len(results)}")

    elapsed = time.time() - t0
    print(f"[5m-lite] DONE: {elapsed:.0f}s, {n_combos/elapsed:.0f}/s, valid={len(results)}")
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:200]


def load_intermediate(tf_label):
    """중간 결과 로드"""
    path = os.path.join(BASE, f"v28_intermediate_{tf_label}.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"[LOAD] {tf_label}: {len(data)} results")
        # combo를 tuple로 변환
        for d in data:
            d['combo'] = tuple(d['combo'])
        return data
    return []


def main():
    total_start = time.time()
    print("=" * 70)
    print("  v28 Phase 2: Remaining TF + Top50 Verification")
    print("=" * 70)

    # 1) 데이터 로드
    print("\n[1] Loading data...")
    df_5m = load_5m_data(BASE)
    df_1h = resample_ohlcv(df_5m, 60)
    print(f"  1h: {len(df_1h)} bars")

    # 2) 1h 스캔
    print("\n[2] Running 1h scan (200,000 combos)...")
    from v28_run_scan import run_single_tf_scan
    top_1h = run_single_tf_scan('1h', df_1h, n_combos=200000)
    for r in top_1h:
        r['tf'] = '1h'
    # 저장
    save = []
    for r in top_1h:
        entry = {k: v for k, v in r.items() if k != 'combo'}
        entry['combo'] = [int(x) if isinstance(x, (np.integer, int)) else float(x) for x in r['combo']]
        save.append(entry)
    with open(os.path.join(BASE, "v28_intermediate_1h.json"), 'w') as f:
        json.dump(save, f, indent=2)
    print(f"  [SAVE] 1h intermediate")

    # 3) 5m 경량 스캔
    print("\n[3] Running 5m lightweight scan (150,000 combos, EMA/WMA/SMA only)...")
    gc.collect()
    top_5m = run_lightweight_5m_scan(df_5m, n_combos=150000)
    for r in top_5m:
        r['tf'] = '5m'
    save = []
    for r in top_5m:
        entry = {k: v for k, v in r.items() if k not in ('combo', 'yearly_data')}
        entry['combo'] = [int(x) if isinstance(x, (np.integer, int)) else float(x) for x in r['combo']]
        save.append(entry)
    with open(os.path.join(BASE, "v28_intermediate_5m.json"), 'w') as f:
        json.dump(save, f, indent=2)
    print(f"  [SAVE] 5m intermediate")

    # 4) 전 TF 결과 통합
    print("\n[4] Merging all TF results...")
    all_results = []

    for tf in ['30m', '15m', '10m']:
        loaded = load_intermediate(tf)
        for r in loaded:
            r['tf'] = tf
            # score 재계산
            ret = max(r.get('return_pct', 0), 1)
            pf_s = min(r.get('pf', 1), 100)
            mdd_p = r.get('mdd', 50) / 100.0
            trade_b = min(r.get('trades', 0) / 50.0, 2.0)
            r['score'] = pf_s * np.log10(ret) * trade_b * (1 - mdd_p * 0.5)
        all_results.extend(loaded)

    all_results.extend(top_1h)
    all_results.extend(top_5m)

    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    print(f"  Total candidates: {len(all_results)}")

    # Top 50 선정
    top50 = all_results[:50]
    ma_names = ['EMA', 'WMA', 'SMA', 'HMA', 'VWMA']

    print("\n  === Unified Top 50 ===")
    for rank, r in enumerate(top50):
        combo = r['combo']
        desc = (f"{ma_names[int(combo[0])]}({int(combo[2])})/"
                f"{ma_names[int(combo[1])]}({int(combo[3])}) "
                f"ADX({int(combo[4])})>={int(combo[5])} RSI{int(combo[6])}-{int(combo[7])}")
        print(f"  #{rank+1:2d} [{r['tf']:>3s}] PF={r['pf']:>7.2f} MDD={r['mdd']:>5.1f}% "
              f"Ret={r['return_pct']:>10,.0f}% Tr={r['trades']:>4d} | {desc}")

    # 5) Top 50 상세 검증 (30회)
    print("\n[5] Verifying Top 50 (30 runs each)...")
    tf_cache = {}

    verified = []
    for rank, r in enumerate(top50):
        tf_label = r['tf']
        combo = tuple(int(x) if float(x) == int(float(x)) else float(x) for x in r['combo'])

        # 데이터 캐시
        if tf_label not in tf_cache:
            if tf_label == '5m':
                df = df_5m
            elif tf_label == '1h':
                df = df_1h
            else:
                mins = int(tf_label.replace('m', '').replace('h', ''))
                if 'h' in tf_label:
                    mins *= 60
                df = resample_ohlcv(df_5m, mins)
            tf_cache[tf_label] = df

        df = tf_cache[tf_label]
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        volume = df['volume'].values.astype(np.float64)
        n = len(close)
        timestamps_epoch = np.zeros(n, dtype=np.float64)

        fast_ma = calc_ma(close, volume, int(combo[0]), int(combo[2]))
        slow_ma = calc_ma(close, volume, int(combo[1]), int(combo[3]))
        adx_arr = calc_adx_wilder(high, low, close, int(combo[4]))
        rsi_arr = calc_rsi_wilder(close, 14)

        balances = []
        full_result = None
        for run in range(30):
            result = backtest_core(
                close, high, low, volume, timestamps_epoch,
                fast_ma, slow_ma, adx_arr, rsi_arr,
                float(combo[5]), float(combo[6]), float(combo[7]),
                float(combo[8]), float(combo[9]), float(combo[10]),
                float(combo[11]), float(combo[12]),
                int(combo[13]), float(combo[14]),
                0.0004, int(combo[15]), n // 3
            )
            balances.append(result[0])
            if run == 0:
                full_result = result

        std_val = float(np.std(balances))
        mean_bal = float(np.mean(balances))
        det = std_val < 0.01

        desc = (f"{ma_names[int(combo[0])]}({int(combo[2])})/"
                f"{ma_names[int(combo[1])]}({int(combo[3])})")
        full_desc = (f"{ma_names[int(combo[0])]}({int(combo[2])})/"
                    f"{ma_names[int(combo[1])]}({int(combo[3])}) "
                    f"ADX({int(combo[4])})>={int(combo[5])} RSI{int(combo[6])}-{int(combo[7])} "
                    f"SL{int(combo[8])}% Trail+{int(combo[9])}/-{int(combo[10])} "
                    f"M{int(combo[11])}% Lev{int(combo[12])}x "
                    f"Delay{int(combo[13])} Offset{float(combo[14])} Skip{int(combo[15])}")

        v = {
            'rank': rank + 1,
            'tf': tf_label,
            'strategy': desc,
            'full_desc': full_desc,
            'combo': [int(x) if float(x) == int(float(x)) else float(x) for x in combo],
            'balance': mean_bal,
            'return_pct': float((mean_bal - 3000) / 3000 * 100),
            'trades': int(full_result[1]),
            'wins': int(full_result[2]),
            'losses': int(full_result[3]),
            'pf': float(full_result[4]),
            'mdd': float(full_result[5]),
            'sl_count': int(full_result[6]),
            'tsl_count': int(full_result[7]),
            'rev_count': int(full_result[8]),
            'max_consec_loss': int(full_result[9]),
            'total_profit': float(full_result[10]),
            'total_loss': float(full_result[11]),
            'win_rate': float(full_result[2] / full_result[1] * 100) if full_result[1] > 0 else 0,
            'avg_win': float(full_result[10] / full_result[2]) if full_result[2] > 0 else 0,
            'avg_loss': float(full_result[11] / full_result[3]) if full_result[3] > 0 else 0,
            'verification_runs': 30,
            'std': std_val,
            'deterministic': det,
            'score': r.get('score', 0),
            'yearly': [float(full_result[12 + i]) for i in range(14)]
        }
        verified.append(v)
        status = "PASS" if det else "FAIL"
        print(f"  #{rank+1:2d} [{tf_label:>3s}] {desc:25s} PF={v['pf']:>8.2f} "
              f"MDD={v['mdd']:>5.1f}% Ret={v['return_pct']:>10,.0f}% "
              f"Tr={v['trades']:>4d} WR={v['win_rate']:>5.1f}% 30x={status}")

    # 6) 최종 저장
    print("\n[6] Saving final results...")
    with open(os.path.join(BASE, "v28_final_results.json"), 'w', encoding='utf-8') as f:
        json.dump(verified, f, indent=2, ensure_ascii=False)
    print(f"  Saved: v28_final_results.json")

    # 요약
    elapsed_total = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  TOTAL TIME: {elapsed_total/60:.1f} minutes")
    combos_total = 200000*3 + 200000 + 150000  # 30m+15m+10m + 1h + 5m
    print(f"  TOTAL COMBOS: {combos_total:,} ({combos_total/1000000:.1f}M)")
    print(f"  VERIFIED: {sum(1 for v in verified if v['deterministic'])}/50 PASS")
    print(f"{'='*70}")

    print(f"\n  === FINAL TOP 10 ===")
    for v in verified[:10]:
        print(f"\n  #{v['rank']} [{v['tf']}] {v['full_desc']}")
        print(f"    Balance: ${v['balance']:,.0f} | Return: +{v['return_pct']:,.0f}%")
        print(f"    PF: {v['pf']:.2f} | MDD: {v['mdd']:.1f}% | Trades: {v['trades']}")
        print(f"    Wins: {v['wins']} | Losses: {v['losses']} | Win Rate: {v['win_rate']:.1f}%")
        print(f"    SL: {v['sl_count']} | TSL: {v['tsl_count']} | REV: {v['rev_count']}")
        print(f"    Avg Win: ${v['avg_win']:,.0f} | Avg Loss: ${v['avg_loss']:,.0f}")
        print(f"    Max Consec Loss: {v['max_consec_loss']}")
        print(f"    30x Verify: {'PASS' if v['deterministic'] else 'FAIL'} (std={v['std']:.4f})")
        yr = v['yearly']
        years = ['2020', '2021', '2022', '2023', '2024', '2025', '2026']
        print(f"    Yearly:")
        for yi in range(7):
            s, e = yr[yi*2], yr[yi*2+1]
            if s > 0 and e > 0:
                ret = (e - s) / s * 100
                print(f"      {years[yi]}: ${s:,.0f} -> ${e:,.0f} ({ret:+.1f}%)")


if __name__ == '__main__':
    main()
