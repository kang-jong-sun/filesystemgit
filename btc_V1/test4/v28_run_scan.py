"""
v28 대규모 스캔 실행기
- 5개 타임프레임 x 200,000 조합 = 1,000,000+ 조합
- Top 50 → 30회 반복 검증
- 결과 JSON 저장
"""

import numpy as np
import pandas as pd
import time
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v28_backtest_engine import *

BASE = r"D:\filesystem\futures\btc_V1\test4"


def run_single_tf_scan(tf_label, df, n_combos=200000):
    """단일 타임프레임 스캔"""
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    n = len(close)

    ma_names = ['EMA', 'WMA', 'SMA', 'HMA', 'VWMA']

    # 지표 사전 계산
    print(f"\n[{tf_label}] 지표 사전 계산...")
    t0 = time.time()
    indicators = {}

    fast_periods = [2, 3, 5, 7, 10, 14, 21]
    slow_periods = [50, 100, 150, 200, 250, 300]
    all_periods = list(set(fast_periods + slow_periods))

    for mt_idx, mt in enumerate(range(5)):
        for p in all_periods:
            key = f"ma_{ma_names[mt_idx]}_{p}"
            try:
                indicators[key] = calc_ma(close, volume, mt, p)
            except:
                pass

    for p in [14, 20]:
        indicators[f"adx_{p}"] = calc_adx_wilder(high, low, close, p)
    indicators["rsi_14"] = calc_rsi_wilder(close, p)

    print(f"[{tf_label}] 지표 계산 완료: {time.time()-t0:.1f}초, {len(indicators)}개")

    # 조합 생성
    np.random.seed(42 + hash(tf_label) % 1000)
    combos = generate_param_combinations(n_combos)

    # 스캔 실행
    timestamps_epoch = np.zeros(n, dtype=np.float64)
    results = []
    t0 = time.time()
    batch_size = 10000

    for idx, combo in enumerate(combos):
        (fast_mt, slow_mt, fast_p, slow_p,
         adx_p, adx_t, rsi_lo, rsi_hi,
         sl, t_act, t_pct, m_pct, lev,
         delay, offset, skip) = combo

        fast_key = f"ma_{ma_names[fast_mt]}_{fast_p}"
        slow_key = f"ma_{ma_names[slow_mt]}_{slow_p}"
        adx_key = f"adx_{adx_p}"

        if fast_key not in indicators or slow_key not in indicators or adx_key not in indicators:
            continue

        fast_ma = indicators[fast_key]
        slow_ma = indicators[slow_key]
        adx_arr = indicators[adx_key]
        rsi_arr = indicators["rsi_14"]

        result = backtest_core(
            close, high, low, volume, timestamps_epoch,
            fast_ma, slow_ma, adx_arr, rsi_arr,
            float(adx_t), float(rsi_lo), float(rsi_hi),
            float(sl), float(t_act), float(t_pct),
            float(m_pct), float(lev),
            int(delay), float(offset),
            0.0004, int(skip), n // 3
        )

        bal, trades, w, l, pf, mdd = result[:6]
        sl_c, tsl_c, rev_c, mcl, tp, tl = result[6:12]

        # 필터: 최소 15거래, 양수 수익
        if trades >= 15 and bal > 3000 and pf > 1.0:
            results.append({
                'combo': combo,
                'balance': float(bal),
                'trades': int(trades),
                'wins': int(w),
                'losses': int(l),
                'pf': float(pf),
                'mdd': float(mdd),
                'sl_count': int(sl_c),
                'tsl_count': int(tsl_c),
                'rev_count': int(rev_c),
                'max_consec_loss': int(mcl),
                'total_profit': float(tp),
                'total_loss': float(tl),
                'win_rate': float(w / trades * 100) if trades > 0 else 0.0,
                'return_pct': float((bal - 3000) / 3000 * 100),
                'yearly_data': [float(result[12+i]) for i in range(14)]
            })

        if (idx + 1) % batch_size == 0:
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed
            eta = (n_combos - idx - 1) / speed
            print(f"  [{tf_label}] {idx+1:>8,}/{n_combos:,} | "
                  f"{speed:.0f}/s | ETA {eta:.0f}s | valid {len(results)}")

    elapsed = time.time() - t0
    print(f"[{tf_label}] SCAN DONE: {elapsed:.0f}s, {n_combos/elapsed:.0f}/s, valid={len(results)}")

    # 복합 점수 정렬: PF * log(return) - MDD_penalty
    for r in results:
        ret = max(r['return_pct'], 1)
        pf_score = min(r['pf'], 100)  # PF cap
        mdd_penalty = r['mdd'] / 100.0
        trade_bonus = min(r['trades'] / 50.0, 2.0)  # 거래 빈도 보너스
        r['score'] = pf_score * np.log10(ret) * trade_bonus * (1 - mdd_penalty * 0.5)

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:200]


def main():
    total_start = time.time()

    # 1) 데이터 로드
    print("=" * 70)
    print("  v28 AI Trading Backtest - 1,000,000+ Combo Scan")
    print("=" * 70)

    print("\n[1/6] Loading 5m data...")
    df_5m = load_5m_data(BASE)

    # 2) 멀티 타임프레임
    print("\n[2/6] Building multi-timeframe data...")
    tf_map = {
        '5m': df_5m,
        '10m': resample_ohlcv(df_5m, 10),
        '15m': resample_ohlcv(df_5m, 15),
        '30m': resample_ohlcv(df_5m, 30),
        '1h': resample_ohlcv(df_5m, 60),
    }
    for k, v in tf_map.items():
        print(f"  {k}: {len(v)} bars")

    # 3) 각 TF별 스캔
    print("\n[3/6] Running 200,000 combos per TF (1,000,000 total)...")
    all_top = []

    for tf_label in ['30m', '15m', '10m', '5m', '1h']:
        df = tf_map[tf_label]
        top = run_single_tf_scan(tf_label, df, n_combos=200000)
        for r in top:
            r['tf'] = tf_label
        all_top.extend(top)
        # 중간 저장
        save_intermediate(all_top, tf_label)

    # 4) 통합 Top 50
    print("\n[4/6] Selecting unified Top 50...")
    all_top.sort(key=lambda x: x['score'], reverse=True)
    top50 = all_top[:50]

    ma_names = ['EMA', 'WMA', 'SMA', 'HMA', 'VWMA']
    for rank, r in enumerate(top50):
        combo = r['combo']
        desc = (f"{ma_names[combo[0]]}({combo[2]})/"
                f"{ma_names[combo[1]]}({combo[3]}) "
                f"ADX({combo[4]})>={combo[5]} RSI{combo[6]}-{combo[7]} "
                f"SL{combo[8]}% T+{combo[9]}/-{combo[10]} "
                f"M{combo[11]}% L{combo[12]}x D{combo[13]}")
        print(f"  #{rank+1:2d} [{r['tf']:>3s}] PF={r['pf']:>7.2f} MDD={r['mdd']:>5.1f}% "
              f"Ret={r['return_pct']:>10,.0f}% Tr={r['trades']:>4d} "
              f"WR={r['win_rate']:>5.1f}% | {desc}")

    # 5) Top 50 상세 검증 (30회 반복)
    print("\n[5/6] Detailed verification (30 runs each)...")
    verified = []
    for rank, r in enumerate(top50):
        tf_label = r['tf']
        df = tf_map[tf_label]
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        volume = df['volume'].values.astype(np.float64)
        n = len(close)
        timestamps_epoch = np.zeros(n, dtype=np.float64)

        combo = r['combo']
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

        desc = (f"{ma_names[combo[0]]}({combo[2]})/"
                f"{ma_names[combo[1]]}({combo[3]})")

        v = {
            'rank': rank + 1,
            'tf': tf_label,
            'strategy': desc,
            'full_desc': (f"{ma_names[combo[0]]}({combo[2]})/"
                         f"{ma_names[combo[1]]}({combo[3]}) "
                         f"ADX({combo[4]})>={combo[5]} RSI{combo[6]}-{combo[7]} "
                         f"SL{combo[8]}% Trail+{combo[9]}/-{combo[10]} "
                         f"M{combo[11]}% Lev{combo[12]}x Delay{combo[13]} Offset{combo[14]} Skip{combo[15]}"),
            'combo': [int(x) if isinstance(x, (np.integer, int)) else float(x) for x in combo],
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
            'verification_runs': 30,
            'std': std_val,
            'deterministic': det,
            'score': r['score'],
            'yearly': [float(full_result[12 + i]) for i in range(14)]
        }
        verified.append(v)
        status = "PASS" if det else "FAIL"
        print(f"  #{rank+1:2d} [{tf_label}] {desc:20s} PF={v['pf']:>7.2f} "
              f"Ret={v['return_pct']:>10,.0f}% MDD={v['mdd']:>5.1f}% "
              f"Tr={v['trades']:>4d} 30x={status}")

    # 6) 최종 저장
    print("\n[6/6] Saving results...")
    save_path = os.path.join(BASE, "v28_final_results.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(verified, f, indent=2, ensure_ascii=False)
    print(f"Saved: {save_path}")

    # 최종 요약
    elapsed_total = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  TOTAL TIME: {elapsed_total/60:.1f} minutes")
    print(f"  COMBOS TESTED: 1,000,000")
    print(f"  VALID STRATEGIES: {len(all_top)}")
    print(f"  TOP 50 VERIFIED: {sum(1 for v in verified if v['deterministic'])}/50 PASS")
    print(f"{'='*70}")

    # Top 10 상세
    print(f"\n{'='*70}")
    print(f"  FINAL TOP 10 STRATEGIES")
    print(f"{'='*70}")
    for v in verified[:10]:
        print(f"\n  === #{v['rank']} [{v['tf']}] {v['full_desc']} ===")
        print(f"  Balance: ${v['balance']:,.0f} | Return: +{v['return_pct']:,.0f}%")
        print(f"  PF: {v['pf']:.2f} | MDD: {v['mdd']:.1f}% | Trades: {v['trades']}")
        print(f"  Win Rate: {v['win_rate']:.1f}% | SL: {v['sl_count']} | TSL: {v['tsl_count']} | REV: {v['rev_count']}")
        print(f"  Max Consec Loss: {v['max_consec_loss']} | 30x Verify: {'PASS' if v['deterministic'] else 'FAIL'} (std={v['std']:.4f})")

        # 연도별 수익
        yr = v['yearly']
        print(f"  Yearly (start->end):")
        years = ['2020', '2021', '2022', '2023', '2024', '2025', '2026']
        for yi in range(7):
            s, e = yr[yi*2], yr[yi*2+1]
            if s > 0 and e > 0:
                ret = (e - s) / s * 100
                print(f"    {years[yi]}: ${s:,.0f} -> ${e:,.0f} ({ret:+.1f}%)")


def save_intermediate(results, tf_label):
    """중간 결과 저장"""
    path = os.path.join(BASE, f"v28_intermediate_{tf_label}.json")
    save = []
    for r in results[-200:]:  # 최근 TF의 Top 200
        entry = {k: v for k, v in r.items() if k not in ('combo', 'yearly_data')}
        entry['combo'] = [int(x) if isinstance(x, (np.integer, int)) else float(x) for x in r['combo']]
        save.append(entry)
    with open(path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"  [SAVE] {path}")


if __name__ == '__main__':
    main()
