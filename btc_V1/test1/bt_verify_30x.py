"""
v14.4 백테스트 30회 반복 검증
- 결정론적(Deterministic) 엔진 확인
- 코드봇(기본) 30회 + 코드봇(동일방향 스킵) 30회
"""

import time
import numpy as np
from bt_engine_v144 import prepare_data, BacktestEngine, CONFIG

DATA_PATH = r"D:\filesystem\futures\btc_V1\test1\btc_usdt_5m_merged.csv"
N = 30


def run_verification():
    print("=" * 80)
    print(f"  v14.4 Backtest x{N} Verification")
    print("=" * 80)

    # 데이터 1회만 로드
    df = prepare_data(DATA_PATH)

    # ============================================================
    # 1. 코드봇 (기본) 30회
    # ============================================================
    print(f"\n[1/2] Code Bot (basic) x {N}")
    print(f"  {'#':<4} {'Balance':>14} {'Return%':>12} {'PF':>8} {'MDD%':>8} {'Trades':>7} {'SL':>4} {'TSL':>4} {'REV':>4}")
    print(f"  {'-'*72}")

    results_basic = []
    t0 = time.time()

    for i in range(1, N + 1):
        engine = BacktestEngine()
        r = engine.run(df, skip_same_direction=False)
        results_basic.append(r)
        print(f"  #{i:<3} ${r['balance']:>12,.0f} {r['return_pct']:>+11.1f}% {r['pf']:>7.2f} {r['mdd']:>7.1f}% {r['trades']:>6} {r['sl_count']:>4} {r['tsl_count']:>4} {r['rev_count']:>4}")

    t1 = time.time()
    print(f"  Time: {t1 - t0:.1f}s ({(t1-t0)/N:.2f}s/run)")

    # 통계
    bal_arr = np.array([r['balance'] for r in results_basic])
    ret_arr = np.array([r['return_pct'] for r in results_basic])
    pf_arr = np.array([r['pf'] for r in results_basic])
    mdd_arr = np.array([r['mdd'] for r in results_basic])
    trades_arr = np.array([r['trades'] for r in results_basic])
    sl_arr = np.array([r['sl_count'] for r in results_basic])
    tsl_arr = np.array([r['tsl_count'] for r in results_basic])
    rev_arr = np.array([r['rev_count'] for r in results_basic])

    print(f"\n  [Statistics - Basic]")
    print(f"  {'Metric':<12} {'Mean':>14} {'Std':>14} {'Min':>14} {'Max':>14}")
    print(f"  {'-'*68}")
    print(f"  {'Balance':<12} ${bal_arr.mean():>12,.2f} {bal_arr.std():>14.6f} ${bal_arr.min():>12,.2f} ${bal_arr.max():>12,.2f}")
    print(f"  {'Return%':<12} {ret_arr.mean():>+13.2f}% {ret_arr.std():>14.6f} {ret_arr.min():>+13.2f}% {ret_arr.max():>+13.2f}%")
    print(f"  {'PF':<12} {pf_arr.mean():>14.6f} {pf_arr.std():>14.6f} {pf_arr.min():>14.6f} {pf_arr.max():>14.6f}")
    print(f"  {'MDD%':<12} {mdd_arr.mean():>13.2f}% {mdd_arr.std():>14.6f} {mdd_arr.min():>13.2f}% {mdd_arr.max():>13.2f}%")
    print(f"  {'Trades':<12} {trades_arr.mean():>14.2f} {trades_arr.std():>14.6f} {trades_arr.min():>14} {trades_arr.max():>14}")
    print(f"  {'SL':<12} {sl_arr.mean():>14.2f} {sl_arr.std():>14.6f} {sl_arr.min():>14} {sl_arr.max():>14}")
    print(f"  {'TSL':<12} {tsl_arr.mean():>14.2f} {tsl_arr.std():>14.6f} {tsl_arr.min():>14} {tsl_arr.max():>14}")
    print(f"  {'REV':<12} {rev_arr.mean():>14.2f} {rev_arr.std():>14.6f} {rev_arr.min():>14} {rev_arr.max():>14}")

    basic_identical = bool(bal_arr.std() < 1e-10 and trades_arr.std() < 1e-10)
    print(f"\n  ==> {N}회 모두 동일: {'PASS' if basic_identical else 'FAIL'}")

    # ============================================================
    # 2. 코드봇 (동일방향 스킵) 30회
    # ============================================================
    print(f"\n[2/2] Code Bot (skip same dir) x {N}")
    print(f"  {'#':<4} {'Balance':>14} {'Return%':>12} {'PF':>8} {'MDD%':>8} {'Trades':>7} {'SL':>4} {'TSL':>4} {'REV':>4}")
    print(f"  {'-'*72}")

    results_skip = []
    t2 = time.time()

    for i in range(1, N + 1):
        engine = BacktestEngine()
        r = engine.run(df, skip_same_direction=True)
        results_skip.append(r)
        print(f"  #{i:<3} ${r['balance']:>12,.0f} {r['return_pct']:>+11.1f}% {r['pf']:>7.2f} {r['mdd']:>7.1f}% {r['trades']:>6} {r['sl_count']:>4} {r['tsl_count']:>4} {r['rev_count']:>4}")

    t3 = time.time()
    print(f"  Time: {t3 - t2:.1f}s ({(t3-t2)/N:.2f}s/run)")

    # 통계
    bal_s = np.array([r['balance'] for r in results_skip])
    ret_s = np.array([r['return_pct'] for r in results_skip])
    pf_s = np.array([r['pf'] for r in results_skip])
    mdd_s = np.array([r['mdd'] for r in results_skip])
    trades_s = np.array([r['trades'] for r in results_skip])
    sl_s = np.array([r['sl_count'] for r in results_skip])
    tsl_s = np.array([r['tsl_count'] for r in results_skip])
    rev_s = np.array([r['rev_count'] for r in results_skip])

    print(f"\n  [Statistics - Skip Same Dir]")
    print(f"  {'Metric':<12} {'Mean':>14} {'Std':>14} {'Min':>14} {'Max':>14}")
    print(f"  {'-'*68}")
    print(f"  {'Balance':<12} ${bal_s.mean():>12,.2f} {bal_s.std():>14.6f} ${bal_s.min():>12,.2f} ${bal_s.max():>12,.2f}")
    print(f"  {'Return%':<12} {ret_s.mean():>+13.2f}% {ret_s.std():>14.6f} {ret_s.min():>+13.2f}% {ret_s.max():>+13.2f}%")
    print(f"  {'PF':<12} {pf_s.mean():>14.6f} {pf_s.std():>14.6f} {pf_s.min():>14.6f} {pf_s.max():>14.6f}")
    print(f"  {'MDD%':<12} {mdd_s.mean():>13.2f}% {mdd_s.std():>14.6f} {mdd_s.min():>13.2f}% {mdd_s.max():>13.2f}%")
    print(f"  {'Trades':<12} {trades_s.mean():>14.2f} {trades_s.std():>14.6f} {trades_s.min():>14} {trades_s.max():>14}")
    print(f"  {'SL':<12} {sl_s.mean():>14.2f} {sl_s.std():>14.6f} {sl_s.min():>14} {sl_s.max():>14}")
    print(f"  {'TSL':<12} {tsl_s.mean():>14.2f} {tsl_s.std():>14.6f} {tsl_s.min():>14} {tsl_s.max():>14}")
    print(f"  {'REV':<12} {rev_s.mean():>14.2f} {rev_s.std():>14.6f} {rev_s.min():>14} {rev_s.max():>14}")

    skip_identical = bool(bal_s.std() < 1e-10 and trades_s.std() < 1e-10)
    print(f"\n  ==> {N}회 모두 동일: {'PASS' if skip_identical else 'FAIL'}")

    # ============================================================
    # 최종 결과
    # ============================================================
    print(f"\n{'='*80}")
    print(f"  FINAL VERIFICATION RESULT")
    print(f"{'='*80}")
    print(f"  Code Bot (basic)          x{N}: {'PASS - ALL IDENTICAL' if basic_identical else 'FAIL - VARIANCE DETECTED'}")
    print(f"    Balance: ${bal_arr[0]:,.0f} | Trades: {int(trades_arr[0])} | PF: {pf_arr[0]:.2f} | MDD: {mdd_arr[0]:.1f}%")
    print(f"    Std: balance={bal_arr.std():.6f}, PF={pf_arr.std():.6f}, MDD={mdd_arr.std():.6f}, trades={trades_arr.std():.6f}")
    print()
    print(f"  Code Bot (skip same dir)  x{N}: {'PASS - ALL IDENTICAL' if skip_identical else 'FAIL - VARIANCE DETECTED'}")
    print(f"    Balance: ${bal_s[0]:,.0f} | Trades: {int(trades_s[0])} | PF: {pf_s[0]:.2f} | MDD: {mdd_s[0]:.1f}%")
    print(f"    Std: balance={bal_s.std():.6f}, PF={pf_s.std():.6f}, MDD={mdd_s.std():.6f}, trades={trades_s.std():.6f}")
    print()

    total_time = t3 - t0
    print(f"  Total: {total_time:.1f}s ({N*2} runs, {total_time/(N*2):.2f}s/run)")
    print(f"  Deterministic Engine: {'CONFIRMED' if (basic_identical and skip_identical) else 'NOT CONFIRMED'}")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_verification()
