"""
PF vs 거래수 효율 프론티어 분석
핵심 질문: "통계적으로 유의한 거래수(30+, 50+, 80+)에서 PF 한계는?"
"""
import numpy as np, time, json
from bt_fast import load_5m_data, build_mtf, IndicatorCache, run_backtest

BASE = r'D:\filesystem\futures\btc_V1\test4'

def main():
    print("=" * 80)
    print("  PF vs 거래수 효율 프론티어 분석")
    print("  '통계적으로 유의한 거래수에서 PF 한계는?'")
    print("=" * 80)

    df5 = load_5m_data(BASE)
    mtf = build_mtf(df5)
    cache = IndicatorCache(mtf)

    # 광범위 탐색
    tfs = ['5m', '10m', '15m', '30m', '1h']
    ma_types = ['ema', 'dema', 'hma', 'wma']
    fast_lens = [3, 5, 7, 10, 14, 21]
    slow_lens = [50, 100, 150, 200, 250]
    adx_ps = [14, 20]
    adx_ms = [25, 30, 35, 40, 45]
    rsi_rs = [(25, 60), (30, 58), (30, 65), (35, 65), (35, 70), (40, 75)]
    sls = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    tas = [0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]
    tps = [0.03, 0.04, 0.05, 0.06]
    levs = [10, 15]
    mgs = [0.20, 0.25, 0.30, 0.35, 0.40]

    configs = []
    for tf in tfs:
        for mt in ma_types:
            for fl in fast_lens:
                for sl in slow_lens:
                    if fl >= sl: continue
                    for ap in adx_ps:
                        for am in adx_ms:
                            for rlo, rhi in rsi_rs:
                                for slp in sls:
                                    for ta in tas:
                                        for tp in tps:
                                            if tp >= ta: continue
                                            for lv in levs:
                                                for mg in mgs:
                                                    if slp >= 1.0/lv - 0.01: continue
                                                    configs.append({
                                                        'timeframe': tf,
                                                        'ma_fast_type': mt, 'ma_slow_type': 'ema',
                                                        'ma_fast': fl, 'ma_slow': sl,
                                                        'adx_period': ap, 'adx_min': am,
                                                        'rsi_period': 14, 'rsi_min': rlo, 'rsi_max': rhi,
                                                        'sl_pct': slp, 'trail_activate': ta, 'trail_pct': tp,
                                                        'leverage': lv, 'margin_normal': mg, 'margin_reduced': mg/2,
                                                        'fee_rate': 0.0004, 'initial_capital': 3000.0,
                                                    })

    total = len(configs)
    np.random.seed(777)
    N = min(total, 80000)
    if total > N:
        idx = np.random.choice(total, N, replace=False)
        configs = [configs[i] for i in idx]

    print(f"\n  전체 공간: {total:,} → 샘플: {len(configs):,}")

    # 실행
    all_results = []
    t0 = time.time()
    for i, cfg in enumerate(configs):
        r = run_backtest(cache, cfg['timeframe'], cfg)
        if r and r.get('liq', 0) == 0 and r['trades'] >= 10 and r.get('ret', 0) > 0:
            all_results.append(r)
        if (i+1) % 20000 == 0:
            print(f"    {i+1:,}/{len(configs):,} ({time.time()-t0:.0f}s) 유효: {len(all_results):,}")

    elapsed = time.time() - t0
    print(f"  완료: {len(all_results):,}개 유효 ({elapsed:.0f}s)")

    # ============================================================
    # 거래수 구간별 PF 프론티어
    # ============================================================
    thresholds = [
        (10, 19, "10~19 (부족)"),
        (20, 29, "20~29 (최소)"),
        (30, 49, "30~49 (유의)"),
        (50, 79, "50~79 (양호)"),
        (80, 150, "80~150 (우수)"),
        (150, 999, "150+ (최다)"),
    ]

    print(f"\n{'='*80}")
    print(f"  거래수 구간별 PF 상한 분석 (FL=0)")
    print(f"{'='*80}")

    for lo, hi, label in thresholds:
        subset = [r for r in all_results if lo <= r['trades'] <= hi]
        if not subset:
            print(f"\n  [{label}] — 데이터 없음")
            continue

        subset.sort(key=lambda x: x['pf'], reverse=True)
        print(f"\n  [{label}] {len(subset):,}개 전략")
        print(f"  {'#':>3} {'잔액':>12} {'PF':>7} {'MDD':>6} {'TR':>4} {'WR':>5} {'수익률':>10} 설정")
        for i, r in enumerate(subset[:5]):
            print(f"  {i+1:>3} ${r['bal']:>10,.0f} {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['ret']:>+9.1f}% {r['cfg']}")

    # ============================================================
    # PF >= 5 + 거래 30+ + MDD <= 40% 조건 (실전 가능)
    # ============================================================
    practical = [r for r in all_results if r['pf'] >= 5 and r['trades'] >= 30 and r['mdd'] <= 40]
    practical.sort(key=lambda x: x['pf'], reverse=True)

    print(f"\n{'='*80}")
    print(f"  실전 가능: PF>=5 + 거래30+ + MDD<=40% → {len(practical)}개")
    print(f"{'='*80}")
    if practical:
        print(f"  {'#':>3} {'잔액':>12} {'PF':>7} {'MDD':>6} {'TR':>4} {'WR':>5} {'수익률':>10} 설정")
        for i, r in enumerate(practical[:15]):
            print(f"  {i+1:>3} ${r['bal']:>10,.0f} {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['ret']:>+9.1f}% {r['cfg']}")
    else:
        print("  없음! PF>=5 + 30거래+ + MDD<=40%는 현 데이터에서 불가")

    # PF >= 4 + 거래 30+
    pf4_30 = [r for r in all_results if r['pf'] >= 4 and r['trades'] >= 30 and r['mdd'] <= 40]
    pf4_30.sort(key=lambda x: x['pf'], reverse=True)
    print(f"\n  PF>=4 + 거래30+ + MDD<=40% → {len(pf4_30)}개")
    if pf4_30:
        for i, r in enumerate(pf4_30[:10]):
            print(f"  {i+1:>3} ${r['bal']:>10,.0f} {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['ret']:>+9.1f}% {r['cfg']}")

    # PF >= 3 + 거래 50+
    pf3_50 = [r for r in all_results if r['pf'] >= 3 and r['trades'] >= 50 and r['mdd'] <= 40]
    pf3_50.sort(key=lambda x: x['pf'], reverse=True)
    print(f"\n  PF>=3 + 거래50+ + MDD<=40% → {len(pf3_50)}개")
    if pf3_50:
        for i, r in enumerate(pf3_50[:10]):
            print(f"  {i+1:>3} ${r['bal']:>10,.0f} {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['ret']:>+9.1f}% {r['cfg']}")

    # ============================================================
    # 최종 요약: 효율 프론티어
    # ============================================================
    print(f"\n{'='*80}")
    print(f"  ★ 효율 프론티어 요약 (FL=0, MDD<=40%)")
    print(f"{'='*80}")
    print(f"  {'거래수':>8} {'최대PF':>8} {'해당잔액':>12} {'해당MDD':>8} {'결론'}")
    print(f"  {'-'*60}")

    for min_tr in [10, 20, 30, 40, 50, 60, 80, 100]:
        sub = [r for r in all_results if r['trades'] >= min_tr and r['mdd'] <= 40]
        if sub:
            sub.sort(key=lambda x: x['pf'], reverse=True)
            best = sub[0]
            print(f"  {min_tr:>6}+ {best['pf']:>7.2f} ${best['bal']:>10,.0f} {best['mdd']:>7.1f}%")
        else:
            print(f"  {min_tr:>6}+     N/A")

    # 저장
    save = {}
    for lo, hi, label in thresholds:
        subset = [r for r in all_results if lo <= r['trades'] <= hi]
        subset.sort(key=lambda x: x['pf'], reverse=True)
        save[label] = [{'bal':r['bal'],'ret':r['ret'],'pf':r['pf'],'mdd':r['mdd'],'tr':r['trades'],
                         'wr':r['wr'],'cfg':r['cfg'],'yr':r.get('yr',{})} for r in subset[:5]]

    with open(f'{BASE}/pf_frontier_results.json', 'w') as f:
        json.dump(save, f, indent=2, ensure_ascii=False)

    print(f"\n분석 완료!")

if __name__ == '__main__':
    main()
