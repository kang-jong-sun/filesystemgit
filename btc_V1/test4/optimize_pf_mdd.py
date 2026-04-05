"""
PF 극대화 + MDD 최소화 중심 재최적화
기존 v15.5(감사보고서) PF=22.5, MDD=20.1% 전략 기반 확장 탐색

핵심 방향:
1. 높은 ADX (40, 45) → 초고선택성
2. 높은 Trail 활성화 (15~25%) → 대형 추세만 포착
3. 멀티TF (5m, 10m, 15m, 30m, 1h)
4. 다양한 MA 조합 + 레버리지 (10x, 15x)
5. 타이트 SL (3~5%) → MDD 감소
"""
import numpy as np, pandas as pd, json, os, time
from bt_fast import load_5m_data, build_mtf, IndicatorCache, run_backtest

BASE = r'D:\filesystem\futures\btc_V1\test4'

def pf_mdd_score(r):
    """PF와 MDD 중심 스코어링"""
    if r is None or r.get('ret', 0) <= 0: return 0
    pf = r.get('pf', 0)
    mdd = r.get('mdd', 100)
    ret = r.get('ret', 0)
    fl = r.get('liq', 0)
    tr = r.get('trades', 0)
    if fl > 0: return 0  # FL 허용 안 함

    # PF 보너스 (핵심)
    pf_b = pf ** 1.5  # PF에 강한 가중치

    # MDD 보너스
    mdd_b = (100 - mdd) / 50  # MDD 낮을수록 높은 점수

    # 수익 보너스 (log 스케일)
    ret_b = np.log1p(ret / 100)

    # 거래수 보너스 (최소 10회, 30회 이상 보너스)
    if tr < 10: return 0
    tr_b = min(tr / 30, 2.0)

    # 최근 연도 가중치
    yr = r.get('yr', {})
    recent = [yr.get(str(y), 0) for y in range(2023, 2027) if str(y) in yr]
    yr_b = 1.5 if recent and np.mean(recent) > 50 else (1.2 if recent and np.mean(recent) > 0 else 1.0)

    return pf_b * mdd_b * ret_b * tr_b * yr_b


def main():
    print("=" * 80)
    print("  PF 극대화 + MDD 최소화 재최적화")
    print("=" * 80)

    df5 = load_5m_data(BASE)
    mtf = build_mtf(df5)
    cache = IndicatorCache(mtf)

    # ============================================================
    # Phase A: 초고선택성 진입 탐색
    # ============================================================
    print("\n" + "=" * 80)
    print("  Phase A: 초고선택성 진입 (ADX 40~50, Trail 10~25%)")
    print("=" * 80)

    configs = []
    tfs = ['5m', '10m', '15m', '30m', '1h']
    ma_types = ['ema', 'hma', 'wma', 'dema']
    fast_lens = [3, 5, 7, 10, 14, 21]
    slow_lens = [100, 150, 200, 250]
    adx_periods = [14, 20]
    adx_mins = [35, 40, 45, 50]
    rsi_ranges = [(30, 65), (35, 65), (35, 70), (40, 75), (30, 75)]
    sl_pcts = [0.03, 0.04, 0.05, 0.06, 0.07]
    trail_acts = [0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    trail_pcts = [0.03, 0.04, 0.05, 0.06]
    leverages = [10, 15]
    margins = [0.20, 0.25, 0.30]

    for tf in tfs:
        for ma_t in ma_types:
            for fl in fast_lens:
                for sl_len in slow_lens:
                    if fl >= sl_len: continue
                    for adx_p in adx_periods:
                        for adx_m in adx_mins:
                            for rsi_lo, rsi_hi in rsi_ranges:
                                for slp in sl_pcts:
                                    for ta in trail_acts:
                                        for tp in trail_pcts:
                                            if tp >= ta: continue
                                            for lev in leverages:
                                                for mg in margins:
                                                    # FL 방지: SL < 강제청산거리
                                                    if slp >= 1.0 / lev - 0.01: continue
                                                    configs.append({
                                                        'timeframe': tf,
                                                        'ma_fast_type': ma_t, 'ma_slow_type': 'ema',
                                                        'ma_fast': fl, 'ma_slow': sl_len,
                                                        'adx_period': adx_p, 'adx_min': adx_m,
                                                        'rsi_period': 14, 'rsi_min': rsi_lo, 'rsi_max': rsi_hi,
                                                        'sl_pct': slp,
                                                        'trail_activate': ta, 'trail_pct': tp,
                                                        'leverage': lev, 'margin_normal': mg, 'margin_reduced': mg/2,
                                                        'fee_rate': 0.0004, 'initial_capital': 3000.0,
                                                    })

    total = len(configs)
    sample_n = min(total, 50000)
    if total > sample_n:
        np.random.seed(42)
        idx = np.random.choice(total, sample_n, replace=False)
        configs = [configs[i] for i in idx]

    print(f"  전체 공간: {total:,} → 샘플: {len(configs):,}")

    results = []
    t0 = time.time()
    for i, cfg in enumerate(configs):
        r = run_backtest(cache, cfg['timeframe'], cfg)
        if r and r['trades'] >= 8 and r.get('liq', 0) == 0:
            r['_score'] = pf_mdd_score(r)
            if r['_score'] > 0:
                results.append(r)
        if (i + 1) % 10000 == 0:
            print(f"    {i+1:,}/{len(configs):,} ({time.time()-t0:.0f}s) 유효: {len(results):,}")

    elapsed = time.time() - t0
    print(f"  완료: {len(results):,}개 유효 ({elapsed:.0f}s)")

    # PF 기준 정렬
    results.sort(key=lambda x: x['pf'], reverse=True)
    print(f"\n  === PF Top 30 (FL=0) ===")
    print(f"  {'#':>3} {'잔액':>12} {'수익률':>10} {'PF':>7} {'MDD':>6} {'TR':>4} {'WR':>5} 설정")
    for i, r in enumerate(results[:30]):
        print(f"  {i+1:>3} ${r['bal']:>10,.0f} {r['ret']:>+9.1f}% {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['cfg']}")

    # MDD 기준 정렬 (PF >= 3)
    mdd_filtered = [r for r in results if r['pf'] >= 3.0]
    mdd_filtered.sort(key=lambda x: x['mdd'])
    print(f"\n  === MDD Top 20 (PF >= 3, FL=0) ===")
    print(f"  {'#':>3} {'잔액':>12} {'수익률':>10} {'PF':>7} {'MDD':>6} {'TR':>4} {'WR':>5} 설정")
    for i, r in enumerate(mdd_filtered[:20]):
        print(f"  {i+1:>3} ${r['bal']:>10,.0f} {r['ret']:>+9.1f}% {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['cfg']}")

    # 스코어 기준 정렬 (균형)
    results.sort(key=lambda x: x['_score'], reverse=True)
    print(f"\n  === 균형 스코어 Top 20 (PF×MDD×수익 최적) ===")
    print(f"  {'#':>3} {'잔액':>12} {'수익률':>10} {'PF':>7} {'MDD':>6} {'TR':>4} {'WR':>5} {'점수':>7} 설정")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:>3} ${r['bal']:>10,.0f} {r['ret']:>+9.1f}% {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['_score']:>6.1f} {r['cfg']}")

    # ============================================================
    # Phase B: 보호 메커니즘 추가 (Top 20에 대해)
    # ============================================================
    print(f"\n{'='*80}")
    print("  Phase B: 보호 + 사이징 최적화")
    print(f"{'='*80}")

    top_for_protection = results[:20]
    ml_opts = [0, -0.10, -0.15, -0.20, -0.25]
    dd_opts = [0, -0.25, -0.30, -0.40]
    mg_opts = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    p2_results = []
    combo = 0
    t0 = time.time()

    for base_r in top_for_protection:
        cfg_str = base_r['cfg']
        parts = cfg_str.split('|')
        tf = parts[0].strip()
        ma_part = parts[1].strip()
        ma_type = ma_part.split('(')[0].strip()
        lens = ma_part.split('(')[1].rstrip(')').split('/')
        ma_f = int(lens[0]); ma_s = int(lens[1])
        adx_part = parts[2].strip()
        adx_p_str = adx_part.split('>=')[0].replace('A','')
        adx_p = int(adx_p_str)
        adx_m = float(adx_part.split('>=')[1])
        rsi_part = parts[3].strip()
        rsi_p_str = rsi_part.split(':')[0].replace('R','')
        rsi_vals = rsi_part.split(':')[1].split('-')
        rsi_min = float(rsi_vals[0]); rsi_max = float(rsi_vals[1])
        sl_part = parts[4].strip()
        slp = float(sl_part.replace('SL', ''))
        trail_part = parts[5].strip()
        ta_vals = trail_part.replace('TA', '').split('/')
        ta = float(ta_vals[0]); tp = float(ta_vals[1])
        lev_part = parts[6].strip()
        lev = int(lev_part.split('x')[0].replace('L',''))

        for mg in mg_opts:
            if slp >= 1.0 / lev - 0.01: continue
            for ml in ml_opts:
                for dd in dd_opts:
                    cfg = {
                        'timeframe': tf,
                        'ma_fast_type': ma_type, 'ma_slow_type': 'ema',
                        'ma_fast': ma_f, 'ma_slow': ma_s,
                        'adx_period': adx_p, 'adx_min': adx_m,
                        'rsi_period': 14, 'rsi_min': rsi_min, 'rsi_max': rsi_max,
                        'sl_pct': slp, 'trail_activate': ta, 'trail_pct': tp,
                        'leverage': lev, 'margin_normal': mg, 'margin_reduced': mg/2,
                        'monthly_loss_limit': ml, 'dd_threshold': dd,
                        'fee_rate': 0.0004, 'initial_capital': 3000.0,
                    }
                    r = run_backtest(cache, tf, cfg)
                    if r and r['trades'] >= 8 and r.get('liq', 0) == 0:
                        r['_score'] = pf_mdd_score(r)
                        if r['_score'] > 0:
                            p2_results.append(r)
                    combo += 1

    elapsed = time.time() - t0
    print(f"  {combo:,}개 조합, {len(p2_results):,}개 유효 ({elapsed:.0f}s)")

    # 최종 정렬
    # 1. PF Top
    p2_results.sort(key=lambda x: x['pf'], reverse=True)
    print(f"\n  === 최종 PF Top 20 ===")
    print(f"  {'#':>3} {'잔액':>14} {'수익률':>12} {'PF':>7} {'MDD':>6} {'TR':>4} {'WR':>5} 설정")
    for i, r in enumerate(p2_results[:20]):
        print(f"  {i+1:>3} ${r['bal']:>12,.0f} {r['ret']:>+11.1f}% {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['cfg']}")

    # 2. MDD Top (PF >= 5)
    mdd_f = [r for r in p2_results if r['pf'] >= 5.0]
    mdd_f.sort(key=lambda x: x['mdd'])
    print(f"\n  === 최종 MDD Top 20 (PF >= 5) ===")
    print(f"  {'#':>3} {'잔액':>14} {'수익률':>12} {'PF':>7} {'MDD':>6} {'TR':>4} {'WR':>5} 설정")
    for i, r in enumerate(mdd_f[:20]):
        print(f"  {i+1:>3} ${r['bal']:>12,.0f} {r['ret']:>+11.1f}% {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['cfg']}")

    # 3. 균형 Top (PF >= 5, MDD <= 35%)
    bal_f = [r for r in p2_results if r['pf'] >= 5.0 and r['mdd'] <= 35]
    bal_f.sort(key=lambda x: x['_score'], reverse=True)
    print(f"\n  === PF>=5 + MDD<=35% 조건 Top 20 ===")
    print(f"  {'#':>3} {'잔액':>14} {'수익률':>12} {'PF':>7} {'MDD':>6} {'TR':>4} {'WR':>5} {'점수':>7} 설정")
    for i, r in enumerate(bal_f[:20]):
        print(f"  {i+1:>3} ${r['bal']:>12,.0f} {r['ret']:>+11.1f}% {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['_score']:>6.1f} {r['cfg']}")

    # 4. PF >= 8 탐색
    pf8 = [r for r in p2_results if r['pf'] >= 8.0]
    pf8.sort(key=lambda x: x['ret'], reverse=True)
    print(f"\n  === PF >= 8 전략 ({len(pf8)}개) ===")
    print(f"  {'#':>3} {'잔액':>14} {'수익률':>12} {'PF':>7} {'MDD':>6} {'TR':>4} {'WR':>5} 설정")
    for i, r in enumerate(pf8[:20]):
        print(f"  {i+1:>3} ${r['bal']:>12,.0f} {r['ret']:>+11.1f}% {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['cfg']}")

    # 저장
    save_data = {
        'total_combos': len(configs) + combo,
        'pf_top20': [{'bal':r['bal'],'ret':r['ret'],'pf':r['pf'],'mdd':r['mdd'],'fl':r['liq'],'tr':r['trades'],'wr':r['wr'],'cfg':r['cfg'],'yr':r.get('yr',{})} for r in p2_results[:20]],
        'mdd_pf5_top20': [{'bal':r['bal'],'ret':r['ret'],'pf':r['pf'],'mdd':r['mdd'],'fl':r['liq'],'tr':r['trades'],'wr':r['wr'],'cfg':r['cfg'],'yr':r.get('yr',{})} for r in mdd_f[:20]],
        'balanced_top20': [{'bal':r['bal'],'ret':r['ret'],'pf':r['pf'],'mdd':r['mdd'],'fl':r['liq'],'tr':r['trades'],'wr':r['wr'],'cfg':r['cfg'],'yr':r.get('yr',{})} for r in bal_f[:20]],
        'pf8_all': [{'bal':r['bal'],'ret':r['ret'],'pf':r['pf'],'mdd':r['mdd'],'fl':r['liq'],'tr':r['trades'],'wr':r['wr'],'cfg':r['cfg'],'yr':r.get('yr',{})} for r in pf8[:20]],
    }
    outf = os.path.join(BASE, 'pf_mdd_optimization_results.json')
    with open(outf, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {outf}")
    print(f"총 조합: {len(configs) + combo:,}")

if __name__ == '__main__':
    main()
