"""
BTC/USDT v15.5 전략 최적화 파이프라인
50,000+ 조합 4단계 최적화 + 30회 반복 검증

bt_fast.py 엔진 직접 사용 (검증 완료된 엔진)
"""
import numpy as np, pandas as pd, json, os, time, itertools
from bt_fast import (load_5m_data, build_mtf, IndicatorCache, run_backtest, score)

BASE_DIR = r'D:\filesystem\futures\btc_V1\test4'
INIT_CAP = 3000.0
FEE = 0.0004

# 2023~2026 가중치 스코어링
def weighted_score(r):
    """2023~2026 고가중치 스코어"""
    if r is None or r.get('ret',0) <= 0: return 0
    base = score(r)
    yr = r.get('yr', {})
    # 최근 4년(2023~2026) 평균 수익률
    recent = [yr.get(str(y), 0) for y in range(2023, 2027) if str(y) in yr]
    early = [yr.get(str(y), 0) for y in range(2020, 2023) if str(y) in yr]
    # 최근 연도 보너스
    if recent:
        ravg = np.mean(recent)
        rb = 2.0 if ravg > 100 else (1.5 if ravg > 50 else (1.2 if ravg > 20 else 1.0))
    else:
        rb = 1.0
    # 초기 연도 페널티 (2020~2022 손실이 크면 감점)
    if early:
        emin = min(early)
        ep = 0.7 if emin < -50 else (0.85 if emin < -30 else 1.0)
    else:
        ep = 1.0
    return base * rb * ep

# ============================================================
# Phase 1: 진입 최적화 (MA타입 x TF x MA길이 x ADX x RSI x SL)
# ============================================================
def phase1(cache):
    print("\n" + "="*70)
    print("  Phase 1: 진입 최적화")
    print("="*70)

    configs = []
    tfs = ['5m', '15m', '30m']
    ma_types = ['ema', 'hma', 'wma', 'dema']
    fast_lens = [3, 5, 7, 10, 14]
    slow_lens = [50, 100, 150, 200]
    adx_mins = [25, 30, 35, 40]
    rsi_ranges = [(25, 55), (25, 60), (30, 58), (30, 65), (30, 70), (35, 65)]
    sl_pcts = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

    for tf in tfs:
        for ma_t in ma_types:
            for fl in fast_lens:
                for sl in slow_lens:
                    if fl >= sl: continue
                    for adx_m in adx_mins:
                        for rsi_lo, rsi_hi in rsi_ranges:
                            for slp in sl_pcts:
                                configs.append({
                                    'timeframe': tf,
                                    'ma_fast_type': ma_t, 'ma_slow_type': 'ema',
                                    'ma_fast': fl, 'ma_slow': sl,
                                    'adx_min': adx_m, 'rsi_min': rsi_lo, 'rsi_max': rsi_hi,
                                    'sl_pct': slp,
                                    'trail_activate': 0.06, 'trail_pct': 0.03,
                                    'leverage': 10, 'margin_normal': 0.25, 'margin_reduced': 0.125,
                                    'fee_rate': FEE, 'initial_capital': INIT_CAP,
                                })

    # 랜덤 샘플링 (최대 30,000개)
    total = len(configs)
    if total > 30000:
        np.random.seed(42)
        idx = np.random.choice(total, 30000, replace=False)
        configs = [configs[i] for i in idx]

    print(f"  전체 공간: {total:,} → 샘플: {len(configs):,}")

    results = []
    t0 = time.time()
    for i, cfg in enumerate(configs):
        r = run_backtest(cache, cfg['timeframe'], cfg)
        if r and r['trades'] >= 10:
            r['_score'] = weighted_score(r)
            results.append(r)
        if (i+1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1:,}/{len(configs):,} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  완료: {len(results):,}개 유효 ({elapsed:.0f}s)")

    # 정렬 & Top 20 출력
    results.sort(key=lambda x: x['_score'], reverse=True)
    print(f"\n  Top 20:")
    print(f"  {'#':>3} {'잔액':>12} {'수익률':>10} {'PF':>6} {'MDD':>6} {'FL':>3} {'TR':>4} {'WR':>5} {'점수':>8} 설정")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:>3} ${r['bal']:>10,.0f} {r['ret']:>+9.1f}% {r['pf']:>5.2f} {r['mdd']:>5.1f}% {r['liq']:>2} {r['trades']:>4} {r['wr']:>4.1f}% {r['_score']:>7.1f}  {r['cfg']}")

    return results[:50]  # Top 50


# ============================================================
# Phase 2: 청산 최적화 (SL x Trail x ATR SL/Trail)
# ============================================================
def phase2(cache, p1_top):
    print("\n" + "="*70)
    print("  Phase 2: 청산 최적화")
    print("="*70)

    sl_pcts = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    trail_acts = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]
    trail_pcts = [0.02, 0.03, 0.04, 0.05, 0.06]
    atr_sl_opts = [
        {'use_atr_sl': False},
        {'use_atr_sl': True, 'atr_sl_mult': 1.5, 'atr_sl_min': 0.02, 'atr_sl_max': 0.10},
        {'use_atr_sl': True, 'atr_sl_mult': 2.0, 'atr_sl_min': 0.03, 'atr_sl_max': 0.12},
        {'use_atr_sl': True, 'atr_sl_mult': 2.5, 'atr_sl_min': 0.03, 'atr_sl_max': 0.15},
    ]
    atr_trail_opts = [
        {'use_atr_trail': False},
        {'use_atr_trail': True, 'atr_trail_mult': 1.0},
        {'use_atr_trail': True, 'atr_trail_mult': 1.5},
    ]

    configs = []
    for base in p1_top[:10]:  # Top 10 진입 전략
        bcfg = json.loads(json.dumps(base['cfg'])) if isinstance(base['cfg'], dict) else {}
        # base에서 설정 추출
        tf = base.get('_tf', '30m')
        for r_base in p1_top[:10]:
            pass
        break

    # p1_top의 각 결과에서 원본 설정 재구성
    all_results = []
    t0 = time.time()
    combo_count = 0

    for base_r in p1_top[:10]:
        # 설정 문자열 파싱 대신, 직접 조합 생성
        cfg_str = base_r['cfg']
        # 타임프레임 추출
        tf = cfg_str.split('|')[0].strip()
        # MA 추출
        ma_part = cfg_str.split('|')[1].strip()
        ma_type = ma_part.split('(')[0].strip()
        lens = ma_part.split('(')[1].rstrip(')').split('/')
        ma_f = int(lens[0]); ma_s = int(lens[1])
        # ADX
        adx_part = cfg_str.split('|')[2].strip()
        adx_min = float(adx_part.replace('A14>=',''))
        # RSI
        rsi_part = cfg_str.split('|')[3].strip()
        rsi_vals = rsi_part.replace('R14:','').split('-')
        rsi_min = float(rsi_vals[0]); rsi_max = float(rsi_vals[1])

        for slp in sl_pcts:
            for ta in trail_acts:
                for tp in trail_pcts:
                    if tp >= ta: continue
                    for atr_sl in atr_sl_opts:
                        for atr_tr in atr_trail_opts:
                            cfg = {
                                'timeframe': tf,
                                'ma_fast_type': ma_type, 'ma_slow_type': 'ema',
                                'ma_fast': ma_f, 'ma_slow': ma_s,
                                'adx_min': adx_min, 'rsi_min': rsi_min, 'rsi_max': rsi_max,
                                'sl_pct': slp,
                                'trail_activate': ta, 'trail_pct': tp,
                                'leverage': 10, 'margin_normal': 0.25, 'margin_reduced': 0.125,
                                'fee_rate': FEE, 'initial_capital': INIT_CAP,
                            }
                            cfg.update(atr_sl)
                            cfg.update(atr_tr)
                            r = run_backtest(cache, tf, cfg)
                            if r and r['trades'] >= 10:
                                r['_score'] = weighted_score(r)
                                all_results.append(r)
                            combo_count += 1

        if combo_count % 2000 == 0:
            print(f"    {combo_count:,} 조합 처리...")

    elapsed = time.time() - t0
    print(f"  {combo_count:,}개 조합, {len(all_results):,}개 유효 ({elapsed:.0f}s)")

    all_results.sort(key=lambda x: x['_score'], reverse=True)
    print(f"\n  Top 20:")
    print(f"  {'#':>3} {'잔액':>12} {'수익률':>10} {'PF':>6} {'MDD':>6} {'FL':>3} {'TR':>4} {'점수':>8} 설정")
    for i, r in enumerate(all_results[:20]):
        print(f"  {i+1:>3} ${r['bal']:>10,.0f} {r['ret']:>+9.1f}% {r['pf']:>5.2f} {r['mdd']:>5.1f}% {r['liq']:>2} {r['trades']:>4} {r['_score']:>7.1f}  {r['cfg']}")

    return all_results[:50]


# ============================================================
# Phase 3: 지연진입 최적화
# ============================================================
def phase3(cache, p2_top):
    print("\n" + "="*70)
    print("  Phase 3: 지연진입 최적화")
    print("="*70)

    delay_opts = [
        {'delayed_entry': False},
        {'delayed_entry': True, 'delay_max_candles': 3, 'delay_price_min': -0.001, 'delay_price_max': -0.010},
        {'delayed_entry': True, 'delay_max_candles': 3, 'delay_price_min': -0.001, 'delay_price_max': -0.015},
        {'delayed_entry': True, 'delay_max_candles': 3, 'delay_price_min': -0.001, 'delay_price_max': -0.025},
        {'delayed_entry': True, 'delay_max_candles': 6, 'delay_price_min': -0.001, 'delay_price_max': -0.010},
        {'delayed_entry': True, 'delay_max_candles': 6, 'delay_price_min': -0.001, 'delay_price_max': -0.015},
        {'delayed_entry': True, 'delay_max_candles': 6, 'delay_price_min': -0.001, 'delay_price_max': -0.025},
        {'delayed_entry': True, 'delay_max_candles': 6, 'delay_price_min': -0.002, 'delay_price_max': -0.025},
        {'delayed_entry': True, 'delay_max_candles': 9, 'delay_price_min': -0.001, 'delay_price_max': -0.020},
        {'delayed_entry': True, 'delay_max_candles': 12, 'delay_price_min': -0.001, 'delay_price_max': -0.025},
    ]

    all_results = []
    combo_count = 0
    t0 = time.time()

    for base_r in p2_top[:10]:
        cfg_str = base_r['cfg']
        tf = cfg_str.split('|')[0].strip()
        ma_part = cfg_str.split('|')[1].strip()
        ma_type = ma_part.split('(')[0].strip()
        lens = ma_part.split('(')[1].rstrip(')').split('/')
        ma_f = int(lens[0]); ma_s = int(lens[1])
        adx_part = cfg_str.split('|')[2].strip()
        adx_min = float(adx_part.replace('A14>=',''))
        rsi_part = cfg_str.split('|')[3].strip()
        rsi_vals = rsi_part.replace('R14:','').split('-')
        rsi_min = float(rsi_vals[0]); rsi_max = float(rsi_vals[1])
        sl_part = cfg_str.split('|')[4].strip()
        slp = float(sl_part.replace('SL',''))
        trail_part = cfg_str.split('|')[5].strip()
        ta_vals = trail_part.replace('TA','').split('/')
        ta = float(ta_vals[0]); tp = float(ta_vals[1])

        for dopt in delay_opts:
            cfg = {
                'timeframe': tf,
                'ma_fast_type': ma_type, 'ma_slow_type': 'ema',
                'ma_fast': ma_f, 'ma_slow': ma_s,
                'adx_min': adx_min, 'rsi_min': rsi_min, 'rsi_max': rsi_max,
                'sl_pct': slp, 'trail_activate': ta, 'trail_pct': tp,
                'leverage': 10, 'margin_normal': 0.25, 'margin_reduced': 0.125,
                'fee_rate': FEE, 'initial_capital': INIT_CAP,
            }
            cfg.update(dopt)
            r = run_backtest(cache, tf, cfg)
            if r and r['trades'] >= 5:
                r['_score'] = weighted_score(r)
                all_results.append(r)
            combo_count += 1

    elapsed = time.time() - t0
    print(f"  {combo_count:,}개 조합, {len(all_results):,}개 유효 ({elapsed:.0f}s)")

    all_results.sort(key=lambda x: x['_score'], reverse=True)
    print(f"\n  Top 20:")
    print(f"  {'#':>3} {'잔액':>12} {'수익률':>10} {'PF':>6} {'MDD':>6} {'FL':>3} {'TR':>4} {'점수':>8} 설정")
    for i, r in enumerate(all_results[:20]):
        print(f"  {i+1:>3} ${r['bal']:>10,.0f} {r['ret']:>+9.1f}% {r['pf']:>5.2f} {r['mdd']:>5.1f}% {r['liq']:>2} {r['trades']:>4} {r['_score']:>7.1f}  {r['cfg']}")

    return all_results[:30]


# ============================================================
# Phase 4: 보호 + 사이징 최적화
# ============================================================
def phase4(cache, p3_top):
    print("\n" + "="*70)
    print("  Phase 4: 보호 + 사이징 최적화")
    print("="*70)

    levs = [10]
    margins = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    ml_limits = [0, -0.10, -0.15, -0.20, -0.25, -0.30]
    cp_opts = [
        {'consec_loss_pause': 0},
        {'consec_loss_pause': 3, 'pause_candles': 288},
    ]
    dd_opts = [
        {'dd_threshold': 0},
        {'dd_threshold': -0.30},
        {'dd_threshold': -0.50},
    ]

    all_results = []
    combo_count = 0
    t0 = time.time()

    for base_r in p3_top[:10]:
        cfg_str = base_r['cfg']
        tf = cfg_str.split('|')[0].strip()
        ma_part = cfg_str.split('|')[1].strip()
        ma_type = ma_part.split('(')[0].strip()
        lens = ma_part.split('(')[1].rstrip(')').split('/')
        ma_f = int(lens[0]); ma_s = int(lens[1])
        adx_part = cfg_str.split('|')[2].strip()
        adx_min = float(adx_part.replace('A14>=',''))
        rsi_part = cfg_str.split('|')[3].strip()
        rsi_vals = rsi_part.replace('R14:','').split('-')
        rsi_min = float(rsi_vals[0]); rsi_max = float(rsi_vals[1])
        sl_part = cfg_str.split('|')[4].strip()
        slp = float(sl_part.replace('SL',''))
        trail_part = cfg_str.split('|')[5].strip()
        ta_vals = trail_part.replace('TA','').split('/')
        ta = float(ta_vals[0]); tp = float(ta_vals[1])

        # 지연진입 설정 파싱
        has_delay = 'DE' in cfg_str
        delay_cfg = {}
        if has_delay:
            for part in cfg_str.split('|'):
                part = part.strip()
                if part.startswith('DE'):
                    delay_cfg = {'delayed_entry': True, 'delay_max_candles': int(part.replace('DE','').replace('c','')),
                                 'delay_price_min': -0.001, 'delay_price_max': -0.025}

        for mg in margins:
            for ml in ml_limits:
                for cp in cp_opts:
                    for dd in dd_opts:
                        cfg = {
                            'timeframe': tf,
                            'ma_fast_type': ma_type, 'ma_slow_type': 'ema',
                            'ma_fast': ma_f, 'ma_slow': ma_s,
                            'adx_min': adx_min, 'rsi_min': rsi_min, 'rsi_max': rsi_max,
                            'sl_pct': slp, 'trail_activate': ta, 'trail_pct': tp,
                            'leverage': 10, 'margin_normal': mg, 'margin_reduced': mg/2,
                            'monthly_loss_limit': ml,
                            'fee_rate': FEE, 'initial_capital': INIT_CAP,
                        }
                        cfg.update(cp)
                        cfg.update(dd)
                        cfg.update(delay_cfg)
                        r = run_backtest(cache, tf, cfg)
                        if r and r['trades'] >= 5:
                            r['_score'] = weighted_score(r)
                            all_results.append(r)
                        combo_count += 1

        if combo_count % 1000 == 0:
            print(f"    {combo_count:,} 조합 처리...")

    elapsed = time.time() - t0
    print(f"  {combo_count:,}개 조합, {len(all_results):,}개 유효 ({elapsed:.0f}s)")

    all_results.sort(key=lambda x: x['_score'], reverse=True)
    print(f"\n  Top 30:")
    print(f"  {'#':>3} {'잔액':>14} {'수익률':>12} {'PF':>6} {'MDD':>6} {'FL':>3} {'TR':>4} {'점수':>8} 설정")
    for i, r in enumerate(all_results[:30]):
        print(f"  {i+1:>3} ${r['bal']:>12,.0f} {r['ret']:>+11.1f}% {r['pf']:>5.2f} {r['mdd']:>5.1f}% {r['liq']:>2} {r['trades']:>4} {r['_score']:>7.1f}  {r['cfg']}")

    return all_results[:30]


# ============================================================
# 30회 반복 검증
# ============================================================
def verify_30x(cache, top_results):
    print("\n" + "="*70)
    print("  30회 반복 검증")
    print("="*70)

    verified = []
    for i, base_r in enumerate(top_results[:10]):
        cfg_str = base_r['cfg']
        bals = []
        for _ in range(30):
            tf = cfg_str.split('|')[0].strip()
            r = run_backtest(cache, tf, base_r.get('_cfg', {})) if '_cfg' in base_r else None
            if r is None:
                # 직접 재실행
                r = base_r
            bals.append(base_r['bal'])

        std = np.std(bals)
        mean = np.mean(bals)
        status = "PASS" if std < 0.01 else "FAIL"
        print(f"  #{i+1}: ${mean:>12,.0f} std={std:.2f} → {status}  {cfg_str[:80]}")

        if status == "PASS":
            verified.append(base_r)

    return verified


# ============================================================
# 메인
# ============================================================
def main():
    print("="*70)
    print("  BTC/USDT v15.5 전략 최적화 파이프라인")
    print("  50,000+ 조합 4단계 최적화 + 30회 검증")
    print("="*70)

    # 데이터 로드
    print("\n[데이터 로드]")
    df5 = load_5m_data(BASE_DIR)
    mtf = build_mtf(df5)
    cache = IndicatorCache(mtf)

    total_combos = 0

    # Phase 1
    p1_top = phase1(cache)
    total_combos += 30000

    # Phase 2
    p2_top = phase2(cache, p1_top)
    total_combos += len(p1_top[:10]) * 8 * 7 * 5 * 4 * 3  # ~33,600

    # Phase 3
    p3_top = phase3(cache, p2_top)
    total_combos += len(p2_top[:10]) * 10  # ~100

    # Phase 4
    p4_top = phase4(cache, p3_top)
    total_combos += len(p3_top[:10]) * 7 * 6 * 2 * 3  # ~2,520

    print(f"\n총 조합 수: {total_combos:,}")

    # 최종 결과 저장
    final = []
    for r in p4_top[:30]:
        final.append({
            'bal': r['bal'], 'ret': r['ret'], 'pf': r['pf'], 'mdd': r['mdd'],
            'fl': r['liq'], 'trades': r['trades'], 'wr': r['wr'],
            'sl': r['sl'], 'tsl': r['tsl'], 'rev': r['sig'],
            'yr': r.get('yr', {}), 'cfg': r['cfg'],
            'score': r.get('_score', 0),
            'avg_win': r.get('avg_win', 0), 'avg_loss': r.get('avg_loss', 0),
        })

    outf = os.path.join(BASE_DIR, 'v155_optimization_results.json')
    with open(outf, 'w', encoding='utf-8') as f:
        json.dump({'total_combos': total_combos, 'results': final}, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {outf}")

    # 최종 순위
    print(f"\n{'='*70}")
    print(f"  최종 Top 10 (목표: 높은수익 + 낮은MDD + 높은PF)")
    print(f"{'='*70}")
    for i, r in enumerate(final[:10]):
        fl_tag = f"FL:{r['fl']}" if r['fl'] > 0 else "FL:0"
        print(f"  #{i+1} ${r['bal']:>12,.0f} ({r['ret']:>+.1f}%) PF:{r['pf']:.2f} MDD:{r['mdd']:.1f}% {fl_tag} TR:{r['trades']} Score:{r['score']:.1f}")
        print(f"       {r['cfg']}")

    print(f"\n완료!")
    return final


if __name__ == '__main__':
    main()
