"""v15.6 기획서용 상세 분석: PF>=8 전략 + 듀얼모델 조합"""
import numpy as np, json, time
from bt_fast import load_5m_data, build_mtf, IndicatorCache, run_backtest

BASE = r'D:\filesystem\futures\btc_V1\test4'

def detail(cache, name, cfg):
    """전략 상세 출력"""
    r = run_backtest(cache, cfg['timeframe'], cfg)
    if not r: print(f"  {name}: 실패"); return None
    print(f"\n  === {name} ===")
    print(f"  잔액: ${r['bal']:,.0f} ({r['ret']:+,.1f}%)")
    print(f"  PF: {r['pf']:.2f} | MDD: {r['mdd']:.1f}% | FL: {r['liq']}")
    print(f"  거래: {r['trades']}회 (SL:{r['sl']} TSL:{r['tsl']} REV:{r['sig']})")
    print(f"  승률: {r['wr']:.1f}% | 평균승: {r['avg_win']:.2f}% | 평균패: {r['avg_loss']:.2f}%")
    if r['avg_loss'] != 0:
        print(f"  손익비: {abs(r['avg_win']/r['avg_loss']):.2f}:1")
    print(f"  연도별:")
    for y in sorted(r.get('yr', {}).keys()):
        print(f"    {y}: {r['yr'][y]:>+8.1f}%")
    # 30회 검증
    bals = [run_backtest(cache, cfg['timeframe'], cfg)['bal'] for _ in range(30)]
    std = np.std(bals)
    print(f"  30회 검증: std={std:.4f} → {'PASS' if std < 0.01 else 'FAIL'}")
    return r

def sweep_margins(cache, name, base_cfg, margins):
    """마진 스윕"""
    print(f"\n  --- {name} 마진 스윕 ---")
    print(f"  {'마진':>6} {'잔액':>14} {'수익률':>10} {'PF':>7} {'MDD':>6} {'FL':>3} {'TR':>4}")
    for mg in margins:
        cfg = dict(base_cfg)
        cfg['margin_normal'] = mg
        cfg['margin_reduced'] = mg / 2
        r = run_backtest(cache, cfg['timeframe'], cfg)
        if r:
            print(f"  {mg:>5.0%} ${r['bal']:>12,.0f} {r['ret']:>+9.1f}% {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['liq']:>2} {r['trades']:>4}")

def main():
    print("=" * 80)
    print("  v15.6 기획서용 상세 분석")
    print("=" * 80)

    df5 = load_5m_data(BASE)
    mtf = build_mtf(df5)
    cache = IndicatorCache(mtf)

    # ============================================================
    # 1. 모델 A: PF 극대화 (15m EMA(3/150) ADX>=45)
    # ============================================================
    print("\n" + "=" * 80)
    print("  [모델 A] PF 극대화형")
    print("=" * 80)

    model_a = {
        'timeframe': '15m',
        'ma_fast_type': 'ema', 'ma_slow_type': 'ema',
        'ma_fast': 3, 'ma_slow': 150,
        'adx_period': 14, 'adx_min': 45,
        'rsi_period': 14, 'rsi_min': 35, 'rsi_max': 70,
        'sl_pct': 0.04, 'trail_activate': 0.10, 'trail_pct': 0.03,
        'leverage': 15, 'margin_normal': 0.40, 'margin_reduced': 0.20,
        'fee_rate': 0.0004, 'initial_capital': 3000.0,
    }
    ra = detail(cache, "모델A: 15m EMA(3/150) ADX>=45 Trail+10/-3 15x", model_a)

    # 모델A 마진 스윕
    sweep_margins(cache, "모델A", model_a, [0.15, 0.20, 0.25, 0.30, 0.35, 0.40])

    # 모델A 변형: ADX 40, 50
    for adx_v in [40, 45, 50]:
        cfg = dict(model_a); cfg['adx_min'] = adx_v
        r = run_backtest(cache, '15m', cfg)
        if r:
            print(f"  ADX>={adx_v}: ${r['bal']:>10,.0f} PF:{r['pf']:>6.2f} MDD:{r['mdd']:.1f}% TR:{r['trades']} WR:{r['wr']:.1f}%")

    # ============================================================
    # 2. 모델 A2: 초고PF (5m DEMA ADX>=50)
    # ============================================================
    print("\n" + "=" * 80)
    print("  [모델 A2] 초고PF형")
    print("=" * 80)

    model_a2 = {
        'timeframe': '5m',
        'ma_fast_type': 'dema', 'ma_slow_type': 'ema',
        'ma_fast': 7, 'ma_slow': 200,
        'adx_period': 14, 'adx_min': 50,
        'rsi_period': 14, 'rsi_min': 35, 'rsi_max': 65,
        'sl_pct': 0.07, 'trail_activate': 0.08, 'trail_pct': 0.05,
        'leverage': 10, 'margin_normal': 0.40, 'margin_reduced': 0.20,
        'fee_rate': 0.0004, 'initial_capital': 3000.0,
    }
    detail(cache, "모델A2: 5m DEMA(7/200) ADX>=50 Trail+8/-5", model_a2)

    # ============================================================
    # 3. 모델 B: 수익 극대화형 (30m EMA(3/200) ADX>=35)
    # ============================================================
    print("\n" + "=" * 80)
    print("  [모델 B] 수익 극대화형")
    print("=" * 80)

    model_b = {
        'timeframe': '30m',
        'ma_fast_type': 'ema', 'ma_slow_type': 'ema',
        'ma_fast': 3, 'ma_slow': 200,
        'adx_period': 14, 'adx_min': 35,
        'rsi_period': 14, 'rsi_min': 35, 'rsi_max': 65,
        'sl_pct': 0.07, 'trail_activate': 0.06, 'trail_pct': 0.05,
        'leverage': 10, 'margin_normal': 0.35, 'margin_reduced': 0.175,
        'monthly_loss_limit': -0.25, 'dd_threshold': -0.30,
        'fee_rate': 0.0004, 'initial_capital': 3000.0,
    }
    rb = detail(cache, "모델B: 30m EMA(3/200) ADX>=35 Trail+6/-5 M35", model_b)

    # ============================================================
    # 4. 추가 PF>=8 후보 탐색 (거래수 15+)
    # ============================================================
    print("\n" + "=" * 80)
    print("  [추가] PF>=8 + 거래 15회+ 탐색")
    print("=" * 80)

    tfs = ['5m', '10m', '15m', '30m']
    ma_types = ['ema', 'dema', 'hma', 'wma']
    fast_lens = [3, 5, 7, 10, 14, 21]
    slow_lens = [100, 150, 200, 250]
    adx_ps = [14, 20]
    adx_ms = [40, 45, 50]
    rsi_rs = [(30, 65), (35, 65), (35, 70), (40, 75)]
    sls = [0.03, 0.04, 0.05, 0.06, 0.07]
    tas = [0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    tps = [0.03, 0.04, 0.05, 0.06]
    levs = [10, 15]
    mgs = [0.25, 0.30, 0.35, 0.40]

    # 샘플링
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
    np.random.seed(123)
    if total > 60000:
        idx = np.random.choice(total, 60000, replace=False)
        configs = [configs[i] for i in idx]
    print(f"  전체: {total:,} → 샘플: {len(configs):,}")

    pf8_15plus = []
    t0 = time.time()
    for i, cfg in enumerate(configs):
        r = run_backtest(cache, cfg['timeframe'], cfg)
        if r and r['pf'] >= 8 and r['trades'] >= 15 and r.get('liq', 0) == 0:
            pf8_15plus.append(r)
        if (i+1) % 15000 == 0:
            print(f"    {i+1:,}/{len(configs):,} ({time.time()-t0:.0f}s) PF>=8&TR>=15: {len(pf8_15plus)}")

    print(f"  완료 ({time.time()-t0:.0f}s): PF>=8 & 거래>=15 → {len(pf8_15plus)}개")

    if pf8_15plus:
        pf8_15plus.sort(key=lambda x: x['bal'], reverse=True)
        print(f"\n  PF>=8 + 거래>=15 Top 15:")
        print(f"  {'#':>3} {'잔액':>12} {'PF':>7} {'MDD':>6} {'TR':>4} {'WR':>5} 설정")
        for i, r in enumerate(pf8_15plus[:15]):
            print(f"  {i+1:>3} ${r['bal']:>10,.0f} {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['cfg']}")

    # ============================================================
    # 5. 최종 비교표
    # ============================================================
    print(f"\n{'='*80}")
    print("  최종 비교표")
    print(f"{'='*80}")
    comps = {
        'v14.4': {'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
                   'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,
                   'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,
                   'margin_normal':0.25,'margin_reduced':0.125,'monthly_loss_limit':-0.20,
                   'fee_rate':0.0004,'initial_capital':3000.0},
        'v15.4': {'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
                   'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,
                   'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,
                   'margin_normal':0.40,'margin_reduced':0.20,'monthly_loss_limit':-0.30,
                   'fee_rate':0.0004,'initial_capital':3000.0},
    }
    print(f"  {'전략':<28} {'잔액':>14} {'PF':>7} {'MDD':>6} {'FL':>3} {'TR':>4} {'WR':>5}")
    print(f"  {'-'*72}")
    for name, cfg in comps.items():
        r = run_backtest(cache, cfg['timeframe'], cfg)
        print(f"  {name:<28} ${r['bal']:>12,.0f} {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['liq']:>2} {r['trades']:>4} {r['wr']:>4.1f}%")

    if ra:
        print(f"  {'모델A (PF형 15m)':<28} ${ra['bal']:>12,.0f} {ra['pf']:>6.2f} {ra['mdd']:>5.1f}% {ra['liq']:>2} {ra['trades']:>4} {ra['wr']:>4.1f}%")
    if rb:
        print(f"  {'모델B (수익형 30m)':<28} ${rb['bal']:>12,.0f} {rb['pf']:>6.2f} {rb['mdd']:>5.1f}% {rb['liq']:>2} {rb['trades']:>4} {rb['wr']:>4.1f}%")

    # 저장
    save = {'model_a': model_a, 'model_a2': model_a2, 'model_b': model_b}
    if pf8_15plus:
        save['pf8_15plus_top'] = [{'bal':r['bal'],'ret':r['ret'],'pf':r['pf'],'mdd':r['mdd'],
            'fl':r['liq'],'tr':r['trades'],'wr':r['wr'],'cfg':r['cfg'],'yr':r.get('yr',{})}
            for r in pf8_15plus[:15]]
    with open(f'{BASE}/v156_analysis.json', 'w') as f:
        json.dump(save, f, indent=2, ensure_ascii=False)

    print(f"\n분석 완료!")

if __name__ == '__main__':
    main()
