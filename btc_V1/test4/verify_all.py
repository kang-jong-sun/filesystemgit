"""
전체 기획서 일괄 30회 검증 + 베스트3 + 폐기10 선정
v10.1 ~ v15.5 (21개 버전)
"""
import sys, os, time, numpy as np, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_fast import load_5m_data, build_mtf, IndicatorCache, run_backtest

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# 모든 기획서 설정 (엔진에서 테스트 가능한 것만)
CONFIGS = {
    'v10.1': {'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
              'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
              'sl_pct':0.09,'trail_activate':0.10,'trail_pct':0.05,
              'leverage':10,'margin_normal':0.20,'margin_reduced':0.10},

    'v11.1': {'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
              'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
              'sl_pct':0.06,'trail_activate':0.08,'trail_pct':0.06,
              'leverage':10,'margin_normal':0.20,'margin_reduced':0.10},

    'v12.0': {'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
              'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
              'sl_pct':0.06,'trail_activate':0.10,'trail_pct':0.03,
              'leverage':10,'margin_normal':0.20,'margin_reduced':0.10},

    'v12.2': {'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
              'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
              'sl_pct':0.09,'trail_activate':0.08,'trail_pct':0.01,
              'leverage':10,'margin_normal':0.20,'margin_reduced':0.10},

    'v12.3': {'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
              'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
              'sl_pct':0.09,'trail_activate':0.08,'trail_pct':0.06,
              'leverage':10,'margin_normal':0.20,'margin_reduced':0.10},

    'v12.5': {'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
              'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
              'sl_pct':0.08,'trail_activate':0.09,'trail_pct':0.05,
              'leverage':10,'margin_normal':0.20,'margin_reduced':0.10},

    'v13.0': {'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
              'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
              'sl_pct':0.09,'trail_activate':0.08,'trail_pct':0.06,
              'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,
              'monthly_loss_limit':-0.15,'consec_loss_pause':3,'pause_candles':288,'dd_threshold':-0.30},

    'v13.3': {'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
              'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
              'sl_pct':0.12,'trail_activate':0.15,'trail_pct':0.01,
              'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,
              'monthly_loss_limit':-0.10,'consec_loss_pause':3,'pause_candles':288,'dd_threshold':-0.30},

    'v13.4A': {'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':5,'ma_slow':100,
               'adx_period':20,'adx_min':25,'rsi_period':14,'rsi_min':30,'rsi_max':58,
               'sl_pct':0.06,'trail_activate':0.06,'trail_pct':0.03,
               'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
               'monthly_loss_limit':-0.15,'dd_threshold':-0.30},

    'v13.5': {'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
              'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
              'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.06,
              'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,
              'monthly_loss_limit':-0.20,'consec_loss_pause':3,'pause_candles':288,'dd_threshold':-0.50},

    'v14.1': {'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
              'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
              'sl_pct':0.09,'trail_activate':0.10,'trail_pct':0.06,
              'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,
              'monthly_loss_limit':-0.25,'dd_threshold':-0.30},

    'v14.2': {'timeframe':'30m','ma_fast_type':'hma','ma_slow_type':'hma','ma_fast':7,'ma_slow':200,
              'adx_period':20,'adx_min':25,'rsi_period':14,'rsi_min':25,'rsi_max':65,
              'sl_pct':0.07,'trail_activate':0.10,'trail_pct':0.01,
              'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,
              'monthly_loss_limit':-0.15,'dd_threshold':-0.40},

    'v14.3': {'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
              'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
              'sl_pct':0.10,'trail_activate':0.08,'trail_pct':0.06,
              'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,
              'monthly_loss_limit':-0.25,'consec_loss_pause':3,'pause_candles':288,'dd_threshold':-0.30},

    'v14.4': {'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
              'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,
              'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
              'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
              'monthly_loss_limit':-0.20},

    'v15.1': {'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
              'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
              'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.06,
              'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,
              'monthly_loss_limit':-0.20,'consec_loss_pause':3,'pause_candles':288},

    'v15.2': {'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
              'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,
              'sl_pct':0.05,'trail_activate':0.06,'trail_pct':0.05,
              'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,
              'monthly_loss_limit':-0.15},

    'v15.3B': {'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
               'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,
               'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
               'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
               'monthly_loss_limit':-0.20},

    'v15.4': {'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
              'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,
              'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
              'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,
              'monthly_loss_limit':-0.30},

    'v15.5': {'timeframe':'15m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':21,'ma_slow':250,
              'adx_period':20,'adx_min':40,'rsi_period':14,'rsi_min':40,'rsi_max':75,
              'sl_pct':0.04,'trail_activate':0.20,'trail_pct':0.05,
              'leverage':15,'margin_normal':0.30,'margin_reduced':0.15},
}

# 공통 기본값
DEFAULTS = {'fee_rate':0.0004,'initial_capital':3000.0,'atr_period':14,
            'monthly_loss_limit':0,'consec_loss_pause':0,'pause_candles':0,'dd_threshold':0}

def main():
    t0 = time.time()
    print("="*110)
    print("  전체 기획서 일괄 30회 검증 (v10.1 ~ v15.5)")
    print("="*110)

    df_5m = load_5m_data(DATA_DIR); mtf = build_mtf(df_5m); cache = IndicatorCache(mtf)

    # 워밍업
    for tf in ['5m','15m','30m']:
        w = {**DEFAULTS,**CONFIGS['v14.4'],'timeframe':tf}
        run_backtest(cache, tf, w)

    # 30회 검증
    print(f"\n{'='*110}")
    print(f"  30회 반복 검증")
    print(f"{'='*110}\n")

    all_verified = {}
    for ver, cfg in CONFIGS.items():
        full = {**DEFAULTS, **cfg}
        tf = full['timeframe']
        results = []
        for _ in range(30):
            r = run_backtest(cache, tf, full)
            if r: results.append(r)
            else: break
        if not results:
            print(f"  {ver:>8}: 결과 없음")
            continue
        r = results[0]
        bals = [x['bal'] for x in results]
        consistent = np.std(bals) == 0
        all_verified[ver] = {
            'bal':r['bal'], 'ret':r['ret'], 'pf':r['pf'], 'mdd':r['mdd'],
            'trades':r['trades'], 'sl':r['sl'], 'tsl':r['tsl'], 'rev':r['sig'],
            'fl':r['liq'], 'wr':r['wr'], 'yr':r.get('yr',{}),
            'consistent': consistent, 'cfg': r['cfg'],
            'avg_win':r.get('avg_win',0), 'avg_loss':r.get('avg_loss',0),
        }
        ck = "PASS" if consistent else "FAIL"
        print(f"  {ver:>8} | ${r['bal']:>12,.0f} | {r['ret']:>+10,.1f}% | PF:{r['pf']:>6.2f} | "
              f"MDD:{r['mdd']:>5.1f}% | FL:{r['liq']:>2} | TR:{r['trades']:>4} | WR:{r['wr']:>5.1f}% | 30x {ck}")

    # 종합 스코어
    print(f"\n{'='*110}")
    print(f"  종합 스코어링 (수익 x PF x (100-MDD) / FL페널티)")
    print(f"{'='*110}\n")

    scored = []
    for ver, d in all_verified.items():
        ret = max(d['ret'], 0.01)
        pf = max(d['pf'], 0.01)
        mdd = d['mdd']
        fl = d['fl']
        trades = d['trades']

        # 종합 스코어: 수익 x PF x 안전성 / FL
        s_ret = np.log1p(ret/100)
        s_pf = min(pf, 30) / 5  # PF cap at 30
        s_mdd = (100 - mdd) / 100
        s_fl = max(0.1, 1 - fl * 0.03)
        s_tr = min(trades / 20, 2.0)  # 거래수 보너스

        score = s_ret * s_pf * s_mdd * s_fl * s_tr * 100

        scored.append((ver, score, d))

    scored.sort(key=lambda x: x[1], reverse=True)

    print(f"  {'#':>3} {'버전':>8} | {'잔액':>12} | {'수익률':>10} | {'PF':>6} | {'MDD':>6} | {'FL':>3} | {'TR':>4} | {'WR':>5} | {'점수':>6}")
    print(f"  {'-'*90}")
    for i, (ver, sc, d) in enumerate(scored):
        print(f"  {i+1:>3} {ver:>8} | ${d['bal']:>10,.0f} | {d['ret']:>+9.1f}% | {d['pf']:>5.2f} | {d['mdd']:>5.1f}% | {d['fl']:>3} | {d['trades']:>4} | {d['wr']:>4.1f}% | {sc:>5.1f}")

    # ===== 베스트 3 =====
    print(f"\n{'='*110}")
    print(f"  BEST 3 선정")
    print(f"{'='*110}")

    for rank, (ver, sc, d) in enumerate(scored[:3]):
        print(f"\n  [{rank+1}위] {ver} (점수: {sc:.1f})")
        print(f"  잔액: ${d['bal']:,.0f} | 수익률: {d['ret']:+,.1f}% | PF: {d['pf']:.2f} | MDD: {d['mdd']:.1f}% | FL: {d['fl']} | 거래: {d['trades']}")
        print(f"  설정: {d['cfg']}")
        if d.get('yr'):
            print(f"  연도별: {' | '.join(f'{k}:{v:+.1f}%' for k,v in sorted(d['yr'].items()))}")

        # 선정 이유
        if rank == 0:
            print(f"\n  [선정 이유]")
            print(f"  - 수익/PF/MDD/FL 4개 지표 종합 최우수")
            if d['pf'] >= 8: print(f"  - PF {d['pf']:.1f}로 목표 PF>=8 달성")
            if d['fl'] == 0: print(f"  - 강제청산 0회로 리스크 관리 완벽")
            if d['mdd'] < 40: print(f"  - MDD {d['mdd']:.1f}%로 낮은 변동성")
        elif rank == 1:
            print(f"\n  [선정 이유]")
            print(f"  - 종합 2위: 1위 대비 다른 강점 보유")
            if d['bal'] > scored[0][2]['bal']: print(f"  - 절대 수익 ${d['bal']:,.0f}로 최고 수준")
            if d['mdd'] < scored[0][2]['mdd']: print(f"  - MDD {d['mdd']:.1f}%로 1위보다 안정")
        elif rank == 2:
            print(f"\n  [선정 이유]")
            print(f"  - 종합 3위: 균형 잡힌 성과")

    # ===== 폐기 10 =====
    print(f"\n{'='*110}")
    print(f"  폐기 권고 10개")
    print(f"{'='*110}")

    discard = scored[-10:] if len(scored) >= 10 else scored[3:]
    discard.reverse()  # 최하위부터

    for i, (ver, sc, d) in enumerate(discard):
        rank = len(scored) - 9 + i if len(scored) >= 10 else i + 4
        print(f"\n  [폐기 {i+1}] {ver} (점수: {sc:.1f}, 순위: {rank}/{len(scored)})")
        print(f"  잔액: ${d['bal']:,.0f} | 수익률: {d['ret']:+,.1f}% | PF: {d['pf']:.2f} | MDD: {d['mdd']:.1f}% | FL: {d['fl']} | 거래: {d['trades']}")

        reasons = []
        if d['mdd'] > 70: reasons.append(f"MDD {d['mdd']:.1f}% (70% 초과, 실전 파산 위험)")
        if d['fl'] > 10: reasons.append(f"강제청산 {d['fl']}회 (10회 초과, 격리마진 반복 파산)")
        if d['fl'] > 0 and d['fl'] <= 10: reasons.append(f"강제청산 {d['fl']}회 (FL 0 달성 버전 대비 열세)")
        if d['ret'] < 1000: reasons.append(f"수익률 {d['ret']:+,.1f}% (+1,000% 미만)")
        if d['pf'] < 1.5: reasons.append(f"PF {d['pf']:.2f} (1.5 미만, 수익/손실 비율 불량)")
        if d['trades'] < 10: reasons.append(f"거래 {d['trades']}회 (10회 미만, 통계 신뢰도 부족)")
        if d['mdd'] > 50 and d['pf'] < 3: reasons.append("높은 MDD + 낮은 PF = 리스크 대비 수익 열악")

        # 상위 버전 대비 열세
        best = scored[0][2]
        if d['bal'] < best['bal'] * 0.1: reasons.append(f"1위 대비 잔액 10% 미만")
        if d['mdd'] > best['mdd'] + 20: reasons.append(f"1위 대비 MDD {d['mdd']-best['mdd']:+.1f}%p 열세")

        if not reasons:
            reasons.append("상위 버전 대비 모든 지표에서 열세")

        print(f"  [폐기 사유]")
        for r in reasons:
            print(f"    - {r}")

    elapsed = time.time() - t0
    print(f"\n{'='*110}")
    print(f"  검증 완료: {len(all_verified)}개 버전, {elapsed:.1f}초")
    print(f"{'='*110}")

    # JSON 저장
    save_data = {'verified': all_verified, 'ranking': [(v,s) for v,s,_ in scored]}
    with open(os.path.join(DATA_DIR, 'verify_all_results.json'), 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  저장: verify_all_results.json")

if __name__=='__main__':
    main()
