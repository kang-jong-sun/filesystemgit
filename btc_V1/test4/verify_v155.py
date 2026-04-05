"""v15.5 최적 전략 30회 반복 검증 + 상세 분석"""
import numpy as np, json
from bt_fast import load_5m_data, build_mtf, IndicatorCache, run_backtest

BASE = r'D:\filesystem\futures\btc_V1\test4'

# 최적 전략 (Phase 4 Top 1)
BEST = {
    'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
    'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':35,'rsi_max':65,
    'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.05,
    'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,
    'monthly_loss_limit':-0.25,'dd_threshold':-0.30,
    'fee_rate':0.0004,'initial_capital':3000.0,
}

# 주요 비교 대상들
ALTS = {
    'v14.4 (검증완료)': {
        'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
        'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,
        'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
        'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
        'monthly_loss_limit':-0.20,
        'fee_rate':0.0004,'initial_capital':3000.0,
    },
    'v15.4 (검증완료)': {
        'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
        'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,
        'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
        'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,
        'monthly_loss_limit':-0.30,
        'fee_rate':0.0004,'initial_capital':3000.0,
    },
    'v15.5-M40 (공격형)': {
        'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
        'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':35,'rsi_max':65,
        'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.05,
        'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,
        'monthly_loss_limit':-0.25,'dd_threshold':-0.30,
        'fee_rate':0.0004,'initial_capital':3000.0,
    },
    'v15.5-M50 (초공격)': {
        'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
        'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':35,'rsi_max':65,
        'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.05,
        'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,
        'monthly_loss_limit':-0.25,'dd_threshold':-0.30,
        'fee_rate':0.0004,'initial_capital':3000.0,
    },
    'v15.5-M25 (안정형)': {
        'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
        'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':35,'rsi_max':65,
        'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.05,
        'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
        'monthly_loss_limit':-0.25,'dd_threshold':-0.30,
        'fee_rate':0.0004,'initial_capital':3000.0,
    },
}

def main():
    print("=" * 70)
    print("  v15.5 최적 전략 30회 반복 검증 + 변형 비교")
    print("=" * 70)

    df5 = load_5m_data(BASE)
    mtf = build_mtf(df5)
    cache = IndicatorCache(mtf)

    # 30회 반복 검증
    print("\n[1] 30회 반복 검증 (v15.5 최적)")
    bals = []
    for i in range(30):
        r = run_backtest(cache, '30m', BEST)
        bals.append(r['bal'])
    print(f"  30회 결과: ${np.mean(bals):,.0f} (std={np.std(bals):.4f})")
    print(f"  → {'PASS' if np.std(bals) < 0.01 else 'FAIL'} (결정론적)")

    # 최적 전략 상세
    print("\n[2] v15.5 최적 전략 상세")
    r = run_backtest(cache, '30m', BEST)
    print(f"  잔액: ${r['bal']:,.0f}")
    print(f"  수익률: {r['ret']:+,.1f}%")
    print(f"  거래: {r['trades']}회 (승률 {r['wr']:.1f}%)")
    print(f"  PF: {r['pf']:.2f}")
    print(f"  MDD: {r['mdd']:.1f}%")
    print(f"  FL: {r['liq']}")
    print(f"  SL/TSL/REV: {r['sl']}/{r['tsl']}/{r['sig']}")
    print(f"  평균승: {r['avg_win']:.2f}% | 평균패: {r['avg_loss']:.2f}%")
    print(f"  손익비: {abs(r['avg_win']/r['avg_loss']) if r['avg_loss'] != 0 else 'INF':.2f}:1")
    print(f"  연도별:")
    for y in sorted(r.get('yr', {}).keys()):
        print(f"    {y}: {r['yr'][y]:>+8.1f}%")

    # 비교
    print("\n[3] 변형 비교")
    print(f"  {'전략':<22} {'잔액':>14} {'수익률':>12} {'PF':>6} {'MDD':>6} {'FL':>3} {'TR':>4}")
    print(f"  {'-'*72}")

    r_best = run_backtest(cache, '30m', BEST)
    print(f"  {'v15.5 (최적 M35)':<22} ${r_best['bal']:>12,.0f} {r_best['ret']:>+10.1f}% {r_best['pf']:>5.2f} {r_best['mdd']:>5.1f}% {r_best['liq']:>2} {r_best['trades']:>4}")

    all_results = {'v15.5 (최적 M35)': r_best}
    for name, cfg in ALTS.items():
        r = run_backtest(cache, '30m', cfg)
        if r:
            print(f"  {name:<22} ${r['bal']:>12,.0f} {r['ret']:>+10.1f}% {r['pf']:>5.2f} {r['mdd']:>5.1f}% {r['liq']:>2} {r['trades']:>4}")
            all_results[name] = r

    # JSON 저장
    out = {}
    for name, r in all_results.items():
        out[name] = {
            'bal': r['bal'], 'ret': r['ret'], 'pf': r['pf'], 'mdd': r['mdd'],
            'fl': r['liq'], 'trades': r['trades'], 'wr': r['wr'],
            'sl': r['sl'], 'tsl': r['tsl'], 'rev': r['sig'],
            'yr': r.get('yr', {}),
            'avg_win': r.get('avg_win', 0), 'avg_loss': r.get('avg_loss', 0),
        }
    with open(f'{BASE}/v155_verify_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n결과 저장: v155_verify_results.json")


if __name__ == '__main__':
    main()
