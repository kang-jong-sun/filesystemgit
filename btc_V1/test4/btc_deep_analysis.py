"""BTC 집중 분석: PF Top vs MDD Top vs 균형 전략 비교"""
import numpy as np, json
from bt_fast import load_5m_data, build_mtf, IndicatorCache, run_backtest

BASE = r'D:\filesystem\futures\btc_V1\test4'

def detail(cache, name, cfg):
    r = run_backtest(cache, cfg['timeframe'], cfg)
    if not r: return None
    # 30회 검증
    bals = [run_backtest(cache, cfg['timeframe'], cfg)['bal'] for _ in range(30)]
    std = np.std(bals)
    print(f"\n  [{name}]")
    print(f"  잔액: ${r['bal']:,.0f} ({r['ret']:+,.1f}%)")
    print(f"  PF: {r['pf']:.2f} | MDD: {r['mdd']:.1f}% | FL: {r['liq']}")
    print(f"  거래: {r['trades']} (SL:{r['sl']} TSL:{r['tsl']} REV:{r['sig']})")
    print(f"  승률: {r['wr']:.1f}% | 평균승: {r['avg_win']:.2f}% | 평균패: {r['avg_loss']:.2f}%")
    if r['avg_loss'] != 0: print(f"  손익비: {abs(r['avg_win']/r['avg_loss']):.2f}:1")
    print(f"  30회: std={std:.4f} {'PASS' if std<0.01 else 'FAIL'}")
    print(f"  연도별:")
    for y in sorted(r.get('yr',{}).keys()):
        print(f"    {y}: {r['yr'][y]:>+8.1f}%")
    return r

def main():
    print("="*80)
    print("  BTC 집중 분석 - 3가지 전략 비교")
    print("="*80)

    df5 = load_5m_data(BASE)
    mtf = build_mtf(df5)
    cache = IndicatorCache(mtf)

    strategies = {
        # A: PF Top (현재 v16.1 최적)
        'A: PF Top (M50%)': {
            'timeframe':'10m','ma_fast_type':'hma','ma_slow_type':'ema',
            'ma_fast':21,'ma_slow':250,'adx_period':20,'adx_min':35,
            'rsi_period':14,'rsi_min':40,'rsi_max':75,
            'sl_pct':0.06,'trail_activate':0.07,'trail_pct':0.03,
            'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,
            'monthly_loss_limit':-0.20,'dd_threshold':-0.25,
            'fee_rate':0.0004,'initial_capital':3000.0,
        },
        # A-2: PF Top 마진 줄여 MDD 감소
        'A-2: PF Top (M25%)': {
            'timeframe':'10m','ma_fast_type':'hma','ma_slow_type':'ema',
            'ma_fast':21,'ma_slow':250,'adx_period':20,'adx_min':35,
            'rsi_period':14,'rsi_min':40,'rsi_max':75,
            'sl_pct':0.06,'trail_activate':0.07,'trail_pct':0.03,
            'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
            'monthly_loss_limit':-0.20,'dd_threshold':-0.25,
            'fee_rate':0.0004,'initial_capital':3000.0,
        },
        # B: MDD Top (낮은 MDD 전략)
        'B: MDD Top (1h)': {
            'timeframe':'1h','ma_fast_type':'ema','ma_slow_type':'ema',
            'ma_fast':3,'ma_slow':100,'adx_period':20,'adx_min':30,
            'rsi_period':14,'rsi_min':35,'rsi_max':70,
            'sl_pct':0.07,'trail_activate':0.07,'trail_pct':0.04,
            'leverage':10,'margin_normal':0.15,'margin_reduced':0.075,
            'fee_rate':0.0004,'initial_capital':3000.0,
        },
        # C: WMA v16.0 (기존)
        'C: v16.0 WMA (M50%)': {
            'timeframe':'30m','ma_fast_type':'wma','ma_slow_type':'ema',
            'ma_fast':3,'ma_slow':200,'adx_period':20,'adx_min':35,
            'rsi_period':14,'rsi_min':35,'rsi_max':65,
            'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,
            'leverage':10,'margin_normal':0.50,'margin_reduced':0.25,
            'fee_rate':0.0004,'initial_capital':3000.0,
        },
        # D: v14.4 검증완료
        'D: v14.4 (검증완료)': {
            'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema',
            'ma_fast':3,'ma_slow':200,'adx_period':14,'adx_min':35,
            'rsi_period':14,'rsi_min':30,'rsi_max':65,
            'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,
            'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
            'monthly_loss_limit':-0.20,
            'fee_rate':0.0004,'initial_capital':3000.0,
        },
        # E: PF Top + M35% (균형)
        'E: PF Top (M35%)': {
            'timeframe':'10m','ma_fast_type':'hma','ma_slow_type':'ema',
            'ma_fast':21,'ma_slow':250,'adx_period':20,'adx_min':35,
            'rsi_period':14,'rsi_min':40,'rsi_max':75,
            'sl_pct':0.06,'trail_activate':0.07,'trail_pct':0.03,
            'leverage':10,'margin_normal':0.35,'margin_reduced':0.175,
            'monthly_loss_limit':-0.20,'dd_threshold':-0.25,
            'fee_rate':0.0004,'initial_capital':3000.0,
        },
    }

    results = {}
    for name, cfg in strategies.items():
        r = detail(cache, name, cfg)
        if r: results[name] = r

    # 비교표
    print(f"\n{'='*80}")
    print("  BTC 전략 비교표")
    print(f"{'='*80}")
    print(f"  {'전략':<24} {'잔액':>12} {'PF':>7} {'MDD':>6} {'거래':>4} {'승률':>5} {'손익비':>7}")
    print(f"  {'-'*70}")
    for name, r in results.items():
        wr_ratio = abs(r['avg_win']/r['avg_loss']) if r['avg_loss']!=0 else 0
        print(f"  {name:<24} ${r['bal']:>10,.0f} {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {wr_ratio:>6.2f}:1")

    print(f"\n  추천:")
    print(f"  - 안정형: B (MDD 14.5%) 또는 A-2 (M25%)")
    print(f"  - 균형형: E (M35%)")
    print(f"  - 공격형: A (M50%)")

if __name__ == '__main__':
    main()
