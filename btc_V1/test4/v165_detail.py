"""v16.5 균형 추천 전략 상세 + 마진별 비교"""
import numpy as np
from bt_fast import load_5m_data, build_mtf, IndicatorCache, run_backtest
BASE = r'D:\filesystem\futures\btc_V1\test4'

def main():
    df5=load_5m_data(BASE); mtf=build_mtf(df5); cache=IndicatorCache(mtf)

    base={
        'timeframe':'10m','ma_fast_type':'hma','ma_slow_type':'ema',
        'ma_fast':21,'ma_slow':250,'adx_period':20,'adx_min':35,
        'rsi_period':14,'rsi_min':40,'rsi_max':75,
        'sl_pct':0.06,'trail_activate':0.07,'trail_pct':0.03,
        'leverage':10,'fee_rate':0.0004,'initial_capital':3000.0,
    }

    # 마진별 비교
    print('='*80)
    print('  v16.5 마진별 비교 (10m HMA(21/250) ADX(20)>=35)')
    print('='*80)
    print(f'  {"마진":>5} {"잔액":>12} {"수익률":>10} {"PF":>7} {"MDD":>6} {"거래":>4} {"승률":>5} {"SL":>3} {"TSL":>4} {"REV":>4}')
    print(f'  {"-"*65}')
    for mg in [0.15,0.20,0.25,0.30,0.35,0.40,0.50]:
        for dd in [0,-0.25]:
            cfg=dict(base); cfg['margin_normal']=mg; cfg['margin_reduced']=mg/2; cfg['dd_threshold']=dd
            r=run_backtest(cache,'10m',cfg)
            if r:
                tag=f' DD-25%' if dd==-0.25 else ''
                print(f'  {mg:>4.0%}{tag:>6} ${r["bal"]:>10,.0f} {r["ret"]:>+9.1f}% {r["pf"]:>6.2f} {r["mdd"]:>5.1f}% {r["trades"]:>4} {r["wr"]:>4.1f}% {r["sl"]:>3} {r["tsl"]:>4} {r["sig"]:>4}')

    # 균형 추천 (M40% DD-25%) 상세
    cfg=dict(base); cfg['margin_normal']=0.40; cfg['margin_reduced']=0.20; cfg['dd_threshold']=-0.25
    r=run_backtest(cache,'10m',cfg)
    print(f'\n{"="*80}')
    print(f'  v16.5 균형 추천 상세')
    print(f'{"="*80}')
    print(f'  잔액: ${r["bal"]:,.0f} ({r["ret"]:+,.1f}%)')
    print(f'  PF: {r["pf"]} | MDD: {r["mdd"]}% | FL: {r["liq"]}')
    print(f'  거래: {r["trades"]} (SL:{r["sl"]} TSL:{r["tsl"]} REV:{r["sig"]})')
    print(f'  승률: {r["wr"]}% | 평균승: {r["avg_win"]:.2f}% | 평균패: {r["avg_loss"]:.2f}%')
    if r['avg_loss']!=0: print(f'  손익비: {abs(r["avg_win"]/r["avg_loss"]):.2f}:1')
    print(f'  연도별:')
    for y in sorted(r.get('yr',{}).keys()):
        print(f'    {y}: {r["yr"][y]:>+8.1f}%')

    # 30m WMA 안정형 상세
    cfg2={
        'timeframe':'30m','ma_fast_type':'wma','ma_slow_type':'ema',
        'ma_fast':3,'ma_slow':200,'adx_period':20,'adx_min':35,
        'rsi_period':14,'rsi_min':35,'rsi_max':65,
        'sl_pct':0.08,'trail_activate':0.04,'trail_pct':0.03,
        'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
        'fee_rate':0.0004,'initial_capital':3000.0,
    }
    r2=run_backtest(cache,'30m',cfg2)
    print(f'\n{"="*80}')
    print(f'  안정형 (30m WMA M25%) 상세')
    print(f'{"="*80}')
    print(f'  잔액: ${r2["bal"]:,.0f} ({r2["ret"]:+,.1f}%)')
    print(f'  PF: {r2["pf"]} | MDD: {r2["mdd"]}% | FL: {r2["liq"]}')
    print(f'  거래: {r2["trades"]} (SL:{r2["sl"]} TSL:{r2["tsl"]} REV:{r2["sig"]})')
    print(f'  승률: {r2["wr"]}% | 평균승: {r2["avg_win"]:.2f}% | 평균패: {r2["avg_loss"]:.2f}%')
    if r2['avg_loss']!=0: print(f'  손익비: {abs(r2["avg_win"]/r2["avg_loss"]):.2f}:1')
    print(f'  연도별:')
    for y in sorted(r2.get('yr',{}).keys()):
        print(f'    {y}: {r2["yr"][y]:>+8.1f}%')

    # 1h EMA 수익형 상세
    cfg3={
        'timeframe':'1h','ma_fast_type':'ema','ma_slow_type':'ema',
        'ma_fast':3,'ma_slow':100,'adx_period':20,'adx_min':30,
        'rsi_period':14,'rsi_min':35,'rsi_max':70,
        'sl_pct':0.07,'trail_activate':0.07,'trail_pct':0.05,
        'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,
        'dd_threshold':-0.25,
        'fee_rate':0.0004,'initial_capital':3000.0,
    }
    r3=run_backtest(cache,'1h',cfg3)
    print(f'\n{"="*80}')
    print(f'  수익형 (1h EMA M40% DD-25%) 상세')
    print(f'{"="*80}')
    print(f'  잔액: ${r3["bal"]:,.0f} ({r3["ret"]:+,.1f}%)')
    print(f'  PF: {r3["pf"]} | MDD: {r3["mdd"]}% | FL: {r3["liq"]}')
    print(f'  거래: {r3["trades"]} (SL:{r3["sl"]} TSL:{r3["tsl"]} REV:{r3["sig"]})')
    print(f'  승률: {r3["wr"]}%')
    print(f'  연도별:')
    for y in sorted(r3.get('yr',{}).keys()):
        print(f'    {y}: {r3["yr"][y]:>+8.1f}%')

if __name__=='__main__':
    main()
