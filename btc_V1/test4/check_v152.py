"""v15.2를 bt_fast/verify_all.py 설정과 동일하게 재실행 (지연진입 없음)"""
import sys
sys.path.insert(0, r'D:\filesystem\futures\btc_V1\test4')
from backtest_v3 import *

base = r'D:\filesystem\futures\btc_V1\test4'
df5 = load_5m(base)
df30 = resample_30m(df5)

# bt_fast/verify_all.py의 v15.2 설정 (지연진입 없음!)
cfg_v152_btfast = {
    'name': 'v15.2 (bt_fast)', 'timeframe': '30m',
    'ema_fast': 3, 'ema_slow': 200,
    'adx_min': 35, 'rsi_min': 30, 'rsi_max': 65,
    'sl_pct': 0.05, 'trail_activate': 0.06, 'trail_pct': 0.05,
    'leverage': 10, 'margin': 0.30, 'margin_reduced': 0.15,
    'fee_rate': 0.0004,
    'monthly_loss_limit': -0.15,
    'delayed_entry': False,  # verify_all.py에는 delayed_entry 없음!
}

df = prep_df(df30.copy(), 3, 200)
r = bt_core(cfg_v152_btfast, df)

exp = {'bal': 353942, 'ret': 11698, 'tr': 93, 'pf': 2.07, 'mdd': 40.2, 'fl': 0, 'wr': 44.1,
       'sl': None, 'tsl': None, 'rev': None}
show('v15.2 (bt_fast 동일 - 지연진입 없음)', r, exp)

print(f"\n감사보고서 bt_fast 결과: $353,942 / 93거래 / PF 2.07 / MDD 40.2%")
print(f"내 v3 결과:              ${r['bal']:,.0f} / {r['trades']}거래 / PF {r['pf']} / MDD {r['mdd']}%")
err = (r['bal'] / 353942 - 1) * 100
print(f"잔액 오차: {err:+.3f}%")
