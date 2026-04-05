"""EMA(21)/EMA(250) 크로스 횟수 - 5분봉, 필터 없이, 월별 집계"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_fast import load_5m_data, calc_ema

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

df = load_5m_data(DATA_DIR)
c = df['close']
df['ema21'] = calc_ema(c, 21)
df['ema250'] = calc_ema(c, 250)

ema21 = df['ema21'].values
ema250 = df['ema250'].values
times = df['time'].values
n = len(df)

monthly = {}
total_cross = 0

for i in range(1, n):
    if np.isnan(ema21[i]) or np.isnan(ema250[i]) or np.isnan(ema21[i-1]) or np.isnan(ema250[i-1]):
        continue

    cross_up = ema21[i] > ema250[i] and ema21[i-1] <= ema250[i-1]
    cross_dn = ema21[i] < ema250[i] and ema21[i-1] >= ema250[i-1]

    mk = pd.Timestamp(times[i]).strftime('%Y-%m')
    if mk not in monthly:
        monthly[mk] = {'up': 0, 'dn': 0, 'total': 0}

    if cross_up:
        monthly[mk]['up'] += 1
        monthly[mk]['total'] += 1
        total_cross += 1
    elif cross_dn:
        monthly[mk]['dn'] += 1
        monthly[mk]['total'] += 1
        total_cross += 1

# 출력
print(f"\n{'='*70}")
print(f"  EMA(21)/EMA(250) 크로스 횟수 (5분봉, 필터 없음)")
print(f"  데이터: {len(df):,}개 캔들 ({df['time'].iloc[0]} ~ {df['time'].iloc[-1]})")
print(f"{'='*70}\n")

print(f"  {'월':>8} | {'상향크로스':>10} | {'하향크로스':>10} | {'합계':>6}")
print(f"  {'-'*50}")

sm = sorted(monthly.keys())
yearly = {}
for mk in sm:
    d = monthly[mk]
    print(f"  {mk:>8} | {d['up']:>10} | {d['dn']:>10} | {d['total']:>6}")

    yr = mk[:4]
    if yr not in yearly:
        yearly[yr] = {'up': 0, 'dn': 0, 'total': 0}
    yearly[yr]['up'] += d['up']
    yearly[yr]['dn'] += d['dn']
    yearly[yr]['total'] += d['total']

    # 연도 구분
    ni = sm.index(mk) + 1
    if ni < len(sm) and sm[ni][:4] != yr:
        y = yearly[yr]
        print(f"  {'-'*50}")
        print(f"  {yr+'합계':>8} | {y['up']:>10} | {y['dn']:>10} | {y['total']:>6}")
        print(f"  {'='*50}")

# 마지막 연도
yr = sm[-1][:4]
y = yearly[yr]
print(f"  {'-'*50}")
print(f"  {yr+'합계':>8} | {y['up']:>10} | {y['dn']:>10} | {y['total']:>6}")

print(f"\n  {'='*50}")
print(f"  전체 합계: 상향 {sum(d['up'] for d in monthly.values())} | 하향 {sum(d['dn'] for d in monthly.values())} | 총 {total_cross}회")
print(f"  월 평균: {total_cross / len(monthly):.1f}회")
print(f"  {'='*50}")

# 연도별 요약
print(f"\n  연도별 요약:")
print(f"  {'연도':>6} | {'상향':>6} | {'하향':>6} | {'합계':>6} | {'월평균':>6}")
print(f"  {'-'*40}")
for yr in sorted(yearly.keys()):
    y = yearly[yr]
    months_in_yr = sum(1 for mk in sm if mk[:4] == yr)
    avg = y['total'] / months_in_yr if months_in_yr > 0 else 0
    print(f"  {yr:>6} | {y['up']:>6} | {y['dn']:>6} | {y['total']:>6} | {avg:>5.1f}")
