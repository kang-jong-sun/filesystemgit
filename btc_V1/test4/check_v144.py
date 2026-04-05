import json

with open(r'D:\filesystem\futures\btc_V1\test4\backtest_v2_results.json') as f:
    d = json.load(f)

# v14.4 연도별 상세 비교
print("=== v14.4 연도별 비교 ===")
print(f"{'연도':<6} {'백테스트':>14} {'기획서':>14} {'차이':>10}")
expected_yearly = {
    '2020': (8781, 192.7),
    '2021': (34388, 291.5),
    '2022': (49150, 42.9),
    '2023': (265291, 439.8),
    '2024': (517380, 95.1),
    '2025': (952258, 84.1),
    '2026': (837212, -12.1),
}
r = d['v14.4']
cum = 3000.0
for y in sorted(r['yearly'].keys()):
    pnl = r['yearly'][y]
    ret = pnl / cum * 100
    cum += pnl
    eb, eret = expected_yearly.get(y, (0, 0))
    print(f"{y:<6} ${cum:>12,.0f} ({ret:>+7.1f}%)   ${eb:>12,} ({eret:>+7.1f}%)  {cum/eb*100-100 if eb else 0:>+7.1f}%")

print(f"\n최종: ${r['final_balance']:,.2f} vs 기획서 $837,212 -> 오차 {r['final_balance']/837212*100-100:+.3f}%")

# v15.4 연도별 비교
print("\n=== v15.4 연도별 비교 ===")
r4 = d['v15.4']
exp4 = {
    '2020': 14042,
    '2021': 104137,
    '2022': 165170,
    '2023': 1891124,
    '2024': 4564337,
    '2025': 10982893,
    '2026': 8717659,
}
cum = 3000.0
for y in sorted(r4['yearly'].keys()):
    pnl = r4['yearly'][y]
    cum += pnl
    eb = exp4.get(y, 0)
    err = (cum / eb - 1) * 100 if eb else 0
    print(f"{y}: ${cum:>14,.0f} vs ${eb:>14,}  오차 {err:>+7.1f}%")

# v15.2 연도별 비교
print("\n=== v15.2 연도별 비교 ===")
r2 = d['v15.2']
exp2_yr = {
    '2020': 7024,
    '2021': 11965,
    '2022': 17257,
    '2023': 68886,
    '2024': 171234,
    '2025': 260358,
    '2026': 243482,
}
cum = 3000.0
for y in sorted(r2['yearly'].keys()):
    pnl = r2['yearly'][y]
    cum += pnl
    eb = exp2_yr.get(y, 0)
    err = (cum / eb - 1) * 100 if eb else 0
    print(f"{y}: ${cum:>14,.0f} vs ${eb:>14,}  오차 {err:>+7.1f}%")
