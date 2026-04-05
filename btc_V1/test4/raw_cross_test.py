"""
EMA(14)/EMA(200) 5분봉 크로스 - 필터 없음 - 75개월 월별 수익률
순수 크로스만으로 진입/청산, SL/트레일링 없음, 역신호 청산만
레버리지 10x, 마진 20%, 수수료 0.04%x2
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_fast import load_5m_data, calc_ema

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

df = load_5m_data(DATA_DIR)
c = df['close']
# 15분봉 리샘플링
from bt_fast import resample_tf
df = resample_tf(df, 15)
print(f"15m: {len(df):,}개")
c = df['close']
df['ema14'] = calc_ema(c, 14)
df['ema200'] = calc_ema(c, 200)

times = df['time'].values
closes = df['close'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)
ema14 = df['ema14'].values.astype(np.float64)
ema200 = df['ema200'].values.astype(np.float64)
n = len(df)

# 설정
FEE = 0.0004
INIT = 3000.0
LEV = 10
MARGIN = 0.20
LIQ = 1.0 / LEV  # 10%

bal = INIT
pb = bal
pos = 0  # 0=없음, 1=롱, -1=숏
ep = 0.0
su = 0.0

cm = ''
monthly = {}
gpeak = INIT
gmdd = 0.0

def mk(i):
    return pd.Timestamp(times[i]).strftime('%Y-%m')

def em(m):
    if m not in monthly:
        monthly[m] = {'sb': bal, 'eb': bal, 'trades': 0, 'w': 0, 'l': 0,
                      'gp': 0.0, 'gl': 0.0, 'fl': 0, 'cross_up': 0, 'cross_dn': 0}

for i in range(1, n):
    if np.isnan(ema14[i]) or np.isnan(ema200[i]) or np.isnan(ema14[i-1]) or np.isnan(ema200[i-1]):
        continue

    m = mk(i)
    if m != cm:
        if cm and cm in monthly:
            monthly[cm]['eb'] = bal
        cm = m
        em(m)
        monthly[m]['sb'] = bal

    # 크로스 감지
    cross_up = ema14[i] > ema200[i] and ema14[i-1] <= ema200[i-1]
    cross_dn = ema14[i] < ema200[i] and ema14[i-1] >= ema200[i-1]

    if cross_up:
        monthly[m]['cross_up'] += 1
    if cross_dn:
        monthly[m]['cross_dn'] += 1

    # MDD
    if bal > gpeak:
        gpeak = bal
    dd = (gpeak - bal) / gpeak if gpeak > 0 else 0
    if dd > gmdd:
        gmdd = dd

    # 포지션 보유 중
    if pos != 0:
        # 강제청산 체크
        if pos == 1:
            lwc = (lows[i] - ep) / ep
        else:
            lwc = (ep - highs[i]) / ep
        if lwc <= -LIQ:
            pu = su * (-LIQ) - su * FEE
            bal += pu
            bal = max(bal, 0)
            monthly[m]['trades'] += 1
            monthly[m]['l'] += 1
            monthly[m]['gl'] += abs(pu)
            monthly[m]['fl'] += 1
            pos = 0
            continue

        # 역신호 청산 (크로스 반대 발생 시)
        if pos == 1 and cross_dn:
            pnl = (closes[i] - ep) / ep
            pu = su * pnl - su * FEE
            bal += pu
            monthly[m]['trades'] += 1
            if pu > 0:
                monthly[m]['w'] += 1
                monthly[m]['gp'] += pu
            else:
                monthly[m]['l'] += 1
                monthly[m]['gl'] += abs(pu)
            pos = 0
            # 즉시 숏 진입
            if bal > 10:
                mu = bal * MARGIN
                su = mu * LEV
                bal -= su * FEE
                pos = -1
                ep = closes[i]
            continue

        elif pos == -1 and cross_up:
            pnl = (ep - closes[i]) / ep
            pu = su * pnl - su * FEE
            bal += pu
            monthly[m]['trades'] += 1
            if pu > 0:
                monthly[m]['w'] += 1
                monthly[m]['gp'] += pu
            else:
                monthly[m]['l'] += 1
                monthly[m]['gl'] += abs(pu)
            pos = 0
            # 즉시 롱 진입
            if bal > 10:
                mu = bal * MARGIN
                su = mu * LEV
                bal -= su * FEE
                pos = 1
                ep = closes[i]
            continue

    # 포지션 없을 때 - 크로스 발생하면 진입
    if pos == 0 and bal > 10:
        if cross_up:
            mu = bal * MARGIN
            su = mu * LEV
            bal -= su * FEE
            pos = 1
            ep = closes[i]
        elif cross_dn:
            mu = bal * MARGIN
            su = mu * LEV
            bal -= su * FEE
            pos = -1
            ep = closes[i]

    if bal > pb:
        pb = bal
    if cm in monthly:
        monthly[cm]['eb'] = bal

# 미청산
if pos != 0:
    m = mk(n - 1)
    if pos == 1:
        pnl = (closes[-1] - ep) / ep
    else:
        pnl = (ep - closes[-1]) / ep
    pu = su * pnl - su * FEE
    bal += pu
    em(m)
    monthly[m]['trades'] += 1
    if pu > 0:
        monthly[m]['w'] += 1
        monthly[m]['gp'] += pu
    else:
        monthly[m]['l'] += 1
        monthly[m]['gl'] += abs(pu)
if cm in monthly:
    monthly[cm]['eb'] = bal

# 출력
print(f"\n{'='*120}")
print(f"  EMA(14)/EMA(200) 15분봉 순수 크로스 매매 (필터 없음)")
print(f"  SL 없음 | 트레일링 없음 | 역신호 청산만 | 10x 20% | 수수료 0.08%")
print(f"{'='*120}\n")

print(f"  {'월':>8} | {'손익률':>8} | {'손익금':>12} | {'계정잔금':>14} | {'거래':>4} | {'승':>3} | {'패':>3} | {'FL':>3} | {'크로스':>6} | {'PF':>6} | 비고")
print(f"  {'-'*110}")

sm = sorted(monthly.keys())
yearly = {}

for mk2 in sm:
    d = monthly[mk2]
    s = d['sb']
    e = d['eb']
    pa = e - s
    pp = (pa / s * 100) if s > 0 else 0
    crosses = d['cross_up'] + d['cross_dn']
    pf = d['gp'] / d['gl'] if d['gl'] > 0 else (0 if d['gp'] == 0 else -1)
    pfs = "  INF" if pf < 0 else f"{pf:5.2f}" if pf > 0 else "    -"

    notes = []
    if pp > 30: notes.append("대형수익")
    elif pp < -15: notes.append("큰손실")
    if d['fl'] > 0: notes.append(f"FL{d['fl']}")

    print(f"  {mk2:>8} | {pp:>+7.1f}% | ${pa:>+10,.0f} | ${e:>12,.0f} | {d['trades']:>4} | {d['w']:>3} | {d['l']:>3} | {d['fl']:>3} | {crosses:>6} | {pfs} | {', '.join(notes)}")

    yr = mk2[:4]
    if yr not in yearly:
        yearly[yr] = {'start': s, 'end': e, 'tr': 0, 'w': 0, 'l': 0, 'fl': 0, 'gp': 0, 'gl': 0, 'cx': 0}
    yearly[yr]['end'] = e
    yearly[yr]['tr'] += d['trades']
    yearly[yr]['w'] += d['w']
    yearly[yr]['l'] += d['l']
    yearly[yr]['fl'] += d['fl']
    yearly[yr]['gp'] += d['gp']
    yearly[yr]['gl'] += d['gl']
    yearly[yr]['cx'] += crosses

    ni = sm.index(mk2) + 1
    if ni < len(sm) and sm[ni][:4] != yr:
        y = yearly[yr]
        yp = y['end'] - y['start']
        ypc = (yp / y['start'] * 100) if y['start'] > 0 else 0
        ypf = y['gp'] / y['gl'] if y['gl'] > 0 else 0
        print(f"  {'-'*110}")
        print(f"  {yr+'합계':>8} | {ypc:>+7.1f}% | ${yp:>+10,.0f} | ${y['end']:>12,.0f} | {y['tr']:>4} | {y['w']:>3} | {y['l']:>3} | {y['fl']:>3} | {y['cx']:>6} | {ypf:5.2f} |")
        print(f"  {'='*110}")

# 마지막 연도
yr = sm[-1][:4]
y = yearly[yr]
yp = y['end'] - y['start']
ypc = (yp / y['start'] * 100) if y['start'] > 0 else 0
ypf = y['gp'] / y['gl'] if y['gl'] > 0 else 0
print(f"  {'-'*110}")
print(f"  {yr+'합계':>8} | {ypc:>+7.1f}% | ${yp:>+10,.0f} | ${y['end']:>12,.0f} | {y['tr']:>4} | {y['w']:>3} | {y['l']:>3} | {y['fl']:>3} | {y['cx']:>6} | {ypf:5.2f} |")

# 전체 요약
ttr = sum(d['trades'] for d in monthly.values())
tw = sum(d['w'] for d in monthly.values())
tl = sum(d['l'] for d in monthly.values())
tfl = sum(d['fl'] for d in monthly.values())
tgp = sum(d['gp'] for d in monthly.values())
tgl = sum(d['gl'] for d in monthly.values())
tpf = tgp / tgl if tgl > 0 else 0
tcx = sum(d['cross_up'] + d['cross_dn'] for d in monthly.values())

print(f"\n  {'='*110}")
print(f"  총계: ${bal:,.0f} ({(bal-INIT)/INIT*100:+,.1f}%) | 거래:{ttr} (승:{tw} 패:{tl} FL:{tfl}) | 승률:{tw/ttr*100:.1f}%" if ttr > 0 else "")
print(f"  PF: {tpf:.2f} | MDD: {gmdd*100:.1f}% | 크로스 총:{tcx}회 (월평균:{tcx/len(monthly):.1f})")
print(f"  {'='*110}")

# 연도별 요약
print(f"\n  연도별 요약:")
print(f"  {'연도':>6} | {'시작':>12} | {'종료':>12} | {'수익률':>8} | {'거래':>4} | {'승률':>5} | {'PF':>5} | {'FL':>3} | {'크로스':>6}")
print(f"  {'-'*80}")
for yr2 in sorted(yearly.keys()):
    y = yearly[yr2]
    yp2 = y['end'] - y['start']
    ypc2 = (yp2 / y['start'] * 100) if y['start'] > 0 else 0
    wr2 = y['w'] / y['tr'] * 100 if y['tr'] > 0 else 0
    ypf2 = y['gp'] / y['gl'] if y['gl'] > 0 else 0
    print(f"  {yr2:>6} | ${y['start']:>10,.0f} | ${y['end']:>10,.0f} | {ypc2:>+7.1f}% | {y['tr']:>4} | {wr2:>4.1f}% | {ypf2:>4.2f} | {y['fl']:>3} | {y['cx']:>6}")
