"""EMA(14)/EMA(200) 순수 크로스 - 5m/15m/30m/1h 비교"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_fast import load_5m_data, resample_tf, calc_ema

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
df_5m = load_5m_data(DATA_DIR)

FEE = 0.0004; INIT = 3000.0; LEV = 10; MARGIN = 0.20; LIQ = 1.0/LEV

def run_raw_cross(df, label):
    c = df['close']; df = df.copy()
    df['ema14'] = calc_ema(c, 14); df['ema200'] = calc_ema(c, 200)
    times=df['time'].values; closes=df['close'].values.astype(np.float64)
    highs=df['high'].values.astype(np.float64); lows=df['low'].values.astype(np.float64)
    ema14=df['ema14'].values.astype(np.float64); ema200=df['ema200'].values.astype(np.float64)
    n=len(df); bal=INIT; pos=0; ep=0.0; su=0.0
    gpeak=INIT; gmdd=0.0; yearly={}; total_trades=0; total_w=0; total_l=0; total_fl=0
    gp=0.0; gl=0.0; total_cx=0

    for i in range(1, n):
        if np.isnan(ema14[i]) or np.isnan(ema200[i]) or np.isnan(ema14[i-1]) or np.isnan(ema200[i-1]):
            continue
        cu = ema14[i]>ema200[i] and ema14[i-1]<=ema200[i-1]
        cd = ema14[i]<ema200[i] and ema14[i-1]>=ema200[i-1]
        if cu or cd: total_cx += 1

        yr = str(pd.Timestamp(times[i]).year)
        if yr not in yearly: yearly[yr] = {'start': bal, 'end': bal}
        yearly[yr]['end'] = bal

        if bal > gpeak: gpeak = bal
        dd = (gpeak - bal) / gpeak if gpeak > 0 else 0
        if dd > gmdd: gmdd = dd

        if pos != 0:
            if pos == 1: lwc = (lows[i] - ep) / ep
            else: lwc = (ep - highs[i]) / ep
            if lwc <= -LIQ:
                pu = su*(-LIQ) - su*FEE; bal += pu; bal = max(bal, 0)
                total_trades += 1; total_l += 1; total_fl += 1; gl += abs(pu); pos = 0; continue

            do_close = False; new_dir = 0
            if pos == 1 and cd: do_close = True; new_dir = -1
            elif pos == -1 and cu: do_close = True; new_dir = 1
            if do_close:
                pnl = (closes[i]-ep)/ep if pos==1 else (ep-closes[i])/ep
                pu = su*pnl - su*FEE; bal += pu
                total_trades += 1
                if pu > 0: total_w += 1; gp += pu
                else: total_l += 1; gl += abs(pu)
                pos = 0
                if bal > 10:
                    mu = bal*MARGIN; su = mu*LEV; bal -= su*FEE
                    pos = new_dir; ep = closes[i]
                continue

        if pos == 0 and bal > 10:
            if cu:
                mu = bal*MARGIN; su = mu*LEV; bal -= su*FEE; pos = 1; ep = closes[i]
            elif cd:
                mu = bal*MARGIN; su = mu*LEV; bal -= su*FEE; pos = -1; ep = closes[i]

    if pos != 0:
        pnl = (closes[-1]-ep)/ep if pos==1 else (ep-closes[-1])/ep
        pu = su*pnl - su*FEE; bal += pu; total_trades += 1
        if pu > 0: total_w += 1; gp += pu
        else: total_l += 1; gl += abs(pu)

    pf = gp/gl if gl > 0 else 0
    wr = total_w/total_trades*100 if total_trades > 0 else 0

    return {
        'label': label, 'bal': bal, 'ret': (bal-INIT)/INIT*100,
        'trades': total_trades, 'w': total_w, 'l': total_l, 'fl': total_fl,
        'wr': wr, 'pf': pf, 'mdd': gmdd*100, 'cx': total_cx,
        'cx_monthly': total_cx / 75, 'yearly': yearly
    }

# 실행
tfs = [
    ('5m', df_5m),
    ('15m', resample_tf(df_5m, 15)),
    ('30m', resample_tf(df_5m, 30)),
    ('1h', resample_tf(df_5m, 60)),
]

results = []
for name, df in tfs:
    print(f"  {name}: {len(df):,}캔들...", flush=True)
    r = run_raw_cross(df, name)
    results.append(r)

# 비교 출력
print(f"\n{'='*100}")
print(f"  EMA(14)/EMA(200) 순수 크로스 매매 - 타임프레임별 비교")
print(f"  필터 없음 | SL 없음 | 트레일링 없음 | 역신호 청산만 | 10x 20% | 수수료 0.08%")
print(f"{'='*100}\n")

print(f"  {'TF':>4} | {'최종잔액':>12} | {'수익률':>10} | {'거래':>6} | {'승률':>6} | {'PF':>5} | {'MDD':>6} | {'FL':>3} | {'크로스/월':>8}")
print(f"  {'-'*80}")
for r in results:
    print(f"  {r['label']:>4} | ${r['bal']:>10,.0f} | {r['ret']:>+9.1f}% | {r['trades']:>6} | {r['wr']:>5.1f}% | {r['pf']:>4.2f} | {r['mdd']:>5.1f}% | {r['fl']:>3} | {r['cx_monthly']:>7.1f}")

# 연도별 비교
print(f"\n{'='*100}")
print(f"  연도별 수익률 비교")
print(f"{'='*100}\n")

all_years = sorted(set(y for r in results for y in r['yearly'].keys()))
print(f"  {'연도':>6}", end='')
for r in results:
    print(f" | {r['label']:>10}", end='')
print()
print(f"  {'-'*55}")

for yr in all_years:
    print(f"  {yr:>6}", end='')
    for r in results:
        y = r['yearly'].get(yr, {'start': 0, 'end': 0})
        if y['start'] > 0:
            pct = (y['end'] - y['start']) / y['start'] * 100
            print(f" | {pct:>+9.1f}%", end='')
        else:
            print(f" | {'N/A':>10}", end='')
    print()

print(f"\n  {'최종':>6}", end='')
for r in results:
    print(f" | ${r['bal']:>9,.0f}", end='')
print()
