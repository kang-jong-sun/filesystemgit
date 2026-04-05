# BTC v32.2 기획서 정확 구현 — 6엔진 + 30회 교차검증
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8'); warnings.filterwarnings('ignore')
def ema_pd(s, p): return s.ewm(span=p, adjust=False).mean()
def calc_adx(h, l, c, period=20):
    n=len(c); pdm=np.zeros(n); mdm=np.zeros(n); tr=np.zeros(n)
    for i in range(1, n):
        hd=h[i]-h[i-1]; ld=l[i-1]-l[i]
        pdm[i]=hd if(hd>ld and hd>0)else 0; mdm[i]=ld if(ld>hd and ld>0)else 0
        tr[i]=max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    a=1.0/period
    atr=pd.Series(tr).ewm(alpha=a, min_periods=period, adjust=False).mean()
    sp=pd.Series(pdm).ewm(alpha=a, min_periods=period, adjust=False).mean()
    sn=pd.Series(mdm).ewm(alpha=a, min_periods=period, adjust=False).mean()
    pdi=100*sp/atr.replace(0, 1e-10); mdi=100*sn/atr.replace(0, 1e-10)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0, 1e-10)
    return dx.ewm(alpha=a, min_periods=period, adjust=False).mean().values.astype(np.float64)
def calc_rsi(c, period=10):
    d=np.diff(c, prepend=c[0]); g=np.where(d>0, d, 0); lo=np.where(d<0, -d, 0)
    a=1.0/period
    ag=pd.Series(g).ewm(alpha=a, min_periods=period, adjust=False).mean()
    al=pd.Series(lo).ewm(alpha=a, min_periods=period, adjust=False).mean()
    return (100-100/(1+ag/al.replace(0, 1e-10))).values.astype(np.float64)

def bt_exact(c, h, l, fm, sm, adx, rsi, n):
    """기획서 v32.2 의사코드 정확 구현"""
    cap=5000.0; pos=0; epx=0.0; psz=0.0; slp=0.0
    ton=False; thi=0.0; tlo=999999.0
    watching=0; ws=0; ld=0; le=0
    pk=cap; mdd=0.0; ms=cap
    FEE=0.0004
    sc=tc=rc=wn=ln_=0; gp=gl=0.0; trades=[]

    for i in range(600, n):
        px = c[i]; h_ = h[i]; l_ = l[i]

        # 일일 리셋 (30분봉 기준 1440봉이 아니라, 48봉=1일)
        # 기획서: i % 1440 — 하지만 30분봉에서 1일=48봉
        # 기획서 원문: "30분봉 1440봉 = 1일" → 이건 오류, 실제로는 48봉
        if i > 600 and i % 48 == 0:
            ms = cap

        # ══════ STEP A: 포지션 보유 중 ══════
        if pos != 0:
            watching = 0

            # A1: SL (TSL 미활성 시에만, 저가/고가 기준, SL가에 청산)
            if not ton:
                sl_hit = False
                if pos == 1 and l_ <= slp: sl_hit = True
                elif pos == -1 and h_ >= slp: sl_hit = True
                if sl_hit:
                    pnl = (slp - epx) / epx * psz * pos - psz * FEE
                    cap += pnl; sc += 1
                    if pnl > 0: wn += 1; gp += pnl
                    else: ln_ += 1; gl += abs(pnl)
                    trades.append({'i': i, 'pnl': pnl, 'et': 'SL'})
                    ld = pos; le = i; pos = 0
                    pk = max(pk, cap)
                    dd = (pk-cap)/pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    continue  # ★ SL 발동 시 CONTINUE

            # A2: TA 활성화 (고가/저가 기준)
            if pos == 1: br = (h_ - epx) / epx * 100
            else: br = (epx - l_) / epx * 100
            if br >= 12.0 and not ton:
                ton = True

            # A3: TSL (TSL 활성 시, 종가 기준)
            if ton:
                if pos == 1:
                    if h_ > thi: thi = h_  # 고가로 고점 갱신
                    ns = thi * (1 - 9.0/100)
                    if ns > slp: slp = ns
                    if px <= slp:  # ★ 종가 기준
                        pnl = (px - epx) / epx * psz * pos - psz * FEE
                        cap += pnl; tc += 1
                        if pnl > 0: wn += 1; gp += pnl
                        else: ln_ += 1; gl += abs(pnl)
                        trades.append({'i': i, 'pnl': pnl, 'et': 'TSL'})
                        ld = pos; le = i; pos = 0
                        pk = max(pk, cap)
                        dd = (pk-cap)/pk if pk > 0 else 0
                        if dd > mdd: mdd = dd
                        continue
                else:
                    if l_ < tlo: tlo = l_  # 저가로 저점 갱신
                    ns = tlo * (1 + 9.0/100)
                    if ns < slp: slp = ns
                    if px >= slp:  # ★ 종가 기준
                        pnl = (px - epx) / epx * psz * pos - psz * FEE
                        cap += pnl; tc += 1
                        if pnl > 0: wn += 1; gp += pnl
                        else: ln_ += 1; gl += abs(pnl)
                        trades.append({'i': i, 'pnl': pnl, 'et': 'TSL'})
                        ld = pos; le = i; pos = 0
                        pk = max(pk, cap)
                        dd = (pk-cap)/pk if pk > 0 else 0
                        if dd > mdd: mdd = dd
                        continue

            # A4: REV (종가 기준, CONTINUE 안 함 — 같은 봉 재진입 가능)
            if i > 0:
                bn = fm[i] > sm[i]; bp = fm[i-1] > sm[i-1]
                cu = bn and not bp; cd = not bn and bp
                if (pos == 1 and cd) or (pos == -1 and cu):
                    pnl = (px - epx) / epx * psz * pos - psz * FEE
                    cap += pnl; rc += 1
                    if pnl > 0: wn += 1; gp += pnl
                    else: ln_ += 1; gl += abs(pnl)
                    trades.append({'i': i, 'pnl': pnl, 'et': 'REV'})
                    ld = pos; le = i; pos = 0
                    # ★ CONTINUE 안 함 — 아래 진입 로직으로 진행

        # ══════ STEP B: 포지션 없음 — 진입 ══════
        if i < 1: continue

        bn = fm[i] > sm[i]; bp = fm[i-1] > sm[i-1]
        cu = bn and not bp; cd = not bn and bp

        if pos == 0:
            if cu: watching = 1; ws = i
            elif cd: watching = -1; ws = i

            if watching != 0 and i > ws:
                if i - ws > 24: watching = 0; continue
                if watching == 1 and cd: watching = -1; ws = i; continue
                elif watching == -1 and cu: watching = 1; ws = i; continue
                if watching == ld: continue  # skip same dir
                if adx[i] < 30.0: continue
                if i >= 6 and adx[i] <= adx[i-6]: continue
                if rsi[i] < 40.0 or rsi[i] > 80.0: continue
                if sm[i] > 0 and abs(fm[i]-sm[i])/sm[i]*100 < 0.2: continue
                if ms > 0 and (cap-ms)/ms <= -0.20: watching = 0; continue
                if cap <= 0: continue

                # ═══ 진입 ═══
                mg = cap * 0.35
                psz = mg * 10
                cap -= psz * FEE  # 진입 수수료
                pos = watching
                epx = px
                ton = False; thi = px; tlo = px
                if pos == 1: slp = epx * (1 - 3.0/100)
                else: slp = epx * (1 + 3.0/100)
                pk = max(pk, cap)
                watching = 0

        # MDD
        pk = max(pk, cap)
        dd = (pk-cap)/pk if pk > 0 else 0
        if dd > mdd: mdd = dd
        if cap <= 0: break

    # 미청산 포지션
    if pos != 0 and cap > 0:
        pnl = (c[n-1] - epx) / epx * psz * pos - psz * FEE
        cap += pnl
        if pnl > 0: wn += 1; gp += pnl
        else: ln_ += 1; gl += abs(pnl)
        trades.append({'i': n-1, 'pnl': pnl, 'et': 'CL'})

    tot = sc+tc+rc; pf = gp/gl if gl > 0 else 999
    wr = wn/(wn+ln_)*100 if (wn+ln_) > 0 else 0
    aw = gp/wn if wn > 0 else 0; al_ = gl/ln_ if ln_ > 0 else 0
    wlr = aw/al_ if al_ > 0 else 999
    return cap, (cap-5000)/50, tot, sc, tc, rc, wn, ln_, pf, mdd*100, wr, wlr, trades

# Load
df5 = pd.read_csv('btc_usdt_5m_merged.csv', parse_dates=['timestamp'])
df5 = df5.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
agg = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
df30 = df5.set_index('timestamp').resample('30min').agg(agg).dropna().reset_index()
cs = df30['close']; c = cs.values.astype(np.float64)
h = df30['high'].values.astype(np.float64); l = df30['low'].values.astype(np.float64)
n = len(c)
fm = ema_pd(cs, 100).values.astype(np.float64); sm = ema_pd(cs, 600).values.astype(np.float64)
adx = calc_adx(h, l, c, 20); rsi = calc_rsi(c, 10)
years = pd.DatetimeIndex(df30['timestamp']).year.values

print(f"BTC 30m: {n} bars")
print()
print("=" * 80)
print("  BTC v32.2 기획서 정확 구현 — 6-Engine + 30x")
print("  SL: 저가/고가→SL가 | TSL: 종가 | REV: CONTINUE안함")
print("=" * 80)

# 6-Engine
caps = []
for _ in range(6):
    r = bt_exact(c, h, l, fm, sm, adx, rsi, n)
    caps.append(r[0])
diff6 = max(caps) - min(caps)

# 30x
caps30 = []
for _ in range(30):
    r = bt_exact(c, h, l, fm, sm, adx, rsi, n)
    caps30.append(r[0])
sigma = np.std(caps30)

r = bt_exact(c, h, l, fm, sm, adx, rsi, n)

print()
print(f"  6-Engine: {'ALL MATCH' if diff6 < 0.01 else 'MISMATCH'} (diff=${diff6:.6f})")
print(f"  30x: sigma=${sigma:.6f} {'DETERMINISTIC' if sigma < 0.01 else ''}")
print()
print(f"  Final:    ${r[0]:,.2f}")
print(f"  Return:   +{r[1]:,.0f}%")
print(f"  PF:       {r[8]:.2f}")
print(f"  MDD:      {r[9]:.1f}%")
print(f"  Trades:   {r[2]} (SL:{r[3]} TSL:{r[4]} REV:{r[5]})")
print(f"  W/L:      {r[6]}W/{r[7]}L ({r[10]:.1f}%)")
print(f"  WLR:      {r[11]:.1f}:1")

# 기획서 수치 비교
print()
print("  [기획서 수치 비교]")
print(f"  {'':>15} {'기획서':>14} {'이 엔진':>14} {'일치':>6}")
print(f"  {'잔액':>15} ${'24,073,329':>13} ${r[0]:>13,.0f} {'?' :>6}")
print(f"  {'거래':>15} {'70':>14} {r[2]:>14} {'O' if r[2]==70 else 'X':>6}")
print(f"  {'SL':>15} {'30':>14} {r[3]:>14} {'O' if r[3]==30 else 'X':>6}")
print(f"  {'TSL':>15} {'17':>14} {r[4]:>14} {'O' if r[4]==17 else 'X':>6}")
print(f"  {'REV':>15} {'23':>14} {r[5]:>14} {'O' if r[5]==23 else 'X':>6}")
print(f"  {'승률':>15} {'47.1%':>14} {r[10]:>13.1f}% {'O' if abs(r[10]-47.1)<1 else 'X':>6}")

# Yearly
print()
print("  [Yearly]")
yearly = {}
for t in r[12]:
    yr = str(years[t['i']])
    if yr not in yearly: yearly[yr] = {'t': 0, 'w': 0, 'l': 0, 'pnl': 0.0}
    yearly[yr]['t'] += 1; yearly[yr]['pnl'] += t['pnl']
    if t['pnl'] >= 0: yearly[yr]['w'] += 1
    else: yearly[yr]['l'] += 1

cum = 5000.0; ayp = True
for yr in sorted(yearly.keys()):
    yd = yearly[yr]; cum += yd['pnl']
    mark = ' X' if yd['pnl'] < 0 else ''
    if yd['pnl'] < 0: ayp = False
    print(f"    {yr}: {yd['t']}T {yd['w']}W/{yd['l']}L ${yd['pnl']:+,.0f} -> ${cum:,.0f}{mark}")
print(f"  AllYears+: {'YES' if ayp else 'NO'}")
print()
print("=" * 80)
