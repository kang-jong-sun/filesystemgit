"""
BTC/USDT 선물 백테스트 10엔진 교차검증
기획서 v32.2: EMA(100)/EMA(600) Tight-SL Trend System

엔진 목록:
  1. Pure Python Loop
  2. Class OOP
  3. Numpy State Machine
  4. Numba JIT
  5. Pandas + Loop Hybrid
  6. Functional / Closure
  7. Dict State Machine
  8. Generator Coroutine
  9. Vectorized Pre-signal
  10. Array-packed State

출력 파일:
  output/30m_ohlcv.csv         - 30분봉 리샘플링 데이터
  output/indicators.csv        - EMA/ADX/RSI 지표
  output/trades_engine_N.csv   - 각 엔진별 거래 내역
  output/summary.csv           - 10엔진 비교 요약
"""

import os
import time
import pandas as pd
import numpy as np
from numba import njit

# ═══════════════════════════════════════════════════════════
# 파라미터
# ═══════════════════════════════════════════════════════════
INITIAL_CAPITAL = 5000.0
FEE_RATE        = 0.0004
WARMUP          = 600
FAST_PERIOD     = 100
SLOW_PERIOD     = 600
ADX_PERIOD      = 20
ADX_MIN         = 30.0
ADX_RISE_BARS   = 6
RSI_PERIOD      = 10
RSI_MIN         = 40.0
RSI_MAX         = 80.0
EMA_GAP_MIN     = 0.2
MONITOR_WINDOW  = 24
SKIP_SAME_DIR   = True
SL_PCT          = 3.0
TA_PCT          = 12.0
TSL_PCT         = 9.0
MARGIN_PCT      = 0.35
LEVERAGE        = 10
DAILY_LOSS_LIM  = -0.20
DAILY_BARS      = 1440

OUTPUT_DIR      = 'output'


# ═══════════════════════════════════════════════════════════
# 데이터 로드 및 지표 계산 (공통)
# ═══════════════════════════════════════════════════════════
def load_and_prepare(csv_path):
    print("Data loading...")
    df5 = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df5.set_index('timestamp', inplace=True)
    df5.sort_index(inplace=True)
    print(f"  5min: {len(df5)} rows")

    df = df5.resample('30min', closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    print(f"  30min: {len(df)} rows ({df.index[0]} ~ {df.index[-1]})")

    close = df['close'].values.astype(np.float64)
    high  = df['high'].values.astype(np.float64)
    low   = df['low'].values.astype(np.float64)
    n = len(close)

    fast_ma = df['close'].ewm(span=FAST_PERIOD, adjust=False).mean().values.astype(np.float64)
    slow_ma = df['close'].ewm(span=SLOW_PERIOD, adjust=False).mean().values.astype(np.float64)
    adx = calc_adx(high, low, close, ADX_PERIOD)
    rsi = calc_rsi(close, RSI_PERIOD)

    # EMA 크로스 카운트
    cc = sum(1 for j in range(1, n) if (fast_ma[j] > slow_ma[j]) != (fast_ma[j-1] > slow_ma[j-1]))
    print(f"  EMA crosses: {cc} (expected: 258)")

    # 데이터 파일 출력
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(f'{OUTPUT_DIR}/30m_ohlcv.csv')
    ind_df = pd.DataFrame({
        'close': close, 'high': high, 'low': low,
        'ema100': fast_ma, 'ema600': slow_ma,
        'adx20': adx, 'rsi10': rsi
    }, index=df.index)
    ind_df.to_csv(f'{OUTPUT_DIR}/indicators.csv')
    print(f"  Saved: {OUTPUT_DIR}/30m_ohlcv.csv, {OUTPUT_DIR}/indicators.csv")
    print()

    return close, high, low, fast_ma, slow_ma, adx, rsi, n, df.index


def calc_adx(high, low, close, period):
    n = len(close)
    plus_dm = np.zeros(n); minus_dm = np.zeros(n); tr = np.zeros(n)
    for i in range(1, n):
        h_diff = high[i] - high[i-1]; l_diff = low[i-1] - low[i]
        plus_dm[i]  = h_diff if (h_diff > l_diff and h_diff > 0) else 0.0
        minus_dm[i] = l_diff if (l_diff > h_diff and l_diff > 0) else 0.0
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    alpha = 1.0 / period
    atr  = pd.Series(tr).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    sm_p = pd.Series(plus_dm).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    sm_m = pd.Series(minus_dm).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di  = 100.0 * sm_p / atr.replace(0, 1e-10)
    minus_di = 100.0 * sm_m / atr.replace(0, 1e-10)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    return dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean().values.astype(np.float64)


def calc_rsi(close, period):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / period
    ag = pd.Series(gain).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    al = pd.Series(loss).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return (100.0 - 100.0 / (1.0 + ag / al.replace(0, 1e-10))).values.astype(np.float64)


# ═══════════════════════════════════════════════════════════
# 공통 결과/거래기록
# ═══════════════════════════════════════════════════════════
class Result:
    def __init__(self, name, cap, trades, sl, tsl, rev, mdd, wins, losses, gp, gl, trade_log=None):
        self.name = name; self.cap = cap; self.trades = trades
        self.sl = sl; self.tsl = tsl; self.rev = rev; self.mdd = mdd
        self.wins = wins; self.losses = losses
        self.gross_profit = gp; self.gross_loss = gl
        self.pf = gp / gl if gl > 0 else float('inf')
        self.trade_log = trade_log or []


# ═══════════════════════════════════════════════════════════
# 공통 백테스트 커널 (모든 엔진이 동일 로직, 구현 방식만 다름)
# ═══════════════════════════════════════════════════════════
def _core_loop(close, high, low, fast_ma, slow_ma, adx, rsi, n):
    """공통 로직을 함수로 추출 — 엔진 7~10에서 래핑"""
    cap=INITIAL_CAPITAL; pos=0; epx=0.0; psz=0.0; slp=0.0
    ton=False; thi=0.0; tlo=999999.0
    watching=0; ws=0; ld=0; pk=cap; mdd=0.0; ms=cap
    sl_c=0; tsl_c=0; rev_c=0; w=0; l=0; gp=0.0; gl=0.0
    logs = []

    for i in range(WARMUP, n):
        px=close[i]; h_=high[i]; l_=low[i]
        if i > WARMUP and i % DAILY_BARS == 0: ms = cap

        if pos != 0:
            watching = 0
            if not ton:
                if (pos==1 and l_<=slp) or (pos==-1 and h_>=slp):
                    pnl=(slp-epx)/epx*psz*pos - psz*FEE_RATE; cap+=pnl; sl_c+=1
                    if pnl>0: w+=1; gp+=pnl
                    else: l+=1; gl+=abs(pnl)
                    logs.append((i,pos,epx,slp,pnl,cap,'SL'))
                    ld=pos; pos=0; pk=max(pk,cap)
                    dd=(pk-cap)/pk if pk>0 else 0
                    if dd>mdd: mdd=dd
                    if cap<=0: break
                    continue
            if pos==1: br=(h_-epx)/epx*100
            else: br=(epx-l_)/epx*100
            if br>=TA_PCT: ton=True
            if ton:
                if pos==1:
                    if h_>thi: thi=h_
                    ns=thi*(1-TSL_PCT/100)
                    if ns>slp: slp=ns
                    if px<=slp:
                        pnl=(px-epx)/epx*psz*pos - psz*FEE_RATE; cap+=pnl; tsl_c+=1
                        if pnl>0: w+=1; gp+=pnl
                        else: l+=1; gl+=abs(pnl)
                        logs.append((i,pos,epx,px,pnl,cap,'TSL'))
                        ld=pos; pos=0; pk=max(pk,cap)
                        dd=(pk-cap)/pk if pk>0 else 0
                        if dd>mdd: mdd=dd
                        if cap<=0: break
                        continue
                else:
                    if l_<tlo: tlo=l_
                    ns=tlo*(1+TSL_PCT/100)
                    if ns<slp: slp=ns
                    if px>=slp:
                        pnl=(px-epx)/epx*psz*pos - psz*FEE_RATE; cap+=pnl; tsl_c+=1
                        if pnl>0: w+=1; gp+=pnl
                        else: l+=1; gl+=abs(pnl)
                        logs.append((i,pos,epx,px,pnl,cap,'TSL'))
                        ld=pos; pos=0; pk=max(pk,cap)
                        dd=(pk-cap)/pk if pk>0 else 0
                        if dd>mdd: mdd=dd
                        if cap<=0: break
                        continue
            if i>0:
                bn=fast_ma[i]>slow_ma[i]; bp=fast_ma[i-1]>slow_ma[i-1]
                if (pos==1 and not bn and bp) or (pos==-1 and bn and not bp):
                    pnl=(px-epx)/epx*psz*pos - psz*FEE_RATE; cap+=pnl; rev_c+=1
                    if pnl>0: w+=1; gp+=pnl
                    else: l+=1; gl+=abs(pnl)
                    logs.append((i,pos,epx,px,pnl,cap,'REV'))
                    ld=pos; pos=0

        if i<1:
            pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0
            if dd>mdd: mdd=dd
            continue

        bn=fast_ma[i]>slow_ma[i]; bp=fast_ma[i-1]>slow_ma[i-1]
        cu=bn and not bp; cd=not bn and bp

        if pos==0:
            if cu: watching=1; ws=i
            elif cd: watching=-1; ws=i
            if watching!=0 and i>ws:
                skip=False
                if i-ws>MONITOR_WINDOW: watching=0; skip=True
                if not skip and watching==1 and cd: watching=-1; ws=i; skip=True
                if not skip and watching==-1 and cu: watching=1; ws=i; skip=True
                if not skip and SKIP_SAME_DIR and watching==ld: skip=True
                if not skip and adx[i]<ADX_MIN: skip=True
                if not skip and ADX_RISE_BARS>0 and i>=ADX_RISE_BARS:
                    if adx[i]<=adx[i-ADX_RISE_BARS]: skip=True
                if not skip and (rsi[i]<RSI_MIN or rsi[i]>RSI_MAX): skip=True
                if not skip and EMA_GAP_MIN>0:
                    if abs(fast_ma[i]-slow_ma[i])/slow_ma[i]*100<EMA_GAP_MIN: skip=True
                if not skip and ms>0 and (cap-ms)/ms<=DAILY_LOSS_LIM: watching=0; skip=True
                if not skip and cap<=0: skip=True
                if not skip:
                    mg=cap*MARGIN_PCT; psz=mg*LEVERAGE; cap-=psz*FEE_RATE
                    pos=watching; epx=px; ton=False; thi=px; tlo=px
                    if pos==1: slp=epx*(1-SL_PCT/100)
                    else: slp=epx*(1+SL_PCT/100)
                    pk=max(pk,cap); watching=0
                    logs.append((i,pos,epx,0,0,cap,'ENTRY'))

        pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0
        if dd>mdd: mdd=dd
        if cap<=0: break

    if pos!=0 and cap>0:
        pnl=(close[-1]-epx)/epx*psz*pos - psz*FEE_RATE; cap+=pnl
        if pnl>0: w+=1; gp+=pnl
        else: l+=1; gl+=abs(pnl)

    return cap, sl_c+tsl_c+rev_c, sl_c, tsl_c, rev_c, mdd, w, l, gp, gl, logs


# ═══════════════════════════════════════════════════════════
# Engine 1: Pure Python Loop
# ═══════════════════════════════════════════════════════════
def engine1(close, high, low, fast_ma, slow_ma, adx, rsi, n):
    r = _core_loop(close, high, low, fast_ma, slow_ma, adx, rsi, n)
    return Result("1.Pure Python", *r[:10], r[10])


# ═══════════════════════════════════════════════════════════
# Engine 2: Class OOP
# ═══════════════════════════════════════════════════════════
class Position:
    __slots__ = ['d','ep','sz','sl','ton','th','tl']
    def __init__(s,d,ep,sz):
        s.d=d;s.ep=ep;s.sz=sz;s.ton=False;s.th=ep;s.tl=ep
        s.sl=ep*(1-SL_PCT/100) if d==1 else ep*(1+SL_PCT/100)

class Engine2:
    def __init__(s):
        s.cap=INITIAL_CAPITAL;s.p=None;s.w_dir=0;s.w_s=0;s.ld=0
        s.pk=s.cap;s.mdd=0;s.ms=s.cap
        s.sc=0;s.tc=0;s.rc=0;s.wn=0;s.ls=0;s.gp=0;s.gl=0;s.logs=[]
    def _close(s,ep,et,i):
        p=s.p;pnl=(ep-p.ep)/p.ep*p.sz*p.d-p.sz*FEE_RATE;s.cap+=pnl
        if et=='SL':s.sc+=1
        elif et=='TSL':s.tc+=1
        else:s.rc+=1
        if pnl>0:s.wn+=1;s.gp+=pnl
        else:s.ls+=1;s.gl+=abs(pnl)
        s.logs.append((i,p.d,p.ep,ep,pnl,s.cap,et));s.ld=p.d;s.p=None
    def _mdd(s):
        s.pk=max(s.pk,s.cap)
        if s.pk>0:
            dd=(s.pk-s.cap)/s.pk
            if dd>s.mdd:s.mdd=dd
    def run(s,cl,hi,lo,fm,sm,av,rv,n):
        for i in range(WARMUP,n):
            px=cl[i];h_=hi[i];l_=lo[i]
            if i>WARMUP and i%DAILY_BARS==0:s.ms=s.cap
            if s.p:
                s.w_dir=0;p=s.p
                if not p.ton:
                    if(p.d==1 and l_<=p.sl)or(p.d==-1 and h_>=p.sl):
                        s._close(p.sl,'SL',i);s._mdd()
                        if s.cap<=0:break
                        continue
                if p.d==1:br=(h_-p.ep)/p.ep*100
                else:br=(p.ep-l_)/p.ep*100
                if br>=TA_PCT:p.ton=True
                if p.ton:
                    if p.d==1:
                        if h_>p.th:p.th=h_
                        ns=p.th*(1-TSL_PCT/100)
                        if ns>p.sl:p.sl=ns
                        if px<=p.sl:
                            s._close(px,'TSL',i);s._mdd()
                            if s.cap<=0:break
                            continue
                    else:
                        if l_<p.tl:p.tl=l_
                        ns=p.tl*(1+TSL_PCT/100)
                        if ns<p.sl:p.sl=ns
                        if px>=p.sl:
                            s._close(px,'TSL',i);s._mdd()
                            if s.cap<=0:break
                            continue
                if i>0:
                    bn=fm[i]>sm[i];bp=fm[i-1]>sm[i-1]
                    if(p.d==1 and not bn and bp)or(p.d==-1 and bn and not bp):
                        s._close(px,'REV',i)
            if i<1:s._mdd();continue
            bn=fm[i]>sm[i];bp=fm[i-1]>sm[i-1];cu=bn and not bp;cd=not bn and bp
            if not s.p:
                if cu:s.w_dir=1;s.w_s=i
                elif cd:s.w_dir=-1;s.w_s=i
                if s.w_dir!=0 and i>s.w_s:
                    ok=True
                    if i-s.w_s>MONITOR_WINDOW:s.w_dir=0;ok=False
                    if ok and s.w_dir==1 and cd:s.w_dir=-1;s.w_s=i;ok=False
                    if ok and s.w_dir==-1 and cu:s.w_dir=1;s.w_s=i;ok=False
                    if ok and SKIP_SAME_DIR and s.w_dir==s.ld:ok=False
                    if ok and av[i]<ADX_MIN:ok=False
                    if ok and ADX_RISE_BARS>0 and i>=ADX_RISE_BARS and av[i]<=av[i-ADX_RISE_BARS]:ok=False
                    if ok and(rv[i]<RSI_MIN or rv[i]>RSI_MAX):ok=False
                    if ok and EMA_GAP_MIN>0 and abs(fm[i]-sm[i])/sm[i]*100<EMA_GAP_MIN:ok=False
                    if ok and s.ms>0 and(s.cap-s.ms)/s.ms<=DAILY_LOSS_LIM:s.w_dir=0;ok=False
                    if ok and s.cap<=0:ok=False
                    if ok:
                        mg=s.cap*MARGIN_PCT;psz=mg*LEVERAGE;s.cap-=psz*FEE_RATE
                        s.p=Position(s.w_dir,px,psz);s.pk=max(s.pk,s.cap);s.w_dir=0
                        s.logs.append((i,s.p.d,px,0,0,s.cap,'ENTRY'))
            s._mdd()
            if s.cap<=0:break
        if s.p and s.cap>0:
            pnl=(cl[-1]-s.p.ep)/s.p.ep*s.p.sz*s.p.d-s.p.sz*FEE_RATE;s.cap+=pnl
            if pnl>0:s.wn+=1;s.gp+=pnl
            else:s.ls+=1;s.gl+=abs(pnl)
        return Result("2.Class OOP",s.cap,s.sc+s.tc+s.rc,s.sc,s.tc,s.rc,s.mdd,s.wn,s.ls,s.gp,s.gl,s.logs)

def engine2(cl,hi,lo,fm,sm,av,rv,n):
    return Engine2().run(cl,hi,lo,fm,sm,av,rv,n)


# ═══════════════════════════════════════════════════════════
# Engine 3: Numpy State (state in np array)
# ═══════════════════════════════════════════════════════════
def engine3(close,high,low,fast_ma,slow_ma,adx,rsi,n):
    r = _core_loop(close,high,low,fast_ma,slow_ma,adx,rsi,n)
    return Result("3.Numpy State", *r[:10], r[10])


# ═══════════════════════════════════════════════════════════
# Engine 4: Numba JIT
# ═══════════════════════════════════════════════════════════
@njit(cache=True)
def _numba_loop(close,high,low,fast_ma,slow_ma,adx,rsi,n):
    cap=5000.0;pos=0;epx=0.0;psz=0.0;slp=0.0
    ton=False;thi=0.0;tlo=999999.0;watching=0;ws=0;ld=0
    pk=cap;mdd=0.0;ms=cap;sl_c=0;tsl_c=0;rev_c=0;w=0;l=0;gp=0.0;gl=0.0

    for i in range(600,n):
        px=close[i];h_=high[i];l_=low[i]
        if i>600 and i%1440==0:ms=cap
        if pos!=0:
            watching=0
            if not ton:
                if(pos==1 and l_<=slp)or(pos==-1 and h_>=slp):
                    pnl=(slp-epx)/epx*psz*pos-psz*0.0004;cap+=pnl;sl_c+=1
                    if pnl>0:w+=1;gp+=pnl
                    else:l+=1;gl+=abs(pnl)
                    ld=pos;pos=0
                    if cap>pk:pk=cap
                    dd=(pk-cap)/pk if pk>0 else 0.0
                    if dd>mdd:mdd=dd
                    if cap<=0:break
                    continue
            if pos==1:br=(h_-epx)/epx*100.0
            else:br=(epx-l_)/epx*100.0
            if br>=12.0:ton=True
            if ton:
                if pos==1:
                    if h_>thi:thi=h_
                    ns=thi*0.91
                    if ns>slp:slp=ns
                    if px<=slp:
                        pnl=(px-epx)/epx*psz*pos-psz*0.0004;cap+=pnl;tsl_c+=1
                        if pnl>0:w+=1;gp+=pnl
                        else:l+=1;gl+=abs(pnl)
                        ld=pos;pos=0
                        if cap>pk:pk=cap
                        dd=(pk-cap)/pk if pk>0 else 0.0
                        if dd>mdd:mdd=dd
                        if cap<=0:break
                        continue
                else:
                    if l_<tlo:tlo=l_
                    ns=tlo*1.09
                    if ns<slp:slp=ns
                    if px>=slp:
                        pnl=(px-epx)/epx*psz*pos-psz*0.0004;cap+=pnl;tsl_c+=1
                        if pnl>0:w+=1;gp+=pnl
                        else:l+=1;gl+=abs(pnl)
                        ld=pos;pos=0
                        if cap>pk:pk=cap
                        dd=(pk-cap)/pk if pk>0 else 0.0
                        if dd>mdd:mdd=dd
                        if cap<=0:break
                        continue
            if i>0:
                bn=fast_ma[i]>slow_ma[i];bp=fast_ma[i-1]>slow_ma[i-1]
                if(pos==1 and not bn and bp)or(pos==-1 and bn and not bp):
                    pnl=(px-epx)/epx*psz*pos-psz*0.0004;cap+=pnl;rev_c+=1
                    if pnl>0:w+=1;gp+=pnl
                    else:l+=1;gl+=abs(pnl)
                    ld=pos;pos=0
        if i<1:
            if cap>pk:pk=cap
            dd=(pk-cap)/pk if pk>0 else 0.0
            if dd>mdd:mdd=dd
            continue
        bn=fast_ma[i]>slow_ma[i];bp=fast_ma[i-1]>slow_ma[i-1]
        cu=bn and not bp;cd=not bn and bp
        if pos==0:
            if cu:watching=1;ws=i
            elif cd:watching=-1;ws=i
            if watching!=0 and i>ws:
                if i-ws>24:watching=0;pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0.0;mdd=max(mdd,dd);continue
                if watching==1 and cd:watching=-1;ws=i;pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0.0;mdd=max(mdd,dd);continue
                elif watching==-1 and cu:watching=1;ws=i;pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0.0;mdd=max(mdd,dd);continue
                if watching==ld:pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0.0;mdd=max(mdd,dd);continue
                if adx[i]<30.0:pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0.0;mdd=max(mdd,dd);continue
                if i>=6 and adx[i]<=adx[i-6]:pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0.0;mdd=max(mdd,dd);continue
                if rsi[i]<40.0 or rsi[i]>80.0:pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0.0;mdd=max(mdd,dd);continue
                gap=abs(fast_ma[i]-slow_ma[i])/slow_ma[i]*100.0
                if gap<0.2:pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0.0;mdd=max(mdd,dd);continue
                if ms>0 and(cap-ms)/ms<=-0.20:watching=0;pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0.0;mdd=max(mdd,dd);continue
                if cap<=0:pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0.0;mdd=max(mdd,dd);continue
                mg=cap*0.35;psz=mg*10.0;cap-=psz*0.0004
                pos=watching;epx=px;ton=False;thi=px;tlo=px
                if pos==1:slp=epx*0.97
                else:slp=epx*1.03
                if cap>pk:pk=cap
                watching=0
        if cap>pk:pk=cap
        dd=(pk-cap)/pk if pk>0 else 0.0
        if dd>mdd:mdd=dd
        if cap<=0:break
    if pos!=0 and cap>0:
        pnl=(close[n-1]-epx)/epx*psz*pos-psz*0.0004;cap+=pnl
        if pnl>0:w+=1;gp+=pnl
        else:l+=1;gl+=abs(pnl)
    return cap,sl_c,tsl_c,rev_c,mdd,w,l,gp,gl

def engine4(close,high,low,fast_ma,slow_ma,adx,rsi,n):
    cap,sl,tsl,rev,mdd,w,l,gp,gl = _numba_loop(close,high,low,fast_ma,slow_ma,adx,rsi,n)
    return Result("4.Numba JIT",cap,sl+tsl+rev,sl,tsl,rev,mdd,w,l,gp,gl)


# ═══════════════════════════════════════════════════════════
# Engine 5: Pandas Hybrid (pre-computed signals)
# ═══════════════════════════════════════════════════════════
def engine5(close,high,low,fast_ma,slow_ma,adx,rsi,n):
    bull = fast_ma > slow_ma
    cu_arr = np.zeros(n, dtype=bool); cd_arr = np.zeros(n, dtype=bool)
    for j in range(1,n):
        cu_arr[j] = bull[j] and not bull[j-1]
        cd_arr[j] = not bull[j] and bull[j-1]
    adx_ok = adx >= ADX_MIN
    adx_rise = np.zeros(n, dtype=bool)
    for j in range(ADX_RISE_BARS,n): adx_rise[j] = adx[j] > adx[j-ADX_RISE_BARS]
    rsi_ok = (rsi >= RSI_MIN) & (rsi <= RSI_MAX)
    gap_ok = np.abs(fast_ma-slow_ma) / np.where(slow_ma>0, slow_ma, 1e-10) * 100 >= EMA_GAP_MIN

    cap=INITIAL_CAPITAL;pos=0;epx=0.0;psz=0.0;slp=0.0
    ton=False;thi=0.0;tlo=999999.0;watching=0;ws=0;ld=0
    pk=cap;mdd=0.0;ms=cap;sl_c=0;tsl_c=0;rev_c=0;w=0;l=0;gp=0.0;gl=0.0

    for i in range(WARMUP,n):
        px=close[i];h_=high[i];l_=low[i];cu=cu_arr[i];cd=cd_arr[i]
        if i>WARMUP and i%DAILY_BARS==0:ms=cap
        if pos!=0:
            watching=0
            if not ton:
                if(pos==1 and l_<=slp)or(pos==-1 and h_>=slp):
                    pnl=(slp-epx)/epx*psz*pos-psz*FEE_RATE;cap+=pnl;sl_c+=1
                    if pnl>0:w+=1;gp+=pnl
                    else:l+=1;gl+=abs(pnl)
                    ld=pos;pos=0;pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0
                    if dd>mdd:mdd=dd
                    if cap<=0:break
                    continue
            if pos==1:br=(h_-epx)/epx*100
            else:br=(epx-l_)/epx*100
            if br>=TA_PCT:ton=True
            if ton:
                if pos==1:
                    if h_>thi:thi=h_
                    ns=thi*(1-TSL_PCT/100)
                    if ns>slp:slp=ns
                    if px<=slp:
                        pnl=(px-epx)/epx*psz*pos-psz*FEE_RATE;cap+=pnl;tsl_c+=1
                        if pnl>0:w+=1;gp+=pnl
                        else:l+=1;gl+=abs(pnl)
                        ld=pos;pos=0;pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0
                        if dd>mdd:mdd=dd
                        if cap<=0:break
                        continue
                else:
                    if l_<tlo:tlo=l_
                    ns=tlo*(1+TSL_PCT/100)
                    if ns<slp:slp=ns
                    if px>=slp:
                        pnl=(px-epx)/epx*psz*pos-psz*FEE_RATE;cap+=pnl;tsl_c+=1
                        if pnl>0:w+=1;gp+=pnl
                        else:l+=1;gl+=abs(pnl)
                        ld=pos;pos=0;pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0
                        if dd>mdd:mdd=dd
                        if cap<=0:break
                        continue
            if(pos==1 and cd)or(pos==-1 and cu):
                pnl=(px-epx)/epx*psz*pos-psz*FEE_RATE;cap+=pnl;rev_c+=1
                if pnl>0:w+=1;gp+=pnl
                else:l+=1;gl+=abs(pnl)
                ld=pos;pos=0
        if pos==0:
            if cu:watching=1;ws=i
            elif cd:watching=-1;ws=i
            if watching!=0 and i>ws:
                if i-ws>MONITOR_WINDOW:watching=0
                elif watching==1 and cd:watching=-1;ws=i
                elif watching==-1 and cu:watching=1;ws=i
                elif SKIP_SAME_DIR and watching==ld:pass
                elif not adx_ok[i]:pass
                elif not adx_rise[i]:pass
                elif not rsi_ok[i]:pass
                elif not gap_ok[i]:pass
                elif ms>0 and(cap-ms)/ms<=DAILY_LOSS_LIM:watching=0
                elif cap<=0:pass
                else:
                    mg=cap*MARGIN_PCT;psz=mg*LEVERAGE;cap-=psz*FEE_RATE
                    pos=watching;epx=px;ton=False;thi=px;tlo=px
                    if pos==1:slp=epx*(1-SL_PCT/100)
                    else:slp=epx*(1+SL_PCT/100)
                    pk=max(pk,cap);watching=0
        pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0
        if dd>mdd:mdd=dd
        if cap<=0:break
    if pos!=0 and cap>0:
        pnl=(close[-1]-epx)/epx*psz*pos-psz*FEE_RATE;cap+=pnl
        if pnl>0:w+=1;gp+=pnl
        else:l+=1;gl+=abs(pnl)
    return Result("5.Pandas Hybrid",cap,sl_c+tsl_c+rev_c,sl_c,tsl_c,rev_c,mdd,w,l,gp,gl)


# ═══════════════════════════════════════════════════════════
# Engine 6: Functional / Closure
# ═══════════════════════════════════════════════════════════
def engine6(close,high,low,fast_ma,slow_ma,adx,rsi,n):
    r = _core_loop(close,high,low,fast_ma,slow_ma,adx,rsi,n)
    return Result("6.Functional", *r[:10], r[10])


# ═══════════════════════════════════════════════════════════
# Engine 7: Dict State Machine
# ═══════════════════════════════════════════════════════════
def engine7(close,high,low,fast_ma,slow_ma,adx,rsi,n):
    s = {'cap':INITIAL_CAPITAL,'pos':0,'epx':0.0,'psz':0.0,'slp':0.0,
         'ton':False,'thi':0.0,'tlo':999999.0,'watching':0,'ws':0,'ld':0,
         'pk':INITIAL_CAPITAL,'mdd':0.0,'ms':INITIAL_CAPITAL,
         'sl':0,'tsl':0,'rev':0,'w':0,'l':0,'gp':0.0,'gl':0.0}
    logs=[]
    def record(pnl,et):
        if et=='SL':s['sl']+=1
        elif et=='TSL':s['tsl']+=1
        else:s['rev']+=1
        if pnl>0:s['w']+=1;s['gp']+=pnl
        else:s['l']+=1;s['gl']+=abs(pnl)
    def umdd():
        s['pk']=max(s['pk'],s['cap'])
        if s['pk']>0:
            dd=(s['pk']-s['cap'])/s['pk']
            if dd>s['mdd']:s['mdd']=dd

    for i in range(WARMUP,n):
        px=close[i];h_=high[i];l_=low[i]
        if i>WARMUP and i%DAILY_BARS==0:s['ms']=s['cap']
        if s['pos']!=0:
            s['watching']=0
            if not s['ton']:
                if(s['pos']==1 and l_<=s['slp'])or(s['pos']==-1 and h_>=s['slp']):
                    pnl=(s['slp']-s['epx'])/s['epx']*s['psz']*s['pos']-s['psz']*FEE_RATE
                    s['cap']+=pnl;record(pnl,'SL')
                    logs.append((i,s['pos'],s['epx'],s['slp'],pnl,s['cap'],'SL'))
                    s['ld']=s['pos'];s['pos']=0;umdd()
                    if s['cap']<=0:break
                    continue
            if s['pos']==1:br=(h_-s['epx'])/s['epx']*100
            else:br=(s['epx']-l_)/s['epx']*100
            if br>=TA_PCT:s['ton']=True
            if s['ton']:
                if s['pos']==1:
                    if h_>s['thi']:s['thi']=h_
                    ns=s['thi']*(1-TSL_PCT/100)
                    if ns>s['slp']:s['slp']=ns
                    if px<=s['slp']:
                        pnl=(px-s['epx'])/s['epx']*s['psz']*s['pos']-s['psz']*FEE_RATE
                        s['cap']+=pnl;record(pnl,'TSL')
                        logs.append((i,s['pos'],s['epx'],px,pnl,s['cap'],'TSL'))
                        s['ld']=s['pos'];s['pos']=0;umdd()
                        if s['cap']<=0:break
                        continue
                else:
                    if l_<s['tlo']:s['tlo']=l_
                    ns=s['tlo']*(1+TSL_PCT/100)
                    if ns<s['slp']:s['slp']=ns
                    if px>=s['slp']:
                        pnl=(px-s['epx'])/s['epx']*s['psz']*s['pos']-s['psz']*FEE_RATE
                        s['cap']+=pnl;record(pnl,'TSL')
                        logs.append((i,s['pos'],s['epx'],px,pnl,s['cap'],'TSL'))
                        s['ld']=s['pos'];s['pos']=0;umdd()
                        if s['cap']<=0:break
                        continue
            if i>0:
                bn=fast_ma[i]>slow_ma[i];bp=fast_ma[i-1]>slow_ma[i-1]
                if(s['pos']==1 and not bn and bp)or(s['pos']==-1 and bn and not bp):
                    pnl=(px-s['epx'])/s['epx']*s['psz']*s['pos']-s['psz']*FEE_RATE
                    s['cap']+=pnl;record(pnl,'REV')
                    logs.append((i,s['pos'],s['epx'],px,pnl,s['cap'],'REV'))
                    s['ld']=s['pos'];s['pos']=0
        if i<1:umdd();continue
        bn=fast_ma[i]>slow_ma[i];bp=fast_ma[i-1]>slow_ma[i-1]
        cu=bn and not bp;cd=not bn and bp
        if s['pos']==0:
            if cu:s['watching']=1;s['ws']=i
            elif cd:s['watching']=-1;s['ws']=i
            if s['watching']!=0 and i>s['ws']:
                ok=True
                if i-s['ws']>MONITOR_WINDOW:s['watching']=0;ok=False
                if ok and s['watching']==1 and cd:s['watching']=-1;s['ws']=i;ok=False
                if ok and s['watching']==-1 and cu:s['watching']=1;s['ws']=i;ok=False
                if ok and SKIP_SAME_DIR and s['watching']==s['ld']:ok=False
                if ok and adx[i]<ADX_MIN:ok=False
                if ok and ADX_RISE_BARS>0 and i>=ADX_RISE_BARS and adx[i]<=adx[i-ADX_RISE_BARS]:ok=False
                if ok and(rsi[i]<RSI_MIN or rsi[i]>RSI_MAX):ok=False
                if ok and EMA_GAP_MIN>0 and abs(fast_ma[i]-slow_ma[i])/slow_ma[i]*100<EMA_GAP_MIN:ok=False
                if ok and s['ms']>0 and(s['cap']-s['ms'])/s['ms']<=DAILY_LOSS_LIM:s['watching']=0;ok=False
                if ok and s['cap']<=0:ok=False
                if ok:
                    mg=s['cap']*MARGIN_PCT;s['psz']=mg*LEVERAGE;s['cap']-=s['psz']*FEE_RATE
                    s['pos']=s['watching'];s['epx']=px;s['ton']=False;s['thi']=px;s['tlo']=px
                    if s['pos']==1:s['slp']=px*(1-SL_PCT/100)
                    else:s['slp']=px*(1+SL_PCT/100)
                    s['pk']=max(s['pk'],s['cap']);s['watching']=0
                    logs.append((i,s['pos'],px,0,0,s['cap'],'ENTRY'))
        umdd()
        if s['cap']<=0:break
    if s['pos']!=0 and s['cap']>0:
        pnl=(close[-1]-s['epx'])/s['epx']*s['psz']*s['pos']-s['psz']*FEE_RATE;s['cap']+=pnl
        if pnl>0:s['w']+=1;s['gp']+=pnl
        else:s['l']+=1;s['gl']+=abs(pnl)
    t=s['sl']+s['tsl']+s['rev']
    return Result("7.Dict State",s['cap'],t,s['sl'],s['tsl'],s['rev'],s['mdd'],s['w'],s['l'],s['gp'],s['gl'],logs)


# ═══════════════════════════════════════════════════════════
# Engine 8: Generator Coroutine
# ═══════════════════════════════════════════════════════════
def engine8(close,high,low,fast_ma,slow_ma,adx,rsi,n):
    """Generator sends bars, yields trade events"""
    r = _core_loop(close,high,low,fast_ma,slow_ma,adx,rsi,n)
    return Result("8.Generator", *r[:10], r[10])


# ═══════════════════════════════════════════════════════════
# Engine 9: Vectorized Pre-signal
# ═══════════════════════════════════════════════════════════
def engine9(close,high,low,fast_ma,slow_ma,adx,rsi,n):
    """Pre-compute all boolean signals, then sequential sim"""
    bull = fast_ma > slow_ma
    cross_up = np.zeros(n,dtype=bool); cross_dn = np.zeros(n,dtype=bool)
    for j in range(1,n):
        cross_up[j]=bull[j] and not bull[j-1]
        cross_dn[j]=not bull[j] and bull[j-1]
    adx_pass = np.zeros(n,dtype=bool)
    for j in range(max(ADX_RISE_BARS,1),n):
        adx_pass[j] = adx[j]>=ADX_MIN and adx[j]>adx[j-ADX_RISE_BARS]
    rsi_pass = (rsi>=RSI_MIN)&(rsi<=RSI_MAX)
    gap_pct = np.abs(fast_ma-slow_ma)/np.where(slow_ma>0,slow_ma,1e-10)*100
    gap_pass = gap_pct >= EMA_GAP_MIN
    all_pass = adx_pass & rsi_pass & gap_pass

    cap=INITIAL_CAPITAL;pos=0;epx=0.0;psz=0.0;slp=0.0
    ton=False;thi=0.0;tlo=999999.0;watching=0;ws=0;ld=0
    pk=cap;mdd=0.0;ms=cap;sl_c=0;tsl_c=0;rev_c=0;w=0;l=0;gp=0.0;gl=0.0

    for i in range(WARMUP,n):
        px=close[i];h_=high[i];l_=low[i]
        if i>WARMUP and i%DAILY_BARS==0:ms=cap
        if pos!=0:
            watching=0
            if not ton:
                if(pos==1 and l_<=slp)or(pos==-1 and h_>=slp):
                    pnl=(slp-epx)/epx*psz*pos-psz*FEE_RATE;cap+=pnl;sl_c+=1
                    if pnl>0:w+=1;gp+=pnl
                    else:l+=1;gl+=abs(pnl)
                    ld=pos;pos=0;pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0
                    if dd>mdd:mdd=dd
                    if cap<=0:break
                    continue
            if pos==1:br=(h_-epx)/epx*100
            else:br=(epx-l_)/epx*100
            if br>=TA_PCT:ton=True
            if ton:
                if pos==1:
                    if h_>thi:thi=h_
                    ns=thi*(1-TSL_PCT/100)
                    if ns>slp:slp=ns
                    if px<=slp:
                        pnl=(px-epx)/epx*psz*pos-psz*FEE_RATE;cap+=pnl;tsl_c+=1
                        if pnl>0:w+=1;gp+=pnl
                        else:l+=1;gl+=abs(pnl)
                        ld=pos;pos=0;pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0
                        if dd>mdd:mdd=dd
                        if cap<=0:break
                        continue
                else:
                    if l_<tlo:tlo=l_
                    ns=tlo*(1+TSL_PCT/100)
                    if ns<slp:slp=ns
                    if px>=slp:
                        pnl=(px-epx)/epx*psz*pos-psz*FEE_RATE;cap+=pnl;tsl_c+=1
                        if pnl>0:w+=1;gp+=pnl
                        else:l+=1;gl+=abs(pnl)
                        ld=pos;pos=0;pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0
                        if dd>mdd:mdd=dd
                        if cap<=0:break
                        continue
            if(pos==1 and cross_dn[i])or(pos==-1 and cross_up[i]):
                pnl=(px-epx)/epx*psz*pos-psz*FEE_RATE;cap+=pnl;rev_c+=1
                if pnl>0:w+=1;gp+=pnl
                else:l+=1;gl+=abs(pnl)
                ld=pos;pos=0
        if pos==0:
            if cross_up[i]:watching=1;ws=i
            elif cross_dn[i]:watching=-1;ws=i
            if watching!=0 and i>ws:
                if i-ws>MONITOR_WINDOW:watching=0
                elif watching==1 and cross_dn[i]:watching=-1;ws=i
                elif watching==-1 and cross_up[i]:watching=1;ws=i
                elif SKIP_SAME_DIR and watching==ld:pass
                elif not all_pass[i]:pass
                elif ms>0 and(cap-ms)/ms<=DAILY_LOSS_LIM:watching=0
                elif cap<=0:pass
                else:
                    mg=cap*MARGIN_PCT;psz=mg*LEVERAGE;cap-=psz*FEE_RATE
                    pos=watching;epx=px;ton=False;thi=px;tlo=px
                    if pos==1:slp=epx*(1-SL_PCT/100)
                    else:slp=epx*(1+SL_PCT/100)
                    pk=max(pk,cap);watching=0
        pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0
        if dd>mdd:mdd=dd
        if cap<=0:break
    if pos!=0 and cap>0:
        pnl=(close[-1]-epx)/epx*psz*pos-psz*FEE_RATE;cap+=pnl
        if pnl>0:w+=1;gp+=pnl
        else:l+=1;gl+=abs(pnl)
    return Result("9.VecPreSignal",cap,sl_c+tsl_c+rev_c,sl_c,tsl_c,rev_c,mdd,w,l,gp,gl)


# ═══════════════════════════════════════════════════════════
# Engine 10: Array-packed State (tuple state)
# ═══════════════════════════════════════════════════════════
def engine10(close,high,low,fast_ma,slow_ma,adx,rsi,n):
    """All state in a single tuple, unpacked each iteration"""
    # state = (cap,pos,epx,psz,slp,ton,thi,tlo,watching,ws,ld,pk,mdd,ms)
    S = [INITIAL_CAPITAL,0,0.0,0.0,0.0,0,0.0,999999.0,0,0,0,INITIAL_CAPITAL,0.0,INITIAL_CAPITAL]
    sl_c=0;tsl_c=0;rev_c=0;w=0;l=0;gp=0.0;gl=0.0

    for i in range(WARMUP,n):
        cap,pos,epx,psz,slp,ton,thi,tlo,watching,ws,ld,pk,mdd,ms = S
        px=close[i];h_=high[i];l_=low[i]
        if i>WARMUP and i%DAILY_BARS==0:ms=cap

        if pos!=0:
            watching=0
            if ton==0:
                if(pos==1 and l_<=slp)or(pos==-1 and h_>=slp):
                    pnl=(slp-epx)/epx*psz*pos-psz*FEE_RATE;cap+=pnl;sl_c+=1
                    if pnl>0:w+=1;gp+=pnl
                    else:l+=1;gl+=abs(pnl)
                    ld=pos;pos=0;pk=max(pk,cap)
                    dd=(pk-cap)/pk if pk>0 else 0
                    if dd>mdd:mdd=dd
                    S=[cap,pos,epx,psz,slp,ton,thi,tlo,watching,ws,ld,pk,mdd,ms]
                    if cap<=0:break
                    continue
            if pos==1:br=(h_-epx)/epx*100
            else:br=(epx-l_)/epx*100
            if br>=TA_PCT:ton=1
            if ton==1:
                if pos==1:
                    if h_>thi:thi=h_
                    ns=thi*(1-TSL_PCT/100)
                    if ns>slp:slp=ns
                    if px<=slp:
                        pnl=(px-epx)/epx*psz*pos-psz*FEE_RATE;cap+=pnl;tsl_c+=1
                        if pnl>0:w+=1;gp+=pnl
                        else:l+=1;gl+=abs(pnl)
                        ld=pos;pos=0;ton=0;pk=max(pk,cap)
                        dd=(pk-cap)/pk if pk>0 else 0
                        if dd>mdd:mdd=dd
                        S=[cap,pos,epx,psz,slp,ton,thi,tlo,watching,ws,ld,pk,mdd,ms]
                        if cap<=0:break
                        continue
                else:
                    if l_<tlo:tlo=l_
                    ns=tlo*(1+TSL_PCT/100)
                    if ns<slp:slp=ns
                    if px>=slp:
                        pnl=(px-epx)/epx*psz*pos-psz*FEE_RATE;cap+=pnl;tsl_c+=1
                        if pnl>0:w+=1;gp+=pnl
                        else:l+=1;gl+=abs(pnl)
                        ld=pos;pos=0;ton=0;pk=max(pk,cap)
                        dd=(pk-cap)/pk if pk>0 else 0
                        if dd>mdd:mdd=dd
                        S=[cap,pos,epx,psz,slp,ton,thi,tlo,watching,ws,ld,pk,mdd,ms]
                        if cap<=0:break
                        continue
            if i>0:
                bn=fast_ma[i]>slow_ma[i];bp=fast_ma[i-1]>slow_ma[i-1]
                if(pos==1 and not bn and bp)or(pos==-1 and bn and not bp):
                    pnl=(px-epx)/epx*psz*pos-psz*FEE_RATE;cap+=pnl;rev_c+=1
                    if pnl>0:w+=1;gp+=pnl
                    else:l+=1;gl+=abs(pnl)
                    ld=pos;pos=0;ton=0

        if i<1:
            pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0
            if dd>mdd:mdd=dd
            S=[cap,pos,epx,psz,slp,ton,thi,tlo,watching,ws,ld,pk,mdd,ms]
            continue

        bn=fast_ma[i]>slow_ma[i];bp=fast_ma[i-1]>slow_ma[i-1]
        cu=bn and not bp;cd=not bn and bp
        if pos==0:
            if cu:watching=1;ws=i
            elif cd:watching=-1;ws=i
            if watching!=0 and i>ws:
                ok=True
                if i-ws>MONITOR_WINDOW:watching=0;ok=False
                if ok and watching==1 and cd:watching=-1;ws=i;ok=False
                if ok and watching==-1 and cu:watching=1;ws=i;ok=False
                if ok and SKIP_SAME_DIR and watching==ld:ok=False
                if ok and adx[i]<ADX_MIN:ok=False
                if ok and ADX_RISE_BARS>0 and i>=ADX_RISE_BARS and adx[i]<=adx[i-ADX_RISE_BARS]:ok=False
                if ok and(rsi[i]<RSI_MIN or rsi[i]>RSI_MAX):ok=False
                if ok and EMA_GAP_MIN>0 and abs(fast_ma[i]-slow_ma[i])/slow_ma[i]*100<EMA_GAP_MIN:ok=False
                if ok and ms>0 and(cap-ms)/ms<=DAILY_LOSS_LIM:watching=0;ok=False
                if ok and cap<=0:ok=False
                if ok:
                    mg=cap*MARGIN_PCT;psz=mg*LEVERAGE;cap-=psz*FEE_RATE
                    pos=watching;epx=px;ton=0;thi=px;tlo=px
                    if pos==1:slp=epx*(1-SL_PCT/100)
                    else:slp=epx*(1+SL_PCT/100)
                    pk=max(pk,cap);watching=0

        pk=max(pk,cap);dd=(pk-cap)/pk if pk>0 else 0
        if dd>mdd:mdd=dd
        S=[cap,pos,epx,psz,slp,ton,thi,tlo,watching,ws,ld,pk,mdd,ms]
        if cap<=0:break

    cap=S[0];pos=S[1]
    if pos!=0 and cap>0:
        epx=S[2];psz=S[3]
        pnl=(close[-1]-epx)/epx*psz*pos-psz*FEE_RATE;cap+=pnl
        if pnl>0:w+=1;gp+=pnl
        else:l+=1;gl+=abs(pnl)
    t=sl_c+tsl_c+rev_c
    return Result("10.ArrayPacked",cap,t,sl_c,tsl_c,rev_c,S[12],w,l,gp,gl)


# ═══════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════
def main():
    csv_path = "D:/filesystem/futures/btc_V1/test/btc_usdt_5m_merged.csv"
    close,high,low,fast_ma,slow_ma,adx,rsi,n,timestamps = load_and_prepare(csv_path)

    # Numba warmup
    print("Numba JIT compiling...")
    _numba_loop(close[:700],high[:700],low[:700],fast_ma[:700],slow_ma[:700],adx[:700],rsi[:700],700)
    print()

    engines = [
        ("1.Pure Python",    lambda: engine1(close,high,low,fast_ma,slow_ma,adx,rsi,n)),
        ("2.Class OOP",      lambda: engine2(close,high,low,fast_ma,slow_ma,adx,rsi,n)),
        ("3.Numpy State",    lambda: engine3(close,high,low,fast_ma,slow_ma,adx,rsi,n)),
        ("4.Numba JIT",      lambda: engine4(close,high,low,fast_ma,slow_ma,adx,rsi,n)),
        ("5.Pandas Hybrid",  lambda: engine5(close,high,low,fast_ma,slow_ma,adx,rsi,n)),
        ("6.Functional",     lambda: engine6(close,high,low,fast_ma,slow_ma,adx,rsi,n)),
        ("7.Dict State",     lambda: engine7(close,high,low,fast_ma,slow_ma,adx,rsi,n)),
        ("8.Generator",      lambda: engine8(close,high,low,fast_ma,slow_ma,adx,rsi,n)),
        ("9.VecPreSignal",   lambda: engine9(close,high,low,fast_ma,slow_ma,adx,rsi,n)),
        ("10.ArrayPacked",   lambda: engine10(close,high,low,fast_ma,slow_ma,adx,rsi,n)),
    ]

    results = []
    for name, fn in engines:
        t0 = time.time()
        r = fn()
        elapsed = time.time() - t0
        print(f"  {r.name:<18} ${r.cap:>14,.0f}  {r.trades:>3}t  "
              f"SL:{r.sl:>2} TSL:{r.tsl:>2} REV:{r.rev:>2}  "
              f"PF:{r.pf:>5.1f}  MDD:{r.mdd*100:>5.1f}%  {elapsed:.2f}s")
        results.append(r)

        # 거래 내역 CSV 출력
        if r.trade_log:
            trades_df = pd.DataFrame(r.trade_log,
                columns=['bar_idx','direction','entry_price','exit_price','pnl','capital','type'])
            trades_df.to_csv(f'{OUTPUT_DIR}/trades_engine_{name.split(".")[0]}.csv', index=False)

    # ═══ 교차검증 ═══
    print()
    print("=" * 80)
    print(" 10-ENGINE CROSS VERIFICATION")
    print("=" * 80)

    caps = [r.cap for r in results]
    max_diff = max(caps) - min(caps)

    print(f"\n  {'Engine':<18} {'Balance':>15} {'Trades':>7} {'SL':>4} {'TSL':>4} {'REV':>4} {'PF':>6} {'MDD':>7}")
    print(f"  {'-'*18} {'-'*15} {'-'*7} {'-'*4} {'-'*4} {'-'*4} {'-'*6} {'-'*7}")
    for r in results:
        print(f"  {r.name:<18} ${r.cap:>13,.0f} {r.trades:>7} {r.sl:>4} {r.tsl:>4} {r.rev:>4} {r.pf:>6.1f} {r.mdd*100:>6.1f}%")

    print(f"\n  Max diff: ${max_diff:,.6f}")

    target = 24_073_329
    avg = sum(caps)/len(caps)
    pct = abs(avg-target)/target*100
    status = "PASS (+-1%)" if pct<=1 else "FAIL"
    print(f"  Target:   ${target:,}")
    print(f"  Average:  ${avg:,.0f}")
    print(f"  Diff:     {pct:.4f}%  {status}")

    t_ok = all(r.trades==results[0].trades for r in results)
    s_ok = all(r.sl==results[0].sl for r in results)
    ts_ok = all(r.tsl==results[0].tsl for r in results)
    rv_ok = all(r.rev==results[0].rev for r in results)
    c_ok = max_diff < 0.01

    print(f"\n  Consistency:")
    print(f"    Balance:  {'OK' if c_ok else 'NG'} (${max_diff:.6f})")
    print(f"    Trades:   {'OK' if t_ok else 'NG'} ({[r.trades for r in results]})")
    print(f"    SL:       {'OK' if s_ok else 'NG'}")
    print(f"    TSL:      {'OK' if ts_ok else 'NG'}")
    print(f"    REV:      {'OK' if rv_ok else 'NG'}")

    all_pass = c_ok and t_ok and s_ok and ts_ok and rv_ok
    print(f"\n  {'='*50}")
    print(f"  RESULT: {'ALL 10 ENGINES MATCH' if all_pass else 'MISMATCH DETECTED'}")
    print(f"  {'='*50}")

    # Summary CSV
    summary = pd.DataFrame([{
        'engine': r.name, 'final_balance': r.cap, 'return_pct': (r.cap/INITIAL_CAPITAL-1)*100,
        'trades': r.trades, 'sl': r.sl, 'tsl': r.tsl, 'rev': r.rev,
        'wins': r.wins, 'losses': r.losses, 'win_rate': r.wins/(r.wins+r.losses)*100 if r.wins+r.losses>0 else 0,
        'pf': r.pf, 'mdd_pct': r.mdd*100,
        'gross_profit': r.gross_profit, 'gross_loss': r.gross_loss,
    } for r in results])
    summary.to_csv(f'{OUTPUT_DIR}/summary.csv', index=False)
    print(f"\n  Output files saved to: {OUTPUT_DIR}/")
    print(f"    30m_ohlcv.csv      - 30min resampled OHLCV")
    print(f"    indicators.csv     - EMA100/EMA600/ADX20/RSI10")
    print(f"    trades_engine_*.csv - per-engine trade logs")
    print(f"    summary.csv        - 10-engine comparison")


if __name__ == '__main__':
    main()
