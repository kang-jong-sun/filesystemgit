"""
BTC/USDT 선물 백테스트 6엔진 교차검증
기획서 v32.2: EMA(100)/EMA(600) Tight-SL Trend System

엔진 목록:
  1. Pure Python Loop
  2. Class OOP
  3. Numpy State Machine
  4. Numba JIT
  5. Pandas + Loop Hybrid
  6. Functional / Closure

목표 수치: $24,073,329 / 70거래 / SL30 TSL17 REV23 / PF 5.8 / MDD 43.5%
"""

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
DAILY_BARS      = 1440  # 30분봉 48개/일 → 기획서는 1440 사용


# ═══════════════════════════════════════════════════════════
# 데이터 로드 및 지표 계산 (공통)
# ═══════════════════════════════════════════════════════════
def load_and_prepare(csv_path):
    """5분봉 CSV → 30분봉 리샘플링 → 지표 계산"""
    print("데이터 로딩...")
    df5 = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df5.set_index('timestamp', inplace=True)
    df5.sort_index(inplace=True)

    print(f"  5분봉: {len(df5)}행")

    # 30분봉 리샘플링
    df = df5.resample('30min', closed='left', label='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    print(f"  30분봉: {len(df)}행 ({df.index[0]} ~ {df.index[-1]})")

    close = df['close'].values.astype(np.float64)
    high  = df['high'].values.astype(np.float64)
    low   = df['low'].values.astype(np.float64)
    n = len(close)

    # EMA (pandas ewm span, adjust=False)
    fast_ma = df['close'].ewm(span=FAST_PERIOD, adjust=False).mean().values.astype(np.float64)
    slow_ma = df['close'].ewm(span=SLOW_PERIOD, adjust=False).mean().values.astype(np.float64)

    # ADX(20) — ewm(alpha=1/20)
    adx = calc_adx(high, low, close, ADX_PERIOD)

    # RSI(10) — ewm(alpha=1/10)
    rsi = calc_rsi(close, RSI_PERIOD)

    # 검증 포인트
    # 2020-01-21 → 인덱스 찾기
    idx_check = None
    for j, ts in enumerate(df.index):
        if ts.strftime('%Y-%m-%d') == '2020-01-21':
            idx_check = j
            break
    if idx_check is not None:
        print(f"\n  검증: 2020-01-21 첫 봉(idx={idx_check})")
        print(f"    EMA(100) = {fast_ma[idx_check]:.2f}  (기대: 8703.63)")
        print(f"    EMA(600) = {slow_ma[idx_check]:.2f}  (기대: 8474.74)")

    # 크로스 카운트
    cross_count = 0
    for j in range(1, n):
        bull_now  = fast_ma[j] > slow_ma[j]
        bull_prev = fast_ma[j-1] > slow_ma[j-1]
        if bull_now != bull_prev:
            cross_count += 1
    print(f"  EMA 크로스: {cross_count}건 (기대: 258)")

    print()
    return close, high, low, fast_ma, slow_ma, adx, rsi, n


def calc_adx(high, low, close, period):
    """ADX 계산 — ewm(alpha=1/period)"""
    n = len(close)
    plus_dm  = np.zeros(n)
    minus_dm = np.zeros(n)
    tr       = np.zeros(n)

    for i in range(1, n):
        h_diff = high[i] - high[i-1]
        l_diff = low[i-1] - low[i]
        plus_dm[i]  = h_diff if (h_diff > l_diff and h_diff > 0) else 0.0
        minus_dm[i] = l_diff if (l_diff > h_diff and l_diff > 0) else 0.0
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))

    s_plus  = pd.Series(plus_dm)
    s_minus = pd.Series(minus_dm)
    s_tr    = pd.Series(tr)

    alpha = 1.0 / period
    atr    = s_tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    sm_p   = s_plus.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    sm_m   = s_minus.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    plus_di  = 100.0 * sm_p / atr.replace(0, 1e-10)
    minus_di = 100.0 * sm_m / atr.replace(0, 1e-10)
    dx_denom = (plus_di + minus_di).replace(0, 1e-10)
    dx = 100.0 * (plus_di - minus_di).abs() / dx_denom
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx.values.astype(np.float64)


def calc_rsi(close, period):
    """RSI 계산 — ewm(alpha=1/period)"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    alpha = 1.0 / period
    avg_gain = pd.Series(gain).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.values.astype(np.float64)


# ═══════════════════════════════════════════════════════════
# 결과 구조
# ═══════════════════════════════════════════════════════════
class Result:
    def __init__(self, name, cap, trades, sl, tsl, rev, mdd, wins, losses, gross_profit, gross_loss, trade_log=None):
        self.name = name
        self.cap = cap
        self.trades = trades
        self.sl = sl
        self.tsl = tsl
        self.rev = rev
        self.mdd = mdd
        self.wins = wins
        self.losses = losses
        self.gross_profit = gross_profit
        self.gross_loss = gross_loss
        self.pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        self.trade_log = trade_log or []


def print_result(r):
    print(f"  {'최종잔액':>12}: ${r.cap:,.0f}")
    print(f"  {'수익률':>12}: {(r.cap / INITIAL_CAPITAL - 1) * 100:,.1f}%")
    print(f"  {'거래수':>12}: {r.trades}건 (SL:{r.sl} TSL:{r.tsl} REV:{r.rev})")
    print(f"  {'승/패':>12}: {r.wins}W / {r.losses}L ({r.wins/(r.wins+r.losses)*100:.1f}%)" if r.wins+r.losses > 0 else "")
    print(f"  {'PF':>12}: {r.pf:.1f}")
    print(f"  {'MDD':>12}: {r.mdd*100:.1f}%")


# ═══════════════════════════════════════════════════════════
# 엔진 1: Pure Python Loop
# ═══════════════════════════════════════════════════════════
def engine1_pure_python(close, high, low, fast_ma, slow_ma, adx, rsi, n):
    cap = INITIAL_CAPITAL
    pos = 0; epx = 0.0; psz = 0.0; slp = 0.0
    ton = False; thi = 0.0; tlo = 999999.0
    watching = 0; ws = 0; le = 0; ld = 0
    pk = cap; mdd = 0.0; ms = cap
    sl_cnt = 0; tsl_cnt = 0; rev_cnt = 0
    wins = 0; losses = 0; gp = 0.0; gl = 0.0
    trades = []

    for i in range(WARMUP, n):
        px = close[i]; h_ = high[i]; l_ = low[i]

        if i > WARMUP and i % DAILY_BARS == 0:
            ms = cap

        # STEP A: 청산 체크
        if pos != 0:
            watching = 0

            # A1: SL (TSL 미활성시만)
            if not ton:
                if (pos == 1 and l_ <= slp) or (pos == -1 and h_ >= slp):
                    pnl = (slp - epx) / epx * psz * pos - psz * FEE_RATE
                    cap += pnl
                    sl_cnt += 1
                    if pnl > 0: wins += 1; gp += pnl
                    else: losses += 1; gl += abs(pnl)
                    trades.append(('SL', i, pos, epx, slp, pnl, cap))
                    ld = pos; le = i; pos = 0
                    pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    if cap <= 0: break
                    continue

            # A2: TA 활성화
            if pos == 1:
                br = (h_ - epx) / epx * 100
            else:
                br = (epx - l_) / epx * 100
            if br >= TA_PCT:
                ton = True

            # A3: TSL (TSL 활성시)
            if ton:
                if pos == 1:
                    if h_ > thi: thi = h_
                    ns = thi * (1 - TSL_PCT / 100)
                    if ns > slp: slp = ns
                    if px <= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * FEE_RATE
                        cap += pnl
                        tsl_cnt += 1
                        if pnl > 0: wins += 1; gp += pnl
                        else: losses += 1; gl += abs(pnl)
                        trades.append(('TSL', i, pos, epx, px, pnl, cap))
                        ld = pos; le = i; pos = 0
                        pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                        if dd > mdd: mdd = dd
                        if cap <= 0: break
                        continue
                else:
                    if l_ < tlo: tlo = l_
                    ns = tlo * (1 + TSL_PCT / 100)
                    if ns < slp: slp = ns
                    if px >= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * FEE_RATE
                        cap += pnl
                        tsl_cnt += 1
                        if pnl > 0: wins += 1; gp += pnl
                        else: losses += 1; gl += abs(pnl)
                        trades.append(('TSL', i, pos, epx, px, pnl, cap))
                        ld = pos; le = i; pos = 0
                        pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                        if dd > mdd: mdd = dd
                        if cap <= 0: break
                        continue

            # A4: REV
            if i > 0:
                bull_now  = fast_ma[i] > slow_ma[i]
                bull_prev = fast_ma[i-1] > slow_ma[i-1]
                cross_up   = bull_now and not bull_prev
                cross_down = not bull_now and bull_prev
                if (pos == 1 and cross_down) or (pos == -1 and cross_up):
                    pnl = (px - epx) / epx * psz * pos - psz * FEE_RATE
                    cap += pnl
                    rev_cnt += 1
                    if pnl > 0: wins += 1; gp += pnl
                    else: losses += 1; gl += abs(pnl)
                    trades.append(('REV', i, pos, epx, px, pnl, cap))
                    ld = pos; le = i; pos = 0

        # STEP B: 진입
        if i < 1:
            pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
            if dd > mdd: mdd = dd
            continue

        bull_now  = fast_ma[i] > slow_ma[i]
        bull_prev = fast_ma[i-1] > slow_ma[i-1]
        cross_up   = bull_now and not bull_prev
        cross_down = not bull_now and bull_prev

        if pos == 0:
            if cross_up:    watching = 1;  ws = i
            elif cross_down: watching = -1; ws = i

            if watching != 0 and i > ws:
                if i - ws > MONITOR_WINDOW:
                    watching = 0
                    pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    continue

                if watching == 1 and cross_down:
                    watching = -1; ws = i
                    pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    continue
                elif watching == -1 and cross_up:
                    watching = 1; ws = i
                    pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    continue

                if SKIP_SAME_DIR and watching == ld:
                    pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    continue

                if adx[i] < ADX_MIN:
                    pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    continue

                if ADX_RISE_BARS > 0 and i >= ADX_RISE_BARS:
                    if adx[i] <= adx[i - ADX_RISE_BARS]:
                        pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                        if dd > mdd: mdd = dd
                        continue

                if rsi[i] < RSI_MIN or rsi[i] > RSI_MAX:
                    pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    continue

                if EMA_GAP_MIN > 0:
                    gap = abs(fast_ma[i] - slow_ma[i]) / slow_ma[i] * 100
                    if gap < EMA_GAP_MIN:
                        pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                        if dd > mdd: mdd = dd
                        continue

                if ms > 0 and (cap - ms) / ms <= DAILY_LOSS_LIM:
                    watching = 0
                    pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    continue

                if cap <= 0:
                    pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    continue

                # 진입
                mg = cap * MARGIN_PCT
                psz = mg * LEVERAGE
                cap -= psz * FEE_RATE
                pos = watching
                epx = px
                ton = False; thi = px; tlo = px
                if pos == 1:
                    slp = epx * (1 - SL_PCT / 100)
                else:
                    slp = epx * (1 + SL_PCT / 100)
                pk = max(pk, cap)
                watching = 0
                trades.append(('ENTRY', i, pos, epx, 0, 0, cap))

        pk = max(pk, cap)
        dd = (pk - cap) / pk if pk > 0 else 0
        if dd > mdd: mdd = dd
        if cap <= 0: break

    # 미청산 포지션
    if pos != 0 and cap > 0:
        pnl = (close[-1] - epx) / epx * psz * pos - psz * FEE_RATE
        cap += pnl
        if pnl > 0: wins += 1; gp += pnl
        else: losses += 1; gl += abs(pnl)

    total = sl_cnt + tsl_cnt + rev_cnt
    return Result("Pure Python", cap, total, sl_cnt, tsl_cnt, rev_cnt, mdd, wins, losses, gp, gl, trades)


# ═══════════════════════════════════════════════════════════
# 엔진 2: Class OOP
# ═══════════════════════════════════════════════════════════
class Position:
    __slots__ = ['direction', 'entry_price', 'size', 'sl_price', 'tsl_active', 'track_high', 'track_low']
    def __init__(self, direction, entry_price, size):
        self.direction = direction
        self.entry_price = entry_price
        self.size = size
        self.tsl_active = False
        self.track_high = entry_price
        self.track_low = entry_price
        if direction == 1:
            self.sl_price = entry_price * (1 - SL_PCT / 100)
        else:
            self.sl_price = entry_price * (1 + SL_PCT / 100)

class TradingEngine:
    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.position = None
        self.watching = 0
        self.watch_start = 0
        self.last_exit_bar = 0
        self.last_exit_dir = 0
        self.peak = INITIAL_CAPITAL
        self.max_dd = 0.0
        self.daily_start = INITIAL_CAPITAL
        self.sl_cnt = 0; self.tsl_cnt = 0; self.rev_cnt = 0
        self.wins = 0; self.losses = 0
        self.gp = 0.0; self.gl = 0.0

    def _close_position(self, exit_price, exit_type, bar_idx):
        p = self.position
        pnl = (exit_price - p.entry_price) / p.entry_price * p.size * p.direction - p.size * FEE_RATE
        self.capital += pnl
        if exit_type == 'SL': self.sl_cnt += 1
        elif exit_type == 'TSL': self.tsl_cnt += 1
        else: self.rev_cnt += 1
        if pnl > 0: self.wins += 1; self.gp += pnl
        else: self.losses += 1; self.gl += abs(pnl)
        self.last_exit_dir = p.direction
        self.last_exit_bar = bar_idx
        self.position = None

    def _update_mdd(self):
        self.peak = max(self.peak, self.capital)
        if self.peak > 0:
            dd = (self.peak - self.capital) / self.peak
            if dd > self.max_dd: self.max_dd = dd

    def _enter(self, direction, price):
        mg = self.capital * MARGIN_PCT
        psz = mg * LEVERAGE
        self.capital -= psz * FEE_RATE
        self.position = Position(direction, price, psz)
        self.peak = max(self.peak, self.capital)
        self.watching = 0

    def run(self, close, high, low, fast_ma, slow_ma, adx, rsi, n):
        for i in range(WARMUP, n):
            px = close[i]; h_ = high[i]; l_ = low[i]

            if i > WARMUP and i % DAILY_BARS == 0:
                self.daily_start = self.capital

            if self.position is not None:
                p = self.position
                self.watching = 0

                # SL
                if not p.tsl_active:
                    if (p.direction == 1 and l_ <= p.sl_price) or \
                       (p.direction == -1 and h_ >= p.sl_price):
                        self._close_position(p.sl_price, 'SL', i)
                        self._update_mdd()
                        if self.capital <= 0: break
                        continue

                # TA
                if p.direction == 1:
                    br = (h_ - p.entry_price) / p.entry_price * 100
                else:
                    br = (p.entry_price - l_) / p.entry_price * 100
                if br >= TA_PCT:
                    p.tsl_active = True

                # TSL
                if p.tsl_active:
                    if p.direction == 1:
                        if h_ > p.track_high: p.track_high = h_
                        ns = p.track_high * (1 - TSL_PCT / 100)
                        if ns > p.sl_price: p.sl_price = ns
                        if px <= p.sl_price:
                            self._close_position(px, 'TSL', i)
                            self._update_mdd()
                            if self.capital <= 0: break
                            continue
                    else:
                        if l_ < p.track_low: p.track_low = l_
                        ns = p.track_low * (1 + TSL_PCT / 100)
                        if ns < p.sl_price: p.sl_price = ns
                        if px >= p.sl_price:
                            self._close_position(px, 'TSL', i)
                            self._update_mdd()
                            if self.capital <= 0: break
                            continue

                # REV
                if i > 0:
                    bn = fast_ma[i] > slow_ma[i]
                    bp = fast_ma[i-1] > slow_ma[i-1]
                    if (p.direction == 1 and not bn and bp) or \
                       (p.direction == -1 and bn and not bp):
                        self._close_position(px, 'REV', i)

            # 진입
            if i < 1:
                self._update_mdd()
                continue

            bn = fast_ma[i] > slow_ma[i]
            bp = fast_ma[i-1] > slow_ma[i-1]
            cu = bn and not bp
            cd = not bn and bp

            if self.position is None:
                if cu: self.watching = 1; self.watch_start = i
                elif cd: self.watching = -1; self.watch_start = i

                if self.watching != 0 and i > self.watch_start:
                    if i - self.watch_start > MONITOR_WINDOW:
                        self.watching = 0; self._update_mdd(); continue
                    if self.watching == 1 and cd:
                        self.watching = -1; self.watch_start = i; self._update_mdd(); continue
                    elif self.watching == -1 and cu:
                        self.watching = 1; self.watch_start = i; self._update_mdd(); continue
                    if SKIP_SAME_DIR and self.watching == self.last_exit_dir:
                        self._update_mdd(); continue
                    if adx[i] < ADX_MIN: self._update_mdd(); continue
                    if ADX_RISE_BARS > 0 and i >= ADX_RISE_BARS:
                        if adx[i] <= adx[i - ADX_RISE_BARS]: self._update_mdd(); continue
                    if rsi[i] < RSI_MIN or rsi[i] > RSI_MAX: self._update_mdd(); continue
                    if EMA_GAP_MIN > 0:
                        gap = abs(fast_ma[i] - slow_ma[i]) / slow_ma[i] * 100
                        if gap < EMA_GAP_MIN: self._update_mdd(); continue
                    if self.daily_start > 0 and (self.capital - self.daily_start) / self.daily_start <= DAILY_LOSS_LIM:
                        self.watching = 0; self._update_mdd(); continue
                    if self.capital <= 0: self._update_mdd(); continue

                    self._enter(self.watching, px)

            self._update_mdd()
            if self.capital <= 0: break

        # 미청산
        if self.position is not None and self.capital > 0:
            p = self.position
            pnl = (close[-1] - p.entry_price) / p.entry_price * p.size * p.direction - p.size * FEE_RATE
            self.capital += pnl
            if pnl > 0: self.wins += 1; self.gp += pnl
            else: self.losses += 1; self.gl += abs(pnl)

        total = self.sl_cnt + self.tsl_cnt + self.rev_cnt
        return Result("Class OOP", self.capital, total, self.sl_cnt, self.tsl_cnt, self.rev_cnt,
                       self.max_dd, self.wins, self.losses, self.gp, self.gl)


def engine2_class_oop(close, high, low, fast_ma, slow_ma, adx, rsi, n):
    eng = TradingEngine()
    return eng.run(close, high, low, fast_ma, slow_ma, adx, rsi, n)


# ═══════════════════════════════════════════════════════════
# 엔진 3: Numpy State Machine
# ═══════════════════════════════════════════════════════════
def engine3_numpy_state(close, high, low, fast_ma, slow_ma, adx, rsi, n):
    # 상태를 numpy 배열로 관리
    state = np.zeros(16, dtype=np.float64)
    # 0:cap, 1:pos, 2:epx, 3:psz, 4:slp, 5:ton, 6:thi, 7:tlo
    # 8:watching, 9:ws, 10:le, 11:ld, 12:pk, 13:mdd, 14:ms
    state[0] = INITIAL_CAPITAL  # cap
    state[7] = 999999.0         # tlo
    state[12] = INITIAL_CAPITAL # pk
    state[14] = INITIAL_CAPITAL # ms

    counters = np.zeros(6, dtype=np.int64)  # sl, tsl, rev, wins, losses, (unused)
    profits = np.zeros(2, dtype=np.float64) # gp, gl

    for i in range(WARMUP, n):
        cap = state[0]; pos = int(state[1]); epx = state[2]; psz = state[3]
        slp = state[4]; ton = state[5] > 0.5; thi = state[6]; tlo = state[7]
        watching = int(state[8]); ws = int(state[9])
        ld = int(state[11]); pk = state[12]; mdd = state[13]; ms = state[14]

        px = close[i]; h_ = high[i]; l_ = low[i]

        if i > WARMUP and i % DAILY_BARS == 0:
            ms = cap

        if pos != 0:
            watching = 0

            if not ton:
                if (pos == 1 and l_ <= slp) or (pos == -1 and h_ >= slp):
                    pnl = (slp - epx) / epx * psz * pos - psz * FEE_RATE
                    cap += pnl
                    counters[0] += 1
                    if pnl > 0: counters[3] += 1; profits[0] += pnl
                    else: counters[4] += 1; profits[1] += abs(pnl)
                    ld = pos; pos = 0
                    pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    state[:] = [cap, pos, epx, psz, slp, 0, thi, tlo, watching, ws, i, ld, pk, mdd, ms, 0]
                    if cap <= 0: break
                    continue

            if pos == 1: br = (h_ - epx) / epx * 100
            else: br = (epx - l_) / epx * 100
            if br >= TA_PCT: ton = True

            if ton:
                if pos == 1:
                    if h_ > thi: thi = h_
                    ns = thi * (1 - TSL_PCT / 100)
                    if ns > slp: slp = ns
                    if px <= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * FEE_RATE
                        cap += pnl; counters[1] += 1
                        if pnl > 0: counters[3] += 1; profits[0] += pnl
                        else: counters[4] += 1; profits[1] += abs(pnl)
                        ld = pos; pos = 0
                        pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                        if dd > mdd: mdd = dd
                        state[:] = [cap, pos, epx, psz, slp, 0, thi, tlo, watching, ws, i, ld, pk, mdd, ms, 0]
                        if cap <= 0: break
                        continue
                else:
                    if l_ < tlo: tlo = l_
                    ns = tlo * (1 + TSL_PCT / 100)
                    if ns < slp: slp = ns
                    if px >= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * FEE_RATE
                        cap += pnl; counters[1] += 1
                        if pnl > 0: counters[3] += 1; profits[0] += pnl
                        else: counters[4] += 1; profits[1] += abs(pnl)
                        ld = pos; pos = 0
                        pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                        if dd > mdd: mdd = dd
                        state[:] = [cap, pos, epx, psz, slp, 0, thi, tlo, watching, ws, i, ld, pk, mdd, ms, 0]
                        if cap <= 0: break
                        continue

            if i > 0:
                bn = fast_ma[i] > slow_ma[i]
                bp = fast_ma[i-1] > slow_ma[i-1]
                if (pos == 1 and not bn and bp) or (pos == -1 and bn and not bp):
                    pnl = (px - epx) / epx * psz * pos - psz * FEE_RATE
                    cap += pnl; counters[2] += 1
                    if pnl > 0: counters[3] += 1; profits[0] += pnl
                    else: counters[4] += 1; profits[1] += abs(pnl)
                    ld = pos; pos = 0

        if i < 1:
            pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
            if dd > mdd: mdd = dd
            state[:] = [cap, pos, epx, psz, slp, 1 if ton else 0, thi, tlo, watching, ws, state[10], ld, pk, mdd, ms, 0]
            continue

        bn = fast_ma[i] > slow_ma[i]
        bp = fast_ma[i-1] > slow_ma[i-1]
        cu = bn and not bp
        cd = not bn and bp

        if pos == 0:
            if cu: watching = 1; ws = i
            elif cd: watching = -1; ws = i

            if watching != 0 and i > ws:
                skip = False
                if i - ws > MONITOR_WINDOW: watching = 0; skip = True
                if not skip and watching == 1 and cd: watching = -1; ws = i; skip = True
                if not skip and watching == -1 and cu: watching = 1; ws = i; skip = True
                if not skip and SKIP_SAME_DIR and watching == ld: skip = True
                if not skip and adx[i] < ADX_MIN: skip = True
                if not skip and ADX_RISE_BARS > 0 and i >= ADX_RISE_BARS:
                    if adx[i] <= adx[i - ADX_RISE_BARS]: skip = True
                if not skip and (rsi[i] < RSI_MIN or rsi[i] > RSI_MAX): skip = True
                if not skip and EMA_GAP_MIN > 0:
                    gap = abs(fast_ma[i] - slow_ma[i]) / slow_ma[i] * 100
                    if gap < EMA_GAP_MIN: skip = True
                if not skip and ms > 0 and (cap - ms) / ms <= DAILY_LOSS_LIM:
                    watching = 0; skip = True
                if not skip and cap <= 0: skip = True

                if not skip:
                    mg = cap * MARGIN_PCT
                    psz = mg * LEVERAGE
                    cap -= psz * FEE_RATE
                    pos = watching; epx = px
                    ton = False; thi = px; tlo = px
                    if pos == 1: slp = epx * (1 - SL_PCT / 100)
                    else: slp = epx * (1 + SL_PCT / 100)
                    pk = max(pk, cap); watching = 0

        pk = max(pk, cap)
        dd = (pk - cap) / pk if pk > 0 else 0
        if dd > mdd: mdd = dd
        state[:] = [cap, pos, epx, psz, slp, 1 if ton else 0, thi, tlo, watching, ws, state[10], ld, pk, mdd, ms, 0]
        if cap <= 0: break

    cap = state[0]; pos = int(state[1])
    if pos != 0 and cap > 0:
        epx = state[2]; psz = state[3]
        pnl = (close[-1] - epx) / epx * psz * pos - psz * FEE_RATE
        cap += pnl
        if pnl > 0: counters[3] += 1; profits[0] += pnl
        else: counters[4] += 1; profits[1] += abs(pnl)

    total = int(counters[0] + counters[1] + counters[2])
    return Result("Numpy State", cap, total, int(counters[0]), int(counters[1]), int(counters[2]),
                   state[13], int(counters[3]), int(counters[4]), profits[0], profits[1])


# ═══════════════════════════════════════════════════════════
# 엔진 4: Numba JIT
# ═══════════════════════════════════════════════════════════
@njit(cache=True)
def _numba_loop(close, high, low, fast_ma, slow_ma, adx, rsi, n):
    cap = 5000.0
    pos = 0; epx = 0.0; psz = 0.0; slp = 0.0
    ton = False; thi = 0.0; tlo = 999999.0
    watching = 0; ws = 0; ld = 0
    pk = cap; mdd = 0.0; ms = cap
    sl_cnt = 0; tsl_cnt = 0; rev_cnt = 0
    wins = 0; losses = 0; gp = 0.0; gl = 0.0

    for i in range(600, n):
        px = close[i]; h_ = high[i]; l_ = low[i]

        if i > 600 and i % 1440 == 0:
            ms = cap

        if pos != 0:
            watching = 0

            if not ton:
                if (pos == 1 and l_ <= slp) or (pos == -1 and h_ >= slp):
                    pnl = (slp - epx) / epx * psz * pos - psz * 0.0004
                    cap += pnl; sl_cnt += 1
                    if pnl > 0: wins += 1; gp += pnl
                    else: losses += 1; gl += abs(pnl)
                    ld = pos; pos = 0
                    if cap > pk: pk = cap
                    dd = (pk - cap) / pk if pk > 0 else 0.0
                    if dd > mdd: mdd = dd
                    if cap <= 0: break
                    continue

            if pos == 1: br = (h_ - epx) / epx * 100.0
            else: br = (epx - l_) / epx * 100.0
            if br >= 12.0: ton = True

            if ton:
                if pos == 1:
                    if h_ > thi: thi = h_
                    ns = thi * 0.91
                    if ns > slp: slp = ns
                    if px <= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * 0.0004
                        cap += pnl; tsl_cnt += 1
                        if pnl > 0: wins += 1; gp += pnl
                        else: losses += 1; gl += abs(pnl)
                        ld = pos; pos = 0
                        if cap > pk: pk = cap
                        dd = (pk - cap) / pk if pk > 0 else 0.0
                        if dd > mdd: mdd = dd
                        if cap <= 0: break
                        continue
                else:
                    if l_ < tlo: tlo = l_
                    ns = tlo * 1.09
                    if ns < slp: slp = ns
                    if px >= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * 0.0004
                        cap += pnl; tsl_cnt += 1
                        if pnl > 0: wins += 1; gp += pnl
                        else: losses += 1; gl += abs(pnl)
                        ld = pos; pos = 0
                        if cap > pk: pk = cap
                        dd = (pk - cap) / pk if pk > 0 else 0.0
                        if dd > mdd: mdd = dd
                        if cap <= 0: break
                        continue

            if i > 0:
                bn = fast_ma[i] > slow_ma[i]
                bp = fast_ma[i-1] > slow_ma[i-1]
                if (pos == 1 and not bn and bp) or (pos == -1 and bn and not bp):
                    pnl = (px - epx) / epx * psz * pos - psz * 0.0004
                    cap += pnl; rev_cnt += 1
                    if pnl > 0: wins += 1; gp += pnl
                    else: losses += 1; gl += abs(pnl)
                    ld = pos; pos = 0

        if i < 1:
            if cap > pk: pk = cap
            dd = (pk - cap) / pk if pk > 0 else 0.0
            if dd > mdd: mdd = dd
            continue

        bn = fast_ma[i] > slow_ma[i]
        bp = fast_ma[i-1] > slow_ma[i-1]
        cu = bn and not bp
        cd = not bn and bp

        if pos == 0:
            if cu: watching = 1; ws = i
            elif cd: watching = -1; ws = i

            if watching != 0 and i > ws:
                if i - ws > 24:
                    watching = 0
                    if cap > pk: pk = cap
                    dd = (pk - cap) / pk if pk > 0 else 0.0
                    if dd > mdd: mdd = dd
                    continue
                if watching == 1 and cd:
                    watching = -1; ws = i
                    if cap > pk: pk = cap
                    dd = (pk - cap) / pk if pk > 0 else 0.0
                    if dd > mdd: mdd = dd
                    continue
                elif watching == -1 and cu:
                    watching = 1; ws = i
                    if cap > pk: pk = cap
                    dd = (pk - cap) / pk if pk > 0 else 0.0
                    if dd > mdd: mdd = dd
                    continue
                if watching == ld:
                    if cap > pk: pk = cap
                    dd = (pk - cap) / pk if pk > 0 else 0.0
                    if dd > mdd: mdd = dd
                    continue
                if adx[i] < 30.0:
                    if cap > pk: pk = cap
                    dd = (pk - cap) / pk if pk > 0 else 0.0
                    if dd > mdd: mdd = dd
                    continue
                if i >= 6:
                    if adx[i] <= adx[i-6]:
                        if cap > pk: pk = cap
                        dd = (pk - cap) / pk if pk > 0 else 0.0
                        if dd > mdd: mdd = dd
                        continue
                if rsi[i] < 40.0 or rsi[i] > 80.0:
                    if cap > pk: pk = cap
                    dd = (pk - cap) / pk if pk > 0 else 0.0
                    if dd > mdd: mdd = dd
                    continue
                gap = abs(fast_ma[i] - slow_ma[i]) / slow_ma[i] * 100.0
                if gap < 0.2:
                    if cap > pk: pk = cap
                    dd = (pk - cap) / pk if pk > 0 else 0.0
                    if dd > mdd: mdd = dd
                    continue
                if ms > 0 and (cap - ms) / ms <= -0.20:
                    watching = 0
                    if cap > pk: pk = cap
                    dd = (pk - cap) / pk if pk > 0 else 0.0
                    if dd > mdd: mdd = dd
                    continue
                if cap <= 0:
                    if cap > pk: pk = cap
                    dd = (pk - cap) / pk if pk > 0 else 0.0
                    if dd > mdd: mdd = dd
                    continue

                mg = cap * 0.35
                psz = mg * 10.0
                cap -= psz * 0.0004
                pos = watching; epx = px
                ton = False; thi = px; tlo = px
                if pos == 1: slp = epx * 0.97
                else: slp = epx * 1.03
                if cap > pk: pk = cap
                watching = 0

        if cap > pk: pk = cap
        dd = (pk - cap) / pk if pk > 0 else 0.0
        if dd > mdd: mdd = dd
        if cap <= 0: break

    if pos != 0 and cap > 0:
        pnl = (close[n-1] - epx) / epx * psz * pos - psz * 0.0004
        cap += pnl
        if pnl > 0: wins += 1; gp += pnl
        else: losses += 1; gl += abs(pnl)

    return cap, sl_cnt, tsl_cnt, rev_cnt, mdd, wins, losses, gp, gl


def engine4_numba(close, high, low, fast_ma, slow_ma, adx, rsi, n):
    cap, sl, tsl, rev, mdd, w, l, gp, gl = _numba_loop(close, high, low, fast_ma, slow_ma, adx, rsi, n)
    return Result("Numba JIT", cap, sl+tsl+rev, sl, tsl, rev, mdd, w, l, gp, gl)


# ═══════════════════════════════════════════════════════════
# 엔진 5: Pandas + Loop Hybrid
# ═══════════════════════════════════════════════════════════
def engine5_pandas_hybrid(close, high, low, fast_ma, slow_ma, adx, rsi, n):
    """Pandas로 신호 전처리, Python 루프로 시뮬레이션"""
    # 신호 전처리
    bull = fast_ma > slow_ma
    cross_up_arr   = np.zeros(n, dtype=bool)
    cross_down_arr = np.zeros(n, dtype=bool)
    for j in range(1, n):
        cross_up_arr[j]   = bull[j] and not bull[j-1]
        cross_down_arr[j] = not bull[j] and bull[j-1]

    adx_ok = adx >= ADX_MIN
    adx_rise = np.zeros(n, dtype=bool)
    for j in range(ADX_RISE_BARS, n):
        adx_rise[j] = adx[j] > adx[j - ADX_RISE_BARS]
    rsi_ok = (rsi >= RSI_MIN) & (rsi <= RSI_MAX)
    gap_arr = np.abs(fast_ma - slow_ma) / np.where(slow_ma > 0, slow_ma, 1e-10) * 100
    gap_ok = gap_arr >= EMA_GAP_MIN

    # 시뮬레이션
    cap = INITIAL_CAPITAL
    pos = 0; epx = 0.0; psz = 0.0; slp = 0.0
    ton = False; thi = 0.0; tlo = 999999.0
    watching = 0; ws = 0; ld = 0
    pk = cap; mdd = 0.0; ms = cap
    sl_cnt = 0; tsl_cnt = 0; rev_cnt = 0
    wins = 0; losses = 0; gp = 0.0; gl = 0.0

    for i in range(WARMUP, n):
        px = close[i]; h_ = high[i]; l_ = low[i]
        cu = cross_up_arr[i]; cd = cross_down_arr[i]

        if i > WARMUP and i % DAILY_BARS == 0:
            ms = cap

        if pos != 0:
            watching = 0

            if not ton:
                if (pos == 1 and l_ <= slp) or (pos == -1 and h_ >= slp):
                    pnl = (slp - epx) / epx * psz * pos - psz * FEE_RATE
                    cap += pnl; sl_cnt += 1
                    if pnl > 0: wins += 1; gp += pnl
                    else: losses += 1; gl += abs(pnl)
                    ld = pos; pos = 0
                    pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                    if dd > mdd: mdd = dd
                    if cap <= 0: break
                    continue

            if pos == 1: br = (h_ - epx) / epx * 100
            else: br = (epx - l_) / epx * 100
            if br >= TA_PCT: ton = True

            if ton:
                if pos == 1:
                    if h_ > thi: thi = h_
                    ns = thi * (1 - TSL_PCT / 100)
                    if ns > slp: slp = ns
                    if px <= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * FEE_RATE
                        cap += pnl; tsl_cnt += 1
                        if pnl > 0: wins += 1; gp += pnl
                        else: losses += 1; gl += abs(pnl)
                        ld = pos; pos = 0
                        pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                        if dd > mdd: mdd = dd
                        if cap <= 0: break
                        continue
                else:
                    if l_ < tlo: tlo = l_
                    ns = tlo * (1 + TSL_PCT / 100)
                    if ns < slp: slp = ns
                    if px >= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * FEE_RATE
                        cap += pnl; tsl_cnt += 1
                        if pnl > 0: wins += 1; gp += pnl
                        else: losses += 1; gl += abs(pnl)
                        ld = pos; pos = 0
                        pk = max(pk, cap); dd = (pk - cap) / pk if pk > 0 else 0
                        if dd > mdd: mdd = dd
                        if cap <= 0: break
                        continue

            if (pos == 1 and cd) or (pos == -1 and cu):
                pnl = (px - epx) / epx * psz * pos - psz * FEE_RATE
                cap += pnl; rev_cnt += 1
                if pnl > 0: wins += 1; gp += pnl
                else: losses += 1; gl += abs(pnl)
                ld = pos; pos = 0

        if pos == 0:
            if cu: watching = 1; ws = i
            elif cd: watching = -1; ws = i

            if watching != 0 and i > ws:
                if i - ws > MONITOR_WINDOW: watching = 0
                elif watching == 1 and cd: watching = -1; ws = i
                elif watching == -1 and cu: watching = 1; ws = i
                elif SKIP_SAME_DIR and watching == ld: pass
                elif not adx_ok[i]: pass
                elif not adx_rise[i]: pass
                elif not rsi_ok[i]: pass
                elif not gap_ok[i]: pass
                elif ms > 0 and (cap - ms) / ms <= DAILY_LOSS_LIM: watching = 0
                elif cap <= 0: pass
                else:
                    mg = cap * MARGIN_PCT
                    psz = mg * LEVERAGE
                    cap -= psz * FEE_RATE
                    pos = watching; epx = px
                    ton = False; thi = px; tlo = px
                    if pos == 1: slp = epx * (1 - SL_PCT / 100)
                    else: slp = epx * (1 + SL_PCT / 100)
                    pk = max(pk, cap); watching = 0

        pk = max(pk, cap)
        dd = (pk - cap) / pk if pk > 0 else 0
        if dd > mdd: mdd = dd
        if cap <= 0: break

    if pos != 0 and cap > 0:
        pnl = (close[-1] - epx) / epx * psz * pos - psz * FEE_RATE
        cap += pnl
        if pnl > 0: wins += 1; gp += pnl
        else: losses += 1; gl += abs(pnl)

    total = sl_cnt + tsl_cnt + rev_cnt
    return Result("Pandas Hybrid", cap, total, sl_cnt, tsl_cnt, rev_cnt, mdd, wins, losses, gp, gl)


# ═══════════════════════════════════════════════════════════
# 엔진 6: Functional / Closure
# ═══════════════════════════════════════════════════════════
def engine6_functional(close, high, low, fast_ma, slow_ma, adx, rsi, n):
    """클로저와 제너레이터 패턴"""

    def make_state():
        return {
            'cap': INITIAL_CAPITAL, 'pos': 0, 'epx': 0.0, 'psz': 0.0,
            'slp': 0.0, 'ton': False, 'thi': 0.0, 'tlo': 999999.0,
            'watching': 0, 'ws': 0, 'le': 0, 'ld': 0,
            'pk': INITIAL_CAPITAL, 'mdd': 0.0, 'ms': INITIAL_CAPITAL,
            'sl': 0, 'tsl': 0, 'rev': 0,
            'wins': 0, 'losses': 0, 'gp': 0.0, 'gl': 0.0
        }

    def record_pnl(s, pnl, exit_type):
        if exit_type == 'SL': s['sl'] += 1
        elif exit_type == 'TSL': s['tsl'] += 1
        else: s['rev'] += 1
        if pnl > 0: s['wins'] += 1; s['gp'] += pnl
        else: s['losses'] += 1; s['gl'] += abs(pnl)

    def update_mdd(s):
        s['pk'] = max(s['pk'], s['cap'])
        if s['pk'] > 0:
            dd = (s['pk'] - s['cap']) / s['pk']
            if dd > s['mdd']: s['mdd'] = dd

    def check_sl(s, h_, l_):
        if s['ton']: return False
        if (s['pos'] == 1 and l_ <= s['slp']) or (s['pos'] == -1 and h_ >= s['slp']):
            pnl = (s['slp'] - s['epx']) / s['epx'] * s['psz'] * s['pos'] - s['psz'] * FEE_RATE
            s['cap'] += pnl
            record_pnl(s, pnl, 'SL')
            s['ld'] = s['pos']; s['pos'] = 0
            return True
        return False

    def check_ta(s, h_, l_):
        if s['pos'] == 1: br = (h_ - s['epx']) / s['epx'] * 100
        else: br = (s['epx'] - l_) / s['epx'] * 100
        if br >= TA_PCT: s['ton'] = True

    def check_tsl(s, px, h_, l_):
        if not s['ton']: return False
        if s['pos'] == 1:
            if h_ > s['thi']: s['thi'] = h_
            ns = s['thi'] * (1 - TSL_PCT / 100)
            if ns > s['slp']: s['slp'] = ns
            if px <= s['slp']:
                pnl = (px - s['epx']) / s['epx'] * s['psz'] * s['pos'] - s['psz'] * FEE_RATE
                s['cap'] += pnl; record_pnl(s, pnl, 'TSL')
                s['ld'] = s['pos']; s['pos'] = 0; return True
        else:
            if l_ < s['tlo']: s['tlo'] = l_
            ns = s['tlo'] * (1 + TSL_PCT / 100)
            if ns < s['slp']: s['slp'] = ns
            if px >= s['slp']:
                pnl = (px - s['epx']) / s['epx'] * s['psz'] * s['pos'] - s['psz'] * FEE_RATE
                s['cap'] += pnl; record_pnl(s, pnl, 'TSL')
                s['ld'] = s['pos']; s['pos'] = 0; return True
        return False

    def check_rev(s, px, i):
        if i <= 0: return False
        bn = fast_ma[i] > slow_ma[i]
        bp = fast_ma[i-1] > slow_ma[i-1]
        if (s['pos'] == 1 and not bn and bp) or (s['pos'] == -1 and bn and not bp):
            pnl = (px - s['epx']) / s['epx'] * s['psz'] * s['pos'] - s['psz'] * FEE_RATE
            s['cap'] += pnl; record_pnl(s, pnl, 'REV')
            s['ld'] = s['pos']; s['pos'] = 0
            return True
        return False

    def try_entry(s, i, px, cu, cd):
        if cu: s['watching'] = 1; s['ws'] = i
        elif cd: s['watching'] = -1; s['ws'] = i

        if s['watching'] == 0 or i <= s['ws']: return
        if i - s['ws'] > MONITOR_WINDOW: s['watching'] = 0; return
        if s['watching'] == 1 and cd: s['watching'] = -1; s['ws'] = i; return
        if s['watching'] == -1 and cu: s['watching'] = 1; s['ws'] = i; return
        if SKIP_SAME_DIR and s['watching'] == s['ld']: return
        if adx[i] < ADX_MIN: return
        if ADX_RISE_BARS > 0 and i >= ADX_RISE_BARS:
            if adx[i] <= adx[i - ADX_RISE_BARS]: return
        if rsi[i] < RSI_MIN or rsi[i] > RSI_MAX: return
        if EMA_GAP_MIN > 0:
            gap = abs(fast_ma[i] - slow_ma[i]) / slow_ma[i] * 100
            if gap < EMA_GAP_MIN: return
        if s['ms'] > 0 and (s['cap'] - s['ms']) / s['ms'] <= DAILY_LOSS_LIM:
            s['watching'] = 0; return
        if s['cap'] <= 0: return

        mg = s['cap'] * MARGIN_PCT
        s['psz'] = mg * LEVERAGE
        s['cap'] -= s['psz'] * FEE_RATE
        s['pos'] = s['watching']; s['epx'] = px
        s['ton'] = False; s['thi'] = px; s['tlo'] = px
        if s['pos'] == 1: s['slp'] = px * (1 - SL_PCT / 100)
        else: s['slp'] = px * (1 + SL_PCT / 100)
        s['pk'] = max(s['pk'], s['cap'])
        s['watching'] = 0

    s = make_state()
    for i in range(WARMUP, n):
        px = close[i]; h_ = high[i]; l_ = low[i]

        if i > WARMUP and i % DAILY_BARS == 0:
            s['ms'] = s['cap']

        if s['pos'] != 0:
            s['watching'] = 0
            if check_sl(s, h_, l_):
                update_mdd(s)
                if s['cap'] <= 0: break
                continue
            check_ta(s, h_, l_)
            if check_tsl(s, px, h_, l_):
                update_mdd(s)
                if s['cap'] <= 0: break
                continue
            check_rev(s, px, i)

        if i >= 1:
            bn = fast_ma[i] > slow_ma[i]
            bp = fast_ma[i-1] > slow_ma[i-1]
            cu = bn and not bp
            cd = not bn and bp
            if s['pos'] == 0:
                try_entry(s, i, px, cu, cd)

        update_mdd(s)
        if s['cap'] <= 0: break

    if s['pos'] != 0 and s['cap'] > 0:
        pnl = (close[-1] - s['epx']) / s['epx'] * s['psz'] * s['pos'] - s['psz'] * FEE_RATE
        s['cap'] += pnl
        if pnl > 0: s['wins'] += 1; s['gp'] += pnl
        else: s['losses'] += 1; s['gl'] += abs(pnl)

    total = s['sl'] + s['tsl'] + s['rev']
    return Result("Functional", s['cap'], total, s['sl'], s['tsl'], s['rev'],
                   s['mdd'], s['wins'], s['losses'], s['gp'], s['gl'])


# ═══════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════
def main():
    csv_path = "D:/filesystem/futures/btc_V1/test/btc_usdt_5m_merged.csv"
    close, high, low, fast_ma, slow_ma, adx, rsi, n = load_and_prepare(csv_path)

    engines = [
        ("Engine 1: Pure Python",    lambda: engine1_pure_python(close, high, low, fast_ma, slow_ma, adx, rsi, n)),
        ("Engine 2: Class OOP",      lambda: engine2_class_oop(close, high, low, fast_ma, slow_ma, adx, rsi, n)),
        ("Engine 3: Numpy State",    lambda: engine3_numpy_state(close, high, low, fast_ma, slow_ma, adx, rsi, n)),
        ("Engine 4: Numba JIT",      lambda: engine4_numba(close, high, low, fast_ma, slow_ma, adx, rsi, n)),
        ("Engine 5: Pandas Hybrid",  lambda: engine5_pandas_hybrid(close, high, low, fast_ma, slow_ma, adx, rsi, n)),
        ("Engine 6: Functional",     lambda: engine6_functional(close, high, low, fast_ma, slow_ma, adx, rsi, n)),
    ]

    # Numba 워밍업
    print("Numba JIT 컴파일 중...")
    _numba_loop(close[:700], high[:700], low[:700], fast_ma[:700], slow_ma[:700], adx[:700], rsi[:700], 700)
    print()

    results = []
    for name, fn in engines:
        print(f"{'='*60}")
        print(f" {name}")
        print(f"{'='*60}")
        t0 = time.time()
        r = fn()
        elapsed = time.time() - t0
        print_result(r)
        print(f"  {'실행시간':>12}: {elapsed:.2f}초")
        results.append(r)
        print()

    # ═══ 교차검증 ═══
    print("=" * 60)
    print(" 교차검증 결과")
    print("=" * 60)

    caps = [r.cap for r in results]
    max_diff = max(caps) - min(caps)
    avg_cap = sum(caps) / len(caps)

    print(f"\n  {'엔진':<20} {'최종잔액':>15} {'거래':>6} {'SL':>4} {'TSL':>4} {'REV':>4} {'PF':>6} {'MDD':>7}")
    print(f"  {'-'*18} {'-'*15} {'-'*6} {'-'*4} {'-'*4} {'-'*4} {'-'*6} {'-'*7}")
    for r in results:
        print(f"  {r.name:<20} ${r.cap:>13,.0f} {r.trades:>6} {r.sl:>4} {r.tsl:>4} {r.rev:>4} {r.pf:>6.1f} {r.mdd*100:>6.1f}%")

    print(f"\n  최대 차이: ${max_diff:,.6f}")
    print(f"  평균 잔액: ${avg_cap:,.0f}")

    # 기획서 검증
    target = 24_073_329
    pct_diff = abs(avg_cap - target) / target * 100
    print(f"\n  [Target] ${target:,}")
    status = "PASS (+-1%)" if pct_diff <= 1 else "FAIL"
    print(f"  [Diff]   {pct_diff:.2f}%  {status}")

    trades_ok = all(r.trades == results[0].trades for r in results)
    sl_ok     = all(r.sl == results[0].sl for r in results)
    tsl_ok    = all(r.tsl == results[0].tsl for r in results)
    rev_ok    = all(r.rev == results[0].rev for r in results)
    cross_ok  = max_diff < 0.01

    print(f"\n  Cross-Engine Consistency:")
    print(f"    Balance:  {'OK' if cross_ok else 'NG'} (diff: ${max_diff:.6f})")
    print(f"    Trades:   {'OK' if trades_ok else 'NG'} ({[r.trades for r in results]})")
    print(f"    SL count: {'OK' if sl_ok else 'NG'} ({[r.sl for r in results]})")
    print(f"    TSL count:{'OK' if tsl_ok else 'NG'} ({[r.tsl for r in results]})")
    print(f"    REV count:{'OK' if rev_ok else 'NG'} ({[r.rev for r in results]})")

    all_pass = cross_ok and trades_ok and sl_ok and tsl_ok and rev_ok
    print(f"\n  {'='*40}")
    print(f"  FINAL: {'ALL 6 ENGINES MATCH' if all_pass else 'MISMATCH DETECTED'}")
    print(f"  {'='*40}")

    # 첫 5거래 상세 (Engine 1 기준)
    if results[0].trade_log:
        print(f"\n  첫 5거래 상세 (Engine 1):")
        entry_count = 0
        for log in results[0].trade_log:
            if log[0] == 'ENTRY':
                entry_count += 1
                dir_str = "LONG" if log[2] == 1 else "SHORT"
                print(f"    #{entry_count} {log[0]} idx={log[1]} {dir_str} @${log[3]:,.2f}")
            else:
                dir_str = "LONG" if log[2] == 1 else "SHORT"
                print(f"       {log[0]} idx={log[1]} {dir_str} exit@${log[4]:,.2f} PnL=${log[5]:,.2f} cap=${log[6]:,.2f}")
            if entry_count >= 5 and log[0] != 'ENTRY':
                break


if __name__ == '__main__':
    main()
