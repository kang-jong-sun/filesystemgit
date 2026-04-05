"""
BTC/USDT 선물 백테스트 — 6엔진 교차 검증
v32.2: EMA(100)/EMA(600) Tight-SL Trend System
v32.3: EMA(75)/SMA(750) Low-MDD Trend System

6개 독립 엔진:
  1. Pure Python (스칼라 변수)
  2. Dict State (상태 딕셔너리)
  3. Class OOP (클래스/메서드)
  4. Numba JIT (@njit 컴파일)
  5. Vectorized Signals (사전 계산 시그널)
  6. Pandas Iterative (DataFrame 기반)
"""
import os, sys, time, math
import numpy as np
import pandas as pd
from numba import njit

# ════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════
DATA_DIR = r"D:\filesystem\futures\btc_V1\BTC_v32"

PARAMS_V32_2 = {
    'name': 'v32.2 EMA(100)/EMA(600)',
    'fast_ma_type': 'EMA', 'fast_ma_period': 100,
    'slow_ma_type': 'EMA', 'slow_ma_period': 600,
    'adx_period': 20, 'adx_min': 30.0, 'adx_rise_bars': 6,
    'rsi_period': 10, 'rsi_min': 40.0, 'rsi_max': 80.0,
    'ema_gap_min': 0.2, 'monitor_window': 24,
    'skip_same_dir': True,
    'sl_pct': 3.0, 'ta_pct': 12.0, 'tsl_pct': 9.0,
    'margin_pct': 0.35, 'leverage': 10,
    'fee_rate': 0.0004, 'initial_capital': 5000.0,
    'warmup_bars': 600, 'daily_loss_limit': -0.20,
    'daily_bars': 1440,  # 기획서 의사코드: i % 1440 (30분봉 1440개 = 30일)
}

PARAMS_V32_3 = {
    'name': 'v32.3 EMA(75)/SMA(750)',
    'fast_ma_type': 'EMA', 'fast_ma_period': 75,
    'slow_ma_type': 'SMA', 'slow_ma_period': 750,
    'adx_period': 20, 'adx_min': 30.0, 'adx_rise_bars': 6,
    'rsi_period': 11, 'rsi_min': 40.0, 'rsi_max': 80.0,
    'ema_gap_min': 0.2, 'monitor_window': 24,
    'skip_same_dir': True,
    'sl_pct': 3.0, 'ta_pct': 12.0, 'tsl_pct': 9.0,
    'margin_pct': 0.35, 'leverage': 10,
    'fee_rate': 0.0004, 'initial_capital': 5000.0,
    'warmup_bars': 600, 'daily_loss_limit': -0.20,
    'daily_bars': 1440,  # 기획서 의사코드: i % 1440 (30분봉 1440개 = 30일)
}

TARGET_V32_2 = {'final_cap': 24_073_329, 'trades': 70, 'sl': 30, 'tsl': 17, 'rev': 23, 'fc': 0, 'pf': 5.8, 'mdd': 43.5}
TARGET_V32_3 = {'final_cap': 13_236_537, 'trades': 69, 'sl': 33, 'tsl': 19, 'rev': 17, 'fc': 0, 'pf': 3.24, 'mdd': 43.5}  # 기획서 6/9절 기준


# ════════════════════════════════════════════════════════════
# DATA LOADING & RESAMPLING
# ════════════════════════════════════════════════════════════
def load_data():
    parts = []
    for i in range(1, 4):
        path = os.path.join(DATA_DIR, f"btc_usdt_5m_2020_to_now_part{i}.csv")
        df = pd.read_csv(path, parse_dates=['timestamp'])
        parts.append(df)
    df5 = pd.concat(parts, ignore_index=True)
    df5 = df5.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    df5 = df5.set_index('timestamp')

    ohlcv = df5.resample('30min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna().reset_index()
    print(f"  5분봉 {len(df5):,}개 -> 30분봉 {len(ohlcv):,}개")
    print(f"  기간: {ohlcv['timestamp'].iloc[0]} ~ {ohlcv['timestamp'].iloc[-1]}")
    return ohlcv


# ════════════════════════════════════════════════════════════
# INDICATOR COMPUTATION (공유 — 지표는 결정적이므로 한 번만 계산)
# ════════════════════════════════════════════════════════════
def compute_adx(high, low, close, period=20):
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr
    denom = plus_di + minus_di
    denom = denom.where(denom != 0, 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean().values


def compute_rsi(close, period=10):
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    alpha = 1.0 / period
    ag = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    al = al.where(al != 0, 1e-10)
    return (100 - 100 / (1 + ag / al)).values


def compute_indicators(df, params):
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    cs = pd.Series(close)
    if params['fast_ma_type'] == 'EMA':
        fast_ma = cs.ewm(span=params['fast_ma_period'], adjust=False).mean().values
    else:
        fast_ma = cs.rolling(params['fast_ma_period']).mean().values
    if params['slow_ma_type'] == 'EMA':
        slow_ma = cs.ewm(span=params['slow_ma_period'], adjust=False).mean().values
    else:
        slow_ma = cs.rolling(params['slow_ma_period']).mean().values
    adx = compute_adx(high, low, close, params['adx_period'])
    rsi = compute_rsi(close, params['rsi_period'])
    return close, high, low, fast_ma, slow_ma, adx, rsi


# ════════════════════════════════════════════════════════════
# HELPER: 결과 딕셔너리 생성
# ════════════════════════════════════════════════════════════
def make_result(cap, trades, sl, tsl, rev, fc, wins, losses, gp, gl, mdd):
    pf = gp / gl if gl > 0 else 999.0
    wr = wins / trades * 100 if trades > 0 else 0.0
    return {'final_cap': cap, 'trades': trades, 'sl': sl, 'tsl': tsl, 'rev': rev, 'fc': fc,
            'wins': wins, 'losses': losses, 'winrate': wr, 'pf': pf, 'mdd': mdd * 100}


# ════════════════════════════════════════════════════════════
# ENGINE 1: PURE PYTHON (스칼라 변수, 직접 루프)
# ════════════════════════════════════════════════════════════
def engine_1_pure_python(c, h, l, fm, sm, av, rv, p):
    n = len(c)
    cap = p['initial_capital']; fee = p['fee_rate']; warmup = p['warmup_bars']
    sl_p = p['sl_pct']; ta_p = p['ta_pct']; tsl_p = p['tsl_pct']
    mg_p = p['margin_pct']; lev = p['leverage']
    a_min = p['adx_min']; a_rise = p['adx_rise_bars']
    r_min = p['rsi_min']; r_max = p['rsi_max']
    g_min = p['ema_gap_min']; m_win = p['monitor_window']
    sk_sd = p['skip_same_dir']; d_ll = p['daily_loss_limit']; d_bars = p['daily_bars']

    pos = 0; epx = 0.0; psz = 0.0; slp = 0.0
    ton = False; thi = 0.0; tlo = 999999.0
    watching = 0; ws = 0; ld = 0
    pk = cap; mdd = 0.0; ms = cap
    trades = 0; sl_n = 0; tsl_n = 0; rev_n = 0; fc_n = 0
    wins = 0; losses = 0; gp = 0.0; gl = 0.0

    for i in range(warmup, n):
        px = c[i]; h_ = h[i]; l_ = l[i]

        # Cross detection
        if i > 0:
            bn = fm[i] > sm[i]; bp = fm[i-1] > sm[i-1]
            cu = bn and not bp; cd = not bn and bp
        else:
            bn = bp = cu = cd = False

        # Daily reset
        if i > warmup and d_bars > 0 and i % d_bars == 0:
            ms = cap

        # ── STEP A: 포지션 청산 체크 ──
        if pos != 0:
            watching = 0

            # A1: SL (ton=False only, 저가/고가, SL가에 청산)
            if not ton:
                if (pos == 1 and l_ <= slp) or (pos == -1 and h_ >= slp):
                    pnl = (slp - epx) / epx * psz * pos - psz * fee
                    cap += pnl
                    if pnl > 0: wins += 1; gp += pnl
                    else: losses += 1; gl += abs(pnl)
                    sl_n += 1; trades += 1; ld = pos; pos = 0
                    continue

            # A1.5: FC (ton=True, intrabar 강제청산)
            if ton:
                lq = 1.0 / lev
                if (pos == 1 and l_ <= epx * (1 - lq)) or (pos == -1 and h_ >= epx * (1 + lq)):
                    liq_px = epx * (1 - lq) if pos == 1 else epx * (1 + lq)
                    pnl = (liq_px - epx) / epx * psz * pos - psz * fee
                    cap += pnl
                    if pnl > 0: wins += 1; gp += pnl
                    else: losses += 1; gl += abs(pnl)
                    fc_n += 1; trades += 1; ld = pos; pos = 0
                    continue

            # A2: TA 활성화 (고가/저가)
            br = (h_ - epx) / epx * 100 if pos == 1 else (epx - l_) / epx * 100
            if br >= ta_p:
                ton = True

            # A3: TSL (ton=True, 종가 기준)
            if ton:
                if pos == 1:
                    if h_ > thi: thi = h_
                    ns = thi * (1 - tsl_p / 100)
                    if ns > slp: slp = ns
                    if px <= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * fee
                        cap += pnl
                        if pnl > 0: wins += 1; gp += pnl
                        else: losses += 1; gl += abs(pnl)
                        tsl_n += 1; trades += 1; ld = pos; pos = 0
                        continue
                else:
                    if l_ < tlo: tlo = l_
                    ns = tlo * (1 + tsl_p / 100)
                    if ns < slp: slp = ns
                    if px >= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * fee
                        cap += pnl
                        if pnl > 0: wins += 1; gp += pnl
                        else: losses += 1; gl += abs(pnl)
                        tsl_n += 1; trades += 1; ld = pos; pos = 0
                        continue

            # A4: REV (종가, EMA 교차 반전) — continue 없음
            if (pos == 1 and cd) or (pos == -1 and cu):
                pnl = (px - epx) / epx * psz * pos - psz * fee
                cap += pnl
                if pnl > 0: wins += 1; gp += pnl
                else: losses += 1; gl += abs(pnl)
                rev_n += 1; trades += 1; ld = pos; pos = 0

        # ── STEP B: 진입 체크 ──
        if pos == 0:
            if cu: watching = 1; ws = i
            elif cd: watching = -1; ws = i

            if watching != 0 and i > ws:
                enter = False
                if i - ws > m_win:
                    watching = 0
                elif watching == 1 and cd:
                    watching = -1; ws = i
                elif watching == -1 and cu:
                    watching = 1; ws = i
                elif sk_sd and watching == ld:
                    pass
                elif av[i] != av[i] or av[i] < a_min:
                    pass
                elif a_rise > 0 and i >= a_rise and av[i] <= av[i - a_rise]:
                    pass
                elif rv[i] != rv[i] or rv[i] < r_min or rv[i] > r_max:
                    pass
                elif g_min > 0:
                    if sm[i] != sm[i] or sm[i] == 0:
                        pass
                    elif abs(fm[i] - sm[i]) / sm[i] * 100 < g_min:
                        pass
                    else:
                        enter = True
                else:
                    enter = True

                if enter:
                    if ms > 0 and (cap - ms) / ms <= d_ll:
                        watching = 0; enter = False
                    elif cap <= 0:
                        enter = False

                if enter:
                    psz = cap * mg_p * lev
                    cap -= psz * fee
                    pos = watching; epx = px; ton = False; thi = px; tlo = px
                    slp = epx * (1 - sl_p / 100) if pos == 1 else epx * (1 + sl_p / 100)
                    pk = max(pk, cap); watching = 0

        # MDD
        pk = max(pk, cap)
        dd = (pk - cap) / pk if pk > 0 else 0.0
        if dd > mdd: mdd = dd
        if cap <= 0: break

    if pos != 0 and cap > 0:
        pnl = (c[n-1] - epx) / epx * psz * pos - psz * fee
        cap += pnl
    return make_result(cap, trades, sl_n, tsl_n, rev_n, fc_n, wins, losses, gp, gl, mdd)


# ════════════════════════════════════════════════════════════
# ENGINE 2: DICT STATE (모든 상태를 딕셔너리로 관리)
# ════════════════════════════════════════════════════════════
def engine_2_dict_state(c, h, l, fm, sm, av, rv, p):
    n = len(c)
    fee = p['fee_rate']; warmup = p['warmup_bars']
    sl_p = p['sl_pct']; ta_p = p['ta_pct']; tsl_p = p['tsl_pct']
    mg_p = p['margin_pct']; lev = p['leverage']; lq = 1.0 / lev
    a_min = p['adx_min']; a_rise = p['adx_rise_bars']
    r_min = p['rsi_min']; r_max = p['rsi_max']
    g_min = p['ema_gap_min']; m_win = p['monitor_window']
    sk_sd = p['skip_same_dir']; d_ll = p['daily_loss_limit']; d_bars = p['daily_bars']

    s = {'cap': p['initial_capital'], 'pos': 0, 'epx': 0.0, 'psz': 0.0, 'slp': 0.0,
         'ton': False, 'thi': 0.0, 'tlo': 999999.0, 'w': 0, 'ws': 0, 'ld': 0,
         'pk': p['initial_capital'], 'mdd': 0.0, 'ms': p['initial_capital'],
         'tr': 0, 'sl': 0, 'tsl': 0, 'rev': 0, 'fc': 0,
         'wi': 0, 'lo': 0, 'gp': 0.0, 'gl': 0.0}

    def do_exit(s, exit_px, etype):
        pnl = (exit_px - s['epx']) / s['epx'] * s['psz'] * s['pos'] - s['psz'] * fee
        s['cap'] += pnl
        if pnl > 0: s['wi'] += 1; s['gp'] += pnl
        else: s['lo'] += 1; s['gl'] += abs(pnl)
        s[etype] += 1; s['tr'] += 1; s['ld'] = s['pos']; s['pos'] = 0

    for i in range(warmup, n):
        px = c[i]; h_ = h[i]; l_ = l[i]
        if i > 0:
            bn = fm[i] > sm[i]; bp = fm[i-1] > sm[i-1]
            cu = bn and not bp; cd = not bn and bp
        else:
            bn = bp = cu = cd = False

        if i > warmup and d_bars > 0 and i % d_bars == 0:
            s['ms'] = s['cap']

        if s['pos'] != 0:
            s['w'] = 0
            if not s['ton']:
                if (s['pos'] == 1 and l_ <= s['slp']) or (s['pos'] == -1 and h_ >= s['slp']):
                    do_exit(s, s['slp'], 'sl'); continue
            if s['ton']:
                if (s['pos'] == 1 and l_ <= s['epx'] * (1 - lq)) or (s['pos'] == -1 and h_ >= s['epx'] * (1 + lq)):
                    liq_px = s['epx'] * (1 - lq) if s['pos'] == 1 else s['epx'] * (1 + lq)
                    do_exit(s, liq_px, 'fc'); continue
            br = (h_ - s['epx']) / s['epx'] * 100 if s['pos'] == 1 else (s['epx'] - l_) / s['epx'] * 100
            if br >= ta_p: s['ton'] = True
            if s['ton']:
                if s['pos'] == 1:
                    if h_ > s['thi']: s['thi'] = h_
                    ns = s['thi'] * (1 - tsl_p / 100)
                    if ns > s['slp']: s['slp'] = ns
                    if px <= s['slp']:
                        do_exit(s, px, 'tsl'); continue
                else:
                    if l_ < s['tlo']: s['tlo'] = l_
                    ns = s['tlo'] * (1 + tsl_p / 100)
                    if ns < s['slp']: s['slp'] = ns
                    if px >= s['slp']:
                        do_exit(s, px, 'tsl'); continue
            if (s['pos'] == 1 and cd) or (s['pos'] == -1 and cu):
                do_exit(s, px, 'rev')

        if s['pos'] == 0:
            if cu: s['w'] = 1; s['ws'] = i
            elif cd: s['w'] = -1; s['ws'] = i
            if s['w'] != 0 and i > s['ws']:
                enter = False
                if i - s['ws'] > m_win:
                    s['w'] = 0
                elif s['w'] == 1 and cd:
                    s['w'] = -1; s['ws'] = i
                elif s['w'] == -1 and cu:
                    s['w'] = 1; s['ws'] = i
                elif sk_sd and s['w'] == s['ld']:
                    pass
                elif av[i] != av[i] or av[i] < a_min:
                    pass
                elif a_rise > 0 and i >= a_rise and av[i] <= av[i - a_rise]:
                    pass
                elif rv[i] != rv[i] or rv[i] < r_min or rv[i] > r_max:
                    pass
                elif g_min > 0:
                    if sm[i] != sm[i] or sm[i] == 0: pass
                    elif abs(fm[i] - sm[i]) / sm[i] * 100 < g_min: pass
                    else: enter = True
                else:
                    enter = True
                if enter and s['ms'] > 0 and (s['cap'] - s['ms']) / s['ms'] <= d_ll:
                    s['w'] = 0; enter = False
                if enter and s['cap'] <= 0:
                    enter = False
                if enter:
                    s['psz'] = s['cap'] * mg_p * lev; s['cap'] -= s['psz'] * fee
                    s['pos'] = s['w']; s['epx'] = px; s['ton'] = False
                    s['thi'] = px; s['tlo'] = px
                    s['slp'] = s['epx'] * (1 - sl_p / 100) if s['pos'] == 1 else s['epx'] * (1 + sl_p / 100)
                    s['pk'] = max(s['pk'], s['cap']); s['w'] = 0

        s['pk'] = max(s['pk'], s['cap'])
        dd = (s['pk'] - s['cap']) / s['pk'] if s['pk'] > 0 else 0.0
        if dd > s['mdd']: s['mdd'] = dd
        if s['cap'] <= 0: break

    if s['pos'] != 0 and s['cap'] > 0:
        pnl = (c[n-1] - s['epx']) / s['epx'] * s['psz'] * s['pos'] - s['psz'] * fee
        s['cap'] += pnl
    return make_result(s['cap'], s['tr'], s['sl'], s['tsl'], s['rev'], s['fc'],
                       s['wi'], s['lo'], s['gp'], s['gl'], s['mdd'])


# ════════════════════════════════════════════════════════════
# ENGINE 3: CLASS OOP (Position/Trade 클래스)
# ════════════════════════════════════════════════════════════
class _Position:
    __slots__ = ['direction', 'entry_px', 'size', 'sl_price', 'tsl_on', 'track_hi', 'track_lo']
    def __init__(self, direction, entry_px, size, sl_pct, leverage):
        self.direction = direction
        self.entry_px = entry_px
        self.size = size
        self.tsl_on = False
        self.track_hi = entry_px
        self.track_lo = entry_px
        self.sl_price = entry_px * (1 - sl_pct / 100) if direction == 1 else entry_px * (1 + sl_pct / 100)

class _BTEngine:
    def __init__(self, params):
        self.p = params
        self.cap = params['initial_capital']
        self.pk = self.cap; self.mdd = 0.0; self.ms = self.cap
        self.pos = None; self.watching = 0; self.ws = 0; self.ld = 0
        self.stats = {'tr': 0, 'sl': 0, 'tsl': 0, 'rev': 0, 'fc': 0,
                      'wi': 0, 'lo': 0, 'gp': 0.0, 'gl': 0.0}

    def close_pos(self, exit_px, etype):
        pos = self.pos
        pnl = (exit_px - pos.entry_px) / pos.entry_px * pos.size * pos.direction - pos.size * self.p['fee_rate']
        self.cap += pnl
        if pnl > 0: self.stats['wi'] += 1; self.stats['gp'] += pnl
        else: self.stats['lo'] += 1; self.stats['gl'] += abs(pnl)
        self.stats[etype] += 1; self.stats['tr'] += 1
        self.ld = pos.direction; self.pos = None

    def open_pos(self, direction, px):
        sz = self.cap * self.p['margin_pct'] * self.p['leverage']
        self.cap -= sz * self.p['fee_rate']
        self.pos = _Position(direction, px, sz, self.p['sl_pct'], self.p['leverage'])
        self.pk = max(self.pk, self.cap); self.watching = 0

    def update_mdd(self):
        self.pk = max(self.pk, self.cap)
        dd = (self.pk - self.cap) / self.pk if self.pk > 0 else 0.0
        if dd > self.mdd: self.mdd = dd

def engine_3_class_oop(c, h, l, fm, sm, av, rv, p):
    n = len(c); eng = _BTEngine(p)
    fee = p['fee_rate']; warmup = p['warmup_bars']
    ta_p = p['ta_pct']; tsl_p = p['tsl_pct']; lq = 1.0 / p['leverage']
    a_min = p['adx_min']; a_rise = p['adx_rise_bars']
    r_min = p['rsi_min']; r_max = p['rsi_max']
    g_min = p['ema_gap_min']; m_win = p['monitor_window']
    sk_sd = p['skip_same_dir']; d_ll = p['daily_loss_limit']; d_bars = p['daily_bars']

    for i in range(warmup, n):
        px = c[i]; h_ = h[i]; l_ = l[i]
        if i > 0:
            bn = fm[i] > sm[i]; bp = fm[i-1] > sm[i-1]
            cu = bn and not bp; cd = not bn and bp
        else:
            bn = bp = cu = cd = False
        if i > warmup and d_bars > 0 and i % d_bars == 0:
            eng.ms = eng.cap

        if eng.pos is not None:
            eng.watching = 0; pos = eng.pos; d = pos.direction
            if not pos.tsl_on:
                if (d == 1 and l_ <= pos.sl_price) or (d == -1 and h_ >= pos.sl_price):
                    eng.close_pos(pos.sl_price, 'sl'); continue
            if pos.tsl_on:
                if (d == 1 and l_ <= pos.entry_px * (1 - lq)) or (d == -1 and h_ >= pos.entry_px * (1 + lq)):
                    liq_px = pos.entry_px * (1 - lq) if d == 1 else pos.entry_px * (1 + lq)
                    eng.close_pos(liq_px, 'fc'); continue
            br = (h_ - pos.entry_px) / pos.entry_px * 100 if d == 1 else (pos.entry_px - l_) / pos.entry_px * 100
            if br >= ta_p: pos.tsl_on = True
            if pos.tsl_on:
                if d == 1:
                    if h_ > pos.track_hi: pos.track_hi = h_
                    ns = pos.track_hi * (1 - tsl_p / 100)
                    if ns > pos.sl_price: pos.sl_price = ns
                    if px <= pos.sl_price:
                        eng.close_pos(px, 'tsl'); continue
                else:
                    if l_ < pos.track_lo: pos.track_lo = l_
                    ns = pos.track_lo * (1 + tsl_p / 100)
                    if ns < pos.sl_price: pos.sl_price = ns
                    if px >= pos.sl_price:
                        eng.close_pos(px, 'tsl'); continue
            if (d == 1 and cd) or (d == -1 and cu):
                eng.close_pos(px, 'rev')

        if eng.pos is None:
            if cu: eng.watching = 1; eng.ws = i
            elif cd: eng.watching = -1; eng.ws = i
            if eng.watching != 0 and i > eng.ws:
                enter = False
                if i - eng.ws > m_win:
                    eng.watching = 0
                elif eng.watching == 1 and cd:
                    eng.watching = -1; eng.ws = i
                elif eng.watching == -1 and cu:
                    eng.watching = 1; eng.ws = i
                elif sk_sd and eng.watching == eng.ld: pass
                elif av[i] != av[i] or av[i] < a_min: pass
                elif a_rise > 0 and i >= a_rise and av[i] <= av[i - a_rise]: pass
                elif rv[i] != rv[i] or rv[i] < r_min or rv[i] > r_max: pass
                elif g_min > 0:
                    if sm[i] != sm[i] or sm[i] == 0: pass
                    elif abs(fm[i] - sm[i]) / sm[i] * 100 < g_min: pass
                    else: enter = True
                else: enter = True
                if enter and eng.ms > 0 and (eng.cap - eng.ms) / eng.ms <= d_ll:
                    eng.watching = 0; enter = False
                if enter and eng.cap <= 0: enter = False
                if enter:
                    eng.open_pos(eng.watching, px)

        eng.update_mdd()
        if eng.cap <= 0: break

    if eng.pos is not None and eng.cap > 0:
        pnl = (c[n-1] - eng.pos.entry_px) / eng.pos.entry_px * eng.pos.size * eng.pos.direction - eng.pos.size * fee
        eng.cap += pnl
    st = eng.stats
    return make_result(eng.cap, st['tr'], st['sl'], st['tsl'], st['rev'], st['fc'],
                       st['wi'], st['lo'], st['gp'], st['gl'], eng.mdd)


# ════════════════════════════════════════════════════════════
# ENGINE 4: NUMBA JIT (@njit 컴파일)
# ════════════════════════════════════════════════════════════
@njit(cache=True)
def _bt_numba(c, h, l, fm, sm, av, rv,
              init_cap, fee, warmup, sl_p, ta_p, tsl_p, mg_p, lev,
              a_min, a_rise, r_min, r_max, g_min, m_win, sk_sd, d_ll, d_bars):
    n = len(c)
    cap = init_cap; pos = 0; epx = 0.0; psz = 0.0; slp = 0.0
    ton = 0; thi = 0.0; tlo = 999999.0
    watching = 0; ws = 0; ld = 0
    pk = cap; mdd_val = 0.0; ms = cap
    lq = 1.0 / lev
    tr = 0; sl_n = 0; tsl_n = 0; rev_n = 0; fc_n = 0
    wi = 0; lo = 0; gp = 0.0; gl = 0.0
    warmup_i = int(warmup); m_win_i = int(m_win); a_rise_i = int(a_rise); d_bars_i = int(d_bars)

    for i in range(warmup_i, n):
        px = c[i]; h_ = h[i]; l_ = l[i]
        cu = False; cd = False; bn = False; bp = False
        if i > 0:
            v1 = fm[i]; v2 = sm[i]; v3 = fm[i-1]; v4 = sm[i-1]
            if v1 == v1 and v2 == v2: bn = v1 > v2
            if v3 == v3 and v4 == v4: bp = v3 > v4
            cu = bn and not bp; cd = not bn and bp
        if i > warmup_i and d_bars_i > 0 and i % d_bars_i == 0:
            ms = cap

        if pos != 0:
            watching = 0
            if ton == 0:
                sl_hit = False
                if pos == 1 and l_ <= slp: sl_hit = True
                elif pos == -1 and h_ >= slp: sl_hit = True
                if sl_hit:
                    pnl = (slp - epx) / epx * psz * pos - psz * fee
                    cap += pnl
                    if pnl > 0: wi += 1; gp += pnl
                    else: lo += 1; gl += abs(pnl)
                    sl_n += 1; tr += 1; ld = pos; pos = 0
                    pk = max(pk, cap)
                    dd = (pk - cap) / pk if pk > 0 else 0.0
                    if dd > mdd_val: mdd_val = dd
                    if cap <= 0: break
                    continue
            if ton == 1:
                fc_hit = False
                if pos == 1 and l_ <= epx * (1 - lq): fc_hit = True
                elif pos == -1 and h_ >= epx * (1 + lq): fc_hit = True
                if fc_hit:
                    liq_px = epx * (1 - lq) if pos == 1 else epx * (1 + lq)
                    pnl = (liq_px - epx) / epx * psz * pos - psz * fee
                    cap += pnl
                    if pnl > 0: wi += 1; gp += pnl
                    else: lo += 1; gl += abs(pnl)
                    fc_n += 1; tr += 1; ld = pos; pos = 0
                    pk = max(pk, cap)
                    dd = (pk - cap) / pk if pk > 0 else 0.0
                    if dd > mdd_val: mdd_val = dd
                    if cap <= 0: break
                    continue
            if pos == 1: br = (h_ - epx) / epx * 100
            else: br = (epx - l_) / epx * 100
            if br >= ta_p: ton = 1
            if ton == 1:
                if pos == 1:
                    if h_ > thi: thi = h_
                    ns = thi * (1 - tsl_p / 100)
                    if ns > slp: slp = ns
                    if px <= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * fee
                        cap += pnl
                        if pnl > 0: wi += 1; gp += pnl
                        else: lo += 1; gl += abs(pnl)
                        tsl_n += 1; tr += 1; ld = pos; pos = 0
                        pk = max(pk, cap)
                        dd = (pk - cap) / pk if pk > 0 else 0.0
                        if dd > mdd_val: mdd_val = dd
                        if cap <= 0: break
                        continue
                else:
                    if l_ < tlo: tlo = l_
                    ns = tlo * (1 + tsl_p / 100)
                    if ns < slp: slp = ns
                    if px >= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * fee
                        cap += pnl
                        if pnl > 0: wi += 1; gp += pnl
                        else: lo += 1; gl += abs(pnl)
                        tsl_n += 1; tr += 1; ld = pos; pos = 0
                        pk = max(pk, cap)
                        dd = (pk - cap) / pk if pk > 0 else 0.0
                        if dd > mdd_val: mdd_val = dd
                        if cap <= 0: break
                        continue
            if (pos == 1 and cd) or (pos == -1 and cu):
                pnl = (px - epx) / epx * psz * pos - psz * fee
                cap += pnl
                if pnl > 0: wi += 1; gp += pnl
                else: lo += 1; gl += abs(pnl)
                rev_n += 1; tr += 1; ld = pos; pos = 0

        if pos == 0:
            if cu: watching = 1; ws = i
            elif cd: watching = -1; ws = i
            if watching != 0 and i > ws:
                enter = False
                if i - ws > m_win_i:
                    watching = 0
                elif watching == 1 and cd:
                    watching = -1; ws = i
                elif watching == -1 and cu:
                    watching = 1; ws = i
                elif sk_sd > 0.5 and watching == ld:
                    enter = False
                elif av[i] != av[i] or av[i] < a_min:
                    enter = False
                elif a_rise_i > 0 and i >= a_rise_i and av[i] <= av[i - a_rise_i]:
                    enter = False
                elif rv[i] != rv[i] or rv[i] < r_min or rv[i] > r_max:
                    enter = False
                elif g_min > 0:
                    if sm[i] != sm[i] or sm[i] == 0.0:
                        enter = False
                    elif abs(fm[i] - sm[i]) / sm[i] * 100 < g_min:
                        enter = False
                    else:
                        enter = True
                else:
                    enter = True
                if enter and ms > 0.0 and (cap - ms) / ms <= d_ll:
                    watching = 0; enter = False
                if enter and cap <= 0:
                    enter = False
                if enter:
                    psz = cap * mg_p * lev; cap -= psz * fee
                    pos = watching; epx = px; ton = 0; thi = px; tlo = px
                    if pos == 1: slp = epx * (1 - sl_p / 100)
                    else: slp = epx * (1 + sl_p / 100)
                    pk = max(pk, cap); watching = 0

        pk = max(pk, cap)
        dd = (pk - cap) / pk if pk > 0 else 0.0
        if dd > mdd_val: mdd_val = dd
        if cap <= 0: break

    if pos != 0 and cap > 0:
        pnl = (c[n-1] - epx) / epx * psz * pos - psz * fee
        cap += pnl
    return cap, tr, sl_n, tsl_n, rev_n, fc_n, wi, lo, gp, gl, mdd_val

def engine_4_numba_jit(c, h, l, fm, sm, av, rv, p):
    res = _bt_numba(c, h, l, fm, sm, av, rv,
                    p['initial_capital'], p['fee_rate'], float(p['warmup_bars']),
                    p['sl_pct'], p['ta_pct'], p['tsl_pct'], p['margin_pct'], float(p['leverage']),
                    p['adx_min'], float(p['adx_rise_bars']),
                    p['rsi_min'], p['rsi_max'], p['ema_gap_min'], float(p['monitor_window']),
                    1.0 if p['skip_same_dir'] else 0.0, p['daily_loss_limit'], float(p['daily_bars']))
    return make_result(*res)


# ════════════════════════════════════════════════════════════
# ENGINE 5: VECTORIZED SIGNALS (사전 계산 시그널 배열)
# ════════════════════════════════════════════════════════════
def engine_5_vectorized(c, h, l, fm, sm, av, rv, p):
    n = len(c); fee = p['fee_rate']; warmup = p['warmup_bars']
    sl_p = p['sl_pct']; ta_p = p['ta_pct']; tsl_p = p['tsl_pct']
    mg_p = p['margin_pct']; lev = p['leverage']; lq = 1.0 / lev
    a_min = p['adx_min']; a_rise = p['adx_rise_bars']
    r_min = p['rsi_min']; r_max = p['rsi_max']
    g_min = p['ema_gap_min']; m_win = p['monitor_window']
    sk_sd = p['skip_same_dir']; d_ll = p['daily_loss_limit']; d_bars = p['daily_bars']

    # Pre-compute boolean signal arrays
    bull = np.zeros(n, dtype=np.bool_)
    cross_up = np.zeros(n, dtype=np.bool_)
    cross_dn = np.zeros(n, dtype=np.bool_)
    adx_ok = np.zeros(n, dtype=np.bool_)
    rsi_ok = np.zeros(n, dtype=np.bool_)
    gap_ok = np.zeros(n, dtype=np.bool_)
    adx_rise_ok = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        if fm[i] == fm[i] and sm[i] == sm[i]:
            bull[i] = fm[i] > sm[i]
    for i in range(1, n):
        cross_up[i] = bull[i] and not bull[i-1]
        cross_dn[i] = not bull[i] and bull[i-1]
        adx_ok[i] = (av[i] == av[i]) and (av[i] >= a_min)
        if a_rise > 0 and i >= a_rise:
            adx_rise_ok[i] = (av[i] == av[i]) and (av[i-a_rise] == av[i-a_rise]) and (av[i] > av[i-a_rise])
        elif a_rise == 0:
            adx_rise_ok[i] = True
        rsi_ok[i] = (rv[i] == rv[i]) and (rv[i] >= r_min) and (rv[i] <= r_max)
        if g_min > 0 and sm[i] == sm[i] and sm[i] != 0:
            gap_ok[i] = abs(fm[i] - sm[i]) / sm[i] * 100 >= g_min
        elif g_min <= 0:
            gap_ok[i] = True

    cap = p['initial_capital']; pos = 0; epx = 0.0; psz = 0.0; slp = 0.0
    ton = False; thi = 0.0; tlo = 999999.0
    watching = 0; ws = 0; ld = 0
    pk = cap; mdd = 0.0; ms = cap
    tr = 0; sl_n = 0; tsl_n = 0; rev_n = 0; fc_n = 0
    wi = 0; lo = 0; gp = 0.0; gl = 0.0

    for i in range(warmup, n):
        px = c[i]; h_ = h[i]; l_ = l[i]
        cu = cross_up[i]; cd = cross_dn[i]
        if i > warmup and d_bars > 0 and i % d_bars == 0: ms = cap

        if pos != 0:
            watching = 0
            if not ton:
                if (pos == 1 and l_ <= slp) or (pos == -1 and h_ >= slp):
                    pnl = (slp - epx) / epx * psz * pos - psz * fee; cap += pnl
                    if pnl > 0: wi += 1; gp += pnl
                    else: lo += 1; gl += abs(pnl)
                    sl_n += 1; tr += 1; ld = pos; pos = 0; continue
            if ton:
                if (pos == 1 and l_ <= epx * (1 - lq)) or (pos == -1 and h_ >= epx * (1 + lq)):
                    liq_px = epx * (1 - lq) if pos == 1 else epx * (1 + lq)
                    pnl = (liq_px - epx) / epx * psz * pos - psz * fee; cap += pnl
                    if pnl > 0: wi += 1; gp += pnl
                    else: lo += 1; gl += abs(pnl)
                    fc_n += 1; tr += 1; ld = pos; pos = 0; continue
            br = (h_ - epx) / epx * 100 if pos == 1 else (epx - l_) / epx * 100
            if br >= ta_p: ton = True
            if ton:
                if pos == 1:
                    if h_ > thi: thi = h_
                    ns = thi * (1 - tsl_p / 100)
                    if ns > slp: slp = ns
                    if px <= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * fee; cap += pnl
                        if pnl > 0: wi += 1; gp += pnl
                        else: lo += 1; gl += abs(pnl)
                        tsl_n += 1; tr += 1; ld = pos; pos = 0; continue
                else:
                    if l_ < tlo: tlo = l_
                    ns = tlo * (1 + tsl_p / 100)
                    if ns < slp: slp = ns
                    if px >= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * fee; cap += pnl
                        if pnl > 0: wi += 1; gp += pnl
                        else: lo += 1; gl += abs(pnl)
                        tsl_n += 1; tr += 1; ld = pos; pos = 0; continue
            if (pos == 1 and cd) or (pos == -1 and cu):
                pnl = (px - epx) / epx * psz * pos - psz * fee; cap += pnl
                if pnl > 0: wi += 1; gp += pnl
                else: lo += 1; gl += abs(pnl)
                rev_n += 1; tr += 1; ld = pos; pos = 0

        if pos == 0:
            if cu: watching = 1; ws = i
            elif cd: watching = -1; ws = i
            if watching != 0 and i > ws:
                enter = False
                if i - ws > m_win: watching = 0
                elif watching == 1 and cd: watching = -1; ws = i
                elif watching == -1 and cu: watching = 1; ws = i
                elif sk_sd and watching == ld: pass
                elif not adx_ok[i]: pass
                elif not adx_rise_ok[i]: pass
                elif not rsi_ok[i]: pass
                elif not gap_ok[i]: pass
                else: enter = True
                if enter and ms > 0 and (cap - ms) / ms <= d_ll: watching = 0; enter = False
                if enter and cap <= 0: enter = False
                if enter:
                    psz = cap * mg_p * lev; cap -= psz * fee
                    pos = watching; epx = px; ton = False; thi = px; tlo = px
                    slp = epx * (1 - sl_p / 100) if pos == 1 else epx * (1 + sl_p / 100)
                    pk = max(pk, cap); watching = 0

        pk = max(pk, cap)
        dd = (pk - cap) / pk if pk > 0 else 0.0
        if dd > mdd: mdd = dd
        if cap <= 0: break

    if pos != 0 and cap > 0:
        pnl = (c[n-1] - epx) / epx * psz * pos - psz * fee; cap += pnl
    return make_result(cap, tr, sl_n, tsl_n, rev_n, fc_n, wi, lo, gp, gl, mdd)


# ════════════════════════════════════════════════════════════
# ENGINE 6: NUMPY STATE ARRAYS (상태 변수를 배열로 추적)
# ════════════════════════════════════════════════════════════
def engine_6_numpy_arrays(c, h, l, fm, sm, av, rv, p):
    n = len(c); fee = p['fee_rate']; warmup = p['warmup_bars']
    sl_p = p['sl_pct']; ta_p = p['ta_pct']; tsl_p = p['tsl_pct']
    mg_p = p['margin_pct']; lev = p['leverage']; lq = 1.0 / lev
    a_min = p['adx_min']; a_rise = p['adx_rise_bars']
    r_min = p['rsi_min']; r_max = p['rsi_max']
    g_min = p['ema_gap_min']; m_win = p['monitor_window']
    sk_sd = p['skip_same_dir']; d_ll = p['daily_loss_limit']; d_bars = p['daily_bars']

    # State arrays for equity curve tracking
    equity = np.empty(n, dtype=np.float64)
    equity[:] = np.nan

    cap = p['initial_capital']; pos = 0; epx = 0.0; psz = 0.0; slp = 0.0
    ton = 0; thi = 0.0; tlo = 999999.0
    watching = 0; ws = 0; ld = 0
    pk = cap; mdd = 0.0; ms = cap
    tr = 0; sl_n = 0; tsl_n = 0; rev_n = 0; fc_n = 0
    wi = 0; lo = 0; gp = 0.0; gl = 0.0

    for i in range(warmup, n):
        px = c[i]; h_ = h[i]; l_ = l[i]
        cu = False; cd = False
        if i > 0:
            f_now = fm[i]; s_now = sm[i]; f_prev = fm[i-1]; s_prev = sm[i-1]
            bn = (f_now == f_now and s_now == s_now) and f_now > s_now
            bp_v = (f_prev == f_prev and s_prev == s_prev) and f_prev > s_prev
            cu = bn and not bp_v; cd = not bn and bp_v
        if i > warmup and d_bars > 0 and i % d_bars == 0: ms = cap

        if pos != 0:
            watching = 0
            if ton == 0:
                if (pos == 1 and l_ <= slp) or (pos == -1 and h_ >= slp):
                    pnl = (slp - epx) / epx * psz * pos - psz * fee; cap += pnl
                    if pnl > 0: wi += 1; gp += pnl
                    else: lo += 1; gl += abs(pnl)
                    sl_n += 1; tr += 1; ld = pos; pos = 0
                    equity[i] = cap; continue
            if ton == 1:
                if (pos == 1 and l_ <= epx * (1 - lq)) or (pos == -1 and h_ >= epx * (1 + lq)):
                    liq_px = epx * (1 - lq) if pos == 1 else epx * (1 + lq)
                    pnl = (liq_px - epx) / epx * psz * pos - psz * fee; cap += pnl
                    if pnl > 0: wi += 1; gp += pnl
                    else: lo += 1; gl += abs(pnl)
                    fc_n += 1; tr += 1; ld = pos; pos = 0
                    equity[i] = cap; continue
            if pos == 1: br = (h_ - epx) / epx * 100
            else: br = (epx - l_) / epx * 100
            if br >= ta_p: ton = 1
            if ton == 1:
                if pos == 1:
                    if h_ > thi: thi = h_
                    ns = thi * (1 - tsl_p / 100)
                    if ns > slp: slp = ns
                    if px <= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * fee; cap += pnl
                        if pnl > 0: wi += 1; gp += pnl
                        else: lo += 1; gl += abs(pnl)
                        tsl_n += 1; tr += 1; ld = pos; pos = 0
                        equity[i] = cap; continue
                else:
                    if l_ < tlo: tlo = l_
                    ns = tlo * (1 + tsl_p / 100)
                    if ns < slp: slp = ns
                    if px >= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * fee; cap += pnl
                        if pnl > 0: wi += 1; gp += pnl
                        else: lo += 1; gl += abs(pnl)
                        tsl_n += 1; tr += 1; ld = pos; pos = 0
                        equity[i] = cap; continue
            if (pos == 1 and cd) or (pos == -1 and cu):
                pnl = (px - epx) / epx * psz * pos - psz * fee; cap += pnl
                if pnl > 0: wi += 1; gp += pnl
                else: lo += 1; gl += abs(pnl)
                rev_n += 1; tr += 1; ld = pos; pos = 0

        if pos == 0:
            if cu: watching = 1; ws = i
            elif cd: watching = -1; ws = i
            if watching != 0 and i > ws:
                enter = False
                if i - ws > m_win: watching = 0
                elif watching == 1 and cd: watching = -1; ws = i
                elif watching == -1 and cu: watching = 1; ws = i
                elif sk_sd and watching == ld: pass
                elif av[i] != av[i] or av[i] < a_min: pass
                elif a_rise > 0 and i >= a_rise and av[i] <= av[i - a_rise]: pass
                elif rv[i] != rv[i] or rv[i] < r_min or rv[i] > r_max: pass
                elif g_min > 0:
                    if sm[i] != sm[i] or sm[i] == 0: pass
                    elif abs(fm[i] - sm[i]) / sm[i] * 100 < g_min: pass
                    else: enter = True
                else: enter = True
                if enter and ms > 0 and (cap - ms) / ms <= d_ll: watching = 0; enter = False
                if enter and cap <= 0: enter = False
                if enter:
                    psz = cap * mg_p * lev; cap -= psz * fee
                    pos = watching; epx = px; ton = 0; thi = px; tlo = px
                    slp = epx * (1 - sl_p / 100) if pos == 1 else epx * (1 + sl_p / 100)
                    pk = max(pk, cap); watching = 0

        equity[i] = cap
        pk = max(pk, cap)
        dd = (pk - cap) / pk if pk > 0 else 0.0
        if dd > mdd: mdd = dd
        if cap <= 0: break

    if pos != 0 and cap > 0:
        pnl = (c[n-1] - epx) / epx * psz * pos - psz * fee; cap += pnl
    return make_result(cap, tr, sl_n, tsl_n, rev_n, fc_n, wi, lo, gp, gl, mdd)


# ════════════════════════════════════════════════════════════
# CROSS-VALIDATION RUNNER
# ════════════════════════════════════════════════════════════
ENGINES = [
    ("1.PurePython", engine_1_pure_python),
    ("2.DictState",  engine_2_dict_state),
    ("3.ClassOOP",   engine_3_class_oop),
    ("4.NumbaJIT",   engine_4_numba_jit),
    ("5.VecSignal",  engine_5_vectorized),
    ("6.NpArrays",   engine_6_numpy_arrays),
]

def run_cross_validation(c, h, l, fm, sm, av, rv, params, target, label):
    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"{'='*90}")

    # Numba warmup (first call compiles)
    print("  Numba JIT 컴파일 중...")
    _ = engine_4_numba_jit(c[:1000], h[:1000], l[:1000], fm[:1000], sm[:1000], av[:1000], rv[:1000], params)

    results = []
    for name, func in ENGINES:
        t0 = time.perf_counter()
        res = func(c, h, l, fm, sm, av, rv, params)
        elapsed = time.perf_counter() - t0
        res['time'] = elapsed
        res['name'] = name
        results.append(res)

    # Print table
    print(f"\n  {'Engine':<14} {'Final Cap':>16} {'Tr':>4} {'SL':>4} {'TSL':>4} {'REV':>4} {'FC':>3} {'W/L':>7} {'PF':>7} {'MDD':>7} {'Time':>8}")
    print(f"  {'─'*14} {'─'*16} {'─'*4} {'─'*4} {'─'*4} {'─'*4} {'─'*3} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")
    for r in results:
        wl = f"{r['wins']}/{r['losses']}"
        print(f"  {r['name']:<14} ${r['final_cap']:>14,.2f} {r['trades']:>4} {r['sl']:>4} {r['tsl']:>4} {r['rev']:>4} {r['fc']:>3} {wl:>7} {r['pf']:>7.2f} {r['mdd']:>6.1f}% {r['time']:>7.3f}s")

    # Target row
    if target:
        print(f"  {'─'*14} {'─'*16} {'─'*4} {'─'*4} {'─'*4} {'─'*4} {'─'*3} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")
        t = target
        print(f"  {'TARGET':<14} ${t['final_cap']:>14,} {t['trades']:>4} {t['sl']:>4} {t['tsl']:>4} {t['rev']:>4} {t['fc']:>3} {'':>7} {t['pf']:>7.2f} {t['mdd']:>6.1f}%")

    # Cross-validation check
    caps = [r['final_cap'] for r in results]
    max_diff = max(caps) - min(caps)
    avg_cap = sum(caps) / len(caps)
    pct_diff = max_diff / avg_cap * 100 if avg_cap != 0 else 0

    trades_list = [r['trades'] for r in results]
    trades_match = len(set(trades_list)) == 1

    print(f"\n  6엔진 교차검증:")
    print(f"    최종잔액 최대차이: ${max_diff:,.6f} ({pct_diff:.8f}%)")
    print(f"    거래수 일치: {'YES' if trades_match else 'NO'} ({set(trades_list)})")

    if max_diff < 0.01 and trades_match:
        print(f"    결과: PASS (6엔진 완전 일치)")
    else:
        print(f"    결과: FAIL (차이 발견)")

    # Target comparison
    if target:
        ref = results[0]
        cap_diff = abs(ref['final_cap'] - target['final_cap'])
        cap_pct = cap_diff / target['final_cap'] * 100 if target['final_cap'] != 0 else 0
        tr_match = ref['trades'] == target['trades']
        print(f"\n  기획서 대비:")
        print(f"    잔액 차이: ${cap_diff:,.2f} ({cap_pct:.2f}%)")
        print(f"    거래수: {'일치' if tr_match else '불일치'} (엔진={ref['trades']}, 기획서={target['trades']})")
        print(f"    SL/TSL/REV/FC: {ref['sl']}/{ref['tsl']}/{ref['rev']}/{ref['fc']} vs {target['sl']}/{target['tsl']}/{target['rev']}/{target['fc']}")

    return results


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 90)
    print("  BTC/USDT 선물 백테스트 — 6엔진 교차 검증")
    print("=" * 90)

    print("\n[1/4] 데이터 로딩 및 30분봉 리샘플링...")
    df = load_data()

    print("\n[2/4] v32.2 지표 계산 (EMA100/EMA600)...")
    c2, h2, l2, fm2, sm2, av2, rv2 = compute_indicators(df, PARAMS_V32_2)

    print("[3/4] v32.3 지표 계산 (EMA75/SMA750)...")
    c3, h3, l3, fm3, sm3, av3, rv3 = compute_indicators(df, PARAMS_V32_3)

    print("\n[4/4] 6엔진 교차 검증 실행...")
    r2 = run_cross_validation(c2, h2, l2, fm2, sm2, av2, rv2, PARAMS_V32_2, TARGET_V32_2,
                              "v32.2: EMA(100)/EMA(600) Tight-SL Trend System")
    r3 = run_cross_validation(c3, h3, l3, fm3, sm3, av3, rv3, PARAMS_V32_3, TARGET_V32_3,
                              "v32.3: EMA(75)/SMA(750) Low-MDD Trend System")

    print(f"\n{'='*90}")
    print("  교차 검증 완료")
    print(f"{'='*90}")
