"""
BTC/USDT v14.4 Enhanced AI Backtest
코드봇 vs GPT-5.4 vs Claude Sonnet 4.6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[변경사항]
- ADX >= 20 에서 AI 질문 (기존 35)
- EMA 3/7/21/100/200 + ADX + RSI + MACD + BB 각 100봉 제공
- TSL 접근시 AI에게 청산/유지 질문
- 2023년~ 실행

Usage:
  python ai_backtest_enhanced.py           # 2023~현재
  python ai_backtest_enhanced.py --dry-run # 시뮬레이션 (API 비용 없음)
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import argparse
import traceback
from datetime import datetime
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
np.seterr(all='ignore')

load_dotenv("D:/filesystem/futures/btc_V1/.env")
load_dotenv()

# ================================================================
# 설정
# ================================================================
IC          = 3000.0
FEE         = 0.0004
LEVERAGE    = 10
MARGIN_PCT  = 0.25
SL_PCT      = 7.0
TSL_ACT     = 6.0
TSL_WIDTH   = 3.0
ADX_THRESH_CODE = 35     # 코드봇 진입 기준
ADX_THRESH_AI   = 20     # AI 질문 기준 (완화)
RSI_LOW     = 30
RSI_HIGH    = 65
ML_LIMIT    = -0.20
WARMUP      = 300
FL_PCT      = 100.0 / LEVERAGE
TSL_PROXIMITY = 1.5      # TSL에 1.5% 이내 접근시 AI에게 질문
CROSS_COOLDOWN = 6       # 크로스 간 최소 간격 (6봉 = 3시간)

DATA_PATH = "D:/filesystem/futures/btc_V1/test1/btc_usdt_5m_merged.csv"
HISTORY_BARS = 100        # AI에게 제공할 히스토리 봉 수

# ================================================================
# 지표 계산 (Wilder Smoothing)
# ================================================================
def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def rsi_calc(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = (-d).where(d < 0, 0.0)
    ag = g.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    al = l.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, 1e-10))

def adx_calc(h, l, c, p=14):
    pdm = h.diff(); mdm = -l.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    pdi = 100 * (pdm.ewm(alpha=1/p, min_periods=p, adjust=False).mean() / atr)
    mdi = 100 * (mdm.ewm(alpha=1/p, min_periods=p, adjust=False).mean() / atr)
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-10)
    return dx.ewm(alpha=1/p, min_periods=p, adjust=False).mean(), pdi, mdi

def macd_calc(c, fast=12, slow=26, signal=9):
    ema_fast = c.ewm(span=fast, adjust=False).mean()
    ema_slow = c.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bb_calc(c, period=20, std_dev=2):
    sma = c.rolling(period).mean()
    std = c.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

# ================================================================
# 데이터 로드 & 지표 계산
# ================================================================
def load_and_prepare():
    print("[1/4] Data loading...")
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    print(f"  5m: {len(df):,} rows")

    df30 = df.resample('30min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    print(f"  30m: {len(df30):,} rows ({df30.index[0]} ~ {df30.index[-1]})")

    print("[2/4] Indicators...")
    c = df30['close']; h = df30['high']; l = df30['low']

    # EMA 5종
    df30['ema3']   = ema(c, 3)
    df30['ema7']   = ema(c, 7)
    df30['ema21']  = ema(c, 21)
    df30['ema100'] = ema(c, 100)
    df30['ema200'] = ema(c, 200)

    # ADX + DI
    adx_s, pdi_s, mdi_s = adx_calc(h, l, c, 14)
    df30['adx'] = adx_s
    df30['plus_di'] = pdi_s
    df30['minus_di'] = mdi_s

    # RSI
    df30['rsi'] = rsi_calc(c, 14)

    # MACD
    macd_l, sig_l, hist_l = macd_calc(c)
    df30['macd'] = macd_l
    df30['macd_signal'] = sig_l
    df30['macd_hist'] = hist_l

    # Bollinger Bands
    bb_u, bb_m, bb_l = bb_calc(c)
    df30['bb_upper'] = bb_u
    df30['bb_middle'] = bb_m
    df30['bb_lower'] = bb_l

    # Cross detection
    bull = df30['ema3'] > df30['ema200']
    prev_bull = bull.shift(1).fillna(False)
    df30['cross_up'] = bull & ~prev_bull
    df30['cross_dn'] = ~bull & prev_bull

    print(f"  Indicators: EMA(3/7/21/100/200), ADX, RSI, MACD, BB")
    return df30


# ================================================================
# AI Client
# ================================================================
def init_ai(dry_run=False):
    if dry_run:
        return None, None
    gpt = claude = None
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        gpt = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return gpt, claude


SYSTEM_PROMPT = """You are a 10-year veteran BTC/USDT futures quant trader.

[Expertise]
- Binance Futures BTC/USDT specialist
- EMA cross + trend following expert
- ADX/RSI/MACD/Bollinger Bands analysis master
- Risk management priority

[Decision Rules]
- Golden Cross + ADX>=35 + RSI 30~65 => strong LONG signal
- Dead Cross + ADX>=35 + RSI 30~65 => strong SHORT signal
- ADX 20~34 => use your judgment with MACD, BB, and candle patterns
- ADX < 20 => HOLD (no trend)
- RSI out of 30~65 => HOLD
- Monthly loss > -20% => HOLD
- Same direction position already held => evaluate if re-entry is beneficial
- If doubtful => HOLD

[For TSL proximity queries]
- Evaluate if the trend is exhausting or continuing
- Check MACD histogram direction, BB position, RSI divergence
- If trend is weakening => let TSL close (respond CLOSE)
- If trend is strong => continue holding (respond HOLD)

[Response Format]
ONLY output JSON. No other text.
{"action": "LONG/SHORT/CLOSE/HOLD", "reason": "one line reason in Korean"}"""


def ask_gpt(client, prompt, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model="gpt-5.4",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            wait = min(30, 5 * (attempt + 1))
            print(f"    [GPT retry {attempt+1}/{retries}] {type(e).__name__} -> {wait}s")
            if attempt < retries - 1:
                time.sleep(wait)
    return None


def ask_claude(client, prompt, retries=3):
    for attempt in range(retries):
        try:
            r = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            text = r.content[0].text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            return None
        except Exception as e:
            wait = min(60, 10 * (attempt + 1))
            print(f"    [Claude retry {attempt+1}/{retries}] {type(e).__name__} -> {wait}s")
            if attempt < retries - 1:
                time.sleep(wait)
    return None


def simulate_ai(event_type, pos_dir, signal, adx_val, rsi_val, macd_hist=0,
                plus_di=0, minus_di=0, tsl_proximity=False):
    """Dry-run: 실제 AI처럼 보수적으로 판단하는 시뮬레이션

    실제 AI는 100봉 컨텍스트를 보고 종합 판단합니다.
    이 시뮬레이션은 그 판단 패턴을 근사합니다.
    """
    if tsl_proximity:
        if adx_val < 25:
            return {"action": "CLOSE", "reason": "ADX 약화, 추세 소진 -> TSL 청산 허용"}
        elif adx_val < 35:
            return {"action": "HOLD", "reason": f"ADX {adx_val:.0f} 보통, 추세 잔존 -> 유지"}
        else:
            return {"action": "HOLD", "reason": f"ADX {adx_val:.0f} 강함, 추세 지속 -> 유지"}

    # 1) 동일방향 보유시 항상 스킵
    if pos_dir == signal:
        return {"action": "HOLD", "reason": f"이미 {signal} 보유 중, 재진입 불필요"}

    # 2) RSI 범위 초과 -> HOLD
    if rsi_val < RSI_LOW or rsi_val > RSI_HIGH:
        return {"action": "HOLD", "reason": f"RSI {rsi_val:.0f} 범위 초과"}

    # 3) ADX >= 35: 코드봇과 동일 -> 진입
    if adx_val >= 35:
        return {"action": signal, "reason": f"ADX {adx_val:.0f}>=35 + RSI {rsi_val:.0f}, 강한 추세 확인"}

    # 4) ADX 30~34: MACD 방향 + DI 크로스 일치시만 진입
    if adx_val >= 30:
        macd_ok = (signal == "LONG" and macd_hist > 0) or (signal == "SHORT" and macd_hist < 0)
        di_ok = (signal == "LONG" and plus_di > minus_di) or (signal == "SHORT" and minus_di > plus_di)
        if macd_ok and di_ok:
            return {"action": signal, "reason": f"ADX {adx_val:.0f}+MACD+DI 일치, {signal} 진입"}
        return {"action": "HOLD", "reason": f"ADX {adx_val:.0f}, MACD/DI 불일치 -> HOLD"}

    # 5) ADX 20~29: 거의 항상 HOLD (추세 약함)
    return {"action": "HOLD", "reason": f"ADX {adx_val:.0f}<30, 추세 불충분 -> HOLD"}


def build_indicator_table(df30, idx, bars=HISTORY_BARS):
    """100봉 지표 히스토리 테이블 생성 (트레이딩뷰 스타일)"""
    start = max(0, idx - bars + 1)
    chunk = df30.iloc[start:idx+1]

    # 핵심 지표만 선별 (토큰 절약)
    cols = ['open', 'high', 'low', 'close', 'volume',
            'ema3', 'ema7', 'ema21', 'ema100', 'ema200',
            'adx', 'plus_di', 'minus_di', 'rsi',
            'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower']

    table = chunk[cols].copy()
    table.index = table.index.strftime('%Y-%m-%d %H:%M')

    # 숫자 포맷팅 (토큰 절약)
    for c in ['open', 'high', 'low', 'close', 'ema3', 'ema7', 'ema21', 'ema100', 'ema200',
              'bb_upper', 'bb_middle', 'bb_lower']:
        if c in table.columns:
            table[c] = table[c].round(1)
    for c in ['adx', 'plus_di', 'minus_di', 'rsi']:
        if c in table.columns:
            table[c] = table[c].round(1)
    for c in ['macd', 'macd_signal', 'macd_hist']:
        if c in table.columns:
            table[c] = table[c].round(2)
    table['volume'] = table['volume'].round(0).astype(int)

    return table.to_string()


def build_cross_prompt(df30, idx, cross_type, balance, pos_str, monthly_pnl, tsl_info=""):
    """크로스 발생시 AI 프롬프트 (100봉 히스토리 포함)"""
    row = df30.iloc[idx]
    ts = df30.index[idx]

    indicator_table = build_indicator_table(df30, idx)

    prompt = f"""=== BTC/USDT 30분봉 크로스 발생 ===
시간: {ts}
이벤트: {cross_type}

=== 현재 지표 ===
가격: ${row['close']:,.1f}
EMA(3): {row['ema3']:.1f} | EMA(7): {row['ema7']:.1f} | EMA(21): {row['ema21']:.1f} | EMA(100): {row['ema100']:.1f} | EMA(200): {row['ema200']:.1f}
ADX(14): {row['adx']:.1f} | +DI: {row['plus_di']:.1f} | -DI: {row['minus_di']:.1f}
RSI(14): {row['rsi']:.1f}
MACD: {row['macd']:.2f} | Signal: {row['macd_signal']:.2f} | Hist: {row['macd_hist']:.2f}
BB: Upper={row['bb_upper']:.1f} | Mid={row['bb_middle']:.1f} | Lower={row['bb_lower']:.1f}

=== 계좌 상태 ===
잔액: ${balance:,.0f}
포지션: {pos_str}
이번달 손익: {monthly_pnl:+.1f}%
{tsl_info}

=== 최근 {HISTORY_BARS}봉 히스토리 (30분봉) ===
{indicator_table}

=== 규칙 ===
골든크로스 + ADX>=35 + RSI 30~65 -> LONG
데드크로스 + ADX>=35 + RSI 30~65 -> SHORT
ADX 20~34 -> 보조지표(MACD, BB, RSI 추세) 종합 판단
ADX < 20 -> HOLD
이번달 -20% 초과 -> HOLD
동일방향 포지션 보유중 -> 재진입 필요성 평가

JSON만 답하세요:
{{"action": "LONG/SHORT/CLOSE/HOLD", "reason": "판단 근거 한 줄"}}"""
    return prompt


def build_tsl_prompt(df30, idx, pos_dir, entry_price, peak_price, tsl_level,
                     current_roi, balance, monthly_pnl):
    """TSL 접근시 AI 프롬프트"""
    row = df30.iloc[idx]
    ts = df30.index[idx]

    indicator_table = build_indicator_table(df30, idx)

    dir_str = "LONG" if pos_dir == 1 else "SHORT"
    if pos_dir == 1:
        dist_pct = (row['close'] - tsl_level) / row['close'] * 100
    else:
        dist_pct = (tsl_level - row['close']) / row['close'] * 100

    prompt = f"""=== BTC/USDT 30분봉 TSL 접근 알림 ===
시간: {ts}
이벤트: 트레일링 스톱 {dist_pct:.1f}% 이내 접근

=== 포지션 상태 ===
방향: {dir_str}
진입가: ${entry_price:,.1f}
현재가: ${row['close']:,.1f}
최고/최저가: ${peak_price:,.1f}
TSL 레벨: ${tsl_level:,.1f} (현재가와 {dist_pct:.1f}% 거리)
현재 ROI: {current_roi:+.1f}%

=== 현재 지표 ===
EMA(3): {row['ema3']:.1f} | EMA(7): {row['ema7']:.1f} | EMA(21): {row['ema21']:.1f} | EMA(100): {row['ema100']:.1f} | EMA(200): {row['ema200']:.1f}
ADX(14): {row['adx']:.1f} | +DI: {row['plus_di']:.1f} | -DI: {row['minus_di']:.1f}
RSI(14): {row['rsi']:.1f}
MACD: {row['macd']:.2f} | Signal: {row['macd_signal']:.2f} | Hist: {row['macd_hist']:.2f}
BB: Upper={row['bb_upper']:.1f} | Mid={row['bb_middle']:.1f} | Lower={row['bb_lower']:.1f}

=== 계좌 ===
잔액: ${balance:,.0f} | 이번달 손익: {monthly_pnl:+.1f}%

=== 최근 {HISTORY_BARS}봉 히스토리 ===
{indicator_table}

=== 판단 요청 ===
TSL에 가격이 접근하고 있습니다.
- CLOSE: TSL 청산을 진행 (수익 확보)
- HOLD: 포지션 유지 (추세가 계속될 것으로 판단)

MACD 히스토그램 방향, BB 위치, RSI 추세, ADX 강도를 종합 판단하세요.

JSON만 답하세요:
{{"action": "CLOSE/HOLD", "reason": "판단 근거 한 줄"}}"""
    return prompt


# ================================================================
# Bot class
# ================================================================
class Bot:
    def __init__(self, name):
        self.name = name
        self.bal = IC
        self.pos = 0          # 1=long, -1=short, 0=none
        self.epx = 0.0
        self.psz = 0.0
        self.margin = 0.0
        self.slp = 0.0
        self.tsl_on = False
        self.peak = 0.0
        self.trough = 999999.0
        self.tsl_level = 0.0
        self.m_start = IC
        self.cur_m = ""
        self.sl_c = 0; self.tsl_c = 0; self.rev_c = 0; self.fl_c = 0
        self.trades = []
        self.monthly = {}
        self.pk_bal = IC
        self.mdd = 0.0
        self.yr_bal = {}
        self.tsl_ai_asked = False  # TSL 접근시 AI 질문 여부
        self.tsl_ai_hold = False   # AI가 HOLD 응답 (TSL 무시)

    def monthly_pnl_pct(self):
        if self.m_start <= 0: return 0.0
        return (self.bal - self.m_start) / self.m_start * 100

    def update_month(self, mk):
        if mk != self.cur_m:
            if self.cur_m and self.cur_m in self.monthly:
                self.monthly[self.cur_m]['eq'] = self.bal
            self.cur_m = mk
            self.m_start = self.bal
            if mk not in self.monthly:
                self.monthly[mk] = {'pnl': 0, 'ent': 0, 'w': 0, 'l': 0,
                                    'sl': 0, 'tsl': 0, 'rev': 0, 'fl': 0,
                                    'eq_s': self.bal, 'eq': self.bal}

    def update_dd(self):
        self.pk_bal = max(self.pk_bal, self.bal)
        if self.pk_bal > 0:
            self.mdd = max(self.mdd, (self.pk_bal - self.bal) / self.pk_bal)

    def enter(self, direction, price, ts):
        if self.bal <= 0: return False
        margin = self.bal * MARGIN_PCT
        if margin < 1: return False
        if self.m_start > 0 and (self.bal - self.m_start) / self.m_start <= ML_LIMIT:
            return False

        self.pos = direction
        self.epx = price
        self.margin = margin
        self.psz = margin * LEVERAGE
        self.bal -= self.psz * FEE
        self.tsl_on = False
        self.tsl_ai_asked = False
        self.tsl_ai_hold = False
        self.peak = price
        self.trough = price
        self.tsl_level = 0.0
        self.slp = price * (1 - SL_PCT / 100) if direction == 1 else price * (1 + SL_PCT / 100)

        mk = self.cur_m
        if mk in self.monthly: self.monthly[mk]['ent'] += 1
        return True

    def close(self, price, reason, ts):
        if self.pos == 0: return 0.0
        if reason == 'fl':
            pnl = -(self.margin + self.psz * FEE)
        else:
            raw = (price - self.epx) / self.epx * self.psz * self.pos
            pnl = raw - self.psz * FEE

        self.bal += pnl
        if self.bal < 0: self.bal = 0

        mk = self.cur_m
        if mk in self.monthly:
            self.monthly[mk]['pnl'] += pnl
            if pnl > 0: self.monthly[mk]['w'] += 1
            else: self.monthly[mk]['l'] += 1
            if reason in ('sl', 'tsl', 'rev', 'fl'):
                self.monthly[mk][reason] += 1

        if reason == 'sl': self.sl_c += 1
        elif reason == 'tsl': self.tsl_c += 1
        elif reason == 'rev': self.rev_c += 1
        elif reason == 'fl': self.fl_c += 1

        exit_price = price if reason != 'fl' else (
            self.epx * (1 - FL_PCT / 100) if self.pos == 1 else
            self.epx * (1 + FL_PCT / 100))

        self.trades.append({
            'ts': str(ts), 'dir': 'LONG' if self.pos == 1 else 'SHORT',
            'entry': self.epx, 'exit': exit_price,
            'pnl': round(pnl, 2), 'reason': reason, 'bal': round(self.bal, 2)
        })

        self.pos = 0; self.epx = 0; self.psz = 0; self.margin = 0
        self.tsl_on = False; self.tsl_ai_asked = False; self.tsl_ai_hold = False
        self.update_dd()
        return pnl

    def check_exit(self, hi, lo, cl):
        if self.pos == 0: return None

        # 1) FL
        if self.pos == 1:
            worst = (lo - self.epx) / self.epx * 100
        else:
            worst = (self.epx - hi) / self.epx * 100
        if worst <= -FL_PCT:
            return ('fl', None)

        # 2) SL
        if not self.tsl_on:
            if self.pos == 1 and lo <= self.slp:
                return ('sl', self.slp)
            if self.pos == -1 and hi >= self.slp:
                return ('sl', self.slp)

        # 3) TSL activation
        if self.pos == 1:
            best = (hi - self.epx) / self.epx * 100
        else:
            best = (self.epx - lo) / self.epx * 100
        if best >= TSL_ACT:
            self.tsl_on = True

        # 4) TSL tracking
        if self.tsl_on:
            if self.pos == 1:
                self.peak = max(self.peak, hi)
                self.tsl_level = self.peak * (1 - TSL_WIDTH / 100)
                self.slp = max(self.slp, self.tsl_level)
                if cl <= self.tsl_level:
                    return ('tsl', cl)
            else:
                self.trough = min(self.trough, lo)
                self.tsl_level = self.trough * (1 + TSL_WIDTH / 100)
                self.slp = min(self.slp, self.tsl_level)
                if cl >= self.tsl_level:
                    return ('tsl', cl)

        return None

    def is_tsl_near(self, close):
        """TSL에 가격이 접근했는지 확인"""
        if not self.tsl_on or self.pos == 0:
            return False
        if self.pos == 1:
            dist = (close - self.tsl_level) / close * 100
            return 0 < dist <= TSL_PROXIMITY
        else:
            dist = (self.tsl_level - close) / close * 100
            return 0 < dist <= TSL_PROXIMITY

    def current_roi(self, close):
        if self.pos == 0: return 0.0
        if self.pos == 1:
            return (close - self.epx) / self.epx * LEVERAGE * 100
        else:
            return (self.epx - close) / self.epx * LEVERAGE * 100

    def pos_str(self):
        if self.pos == 0: return "None"
        d = "LONG" if self.pos == 1 else "SHORT"
        return f"{d} @ ${self.epx:,.0f}"

    def stats(self):
        n = len(self.trades)
        w = sum(1 for t in self.trades if t['pnl'] > 0)
        l = sum(1 for t in self.trades if t['pnl'] <= 0)
        gp = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gl = abs(sum(t['pnl'] for t in self.trades if t['pnl'] <= 0))
        avg_w = np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0]) if w > 0 else 0
        avg_l = np.mean([t['pnl'] for t in self.trades if t['pnl'] <= 0]) if l > 0 else 0
        return {
            'bal': self.bal,
            'ret': (self.bal - IC) / IC * 100,
            'pf': gp / max(gl, 0.001),
            'mdd': self.mdd * 100,
            'n': n, 'w': w, 'l': l,
            'wr': w / max(n, 1) * 100,
            'avg_w': avg_w, 'avg_l': avg_l,
            'sl': self.sl_c, 'tsl': self.tsl_c, 'rev': self.rev_c, 'fl': self.fl_c
        }


# ================================================================
# Main backtest
# ================================================================
def run(dry_run=False):
    print("=" * 70)
    print("  BTC/USDT v14.4 Enhanced AI Backtest")
    print(f"  Period: 2023~ | ADX trigger: >= {ADX_THRESH_AI}")
    print(f"  Dry-run: {dry_run}")
    print("=" * 70)

    df30 = load_and_prepare()

    # AI init
    print("[3/4] AI clients...")
    gpt_cli, cld_cli = init_ai(dry_run)
    if dry_run:
        print("  DRY-RUN mode: simulated AI responses")
    else:
        print(f"  GPT-5.4: {'OK' if gpt_cli else 'NO KEY'}")
        print(f"  Claude Sonnet 4.6: {'OK' if cld_cli else 'NO KEY'}")

    # Index range
    ts_arr = df30.index
    n = len(df30)
    start_i = max(WARMUP, np.searchsorted(ts_arr, pd.Timestamp('2023-01-01')))
    end_i = n
    print(f"  Range: {ts_arr[start_i]} ~ {ts_arr[end_i-1]} ({end_i - start_i:,} candles)")

    # Numpy arrays
    cl = df30['close'].values; hi = df30['high'].values; lo = df30['low'].values
    adx_v = df30['adx'].values
    rsi_v = df30['rsi'].values
    gx = df30['cross_up'].values; dx = df30['cross_dn'].values

    # Bots
    code = Bot("CodeBot"); gpt_bot = Bot("GPT-5.4"); cla_bot = Bot("Claude")
    bots = [code, gpt_bot, cla_bot]

    cross_log = []
    tsl_log = []
    api_calls = 0
    cross_cnt = 0
    tsl_query_cnt = 0
    last_cross_i = -CROSS_COOLDOWN - 1   # AI 쿨다운 추적
    last_cross_dir = 0                    # 마지막 크로스 방향 (노이즈 필터)
    macd_hist_v = df30['macd_hist'].values

    print(f"\n[4/4] Running backtest...\n")

    for i in range(start_i, end_i):
        t = ts_arr[i]
        h_ = hi[i]; l_ = lo[i]; c_ = cl[i]
        mk = f"{t.year}-{t.month:02d}"

        for b in bots: b.update_month(mk)

        # === SL/TSL/FL auto check ===
        for b in bots:
            # AI봇의 TSL: AI가 HOLD 응답한 경우 TSL 청산 스킵
            if b.name != "CodeBot" and b.tsl_ai_hold:
                # TSL을 무시하되, SL/FL은 체크
                ex = b.check_exit(h_, l_, c_)
                if ex and ex[0] in ('fl', 'sl'):
                    b.close(ex[1] if ex[1] else c_, ex[0], t)
                elif ex and ex[0] == 'tsl':
                    pass  # AI가 HOLD -> TSL 무시
                    # 하지만 다음 캔들에서 다시 체크 (1회만 무시)
                    b.tsl_ai_hold = False
                continue

            ex = b.check_exit(h_, l_, c_)
            if ex:
                reason, ex_px = ex
                if reason == 'fl':
                    b.close(c_, 'fl', t)
                elif reason == 'sl':
                    b.close(ex_px, 'sl', t)
                elif reason == 'tsl':
                    b.close(ex_px, 'tsl', t)

        # === TSL proximity check for AI bots ===
        for b in [gpt_bot, cla_bot]:
            if b.is_tsl_near(c_) and not b.tsl_ai_asked:
                b.tsl_ai_asked = True
                tsl_query_cnt += 1

                pos_dir_str = "LONG" if b.pos == 1 else "SHORT"
                peak_p = b.peak if b.pos == 1 else b.trough
                roi = b.current_roi(c_)

                if dry_run:
                    result = simulate_ai("tsl", b.pos, None, adx_v[i], rsi_v[i], tsl_proximity=True)
                elif b.name == "GPT-5.4" and gpt_cli:
                    prompt = build_tsl_prompt(df30, i, b.pos, b.epx, peak_p, b.tsl_level, roi, b.bal, b.monthly_pnl_pct())
                    result = ask_gpt(gpt_cli, prompt)
                    api_calls += 1
                elif b.name == "Claude" and cld_cli:
                    prompt = build_tsl_prompt(df30, i, b.pos, b.epx, peak_p, b.tsl_level, roi, b.bal, b.monthly_pnl_pct())
                    result = ask_claude(cld_cli, prompt)
                    api_calls += 1
                else:
                    result = None

                if result:
                    action = result.get('action', 'HOLD')
                    reason = result.get('reason', '')
                    if action == 'HOLD':
                        b.tsl_ai_hold = True
                        print(f"  [TSL-AI] {t} | {b.name} | {pos_dir_str} | ROI={roi:+.1f}% | AI=HOLD | {reason[:40]}")
                    else:
                        print(f"  [TSL-AI] {t} | {b.name} | {pos_dir_str} | ROI={roi:+.1f}% | AI=CLOSE | {reason[:40]}")

                    tsl_log.append({
                        'timestamp': str(t), 'bot': b.name, 'direction': pos_dir_str,
                        'roi': round(roi, 1), 'tsl_level': round(b.tsl_level, 1),
                        'ai_action': action, 'ai_reason': reason
                    })

        # === Cross detection ===
        is_gx = bool(gx[i]); is_dx = bool(dx[i])
        if not (is_gx or is_dx):
            for b in bots: b.yr_bal[t.year] = b.bal
            continue

        a_v = adx_v[i]; r_v = rsi_v[i]

        a_v = adx_v[i]; r_v = rsi_v[i]
        new_dir = 1 if is_gx else -1

        # 동일방향 반복 크로스 필터: 방향이 바뀔 때만 의미있는 이벤트
        if new_dir == last_cross_dir:
            for b in bots: b.yr_bal[t.year] = b.bal
            continue
        last_cross_dir = new_dir

        # 코드봇은 ADX >= 35 + RSI 30~65 조건 (항상 처리)
        code_trigger = a_v >= ADX_THRESH_CODE and RSI_LOW <= r_v <= RSI_HIGH

        # AI 트리거: ADX >= 20
        ai_trigger = a_v >= ADX_THRESH_AI

        if not (ai_trigger or code_trigger):
            for b in bots: b.yr_bal[t.year] = b.bal
            continue

        cross_cnt += 1
        cross_type = "Golden Cross" if is_gx else "Dead Cross"
        signal_str = "LONG" if new_dir == 1 else "SHORT"

        # === CodeBot: REV + entry (ADX>=35 + RSI 30~65 only) ===
        code_action = "HOLD"
        if code_trigger:
            if code.pos != 0 and code.pos != new_dir:
                code.close(c_, 'rev', t)
            if code.enter(new_dir, c_, t):
                code_action = signal_str

        # === GPT judgment ===
        gpt_action = "HOLD"; gpt_reason = ""
        mh = macd_hist_v[i]
        if ai_trigger:
            if dry_run:
                pos_dir = gpt_bot.pos
                result = simulate_ai("cross", pos_dir, signal_str, a_v, r_v, macd_hist=mh,
                                     plus_di=df30.iloc[i]['plus_di'], minus_di=df30.iloc[i]['minus_di'])
            elif gpt_cli:
                prompt = build_cross_prompt(df30, i, cross_type, gpt_bot.bal,
                                            gpt_bot.pos_str(), gpt_bot.monthly_pnl_pct())
                result = ask_gpt(gpt_cli, prompt)
                api_calls += 1
            else:
                result = None

            if result:
                gpt_action = result.get('action', 'HOLD')
                gpt_reason = result.get('reason', '')
            else:
                gpt_action = signal_str if code_trigger else "HOLD"
                gpt_reason = "API fallback"

        # === Claude judgment ===
        cla_action = "HOLD"; cla_reason = ""
        if ai_trigger:
            if dry_run:
                pos_dir = cla_bot.pos
                result = simulate_ai("cross", pos_dir, signal_str, a_v, r_v, macd_hist=mh,
                                     plus_di=df30.iloc[i]['plus_di'], minus_di=df30.iloc[i]['minus_di'])
            elif cld_cli:
                prompt = build_cross_prompt(df30, i, cross_type, cla_bot.bal,
                                            cla_bot.pos_str(), cla_bot.monthly_pnl_pct())
                result = ask_claude(cld_cli, prompt)
                api_calls += 1
            else:
                result = None

            if result:
                cla_action = result.get('action', 'HOLD')
                cla_reason = result.get('reason', '')
            else:
                cla_action = signal_str if code_trigger else "HOLD"
                cla_reason = "API fallback"

        # === AI bot execution (REV close + entry only on AI decision) ===
        gpt_entered = False
        if gpt_action in ("LONG", "SHORT"):
            ai_dir = 1 if gpt_action == "LONG" else -1
            if gpt_bot.pos != 0 and gpt_bot.pos != ai_dir:
                gpt_bot.close(c_, 'rev', t)
            gpt_entered = gpt_bot.enter(ai_dir, c_, t)

        cla_entered = False
        if cla_action in ("LONG", "SHORT"):
            ai_dir = 1 if cla_action == "LONG" else -1
            if cla_bot.pos != 0 and cla_bot.pos != ai_dir:
                cla_bot.close(c_, 'rev', t)
            cla_entered = cla_bot.enter(ai_dir, c_, t)

        # === Log ===
        cross_log.append({
            'timestamp': str(t), 'cross': cross_type,
            'price': round(c_, 1), 'adx': round(a_v, 1), 'rsi': round(r_v, 1),
            'code_action': code_action,
            'gpt_action': gpt_action, 'gpt_reason': gpt_reason,
            'claude_action': cla_action, 'claude_reason': cla_reason,
            'code_bal': round(code.bal, 0),
            'gpt_bal': round(gpt_bot.bal, 0),
            'claude_bal': round(cla_bot.bal, 0)
        })

        # Print
        adx_marker = "*" if a_v >= ADX_THRESH_CODE else " "
        print(f"[{cross_cnt:>3}] {str(t)[:16]} | {cross_type:>12} | ADX={a_v:5.1f}{adx_marker} RSI={r_v:5.1f} | "
              f"Code={code_action:>5} GPT={gpt_action:>5} Claude={cla_action:>6} | "
              f"${code.bal:>8,.0f} ${gpt_bot.bal:>8,.0f} ${cla_bot.bal:>8,.0f}")

        for b in bots: b.yr_bal[t.year] = b.bal

    # === Close remaining positions ===
    last_px = cl[end_i - 1]; last_t = ts_arr[end_i - 1]
    for b in bots:
        if b.pos != 0:
            b.close(last_px, 'end', last_t)
        if b.cur_m in b.monthly:
            b.monthly[b.cur_m]['eq'] = b.bal
        b.yr_bal[last_t.year] = b.bal

    # === Report ===
    print_report(code, gpt_bot, cla_bot, cross_log, tsl_log, api_calls, cross_cnt, tsl_query_cnt)
    save_results(code, gpt_bot, cla_bot, cross_log, tsl_log)


# ================================================================
# Report
# ================================================================
def print_report(code, gpt_bot, cla_bot, cross_log, tsl_log, api_calls, cross_cnt, tsl_query_cnt):
    cs = code.stats(); gs = gpt_bot.stats(); ls = cla_bot.stats()

    print(f"\n\n{'='*75}")
    print(f"  ENHANCED AI BACKTEST REPORT (2023~)")
    print(f"{'='*75}")
    print(f"  {'':>16} {'CodeBot':>14} {'GPT-5.4':>14} {'Claude':>14}")
    print(f"  {'-'*60}")
    print(f"  {'Balance':>16} ${cs['bal']:>12,.0f} ${gs['bal']:>12,.0f} ${ls['bal']:>12,.0f}")
    print(f"  {'Return':>16} {cs['ret']:>+12.1f}% {gs['ret']:>+12.1f}% {ls['ret']:>+12.1f}%")
    print(f"  {'PF':>16} {cs['pf']:>14.2f} {gs['pf']:>14.2f} {ls['pf']:>14.2f}")
    print(f"  {'MDD':>16} {cs['mdd']:>13.1f}% {gs['mdd']:>13.1f}% {ls['mdd']:>13.1f}%")
    print(f"  {'Trades':>16} {cs['n']:>14} {gs['n']:>14} {ls['n']:>14}")
    print(f"  {'Win Rate':>16} {cs['wr']:>13.1f}% {gs['wr']:>13.1f}% {ls['wr']:>13.1f}%")
    print(f"  {'Avg Win':>16} ${cs['avg_w']:>12,.0f} ${gs['avg_w']:>12,.0f} ${ls['avg_w']:>12,.0f}")
    print(f"  {'Avg Loss':>16} ${cs['avg_l']:>12,.0f} ${gs['avg_l']:>12,.0f} ${ls['avg_l']:>12,.0f}")
    print(f"  {'SL':>16} {cs['sl']:>14} {gs['sl']:>14} {ls['sl']:>14}")
    print(f"  {'TSL':>16} {cs['tsl']:>14} {gs['tsl']:>14} {ls['tsl']:>14}")
    print(f"  {'REV':>16} {cs['rev']:>14} {gs['rev']:>14} {ls['rev']:>14}")
    print(f"  {'FL':>16} {cs['fl']:>14} {gs['fl']:>14} {ls['fl']:>14}")

    # Agreement rate
    tot = len(cross_log)
    if tot > 0:
        all3 = sum(1 for x in cross_log if x['code_action'] == x['gpt_action'] == x['claude_action'])
        gpt_dif = sum(1 for x in cross_log if x['gpt_action'] != x['code_action'] and x['claude_action'] == x['code_action'])
        cla_dif = sum(1 for x in cross_log if x['claude_action'] != x['code_action'] and x['gpt_action'] == x['code_action'])
        both_dif = sum(1 for x in cross_log if x['gpt_action'] != x['code_action'] and x['claude_action'] != x['code_action'])

        print(f"\n  [Agreement Rate] ({tot} crosses)")
        print(f"  Code=GPT=Claude: {all3:>4} ({all3/tot*100:.1f}%)")
        print(f"  GPT only diff:   {gpt_dif:>4} ({gpt_dif/tot*100:.1f}%)")
        print(f"  Claude only diff:{cla_dif:>4} ({cla_dif/tot*100:.1f}%)")
        print(f"  Both diff:       {both_dif:>4} ({both_dif/tot*100:.1f}%)")

    # AI divergences
    diverged = [x for x in cross_log if x['gpt_action'] != x['code_action'] or x['claude_action'] != x['code_action']]
    if diverged:
        print(f"\n  [AI Divergences] ({len(diverged)} events)")
        print(f"  {'Date':>20} {'ADX':>5} {'RSI':>5} {'Code':>6} {'GPT':>6} {'Claude':>7} {'GPT Reason':>30} {'Claude Reason':>30}")
        print(f"  {'-'*120}")
        for d in diverged[:20]:
            print(f"  {d['timestamp'][:16]:>20} {d['adx']:>5.1f} {d['rsi']:>5.1f} {d['code_action']:>6} "
                  f"{d['gpt_action']:>6} {d['claude_action']:>7} {d['gpt_reason'][:30]:>30} {d['claude_reason'][:30]:>30}")
        if len(diverged) > 20:
            print(f"  ... +{len(diverged)-20} more")

    # TSL AI queries
    if tsl_log:
        print(f"\n  [TSL AI Queries] ({len(tsl_log)} events)")
        print(f"  {'Date':>20} {'Bot':>8} {'Dir':>5} {'ROI':>7} {'Action':>6} {'Reason':>40}")
        print(f"  {'-'*90}")
        for t in tsl_log[:20]:
            print(f"  {t['timestamp'][:16]:>20} {t['bot']:>8} {t['direction']:>5} {t['roi']:>+6.1f}% "
                  f"{t['ai_action']:>6} {t['ai_reason'][:40]:>40}")

    # Yearly comparison
    all_yrs = sorted(set(list(code.yr_bal) + list(gpt_bot.yr_bal) + list(cla_bot.yr_bal)))
    if all_yrs:
        print(f"\n  [Yearly Balance]")
        print(f"  {'Year':>6} {'CodeBot':>12} {'GPT-5.4':>12} {'Claude':>12} {'Best':>8}")
        print(f"  {'-'*55}")
        for yr in all_yrs:
            cb = code.yr_bal.get(yr, 0)
            gb = gpt_bot.yr_bal.get(yr, 0)
            lb = cla_bot.yr_bal.get(yr, 0)
            best = max(cb, gb, lb)
            winner = "Code" if best == cb else ("GPT" if best == gb else "Claude")
            print(f"  {yr:>6} ${cb:>10,.0f} ${gb:>10,.0f} ${lb:>10,.0f} {winner:>8}")

    # Monthly detail
    months = sorted(set(list(code.monthly) + list(gpt_bot.monthly) + list(cla_bot.monthly)))
    if months:
        print(f"\n  [Monthly Balance]")
        print(f"  {'Month':>7} {'Code':>10} {'GPT':>10} {'Claude':>10} {'Code PnL':>10} {'GPT PnL':>10} {'Cla PnL':>10}")
        print(f"  {'-'*70}")
        for mk in months:
            cd = code.monthly.get(mk, {}); gd = gpt_bot.monthly.get(mk, {}); ld = cla_bot.monthly.get(mk, {})
            print(f"  {mk:>7} ${cd.get('eq',0):>8,.0f} ${gd.get('eq',0):>8,.0f} ${ld.get('eq',0):>8,.0f} "
                  f"${cd.get('pnl',0):>+8,.0f} ${gd.get('pnl',0):>+8,.0f} ${ld.get('pnl',0):>+8,.0f}")

    # Final ranking
    ranking = sorted([(cs['bal'], 'CodeBot'), (gs['bal'], 'GPT-5.4'), (ls['bal'], 'Claude')], reverse=True)
    print(f"\n  {'='*55}")
    print(f"  FINAL RANKING:")
    for rank, (bal, name) in enumerate(ranking, 1):
        medal = {1: '1st', 2: '2nd', 3: '3rd'}[rank]
        print(f"    {medal}: {name:>8}  ${bal:>12,.0f}  ({(bal-IC)/IC*100:+,.1f}%)")

    best_ai = max(gs['bal'], ls['bal'])
    if best_ai > cs['bal'] * 1.05:
        verdict = 'AI > Code -> "AI bot recommended"'
    elif best_ai > cs['bal'] * 0.95:
        verdict = 'AI ~ Code -> "Code bot sufficient"'
    else:
        verdict = 'AI < Code -> "AI unnecessary"'
    print(f"\n  Verdict: {verdict}")

    print(f"\n  Stats: {cross_cnt} crosses | {tsl_query_cnt} TSL queries | {api_calls} API calls")
    print(f"{'='*75}")


def save_results(code, gpt_bot, cla_bot, cross_log, tsl_log):
    base = "D:/filesystem/futures/btc_V1/test1"

    if cross_log:
        df = pd.DataFrame(cross_log)
        path = f"{base}/enhanced_ai_trades.csv"
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"\nSaved: {path} ({len(cross_log)} rows)")

    if tsl_log:
        df = pd.DataFrame(tsl_log)
        path = f"{base}/enhanced_ai_tsl_queries.csv"
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"Saved: {path} ({len(tsl_log)} rows)")

    # Monthly
    months = sorted(set(list(code.monthly) + list(gpt_bot.monthly) + list(cla_bot.monthly)))
    rows = []
    for mk in months:
        cd = code.monthly.get(mk, {}); gd = gpt_bot.monthly.get(mk, {}); ld = cla_bot.monthly.get(mk, {})
        rows.append({
            'month': mk,
            'code_bal': round(cd.get('eq', 0)), 'code_pnl': round(cd.get('pnl', 0)),
            'code_trades': cd.get('ent', 0), 'code_sl': cd.get('sl', 0),
            'code_tsl': cd.get('tsl', 0), 'code_rev': cd.get('rev', 0),
            'gpt_bal': round(gd.get('eq', 0)), 'gpt_pnl': round(gd.get('pnl', 0)),
            'gpt_trades': gd.get('ent', 0),
            'claude_bal': round(ld.get('eq', 0)), 'claude_pnl': round(ld.get('pnl', 0)),
            'claude_trades': ld.get('ent', 0),
        })
    if rows:
        df = pd.DataFrame(rows)
        path = f"{base}/enhanced_ai_monthly.csv"
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"Saved: {path} ({len(rows)} rows)")


# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Simulate AI (no API cost)')
    args = parser.parse_args()
    run(dry_run=args.dry_run)
