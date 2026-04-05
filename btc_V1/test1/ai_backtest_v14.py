"""
BTC/USDT v14.4 AI Backtest Comparison
코드봇 vs GPT-5.4 vs Claude Sonnet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage:
  python ai_backtest_v14.py 2024     # 1단계: 2024년만
  python ai_backtest_v14.py full     # 2단계: 전체 2020~2026
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import traceback
from datetime import datetime
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
np.seterr(all='ignore')

# .env 로드 (상위 디렉토리 포함)
load_dotenv("D:/filesystem/futures/btc_V1/.env")
load_dotenv()  # 현재 디렉토리 .env도 시도

# ═══════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════
IC          = 3000.0
FEE         = 0.0004       # 0.04% per side (총 0.08%)
LEVERAGE    = 10
MARGIN_PCT  = 0.25         # 잔액의 25%
SL_PCT      = 7.0          # 손절 -7%
TSL_ACT     = 6.0          # TSL 활성화 +6%
TSL_WIDTH   = 3.0          # 트레일링 3%
ADX_THRESH  = 35
RSI_LOW     = 30
RSI_HIGH    = 65
ML_LIMIT    = -0.20        # 월간손실한도 -20%
WARMUP      = 300
FL_PCT      = 100.0 / LEVERAGE          # 강제청산 10.0%

BASE = "D:/filesystem/futures/btc_V1/test2"

# ═══════════════════════════════════════════
# 지표 (Wilder Smoothing, rolling 금지)
# ═══════════════════════════════════════════
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
    return dx.ewm(alpha=1/p, min_periods=p, adjust=False).mean()

# ═══════════════════════════════════════════
# 데이터 로드
# ═══════════════════════════════════════════
def load_data():
    dfs = []
    for i in [1, 2, 3]:
        path = f"{BASE}/btc_usdt_5m_2020_to_now_part{i}.csv"
        dfs.append(pd.read_csv(path, parse_dates=['timestamp']))
    df = pd.concat(dfs).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']].astype(float)

def resample_30m(df):
    return df.resample('30min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()

# ═══════════════════════════════════════════
# AI 클라이언트
# ═══════════════════════════════════════════
def init_ai():
    gpt = claude = None
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        gpt = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return gpt, claude

def ask_gpt(client, prompt, retries=5):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model="gpt-5.4",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",
                     "content": """당신은 10년 경력의 BTC/USDT 선물 퀀트 트레이더입니다.

[전문성]
Binance Futures BTC/USDT 전문
EMA 크로스 + 추세 추종 전략 전문가
ADX/RSI 필터링 숙련자
리스크 관리 최우선

[판단 원칙]
ADX >= 35 필수 (미달 시 무조건 HOLD)
RSI 30~65 범위 필수 (초과 시 무조건 HOLD)
이번달 -20% 초과 시 무조건 HOLD
캔들 패턴으로 추세 강도 추가 확인
의심스러우면 HOLD

[성격]
보수적이고 냉철함
숫자 기반 판단 (감정 없음)
손실 방어 최우선
규칙 절대 준수

[출력 형식]
반드시 아래 JSON만 출력. 다른 텍스트 절대 금지.
{"action": "LONG/SHORT/CLOSE/HOLD", "execute": true/false, "reason": "한 줄 이유"}"""},
                    {"role": "user", "content": prompt}
                ]
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            wait = min(30, 5 * (attempt + 1))
            print(f"    [GPT 재시도 {attempt+1}/{retries}] {type(e).__name__} → {wait}s 대기")
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                return None

def ask_claude(client, prompt, retries=5):
    for attempt in range(retries):
        try:
            r = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                temperature=0,
                system="""당신은 10년 경력의 BTC/USDT 선물 퀀트 트레이더입니다.

[전문성]
Binance Futures BTC/USDT 전문
EMA 크로스 + 추세 추종 전략 전문가
ADX/RSI 필터링 숙련자
리스크 관리 최우선

[판단 원칙]
ADX >= 35 필수 (미달 시 무조건 HOLD)
RSI 30~65 범위 필수 (초과 시 무조건 HOLD)
이번달 -20% 초과 시 무조건 HOLD
캔들 패턴으로 추세 강도 추가 확인
의심스러우면 HOLD

[성격]
보수적이고 냉철함
숫자 기반 판단 (감정 없음)
손실 방어 최우선
규칙 절대 준수

[출력 형식]
반드시 아래 JSON만 출력. 다른 텍스트 절대 금지.
{"action": "LONG/SHORT/CLOSE/HOLD", "execute": true/false, "reason": "한 줄 이유"}""",
                messages=[{"role": "user", "content": prompt}]
            )
            text = r.content[0].text
            start = text.find('{')
            end = text.rfind('}') + 1
            return json.loads(text[start:end])
        except Exception as e:
            wait = min(60, 10 * (attempt + 1))
            print(f"    [Claude 재시도 {attempt+1}/{retries}] {type(e).__name__} → {wait}s 대기")
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                return None

def build_prompt(ts, cross_type, price, ema3, ema200, adx_v, rsi_v,
                 recent_50, balance, pos_str, monthly_pnl):
    return f"""지금은 {ts} 입니다.
BTC/USDT 30분봉에서 {cross_type}이 발생했습니다.

=== 현재 지표 ===
현재 가격:  ${price:,.0f}
EMA(3):    {ema3:.2f}
EMA(200):  {ema200:.2f}
ADX(14):   {adx_v:.1f}  (기준: 35 이상)
RSI(14):   {rsi_v:.1f}  (기준: 30~65)

=== 최근 50캔들 (30분봉 OHLCV) ===
{recent_50}

=== 계좌 상태 ===
잔액:         ${balance:,.0f}
포지션:       {pos_str}
이번달 손익:  {monthly_pnl:+.1f}%

=== 반드시 지켜야 할 규칙 ===
골든크로스 + ADX>=35 + RSI 30~65 → LONG
데드크로스 + ADX>=35 + RSI 30~65 → SHORT
ADX < 35 → HOLD
RSI 범위 초과 → HOLD
이번달 -20% 초과 → HOLD
현재 LONG + 데드크로스 → CLOSE
현재 SHORT + 골든크로스 → CLOSE

아래 JSON으로만 답하세요:
{{"action": "LONG/SHORT/CLOSE/HOLD", "execute": true/false, "reason": "판단 근거 한 줄"}}
"""

# ═══════════════════════════════════════════
# 봇 클래스
# ═══════════════════════════════════════════
class Bot:
    def __init__(self, name):
        self.name = name
        self.bal = IC
        self.pos = 0          # 1=long, -1=short, 0=none
        self.epx = 0.0        # entry price
        self.psz = 0.0        # position size ($)
        self.margin = 0.0     # margin used
        self.slp = 0.0        # SL price
        self.tsl_on = False
        self.peak = 0.0
        self.trough = 999999.0
        self.m_start = IC     # month start balance
        self.cur_m = ""
        # 통계
        self.sl_c = 0; self.tsl_c = 0; self.rev_c = 0; self.fl_c = 0
        self.trades = []
        self.monthly = {}
        self.pk_bal = IC
        self.mdd = 0.0
        self.yr_bal = {}

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
        # 월간손실한도
        if self.m_start > 0 and (self.bal - self.m_start) / self.m_start <= ML_LIMIT:
            return False

        self.pos = direction
        self.epx = price
        self.margin = margin
        self.psz = margin * LEVERAGE
        self.bal -= self.psz * FEE          # 진입 수수료
        self.tsl_on = False
        self.peak = price
        self.trough = price
        self.slp = price * (1 - SL_PCT / 100) if direction == 1 else price * (1 + SL_PCT / 100)

        mk = self.cur_m
        if mk in self.monthly: self.monthly[mk]['ent'] += 1
        return True

    def close(self, price, reason, ts):
        """포지션 청산. reason: sl/tsl/rev/fl/end"""
        if self.pos == 0: return 0.0

        if reason == 'fl':
            # 강제청산: 마진 전액 손실 + 청산 수수료
            pnl = -(self.margin + self.psz * FEE)
        else:
            raw = (price - self.epx) / self.epx * self.psz * self.pos
            pnl = raw - self.psz * FEE      # 청산 수수료

        self.bal += pnl
        if self.bal < 0: self.bal = 0

        mk = self.cur_m
        if mk in self.monthly:
            self.monthly[mk]['pnl'] += pnl
            if pnl > 0: self.monthly[mk]['w'] += 1
            else:        self.monthly[mk]['l'] += 1
            if reason in ('sl', 'tsl', 'rev', 'fl'):
                self.monthly[mk][reason] += 1

        if   reason == 'sl':  self.sl_c  += 1
        elif reason == 'tsl': self.tsl_c += 1
        elif reason == 'rev': self.rev_c += 1
        elif reason == 'fl':  self.fl_c  += 1

        exit_price = price if reason != 'fl' else (
            self.epx * (1 - FL_PCT / 100) if self.pos == 1 else
            self.epx * (1 + FL_PCT / 100))

        self.trades.append({
            'ts': str(ts), 'dir': 'LONG' if self.pos == 1 else 'SHORT',
            'entry': self.epx, 'exit': exit_price,
            'pnl': round(pnl, 2), 'reason': reason, 'bal': round(self.bal, 2)
        })

        self.pos = 0; self.epx = 0; self.psz = 0; self.margin = 0
        self.update_dd()
        return pnl

    def check_exit(self, hi, lo, cl):
        """SL/TSL/FL 확인. 반환: (reason, exit_price) or None"""
        if self.pos == 0: return None

        # 1) 강제청산 (격리마진)
        if self.pos == 1:
            worst_pct = (lo - self.epx) / self.epx * 100
        else:
            worst_pct = (self.epx - hi) / self.epx * 100
        if worst_pct <= -FL_PCT:
            return ('fl', None)

        # 2) SL (캔들 high/low 기준)
        if not self.tsl_on:
            if self.pos == 1 and lo <= self.slp:
                return ('sl', self.slp)
            if self.pos == -1 and hi >= self.slp:
                return ('sl', self.slp)

        # 3) TSL 활성화 체크
        if self.pos == 1:
            best_pct = (hi - self.epx) / self.epx * 100
        else:
            best_pct = (self.epx - lo) / self.epx * 100
        if best_pct >= TSL_ACT:
            self.tsl_on = True

        # 4) TSL 추적 및 종가 청산
        if self.tsl_on:
            if self.pos == 1:
                self.peak = max(self.peak, hi)
                tsl_level = self.peak * (1 - TSL_WIDTH / 100)
                # TSL이 원래 SL보다 높으면 SL도 이동
                self.slp = max(self.slp, tsl_level)
                if cl <= tsl_level:
                    return ('tsl', cl)
            else:
                self.trough = min(self.trough, lo)
                tsl_level = self.trough * (1 + TSL_WIDTH / 100)
                self.slp = min(self.slp, tsl_level)
                if cl >= tsl_level:
                    return ('tsl', cl)

        return None

    def pos_str(self):
        if self.pos == 0: return "없음"
        d = "LONG" if self.pos == 1 else "SHORT"
        roi = (self.bal - IC) / IC * 100
        return f"{d} @ ${self.epx:,.0f}"

    def stats(self):
        n = len(self.trades)
        w = sum(1 for t in self.trades if t['pnl'] > 0)
        gp = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gl = abs(sum(t['pnl'] for t in self.trades if t['pnl'] <= 0))
        return {
            'bal': self.bal, 'ret': (self.bal - IC) / IC * 100,
            'pf': gp / max(gl, 0.001), 'mdd': self.mdd * 100,
            'n': n, 'wr': w / max(n, 1) * 100,
            'sl': self.sl_c, 'tsl': self.tsl_c, 'rev': self.rev_c, 'fl': self.fl_c
        }

# ═══════════════════════════════════════════
# 메인 백테스트
# ═══════════════════════════════════════════
def run(mode="2024"):
    print("=" * 60)
    print("  BTC/USDT v14.4 AI Backtest Comparison")
    print(f"  Mode: {mode} | {datetime.now()}")
    print("=" * 60)

    # ── 데이터 ──
    print("\n[1/4] 데이터 로딩...")
    df5 = load_data()
    df30 = resample_30m(df5)
    print(f"  5m: {len(df5):,} → 30m: {len(df30):,}")

    # ── 지표 (전체 기간에 계산) ──
    print("[2/4] 지표 계산...")
    ema3_s  = ema(df30['close'], 3)
    ema200_s = ema(df30['close'], 200)
    adx_s   = adx_calc(df30['high'], df30['low'], df30['close'], 14)
    rsi_s   = rsi_calc(df30['close'], 14)

    # 크로스 감지
    bull = ema3_s > ema200_s
    prev_bull = bull.shift(1).fillna(False)
    golden_x = bull & ~prev_bull
    dead_x   = ~bull & prev_bull
    valid_f  = (adx_s >= ADX_THRESH) & (rsi_s >= RSI_LOW) & (rsi_s <= RSI_HIGH)

    # numpy 변환
    cl = df30['close'].values; hi = df30['high'].values; lo = df30['low'].values
    ts_arr = df30.index
    ema3_v = ema3_s.values; ema200_v = ema200_s.values
    adx_v = adx_s.values; rsi_v = rsi_s.values
    gx = golden_x.values; dx = dead_x.values; vf = valid_f.values

    # 범위 결정
    n = len(df30)
    if mode == "2024":
        start_i = max(WARMUP, np.searchsorted(ts_arr, pd.Timestamp('2024-01-01')))
        end_i = min(n, np.searchsorted(ts_arr, pd.Timestamp('2025-01-01')))
    elif mode == "2021":
        start_i = max(WARMUP, np.searchsorted(ts_arr, pd.Timestamp('2021-01-01')))
        end_i = min(n, np.searchsorted(ts_arr, pd.Timestamp('2022-01-01')))
    elif mode == "2025":
        start_i = max(WARMUP, np.searchsorted(ts_arr, pd.Timestamp('2025-01-01')))
        end_i = min(n, np.searchsorted(ts_arr, pd.Timestamp('2026-01-01')))
    elif mode == "years":
        # 복수 연도: YEARS 환경변수로 전달
        years = os.getenv("BT_YEARS", "").split(",")
        ranges = []
        for y in years:
            y = y.strip()
            if y:
                si = max(WARMUP, np.searchsorted(ts_arr, pd.Timestamp(f'{y}-01-01')))
                ei = min(n, np.searchsorted(ts_arr, pd.Timestamp(f'{int(y)+1}-01-01')))
                ranges.append((si, ei))
        start_i = ranges[0][0] if ranges else WARMUP
        end_i = ranges[-1][1] if ranges else n
        # 범위 리스트를 저장해서 루프에서 사용
        year_ranges = ranges
    else:
        start_i = WARMUP
        end_i = n

    # years 모드: 실행할 인덱스 집합 구성
    if mode == "years":
        run_indices = set()
        for si, ei in year_ranges:
            run_indices.update(range(si, ei))
        total = len(run_indices)
        yr_labels = os.getenv("BT_YEARS", "")
        print(f"  범위: {yr_labels} ({total:,} 캔들)")
    else:
        run_indices = None
        total = end_i - start_i
        print(f"  범위: {ts_arr[start_i]} ~ {ts_arr[end_i-1]}")
        print(f"  캔들: {total:,}")

    # ── AI 초기화 ──
    print("[3/4] AI 클라이언트...")
    gpt_cli, cld_cli = init_ai()
    gpt_ok = "OK" if gpt_cli else "없음(코드규칙대체)"
    cld_ok = "OK" if cld_cli else "없음(코드규칙대체)"
    print(f"  GPT-5.4: {gpt_ok} | Claude: {cld_ok}")

    # ── 봇 초기화 ──
    code = Bot("코드봇"); gpt = Bot("GPT-5.4"); cla = Bot("Claude Sonnet")
    bots = [code, gpt, cla]
    cross_log = []
    errs = {'gpt': 0, 'claude': 0, 'gpt_fb': 0, 'claude_fb': 0}
    cross_cnt = 0; api_calls = 0; processed = 0

    print(f"\n[4/4] 백테스트 실행...\n")

    for i in range(start_i, end_i):
        # years 모드: 해당 연도 범위 밖이면 스킵
        if run_indices is not None and i not in run_indices:
            continue
        processed += 1
        t  = ts_arr[i]
        h_ = hi[i]; l_ = lo[i]; c_ = cl[i]
        mk = f"{t.year}-{t.month:02d}"

        # 월 업데이트
        for b in bots: b.update_month(mk)

        # ── SL / TSL / FL 자동 체크 ──
        for b in bots:
            ex = b.check_exit(h_, l_, c_)
            if ex:
                reason, ex_px = ex
                if reason == 'fl':
                    b.close(c_, 'fl', t)
                elif reason == 'sl':
                    b.close(ex_px, 'sl', t)
                elif reason == 'tsl':
                    b.close(ex_px, 'tsl', t)

        # ── 크로스 감지 ──
        is_gx = bool(gx[i]); is_dx = bool(dx[i]); is_vf = bool(vf[i])
        if not ((is_gx or is_dx) and is_vf):
            # 진행률 (1000캔들마다)
            if processed > 0 and processed % 1000 == 0:
                pct = processed / total * 100
                print(f"[진행 {pct:4.0f}%] {t.strftime('%Y-%m')} | "
                      f"코드: ${code.bal:,.0f} | GPT: ${gpt.bal:,.0f} | Claude: ${cla.bal:,.0f}")
            # 연도 갱신
            for b in bots: b.yr_bal[t.year] = b.bal
            continue

        cross_cnt += 1
        cross_type = "골든크로스" if is_gx else "데드크로스"
        new_dir = 1 if is_gx else -1
        a_v = adx_v[i]; r_v = rsi_v[i]; e3 = ema3_v[i]; e200 = ema200_v[i]

        # ── REV 청산 (모든 봇, 자동) ──
        for b in bots:
            if b.pos != 0 and b.pos != new_dir:
                b.close(c_, 'rev', t)

        # ── 코드봇: 자동 진입 ──
        code_entered = code.enter(new_dir, c_, t)
        code_action = ("LONG" if new_dir == 1 else "SHORT") if code_entered else "HOLD"

        # ── 최근 50캔들 문자열 ──
        s50 = max(0, i - 49)
        r50 = df30.iloc[s50:i+1]
        r50_str = r50[['open','high','low','close','volume']].tail(20).to_string()

        # ── GPT 판단 ──
        gpt_action = gpt_reason = ""
        if gpt_cli:
            prompt = build_prompt(str(t), cross_type, c_, e3, e200, a_v, r_v,
                                  r50_str, gpt.bal, gpt.pos_str(), gpt.monthly_pnl_pct())
            result = ask_gpt(gpt_cli, prompt)
            api_calls += 1
            if result:
                gpt_action = result.get('action', 'HOLD')
                gpt_reason = result.get('reason', '')
            else:
                errs['gpt'] += 1; errs['gpt_fb'] += 1
                gpt_action = "LONG" if is_gx else "SHORT"
                gpt_reason = "API오류-코드규칙대체"
        else:
            gpt_action = "LONG" if is_gx else "SHORT"
            gpt_reason = "API키없음-코드규칙대체"

        # ── Claude 판단 ──
        cla_action = cla_reason = ""
        if cld_cli:
            prompt = build_prompt(str(t), cross_type, c_, e3, e200, a_v, r_v,
                                  r50_str, cla.bal, cla.pos_str(), cla.monthly_pnl_pct())
            result = ask_claude(cld_cli, prompt)
            api_calls += 1
            if result:
                cla_action = result.get('action', 'HOLD')
                cla_reason = result.get('reason', '')
            else:
                errs['claude'] += 1; errs['claude_fb'] += 1
                cla_action = "LONG" if is_gx else "SHORT"
                cla_reason = "API오류-코드규칙대체"
        else:
            cla_action = "LONG" if is_gx else "SHORT"
            cla_reason = "API키없음-코드규칙대체"

        # ── AI 봇 진입 실행 ──
        gpt_entered = False
        if gpt_action in ("LONG", "SHORT"):
            ai_dir = 1 if gpt_action == "LONG" else -1
            if ai_dir == new_dir:
                gpt_entered = gpt.enter(ai_dir, c_, t)

        cla_entered = False
        if cla_action in ("LONG", "SHORT"):
            ai_dir = 1 if cla_action == "LONG" else -1
            if ai_dir == new_dir:
                cla_entered = cla.enter(ai_dir, c_, t)

        # ── 크로스 로그 ──
        cross_log.append({
            'timestamp': str(t), 'cross': cross_type,
            'price': c_, 'adx': round(a_v, 1), 'rsi': round(r_v, 1),
            'code_action': code_action,
            'gpt_action': gpt_action, 'gpt_reason': gpt_reason,
            'claude_action': cla_action, 'claude_reason': cla_reason,
            'code_bal': round(code.bal, 0),
            'gpt_bal': round(gpt.bal, 0),
            'claude_bal': round(cla.bal, 0)
        })

        # ── 출력 ──
        print(f"\n{'━'*60}")
        print(f"[{t}] {cross_type} #{cross_cnt}")
        print(f"ADX={a_v:.1f} | RSI={r_v:.1f} | 가격=${c_:,.0f}")
        print(f"{'━'*60}")
        g_str = f"{gpt_action:>5}" + (f'  | "{gpt_reason[:30]}"' if gpt_reason else "")
        c_str = f"{cla_action:>5}" + (f'  | "{cla_reason[:30]}"' if cla_reason else "")
        print(f"코드봇: {code_action:>5} {'진입' if code_entered else ''}")
        print(f"GPT:    {g_str}")
        print(f"Claude: {c_str}")
        print(f"{'━'*60}")
        print(f"잔액 | 코드: ${code.bal:,.0f} | GPT: ${gpt.bal:,.0f} | Claude: ${cla.bal:,.0f}")

        # 연도 갱신
        for b in bots: b.yr_bal[t.year] = b.bal

        # 잔액 0 체크
        for b in bots:
            if b.bal <= 0:
                print(f"\n  *** {b.name} 파산! ***")

    # ── 잔여 포지션 청산 ──
    last_px = cl[end_i - 1]; last_t = ts_arr[end_i - 1]
    for b in bots:
        if b.pos != 0:
            b.close(last_px, 'end', last_t)
        # 마지막 월 eq
        if b.cur_m in b.monthly:
            b.monthly[b.cur_m]['eq'] = b.bal
        b.yr_bal[last_t.year] = b.bal

    # ── 결과 출력 ──
    print_results(code, gpt, cla, cross_log, errs, api_calls)
    save_csvs(code, gpt, cla, cross_log, mode)


# ═══════════════════════════════════════════
# 결과 출력
# ═══════════════════════════════════════════
def print_results(code, gpt, cla, cross_log, errs, api_calls):
    cs = code.stats(); gs = gpt.stats(); ls = cla.stats()

    print(f"\n\n{'='*60}")
    print(f"         {'코드봇':>12}  {'GPT-5.4':>12}  {'Claude':>12}")
    print(f"{'='*60}")
    print(f"최종잔액  ${cs['bal']:>10,.0f}  ${gs['bal']:>10,.0f}  ${ls['bal']:>10,.0f}")
    print(f"수익률   {cs['ret']:>+10,.1f}%  {gs['ret']:>+10,.1f}%  {ls['ret']:>+10,.1f}%")
    print(f"PF       {cs['pf']:>10.2f}   {gs['pf']:>10.2f}   {ls['pf']:>10.2f}")
    print(f"MDD      {cs['mdd']:>10.1f}%  {gs['mdd']:>10.1f}%  {ls['mdd']:>10.1f}%")
    print(f"거래수   {cs['n']:>10}회  {gs['n']:>10}회  {ls['n']:>10}회")
    print(f"SL       {cs['sl']:>10}   {gs['sl']:>10}   {ls['sl']:>10}")
    print(f"TSL      {cs['tsl']:>10}   {gs['tsl']:>10}   {ls['tsl']:>10}")
    print(f"REV      {cs['rev']:>10}   {gs['rev']:>10}   {ls['rev']:>10}")
    print(f"FL       {cs['fl']:>10}   {gs['fl']:>10}   {ls['fl']:>10}")
    print(f"{'='*60}")

    # ── 판단 일치율 ──
    tot = len(cross_log)
    if tot > 0:
        all_eq  = sum(1 for x in cross_log if x['code_action'] == x['gpt_action'] == x['claude_action'])
        gpt_dif = sum(1 for x in cross_log
                      if x['gpt_action'] != x['code_action'] and x['claude_action'] == x['code_action'])
        cla_dif = sum(1 for x in cross_log
                      if x['claude_action'] != x['code_action'] and x['gpt_action'] == x['code_action'])
        both_dif = sum(1 for x in cross_log
                       if x['gpt_action'] != x['code_action'] and x['claude_action'] != x['code_action'])

        print(f"\n판단 일치율 (총 {tot}회 크로스):")
        print(f"  코드=GPT=Claude: {all_eq:>4} ({all_eq/tot*100:5.1f}%)")
        print(f"  GPT만 다름:      {gpt_dif:>4} ({gpt_dif/tot*100:5.1f}%)")
        print(f"  Claude만 다름:   {cla_dif:>4} ({cla_dif/tot*100:5.1f}%)")
        print(f"  둘 다 다름:      {both_dif:>4} ({both_dif/tot*100:5.1f}%)")

    # ── AI 거부 목록 ──
    rejected = [x for x in cross_log if x['gpt_action'] == 'HOLD' or x['claude_action'] == 'HOLD']
    if rejected:
        print(f"\nAI 거부 신호 ({len(rejected)}건):")
        print(f"  {'날짜':>20} | {'코드':>5} | {'GPT':>5} | {'Claude':>6} | {'GPT이유':>25} | {'Claude이유':>25}")
        print("  " + "-" * 100)
        for r in rejected[:30]:
            print(f"  {r['timestamp'][:16]:>20} | {r['code_action']:>5} | {r['gpt_action']:>5} | "
                  f"{r['claude_action']:>6} | {r['gpt_reason'][:25]:>25} | {r['claude_reason'][:25]:>25}")
        if len(rejected) > 30:
            print(f"  ... 외 {len(rejected)-30}건")

    # ── 연도별 잔액 ──
    all_yrs = sorted(set(list(code.yr_bal) + list(gpt.yr_bal) + list(cla.yr_bal)))
    if all_yrs:
        print(f"\n연도별 잔액 비교:")
        print(f"  {'연도':>6} | {'코드봇':>12} | {'GPT봇':>12} | {'Claude봇':>12}")
        print("  " + "-" * 55)
        for yr in all_yrs:
            cb = code.yr_bal.get(yr, 0)
            gb = gpt.yr_bal.get(yr, 0)
            lb = cla.yr_bal.get(yr, 0)
            print(f"  {yr:>6} | ${cb:>10,.0f} | ${gb:>10,.0f} | ${lb:>10,.0f}")

    # ── 월별 요약 ──
    months = sorted(set(list(code.monthly) + list(gpt.monthly) + list(cla.monthly)))
    if months:
        print(f"\n월별 잔액:")
        print(f"  {'월':>7} | {'코드봇':>10} | {'GPT봇':>10} | {'Claude봇':>10} | {'코드PnL':>9} | {'GPT PnL':>9} | {'Claude PnL':>9}")
        print("  " + "-" * 80)
        for mk in months:
            cd = code.monthly.get(mk, {}); gd = gpt.monthly.get(mk, {}); ld = cla.monthly.get(mk, {})
            ceq = cd.get('eq', 0); geq = gd.get('eq', 0); leq = ld.get('eq', 0)
            cp = cd.get('pnl', 0); gp = gd.get('pnl', 0); lp = ld.get('pnl', 0)
            print(f"  {mk:>7} | ${ceq:>8,.0f} | ${geq:>8,.0f} | ${leq:>8,.0f} | "
                  f"${cp:>+8,.0f} | ${gp:>+8,.0f} | ${lp:>+8,.0f}")

    # ── 최종 순위 ──
    ranking = sorted([
        (cs['bal'], '코드봇'), (gs['bal'], 'GPT-5.4'), (ls['bal'], 'Claude')
    ], reverse=True)

    print(f"\n{'='*60}")
    print("최종 결론:")
    for i, (bal, name) in enumerate(ranking):
        print(f"  {i+1}등: {name:>8}  (${bal:,.0f})")

    best_ai = max(gs['bal'], ls['bal'])
    if best_ai > cs['bal'] * 1.05:
        verdict = 'AI > 코드 → "AI 실전 봇 도입 검토"'
    elif best_ai > cs['bal'] * 0.95:
        verdict = 'AI = 코드 → "코드 봇 유지"'
    else:
        verdict = 'AI < 코드 → "AI 사용 불필요"'
    print(f"\n  판단: {verdict}")

    print(f"\nAPI 호출: {api_calls}회 | "
          f"오류: GPT {errs['gpt']}회(대체{errs['gpt_fb']}) Claude {errs['claude']}회(대체{errs['claude_fb']})")
    print(f"{'='*60}")


# ═══════════════════════════════════════════
# CSV 저장
# ═══════════════════════════════════════════
def save_csvs(code, gpt, cla, cross_log, mode):
    suffix = f"_{mode}" if mode != "full" else ""

    # 크로스별 비교
    if cross_log:
        df = pd.DataFrame(cross_log)
        path = f"{BASE}/ai_comparison_trades{suffix}.csv"
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"\n저장: {os.path.basename(path)} ({len(cross_log)}건)")

    # 월별 비교
    months = sorted(set(list(code.monthly) + list(gpt.monthly) + list(cla.monthly)))
    rows = []
    for mk in months:
        cd = code.monthly.get(mk, {}); gd = gpt.monthly.get(mk, {}); ld = cla.monthly.get(mk, {})
        rows.append({
            'month': mk,
            'code_bal': round(cd.get('eq', 0), 0),   'code_pnl': round(cd.get('pnl', 0), 0),
            'code_trades': cd.get('ent', 0),           'code_sl': cd.get('sl', 0),
            'code_tsl': cd.get('tsl', 0),              'code_rev': cd.get('rev', 0),
            'gpt_bal': round(gd.get('eq', 0), 0),     'gpt_pnl': round(gd.get('pnl', 0), 0),
            'gpt_trades': gd.get('ent', 0),
            'claude_bal': round(ld.get('eq', 0), 0),  'claude_pnl': round(ld.get('pnl', 0), 0),
            'claude_trades': ld.get('ent', 0),
        })
    if rows:
        df = pd.DataFrame(rows)
        path = f"{BASE}/ai_comparison_monthly{suffix}.csv"
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"저장: {os.path.basename(path)} ({len(rows)}건)")


# ═══════════════════════════════════════════
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "2024"
    # 콤마 구분 연도 지원: python ai_backtest_v14.py 2021,2025
    if "," in mode:
        os.environ["BT_YEARS"] = mode
        mode = "years"
    valid = ("2024", "2021", "2025", "full", "years")
    if mode not in valid:
        print(f"Usage: python ai_backtest_v14.py [{'/'.join(valid)}|2021,2025]")
        sys.exit(1)
    run(mode)
