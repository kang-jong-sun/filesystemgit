"""
BTC/USDT AI 자유재량 백테스트
코드봇(v14.4 규칙) vs GPT-5.4(자유판단) vs Claude Sonnet(자유판단)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI에게 고정 규칙 없이 모든 지표를 주고 자유 판단
ADX>=30 크로스만 AI 호출 (ADX<30은 자동 HOLD)
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
np.seterr(all='ignore')
load_dotenv("D:/filesystem/futures/btc_V1/.env")
load_dotenv()

# ═══ 설정 ═══
IC = 3000.0; FEE = 0.0004; LEVERAGE = 10; MARGIN_PCT = 0.25
SL_PCT = 7.0; TSL_ACT = 6.0; TSL_WIDTH = 3.0
ADX_THRESH = 35; RSI_LOW = 30; RSI_HIGH = 65; ML_LIMIT = -0.20
WARMUP = 300; FL_PCT = 100.0 / LEVERAGE
AI_ADX_MIN = 30  # AI 호출 최소 ADX
BASE = "D:/filesystem/futures/btc_V1/test2"

# ═══ 지표 ═══
def ema(s, p): return s.ewm(span=p, adjust=False).mean()

def rsi_calc(s, p=14):
    d = s.diff(); g = d.where(d > 0, 0.0); l = (-d).where(d < 0, 0.0)
    ag = g.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    al = l.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, 1e-10))

def adx_calc(h, l, c, p=14):
    pdm = h.diff(); mdm = -l.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    pdi = 100 * (pdm.ewm(alpha=1/p, min_periods=p, adjust=False).mean() / atr)
    mdi = 100 * (mdm.ewm(alpha=1/p, min_periods=p, adjust=False).mean() / atr)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-10)
    return dx.ewm(alpha=1/p, min_periods=p, adjust=False).mean()

def macd_calc(c, fast=12, slow=26, sig=9):
    ema_f = c.ewm(span=fast, adjust=False).mean()
    ema_s = c.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def bb_calc(c, p=20, std=2):
    mid = c.ewm(span=p, adjust=False).mean()
    s = c.rolling(p).std()
    return mid + std * s, mid, mid - std * s

def atr_calc(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, min_periods=p, adjust=False).mean()

def stoch_calc(h, l, c, k_p=14, d_p=3):
    ll = l.rolling(k_p).min(); hh = h.rolling(k_p).max()
    k = 100 * (c - ll) / (hh - ll).replace(0, 1e-10)
    d = k.rolling(d_p).mean()
    return k, d

# ═══ 데이터 ═══
def load_data():
    dfs = []
    for i in [1, 2, 3]:
        dfs.append(pd.read_csv(f"{BASE}/btc_usdt_5m_2020_to_now_part{i}.csv", parse_dates=['timestamp']))
    df = pd.concat(dfs).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']].astype(float)

def resample_30m(df):
    return df.resample('30min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

# ═══ AI ═══
def init_ai():
    gpt = claude = None
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        gpt = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return gpt, claude

GPT_SYSTEM = """당신은 15년 경력의 BTC/USDT 선물 퀀트 트레이더입니다.

[역할]
EMA 크로스 발생 시점에 모든 기술적 지표를 분석하여 진입 여부를 독자적으로 판단합니다.
고정 규칙 없이 당신의 전문적 판단으로 결정하세요.

[분석 관점]
- 추세 강도: ADX, EMA 배열, 가격 위치
- 모멘텀: RSI, MACD, 스토캐스틱
- 변동성: 볼린저밴드, ATR
- 거래량: 평균 대비 현재 거래량
- 캔들 패턴: 최근 가격 흐름

[판단 기준]
- 여러 지표가 동시에 같은 방향을 지지할 때만 진입
- 추세가 약하거나 지표가 엇갈리면 HOLD
- 의심스러우면 HOLD (놓치는 것보다 잘못 진입하는 게 더 나쁨)
- 이미 같은 방향 포지션이면 재진입 불필요

[출력 형식]
반드시 아래 JSON만 출력. 다른 텍스트 절대 금지.
{"action": "LONG/SHORT/HOLD", "confidence": 1~10, "reason": "한 줄 이유"}"""

CLAUDE_SYSTEM = """당신은 15년 경력의 BTC/USDT 선물 퀀트 트레이더입니다.

[역할]
EMA 크로스 발생 시점에 모든 기술적 지표를 분석하여 진입 여부를 독자적으로 판단합니다.
고정 규칙 없이 당신의 전문적 판단으로 결정하세요.

[분석 관점]
- 추세 강도: ADX, EMA 배열, 가격 위치
- 모멘텀: RSI, MACD, 스토캐스틱
- 변동성: 볼린저밴드, ATR
- 거래량: 평균 대비 현재 거래량
- 캔들 패턴: 최근 가격 흐름

[판단 기준]
- 여러 지표가 동시에 같은 방향을 지지할 때만 진입
- 추세가 약하거나 지표가 엇갈리면 HOLD
- 의심스러우면 HOLD (놓치는 것보다 잘못 진입하는 게 더 나쁨)
- 이미 같은 방향 포지션이면 재진입 불필요

[출력 형식]
반드시 아래 JSON만 출력. 다른 텍스트 절대 금지.
{"action": "LONG/SHORT/HOLD", "confidence": 1~10, "reason": "한 줄 이유"}"""

def ask_gpt(client, prompt, retries=5):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model="gpt-5.4", temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": GPT_SYSTEM},
                    {"role": "user", "content": prompt}
                ]
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            wait = min(30, 5 * (attempt + 1))
            print(f"    [GPT retry {attempt+1}/{retries}] {type(e).__name__} {wait}s")
            if attempt < retries - 1: time.sleep(wait)
            else: return None

def ask_claude(client, prompt, retries=5):
    for attempt in range(retries):
        try:
            r = client.messages.create(
                model="claude-sonnet-4-6", max_tokens=300, temperature=0,
                system=CLAUDE_SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            text = r.content[0].text
            s = text.find('{'); e = text.rfind('}') + 1
            return json.loads(text[s:e])
        except Exception as e:
            wait = min(60, 10 * (attempt + 1))
            print(f"    [Claude retry {attempt+1}/{retries}] {type(e).__name__} {wait}s")
            if attempt < retries - 1: time.sleep(wait)
            else: return None

def build_freehand_prompt(ts, cross_type, candle, indicators, recent_10, balance, pos_str, monthly_pnl):
    return f"""시각: {ts}
BTC/USDT 30분봉에서 **{cross_type}** 발생

=== 현재 캔들 ===
시가: ${candle['open']:,.0f} | 고가: ${candle['high']:,.0f} | 저가: ${candle['low']:,.0f} | 종가: ${candle['close']:,.0f}
거래량: {candle['volume']:,.0f} (20봉 평균 대비 {indicators['vol_ratio']:.1f}배)

=== 추세 지표 ===
EMA(3):   {indicators['ema3']:.1f}
EMA(21):  {indicators['ema21']:.1f}
EMA(50):  {indicators['ema50']:.1f}
EMA(200): {indicators['ema200']:.1f}
ADX(14):  {indicators['adx']:.1f}

=== 모멘텀 ===
RSI(14):       {indicators['rsi']:.1f}
MACD:          {indicators['macd']:.1f}
MACD Signal:   {indicators['macd_sig']:.1f}
MACD Hist:     {indicators['macd_hist']:.1f}
Stoch %K/%D:   {indicators['stoch_k']:.1f} / {indicators['stoch_d']:.1f}

=== 변동성 ===
BB 상단: {indicators['bb_upper']:.0f}
BB 중간: {indicators['bb_mid']:.0f}
BB 하단: {indicators['bb_lower']:.0f}
ATR(14): {indicators['atr']:.0f}
가격 위치: BB {indicators['bb_pct']:.0f}% (0=하단, 100=상단)

=== 최근 10캔들 요약 ===
{recent_10}

=== 계좌 ===
잔액: ${balance:,.0f} | 포지션: {pos_str} | 이번달 손익: {monthly_pnl:+.1f}%

진입할 가치가 있습니까?
{{"action": "LONG/SHORT/HOLD", "confidence": 1~10, "reason": "한 줄 이유"}}"""

# ═══ Bot ═══
class Bot:
    def __init__(self, name):
        self.name = name
        self.bal = IC; self.pos = 0; self.epx = 0.0; self.psz = 0.0
        self.margin = 0.0; self.slp = 0.0; self.tsl_on = False
        self.peak = 0.0; self.trough = 999999.0
        self.m_start = IC; self.cur_m = ""
        self.sl_c = 0; self.tsl_c = 0; self.rev_c = 0; self.fl_c = 0
        self.trades = []; self.monthly = {}
        self.pk_bal = IC; self.mdd = 0.0; self.yr_bal = {}

    def monthly_pnl_pct(self):
        return (self.bal - self.m_start) / self.m_start * 100 if self.m_start > 0 else 0.0

    def update_month(self, mk):
        if mk != self.cur_m:
            if self.cur_m and self.cur_m in self.monthly: self.monthly[self.cur_m]['eq'] = self.bal
            self.cur_m = mk; self.m_start = self.bal
            if mk not in self.monthly:
                self.monthly[mk] = {'pnl':0,'ent':0,'w':0,'l':0,'sl':0,'tsl':0,'rev':0,'fl':0,'eq_s':self.bal,'eq':self.bal}

    def update_dd(self):
        self.pk_bal = max(self.pk_bal, self.bal)
        if self.pk_bal > 0: self.mdd = max(self.mdd, (self.pk_bal - self.bal) / self.pk_bal)

    def enter(self, direction, price, ts, check_ml=True):
        if self.bal <= 0 or self.bal * MARGIN_PCT < 1: return False
        if check_ml and self.m_start > 0 and (self.bal - self.m_start) / self.m_start <= ML_LIMIT: return False
        self.pos = direction; self.epx = price
        self.margin = self.bal * MARGIN_PCT; self.psz = self.margin * LEVERAGE
        self.bal -= self.psz * FEE
        self.tsl_on = False; self.peak = price; self.trough = price
        self.slp = price * (1 - SL_PCT/100) if direction == 1 else price * (1 + SL_PCT/100)
        mk = self.cur_m
        if mk in self.monthly: self.monthly[mk]['ent'] += 1
        return True

    def close(self, price, reason, ts):
        if self.pos == 0: return 0.0
        if reason == 'fl': pnl = -(self.margin + self.psz * FEE)
        else: pnl = (price - self.epx) / self.epx * self.psz * self.pos - self.psz * FEE
        self.bal += pnl
        if self.bal < 0: self.bal = 0
        mk = self.cur_m
        if mk in self.monthly:
            self.monthly[mk]['pnl'] += pnl
            if pnl > 0: self.monthly[mk]['w'] += 1
            else: self.monthly[mk]['l'] += 1
            if reason in ('sl','tsl','rev','fl'): self.monthly[mk][reason] += 1
        if reason == 'sl': self.sl_c += 1
        elif reason == 'tsl': self.tsl_c += 1
        elif reason == 'rev': self.rev_c += 1
        elif reason == 'fl': self.fl_c += 1
        self.trades.append({'ts':str(ts),'dir':'LONG' if self.pos==1 else 'SHORT',
            'entry':self.epx,'exit':price,'pnl':round(pnl,2),'reason':reason,'bal':round(self.bal,2)})
        self.pos = 0; self.epx = 0; self.psz = 0; self.margin = 0; self.update_dd()
        return pnl

    def check_exit(self, hi, lo, cl):
        if self.pos == 0: return None
        if self.pos == 1: worst = (lo - self.epx) / self.epx * 100
        else: worst = (self.epx - hi) / self.epx * 100
        if worst <= -FL_PCT: return ('fl', None)
        if not self.tsl_on:
            if self.pos == 1 and lo <= self.slp: return ('sl', self.slp)
            if self.pos == -1 and hi >= self.slp: return ('sl', self.slp)
        if self.pos == 1: best = (hi - self.epx) / self.epx * 100
        else: best = (self.epx - lo) / self.epx * 100
        if best >= TSL_ACT: self.tsl_on = True
        if self.tsl_on:
            if self.pos == 1:
                self.peak = max(self.peak, hi); tl = self.peak * (1 - TSL_WIDTH/100)
                self.slp = max(self.slp, tl)
                if cl <= tl: return ('tsl', cl)
            else:
                self.trough = min(self.trough, lo); tl = self.trough * (1 + TSL_WIDTH/100)
                self.slp = min(self.slp, tl)
                if cl >= tl: return ('tsl', cl)
        return None

    def pos_str(self):
        if self.pos == 0: return "없음"
        return f"{'LONG' if self.pos==1 else 'SHORT'} @ ${self.epx:,.0f}"

    def stats(self):
        n = len(self.trades); w = sum(1 for t in self.trades if t['pnl'] > 0)
        gp = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gl = abs(sum(t['pnl'] for t in self.trades if t['pnl'] <= 0))
        return {'bal':self.bal,'ret':(self.bal-IC)/IC*100,'pf':gp/max(gl,0.001),
                'mdd':self.mdd*100,'n':n,'wr':w/max(n,1)*100,
                'sl':self.sl_c,'tsl':self.tsl_c,'rev':self.rev_c,'fl':self.fl_c}

# ═══ Main ═══
def run():
    print("=" * 70)
    print("  BTC/USDT AI 자유재량 백테스트 (코드봇 v14.4 vs AI 자유판단)")
    print(f"  {datetime.now()}")
    print("=" * 70)

    print("\n[1/4] 데이터...")
    df5 = load_data(); df30 = resample_30m(df5)
    print(f"  30m: {len(df30):,} candles")

    print("[2/4] 지표 계산...")
    c = df30['close']; h = df30['high']; l = df30['low']
    ema3_s = ema(c, 3); ema21_s = ema(c, 21); ema50_s = ema(c, 50); ema200_s = ema(c, 200)
    adx_s = adx_calc(h, l, c, 14); rsi_s = rsi_calc(c, 14)
    macd_s, macd_sig_s, macd_hist_s = macd_calc(c)
    bb_up_s, bb_mid_s, bb_lo_s = bb_calc(c)
    atr_s = atr_calc(h, l, c, 14)
    stk_s, std_s = stoch_calc(h, l, c)
    vol_ma = df30['volume'].rolling(20).mean()

    bull = ema3_s > ema200_s
    prev_bull = bull.shift(1).fillna(False)
    golden = (bull & ~prev_bull).values
    dead = (~bull & prev_bull).values
    v14_filter = ((adx_s >= ADX_THRESH) & (rsi_s >= RSI_LOW) & (rsi_s <= RSI_HIGH)).values

    ts_arr = df30.index; cl = c.values; hi = h.values; lo = l.values
    n = len(df30)

    print("[3/4] AI 초기화...")
    gpt_cli, cld_cli = init_ai()
    print(f"  GPT-5.4: {'OK' if gpt_cli else 'X'} | Claude: {'OK' if cld_cli else 'X'}")

    code = Bot("코드봇 v14.4"); gpt = Bot("GPT-5.4"); cla = Bot("Claude Sonnet")
    bots = [code, gpt, cla]
    cross_log = []; errs = {'gpt':0,'claude':0}
    cross_cnt = 0; api_calls = 0; processed = 0

    print(f"\n[4/4] 백테스트 실행...\n")

    for i in range(WARMUP, n):
        t = ts_arr[i]; h_ = hi[i]; l_ = lo[i]; c_ = cl[i]
        mk = f"{t.year}-{t.month:02d}"
        for b in bots: b.update_month(mk)

        # SL/TSL/FL
        for b in bots:
            ex = b.check_exit(h_, l_, c_)
            if ex:
                reason, px = ex
                if reason == 'fl': b.close(c_, 'fl', t)
                elif reason == 'sl': b.close(px, 'sl', t)
                elif reason == 'tsl': b.close(px, 'tsl', t)

        # Cross?
        is_gx = bool(golden[i]); is_dx = bool(dead[i])
        if not (is_gx or is_dx):
            processed += 1
            if processed > 0 and processed % 2000 == 0:
                pct = processed / (n - WARMUP) * 100
                print(f"[{pct:4.0f}%] {t.strftime('%Y-%m')} | 코드:${code.bal:,.0f} GPT:${gpt.bal:,.0f} Claude:${cla.bal:,.0f} | 크로스:{cross_cnt}")
            for b in bots: b.yr_bal[t.year] = b.bal
            continue

        cross_cnt += 1
        cross_type = "골든크로스" if is_gx else "데드크로스"
        new_dir = 1 if is_gx else -1
        a_v = adx_s.values[i]; r_v = rsi_s.values[i]
        is_filtered = bool(v14_filter[i])

        # ── 코드봇: v14.4 규칙 ──
        # REV
        if code.pos != 0 and code.pos != new_dir and is_filtered:
            code.close(c_, 'rev', t)
        code_action = "HOLD"
        if is_filtered and code.pos != new_dir:
            if code.enter(new_dir, c_, t):
                code_action = "LONG" if new_dir == 1 else "SHORT"

        # ── AI 봇: ADX>=30일 때만 호출 ──
        gpt_action = gpt_reason = cla_action = cla_reason = ""
        gpt_conf = cla_conf = 0

        if a_v >= AI_ADX_MIN:
            # 지표 딕셔너리
            vm = vol_ma.values[i] if not np.isnan(vol_ma.values[i]) else 1
            ind = {
                'ema3': ema3_s.values[i], 'ema21': ema21_s.values[i],
                'ema50': ema50_s.values[i], 'ema200': ema200_s.values[i],
                'adx': a_v, 'rsi': r_v,
                'macd': macd_s.values[i], 'macd_sig': macd_sig_s.values[i], 'macd_hist': macd_hist_s.values[i],
                'bb_upper': bb_up_s.values[i], 'bb_mid': bb_mid_s.values[i], 'bb_lower': bb_lo_s.values[i],
                'atr': atr_s.values[i],
                'stoch_k': stk_s.values[i], 'stoch_d': std_s.values[i],
                'vol_ratio': df30['volume'].values[i] / max(vm, 1),
                'bb_pct': (c_ - bb_lo_s.values[i]) / max(bb_up_s.values[i] - bb_lo_s.values[i], 1) * 100,
            }
            candle = {'open': df30['open'].values[i], 'high': h_, 'low': l_, 'close': c_,
                      'volume': df30['volume'].values[i]}

            # 최근 10캔들 요약
            s10 = max(0, i-9)
            r10 = df30.iloc[s10:i+1]
            ups = sum(r10['close'] > r10['open']); dns = len(r10) - ups
            r10_str = f"양봉 {ups}개, 음봉 {dns}개\n"
            r10_str += r10[['open','high','low','close','volume']].tail(5).to_string()

            prompt = build_freehand_prompt(str(t), cross_type, candle, ind, r10_str,
                                           gpt.bal, gpt.pos_str(), gpt.monthly_pnl_pct())

            # GPT
            if gpt_cli:
                res = ask_gpt(gpt_cli, prompt); api_calls += 1
                if res:
                    gpt_action = res.get('action', 'HOLD')
                    gpt_reason = res.get('reason', '')
                    gpt_conf = res.get('confidence', 0)
                else:
                    errs['gpt'] += 1; gpt_action = "HOLD"; gpt_reason = "API오류"
            else:
                gpt_action = "LONG" if is_gx else "SHORT" if is_filtered else "HOLD"
                gpt_reason = "API없음"

            # Claude (별도 잔액 기준 프롬프트)
            prompt_c = build_freehand_prompt(str(t), cross_type, candle, ind, r10_str,
                                              cla.bal, cla.pos_str(), cla.monthly_pnl_pct())
            if cld_cli:
                res = ask_claude(cld_cli, prompt_c); api_calls += 1
                if res:
                    cla_action = res.get('action', 'HOLD')
                    cla_reason = res.get('reason', '')
                    cla_conf = res.get('confidence', 0)
                else:
                    errs['claude'] += 1; cla_action = "HOLD"; cla_reason = "API오류"
            else:
                cla_action = "LONG" if is_gx else "SHORT" if is_filtered else "HOLD"
                cla_reason = "API없음"
        else:
            gpt_action = "HOLD"; gpt_reason = f"ADX {a_v:.0f}<30 자동HOLD"
            cla_action = "HOLD"; cla_reason = f"ADX {a_v:.0f}<30 자동HOLD"

        # ── AI 봇 실행 ──
        for bot, action in [(gpt, gpt_action), (cla, cla_action)]:
            if action in ("LONG", "SHORT"):
                ai_dir = 1 if action == "LONG" else -1
                # 반대 포지션이면 청산
                if bot.pos != 0 and bot.pos != ai_dir:
                    bot.close(c_, 'rev', t)
                # 같은 방향이면 스킵
                if bot.pos == 0:
                    bot.enter(ai_dir, c_, t)

        # 로그
        cross_log.append({
            'timestamp': str(t), 'cross': cross_type, 'price': c_,
            'adx': round(a_v, 1), 'rsi': round(r_v, 1), 'filtered': is_filtered,
            'code_action': code_action,
            'gpt_action': gpt_action, 'gpt_conf': gpt_conf, 'gpt_reason': gpt_reason,
            'claude_action': cla_action, 'claude_conf': cla_conf, 'claude_reason': cla_reason,
            'code_bal': round(code.bal, 0), 'gpt_bal': round(gpt.bal, 0), 'claude_bal': round(cla.bal, 0)
        })

        # 출력 (ADX>=30만)
        if a_v >= AI_ADX_MIN:
            filt_mark = "V14" if is_filtered else "NEW"
            print(f"\n{'━'*65}")
            print(f"[{t}] {cross_type} #{cross_cnt} [{filt_mark}]")
            print(f"ADX={a_v:.1f} RSI={r_v:.1f} ${c_:,.0f} | MACD={ind['macd']:.0f} BB={ind['bb_pct']:.0f}%")
            print(f"{'━'*65}")
            print(f"코드봇: {code_action:>5}")
            print(f"GPT:    {gpt_action:>5} conf={gpt_conf} | {gpt_reason[:45]}")
            print(f"Claude: {cla_action:>5} conf={cla_conf} | {cla_reason[:45]}")
            print(f"{'━'*65}")
            print(f"잔액 | 코드:${code.bal:,.0f} GPT:${gpt.bal:,.0f} Claude:${cla.bal:,.0f}")

        for b in bots: b.yr_bal[t.year] = b.bal
        processed += 1

    # 잔여 청산
    last_px = cl[-1]; last_t = ts_arr[-1]
    for b in bots:
        if b.pos != 0: b.close(last_px, 'end', last_t)
        if b.cur_m in b.monthly: b.monthly[b.cur_m]['eq'] = b.bal
        b.yr_bal[last_t.year] = b.bal

    print_results(code, gpt, cla, cross_log, errs, api_calls)
    save_csvs(code, gpt, cla, cross_log)


def print_results(code, gpt, cla, cross_log, errs, api_calls):
    cs = code.stats(); gs = gpt.stats(); ls = cla.stats()

    print(f"\n\n{'='*70}")
    print(f"{'':>12}{'코드봇v14.4':>14}{'GPT-5.4 자유':>14}{'Claude 자유':>14}")
    print(f"{'='*70}")
    print(f"{'최종잔액':>12} ${cs['bal']:>10,.0f} ${gs['bal']:>10,.0f} ${ls['bal']:>10,.0f}")
    print(f"{'수익률':>12} {cs['ret']:>+10,.1f}% {gs['ret']:>+10,.1f}% {ls['ret']:>+10,.1f}%")
    print(f"{'PF':>12} {cs['pf']:>10.2f} {gs['pf']:>10.2f} {ls['pf']:>10.2f}")
    print(f"{'MDD':>12} {cs['mdd']:>10.1f}% {gs['mdd']:>10.1f}% {ls['mdd']:>10.1f}%")
    print(f"{'거래수':>12} {cs['n']:>10}회 {gs['n']:>10}회 {ls['n']:>10}회")
    print(f"{'SL':>12} {cs['sl']:>10} {gs['sl']:>10} {ls['sl']:>10}")
    print(f"{'TSL':>12} {cs['tsl']:>10} {gs['tsl']:>10} {ls['tsl']:>10}")
    print(f"{'REV':>12} {cs['rev']:>10} {gs['rev']:>10} {ls['rev']:>10}")
    print(f"{'='*70}")

    # 크로스 분석
    tot = len(cross_log)
    ai_called = [x for x in cross_log if x['adx'] >= AI_ADX_MIN]
    filtered = [x for x in cross_log if x['filtered']]
    new_only = [x for x in ai_called if not x['filtered']]

    print(f"\n크로스 분석:")
    print(f"  전체 크로스: {tot}건")
    print(f"  ADX>=30 (AI호출): {len(ai_called)}건")
    print(f"  v14.4 필터 통과:  {len(filtered)}건")
    print(f"  필터 미통과 (NEW): {len(new_only)}건")

    # AI가 NEW 구간에서 진입한 건
    gpt_new = [x for x in new_only if x['gpt_action'] in ('LONG','SHORT')]
    cla_new = [x for x in new_only if x['claude_action'] in ('LONG','SHORT')]
    print(f"\n  AI가 필터 미통과 구간에서 진입:")
    print(f"    GPT:    {len(gpt_new)}건 / {len(new_only)}건")
    print(f"    Claude: {len(cla_new)}건 / {len(new_only)}건")

    # AI가 필터 통과 구간에서 거부한 건
    gpt_reject = [x for x in filtered if x['gpt_action'] == 'HOLD']
    cla_reject = [x for x in filtered if x['claude_action'] == 'HOLD']
    print(f"\n  AI가 필터 통과 구간에서 거부:")
    print(f"    GPT:    {len(gpt_reject)}건 / {len(filtered)}건")
    print(f"    Claude: {len(cla_reject)}건 / {len(filtered)}건")

    # 연도별
    all_yrs = sorted(set(list(code.yr_bal) + list(gpt.yr_bal) + list(cla.yr_bal)))
    print(f"\n연도별 잔액:")
    print(f"  {'연도':>6} | {'코드봇':>12} | {'GPT 자유':>12} | {'Claude 자유':>12}")
    print("  " + "-" * 55)
    for yr in all_yrs:
        print(f"  {yr:>6} | ${code.yr_bal.get(yr,0):>10,.0f} | ${gpt.yr_bal.get(yr,0):>10,.0f} | ${cla.yr_bal.get(yr,0):>10,.0f}")

    # 순위
    ranking = sorted([(cs['bal'],'코드봇'),(gs['bal'],'GPT-5.4'),(ls['bal'],'Claude')], reverse=True)
    print(f"\n최종 순위:")
    for i,(bal,name) in enumerate(ranking):
        print(f"  {i+1}등: {name:>10} ${bal:,.0f}")

    best_ai = max(gs['bal'], ls['bal'])
    if best_ai > cs['bal'] * 1.05: v = 'AI > 코드 -> "AI 자유재량 도입 검토"'
    elif best_ai > cs['bal'] * 0.95: v = 'AI = 코드 -> "코드 봇 유지"'
    else: v = 'AI < 코드 -> "고정 규칙이 더 나음"'
    print(f"\n  판단: {v}")
    print(f"\nAPI: {api_calls}회 | 오류: GPT {errs['gpt']} Claude {errs['claude']}")
    print("=" * 70)


def save_csvs(code, gpt, cla, cross_log):
    if cross_log:
        df = pd.DataFrame(cross_log)
        path = f"{BASE}/ai_freehand_trades.csv"
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"\n저장: ai_freehand_trades.csv ({len(cross_log)}건)")

    months = sorted(set(list(code.monthly) + list(gpt.monthly) + list(cla.monthly)))
    rows = []
    for mk in months:
        cd = code.monthly.get(mk,{}); gd = gpt.monthly.get(mk,{}); ld = cla.monthly.get(mk,{})
        rows.append({'month':mk,
            'code_bal':round(cd.get('eq',0),0),'code_pnl':round(cd.get('pnl',0),0),'code_trades':cd.get('ent',0),
            'gpt_bal':round(gd.get('eq',0),0),'gpt_pnl':round(gd.get('pnl',0),0),'gpt_trades':gd.get('ent',0),
            'claude_bal':round(ld.get('eq',0),0),'claude_pnl':round(ld.get('pnl',0),0),'claude_trades':ld.get('ent',0)})
    if rows:
        df = pd.DataFrame(rows)
        path = f"{BASE}/ai_freehand_monthly.csv"
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"저장: ai_freehand_monthly.csv ({len(rows)}건)")


if __name__ == "__main__":
    run()
