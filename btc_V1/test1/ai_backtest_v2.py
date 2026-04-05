"""
BTC/USDT v14.4 Enhanced AI Backtest V2 - Full Indicators
코드봇 vs GPT-5.4 vs Claude Sonnet 4.6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[V2 변경사항 - 모든 지표 100봉 히스토리]
- 메인: EMA 3/7/21/100/200, ADX/DI, RSI, MACD, BB (100봉)
- 추가: ATR, OBV, StochRSI, 캔들패턴, 변화율 (100봉)
- 다중TF: 1h/4h/1D 각 EMA/ADX/RSI (100봉)
- 컨텍스트: 피보나치, 지지/저항, 다이버전스, 거래이력

Usage:
  python ai_backtest_v2.py              # 실제 API
  python ai_backtest_v2.py --dry-run    # 시뮬레이션
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
# nohup 리다이렉트시 flush 보장
import functools
print = functools.partial(print, flush=True)
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
ADX_THRESH_CODE = 35
ADX_THRESH_AI   = 20
RSI_LOW     = 30
RSI_HIGH    = 65
ML_LIMIT    = -0.20
WARMUP      = 300
FL_PCT      = 100.0 / LEVERAGE
TSL_PROXIMITY = 1.5
HISTORY_BARS = 100

DATA_PATH = "D:/filesystem/futures/btc_V1/test1/btc_usdt_5m_merged.csv"

# ================================================================
# 지표 계산
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
    return dx.ewm(alpha=1/p, min_periods=p, adjust=False).mean(), pdi, mdi, atr

def macd_calc(c, fast=12, slow=26, signal=9):
    ef = c.ewm(span=fast, adjust=False).mean()
    es = c.ewm(span=slow, adjust=False).mean()
    ml = ef - es
    sl = ml.ewm(span=signal, adjust=False).mean()
    return ml, sl, ml - sl

def bb_calc(c, period=20, std_dev=2):
    sma = c.rolling(period).mean()
    std = c.rolling(period).std()
    return sma + std_dev * std, sma, sma - std_dev * std

def stoch_rsi_calc(c, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    r = rsi_calc(c, rsi_period)
    lowest = r.rolling(stoch_period).min()
    highest = r.rolling(stoch_period).max()
    stoch_rsi = (r - lowest) / (highest - lowest).replace(0, 1e-10) * 100
    k = stoch_rsi.rolling(k_period).mean()
    d = k.rolling(d_period).mean()
    return k, d

def obv_calc(c, v):
    direction = np.sign(c.diff()).fillna(0)
    return (v * direction).cumsum()

def detect_candle_pattern(o, h, l, c):
    """캔들 패턴 감지: 0=없음, 1=망치/역망치, 2=도지, 3=장악형, -1=유성, -2=도지, -3=하락장악"""
    body = c - o
    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l
    body_abs = body.abs()
    candle_range = h - l

    patterns = pd.Series(0, index=o.index)

    # 도지 (body < 10% of range)
    doji = body_abs < candle_range * 0.1
    patterns = patterns.where(~doji, 2)

    # 망치 (하방 그림자 > body*2, 상방 그림자 작음)
    hammer = (lower_shadow > body_abs * 2) & (upper_shadow < body_abs * 0.5) & (body > 0)
    patterns = patterns.where(~hammer, 1)

    # 유성 (상방 그림자 > body*2, 하방 그림자 작음)
    shooting = (upper_shadow > body_abs * 2) & (lower_shadow < body_abs * 0.5) & (body < 0)
    patterns = patterns.where(~shooting, -1)

    # 상승장악 (이전 음봉 → 현재 양봉이 완전히 감쌈)
    prev_bearish = o.shift(1) > c.shift(1)
    engulf_up = prev_bearish & (body > 0) & (o <= c.shift(1)) & (c >= o.shift(1))
    patterns = patterns.where(~engulf_up, 3)

    # 하락장악
    prev_bullish = c.shift(1) > o.shift(1)
    engulf_dn = prev_bullish & (body < 0) & (o >= c.shift(1)) & (c <= o.shift(1))
    patterns = patterns.where(~engulf_dn, -3)

    return patterns

def find_support_resistance(h, l, c, lookback=100):
    """최근 lookback 봉에서 주요 지지/저항 수준 (피벗 기반)"""
    sr_levels = []
    for i in range(2, min(lookback, len(c)) - 2):
        idx = len(c) - lookback + i
        if idx < 2 or idx >= len(c) - 2:
            continue
        # 피벗 하이
        if h.iloc[idx] > h.iloc[idx-1] and h.iloc[idx] > h.iloc[idx-2] and \
           h.iloc[idx] > h.iloc[idx+1] and h.iloc[idx] > h.iloc[idx+2]:
            sr_levels.append(('R', round(h.iloc[idx], 1)))
        # 피벗 로우
        if l.iloc[idx] < l.iloc[idx-1] and l.iloc[idx] < l.iloc[idx-2] and \
           l.iloc[idx] < l.iloc[idx+1] and l.iloc[idx] < l.iloc[idx+2]:
            sr_levels.append(('S', round(l.iloc[idx], 1)))
    return sr_levels[-10:] if sr_levels else []

def calc_fibonacci(high_val, low_val):
    """피보나치 레벨 계산"""
    diff = high_val - low_val
    return {
        '0.0%': round(high_val, 1),
        '23.6%': round(high_val - diff * 0.236, 1),
        '38.2%': round(high_val - diff * 0.382, 1),
        '50.0%': round(high_val - diff * 0.5, 1),
        '61.8%': round(high_val - diff * 0.618, 1),
        '100%': round(low_val, 1),
    }

def detect_divergence(price, indicator, lookback=20):
    """가격-지표 다이버전스 감지"""
    if len(price) < lookback:
        return "NONE"
    p_recent = price.iloc[-lookback:]
    i_recent = indicator.iloc[-lookback:]

    p_rising = p_recent.iloc[-1] > p_recent.iloc[0]
    i_rising = i_recent.iloc[-1] > i_recent.iloc[0]

    if p_rising and not i_rising:
        return "BEARISH_DIV"  # 가격 상승 + 지표 하락
    elif not p_rising and i_rising:
        return "BULLISH_DIV"  # 가격 하락 + 지표 상승
    return "NONE"

# ================================================================
# 데이터 로드 & 지표
# ================================================================
def load_and_prepare():
    print("[1/4] Data loading...")
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    print(f"  5m: {len(df):,} rows")

    # 다중 타임프레임 리샘플링
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    df30 = df.resample('30min').agg(agg).dropna()
    df1h = df.resample('1h').agg(agg).dropna()
    df4h = df.resample('4h').agg(agg).dropna()
    df1d = df.resample('1D').agg(agg).dropna()

    print(f"  30m: {len(df30):,} | 1h: {len(df1h):,} | 4h: {len(df4h):,} | 1D: {len(df1d):,}")

    print("[2/4] Indicators (all timeframes)...")

    # === 30분봉 지표 ===
    c = df30['close']; h = df30['high']; l = df30['low']; o = df30['open']; v = df30['volume']
    df30['ema3'] = ema(c, 3); df30['ema7'] = ema(c, 7); df30['ema21'] = ema(c, 21)
    df30['ema100'] = ema(c, 100); df30['ema200'] = ema(c, 200)
    adx_s, pdi_s, mdi_s, atr_s = adx_calc(h, l, c, 14)
    df30['adx'] = adx_s; df30['plus_di'] = pdi_s; df30['minus_di'] = mdi_s; df30['atr'] = atr_s
    df30['rsi'] = rsi_calc(c, 14)
    ml, sl_, hist = macd_calc(c); df30['macd'] = ml; df30['macd_signal'] = sl_; df30['macd_hist'] = hist
    bb_u, bb_m, bb_l = bb_calc(c); df30['bb_upper'] = bb_u; df30['bb_mid'] = bb_m; df30['bb_lower'] = bb_l
    stk, std_ = stoch_rsi_calc(c); df30['stoch_k'] = stk; df30['stoch_d'] = std_
    df30['obv'] = obv_calc(c, v)
    df30['candle_pattern'] = detect_candle_pattern(o, h, l, c)
    # 변화율
    df30['chg_1h'] = c.pct_change(2) * 100    # 2봉 = 1시간
    df30['chg_4h'] = c.pct_change(8) * 100    # 8봉 = 4시간
    df30['chg_1d'] = c.pct_change(48) * 100   # 48봉 = 24시간
    df30['chg_1w'] = c.pct_change(336) * 100  # 336봉 = 7일

    # Cross detection
    bull = df30['ema3'] > df30['ema200']
    prev_bull = bull.shift(1).fillna(False)
    df30['cross_up'] = bull & ~prev_bull
    df30['cross_dn'] = ~bull & prev_bull

    # === 1시간봉 지표 ===
    c1 = df1h['close']; h1 = df1h['high']; l1 = df1h['low']
    df1h['ema7'] = ema(c1, 7); df1h['ema21'] = ema(c1, 21); df1h['ema50'] = ema(c1, 50)
    adx1, pdi1, mdi1, atr1 = adx_calc(h1, l1, c1, 14)
    df1h['adx'] = adx1; df1h['rsi'] = rsi_calc(c1, 14)
    ml1, sl1_, h1_ = macd_calc(c1); df1h['macd_hist'] = h1_
    df1h['trend'] = np.where(df1h['ema7'] > df1h['ema21'], 1, -1)

    # === 4시간봉 지표 ===
    c4 = df4h['close']; h4 = df4h['high']; l4 = df4h['low']
    df4h['ema7'] = ema(c4, 7); df4h['ema21'] = ema(c4, 21); df4h['ema50'] = ema(c4, 50)
    adx4, pdi4, mdi4, atr4 = adx_calc(h4, l4, c4, 14)
    df4h['adx'] = adx4; df4h['rsi'] = rsi_calc(c4, 14)
    ml4, sl4_, h4_ = macd_calc(c4); df4h['macd_hist'] = h4_
    df4h['trend'] = np.where(df4h['ema7'] > df4h['ema21'], 1, -1)

    # === 일봉 지표 ===
    cd = df1d['close']; hd = df1d['high']; ld = df1d['low']
    df1d['ema7'] = ema(cd, 7); df1d['ema21'] = ema(cd, 21); df1d['ema50'] = ema(cd, 50)
    adxd, pdid, mdid, atrd = adx_calc(hd, ld, cd, 14)
    df1d['adx'] = adxd; df1d['rsi'] = rsi_calc(cd, 14)
    mld, sld_, hd_ = macd_calc(cd); df1d['macd_hist'] = hd_
    df1d['trend'] = np.where(df1d['ema7'] > df1d['ema21'], 1, -1)

    print(f"  30m: EMA/ADX/RSI/MACD/BB/ATR/OBV/StochRSI/Pattern/ChgRate")
    print(f"  1h/4h/1D: EMA/ADX/RSI/MACD/Trend")

    return df30, df1h, df4h, df1d


# ================================================================
# 프롬프트 빌더 (풀 지표)
# ================================================================
def build_main_table(df30, idx, bars=HISTORY_BARS):
    """메인 지표 100봉 테이블"""
    start = max(0, idx - bars + 1)
    chunk = df30.iloc[start:idx+1]
    cols = ['open', 'high', 'low', 'close', 'volume',
            'ema3', 'ema7', 'ema21', 'ema100', 'ema200',
            'adx', 'plus_di', 'minus_di', 'rsi',
            'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_mid', 'bb_lower']
    t = chunk[cols].copy()
    t.index = t.index.strftime('%m-%d %H:%M')
    for c in ['open','high','low','close','ema3','ema7','ema21','ema100','ema200','bb_upper','bb_mid','bb_lower']:
        t[c] = t[c].round(0).astype(int)
    for c in ['adx','plus_di','minus_di','rsi']:
        t[c] = t[c].round(1)
    for c in ['macd','macd_signal','macd_hist']:
        t[c] = t[c].round(1)
    t['volume'] = (t['volume']/1000).round(0).astype(int)
    t = t.rename(columns={'volume': 'vol_k'})
    return t.to_string()


def build_extra_table(df30, idx, bars=HISTORY_BARS):
    """보조 지표 100봉 테이블"""
    start = max(0, idx - bars + 1)
    chunk = df30.iloc[start:idx+1]
    cols = ['atr', 'stoch_k', 'stoch_d', 'obv', 'candle_pattern',
            'chg_1h', 'chg_4h', 'chg_1d', 'chg_1w']
    t = chunk[cols].copy()
    t.index = t.index.strftime('%m-%d %H:%M')
    t['atr'] = t['atr'].round(1)
    t['stoch_k'] = t['stoch_k'].round(1)
    t['stoch_d'] = t['stoch_d'].round(1)
    t['obv'] = (t['obv']/1e6).round(1)
    t = t.rename(columns={'obv': 'obv_M'})
    for c in ['chg_1h','chg_4h','chg_1d','chg_1w']:
        t[c] = t[c].round(2)
    # 패턴 라벨
    pat_map = {0: '-', 1: 'HAM', -1: 'SHOOT', 2: 'DOJI', -2: 'DOJI', 3: 'ENGULF+', -3: 'ENGULF-'}
    t['candle_pattern'] = t['candle_pattern'].map(pat_map).fillna('-')
    t = t.rename(columns={'candle_pattern': 'pattern'})
    return t.to_string()


def build_mtf_section(df30, df1h, df4h, df1d, idx_30m, bars=50):
    """다중 타임프레임 섹션 (각 50봉)"""
    ts = df30.index[idx_30m]

    sections = []
    for label, df_tf, b in [('1H', df1h, bars), ('4H', df4h, min(bars, 30)), ('1D', df1d, min(bars, 20))]:
        # 해당 시점 이전 데이터
        mask = df_tf.index <= ts
        if mask.sum() < 5:
            continue
        chunk = df_tf[mask].tail(b)
        cols = ['close', 'ema7', 'ema21', 'ema50', 'adx', 'rsi', 'macd_hist', 'trend']
        available = [c for c in cols if c in chunk.columns]
        t = chunk[available].copy()
        t.index = t.index.strftime('%m-%d %H:%M')
        for c in ['close', 'ema7', 'ema21', 'ema50']:
            if c in t.columns: t[c] = t[c].round(0).astype(int)
        for c in ['adx', 'rsi']:
            if c in t.columns: t[c] = t[c].round(1)
        if 'macd_hist' in t.columns: t['macd_hist'] = t['macd_hist'].round(1)
        if 'trend' in t.columns:
            t['trend'] = t['trend'].map({1: 'UP', -1: 'DN'})
        sections.append(f"[{label} - {len(t)} candles]\n{t.to_string()}")

    return "\n\n".join(sections)


def build_context_section(df30, idx, trade_history):
    """피보나치, 지지/저항, 다이버전스, 거래이력"""
    start = max(0, idx - HISTORY_BARS + 1)
    chunk = df30.iloc[start:idx+1]
    c = chunk['close']; h = chunk['high']; l = chunk['low']

    lines = []

    # 피보나치
    high_100 = h.max(); low_100 = l.min()
    fib = calc_fibonacci(high_100, low_100)
    fib_str = " | ".join([f"{k}={v}" for k, v in fib.items()])
    lines.append(f"Fibonacci (100-bar range): {fib_str}")

    # 지지/저항
    sr = find_support_resistance(h, l, c)
    if sr:
        r_levels = sorted(set([lv for t, lv in sr if t == 'R']), reverse=True)[:5]
        s_levels = sorted(set([lv for t, lv in sr if t == 'S']))[:5]
        lines.append(f"Resistance: {r_levels}")
        lines.append(f"Support: {s_levels}")

    # 다이버전스
    rsi_s = chunk['rsi'] if 'rsi' in chunk.columns else None
    macd_s = chunk['macd_hist'] if 'macd_hist' in chunk.columns else None
    if rsi_s is not None:
        rsi_div = detect_divergence(c, rsi_s)
        lines.append(f"RSI Divergence: {rsi_div}")
    if macd_s is not None:
        macd_div = detect_divergence(c, macd_s)
        lines.append(f"MACD Divergence: {macd_div}")

    # 최근 거래 이력
    if trade_history:
        recent = trade_history[-5:]
        th_lines = []
        for t in recent:
            th_lines.append(f"  {t['dir']} entry=${t['entry']:,.0f} exit=${t['exit']:,.0f} pnl=${t['pnl']:+,.0f} ({t['reason']})")
        lines.append(f"Recent Trades ({len(recent)}):\n" + "\n".join(th_lines))

    return "\n".join(lines)


def build_full_prompt(df30, df1h, df4h, df1d, idx, event_type, balance, pos_str,
                      monthly_pnl, trade_history, tsl_info=""):
    """V2 풀 프롬프트"""
    row = df30.iloc[idx]
    ts = df30.index[idx]

    main_table = build_main_table(df30, idx)
    extra_table = build_extra_table(df30, idx)
    mtf_section = build_mtf_section(df30, df1h, df4h, df1d, idx)
    context = build_context_section(df30, idx, trade_history)

    prompt = f"""=== BTC/USDT 30분봉 {event_type} ===
Time: {ts}

=== Current Indicators ===
Price: ${row['close']:,.1f}
EMA: 3={row['ema3']:.0f} | 7={row['ema7']:.0f} | 21={row['ema21']:.0f} | 100={row['ema100']:.0f} | 200={row['ema200']:.0f}
ADX: {row['adx']:.1f} | +DI: {row['plus_di']:.1f} | -DI: {row['minus_di']:.1f} | ATR: {row['atr']:.1f}
RSI: {row['rsi']:.1f} | StochRSI: K={row['stoch_k']:.1f} D={row['stoch_d']:.1f}
MACD: {row['macd']:.1f} | Signal: {row['macd_signal']:.1f} | Hist: {row['macd_hist']:.1f}
BB: Upper={row['bb_upper']:.0f} | Mid={row['bb_mid']:.0f} | Lower={row['bb_lower']:.0f}
Change: 1h={row['chg_1h']:+.2f}% | 4h={row['chg_4h']:+.2f}% | 1D={row['chg_1d']:+.2f}% | 1W={row['chg_1w']:+.2f}%

=== Account ===
Balance: ${balance:,.0f} | Position: {pos_str} | Monthly PnL: {monthly_pnl:+.1f}%
{tsl_info}

=== Context ===
{context}

=== 30m Main Indicators ({HISTORY_BARS} bars) ===
{main_table}

=== 30m Extra Indicators ({HISTORY_BARS} bars) ===
{extra_table}

=== Multi-Timeframe ===
{mtf_section}

=== Rules ===
Golden Cross + ADX>=35 + RSI 30~65 -> LONG
Dead Cross + ADX>=35 + RSI 30~65 -> SHORT
ADX 20~34 -> judge using ALL indicators (MACD, BB, StochRSI, MTF, patterns, divergence)
ADX < 20 -> HOLD
Monthly loss > -20% -> HOLD
Same direction held -> evaluate re-entry need

JSON only:
{{"action": "LONG/SHORT/CLOSE/HOLD", "reason": "one line reason in Korean"}}"""
    return prompt


def build_tsl_prompt_v2(df30, df1h, df4h, df1d, idx, pos_dir, entry_price,
                        peak_price, tsl_level, current_roi, balance, monthly_pnl, trade_history):
    """V2 TSL 프롬프트"""
    row = df30.iloc[idx]
    ts = df30.index[idx]
    dir_str = "LONG" if pos_dir == 1 else "SHORT"
    if pos_dir == 1:
        dist_pct = (row['close'] - tsl_level) / row['close'] * 100
    else:
        dist_pct = (tsl_level - row['close']) / row['close'] * 100

    main_table = build_main_table(df30, idx)
    extra_table = build_extra_table(df30, idx)
    mtf_section = build_mtf_section(df30, df1h, df4h, df1d, idx)
    context = build_context_section(df30, idx, trade_history)

    tsl_info = f"""=== TSL Status ===
Direction: {dir_str} | Entry: ${entry_price:,.0f} | Current: ${row['close']:,.0f}
Peak: ${peak_price:,.0f} | TSL Level: ${tsl_level:,.0f} ({dist_pct:.1f}% away)
ROI: {current_roi:+.1f}%"""

    prompt = f"""=== BTC/USDT 30m TSL Proximity Alert ===
Time: {ts}

{tsl_info}

=== Current Indicators ===
EMA: 3={row['ema3']:.0f} | 7={row['ema7']:.0f} | 21={row['ema21']:.0f} | 100={row['ema100']:.0f} | 200={row['ema200']:.0f}
ADX: {row['adx']:.1f} | +DI: {row['plus_di']:.1f} | -DI: {row['minus_di']:.1f} | ATR: {row['atr']:.1f}
RSI: {row['rsi']:.1f} | StochRSI: K={row['stoch_k']:.1f} D={row['stoch_d']:.1f}
MACD: {row['macd']:.1f} | Signal: {row['macd_signal']:.1f} | Hist: {row['macd_hist']:.1f}
BB: Upper={row['bb_upper']:.0f} | Mid={row['bb_mid']:.0f} | Lower={row['bb_lower']:.0f}
Change: 1h={row['chg_1h']:+.2f}% | 4h={row['chg_4h']:+.2f}% | 1D={row['chg_1d']:+.2f}%

=== Account ===
Balance: ${balance:,.0f} | Monthly PnL: {monthly_pnl:+.1f}%

=== Context ===
{context}

=== 30m Main ({HISTORY_BARS} bars) ===
{main_table}

=== 30m Extra ({HISTORY_BARS} bars) ===
{extra_table}

=== Multi-Timeframe ===
{mtf_section}

=== Decision ===
TSL is {dist_pct:.1f}% away. Evaluate trend exhaustion vs continuation.
- CLOSE: take profit (trend exhausting)
- HOLD: keep position (trend continuing)

JSON only:
{{"action": "CLOSE/HOLD", "reason": "one line reason in Korean"}}"""
    return prompt


# ================================================================
# AI Client
# ================================================================
SYSTEM_PROMPT = """You are a 10-year veteran BTC/USDT futures quant trader with deep expertise in:
- Multi-timeframe analysis (30m/1h/4h/1D)
- EMA cross + ADX/RSI/MACD/BB/StochRSI
- Volume analysis (OBV), ATR volatility
- Candle patterns, Fibonacci, Support/Resistance
- RSI/MACD divergence detection
- Risk management (monthly loss limits, trailing stops)

[Rules]
- ADX>=35 + RSI 30~65: strong signal -> follow cross direction
- ADX 20~34: use ALL indicators to judge (MTF alignment, MACD momentum, StochRSI, BB position, divergence, patterns)
- ADX < 20: HOLD
- RSI outside 30~65: HOLD
- Monthly loss > -20%: HOLD
- Check MTF alignment: if 1h/4h/1D all disagree with 30m signal -> HOLD
- Check divergence: bearish divergence on LONG signal -> HOLD

[TSL queries]
- Check MACD histogram trend, ADX strength, StochRSI, MTF alignment
- Weakening trend -> CLOSE
- Strong trend continuing -> HOLD

[Output]
JSON ONLY: {"action": "LONG/SHORT/CLOSE/HOLD", "reason": "한 줄 이유"}"""


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


def ask_gpt(client, prompt, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model="gpt-5.4", temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}]
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            wait = min(30, 5 * (attempt + 1))
            print(f"    [GPT retry {attempt+1}] {type(e).__name__} -> {wait}s")
            if attempt < retries - 1: time.sleep(wait)
    return None


def ask_claude(client, prompt, retries=3):
    for attempt in range(retries):
        try:
            r = client.messages.create(
                model="claude-sonnet-4-6", max_tokens=300, temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            text = r.content[0].text
            s = text.find('{'); e = text.rfind('}') + 1
            if s >= 0 and e > s:
                return json.loads(text[s:e])
            return None
        except Exception as e:
            wait = min(60, 10 * (attempt + 1))
            print(f"    [Claude retry {attempt+1}] {type(e).__name__} -> {wait}s")
            if attempt < retries - 1: time.sleep(wait)
    return None


def simulate_ai(pos_dir, signal, adx_val, rsi_val, macd_hist,
                plus_di, minus_di, stoch_k, trend_1h, trend_4h, tsl_proximity=False):
    """Dry-run simulation with MTF awareness"""
    if tsl_proximity:
        if adx_val < 25: return {"action": "CLOSE", "reason": "ADX 약화, 추세 소진"}
        return {"action": "HOLD", "reason": f"ADX {adx_val:.0f}, 추세 지속"}

    if pos_dir == signal:
        return {"action": "HOLD", "reason": f"이미 {signal} 보유 중"}
    if rsi_val < RSI_LOW or rsi_val > RSI_HIGH:
        return {"action": "HOLD", "reason": f"RSI {rsi_val:.0f} 범위 초과"}
    if adx_val >= 35:
        return {"action": signal, "reason": f"ADX {adx_val:.0f}>=35, 강한 추세"}
    if adx_val >= 30:
        macd_ok = (signal == "LONG" and macd_hist > 0) or (signal == "SHORT" and macd_hist < 0)
        di_ok = (signal == "LONG" and plus_di > minus_di) or (signal == "SHORT" and minus_di > plus_di)
        mtf_ok = (signal == "LONG" and trend_1h == 1) or (signal == "SHORT" and trend_1h == -1)
        if macd_ok and di_ok and mtf_ok:
            return {"action": signal, "reason": f"ADX {adx_val:.0f}+MACD+DI+1H MTF 일치"}
        return {"action": "HOLD", "reason": f"ADX {adx_val:.0f}, 보조지표 불일치"}
    return {"action": "HOLD", "reason": f"ADX {adx_val:.0f}<30, 추세 불충분"}


# ================================================================
# Bot class (same as v1)
# ================================================================
class Bot:
    def __init__(self, name):
        self.name = name; self.bal = IC; self.pos = 0; self.epx = 0.0
        self.psz = 0.0; self.margin = 0.0; self.slp = 0.0
        self.tsl_on = False; self.peak = 0.0; self.trough = 999999.0; self.tsl_level = 0.0
        self.m_start = IC; self.cur_m = ""
        self.sl_c = 0; self.tsl_c = 0; self.rev_c = 0; self.fl_c = 0
        self.trades = []; self.monthly = {}; self.pk_bal = IC; self.mdd = 0.0; self.yr_bal = {}
        self.tsl_ai_asked = False; self.tsl_ai_hold = False

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

    def enter(self, direction, price, ts):
        if self.bal <= 0: return False
        margin = self.bal * MARGIN_PCT
        if margin < 1: return False
        if self.m_start > 0 and (self.bal - self.m_start) / self.m_start <= ML_LIMIT: return False
        self.pos = direction; self.epx = price; self.margin = margin; self.psz = margin * LEVERAGE
        self.bal -= self.psz * FEE; self.tsl_on = False; self.tsl_ai_asked = False; self.tsl_ai_hold = False
        self.peak = price; self.trough = price; self.tsl_level = 0.0
        self.slp = price * (1 - SL_PCT/100) if direction == 1 else price * (1 + SL_PCT/100)
        if self.cur_m in self.monthly: self.monthly[self.cur_m]['ent'] += 1
        return True

    def close(self, price, reason, ts):
        if self.pos == 0: return 0.0
        pnl = -(self.margin + self.psz * FEE) if reason == 'fl' else \
              (price - self.epx) / self.epx * self.psz * self.pos - self.psz * FEE
        self.bal = max(0, self.bal + pnl)
        mk = self.cur_m
        if mk in self.monthly:
            self.monthly[mk]['pnl'] += pnl
            self.monthly[mk]['w' if pnl > 0 else 'l'] += 1
            if reason in ('sl','tsl','rev','fl'): self.monthly[mk][reason] += 1
        for attr in ['sl_c','tsl_c','rev_c','fl_c']:
            if reason == attr.replace('_c',''): setattr(self, attr, getattr(self, attr) + 1)
        exit_px = price if reason != 'fl' else (self.epx * (1-FL_PCT/100) if self.pos==1 else self.epx * (1+FL_PCT/100))
        self.trades.append({'ts':str(ts),'dir':'LONG' if self.pos==1 else 'SHORT',
                           'entry':self.epx,'exit':exit_px,'pnl':round(pnl,2),'reason':reason,'bal':round(self.bal,2)})
        self.pos=0; self.epx=0; self.psz=0; self.margin=0
        self.tsl_on=False; self.tsl_ai_asked=False; self.tsl_ai_hold=False
        self.update_dd()
        return pnl

    def check_exit(self, hi, lo, cl):
        if self.pos == 0: return None
        worst = ((lo-self.epx)/self.epx*100) if self.pos==1 else ((self.epx-hi)/self.epx*100)
        if worst <= -FL_PCT: return ('fl', None)
        if not self.tsl_on:
            if self.pos==1 and lo<=self.slp: return ('sl', self.slp)
            if self.pos==-1 and hi>=self.slp: return ('sl', self.slp)
        best = ((hi-self.epx)/self.epx*100) if self.pos==1 else ((self.epx-lo)/self.epx*100)
        if best >= TSL_ACT: self.tsl_on = True
        if self.tsl_on:
            if self.pos==1:
                self.peak = max(self.peak, hi); self.tsl_level = self.peak*(1-TSL_WIDTH/100)
                self.slp = max(self.slp, self.tsl_level)
                if cl <= self.tsl_level: return ('tsl', cl)
            else:
                self.trough = min(self.trough, lo); self.tsl_level = self.trough*(1+TSL_WIDTH/100)
                self.slp = min(self.slp, self.tsl_level)
                if cl >= self.tsl_level: return ('tsl', cl)
        return None

    def is_tsl_near(self, close):
        if not self.tsl_on or self.pos==0: return False
        dist = ((close-self.tsl_level)/close*100) if self.pos==1 else ((self.tsl_level-close)/close*100)
        return 0 < dist <= TSL_PROXIMITY

    def current_roi(self, close):
        if self.pos==0: return 0.0
        return ((close-self.epx)/self.epx if self.pos==1 else (self.epx-close)/self.epx) * LEVERAGE * 100

    def pos_str(self):
        return "None" if self.pos==0 else f"{'LONG' if self.pos==1 else 'SHORT'} @ ${self.epx:,.0f}"

    def stats(self):
        n=len(self.trades); w=sum(1 for t in self.trades if t['pnl']>0); l=n-w
        gp=sum(t['pnl'] for t in self.trades if t['pnl']>0)
        gl=abs(sum(t['pnl'] for t in self.trades if t['pnl']<=0))
        return {'bal':self.bal,'ret':(self.bal-IC)/IC*100,'pf':gp/max(gl,.001),'mdd':self.mdd*100,
                'n':n,'w':w,'l':l,'wr':w/max(n,1)*100,'sl':self.sl_c,'tsl':self.tsl_c,'rev':self.rev_c,'fl':self.fl_c}


# ================================================================
# Main
# ================================================================
def get_mtf_trend(df_tf, ts):
    """특정 시점의 MTF 트렌드 가져오기"""
    mask = df_tf.index <= ts
    if mask.sum() == 0: return 0
    return int(df_tf.loc[mask, 'trend'].iloc[-1]) if 'trend' in df_tf.columns else 0


args_ref = [None]  # 배치 모드 참조용

def run(dry_run=False):
    print("=" * 70)
    print("  BTC/USDT v14.4 Enhanced AI Backtest V2 - FULL INDICATORS")
    print(f"  Period: 2023~ | Dry-run: {dry_run}")
    print("=" * 70)

    df30, df1h, df4h, df1d = load_and_prepare()

    print("[3/4] AI clients...")
    gpt_cli, cld_cli = init_ai(dry_run)
    if dry_run:
        print("  DRY-RUN mode")
    else:
        print(f"  GPT-5.4: {'OK' if gpt_cli else 'NO KEY'} | Claude: {'OK' if cld_cli else 'NO KEY'}")

    ts_arr = df30.index; n = len(df30)
    start_i = max(WARMUP, np.searchsorted(ts_arr, pd.Timestamp('2023-01-01')))
    end_i = n
    print(f"  Range: {ts_arr[start_i]} ~ {ts_arr[end_i-1]} ({end_i-start_i:,} candles)")

    cl = df30['close'].values; hi = df30['high'].values; lo = df30['low'].values
    adx_v = df30['adx'].values; rsi_v = df30['rsi'].values
    gx = df30['cross_up'].values; dx = df30['cross_dn'].values

    code = Bot("CodeBot"); gpt_bot = Bot("GPT-5.4"); cla_bot = Bot("Claude")
    bots = [code, gpt_bot, cla_bot]
    cross_log = []; tsl_log = []; api_calls = 0; cross_cnt = 0; tsl_cnt = 0
    last_cross_dir = 0

    # 체크포인트 이어하기
    import pickle
    CKPT = "D:/filesystem/futures/btc_V1/test1/v2_checkpoint.pkl"
    resume_after_cross = 0
    if os.path.exists(CKPT):
        try:
            ckpt = pickle.load(open(CKPT, 'rb'))
            code.__dict__.update(ckpt['code']); gpt_bot.__dict__.update(ckpt['gpt']); cla_bot.__dict__.update(ckpt['claude'])
            cross_log = ckpt['cross_log']; tsl_log = ckpt['tsl_log']
            api_calls = ckpt['api_calls']; cross_cnt = ckpt['cross_cnt']; tsl_cnt = ckpt['tsl_cnt']
            last_cross_dir = ckpt['last_cross_dir']; resume_after_cross = ckpt['cross_cnt']
            print(f"  [RESUME] checkpoint loaded: {resume_after_cross} crosses done")
            print(f"  Code=${code.bal:,.0f} GPT=${gpt_bot.bal:,.0f} Claude=${cla_bot.bal:,.0f}")
        except Exception as e:
            print(f"  [RESUME] failed: {e}, starting fresh")

    print(f"\n[4/4] Running backtest...\n")

    for i in range(start_i, end_i):
        t = ts_arr[i]; h_ = hi[i]; l_ = lo[i]; c_ = cl[i]
        mk = f"{t.year}-{t.month:02d}"
        for b in bots: b.update_month(mk)

        # SL/TSL/FL
        for b in bots:
            if b.name != "CodeBot" and b.tsl_ai_hold:
                ex = b.check_exit(h_, l_, c_)
                if ex and ex[0] in ('fl','sl'): b.close(ex[1] if ex[1] else c_, ex[0], t)
                elif ex and ex[0] == 'tsl': b.tsl_ai_hold = False
                continue
            ex = b.check_exit(h_, l_, c_)
            if ex:
                r, px = ex
                b.close(px if px else c_, r, t)

        # TSL proximity for AI bots
        for b in [gpt_bot, cla_bot]:
            if b.is_tsl_near(c_) and not b.tsl_ai_asked:
                b.tsl_ai_asked = True; tsl_cnt += 1
                peak_p = b.peak if b.pos==1 else b.trough
                roi = b.current_roi(c_)
                if dry_run:
                    result = simulate_ai(0, None, adx_v[i], rsi_v[i], 0, 0, 0, 0, 0, 0, tsl_proximity=True)
                elif b.name == "GPT-5.4" and gpt_cli:
                    prompt = build_tsl_prompt_v2(df30, df1h, df4h, df1d, i, b.pos, b.epx, peak_p, b.tsl_level, roi, b.bal, b.monthly_pnl_pct(), b.trades)
                    result = ask_gpt(gpt_cli, prompt); api_calls += 1
                elif b.name == "Claude" and cld_cli:
                    prompt = build_tsl_prompt_v2(df30, df1h, df4h, df1d, i, b.pos, b.epx, peak_p, b.tsl_level, roi, b.bal, b.monthly_pnl_pct(), b.trades)
                    result = ask_claude(cld_cli, prompt); api_calls += 1
                else: result = None
                if result:
                    act = result.get('action','HOLD'); rsn = result.get('reason','')
                    if act == 'HOLD': b.tsl_ai_hold = True
                    tsl_log.append({'timestamp':str(t),'bot':b.name,'direction':'LONG' if b.pos==1 else 'SHORT',
                                   'roi':round(roi,1),'ai_action':act,'ai_reason':rsn})
                    print(f"  [TSL] {str(t)[:16]} | {b.name} | ROI={roi:+.1f}% | AI={act} | {rsn[:50]}")

        # Cross
        is_gx = bool(gx[i]); is_dx = bool(dx[i])
        if not (is_gx or is_dx):
            for b in bots: b.yr_bal[t.year] = b.bal
            continue

        a_v = adx_v[i]; r_v = rsi_v[i]; new_dir = 1 if is_gx else -1
        if new_dir == last_cross_dir:
            for b in bots: b.yr_bal[t.year] = b.bal
            continue
        last_cross_dir = new_dir

        code_trigger = a_v >= ADX_THRESH_CODE and RSI_LOW <= r_v <= RSI_HIGH
        ai_trigger = a_v >= ADX_THRESH_AI
        if not (ai_trigger or code_trigger):
            for b in bots: b.yr_bal[t.year] = b.bal
            continue

        cross_cnt += 1

        # 이어하기: 이미 처리된 크로스 스킵 (코드봇은 SL/TSL에서 이미 처리됨)
        if cross_cnt <= resume_after_cross:
            # 코드봇만 진행 (SL/TSL은 위에서 처리, 진입/REV만)
            if code_trigger:
                if code.pos != 0 and code.pos != new_dir: code.close(c_, 'rev', t)
                code.enter(new_dir, c_, t)
            for b in bots: b.yr_bal[t.year] = b.bal
            continue

        cross_type = "Golden Cross" if is_gx else "Dead Cross"
        signal_str = "LONG" if new_dir == 1 else "SHORT"
        mh = df30.iloc[i]['macd_hist']

        # Code bot
        code_action = "HOLD"
        if code_trigger:
            if code.pos != 0 and code.pos != new_dir: code.close(c_, 'rev', t)
            if code.enter(new_dir, c_, t): code_action = signal_str

        # MTF trends
        t1h = get_mtf_trend(df1h, t); t4h = get_mtf_trend(df4h, t)

        # GPT
        gpt_action = "HOLD"; gpt_reason = ""
        if ai_trigger:
            if dry_run:
                result = simulate_ai(gpt_bot.pos, signal_str, a_v, r_v, mh,
                                     df30.iloc[i]['plus_di'], df30.iloc[i]['minus_di'],
                                     df30.iloc[i]['stoch_k'], t1h, t4h)
            elif gpt_cli:
                prompt = build_full_prompt(df30, df1h, df4h, df1d, i, cross_type,
                                          gpt_bot.bal, gpt_bot.pos_str(), gpt_bot.monthly_pnl_pct(), gpt_bot.trades)
                result = ask_gpt(gpt_cli, prompt); api_calls += 1
            else: result = None
            if result: gpt_action = result.get('action','HOLD'); gpt_reason = result.get('reason','')

        # Claude
        cla_action = "HOLD"; cla_reason = ""
        if ai_trigger:
            if dry_run:
                result = simulate_ai(cla_bot.pos, signal_str, a_v, r_v, mh,
                                     df30.iloc[i]['plus_di'], df30.iloc[i]['minus_di'],
                                     df30.iloc[i]['stoch_k'], t1h, t4h)
            elif cld_cli:
                prompt = build_full_prompt(df30, df1h, df4h, df1d, i, cross_type,
                                          cla_bot.bal, cla_bot.pos_str(), cla_bot.monthly_pnl_pct(), cla_bot.trades)
                result = ask_claude(cld_cli, prompt); api_calls += 1
            else: result = None
            if result: cla_action = result.get('action','HOLD'); cla_reason = result.get('reason','')

        # AI bot execution
        for b, act in [(gpt_bot, gpt_action), (cla_bot, cla_action)]:
            if act in ("LONG","SHORT"):
                d = 1 if act == "LONG" else -1
                if b.pos != 0 and b.pos != d: b.close(c_, 'rev', t)
                b.enter(d, c_, t)

        cross_log.append({'timestamp':str(t),'cross':cross_type,'price':round(c_,1),
                         'adx':round(a_v,1),'rsi':round(r_v,1),
                         'code_action':code_action,'gpt_action':gpt_action,'gpt_reason':gpt_reason,
                         'claude_action':cla_action,'claude_reason':cla_reason,
                         'code_bal':round(code.bal,0),'gpt_bal':round(gpt_bot.bal,0),'claude_bal':round(cla_bot.bal,0)})

        adx_m = "*" if a_v >= ADX_THRESH_CODE else " "
        print(f"[{cross_cnt:>3}] {str(t)[:16]} | {cross_type:>12} | ADX={a_v:5.1f}{adx_m} RSI={r_v:5.1f} | "
              f"Code={code_action:>5} GPT={gpt_action:>5} Claude={cla_action:>6} | "
              f"${code.bal:>8,.0f} ${gpt_bot.bal:>8,.0f} ${cla_bot.bal:>8,.0f}")

        for b in bots: b.yr_bal[t.year] = b.bal

        # 체크포인트 저장 (10건마다)
        if cross_cnt % 10 == 0:
            pickle.dump({
                'code': code.__dict__, 'gpt': gpt_bot.__dict__, 'claude': cla_bot.__dict__,
                'cross_log': cross_log, 'tsl_log': tsl_log,
                'api_calls': api_calls, 'cross_cnt': cross_cnt, 'tsl_cnt': tsl_cnt,
                'last_cross_dir': last_cross_dir,
            }, open(CKPT, 'wb'))
            print(f"  [CHECKPOINT] {cross_cnt}/772 saved")

        # 배치 모드: --batch N 옵션시 N건 후 자동 종료
        if hasattr(args_ref[0], 'batch') and args_ref[0].batch and (cross_cnt - resume_after_cross) >= args_ref[0].batch:
            print(f"  [BATCH] {args_ref[0].batch} crosses done, stopping for resume")
            pickle.dump({
                'code': code.__dict__, 'gpt': gpt_bot.__dict__, 'claude': cla_bot.__dict__,
                'cross_log': cross_log, 'tsl_log': tsl_log,
                'api_calls': api_calls, 'cross_cnt': cross_cnt, 'tsl_cnt': tsl_cnt,
                'last_cross_dir': last_cross_dir,
            }, open(CKPT, 'wb'))
            return  # 배치 종료

    # 완료 - 체크포인트 삭제
    if os.path.exists(CKPT): os.remove(CKPT)

    # Close remaining
    last_px = cl[end_i-1]; last_t = ts_arr[end_i-1]
    for b in bots:
        if b.pos != 0: b.close(last_px, 'end', last_t)
        if b.cur_m in b.monthly: b.monthly[b.cur_m]['eq'] = b.bal
        b.yr_bal[last_t.year] = b.bal

    # Report
    cs = code.stats(); gs = gpt_bot.stats(); ls = cla_bot.stats()
    print(f"\n{'='*75}")
    print(f"  V2 FULL INDICATORS BACKTEST REPORT (2023~)")
    print(f"{'='*75}")
    print(f"  {'':>16} {'CodeBot':>14} {'GPT-5.4':>14} {'Claude':>14}")
    print(f"  {'-'*60}")
    for label, key in [('Balance','bal'),('Return%','ret'),('PF','pf'),('MDD%','mdd'),
                       ('Trades','n'),('Win Rate%','wr'),('SL','sl'),('TSL','tsl'),('REV','rev')]:
        cv=cs[key]; gv=gs[key]; lv=ls[key]
        if key=='bal': print(f"  {label:>16} ${cv:>12,.0f} ${gv:>12,.0f} ${lv:>12,.0f}")
        elif key in ('ret','mdd','wr'): print(f"  {label:>16} {cv:>13.1f}% {gv:>13.1f}% {lv:>13.1f}%")
        elif key=='pf': print(f"  {label:>16} {cv:>14.2f} {gv:>14.2f} {lv:>14.2f}")
        else: print(f"  {label:>16} {cv:>14} {gv:>14} {lv:>14}")

    # Yearly
    all_yrs = sorted(set(list(code.yr_bal)+list(gpt_bot.yr_bal)+list(cla_bot.yr_bal)))
    print(f"\n  [Yearly]")
    for y in all_yrs:
        print(f"  {y}: Code ${code.yr_bal.get(y,0):>10,.0f} | GPT ${gpt_bot.yr_bal.get(y,0):>10,.0f} | Claude ${cla_bot.yr_bal.get(y,0):>10,.0f}")

    # Ranking
    ranking = sorted([(cs['bal'],'CodeBot'),(gs['bal'],'GPT-5.4'),(ls['bal'],'Claude')], reverse=True)
    print(f"\n  RANKING: ", end="")
    for i,(bal,name) in enumerate(ranking): print(f"{i+1}.{name}(${bal:,.0f})", end="  ")
    print(f"\n  API calls: {api_calls} | Crosses: {cross_cnt} | TSL queries: {tsl_cnt}")
    print(f"{'='*75}")

    # Save
    base = "D:/filesystem/futures/btc_V1/test1"
    if cross_log:
        pd.DataFrame(cross_log).to_csv(f"{base}/v2_ai_trades.csv", index=False, encoding='utf-8-sig')
        print(f"Saved: v2_ai_trades.csv ({len(cross_log)} rows)")
    if tsl_log:
        pd.DataFrame(tsl_log).to_csv(f"{base}/v2_ai_tsl.csv", index=False, encoding='utf-8-sig')
        print(f"Saved: v2_ai_tsl.csv ({len(tsl_log)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--batch', type=int, default=0, help='Process N crosses then stop (0=all)')
    args = parser.parse_args()
    args_ref[0] = args
    run(dry_run=args.dry_run)
