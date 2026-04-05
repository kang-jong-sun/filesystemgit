"""
GPT-5 제안 8대 실험 통합 백테스트 엔진
1. 시장 레짐 분류 (추세/횡보/고변동)
2. ATR 기반 동적 SL + Chandelier Exit
3. 풀백 재진입 (Stoch + EMA20/50)
4. BB Squeeze 돌파 (BB + KC + Volume)
5. 다단계 분할청산 (40/30/30)
6. 리스크 기반 포지션 사이징 (계정위험 1%)
7. 피라미딩 (+1R 후 추가, 최대 2회)
8. 포트폴리오 동시 운용
"""
import sys, time, json, os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = r"D:\filesystem\futures\btc_V1\test1\btc_usdt_5m_merged.csv"
OUTPUT_DIR = r"D:\filesystem\futures\btc_V1\test1\gpt5_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# INDICATORS
# ============================================================
def ema(data, p):
    return pd.Series(data).ewm(span=p, adjust=False).mean().values

def sma(data, p):
    return pd.Series(data).rolling(p, min_periods=p).mean().values

def wma(data, p):
    w = np.arange(1, p+1, dtype=float)
    return pd.Series(data).rolling(p).apply(lambda x: np.dot(x, w)/w.sum(), raw=True).values

def rsi(close, p=14):
    d = np.diff(close, prepend=close[0])
    g = np.where(d>0, d, 0.0); l = np.where(d<0, -d, 0.0)
    ga = pd.Series(g).ewm(alpha=1/p, min_periods=p, adjust=False).mean().values
    la = pd.Series(l).ewm(alpha=1/p, min_periods=p, adjust=False).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(la>0, ga/la, 100.0)
    return 100 - 100/(1+rs)

def adx(high, low, close, p=14):
    tr1=high-low; tr2=np.abs(high-np.roll(close,1)); tr3=np.abs(low-np.roll(close,1))
    tr1[0]=tr2[0]=tr3[0]=0
    tr=np.maximum(np.maximum(tr1,tr2),tr3)
    up=high-np.roll(high,1); dn=np.roll(low,1)-low; up[0]=dn[0]=0
    pdm=np.where((up>dn)&(up>0),up,0.0); mdm=np.where((dn>up)&(dn>0),dn,0.0)
    a=1.0/p
    atr_v=pd.Series(tr).ewm(alpha=a,min_periods=p,adjust=False).mean().values
    pds=pd.Series(pdm).ewm(alpha=a,min_periods=p,adjust=False).mean().values
    mds=pd.Series(mdm).ewm(alpha=a,min_periods=p,adjust=False).mean().values
    with np.errstate(divide='ignore',invalid='ignore'):
        pdi=np.where(atr_v>0,100*pds/atr_v,0); mdi=np.where(atr_v>0,100*mds/atr_v,0)
        ds=pdi+mdi; dx=np.where(ds>0,100*np.abs(pdi-mdi)/ds,0)
    return pd.Series(dx).ewm(alpha=a,min_periods=p,adjust=False).mean().values

def atr(high, low, close, p=14):
    tr1=high-low; tr2=np.abs(high-np.roll(close,1)); tr3=np.abs(low-np.roll(close,1))
    tr1[0]=tr2[0]=tr3[0]=0
    tr=np.maximum(np.maximum(tr1,tr2),tr3)
    return pd.Series(tr).ewm(alpha=1.0/p,min_periods=p,adjust=False).mean().values

def stochastic(high, low, close, k_period=14, d_period=3, smooth=3):
    """Stochastic oscillator: %K and %D"""
    n = len(close)
    raw_k = np.full(n, 50.0)
    for i in range(k_period-1, n):
        hh = np.max(high[i-k_period+1:i+1])
        ll = np.min(low[i-k_period+1:i+1])
        if hh != ll:
            raw_k[i] = (close[i] - ll) / (hh - ll) * 100
    k = pd.Series(raw_k).rolling(smooth, min_periods=1).mean().values
    d = pd.Series(k).rolling(d_period, min_periods=1).mean().values
    return k, d

def bollinger(close, p=20, std=2.0):
    s = pd.Series(close)
    mid = s.rolling(p, min_periods=p).mean().values
    sd = s.rolling(p, min_periods=p).std().values
    upper = mid + std * sd
    lower = mid - std * sd
    with np.errstate(divide='ignore', invalid='ignore'):
        width = np.where(mid > 0, (upper - lower) / mid, 0)
    return upper, mid, lower, width

def keltner(close, high, low, p=20, atr_mult=1.5):
    mid = ema(close, p)
    atr_v = atr(high, low, close, p)
    upper = mid + atr_mult * atr_v
    lower = mid - atr_mult * atr_v
    return upper, mid, lower

def obv(close, volume):
    """On-Balance Volume"""
    n = len(close)
    o = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]: o[i] = o[i-1] + volume[i]
        elif close[i] < close[i-1]: o[i] = o[i-1] - volume[i]
        else: o[i] = o[i-1]
    return o

def cci(high, low, close, p=20):
    tp = (high + low + close) / 3.0
    tp_sma = sma(tp, p)
    md = pd.Series(tp).rolling(p).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(md > 0, (tp - tp_sma) / (0.015 * md), 0)

def ichimoku(high, low, close, tenkan_p=9, kijun_p=26, senkou_p=52):
    n = len(close)
    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    span_a = np.full(n, np.nan)
    span_b = np.full(n, np.nan)

    for i in range(tenkan_p-1, n):
        tenkan[i] = (np.max(high[i-tenkan_p+1:i+1]) + np.min(low[i-tenkan_p+1:i+1])) / 2
    for i in range(kijun_p-1, n):
        kijun[i] = (np.max(high[i-kijun_p+1:i+1]) + np.min(low[i-kijun_p+1:i+1])) / 2

    for i in range(kijun_p-1, n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            span_a[i] = (tenkan[i] + kijun[i]) / 2
    for i in range(senkou_p-1, n):
        span_b[i] = (np.max(high[i-senkou_p+1:i+1]) + np.min(low[i-senkou_p+1:i+1])) / 2

    cloud_top = np.maximum(np.nan_to_num(span_a, 0), np.nan_to_num(span_b, 0))
    cloud_bot = np.minimum(np.where(np.isnan(span_a), 1e18, span_a),
                           np.where(np.isnan(span_b), 1e18, span_b))
    return tenkan, kijun, span_a, span_b, cloud_top, cloud_bot

def chandelier_exit(high, low, close, atr_v, period=22, mult=3.0):
    """Chandelier Exit: trailing stop based on highest high - ATR*mult"""
    n = len(close)
    ce_long = np.full(n, np.nan)
    ce_short = np.full(n, np.nan)
    for i in range(period-1, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        ce_long[i] = hh - mult * atr_v[i]
        ce_short[i] = ll + mult * atr_v[i]
    return ce_long, ce_short

def regime_score(close, high, low, volume, adx_v, bb_width, ema200):
    """Market regime: 0-5 score (higher = more trending)"""
    n = len(close)
    score = np.zeros(n)
    obv_v = obv(close, volume)
    obv_ema = ema(obv_v, 20)
    ema50 = ema(close, 50)

    for i in range(200, n):
        s = 0
        if close[i] > ema200[i]: s += 1      # Above EMA200
        if ema50[i] > ema200[i]: s += 1       # EMA50 > EMA200
        if adx_v[i] > 20: s += 1              # Trending
        if adx_v[i] > 35: s += 1              # Strong trend
        if obv_v[i] > obv_ema[i]: s += 1      # Volume confirms
        score[i] = s
    return score


# ============================================================
# DATA LOADING
# ============================================================
def load_and_prepare():
    print('Loading data...')
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    print('  %d rows' % len(df))

    # Resample to multiple timeframes
    data = {}
    for name, mins in [('5min',5),('15min',15),('30min',30),('1h',60),('4h',240)]:
        if mins == 5:
            r = df.copy()
        else:
            d2 = df.set_index('timestamp')
            rule = '%dmin' % mins if mins < 60 else '%dh' % (mins//60)
            r = d2.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()

        ts = pd.to_datetime(r['timestamp'].values)
        cl = r['close'].values.astype(float)
        hi = r['high'].values.astype(float)
        lo = r['low'].values.astype(float)
        vo = r['volume'].values.astype(float)

        # Pre-compute all indicators
        data[name] = {
            'ts': r['timestamp'].values, 'cl': cl, 'hi': hi, 'lo': lo, 'vo': vo,
            'yr': ts.year.values.astype(np.int32),
            'mk': (ts.year.values*100 + ts.month.values).astype(np.int32),
            'n': len(r),
            'ema20': ema(cl,20), 'ema50': ema(cl,50), 'ema200': ema(cl,200),
            'wma3': wma(cl,3), 'sma14': sma(cl,14),
            'adx14': adx(hi,lo,cl,14), 'adx20': adx(hi,lo,cl,20),
            'rsi14': rsi(cl,14), 'rsi5': rsi(cl,5),
            'atr14': atr(hi,lo,cl,14),
            'stoch_k': stochastic(hi,lo,cl,14,3,3)[0],
            'stoch_d': stochastic(hi,lo,cl,14,3,3)[1],
            'bb_upper': bollinger(cl,20,2.0)[0],
            'bb_mid': bollinger(cl,20,2.0)[1],
            'bb_lower': bollinger(cl,20,2.0)[2],
            'bb_width': bollinger(cl,20,2.0)[3],
            'kc_upper': keltner(cl,hi,lo,20,1.5)[0],
            'obv': obv(cl,vo),
            'obv_ema20': ema(obv(cl,vo),20),
            'cci20': cci(hi,lo,cl,20),
        }

        # Ichimoku
        tk,kj,sa,sb,ct,cb = ichimoku(hi,lo,cl,9,26,52)
        data[name]['ichi_tenkan'] = tk
        data[name]['ichi_kijun'] = kj
        data[name]['ichi_cloud_top'] = ct
        data[name]['ichi_cloud_bot'] = cb
        data[name]['ichi_span_a'] = sa
        data[name]['ichi_span_b'] = sb

        # Chandelier Exit
        ce_l, ce_s = chandelier_exit(hi,lo,cl,data[name]['atr14'],22,3.0)
        data[name]['ce_long'] = ce_l
        data[name]['ce_short'] = ce_s

        # Regime
        data[name]['regime'] = regime_score(cl,hi,lo,vo,data[name]['adx14'],data[name]['bb_width'],data[name]['ema200'])

        # BB Squeeze (BB inside KC)
        data[name]['squeeze'] = data[name]['bb_upper'] < data[name]['kc_upper']

        # Swing high/low (5-bar)
        sh = np.zeros(len(cl)); sl_arr = np.full(len(cl), 1e18)
        for i in range(5, len(cl)-5):
            if hi[i] == np.max(hi[i-5:i+6]): sh[i] = hi[i]
            if lo[i] == np.min(lo[i-5:i+6]): sl_arr[i] = lo[i]
        # Running swing high/low
        rsh = np.zeros(len(cl)); rsl = np.full(len(cl), 1e18)
        for i in range(1, len(cl)):
            rsh[i] = max(rsh[i-1], sh[i]) if sh[i] > 0 else rsh[i-1]
            rsl[i] = min(rsl[i-1], sl_arr[i]) if sl_arr[i] < 1e18 else rsl[i-1]
        data[name]['swing_high'] = rsh
        data[name]['swing_low'] = rsl

        # 20-bar high/low for breakout
        hh20 = pd.Series(hi).rolling(20, min_periods=20).max().values
        ll20 = pd.Series(lo).rolling(20, min_periods=20).min().values
        data[name]['hh20'] = hh20
        data[name]['ll20'] = ll20

        # Volume SMA
        data[name]['vol_sma20'] = sma(vo, 20)

        print('  %s: %d bars, indicators computed' % (name, len(r)))

    return data


# ============================================================
# STRATEGY A: TREND CORE (Ichimoku + OBV + EMA200)
# ============================================================
def strategy_a_trend(d, htf_d=None, init_bal=5000.0, risk_pct=0.01, atr_sl_mult=2.2):
    """Trend following: Ichimoku + OBV + ADX on 30m, 4H filter"""
    cl,hi,lo = d['cl'],d['hi'],d['lo']
    yr,mk = d['yr'],d['mk']
    n = len(cl)
    bal = init_bal; peak = init_bal; mdd = 0.0
    trades = []; in_pos = False; p_dir = 0; p_entry = 0.0; p_size = 0.0
    p_sl = 0.0; p_high = 0.0; p_low = 0.0; p_stages = [1.0, 1.0, 1.0]  # 40/30/30
    p_add_count = 0; p_r = 0.0  # For pyramiding
    cur_m = -1; m_start = bal; m_locked = False

    # Higher TF filter (4H or 1H)
    htf_bullish = np.ones(n, dtype=bool)  # Default: always bullish if no HTF
    if htf_d is not None:
        # Map HTF trend to this TF
        htf_ema50 = htf_d.get('ema50', np.zeros(1))
        htf_ema200 = htf_d.get('ema200', np.zeros(1))
        htf_adx = htf_d.get('adx14', np.zeros(1))
        htf_ts = htf_d['ts']
        # Simple: use last known HTF bar
        htf_idx = 0
        for i in range(n):
            while htf_idx < len(htf_ts)-1 and htf_ts[htf_idx+1] <= d['ts'][i]:
                htf_idx += 1
            if htf_idx < len(htf_ema50):
                htf_bullish[i] = (htf_ema50[htf_idx] > htf_ema200[htf_idx]) and (htf_adx[htf_idx] > 20)

    fee = 0.0004
    warmup = 250

    for i in range(warmup, n):
        c = cl[i]; h = hi[i]; lo_v = lo[i]
        # Month check
        m = mk[i]
        if m != cur_m: cur_m = m; m_start = bal; m_locked = False
        if not m_locked and m_start > 0:
            if (bal - m_start) / m_start <= -0.20: m_locked = True

        atr_v = d['atr14'][i]
        if np.isnan(atr_v) or atr_v == 0: atr_v = c * 0.02

        if in_pos:
            # Chandelier trailing
            if p_dir == 1:
                if h > p_high: p_high = h
                ce = p_high - 3.0 * atr_v
                if ce > p_sl: p_sl = ce  # Ratchet up
                # SL check
                if lo_v <= p_sl:
                    pnl = p_size * (p_sl - p_entry) / p_entry * p_dir - p_size * fee
                    bal += pnl
                    reason = 'TSL' if p_sl > p_entry else 'SL'
                    trades.append({'yr':yr[i],'pnl':pnl,'reason':reason,'bal':bal})
                    in_pos = False
            else:
                if lo_v < p_low: p_low = lo_v
                ce = p_low + 3.0 * atr_v
                if ce < p_sl or p_sl == 0: p_sl = ce
                if hi[i] >= p_sl:
                    pnl = p_size * (p_entry - p_sl) / p_entry - p_size * fee
                    bal += pnl
                    reason = 'TSL' if p_sl < p_entry else 'SL'
                    trades.append({'yr':yr[i],'pnl':pnl,'reason':reason,'bal':bal})
                    in_pos = False

            # Pyramiding: if +1R and < 2 adds
            if in_pos and p_add_count < 2:
                curr_r = (c - p_entry) / p_entry * p_dir / (atr_sl_mult * atr_v / p_entry)
                if curr_r >= 1.0:
                    add_size = bal * risk_pct / (atr_sl_mult * atr_v / c)
                    add_size = min(add_size, bal * 0.15 * 10)  # Cap
                    bal -= add_size * fee
                    p_size += add_size
                    p_add_count += 1
                    p_entry = (p_entry * (p_size - add_size) + c * add_size) / p_size  # Avg entry

        # Signal detection
        if i < warmup + 1: continue

        sig = 0
        tk = d['ichi_tenkan'][i]; kj = d['ichi_kijun'][i]
        ct = d['ichi_cloud_top'][i]; sa = d['ichi_span_a'][i]; sb = d['ichi_span_b'][i]
        obv_ok = d['obv'][i] > d['obv_ema20'][i]
        adx_ok = d['adx14'][i] >= 25

        if not np.isnan(tk) and not np.isnan(kj) and not np.isnan(ct):
            # Long
            if (tk > kj and c > ct and sa > sb and obv_ok and adx_ok and htf_bullish[i]):
                # Check if just crossed (tk crossed above kj)
                tk_p = d['ichi_tenkan'][i-1]; kj_p = d['ichi_kijun'][i-1]
                if not np.isnan(tk_p) and not np.isnan(kj_p):
                    if tk > kj and tk_p <= kj_p: sig = 1
            # Short
            if (tk < kj and c < d['ichi_cloud_bot'][i] and sa < sb and not obv_ok and adx_ok and not htf_bullish[i]):
                tk_p = d['ichi_tenkan'][i-1]; kj_p = d['ichi_kijun'][i-1]
                if not np.isnan(tk_p) and not np.isnan(kj_p):
                    if tk < kj and tk_p >= kj_p: sig = -1

        if sig != 0:
            if in_pos and p_dir != sig:
                pnl = p_size * (c - p_entry) / p_entry * p_dir - p_size * fee
                bal += pnl
                trades.append({'yr':yr[i],'pnl':pnl,'reason':'REV','bal':bal})
                in_pos = False

            if not in_pos and not m_locked and bal > 10:
                sl_dist = atr_sl_mult * atr_v
                pos_size = bal * risk_pct / (sl_dist / c)
                pos_size = min(pos_size, bal * 0.25 * 15)  # Cap at M25% L15x
                bal -= pos_size * fee
                p_dir = sig; p_entry = c; p_size = pos_size
                p_sl = c - sl_dist if sig == 1 else c + sl_dist
                p_high = c; p_low = c; p_add_count = 0
                in_pos = True

        if bal > peak: peak = bal
        if peak > 0:
            dd = (peak - bal) / peak
            if dd > mdd: mdd = dd

    if in_pos:
        pnl = p_size * (cl[-1] - p_entry) / p_entry * p_dir - p_size * fee
        bal += pnl
        trades.append({'yr':yr[-1],'pnl':pnl,'reason':'END','bal':bal})

    return compile_results(trades, bal, init_bal, mdd, 'A_Trend')


# ============================================================
# STRATEGY B: PULLBACK RE-ENTRY (Stoch + EMA20/50)
# ============================================================
def strategy_b_pullback(d, htf_d=None, init_bal=5000.0, risk_pct=0.01, atr_sl_mult=1.5):
    cl,hi,lo = d['cl'],d['hi'],d['lo']
    yr,mk = d['yr'],d['mk']
    n = len(cl)
    bal = init_bal; peak = init_bal; mdd = 0.0
    trades = []; in_pos = False; p_dir = 0; p_entry = 0.0; p_size = 0.0
    p_sl = 0.0; p_high = 0.0; p_low = 0.0; p_tp1 = 0.0; p_tp1_done = False
    cur_m = -1; m_start = bal; m_locked = False
    fee = 0.0004; warmup = 250

    for i in range(warmup, n):
        c = cl[i]; h = hi[i]; lo_v = lo[i]
        m = mk[i]
        if m != cur_m: cur_m = m; m_start = bal; m_locked = False
        if not m_locked and m_start > 0:
            if (bal - m_start) / m_start <= -0.20: m_locked = True

        atr_v = d['atr14'][i]
        if np.isnan(atr_v) or atr_v == 0: atr_v = c * 0.02

        if in_pos:
            # Split exit: 50% at 1R, 50% trail 2.5 ATR
            if not p_tp1_done:
                r_mult = (c - p_entry) / p_entry * p_dir / (atr_sl_mult * atr_v / p_entry)
                if r_mult >= 1.0:
                    exit_amt = p_size * 0.5
                    pnl_p = exit_amt * (c - p_entry) / p_entry * p_dir - exit_amt * fee
                    bal += pnl_p; p_size -= exit_amt; p_tp1_done = True
                    p_sl = p_entry  # Move to breakeven

            # Trail remaining
            if p_dir == 1:
                if h > p_high: p_high = h
                trail = p_high - 2.5 * atr_v
                if trail > p_sl: p_sl = trail
                if lo_v <= p_sl:
                    pnl = p_size * (p_sl - p_entry) / p_entry - p_size * fee
                    bal += pnl
                    trades.append({'yr':yr[i],'pnl':pnl,'reason':'TSL' if p_sl > p_entry else 'SL','bal':bal})
                    in_pos = False
            else:
                if lo_v < p_low: p_low = lo_v
                trail = p_low + 2.5 * atr_v
                if trail < p_sl or p_sl == 0: p_sl = trail
                if h >= p_sl:
                    pnl = p_size * (p_entry - p_sl) / p_entry - p_size * fee
                    bal += pnl
                    trades.append({'yr':yr[i],'pnl':pnl,'reason':'TSL' if p_sl < p_entry else 'SL','bal':bal})
                    in_pos = False

        # Pullback signal
        if i < warmup + 1 or in_pos: continue

        sig = 0
        ema20 = d['ema20'][i]; ema50 = d['ema50'][i]; ema200 = d['ema200'][i]
        sk = d['stoch_k'][i]; sd = d['stoch_d'][i]
        sk_p = d['stoch_k'][i-1]; sd_p = d['stoch_d'][i-1]
        rsi_v = d['rsi14'][i]

        if not np.isnan(ema20) and not np.isnan(ema50) and not np.isnan(ema200):
            # Long pullback
            if (ema20 > ema50 > ema200 and d['adx14'][i] >= 20):
                # Price near EMA20 (within 0.5%)
                if abs(c - ema20) / c < 0.005 or (lo_v <= ema20 <= h):
                    # Stoch golden cross from oversold
                    if sk_p < 25 and sk > sd and sk_p <= sd_p:
                        if 45 <= rsi_v <= 65:
                            sig = 1

            # Short pullback
            if (ema20 < ema50 < ema200 and d['adx14'][i] >= 20):
                if abs(c - ema20) / c < 0.005 or (lo_v <= ema20 <= h):
                    if sk_p > 75 and sk < sd and sk_p >= sd_p:
                        if 35 <= rsi_v <= 55:
                            sig = -1

        if sig != 0 and not m_locked and bal > 10:
            sl_dist = atr_sl_mult * atr_v
            pos_size = bal * risk_pct / (sl_dist / c)
            pos_size = min(pos_size, bal * 0.20 * 15)
            bal -= pos_size * fee
            p_dir = sig; p_entry = c; p_size = pos_size
            p_sl = c - sl_dist if sig == 1 else c + sl_dist
            p_high = c; p_low = c; p_tp1_done = False
            in_pos = True

        if bal > peak: peak = bal
        if peak > 0:
            dd = (peak - bal) / peak
            if dd > mdd: mdd = dd

    if in_pos:
        pnl = p_size * (cl[-1] - p_entry) / p_entry * p_dir - p_size * fee
        bal += pnl
        trades.append({'yr':yr[-1],'pnl':pnl,'reason':'END','bal':bal})

    return compile_results(trades, bal, init_bal, mdd, 'B_Pullback')


# ============================================================
# STRATEGY C: BB SQUEEZE BREAKOUT
# ============================================================
def strategy_c_breakout(d, init_bal=5000.0, risk_pct=0.01, atr_sl_mult=1.8):
    cl,hi,lo = d['cl'],d['hi'],d['lo']
    yr,mk = d['yr'],d['mk']
    n = len(cl)
    bal = init_bal; peak = init_bal; mdd = 0.0
    trades = []; in_pos = False; p_dir = 0; p_entry = 0.0; p_size = 0.0
    p_sl = 0.0; p_high = 0.0; p_low = 0.0
    p_stages = [True, True, True]  # 30/30/40 exit stages
    p_orig_size = 0.0
    cur_m = -1; m_start = bal; m_locked = False
    fee = 0.0004; warmup = 250

    # BB width percentile
    bw = d['bb_width']
    bw_pct = np.zeros(n)
    for i in range(100, n):
        window = bw[i-100:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            bw_pct[i] = np.sum(valid < bw[i]) / len(valid)

    for i in range(warmup, n):
        c = cl[i]; h = hi[i]; lo_v = lo[i]
        m = mk[i]
        if m != cur_m: cur_m = m; m_start = bal; m_locked = False
        if not m_locked and m_start > 0:
            if (bal - m_start) / m_start <= -0.20: m_locked = True

        atr_v = d['atr14'][i]
        if np.isnan(atr_v) or atr_v == 0: atr_v = c * 0.02

        if in_pos:
            # 3-stage exit: 30% at 1R, 30% at 2R, 40% trail
            r = (c - p_entry) / p_entry * p_dir / (atr_sl_mult * atr_v / p_entry)
            if p_stages[0] and r >= 1.0:
                ea = p_orig_size * 0.30
                pnl_p = ea * (c - p_entry) / p_entry * p_dir - ea * fee
                bal += pnl_p; p_size -= ea; p_stages[0] = False
                p_sl = p_entry  # Breakeven
            if p_stages[1] and r >= 2.0:
                ea = p_orig_size * 0.30
                if ea > p_size: ea = p_size
                pnl_p = ea * (c - p_entry) / p_entry * p_dir - ea * fee
                bal += pnl_p; p_size -= ea; p_stages[1] = False

            # Trail remaining
            if p_dir == 1:
                if h > p_high: p_high = h
                trail = p_high - 2.0 * atr_v
                if trail > p_sl: p_sl = trail
                if lo_v <= p_sl or p_size <= 0:
                    if p_size > 0:
                        pnl = p_size * (p_sl - p_entry) / p_entry - p_size * fee
                        bal += pnl
                    trades.append({'yr':yr[i],'pnl':bal - trades[-1]['bal'] if trades else bal - init_bal,'reason':'TSL','bal':bal})
                    in_pos = False
            else:
                if lo_v < p_low: p_low = lo_v
                trail = p_low + 2.0 * atr_v
                if trail < p_sl or p_sl == 0: p_sl = trail
                if h >= p_sl or p_size <= 0:
                    if p_size > 0:
                        pnl = p_size * (p_entry - p_sl) / p_entry - p_size * fee
                        bal += pnl
                    trades.append({'yr':yr[i],'pnl':bal - trades[-1]['bal'] if trades else bal - init_bal,'reason':'TSL','bal':bal})
                    in_pos = False

            # Time-based exit: 36 bars max without progress
            if in_pos and hasattr(self if False else type('',(),{'entry_bar':0}), 'entry_bar'):
                pass  # Simplified

        # Breakout signal
        if in_pos: continue

        sig = 0
        squeeze = d['squeeze'][i] if i > 0 else False
        squeeze_prev = d['squeeze'][i-1] if i > 1 else False

        # Squeeze release: was squeezed, now expanding
        if squeeze_prev and not squeeze:
            vol_surge = d['vo'][i] > d['vol_sma20'][i] * 1.5 if not np.isnan(d['vol_sma20'][i]) else False
            ema200 = d['ema200'][i]

            if c > d['bb_upper'][i] and vol_surge and not np.isnan(ema200):
                if c > ema200: sig = 1
            elif c < d['bb_lower'][i] and vol_surge and not np.isnan(ema200):
                if c < ema200: sig = -1

        # Also: 20-bar high breakout with volume
        if sig == 0 and not np.isnan(d['hh20'][i]):
            if c > d['hh20'][i-1] and d['vo'][i] > d['vol_sma20'][i] * 1.5:
                if c > d['ema200'][i] and bw_pct[i] < 0.25:
                    sig = 1
            elif c < d['ll20'][i-1] and d['vo'][i] > d['vol_sma20'][i] * 1.5:
                if c < d['ema200'][i] and bw_pct[i] < 0.25:
                    sig = -1

        if sig != 0 and not m_locked and bal > 10:
            sl_dist = atr_sl_mult * atr_v
            pos_size = bal * risk_pct / (sl_dist / c)
            pos_size = min(pos_size, bal * 0.20 * 15)
            bal -= pos_size * fee
            p_dir = sig; p_entry = c; p_size = pos_size; p_orig_size = pos_size
            p_sl = c - sl_dist if sig == 1 else c + sl_dist
            p_high = c; p_low = c; p_stages = [True, True, True]
            in_pos = True

        if bal > peak: peak = bal
        if peak > 0:
            dd = (peak - bal) / peak
            if dd > mdd: mdd = dd

    if in_pos and p_size > 0:
        pnl = p_size * (cl[-1] - p_entry) / p_entry * p_dir - p_size * fee
        bal += pnl
        trades.append({'yr':yr[-1],'pnl':pnl,'reason':'END','bal':bal})

    return compile_results(trades, bal, init_bal, mdd, 'C_Breakout')


# ============================================================
# STRATEGY D: MEAN REVERSION (CCI + BB + RSI5)
# ============================================================
def strategy_d_meanrev(d, htf_d=None, init_bal=5000.0, risk_pct=0.0075, atr_sl_mult=1.2):
    cl,hi,lo = d['cl'],d['hi'],d['lo']
    yr,mk = d['yr'],d['mk']
    n = len(cl)
    bal = init_bal; peak = init_bal; mdd = 0.0
    trades = []; in_pos = False; p_dir = 0; p_entry = 0.0; p_size = 0.0
    p_sl = 0.0; p_target = 0.0; max_bars = 24; entry_bar = 0
    cur_m = -1; m_start = bal; m_locked = False
    fee = 0.0004; warmup = 250

    # HTF trend filter
    htf_bull = np.ones(n, dtype=bool)
    if htf_d is not None:
        htf_ema200 = htf_d.get('ema200', np.zeros(1))
        htf_ts = htf_d['ts']
        htf_idx = 0
        for i in range(n):
            while htf_idx < len(htf_ts)-1 and htf_ts[htf_idx+1] <= d['ts'][i]:
                htf_idx += 1
            if htf_idx < len(htf_ema200):
                htf_bull[i] = d['cl'][0] > htf_ema200[htf_idx]  # Simplified

    for i in range(warmup, n):
        c = cl[i]; h = hi[i]; lo_v = lo[i]
        m = mk[i]
        if m != cur_m: cur_m = m; m_start = bal; m_locked = False
        if not m_locked and m_start > 0:
            if (bal - m_start) / m_start <= -0.20: m_locked = True

        atr_v = d['atr14'][i]
        if np.isnan(atr_v) or atr_v == 0: atr_v = c * 0.02

        if in_pos:
            # Target: BB mid or 1R
            if p_dir == 1:
                if c >= p_target or lo_v <= p_sl or (i - entry_bar) >= max_bars:
                    ep = min(c, p_target) if c >= p_target else (p_sl if lo_v <= p_sl else c)
                    pnl = p_size * (ep - p_entry) / p_entry - p_size * fee
                    bal += pnl
                    reason = 'TP' if c >= p_target else ('SL' if lo_v <= p_sl else 'TIME')
                    trades.append({'yr':yr[i],'pnl':pnl,'reason':reason,'bal':bal})
                    in_pos = False
            else:
                if c <= p_target or h >= p_sl or (i - entry_bar) >= max_bars:
                    ep = max(c, p_target) if c <= p_target else (p_sl if h >= p_sl else c)
                    pnl = p_size * (p_entry - ep) / p_entry - p_size * fee
                    bal += pnl
                    reason = 'TP' if c <= p_target else ('SL' if h >= p_sl else 'TIME')
                    trades.append({'yr':yr[i],'pnl':pnl,'reason':reason,'bal':bal})
                    in_pos = False

        if in_pos: continue

        # Mean reversion signal
        sig = 0
        cci_v = d['cci20'][i]; rsi5 = d['rsi5'][i]
        bb_lo = d['bb_lower'][i]; bb_up = d['bb_upper'][i]; bb_m = d['bb_mid'][i]

        if not np.isnan(cci_v) and not np.isnan(bb_lo):
            # Long: oversold bounce
            if cci_v < -100 and c <= bb_lo and rsi5 < 25:
                if c > d['ema200'][i]:  # Only in uptrend
                    # Confirm: close above prev high
                    if i > 0 and c > hi[i-1]:
                        sig = 1
            # Short: overbought drop
            if cci_v > 100 and c >= bb_up and rsi5 > 75:
                if c < d['ema200'][i]:
                    if i > 0 and c < lo[i-1]:
                        sig = -1

        if sig != 0 and not m_locked and bal > 10:
            sl_dist = atr_sl_mult * atr_v
            pos_size = bal * risk_pct / (sl_dist / c)
            pos_size = min(pos_size, bal * 0.15 * 10)
            bal -= pos_size * fee
            p_dir = sig; p_entry = c; p_size = pos_size
            p_sl = c - sl_dist if sig == 1 else c + sl_dist
            p_target = bb_m if sig == 1 else bb_m  # Target: BB mid
            entry_bar = i
            in_pos = True

        if bal > peak: peak = bal
        if peak > 0:
            dd = (peak - bal) / peak
            if dd > mdd: mdd = dd

    if in_pos:
        pnl = p_size * (cl[-1] - p_entry) / p_entry * p_dir - p_size * fee
        bal += pnl
        trades.append({'yr':yr[-1],'pnl':pnl,'reason':'END','bal':bal})

    return compile_results(trades, bal, init_bal, mdd, 'D_MeanRev')


# ============================================================
# BASELINE: WMA3/EMA200 (v16.6 style for comparison)
# ============================================================
def strategy_baseline(d, init_bal=5000.0, margin_pct=0.25, leverage=15):
    cl,hi,lo = d['cl'],d['hi'],d['lo']
    yr,mk = d['yr'],d['mk']
    n = len(cl)
    bal = init_bal; peak = init_bal; mdd = 0.0
    trades = []; in_pos = False; p_dir = 0; p_entry = 0.0; p_size = 0.0
    p_sl = 0.0; p_high = 0.0; p_low = 0.0
    t_active = False; t_sl = 0.0
    cur_m = -1; m_start = bal; m_locked = False
    fee = 0.0004; warmup = 250
    sl_pct = min(0.05, 0.9/leverage)

    wma3 = d['wma3']; ema200 = d['ema200']
    adx20 = d['adx20']; rsi14 = d['rsi14']
    valid = ~(np.isnan(wma3)|np.isnan(ema200))
    above = wma3 > ema200
    cup = np.zeros(n,dtype=bool); cdn = np.zeros(n,dtype=bool)
    for i in range(warmup, n):
        if valid[i] and valid[i-1]:
            if above[i] and not above[i-1]: cup[i] = True
            elif not above[i] and above[i-1]: cdn[i] = True

    pend = 0; pcnt = 0
    for i in range(warmup, n):
        c = cl[i]; h = hi[i]; lo_v = lo[i]
        m = mk[i]
        if m != cur_m: cur_m = m; m_start = bal; m_locked = False
        if not m_locked and m_start > 0:
            if (bal - m_start) / m_start <= -0.20: m_locked = True

        if in_pos:
            if p_dir == 1:
                if lo_v <= p_sl:
                    pnl = p_size * (p_sl - p_entry)/p_entry - p_size*fee
                    bal += pnl; trades.append({'yr':yr[i],'pnl':pnl,'reason':'SL','bal':bal}); in_pos = False
                else:
                    if h > p_high: p_high = h
                    if (p_high-p_entry)/p_entry*leverage >= 0.03*leverage:
                        t_active = True
                        ns = p_high*(1-0.02)
                        if ns > t_sl: t_sl = ns
                    if t_active and c <= t_sl:
                        pnl = p_size*(c-p_entry)/p_entry - p_size*fee
                        bal += pnl; trades.append({'yr':yr[i],'pnl':pnl,'reason':'TSL','bal':bal}); in_pos = False
            else:
                if h >= p_sl:
                    pnl = p_size*(p_entry-p_sl)/p_entry - p_size*fee
                    bal += pnl; trades.append({'yr':yr[i],'pnl':pnl,'reason':'SL','bal':bal}); in_pos = False
                else:
                    if lo_v < p_low: p_low = lo_v
                    if (p_entry-p_low)/p_entry*leverage >= 0.03*leverage:
                        t_active = True
                        ns = p_low*(1+0.02)
                        if t_sl == 0 or ns < t_sl: t_sl = ns
                    if t_active and c >= t_sl:
                        pnl = p_size*(p_entry-c)/p_entry - p_size*fee
                        bal += pnl; trades.append({'yr':yr[i],'pnl':pnl,'reason':'TSL','bal':bal}); in_pos = False

        sig = 0
        adx_s = adx20[i] if not np.isnan(adx20[i]) else 0
        rsi_s = rsi14[i] if not np.isnan(rsi14[i]) else 50
        if cup[i] and adx_s >= 35 and 30 <= rsi_s <= 70: sig = 1
        elif cdn[i] and adx_s >= 35 and 30 <= rsi_s <= 70: sig = -1

        if sig != 0:
            pend = sig; pcnt = 5; sig = 0
        if pcnt > 0:
            pcnt -= 1
            if pcnt == 0: sig = pend; pend = 0

        if sig != 0:
            if in_pos:
                pnl = p_size*(c-p_entry)/p_entry*p_dir - p_size*fee
                bal += pnl; trades.append({'yr':yr[i],'pnl':pnl,'reason':'REV','bal':bal}); in_pos = False
            if not in_pos and not m_locked and bal > 10:
                mg = bal*margin_pct; sz = mg*leverage
                bal -= sz*fee
                p_dir = sig; p_entry = c; p_size = sz
                p_sl = c*(1-sl_pct) if sig==1 else c*(1+sl_pct)
                p_high = c; p_low = c; t_active = False; t_sl = 0
                in_pos = True

        if bal > peak: peak = bal
        if peak > 0:
            dd = (peak-bal)/peak
            if dd > mdd: mdd = dd

    if in_pos:
        pnl = p_size*(cl[-1]-p_entry)/p_entry*p_dir - p_size*fee
        bal += pnl; trades.append({'yr':yr[-1],'pnl':pnl,'reason':'END','bal':bal})

    return compile_results(trades, bal, init_bal, mdd, 'Baseline_v166')


# ============================================================
# PORTFOLIO: Run all strategies simultaneously
# ============================================================
def portfolio_backtest(data, init_bal=5000.0, allocations=None):
    if allocations is None:
        allocations = {'A': 0.40, 'B': 0.25, 'C': 0.20, 'D': 0.15}

    results = {}
    for name, alloc in allocations.items():
        sub_bal = init_bal * alloc
        d30 = data['30min']
        d1h = data.get('1h')
        d15 = data.get('15min')

        if name == 'A':
            r = strategy_a_trend(d30, htf_d=d1h, init_bal=sub_bal)
        elif name == 'B':
            r = strategy_b_pullback(d15 if d15 else d30, htf_d=d1h, init_bal=sub_bal)
        elif name == 'C':
            r = strategy_c_breakout(d30, init_bal=sub_bal)
        elif name == 'D':
            r = strategy_d_meanrev(d15 if d15 else d30, htf_d=d1h, init_bal=sub_bal)
        results[name] = r

    total_bal = sum(r['balance'] for r in results.values())
    total_trades = sum(r['trades'] for r in results.values())
    total_ret = (total_bal - init_bal) / init_bal * 100

    # Estimate combined MDD (simplified: max of individual MDDs weighted)
    max_mdd = max(r['mdd'] for r in results.values())

    return {
        'engines': results,
        'total_balance': round(total_bal, 2),
        'total_return': round(total_ret, 1),
        'total_trades': total_trades,
        'max_mdd': round(max_mdd, 1),
    }


# ============================================================
# HELPERS
# ============================================================
def compile_results(trades, bal, init_bal, mdd, name):
    tc = len(trades)
    if tc == 0:
        return {'name':name,'balance':bal,'return_pct':0,'trades':0,'pf':0,'mdd':0,
                'win_rate':0,'sl':0,'tsl':0,'rev':0,'tp':0,'time':0,'fl':0}
    pnls = np.array([t['pnl'] for t in trades])
    w = pnls > 0; l = pnls <= 0
    gp = pnls[w].sum() if w.any() else 0
    gl = abs(pnls[l].sum()) if l.any() else 0.001
    pf = min(gp/gl, 999.99)
    wr = w.sum()/tc*100

    yrs = {}
    for t in trades:
        y = t['yr']
        if y not in yrs: yrs[y] = {'trades':0,'pnl':0}
        yrs[y]['trades'] += 1; yrs[y]['pnl'] += t['pnl']

    return {
        'name': name,
        'balance': round(bal, 2),
        'return_pct': round((bal-init_bal)/init_bal*100, 1),
        'trades': tc,
        'pf': round(pf, 2),
        'mdd': round(mdd*100, 1),
        'win_rate': round(wr, 1),
        'sl': sum(1 for t in trades if t['reason']=='SL'),
        'tsl': sum(1 for t in trades if t['reason']=='TSL'),
        'rev': sum(1 for t in trades if t['reason']=='REV'),
        'tp': sum(1 for t in trades if t['reason']=='TP'),
        'time': sum(1 for t in trades if t['reason']=='TIME'),
        'fl': sum(1 for t in trades if t['reason']=='FL'),
        'yearly': yrs,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print('='*70)
    print('  GPT-5 8-Strategy Experiment Engine')
    print('='*70)

    data = load_and_prepare()

    # Run each strategy individually
    print('\n' + '='*70)
    print('  INDIVIDUAL STRATEGY RESULTS')
    print('='*70)

    d30 = data['30min']
    d15 = data['15min']
    d1h = data['1h']

    strategies = [
        ('A_Trend (Ichimoku+OBV)', lambda: strategy_a_trend(d30, htf_d=d1h)),
        ('B_Pullback (Stoch+EMA)', lambda: strategy_b_pullback(d30, htf_d=d1h)),
        ('C_Breakout (BB Squeeze)', lambda: strategy_c_breakout(d30)),
        ('D_MeanRev (CCI+BB)', lambda: strategy_d_meanrev(d15, htf_d=d1h)),
        ('Baseline (WMA3 v166)', lambda: strategy_baseline(d30)),
    ]

    all_results = []
    for name, fn in strategies:
        print('\n  --- %s ---' % name)
        r = fn()
        all_results.append(r)
        print('  Balance: $%s | Return: %+.1f%% | Trades: %d | PF: %.2f | MDD: %.1f%%' % (
            '{:,.0f}'.format(r['balance']), r['return_pct'], r['trades'], r['pf'], r['mdd']))
        print('  WR: %.1f%% | SL:%d TSL:%d REV:%d TP:%d TIME:%d FL:%d' % (
            r['win_rate'], r['sl'], r['tsl'], r['rev'], r['tp'], r['time'], r['fl']))
        if r.get('yearly'):
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                print('    %d: %d trades, PnL $%s' % (y, ys['trades'], '{:,.0f}'.format(ys['pnl'])))

    # Comparison table
    print('\n' + '='*70)
    print('  COMPARISON TABLE')
    print('='*70)
    print('  %20s %10s %6s %7s %6s %5s %4s' % ('Strategy','Return','Tr','PF','MDD%','WR%','FL'))
    print('  '+'-'*65)
    for r in all_results:
        print('  %20s %+9.1f%% %6d %6.2f %5.1f%% %4.1f %4d' % (
            r['name'], r['return_pct'], r['trades'], r['pf'], r['mdd'], r['win_rate'], r['fl']))

    # Portfolio
    print('\n' + '='*70)
    print('  PORTFOLIO BACKTEST (40/25/20/15 allocation)')
    print('='*70)

    port = portfolio_backtest(data)
    print('  Total Balance: $%s' % '{:,.0f}'.format(port['total_balance']))
    print('  Total Return: %+.1f%%' % port['total_return'])
    print('  Total Trades: %d' % port['total_trades'])
    print('  Max MDD: %.1f%%' % port['max_mdd'])
    print('\n  Per-Engine:')
    for name, r in port['engines'].items():
        print('    %s: $%s (%+.1f%%) Tr=%d PF=%.2f MDD=%.1f%%' % (
            name, '{:,.0f}'.format(r['balance']), r['return_pct'], r['trades'], r['pf'], r['mdd']))

    # 30x Validation for top strategies
    print('\n' + '='*70)
    print('  30x VALIDATION (slippage simulation)')
    print('='*70)

    for strat_fn, strat_name in [
        (lambda: strategy_a_trend(d30, htf_d=d1h), 'A_Trend'),
        (lambda: strategy_baseline(d30), 'Baseline'),
    ]:
        print('\n  --- %s ---' % strat_name)
        runs = []
        for run in range(30):
            np.random.seed(run * 42)
            slip = 1 + np.random.uniform(-0.0005, 0.0005, len(d30['cl']))
            # Create slipped data
            d_slip = dict(d30)
            d_slip['cl'] = d30['cl'] * slip
            d_slip['hi'] = d30['hi'] * slip
            d_slip['lo'] = d30['lo'] * slip
            if strat_name == 'A_Trend':
                # Recompute indicators for slipped data
                d_slip['ichi_tenkan'],d_slip['ichi_kijun'],d_slip['ichi_span_a'],d_slip['ichi_span_b'],d_slip['ichi_cloud_top'],d_slip['ichi_cloud_bot'] = ichimoku(d_slip['hi'],d_slip['lo'],d_slip['cl'])
                d_slip['obv'] = obv(d_slip['cl'], d30['vo'])
                d_slip['obv_ema20'] = ema(d_slip['obv'], 20)
                d_slip['adx14'] = adx(d_slip['hi'],d_slip['lo'],d_slip['cl'],14)
                d_slip['atr14'] = atr(d_slip['hi'],d_slip['lo'],d_slip['cl'],14)
                r = strategy_a_trend(d_slip, htf_d=d1h)
            else:
                d_slip['wma3'] = wma(d_slip['cl'],3)
                d_slip['ema200'] = ema(d_slip['cl'],200)
                d_slip['adx20'] = adx(d_slip['hi'],d_slip['lo'],d_slip['cl'],20)
                d_slip['rsi14'] = rsi(d_slip['cl'],14)
                r = strategy_baseline(d_slip)
            runs.append(r)

        bals = [r['balance'] for r in runs]
        pfs = [r['pf'] for r in runs]
        mdds = [r['mdd'] for r in runs]
        print('  Bal: $%.0f +/- $%.0f (min $%.0f)' % (np.mean(bals), np.std(bals), np.min(bals)))
        print('  PF: %.2f +/- %.2f (min %.2f)' % (np.mean(pfs), np.std(pfs), np.min(pfs)))
        print('  MDD: %.1f%% (max %.1f%%)' % (np.mean(mdds), np.max(mdds)))

    # Save
    with open(os.path.join(OUTPUT_DIR, 'all_results.json'), 'w') as f:
        json.dump({r['name']: r for r in all_results}, f, indent=2, default=str)

    print('\n' + '='*70)
    print('  COMPLETE in %.1f min' % ((time.time()-t0)/60))
    print('='*70)


if __name__ == '__main__':
    main()
