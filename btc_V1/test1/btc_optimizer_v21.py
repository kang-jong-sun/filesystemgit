"""
BTC/USDT Futures Optimizer v21.5
- $5,000 initial, L<=15x, M<=25%
- Cross margin vs Isolated comparison
- Target: PF>=5, MDD<=20%, Return>=100,000%
- 4-phase hierarchical optimization
- 30x slippage validation
"""
import sys, time, json, os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = r"D:\filesystem\futures\btc_V1\test1\btc_usdt_5m_merged.csv"
OUTPUT_DIR = r"D:\filesystem\futures\btc_V1\test1\optimization_v21"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# INDICATORS (reuse from v17)
# ============================================================
def calc_ema(data, period):
    return pd.Series(data).ewm(span=period, adjust=False).mean().values

def calc_sma(data, period):
    return pd.Series(data).rolling(period, min_periods=period).mean().values

def calc_wma(data, period):
    weights = np.arange(1, period + 1, dtype=float)
    s = pd.Series(data)
    return s.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).values

def calc_hma(data, period):
    half = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    wma_half = calc_wma(data, half)
    wma_full = calc_wma(data, period)
    diff = 2 * wma_half - wma_full
    mask = ~np.isnan(diff)
    result = np.full_like(data, np.nan, dtype=float)
    if mask.sum() > sqrt_p:
        result[mask] = calc_wma(diff[mask], sqrt_p)
    return result

def calc_dema(data, period):
    e1 = calc_ema(data, period)
    e2 = calc_ema(e1, period)
    return 2 * e1 - e2

def calc_vwma(close, volume, period):
    cv = pd.Series(close * volume).rolling(period, min_periods=period).sum().values
    v = pd.Series(volume).rolling(period, min_periods=period).sum().values
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(v > 0, cv / v, np.nan)

def calc_rsi(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ga = pd.Series(gain).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    la = pd.Series(loss).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(la > 0, ga / la, 100.0)
    return 100 - 100 / (1 + rs)

def calc_adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    up = high - np.roll(high, 1); dn = np.roll(low, 1) - low
    up[0] = dn[0] = 0
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = 1.0 / period
    atr = pd.Series(tr).ewm(alpha=a, min_periods=period, adjust=False).mean().values
    pdi_s = pd.Series(pdm).ewm(alpha=a, min_periods=period, adjust=False).mean().values
    mdi_s = pd.Series(mdm).ewm(alpha=a, min_periods=period, adjust=False).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        pdi = np.where(atr > 0, 100 * pdi_s / atr, 0)
        mdi = np.where(atr > 0, 100 * mdi_s / atr, 0)
        ds = pdi + mdi
        dx = np.where(ds > 0, 100 * np.abs(pdi - mdi) / ds, 0)
    return pd.Series(dx).ewm(alpha=a, min_periods=period, adjust=False).mean().values

def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    return pd.Series(tr).ewm(alpha=1.0/period, min_periods=period, adjust=False).mean().values

def calc_ma(data, ma_type, period, volume=None):
    if ma_type == 'EMA': return calc_ema(data, period)
    elif ma_type == 'SMA': return calc_sma(data, period)
    elif ma_type == 'WMA': return calc_wma(data, period)
    elif ma_type == 'HMA': return calc_hma(data, period)
    elif ma_type == 'DEMA': return calc_dema(data, period)
    elif ma_type == 'VWMA' and volume is not None: return calc_vwma(data, volume, period)
    return calc_ema(data, period)

# ============================================================
# DATA LOADING
# ============================================================
TF_MINUTES = {'5min': 5, '10min': 10, '15min': 15, '30min': 30, '1h': 60}

def load_data():
    print('Loading 5m data...')
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    print('  %d rows (%s ~ %s)' % (len(df), df.timestamp.iloc[0], df.timestamp.iloc[-1]))
    return df

def resample(df_5m, tf_min):
    df = df_5m.set_index('timestamp')
    rule = '%dmin' % tf_min if tf_min < 60 else '%dh' % (tf_min // 60)
    return df.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()

def precompute(df_5m):
    data = {}
    for name, mins in TF_MINUTES.items():
        r = df_5m.copy() if mins == 5 else resample(df_5m, mins)
        ts = pd.to_datetime(r['timestamp'].values)
        data[name] = {
            'timestamp': r['timestamp'].values,
            'open': r['open'].values.astype(float),
            'high': r['high'].values.astype(float),
            'low': r['low'].values.astype(float),
            'close': r['close'].values.astype(float),
            'volume': r['volume'].values.astype(float),
            'years': ts.year.values.astype(np.int32),
            'month_keys': (ts.year.values * 100 + ts.month.values).astype(np.int32),
            'n': len(r),
        }
        print('  %s: %d bars' % (name, data[name]['n']))
    return data

# ============================================================
# FAST BACKTEST (supports cross/isolated margin)
# ============================================================
def fast_bt(close, high, low, ma_fast, ma_slow, adx, rsi, atr,
            years, mkeys,
            adx_min=35, rsi_min=30, rsi_max=65, entry_delay=0,
            sl_pct=0.05, trail_act=0.06, trail_pct=0.03,
            tp1_roi=None, tp1_ratio=0.30,
            leverage=10, margin_pct=0.20,
            ml_limit=-0.20, fee=0.0004, init_bal=5000.0,
            sl_atr_mult=0.0, warmup=300,
            cross_margin=False):
    """Fast backtest. cross_margin=True uses full balance as margin."""
    n = len(close)
    bal = init_bal
    peak = init_bal
    mdd = 0.0
    in_pos = False
    p_dir = 0; p_entry = 0.0; p_size = 0.0; p_margin = 0.0; p_sl = 0.0
    t_active = False; t_sl = 0.0; p_high = 0.0; p_low = 0.0
    p_partial = False; p_orig = 0.0
    pend_sig = 0; pend_cnt = 0
    cur_m = -1; m_start = init_bal; m_locked = False
    t_pnl = []; t_roi = []; t_reason = []; t_year = []
    tc = 0
    inv_lev = 1.0 / leverage
    ta_lev = trail_act * leverage

    # Pre-compute crosses
    valid = ~(np.isnan(ma_fast) | np.isnan(ma_slow))
    above = ma_fast > ma_slow
    cup = np.zeros(n, dtype=bool); cdn = np.zeros(n, dtype=bool)
    for i in range(max(warmup, 1), n):
        if valid[i] and valid[i-1]:
            if above[i] and not above[i-1]: cup[i] = True
            elif not above[i] and above[i-1]: cdn[i] = True
    adx_s = np.where(np.isnan(adx), 0.0, adx)
    rsi_s = np.where(np.isnan(rsi), 50.0, rsi)
    atr_s = np.where(np.isnan(atr), 0.0, atr)

    for i in range(max(warmup, 1), n):
        c = close[i]; h = high[i]; lo = low[i]
        mk = mkeys[i]
        if mk != cur_m:
            cur_m = mk; m_start = bal; m_locked = False
        if not m_locked and m_start > 0:
            if (bal - m_start) / m_start <= ml_limit: m_locked = True

        if in_pos:
            # FL check
            if cross_margin:
                # Cross margin: liquidation when unrealized loss >= account balance
                if p_dir == 1:
                    unreal = p_size * (lo - p_entry) / p_entry
                    if bal + unreal <= 0:
                        bal = 10.0  # Near-zero (account wiped)
                        t_pnl.append(-init_bal); t_roi.append(-1.0); t_reason.append('FL'); t_year.append(years[i])
                        tc += 1; in_pos = False
                        if bal > peak: peak = bal
                        dd = (peak - bal) / peak if peak > 0 else 0
                        if dd > mdd: mdd = dd
                        continue
                else:
                    unreal = p_size * (p_entry - h) / p_entry
                    if bal + unreal <= 0:
                        bal = 10.0
                        t_pnl.append(-init_bal); t_roi.append(-1.0); t_reason.append('FL'); t_year.append(years[i])
                        tc += 1; in_pos = False
                        if bal > peak: peak = bal
                        dd = (peak - bal) / peak if peak > 0 else 0
                        if dd > mdd: mdd = dd
                        continue
            else:
                # Isolated: liquidation at margin depletion
                if p_dir == 1:
                    fl_p = p_entry * (1 - inv_lev)
                    if lo <= fl_p:
                        bal -= p_margin; f2 = p_size * fee; bal -= f2
                        t_pnl.append(-p_margin - f2); t_roi.append(-1.0); t_reason.append('FL'); t_year.append(years[i])
                        tc += 1; in_pos = False
                        if bal > peak: peak = bal
                        dd = (peak - bal) / peak if peak > 0 else 0
                        if dd > mdd: mdd = dd
                        continue
                else:
                    fl_p = p_entry * (1 + inv_lev)
                    if h >= fl_p:
                        bal -= p_margin; f2 = p_size * fee; bal -= f2
                        t_pnl.append(-p_margin - f2); t_roi.append(-1.0); t_reason.append('FL'); t_year.append(years[i])
                        tc += 1; in_pos = False
                        if bal > peak: peak = bal
                        dd = (peak - bal) / peak if peak > 0 else 0
                        if dd > mdd: mdd = dd
                        continue

            # SL check
            sl_hit = False
            if p_dir == 1 and lo <= p_sl: sl_hit = True; ep = p_sl
            elif p_dir == -1 and h >= p_sl: sl_hit = True; ep = p_sl

            if sl_hit:
                pnl = p_size * (ep - p_entry) / p_entry * p_dir - p_size * fee
                bal += pnl; roi = (ep - p_entry) / p_entry * p_dir * leverage
                t_pnl.append(pnl); t_roi.append(roi); t_reason.append('SL'); t_year.append(years[i])
                tc += 1; in_pos = False
            else:
                # Trailing update
                if p_dir == 1:
                    if h > p_high: p_high = h
                    if (p_high - p_entry) / p_entry * leverage >= ta_lev:
                        t_active = True
                        ns = p_high * (1 - trail_pct)
                        if ns > t_sl: t_sl = ns
                else:
                    if lo < p_low: p_low = lo
                    if (p_entry - p_low) / p_entry * leverage >= ta_lev:
                        t_active = True
                        ns = p_low * (1 + trail_pct)
                        if t_sl == 0 or ns < t_sl: t_sl = ns

                # Partial TP
                if tp1_roi is not None and not p_partial:
                    cr = (c - p_entry) / p_entry * p_dir * leverage
                    if cr >= tp1_roi:
                        ea = p_orig * tp1_ratio
                        pp = ea * (c - p_entry) / p_entry * p_dir - ea * fee
                        bal += pp; p_size -= ea; p_partial = True

                # TSL check
                if t_active:
                    if (p_dir == 1 and c <= t_sl) or (p_dir == -1 and c >= t_sl):
                        pnl = p_size * (c - p_entry) / p_entry * p_dir - p_size * fee
                        bal += pnl; roi = (c - p_entry) / p_entry * p_dir * leverage
                        t_pnl.append(pnl); t_roi.append(roi); t_reason.append('TSL'); t_year.append(years[i])
                        tc += 1; in_pos = False

        # Signal detection
        sig = 0
        if cup[i]:
            if adx_s[i] >= adx_min and rsi_min <= rsi_s[i] <= rsi_max: sig = 1
        elif cdn[i]:
            if adx_s[i] >= adx_min and rsi_min <= rsi_s[i] <= rsi_max: sig = -1

        if sig != 0 and entry_delay > 0:
            pend_sig = sig; pend_cnt = entry_delay; sig = 0
        if pend_cnt > 0:
            pend_cnt -= 1
            if pend_cnt == 0: sig = pend_sig; pend_sig = 0

        if sig != 0:
            if in_pos and p_dir != sig:
                pnl = p_size * (c - p_entry) / p_entry * p_dir - p_size * fee
                bal += pnl; roi = (c - p_entry) / p_entry * p_dir * leverage
                t_pnl.append(pnl); t_roi.append(roi); t_reason.append('REV'); t_year.append(years[i])
                tc += 1; in_pos = False
            elif in_pos:
                pnl = p_size * (c - p_entry) / p_entry * p_dir - p_size * fee
                bal += pnl; roi = (c - p_entry) / p_entry * p_dir * leverage
                t_pnl.append(pnl); t_roi.append(roi); t_reason.append('REV'); t_year.append(years[i])
                tc += 1; in_pos = False

            if not in_pos and not m_locked and bal > 10:
                mg = bal * margin_pct
                sz = mg * leverage
                bal -= sz * fee  # entry fee
                p_dir = sig; p_entry = c; p_size = sz; p_orig = sz; p_margin = mg
                t_active = False; t_sl = 0.0; p_high = c; p_low = c; p_partial = False

                if sl_atr_mult > 0 and atr_s[i] > 0:
                    sd = atr_s[i] * sl_atr_mult / p_entry
                    sd = max(0.01, min(sd, inv_lev * 0.9))
                else:
                    sd = min(sl_pct, inv_lev * 0.9)  # Auto-cap SL below FL

                p_sl = p_entry * (1 - sd) if p_dir == 1 else p_entry * (1 + sd)
                in_pos = True

        if bal > peak: peak = bal
        if peak > 0:
            dd = (peak - bal) / peak
            if dd > mdd: mdd = dd

    # Close remaining
    if in_pos:
        c = close[-1]
        pnl = p_size * (c - p_entry) / p_entry * p_dir - p_size * fee
        bal += pnl; roi = (c - p_entry) / p_entry * p_dir * leverage
        t_pnl.append(pnl); t_roi.append(roi); t_reason.append('END'); t_year.append(years[-1])
        tc += 1

    if tc == 0:
        return {'balance':bal,'return_pct':0,'trades':0,'pf':0,'wpf':0,'mdd':0,
                'win_rate':0,'payoff':0,'avg_win':0,'avg_loss':0,
                'sl':0,'tsl':0,'rev':0,'fl':0,'tpm':0,'score':0}

    pa = np.array(t_pnl); ra = np.array(t_roi); ya = np.array(t_year)
    w = pa > 0; l = pa <= 0
    gp = pa[w].sum() if w.any() else 0
    gl = abs(pa[l].sum()) if l.any() else 0.001
    pf = min(gp / gl, 999.99) if gl > 0 else 999.99
    wr = w.sum() / tc * 100
    aw = ra[w].mean() * 100 if w.any() else 0
    al = ra[l].mean() * 100 if l.any() else -0.001
    po = min(abs(aw / al), 999.99) if al != 0 else 999.99

    wt = np.where(ya >= 2023, 1.5, 0.5)
    wp = (pa[w] * wt[w]).sum() if w.any() else 0
    wl = abs((pa[l] * wt[l]).sum()) if l.any() else 0.001
    wpf = min(wp / wl, 999.99) if wl > 0 else 999.99

    sl_c = sum(1 for r in t_reason if r == 'SL')
    tsl_c = sum(1 for r in t_reason if r == 'TSL')
    rev_c = sum(1 for r in t_reason if r == 'REV')
    fl_c = sum(1 for r in t_reason if r == 'FL')

    ret = (bal - init_bal) / init_bal * 100
    tpm = tc / 75.0

    # Composite score: prioritize return while filtering MDD<=20% and PF>=5
    if mdd * 100 <= 25 and pf >= 3 and tc >= 10:
        score = ret * min(pf, 50) / max(mdd * 100, 5)
    else:
        score = ret / max(mdd * 100, 10) * min(pf, 10) * 0.1

    return {
        'balance': round(bal, 2), 'return_pct': round(ret, 1),
        'trades': tc, 'pf': round(pf, 2), 'wpf': round(wpf, 2),
        'mdd': round(mdd * 100, 1), 'win_rate': round(wr, 1),
        'payoff': round(po, 2), 'avg_win': round(aw, 2), 'avg_loss': round(al, 2),
        'sl': sl_c, 'tsl': tsl_c, 'rev': rev_c, 'fl': fl_c,
        'tpm': round(tpm, 2), 'score': round(score, 1),
    }


# ============================================================
# DETAILED ANALYSIS
# ============================================================
def detail_bt(all_data, p):
    tf = p['tf']; d = all_data[tf]
    cl, hi, lo = d['close'], d['high'], d['low']
    vol, ts = d['volume'], d['timestamp']
    yr, mk = d['years'], d['month_keys']
    mf = calc_ma(cl, p['ma_type'], p['fast_period'], vol)
    ms = calc_ma(cl, 'EMA', p['slow_period'], vol)
    ax = calc_adx(hi, lo, cl, p.get('adx_period', 20))
    rs = calc_rsi(cl, 14)
    at = calc_atr(hi, lo, cl, 14)
    wu = p['slow_period'] + 50

    # Run with trade tracking
    n = len(cl); bal = p.get('init_bal', 5000.0); init = bal
    pk = bal; mdd = 0.0; trades = []
    in_pos = False; p_dir = 0; p_entry = 0.0; p_size = 0.0; p_sl = 0.0
    t_active = False; t_sl = 0.0; p_high = 0.0; p_low = 0.0
    p_partial = False; p_orig = 0.0; p_margin = 0.0; entry_time = None
    pend_sig = 0; pend_cnt = 0
    cur_m = -1; m_start = bal; m_locked = False
    lev = p.get('leverage', 10); mpct = p.get('margin_pct', 0.20)
    sl_p = p.get('sl_pct', 0.05); ta = p.get('trail_act', 0.06); tp = p.get('trail_pct', 0.03)
    tp1 = p.get('tp1_roi', None); dl = p.get('entry_delay', 0)
    ml = p.get('ml_limit', -0.20); sa = p.get('sl_atr_mult', 0)
    cross = p.get('cross_margin', False)
    inv_lev = 1.0/lev; ta_lev = ta*lev; fee = 0.0004

    valid = ~(np.isnan(mf) | np.isnan(ms))
    above = mf > ms
    cup_a = np.zeros(n, dtype=bool); cdn_a = np.zeros(n, dtype=bool)
    for i in range(max(wu,1), n):
        if valid[i] and valid[i-1]:
            if above[i] and not above[i-1]: cup_a[i] = True
            elif not above[i] and above[i-1]: cdn_a[i] = True
    adx_s = np.where(np.isnan(ax), 0.0, ax)
    rsi_s = np.where(np.isnan(rs), 50.0, rs)
    atr_s = np.where(np.isnan(at), 0.0, at)
    ts_pd = pd.to_datetime(ts)

    def close_p(ep, reason):
        nonlocal bal, in_pos, pk, mdd
        pnl = p_size * (ep - p_entry) / p_entry * p_dir - p_size * fee
        roi = (ep - p_entry) / p_entry * p_dir * lev
        bal += pnl
        trades.append({'et': entry_time, 'xt': ts_pd[i], 'dir': 'LONG' if p_dir==1 else 'SHORT',
                      'ep': p_entry, 'xp': ep, 'pnl': pnl, 'roi': roi, 'reason': reason, 'bal': bal})
        in_pos = False
        if bal > pk: pk = bal
        if pk > 0:
            dd = (pk - bal) / pk
            if dd > mdd: mdd = dd

    for i in range(wu, n):
        c = cl[i]; h = hi[i]; l = lo[i]
        m = mk[i]
        if m != cur_m: cur_m = m; m_start = bal; m_locked = False
        if not m_locked and m_start > 0:
            if (bal - m_start) / m_start <= ml: m_locked = True

        if in_pos:
            if not cross:
                if p_dir == 1 and l <= p_entry*(1-inv_lev):
                    bal -= p_margin; trades.append({'et':entry_time,'xt':ts_pd[i],'dir':'LONG','ep':p_entry,'xp':p_entry*(1-inv_lev),'pnl':-p_margin,'roi':-1.0,'reason':'FL','bal':bal})
                    in_pos = False; continue
                if p_dir == -1 and h >= p_entry*(1+inv_lev):
                    bal -= p_margin; trades.append({'et':entry_time,'xt':ts_pd[i],'dir':'SHORT','ep':p_entry,'xp':p_entry*(1+inv_lev),'pnl':-p_margin,'roi':-1.0,'reason':'FL','bal':bal})
                    in_pos = False; continue

            sl_hit = False
            if p_dir == 1 and l <= p_sl: sl_hit = True; ep = p_sl
            elif p_dir == -1 and h >= p_sl: sl_hit = True; ep = p_sl
            if sl_hit: close_p(ep, 'SL')
            elif in_pos:
                if p_dir == 1:
                    if h > p_high: p_high = h
                    if (p_high-p_entry)/p_entry*lev >= ta_lev:
                        t_active = True; ns = p_high*(1-tp)
                        if ns > t_sl: t_sl = ns
                else:
                    if l < p_low: p_low = l
                    if (p_entry-p_low)/p_entry*lev >= ta_lev:
                        t_active = True; ns = p_low*(1+tp)
                        if t_sl == 0 or ns < t_sl: t_sl = ns
                if t_active:
                    if (p_dir==1 and c<=t_sl) or (p_dir==-1 and c>=t_sl):
                        close_p(c, 'TSL')

        sig = 0
        if cup_a[i]:
            if adx_s[i] >= p.get('adx_min',35) and p.get('rsi_min',30) <= rsi_s[i] <= p.get('rsi_max',65): sig = 1
        elif cdn_a[i]:
            if adx_s[i] >= p.get('adx_min',35) and p.get('rsi_min',30) <= rsi_s[i] <= p.get('rsi_max',65): sig = -1
        if sig != 0 and dl > 0: pend_sig = sig; pend_cnt = dl; sig = 0
        if pend_cnt > 0:
            pend_cnt -= 1
            if pend_cnt == 0: sig = pend_sig; pend_sig = 0

        if sig != 0:
            if in_pos: close_p(c, 'REV')
            if not in_pos and not m_locked and bal > 10:
                mg = bal * mpct; sz = mg * lev
                bal -= sz * fee
                p_dir = sig; p_entry = c; p_size = sz; p_orig = sz; p_margin = mg
                t_active = False; t_sl = 0.0; p_high = c; p_low = c; p_partial = False
                entry_time = ts_pd[i]
                if sa > 0 and atr_s[i] > 0:
                    sd = atr_s[i]*sa/p_entry; sd = max(0.01, min(sd, inv_lev*0.9))
                else:
                    sd = min(sl_p, inv_lev*0.9)
                p_sl = p_entry*(1-sd) if p_dir==1 else p_entry*(1+sd)
                in_pos = True

        if bal > pk: pk = bal
        if pk > 0:
            dd = (pk-bal)/pk
            if dd > mdd: mdd = dd

    if in_pos: close_p(cl[-1], 'END')

    # Monthly/Yearly stats
    monthly = {}; yearly = {}; rb = init
    for t in trades:
        mk2 = t['et'].strftime('%Y-%m')
        if mk2 not in monthly: monthly[mk2] = {'start':rb,'trades':0,'pnl':0,'sl':0,'tsl':0,'rev':0,'fl':0}
        monthly[mk2]['trades'] += 1; monthly[mk2]['pnl'] += t['pnl']
        monthly[mk2][t['reason'].lower()] = monthly[mk2].get(t['reason'].lower(),0)+1
        monthly[mk2]['end'] = t['bal']; rb = t['bal']

        y = t['et'].year
        if y not in yearly: yearly[y] = {'start':rb - t['pnl'],'trades':0,'pnl':0,'sl':0,'tsl':0,'rev':0,'fl':0}
        yearly[y]['trades'] += 1; yearly[y]['pnl'] += t['pnl']
        yearly[y][t['reason'].lower()] = yearly[y].get(t['reason'].lower(),0)+1
        yearly[y]['end'] = t['bal']

    for m2 in monthly:
        s = monthly[m2]['start']; e = monthly[m2].get('end',s)
        monthly[m2]['ret'] = (e-s)/s*100 if s > 0 else 0
    for y in yearly:
        s = yearly[y]['start']; e = yearly[y].get('end',s)
        yearly[y]['ret'] = (e-s)/s*100 if s > 0 else 0

    return {'trades':trades,'monthly':monthly,'yearly':yearly,'balance':bal,'mdd':mdd*100,'tc':len(trades)}


# ============================================================
# MAIN OPTIMIZATION
# ============================================================
def main():
    t_start = time.time()
    print('='*70)
    print('  BTC/USDT Optimizer v21.5')
    print('  Target: PF>=5, MDD<=20%%, Return>=100,000%%')
    print('='*70)

    df_5m = load_data()
    all_data = precompute(df_5m)

    # ===== PHASE 1: MA × TF × Period =====
    print('\n' + '='*70)
    print('  PHASE 1: MA x TF x Period Scan')
    print('='*70)

    MA_TYPES = ['EMA', 'WMA', 'HMA', 'SMA', 'DEMA']
    TFS = ['15min', '30min', '1h']
    FAST_P = [2, 3, 5, 7, 10, 14, 21]
    SLOW_P = [100, 150, 200, 250, 300]

    p1 = []; cnt = 0; t0 = time.time()
    for mt in MA_TYPES:
        for tf in TFS:
            d = all_data[tf]
            cl,hi,lo,vol = d['close'],d['high'],d['low'],d['volume']
            ax = calc_adx(hi, lo, cl, 20)
            rs = calc_rsi(cl, 14); at = calc_atr(hi, lo, cl, 14)
            yr, mk = d['years'], d['month_keys']
            for fp in FAST_P:
                mf = calc_ma(cl, mt, fp, vol)
                for sp in SLOW_P:
                    if fp >= sp: cnt += 1; continue
                    ms2 = calc_ma(cl, 'EMA', sp, vol)
                    for lev in [10, 15]:
                        for mpct in [0.15, 0.20, 0.25]:
                            sl = min(0.05, 0.9/lev)  # Auto-cap SL
                            r = fast_bt(cl, hi, lo, mf, ms2, ax, rs, at, yr, mk,
                                       adx_min=35, rsi_min=30, rsi_max=65,
                                       sl_pct=sl, trail_act=0.06, trail_pct=0.03,
                                       leverage=lev, margin_pct=mpct, ml_limit=-0.20,
                                       init_bal=5000.0, warmup=sp+50)
                            r.update({'ma_type':mt,'tf':tf,'fast_period':fp,'slow_period':sp,
                                     'leverage':lev,'margin_pct':mpct,'sl_pct':sl})
                            p1.append(r)
                            cnt += 1
                    if cnt % 200 == 0: print('  P1: %d (%ds)' % (cnt, time.time()-t0))

    print('  P1: %d combos in %ds' % (len(p1), time.time()-t0))
    p1.sort(key=lambda x: x['score'], reverse=True)

    print('\n  Top 30:')
    print('  %4s %5s %5s %3s %3s %3s %4s %4s %10s %7s %6s %5s %4s %4s %7s' %
          ('R','MA','TF','F','S','L','M%','Tr','Ret%','PF','MDD%','WR%','SL','FL','Score'))
    print('  '+'-'*95)
    for i, r in enumerate(p1[:30]):
        print('  %4d %5s %5s %3d %3d %2dx %3.0f%% %4d %+9.1f%% %6.2f %5.1f%% %4.1f %4d %4d %6.1f' %
              (i+1, r['ma_type'], r['tf'], r['fast_period'], r['slow_period'],
               r['leverage'], r['margin_pct']*100, r['trades'], r['return_pct'],
               r['pf'], r['mdd'], r['win_rate'], r['sl'], r['fl'], r['score']))

    with open(os.path.join(OUTPUT_DIR, 'p1.json'), 'w') as f:
        json.dump(p1[:100], f, indent=2, default=str)

    # ===== PHASE 2: Entry optimization =====
    print('\n' + '='*70)
    print('  PHASE 2: Entry Optimization')
    print('='*70)

    ADX_T = [25, 30, 35, 40, 45]
    RSI_R = [(20,80), (25,75), (30,70), (35,65), (30,65)]
    DELAYS = [0, 1, 3, 5]

    top1 = [r for r in p1 if r['trades'] >= 10][:15]
    p2 = []; cnt = 0; t0 = time.time()
    for base in top1:
        tf = base['tf']; d = all_data[tf]
        cl,hi,lo,vol = d['close'],d['high'],d['low'],d['volume']
        yr, mk = d['years'], d['month_keys']
        mf = calc_ma(cl, base['ma_type'], base['fast_period'], vol)
        ms2 = calc_ma(cl, 'EMA', base['slow_period'], vol)
        at = calc_atr(hi, lo, cl, 14)
        wu = base['slow_period'] + 50
        for adx_p in [14, 20]:
            ax = calc_adx(hi, lo, cl, adx_p)
            for at2 in ADX_T:
                for rmin, rmax in RSI_R:
                    rs = calc_rsi(cl, 14)
                    for dl in DELAYS:
                        r = fast_bt(cl, hi, lo, mf, ms2, ax, rs, at, yr, mk,
                                   adx_min=at2, rsi_min=rmin, rsi_max=rmax, entry_delay=dl,
                                   sl_pct=base['sl_pct'], trail_act=0.06, trail_pct=0.03,
                                   leverage=base['leverage'], margin_pct=base['margin_pct'],
                                   ml_limit=-0.20, init_bal=5000.0, warmup=wu)
                        for k in ['ma_type','tf','fast_period','slow_period','leverage','margin_pct','sl_pct']:
                            r[k] = base[k]
                        r.update({'adx_period':adx_p,'adx_min':at2,'rsi_min':rmin,'rsi_max':rmax,'entry_delay':dl})
                        p2.append(r)
                        cnt += 1
                        if cnt % 500 == 0: print('  P2: %d (%ds)' % (cnt, time.time()-t0))

    print('  P2: %d combos in %ds' % (len(p2), time.time()-t0))
    p2.sort(key=lambda x: x['score'], reverse=True)

    print('\n  Top 20:')
    for i, r in enumerate(p2[:20]):
        print('  %3d %5s %5s F%d/S%d L%dx M%.0f%% ADX>=%d RSI%d-%d D%d Tr=%d Ret=%+.1f%% PF=%.2f MDD=%.1f%% Score=%.1f' %
              (i+1, r['ma_type'], r['tf'], r['fast_period'], r['slow_period'],
               r['leverage'], r['margin_pct']*100, r['adx_min'], r['rsi_min'], r['rsi_max'],
               r['entry_delay'], r['trades'], r['return_pct'], r['pf'], r['mdd'], r['score']))

    with open(os.path.join(OUTPUT_DIR, 'p2.json'), 'w') as f:
        json.dump(p2[:100], f, indent=2, default=str)

    # ===== PHASE 3: Exit optimization =====
    print('\n' + '='*70)
    print('  PHASE 3: Exit Optimization')
    print('='*70)

    SL_PCTS = [0.03, 0.04, 0.05, 0.06]
    TRAIL_A = [0.02, 0.03, 0.05, 0.06, 0.08, 0.10]
    TRAIL_P = [0.02, 0.03, 0.05]
    TP1S = [None, 0.15, 0.30]

    top2 = [r for r in p2 if r['trades'] >= 10][:20]
    p3 = []; cnt = 0; t0 = time.time()
    for base in top2:
        tf = base['tf']; d = all_data[tf]
        cl,hi,lo,vol = d['close'],d['high'],d['low'],d['volume']
        yr, mk = d['years'], d['month_keys']
        mf = calc_ma(cl, base['ma_type'], base['fast_period'], vol)
        ms2 = calc_ma(cl, 'EMA', base['slow_period'], vol)
        ax = calc_adx(hi, lo, cl, base.get('adx_period', 20))
        rs = calc_rsi(cl, 14); at = calc_atr(hi, lo, cl, 14)
        wu = base['slow_period'] + 50
        for sl in SL_PCTS:
            capped_sl = min(sl, 0.9 / base['leverage'])
            for ta2 in TRAIL_A:
                for tp2 in TRAIL_P:
                    if tp2 >= ta2: continue
                    for tp1 in TP1S:
                        r = fast_bt(cl, hi, lo, mf, ms2, ax, rs, at, yr, mk,
                                   adx_min=base.get('adx_min',35), rsi_min=base.get('rsi_min',30),
                                   rsi_max=base.get('rsi_max',65), entry_delay=base.get('entry_delay',0),
                                   sl_pct=capped_sl, trail_act=ta2, trail_pct=tp2, tp1_roi=tp1,
                                   leverage=base['leverage'], margin_pct=base['margin_pct'],
                                   ml_limit=-0.20, init_bal=5000.0, warmup=wu)
                        for k in ['ma_type','tf','fast_period','slow_period','leverage','margin_pct',
                                  'adx_period','adx_min','rsi_min','rsi_max','entry_delay']:
                            if k in base: r[k] = base[k]
                        r.update({'sl_pct':capped_sl,'trail_act':ta2,'trail_pct':tp2,'tp1_roi':tp1})
                        p3.append(r)
                        cnt += 1
                        if cnt % 1000 == 0: print('  P3: %d (%ds)' % (cnt, time.time()-t0))

    print('  P3: %d combos in %ds' % (len(p3), time.time()-t0))
    p3.sort(key=lambda x: x['score'], reverse=True)

    print('\n  Top 20:')
    for i, r in enumerate(p3[:20]):
        tp1s = '%.0f%%' % (r['tp1_roi']*100) if r['tp1_roi'] else 'None'
        print('  %3d %5s %5s SL%.0f%% TA%.0f%% TP%.0f%% TP1=%s Tr=%d Ret=%+.1f%% PF=%.2f MDD=%.1f%% Sc=%.1f' %
              (i+1, r['ma_type'], r['tf'], r['sl_pct']*100, r['trail_act']*100,
               r['trail_pct']*100, tp1s, r['trades'], r['return_pct'], r['pf'], r['mdd'], r['score']))

    with open(os.path.join(OUTPUT_DIR, 'p3.json'), 'w') as f:
        json.dump(p3[:100], f, indent=2, default=str)

    # ===== PHASE 4: Cross vs Isolated + final sizing =====
    print('\n' + '='*70)
    print('  PHASE 4: Cross vs Isolated Margin Comparison')
    print('='*70)

    top3 = [r for r in p3 if r['trades'] >= 10][:15]
    p4 = []; cnt = 0; t0 = time.time()
    for base in top3:
        tf = base['tf']; d = all_data[tf]
        cl,hi,lo,vol = d['close'],d['high'],d['low'],d['volume']
        yr, mk = d['years'], d['month_keys']
        mf = calc_ma(cl, base['ma_type'], base['fast_period'], vol)
        ms2 = calc_ma(cl, 'EMA', base['slow_period'], vol)
        ax = calc_adx(hi, lo, cl, base.get('adx_period', 20))
        rs = calc_rsi(cl, 14); at = calc_atr(hi, lo, cl, 14)
        wu = base['slow_period'] + 50
        for cm in [False, True]:
            for lev in [7, 10, 15]:
                for mpct in [0.10, 0.15, 0.20, 0.25]:
                    csl = min(base.get('sl_pct', 0.05), 0.9/lev)
                    for ml in [-0.15, -0.20, -0.25]:
                        r = fast_bt(cl, hi, lo, mf, ms2, ax, rs, at, yr, mk,
                                   adx_min=base.get('adx_min',35), rsi_min=base.get('rsi_min',30),
                                   rsi_max=base.get('rsi_max',65), entry_delay=base.get('entry_delay',0),
                                   sl_pct=csl, trail_act=base.get('trail_act',0.06),
                                   trail_pct=base.get('trail_pct',0.03), tp1_roi=base.get('tp1_roi'),
                                   leverage=lev, margin_pct=mpct, ml_limit=ml,
                                   init_bal=5000.0, warmup=wu, cross_margin=cm)
                        for k in ['ma_type','tf','fast_period','slow_period','adx_period',
                                  'adx_min','rsi_min','rsi_max','entry_delay','trail_act','trail_pct','tp1_roi']:
                            if k in base: r[k] = base[k]
                        r.update({'leverage':lev,'margin_pct':mpct,'sl_pct':csl,
                                 'ml_limit':ml,'cross_margin':cm})
                        p4.append(r)
                        cnt += 1
                        if cnt % 500 == 0: print('  P4: %d (%ds)' % (cnt, time.time()-t0))

    print('  P4: %d combos in %ds' % (len(p4), time.time()-t0))
    p4.sort(key=lambda x: x['score'], reverse=True)

    # Print cross vs isolated comparison
    iso = sorted([r for r in p4 if not r.get('cross_margin')], key=lambda x: x['score'], reverse=True)
    crs = sorted([r for r in p4 if r.get('cross_margin')], key=lambda x: x['score'], reverse=True)

    print('\n  === ISOLATED MARGIN Top 10 ===')
    for i, r in enumerate(iso[:10]):
        print('  %3d %5s %5s L%dx M%.0f%% Tr=%d Ret=%+.1f%% PF=%.2f MDD=%.1f%% FL=%d Sc=%.1f' %
              (i+1, r['ma_type'], r['tf'], r['leverage'], r['margin_pct']*100,
               r['trades'], r['return_pct'], r['pf'], r['mdd'], r['fl'], r['score']))

    print('\n  === CROSS MARGIN Top 10 ===')
    for i, r in enumerate(crs[:10]):
        print('  %3d %5s %5s L%dx M%.0f%% Tr=%d Ret=%+.1f%% PF=%.2f MDD=%.1f%% FL=%d Sc=%.1f' %
              (i+1, r['ma_type'], r['tf'], r['leverage'], r['margin_pct']*100,
               r['trades'], r['return_pct'], r['pf'], r['mdd'], r['fl'], r['score']))

    with open(os.path.join(OUTPUT_DIR, 'p4.json'), 'w') as f:
        json.dump(p4[:100], f, indent=2, default=str)

    # ===== VALIDATION =====
    print('\n' + '='*70)
    print('  VALIDATION (30x slippage)')
    print('='*70)

    # Select diverse top strategies
    final = {}
    for label, lst in [('BEST', p4), ('ISO', iso), ('CROSS', crs)]:
        for r in lst[:5]:
            key = '%s_%s_%d_%d_%d_%.2f_%s' % (r['ma_type'],r['tf'],r['fast_period'],
                  r['slow_period'],r['leverage'],r['margin_pct'],r.get('cross_margin',False))
            if key not in final: r['_label'] = label; final[key] = r

    print('  %d unique strategies' % len(final))
    validated = []
    for key, strat in list(final.items())[:10]:
        cm_str = 'CROSS' if strat.get('cross_margin') else 'ISO'
        print('\n  Validating: %s %s F%d/S%d L%dx M%.0f%% %s' % (
            strat['ma_type'], strat['tf'], strat['fast_period'], strat['slow_period'],
            strat['leverage'], strat['margin_pct']*100, cm_str))

        tf = strat['tf']; d = all_data[tf]
        cl,hi,lo,vol = d['close'],d['high'],d['low'],d['volume']
        yr, mk = d['years'], d['month_keys']
        mf = calc_ma(cl, strat['ma_type'], strat['fast_period'], vol)
        ms2 = calc_ma(cl, 'EMA', strat['slow_period'], vol)
        ax = calc_adx(hi, lo, cl, strat.get('adx_period', 20))
        rs = calc_rsi(cl, 14); at = calc_atr(hi, lo, cl, 14)
        wu = strat['slow_period'] + 50

        runs = []
        for run in range(30):
            np.random.seed(run * 42)
            slip = 1 + np.random.uniform(-0.0005, 0.0005, len(cl))
            r = fast_bt(cl*slip, hi*slip, lo*slip, mf, ms2, ax, rs, at, yr, mk,
                       adx_min=strat.get('adx_min',35), rsi_min=strat.get('rsi_min',30),
                       rsi_max=strat.get('rsi_max',65), entry_delay=strat.get('entry_delay',0),
                       sl_pct=strat.get('sl_pct',0.05), trail_act=strat.get('trail_act',0.06),
                       trail_pct=strat.get('trail_pct',0.03), tp1_roi=strat.get('tp1_roi'),
                       leverage=strat['leverage'], margin_pct=strat['margin_pct'],
                       ml_limit=strat.get('ml_limit',-0.20), init_bal=5000.0,
                       warmup=wu, cross_margin=strat.get('cross_margin',False))
            runs.append(r)

        bals = [r['balance'] for r in runs]
        pfs = [r['pf'] for r in runs]
        mdds = [r['mdd'] for r in runs]
        rets = [r['return_pct'] for r in runs]

        stats = {
            'bal_mean': np.mean(bals), 'bal_std': np.std(bals), 'bal_min': np.min(bals), 'bal_max': np.max(bals),
            'ret_mean': np.mean(rets), 'ret_std': np.std(rets),
            'pf_mean': np.mean(pfs), 'pf_std': np.std(pfs), 'pf_min': np.min(pfs),
            'mdd_mean': np.mean(mdds), 'mdd_max': np.max(mdds),
            'trades': np.mean([r['trades'] for r in runs]),
            'wr': np.mean([r['win_rate'] for r in runs]),
        }
        strat['val'] = stats
        validated.append(strat)

        print('    Bal: $%s +/- $%s (min $%s)' % ('{:,.0f}'.format(stats['bal_mean']),
              '{:,.0f}'.format(stats['bal_std']), '{:,.0f}'.format(stats['bal_min'])))
        print('    Ret: %+,.1f%% PF: %.2f (min %.2f) MDD: %.1f%% (max %.1f%%)' % (
              stats['ret_mean'], stats['pf_mean'], stats['pf_min'], stats['mdd_mean'], stats['mdd_max']))

    # ===== DETAILED TOP 3 =====
    print('\n' + '='*70)
    print('  DETAILED ANALYSIS - TOP 3')
    print('='*70)

    validated.sort(key=lambda x: x.get('val',{}).get('ret_mean',0) * min(x.get('val',{}).get('pf_mean',1),50) / max(x.get('val',{}).get('mdd_max',1),1), reverse=True)

    for idx, strat in enumerate(validated[:3]):
        cm_str = 'CROSS' if strat.get('cross_margin') else 'ISOLATED'
        print('\n' + '='*70)
        print('  STRATEGY #%d: %s %s F%d/S%d L%dx M%.0f%% %s' % (
            idx+1, strat['ma_type'], strat['tf'], strat['fast_period'], strat['slow_period'],
            strat['leverage'], strat['margin_pct']*100, cm_str))
        print('  ADX>=%d RSI %d-%d D=%d SL=%.1f%% Trail=%.0f%%/%.0f%%' % (
            strat.get('adx_min',35), strat.get('rsi_min',30), strat.get('rsi_max',65),
            strat.get('entry_delay',0), strat.get('sl_pct',0.05)*100,
            strat.get('trail_act',0.06)*100, strat.get('trail_pct',0.03)*100))
        print('='*70)

        p_detail = dict(strat)
        p_detail['init_bal'] = 5000.0
        p_detail['ml_limit'] = strat.get('ml_limit', -0.20)
        detail = detail_bt(all_data, p_detail)

        print('\n  [Yearly]')
        print('  %6s %12s %12s %10s %7s %4s %5s %5s %4s' % ('Year','Start','End','Return','Trades','SL','TSL','REV','FL'))
        print('  '+'-'*70)
        for y in sorted(detail['yearly'].keys()):
            ys = detail['yearly'][y]
            print('  %6d $%10s $%10s %+9.1f%% %6d %4d %5d %5d %4d' % (
                y, '{:,.0f}'.format(ys['start']), '{:,.0f}'.format(ys.get('end',ys['start'])),
                ys.get('ret',0), ys['trades'], ys.get('sl',0), ys.get('tsl',0), ys.get('rev',0), ys.get('fl',0)))

        print('\n  [Monthly]')
        print('  %8s %10s %7s %12s %12s' % ('Month','Return','Trades','PnL','Balance'))
        print('  '+'-'*55)
        for mk2 in sorted(detail['monthly'].keys()):
            ms = detail['monthly'][mk2]
            if ms['trades'] > 0:
                print('  %8s %+9.1f%% %6d $%10s $%10s' % (
                    mk2, ms.get('ret',0), ms['trades'],
                    '{:,.0f}'.format(ms['pnl']), '{:,.0f}'.format(ms.get('end',ms['start']))))

    # Save all
    total = len(p1) + len(p2) + len(p3) + len(p4)
    with open(os.path.join(OUTPUT_DIR, 'validated.json'), 'w') as f:
        json.dump(validated[:5], f, indent=2, default=str)

    print('\n' + '='*70)
    print('  TOTAL: %s combos tested in %.1f min' % ('{:,}'.format(total), (time.time()-t_start)/60))
    print('='*70)


if __name__ == '__main__':
    main()
