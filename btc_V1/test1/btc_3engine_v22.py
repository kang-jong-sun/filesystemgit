"""
v22.0 3-Engine Portfolio System (GPT-5 Collaboration)
Engine 1: Core Trend - WMA(3)/EMA(200) ADX>=35 D5
Engine 2: Accelerated - SMA(14)/EMA(200) ADX>=32 D3
Engine 3: Re-entry - WMA(3) pullback in trend ADX>=30
+ ATR dynamic SL, 2-stage partial exit, regime filter, dynamic sizing
"""
import sys, time, json, os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = r"D:\filesystem\futures\btc_V1\test1\btc_usdt_5m_merged.csv"
OUTPUT_DIR = r"D:\filesystem\futures\btc_V1\test1\v22_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ INDICATORS ============
def ema(d, p): return pd.Series(d).ewm(span=p, adjust=False).mean().values
def sma(d, p): return pd.Series(d).rolling(p, min_periods=p).mean().values
def wma(d, p):
    w = np.arange(1, p+1, dtype=float)
    return pd.Series(d).rolling(p).apply(lambda x: np.dot(x, w)/w.sum(), raw=True).values
def calc_rsi(cl, p=14):
    delta = np.diff(cl, prepend=cl[0])
    g = np.where(delta>0, delta, 0.0); l = np.where(delta<0, -delta, 0.0)
    ga = pd.Series(g).ewm(alpha=1/p, min_periods=p, adjust=False).mean().values
    la = pd.Series(l).ewm(alpha=1/p, min_periods=p, adjust=False).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(la>0, ga/la, 100.0)
    return 100 - 100/(1+rs)
def calc_adx(hi, lo, cl, p=20):
    tr1=hi-lo; tr2=np.abs(hi-np.roll(cl,1)); tr3=np.abs(lo-np.roll(cl,1))
    tr1[0]=tr2[0]=tr3[0]=0; tr=np.maximum(np.maximum(tr1,tr2),tr3)
    up=hi-np.roll(hi,1); dn=np.roll(lo,1)-lo; up[0]=dn[0]=0
    pdm=np.where((up>dn)&(up>0),up,0.0); mdm=np.where((dn>up)&(dn>0),dn,0.0)
    a=1.0/p
    atr_v=pd.Series(tr).ewm(alpha=a,min_periods=p,adjust=False).mean().values
    pds=pd.Series(pdm).ewm(alpha=a,min_periods=p,adjust=False).mean().values
    mds=pd.Series(mdm).ewm(alpha=a,min_periods=p,adjust=False).mean().values
    with np.errstate(divide='ignore',invalid='ignore'):
        pdi=np.where(atr_v>0,100*pds/atr_v,0); mdi=np.where(atr_v>0,100*mds/atr_v,0)
        ds=pdi+mdi; dx=np.where(ds>0,100*np.abs(pdi-mdi)/ds,0)
    return pd.Series(dx).ewm(alpha=a,min_periods=p,adjust=False).mean().values
def calc_atr(hi, lo, cl, p=14):
    tr1=hi-lo; tr2=np.abs(hi-np.roll(cl,1)); tr3=np.abs(lo-np.roll(cl,1))
    tr1[0]=tr2[0]=tr3[0]=0; tr=np.maximum(np.maximum(tr1,tr2),tr3)
    return pd.Series(tr).ewm(alpha=1.0/p,min_periods=p,adjust=False).mean().values

# ============ DATA ============
def load_data():
    print('Loading data...')
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    d = df.set_index('timestamp')
    r30 = d.resample('30min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
    ts = pd.to_datetime(r30['timestamp'].values)
    cl = r30['close'].values.astype(float)
    hi = r30['high'].values.astype(float)
    lo = r30['low'].values.astype(float)
    vo = r30['volume'].values.astype(float)
    print('  30min: %d bars' % len(cl))
    return {
        'cl':cl, 'hi':hi, 'lo':lo, 'vo':vo,
        'ts': r30['timestamp'].values,
        'yr': ts.year.values.astype(np.int32),
        'mk': (ts.year.values*100 + ts.month.values).astype(np.int32),
        'n': len(cl),
        'wma3': wma(cl,3), 'sma14': sma(cl,14), 'ema200': ema(cl,200),
        'adx20': calc_adx(hi,lo,cl,20), 'rsi14': calc_rsi(cl,14),
        'atr14': calc_atr(hi,lo,cl,14),
    }

# ============ REGIME ============
def calc_regime(d):
    """0=Range, 1=Trend Bull, -1=Trend Bear"""
    n = d['n']; cl = d['cl']; e200 = d['ema200']; adx_v = d['adx20']
    regime = np.zeros(n, dtype=int)
    for i in range(220, n):
        slope = (e200[i] - e200[i-20]) / e200[i-20] * 100 if e200[i-20] > 0 else 0
        if cl[i] > e200[i] and slope > 0.25 and adx_v[i] >= 30:
            regime[i] = 1
        elif cl[i] < e200[i] and slope < -0.25 and adx_v[i] >= 30:
            regime[i] = -1
    return regime

# ============ SINGLE ENGINE BACKTEST ============
def run_engine(d, regime, engine_type, base_margin=0.20, leverage=10, init_bal=5000.0):
    """
    engine_type: 'core', 'accel', 'reentry'
    Returns dict with all results
    """
    cl,hi,lo = d['cl'],d['hi'],d['lo']
    yr,mk = d['yr'],d['mk']
    wma3,sma14,e200 = d['wma3'],d['sma14'],d['ema200']
    adx_v,atr_v,rsi_v = d['adx20'],d['atr14'],d['rsi14']
    n = d['n']

    # Engine-specific params
    if engine_type == 'core':
        ma_fast = wma3; adx_min = 35; delay = 5
        atr_sl_mult = 2.2; sl_max = 0.032; sl_min = 0.014
        trail_start = 0.030; risk_per_trade = 0.012
        tp1_pct = 0.025; tp2_pct = 0.050
        time_exit = 40  # bars
        time_min_profit = 0.012
    elif engine_type == 'accel':
        ma_fast = sma14; adx_min = 32; delay = 3
        atr_sl_mult = 1.9; sl_max = 0.028; sl_min = 0.012
        trail_start = 0.025; risk_per_trade = 0.009
        tp1_pct = 0.025; tp2_pct = 0.050
        time_exit = 24
        time_min_profit = 0.008
    else:  # reentry
        ma_fast = wma3; adx_min = 30; delay = 0
        atr_sl_mult = 1.5; sl_max = 0.020; sl_min = 0.010
        trail_start = 0.018; risk_per_trade = 0.007
        tp1_pct = 0.018; tp2_pct = 0.035
        time_exit = 20  # 10 hours
        time_min_profit = 0.005

    fee = 0.0004
    warmup = 250
    bal = init_bal; peak = init_bal; mdd = 0.0
    trades = []
    in_pos = False; p_dir = 0; p_entry = 0.0; p_size = 0.0; p_orig = 0.0
    p_sl = 0.0; p_high = 0.0; p_low = 0.0
    tp1_done = False; tp2_done = False; entry_bar = 0
    be_done = False  # breakeven moved

    # Pending signal
    pend = 0; pcnt = 0

    # Monthly loss limit
    cur_m = -1; m_start = bal; m_locked = False

    # Recent results for dynamic sizing
    recent_pnl = []

    # Cross detection
    valid = ~(np.isnan(ma_fast) | np.isnan(e200))
    above = ma_fast > e200

    # For reentry: detect pullback
    # Pullback = price dipped below WMA3 for 1-3 bars then recovered
    if engine_type == 'reentry':
        below_wma = cl < wma3
        above_wma = cl > wma3

    for i in range(warmup, n):
        c = cl[i]; h = hi[i]; l = lo[i]
        m = mk[i]
        if m != cur_m:
            cur_m = m; m_start = bal; m_locked = False
        if not m_locked and m_start > 0:
            if (bal - m_start) / m_start <= -0.20:
                m_locked = True

        # Daily loss check (simplified: 48 bars = 24h)
        # Weekly loss check: skipped for simplicity

        at = atr_v[i]
        if np.isnan(at) or at <= 0: at = c * 0.015

        # Regime filter
        r = regime[i]
        if engine_type == 'reentry' and r == 0:
            # Range: skip reentry entirely
            if not in_pos:
                if pend != 0: pend = 0; pcnt = 0
                continue
        if engine_type == 'accel' and r == 0:
            # Range: tighten conditions
            adx_threshold = 38
        else:
            adx_threshold = adx_min

        # ---- POSITION MANAGEMENT ----
        if in_pos:
            bars_held = i - entry_bar

            # Dynamic trail width based on ADX
            adx_now = adx_v[i] if not np.isnan(adx_v[i]) else 30
            if adx_now >= 45:
                trail_pct = 0.026
            elif adx_now >= 35:
                trail_pct = 0.018
            else:
                trail_pct = 0.016

            if p_dir == 1:
                # SL check
                if l <= p_sl:
                    pnl = p_size * (p_sl - p_entry) / p_entry - p_size * fee
                    bal += pnl
                    trades.append({'yr':yr[i],'pnl':pnl,'reason':'SL','bal':bal,'bars':bars_held})
                    recent_pnl.append(pnl)
                    in_pos = False
                    continue

                if h > p_high: p_high = h
                profit_pct = (c - p_entry) / p_entry

                # Breakeven move
                if not be_done and profit_pct >= 0.018:
                    new_sl = p_entry + p_entry * 0.001  # Entry + 0.1%
                    if new_sl > p_sl: p_sl = new_sl
                    be_done = True

                # Partial TP1: 30% at +2.5%
                if not tp1_done and profit_pct >= tp1_pct:
                    ea = p_orig * 0.30
                    pp = ea * (c - p_entry) / p_entry - ea * fee
                    bal += pp; p_size -= ea; tp1_done = True

                # Partial TP2: 30% at +5.0%
                if not tp2_done and profit_pct >= tp2_pct:
                    ea = min(p_orig * 0.30, p_size)
                    pp = ea * (c - p_entry) / p_entry - ea * fee
                    bal += pp; p_size -= ea; tp2_done = True

                # Trailing
                if profit_pct >= trail_start:
                    trail_sl = p_high * (1 - trail_pct)
                    if trail_sl > p_sl: p_sl = trail_sl

                # TSL check
                if c <= p_sl and p_sl > p_entry:
                    pnl = p_size * (p_sl - p_entry) / p_entry - p_size * fee
                    bal += pnl
                    trades.append({'yr':yr[i],'pnl':pnl,'reason':'TSL','bal':bal,'bars':bars_held})
                    recent_pnl.append(pnl)
                    in_pos = False
                    continue

                # Time exit
                if bars_held >= time_exit and profit_pct < time_min_profit:
                    pnl = p_size * (c - p_entry) / p_entry - p_size * fee
                    bal += pnl
                    trades.append({'yr':yr[i],'pnl':pnl,'reason':'TIME','bal':bal,'bars':bars_held})
                    recent_pnl.append(pnl)
                    in_pos = False
                    continue

            else:  # SHORT
                if h >= p_sl:
                    pnl = p_size * (p_entry - p_sl) / p_entry - p_size * fee
                    bal += pnl
                    trades.append({'yr':yr[i],'pnl':pnl,'reason':'SL','bal':bal,'bars':bars_held})
                    recent_pnl.append(pnl)
                    in_pos = False
                    continue

                if l < p_low: p_low = l
                profit_pct = (p_entry - c) / p_entry

                if not be_done and profit_pct >= 0.018:
                    new_sl = p_entry - p_entry * 0.001
                    if new_sl < p_sl or p_sl == 0: p_sl = new_sl
                    be_done = True

                if not tp1_done and profit_pct >= tp1_pct:
                    ea = p_orig * 0.30
                    pp = ea * (p_entry - c) / p_entry - ea * fee
                    bal += pp; p_size -= ea; tp1_done = True

                if not tp2_done and profit_pct >= tp2_pct:
                    ea = min(p_orig * 0.30, p_size)
                    pp = ea * (p_entry - c) / p_entry - ea * fee
                    bal += pp; p_size -= ea; tp2_done = True

                if profit_pct >= trail_start:
                    trail_sl = p_low * (1 + trail_pct)
                    if trail_sl < p_sl or p_sl > p_entry: p_sl = trail_sl

                if c >= p_sl and p_sl < p_entry and p_sl > 0:
                    pnl = p_size * (p_entry - p_sl) / p_entry - p_size * fee
                    bal += pnl
                    trades.append({'yr':yr[i],'pnl':pnl,'reason':'TSL','bal':bal,'bars':bars_held})
                    recent_pnl.append(pnl)
                    in_pos = False
                    continue

                if bars_held >= time_exit and profit_pct < time_min_profit:
                    pnl = p_size * (p_entry - c) / p_entry - p_size * fee
                    bal += pnl
                    trades.append({'yr':yr[i],'pnl':pnl,'reason':'TIME','bal':bal,'bars':bars_held})
                    recent_pnl.append(pnl)
                    in_pos = False
                    continue

        # ---- SIGNAL DETECTION ----
        if i < warmup + 1: continue

        sig = 0
        adx_now = adx_v[i] if not np.isnan(adx_v[i]) else 0

        if engine_type in ('core', 'accel'):
            # MA cross detection
            if valid[i] and valid[i-1]:
                cross_up = above[i] and not above[i-1]
                cross_dn = not above[i] and above[i-1]

                if cross_up and adx_now >= adx_threshold:
                    sig = 1
                elif cross_dn and adx_now >= adx_threshold:
                    sig = -1

            # Apply delay
            if sig != 0 and delay > 0:
                pend = sig; pcnt = delay; sig = 0
            if pcnt > 0:
                pcnt -= 1
                if pcnt == 0:
                    # Verify cross still valid
                    if pend == 1 and ma_fast[i] > e200[i]:
                        sig = pend
                    elif pend == -1 and ma_fast[i] < e200[i]:
                        sig = pend
                    pend = 0

        else:  # reentry
            # Pullback re-entry: price dipped below WMA3 then recovered
            if adx_now >= adx_threshold and valid[i]:
                if ma_fast[i] > e200[i] and r == 1:  # Uptrend
                    # Check: recent dip below WMA3 (within 10 bars)
                    dip_found = False
                    for j in range(max(i-10, warmup), i):
                        if cl[j] < wma3[j]:
                            dip_found = True
                            break
                    if dip_found and c > wma3[i]:
                        # Confirm: close above 5-bar high
                        if i >= 5 and c > max(hi[i-5:i]):
                            sig = 1

                elif ma_fast[i] < e200[i] and r == -1:  # Downtrend
                    dip_found = False
                    for j in range(max(i-10, warmup), i):
                        if cl[j] > wma3[j]:
                            dip_found = True
                            break
                    if dip_found and c < wma3[i]:
                        if i >= 5 and c < min(lo[i-5:i]):
                            sig = -1

        # REV: reverse signal closes existing position
        if sig != 0 and in_pos and p_dir != sig:
            pnl = p_size * (c - p_entry) / p_entry * p_dir - p_size * fee
            bal += pnl
            trades.append({'yr':yr[i],'pnl':pnl,'reason':'REV','bal':bal,'bars':i-entry_bar})
            recent_pnl.append(pnl)
            in_pos = False

        # Same direction signal while in position: skip
        if sig != 0 and in_pos:
            continue

        # New entry
        if sig != 0 and not in_pos and not m_locked and bal > 10:
            # Dynamic sizing
            regime_mult = 1.0 if r != 0 else 0.5
            if adx_now >= 45: regime_mult = 1.15

            # Recent performance factor
            perf_mult = 1.0
            if len(recent_pnl) >= 5:
                if sum(recent_pnl[-5:]) <= 0: perf_mult = 0.75
            if len(recent_pnl) >= 3:
                if all(p <= 0 for p in recent_pnl[-3:]): perf_mult = 0.5

            actual_margin = base_margin * regime_mult * perf_mult
            actual_margin = max(0.05, min(actual_margin, 0.30))  # Cap

            # Risk-based sizing
            sl_dist = at * atr_sl_mult
            sl_pct = sl_dist / c
            sl_pct = max(sl_min, min(sl_pct, sl_max))
            sl_dist = c * sl_pct

            risk_amount = bal * risk_per_trade * regime_mult * perf_mult
            pos_size = risk_amount / sl_pct
            pos_size = min(pos_size, bal * actual_margin * leverage)  # Cap by margin

            if pos_size < 10: continue

            bal -= pos_size * fee
            p_dir = sig; p_entry = c; p_size = pos_size; p_orig = pos_size
            p_sl = c - sl_dist if sig == 1 else c + sl_dist
            p_high = c; p_low = c
            tp1_done = False; tp2_done = False; be_done = False
            entry_bar = i
            in_pos = True

        # MDD tracking
        if bal > peak: peak = bal
        if peak > 0:
            dd = (peak - bal) / peak
            if dd > mdd: mdd = dd

    # Close remaining
    if in_pos and p_size > 0:
        c = cl[-1]
        pnl = p_size * (c - p_entry) / p_entry * p_dir - p_size * fee
        bal += pnl
        trades.append({'yr':yr[-1],'pnl':pnl,'reason':'END','bal':bal,'bars':n-entry_bar})

    return compile(trades, bal, init_bal, mdd, engine_type)


def compile(trades, bal, init, mdd, name):
    tc = len(trades)
    if tc == 0:
        return {'name':name,'balance':bal,'ret':0,'trades':0,'pf':0,'mdd':0,
                'wr':0,'sl':0,'tsl':0,'rev':0,'tp':0,'time':0,'yearly':{}}
    pnls = np.array([t['pnl'] for t in trades])
    w = pnls > 0; l = pnls <= 0
    gp = pnls[w].sum() if w.any() else 0
    gl = abs(pnls[l].sum()) if l.any() else 0.001
    pf = min(gp/gl, 999.99)

    yrs = {}
    for t in trades:
        y = int(t['yr'])
        if y not in yrs: yrs[y] = {'trades':0,'pnl':0.0,'sl':0,'tsl':0,'rev':0}
        yrs[y]['trades'] += 1; yrs[y]['pnl'] += t['pnl']
        if t['reason'] in yrs[y]: yrs[y][t['reason'].lower()] = yrs[y].get(t['reason'].lower(),0) + 1

    return {
        'name':name, 'balance':round(bal,2), 'ret':round((bal-init)/init*100,1),
        'trades':tc, 'pf':round(pf,2), 'mdd':round(mdd*100,1),
        'wr':round(w.sum()/tc*100,1),
        'sl':sum(1 for t in trades if t['reason']=='SL'),
        'tsl':sum(1 for t in trades if t['reason']=='TSL'),
        'rev':sum(1 for t in trades if t['reason']=='REV'),
        'time':sum(1 for t in trades if t['reason']=='TIME'),
        'yearly':yrs,
        'avg_bars': round(np.mean([t.get('bars',0) for t in trades]),1),
    }


# ============ PORTFOLIO ============
def run_portfolio(d, regime, allocations, base_margin=0.20, leverage=10, init_bal=5000.0):
    results = {}
    for eng, alloc in allocations.items():
        sub = init_bal * alloc
        r = run_engine(d, regime, eng, base_margin=base_margin, leverage=leverage, init_bal=sub)
        r['alloc'] = alloc
        results[eng] = r

    total = sum(r['balance'] for r in results.values())
    total_trades = sum(r['trades'] for r in results.values())
    return {
        'engines': results,
        'total_bal': round(total, 2),
        'total_ret': round((total - init_bal) / init_bal * 100, 1),
        'total_trades': total_trades,
        'max_mdd': round(max(r['mdd'] for r in results.values()), 1),
    }


# ============ MAIN ============
def main():
    t0 = time.time()
    print('='*70)
    print('  v22.0 3-Engine Portfolio Backtest')
    print('='*70)

    d = load_data()
    regime = calc_regime(d)

    # Count regime bars
    trend_pct = np.sum(regime != 0) / len(regime) * 100
    print('  Regime: %.1f%% trending, %.1f%% ranging' % (trend_pct, 100-trend_pct))

    # ===== Individual engines =====
    print('\n' + '='*70)
    print('  INDIVIDUAL ENGINE RESULTS (M20%% L10x)')
    print('='*70)

    for eng in ['core', 'accel', 'reentry']:
        r = run_engine(d, regime, eng, base_margin=0.20, leverage=10, init_bal=5000.0)
        print('\n  --- %s ---' % r['name'])
        print('  $%.0f (%+.1f%%) | Tr=%d PF=%.2f MDD=%.1f%% WR=%.1f%%' % (
            r['balance'], r['ret'], r['trades'], r['pf'], r['mdd'], r['wr']))
        print('  SL=%d TSL=%d REV=%d TIME=%d | AvgBars=%.0f' % (
            r['sl'], r['tsl'], r['rev'], r['time'], r['avg_bars']))
        for y in sorted(r['yearly'].keys()):
            ys = r['yearly'][y]
            print('    %d: %d trades, PnL $%.0f' % (y, ys['trades'], ys['pnl']))

    # ===== Margin/Leverage sweep =====
    print('\n' + '='*70)
    print('  MARGIN/LEVERAGE SWEEP (Core engine)')
    print('='*70)
    print('  %3s %4s %10s %7s %6s %4s %4s' % ('L','M%','Return','PF','MDD%','Tr','SL'))
    print('  '+'-'*45)
    for lev in [7, 10, 15]:
        for mpct in [0.15, 0.20, 0.25, 0.30]:
            r = run_engine(d, regime, 'core', base_margin=mpct, leverage=lev, init_bal=5000.0)
            print('  %2dx %3.0f%% %+9.1f%% %6.2f %5.1f%% %4d %4d' % (
                lev, mpct*100, r['ret'], r['pf'], r['mdd'], r['trades'], r['sl']))

    # ===== Portfolio combos =====
    print('\n' + '='*70)
    print('  PORTFOLIO COMBINATIONS')
    print('='*70)

    combos = [
        ('50/30/20 M20 L10', {'core':0.50,'accel':0.30,'reentry':0.20}, 0.20, 10),
        ('50/30/20 M25 L10', {'core':0.50,'accel':0.30,'reentry':0.20}, 0.25, 10),
        ('50/30/20 M20 L15', {'core':0.50,'accel':0.30,'reentry':0.20}, 0.20, 15),
        ('60/25/15 M20 L10', {'core':0.60,'accel':0.25,'reentry':0.15}, 0.20, 10),
        ('60/25/15 M25 L10', {'core':0.60,'accel':0.25,'reentry':0.15}, 0.25, 10),
        ('70/30/0  M20 L10', {'core':0.70,'accel':0.30,'reentry':0.00}, 0.20, 10),
        ('Core100  M20 L10', {'core':1.00,'accel':0.00,'reentry':0.00}, 0.20, 10),
        ('Core100  M25 L10', {'core':1.00,'accel':0.00,'reentry':0.00}, 0.25, 10),
    ]

    print('  %22s %10s %5s %7s %6s' % ('Config','Return','Tr','PF_max','MDD%'))
    print('  '+'-'*55)
    best_port = None; best_score = 0
    for label, alloc, bm, lev in combos:
        p = run_portfolio(d, regime, alloc, base_margin=bm, leverage=lev, init_bal=5000.0)
        # Blended PF estimate
        pf_best = max(r['pf'] for r in p['engines'].values() if r['trades'] > 0) if p['total_trades'] > 0 else 0
        print('  %22s %+9.1f%% %5d %6.2f %5.1f%%' % (
            label, p['total_ret'], p['total_trades'], pf_best, p['max_mdd']))
        for eng_name, r in p['engines'].items():
            if r['trades'] > 0:
                print('    %8s: $%.0f (%+.1f%%) Tr=%d PF=%.2f MDD=%.1f%%' % (
                    eng_name, r['balance'], r['ret'], r['trades'], r['pf'], r['mdd']))

        score = p['total_ret'] / max(p['max_mdd'], 5)
        if score > best_score:
            best_score = score; best_port = (label, p)

    # ===== 30x Validation for best =====
    print('\n' + '='*70)
    print('  30x VALIDATION')
    print('='*70)

    for eng_name in ['core', 'accel', 'reentry']:
        print('\n  --- %s ---' % eng_name)
        runs = []
        for run in range(30):
            np.random.seed(run * 42)
            slip = 1 + np.random.uniform(-0.0005, 0.0005, d['n'])
            d_s = dict(d)
            d_s['cl'] = d['cl']*slip; d_s['hi'] = d['hi']*slip; d_s['lo'] = d['lo']*slip
            d_s['wma3'] = wma(d_s['cl'],3); d_s['sma14'] = sma(d_s['cl'],14)
            d_s['ema200'] = ema(d_s['cl'],200)
            d_s['adx20'] = calc_adx(d_s['hi'],d_s['lo'],d_s['cl'],20)
            d_s['atr14'] = calc_atr(d_s['hi'],d_s['lo'],d_s['cl'],14)
            d_s['rsi14'] = calc_rsi(d_s['cl'],14)
            reg_s = calc_regime(d_s)
            r = run_engine(d_s, reg_s, eng_name, base_margin=0.20, leverage=10, init_bal=5000.0)
            runs.append(r)
        bals = [r['balance'] for r in runs]
        pfs = [r['pf'] for r in runs]
        mdds = [r['mdd'] for r in runs]
        print('  Bal: $%.0f +/- $%.0f (min $%.0f)' % (np.mean(bals), np.std(bals), np.min(bals)))
        print('  PF: %.2f +/- %.2f (min %.2f)' % (np.mean(pfs), np.std(pfs), np.min(pfs)))
        print('  MDD: %.1f%% (max %.1f%%)' % (np.mean(mdds), np.max(mdds)))
        print('  Trades: %.0f' % np.mean([r['trades'] for r in runs]))

    # Save
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump({'best_portfolio': best_port[0] if best_port else 'N/A'}, f, indent=2)

    print('\n' + '='*70)
    print('  COMPLETE in %.1f min' % ((time.time()-t0)/60))
    print('='*70)

if __name__ == '__main__':
    main()
