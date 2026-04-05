"""
v16.2 BTC/USDT Futures Backtest Engine v3
- State-based entry (trend state transitions, not single-candle cross)
- Proper signal forward-propagation
- All 75 months coverage
- Triple Engine with distinct timeframes
"""

import pandas as pd
import numpy as np
import warnings, os, time
warnings.filterwarnings('ignore')

DATA_DIR = r"D:\filesystem\futures\btc_V1\test3"

# ============================================================
# INDICATORS
# ============================================================
def calc_wma(s, p):
    w = np.arange(1, p+1, dtype=float)
    return s.rolling(p).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)

def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()

def calc_hma(s, p):
    h = max(int(p/2),1); sq = max(int(np.sqrt(p)),1)
    return calc_wma(2*calc_wma(s,h)-calc_wma(s,p), sq)

def calc_rsi(s, p=14):
    d = s.diff(); g = d.where(d>0,0.0); l = (-d).where(d<0,0.0)
    ag = g.ewm(alpha=1/p, min_periods=p).mean()
    al = l.ewm(alpha=1/p, min_periods=p).mean()
    return 100 - (100 / (1 + ag/al.replace(0, np.nan)))

def calc_adx(h, l, c, p=14):
    tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/p, min_periods=p).mean()
    um = h - h.shift(1); dm = l.shift(1) - l
    pdm = pd.Series(np.where((um>dm)&(um>0), um, 0.0), index=h.index).ewm(alpha=1/p, min_periods=p).mean()
    mdm = pd.Series(np.where((dm>um)&(dm>0), dm, 0.0), index=h.index).ewm(alpha=1/p, min_periods=p).mean()
    pdi = 100*pdm/atr.replace(0,np.nan); mdi = 100*mdm/atr.replace(0,np.nan)
    dx = 100*abs(pdi-mdi)/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/p, min_periods=p).mean(), pdi, mdi, atr

def calc_atr(h, l, c, p=14):
    tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, min_periods=p).mean()

def calc_macd(s, f=12, sl=26, sg=9):
    ml = calc_ema(s,f)-calc_ema(s,sl); return ml, calc_ema(ml,sg), ml-calc_ema(ml,sg)

def calc_bb_pctb(s, p=20, sd=2.0):
    m = s.rolling(p).mean(); st = s.rolling(p).std()
    return (s - (m-sd*st)) / ((m+sd*st)-(m-sd*st))


# ============================================================
# DATA LOAD & INDICATOR COMPUTATION
# ============================================================
def load_data():
    print("[1] Loading 5m data...")
    t0 = time.time()
    files = [os.path.join(DATA_DIR, f"btc_usdt_5m_2020_to_now_part{i}.csv") for i in range(1,4)]
    df = pd.concat([pd.read_csv(f, parse_dates=['timestamp']) for f in files], ignore_index=True)
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    print(f"  {len(df):,} rows | {df.index[0]} ~ {df.index[-1]} | {time.time()-t0:.1f}s")
    return df


def build_tf_signals(df_5m):
    """Build all TF data and compute indicators, return dict of TF dataframes"""
    print("[2] Building multi-timeframe signals...")
    t0 = time.time()

    def resample(df, rule):
        return df.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last',
                                       'volume':'sum','quote_volume':'sum','trades':'sum'}).dropna()

    tfs = {
        '5m': df_5m.copy(),
        '15m': resample(df_5m, '15min'),
        '30m': resample(df_5m, '30min'),
        '1h': resample(df_5m, '1h'),
    }

    # Indicator configs per engine's primary TF
    configs = {
        '30m': {'fast':'WMA','fl':3,'slow':'EMA','sl':200,'adx_p':20},
        '15m': {'fast':'WMA','fl':3,'slow':'EMA','sl':150,'adx_p':20},
        '5m':  {'fast':'HMA','fl':5,'slow':'EMA','sl':100,'adx_p':14},
        '1h':  {'fast':'WMA','fl':3,'slow':'EMA','sl':200,'adx_p':20},
    }

    for tf_key, cfg in configs.items():
        d = tfs[tf_key]
        c = d['close']; h = d['high']; l = d['low']; v = d['volume']

        # Fast MA
        if cfg['fast'] == 'WMA': d['fast_ma'] = calc_wma(c, cfg['fl'])
        elif cfg['fast'] == 'HMA': d['fast_ma'] = calc_hma(c, cfg['fl'])
        elif cfg['fast'] == 'EMA': d['fast_ma'] = calc_ema(c, cfg['fl'])

        d['slow_ma'] = calc_ema(c, cfg['sl'])
        d['adx'], d['pdi'], d['mdi'], _ = calc_adx(h, l, c, cfg['adx_p'])
        d['rsi'] = calc_rsi(c, 14)
        d['atr14'] = calc_atr(h, l, c, 14)
        d['atr_sma50'] = d['atr14'].rolling(50).mean()
        _, _, d['macd_hist'] = calc_macd(c, 12, 26, 9)
        d['bb_pctb'] = calc_bb_pctb(c, 20, 2.0)
        d['vol_ratio'] = v / v.rolling(20).mean()
        d['adx_slope'] = d['adx'] - d['adx'].shift(3)

        # Trend state: 1 = bullish (fast > slow), 0 = bearish
        d['trend'] = (d['fast_ma'] > d['slow_ma']).astype(int)
        # State change detection
        d['trend_change'] = d['trend'].diff().fillna(0).astype(int)
        # trend_change == 1 → just turned bullish; trend_change == -1 → just turned bearish

    print(f"  Done | {time.time()-t0:.1f}s")
    return tfs


def align_signals(tfs, idx_5m):
    """Align all TF signals to 5m index using forward-fill"""
    print("[3] Aligning to 5m timeline...")
    t0 = time.time()

    cols_needed = ['trend','trend_change','adx','rsi','atr14','atr_sma50',
                   'macd_hist','bb_pctb','vol_ratio','adx_slope','close','fast_ma','slow_ma']

    aligned = {}
    for tf_key in ['5m','15m','30m','1h']:
        d = tfs[tf_key]
        if tf_key == '5m':
            sub = d[cols_needed].copy()
        else:
            sub = d[cols_needed].reindex(idx_5m, method='ffill')

        # For trend_change, we need special handling:
        # After ffill, the trend_change would persist. We need it to be 0 after the first bar.
        # Instead, create a "time since last state change" column
        if tf_key != '5m':
            # Reindex trend_change without ffill (NaN for non-aligned bars)
            tc_raw = d['trend_change'].reindex(idx_5m)
            # Forward fill with 0 (only the transition bar has ±1)
            sub['trend_change'] = tc_raw.fillna(0).astype(int)

        sub.columns = [f'{tf_key}_{c}' for c in cols_needed]
        aligned[tf_key] = sub

    master = pd.concat([tfs['5m'][['open','high','low','close','volume']]] +
                       list(aligned.values()), axis=1)

    # Create entry windows: bars since last trend change per TF
    # This tells us "how many 5m bars since the trend changed on this TF"
    for tf_key in ['5m','15m','30m']:
        tc_col = f'{tf_key}_trend_change'
        # Find bars where trend changed (value != 0)
        changed = master[tc_col] != 0
        # Create cumulative group ID
        groups = changed.cumsum()
        # Count bars within each group
        master[f'{tf_key}_bars_since_change'] = groups.groupby(groups).cumcount()

    print(f"  Shape: {master.shape} | {time.time()-t0:.1f}s")
    return master


# ============================================================
# ENGINE CONFIG
# ============================================================
ENGINE = {
    'A': {'name':'Sniper','tf':'30m','adx_min':35,'rsi_lo':30,'rsi_hi':68,
           'min_score':45,'lev':10,'ratio':0.20,'atr_sl':2.0,'atr_tp1':5.0,'atr_tp2':9.0,
           'trail_atr':2.5,'trail_act_pct':0.08,
           'entry_window':12,'min_gap':36,'max_pos':1,  # 12 bars=60min window, 36 bars=3h gap
           'mtf_tfs':['30m','15m','1h']},
    'B': {'name':'Core','tf':'15m','adx_min':28,'rsi_lo':25,'rsi_hi':72,
           'min_score':30,'lev':7,'ratio':0.18,'atr_sl':2.5,'atr_tp1':4.0,'atr_tp2':7.0,
           'trail_atr':2.5,'trail_act_pct':0.06,
           'entry_window':18,'min_gap':18,'max_pos':1,  # 18 bars=90min window
           'mtf_tfs':['15m','30m','5m']},
    'C': {'name':'Swing','tf':'5m','adx_min':22,'rsi_lo':22,'rsi_hi':78,
           'min_score':20,'lev':5,'ratio':0.12,'atr_sl':3.0,'atr_tp1':3.5,'atr_tp2':6.0,
           'trail_atr':3.0,'trail_act_pct':0.05,
           'entry_window':24,'min_gap':12,'max_pos':1,  # 24 bars=2h window
           'mtf_tfs':['5m','15m']},
}


# ============================================================
# BACKTEST
# ============================================================
class Pos:
    __slots__ = ['d','ep','et','sz','lev','sl','tp1','tp2','eng','sc','atr',
                 'hi','lo','pkr','t1','t2','rem','ta','tsl','pp','slp']
    def __init__(s, d, ep, et, sz, lev, sl, tp1, tp2, eng, sc, atr, slp):
        s.d=d; s.ep=ep; s.et=et; s.sz=sz; s.lev=lev; s.sl=sl; s.tp1=tp1; s.tp2=tp2
        s.eng=eng; s.sc=sc; s.atr=atr; s.slp=slp
        s.hi=ep; s.lo=ep; s.pkr=0.0; s.t1=False; s.t2=False; s.rem=1.0
        s.ta=False; s.tsl=None; s.pp=0.0


def score(adx_sl, vr, mtf_n, bbp, mh, rsi, d):
    s = 0
    if not np.isnan(adx_sl) and adx_sl > 0: s += 10
    if not np.isnan(vr) and vr > 1.0: s += min(10, int(vr*5))
    s += min(20, mtf_n * 7)
    if not np.isnan(bbp):
        if d==1 and bbp < 0.75: s += 8
        elif d==-1 and bbp > 0.25: s += 8
    if not np.isnan(mh):
        if (d==1 and mh>0) or (d==-1 and mh<0): s += 10
    if not np.isnan(rsi):
        if d==1 and 30<=rsi<=62: s += 8
        elif d==-1 and 38<=rsi<=70: s += 8
    return s


def run():
    FEE = 0.0004
    INIT = 3000.0

    df_5m = load_data()
    tfs = build_tf_signals(df_5m)
    master = align_signals(tfs, df_5m.index)

    # Convert to numpy for speed
    h_arr = master['high'].values
    l_arr = master['low'].values
    c_arr = master['close'].values
    ts_arr = master.index
    n = len(master)

    # Pre-extract numpy arrays for each TF's signals
    sig = {}
    for tf in ['5m','15m','30m','1h']:
        sig[tf] = {
            'trend': master[f'{tf}_trend'].values,
            'tc': master[f'{tf}_trend_change'].values,
            'adx': master[f'{tf}_adx'].values,
            'rsi': master[f'{tf}_rsi'].values,
            'atr': master[f'{tf}_atr14'].values,
            'atr50': master[f'{tf}_atr_sma50'].values,
            'mh': master[f'{tf}_macd_hist'].values,
            'bbp': master[f'{tf}_bb_pctb'].values,
            'vr': master[f'{tf}_vol_ratio'].values,
            'adxs': master[f'{tf}_adx_slope'].values,
        }

    # Bars since change
    bsc = {}
    for tf in ['5m','15m','30m']:
        bsc[tf] = master[f'{tf}_bars_since_change'].values

    bal = INIT
    pk_bal = INIT
    positions = []
    trades = []
    consec_loss = 0
    cooldown_end = None
    mo_start = {}
    last_entry_bar = {'A': -999, 'B': -999, 'C': -999}
    warmup = 3000

    print(f"\n[4] Running backtest ({n:,} bars)...")
    t0 = time.time()

    for i in range(warmup, n):
        ts = ts_arr[i]
        hi = h_arr[i]; lo = l_arr[i]; cl = c_arr[i]

        # ---- Check positions ----
        rm = []
        for pi, p in enumerate(positions):
            xr = None; xp = cl

            if p.d == 'LONG':
                p.hi = max(p.hi, hi)
                cr = (cl - p.ep) / p.ep
                p.pkr = max(p.pkr, (p.hi - p.ep) / p.ep)
                if lo <= p.sl: xp = p.sl; xr = 'SL'
                else:
                    if not p.t1 and hi >= p.tp1:
                        ps = p.sz * 0.30
                        p.pp += ps*(p.tp1-p.ep)/p.ep - ps*FEE
                        p.rem -= 0.30; p.t1 = True; p.sl = p.ep
                    if p.t1 and not p.t2 and hi >= p.tp2:
                        ps = p.sz * 0.30
                        p.pp += ps*(p.tp2-p.ep)/p.ep - ps*FEE
                        p.rem -= 0.30; p.t2 = True; p.sl = p.tp1
                    cfg = ENGINE[p.eng]
                    if cr >= cfg['trail_act_pct'] or p.t1:
                        p.ta = True
                        nt = p.hi - p.atr * cfg['trail_atr']
                        if p.tsl is None or nt > p.tsl: p.tsl = max(nt, p.sl)
                    if p.ta and p.tsl and lo <= p.tsl: xp = p.tsl; xr = 'TRAIL'
            else:
                p.lo = min(p.lo, lo)
                cr = (p.ep - cl) / p.ep
                p.pkr = max(p.pkr, (p.ep - p.lo) / p.ep)
                if hi >= p.sl: xp = p.sl; xr = 'SL'
                else:
                    if not p.t1 and lo <= p.tp1:
                        ps = p.sz * 0.30
                        p.pp += ps*(p.ep-p.tp1)/p.ep - ps*FEE
                        p.rem -= 0.30; p.t1 = True; p.sl = p.ep
                    if p.t1 and not p.t2 and lo <= p.tp2:
                        ps = p.sz * 0.30
                        p.pp += ps*(p.ep-p.tp2)/p.ep - ps*FEE
                        p.rem -= 0.30; p.t2 = True; p.sl = p.tp1
                    cfg = ENGINE[p.eng]
                    if cr >= cfg['trail_act_pct'] or p.t1:
                        p.ta = True
                        nt = p.lo + p.atr * cfg['trail_atr']
                        if p.tsl is None or nt < p.tsl: p.tsl = min(nt, p.sl)
                    if p.ta and p.tsl and hi >= p.tsl: xp = p.tsl; xr = 'TRAIL'

            if xr:
                rs = p.sz * p.rem
                rpnl = rs * ((xp-p.ep)/p.ep if p.d=='LONG' else (p.ep-xp)/p.ep)
                tpnl = rpnl + p.pp - rs*FEE
                bal += tpnl
                pk_bal = max(pk_bal, bal)
                consec_loss = consec_loss+1 if tpnl<0 else 0
                trades.append({
                    'eng':p.eng,'dir':p.d,'et':p.et,'xt':ts,'ep':p.ep,'xp':xp,
                    'xr':xr,'sz':p.sz,'lev':p.lev,'margin':p.sz/p.lev,'pnl':tpnl,
                    'roi':tpnl/(p.sz/p.lev)*100,'pkroi':p.pkr*100,
                    't1':p.t1,'t2':p.t2,'sc':p.sc,'bal':bal,'slp':p.slp*100,
                    'hold':(ts-p.et).total_seconds()/60
                })
                rm.append(pi)

        for pi in sorted(rm, reverse=True): positions.pop(pi)

        # ---- Risk checks ----
        if cooldown_end:
            if ts < cooldown_end: continue
            cooldown_end = None

        dd = (bal - pk_bal) / pk_bal if pk_bal > 0 else 0
        if dd < -0.35:
            cooldown_end = ts + pd.Timedelta(hours=48)
            continue
        dd_m = 0.5 if dd < -0.20 else 1.0
        st_m = max(0.3, 1.0 - consec_loss * 0.12)

        mk = f"{ts.year}-{ts.month:02d}"
        if mk not in mo_start: mo_start[mk] = bal
        mo_pnl = (bal - mo_start[mk]) / mo_start[mk] if mo_start[mk] > 0 else 0
        if mo_pnl < -0.15: continue

        if len(positions) >= 3: continue

        # ---- Try each engine ----
        for ek, cfg in ENGINE.items():
            tf = cfg['tf']
            s = sig[tf]

            # Check if within entry window of a recent trend change
            bars_since = bsc[tf][i] if tf in bsc else 999
            if bars_since > cfg['entry_window']:
                continue

            # Minimum gap between entries
            if i - last_entry_bar[ek] < cfg['min_gap']:
                continue

            # Already have position for this engine?
            if sum(1 for p in positions if p.eng == ek) >= cfg['max_pos']:
                continue

            # Determine direction from trend state
            trend = s['trend'][i]
            if np.isnan(trend): continue
            direction = 1 if trend > 0.5 else -1
            dir_str = 'LONG' if direction == 1 else 'SHORT'

            # Filter: ADX
            adx_v = s['adx'][i]
            if np.isnan(adx_v) or adx_v < cfg['adx_min']: continue

            # Filter: RSI
            rsi_v = s['rsi'][i]
            if np.isnan(rsi_v) or not (cfg['rsi_lo'] <= rsi_v <= cfg['rsi_hi']): continue

            # ATR
            atr_v = s['atr'][i]
            if np.isnan(atr_v) or atr_v <= 0: continue

            # MTF alignment
            mtf_n = 0
            for mtf_tf in cfg['mtf_tfs']:
                mt = sig[mtf_tf]['trend'][i]
                if not np.isnan(mt):
                    if (direction == 1 and mt > 0.5) or (direction == -1 and mt < 0.5):
                        mtf_n += 1
            if mtf_n < 1: continue

            # Score
            sc = score(s['adxs'][i], s['vr'][i], mtf_n, s['bbp'][i], s['mh'][i], rsi_v, direction)
            if sc < cfg['min_score']: continue

            # Regime multiplier
            a50 = s['atr50'][i]
            vrat = atr_v / a50 if (not np.isnan(a50) and a50 > 0) else 1.0
            reg_m = 0.7 if vrat > 1.3 else (1.3 if vrat < 0.7 else 1.0)

            # Close opposite positions for this engine
            opp = [pi for pi, p in enumerate(positions) if p.eng == ek and p.d != dir_str]
            for pi in sorted(opp, reverse=True):
                p = positions[pi]
                rs = p.sz * p.rem
                rpnl = rs * ((cl-p.ep)/p.ep if p.d=='LONG' else (p.ep-cl)/p.ep)
                tpnl = rpnl + p.pp - rs*FEE
                bal += tpnl
                pk_bal = max(pk_bal, bal)
                consec_loss = consec_loss+1 if tpnl<0 else 0
                trades.append({
                    'eng':p.eng,'dir':p.d,'et':p.et,'xt':ts,'ep':p.ep,'xp':cl,
                    'xr':'REVERSE','sz':p.sz,'lev':p.lev,'margin':p.sz/p.lev,'pnl':tpnl,
                    'roi':tpnl/(p.sz/p.lev)*100,'pkroi':p.pkr*100,
                    't1':p.t1,'t2':p.t2,'sc':p.sc,'bal':bal,'slp':p.slp*100,
                    'hold':(ts-p.et).total_seconds()/60
                })
                positions.pop(pi)

            # Position sizing
            size = bal * cfg['ratio'] * reg_m * dd_m * st_m
            if size < 5: continue
            lev = cfg['lev']
            notional = size * lev

            # SL/TP
            sl_pct = max(0.015, min(0.08, atr_v * cfg['atr_sl'] / cl))
            sd = cl * sl_pct
            t1d = atr_v * cfg['atr_tp1']
            t2d = atr_v * cfg['atr_tp2']

            if direction == 1:
                sl_p = cl - sd; tp1_p = cl + t1d; tp2_p = cl + t2d
            else:
                sl_p = cl + sd; tp1_p = cl - t1d; tp2_p = cl - t2d

            bal -= notional * FEE
            positions.append(Pos(dir_str, cl, ts, notional, lev, sl_p, tp1_p, tp2_p,
                                ek, sc, atr_v, sl_pct))
            last_entry_bar[ek] = i

        # Progress
        if (i - warmup) % 130000 == 0 and i > warmup:
            pct = (i-warmup)/(n-warmup)*100
            print(f"  {pct:.0f}% | {ts.strftime('%Y-%m-%d')} | Bal:${bal:,.0f} | Trades:{len(trades)} | Pos:{len(positions)}")

    elapsed = time.time() - t0
    print(f"  100% | Done in {elapsed:.1f}s | Total trades: {len(trades)}")

    return trades, bal, pk_bal, INIT


# ============================================================
# REPORTING
# ============================================================
def report(trades, fbal, pkbal, init):
    if not trades:
        print("\n  NO TRADES!"); return

    df = pd.DataFrame(trades)
    T = len(df)
    W = df[df['pnl']>0]; L = df[df['pnl']<=0]
    wr = len(W)/T*100
    gp = W['pnl'].sum() if len(W) else 0
    gl = abs(L['pnl'].sum()) if len(L) else 0
    pf = gp/gl if gl>0 else float('inf')
    aw = W['roi'].mean() if len(W) else 0
    al = L['roi'].mean() if len(L) else 0
    rr = abs(aw/al) if al!=0 else float('inf')
    tret = (fbal-init)/init*100

    bals = [init]+[t['bal'] for t in trades]
    pk=bals[0]; mdd=0
    for b in bals:
        pk=max(pk,b); dd=(b-pk)/pk; mdd=min(mdd,dd)

    mc=0;cc=0
    for t in trades:
        if t['pnl']<=0: cc+=1; mc=max(mc,cc)
        else: cc=0

    sl_n=len(df[df['xr']=='SL']); tr_n=len(df[df['xr']=='TRAIL']); rv_n=len(df[df['xr']=='REVERSE'])
    t1_n=df['t1'].sum(); t2_n=df['t2'].sum()
    mo_span = max(1, (df['xt'].max()-df['et'].min()).days/30)

    print("\n" + "="*120)
    print("  v16.2 TRIPLE ENGINE BACKTEST RESULTS")
    print("="*120)
    print(f"""
  Initial:       ${init:,.0f}
  Final:         ${fbal:,.2f}
  Total Return:  {tret:+,.1f}%
  Profit Factor: {pf:.2f}
  MDD:           {mdd*100:.1f}%
  Trades:        {T}  ({mo_span:.0f} months, {T/mo_span:.1f}/mo)
  Win Rate:      {wr:.1f}%
  Avg Win:       {aw:+.2f}%     Avg Loss: {al:+.2f}%
  R:R Ratio:     {rr:.2f}
  Max Consec L:  {mc}
  Exits: SL={sl_n}({sl_n/T*100:.0f}%) TRAIL={tr_n}({tr_n/T*100:.0f}%) REV={rv_n}({rv_n/T*100:.0f}%)
  TP1: {t1_n:.0f}({t1_n/T*100:.0f}%)  TP2: {t2_n:.0f}({t2_n/T*100:.0f}%)
""")

    # ENGINE BREAKDOWN
    print("="*120)
    print("  ENGINE BREAKDOWN (엔진별 성과)")
    print("="*120)
    hdr = f"  {'Engine':<14} {'Trades':>6} {'Wins':>5} {'WR%':>6} {'GrossP':>11} {'GrossL':>11} {'PF':>6} {'AvgW%':>7} {'AvgL%':>7} {'R:R':>5} {'NetPnL':>11}"
    print(hdr); print("  "+"-"*110)
    for ek in ['A','B','C']:
        et = df[df['eng']==ek]
        if len(et)==0: print(f"  {ek}({ENGINE[ek]['name']:<7}) {'0':>6}"); continue
        ew=et[et['pnl']>0]; el=et[et['pnl']<=0]
        ewr=len(ew)/len(et)*100; egp=ew['pnl'].sum() if len(ew) else 0
        egl=abs(el['pnl'].sum()) if len(el) else 0; epf=egp/egl if egl>0 else float('inf')
        eaw=ew['roi'].mean() if len(ew) else 0; eal=el['roi'].mean() if len(el) else 0
        err=abs(eaw/eal) if eal!=0 else float('inf')
        pfs=f"{epf:.2f}" if epf<999 else "INF"
        print(f"  {ek}({ENGINE[ek]['name']:<7}) {len(et):>5} {len(ew):>5} {ewr:>5.1f}% ${egp:>9,.2f} ${egl:>9,.2f} {pfs:>5} {eaw:>+6.2f}% {eal:>+6.2f}% {err:>4.2f} ${et['pnl'].sum():>+9,.2f}")

    # ENTRY STRUCTURE
    print("\n" + "="*120)
    print("  POSITION ENTRY STRUCTURE (진입 구조 상세)")
    print("="*120)

    lo=df[df['dir']=='LONG']; sh=df[df['dir']=='SHORT']
    print(f"\n  [방향별]")
    for label, sub in [('LONG', lo), ('SHORT', sh)]:
        if len(sub)==0: continue
        sw = sub[sub['pnl']>0]
        print(f"    {label:>5}: {len(sub):>5} ({len(sub)/T*100:.1f}%) WR:{len(sw)/len(sub)*100:.1f}% AvgROI:{sub['roi'].mean():+.2f}% PnL:${sub['pnl'].sum():+,.2f}")

    print(f"\n  [진입 스코어별 성과]")
    for lo_s, hi_s in [(0,20),(20,30),(30,40),(40,50),(50,60),(60,80)]:
        mask=(df['sc']>=lo_s)&(df['sc']<hi_s); bt=df[mask]
        if len(bt):
            bw=bt[bt['pnl']>0]
            print(f"    Score {lo_s:>2}~{hi_s:>2}: {len(bt):>5} trades WR:{len(bw)/len(bt)*100:.1f}% AvgROI:{bt['roi'].mean():+.2f}% PnL:${bt['pnl'].sum():+,.2f}")

    print(f"\n  [보유시간별 성과]")
    df['hold_h']=df['hold']/60
    for lo_h,hi_h,lb in [(0,0.5,'<30m'),(0.5,2,'30m~2h'),(2,6,'2~6h'),(6,12,'6~12h'),(12,24,'12~24h'),(24,72,'1~3d'),(72,168,'3~7d'),(168,99999,'7d+')]:
        mask=(df['hold_h']>=lo_h)&(df['hold_h']<hi_h); ht=df[mask]
        if len(ht):
            hw=ht[ht['pnl']>0]
            print(f"    {lb:>8}: {len(ht):>5} trades WR:{len(hw)/len(ht)*100:.1f}% AvgROI:{ht['roi'].mean():+.2f}% PnL:${ht['pnl'].sum():+,.2f}")

    print(f"\n  [SL 거리 분포]")
    for lo_s,hi_s in [(1,2),(2,3),(3,4),(4,6),(6,8)]:
        mask=(df['slp']>=lo_s)&(df['slp']<hi_s); st=df[mask]
        if len(st):
            sw2=st[st['pnl']>0]
            print(f"    SL {lo_s}~{hi_s}%: {len(st):>5} trades WR:{len(sw2)/len(st)*100:.1f}% AvgROI:{st['roi'].mean():+.2f}%")

    print(f"\n  [엔진별 청산사유]")
    for ek in ['A','B','C']:
        et=df[df['eng']==ek]
        if len(et)==0: continue
        print(f"    Engine {ek} ({ENGINE[ek]['name']}):")
        for r in ['SL','TRAIL','REVERSE']:
            rt=et[et['xr']==r]
            if len(rt): print(f"      {r:>8}: {len(rt):>4} ({len(rt)/len(et)*100:.0f}%) AvgROI:{rt['roi'].mean():+.2f}%")

    # MONTHLY PERFORMANCE
    print("\n" + "="*120)
    print("  MONTHLY PERFORMANCE (월별 상세)")
    print("="*120)
    df['month']=pd.to_datetime(df['xt']).dt.to_period('M')
    mg=df.groupby('month')
    print(f"\n  {'Month':>8} {'Trd':>4} {'W':>3} {'L':>3} {'WR%':>6} {'GrossP':>10} {'GrossL':>10} {'NetPnL':>10} {'PF':>5} {'Balance':>12} {'MoRet%':>8} {'EngA':>4} {'EngB':>4} {'EngC':>4}")
    print("  "+"-"*115)

    rb=init; yearly={}; loss_mo=0; total_mo=0
    for mo in sorted(mg.groups.keys()):
        g=mg.get_group(mo); nt=len(g); nw=len(g[g['pnl']>0]); nl=nt-nw
        wr2=nw/nt*100 if nt else 0
        gp2=g[g['pnl']>0]['pnl'].sum() if nw else 0
        gl2=abs(g[g['pnl']<=0]['pnl'].sum()) if nl else 0
        net=g['pnl'].sum(); mpf=gp2/gl2 if gl2>0 else float('inf')
        sbr=rb; rb+=net; mret=net/sbr*100 if sbr>0 else 0
        total_mo+=1
        if net<0: loss_mo+=1

        # Engine counts
        ea=len(g[g['eng']=='A']); eb=len(g[g['eng']=='B']); ec=len(g[g['eng']=='C'])

        yr=str(mo)[:4]
        if yr not in yearly: yearly[yr]={'pnl':0,'trades':0,'wins':0,'losses':0,'gp':0,'gl':0,'sb':sbr}
        yearly[yr]['pnl']+=net; yearly[yr]['trades']+=nt
        yearly[yr]['wins']+=nw; yearly[yr]['losses']+=nl
        yearly[yr]['gp']+=gp2; yearly[yr]['gl']+=gl2; yearly[yr]['eb']=rb

        pfs=f"{mpf:.1f}" if mpf<999 else "INF"
        mk=" <<LOSS" if net<0 else ""
        print(f"  {str(mo):>8} {nt:>4} {nw:>3} {nl:>3} {wr2:>5.1f}% ${gp2:>8,.1f} ${gl2:>8,.1f} ${net:>+8,.1f} {pfs:>4} ${rb:>10,.1f} {mret:>+7.1f}% {ea:>4} {eb:>4} {ec:>4}{mk}")

    # YEARLY
    print("\n" + "="*120)
    print("  YEARLY PERFORMANCE (연도별)")
    print("="*120)
    print(f"\n  {'Year':>6} {'Trd':>5} {'W':>4} {'L':>4} {'WR%':>6} {'GrossP':>12} {'GrossL':>12} {'NetPnL':>12} {'PF':>6} {'YrRet%':>9}")
    print("  "+"-"*95)
    for yr in sorted(yearly.keys()):
        y=yearly[yr]
        ywr=y['wins']/y['trades']*100 if y['trades'] else 0
        ypf=y['gp']/y['gl'] if y['gl']>0 else float('inf')
        yret=y['pnl']/y['sb']*100 if y['sb']>0 else 0
        pfs=f"{ypf:.2f}" if ypf<999 else "INF"
        print(f"  {yr:>6} {y['trades']:>4} {y['wins']:>4} {y['losses']:>4} {ywr:>5.1f}% ${y['gp']:>10,.2f} ${y['gl']:>10,.2f} ${y['pnl']:>+10,.2f} {pfs:>5} {yret:>+8.1f}%")

    pyrs=sum(1 for y in yearly.values() if y['pnl']>0)
    print(f"\n  Loss Months: {loss_mo}/{total_mo} ({loss_mo/max(1,total_mo)*100:.0f}%)")
    print(f"  Profitable Years: {pyrs}/{len(yearly)}")

    # ENGINE x YEAR
    print("\n" + "="*120)
    print("  ENGINE x YEAR (엔진별 연도 성과)")
    print("="*120)
    df['year']=pd.to_datetime(df['xt']).dt.year
    for ek in ['A','B','C']:
        et=df[df['eng']==ek]
        if len(et)==0: print(f"\n  Engine {ek} ({ENGINE[ek]['name']}): No trades"); continue
        print(f"\n  Engine {ek} ({ENGINE[ek]['name']}):")
        print(f"    {'Year':>6} {'Trd':>4} {'WR%':>6} {'PnL':>11} {'AvgROI%':>8} {'PF':>6}")
        for yr in sorted(et['year'].unique()):
            yt=et[et['year']==yr]; yw=yt[yt['pnl']>0]; yl=yt[yt['pnl']<=0]
            ywr=len(yw)/len(yt)*100; ygp=yw['pnl'].sum() if len(yw) else 0
            ygl=abs(yl['pnl'].sum()) if len(yl) else 0; ypf=ygp/ygl if ygl>0 else float('inf')
            pfs=f"{ypf:.2f}" if ypf<999 else "INF"
            print(f"    {yr:>6} {len(yt):>3} {ywr:>5.1f}% ${yt['pnl'].sum():>+9,.2f} {yt['roi'].mean():>+7.2f}% {pfs:>5}")

    # TOP/BOTTOM
    print("\n" + "="*120)
    print("  TOP 10 / BOTTOM 10 TRADES")
    print("="*120)
    ds=df.sort_values('pnl',ascending=False)
    print(f"\n  Top 10:")
    print(f"    {'#':>3} {'Eng':>3} {'Dir':>5} {'Entry':>20} {'Exit':>20} {'Rsn':>7} {'ROI%':>7} {'PnL':>10} {'Sc':>3}")
    for i2,(_, r) in enumerate(ds.head(10).iterrows()):
        print(f"    {i2+1:>3} {r['eng']:>3} {r['dir']:>5} {str(r['et'])[:19]:>20} {str(r['xt'])[:19]:>20} {r['xr']:>7} {r['roi']:>+6.1f}% ${r['pnl']:>8,.1f} {r['sc']:>2.0f}")
    print(f"\n  Bottom 10:")
    for i2,(_, r) in enumerate(ds.tail(10).iterrows()):
        print(f"    {i2+1:>3} {r['eng']:>3} {r['dir']:>5} {str(r['et'])[:19]:>20} {str(r['xt'])[:19]:>20} {r['xr']:>7} {r['roi']:>+6.1f}% ${r['pnl']:>8,.1f} {r['sc']:>2.0f}")

    print("\n" + "="*120)
    print("  COMPLETE")
    print("="*120)


if __name__ == "__main__":
    print("="*120)
    print("  v16.2 Triple Engine Backtest - BTC/USDT Futures [v3 - State-Based]")
    print("="*120)
    trades, fbal, pkbal, init = run()
    report(trades, fbal, pkbal, init)
