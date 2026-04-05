"""
TOP 10 Strategy Verification Backtest
- 10 BT (rolling window splits) + 10 Verify (shifted window splits) = 20 runs
- 5m data resampled to 10m/15m/30m/1h
- Wilder ADX, EMA/WMA/HMA/SMA
- Pure strategy test: no DD protection, no monthly limits
- Each run uses a different time window for true out-of-sample verification
"""
import pandas as pd, numpy as np, os, time, warnings, sys, io, csv
from datetime import datetime
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DIR = r"D:\filesystem\futures\btc_V1\test3"
FEE = 0.0004          # 0.04% per side
INIT = 3000.0
LEV = 10
N_BT = 10
N_VER = 10
N_TOTAL = N_BT + N_VER

# ============================================================
# INDICATORS
# ============================================================
def ema_calc(s, p):
    return s.ewm(span=p, adjust=False).mean()

def sma_calc(s, p):
    return s.rolling(p).mean()

def wma_calc(s, p):
    w = np.arange(1, p+1, dtype=float)
    return s.rolling(p).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def hma_calc(s, p):
    h = max(int(p / 2), 1)
    sq = max(int(np.sqrt(p)), 1)
    return wma_calc(2 * wma_calc(s, h) - wma_calc(s, p), sq)

def rsi_calc(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = (-d).where(d < 0, 0.0)
    ag = g.ewm(alpha=1/p, min_periods=p).mean()
    al = l.ewm(alpha=1/p, min_periods=p).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def wilder(arr, p):
    """Wilder smoothing"""
    out = np.full(len(arr), np.nan)
    s = 0
    while s < len(arr) and np.isnan(arr[s]):
        s += 1
    if s + p > len(arr):
        return out
    out[s + p - 1] = np.nanmean(arr[s:s + p])
    for i in range(s + p, len(arr)):
        if not np.isnan(arr[i]) and not np.isnan(out[i - 1]):
            out[i] = (out[i - 1] * (p - 1) + arr[i]) / p
    return out

def adx_wilder(h_s, l_s, c_s, p=14):
    """True Wilder ADX"""
    h = h_s.values; l = l_s.values; c = c_s.values
    n = len(h)
    tr = np.full(n, np.nan)
    pdm = np.full(n, np.nan)
    mdm = np.full(n, np.nan)
    for i in range(1, n):
        hl = h[i] - l[i]
        hc = abs(h[i] - c[i-1])
        lc = abs(l[i] - c[i-1])
        tr[i] = max(hl, hc, lc)
        um = h[i] - h[i-1]
        dm = l[i-1] - l[i]
        pdm[i] = um if (um > dm and um > 0) else 0.0
        mdm[i] = dm if (dm > um and dm > 0) else 0.0
    atr = wilder(tr, p)
    s_pdm = wilder(pdm, p)
    s_mdm = wilder(mdm, p)
    pdi = np.where(atr > 0, 100 * s_pdm / atr, 0)
    mdi = np.where(atr > 0, 100 * s_mdm / atr, 0)
    dsum = pdi + mdi
    dx = np.where(dsum > 0, 100 * np.abs(pdi - mdi) / dsum, 0)
    dx = np.where(np.isnan(pdi) | np.isnan(mdi), np.nan, dx)
    adx = wilder(dx, p)
    return pd.Series(adx, index=h_s.index), pd.Series(atr, index=h_s.index)

# ============================================================
# DATA LOADING
# ============================================================
def load_5m():
    print("Loading 5m CSV data...")
    t0 = time.time()
    fs = [os.path.join(DIR, f"btc_usdt_5m_2020_to_now_part{i}.csv") for i in range(1, 4)]
    dfs = []
    for f in fs:
        d = pd.read_csv(f, parse_dates=['timestamp'])
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df.set_index('timestamp', inplace=True)
    print(f"  Loaded {len(df)} rows in {time.time()-t0:.1f}s  ({df.index[0]} ~ {df.index[-1]})")
    return df

def resample_tf(df5, tf):
    d = df5.resample(tf).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    return d

# ============================================================
# STRATEGY DEFINITIONS
# ============================================================
STRATEGIES = {
    'v15.4': dict(
        fast_type='ema', fast_p=3, slow_type='ema', slow_p=200,
        tf='30min', adx_p=14, adx_th=35, rsi_p=14, rsi_lo=30, rsi_hi=65,
        delay=0, sl_pct=-7, trail_on=6, trail_off=-3, margin_pct=40, lev=10
    ),
    'v22.3': dict(
        fast_type='ema', fast_p=3, slow_type='wma', slow_p=250,
        tf='30min', adx_p=20, adx_th=25, rsi_p=14, rsi_lo=35, rsi_hi=65,
        delay=0, sl_pct=-8, trail_on=5, trail_off=-4, margin_pct=60, lev=10
    ),
    'v22.2': dict(
        fast_type='ema', fast_p=3, slow_type='ema', slow_p=200,
        tf='30min', adx_p=14, adx_th=35, rsi_p=14, rsi_lo=30, rsi_hi=65,
        delay=0, sl_pct=-7, trail_on=6, trail_off=-3, margin_pct=60, lev=10
    ),
    'v25.2A': dict(
        fast_type='ema', fast_p=3, slow_type='ema', slow_p=200,
        tf='30min', adx_p=20, adx_th=35, rsi_p=14, rsi_lo=40, rsi_hi=75,
        delay=0, sl_pct=-7, trail_on=7, trail_off=-3, margin_pct=40, lev=10
    ),
    'v25.1A': dict(
        fast_type='hma', fast_p=21, slow_type='ema', slow_p=250,
        tf='10min', adx_p=20, adx_th=35, rsi_p=14, rsi_lo=40, rsi_hi=75,
        delay=0, sl_pct=-6, trail_on=7, trail_off=-3, margin_pct=50, lev=10
    ),
    'v16.6': dict(
        fast_type='wma', fast_p=3, slow_type='ema', slow_p=200,
        tf='30min', adx_p=20, adx_th=35, rsi_p=14, rsi_lo=30, rsi_hi=70,
        delay=5, sl_pct=-8, trail_on=3, trail_off=-2, margin_pct=50, lev=10
    ),
    'v25.2B': dict(
        fast_type='ema', fast_p=7, slow_type='ema', slow_p=100,
        tf='30min', adx_p=20, adx_th=35, rsi_p=14, rsi_lo=40, rsi_hi=75,
        delay=0, sl_pct=-7, trail_on=6, trail_off=-3, margin_pct=40, lev=10
    ),
    'v22.1': dict(
        fast_type='wma', fast_p=3, slow_type='ema', slow_p=200,
        tf='30min', adx_p=20, adx_th=35, rsi_p=14, rsi_lo=35, rsi_hi=60,
        delay=0, sl_pct=-8, trail_on=4, trail_off=-1, margin_pct=50, lev=10
    ),
    'v16.4': dict(
        fast_type='wma', fast_p=3, slow_type='ema', slow_p=200,
        tf='30min', adx_p=20, adx_th=35, rsi_p=14, rsi_lo=35, rsi_hi=65,
        delay=0, sl_pct=-8, trail_on=4, trail_off=-3, margin_pct=30, lev=10
    ),
    'v25.2C': dict(
        fast_type='ema', fast_p=5, slow_type='ema', slow_p=300,
        tf='10min', adx_p=20, adx_th=35, rsi_p=14, rsi_lo=40, rsi_hi=75,
        delay=0, sl_pct=-7, trail_on=8, trail_off=-3, margin_pct=40, lev=10
    ),
}

# ============================================================
# COMPUTE MA HELPER
# ============================================================
def compute_ma(series, ma_type, period):
    if ma_type == 'ema': return ema_calc(series, period)
    elif ma_type == 'wma': return wma_calc(series, period)
    elif ma_type == 'hma': return hma_calc(series, period)
    elif ma_type == 'sma': return sma_calc(series, period)
    else: return ema_calc(series, period)

# ============================================================
# PREPARE DATA
# ============================================================
def prepare_tf_data(df5):
    tf_data = {}
    needed = set()
    for s in STRATEGIES.values():
        needed.add(s['tf'])
    for tf in needed:
        print(f"  Resampling to {tf}...")
        tf_data[tf] = resample_tf(df5, tf)
        print(f"    -> {len(tf_data[tf])} bars")
    return tf_data

def prepare_indicators(df_tf, cfg):
    c = df_tf['close']; h = df_tf['high']; l = df_tf['low']; o = df_tf['open']
    fast = compute_ma(c, cfg['fast_type'], cfg['fast_p'])
    slow = compute_ma(c, cfg['slow_type'], cfg['slow_p'])
    adx, atr = adx_wilder(h, l, c, cfg['adx_p'])
    rsi = rsi_calc(c, cfg['rsi_p'])
    return {
        'open': o.values, 'high': h.values, 'low': l.values, 'close': c.values,
        'fast': fast.values, 'slow': slow.values,
        'adx': adx.values, 'rsi': rsi.values, 'atr': atr.values,
        'idx': df_tf.index
    }

# ============================================================
# GENERATE 20 TIME WINDOWS
# ============================================================
def generate_windows(n_bars, warmup=500):
    """
    Generate 20 different (start, end) windows for backtesting.
    BT01-10: 10 expanding windows from 60% to 100% of data
    VR01-10: 10 rolling windows of ~30% length stepping through data
    """
    windows = []
    usable = n_bars - warmup

    # BT runs: expanding windows ending at dataset end
    # BT01 starts at 40% of usable, BT10 starts at warmup (full data)
    for i in range(N_BT):
        frac = 0.4 + 0.06 * i   # 0.40, 0.46, 0.52, ..., 0.94
        start = warmup + int(usable * (1.0 - frac))
        end = n_bars
        windows.append((start, end, f"BT{i+1:02d}"))

    # VR runs: rolling windows of ~45% length stepping through
    win_len = int(usable * 0.45)
    step = int((usable - win_len) / max(N_VER - 1, 1))
    for i in range(N_VER):
        start = warmup + i * step
        end = min(start + win_len, n_bars)
        windows.append((start, end, f"VR{i+1:02d}"))

    return windows

# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(ind, cfg, start_bar, end_bar):
    """Single backtest run on bars [start_bar, end_bar)"""
    fast = ind['fast']; slow = ind['slow']
    adx = ind['adx']; rsi = ind['rsi']
    hi = ind['high']; lo = ind['low']; cl = ind['close']

    sl_pct = cfg['sl_pct'] / 100.0
    trail_on = cfg['trail_on'] / 100.0
    trail_off = cfg['trail_off'] / 100.0
    margin_frac = cfg['margin_pct'] / 100.0
    lev = cfg['lev']
    adx_th = cfg['adx_th']
    rsi_lo = cfg['rsi_lo']; rsi_hi = cfg['rsi_hi']
    delay = cfg['delay']

    balance = INIT
    pos = 0
    entry_price = 0.0
    entry_margin = 0.0
    trail_active = False
    trail_peak = 0.0

    trades = 0; wins = 0; sl_hits = 0
    pnl_list = []
    peak_bal = INIT; max_dd = 0.0
    signal_count = 0

    for i in range(max(start_bar, 1), end_bar):
        if np.isnan(fast[i]) or np.isnan(slow[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            continue
        if np.isnan(fast[i-1]) or np.isnan(slow[i-1]):
            continue

        price = cl[i]; bar_hi = hi[i]; bar_lo = lo[i]

        # -- SL / Trail check --
        if pos != 0:
            roi_hi = (bar_hi / entry_price - 1) * pos * lev
            roi_lo = (bar_lo / entry_price - 1) * pos * lev
            worst_roi = roi_lo if pos == 1 else roi_hi

            # SL hit
            if worst_roi <= sl_pct:
                exit_pnl = sl_pct * entry_margin
                balance += entry_margin + exit_pnl - entry_margin * FEE
                pnl_list.append(sl_pct)
                if exit_pnl > 0: wins += 1
                sl_hits += 1; trades += 1; pos = 0; trail_active = False; trail_peak = 0
                peak_bal = max(peak_bal, balance)
                dd = (peak_bal - balance) / peak_bal if peak_bal > 0 else 0
                max_dd = max(max_dd, dd)
                continue

            # Trail activation
            best_roi = roi_hi if pos == 1 else roi_lo
            if best_roi >= trail_on:
                trail_active = True

            # Trail stop on close
            if trail_active:
                if pos == 1:
                    trail_peak = max(trail_peak, bar_hi) if trail_peak > 0 else bar_hi
                    drop_from_peak = (cl[i] / trail_peak - 1) * lev
                    if drop_from_peak <= trail_off:
                        act_roi = (cl[i] / entry_price - 1) * lev
                        exit_pnl = act_roi * entry_margin
                        balance += entry_margin + exit_pnl - entry_margin * FEE
                        pnl_list.append(act_roi)
                        if exit_pnl > 0: wins += 1
                        trades += 1; pos = 0; trail_active = False; trail_peak = 0
                        peak_bal = max(peak_bal, balance)
                        dd = (peak_bal - balance) / peak_bal if peak_bal > 0 else 0
                        max_dd = max(max_dd, dd)
                        continue
                else:
                    trail_peak = min(trail_peak, bar_lo) if trail_peak > 0 else bar_lo
                    rise_from_peak = (cl[i] / trail_peak - 1) * lev
                    if rise_from_peak >= abs(trail_off):
                        act_roi = (entry_price / cl[i] - 1) * lev
                        exit_pnl = act_roi * entry_margin
                        balance += entry_margin + exit_pnl - entry_margin * FEE
                        pnl_list.append(act_roi)
                        if exit_pnl > 0: wins += 1
                        trades += 1; pos = 0; trail_active = False; trail_peak = 0
                        peak_bal = max(peak_bal, balance)
                        dd = (peak_bal - balance) / peak_bal if peak_bal > 0 else 0
                        max_dd = max(max_dd, dd)
                        continue

        # -- Signal detection --
        cross_long = fast[i] > slow[i] and fast[i-1] <= slow[i-1]
        cross_short = fast[i] < slow[i] and fast[i-1] >= slow[i-1]
        adx_ok = adx[i] >= adx_th
        rsi_ok = rsi_lo <= rsi[i] <= rsi_hi
        sig = 0
        if cross_long and adx_ok and rsi_ok: sig = 1
        elif cross_short and adx_ok and rsi_ok: sig = -1

        # Reverse signal -> close + flip
        if pos != 0 and sig != 0 and sig != pos:
            exit_roi = (price / entry_price - 1) * pos * lev
            exit_pnl = exit_roi * entry_margin
            balance += entry_margin + exit_pnl - entry_margin * FEE
            pnl_list.append(exit_roi)
            if exit_pnl > 0: wins += 1
            trades += 1; pos = 0; trail_active = False; trail_peak = 0
            peak_bal = max(peak_bal, balance)
            dd = (peak_bal - balance) / peak_bal if peak_bal > 0 else 0
            max_dd = max(max_dd, dd)

        # Delay
        if sig != 0 and delay > 0:
            signal_count += 1
            if signal_count < delay: sig = 0
            else: signal_count = 0
        elif sig == 0:
            signal_count = 0

        # Same direction skip
        if sig != 0 and pos == sig: sig = 0

        # Open new
        if sig != 0 and pos == 0 and balance > 10:
            margin = balance * margin_frac
            if margin < 5: continue
            entry_price = price
            entry_margin = margin
            balance -= margin + margin * FEE
            pos = sig; trail_active = False; trail_peak = 0

    # Close open position at end
    if pos != 0:
        price = cl[min(end_bar - 1, len(cl) - 1)]
        exit_roi = (price / entry_price - 1) * pos * lev
        exit_pnl = exit_roi * entry_margin
        balance += entry_margin + exit_pnl - entry_margin * FEE
        pnl_list.append(exit_roi)
        if exit_pnl > 0: wins += 1
        trades += 1
        peak_bal = max(peak_bal, balance)
        dd = (peak_bal - balance) / peak_bal if peak_bal > 0 else 0
        max_dd = max(max_dd, dd)

    ret_pct = (balance - INIT) / INIT * 100
    gross_win = sum(p for p in pnl_list if p > 0) if pnl_list else 0
    gross_loss = abs(sum(p for p in pnl_list if p <= 0)) if pnl_list else 1e-9
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    wr = wins / trades * 100 if trades > 0 else 0

    return {
        'final_balance': round(balance, 2),
        'return_pct': round(ret_pct, 2),
        'pf': round(pf, 3),
        'mdd': round(max_dd * 100, 2),
        'trades': trades,
        'wins': wins,
        'wr': round(wr, 1),
        'sl_hits': sl_hits,
    }


def run_backtest_monthly(ind, cfg):
    """Full-data run with monthly P&L tracking"""
    n = len(ind['close'])
    fast = ind['fast']; slow = ind['slow']
    adx = ind['adx']; rsi = ind['rsi']
    hi = ind['high']; lo = ind['low']; cl = ind['close']
    timestamps = ind['idx']

    sl_pct = cfg['sl_pct'] / 100.0
    trail_on = cfg['trail_on'] / 100.0
    trail_off = cfg['trail_off'] / 100.0
    margin_frac = cfg['margin_pct'] / 100.0
    lev = cfg['lev']
    adx_th = cfg['adx_th']
    rsi_lo = cfg['rsi_lo']; rsi_hi = cfg['rsi_hi']
    delay = cfg['delay']

    balance = INIT; pos = 0; entry_price = 0.0; entry_margin = 0.0
    trail_active = False; trail_peak = 0.0; signal_count = 0
    monthly = {}
    current_month = None; month_start_bal = INIT
    warmup = 500

    for i in range(max(warmup, 1), n):
        if np.isnan(fast[i]) or np.isnan(slow[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            continue
        if np.isnan(fast[i-1]) or np.isnan(slow[i-1]):
            continue

        ts = timestamps[i]
        ym = f"{ts.year}-{ts.month:02d}"
        if current_month is None:
            current_month = ym; month_start_bal = balance
        elif ym != current_month:
            monthly[current_month] = {
                'start_bal': round(month_start_bal, 2),
                'end_bal': round(balance, 2),
                'ret_pct': round((balance - month_start_bal) / month_start_bal * 100, 2) if month_start_bal > 0 else 0
            }
            current_month = ym; month_start_bal = balance

        price = cl[i]; bar_hi = hi[i]; bar_lo = lo[i]

        if pos != 0:
            roi_hi = (bar_hi / entry_price - 1) * pos * lev
            roi_lo = (bar_lo / entry_price - 1) * pos * lev
            worst_roi = roi_lo if pos == 1 else roi_hi
            if worst_roi <= sl_pct:
                exit_pnl = sl_pct * entry_margin
                balance += entry_margin + exit_pnl - entry_margin * FEE
                pos = 0; trail_active = False; trail_peak = 0; continue
            best_roi = roi_hi if pos == 1 else roi_lo
            if best_roi >= trail_on: trail_active = True
            if trail_active:
                if pos == 1:
                    trail_peak = max(trail_peak, bar_hi) if trail_peak > 0 else bar_hi
                    drop = (cl[i] / trail_peak - 1) * lev
                    if drop <= trail_off:
                        act_roi = (cl[i] / entry_price - 1) * lev
                        balance += entry_margin + act_roi * entry_margin - entry_margin * FEE
                        pos = 0; trail_active = False; trail_peak = 0; continue
                else:
                    trail_peak = min(trail_peak, bar_lo) if trail_peak > 0 else bar_lo
                    rise = (cl[i] / trail_peak - 1) * lev
                    if rise >= abs(trail_off):
                        act_roi = (entry_price / cl[i] - 1) * lev
                        balance += entry_margin + act_roi * entry_margin - entry_margin * FEE
                        pos = 0; trail_active = False; trail_peak = 0; continue

        cross_long = fast[i] > slow[i] and fast[i-1] <= slow[i-1]
        cross_short = fast[i] < slow[i] and fast[i-1] >= slow[i-1]
        adx_ok = adx[i] >= adx_th
        rsi_ok = rsi_lo <= rsi[i] <= rsi_hi
        sig = 0
        if cross_long and adx_ok and rsi_ok: sig = 1
        elif cross_short and adx_ok and rsi_ok: sig = -1
        if pos != 0 and sig != 0 and sig != pos:
            exit_roi = (price / entry_price - 1) * pos * lev
            balance += entry_margin + exit_roi * entry_margin - entry_margin * FEE
            pos = 0; trail_active = False; trail_peak = 0
        if sig != 0 and delay > 0:
            signal_count += 1
            if signal_count < delay: sig = 0
            else: signal_count = 0
        elif sig == 0: signal_count = 0
        if sig != 0 and pos == sig: sig = 0
        if sig != 0 and pos == 0 and balance > 10:
            margin = balance * margin_frac
            if margin < 5: continue
            entry_price = price; entry_margin = margin
            balance -= margin + margin * FEE
            pos = sig; trail_active = False; trail_peak = 0

    if pos != 0:
        price = cl[n-1]
        exit_roi = (price / entry_price - 1) * pos * lev
        balance += entry_margin + exit_roi * entry_margin - entry_margin * FEE
    if current_month and current_month not in monthly:
        monthly[current_month] = {
            'start_bal': round(month_start_bal, 2), 'end_bal': round(balance, 2),
            'ret_pct': round((balance - month_start_bal) / month_start_bal * 100, 2) if month_start_bal > 0 else 0
        }
    return monthly, round(balance, 2)


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    print("=" * 90)
    print("  TOP 10 STRATEGY VERIFICATION BACKTEST")
    print("  10 BT (expanding windows) + 10 VR (rolling windows) = 20 runs per strategy")
    print("=" * 90)

    df5 = load_5m()
    tf_data = prepare_tf_data(df5)

    print("\nComputing indicators...")
    strat_ind = {}
    for name, cfg in STRATEGIES.items():
        print(f"  {name}: {cfg['fast_type'].upper()}({cfg['fast_p']})/{cfg['slow_type'].upper()}({cfg['slow_p']}) on {cfg['tf']}")
        df_tf = tf_data[cfg['tf']]
        strat_ind[name] = prepare_indicators(df_tf, cfg)

    print("\n" + "=" * 90)
    print("  RUNNING 20 RUNS PER STRATEGY")
    print("=" * 90)

    all_results = {}
    all_run_details = {}

    for name, cfg in STRATEGIES.items():
        ind = strat_ind[name]
        n_bars = len(ind['close'])
        windows = generate_windows(n_bars, warmup=500)
        print(f"\n--- {name} ({cfg['fast_type'].upper()}({cfg['fast_p']})/{cfg['slow_type'].upper()}({cfg['slow_p']}) {cfg['tf']}, {n_bars} bars) ---")

        runs = []
        details = []
        for w_start, w_end, label in windows:
            res = run_backtest(ind, cfg, w_start, w_end)
            runs.append(res)
            details.append({'label': label, 'start': w_start, 'end': w_end, **res})

            if label in ('BT01', 'BT05', 'BT10', 'VR01', 'VR05', 'VR10'):
                bars_used = w_end - w_start
                print(f"  [{label}] bars {w_start}-{w_end} ({bars_used:,}) | "
                      f"${res['final_balance']:>9,.2f} | {res['return_pct']:>7.1f}% | "
                      f"PF {res['pf']:.2f} | MDD {res['mdd']:.1f}% | "
                      f"{res['trades']} trades | WR {res['wr']:.0f}% | SL {res['sl_hits']}")

        all_results[name] = runs
        all_run_details[name] = details

    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    print("\n\n" + "=" * 110)
    print("  FINAL COMPARISON TABLE (20-run aggregation)")
    print("=" * 110)

    header = f"{'Strategy':<10} | {'Avg Final$':>11} | {'Avg Ret%':>9} | {'Avg PF':>7} | {'Avg MDD%':>8} | {'Avg Trd':>7} | {'Avg WR%':>7} | {'Avg SL':>6} | {'Ret Std%':>8} | {'Det?':>4}"
    print(header)
    print("-" * len(header))

    summary_rows = []
    best_name = None; best_score = -999999

    for name in STRATEGIES:
        runs = all_results[name]
        rets = [r['return_pct'] for r in runs]
        pfs = [r['pf'] for r in runs]
        mdds = [r['mdd'] for r in runs]
        trs = [r['trades'] for r in runs]
        wrs = [r['wr'] for r in runs]
        sls = [r['sl_hits'] for r in runs]
        finals = [r['final_balance'] for r in runs]

        avg_final = np.mean(finals); avg_ret = np.mean(rets)
        avg_pf = np.mean(pfs); avg_mdd = np.mean(mdds)
        avg_tr = np.mean(trs); avg_wr = np.mean(wrs)
        avg_sl = np.mean(sls); std_ret = np.std(rets)
        med_ret = np.median(rets)
        min_ret = np.min(rets); max_ret = np.max(rets)

        # Deterministic: median return > 0 and avg PF > 1
        profitable_runs = sum(1 for r in rets if r > 0)
        det = "Y" if profitable_runs >= 14 and avg_pf > 1.0 else "N"

        row = (f"{name:<10} | ${avg_final:>10,.0f} | {avg_ret:>8.1f}% | {avg_pf:>6.2f} | "
               f"{avg_mdd:>7.1f}% | {avg_tr:>6.0f} | {avg_wr:>6.0f}% | {avg_sl:>5.0f} | "
               f"{std_ret:>7.2f}% | {det:>4}")
        print(row)

        summary_rows.append({
            'strategy': name,
            'avg_final': round(avg_final, 2), 'avg_return_pct': round(avg_ret, 2),
            'med_return_pct': round(med_ret, 2),
            'min_return_pct': round(min_ret, 2), 'max_return_pct': round(max_ret, 2),
            'avg_pf': round(avg_pf, 3), 'avg_mdd': round(avg_mdd, 2),
            'avg_trades': round(avg_tr, 1), 'avg_wr': round(avg_wr, 1),
            'avg_sl': round(avg_sl, 1), 'std_ret': round(std_ret, 2),
            'profitable_runs': profitable_runs,
            'deterministic': det,
        })

        # Score: return / (1 + mdd) * consistency
        score = avg_ret / (1 + avg_mdd) * (profitable_runs / N_TOTAL)
        if score > best_score:
            best_score = score; best_name = name

    # ============================================================
    # RANKING
    # ============================================================
    print("\n\n" + "=" * 110)
    print("  RANKING BY COMPOSITE SCORE (Return / MDD * Consistency)")
    print("=" * 110)

    for r in summary_rows:
        r['score'] = r['avg_return_pct'] / (1 + r['avg_mdd']) * (r['profitable_runs'] / N_TOTAL)

    ranked = sorted(summary_rows, key=lambda x: x['score'], reverse=True)
    print(f"{'Rank':<5} {'Strategy':<10} | {'Avg Ret%':>9} | {'Med Ret%':>9} | {'PF':>6} | {'MDD%':>6} | {'Trades':>6} | {'WR%':>5} | {'Std%':>8} | {'Win/20':>6} | {'Det?':>4} | {'Score':>7}")
    print("-" * 105)
    for i, r in enumerate(ranked):
        print(f"  #{i+1:<3} {r['strategy']:<10} | {r['avg_return_pct']:>8.1f}% | {r['med_return_pct']:>8.1f}% | "
              f"{r['avg_pf']:>5.2f} | {r['avg_mdd']:>5.1f}% | {r['avg_trades']:>5.0f} | "
              f"{r['avg_wr']:>4.0f}% | {r['std_ret']:>7.2f}% | {r['profitable_runs']:>3d}/20 | "
              f"{r['deterministic']:>4} | {r['score']:>6.2f}")

    # ============================================================
    # MONTHLY BREAKDOWN FOR BEST
    # ============================================================
    print(f"\n\n{'='*90}")
    print(f"  MONTHLY BREAKDOWN: {best_name} (Best Composite Score)")
    print(f"{'='*90}")

    cfg = STRATEGIES[best_name]
    ind = strat_ind[best_name]
    monthly, final_bal = run_backtest_monthly(ind, cfg)

    print(f"\n{'Month':<10} | {'Start$':>10} | {'End$':>10} | {'Return%':>9}")
    print("-" * 50)
    pos_months = 0; neg_months = 0; zero_months = 0
    for ym in sorted(monthly.keys()):
        m = monthly[ym]
        if m['ret_pct'] > 0:
            marker = "+"; pos_months += 1
        elif m['ret_pct'] < 0:
            marker = "-"; neg_months += 1
        else:
            marker = " "; zero_months += 1
        print(f"{ym:<10} | ${m['start_bal']:>9,.2f} | ${m['end_bal']:>9,.2f} | {marker}{abs(m['ret_pct']):>7.2f}%")

    total_active = pos_months + neg_months
    print(f"\nPositive months: {pos_months}/{total_active} ({pos_months/total_active*100:.0f}%)" if total_active > 0 else "")
    print(f"Zero-activity months: {zero_months}")
    print(f"Final balance: ${final_bal:,.2f}")

    # ============================================================
    # INDIVIDUAL RUN DETAILS
    # ============================================================
    print(f"\n\n{'='*110}")
    print(f"  ALL 20 RUNS PER STRATEGY")
    print(f"{'='*110}")

    for name in STRATEGIES:
        details = all_run_details[name]
        print(f"\n--- {name} ---")
        print(f"  {'Run':<6} {'Bars':>12} {'Final$':>10} {'Ret%':>8} {'PF':>6} {'MDD%':>6} {'Trades':>6} {'WR%':>5} {'SL':>4}")
        for d in details:
            brange = f"{d['start']}-{d['end']}"
            print(f"  {d['label']:<6} {brange:>12} ${d['final_balance']:>9,.2f} {d['return_pct']:>7.1f}% "
                  f"{d['pf']:>5.2f} {d['mdd']:>5.1f}% {d['trades']:>5d} {d['wr']:>5.0f}% {d['sl_hits']:>3d}")

    # ============================================================
    # SAVE CSV
    # ============================================================
    csv_path = os.path.join(DIR, "best10_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['strategy', 'run', 'start_bar', 'end_bar', 'bars',
                     'final_balance', 'return_pct', 'pf', 'mdd', 'trades', 'wins', 'wr', 'sl_hits'])
        for name in STRATEGIES:
            for d in all_run_details[name]:
                w.writerow([name, d['label'], d['start'], d['end'], d['end']-d['start'],
                            d['final_balance'], d['return_pct'], d['pf'],
                            d['mdd'], d['trades'], d['wins'], d['wr'], d['sl_hits']])
    print(f"\nSaved: {csv_path}")

    # ============================================================
    # SAVE REPORT
    # ============================================================
    rpt_path = os.path.join(DIR, "best10_report.txt")
    with open(rpt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 110 + "\n")
        f.write("  TOP 10 STRATEGY VERIFICATION BACKTEST REPORT\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Data: BTC/USDT 5m resampled | Init: ${INIT:,.0f} | Fee: {FEE*100:.2f}%/side | 20 runs/strategy\n")
        f.write("=" * 110 + "\n\n")

        f.write("STRATEGY DEFINITIONS:\n")
        f.write("-" * 90 + "\n")
        for name, cfg in STRATEGIES.items():
            f.write(f"  {name}: {cfg['fast_type'].upper()}({cfg['fast_p']})/{cfg['slow_type'].upper()}({cfg['slow_p']}) "
                    f"{cfg['tf']} ADX({cfg['adx_p']})>={cfg['adx_th']} RSI({cfg['rsi_lo']}-{cfg['rsi_hi']}) "
                    f"D{cfg['delay']} SL{cfg['sl_pct']}% Trail+{cfg['trail_on']}/{cfg['trail_off']} "
                    f"M{cfg['margin_pct']}% Lev{cfg['lev']}x\n")

        f.write(f"\n\nRANKING BY COMPOSITE SCORE:\n")
        f.write("-" * 105 + "\n")
        f.write(f"{'Rank':<5} {'Strategy':<10} | {'Avg Ret%':>9} | {'Med Ret%':>9} | {'PF':>6} | {'MDD%':>6} | {'Trd':>5} | {'WR%':>5} | {'Std%':>8} | {'Win/20':>6} | {'Det?':>4} | {'Score':>7}\n")
        f.write("-" * 105 + "\n")
        for i, r in enumerate(ranked):
            f.write(f"  #{i+1:<3} {r['strategy']:<10} | {r['avg_return_pct']:>8.1f}% | {r['med_return_pct']:>8.1f}% | "
                    f"{r['avg_pf']:>5.2f} | {r['avg_mdd']:>5.1f}% | {r['avg_trades']:>4.0f} | "
                    f"{r['avg_wr']:>4.0f}% | {r['std_ret']:>7.2f}% | {r['profitable_runs']:>3d}/20 | "
                    f"{r['deterministic']:>4} | {r['score']:>6.2f}\n")

        f.write(f"\n\nBEST STRATEGY MONTHLY BREAKDOWN: {best_name}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Month':<10} | {'Start$':>10} | {'End$':>10} | {'Return%':>9}\n")
        f.write("-" * 50 + "\n")
        for ym in sorted(monthly.keys()):
            m = monthly[ym]
            marker = "+" if m['ret_pct'] > 0 else "-" if m['ret_pct'] < 0 else " "
            f.write(f"{ym:<10} | ${m['start_bal']:>9,.2f} | ${m['end_bal']:>9,.2f} | {marker}{abs(m['ret_pct']):>7.2f}%\n")
        f.write(f"\nPositive months: {pos_months}/{total_active} ({pos_months/total_active*100:.0f}%)\n" if total_active > 0 else "\n")
        f.write(f"Final balance: ${final_bal:,.2f}\n")

        f.write(f"\n\nALL 20 RUNS PER STRATEGY:\n")
        f.write("=" * 100 + "\n")
        for name in STRATEGIES:
            details = all_run_details[name]
            f.write(f"\n--- {name} ---\n")
            f.write(f"  {'Run':<6} {'Bars':>12} {'Final$':>10} {'Ret%':>8} {'PF':>6} {'MDD%':>6} {'Trd':>5} {'WR%':>5} {'SL':>4}\n")
            for d in details:
                brange = f"{d['start']}-{d['end']}"
                f.write(f"  {d['label']:<6} {brange:>12} ${d['final_balance']:>9,.2f} {d['return_pct']:>7.1f}% "
                        f"{d['pf']:>5.2f} {d['mdd']:>5.1f}% {d['trades']:>5d} {d['wr']:>5.0f}% {d['sl_hits']:>3d}\n")

    print(f"Saved: {rpt_path}")
    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")
    print("DONE.")


if __name__ == '__main__':
    main()
