# -*- coding: utf-8 -*-
"""
21 Versions x 30 Backtest Verification (Self-contained)
Built-in backtest engine - no external dependencies except pandas/numpy
"""
import sys, os, time, json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# BUILT-IN BACKTEST ENGINE
# ============================================================
def load_5m():
    parts = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.startswith('btc_usdt_5m') and f.endswith('.csv'):
            df = pd.read_csv(os.path.join(DATA_DIR, f), parse_dates=['timestamp'])
            parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    df = df.rename(columns={'timestamp': 'time'})
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    print(f"5m: {len(df):,} candles ({df['time'].iloc[0]} ~ {df['time'].iloc[-1]})")
    return df

def resample(df, mins):
    d = df.set_index('time').resample(f'{mins}min').agg(
        {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
    return d

def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(p).mean()
def wma(s, p):
    w = np.arange(1, p+1, dtype=float)
    return s.rolling(p).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
def hma(s, p):
    h = max(int(p/2),1); sq = max(int(np.sqrt(p)),1)
    return wma(2*wma(s,h) - wma(s,p), sq)

def calc_ma(s, t, p, v=None):
    if t == 'ema': return ema(s, p)
    elif t == 'sma': return sma(s, p)
    elif t == 'wma': return wma(s, p)
    elif t == 'hma': return hma(s, p)
    return ema(s, p)

def rsi_w(c, p=14):
    d = c.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    al = l.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    return 100 - 100/(1 + ag/al.replace(0, np.nan))

def adx_w(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    up = h - h.shift(1); dn = l.shift(1) - l
    pdm = np.where((up>dn)&(up>0), up, 0.0)
    mdm = np.where((dn>up)&(dn>0), dn, 0.0)
    atr = pd.Series(tr, index=c.index).ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    pdi = 100*pd.Series(pdm, index=c.index).ewm(alpha=1/p, min_periods=p, adjust=False).mean()/atr
    mdi = 100*pd.Series(mdm, index=c.index).ewm(alpha=1/p, min_periods=p, adjust=False).mean()/atr
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/p, min_periods=p, adjust=False).mean()

def atr_w(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, min_periods=p, adjust=False).mean()


def backtest(df, cfg):
    """Core backtest engine"""
    n = len(df)
    if n < 200: return None

    # Unpack config
    adx_min = cfg['adx_min']; rsi_min = cfg['rsi_min']; rsi_max = cfg['rsi_max']
    sl_pct = cfg['sl']; ta_pct = cfg['ta']; tw_pct = cfg['tw']
    lev = cfg['lev']; margin = cfg['margin']
    ml = cfg.get('ml', 0); cp = cfg.get('cp', 0); cp_dur = cfg.get('cp_dur', 0)
    dd_thresh = cfg.get('dd', 0)
    fee = 0.0004; capital = 3000.0

    times = df['time'].values
    closes = df['close'].values; highs = df['high'].values; lows = df['low'].values
    mf = df['ma_fast'].values; ms = df['ma_slow'].values
    rsi_v = df['rsi'].values; adx_v = df['adx'].values

    balance = capital; peak_bal = balance
    pos = 0; entry_p = 0.0; entry_i = 0; sz = 0.0
    peak_price = 0.0; trail_on = False; rem = 1.0
    m_start = balance; cur_m = ''; m_paused = False
    c_loss = 0; pause_til = 0
    liq_dist = 1.0 / lev

    trades = []; monthly = {}
    m_tr = 0; m_sl = 0; m_tsl = 0; m_rev = 0; m_fl = 0

    def save_m():
        nonlocal m_tr, m_sl, m_tsl, m_rev, m_fl
        if cur_m:
            monthly[cur_m] = {'pnl': balance - m_start,
                'pct': (balance - m_start)/m_start*100 if m_start > 0 else 0,
                'bal': round(balance), 'tr': m_tr, 'sl': m_sl, 'tsl': m_tsl, 'rev': m_rev, 'fl': m_fl}
        m_tr = 0; m_sl = 0; m_tsl = 0; m_rev = 0; m_fl = 0

    def close_p(ep, et, idx):
        nonlocal balance, pos, c_loss, pause_til, peak_bal, m_tr, m_sl, m_tsl, m_rev, m_fl
        pp = pos * (ep - entry_p) / entry_p
        pu = sz * rem * pp; fe = sz * rem * fee
        balance += pu - fe
        if et == 'FL': balance = max(balance, 0)
        m_tr += 1
        if et == 'SL': m_sl += 1
        elif et == 'TSL': m_tsl += 1
        elif et == 'REV': m_rev += 1
        elif et == 'FL': m_fl += 1
        trades.append({'pnl': pp, 'type': et, 'time': str(times[idx])[:19]})
        if pp > 0: c_loss = 0
        else:
            c_loss += 1
            if cp > 0 and c_loss >= cp: pause_til = idx + cp_dur
        pos = 0; peak_bal = max(peak_bal, balance)

    def enter_p(d, price, idx):
        nonlocal balance, pos, entry_p, entry_i, sz, peak_price, trail_on, rem, peak_bal
        mg = margin
        if peak_bal > 0 and dd_thresh < 0:
            if (peak_bal - balance)/peak_bal > abs(dd_thresh): mg = margin/2
        pos = d; entry_p = price; entry_i = idx
        sz = balance * mg * lev
        balance -= sz * fee
        peak_price = price; trail_on = False; rem = 1.0
        peak_bal = max(peak_bal, balance)

    for i in range(1, n):
        t = str(times[i])[:19]; cc = closes[i]; hh = highs[i]; ll = lows[i]
        if np.isnan(mf[i]) or np.isnan(ms[i]) or np.isnan(rsi_v[i]) or np.isnan(adx_v[i]):
            continue
        mk = t[:7]
        if mk != cur_m:
            save_m(); cur_m = mk; m_start = balance; m_paused = False

        if pos != 0:
            w = ll if pos == 1 else hh
            if pos*(w - entry_p)/entry_p <= -liq_dist:
                close_p(entry_p*(1 - pos*liq_dist), 'FL', i); continue
            if pos*(w - entry_p)/entry_p <= -sl_pct:
                close_p(entry_p*(1 - pos*sl_pct), 'SL', i); continue
            if pos == 1: peak_price = max(peak_price, hh)
            else: peak_price = min(peak_price, ll)
            ppnl = pos*(peak_price - entry_p)/entry_p
            if ppnl >= ta_pct: trail_on = True
            if trail_on:
                if pos == 1:
                    tsl = peak_price*(1 - tw_pct)
                    if cc <= tsl: close_p(tsl, 'TSL', i); continue
                else:
                    tsl = peak_price*(1 + tw_pct)
                    if cc >= tsl: close_p(tsl, 'TSL', i); continue
            cu = mf[i]>ms[i] and mf[i-1]<=ms[i-1]
            cd = mf[i]<ms[i] and mf[i-1]>=ms[i-1]
            ao = adx_v[i] >= adx_min; ro = rsi_min <= rsi_v[i] <= rsi_max
            nd = 0
            if pos == 1 and cd and ao and ro: nd = -1
            elif pos == -1 and cu and ao and ro: nd = 1
            if nd != 0:
                close_p(cc, 'REV', i)
                if balance > 10 and not m_paused and i >= pause_til:
                    enter_p(nd, cc, i)
                continue
            if ml < 0 and m_start > 0:
                ur = sz*rem*pos*(cc-entry_p)/entry_p
                if (balance+ur-m_start)/m_start < ml: m_paused = True

        if pos == 0 and balance > 10:
            if ml < 0 and m_start > 0:
                if (balance - m_start)/m_start < ml: m_paused = True
            if m_paused or i < pause_til: continue
            cu = mf[i]>ms[i] and mf[i-1]<=ms[i-1]
            cd = mf[i]<ms[i] and mf[i-1]>=ms[i-1]
            ao = adx_v[i] >= adx_min; ro = rsi_min <= rsi_v[i] <= rsi_max
            if cu and ao and ro: enter_p(1, cc, i)
            elif cd and ao and ro: enter_p(-1, cc, i)
        peak_bal = max(peak_bal, balance)

    if pos != 0: close_p(closes[-1], 'END', n-1)
    save_m()

    # Compile
    total = len(trades)
    if total == 0: return None
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gp = sum(t['pnl']*sz for t in wins) if wins else 0  # approximate
    gl = abs(sum(t['pnl']*sz for t in losses)) if losses else 1
    # Better PF from actual balances
    gross_win = sum(monthly[m]['pnl'] for m in monthly if monthly[m]['pnl'] > 0)
    gross_loss = abs(sum(monthly[m]['pnl'] for m in monthly if monthly[m]['pnl'] < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else gross_win

    pk = 3000; mdd = 0
    for m in sorted(monthly.keys()):
        b = monthly[m]['bal']; pk = max(pk, b)
        d = (pk-b)/pk if pk > 0 else 0; mdd = max(mdd, d)

    yr = {}
    yr_start = {}; yr_end = {}
    for m in sorted(monthly.keys()):
        y = m[:4]
        if y not in yr_start: yr_start[y] = monthly[m]['bal'] - monthly[m]['pnl']
        yr_end[y] = monthly[m]['bal']
    for y in yr_start:
        s = yr_start[y]; e = yr_end[y]
        yr[y] = round((e-s)/s*100, 1) if s > 0 else 0

    t_sl = sum(monthly[m]['sl'] for m in monthly)
    t_tsl = sum(monthly[m]['tsl'] for m in monthly)
    t_rev = sum(monthly[m]['rev'] for m in monthly)
    t_fl = sum(monthly[m]['fl'] for m in monthly)
    loss_m = sum(1 for m in monthly if monthly[m]['pnl'] < 0)

    return {
        'bal': round(balance), 'ret': round((balance-3000)/3000*100, 1),
        'pf': round(pf, 2), 'mdd': round(mdd*100, 1),
        'trades': total, 'wr': round(len(wins)/total*100, 1),
        'sl': t_sl, 'tsl': t_tsl, 'rev': t_rev, 'fl': t_fl,
        'loss_m': loss_m, 'yr': yr, 'monthly': monthly,
    }


# ============================================================
# VERSIONS
# ============================================================
VERSIONS = [
    # (name, tf, mf_t, mf_p, ms_t, ms_p, adx_p, adx_min, rsi_min, rsi_max,
    #  sl, ta, tw, lev, margin, ml, cp, cp_dur, dd, c_bal, c_ret, c_pf, c_mdd)
    ("v10.1","5m","ema",7,"ema",100,14,30,30,60, 0.09,0.20,0.05,10,0.20, 0,0,0,0, 6144,105,1.36,98.1),
    ("v11.1","5m","ema",7,"ema",100,14,20,30,60, 0.06,0.08,0.06,10,0.20, 0,0,0,0, 35492,1083,0,94.6),
    ("v12.0","5m","ema",7,"ema",100,14,30,30,58, 0.09,0.08,0.06,10,0.20, 0,0,0,0, 20216,574,0,96.1),
    ("v12.2","5m","ema",7,"ema",100,14,30,30,58, 0.09,0.08,0.01,10,0.20, 0,0,0,0, 31014,934,6.34,14.4),
    ("v12.3","5m","ema",7,"ema",100,14,30,30,58, 0.09,0.08,0.06,10,0.20, 0,0,0,0, 252892,8330,13.34,39.9),
    ("v12.5","5m","ema",7,"ema",100,14,30,30,58, 0.08,0.09,0.05,10,0.20, 0,0,0,0, 233881,7696,0,41.0),
    ("v13.0","5m","ema",7,"ema",100,14,30,30,58, 0.09,0.08,0.06,10,0.20, -0.15,3,288,-0.30, 254105,8370,9.00,68.3),
    ("v13.2","5m","ema",30,"ema",200,14,35,35,65, 0.04,0.08,0.03,10,0.10, 0,0,0,0, 17091,470,1.4,37.4),
    ("v13.3","5m","ema",7,"ema",100,14,30,30,58, 0.07,0.08,0.06,10,0.20, -0.20,3,288,-0.50, 468530,15518,6.87,74.3),
    ("v13.4","30m","ema",5,"ema",100,20,25,30,58, 0.06,0.06,0.03,10,0.25, -0.15,0,0,-0.30, 161046,5268,2.54,29.9),
    ("v13.5","5m","ema",7,"ema",100,14,30,30,58, 0.07,0.08,0.06,10,0.20, -0.20,3,288,-0.50, 468530,15518,6.87,74.3),
    ("v14.1","5m","ema",7,"ema",100,14,30,30,58, 0.09,0.10,0.06,10,0.20, -0.25,3,288,-0.30, 124882,4063,7.05,77.9),
    ("v14.2F","30m","hma",7,"ema",200,20,25,25,65, 0.07,0.10,0.01,10,0.30, -0.15,0,0,-0.40, 798358,26512,1.29,57.8),
    ("v14.3","5m","ema",7,"ema",100,14,30,30,58, 0.10,0.08,0.06,10,0.30, -0.25,3,288,-0.30, 3210275,106909,22.50,72.1),
    ("v14.4","30m","ema",3,"ema",200,14,35,30,65, 0.07,0.06,0.03,10,0.25, -0.20,0,0,0, 837212,27807,2.04,36.9),
    ("v15.1","5m","ema",7,"ema",100,14,30,30,58, 0.07,0.08,0.06,10,0.20, -0.20,3,288,0, 206282,6776,5.60,79.9),
    ("v15.2","30m","ema",3,"ema",200,14,35,30,65, 0.05,0.06,0.05,10,0.30, -0.15,0,0,0, 243482,8016,2.48,27.6),
    ("v15.3A","5m","ema",7,"ema",100,14,30,30,58, 0.07,0.08,0.06,10,0.30, -0.25,3,288,0, 1240536,41320,10.82,76.8),
    ("v15.4","30m","ema",3,"ema",200,14,35,30,65, 0.07,0.06,0.03,10,0.40, -0.30,0,0,0, 8717659,290489,1.65,54.2),
    ("v15.5","15m","ema",21,"ema",250,20,40,40,75, 0.04,0.20,0.05,15,0.30, -0.30,0,0,0, 192281,6309,22.50,20.1),
]


def main():
    t0 = time.time()
    print("="*90)
    print("  21 VERSIONS x 30 BACKTEST VERIFICATION")
    print("="*90); sys.stdout.flush()

    df_5m = load_5m()
    mtf = {'5m': df_5m}
    for mins, name in [(10,'10m'),(15,'15m'),(30,'30m'),(60,'1h')]:
        mtf[name] = resample(df_5m, mins)
        print(f"  {name}: {len(mtf[name]):,}"); sys.stdout.flush()

    results = []

    for vi, v in enumerate(VERSIONS):
        name, tf, mf_t, mf_p, ms_t, ms_p, adx_p, adx_min, rsi_min, rsi_max, \
            sl, ta, tw, lev, mg, ml, cp, cp_dur, dd, c_bal, c_ret, c_pf, c_mdd = v

        print(f"\n--- [{vi+1}/{len(VERSIONS)}] {name} ({tf} {mf_t}({mf_p}/{ms_p})) ---"); sys.stdout.flush()

        df_raw = mtf[tf]
        # Pre-compute indicators
        df = df_raw.copy()
        c_s, h_s, l_s, vol = df['close'], df['high'], df['low'], df['volume']
        df['ma_fast'] = calc_ma(c_s, mf_t, mf_p, vol)
        df['ma_slow'] = calc_ma(c_s, ms_t, ms_p, vol)
        df['adx'] = adx_w(h_s, l_s, c_s, adx_p)
        df['rsi'] = rsi_w(c_s, 14)

        cfg = {'adx_min': adx_min, 'rsi_min': rsi_min, 'rsi_max': rsi_max,
               'sl': sl, 'ta': ta, 'tw': tw, 'lev': lev, 'margin': mg,
               'ml': ml, 'cp': cp, 'cp_dur': cp_dur, 'dd': dd}

        # Run 30 times (deterministic = same result, but we verify consistency)
        r = backtest(df, cfg)
        if r is None:
            print(f"  FAILED"); sys.stdout.flush()
            results.append({'ver': name, 'status': 'FAIL', 'actual_bal':0, 'actual_ret':0,
                           'actual_pf':0, 'actual_mdd':0, 'claimed_bal':c_bal,
                           'claimed_ret':c_ret, 'claimed_pf':c_pf, 'claimed_mdd':c_mdd,
                           'score':-999, 'tf': tf})
            continue

        # Verify 30x consistency
        consistent = True
        for run in range(29):
            r2 = backtest(df, cfg)
            if r2 is None or abs(r2['bal'] - r['bal']) > 1:
                consistent = False; break

        bd = abs(r['bal'] - c_bal) / max(c_bal, 1) * 100 if c_bal > 0 else 0
        status = "MATCH" if bd < 25 else ("CLOSE" if bd < 50 else "DIFFER")
        if not consistent: status = "UNSTABLE"

        print(f"  Actual:  ${r['bal']:>12,} | Ret:{r['ret']:>+10,.1f}% | PF:{r['pf']:>6.2f} | MDD:{r['mdd']:>5.1f}% | TR:{r['trades']} SL:{r['sl']} TSL:{r['tsl']} FL:{r['fl']}")
        print(f"  Claimed: ${c_bal:>12,} | Ret:{c_ret:>+10,.1f}% | PF:{c_pf:>6.2f} | MDD:{c_mdd:>5.1f}%")
        print(f"  30x Consistent: {'YES' if consistent else 'NO'} | Bal diff: {bd:.1f}% | Status: {status}")
        yr = r.get('yr', {})
        if yr:
            ys = ' '.join([f"{k}:{v:+.0f}%" for k,v in sorted(yr.items())])
            print(f"  Yearly: {ys}")
        sys.stdout.flush()

        # Score
        pf = r['pf']; ret = r['ret']; mdd = max(r['mdd'], 1); fl = r['fl']; tr = r['trades']
        sc = -999
        if ret > 0 and pf > 0 and tr >= 10:
            mm = 1.0
            if mdd > 80: mm = 0.2
            elif mdd > 60: mm = 0.4
            elif mdd > 40: mm = 0.7
            elif mdd > 30: mm = 0.85
            sc = pf**1.5 * np.log1p(ret/100) * mm * max(0.1, 1 - fl*0.15)

        results.append({
            'ver': name, 'status': status, 'consistent': consistent, 'tf': tf,
            'actual_bal': r['bal'], 'actual_ret': r['ret'],
            'actual_pf': r['pf'], 'actual_mdd': r['mdd'],
            'actual_tr': r['trades'], 'actual_fl': r['fl'],
            'actual_sl': r['sl'], 'actual_tsl': r['tsl'], 'actual_rev': r['rev'],
            'claimed_bal': c_bal, 'claimed_ret': c_ret,
            'claimed_pf': c_pf, 'claimed_mdd': c_mdd,
            'bal_diff': bd, 'score': sc, 'yearly': yr, 'loss_m': r['loss_m'],
        })

    # RANKING
    elapsed = time.time() - t0
    results.sort(key=lambda x: x.get('score', -999), reverse=True)

    print(f"\n\n{'='*100}")
    print(f"  VERIFICATION COMPLETE: {len(results)} versions, 30x each, {elapsed:.0f}s")
    print(f"{'='*100}")
    print(f"\n{'#':>3} {'Ver':>8} {'TF':>4} {'Actual Bal':>14} {'Ret':>10} {'PF':>7} {'MDD':>7} {'TR':>4} {'FL':>3} {'Score':>8} {'Status':>8}")
    print("-"*90)
    for i, r in enumerate(results):
        print(f"{i+1:>3} {r['ver']:>8} {r['tf']:>4} "
              f"${r['actual_bal']:>12,} {r['actual_ret']:>+9,.1f}% "
              f"{r['actual_pf']:>6.2f} {r['actual_mdd']:>6.1f}% "
              f"{r['actual_tr']:>3} {r['actual_fl']:>2} "
              f"{r['score']:>8.1f} {r['status']:>8}")
    sys.stdout.flush()

    with open(os.path.join(DATA_DIR, 'verify_all_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to verify_all_results.json")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
