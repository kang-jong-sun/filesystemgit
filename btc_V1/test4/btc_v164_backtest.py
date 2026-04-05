"""
BTC/USDT v16.4 AMCS (Adaptive Multi-Model Convergence System) 백테스트
- Model Α "Sniper": PF 극대화 (ADX≥45, 1H/4H 필터)
- Model Β "Machine Gun": 수익 극대화 (ADX≥30, 분할청산)
- Model Γ "Chameleon": 적응형 (1H ADX 국면 전환)
"""
import pandas as pd
import numpy as np
import time, sys, os, io
from pathlib import Path

# Fix Windows console encoding (only when run directly, not on import)
if sys.platform == 'win32' and __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

DATA_DIR = Path(r'D:\filesystem\futures\btc_V1\test4')
FEE_RATE = 0.0004  # 0.04% taker

# ====================================================================
# 1. DATA LOADING & RESAMPLING
# ====================================================================
def load_5m_data():
    files = ['btc_usdt_5m_2020_to_now_part1.csv',
             'btc_usdt_5m_2020_to_now_part2.csv',
             'btc_usdt_5m_2020_to_now_part3.csv']
    dfs = []
    for f in files:
        fp = DATA_DIR / f
        df = pd.read_csv(fp, parse_dates=['timestamp'])
        dfs.append(df)
        print(f"  {f}: {len(df):,} rows")
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    print(f"  Total: {len(df):,} rows  [{df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}]")
    return df

def resample(df_5m, rule):
    df = df_5m.set_index('timestamp')
    agg = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    r = df.resample(rule).agg(agg).dropna(subset=['open']).reset_index()
    return r

# ====================================================================
# 2. INDICATORS (Wilder's Smoothing for ADX/RSI/ATR)
# ====================================================================
def calc_wma(s, n):
    w = np.arange(1, n+1, dtype=np.float64)
    ws = w.sum()
    return s.rolling(n).apply(lambda x: np.dot(x, w)/ws, raw=True)

def calc_ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def calc_sma(s, n):
    return s.rolling(n).mean()

def calc_adx(high, low, close, period=14):
    h, l, c = high.values, low.values, close.values
    n = len(h)
    tr = np.zeros(n); pdm = np.zeros(n); ndm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        up = h[i]-h[i-1]; dn = l[i-1]-l[i]
        pdm[i] = up if (up > dn and up > 0) else 0
        ndm[i] = dn if (dn > up and dn > 0) else 0
    # Wilder's smoothing for TR/DM: first=SUM, then prev*(N-1)/N + current
    def wilder_sum(arr, p, start=1):
        out = np.full(n, np.nan)
        if start+p <= n:
            out[start+p-1] = arr[start:start+p].sum()
            for i in range(start+p, n):
                out[i] = out[i-1]*(p-1)/p + arr[i]
        return out
    str_ = wilder_sum(tr, period); spdm = wilder_sum(pdm, period); sndm = wilder_sum(ndm, period)
    with np.errstate(divide='ignore', invalid='ignore'):
        pdi = np.where(str_>0, spdm/str_*100, 0)
        ndi = np.where(str_>0, sndm/str_*100, 0)
        ds = pdi+ndi
        dx = np.where(ds>0, np.abs(pdi-ndi)/ds*100, 0)
    # ADX: Wilder's smoothing of DX with MEAN init and /N formula
    # DX valid from index=period (where smoothed TR/DM first appear)
    adx_arr = np.full(n, np.nan)
    dx_start = period  # first valid DX index
    if dx_start + period <= n:
        adx_arr[dx_start+period-1] = np.mean(dx[dx_start:dx_start+period])
        for i in range(dx_start+period, n):
            adx_arr[i] = (adx_arr[i-1]*(period-1) + dx[i]) / period
    return pd.Series(adx_arr, index=close.index)

def calc_rsi(close, period=14):
    d = close.diff().values
    g = np.where(d>0, d, 0.0)
    l = np.where(d<0, -d, 0.0)
    n = len(close)
    ag = np.full(n, np.nan); al = np.full(n, np.nan)
    if period+1 <= n:
        ag[period] = g[1:period+1].mean()
        al[period] = l[1:period+1].mean()
        for i in range(period+1, n):
            ag[i] = (ag[i-1]*(period-1)+g[i])/period
            al[i] = (al[i-1]*(period-1)+l[i])/period
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(al>0, ag/al, 100.0)
        rsi = 100.0 - 100.0/(1.0+rs)
    rsi[np.isnan(ag)] = np.nan
    return pd.Series(rsi, index=close.index)

def calc_atr(high, low, close, period=14):
    h, l, c = high.values, low.values, close.values
    n = len(h)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    atr = np.full(n, np.nan)
    if period+1 <= n:
        atr[period] = tr[1:period+1].mean()
        for i in range(period+1, n):
            atr[i] = (atr[i-1]*(period-1)+tr[i])/period
    return pd.Series(atr, index=close.index)

def add_indicators(df):
    df['wma3'] = calc_wma(df['close'], 3)
    df['ema3'] = calc_ema(df['close'], 3)
    df['ema50'] = calc_ema(df['close'], 50)
    df['ema100'] = calc_ema(df['close'], 100)
    df['ema200'] = calc_ema(df['close'], 200)
    df['adx14'] = calc_adx(df['high'], df['low'], df['close'], 14)
    df['adx20'] = calc_adx(df['high'], df['low'], df['close'], 20)
    df['rsi14'] = calc_rsi(df['close'], 14)
    df['atr14'] = calc_atr(df['high'], df['low'], df['close'], 14)
    df['vol_sma20'] = calc_sma(df['volume'], 20)
    return df

# ====================================================================
# 3. BACKTEST ENGINE
# ====================================================================
def run_backtest(df_30m, df_1h, df_4h, p, capital):
    """
    p = params dict:
      fast_ma, slow_ma, adx_col, adx_min, rsi_min, rsi_max,
      sl_pct, sl_dynamic(bool), atr_mult,
      ts_act, ts_trail, ts_accel(bool),
      partial_exits=[(roi,pct),...], reverse('flip'/'ignore'),
      time_sl(candles or None), h1_filter(bool), h4_filter(bool),
      margin, leverage, re_entry_block(candles or 0)
    """
    bal = capital
    pos = None  # (dir, entry_px, entry_time, size, peak_roi, highest, lowest, entry_i, partial_done, remain_pct)
    trades = []; monthly = {}
    fast = df_30m[p['fast_ma']].values
    slow = df_30m[p['slow_ma']].values
    adx = df_30m[p['adx_col']].values
    rsi = df_30m['rsi14'].values
    atr = df_30m['atr14'].values
    closes = df_30m['close'].values
    highs = df_30m['high'].values
    lows = df_30m['low'].values
    ts_arr = df_30m['timestamp'].values

    # Pre-merge 1H/4H trend
    h1_ema_up = np.full(len(df_30m), True)
    h4_ema_up = np.full(len(df_30m), True)
    h1_ema_dn = np.full(len(df_30m), True)
    h4_ema_dn = np.full(len(df_30m), True)
    if p.get('h1_filter'):
        h1_ts = df_1h['timestamp'].values; h1_e50 = df_1h['ema50'].values
        j = 0
        for i in range(len(df_30m)):
            while j+1 < len(h1_ts) and h1_ts[j+1] <= ts_arr[i]: j += 1
            if j > 0 and not np.isnan(h1_e50[j]) and not np.isnan(h1_e50[j-1]):
                h1_ema_up[i] = h1_e50[j] > h1_e50[j-1]
                h1_ema_dn[i] = h1_e50[j] < h1_e50[j-1]
            else:
                h1_ema_up[i] = True; h1_ema_dn[i] = True
    if p.get('h4_filter'):
        h4_ts = df_4h['timestamp'].values; h4_e100 = df_4h['ema100'].values
        j = 0
        for i in range(len(df_30m)):
            while j+1 < len(h4_ts) and h4_ts[j+1] <= ts_arr[i]: j += 1
            if j > 0 and not np.isnan(h4_e100[j]) and not np.isnan(h4_e100[j-1]):
                h4_ema_up[i] = h4_e100[j] > h4_e100[j-1]
                h4_ema_dn[i] = h4_e100[j] < h4_e100[j-1]
            else:
                h4_ema_up[i] = True; h4_ema_dn[i] = True

    peak_bal = capital; mdd = 0.0
    last_sl_dir = None; last_sl_i = -9999
    margin = p['margin']; lev = p['leverage']
    sl_pct = p['sl_pct']; ts_act = p['ts_act']; ts_trail = p['ts_trail']
    partial_cfg = p.get('partial_exits', [])
    rev_mode = p.get('reverse', 'flip')
    time_sl = p.get('time_sl', None)
    re_block = p.get('re_entry_block', 0)

    def mk(i):
        return pd.Timestamp(ts_arr[i]).strftime('%Y-%m')
    def init_month(m):
        if m not in monthly:
            monthly[m] = {'start_bal': bal, 'end_bal': bal, 'trades':0, 'sl':0, 'tsl':0, 'rev':0, 'pnl':0.0}

    def close_pos(exit_px, reason, i):
        nonlocal bal, pos, last_sl_dir, last_sl_i
        d, epx, etime, sz, pk, hi, lo, ei, pd_done, rem = pos
        roi = (exit_px/epx - 1) if d=='long' else (1 - exit_px/epx)
        gross = sz * lev * roi * rem
        fee = sz * lev * rem * FEE_RATE * 2
        pnl = gross - fee
        bal += pnl
        m = mk(i); init_month(m)
        monthly[m]['pnl'] += pnl; monthly[m]['trades'] += 1
        if reason=='SL': monthly[m]['sl'] += 1
        elif reason=='TSL': monthly[m]['tsl'] += 1
        else: monthly[m]['rev'] += 1
        trades.append({'entry_time': etime, 'exit_time': pd.Timestamp(ts_arr[i]),
                        'dir': d, 'entry_px': epx, 'exit_px': exit_px,
                        'roi': roi, 'pnl': pnl, 'reason': reason,
                        'peak_roi': pk, 'bal_after': bal})
        if reason == 'SL': last_sl_dir = d; last_sl_i = i
        pos = None
        return pnl

    for i in range(1, len(df_30m)):
        if np.isnan(fast[i]) or np.isnan(slow[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            continue
        m = mk(i); init_month(m)
        cpx = closes[i]; hi_px = highs[i]; lo_px = lows[i]

        # --- Position management ---
        if pos is not None:
            d, epx, etime, sz, pk, hi, lo, ei, pd_done, rem = pos
            # Update hi/lo
            hi = max(hi, hi_px); lo = min(lo, lo_px)
            roi = (cpx/epx - 1) if d=='long' else (1 - cpx/epx)
            best_roi = (hi/epx - 1) if d=='long' else (1 - lo/epx)
            pk = max(pk, best_roi)
            pos = (d, epx, etime, sz, pk, hi, lo, ei, pd_done, rem)

            exit_reason = None; exit_px = cpx

            # SL check (intra-candle: use low for long, high for short)
            if p.get('sl_dynamic') and not np.isnan(atr[i]):
                dyn_sl = min(sl_pct, atr[i] * p['atr_mult'] / epx)
            else:
                dyn_sl = sl_pct
            if d == 'long':
                sl_price = epx * (1 - dyn_sl)
                if lo_px <= sl_price:
                    exit_reason = 'SL'; exit_px = sl_price
            else:
                sl_price = epx * (1 + dyn_sl)
                if hi_px >= sl_price:
                    exit_reason = 'SL'; exit_px = sl_price

            # Time SL
            if exit_reason is None and time_sl and (i - ei) >= time_sl and roi <= 0:
                exit_reason = 'TIME_SL'; exit_px = cpx

            # Trailing stop
            if exit_reason is None:
                if pk >= ts_act:
                    tw = ts_trail
                    if p.get('ts_accel'):
                        if pk >= ts_act*4: tw = ts_trail*0.5
                        elif pk >= ts_act*3: tw = ts_trail*0.6
                        elif pk >= ts_act*2: tw = ts_trail*0.8
                    if pk - roi >= tw:
                        exit_reason = 'TSL'; exit_px = cpx

            # Partial exits
            if exit_reason is None and pd_done < len(partial_cfg):
                pe_roi, pe_pct = partial_cfg[pd_done]
                if roi >= pe_roi:
                    part_gross = sz * lev * roi * rem * pe_pct
                    part_fee = sz * lev * rem * pe_pct * FEE_RATE * 2
                    part_pnl = part_gross - part_fee
                    bal += part_pnl
                    monthly[m]['pnl'] += part_pnl
                    rem *= (1 - pe_pct)
                    pd_done += 1
                    pos = (d, epx, etime, sz, pk, hi, lo, ei, pd_done, rem)

            # Reverse signal
            if exit_reason is None and rev_mode == 'flip':
                gc = fast[i-1] <= slow[i-1] and fast[i] > slow[i]
                dc = fast[i-1] >= slow[i-1] and fast[i] < slow[i]
                if d=='long' and dc and adx[i]>=p['adx_min'] and p['rsi_min']<=rsi[i]<=p['rsi_max']:
                    if roi < 0.20:
                        exit_reason = 'REV'; exit_px = cpx
                elif d=='short' and gc and adx[i]>=p['adx_min'] and p['rsi_min']<=rsi[i]<=p['rsi_max']:
                    if roi < 0.20:
                        exit_reason = 'REV'; exit_px = cpx

            if exit_reason:
                close_pos(exit_px, exit_reason, i)
                # Flip on REV
                if exit_reason == 'REV' and rev_mode == 'flip' and bal > 0:
                    new_dir = 'short' if d=='long' else 'long'
                    sz_new = bal * margin
                    pos = (new_dir, cpx, pd.Timestamp(ts_arr[i]), sz_new, 0.0, cpx, cpx, i, 0, 1.0)

        # --- Entry logic ---
        if pos is None and bal > 0:
            gc = fast[i-1] <= slow[i-1] and fast[i] > slow[i]
            dc = fast[i-1] >= slow[i-1] and fast[i] < slow[i]
            sig = None
            if gc: sig = 'long'
            elif dc: sig = 'short'

            if sig:
                if adx[i] < p['adx_min']: sig = None
                if sig and not (p['rsi_min'] <= rsi[i] <= p['rsi_max']): sig = None
                if sig and re_block > 0:
                    if sig == last_sl_dir and (i - last_sl_i) < re_block: sig = None
                if sig and p.get('h1_filter'):
                    if sig=='long' and not h1_ema_up[i]: sig = None
                    if sig=='short' and not h1_ema_dn[i]: sig = None
                if sig and p.get('h4_filter'):
                    if sig=='long' and not h4_ema_up[i]: sig = None
                    if sig=='short' and not h4_ema_dn[i]: sig = None
                if sig:
                    sz = bal * margin
                    pos = (sig, cpx, pd.Timestamp(ts_arr[i]), sz, 0.0, cpx, cpx, i, 0, 1.0)

        monthly[m]['end_bal'] = bal
        if bal > peak_bal: peak_bal = bal
        dd = (peak_bal - bal) / peak_bal if peak_bal > 0 else 0
        if dd > mdd: mdd = dd

    # Close remaining position
    if pos is not None:
        close_pos(closes[-1], 'END', len(df_30m)-1)

    # Stats
    nt = len(trades)
    wins = [t for t in trades if t['pnl']>0]
    losses = [t for t in trades if t['pnl']<=0]
    tp = sum(t['pnl'] for t in wins)
    tl = sum(abs(t['pnl']) for t in losses)
    pf = tp/tl if tl>0 else float('inf')
    wr = len(wins)/nt*100 if nt>0 else 0
    aw = np.mean([t['roi'] for t in wins])*100 if wins else 0
    al_ = np.mean([t['roi'] for t in losses])*100 if losses else 0
    rr = abs(aw/al_) if al_!=0 else float('inf')
    sl_n = sum(1 for t in trades if t['reason']=='SL')
    tsl_n = sum(1 for t in trades if t['reason']=='TSL')
    rev_n = sum(1 for t in trades if t['reason'] in ('REV','TIME_SL','END'))

    return {'bal':bal, 'ret':(bal-capital)/capital*100, 'trades':nt, 'wins':len(wins),
            'losses':len(losses), 'wr':wr, 'pf':pf, 'mdd':mdd*100,
            'aw':aw, 'al':al_, 'rr':rr, 'sl':sl_n, 'tsl':tsl_n, 'rev':rev_n,
            'trade_list':trades, 'monthly':monthly, 'capital':capital}

# ====================================================================
# 4. MODEL Γ (Regime-switching)
# ====================================================================
def run_gamma(df_30m, df_1h, df_4h, capital):
    """Model Γ: 1H ADX 기반 국면 전환"""
    # Merge 1H ADX into 30m
    df = df_30m.copy()
    h1_adx = df_1h[['timestamp','adx20']].rename(columns={'adx20':'h1_adx'})
    df = pd.merge_asof(df.sort_values('timestamp'), h1_adx.sort_values('timestamp'),
                       on='timestamp', direction='backward')
    h1_adx_vals = df['h1_adx'].values

    # Same engine but dynamic params
    bal = capital; pos = None; trades = []; monthly = {}
    fast = df['wma3'].values; slow = df['ema200'].values
    adx = df['adx20'].values; rsi = df['rsi14'].values; atr = df['atr14'].values
    closes = df['close'].values; highs = df['high'].values; lows = df['low'].values
    ts_arr = df['timestamp'].values

    # 1H trend
    h1_e50 = df_1h['ema50'].values; h1_ts = df_1h['timestamp'].values
    h1_up = np.full(len(df), True); h1_dn = np.full(len(df), True)
    j = 0
    for i in range(len(df)):
        while j+1 < len(h1_ts) and h1_ts[j+1] <= ts_arr[i]: j += 1
        if j > 0 and not np.isnan(h1_e50[j]) and not np.isnan(h1_e50[j-1]):
            h1_up[i] = h1_e50[j] > h1_e50[j-1]
            h1_dn[i] = h1_e50[j] < h1_e50[j-1]

    peak_bal = capital; mdd = 0.0
    last_sl_dir = None; last_sl_i = -9999
    lev = 10

    def mk(i): return pd.Timestamp(ts_arr[i]).strftime('%Y-%m')
    def init_m(m):
        if m not in monthly:
            monthly[m] = {'start_bal': bal, 'end_bal': bal, 'trades':0, 'sl':0, 'tsl':0, 'rev':0, 'pnl':0.0}

    def get_regime(i):
        v = h1_adx_vals[i]
        if np.isnan(v): return 'sideways'
        if v >= 40: return 'strong'
        if v >= 25: return 'weak'
        return 'sideways'

    # Regime-specific params
    R = {
        'strong': {'adx_min':40, 'rsi_min':35, 'rsi_max':65, 'sl':0.08, 'sl_dyn':False,
                   'atr_m':3.0, 'ts_act':0.04, 'ts_trail':0.03, 'margin':0.25,
                   'rev':'flip', 'time_sl':None, 'h1':False, 'accel':True},
        'weak':   {'adx_min':35, 'rsi_min':35, 'rsi_max':65, 'sl':0.07, 'sl_dyn':False,
                   'atr_m':3.0, 'ts_act':0.06, 'ts_trail':0.05, 'margin':0.30,
                   'rev':'flip', 'time_sl':None, 'h1':False, 'accel':False},
    }

    pos_regime = None  # regime when position was opened

    for i in range(1, len(df)):
        if np.isnan(fast[i]) or np.isnan(slow[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            continue
        m = mk(i); init_m(m)
        cpx = closes[i]; hi_px = highs[i]; lo_px = lows[i]
        regime = get_regime(i)

        # --- Position mgmt (use entry regime params) ---
        if pos is not None:
            d, epx, etime, sz, pk, hi, lo, ei, pd_done, rem = pos
            rp = R[pos_regime]
            hi = max(hi, hi_px); lo = min(lo, lo_px)
            roi = (cpx/epx-1) if d=='long' else (1-cpx/epx)
            best = (hi/epx-1) if d=='long' else (1-lo/epx)
            pk = max(pk, best)
            pos = (d, epx, etime, sz, pk, hi, lo, ei, pd_done, rem)

            exit_r = None; exit_px = cpx
            # SL
            if rp['sl_dyn'] and not np.isnan(atr[i]):
                dsl = min(rp['sl'], atr[i]*rp['atr_m']/epx)
            else:
                dsl = rp['sl']
            if d=='long':
                slp = epx*(1-dsl)
                if lo_px <= slp: exit_r='SL'; exit_px=slp
            else:
                slp = epx*(1+dsl)
                if hi_px >= slp: exit_r='SL'; exit_px=slp
            # Time SL
            if exit_r is None and rp['time_sl'] and (i-ei)>=rp['time_sl'] and roi<=0:
                exit_r='TIME_SL'; exit_px=cpx
            # TSL
            if exit_r is None and pk >= rp['ts_act']:
                tw = rp['ts_trail']
                if rp['accel']:
                    if pk>=rp['ts_act']*4: tw*=0.5
                    elif pk>=rp['ts_act']*3: tw*=0.6
                    elif pk>=rp['ts_act']*2: tw*=0.8
                if pk-roi >= tw: exit_r='TSL'; exit_px=cpx
            # REV
            if exit_r is None and rp['rev']=='flip':
                gc = fast[i-1]<=slow[i-1] and fast[i]>slow[i]
                dc = fast[i-1]>=slow[i-1] and fast[i]<slow[i]
                if d=='long' and dc and adx[i]>=rp['adx_min'] and rp['rsi_min']<=rsi[i]<=rp['rsi_max']:
                    if roi<0.20: exit_r='REV'; exit_px=cpx
                elif d=='short' and gc and adx[i]>=rp['adx_min'] and rp['rsi_min']<=rsi[i]<=rp['rsi_max']:
                    if roi<0.20: exit_r='REV'; exit_px=cpx

            if exit_r:
                gross = sz*lev*((exit_px/epx-1) if d=='long' else (1-exit_px/epx))*rem
                fee = sz*lev*rem*FEE_RATE*2; pnl = gross-fee; bal += pnl
                monthly[m]['pnl']+=pnl; monthly[m]['trades']+=1
                if exit_r=='SL': monthly[m]['sl']+=1; last_sl_dir=d; last_sl_i=i
                elif exit_r=='TSL': monthly[m]['tsl']+=1
                else: monthly[m]['rev']+=1
                trades.append({'entry_time':etime,'exit_time':pd.Timestamp(ts_arr[i]),
                               'dir':d,'entry_px':epx,'exit_px':exit_px,'roi':(exit_px/epx-1) if d=='long' else (1-exit_px/epx),
                               'pnl':pnl,'reason':exit_r,'peak_roi':pk,'bal_after':bal})
                new_dir = None
                if exit_r=='REV' and rp['rev']=='flip': new_dir = 'short' if d=='long' else 'long'
                pos = None
                if new_dir and bal>0 and regime!='sideways':
                    rr2 = R.get(regime, R['weak'])
                    sz2 = bal*rr2['margin']
                    pos = (new_dir, cpx, pd.Timestamp(ts_arr[i]), sz2, 0.0, cpx, cpx, i, 0, 1.0)
                    pos_regime = regime

        # --- Entry ---
        if pos is None and bal > 0 and regime != 'sideways':
            rp = R[regime]
            gc = fast[i-1]<=slow[i-1] and fast[i]>slow[i]
            dc = fast[i-1]>=slow[i-1] and fast[i]<slow[i]
            sig = None
            if gc: sig='long'
            elif dc: sig='short'
            if sig and adx[i]<rp['adx_min']: sig=None
            if sig and not (rp['rsi_min']<=rsi[i]<=rp['rsi_max']): sig=None
            if sig and sig==last_sl_dir and (i-last_sl_i)<24: sig=None
            if sig and rp['h1']:
                if sig=='long' and not h1_up[i]: sig=None
                if sig=='short' and not h1_dn[i]: sig=None
            if sig:
                sz = bal*rp['margin']
                pos = (sig, cpx, pd.Timestamp(ts_arr[i]), sz, 0.0, cpx, cpx, i, 0, 1.0)
                pos_regime = regime

        monthly[m]['end_bal'] = bal
        if bal>peak_bal: peak_bal=bal
        dd = (peak_bal-bal)/peak_bal if peak_bal>0 else 0
        if dd>mdd: mdd=dd

    if pos is not None:
        d, epx, etime, sz, pk, hi, lo, ei, pd_done, rem = pos
        gross = sz*lev*((closes[-1]/epx-1) if d=='long' else (1-closes[-1]/epx))*rem
        fee = sz*lev*rem*FEE_RATE*2; pnl = gross-fee; bal += pnl
        m2 = pd.Timestamp(ts_arr[-1]).strftime('%Y-%m')
        if m2 in monthly: monthly[m2]['pnl']+=pnl; monthly[m2]['trades']+=1; monthly[m2]['end_bal']=bal
        trades.append({'entry_time':etime,'exit_time':pd.Timestamp(ts_arr[-1]),'dir':d,
                       'entry_px':epx,'exit_px':closes[-1],'roi':(closes[-1]/epx-1) if d=='long' else (1-closes[-1]/epx),
                       'pnl':pnl,'reason':'END','peak_roi':pk,'bal_after':bal})

    nt=len(trades); wins=[t for t in trades if t['pnl']>0]; losses=[t for t in trades if t['pnl']<=0]
    tp=sum(t['pnl'] for t in wins); tl=sum(abs(t['pnl']) for t in losses)
    pf=tp/tl if tl>0 else float('inf')
    wr=len(wins)/nt*100 if nt>0 else 0
    aw=np.mean([t['roi'] for t in wins])*100 if wins else 0
    al_=np.mean([t['roi'] for t in losses])*100 if losses else 0
    rr_=abs(aw/al_) if al_!=0 else float('inf')
    return {'bal':bal,'ret':(bal-capital)/capital*100,'trades':nt,'wins':len(wins),
            'losses':len(losses),'wr':wr,'pf':pf,'mdd':mdd*100,
            'aw':aw,'al':al_,'rr':rr_,
            'sl':sum(1 for t in trades if t['reason']=='SL'),
            'tsl':sum(1 for t in trades if t['reason']=='TSL'),
            'rev':sum(1 for t in trades if t['reason'] in ('REV','TIME_SL','END')),
            'trade_list':trades,'monthly':monthly,'capital':capital}

# ====================================================================
# 5. REPORTING
# ====================================================================
def print_model(res, name):
    cap = res['capital']
    print(f"\n{'='*90}")
    print(f"  {name}")
    print(f"{'='*90}")
    print(f"  초기자본: ${cap:,.0f} → 최종잔액: ${res['bal']:,.0f} ({res['ret']:+,.1f}%)")
    print(f"  PF: {res['pf']:.2f} | MDD: {res['mdd']:.1f}% | 승률: {res['wr']:.1f}% ({res['wins']}W/{res['losses']}L)")
    print(f"  거래: {res['trades']}건 (SL:{res['sl']} TSL:{res['tsl']} REV:{res['rev']})")
    print(f"  평균수익: {res['aw']:+.2f}% | 평균손실: {res['al']:.2f}% | 손익비: {res['rr']:.2f}:1")
    print()
    hdr = f"  {'월':^8} | {'수익%':>8} | {'PnL($)':>11} | {'잔액($)':>12} | {'거래':>3} | {'SL':>2} | {'TSL':>3} | {'REV':>3} | 비고"
    print(hdr)
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*11}-+-{'-'*12}-+-{'-'*3}-+-{'-'*2}-+-{'-'*3}-+-{'-'*3}-+------")
    mo = res['monthly']
    sorted_m = sorted(mo.keys())
    cur_yr = None; yr_start = cap
    for mk in sorted_m:
        md = mo[mk]; yr = mk[:4]
        if yr != cur_yr:
            if cur_yr is not None:
                yr_ret = (prev_end - yr_start)/yr_start*100 if yr_start>0 else 0
                print(f"  {'['+cur_yr+']':^8} | {yr_ret:>+7.1f}% |             | ${prev_end:>11,.0f} |     |    |     |     | 연간합계")
                print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*11}-+-{'-'*12}-+-{'-'*3}-+-{'-'*2}-+-{'-'*3}-+-{'-'*3}-+------")
            cur_yr = yr; yr_start = md['start_bal']
        sb = md['start_bal']; eb = md['end_bal']; pnl = md['pnl']
        pct = (eb-sb)/sb*100 if sb>0 else 0
        note = ''
        if pct >= 30: note = '++LARGE'
        elif pct <= -15: note = '--LOSS'
        elif md['sl']>0: note = f'SL×{md["sl"]}'
        elif md['trades']==0: note = '-'
        print(f"  {mk:^8} | {pct:>+7.1f}% | ${pnl:>+10,.0f} | ${eb:>11,.0f} | {md['trades']:>3} | {md['sl']:>2} | {md['tsl']:>3} | {md['rev']:>3} | {note}")
        prev_end = eb
    if cur_yr:
        yr_ret = (prev_end-yr_start)/yr_start*100 if yr_start>0 else 0
        print(f"  {'['+cur_yr+']':^8} | {yr_ret:>+7.1f}% |             | ${prev_end:>11,.0f} |     |    |     |     | 연간합계")

    # Yearly summary table
    print(f"\n  [연간 요약]")
    print(f"  {'연도':^6} | {'시작잔액':>12} | {'종료잔액':>12} | {'수익률':>8} | {'거래':>4} | {'SL':>3} | {'TSL':>3} | {'REV':>3}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*4}-+-{'-'*3}-+-{'-'*3}-+-{'-'*3}")
    years = sorted(set(mk[:4] for mk in sorted_m))
    for yr in years:
        yr_months = [mk for mk in sorted_m if mk.startswith(yr)]
        yr_sb = mo[yr_months[0]]['start_bal']
        yr_eb = mo[yr_months[-1]]['end_bal']
        yr_ret = (yr_eb-yr_sb)/yr_sb*100 if yr_sb>0 else 0
        yr_tr = sum(mo[m_]['trades'] for m_ in yr_months)
        yr_sl = sum(mo[m_]['sl'] for m_ in yr_months)
        yr_tsl = sum(mo[m_]['tsl'] for m_ in yr_months)
        yr_rev = sum(mo[m_]['rev'] for m_ in yr_months)
        print(f"  {yr:^6} | ${yr_sb:>11,.0f} | ${yr_eb:>11,.0f} | {yr_ret:>+7.1f}% | {yr_tr:>4} | {yr_sl:>3} | {yr_tsl:>3} | {yr_rev:>3}")

def print_composite(ra, rb, rg, total_cap):
    print(f"\n{'='*90}")
    print(f"  v16.4 AMCS 복합 시스템 종합 결과")
    print(f"{'='*90}")
    comp_bal = ra['bal'] + rb['bal'] + rg['bal']
    comp_ret = (comp_bal - total_cap)/total_cap*100
    print(f"  초기자본: ${total_cap:,.0f} → 최종잔액: ${comp_bal:,.0f} ({comp_ret:+,.1f}%)")
    print(f"")
    print(f"  {'모델':^20} | {'자본':>8} | {'최종잔액':>12} | {'수익률':>10} | {'PF':>6} | {'MDD':>6} | {'거래':>4} | {'승률':>6}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*4}-+-{'-'*6}")
    for nm, r in [("A Sniper", ra), ("B Machine Gun", rb), ("G Chameleon", rg)]:
        print(f"  {nm:^20} | ${r['capital']:>7,.0f} | ${r['bal']:>11,.0f} | {r['ret']:>+9.1f}% | {r['pf']:>5.2f} | {r['mdd']:>5.1f}% | {r['trades']:>4} | {r['wr']:>5.1f}%")
    print(f"  {'복합 합계':^20} | ${total_cap:>7,.0f} | ${comp_bal:>11,.0f} | {comp_ret:>+9.1f}% |       |       |      |")

    # Composite monthly
    all_months = sorted(set(list(ra['monthly'].keys()) + list(rb['monthly'].keys()) + list(rg['monthly'].keys())))
    print(f"\n  [복합 월별 상세]")
    print(f"  {'월':^8} | {'Α잔액':>10} | {'Β잔액':>10} | {'Γ잔액':>10} | {'합계잔액':>12} | {'월수익%':>8} | 비고")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}-+------")
    prev_total = total_cap
    cur_yr = None
    for mk in all_months:
        yr = mk[:4]
        a_eb = ra['monthly'].get(mk, {}).get('end_bal', 0)
        b_eb = rb['monthly'].get(mk, {}).get('end_bal', 0)
        g_eb = rg['monthly'].get(mk, {}).get('end_bal', 0)
        # Use last known balance if month not present
        if mk not in ra['monthly']:
            prev_months_a = [m for m in sorted(ra['monthly'].keys()) if m < mk]
            a_eb = ra['monthly'][prev_months_a[-1]]['end_bal'] if prev_months_a else ra['capital']
        if mk not in rb['monthly']:
            prev_months_b = [m for m in sorted(rb['monthly'].keys()) if m < mk]
            b_eb = rb['monthly'][prev_months_b[-1]]['end_bal'] if prev_months_b else rb['capital']
        if mk not in rg['monthly']:
            prev_months_g = [m for m in sorted(rg['monthly'].keys()) if m < mk]
            g_eb = rg['monthly'][prev_months_g[-1]]['end_bal'] if prev_months_g else rg['capital']
        total_eb = a_eb + b_eb + g_eb
        m_ret = (total_eb - prev_total)/prev_total*100 if prev_total>0 else 0

        if yr != cur_yr:
            if cur_yr is not None:
                print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}-+------")
            cur_yr = yr

        note = ''
        if m_ret >= 30: note = '++'
        elif m_ret <= -15: note = '--'
        print(f"  {mk:^8} | ${a_eb:>9,.0f} | ${b_eb:>9,.0f} | ${g_eb:>9,.0f} | ${total_eb:>11,.0f} | {m_ret:>+7.1f}% | {note}")
        prev_total = total_eb

# ====================================================================
# 6. MAIN
# ====================================================================
def main():
    print("="*90)
    print("  BTC/USDT v16.4 AMCS 백테스트 시스템")
    print("  Adaptive Multi-Model Convergence System")
    print("="*90)

    t0 = time.time()
    print("\n[1/4] 5분봉 데이터 로딩...")
    df_5m = load_5m_data()

    print("\n[2/4] 멀티 타임프레임 리샘플링 + 인디케이터 계산...")
    print("  30m 리샘플링...")
    df_30m = resample(df_5m, '30min')
    print(f"    {len(df_30m):,} candles")
    print("  1H 리샘플링...")
    df_1h = resample(df_5m, '1h')
    print(f"    {len(df_1h):,} candles")
    print("  4H 리샘플링...")
    df_4h = resample(df_5m, '4h')
    print(f"    {len(df_4h):,} candles")

    print("  인디케이터 계산 (30m)...")
    df_30m = add_indicators(df_30m)
    print("  인디케이터 계산 (1H)...")
    df_1h = add_indicators(df_1h)
    print("  인디케이터 계산 (4H)...")
    df_4h = add_indicators(df_4h)
    print(f"  데이터 준비 완료: {time.time()-t0:.1f}초")

    TOTAL = 3000

    print("\n[3/4] 백테스트 실행...")

    # ========================
    # STEP 0: v16.0 재현 검증 (엔진 검증용)
    # ========================
    print("  > [검증] v16.0 재현 (WMA3/EMA200, ADX20>=35, SL-8%, TS+4/-3, REV, M50%)...")
    t1 = time.time()
    v16_ref = run_backtest(df_30m, df_1h, df_4h, {
        'fast_ma':'wma3', 'slow_ma':'ema200',
        'adx_col':'adx20', 'adx_min':35,
        'rsi_min':35, 'rsi_max':65,
        'sl_pct':0.08, 'sl_dynamic':False, 'atr_mult':3.0,
        'ts_act':0.04, 'ts_trail':0.03, 'ts_accel':False,
        'partial_exits':[], 'reverse':'flip',
        'time_sl':None,
        'h1_filter':False, 'h4_filter':False,
        'margin':0.50, 'leverage':10,
        're_entry_block':0,
    }, TOTAL)
    print(f"    완료: {time.time()-t1:.1f}초  |  PF:{v16_ref['pf']:.2f} MDD:{v16_ref['mdd']:.1f}% 거래:{v16_ref['trades']} SL:{v16_ref['sl']}")

    # ========================
    # Model A "Sniper" - v16.0 기반 + ADX>=45 + 1H 필터
    # REV 활성화 (v16.0의 0 SL 핵심 메커니즘)
    # ========================
    print("  > Model A 'Sniper' (PF) - ADX>=45, REV ON, M30%...")
    t1 = time.time()
    alpha = run_backtest(df_30m, df_1h, df_4h, {
        'fast_ma':'wma3', 'slow_ma':'ema200',
        'adx_col':'adx20', 'adx_min':45,
        'rsi_min':35, 'rsi_max':65,
        'sl_pct':0.08, 'sl_dynamic':False, 'atr_mult':3.0,
        'ts_act':0.04, 'ts_trail':0.03, 'ts_accel':True,
        'partial_exits':[], 'reverse':'flip',
        'time_sl':None,
        'h1_filter':False, 'h4_filter':False,
        'margin':0.30, 'leverage':10,
        're_entry_block':0,
    }, TOTAL * 0.30)
    print(f"    완료: {time.time()-t1:.1f}초")

    # ========================
    # Model B "Machine Gun" - v14.4 WMA 업그레이드
    # ADX>=35, Trail +6/-5 (v15.5 검증), REV, M35%
    # ========================
    print("  > Model B 'Machine Gun' (Profit) - ADX>=35, TS+6/-5, M35%...")
    t1 = time.time()
    beta = run_backtest(df_30m, df_1h, df_4h, {
        'fast_ma':'wma3', 'slow_ma':'ema200',
        'adx_col':'adx20', 'adx_min':35,
        'rsi_min':35, 'rsi_max':65,
        'sl_pct':0.07, 'sl_dynamic':False, 'atr_mult':3.0,
        'ts_act':0.06, 'ts_trail':0.05, 'ts_accel':False,
        'partial_exits':[],
        'reverse':'flip',
        'time_sl':None,
        'h1_filter':False, 'h4_filter':False,
        'margin':0.35, 'leverage':10,
        're_entry_block':0,
    }, TOTAL * 0.40)
    print(f"    완료: {time.time()-t1:.1f}초")

    # ========================
    # Model G "Chameleon" - 국면 전환 (REV 활성화)
    # ========================
    print("  > Model G 'Chameleon' (Adaptive) - regime switch, REV ON...")
    t1 = time.time()
    gamma = run_gamma(df_30m, df_1h, df_4h, TOTAL * 0.30)
    print(f"    완료: {time.time()-t1:.1f}초")

    print(f"\n  전체 백테스트 완료: {time.time()-t0:.1f}초")

    # ========================
    # 4. Reports
    # ========================
    print("\n[4/4] 리포트 생성...")
    print_model(v16_ref, "[REF] v16.0 Reproduction - $3,000 (100%)")
    print_model(alpha, "Model A 'Sniper' (PF) - $900 (30%)")
    print_model(beta, "Model B 'Machine Gun' (Profit) - $1,200 (40%)")
    print_model(gamma, "Model G 'Chameleon' (Adaptive) - $900 (30%)")
    print_composite(alpha, beta, gamma, TOTAL)

    # Save results
    print(f"\n  결과 저장 중...")
    all_trades = []
    for nm, r in [('Alpha', alpha), ('Beta', beta), ('Gamma', gamma)]:
        for t in r['trade_list']:
            t2 = t.copy(); t2['model'] = nm; all_trades.append(t2)
    df_trades = pd.DataFrame(all_trades)
    df_trades.to_csv(DATA_DIR / 'v164_backtest_trades.csv', index=False)
    print(f"  거래 내역 저장: v164_backtest_trades.csv ({len(df_trades)}건)")

    print("\n" + "="*90)
    print("  v16.4 AMCS 백테스트 완료")
    print("="*90)

if __name__ == '__main__':
    main()
