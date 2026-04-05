"""
ETH V7.3 GPT-5.4 Collaboration Build — CUDA GPU Accelerated (Stage 2)
Runs thousands of backtests simultaneously on GPU (RTX 4060 Ti).
CPU stays free for trading bots.
"""
import sys, os, time, warnings, json
import pandas as pd
import numpy as np
from numba import cuda, njit
import math
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
warnings.filterwarnings('ignore')

# ==================== INDICATORS (pandas, same as original) ====================
def ema(s,p): return s.ewm(span=p,adjust=False).mean()
def wma(s,p):
    w=np.arange(1,p+1,dtype=float)
    return s.rolling(p).apply(lambda x:np.dot(x,w)/w.sum(),raw=True)
def hma(s,p):
    h=max(int(p/2),1);sq=max(int(np.sqrt(p)),1)
    return wma(2*wma(s,h)-wma(s,p),sq)
def dema(s,p): e1=ema(s,p); return 2*e1-ema(e1,p)
def calc_ma(s,mt,p,v=None):
    if mt=='HMA': return hma(s,p)
    if mt=='DEMA': return dema(s,p)
    if mt=='EMA': return ema(s,p)
    return ema(s,p)
def calc_rsi(c,p):
    d=c.diff();g=d.where(d>0,0.0);l=(-d).where(d<0,0.0)
    a=1.0/p; ag=g.ewm(alpha=a,min_periods=p,adjust=False).mean()
    al=l.ewm(alpha=a,min_periods=p,adjust=False).mean()
    return (100-100/(1+ag/al.replace(0,1e-10))).fillna(50)
def calc_adx(h,l,c,p):
    a=1.0/p; tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    up=h-h.shift(1);dn=l.shift(1)-l
    pdm=pd.Series(np.where((up>dn)&(up>0),up,0),index=c.index)
    mdm=pd.Series(np.where((dn>up)&(dn>0),dn,0),index=c.index)
    atr=tr.ewm(alpha=a,min_periods=p,adjust=False).mean()
    pdi=100*pdm.ewm(alpha=a,min_periods=p,adjust=False).mean()/atr
    mdi=100*mdm.ewm(alpha=a,min_periods=p,adjust=False).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,1e-10)
    adx=dx.ewm(alpha=a,min_periods=p,adjust=False).mean()
    return adx.fillna(0),atr.fillna(0)
def calc_stoch(h,l,c,k_p=14,d_p=3):
    lo=l.rolling(k_p).min();hi=h.rolling(k_p).max()
    k=100*(c-lo)/(hi-lo+1e-10);d=k.rolling(d_p).mean()
    return k.fillna(50),d.fillna(50)
def calc_cmf(h,l,c,v,p=20):
    mfv=((c-l)-(h-c))/(h-l+1e-10)*v
    return (mfv.rolling(p).sum()/v.rolling(p).sum()).fillna(0)
def calc_bb_w(c,p=20,std=2.0):
    mid=c.rolling(p).mean();s=c.rolling(p).std()
    return (((mid+std*s)-(mid-std*s))/mid*100).fillna(0)
def calc_cci(h,l,c,p=20):
    tp=(h+l+c)/3;sma_tp=tp.rolling(p).mean()
    mad=tp.rolling(p).apply(lambda x:np.abs(x-x.mean()).mean(),raw=True)
    return ((tp-sma_tp)/(0.015*mad+1e-10)).fillna(0)

# ==================== CUDA DEVICE FUNCTIONS ====================
@cuda.jit(device=True)
def d_sc_adx(v, prev):
    if v >= 40: a = 100.0
    elif v >= 30: a = 80.0
    elif v >= 25: a = 65.0
    elif v >= 20: a = 50.0
    elif v >= 15: a = 30.0
    else: a = 10.0
    r = 70.0 if v > prev else 30.0
    return a * 0.5 + r * 0.5

@cuda.jit(device=True)
def d_sc_rsi(v, d):
    if d == 1:
        if 45 <= v <= 60: return 100.0
        elif 40 <= v <= 65: return 80.0
        elif 35 <= v <= 70: return 60.0
        elif 30 <= v <= 75: return 40.0
        else: return 10.0
    else:
        if 40 <= v <= 55: return 100.0
        elif 35 <= v <= 60: return 80.0
        elif 30 <= v <= 65: return 60.0
        elif 25 <= v <= 70: return 40.0
        else: return 10.0

@cuda.jit(device=True)
def d_sc_vol(v, m):
    r = v / m if m > 0 else 1.0
    if r >= 2: return 100.0
    elif r >= 1.5: return 85.0
    elif r >= 1.2: return 70.0
    elif r >= 1: return 55.0
    elif r >= 0.8: return 40.0
    else: return 20.0

@cuda.jit(device=True)
def d_sc_gap(f, s):
    g = abs(f - s) / s * 100 if s > 0 else 0.0
    if 0.5 <= g <= 3: return 100.0
    elif 0.3 <= g <= 5: return 75.0
    elif 0.2 <= g <= 8: return 50.0
    elif g < 0.2: return 15.0
    else: return 30.0

@cuda.jit(device=True)
def d_sc_stoch(k, d, dr):
    b = 20.0 if (dr == 1 and k > d) or (dr == -1 and k < d) else 0.0
    if (dr == 1 and 25 <= k <= 60) or (dr == -1 and 40 <= k <= 75): base = 80.0
    elif (dr == 1 and 20 <= k <= 70) or (dr == -1 and 30 <= k <= 80): base = 60.0
    else: base = 30.0
    v = base + b
    if v > 100.0: v = 100.0
    return v

@cuda.jit(device=True)
def d_sc_cmf(v, d):
    al = (d == 1 and v > 0) or (d == -1 and v < 0)
    s = abs(v)
    if al:
        if s > 0.15: return 100.0
        elif s > 0.1: return 85.0
        elif s > 0.05: return 70.0
        else: return 55.0
    else:
        if s > 0.15: return 10.0
        elif s > 0.05: return 30.0
        else: return 45.0

@cuda.jit(device=True)
def d_sc_bbw(w, m):
    r = w / m if m > 0 else 1.0
    if r >= 1.5: return 95.0
    elif r >= 1.2: return 80.0
    elif r >= 1: return 65.0
    elif r >= 0.8: return 45.0
    elif r >= 0.6: return 25.0
    else: return 10.0

@cuda.jit(device=True)
def d_sc_atrr(v, m):
    r = v / m if m > 0 else 1.0
    if 1 <= r <= 2: return 90.0
    elif 0.8 <= r <= 2.5: return 70.0
    elif r < 0.8: return 30.0
    else: return 40.0

# ==================== CUDA KERNEL ====================
# params_matrix: (num_jobs, NUM_PARAMS) — each row is one param set
# results_matrix: (num_jobs, NUM_RESULTS) — each row is output
# All market data arrays are 1D shared across all threads (read-only)
# MA arrays: we pass a set of pre-computed MAs and use indices to select which one

NUM_PARAMS = 48
NUM_RESULTS = 15
# Results: [0]=final, [1]=ret, [2]=pf, [3]=mdd, [4]=tt, [5]=wins, [6]=losses,
#          [7]=wr, [8]=sl, [9]=tsl, [10]=rev, [11]=fc, [12]=tp_sum, [13]=tl_sum, [14]=mcl

@cuda.jit
def backtest_kernel(close, high, low, volume, day_vals,
                    pma_all, sma2_all, slma_all,  # (num_ma_combos, n) — pre-selected per job
                    adx, atr, rsi, sk, sd, cmf, bbw, vm, bwm, am, htf_trend,
                    params_matrix, results_matrix, n):
    tid = cuda.grid(1)
    if tid >= params_matrix.shape[0]:
        return

    # Read params for this thread
    p = params_matrix[tid]
    fee = p[0]; warmup_f = p[1]; init_cap = p[2]; dual_mode = p[3]
    monitor_window = p[4]; entry_delay = p[5]; skip_same = p[6]; daily_loss_limit = p[7]
    use_eqs = p[8]; eqs_threshold = p[9]
    w_adx = p[10]; w_rsi = p[11]; w_vol = p[12]; w_gap = p[13]
    w_stoch = p[14]; w_cmf = p[15]; w_bb = p[16]; w_atr = p[17]
    adx_rise = p[18]; sl_pct = p[19]; ta_pct = p[20]; tsl_pct = p[21]
    use_partial = p[22]; partial_trigger = p[23]; partial_pct_v = p[24]
    lev = p[25]
    use_ranging_filter = p[26]; ranging_adx_th = p[27]; ranging_bbw_th = p[28]; ranging_atr_ratio_th = p[29]
    use_mtf_filter = p[30]
    use_reentry = p[31]; reentry_wait = p[32]; reentry_adx_min = p[33]; max_reentries = p[34]
    margin_fixed = p[35]; use_compound = p[36]; base_margin = p[37]; margin_pct = p[38]
    stage2_th = p[39]; stage3_th = p[40]; growth_rate = p[41]; accel_rate = p[42]
    max_margin_pct = p[43]
    dd_th1 = p[44]; dd_sc1 = p[45]; dd_th2 = p[46]; dd_sc2 = p[47]
    # dd_th3, consec_limit, consec_scale packed at end or reuse slots
    # We have exactly 48 params. Let's use a fixed layout.
    # Remap: dd_th3=0.30 fixed, consec_limit=5 fixed, consec_scale=0.50 fixed
    dd_th3 = 0.30; consec_limit_f = 5.0; consec_scale = 0.50

    warmup_i = int(warmup_f)

    cap = init_cap; pos = 0; epx = 0.0; psz = 0.0; slp = 0.0
    ton = 0; partial_done = 0; thi = 0.0; tlo = 999999.0
    watching = 0; ws = 0; le = 0; ld = 0
    pk = cap; mdd_v = 0.0; ms = cap; consec_losses = 0; lev_used = lev; margin_used = 0.0
    sl_c = 0; tsl_c = 0; rev_c = 0; fc_c = 0
    tp_sum = 0.0; tl_sum = 0.0; wins = 0; losses = 0
    max_cl = 0; cur_cl = 0; prev_day = -1
    reentry_count = 0

    for i in range(warmup_i, n):
        px = close[i]; h_ = high[i]; l_ = low[i]
        pma_i = pma_all[tid, i]; slma_i = slma_all[tid, i]; sma2_i = sma2_all[tid, i]

        if math.isnan(pma_i) or math.isnan(slma_i):
            continue
        cd = int(day_vals[i])
        if cd != prev_day:
            ms = cap; prev_day = cd

        if pos != 0:
            watching = 0
            # FC Long
            if pos == 1 and l_ <= epx * (1 - 1.0 / lev_used * 0.98):
                pnl = -margin_used * 0.98; cap += pnl; fc_c += 1
                tl_sum += abs(pnl); losses += 1; cur_cl += 1; consec_losses += 1
                if cur_cl > max_cl: max_cl = cur_cl
                ld = pos; le = i; pos = 0; reentry_count = 0
                if cap > pk: pk = cap
                dd = (pk - cap) / pk if pk > 0 else 0.0
                if dd > mdd_v: mdd_v = dd
                continue
            # FC Short
            if pos == -1 and h_ >= epx * (1 + 1.0 / lev_used * 0.98):
                pnl = -margin_used * 0.98; cap += pnl; fc_c += 1
                tl_sum += abs(pnl); losses += 1; cur_cl += 1; consec_losses += 1
                if cur_cl > max_cl: max_cl = cur_cl
                ld = pos; le = i; pos = 0; reentry_count = 0
                if cap > pk: pk = cap
                dd = (pk - cap) / pk if pk > 0 else 0.0
                if dd > mdd_v: mdd_v = dd
                continue
            # SL
            if ton == 0:
                hit = (pos == 1 and l_ <= slp) or (pos == -1 and h_ >= slp)
                if hit:
                    pnl = (slp - epx) / epx * psz * pos - psz * fee; cap += pnl; sl_c += 1
                    if pnl >= 0:
                        tp_sum += pnl; wins += 1; cur_cl = 0; consec_losses = 0
                    else:
                        tl_sum += abs(pnl); losses += 1; cur_cl += 1; consec_losses += 1
                    if cur_cl > max_cl: max_cl = cur_cl
                    ld = pos; le = i; pos = 0; reentry_count = 0
                    if cap > pk: pk = cap
                    dd = (pk - cap) / pk if pk > 0 else 0.0
                    if dd > mdd_v: mdd_v = dd
                    continue
            # Partial
            if use_partial > 0.5 and partial_done == 0:
                if pos == 1: roi = (h_ - epx) / epx * 100
                else: roi = (epx - l_) / epx * 100
                if roi >= partial_trigger:
                    pp = psz * partial_pct_v
                    ppnl = (px - epx) / epx * pp * pos - pp * fee
                    cap += ppnl; psz -= pp; partial_done = 1
                    if ppnl >= 0: tp_sum += ppnl
                    else: tl_sum += abs(ppnl)
            # Trail activation
            if pos == 1: br = (h_ - epx) / epx * 100
            else: br = (epx - l_) / epx * 100
            if br >= ta_pct: ton = 1
            # TSL
            if ton == 1:
                if pos == 1:
                    if h_ > thi: thi = h_
                    ns = thi * (1 - tsl_pct / 100)
                    if ns > slp: slp = ns
                    if px <= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * fee; cap += pnl; tsl_c += 1
                        tp_sum += pnl; wins += 1; cur_cl = 0; consec_losses = 0
                        ld = pos; le = i; pos = 0; reentry_count = 0
                        if cap > pk: pk = cap
                        dd = (pk - cap) / pk if pk > 0 else 0.0
                        if dd > mdd_v: mdd_v = dd
                        continue
                else:
                    if l_ < tlo: tlo = l_
                    ns = tlo * (1 + tsl_pct / 100)
                    if ns < slp: slp = ns
                    if px >= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * fee; cap += pnl; tsl_c += 1
                        tp_sum += pnl; wins += 1; cur_cl = 0; consec_losses = 0
                        ld = pos; le = i; pos = 0; reentry_count = 0
                        if cap > pk: pk = cap
                        dd = (pk - cap) / pk if pk > 0 else 0.0
                        if dd > mdd_v: mdd_v = dd
                        continue
            # REV
            if i >= 1 and not math.isnan(pma_all[tid, i-1]):
                p_up = pma_i > slma_i and pma_all[tid, i-1] <= slma_all[tid, i-1]
                p_dn = pma_i < slma_i and pma_all[tid, i-1] >= slma_all[tid, i-1]
                if (pos == 1 and p_dn) or (pos == -1 and p_up):
                    pnl = (px - epx) / epx * psz * pos - psz * fee; cap += pnl; rev_c += 1
                    if pnl >= 0:
                        tp_sum += pnl; wins += 1; cur_cl = 0; consec_losses = 0
                    else:
                        tl_sum += abs(pnl); losses += 1; cur_cl += 1; consec_losses += 1
                    if cur_cl > max_cl: max_cl = cur_cl
                    ld = pos; le = i; pos = 0; reentry_count = 0

        # Entry
        if pos == 0 and i >= 1 and not math.isnan(pma_all[tid, i-1]):
            pma_i = pma_all[tid, i]; slma_i = slma_all[tid, i]
            p_up = pma_i > slma_i and pma_all[tid, i-1] <= slma_all[tid, i-1]
            p_dn = pma_i < slma_i and pma_all[tid, i-1] >= slma_all[tid, i-1]

            if dual_mode == 0:  # SINGLE
                c_up = p_up; c_dn = p_dn
            elif not math.isnan(sma2_i) and not math.isnan(sma2_all[tid, i-1]):
                sa_up = sma2_i > slma_i; sa_dn = sma2_i < slma_i
                if dual_mode == 1:  # BOTH
                    c_up = p_up and sa_up; c_dn = p_dn and sa_dn
                else:  # EITHER
                    s_up = sma2_i > slma_i and sma2_all[tid, i-1] <= slma_all[tid, i-1]
                    s_dn = sma2_i < slma_i and sma2_all[tid, i-1] >= slma_all[tid, i-1]
                    c_up = p_up or (s_up and sa_up)
                    c_dn = p_dn or (s_dn and sa_dn)
            else:
                c_up = p_up; c_dn = p_dn

            if c_up: watching = 1; ws = i
            if c_dn: watching = -1; ws = i

            if watching != 0 and i >= ws + int(entry_delay):
                if i - ws > int(monitor_window):
                    watching = 0
                else:
                    can = 1
                    if watching == ld:
                        if use_reentry > 0.5:
                            if i - le < int(reentry_wait) or adx[i] < reentry_adx_min or reentry_count >= int(max_reentries):
                                can = 0
                        elif skip_same > 0.5:
                            can = 0
                    if can == 1 and use_ranging_filter > 0.5:
                        if adx[i] < ranging_adx_th and bbw[i] < ranging_bbw_th:
                            if am[i] > 0 and atr[i] / am[i] < ranging_atr_ratio_th:
                                can = 0
                    if can == 1 and use_mtf_filter > 0.5:
                        if watching == 1 and htf_trend[i] < 0: can = 0
                        if watching == -1 and htf_trend[i] > 0: can = 0
                    if can == 1 and use_eqs > 0.5:
                        ri = i - int(adx_rise)
                        if ri < 0: ri = 0
                        eqs = (d_sc_adx(adx[i], adx[ri]) * w_adx +
                               d_sc_rsi(rsi[i], watching) * w_rsi +
                               d_sc_vol(volume[i], vm[i]) * w_vol +
                               d_sc_gap(pma_i, slma_i) * w_gap +
                               d_sc_stoch(sk[i], sd[i], watching) * w_stoch +
                               d_sc_cmf(cmf[i], watching) * w_cmf +
                               d_sc_bbw(bbw[i], bwm[i]) * w_bb +
                               d_sc_atrr(atr[i], am[i]) * w_atr)
                        if eqs < eqs_threshold: can = 0
                    if can == 1 and ms > 0 and (cap - ms) / ms <= daily_loss_limit:
                        can = 0
                    if can == 1 and cap > 0:
                        if margin_fixed > 0:
                            margin = margin_fixed
                        elif use_compound > 0.5:
                            profit = cap - init_cap
                            if profit < 0: profit = 0.0
                            if cap < stage2_th: margin = base_margin
                            elif cap < stage3_th:
                                margin = base_margin + profit * growth_rate
                                if margin > cap * 0.25: margin = cap * 0.25
                            else:
                                margin = base_margin + profit * accel_rate
                                if margin > cap * max_margin_pct: margin = cap * max_margin_pct
                            dd_now = (pk - cap) / pk if pk > 0 else 0.0
                            if dd_now > dd_th3: margin = base_margin
                            elif dd_now > dd_th2: margin *= dd_sc2
                            elif dd_now > dd_th1: margin *= dd_sc1
                            if consec_losses >= int(consec_limit_f): margin *= consec_scale
                            if margin < 300: margin = 300.0
                            if margin > cap * 0.50: margin = cap * 0.50
                        else:
                            margin = cap * margin_pct
                            if margin < 300: margin = 300.0
                        if margin > cap * 0.95: margin = cap * 0.95
                        if margin >= 100:
                            lev_used = lev; psz = margin * lev_used; margin_used = margin
                            cap -= psz * fee; pos = watching; epx = px
                            ton = 0; partial_done = 0; thi = epx; tlo = epx
                            slp = epx * (1 - sl_pct / 100) if pos == 1 else epx * (1 + sl_pct / 100)
                            if watching == ld: reentry_count += 1
                            else: reentry_count = 0
                            watching = 0

        if cap > pk: pk = cap
        dd = (pk - cap) / pk if pk > 0 else 0.0
        if dd > mdd_v: mdd_v = dd
        if cap <= 0: break

    tt = sl_c + tsl_c + rev_c + fc_c
    pf = tp_sum / tl_sum if tl_sum > 0 else 999.0
    wr = wins / tt * 100.0 if tt > 0 else 0.0
    ret = (cap - init_cap) / init_cap * 100.0

    results_matrix[tid, 0] = cap
    results_matrix[tid, 1] = ret
    results_matrix[tid, 2] = pf
    results_matrix[tid, 3] = mdd_v * 100.0
    results_matrix[tid, 4] = float(tt)
    results_matrix[tid, 5] = float(wins)
    results_matrix[tid, 6] = float(losses)
    results_matrix[tid, 7] = wr
    results_matrix[tid, 8] = float(sl_c)
    results_matrix[tid, 9] = float(tsl_c)
    results_matrix[tid, 10] = float(rev_c)
    results_matrix[tid, 11] = float(fc_c)
    results_matrix[tid, 12] = tp_sum
    results_matrix[tid, 13] = tl_sum
    results_matrix[tid, 14] = float(max_cl)


# ==================== HOST FUNCTIONS ====================

def build_params_matrix(jobs_list):
    """Convert list of param dicts → (N, NUM_PARAMS) float64 array."""
    N = len(jobs_list)
    mat = np.zeros((N, NUM_PARAMS), dtype=np.float64)
    dual_map = {'SINGLE': 0.0, 'BOTH': 1.0, 'EITHER': 2.0}
    for j, p in enumerate(jobs_list):
        mat[j, 0] = p.get('fee_rate', 0.0004)
        mat[j, 1] = float(p.get('warmup', 800))
        mat[j, 2] = p.get('initial_capital', 5000.0)
        mat[j, 3] = dual_map.get(p.get('dual_mode', 'BOTH'), 1.0)
        mat[j, 4] = float(p.get('monitor_window', 6))
        mat[j, 5] = float(p.get('entry_delay', 0))
        mat[j, 6] = 1.0 if p.get('skip_same_dir', True) else 0.0
        mat[j, 7] = p.get('daily_loss_limit', -0.25)
        mat[j, 8] = 1.0 if p.get('use_eqs', True) else 0.0
        mat[j, 9] = float(p.get('eqs_threshold', 45))
        mat[j, 10] = p.get('w_adx', 0.15)
        mat[j, 11] = p.get('w_rsi', 0.10)
        mat[j, 12] = p.get('w_vol', 0.05)
        mat[j, 13] = p.get('w_gap', 0.15)
        mat[j, 14] = p.get('w_stoch', 0.05)
        mat[j, 15] = p.get('w_cmf', 0.05)
        mat[j, 16] = p.get('w_bb', 0.25)
        mat[j, 17] = p.get('w_atr', 0.20)
        mat[j, 18] = float(p.get('adx_rise_bars', 6))
        mat[j, 19] = p.get('sl_pct', 1.0)
        mat[j, 20] = float(p.get('ta_pct', 30))
        mat[j, 21] = float(p.get('tsl_pct', 9))
        mat[j, 22] = 1.0 if p.get('use_partial', False) else 0.0
        mat[j, 23] = p.get('partial_trigger', 15.0)
        mat[j, 24] = p.get('partial_pct', 0.5)
        mat[j, 25] = float(p.get('leverage', 10))
        mat[j, 26] = 1.0 if p.get('use_ranging_filter', True) else 0.0
        mat[j, 27] = float(p.get('ranging_adx_th', 20))
        mat[j, 28] = p.get('ranging_bbw_th', 2.5)
        mat[j, 29] = p.get('ranging_atr_ratio_th', 0.8)
        mat[j, 30] = 1.0 if p.get('use_mtf_filter', True) else 0.0
        mat[j, 31] = 1.0 if p.get('use_reentry', False) else 0.0
        mat[j, 32] = float(p.get('reentry_wait', 5))
        mat[j, 33] = float(p.get('reentry_adx_min', 25))
        mat[j, 34] = float(p.get('max_reentries', 3))
        mat[j, 35] = p.get('margin_fixed', -1.0) if p.get('margin_fixed') is not None else -1.0
        mat[j, 36] = 1.0 if p.get('use_compound', True) else 0.0
        mat[j, 37] = float(p.get('base_margin', 1000))
        mat[j, 38] = p.get('margin_pct', 0.20)
        mat[j, 39] = float(p.get('stage2_th', 15000))
        mat[j, 40] = float(p.get('stage3_th', 75000))
        mat[j, 41] = p.get('growth_rate', 0.15)
        mat[j, 42] = p.get('accel_rate', 0.35)
        mat[j, 43] = p.get('max_margin_pct', 0.40)
        mat[j, 44] = p.get('dd_th1', 0.10)
        mat[j, 45] = p.get('dd_sc1', 0.70)
        mat[j, 46] = p.get('dd_th2', 0.20)
        mat[j, 47] = p.get('dd_sc2', 0.50)
    return mat


def build_ma_arrays(jobs_list, ma_dict, n):
    """Build per-job MA arrays: (num_jobs, n) for pma, sma2, slma."""
    N = len(jobs_list)
    pma_all = np.empty((N, n), dtype=np.float64)
    sma2_all = np.empty((N, n), dtype=np.float64)
    slma_all = np.empty((N, n), dtype=np.float64)
    for j, p in enumerate(jobs_list):
        pma_all[j] = ma_dict[p.get('primary_ma_key', 'DEMA_125')]
        sma2_key = p.get('secondary_ma_key', 'HMA_125')
        sma2_all[j] = ma_dict.get(sma2_key, ma_dict[p.get('primary_ma_key', 'DEMA_125')])
        slma_all[j] = ma_dict[p.get('slow_ma_key', 'EMA_800')]
    return pma_all, sma2_all, slma_all


def run_cuda_batch(jobs_batch, ma_dict, close, high, low, volume, day_vals,
                   adx, atr, rsi, sk, sd, cmf, bbw, vm, bwm, am, htf_trend):
    """Run a batch of jobs on GPU."""
    N = len(jobs_batch)
    n = len(close)

    # Build matrices
    params_mat = build_params_matrix(jobs_batch)
    pma_all, sma2_all, slma_all = build_ma_arrays(jobs_batch, ma_dict, n)

    # Transfer to GPU
    d_close = cuda.to_device(close)
    d_high = cuda.to_device(high)
    d_low = cuda.to_device(low)
    d_volume = cuda.to_device(volume)
    d_day_vals = cuda.to_device(day_vals)
    d_pma = cuda.to_device(pma_all)
    d_sma2 = cuda.to_device(sma2_all)
    d_slma = cuda.to_device(slma_all)
    d_adx = cuda.to_device(adx)
    d_atr = cuda.to_device(atr)
    d_rsi = cuda.to_device(rsi)
    d_sk = cuda.to_device(sk)
    d_sd = cuda.to_device(sd)
    d_cmf = cuda.to_device(cmf)
    d_bbw = cuda.to_device(bbw)
    d_vm = cuda.to_device(vm)
    d_bwm = cuda.to_device(bwm)
    d_am = cuda.to_device(am)
    d_htf = cuda.to_device(htf_trend)
    d_params = cuda.to_device(params_mat)
    d_results = cuda.device_array((N, NUM_RESULTS), dtype=np.float64)

    # Launch kernel
    threads_per_block = 128
    blocks = (N + threads_per_block - 1) // threads_per_block
    backtest_kernel[blocks, threads_per_block](
        d_close, d_high, d_low, d_volume, d_day_vals,
        d_pma, d_sma2, d_slma,
        d_adx, d_atr, d_rsi, d_sk, d_sd, d_cmf, d_bbw, d_vm, d_bwm, d_am, d_htf,
        d_params, d_results, n
    )
    cuda.synchronize()

    # Copy results back
    results_mat = d_results.copy_to_host()
    return results_mat


def results_to_dicts(results_mat, jobs_list):
    """Convert results matrix to list of dicts (same format as original)."""
    out = []
    for j in range(len(jobs_list)):
        r = results_mat[j]
        tt = int(r[4]); wins = int(r[5]); losses = int(r[6])
        if tt < 15: continue
        pf = r[2]; mdd_v = r[3]; ret = r[1]
        # Score
        if pf <= 0 or mdd_v <= 0 or tt < 20:
            score = 0.0
        else:
            lr = np.log10(max(ret, 1.0))
            score = (pf ** 1.5) * lr * 10.0 / max(mdd_v, 1.0)
            score *= min(tt / 100.0, 2.0)
        out.append({
            'final': r[0], 'ret': ret, 'pf': pf, 'mdd': mdd_v,
            'tt': tt, 'wins': wins, 'losses': losses, 'wr': r[7],
            'sl': int(r[8]), 'tsl': int(r[9]), 'rev': int(r[10]), 'fc': int(r[11]),
            'tp': r[12], 'tl': r[13],
            'aw': r[12]/wins if wins > 0 else 0,
            'al': r[13]/losses if losses > 0 else 0,
            'mcl': int(r[14]),
            'score': score,
            'params': jobs_list[j]
        })
    return out


def main():
    print("=" * 90)
    print("  ETH V7.3 GPT-5.4 Collaboration — CUDA GPU Accelerated")
    print("=" * 90)

    # Check GPU
    if not cuda.is_available():
        print("  ERROR: CUDA not available! Install numba with CUDA support.")
        return
    gpu = cuda.get_current_device()
    print(f"  GPU: {gpu.name}")
    mem = cuda.current_context().get_memory_info()
    print(f"  VRAM: {mem[1]/1024**3:.1f}GB total, {mem[0]/1024**3:.1f}GB free")

    print("\n[1] Loading...", flush=True)
    df5 = pd.read_csv('eth_usdt_5m_merged.csv', parse_dates=['timestamp'])
    df5 = df5.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    agg = {'open':'first','high':'max','low':'min','close':'last','volume':'sum','quote_volume':'sum','trades':'sum'}
    df30 = df5.set_index('timestamp').resample('30min').agg(agg).dropna().reset_index()
    df1h = df5.set_index('timestamp').resample('1h').agg(agg).dropna().reset_index()
    print(f"  30m:{len(df30)} 1h:{len(df1h)}", flush=True)

    print("\n[2] Indicators...", flush=True)
    t0 = time.time()
    cs = df30['close']; hs = df30['high']; ls = df30['low']; vs = df30['volume']
    ma = {}
    for mt in ['HMA','DEMA','EMA']:
        for p_ in [50,75,100,125,150,200,400,500,600,700,800]:
            ma[f"{mt}_{p_}"] = calc_ma(cs, mt, p_, vs).values.astype(np.float64)
    ind = {}
    for p_ in [7, 14]: ind[f"RSI_{p_}"] = calc_rsi(cs, p_).values.astype(np.float64)
    adx_v, atr_v = calc_adx(hs, ls, cs, 20)
    ind['ADX'] = adx_v.values.astype(np.float64); ind['ATR'] = atr_v.values.astype(np.float64)
    sk, sd = calc_stoch(hs, ls, cs)
    ind['SK'] = sk.values.astype(np.float64); ind['SD'] = sd.values.astype(np.float64)
    ind['CMF'] = calc_cmf(hs, ls, cs, vs, 20).values.astype(np.float64)
    ind['BBW'] = calc_bb_w(cs, 20, 2.0).values.astype(np.float64)
    ind['VM'] = vs.rolling(20).mean().fillna(vs.iloc[0]).values.astype(np.float64)
    ind['BWM'] = pd.Series(ind['BBW']).rolling(50).mean().fillna(ind['BBW'][0]).values.astype(np.float64)
    ind['AM'] = pd.Series(ind['ATR']).rolling(50).mean().fillna(ind['ATR'][0]).values.astype(np.float64)

    # HTF
    htf_ema = ema(df1h['close'], 200).values; htf_cl = df1h['close'].values; htf_ts = df1h['timestamp'].values
    htf_trend = np.zeros(len(df30), dtype=np.float64); j = 0
    for i in range(len(df30)):
        ts30 = df30['timestamp'].iloc[i]
        while j < len(df1h) - 1 and htf_ts[j+1] <= ts30: j += 1
        if j < len(htf_ema):
            htf_trend[i] = 1.0 if htf_cl[j] > htf_ema[j] else -1.0

    ts_ = pd.DatetimeIndex(df30['timestamp'])
    dv_ = ts_.day.values.astype(np.float64)
    n = len(cs)
    close_arr = cs.values.astype(np.float64)
    high_arr = hs.values.astype(np.float64)
    low_arr = ls.values.astype(np.float64)
    vol_arr = vs.values.astype(np.float64)
    print(f"  Done {time.time()-t0:.0f}s", flush=True)

    # Base params
    base = {'initial_capital':5000.0,'fee_rate':0.0004,'warmup':800,'leverage':10,
        'daily_loss_limit':-0.25,'adx_rise_bars':6,'rsi_period':7,
        'use_eqs':True,'eqs_threshold':45,
        'w_adx':.15,'w_rsi':.10,'w_vol':.05,'w_gap':.15,
        'w_stoch':.05,'w_cmf':.05,'w_bb':.25,'w_atr':.20,
        'use_partial':False,'use_mtf_filter':True,
        'use_ranging_filter':True,'ranging_adx_th':20,'ranging_bbw_th':2.5,'ranging_atr_ratio_th':0.8}

    # Build jobs (same as original)
    print("\n[3] Building jobs...", flush=True)
    jobs = []
    for fma in ['DEMA_100','DEMA_125','DEMA_150','HMA_100','HMA_125']:
        for slma_ in ['EMA_600','EMA_700','EMA_800']:
            for dm in ['SINGLE','BOTH']:
                for sl in [1.0, 1.5, 2.0]:
                    for ta in [20, 25, 30]:
                        for tsl in [5, 7, 9]:
                            for mon in [6, 12]:
                                for delay in [0, 4]:
                                    stype = 'HMA' if 'DEMA' in fma else 'DEMA'
                                    sper = fma.split('_')[1]
                                    for sizing_label, sz in [
                                        ('pct15',{'margin_fixed':None,'use_compound':False,'margin_pct':0.15}),
                                        ('pct20',{'margin_fixed':None,'use_compound':False,'margin_pct':0.20}),
                                        ('pct25',{'margin_fixed':None,'use_compound':False,'margin_pct':0.25}),
                                        ('antidd',{'margin_fixed':None,'use_compound':True,'base_margin':1000,
                                            'stage2_th':15000,'stage3_th':75000,'growth_rate':0.15,'accel_rate':0.35,
                                            'max_margin_pct':0.40,'dd_th1':0.10,'dd_sc1':0.70,'dd_th2':0.20,
                                            'dd_sc2':0.50,'dd_th3':0.30,'consec_limit':5,'consec_scale':0.50}),
                                        ('fixed1k',{'margin_fixed':1000.0,'use_compound':False}),
                                    ]:
                                        for rf in [True, False]:
                                            for mtf in [True, False]:
                                                for eqs_on in [True, False]:
                                                    for reentry in [False]:
                                                        p = dict(base)
                                                        p['primary_ma_key'] = fma
                                                        p['secondary_ma_key'] = f"{stype}_{sper}"
                                                        p['slow_ma_key'] = slma_
                                                        p['dual_mode'] = dm
                                                        p['sl_pct'] = sl; p['ta_pct'] = ta; p['tsl_pct'] = tsl
                                                        p['monitor_window'] = mon; p['entry_delay'] = delay
                                                        p['skip_same_dir'] = True
                                                        p.update(sz)
                                                        p['use_ranging_filter'] = rf
                                                        p['use_mtf_filter'] = mtf
                                                        p['use_eqs'] = eqs_on
                                                        p['use_reentry'] = reentry
                                                        jobs.append(p)
    # Deduplicate
    jobs_filtered = []; seen = set()
    for p in jobs:
        key = (p['primary_ma_key'], p['slow_ma_key'], p['dual_mode'],
               p['sl_pct'], p['ta_pct'], p['tsl_pct'], p['monitor_window'], p['entry_delay'],
               str(p.get('margin_fixed')), str(p.get('use_compound')), str(p.get('margin_pct', '')),
               p['use_ranging_filter'], p['use_mtf_filter'], p['use_eqs'])
        if key not in seen:
            seen.add(key); jobs_filtered.append(p)
    jobs = jobs_filtered
    print(f"  Total unique combos: {len(jobs)}", flush=True)

    # CUDA batched execution
    # VRAM limit: each job needs ~n*8*3 bytes for MA arrays (pma, sma2, slma)
    # n=109330, 3 arrays × 8 bytes = ~2.5MB per job
    # With 8GB VRAM, safe batch = ~2000 jobs (using ~5GB for MAs + overhead)
    BATCH_SIZE = 1500  # conservative for 8GB GPU
    print(f"\n[4] Running CUDA batches (batch_size={BATCH_SIZE})...", flush=True)
    t0 = time.time()
    all_results = []
    total_batches = (len(jobs) + BATCH_SIZE - 1) // BATCH_SIZE

    for b in range(total_batches):
        start = b * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(jobs))
        batch = jobs[start:end]

        bt0 = time.time()
        results_mat = run_cuda_batch(
            batch, ma, close_arr, high_arr, low_arr, vol_arr, dv_,
            ind['ADX'], ind['ATR'], ind[f"RSI_{base['rsi_period']}"],
            ind['SK'], ind['SD'], ind['CMF'], ind['BBW'],
            ind['VM'], ind['BWM'], ind['AM'], htf_trend
        )
        batch_results = results_to_dicts(results_mat, batch)
        all_results.extend(batch_results)
        bt = time.time() - bt0
        print(f"    Batch {b+1}/{total_batches}: {len(batch)} jobs in {bt:.1f}s ({len(batch)/bt:.0f} jobs/s) pass={len(batch_results)}", flush=True)

    total_time = time.time() - t0
    all_results.sort(key=lambda x: x['score'], reverse=True)
    print(f"  Done: {len(all_results)} pass in {total_time:.0f}s ({len(jobs)/total_time:.0f} jobs/s)", flush=True)

    # TOP results
    print(f"\n{'='*90}\n  TOP 10 BY SCORE\n{'='*90}")
    print(f"  {'#':>3} {'Balance':>14} {'Return':>10} {'PF':>6} {'MDD':>6} {'T':>5} {'W':>4} {'L':>4} {'SL':>4} {'TSL':>4} {'Score':>8}")
    for i, r in enumerate(all_results[:10]):
        print(f"  {i+1:>3} ${r['final']:>13,.0f} {r['ret']:>9,.0f}% {r['pf']:>5.2f} {r['mdd']:>5.1f}% {r['tt']:>5} {r['wins']:>4} {r['losses']:>4} {r['sl']:>4} {r['tsl']:>4} {r['score']:>8.1f}")

    by_ret = [r for r in all_results if r['mdd'] <= 30]
    by_ret.sort(key=lambda x: x['ret'], reverse=True)
    print(f"\n  TOP 10 BY RETURN (MDD<=30%)")
    for i, r in enumerate(by_ret[:10]):
        print(f"  {i+1:>3} ${r['final']:>13,.0f} {r['ret']:>9,.0f}% {r['pf']:>5.2f} {r['mdd']:>5.1f}% {r['tt']:>5} {r['wins']:>4} {r['losses']:>4}")

    best = by_ret[0] if by_ret else all_results[0] if all_results else None
    if best:
        bp = best['params']
        print(f"\n  BEST: ${best['final']:,.0f} | PF: {best['pf']:.2f} | MDD: {best['mdd']:.1f}% | T: {best['tt']}")
        print(f"\n  Key params:")
        for k in ['primary_ma_key','secondary_ma_key','slow_ma_key','dual_mode','sl_pct','ta_pct','tsl_pct',
                  'monitor_window','entry_delay','use_eqs','use_ranging_filter','use_mtf_filter',
                  'use_compound','margin_pct','margin_fixed','accel_rate','growth_rate']:
            if k in bp: print(f"    {k:30s} = {bp[k]}")

        with open('v7_3_cuda_results.json', 'w') as f:
            json.dump({'params': bp, 'final': best['final'], 'ret': best['ret'],
                       'pf': best['pf'], 'mdd': best['mdd'], 'trades': best['tt']}, f, indent=2, default=str)
        print(f"\n  Saved to v7_3_cuda_results.json")

    print(f"\n{'='*90}\n  V7.3 GPT Collab (CUDA GPU) Complete!\n{'='*90}")


if __name__ == '__main__':
    main()
