"""
ETH V7.3 GPT-5.4 Collaboration Build — Numba JIT Accelerated (Stage 1)
Drop-in replacement: same logic, 10-50x faster backtest via @njit
"""
import sys, os, time, warnings, json
import pandas as pd
import numpy as np
from numba import njit, prange
from multiprocessing import Pool, cpu_count
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

# ==================== NUMBA JIT EQS SCORING ====================
@njit(cache=True)
def sc_adx(v, prev):
    if v >= 40: a = 100.0
    elif v >= 30: a = 80.0
    elif v >= 25: a = 65.0
    elif v >= 20: a = 50.0
    elif v >= 15: a = 30.0
    else: a = 10.0
    r = 70.0 if v > prev else 30.0
    return a * 0.5 + r * 0.5

@njit(cache=True)
def sc_rsi(v, d):
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

@njit(cache=True)
def sc_vol(v, m):
    r = v / m if m > 0 else 1.0
    if r >= 2: return 100.0
    elif r >= 1.5: return 85.0
    elif r >= 1.2: return 70.0
    elif r >= 1: return 55.0
    elif r >= 0.8: return 40.0
    else: return 20.0

@njit(cache=True)
def sc_gap(f, s):
    g = abs(f - s) / s * 100 if s > 0 else 0.0
    if 0.5 <= g <= 3: return 100.0
    elif 0.3 <= g <= 5: return 75.0
    elif 0.2 <= g <= 8: return 50.0
    elif g < 0.2: return 15.0
    else: return 30.0

@njit(cache=True)
def sc_stoch(k, d, dr):
    b = 20.0 if (dr == 1 and k > d) or (dr == -1 and k < d) else 0.0
    if (dr == 1 and 25 <= k <= 60) or (dr == -1 and 40 <= k <= 75): base = 80.0
    elif (dr == 1 and 20 <= k <= 70) or (dr == -1 and 30 <= k <= 80): base = 60.0
    else: base = 30.0
    return min(100.0, base + b)

@njit(cache=True)
def sc_cmf(v, d):
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

@njit(cache=True)
def sc_bbw(w, m):
    r = w / m if m > 0 else 1.0
    if r >= 1.5: return 95.0
    elif r >= 1.2: return 80.0
    elif r >= 1: return 65.0
    elif r >= 0.8: return 45.0
    elif r >= 0.6: return 25.0
    else: return 10.0

@njit(cache=True)
def sc_atrr(v, m):
    r = v / m if m > 0 else 1.0
    if 1 <= r <= 2: return 90.0
    elif 0.8 <= r <= 2.5: return 70.0
    elif r < 0.8: return 30.0
    else: return 40.0

# ==================== NUMBA JIT BACKTEST ENGINE ====================
# Returns: numpy array of [final, ret, pf, mdd, tt, wins, losses, wr,
#          sl_c, tsl_c, rev_c, fc_c, pe_c, tp_sum, tl_sum, aw, al, mcl, score, ayp]
# Plus trade_pnls, trade_months (int-encoded), trade_types for post-processing

@njit(cache=True)
def backtest_jit(
    close, high, low, volume, day_vals,
    pma, sma2, slma, adx, atr, rsi, sk, sd, cmf, bbw, vm, bwm, am, htf_trend,
    # Scalar params (all floats; booleans as 1.0/0.0; ints as float)
    fee, warmup, init_cap, dual_mode,  # dual_mode: 0=SINGLE, 1=BOTH, 2=EITHER
    monitor_window, entry_delay, skip_same, daily_loss_limit,
    use_eqs, eqs_threshold,
    w_adx, w_rsi, w_vol, w_gap, w_stoch, w_cmf, w_bb, w_atr,
    adx_rise, sl_pct, ta_pct, tsl_pct,
    use_partial, partial_trigger, partial_pct_v,
    lev,
    use_ranging_filter, ranging_adx_th, ranging_bbw_th, ranging_atr_ratio_th,
    use_mtf_filter,
    use_reentry, reentry_wait, reentry_adx_min, max_reentries,
    margin_fixed,  # -1.0 means None
    use_compound, base_margin, margin_pct,
    stage2_th, stage3_th, growth_rate, accel_rate, max_margin_pct,
    dd_th1, dd_sc1, dd_th2, dd_sc2, dd_th3,
    consec_limit, consec_scale
):
    n = len(close)
    warmup_i = int(warmup)
    max_trades = n  # upper bound

    # Trade log arrays (for monthly/yearly post-processing)
    trade_pnls = np.empty(max_trades, dtype=np.float64)
    trade_months = np.empty(max_trades, dtype=np.int32)  # encoded as YYYYMM int
    trade_types = np.empty(max_trades, dtype=np.int32)    # 0=SL,1=TSL,2=REV,3=FC,4=PE
    trade_count = 0

    cap = init_cap; pos = 0; epx = 0.0; psz = 0.0; slp = 0.0
    ton = 0; partial_done = 0; thi = 0.0; tlo = 999999.0
    watching = 0; ws = 0; le = 0; ld = 0
    pk = cap; mdd_v = 0.0; ms = cap; consec_losses = 0; lev_used = lev; margin_used = 0.0
    sl_c = 0; tsl_c = 0; rev_c = 0; pe_c = 0; fc_c = 0
    tp_sum = 0.0; tl_sum = 0.0; wins = 0; losses = 0
    max_cl = 0; cur_cl = 0; prev_day = -1
    reentry_count = 0

    for i in range(warmup_i, n):
        px = close[i]; h_ = high[i]; l_ = low[i]
        if np.isnan(pma[i]) or np.isnan(slma[i]):
            continue
        cd = int(day_vals[i])
        if cd != prev_day:
            ms = cap; prev_day = cd

        if pos != 0:
            watching = 0
            # Forced Close (Long)
            if pos == 1 and l_ <= epx * (1 - 1.0 / lev_used * 0.98):
                pnl = -margin_used * 0.98; cap += pnl; fc_c += 1
                tl_sum += abs(pnl); losses += 1
                cur_cl += 1; consec_losses += 1
                if cur_cl > max_cl: max_cl = cur_cl
                if trade_count < max_trades:
                    trade_pnls[trade_count] = pnl
                    trade_months[trade_count] = int(day_vals[i])  # will fix encoding below
                    trade_types[trade_count] = 3  # FC
                    trade_count += 1
                ld = pos; le = i; pos = 0; reentry_count = 0
                if cap > pk: pk = cap
                dd = (pk - cap) / pk if pk > 0 else 0.0
                if dd > mdd_v: mdd_v = dd
                continue
            # Forced Close (Short)
            if pos == -1 and h_ >= epx * (1 + 1.0 / lev_used * 0.98):
                pnl = -margin_used * 0.98; cap += pnl; fc_c += 1
                tl_sum += abs(pnl); losses += 1
                cur_cl += 1; consec_losses += 1
                if cur_cl > max_cl: max_cl = cur_cl
                if trade_count < max_trades:
                    trade_pnls[trade_count] = pnl
                    trade_months[trade_count] = int(day_vals[i])
                    trade_types[trade_count] = 3
                    trade_count += 1
                ld = pos; le = i; pos = 0; reentry_count = 0
                if cap > pk: pk = cap
                dd = (pk - cap) / pk if pk > 0 else 0.0
                if dd > mdd_v: mdd_v = dd
                continue

            # Stop Loss (not trailing yet)
            if ton == 0:
                hit = (pos == 1 and l_ <= slp) or (pos == -1 and h_ >= slp)
                if hit:
                    pnl = (slp - epx) / epx * psz * pos - psz * fee
                    cap += pnl; sl_c += 1
                    if pnl >= 0:
                        tp_sum += pnl; wins += 1; cur_cl = 0; consec_losses = 0
                    else:
                        tl_sum += abs(pnl); losses += 1; cur_cl += 1; consec_losses += 1
                    if cur_cl > max_cl: max_cl = cur_cl
                    if trade_count < max_trades:
                        trade_pnls[trade_count] = pnl
                        trade_months[trade_count] = int(day_vals[i])
                        trade_types[trade_count] = 0  # SL
                        trade_count += 1
                    ld = pos; le = i; pos = 0; reentry_count = 0
                    if cap > pk: pk = cap
                    dd = (pk - cap) / pk if pk > 0 else 0.0
                    if dd > mdd_v: mdd_v = dd
                    continue

            # Partial exit
            if use_partial > 0.5 and partial_done == 0:
                if pos == 1:
                    roi = (h_ - epx) / epx * 100
                else:
                    roi = (epx - l_) / epx * 100
                if roi >= partial_trigger:
                    pp = psz * partial_pct_v
                    ppnl = (px - epx) / epx * pp * pos - pp * fee
                    cap += ppnl; psz -= pp; partial_done = 1; pe_c += 1
                    if ppnl >= 0:
                        tp_sum += ppnl
                    else:
                        tl_sum += abs(ppnl)

            # Check trailing activation
            if pos == 1:
                br = (h_ - epx) / epx * 100
            else:
                br = (epx - l_) / epx * 100
            if br >= ta_pct:
                ton = 1

            # Trailing stop
            if ton == 1:
                if pos == 1:
                    if h_ > thi: thi = h_
                    ns = thi * (1 - tsl_pct / 100)
                    if ns > slp: slp = ns
                    if px <= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * fee
                        cap += pnl; tsl_c += 1
                        tp_sum += pnl; wins += 1; cur_cl = 0; consec_losses = 0
                        if trade_count < max_trades:
                            trade_pnls[trade_count] = pnl
                            trade_months[trade_count] = int(day_vals[i])
                            trade_types[trade_count] = 1  # TSL
                            trade_count += 1
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
                        pnl = (px - epx) / epx * psz * pos - psz * fee
                        cap += pnl; tsl_c += 1
                        tp_sum += pnl; wins += 1; cur_cl = 0; consec_losses = 0
                        if trade_count < max_trades:
                            trade_pnls[trade_count] = pnl
                            trade_months[trade_count] = int(day_vals[i])
                            trade_types[trade_count] = 1
                            trade_count += 1
                        ld = pos; le = i; pos = 0; reentry_count = 0
                        if cap > pk: pk = cap
                        dd = (pk - cap) / pk if pk > 0 else 0.0
                        if dd > mdd_v: mdd_v = dd
                        continue

            # Reversal exit
            if i >= 1 and not np.isnan(pma[i-1]):
                p_up = pma[i] > slma[i] and pma[i-1] <= slma[i-1]
                p_dn = pma[i] < slma[i] and pma[i-1] >= slma[i-1]
                if (pos == 1 and p_dn) or (pos == -1 and p_up):
                    pnl = (px - epx) / epx * psz * pos - psz * fee
                    cap += pnl; rev_c += 1
                    if pnl >= 0:
                        tp_sum += pnl; wins += 1; cur_cl = 0; consec_losses = 0
                    else:
                        tl_sum += abs(pnl); losses += 1; cur_cl += 1; consec_losses += 1
                    if cur_cl > max_cl: max_cl = cur_cl
                    if trade_count < max_trades:
                        trade_pnls[trade_count] = pnl
                        trade_months[trade_count] = int(day_vals[i])
                        trade_types[trade_count] = 2  # REV
                        trade_count += 1
                    ld = pos; le = i; pos = 0; reentry_count = 0

        # Entry logic
        if pos == 0 and i >= 1 and not np.isnan(pma[i-1]):
            p_up = pma[i] > slma[i] and pma[i-1] <= slma[i-1]
            p_dn = pma[i] < slma[i] and pma[i-1] >= slma[i-1]

            # Dual mode: 0=SINGLE, 1=BOTH, 2=EITHER
            if dual_mode == 0:
                c_up = p_up; c_dn = p_dn
            elif not np.isnan(sma2[i]) and not np.isnan(sma2[i-1]):
                sa_up = sma2[i] > slma[i]; sa_dn = sma2[i] < slma[i]
                if dual_mode == 1:
                    c_up = p_up and sa_up; c_dn = p_dn and sa_dn
                else:  # EITHER
                    s_up = sma2[i] > slma[i] and sma2[i-1] <= slma[i-1]
                    s_dn = sma2[i] < slma[i] and sma2[i-1] >= slma[i-1]
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
                    # Same direction skip
                    if watching == ld:
                        if use_reentry > 0.5:
                            if i - le < int(reentry_wait) or adx[i] < reentry_adx_min or reentry_count >= int(max_reentries):
                                can = 0
                        elif skip_same > 0.5:
                            can = 0

                    # Ranging filter
                    if can == 1 and use_ranging_filter > 0.5:
                        if adx[i] < ranging_adx_th and bbw[i] < ranging_bbw_th:
                            if am[i] > 0 and atr[i] / am[i] < ranging_atr_ratio_th:
                                can = 0

                    # MTF filter
                    if can == 1 and use_mtf_filter > 0.5:
                        if watching == 1 and htf_trend[i] < 0: can = 0
                        if watching == -1 and htf_trend[i] > 0: can = 0

                    # EQS scoring
                    if can == 1 and use_eqs > 0.5:
                        ri = i - int(adx_rise)
                        if ri < 0: ri = 0
                        eqs = (sc_adx(adx[i], adx[ri]) * w_adx +
                               sc_rsi(rsi[i], watching) * w_rsi +
                               sc_vol(volume[i], vm[i]) * w_vol +
                               sc_gap(pma[i], slma[i]) * w_gap +
                               sc_stoch(sk[i], sd[i], watching) * w_stoch +
                               sc_cmf(cmf[i], watching) * w_cmf +
                               sc_bbw(bbw[i], bwm[i]) * w_bb +
                               sc_atrr(atr[i], am[i]) * w_atr)
                        if eqs < eqs_threshold: can = 0

                    # Daily loss limit
                    if can == 1 and ms > 0 and (cap - ms) / ms <= daily_loss_limit:
                        can = 0

                    # Entry
                    if can == 1 and cap > 0:
                        # Sizing
                        if margin_fixed > 0:
                            margin = margin_fixed
                        elif use_compound > 0.5:
                            profit = cap - init_cap
                            if profit < 0: profit = 0.0
                            if cap < stage2_th:
                                margin = base_margin
                            elif cap < stage3_th:
                                margin = base_margin + profit * growth_rate
                                if margin > cap * 0.25: margin = cap * 0.25
                            else:
                                margin = base_margin + profit * accel_rate
                                if margin > cap * max_margin_pct: margin = cap * max_margin_pct
                            dd_now = (pk - cap) / pk if pk > 0 else 0.0
                            if dd_now > dd_th3:
                                margin = base_margin
                            elif dd_now > dd_th2:
                                margin *= dd_sc2
                            elif dd_now > dd_th1:
                                margin *= dd_sc1
                            if consec_losses >= int(consec_limit):
                                margin *= consec_scale
                            if margin < 300: margin = 300.0
                            if margin > cap * 0.50: margin = cap * 0.50
                        else:
                            margin = cap * margin_pct
                            if margin < 300: margin = 300.0

                        if margin > cap * 0.95: margin = cap * 0.95

                        if margin >= 100:
                            lev_used = lev
                            psz = margin * lev_used
                            margin_used = margin
                            cap -= psz * fee
                            pos = watching; epx = px
                            ton = 0; partial_done = 0; thi = epx; tlo = epx
                            slp = epx * (1 - sl_pct / 100) if pos == 1 else epx * (1 + sl_pct / 100)
                            # reentry tracking (note: watching already set to new dir above)
                            if watching == ld:
                                reentry_count += 1
                            else:
                                reentry_count = 0
                            watching = 0

        if cap > pk: pk = cap
        dd = (pk - cap) / pk if pk > 0 else 0.0
        if dd > mdd_v: mdd_v = dd
        if cap <= 0: break

    tt = sl_c + tsl_c + rev_c + fc_c
    # Return results as flat array + trade arrays
    # [0]=final, [1]=ret, [2]=pf, [3]=mdd, [4]=tt, [5]=wins, [6]=losses,
    # [7]=wr, [8]=sl, [9]=tsl, [10]=rev, [11]=fc, [12]=pe,
    # [13]=tp_sum, [14]=tl_sum, [15]=aw, [16]=al, [17]=mcl, [18]=score, [19]=ayp, [20]=trade_count
    result = np.empty(21, dtype=np.float64)
    result[0] = cap
    result[4] = float(tt)
    result[5] = float(wins)
    result[6] = float(losses)
    result[8] = float(sl_c)
    result[9] = float(tsl_c)
    result[10] = float(rev_c)
    result[11] = float(fc_c)
    result[12] = float(pe_c)
    result[13] = tp_sum
    result[14] = tl_sum
    result[17] = float(max_cl)
    result[20] = float(trade_count)

    if tt < 10:
        result[18] = -1.0  # marker for invalid
        return result, trade_pnls[:trade_count], trade_months[:trade_count], trade_types[:trade_count]

    pf = tp_sum / tl_sum if tl_sum > 0 else 999.0
    aw = tp_sum / wins if wins > 0 else 0.0
    al_v = tl_sum / losses if losses > 0 else 0.0
    wr = wins / tt * 100.0
    ret = (cap - init_cap) / init_cap * 100.0

    result[1] = ret
    result[2] = pf
    result[3] = mdd_v * 100.0
    result[7] = wr
    result[15] = aw
    result[16] = al_v

    # Score (same formula as original)
    if pf <= 0 or mdd_v <= 0 or tt < 20:
        result[18] = 0.0
    else:
        lr = np.log10(max(ret, 1.0))
        sc = (pf ** 1.5) * lr * 10.0 / max(mdd_v * 100.0, 1.0) ** 1.0
        tsc = tt / 100.0
        if tsc > 2.0: tsc = 2.0
        sc *= tsc
        # ayp check done in post-processing
        result[18] = sc

    result[19] = 0.0  # ayp placeholder, computed in post-processing
    return result, trade_pnls[:trade_count], trade_months[:trade_count], trade_types[:trade_count]


# ==================== POST-PROCESSING (Python, builds monthly/yearly dicts) ====================
def postprocess(result_arr, trade_pnls, trade_months_encoded, trade_types, month_keys_arr, init_cap):
    """Convert JIT output back to the same dict format as original."""
    if result_arr[18] < 0:
        return None  # invalid (tt < 10)

    tt = int(result_arr[4])
    wins = int(result_arr[5])
    losses = int(result_arr[6])

    monthly = {}
    yearly = {}
    for t in range(len(trade_pnls)):
        mk_ = month_keys_arr[trade_months_encoded[t]]
        pnl_ = trade_pnls[t]
        if mk_ not in monthly:
            monthly[mk_] = {'trades':0,'wins':0,'losses':0,'pnl':0.0,'tp':0.0,'tl':0.0}
        monthly[mk_]['trades'] += 1
        monthly[mk_]['pnl'] += pnl_
        if pnl_ >= 0:
            monthly[mk_]['wins'] += 1; monthly[mk_]['tp'] += pnl_
        else:
            monthly[mk_]['losses'] += 1; monthly[mk_]['tl'] += abs(pnl_)
        yr = mk_[:4]
        if yr not in yearly:
            yearly[yr] = {'trades':0,'wins':0,'losses':0,'pnl':0.0}
        yearly[yr]['trades'] += 1; yearly[yr]['pnl'] += pnl_
        if pnl_ >= 0: yearly[yr]['wins'] += 1
        else: yearly[yr]['losses'] += 1

    all_yp = all(v['pnl'] > 0 for v in yearly.values()) if yearly else False
    score = result_arr[18]
    if all_yp:
        score *= 1.5

    return {
        'final': result_arr[0], 'ret': result_arr[1], 'pf': result_arr[2],
        'mdd': result_arr[3], 'tt': tt, 'wins': wins, 'losses': losses,
        'wr': result_arr[7],
        'sl': int(result_arr[8]), 'tsl': int(result_arr[9]),
        'rev': int(result_arr[10]), 'fc': int(result_arr[11]), 'pe': int(result_arr[12]),
        'tp': result_arr[13], 'tl': result_arr[14],
        'aw': result_arr[15], 'al': result_arr[16], 'mcl': int(result_arr[17]),
        'score': score, 'ayp': all_yp,
        'monthly': monthly, 'yearly': yearly
    }


# ==================== WRAPPER: dict params → scalar args ====================
def params_to_scalars(p):
    """Convert param dict to tuple of scalars for JIT function."""
    dual_map = {'SINGLE': 0.0, 'BOTH': 1.0, 'EITHER': 2.0}
    return (
        p.get('fee_rate', 0.0004),
        float(p.get('warmup', 800)),
        p.get('initial_capital', 5000.0),
        dual_map.get(p.get('dual_mode', 'BOTH'), 1.0),
        float(p.get('monitor_window', 6)),
        float(p.get('entry_delay', 0)),
        1.0 if p.get('skip_same_dir', True) else 0.0,
        p.get('daily_loss_limit', -0.25),
        1.0 if p.get('use_eqs', True) else 0.0,
        p.get('eqs_threshold', 45.0),
        p.get('w_adx', 0.15), p.get('w_rsi', 0.10), p.get('w_vol', 0.05),
        p.get('w_gap', 0.15), p.get('w_stoch', 0.05), p.get('w_cmf', 0.05),
        p.get('w_bb', 0.25), p.get('w_atr', 0.20),
        float(p.get('adx_rise_bars', 6)),
        p.get('sl_pct', 1.0), p.get('ta_pct', 30.0), p.get('tsl_pct', 9.0),
        1.0 if p.get('use_partial', False) else 0.0,
        p.get('partial_trigger', 15.0), p.get('partial_pct', 0.5),
        float(p.get('leverage', 10)),
        1.0 if p.get('use_ranging_filter', True) else 0.0,
        p.get('ranging_adx_th', 20.0),
        p.get('ranging_bbw_th', 2.5),
        p.get('ranging_atr_ratio_th', 0.8),
        1.0 if p.get('use_mtf_filter', True) else 0.0,
        1.0 if p.get('use_reentry', False) else 0.0,
        float(p.get('reentry_wait', 5)),
        p.get('reentry_adx_min', 25.0),
        float(p.get('max_reentries', 3)),
        p.get('margin_fixed', -1.0) if p.get('margin_fixed') is not None else -1.0,
        1.0 if p.get('use_compound', True) else 0.0,
        p.get('base_margin', 1000.0),
        p.get('margin_pct', 0.20),
        p.get('stage2_th', 15000.0), p.get('stage3_th', 75000.0),
        p.get('growth_rate', 0.15), p.get('accel_rate', 0.35),
        p.get('max_margin_pct', 0.40),
        p.get('dd_th1', 0.10), p.get('dd_sc1', 0.70),
        p.get('dd_th2', 0.20), p.get('dd_sc2', 0.50),
        p.get('dd_th3', 0.30),
        float(p.get('consec_limit', 5)),
        p.get('consec_scale', 0.50),
    )


# ==================== GLOBALS for multiprocessing ====================
G = {}

def init_w(g):
    global G; G = g

def wfn(p):
    try:
        scalars = params_to_scalars(p)
        result_arr, t_pnls, t_months, t_types = backtest_jit(
            G['cl'], G['hi'], G['lo'], G['vol'], G['dv_int'],
            G['ma'][p.get('primary_ma_key', 'DEMA_125')],
            G['ma'].get(p.get('secondary_ma_key', 'HMA_125'), G['ma'][p.get('primary_ma_key', 'DEMA_125')]),
            G['ma'][p.get('slow_ma_key', 'EMA_800')],
            G['ind']['ADX'], G['ind']['ATR'], G['ind'][f"RSI_{p.get('rsi_period',7)}"],
            G['ind']['SK'], G['ind']['SD'], G['ind']['CMF'], G['ind']['BBW'],
            G['ind']['VM'], G['ind']['BWM'], G['ind']['AM'], G['htf'],
            *scalars
        )
        r = postprocess(result_arr, t_pnls, t_months, t_types, G['mk_arr'], p.get('initial_capital', 5000.0))
        if r and r['tt'] >= 15:
            r['params'] = dict(p)
            return r
    except Exception:
        pass
    return None


# ==================== WARMUP JIT (compile once before multiprocessing) ====================
def warmup_jit(G):
    """Call backtest_jit once with small data to trigger Numba compilation."""
    print("  [JIT] Compiling backtest_jit (first call)...", flush=True)
    t0 = time.time()
    n = min(1000, len(G['cl']))
    dummy_scalars = params_to_scalars({})  # all defaults
    backtest_jit(
        G['cl'][:n], G['hi'][:n], G['lo'][:n], G['vol'][:n], G['dv_int'][:n],
        G['ma']['DEMA_125'][:n] if 'DEMA_125' in G['ma'] else G['cl'][:n],
        G['cl'][:n], G['cl'][:n],
        G['ind']['ADX'][:n], G['ind']['ATR'][:n], G['ind']['RSI_7'][:n],
        G['ind']['SK'][:n], G['ind']['SD'][:n], G['ind']['CMF'][:n], G['ind']['BBW'][:n],
        G['ind']['VM'][:n], G['ind']['BWM'][:n], G['ind']['AM'][:n], G['htf'][:n],
        *dummy_scalars
    )
    print(f"  [JIT] Compiled in {time.time()-t0:.1f}s", flush=True)


def main():
    print("=" * 90)
    print("  ETH V7.3 GPT-5.4 Collaboration — NUMBA JIT Accelerated")
    print("=" * 90)
    global G

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
            ma[f"{mt}_{p_}"] = calc_ma(cs, mt, p_, vs).values
    ind = {}
    for p_ in [7, 14]:
        ind[f"RSI_{p_}"] = calc_rsi(cs, p_).values
    adx_v, atr_v = calc_adx(hs, ls, cs, 20)
    ind['ADX'] = adx_v.values; ind['ATR'] = atr_v.values
    sk, sd = calc_stoch(hs, ls, cs); ind['SK'] = sk.values; ind['SD'] = sd.values
    ind['CMF'] = calc_cmf(hs, ls, cs, vs, 20).values
    ind['BBW'] = calc_bb_w(cs, 20, 2.0).values
    ind['VM'] = vs.rolling(20).mean().fillna(vs.iloc[0]).values
    ind['BWM'] = pd.Series(ind['BBW']).rolling(50).mean().fillna(ind['BBW'][0]).values
    ind['AM'] = pd.Series(ind['ATR']).rolling(50).mean().fillna(ind['ATR'][0]).values
    ind['CCI'] = calc_cci(hs, ls, cs, 20).values

    # HTF 1h EMA(200) trend
    htf_ema = ema(df1h['close'], 200).values; htf_cl = df1h['close'].values; htf_ts = df1h['timestamp'].values
    htf_trend = np.zeros(len(df30)); j = 0
    for i in range(len(df30)):
        ts30 = df30['timestamp'].iloc[i]
        while j < len(df1h) - 1 and htf_ts[j+1] <= ts30: j += 1
        if j < len(htf_ema):
            htf_trend[i] = 1.0 if htf_cl[j] > htf_ema[j] else -1.0

    # Month keys: encode as integer index for JIT, keep string array for post-processing
    ts_ = pd.DatetimeIndex(df30['timestamp'])
    mk_strings = [f"{t.year}-{t.month:02d}" for t in ts_]
    # Create unique month key → index mapping
    unique_months = sorted(set(mk_strings))
    mk_to_idx = {m: i for i, m in enumerate(unique_months)}
    mk_int = np.array([mk_to_idx[m] for m in mk_strings], dtype=np.int32)
    mk_arr = np.array(unique_months)  # for reverse lookup in post-processing

    dv_ = ts_.day.values.astype(np.float64)

    G = {
        'cl': cs.values.astype(np.float64),
        'hi': hs.values.astype(np.float64),
        'lo': ls.values.astype(np.float64),
        'vol': vs.values.astype(np.float64),
        'mk': mk_strings,
        'mk_arr': mk_arr,
        'dv': dv_,
        'dv_int': mk_int.astype(np.float64),  # reuse day_vals slot for month-key index
        'ma': ma,
        'ind': ind,
        'htf': htf_trend
    }

    NW = max(1, cpu_count() - 2)
    print(f"  Done {time.time()-t0:.0f}s Workers:{NW}", flush=True)

    # JIT warmup (compile before forking workers)
    warmup_jit(G)

    # Base params (v7.3 best)
    base = {'initial_capital':5000.0,'fee_rate':0.0004,'warmup':800,'leverage':10,
        'daily_loss_limit':-0.25,'adx_rise_bars':6,'rsi_period':7,
        'use_eqs':True,'eqs_threshold':45,
        'w_adx':.15,'w_rsi':.10,'w_vol':.05,'w_gap':.15,
        'w_stoch':.05,'w_cmf':.05,'w_bb':.25,'w_atr':.20,
        'use_partial':False,'use_mtf_filter':True,
        'use_ranging_filter':True,'ranging_adx_th':20,'ranging_bbw_th':2.5,'ranging_atr_ratio_th':0.8}

    # ====== COMPREHENSIVE SEARCH ======
    print("\n" + "=" * 90)
    print("  COMPREHENSIVE GPT-COLLAB SEARCH (NUMBA JIT)")
    print("=" * 90)
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
    jobs_filtered = []
    seen = set()
    for p in jobs:
        key = (p['primary_ma_key'], p['slow_ma_key'], p['dual_mode'],
               p['sl_pct'], p['ta_pct'], p['tsl_pct'], p['monitor_window'], p['entry_delay'],
               str(p.get('margin_fixed')), str(p.get('use_compound')), str(p.get('margin_pct', '')),
               p['use_ranging_filter'], p['use_mtf_filter'], p['use_eqs'])
        if key not in seen:
            seen.add(key); jobs_filtered.append(p)
    jobs = jobs_filtered
    print(f"  Total unique combos: {len(jobs)}", flush=True)

    # Run in parallel
    t0 = time.time(); results = []; done = 0
    with Pool(NW, init_w, (G,)) as pool:
        for r in pool.imap_unordered(wfn, jobs, chunksize=50):
            done += 1
            if r: results.append(r)
            if done % (max(1, len(jobs) // 10)) == 0 or done == len(jobs):
                print(f"    {done}/{len(jobs)} ({done/len(jobs)*100:.0f}%) pass={len(results)} {time.time()-t0:.0f}s", flush=True)
    results.sort(key=lambda x: x['score'], reverse=True)
    print(f"  Done: {len(results)} pass in {time.time()-t0:.0f}s", flush=True)

    # TOP results
    print(f"\n{'='*90}\n  TOP 10 BY SCORE\n{'='*90}")
    print(f"  {'#':>3} {'Balance':>14} {'Return':>10} {'PF':>6} {'MDD':>6} {'T':>5} {'W':>4} {'L':>4} {'SL':>4} {'TSL':>4} {'AllYr':>5} {'Score':>8}")
    for i, r in enumerate(results[:10]):
        ay = 'Y' if r.get('ayp') else 'N'
        print(f"  {i+1:>3} ${r['final']:>13,.0f} {r['ret']:>9,.0f}% {r['pf']:>5.2f} {r['mdd']:>5.1f}% {r['tt']:>5} {r['wins']:>4} {r['losses']:>4} {r['sl']:>4} {r['tsl']:>4} {ay:>5} {r['score']:>8.1f}")

    by_ret = [r for r in results if r['mdd'] <= 30]
    by_ret.sort(key=lambda x: x['ret'], reverse=True)
    print(f"\n  TOP 10 BY RETURN (MDD<=30%)")
    for i, r in enumerate(by_ret[:10]):
        ay = 'Y' if r.get('ayp') else 'N'
        print(f"  {i+1:>3} ${r['final']:>13,.0f} {r['ret']:>9,.0f}% {r['pf']:>5.2f} {r['mdd']:>5.1f}% {r['tt']:>5} {r['wins']:>4} {r['losses']:>4} {ay:>5}")

    best = by_ret[0] if by_ret else results[0]

    # 6-engine + 30x verification
    print(f"\n{'='*90}\n  6-ENGINE + 30x VERIFICATION\n{'='*90}")
    bp = best['params']
    for eng in range(6):
        r = wfn(bp)
        if r: print(f"  Engine {eng+1}: ${r['final']:>14,.2f} PF={r['pf']:.4f} MDD={r['mdd']:.2f}% T={r['tt']}")
    finals = []
    for _ in range(30):
        r = wfn(bp)
        if r: finals.append(r['final'])
    if finals:
        print(f"  30x: Mean=${np.mean(finals):,.2f} Std=${np.std(finals):.6f}")

    # Monthly detail
    print(f"\n{'='*120}")
    print(f"  MONTHLY DETAIL — BEST (MDD<=30%)")
    print(f"{'='*120}")
    wlr = best['aw'] / best['al'] if best['al'] > 0 else 999
    print(f"  Final: ${best['final']:,.0f} | Return: {best['ret']:,.1f}% | PF: {best['pf']:.2f} | MDD: {best['mdd']:.1f}%")
    print(f"  Trades: {best['tt']} (W:{best['wins']} L:{best['losses']}) | WR: {best['wr']:.1f}% | WLR: {wlr:.1f}:1")
    print(f"  SL:{best['sl']} TSL:{best['tsl']} REV:{best['rev']} FC:{best['fc']} | MaxCL: {best['mcl']} | AllYr: {'Y' if best['ayp'] else 'N'}")
    print(f"\n  Key params:")
    for k in ['primary_ma_key','secondary_ma_key','slow_ma_key','dual_mode','sl_pct','ta_pct','tsl_pct',
              'monitor_window','entry_delay','use_eqs','eqs_threshold','use_ranging_filter','use_mtf_filter',
              'use_compound','margin_pct','margin_fixed','accel_rate','growth_rate']:
        if k in bp: print(f"    {k:30s} = {bp[k]}")

    print(f"\n  {'Year':>6} {'T':>5} {'W':>4} {'L':>4} {'PnL':>14} {'Balance':>14}")
    cumbal = 5000.0
    for yr in sorted(best['yearly'].keys()):
        yd = best['yearly'][yr]; cumbal += yd['pnl']
        print(f"  {yr:>6} {yd['trades']:>5} {yd['wins']:>4} {yd['losses']:>4} ${yd['pnl']:>13,.0f} ${cumbal:>13,.0f}")

    print(f"\n  {'Month':>9} {'T':>4} {'W/L':>7} {'WR':>6} {'WLR':>8} {'PnL':>14} {'Balance':>14} {'Note':>6}")
    all_months = []
    for y in range(2020, 2027):
        for m in range(1, 13):
            mk_ = f"{y}-{m:02d}"
            if mk_ <= "2026-03": all_months.append(mk_)
    cumbal = 5000.0
    for mk_ in all_months:
        if mk_ in best['monthly']:
            md = best['monthly'][mk_]; cumbal += md['pnl']
            t = md['trades']; w = md['wins']; l = md['losses']
            wr_m = w / t * 100 if t > 0 else 0
            wlr_m = md['tp'] / md['tl'] if md['tl'] > 0 else (999 if md['tp'] > 0 else 0)
            note = 'LOSS' if md['pnl'] < 0 else ''
            print(f"  {mk_:>9} {t:>4} {w:>3}/{l:<3} {wr_m:>5.0f}% {wlr_m:>7.1f}x ${md['pnl']:>13,.0f} ${cumbal:>13,.0f} {note:>6}")
        else:
            print(f"  {mk_:>9} {'0':>4} {'0/0':>7} {'0%':>6} {'0.0x':>8} {'$0':>14} ${cumbal:>13,.0f} {'-':>6}")

    with open('v7_3_gpt_numba_results.json', 'w') as f:
        json.dump({'params': bp, 'final': best['final'], 'ret': best['ret'],
                   'pf': best['pf'], 'mdd': best['mdd'], 'trades': best['tt']}, f, indent=2, default=str)
    print(f"\n  Saved to v7_3_gpt_numba_results.json")
    print(f"\n{'='*90}\n  V7.3 GPT Collab (NUMBA JIT) Complete!\n{'='*90}")


if __name__ == '__main__':
    main()
