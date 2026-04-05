# -*- coding: utf-8 -*-
"""
ETH V9.5 CUDA GPU Mega-Optimizer
Combines best elements from V8.3~V8.31:
  - 10m timeframe (V8.16: 3x precision, PF 3.06)
  - Wide MA grid: EMA 30~300 fast × 300~2000 slow
  - TA/TSL grid: TA 15~60 × TSL 5~15 (V8.13 discovery: TA54/TSL8)
  - SL grid: 1.0~3.0% (V8.3: 2.0%, V8.13: 2.3%)
  - Monitor grid: 6~36 bars (V8.4: 24, V8.16: 18)
  - SKIP_SAME_DIR: True/False (V8.16: True, V8.13: False)
  - MTF filter: ON/OFF (V8.31: ON, V8.16: OFF)
  - Fixed $1,000 margin, 10x leverage
"""
import sys, os, time, warnings, json, gc
import pandas as pd
import numpy as np
from numba import njit, cuda
from datetime import datetime
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
warnings.filterwarnings('ignore')

# ============ INDICATORS ============
def ema_pd(s, p): return s.ewm(span=p, adjust=False).mean()

def calc_adx_atr(h, l, c, period=20):
    a = 1.0/period
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    up = h - h.shift(1); dn = l.shift(1) - l
    pdm = pd.Series(np.where((up>dn)&(up>0), up, 0), index=c.index)
    mdm = pd.Series(np.where((dn>up)&(dn>0), dn, 0), index=c.index)
    atr = tr.ewm(alpha=a, min_periods=period, adjust=False).mean()
    pdi = 100*pdm.ewm(alpha=a, min_periods=period, adjust=False).mean()/atr.replace(0, 1e-10)
    mdi = 100*mdm.ewm(alpha=a, min_periods=period, adjust=False).mean()/atr.replace(0, 1e-10)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, 1e-10)
    adx = dx.ewm(alpha=a, min_periods=period, adjust=False).mean()
    return adx.fillna(0).values.astype(np.float64), atr.fillna(0).values.astype(np.float64)


# ============ JIT BACKTEST (simple, no EQS — V8.3+ style) ============
@njit(cache=True)
def bt_v95(close, high, low, fm, sm, htf_trend, n,
           SL, TA, TSL, MON, SKIP, USE_MTF, FEE, WU):
    cap = 5000.0; pos = 0; epx = 0.0; psz = 10000.0; slp = 0.0
    ton = 0; thi = 0.0; tlo = 999999.0
    w = 0; ws = 0; ld = 0; pk = cap; mdd = 0.0
    sc = 0; tc = 0; rc = 0; wn = 0; ln = 0; gp = 0.0; gl = 0.0

    for i in range(WU, n):
        px = close[i]; hi = high[i]; lo_ = low[i]

        if pos != 0:
            w = 0
            # SL
            if ton == 0 and SL > 0:
                if (pos == 1 and lo_ <= slp) or (pos == -1 and hi >= slp):
                    pnl = (slp-epx)/epx*psz*pos - psz*FEE; cap += pnl; sc += 1
                    if pnl > 0: wn += 1; gp += pnl
                    else: ln += 1; gl += abs(pnl)
                    ld = pos; pos = 0
                    if cap > pk: pk = cap
                    dd = (pk-cap)/pk if pk > 0 else 0.0
                    if dd > mdd: mdd = dd
                    if cap <= 0: break
                    continue
            # TA activation
            br = ((hi-epx)/epx*100) if pos == 1 else ((epx-lo_)/epx*100)
            if br >= TA and ton == 0: ton = 1
            # TSL
            if ton == 1 and TSL > 0:
                ex_hit = 0
                if pos == 1:
                    if hi > thi: thi = hi
                    ns = thi*(1-TSL/100)
                    if ns > slp: slp = ns
                    if px <= slp: ex_hit = 1
                else:
                    if lo_ < tlo: tlo = lo_
                    ns = tlo*(1+TSL/100)
                    if ns < slp: slp = ns
                    if px >= slp: ex_hit = 1
                if ex_hit == 1:
                    pnl = (px-epx)/epx*psz*pos - psz*FEE; cap += pnl; tc += 1
                    if pnl > 0: wn += 1; gp += pnl
                    else: ln += 1; gl += abs(pnl)
                    ld = pos; pos = 0
                    if cap > pk: pk = cap
                    dd = (pk-cap)/pk if pk > 0 else 0.0
                    if dd > mdd: mdd = dd
                    if cap <= 0: break
                    continue
            # REV
            if i > 0:
                bn = fm[i] > sm[i]; bp = fm[i-1] > sm[i-1]
                cu = bn and not bp; cd = not bn and bp
                if (pos == 1 and cd) or (pos == -1 and cu):
                    pnl = (px-epx)/epx*psz*pos - psz*FEE; cap += pnl; rc += 1
                    if pnl > 0: wn += 1; gp += pnl
                    else: ln += 1; gl += abs(pnl)
                    ld = pos; pos = 0
        # Entry
        if pos == 0 and i > 0:
            bn = fm[i] > sm[i]; bp = fm[i-1] > sm[i-1]
            cu = bn and not bp; cd = not bn and bp
            if cu: w = 1; ws = i
            elif cd: w = -1; ws = i
            if w != 0 and i > ws:
                if MON > 0 and i - ws > int(MON): w = 0; continue
                if SKIP > 0 and w == ld: continue
                # MTF filter
                if USE_MTF > 0:
                    if w == 1 and htf_trend[i] < 0: continue
                    if w == -1 and htf_trend[i] > 0: continue
                if cap < 1000: continue
                cap -= psz*FEE; pos = w; epx = px; ton = 0; thi = px; tlo = px
                if pos == 1: slp = px*(1-SL/100)
                else: slp = px*(1+SL/100)
                if cap > pk: pk = cap
                w = 0
        if cap > pk: pk = cap
        dd = (pk-cap)/pk if pk > 0 else 0.0
        if dd > mdd: mdd = dd
        if cap <= 0: break

    # Close open position
    if pos != 0 and cap > 0:
        pnl = (close[n-1]-epx)/epx*psz*pos - psz*FEE; cap += pnl
        if pnl > 0: wn += 1; gp += pnl
        else: ln += 1; gl += abs(pnl)

    tot = sc+tc+rc
    pf = gp/gl if gl > 0 else 0.0
    wr = wn/(wn+ln)*100.0 if (wn+ln) > 0 else 0.0
    return cap, gp-gl, tot, sc, tc, rc, wn, ln, wr, pf, mdd*100


# ============ CUDA KERNEL ============
@cuda.jit
def bt_v95_kernel(close, high, low, fm_all, sm_all, htf_all,
                  params_matrix, results_matrix, n_arr):
    tid = cuda.grid(1)
    if tid >= params_matrix.shape[0]: return
    n = n_arr[0]
    p = params_matrix[tid]
    SL=p[0]; TA=p[1]; TSL=p[2]; MON=int(p[3]); SKIP=int(p[4]); USE_MTF=int(p[5]); FEE=p[6]; WU=int(p[7])

    cap=5000.0;pos=0;epx=0.0;psz=10000.0;slp=0.0
    ton=0;thi=0.0;tlo=999999.0
    w=0;ws=0;ld=0;pk=cap;mdd=0.0
    sc=0;tc=0;rc=0;wn=0;ln=0;gp=0.0;gl=0.0

    for i in range(WU, n):
        px=close[i];hi=high[i];lo_=low[i]
        fm_i=fm_all[tid,i];sm_i=sm_all[tid,i]
        if pos!=0:
            w=0
            if ton==0 and SL>0:
                if (pos==1 and lo_<=slp) or (pos==-1 and hi>=slp):
                    pnl=(slp-epx)/epx*psz*pos-psz*FEE;cap+=pnl;sc+=1
                    if pnl>0:wn+=1;gp+=pnl
                    else:ln+=1;gl+=abs(pnl)
                    ld=pos;pos=0
                    if cap>pk:pk=cap
                    dd=(pk-cap)/pk if pk>0 else 0.0
                    if dd>mdd:mdd=dd
                    if cap<=0:break
                    continue
            br=((hi-epx)/epx*100) if pos==1 else ((epx-lo_)/epx*100)
            if br>=TA and ton==0:ton=1
            if ton==1 and TSL>0:
                ex_hit=0
                if pos==1:
                    if hi>thi:thi=hi
                    ns=thi*(1-TSL/100)
                    if ns>slp:slp=ns
                    if px<=slp:ex_hit=1
                else:
                    if lo_<tlo:tlo=lo_
                    ns=tlo*(1+TSL/100)
                    if ns<slp:slp=ns
                    if px>=slp:ex_hit=1
                if ex_hit==1:
                    pnl=(px-epx)/epx*psz*pos-psz*FEE;cap+=pnl;tc+=1
                    if pnl>0:wn+=1;gp+=pnl
                    else:ln+=1;gl+=abs(pnl)
                    ld=pos;pos=0
                    if cap>pk:pk=cap
                    dd=(pk-cap)/pk if pk>0 else 0.0
                    if dd>mdd:mdd=dd
                    if cap<=0:break
                    continue
            if i>0:
                fm_prev=fm_all[tid,i-1];sm_prev=sm_all[tid,i-1]
                bn=fm_i>sm_i;bp=fm_prev>sm_prev
                cu=bn and not bp;cd=not bn and bp
                if (pos==1 and cd) or (pos==-1 and cu):
                    pnl=(px-epx)/epx*psz*pos-psz*FEE;cap+=pnl;rc+=1
                    if pnl>0:wn+=1;gp+=pnl
                    else:ln+=1;gl+=abs(pnl)
                    ld=pos;pos=0
        if pos==0 and i>0:
            fm_prev=fm_all[tid,i-1];sm_prev=sm_all[tid,i-1]
            bn=fm_i>sm_i;bp=fm_prev>sm_prev
            cu=bn and not bp;cd=not bn and bp
            if cu:w=1;ws=i
            elif cd:w=-1;ws=i
            if w!=0 and i>ws:
                if MON>0 and i-ws>MON:w=0;continue
                if SKIP>0 and w==ld:continue
                if USE_MTF>0:
                    htf_i=htf_all[tid,i]
                    if w==1 and htf_i<0:continue
                    if w==-1 and htf_i>0:continue
                if cap<1000:continue
                cap-=psz*FEE;pos=w;epx=px;ton=0;thi=px;tlo=px
                if pos==1:slp=px*(1-SL/100)
                else:slp=px*(1+SL/100)
                if cap>pk:pk=cap
                w=0
        if cap>pk:pk=cap
        dd=(pk-cap)/pk if pk>0 else 0.0
        if dd>mdd:mdd=dd
        if cap<=0:break

    if pos!=0 and cap>0:
        pnl=(close[n-1]-epx)/epx*psz*pos-psz*FEE;cap+=pnl
        if pnl>0:wn+=1;gp+=pnl
        else:ln+=1;gl+=abs(pnl)
    tot=sc+tc+rc;pf=gp/gl if gl>0 else 0.0;wr=wn/(wn+ln)*100.0 if(wn+ln)>0 else 0.0
    results_matrix[tid,0]=cap;results_matrix[tid,1]=gp-gl
    results_matrix[tid,2]=float(tot);results_matrix[tid,3]=float(sc);results_matrix[tid,4]=float(tc)
    results_matrix[tid,5]=float(rc);results_matrix[tid,6]=float(wn);results_matrix[tid,7]=float(ln)
    results_matrix[tid,8]=wr;results_matrix[tid,9]=pf;results_matrix[tid,10]=mdd*100


NUM_RESULTS = 11

def run_cuda_grouped(ma_groups, close, high, low, BATCH_SIZE=600):
    n = len(close)
    all_results = []
    d_close = cuda.to_device(close); d_high = cuda.to_device(high); d_low = cuda.to_device(low)
    d_n = cuda.to_device(np.array([n], dtype=np.int64))

    for fm_arr, sm_arr, htf_arr, param_list, meta_list in ma_groups:
        N = len(param_list)
        for b_start in range(0, N, BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, N); bs = b_end - b_start
            params_mat = np.array(param_list[b_start:b_end], dtype=np.float64)
            fm_t = np.broadcast_to(fm_arr[np.newaxis,:], (bs,n)).copy()
            sm_t = np.broadcast_to(sm_arr[np.newaxis,:], (bs,n)).copy()
            htf_t = np.broadcast_to(htf_arr[np.newaxis,:], (bs,n)).copy()
            d_fm=cuda.to_device(fm_t);d_sm=cuda.to_device(sm_t);d_htf=cuda.to_device(htf_t)
            d_params=cuda.to_device(params_mat)
            d_results=cuda.device_array((bs,NUM_RESULTS),dtype=np.float64)
            threads=128;blocks=(bs+threads-1)//threads
            bt_v95_kernel[blocks,threads](d_close,d_high,d_low,d_fm,d_sm,d_htf,d_params,d_results,d_n)
            cuda.synchronize()
            res_mat=d_results.copy_to_host()
            batch_meta=meta_list[b_start:b_end]
            for j in range(bs):
                r=res_mat[j];tot=int(r[2]);pf=r[9];mdd_v=r[10]
                if tot>=30 and pf>1.5 and mdd_v<30:
                    all_results.append({
                        'cap':r[0],'net':r[1],'tot':tot,'sc':int(r[3]),'tc':int(r[4]),'rc':int(r[5]),
                        'wn':int(r[6]),'ln':int(r[7]),'wr':r[8],'pf':pf,'mdd':mdd_v,
                        **batch_meta[j]})
            del d_fm,d_sm,d_htf,d_params,d_results,fm_t,sm_t,htf_t,params_mat
    return all_results


def main():
    if not cuda.is_available():
        print("ERROR: CUDA not available!"); return
    gpu=cuda.get_current_device();mem=cuda.current_context().get_memory_info()
    print("="*90)
    print("  ETH V9.5 CUDA Mega-Optimizer")
    print(f"  GPU: {gpu.name} ({mem[0]//1024**3}GB free)")
    print(f"  10m+30m dual TF | Wide MA grid | Fixed $1K | 10x")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*90, flush=True)

    # Load data
    print("\n[1] Loading...", flush=True)
    df5 = pd.read_csv('eth_usdt_5m_2020_TO_NOW_merged.csv', parse_dates=['timestamp'])
    df5 = df5.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    agg = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    if 'quote_volume' in df5.columns: agg['quote_volume'] = 'sum'
    if 'trades' in df5.columns: agg['trades'] = 'sum'

    # Multi-timeframe
    tfs = {
        '10m': df5.set_index('timestamp').resample('10min').agg(agg).dropna().reset_index(),
        '30m': df5.set_index('timestamp').resample('30min').agg(agg).dropna().reset_index(),
    }
    df1h = df5.set_index('timestamp').resample('1h').agg(agg).dropna().reset_index()
    for k, v in tfs.items():
        print(f"  {k}: {len(v)} bars", flush=True)

    # Parameter grid (V8 best elements combined)
    fast_periods = [30, 50, 75, 83, 100, 125, 150, 200, 250, 300]
    slow_periods = [300, 400, 500, 528, 600, 700, 800, 1000, 1070, 1200, 1575, 2000]
    sls = [1.0, 1.5, 2.0, 2.3, 2.5, 3.0]
    ta_tsls = [(15,5),(20,7),(25,8),(30,8),(30,9),(35,8),(40,8),(40,9),(40,12),
               (50,8),(50,9),(54,8),(54,9),(60,8),(60,12)]
    monitors = [6, 12, 18, 24, 36]
    skips = [0, 1]
    mtfs = [0, 1]

    all_results = []
    grand_total = 0

    for tf_name, df_tf in tfs.items():
        print(f"\n[TF: {tf_name}] Computing indicators...", flush=True)
        cs = df_tf['close']; hs = df_tf['high']; ls = df_tf['low']
        c = cs.values.astype(np.float64); h = hs.values.astype(np.float64); l = ls.values.astype(np.float64)
        n = len(c)

        # HTF alignment
        htf_ema = ema_pd(df1h['close'], 200).values; htf_cl = df1h['close'].values; htf_ts = df1h['timestamp'].values
        htf_trend = np.zeros(n, dtype=np.float64); j = 0
        for i in range(n):
            ts_i = df_tf['timestamp'].iloc[i]
            while j < len(df1h)-1 and htf_ts[j+1] <= ts_i: j += 1
            if j < len(htf_ema): htf_trend[i] = 1.0 if htf_cl[j] > htf_ema[j] else -1.0

        # Pre-compute MAs
        mas = {}
        for p in fast_periods + slow_periods:
            if p not in mas:
                mas[p] = ema_pd(cs, p).values.astype(np.float64)
        print(f"  {len(mas)} MAs computed", flush=True)

        # Build grouped jobs
        ma_groups = []
        tf_total = 0
        for fp in fast_periods:
            for sp in slow_periods:
                if fp >= sp: continue
                fm = mas[fp]; sm = mas[sp]
                wu = max(sp, 50)
                param_list = []; meta_list = []
                for sl in sls:
                    for ta, tsl in ta_tsls:
                        for mon in monitors:
                            for skip in skips:
                                for mtf in mtfs:
                                    param_list.append([sl, float(ta), float(tsl), float(mon), float(skip), float(mtf), 0.0004, float(wu)])
                                    meta_list.append({'tf':tf_name,'fp':fp,'sp':sp,'sl':sl,'ta':ta,'tsl':tsl,'mon':mon,'skip':bool(skip),'mtf':bool(mtf)})
                if param_list:
                    ma_groups.append((fm, sm, htf_trend, param_list, meta_list))
                    tf_total += len(param_list)

        grand_total += tf_total
        print(f"  {tf_total:,} jobs in {len(ma_groups)} MA groups", flush=True)

        t0 = time.time()
        tf_results = run_cuda_grouped(ma_groups, c, h, l, BATCH_SIZE=600)
        el = time.time() - t0
        all_results.extend(tf_results)
        print(f"  -> {len(tf_results)} passed in {el:.0f}s ({tf_total/max(el,0.01):.0f} j/s)", flush=True)

        del mas, ma_groups, tf_results; gc.collect()

    # Score & rank
    for r in all_results:
        ret = (r['cap']-5000)/5000*100
        r['ret'] = ret
        pf=r['pf'];mdd=r['mdd'];tt=r['tot']
        if pf<=0 or mdd<=0 or ret<=0 or tt<20: r['score']=0
        else:
            lr=np.log10(max(ret,1));sc=(pf**1.5)*lr*10/max(mdd,1);sc*=min(tt/100,2.0)
            r['score']=sc

    all_results.sort(key=lambda x: x['score'], reverse=True)

    # TOP 30
    print(f"\n{'='*120}")
    print(f"  V9.5 TOP 30 (Total: {grand_total:,} combos | Passed: {len(all_results)})")
    print(f"{'='*120}")
    print(f"  {'#':>3} {'TF':>4} {'Fast':>5} {'Slow':>5} {'SL':>4} {'TA':>4} {'TSL':>4} {'Mon':>4} {'Skip':>4} {'MTF':>4} {'$Final':>12} {'Ret%':>8} {'PF':>6} {'MDD':>6} {'T':>4} {'W':>3} {'L':>4} {'Score':>7}")
    print(f"  {'-'*110}")
    for i, r in enumerate(all_results[:30], 1):
        print(f"  {i:>3} {r['tf']:>4} {r['fp']:>5} {r['sp']:>5} {r['sl']:>4.1f} {r['ta']:>4} {r['tsl']:>4} {r['mon']:>4} {'Y' if r['skip'] else 'N':>4} {'Y' if r['mtf'] else 'N':>4} ${r['cap']:>11,.0f} {r['ret']:>7,.0f}% {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['tot']:>4} {r['wn']:>3} {r['ln']:>4} {r['score']:>7.1f}")

    # Best by return (MDD<=15%)
    safe = [r for r in all_results if r['mdd'] <= 15]
    safe.sort(key=lambda x: x['ret'], reverse=True)
    print(f"\n  TOP 10 BY RETURN (MDD<=15%):")
    for i, r in enumerate(safe[:10], 1):
        print(f"  {i:>3} {r['tf']:>4} EMA({r['fp']})/EMA({r['sp']}) SL={r['sl']:.1f} TA={r['ta']} TSL={r['tsl']} | ${r['cap']:>10,.0f} ({r['ret']:,.0f}%) PF={r['pf']:.2f} MDD={r['mdd']:.1f}% T={r['tot']}")

    # 6-engine JIT verify best
    best = safe[0] if safe else all_results[0] if all_results else None
    if best:
        print(f"\n{'='*90}")
        print(f"  6-ENGINE VERIFICATION — V9.5 BEST")
        print(f"{'='*90}")
        tf_name = best['tf']
        df_tf = tfs[tf_name]
        cs = df_tf['close']; c = cs.values.astype(np.float64)
        h = df_tf['high'].values.astype(np.float64); l = df_tf['low'].values.astype(np.float64)
        n = len(c)
        fm = ema_pd(cs, best['fp']).values.astype(np.float64)
        sm = ema_pd(cs, best['sp']).values.astype(np.float64)
        htf_ema = ema_pd(df1h['close'], 200).values; htf_cl = df1h['close'].values; htf_ts = df1h['timestamp'].values
        htf_trend = np.zeros(n, dtype=np.float64); j = 0
        for i in range(n):
            ts_i = df_tf['timestamp'].iloc[i]
            while j < len(df1h)-1 and htf_ts[j+1] <= ts_i: j += 1
            if j < len(htf_ema): htf_trend[i] = 1.0 if htf_cl[j] > htf_ema[j] else -1.0

        wu = max(best['sp'], 50)
        caps = []
        for _ in range(6):
            r = bt_v95(c,h,l,fm,sm,htf_trend,n,
                       best['sl'],float(best['ta']),float(best['tsl']),float(best['mon']),
                       1 if best['skip'] else 0, 1 if best['mtf'] else 0, 0.0004, wu)
            caps.append(r[0])

        diff = max(caps)-min(caps)
        r = bt_v95(c,h,l,fm,sm,htf_trend,n,
                   best['sl'],float(best['ta']),float(best['tsl']),float(best['mon']),
                   1 if best['skip'] else 0, 1 if best['mtf'] else 0, 0.0004, wu)
        aw = r[6]/(r[6]+1e-10) and gp/r[6] if r[6] > 0 else 0
        # Simpler
        net = r[1]; tot = int(r[2]); pf = r[9]; mdd_v = r[10]
        wn = int(r[6]); ln = int(r[7])
        gp_v = 0; gl_v = 0
        if wn > 0: gp_v = (r[0]-5000+abs(net))/2 if net > 0 else 0

        print(f"  6x: {'ALL MATCH' if diff < 0.01 else 'MISMATCH'} (diff=${diff:.6f})")
        print(f"  TF: {best['tf']} | EMA({best['fp']})/EMA({best['sp']})")
        print(f"  SL: {best['sl']}% | TA: {best['ta']}% | TSL: {best['tsl']}%")
        print(f"  Monitor: {best['mon']} | Skip: {best['skip']} | MTF: {best['mtf']}")
        print(f"  Final: ${r[0]:,.0f} | Ret: +{(r[0]-5000)/50:,.0f}% | PF: {pf:.2f} | MDD: {mdd_v:.1f}%")
        print(f"  Trades: {tot} (SL:{int(r[3])} TSL:{int(r[4])} REV:{int(r[5])})")
        print(f"  W/L: {wn}W/{ln}L ({r[8]:.1f}%)")

    # Save
    save = {'version':'V9.5','total_combos':grand_total,'passed':len(all_results),
            'best':best,'top30':all_results[:30]}
    with open('v95_cuda_results.json','w') as f:
        json.dump(save,f,indent=2,default=str)
    print(f"\n  Saved: v95_cuda_results.json")
    print(f"\n{'='*90}")
    print(f"  V9.5 COMPLETE | {grand_total:,} combos")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
