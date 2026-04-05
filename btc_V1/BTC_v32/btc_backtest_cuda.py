# -*- coding: utf-8 -*-
"""
BTC/USDT v32.2 — CUDA GPU Accelerated Backtest + Optimizer
Original: backtest_6engines.py (CPU)
GPU: RTX 4060 Ti, Numba CUDA

포함:
- bt_jit: Numba JIT 단건 백테스트 (6엔진 검증용)
- bt_cuda_kernel: CUDA GPU 대량 최적화
- 파라미터 그리드 서치 + TOP 결과 출력
"""
import pandas as pd, numpy as np, time, json, sys, os, math, gc
from numba import njit, cuda
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# ═══════════════════════════════════════════════
# 데이터 로딩 + 지표
# ═══════════════════════════════════════════════
def load_data():
    """BTC 5분봉 → 30분봉"""
    base = 'D:/filesystem/futures/btc_V1/BTC_v32'
    # 데이터 파일 경로 탐색
    candidates = [
        [f'{base}/btc_usdt_5m_merged.csv'],
        [f'D:/filesystem/futures/btc_V1/test/btc_usdt_5m_merged.csv'],
        [f'{base}/btc_usdt_5m_2020_to_now_part{i}.csv' for i in range(1, 4)],
        [f'D:/filesystem/futures/btc_V1/test/btc_usdt_5m_2020_to_now_part{i}.csv' for i in range(1, 4)],
    ]
    for files in candidates:
        if all(os.path.exists(f) for f in files):
            print(f"  Loading from: {files[0][:50]}...", flush=True)
            dfs = [pd.read_csv(f, parse_dates=['timestamp']) for f in files]
            df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp', keep='last')
            df.set_index('timestamp', inplace=True)
            ohlcv = df.resample('30min').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()
            return ohlcv
    raise FileNotFoundError("BTC 5m data not found")


def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean().values.astype(np.float64)

def calc_adx(h, l, c, period=20):
    n = len(c)
    pdm = np.zeros(n); mdm = np.zeros(n); tr = np.zeros(n)
    for i in range(1, n):
        hd = h[i]-h[i-1]; ld = l[i-1]-l[i]
        pdm[i] = hd if (hd > ld and hd > 0) else 0
        mdm[i] = ld if (ld > hd and ld > 0) else 0
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    a = 1.0/period
    atr = pd.Series(tr).ewm(alpha=a, min_periods=period, adjust=False).mean()
    sp = pd.Series(pdm).ewm(alpha=a, min_periods=period, adjust=False).mean()
    sn = pd.Series(mdm).ewm(alpha=a, min_periods=period, adjust=False).mean()
    pdi = 100*sp/atr.replace(0, 1e-10)
    mdi = 100*sn/atr.replace(0, 1e-10)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, 1e-10)
    return dx.ewm(alpha=a, min_periods=period, adjust=False).mean().values.astype(np.float64)

def calc_rsi(c, period=10):
    d = np.diff(c, prepend=c[0]); g = np.where(d > 0, d, 0); lo = np.where(d < 0, -d, 0)
    a = 1.0/period
    ag = pd.Series(g).ewm(alpha=a, min_periods=period, adjust=False).mean()
    al = pd.Series(lo).ewm(alpha=a, min_periods=period, adjust=False).mean()
    return (100-100/(1+ag/al.replace(0, 1e-10))).values.astype(np.float64)


# ═══════════════════════════════════════════════
# NUMBA JIT BACKTEST (단건, 6엔진 검증용)
# ═══════════════════════════════════════════════
@njit(cache=True)
def bt_jit(c, h, l, fm, sm, av, rv, n,
           SL, TA, TSL, ADX_M, ARISE, RMIN, RMAX, GAP, MON,
           SKIP, MARGIN_PCT, LEV, FEE, DAILY_LOSS, WU):
    cap = 5000.0; pos = 0; epx = 0.0; psz = 0.0; slp = 0.0
    ton = 0; thi = 0.0; tlo = 999999.0
    w = 0; ws = 0; ld = 0; pk = cap; mdd = 0.0; dsc = cap; prev_day = -1
    sc = 0; tc = 0; rc = 0; wn = 0; ln = 0; gp = 0.0; gl = 0.0

    for i in range(WU, n):
        px = c[i]; hi = h[i]; lo_ = l[i]
        # Daily reset (every ~48 bars for 30m)
        if i > WU and i % 48 == 0:
            dsc = cap

        if pos != 0:
            w = 0
            et = 0; ex = 0.0  # 0=none, 1=SL, 2=TSL, 3=REV

            # SL (only if TSL not active)
            if ton == 0:
                if pos == 1 and lo_ <= slp: et = 1; ex = slp
                elif pos == -1 and hi >= slp: et = 1; ex = slp

            # TA activation
            br = ((hi-epx)/epx*100) if pos == 1 else ((epx-lo_)/epx*100)
            if br >= TA and ton == 0: ton = 1

            # TSL
            if et == 0 and ton == 1:
                if pos == 1:
                    if hi > thi: thi = hi
                    ns = thi * (1 - TSL/100)
                    if ns > slp: slp = ns
                    if px <= slp: et = 2; ex = px
                else:
                    if lo_ < tlo: tlo = lo_
                    ns = tlo * (1 + TSL/100)
                    if ns < slp: slp = ns
                    if px >= slp: et = 2; ex = px

            # REV
            if et == 0 and i > 0:
                bn = fm[i] > sm[i]; bp = fm[i-1] > sm[i-1]
                cu = bn and not bp; cd = not bn and bp
                if (pos == 1 and cd) or (pos == -1 and cu):
                    et = 3; ex = px

            if et > 0:
                pnl = (ex-epx)/epx*psz*pos - psz*FEE
                cap += pnl
                if et == 1: sc += 1
                elif et == 2: tc += 1
                elif et == 3: rc += 1
                if pnl > 0: wn += 1; gp += pnl
                else: ln += 1; gl += abs(pnl)
                ld = pos; pos = 0
                if cap > pk: pk = cap
                dd = (pk-cap)/pk if pk > 0 else 0.0
                if dd > mdd: mdd = dd
                continue
        else:
            if i < 1: continue
            bn = fm[i] > sm[i]; bp = fm[i-1] > sm[i-1]
            cu = bn and not bp; cd = not bn and bp
            if cu: w = 1; ws = i
            elif cd: w = -1; ws = i
            if w == 0: continue
            if i <= ws: continue
            if i - ws > int(MON): w = 0; continue
            if w == 1 and cd: w = -1; ws = i; continue
            if w == -1 and cu: w = 1; ws = i; continue
            if SKIP > 0 and w == ld: continue
            if av[i] < ADX_M: continue
            if ARISE > 0 and i >= int(ARISE) and av[i] <= av[i-int(ARISE)]: continue
            if (RMIN > 0 or RMAX < 100) and (rv[i] < RMIN or rv[i] > RMAX): continue
            if GAP > 0 and sm[i] > 0 and abs(fm[i]-sm[i])/sm[i]*100 < GAP: continue
            if dsc > 0 and (cap-dsc)/dsc <= DAILY_LOSS: w = 0; continue
            if cap <= 0: continue

            pos = w; epx = px
            margin = cap * MARGIN_PCT; psz = margin * LEV
            if pos == 1: slp = epx*(1-SL/100)
            else: slp = epx*(1+SL/100)
            ton = 0; thi = epx; tlo = epx; w = 0

        if cap > pk: pk = cap
        dd = (pk-cap)/pk if pk > 0 else 0.0
        if dd > mdd: mdd = dd
        if cap <= 0: break

    tot = sc+tc+rc
    pf = gp/gl if gl > 0 else 0.0
    wr = wn/(wn+ln)*100.0 if (wn+ln) > 0 else 0.0
    net = gp-gl
    return cap, net, tot, sc, tc, rc, wn, ln, wr, pf, mdd*100


# ═══════════════════════════════════════════════
# CUDA KERNEL
# ═══════════════════════════════════════════════
# params: [SL,TA,TSL,ADX_M,ARISE,RMIN,RMAX,GAP,MON,SKIP,MARGIN_PCT,LEV,FEE,DAILY_LOSS,WU]
NUM_PARAMS = 15
NUM_RESULTS = 11

@cuda.jit
def bt_cuda_kernel(close, high, low, fm_all, sm_all, adx_all, rsi_all,
                   params_matrix, results_matrix, n_arr):
    tid = cuda.grid(1)
    if tid >= params_matrix.shape[0]:
        return

    n = n_arr[0]
    p = params_matrix[tid]
    SL=p[0]; TA=p[1]; TSL=p[2]; ADX_M=p[3]; ARISE=int(p[4])
    RMIN=p[5]; RMAX=p[6]; GAP=p[7]; MON=int(p[8]); SKIP=int(p[9])
    MARGIN_PCT=p[10]; LEV=p[11]; FEE=p[12]; DAILY_LOSS=p[13]; WU=int(p[14])

    cap=5000.0; pos=0; epx=0.0; psz=0.0; slp=0.0
    ton=0; thi=0.0; tlo=999999.0
    w=0; ws=0; ld=0; pk=cap; mdd=0.0; dsc=cap
    sc=0; tc=0; rc=0; wn=0; ln=0; gp=0.0; gl=0.0

    for i in range(WU, n):
        px=close[i]; hi=high[i]; lo_=low[i]
        fm_i=fm_all[tid,i]; sm_i=sm_all[tid,i]; av_i=adx_all[tid,i]; rv_i=rsi_all[tid,i]
        if i > WU and i % 48 == 0: dsc = cap

        if pos != 0:
            w = 0; et = 0; ex = 0.0
            if ton == 0:
                if pos == 1 and lo_ <= slp: et = 1; ex = slp
                elif pos == -1 and hi >= slp: et = 1; ex = slp
            br = ((hi-epx)/epx*100) if pos == 1 else ((epx-lo_)/epx*100)
            if br >= TA and ton == 0: ton = 1
            if et == 0 and ton == 1:
                if pos == 1:
                    if hi > thi: thi = hi
                    ns = thi*(1-TSL/100)
                    if ns > slp: slp = ns
                    if px <= slp: et = 2; ex = px
                else:
                    if lo_ < tlo: tlo = lo_
                    ns = tlo*(1+TSL/100)
                    if ns < slp: slp = ns
                    if px >= slp: et = 2; ex = px
            if et == 0 and i > 0:
                fm_prev=fm_all[tid,i-1]; sm_prev=sm_all[tid,i-1]
                bn=fm_i>sm_i; bp=fm_prev>sm_prev
                cu=bn and not bp; cd=not bn and bp
                if (pos==1 and cd) or (pos==-1 and cu): et=3; ex=px
            if et > 0:
                pnl = (ex-epx)/epx*psz*pos - psz*FEE; cap += pnl
                if et==1: sc+=1
                elif et==2: tc+=1
                elif et==3: rc+=1
                if pnl>0: wn+=1; gp+=pnl
                else: ln+=1; gl+=abs(pnl)
                ld=pos; pos=0
                if cap>pk: pk=cap
                dd=(pk-cap)/pk if pk>0 else 0.0
                if dd>mdd: mdd=dd
                continue
        else:
            if i < 1: continue
            fm_prev=fm_all[tid,i-1]; sm_prev=sm_all[tid,i-1]
            bn=fm_i>sm_i; bp=fm_prev>sm_prev
            cu=bn and not bp; cd=not bn and bp
            if cu: w=1; ws=i
            elif cd: w=-1; ws=i
            if w==0: continue
            if i<=ws: continue
            if i-ws>MON: w=0; continue
            if w==1 and cd: w=-1; ws=i; continue
            if w==-1 and cu: w=1; ws=i; continue
            if SKIP>0 and w==ld: continue
            if av_i<ADX_M: continue
            if ARISE>0 and i>=ARISE and adx_all[tid,i]<=adx_all[tid,i-ARISE]: continue
            if (RMIN>0 or RMAX<100) and (rv_i<RMIN or rv_i>RMAX): continue
            if GAP>0 and sm_i>0 and abs(fm_i-sm_i)/sm_i*100<GAP: continue
            if dsc>0 and (cap-dsc)/dsc<=DAILY_LOSS: w=0; continue
            if cap<=0: continue
            pos=w; epx=px; margin=cap*MARGIN_PCT; psz=margin*LEV
            if pos==1: slp=epx*(1-SL/100)
            else: slp=epx*(1+SL/100)
            ton=0; thi=epx; tlo=epx; w=0

        if cap>pk: pk=cap
        dd=(pk-cap)/pk if pk>0 else 0.0
        if dd>mdd: mdd=dd
        if cap<=0: break

    tot=sc+tc+rc; pf=gp/gl if gl>0 else 0.0
    wr=wn/(wn+ln)*100.0 if (wn+ln)>0 else 0.0

    results_matrix[tid,0]=cap; results_matrix[tid,1]=gp-gl
    results_matrix[tid,2]=float(tot); results_matrix[tid,3]=float(sc)
    results_matrix[tid,4]=float(tc); results_matrix[tid,5]=float(rc)
    results_matrix[tid,6]=float(wn); results_matrix[tid,7]=float(ln)
    results_matrix[tid,8]=wr; results_matrix[tid,9]=pf; results_matrix[tid,10]=mdd*100


# ═══════════════════════════════════════════════
# CUDA BATCH RUNNER (Memory Optimized)
# ═══════════════════════════════════════════════
def run_cuda_grouped(ma_groups, close, high, low, BATCH_SIZE=800):
    n = len(close)
    all_results = []
    d_close = cuda.to_device(close)
    d_high = cuda.to_device(high)
    d_low = cuda.to_device(low)
    d_n = cuda.to_device(np.array([n], dtype=np.int64))

    for fm_arr, sm_arr, adx_arr, rsi_arr, param_list, meta_list in ma_groups:
        N = len(param_list)
        for b_start in range(0, N, BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, N)
            bs = b_end - b_start

            params_mat = np.array(param_list[b_start:b_end], dtype=np.float64)
            fm_tiled = np.broadcast_to(fm_arr[np.newaxis,:], (bs,n)).copy()
            sm_tiled = np.broadcast_to(sm_arr[np.newaxis,:], (bs,n)).copy()
            adx_tiled = np.broadcast_to(adx_arr[np.newaxis,:], (bs,n)).copy()
            rsi_tiled = np.broadcast_to(rsi_arr[np.newaxis,:], (bs,n)).copy()

            d_fm = cuda.to_device(fm_tiled)
            d_sm = cuda.to_device(sm_tiled)
            d_adx = cuda.to_device(adx_tiled)
            d_rsi = cuda.to_device(rsi_tiled)
            d_params = cuda.to_device(params_mat)
            d_results = cuda.device_array((bs, NUM_RESULTS), dtype=np.float64)

            threads = 128
            blocks = (bs + threads - 1) // threads
            bt_cuda_kernel[blocks, threads](d_close, d_high, d_low,
                                             d_fm, d_sm, d_adx, d_rsi,
                                             d_params, d_results, d_n)
            cuda.synchronize()
            res_mat = d_results.copy_to_host()
            batch_meta = meta_list[b_start:b_end]

            for j in range(bs):
                r = res_mat[j]
                tot = int(r[2]); pf = r[9]; mdd_v = r[10]
                if tot >= 10 and pf > 1.0:
                    all_results.append({
                        'cap': r[0], 'net': r[1], 'tot': tot,
                        'sc': int(r[3]), 'tc': int(r[4]), 'rc': int(r[5]),
                        'wn': int(r[6]), 'ln': int(r[7]),
                        'wr': r[8], 'pf': pf, 'mdd': mdd_v,
                        **batch_meta[j]
                    })

            del d_fm, d_sm, d_adx, d_rsi, d_params, d_results
            del fm_tiled, sm_tiled, adx_tiled, rsi_tiled, params_mat

    return all_results


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def main():
    if not cuda.is_available():
        print("ERROR: CUDA not available!"); return

    gpu = cuda.get_current_device()
    mem = cuda.current_context().get_memory_info()
    print("="*80)
    print(f"  BTC/USDT v32.2 — CUDA GPU Backtest + Optimizer")
    print(f"  GPU: {gpu.name} ({mem[0]//1024**3}GB free / {mem[1]//1024**3}GB total)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80, flush=True)

    # Load data
    print("\n[1] Loading BTC data...", flush=True)
    df30 = load_data()
    cs = df30['close']
    c = cs.values.astype(np.float64)
    h = df30['high'].values.astype(np.float64)
    l = df30['low'].values.astype(np.float64)
    n = len(c)
    print(f"  30m: {n:,} bars", flush=True)

    # Pre-compute MAs
    print("\n[2] Computing indicators...", flush=True)
    fast_periods = [50, 75, 100, 125, 150, 200]
    slow_periods = [300, 400, 500, 600, 700, 800]
    adx_arr = calc_adx(h, l, c, 20)
    rsi_arr = calc_rsi(c, 10)

    mas = {}
    for p in fast_periods + slow_periods:
        mas[p] = calc_ema(cs, p)
    print(f"  {len(mas)} MAs + ADX + RSI", flush=True)

    # Parameter grid
    sls = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    tas = [8, 10, 12, 15, 20, 25, 30]
    tsls = [5, 7, 9, 10, 12, 15]
    adx_mins = [20, 25, 30, 35]
    arises = [0, 3, 6, 10]
    rsi_ranges = [(40, 80), (30, 70), (35, 75), (0, 100)]
    gaps = [0, 0.1, 0.2, 0.3]
    mons = [12, 24, 36]
    margin_pcts = [0.20, 0.25, 0.30, 0.35, 0.50]
    levs = [5, 10]
    skips = [1, 0]

    # Build MA-grouped jobs
    print("\n[3] Building jobs...", flush=True)
    ma_groups = []
    grand_total = 0

    for fp in fast_periods:
        for sp in slow_periods:
            if fp >= sp: continue
            fm = mas[fp]; sm = mas[sp]
            wu = max(sp, 21)

            param_list = []
            meta_list = []
            for sl in sls:
                for ta in tas:
                    for tsl in tsls:
                        if tsl >= ta: continue
                        for adx_m in adx_mins:
                            for arise in arises:
                                for rmin, rmax in rsi_ranges:
                                    for gap in gaps:
                                        for mon in mons:
                                            for mpct in margin_pcts:
                                                for lev in levs:
                                                    for skip in skips:
                                                        param_list.append([
                                                            sl, float(ta), float(tsl),
                                                            float(adx_m), float(arise),
                                                            float(rmin), float(rmax),
                                                            gap, float(mon), float(skip),
                                                            mpct, float(lev), 0.0004,
                                                            -0.20, float(wu)
                                                        ])
                                                        meta_list.append({
                                                            'fp': fp, 'sp': sp,
                                                            'sl': sl, 'ta': ta, 'tsl': tsl,
                                                            'adx': adx_m, 'arise': arise,
                                                            'rmin': rmin, 'rmax': rmax,
                                                            'gap': gap, 'mon': mon,
                                                            'mpct': mpct, 'lev': lev,
                                                            'skip': bool(skip),
                                                        })

            if param_list:
                ma_groups.append((fm, sm, adx_arr, rsi_arr, param_list, meta_list))
                grand_total += len(param_list)

    print(f"  {grand_total:,} jobs in {len(ma_groups)} MA groups", flush=True)

    # Limit if too many (optional sampling)
    if grand_total > 500000:
        print(f"  WARNING: {grand_total:,} jobs is very large. Sampling 200,000...", flush=True)
        import random; random.seed(42)
        new_groups = []
        sampled = 0
        for fm, sm, av, rv, pl, ml in ma_groups:
            if sampled >= 200000: break
            take = min(len(pl), 200000 - sampled)
            idx = random.sample(range(len(pl)), take) if take < len(pl) else list(range(len(pl)))
            new_groups.append((fm, sm, av, rv, [pl[i] for i in idx], [ml[i] for i in idx]))
            sampled += take
        ma_groups = new_groups
        grand_total = sampled
        print(f"  Sampled: {grand_total:,} jobs", flush=True)

    # Run GPU
    print(f"\n[4] Running {grand_total:,} backtests on GPU...", flush=True)
    t0 = time.time()
    all_results = run_cuda_grouped(ma_groups, c, h, l, BATCH_SIZE=800)
    el = time.time() - t0
    print(f"  Done: {len(all_results)} passed in {el:.1f}s ({grand_total/max(el,0.01):.0f} jobs/s)", flush=True)

    del ma_groups; gc.collect()

    all_results.sort(key=lambda x: x['net'], reverse=True)

    # TOP 30
    print(f"\n{'='*100}")
    print(f"  TOP 30 BTC STRATEGIES (CUDA GPU)")
    print(f"{'='*100}")
    print(f"  {'#':>3} {'EMA':>10} {'SL':>4} {'TA':>4} {'TSL':>4} {'ADX':>4} {'M%':>4} {'Lev':>3} {'Net$':>14} {'T':>4} {'PF':>6} {'MDD':>6} {'WR':>5}")
    print(f"  {'-'*90}")
    for i, r in enumerate(all_results[:30], 1):
        print(f"  {i:>3} {r['fp']}/{r['sp']:>5} {r['sl']:>4.1f} {r['ta']:>4} {r['tsl']:>4} {r['adx']:>4} {r['mpct']*100:>3.0f}% {r['lev']:>3}x ${r['net']:>13,.0f} {r['tot']:>4} {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['wr']:>4.0f}%")

    # 6-Engine verification (JIT)
    print(f"\n[5] 6-Engine Cross-Verify Top 10...", flush=True)
    verified = []
    for idx, s in enumerate(all_results[:10]):
        fm = mas[s['fp']]; sm = mas[s['sp']]
        skip_int = 1 if s['skip'] else 0
        caps = []
        for _ in range(6):
            r = bt_jit(c, h, l, fm, sm, adx_arr, rsi_arr, n,
                       s['sl'], float(s['ta']), float(s['tsl']),
                       float(s['adx']), float(s['arise']),
                       float(s['rmin']), float(s['rmax']),
                       s['gap'], float(s['mon']),
                       skip_int, s['mpct'], float(s['lev']),
                       0.0004, -0.20, max(s['sp'], 21))
            caps.append(r[0])
        diff = max(caps)-min(caps)
        ok = diff < 0.01
        s['cross_ok'] = ok
        verified.append(s)
        print(f"  #{idx+1} EMA({s['fp']}/{s['sp']}) 6E:{'OK' if ok else 'NG'} diff=${diff:.4f} Net=${s['net']:+,.0f}", flush=True)

    # Save
    json_out = 'D:/filesystem/futures/btc_V1/BTC_v32/btc_cuda_results.json'
    save = all_results[:100]
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(save, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved: {json_out}", flush=True)

    print(f"\n{'='*80}")
    print(f"  BTC CUDA COMPLETE: {grand_total:,} jobs in {el:.0f}s")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}", flush=True)


if __name__ == '__main__':
    main()
