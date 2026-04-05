"""
v24.2 Focused Max-Return: Proven entries x targeted exit/margin/leverage grid
Uses run_backtest (supports PE) for accuracy
"""
import pandas as pd, numpy as np, time, sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(r'D:\filesystem\futures\btc_V1\test')))
from btc_v164_backtest import (load_5m_data, resample, calc_wma, calc_ema, calc_sma,
                                calc_adx, calc_rsi, calc_atr, add_indicators,
                                run_backtest, print_model, FEE_RATE)
DATA_DIR = Path(r'D:\filesystem\futures\btc_V1\test')
import btc_v164_backtest as bt; bt.DATA_DIR = DATA_DIR
CAPITAL = 3000

def calc_hma(s,p):
    h=max(int(p/2),1); sq=max(int(np.sqrt(p)),1)
    return calc_wma(2*calc_wma(s,h)-calc_wma(s,p), sq)

def main():
    print("="*90, flush=True)
    print("  v24.2 FOCUSED MAX-RETURN OPTIMIZER", flush=True)
    print("="*90, flush=True)
    t0=time.time()

    df_5m=load_5m_data()
    tfs={}
    for rule,label in [('30min','30m'),('60min','1h')]:
        df=resample(df_5m,rule); df=add_indicators(df)
        df['sma300']=calc_sma(df['close'],300)
        df['ema150']=calc_ema(df['close'],150)
        df['ema100_s']=calc_ema(df['close'],100)
        tfs[label]=df
        print(f"  {label}: {len(df):,}", flush=True)
    print(f"  Ready: {time.time()-t0:.1f}s\n", flush=True)

    # PROVEN TOP ENTRIES (from v17 scan + v16.4 optimization)
    proven_entries = [
        ('30m','wma3','ema200','adx20',35,35,65,'WMA3/EMA200 RSI35-65'),
        ('30m','wma3','ema200','adx20',35,30,70,'WMA3/EMA200 RSI30-70'),
        ('30m','ema3','sma300','adx20',40,30,70,'EMA3/SMA300 ADX40 RSI30-70'),
        ('30m','ema3','sma300','adx20',40,35,65,'EMA3/SMA300 ADX40 RSI35-65'),
        ('30m','ema3','ema200','adx20',35,30,70,'EMA3/EMA200 RSI30-70'),
        ('1h','ema3','ema100_s','adx20',30,30,70,'1h EMA3/EMA100 RSI30-70'),
        ('1h','ema3','ema100_s','adx20',30,35,65,'1h EMA3/EMA100 RSI35-65'),
        ('30m','wma3','ema150','adx20',35,30,70,'WMA3/EMA150 RSI30-70'),
    ]

    # EXIT GRID: SL x TS_act x TS_trail x Margin x Leverage x PE
    exit_grid = list(itertools.product(
        [0.04, 0.06, 0.08, 0.10, 0.15],           # SL (5)
        [0.03, 0.04, 0.06, 0.08, 0.10, 0.15],     # TS act (6)
        [0.02, 0.03, 0.05],                         # TS trail (3)
        [0.40, 0.50, 0.60, 0.70],                   # Margin (4)
        [10, 15, 20],                                # Leverage (3)
        [                                             # PE configs (4)
            [],
            [(0.10, 0.25)],
            [(0.15, 0.30)],
            [(0.10, 0.25), (0.20, 0.25)],
        ],
    ))
    total = len(proven_entries) * len(exit_grid)
    print(f"  {len(proven_entries)} entries x {len(exit_grid)} exit combos = {total:,} total", flush=True)
    print(f"  Parameter space: 2,177,280 (full grid)", flush=True)

    results = []; done = 0; t1 = time.time()
    for tl, fma_col, sma_col, adx_col, am, rmin, rmax, name in proven_entries:
        df = tfs[tl]
        for sl, ta, tt, m, lv, pe in exit_grid:
            fl_dist = 1.0 / lv
            if sl >= fl_dist - 0.01: continue
            params = {
                'fast_ma':fma_col, 'slow_ma':sma_col,
                'adx_col':adx_col, 'adx_min':am,
                'rsi_min':rmin, 'rsi_max':rmax,
                'sl_pct':sl, 'sl_dynamic':False, 'atr_mult':3.0,
                'ts_act':ta, 'ts_trail':tt, 'ts_accel':False,
                'partial_exits':pe, 'reverse':'flip', 'time_sl':None,
                'h1_filter':False, 'h4_filter':False,
                'margin':m, 'leverage':lv, 're_entry_block':0,
            }
            r = run_backtest(df, tfs.get('1h', df), df, params, CAPITAL)
            if r.get('fl', 0) > 0 or r['trades'] < 3: continue
            r['name']=name; r['tl']=tl; r['sl_']=sl; r['ta_']=ta; r['tt_']=tt
            r['m_']=m; r['lv_']=lv; r['pe_']=str(pe) if pe else 'OFF'
            r['fma']=fma_col; r['sma_c']=sma_col; r['adx_c']=adx_col
            r['am_']=am; r['rmin_']=rmin; r['rmax_']=rmax
            results.append(r)
            done += 1
        elapsed = time.time()-t1
        rate = done/elapsed if elapsed>0 else 1
        print(f"  {name}: {done:,} valid ({elapsed:.0f}s, {rate:.1f}/s)", flush=True)

    print(f"\n  Total valid: {len(results):,} in {time.time()-t1:.0f}s", flush=True)

    # SORT BY RETURN
    results.sort(key=lambda x: x['bal'], reverse=True)

    hdr = f"  {'#':>2} | {'Strategy':>25} | {'SL':>4} | {'TS':>7} | {'M':>3} | {'Lv':>2} | {'PE':>10} | {'PF':>5} | {'MDD':>5} | {'Ret%':>11} | {'$':>11} | {'Tr':>3} | {'SL':>2}"
    sep = f"  {'-'*2}-+-{'-'*25}-+-{'-'*4}-+-{'-'*7}-+-{'-'*3}-+-{'-'*2}-+-{'-'*10}-+-{'-'*5}-+-{'-'*5}-+-{'-'*11}-+-{'-'*11}-+-{'-'*3}-+-{'-'*2}"
    def prow(i,r):
        pe_s = 'OFF' if r['pe_']=='OFF' else 'YES'
        print(f"  {i:>2} | {r['name']:>25} | {r['sl_']:.0%} | +{r['ta_']:.0%}/-{r['tt_']:.0%} | {r['m_']:.0%} | {r['lv_']:>2} | {pe_s:>10} | {r['pf']:>4.1f} | {r['mdd']:>4.1f}% | {r['ret']:>+10.1f}% | ${r['bal']:>10,.0f} | {r['trades']:>3} | {r['sl']:>2}", flush=True)

    print(f"\n  [TOP 30 by RETURN]", flush=True)
    print(hdr, flush=True); print(sep, flush=True)
    for i,r in enumerate(results[:30]): prow(i+1, r)

    # Best PF with high return
    pf_top = [r for r in results if r['trades']>=10 and r['ret']>500]
    pf_top.sort(key=lambda x: x['pf'], reverse=True)
    print(f"\n  [TOP 15 by PF (trades>=10, ret>500%)]", flush=True)
    print(hdr, flush=True); print(sep, flush=True)
    for i,r in enumerate(pf_top[:15]): prow(i+1, r)

    # Best with SL=0
    sl0 = [r for r in results if r['sl']==0]
    sl0.sort(key=lambda x: x['bal'], reverse=True)
    print(f"\n  [TOP 15 with SL=0]", flush=True)
    print(hdr, flush=True); print(sep, flush=True)
    for i,r in enumerate(sl0[:15]): prow(i+1, r)

    # ============================================================
    # 10x BACKTEST + 10x VERIFY FOR #1
    # ============================================================
    best = results[0]
    print(f"\n{'='*90}", flush=True)
    print(f"  WINNER: {best['name']}", flush=True)
    print(f"  SL={best['sl_']:.0%} TS=+{best['ta_']:.0%}/-{best['tt_']:.0%} M={best['m_']:.0%} Lv={best['lv_']} PE={best['pe_']}", flush=True)
    print("="*90, flush=True)

    best_params = {
        'fast_ma':best['fma'], 'slow_ma':best['sma_c'],
        'adx_col':best['adx_c'], 'adx_min':best['am_'],
        'rsi_min':best['rmin_'], 'rsi_max':best['rmax_'],
        'sl_pct':best['sl_'], 'sl_dynamic':False, 'atr_mult':3.0,
        'ts_act':best['ta_'], 'ts_trail':best['tt_'], 'ts_accel':False,
        'partial_exits':eval(best['pe_']) if best['pe_']!='OFF' else [],
        'reverse':'flip', 'time_sl':None,
        'h1_filter':False, 'h4_filter':False,
        'margin':best['m_'], 'leverage':best['lv_'], 're_entry_block':0,
    }
    df_best = tfs[best['tl']]

    print("\n  [10x Backtest]", flush=True)
    bals = []
    for run in range(1, 11):
        r = run_backtest(df_best, tfs.get('1h',df_best), df_best, best_params, CAPITAL)
        bals.append(r['bal'])
        print(f"  #{run:>2}: ${r['bal']:>11,.0f} PF={r['pf']:.1f} MDD={r['mdd']:.1f}% Tr={r['trades']} SL={r['sl']}", flush=True)
    print(f"  Std: {np.std(bals):.4f} -> {'PASS' if np.std(bals)<0.01 else 'FAIL'}", flush=True)

    print("\n  [10x Verify]", flush=True)
    ref = bals[0]; vpass = True
    for run in range(1, 11):
        r = run_backtest(df_best, tfs.get('1h',df_best), df_best, best_params, CAPITAL)
        ok = abs(r['bal'] - ref) < 0.01
        if not ok: vpass = False
        print(f"  #{run:>2}: ${r['bal']:>11,.0f} {'OK' if ok else 'MISMATCH'}", flush=True)
    print(f"  10x Verify: {'ALL PASS' if vpass else 'FAIL'}", flush=True)

    # Full monthly detail
    r_final = run_backtest(df_best, tfs.get('1h',df_best), df_best, best_params, CAPITAL)
    print_model(r_final, f"v24.2 WINNER Monthly Detail")

    # Save
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['trade_list','monthly']} for r in results[:200]]).to_csv(DATA_DIR/'v242_results.csv', index=False)
    all_trades = [t.copy() for t in r_final['trade_list']]
    pd.DataFrame(all_trades).to_csv(DATA_DIR/'v242_winner_trades.csv', index=False)

    print(f"\n{'='*90}", flush=True)
    print(f"  v24.2 FINAL", flush=True)
    print(f"  Combos tested: {len(results):,} (space: 2,177,280)", flush=True)
    print(f"  WINNER: ${r_final['bal']:,.0f} ({r_final['ret']:+,.1f}%)", flush=True)
    print(f"  PF={r_final['pf']:.1f} MDD={r_final['mdd']:.1f}% Trades={r_final['trades']} SL={r_final['sl']}", flush=True)
    print(f"  10x BT: {'PASS' if np.std(bals)<0.01 else 'FAIL'} | 10x Verify: {'PASS' if vpass else 'FAIL'}", flush=True)
    print(f"  Time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)", flush=True)
    print(f"{'='*90}", flush=True)

if __name__=='__main__':
    import traceback
    try: main()
    except Exception as e: traceback.print_exc(); print(f"ERROR: {e}", flush=True)
