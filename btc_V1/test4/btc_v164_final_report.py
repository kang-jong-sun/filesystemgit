"""
v16.4 Final Report: Top 3 strategies monthly detail + v16.0 baseline
"""
import pandas as pd, numpy as np, time, sys
from pathlib import Path
from btc_v164_backtest import (load_5m_data, resample, calc_wma, calc_ema,
                                calc_adx, calc_rsi, calc_atr, add_indicators,
                                run_backtest, print_model, FEE_RATE)

DATA_DIR = Path(r'D:\filesystem\futures\btc_V1\test4')

def main():
    t0 = time.time()
    print("=" * 90, flush=True)
    print("  v16.4 FINAL REPORT - Top 3 Strategies Monthly Detail", flush=True)
    print("=" * 90, flush=True)

    print("\n[DATA]", flush=True)
    df_5m = load_5m_data()
    df_30m = resample(df_5m, '30min')
    df_1h = resample(df_5m, '1h')
    df_4h = resample(df_5m, '4h')
    df_30m = add_indicators(df_30m)
    df_1h = add_indicators(df_1h)
    df_4h = add_indicators(df_4h)
    print(f"  Ready: {time.time()-t0:.1f}s", flush=True)

    TOTAL = 3000
    strategies = {
        'v16.0 Baseline (TS+4/-3, M50%)': {
            'fast_ma':'wma3','slow_ma':'ema200','adx_col':'adx20','adx_min':35,
            'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'sl_dynamic':False,'atr_mult':3.0,
            'ts_act':0.04,'ts_trail':0.03,'ts_accel':False,
            'partial_exits':[],'reverse':'flip','time_sl':None,
            'h1_filter':False,'h4_filter':False,'margin':0.50,'leverage':10,'re_entry_block':0,
        },
        '[1] Ultra-Safe (TS+3/-3, M15%)': {
            'fast_ma':'wma3','slow_ma':'ema200','adx_col':'adx20','adx_min':35,
            'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'sl_dynamic':False,'atr_mult':3.0,
            'ts_act':0.03,'ts_trail':0.03,'ts_accel':False,
            'partial_exits':[],'reverse':'flip','time_sl':None,
            'h1_filter':False,'h4_filter':False,'margin':0.15,'leverage':10,'re_entry_block':0,
        },
        '[2] Balanced (TS+4/-3, M30%)': {
            'fast_ma':'wma3','slow_ma':'ema200','adx_col':'adx20','adx_min':35,
            'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'sl_dynamic':False,'atr_mult':3.0,
            'ts_act':0.04,'ts_trail':0.03,'ts_accel':False,
            'partial_exits':[],'reverse':'flip','time_sl':None,
            'h1_filter':False,'h4_filter':False,'margin':0.30,'leverage':10,'re_entry_block':0,
        },
        '[3] Max-Return (TS+6/-3, M50%)': {
            'fast_ma':'wma3','slow_ma':'ema200','adx_col':'adx20','adx_min':35,
            'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'sl_dynamic':False,'atr_mult':3.0,
            'ts_act':0.06,'ts_trail':0.03,'ts_accel':False,
            'partial_exits':[],'reverse':'flip','time_sl':None,
            'h1_filter':False,'h4_filter':False,'margin':0.50,'leverage':10,'re_entry_block':0,
        },
        '[NEW] TS+3/-3 M50% (PF+Return)': {
            'fast_ma':'wma3','slow_ma':'ema200','adx_col':'adx20','adx_min':35,
            'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'sl_dynamic':False,'atr_mult':3.0,
            'ts_act':0.03,'ts_trail':0.03,'ts_accel':False,
            'partial_exits':[],'reverse':'flip','time_sl':None,
            'h1_filter':False,'h4_filter':False,'margin':0.50,'leverage':10,'re_entry_block':0,
        },
        '[NEW] TS+3/-2 M50% (Tight Trail)': {
            'fast_ma':'wma3','slow_ma':'ema200','adx_col':'adx20','adx_min':35,
            'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'sl_dynamic':False,'atr_mult':3.0,
            'ts_act':0.03,'ts_trail':0.02,'ts_accel':False,
            'partial_exits':[],'reverse':'flip','time_sl':None,
            'h1_filter':False,'h4_filter':False,'margin':0.50,'leverage':10,'re_entry_block':0,
        },
    }

    results = {}
    for name, params in strategies.items():
        print(f"\n  Running: {name}...", flush=True)
        t1 = time.time()
        r = run_backtest(df_30m, df_1h, df_4h, params, TOTAL)
        print(f"    Done: {time.time()-t1:.1f}s", flush=True)
        results[name] = r

    # Print comparison table
    print("\n" + "=" * 90, flush=True)
    print("  STRATEGY COMPARISON TABLE", flush=True)
    print("=" * 90, flush=True)
    print(f"  {'Strategy':<35} | {'PF':>5} | {'MDD':>6} | {'Ret%':>10} | {'$Final':>10} | {'Tr':>3} | {'SL':>2} | {'TSL':>3} | {'REV':>3} | {'WR':>5}", flush=True)
    print(f"  {'-'*35}-+-{'-'*5}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*3}-+-{'-'*2}-+-{'-'*3}-+-{'-'*3}-+-{'-'*5}", flush=True)
    for name, r in results.items():
        print(f"  {name:<35} | {r['pf']:>5.2f} | {r['mdd']:>5.1f}% | {r['ret']:>+9.1f}% | ${r['bal']:>9,.0f} | {r['trades']:>3} | {r['sl']:>2} | {r['tsl']:>3} | {r['rev']:>3} | {r['wr']:>4.1f}%", flush=True)

    # Print yearly comparison
    print("\n" + "=" * 90, flush=True)
    print("  YEARLY RETURN COMPARISON", flush=True)
    print("=" * 90, flush=True)
    names = list(results.keys())
    short_names = ['v16.0', 'Safe', 'Bal', 'MaxR', 'TS33', 'TS32']
    print(f"  {'Year':<6}", end="", flush=True)
    for sn in short_names: print(f" | {sn:>10}", end="", flush=True)
    print(flush=True)
    print(f"  {'-'*6}" + "".join(f"-+-{'-'*10}" for _ in short_names), flush=True)

    all_months = set()
    for r in results.values():
        all_months.update(r['monthly'].keys())
    years = sorted(set(m[:4] for m in all_months))

    for yr in years:
        print(f"  {yr:<6}", end="", flush=True)
        for name in names:
            mo = results[name]['monthly']
            yr_months = sorted([m for m in mo if m.startswith(yr)])
            if yr_months:
                ys = mo[yr_months[0]]['start_bal']
                ye = mo[yr_months[-1]]['end_bal']
                yr_ret = (ye-ys)/ys*100 if ys > 0 else 0
                print(f" | {yr_ret:>+9.1f}%", end="", flush=True)
            else:
                print(f" | {'N/A':>10}", end="", flush=True)
        print(flush=True)

    # Print full monthly detail for each strategy
    for name, r in results.items():
        print_model(r, name)

    # Save trade details
    all_trades = []
    for name, r in results.items():
        for t in r['trade_list']:
            t2 = t.copy(); t2['strategy'] = name; all_trades.append(t2)
    pd.DataFrame(all_trades).to_csv(DATA_DIR / 'v164_final_trades.csv', index=False)

    print(f"\n  Total time: {time.time()-t0:.0f}s", flush=True)
    print(f"  Trade details saved: v164_final_trades.csv ({len(all_trades)} trades)", flush=True)

if __name__ == '__main__':
    import traceback
    try: main()
    except Exception as e: traceback.print_exc(); print(f"ERROR: {e}", flush=True)
