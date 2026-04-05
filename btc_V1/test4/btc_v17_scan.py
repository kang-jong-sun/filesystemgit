"""
v17.0 Quick Scan - Uses PROVEN v16.4 run_backtest engine
Tests multiple TFs, HMA discovery (v16.5), delayed entry (v16.6)
"""
import pandas as pd, numpy as np, time, sys, itertools
from pathlib import Path
from btc_v164_backtest import (load_5m_data, resample, calc_wma, calc_ema, calc_sma,
                                calc_adx, calc_rsi, calc_atr, run_backtest, print_model,
                                add_indicators, FEE_RATE)

DATA_DIR = Path(r'D:\filesystem\futures\btc_V1\test4')

def calc_hma(s, period):
    h = max(int(period/2),1); sq = max(int(np.sqrt(period)),1)
    wh = calc_wma(s,h); wf = calc_wma(s,period)
    return calc_wma(2*wh - wf, sq)

def main():
    print("="*90, flush=True)
    print("  v17.0 Multi-TF Scan (using proven v16.4 engine)", flush=True)
    print("="*90, flush=True)
    t0 = time.time()

    df_5m = load_5m_data()

    # Build multiple TFs
    tfs = {}
    for rule, label in [('30min','30m'),('60min','1h')]:
        df = resample(df_5m, rule)
        df = add_indicators(df)
        # Add extra MAs
        df['hma5'] = calc_hma(df['close'], 5)
        df['hma7'] = calc_hma(df['close'], 7)
        df['hma14'] = calc_hma(df['close'], 14)
        df['hma21'] = calc_hma(df['close'], 21)
        df['wma4'] = calc_wma(df['close'], 4)
        df['wma5'] = calc_wma(df['close'], 5)
        df['wma7'] = calc_wma(df['close'], 7)
        df['ema5'] = calc_ema(df['close'], 5)
        df['ema250'] = calc_ema(df['close'], 250)
        df['ema300'] = calc_ema(df['close'], 300)
        df['sma250'] = calc_sma(df['close'], 250)
        df['sma300'] = calc_sma(df['close'], 300)
        df['ema100_s'] = calc_ema(df['close'], 100)
        df['ema150'] = calc_ema(df['close'], 150)
        tfs[label] = df
        print(f"  {label}: {len(df):,} candles", flush=True)

    print(f"  Ready: {time.time()-t0:.1f}s\n", flush=True)

    # Build test matrix using proven v16.4 run_backtest
    # Each combo: (name, tf_label, params_dict)
    TOTAL = 3000
    tests = []

    for tf_label in ['30m', '1h']:
        for fast_col in ['wma3','wma4','wma5','wma7','ema3','ema5','hma5','hma7','hma14','hma21']:
            for slow_col in ['ema200','ema250','ema300','ema150','sma250','sma300','ema100_s']:
                for adx_col in ['adx20']:
                    for adx_min in [30, 35, 40]:
                        for rsi_min, rsi_max in [(35,65),(30,70)]:
                            for sl, ta, tt in [(0.08,0.04,0.03),(0.07,0.06,0.05),(0.08,0.03,0.02),(0.06,0.08,0.03)]:
                                for margin in [0.30, 0.50]:
                                    nm = f"{tf_label}_{fast_col}/{slow_col}_ADX{adx_min}_RSI{rsi_min}{rsi_max}_SL{int(sl*100)}TS{int(ta*100)}{int(tt*100)}_M{int(margin*100)}"
                                    tests.append((nm, tf_label, {
                                        'fast_ma':fast_col,'slow_ma':slow_col,
                                        'adx_col':adx_col,'adx_min':adx_min,
                                        'rsi_min':rsi_min,'rsi_max':rsi_max,
                                        'sl_pct':sl,'sl_dynamic':False,'atr_mult':3.0,
                                        'ts_act':ta,'ts_trail':tt,'ts_accel':False,
                                        'partial_exits':[],'reverse':'flip','time_sl':None,
                                        'h1_filter':False,'h4_filter':False,
                                        'margin':margin,'leverage':10,'re_entry_block':0,
                                    }))

    print(f"  Total tests: {len(tests):,}", flush=True)

    # Run all
    results = []
    done = 0
    for nm, tf_label, params in tests:
        df = tfs[tf_label]
        dummy_1h = tfs.get('1h', df)
        dummy_4h = df  # not used (filters off)
        r = run_backtest(df, dummy_1h, dummy_4h, params, TOTAL)
        r['name'] = nm
        r['tf'] = tf_label
        r['fast'] = params['fast_ma']
        r['slow'] = params['slow_ma']
        r['adx_min'] = params['adx_min']
        r['rsi_r'] = f"{params['rsi_min']}-{params['rsi_max']}"
        r['sl_pct'] = params['sl_pct']
        r['ts_act'] = params['ts_act']
        r['ts_trail'] = params['ts_trail']
        r['margin_pct'] = params['margin']
        results.append(r)
        done += 1
        if done % 1000 == 0:
            print(f"    {done:,}/{len(tests):,} ({done/len(tests)*100:.0f}%) {time.time()-t0:.0f}s", flush=True)

    print(f"  Done: {len(tests):,} in {time.time()-t0:.0f}s", flush=True)

    # Filter valid
    valid = [r for r in results if r['pf']>=2.0 and r['trades']>=10 and r.get('fl',0)==0]
    valid.sort(key=lambda x: x['pf']*(1-x['mdd']/100)*np.log(x['trades']+1), reverse=True)
    print(f"  Valid (PF>=2, tr>=10): {len(valid)}", flush=True)

    # Print results
    hdr = f"  {'#':>2} | {'TF':>3} | {'Fast':>6} | {'Slow':>7} | {'ADX':>3} | {'RSI':>5} | {'SL':>3} | {'TS':>5} | {'M':>3} | {'PF':>6} | {'MDD':>5} | {'Ret%':>9} | {'$':>9} | {'Tr':>3} | {'SL':>2} | {'WR':>5}"
    sep = f"  {'-'*2}-+-{'-'*3}-+-{'-'*6}-+-{'-'*7}-+-{'-'*3}-+-{'-'*5}-+-{'-'*3}-+-{'-'*5}-+-{'-'*3}-+-{'-'*6}-+-{'-'*5}-+-{'-'*9}-+-{'-'*9}-+-{'-'*3}-+-{'-'*2}-+-{'-'*5}"
    def prow(i,r):
        print(f"  {i:>2} | {r['tf']:>3} | {r['fast']:>6} | {r['slow']:>7} | {r['adx_min']:>3} | {r['rsi_r']:>5} | {int(r['sl_pct']*100):>2}% | {int(r['ts_act']*100)}/-{int(r['ts_trail']*100)} | {int(r['margin_pct']*100):>2}% | {r['pf']:>5.1f} | {r['mdd']:>4.1f}% | {r['ret']:>+8.1f}% | ${r['bal']:>8,.0f} | {r['trades']:>3} | {r['sl']:>2} | {r['wr']:>4.1f}%", flush=True)

    print(f"\n  [TOP 30 by Score]", flush=True)
    print(hdr, flush=True); print(sep, flush=True)
    for i,r in enumerate(valid[:30]): prow(i+1, r)

    # Top by PF
    pf_top = [r for r in valid if r['trades']>=15]
    pf_top.sort(key=lambda x: x['pf'], reverse=True)
    print(f"\n  [TOP 20 by PF (trades>=15)]", flush=True)
    print(hdr, flush=True); print(sep, flush=True)
    for i,r in enumerate(pf_top[:20]): prow(i+1, r)

    # Top by trades with good PF
    tr_top = [r for r in valid if r['pf']>=5]
    tr_top.sort(key=lambda x: x['trades'], reverse=True)
    print(f"\n  [TOP 20 by Trade Count (PF>=5)]", flush=True)
    print(hdr, flush=True); print(sep, flush=True)
    for i,r in enumerate(tr_top[:20]): prow(i+1, r)

    # Top by return
    ret_top = sorted(valid, key=lambda x: x['bal'], reverse=True)
    print(f"\n  [TOP 20 by Return]", flush=True)
    print(hdr, flush=True); print(sep, flush=True)
    for i,r in enumerate(ret_top[:20]): prow(i+1, r)

    # Lowest MDD
    mdd_top = [r for r in valid if r['ret']>300]
    mdd_top.sort(key=lambda x: x['mdd'])
    print(f"\n  [TOP 20 by Lowest MDD (ret>300%)]", flush=True)
    print(hdr, flush=True); print(sep, flush=True)
    for i,r in enumerate(mdd_top[:20]): prow(i+1, r)

    pd.DataFrame([{k:v for k,v in r.items() if k not in ['trade_list','monthly']} for r in results]).to_csv(DATA_DIR/'v17_scan_results.csv', index=False)
    print(f"\n  Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)", flush=True)
    print(f"  Saved: v17_scan_results.csv", flush=True)

    # Print monthly detail for top 3
    print(f"\n  [Monthly Detail for Top 3]", flush=True)
    for i,r in enumerate(valid[:3]):
        print_model(r, f"#{i+1}: {r['name']}")

if __name__ == '__main__':
    import traceback
    try: main()
    except Exception as e: traceback.print_exc(); print(f"ERROR: {e}", flush=True)
