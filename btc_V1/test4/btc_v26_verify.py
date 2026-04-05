"""
v26.0 Verification: Test key strategies from v22/v23/v25 + our v17 findings
Using the proven v16.4 backtest engine
"""
import pandas as pd, numpy as np, time, sys
from pathlib import Path
from btc_v164_backtest import (load_5m_data, resample, calc_wma, calc_ema, calc_sma,
                                calc_adx, calc_rsi, calc_atr, add_indicators,
                                run_backtest, print_model, FEE_RATE)

DATA_DIR = Path(r'D:\filesystem\futures\btc_V1\test4')

def calc_hma(s, period):
    h = max(int(period/2),1); sq = max(int(np.sqrt(period)),1)
    return calc_wma(2*calc_wma(s,h)-calc_wma(s,period), sq)

def main():
    print("="*90, flush=True)
    print("  v26.0 Key Strategy Verification + Monthly Detail", flush=True)
    print("="*90, flush=True)
    t0 = time.time()

    df_5m = load_5m_data()
    # Build TFs
    tfs = {}
    for rule, label in [('30min','30m'),('60min','1h')]:
        df = resample(df_5m, rule)
        df = add_indicators(df)
        df['hma3'] = calc_hma(df['close'], 3)
        df['hma5'] = calc_hma(df['close'], 5)
        df['wma7'] = calc_wma(df['close'], 7)
        df['ema5'] = calc_ema(df['close'], 5)
        df['ema7'] = calc_ema(df['close'], 7)
        df['ema250'] = calc_ema(df['close'], 250)
        df['ema300'] = calc_ema(df['close'], 300)
        df['sma250'] = calc_sma(df['close'], 250)
        df['sma300'] = calc_sma(df['close'], 300)
        df['ema100_s'] = calc_ema(df['close'], 100)
        df['ema150'] = calc_ema(df['close'], 150)
        df['wma300'] = calc_wma(df['close'], 300)
        tfs[label] = df
        print(f"  {label}: {len(df):,} candles", flush=True)
    print(f"  Ready: {time.time()-t0:.1f}s\n", flush=True)

    TOTAL = 3000
    strategies = {
        # === v16.0/v16.4 PROVEN BASELINES ===
        'v16.0 WMA3/EMA200 M50%': {
            'fast_ma':'wma3','slow_ma':'ema200','adx_col':'adx20','adx_min':35,
            'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'sl_dynamic':False,'atr_mult':3.0,
            'ts_act':0.04,'ts_trail':0.03,'ts_accel':False,
            'partial_exits':[],'reverse':'flip','time_sl':None,
            'h1_filter':False,'h4_filter':False,'margin':0.50,'leverage':10,'re_entry_block':0,
        },
        # === v17 SCAN BEST: High-Trade (51tr, PF 9.4) ===
        'v17 WMA3/EMA200 RSI30-70 TS32 M30%': {
            'fast_ma':'wma3','slow_ma':'ema200','adx_col':'adx20','adx_min':35,
            'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'sl_dynamic':False,'atr_mult':3.0,
            'ts_act':0.03,'ts_trail':0.02,'ts_accel':False,
            'partial_exits':[],'reverse':'flip','time_sl':None,
            'h1_filter':False,'h4_filter':False,'margin':0.30,'leverage':10,'re_entry_block':0,
        },
        # === v17 SCAN BEST: High-PF (PF 39.4) ===
        'v17 EMA3/SMA300 ADX40 M50%': {
            'fast_ma':'ema3','slow_ma':'sma300','adx_col':'adx20','adx_min':40,
            'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'sl_dynamic':False,'atr_mult':3.0,
            'ts_act':0.04,'ts_trail':0.03,'ts_accel':False,
            'partial_exits':[],'reverse':'flip','time_sl':None,
            'h1_filter':False,'h4_filter':False,'margin':0.50,'leverage':10,'re_entry_block':0,
        },
        # === 1h EMA3/EMA100 (high return from v17 scan) ===
        '1h EMA3/EMA100 ADX30 TS65 M50%': {
            'fast_ma':'ema3','slow_ma':'ema100_s','adx_col':'adx20','adx_min':30,
            'rsi_min':30,'rsi_max':70,'sl_pct':0.07,'sl_dynamic':False,'atr_mult':3.0,
            'ts_act':0.06,'ts_trail':0.05,'ts_accel':False,
            'partial_exits':[],'reverse':'flip','time_sl':None,
            'h1_filter':False,'h4_filter':False,'margin':0.50,'leverage':10,'re_entry_block':0,
        },
        # === v23 Tier2 HMA3/EMA250 ADX45 ===
        'v23 HMA3/EMA250 ADX45 TS32 M30%': {
            'fast_ma':'hma3','slow_ma':'ema250','adx_col':'adx14','adx_min':45,
            'rsi_min':35,'rsi_max':70,'sl_pct':0.08,'sl_dynamic':False,'atr_mult':3.0,
            'ts_act':0.03,'ts_trail':0.02,'ts_accel':False,
            'partial_exits':[],'reverse':'flip','time_sl':None,
            'h1_filter':False,'h4_filter':False,'margin':0.30,'leverage':10,'re_entry_block':0,
        },
        # === NEW: WMA3/EMA200 RSI30-70 TS43 M50% (high trade + high M) ===
        'WMA3/EMA200 RSI30-70 TS43 M50%': {
            'fast_ma':'wma3','slow_ma':'ema200','adx_col':'adx20','adx_min':35,
            'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'sl_dynamic':False,'atr_mult':3.0,
            'ts_act':0.04,'ts_trail':0.03,'ts_accel':False,
            'partial_exits':[],'reverse':'flip','time_sl':None,
            'h1_filter':False,'h4_filter':False,'margin':0.50,'leverage':10,'re_entry_block':0,
        },
        # === NEW: EMA3/SMA300 ADX40 RSI30-70 M50% (wider RSI) ===
        'EMA3/SMA300 ADX40 RSI30-70 M50%': {
            'fast_ma':'ema3','slow_ma':'sma300','adx_col':'adx20','adx_min':40,
            'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'sl_dynamic':False,'atr_mult':3.0,
            'ts_act':0.04,'ts_trail':0.03,'ts_accel':False,
            'partial_exits':[],'reverse':'flip','time_sl':None,
            'h1_filter':False,'h4_filter':False,'margin':0.50,'leverage':10,'re_entry_block':0,
        },
        # === NEW: HMA5/EMA200 ADX35 TS43 M50% ===
        'HMA5/EMA200 ADX35 TS43 M50%': {
            'fast_ma':'hma5','slow_ma':'ema200','adx_col':'adx20','adx_min':35,
            'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'sl_dynamic':False,'atr_mult':3.0,
            'ts_act':0.04,'ts_trail':0.03,'ts_accel':False,
            'partial_exits':[],'reverse':'flip','time_sl':None,
            'h1_filter':False,'h4_filter':False,'margin':0.50,'leverage':10,'re_entry_block':0,
        },
    }

    # Run all on 30m (except 1h strategy)
    results = {}
    for name, params in strategies.items():
        tf = '1h' if name.startswith('1h') else '30m'
        df = tfs[tf]
        print(f"  {name}...", flush=True)
        r = run_backtest(df, tfs['1h'], tfs['30m'], params, TOTAL)
        results[name] = r

    # Comparison table
    print("\n" + "="*90, flush=True)
    print("  STRATEGY COMPARISON", flush=True)
    print("="*90, flush=True)
    print(f"  {'Strategy':<38} | {'PF':>5} | {'MDD':>6} | {'Ret%':>10} | {'$':>10} | {'Tr':>3} | {'SL':>2} | {'WR':>5}", flush=True)
    print(f"  {'-'*38}-+-{'-'*5}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*3}-+-{'-'*2}-+-{'-'*5}", flush=True)
    for name, r in results.items():
        print(f"  {name:<38} | {r['pf']:>5.1f} | {r['mdd']:>5.1f}% | {r['ret']:>+9.1f}% | ${r['bal']:>9,.0f} | {r['trades']:>3} | {r['sl']:>2} | {r['wr']:>4.1f}%", flush=True)

    # Monthly detail for top strategies
    for name in list(results.keys()):
        print_model(results[name], name)

    # Save
    all_trades = []
    for nm, r in results.items():
        for t in r['trade_list']:
            t2 = t.copy(); t2['strategy'] = nm; all_trades.append(t2)
    pd.DataFrame(all_trades).to_csv(DATA_DIR / 'v26_trades.csv', index=False)
    print(f"\n  Total time: {time.time()-t0:.0f}s", flush=True)
    print(f"  Saved: v26_trades.csv ({len(all_trades)} trades)", flush=True)

if __name__ == '__main__':
    import traceback
    try: main()
    except Exception as e: traceback.print_exc(); print(f"ERROR: {e}", flush=True)
