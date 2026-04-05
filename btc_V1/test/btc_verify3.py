"""
검증3: v26.0 Dual-Track 전략 10회 백테스트 + 10회 검증
Track A: EMA(3)/SMA(300), ADX(20)>=40, RSI 30-70, TS+4/-3, M50%
Track B: WMA(3)/EMA(200), ADX(20)>=35, RSI 30-70, TS+3/-2, M30%
"""
import pandas as pd, numpy as np, time, sys, os
from pathlib import Path

# Fix path for import
sys.path.insert(0, str(Path(r'D:\filesystem\futures\btc_V1\test')))
from btc_v164_backtest import (load_5m_data, resample, calc_wma, calc_ema, calc_sma,
                                calc_adx, calc_rsi, calc_atr, add_indicators,
                                run_backtest, print_model, FEE_RATE)

DATA_DIR = Path(r'D:\filesystem\futures\btc_V1\test')
TOTAL = 3000

def main():
    print("="*90, flush=True)
    print("  [검증3] v26.0 Dual-Track 10회 백테스트 + 10회 검증", flush=True)
    print("="*90, flush=True)
    t0 = time.time()

    # Override DATA_DIR in the imported module
    import btc_v164_backtest as bt
    bt.DATA_DIR = DATA_DIR

    print("\n[DATA]", flush=True)
    df_5m = load_5m_data()
    df_30m = resample(df_5m, '30min')
    df_1h = resample(df_5m, '1h')
    df_30m = add_indicators(df_30m)
    df_1h = add_indicators(df_1h)
    # Extra MAs
    df_30m['sma300'] = calc_sma(df_30m['close'], 300)
    df_30m['ema250'] = calc_ema(df_30m['close'], 250)
    print(f"  30m: {len(df_30m):,}, 1h: {len(df_1h):,} candles", flush=True)
    print(f"  Ready: {time.time()-t0:.1f}s\n", flush=True)

    # Strategy definitions
    track_a_params = {
        'fast_ma':'ema3','slow_ma':'sma300','adx_col':'adx20','adx_min':40,
        'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'sl_dynamic':False,'atr_mult':3.0,
        'ts_act':0.04,'ts_trail':0.03,'ts_accel':False,
        'partial_exits':[],'reverse':'flip','time_sl':None,
        'h1_filter':False,'h4_filter':False,'margin':0.50,'leverage':10,'re_entry_block':0,
    }
    track_b_params = {
        'fast_ma':'wma3','slow_ma':'ema200','adx_col':'adx20','adx_min':35,
        'rsi_min':30,'rsi_max':70,'sl_pct':0.08,'sl_dynamic':False,'atr_mult':3.0,
        'ts_act':0.03,'ts_trail':0.02,'ts_accel':False,
        'partial_exits':[],'reverse':'flip','time_sl':None,
        'h1_filter':False,'h4_filter':False,'margin':0.30,'leverage':10,'re_entry_block':0,
    }

    # ============================================================
    # PHASE 1: 10회 백테스트
    # ============================================================
    print("="*90, flush=True)
    print("  PHASE 1: 10회 백테스트 실행", flush=True)
    print("="*90, flush=True)

    a_results = []; b_results = []
    for run in range(1, 11):
        ra = run_backtest(df_30m, df_1h, df_30m, track_a_params, TOTAL * 0.60)  # $1,800
        rb = run_backtest(df_30m, df_1h, df_30m, track_b_params, TOTAL * 0.40)  # $1,200
        a_results.append(ra)
        b_results.append(rb)
        combined = ra['bal'] + rb['bal']
        print(f"  Run #{run:>2}: A=${ra['bal']:>9,.0f}(PF{ra['pf']:>5.1f}) B=${rb['bal']:>9,.0f}(PF{rb['pf']:>5.1f}) | Combined=${combined:>10,.0f}", flush=True)

    # Summary stats
    a_bals = [r['bal'] for r in a_results]
    b_bals = [r['bal'] for r in b_results]
    c_bals = [a+b for a,b in zip(a_bals, b_bals)]

    print(f"\n  [10회 백테스트 통계]", flush=True)
    print(f"  {'':>12} | {'평균':>12} | {'표준편차':>10} | {'최소':>12} | {'최대':>12} | {'동일여부':>8}", flush=True)
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}", flush=True)
    for nm, vals in [('Track A', a_bals), ('Track B', b_bals), ('Combined', c_bals)]:
        avg = np.mean(vals); std = np.std(vals); mn = min(vals); mx = max(vals)
        same = "PASS" if std < 0.01 else "FAIL"
        print(f"  {nm:>12} | ${avg:>11,.0f} | {std:>9.4f} | ${mn:>11,.0f} | ${mx:>11,.0f} | {same:>8}", flush=True)

    # ============================================================
    # PHASE 2: 10회 검증 (동일 데이터, 결정론적 확인)
    # ============================================================
    print(f"\n{'='*90}", flush=True)
    print("  PHASE 2: 10회 검증 (결정론적 확인)", flush=True)
    print("="*90, flush=True)

    verify_pass = True
    ref_a = a_results[0]['bal']; ref_b = b_results[0]['bal']
    for run in range(1, 11):
        ra = run_backtest(df_30m, df_1h, df_30m, track_a_params, TOTAL * 0.60)
        rb = run_backtest(df_30m, df_1h, df_30m, track_b_params, TOTAL * 0.40)
        a_match = abs(ra['bal'] - ref_a) < 0.01
        b_match = abs(rb['bal'] - ref_b) < 0.01
        status = "PASS" if (a_match and b_match) else "FAIL"
        if not (a_match and b_match): verify_pass = False
        print(f"  Verify #{run:>2}: A=${ra['bal']:>9,.0f} {'OK' if a_match else 'MISMATCH'} | B=${rb['bal']:>9,.0f} {'OK' if b_match else 'MISMATCH'} | {status}", flush=True)

    print(f"\n  10회 검증 결과: {'*** ALL PASS ***' if verify_pass else '*** FAIL ***'}", flush=True)

    # ============================================================
    # PHASE 3: 상세 결과 리포트
    # ============================================================
    print(f"\n{'='*90}", flush=True)
    print("  PHASE 3: 상세 결과 리포트", flush=True)
    print("="*90, flush=True)

    ra = a_results[0]; rb = b_results[0]
    combined_bal = ra['bal'] + rb['bal']
    combined_ret = (combined_bal - TOTAL) / TOTAL * 100

    print(f"\n  [v26.0 Dual-Track 최종 성과]", flush=True)
    print(f"  초기자본: ${TOTAL:,.0f} (A:${TOTAL*0.6:,.0f} + B:${TOTAL*0.4:,.0f})", flush=True)
    print(f"  최종잔액: ${combined_bal:,.0f} ({combined_ret:+,.1f}%)", flush=True)
    print(f"", flush=True)
    print(f"  {'트랙':<15} | {'자본':>8} | {'최종잔액':>10} | {'수익률':>10} | {'PF':>6} | {'MDD':>6} | {'거래':>4} | {'SL':>3} | {'승률':>5}", flush=True)
    print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*4}-+-{'-'*3}-+-{'-'*5}", flush=True)
    print(f"  {'A Sniper':<15} | ${TOTAL*0.6:>7,.0f} | ${ra['bal']:>9,.0f} | {ra['ret']:>+9.1f}% | {ra['pf']:>5.1f} | {ra['mdd']:>5.1f}% | {ra['trades']:>4} | {ra['sl']:>3} | {ra['wr']:>4.1f}%", flush=True)
    print(f"  {'B Compounder':<15} | ${TOTAL*0.4:>7,.0f} | ${rb['bal']:>9,.0f} | {rb['ret']:>+9.1f}% | {rb['pf']:>5.1f} | {rb['mdd']:>5.1f}% | {rb['trades']:>4} | {rb['sl']:>3} | {rb['wr']:>4.1f}%", flush=True)
    print(f"  {'Combined':<15} | ${TOTAL:>7,.0f} | ${combined_bal:>9,.0f} | {combined_ret:>+9.1f}% |       |       |      |     |", flush=True)

    # Yearly combined
    print(f"\n  [연도별 복합 성과]", flush=True)
    all_months = sorted(set(list(ra['monthly'].keys()) + list(rb['monthly'].keys())))
    years = sorted(set(m[:4] for m in all_months))

    print(f"  {'연도':>6} | {'A잔액':>10} | {'B잔액':>10} | {'합계':>12} | {'A거래':>4} | {'B거래':>4} | {'합계거래':>6} | {'A-SL':>4} | {'B-SL':>4}", flush=True)
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*4}-+-{'-'*4}-+-{'-'*6}-+-{'-'*4}-+-{'-'*4}", flush=True)

    for yr in years:
        yr_m_a = [m for m in all_months if m.startswith(yr) and m in ra['monthly']]
        yr_m_b = [m for m in all_months if m.startswith(yr) and m in rb['monthly']]
        a_end = ra['monthly'][yr_m_a[-1]]['end_bal'] if yr_m_a else 0
        b_end = rb['monthly'][yr_m_b[-1]]['end_bal'] if yr_m_b else 0
        a_tr = sum(ra['monthly'][m]['trades'] for m in yr_m_a)
        b_tr = sum(rb['monthly'][m]['trades'] for m in yr_m_b)
        a_sl = sum(ra['monthly'][m]['sl'] for m in yr_m_a)
        b_sl = sum(rb['monthly'][m]['sl'] for m in yr_m_b)
        print(f"  {yr:>6} | ${a_end:>9,.0f} | ${b_end:>9,.0f} | ${a_end+b_end:>11,.0f} | {a_tr:>4} | {b_tr:>4} | {a_tr+b_tr:>6} | {a_sl:>4} | {b_sl:>4}", flush=True)

    # Monthly detail for both tracks
    print_model(ra, "Track A 'Sniper' EMA3/SMA300 ADX>=40 M50% ($1,800)")
    print_model(rb, "Track B 'Compounder' WMA3/EMA200 RSI30-70 TS32 M30% ($1,200)")

    # Combined monthly
    print(f"\n  [복합 월별 상세]", flush=True)
    print(f"  {'월':>8} | {'A잔액':>10} | {'B잔액':>10} | {'합계':>12} | {'월수익%':>8}", flush=True)
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}", flush=True)
    prev_total = TOTAL
    cur_yr = None
    for mk in all_months:
        yr = mk[:4]
        a_eb = ra['monthly'].get(mk, {}).get('end_bal', 0)
        b_eb = rb['monthly'].get(mk, {}).get('end_bal', 0)
        if mk not in ra['monthly']:
            prev_a = [m for m in sorted(ra['monthly'].keys()) if m < mk]
            a_eb = ra['monthly'][prev_a[-1]]['end_bal'] if prev_a else TOTAL*0.6
        if mk not in rb['monthly']:
            prev_b = [m for m in sorted(rb['monthly'].keys()) if m < mk]
            b_eb = rb['monthly'][prev_b[-1]]['end_bal'] if prev_b else TOTAL*0.4
        total_eb = a_eb + b_eb
        m_ret = (total_eb - prev_total)/prev_total*100 if prev_total > 0 else 0
        if yr != cur_yr:
            if cur_yr: print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}", flush=True)
            cur_yr = yr
        note = ' ++' if m_ret >= 20 else (' --' if m_ret <= -10 else '')
        print(f"  {mk:>8} | ${a_eb:>9,.0f} | ${b_eb:>9,.0f} | ${total_eb:>11,.0f} | {m_ret:>+7.1f}%{note}", flush=True)
        prev_total = total_eb

    # Save
    all_trades = []
    for t in ra['trade_list']:
        t2 = t.copy(); t2['track'] = 'A'; all_trades.append(t2)
    for t in rb['trade_list']:
        t2 = t.copy(); t2['track'] = 'B'; all_trades.append(t2)
    pd.DataFrame(all_trades).to_csv(DATA_DIR / 'verify3_trades.csv', index=False)

    print(f"\n{'='*90}", flush=True)
    print(f"  [검증3 최종 결과]", flush=True)
    print(f"  10회 백테스트: {'ALL IDENTICAL (std=0.0000)' if np.std(c_bals) < 0.01 else 'VARIANCE DETECTED'}", flush=True)
    print(f"  10회 검증: {'ALL PASS' if verify_pass else 'FAIL'}", flush=True)
    print(f"  Track A: ${ra['bal']:,.0f} | PF {ra['pf']:.1f} | MDD {ra['mdd']:.1f}% | {ra['trades']}tr | SL {ra['sl']}", flush=True)
    print(f"  Track B: ${rb['bal']:,.0f} | PF {rb['pf']:.1f} | MDD {rb['mdd']:.1f}% | {rb['trades']}tr | SL {rb['sl']}", flush=True)
    print(f"  Combined: ${combined_bal:,.0f} ({combined_ret:+,.1f}%)", flush=True)
    print(f"  Total time: {time.time()-t0:.0f}s", flush=True)
    print(f"  Saved: verify3_trades.csv ({len(all_trades)} trades)", flush=True)
    print(f"{'='*90}", flush=True)

if __name__ == '__main__':
    import traceback
    try: main()
    except Exception as e: traceback.print_exc(); print(f"ERROR: {e}", flush=True)
