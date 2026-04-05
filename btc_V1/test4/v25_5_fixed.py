"""
v25.5 Fixed - 레버리지 PnL 계산 버그 수정 + 최적화
==================================================
BUG: pnl = position_size * price * pnl_pct / leverage ← leverage 상쇄!
FIX: pnl = margin * leveraged_return (leverage 정상 적용)

실제 선물 거래:
- 마진(담보) = 자본 × 마진율
- 포지션 명목가 = 마진 × 레버리지
- 손익 = 포지션 명목가 × 가격변동% = 마진 × 레버리지 × 가격변동%
- 수수료 = 포지션 명목가 × 수수료율
"""
import pandas as pd
import numpy as np
from numba import njit
import time, os, json, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

from v27_backtest_engine import load_5m_data, map_tf_index
from v27_1_engine import build_extended_cache


@njit(cache=True)
def run_backtest_fixed(
    close_5m, high_5m, low_5m, ts_i64,
    cross_signal, adx_values, rsi_values, atr_values, macd_hist,
    ichimoku_cloud,
    adx_min, rsi_min, rsi_max,
    use_ichimoku,
    sl_pct, trail_activate, trail_pct,
    leverage, margin_pct, margin_reduced, dd_threshold,
    fee_rate, initial_capital,
    entry_delay_bars, entry_price_tol,
    rev_mode, min_bars_between,
):
    """레버리지 정상 적용 백테스트 엔진"""
    n = len(close_5m)
    capital = initial_capital
    position = 0
    entry_price = 0.0
    margin_used = 0.0  # 실제 투입 마진 ($)
    peak_roi = 0.0
    trail_active = False
    peak_capital = initial_capital
    last_exit_bar = -9999
    last_cross_bar = -9999
    last_cross_dir = 0
    last_cross_price = 0.0

    max_trades = 5000
    trade_roi = np.zeros(max_trades)
    trade_pnl = np.zeros(max_trades)
    trade_peak_roi = np.zeros(max_trades)
    trade_exit_type = np.zeros(max_trades, dtype=np.int64)
    trade_balance = np.zeros(max_trades)
    trade_dir = np.zeros(max_trades, dtype=np.int64)
    trade_count = 0

    eq_len = n // 288 + 2
    equity_curve = np.zeros(eq_len)
    eq_idx = 0

    for i in range(300, n):
        price = close_5m[i]
        hi = high_5m[i]
        lo = low_5m[i]

        dd = (capital - peak_capital) / peak_capital if peak_capital > 0 else 0.0
        current_margin = margin_reduced if dd < dd_threshold else margin_pct

        if cross_signal[i] != 0 and cross_signal[i] != cross_signal[i-1]:
            last_cross_bar = i
            last_cross_dir = cross_signal[i]
            last_cross_price = price

        # ---- POSITION MANAGEMENT ----
        if position != 0:
            # 가격 변동률 (순수)
            if position == 1:
                price_change = (price - entry_price) / entry_price
                max_change = (hi - entry_price) / entry_price
                min_change = (lo - entry_price) / entry_price
            else:
                price_change = (entry_price - price) / entry_price
                max_change = (entry_price - lo) / entry_price
                min_change = (entry_price - hi) / entry_price

            # 레버리지 적용 ROI (마진 대비 수익률)
            lev_roi = price_change * leverage
            max_lev_roi = max_change * leverage
            min_lev_roi = min_change * leverage

            if max_lev_roi > peak_roi:
                peak_roi = max_lev_roi

            exit_type = -1

            # SL: 가격이 sl_pct% 역행하면 발동
            # 실제 손실 = 마진 × sl_pct × leverage
            if min_change <= -sl_pct:
                exit_type = 0

            # TSL: 최고점에서 trail_pct% 하락하면 발동
            if exit_type < 0:
                if max_change >= trail_activate:
                    trail_active = True
                    if price_change <= max_change - trail_pct:
                        exit_type = 1

            # REV
            if exit_type < 0 and rev_mode > 0:
                new_sig = cross_signal[i]
                if new_sig != 0 and new_sig != position:
                    if rev_mode == 1:
                        a_ok = adx_values[i] >= adx_min
                        r_ok = rsi_values[i] >= rsi_min and rsi_values[i] <= rsi_max
                        m_ok = (new_sig == 1 and macd_hist[i] > 0) or (new_sig == -1 and macd_hist[i] < 0)
                        ich_ok = True
                        if use_ichimoku > 0:
                            if new_sig == 1: ich_ok = ichimoku_cloud[i] >= 0
                            else: ich_ok = ichimoku_cloud[i] <= 0
                        if a_ok and r_ok and m_ok and ich_ok:
                            exit_type = 2
                    elif rev_mode == 2:
                        a_ok = adx_values[i] >= adx_min * 0.7
                        m_ok = (new_sig == 1 and macd_hist[i] > 0) or (new_sig == -1 and macd_hist[i] < 0)
                        if a_ok and m_ok:
                            exit_type = 2

            if exit_type >= 0:
                # 실제 가격 변동률 결정
                if exit_type == 0:
                    actual_change = -sl_pct  # SL 발동 가격
                elif exit_type == 1:
                    actual_change = max_change - trail_pct  # TSL 발동 가격
                else:
                    actual_change = price_change  # 현재가

                # ★ 핵심: 레버리지 정상 적용 PnL ★
                # PnL = 마진 × 레버리지 × 가격변동률
                pnl_dollar = margin_used * leverage * actual_change

                # 수수료 = 포지션 명목가 × 수수료율 (진입+청산)
                nominal_value = margin_used * leverage
                fee = nominal_value * fee_rate  # 청산 수수료

                capital += pnl_dollar - fee
                if capital < 0:
                    capital = 0  # 파산 방지

                if capital > peak_capital:
                    peak_capital = capital

                if trade_count < max_trades:
                    trade_dir[trade_count] = position
                    trade_roi[trade_count] = actual_change * 100  # 순수 가격 변동%
                    trade_pnl[trade_count] = pnl_dollar - fee
                    trade_peak_roi[trade_count] = max_change * 100
                    trade_exit_type[trade_count] = exit_type
                    trade_balance[trade_count] = capital
                    trade_count += 1

                last_exit_bar = i

                if exit_type == 2 and capital > 0:
                    position = -position
                    entry_price = price
                    margin_used = capital * current_margin
                    fee_entry = margin_used * leverage * fee_rate
                    capital -= fee_entry
                    margin_used = capital * current_margin  # 수수료 후 재계산
                    peak_roi = 0.0
                    trail_active = False
                else:
                    position = 0
                    entry_price = 0.0
                    margin_used = 0.0
                    peak_roi = 0.0
                    trail_active = False

        # ---- ENTRY ----
        if position == 0 and last_cross_dir != 0 and capital > 0:
            bars_since = i - last_cross_bar
            bars_since_exit = i - last_exit_bar

            if 0 <= bars_since <= entry_delay_bars and bars_since_exit >= min_bars_between:
                pdiff = (price - last_cross_price) / last_cross_price * 100
                entry_ok = False
                if last_cross_dir == 1 and -entry_price_tol <= pdiff <= 0.5:
                    entry_ok = True
                elif last_cross_dir == -1 and -0.5 <= pdiff <= entry_price_tol:
                    entry_ok = True

                if entry_ok:
                    a_ok = adx_values[i] >= adx_min
                    r_ok = rsi_values[i] >= rsi_min and rsi_values[i] <= rsi_max
                    m_ok = (last_cross_dir == 1 and macd_hist[i] > 0) or (last_cross_dir == -1 and macd_hist[i] < 0)
                    ich_ok = True
                    if use_ichimoku > 0:
                        if last_cross_dir == 1: ich_ok = ichimoku_cloud[i] >= 0
                        else: ich_ok = ichimoku_cloud[i] <= 0

                    if a_ok and r_ok and m_ok and ich_ok:
                        position = last_cross_dir
                        entry_price = price
                        margin_used = capital * current_margin
                        # 진입 수수료
                        fee_entry = margin_used * leverage * fee_rate
                        capital -= fee_entry
                        margin_used = capital * current_margin  # 수수료 후
                        peak_roi = 0.0
                        trail_active = False
                        last_cross_dir = 0

        if i % 288 == 0 and eq_idx < eq_len:
            equity_curve[eq_idx] = capital
            eq_idx += 1

    # Close open
    if position != 0 and capital > 0:
        price = close_5m[n-1]
        if position == 1: pc = (price - entry_price) / entry_price
        else: pc = (entry_price - price) / entry_price
        pnl = margin_used * leverage * pc
        fee = margin_used * leverage * fee_rate
        capital += pnl - fee
        if trade_count < max_trades:
            trade_dir[trade_count] = position
            trade_roi[trade_count] = pc * 100
            trade_pnl[trade_count] = pnl - fee
            trade_balance[trade_count] = capital
            trade_exit_type[trade_count] = 1
            trade_count += 1

    return (
        capital, trade_count,
        trade_dir[:trade_count], trade_roi[:trade_count],
        trade_pnl[:trade_count], trade_peak_roi[:trade_count],
        trade_exit_type[:trade_count], trade_balance[:trade_count],
        equity_curve[:eq_idx], peak_capital
    )


def main():
    print("="*70, flush=True)
    print("v25.5 FIXED - LEVERAGE BUG CORRECTED", flush=True)
    print("$5,000 | Lev 10-15x | Margin ≤30%", flush=True)
    print("="*70, flush=True)

    df_5m = load_5m_data()
    print("\nBuilding cache...", flush=True)
    cache = build_extended_cache(df_5m)

    ts_5m = cache['5m']['timestamp']
    close_5m = cache['5m']['close']
    high_5m = cache['5m']['high']
    low_5m = cache['5m']['low']
    n = len(close_5m)
    ts_i64 = ts_5m.astype('int64')

    tf_maps = {}
    for tf in ['10m','15m','30m','1h']:
        tf_maps[tf] = map_tf_index(ts_5m, cache[tf]['timestamp'])

    # JIT warmup
    nw = 1000
    _ = run_backtest_fixed(
        close_5m[:nw],high_5m[:nw],low_5m[:nw],ts_i64[:nw],
        np.zeros(nw,dtype=np.int64),np.zeros(nw),np.zeros(nw),np.zeros(nw),np.zeros(nw),np.zeros(nw),
        25,30,70,0,0.05,0.05,0.03,10,0.30,0.15,-0.20,0.0004,5000.0,6,1.0,0,6
    )
    print("JIT done.", flush=True)

    ich=cache['5m']['ichimoku_cloud']
    atr_v=cache['5m']['atr'][14]

    cross_configs = [
        ('30m','ema',3,'ema',200, '15m',20, '5m',14),
        ('30m','ema',3,'ema',150, '15m',20, '5m',14),
        ('30m','ema',3,'ema',100, '15m',20, '5m',14),
        ('30m','ema',5,'ema',200, '15m',20, '5m',14),
        ('30m','ema',5,'ema',100, '15m',20, '5m',14),
        ('30m','ema',7,'ema',100, '15m',20, '10m',14),
        ('30m','hma',9,'ema',200, '15m',20, '5m',14),
        ('30m','hma',14,'ema',200,'15m',20, '5m',14),
        ('10m','ema',5,'ema',300, '15m',20, '5m',14),
        ('10m','ema',5,'ema',200, '15m',20, '5m',14),
        ('15m','ema',3,'ema',200, '30m',20, '5m',14),
    ]

    cross_sigs={}; adx_m={}; rsi_m={}
    for ctf,ft,fl,st,sl_len,atf,ap,rtf,rp in cross_configs:
        label=f"{ctf}_{ft}{fl}_{st}{sl_len}"
        c_data=cache[ctf]
        fm=c_data.get(ft,c_data['ema']).get(fl) if isinstance(c_data.get(ft),dict) else c_data['ema'].get(fl)
        sm=c_data.get(st,c_data['ema']).get(sl_len) if isinstance(c_data.get(st),dict) else c_data['ema'].get(sl_len)
        if fm is None or sm is None: continue
        sig=np.zeros(len(fm),dtype=np.int64)
        for j in range(1,len(fm)):
            if fm[j]>sm[j]: sig[j]=1
            elif fm[j]<sm[j]: sig[j]=-1
        cross_sigs[label]=sig if ctf=='5m' else sig[tf_maps[ctf]]
        ak=f"{atf}_{ap}"
        if ak not in adx_m:
            v=cache[atf]['adx'][ap]; adx_m[ak]=v if atf=='5m' else v[tf_maps[atf]]
        rk=f"{rtf}_{rp}"
        if rk not in rsi_m:
            v=cache[rtf]['rsi'][rp]; rsi_m[rk]=v if rtf=='5m' else v[tf_maps[rtf]]

    macd_m={mk:cache['5m']['macd'][mk]['hist'] for mk in ['5_35_5','8_21_5','12_26_9']}

    # Optimization grid
    leverages = [10, 12, 15]
    adx_mins = [25, 30, 35, 40]
    rsi_ranges = [(30,70),(35,70),(35,75),(40,75)]
    macd_keys = ['5_35_5','8_21_5','12_26_9']
    ich_opts = [0, 1]

    sl_trail_by_lev = {
        10: [(0.05,0.06,0.03),(0.07,0.07,0.03),(0.07,0.08,0.03),(0.07,0.10,0.04)],
        12: [(0.04,0.05,0.025),(0.05,0.06,0.03),(0.05,0.07,0.03),(0.07,0.07,0.03)],
        15: [(0.03,0.04,0.02),(0.03,0.05,0.025),(0.04,0.05,0.025),(0.05,0.06,0.03)],
    }
    margins = [(0.20,0.10),(0.25,0.12),(0.30,0.15)]
    entry_delays = [0, 6, 12]
    entry_tols = [0.5, 1.0]
    rev_modes = [0, 1]
    min_bars = [1, 12, 24]

    total = 0
    for lev in leverages:
        total += (len(cross_configs)*len(adx_mins)*len(rsi_ranges)*len(macd_keys)*
                  len(ich_opts)*len(sl_trail_by_lev[lev])*len(margins)*
                  len(entry_delays)*len(entry_tols)*len(rev_modes)*len(min_bars))
    print(f"\n  Total: {total:,}", flush=True)

    all_results=[]
    tested=0; t0=time.time()

    for lev in leverages:
        sl_trails=sl_trail_by_lev[lev]
        print(f"\n  === Leverage {lev}x ===", flush=True)

        for ctf,ft,fl,st,sl_len,atf,ap,rtf,rp in cross_configs:
            label=f"{ctf}_{ft}{fl}_{st}{sl_len}"
            if label not in cross_sigs: continue
            csig=cross_sigs[label]
            adx_val=adx_m[f"{atf}_{ap}"]
            rsi_val=rsi_m[f"{rtf}_{rp}"]

            for amin in adx_mins:
                for rmin,rmax in rsi_ranges:
                    for mk in macd_keys:
                        mv=macd_m[mk]
                        for ich_use in ich_opts:
                            for sl,ta,tp in sl_trails:
                                for mn,mr in margins:
                                    for ed in entry_delays:
                                        for et in entry_tols:
                                            for rm in rev_modes:
                                                for mb in min_bars:
                                                    tested+=1
                                                    result=run_backtest_fixed(
                                                        close_5m,high_5m,low_5m,ts_i64,
                                                        csig,adx_val,rsi_val,atr_v,mv,ich,
                                                        amin,rmin,rmax,ich_use,
                                                        sl,ta,tp,
                                                        lev,mn,mr,-0.20,0.0004,5000.0,
                                                        ed,et,rm,mb
                                                    )
                                                    fc,tc=result[0],result[1]
                                                    if tc>=10 and fc>10000:
                                                        rois=result[3];pnls=result[4]
                                                        wins=np.sum(pnls>0)
                                                        tpro=np.sum(pnls[pnls>0])
                                                        tlos=abs(np.sum(pnls[pnls<0]))+1e-10
                                                        pf=tpro/tlos
                                                        ret=(fc-5000)/5000*100
                                                        eq=result[8];mdd=0
                                                        if len(eq)>0:
                                                            peq=np.maximum.accumulate(eq)
                                                            dd=(eq-peq)/(peq+1e-10)
                                                            mdd=abs(np.min(dd))*100
                                                        etypes=result[6]
                                                        if mdd<65:
                                                            all_results.append({
                                                                'cross':label,'atf':atf,'ap':ap,'rtf':rtf,'rp':rp,
                                                                'adx_min':amin,'rsi_min':rmin,'rsi_max':rmax,
                                                                'macd':mk,'ichimoku':ich_use,
                                                                'sl':sl,'trail_act':ta,'trail_pct':tp,
                                                                'leverage':lev,'margin':mn,'margin_dd':mr,
                                                                'entry_delay':ed,'entry_tol':et,
                                                                'rev_mode':rm,'min_bars':mb,
                                                                'final_cap':float(fc),'trades':int(tc),
                                                                'win_rate':float(wins/tc*100),
                                                                'pf':float(pf),'mdd':float(mdd),
                                                                'return_pct':float(ret),
                                                                'avg_win':float(np.mean(rois[pnls>0])) if wins>0 else 0,
                                                                'avg_loss':float(np.mean(rois[pnls<=0])) if (tc-wins)>0 else 0,
                                                                'sl_c':int(np.sum(etypes==0)),
                                                                'tsl_c':int(np.sum(etypes==1)),
                                                                'rev_c':int(np.sum(etypes==2)),
                                                            })
                                                    if tested%100000==0:
                                                        elapsed=time.time()-t0
                                                        rate=tested/elapsed
                                                        rem=(total-tested)/rate/60 if rate>0 else 999
                                                        top_ret=max([r['return_pct'] for r in all_results]) if all_results else 0
                                                        top_fc=max([r['final_cap'] for r in all_results]) if all_results else 0
                                                        pf5=len([r for r in all_results if r['pf']>=5])
                                                        print(f"  {tested:,}/{total:,} ({tested/total*100:.1f}%) | {rate:.0f}/s | ~{rem:.0f}min | {len(all_results)} passed | TopRet:{top_ret:,.0f}% Top$:{top_fc:,.0f} PF5:{pf5}", flush=True)

    elapsed=time.time()-t0
    print(f"\n  Done: {tested:,} in {elapsed:.0f}s", flush=True)

    by_ret=sorted(all_results, key=lambda x: x['return_pct'], reverse=True)
    pf5=[r for r in all_results if r['pf']>=5]
    pf3=[r for r in all_results if r['pf']>=3]
    print(f"  Total passed: {len(all_results)} | PF>=3: {len(pf3)} | PF>=5: {len(pf5)}", flush=True)

    print(f"\n  === Top 15 by RETURN ===", flush=True)
    for i,r in enumerate(by_ret[:15]):
        print(f"  #{i+1}: {r['cross']} Lev:{r['leverage']}x ADX>{r['adx_min']} RSI:{r['rsi_min']}-{r['rsi_max']} ICH:{r['ichimoku']} "
              f"SL:{r['sl']*100:.0f}% T+{r['trail_act']*100:.0f}%/-{r['trail_pct']*100:.0f}% M:{r['margin']*100:.0f}% "
              f"REV:{r['rev_mode']} | Ret:{r['return_pct']:,.0f}% PF:{r['pf']:.2f} MDD:{r['mdd']:.1f}% T:{r['trades']} ${r['final_cap']:,.0f}", flush=True)

    # Select
    model_a = by_ret[0] if by_ret else None
    good_pf = [r for r in all_results if r['pf']>=3]
    model_b = max(good_pf, key=lambda x: x['return_pct']) if good_pf else by_ret[1] if len(by_ret)>1 else model_a
    model_c = max(all_results[:50], key=lambda x: x['pf']/(x['mdd']+3)) if all_results else model_a

    models={'A':model_a,'B':model_b,'C':model_c}
    for name,m in models.items():
        if m is None: continue
        print(f"\n  Model {name}: {m['cross']} Lev:{m['leverage']}x M:{m['margin']*100:.0f}%", flush=True)
        print(f"    Ret:{m['return_pct']:,.0f}% PF:{m['pf']:.2f} MDD:{m['mdd']:.1f}% T:{m['trades']} WR:{m['win_rate']:.1f}%", flush=True)
        print(f"    Final:${m['final_cap']:,.0f} | SL:{m['sl_c']} TSL:{m['tsl_c']} REV:{m['rev_c']}", flush=True)

    # 30x validation
    print("\n[30x Validation]", flush=True)
    validation={}
    for name,m in models.items():
        if m is None: continue
        csig=cross_sigs[m['cross']]
        adx_val=adx_m[f"{m['atf']}_{m['ap']}"]
        rsi_val=rsi_m[f"{m['rtf']}_{m['rp']}"]
        mv=macd_m[m['macd']]
        vals=[]
        for run in range(30):
            s=int(n*(0.02+(run/30)*0.13))
            ml=min(n-s,len(csig)-s,len(adx_val)-s,len(rsi_val)-s)
            r=run_backtest_fixed(
                close_5m[s:s+ml],high_5m[s:s+ml],low_5m[s:s+ml],ts_i64[s:s+ml],
                csig[s:s+ml],adx_val[s:s+ml],rsi_val[s:s+ml],atr_v[s:s+ml],mv[s:s+ml],ich[s:s+ml],
                m['adx_min'],m['rsi_min'],m['rsi_max'],m['ichimoku'],
                m['sl'],m['trail_act'],m['trail_pct'],
                m['leverage'],m['margin'],m['margin_dd'],-0.20,0.0004,5000.0,
                m['entry_delay'],m['entry_tol'],m['rev_mode'],m['min_bars']
            )
            fc,tc=r[0],r[1];pnls=r[4]
            w=np.sum(pnls>0) if tc>0 else 0
            pf=np.sum(pnls[pnls>0])/(abs(np.sum(pnls[pnls<0]))+1e-10)
            eq=r[8];mdd=0
            if len(eq)>0:peq=np.maximum.accumulate(eq);dd=(eq-peq)/(peq+1e-10);mdd=abs(np.min(dd))*100
            vals.append({'run':run+1,'fc':float(fc),'ret':float((fc-5000)/5000*100),
                        'trades':int(tc),'pf':float(pf),'mdd':float(mdd)})
        validation[name]=vals
        rets=[v['ret'] for v in vals];pfs=[v['pf'] for v in vals];fcs=[v['fc'] for v in vals]
        print(f"  {name}: Ret={np.mean(rets):,.0f}%+/-{np.std(rets):,.0f}% PF={np.mean(pfs):.2f} MDD={np.mean([v['mdd'] for v in vals]):.1f}%", flush=True)
        print(f"      Min${np.min(fcs):,.0f} Max${np.max(fcs):,.0f} MinRet:{np.min(rets):,.0f}%", flush=True)

    # Save
    output={'version':'v25.5_fixed','initial_capital':5000,
            'total_tested':tested,'bug_fix':'leverage PnL calculation corrected',
            'models':{n:{k:v for k,v in m.items()} for n,m in models.items() if m},
            'validation':validation,
            'top20':[{k:v for k,v in r.items()} for r in by_ret[:20]]}
    def conv(o):
        if isinstance(o,(np.integer,)):return int(o)
        if isinstance(o,(np.floating,)):return float(o)
        if isinstance(o,np.ndarray):return o.tolist()
        return o
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'v25_5_fixed_results.json')
    with open(path,'w',encoding='utf-8') as f:
        json.dump(output,f,indent=2,default=conv,ensure_ascii=False)
    print(f"\nSaved: {path}\n{'='*70}\nCOMPLETE", flush=True)

if __name__=='__main__':
    main()
