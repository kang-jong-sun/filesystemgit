"""v27.1 월별 상세 데이터 추출 - Model A (30% 마진) + Model B (20% 마진)"""
import pandas as pd
import numpy as np
import time, os, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

from v27_backtest_engine import load_5m_data, map_tf_index
from v27_1_engine import build_extended_cache, run_backtest_v271


def run_detailed(cache, cfg, label):
    ts_5m = cache['5m']['timestamp']
    close_5m = cache['5m']['close']
    high_5m = cache['5m']['high']
    low_5m = cache['5m']['low']
    n = len(close_5m)
    tf_maps = {}
    for tf in ['10m','15m','30m','1h']:
        tf_maps[tf] = map_tf_index(ts_5m, cache[tf]['timestamp'])

    ctf = cfg['ctf']; c_data = cache[ctf]
    fm = c_data[cfg['ft']][cfg['fl']]; sm = c_data[cfg['st']][cfg['sl_len']]
    sig = np.zeros(len(fm), dtype=np.int64)
    for j in range(1,len(fm)):
        if fm[j]>sm[j]: sig[j]=1
        elif fm[j]<sm[j]: sig[j]=-1
    csig = sig if ctf=='5m' else sig[tf_maps[ctf]]

    atf=cfg['atf']; adx_v=cache[atf]['adx'][cfg['ap']]
    if atf!='5m': adx_v=adx_v[tf_maps[atf]]
    rtf=cfg['rtf']; rsi_v=cache[rtf]['rsi'][cfg['rp']]
    if rtf!='5m': rsi_v=rsi_v[tf_maps[rtf]]
    macd_v=cache['5m']['macd'][cfg['macd']]['hist']
    atr_v=cache['5m']['atr'][14]
    sk=cache['5m']['stoch_k']; sd=cache['5m']['stoch_d']
    cci=cache['5m']['cci']; obv=cache['5m']['obv_slope']
    bbw=cache['5m']['bb_width']; ich=cache['5m']['ichimoku_cloud']

    # Manual backtest for trade-level detail
    leverage=10; fee_rate=0.0004; capital=3000.0
    position=0; entry_price=0.0; position_size=0.0; peak_roi=0.0
    trail_active=False; peak_capital=3000.0
    last_cross_bar=-9999; last_cross_dir=0; last_cross_price=0.0
    last_exit_bar=-9999; entry_bar=0
    trades=[]

    for i in range(300, n):
        price=close_5m[i]; hi=high_5m[i]; lo=low_5m[i]
        dd=(capital-peak_capital)/peak_capital if peak_capital>0 else 0
        cm=cfg['mr'] if dd<-0.20 else cfg['mn']

        if csig[i]!=0 and csig[i]!=csig[max(0,i-1)]:
            last_cross_bar=i; last_cross_dir=csig[i]; last_cross_price=price

        if position!=0:
            if position==1:
                cr=(price-entry_price)/entry_price*leverage
                mx=(hi-entry_price)/entry_price*leverage
                mn_r=(lo-entry_price)/entry_price*leverage
            else:
                cr=(entry_price-price)/entry_price*leverage
                mx=(entry_price-lo)/entry_price*leverage
                mn_r=(entry_price-hi)/entry_price*leverage
            if mx>peak_roi: peak_roi=mx
            asl=-cfg['sl_v']*leverage; ta_v=cfg['ta']*leverage; tp_v=cfg['tp']*leverage
            et=-1

            if mn_r<=asl: et=0
            if et<0 and peak_roi>=ta_v:
                trail_active=True
                if cr<=peak_roi-tp_v: et=1

            # NO REV (rev_mode=0)

            if et>=0:
                if et==0: pp=asl
                elif et==1: pp=peak_roi-tp_v
                else: pp=cr
                fee=abs(position_size*price*fee_rate)
                pnl=position_size*price*pp/leverage-fee
                capital+=pnl
                if capital>peak_capital: peak_capital=capital
                exit_ts=pd.Timestamp(ts_5m[i]); entry_ts=pd.Timestamp(ts_5m[entry_bar])
                trades.append({
                    'entry_time':entry_ts,'exit_time':exit_ts,
                    'direction':'long' if position==1 else 'short',
                    'entry_price':entry_price,'exit_price':price,
                    'roi_pct':pp/leverage*100,'pnl':pnl,
                    'exit_type':['SL','TSL','REV'][et],
                    'peak_roi_pct':peak_roi/leverage*100,'balance':capital,
                    'year':exit_ts.year,'month':exit_ts.month,
                })
                last_exit_bar=i; position=0; entry_price=0; peak_roi=0; trail_active=False

        if position==0 and last_cross_dir!=0:
            bs=i-last_cross_bar; be=i-last_exit_bar
            if 0<=bs<=0 and be>=48:  # entry_delay=0, min_bars=48
                pd_pct=(price-last_cross_price)/last_cross_price*100
                eok=False
                if last_cross_dir==1 and -0.5<=pd_pct<=0.5: eok=True
                elif last_cross_dir==-1 and -0.5<=pd_pct<=0.5: eok=True

                if eok:
                    a_ok=adx_v[i]>=cfg['amin']
                    r_ok=rsi_v[i]>=cfg['rmin'] and rsi_v[i]<=cfg['rmax']
                    m_ok=(last_cross_dir==1 and macd_v[i]>0) or (last_cross_dir==-1 and macd_v[i]<0)
                    ich_ok=True
                    if cfg['ichimoku']:
                        if last_cross_dir==1: ich_ok=ich[i]>=0
                        else: ich_ok=ich[i]<=0

                    if a_ok and r_ok and m_ok and ich_ok:
                        position=last_cross_dir; entry_price=price; entry_bar=i
                        pv=capital*cm; position_size=pv/price
                        fe=position_size*price*fee_rate; capital-=fe
                        peak_roi=0; trail_active=False; last_cross_dir=0

    if position!=0:
        price=close_5m[n-1]
        if position==1: cr=(price-entry_price)/entry_price*leverage
        else: cr=(entry_price-price)/entry_price*leverage
        fee=abs(position_size*price*fee_rate)
        pnl=position_size*price*cr/leverage-fee; capital+=pnl
        trades.append({
            'entry_time':pd.Timestamp(ts_5m[entry_bar]),'exit_time':pd.Timestamp(ts_5m[n-1]),
            'direction':'long' if position==1 else 'short',
            'entry_price':entry_price,'exit_price':price,
            'roi_pct':cr/leverage*100,'pnl':pnl,'exit_type':'END',
            'peak_roi_pct':peak_roi/leverage*100,'balance':capital,
            'year':pd.Timestamp(ts_5m[n-1]).year,'month':pd.Timestamp(ts_5m[n-1]).month,
        })
    return trades, capital


def print_all(trades, label):
    df=pd.DataFrame(trades)
    if len(df)==0:
        print(f"  No trades for {label}"); return

    # Print individual trades
    print(f"\n{'='*140}")
    print(f"  {label} - 전체 거래 내역")
    print(f"{'='*140}")
    print(f"| {'#':>3} | {'진입시간':^19} | {'청산시간':^19} | {'방향':^5} | {'진입가':>10} | {'청산가':>10} | {'ROI%':>8} | {'손익$':>12} | {'사유':^4} | {'최고ROI%':>8} | {'잔액$':>12} |")
    print(f"|{'-'*5}|{'-'*21}|{'-'*21}|{'-'*7}|{'-'*12}|{'-'*12}|{'-'*10}|{'-'*14}|{'-'*6}|{'-'*10}|{'-'*14}|")

    for i, t in df.iterrows():
        print(f"| {i+1:>3} | {str(t['entry_time'])[:16]:^19} | {str(t['exit_time'])[:16]:^19} | "
              f"{'long' if t['direction']=='long' else 'short':^5} | "
              f"{t['entry_price']:>10,.0f} | {t['exit_price']:>10,.0f} | "
              f"{t['roi_pct']:>+7.2f}% | {t['pnl']:>+11,.0f} | {t['exit_type']:^4} | "
              f"{t['peak_roi_pct']:>+7.2f}% | {t['balance']:>11,.0f} |")

    # Monthly summary
    df['ym'] = df['exit_time'].dt.to_period('M')
    all_months = pd.period_range('2020-01','2026-03',freq='M')

    print(f"\n{'='*140}")
    print(f"  {label} - 월별 상세 데이터 (75개월)")
    print(f"{'='*140}")
    print(f"| {'월':^8} | {'거래':^4} | {'승':^3} | {'패':^3} | {'승률':^6} | {'총이익$':^12} | {'총손실$':^12} | {'순손익$':^12} | {'누적$':^12} | {'PF':^6} | {'SL':^3} | {'TSL':^4} | {'Avg Win%':^9} | {'Avg Loss%':^9} |")
    print(f"|{'-'*10}|{'-'*6}|{'-'*5}|{'-'*5}|{'-'*8}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*8}|{'-'*5}|{'-'*6}|{'-'*11}|{'-'*11}|")

    cum=0; yearly={}
    for m in all_months:
        mt=df[df['ym']==m]; tc=len(mt)
        yr=m.year
        if yr not in yearly: yearly[yr]={'t':0,'w':0,'p':0,'l':0}

        if tc==0:
            print(f"| {str(m):^8} | {0:^4} | {'-':^3} | {'-':^3} | {'-':^6} | {'-':^12} | {'-':^12} | {'-':^12} | {cum:>+11,.0f} | {'-':^6} | {'-':^3} | {'-':^4} | {'-':^9} | {'-':^9} |")
        else:
            w=len(mt[mt['pnl']>0]); l=tc-w; wr=w/tc*100
            tp=mt[mt['pnl']>0]['pnl'].sum(); tl=mt[mt['pnl']<=0]['pnl'].sum()
            net=tp+tl; cum+=net
            pf=tp/abs(tl) if tl!=0 else 999
            sl_c=len(mt[mt['exit_type']=='SL']); tsl_c=len(mt[mt['exit_type']=='TSL'])
            aw=mt[mt['pnl']>0]['roi_pct'].mean() if w>0 else 0
            al=mt[mt['pnl']<=0]['roi_pct'].mean() if l>0 else 0
            pfs=f"{pf:.2f}" if pf<100 else "INF"

            yearly[yr]['t']+=tc; yearly[yr]['w']+=w
            yearly[yr]['p']+=tp; yearly[yr]['l']+=tl

            print(f"| {str(m):^8} | {tc:^4} | {w:^3} | {l:^3} | {wr:>5.1f}% | {tp:>+11,.0f} | {tl:>+11,.0f} | {net:>+11,.0f} | {cum:>+11,.0f} | {pfs:^6} | {sl_c:^3} | {tsl_c:^4} | {aw:>+8.1f}% | {al:>+8.1f}% |")

    # Yearly
    print(f"\n{'='*100}")
    print(f"  {label} - 연도별 요약")
    print(f"{'='*100}")
    print(f"| {'연도':^6} | {'거래':^4} | {'승':^3} | {'패':^3} | {'승률':^7} | {'총이익$':^12} | {'총손실$':^12} | {'순손익$':^12} | {'PF':^6} |")
    print(f"|{'-'*8}|{'-'*6}|{'-'*5}|{'-'*5}|{'-'*9}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*8}|")
    for yr in sorted(yearly.keys()):
        y=yearly[yr]; tc=y['t']; w=y['w']; l=tc-w; wr=w/tc*100 if tc>0 else 0
        net=y['p']+y['l']; pf=y['p']/abs(y['l']) if y['l']!=0 else 999
        pfs=f"{pf:.2f}" if pf<100 else "INF"
        print(f"| {yr:^6} | {tc:^4} | {w:^3} | {l:^3} | {wr:>6.1f}% | {y['p']:>+11,.0f} | {y['l']:>+11,.0f} | {net:>+11,.0f} | {pfs:^6} |")

    tt=len(df); tw=len(df[df['pnl']>0]); ttp=df[df['pnl']>0]['pnl'].sum()
    ttl=df[df['pnl']<=0]['pnl'].sum(); tpf=ttp/abs(ttl) if ttl!=0 else 999
    print(f"|{'─'*8}|{'─'*6}|{'─'*5}|{'─'*5}|{'─'*9}|{'─'*14}|{'─'*14}|{'─'*14}|{'─'*8}|")
    print(f"| {'합계':^5} | {tt:^4} | {tw:^3} | {tt-tw:^3} | {tw/tt*100:>6.1f}% | {ttp:>+11,.0f} | {ttl:>+11,.0f} | {ttp+ttl:>+11,.0f} | {tpf:.2f} |")
    print(f"\n  최종 잔액: ${df.iloc[-1]['balance']:,.0f} | 수익률: {(df.iloc[-1]['balance']-3000)/3000*100:,.1f}% | 거래 {tt}회")


def main():
    print("Loading...", flush=True)
    df_5m = load_5m_data()
    print("Building cache...", flush=True)
    cache = build_extended_cache(df_5m)

    # Model A: v25.2_A + Ichimoku + 30% margin + NO REV
    cfg_a = {'ctf':'30m','ft':'ema','fl':3,'st':'ema','sl_len':200,
             'atf':'15m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':40,'rmax':75,
             'macd':'5_35_5','sl_v':0.07,'ta':0.07,'tp':0.03,
             'mn':0.30,'mr':0.15,'ichimoku':True}

    # Model B: same + 20% margin
    cfg_b = {'ctf':'30m','ft':'ema','fl':3,'st':'ema','sl_len':200,
             'atf':'15m','ap':20,'amin':35,'rtf':'5m','rp':14,'rmin':40,'rmax':75,
             'macd':'5_35_5','sl_v':0.07,'ta':0.07,'tp':0.03,
             'mn':0.20,'mr':0.10,'ichimoku':True}

    print("\n\nModel A (30% margin)...", flush=True)
    trades_a, cap_a = run_detailed(cache, cfg_a, "Model A")
    print_all(trades_a, "v27.1 Model A (PF 5.05, MDD 6.1%, Margin 30%)")

    print("\n\nModel B (20% margin)...", flush=True)
    trades_b, cap_b = run_detailed(cache, cfg_b, "Model B")
    print_all(trades_b, "v27.1 Model B (PF 5.18, MDD 4.1%, Margin 20%)")

    print("\n\nDONE.", flush=True)

if __name__ == '__main__':
    main()
