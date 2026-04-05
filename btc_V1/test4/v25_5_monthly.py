"""v25.5 월별 상세 데이터 - Model A (50,982%) + Model B (4,482%)"""
import pandas as pd
import numpy as np
import time,os,sys,warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

from v27_backtest_engine import load_5m_data, map_tf_index
from v27_1_engine import build_extended_cache


def run_detailed(cache, cfg, tf_maps):
    ts_5m=cache['5m']['timestamp']; close=cache['5m']['close']; high=cache['5m']['high']; low=cache['5m']['low']
    n=len(close); ich=cache['5m']['ichimoku_cloud']; atr=cache['5m']['atr'][14]

    ctf=cfg['ctf']; c_data=cache[ctf]
    fm=c_data.get(cfg['ft'],c_data['ema']).get(cfg['fl']) if isinstance(c_data.get(cfg['ft']),dict) else c_data['ema'].get(cfg['fl'])
    sm=c_data.get(cfg['st'],c_data['ema']).get(cfg['sl_len']) if isinstance(c_data.get(cfg['st']),dict) else c_data['ema'].get(cfg['sl_len'])
    sig=np.zeros(len(fm),dtype=np.int64)
    for j in range(1,len(fm)):
        if fm[j]>sm[j]: sig[j]=1
        elif fm[j]<sm[j]: sig[j]=-1
    csig=sig if ctf=='5m' else sig[tf_maps[ctf]]

    atf=cfg['atf']; adx_v=cache[atf]['adx'][cfg['ap']]
    if atf!='5m': adx_v=adx_v[tf_maps[atf]]
    rtf=cfg['rtf']; rsi_v=cache[rtf]['rsi'][cfg['rp']]
    if rtf!='5m': rsi_v=rsi_v[tf_maps[rtf]]
    macd_v=cache['5m']['macd'][cfg['macd']]['hist']

    lev=cfg['lev']; fee_rate=0.0004; capital=5000.0
    position=0; entry_price=0.0; margin_used=0.0; peak_change=0.0; max_change=0.0
    trail_active=False; peak_capital=5000.0
    last_cross_bar=-9999; last_cross_dir=0; last_cross_price=0.0
    last_exit_bar=-9999; entry_bar=0
    trades=[]

    for i in range(300,n):
        price=close[i]; hi=high[i]; lo=low[i]
        dd=(capital-peak_capital)/peak_capital if peak_capital>0 else 0
        cm=cfg['mr'] if dd<-0.20 else cfg['mn']

        if csig[i]!=0 and csig[i]!=csig[max(0,i-1)]:
            last_cross_bar=i; last_cross_dir=csig[i]; last_cross_price=price

        if position!=0:
            if position==1:
                pc=(price-entry_price)/entry_price; mc=(hi-entry_price)/entry_price; mnc=(lo-entry_price)/entry_price
            else:
                pc=(entry_price-price)/entry_price; mc=(entry_price-lo)/entry_price; mnc=(entry_price-hi)/entry_price
            if mc>max_change: max_change=mc
            et=-1

            if mnc<=-cfg['sl_v']: et=0
            if et<0 and max_change>=cfg['ta']:
                trail_active=True
                if pc<=max_change-cfg['tp']: et=1

            # REV
            if et<0 and cfg['rev']>0:
                ns=csig[i]
                if ns!=0 and ns!=position:
                    a_ok=adx_v[i]>=cfg['amin']; r_ok=rsi_v[i]>=cfg['rmin'] and rsi_v[i]<=cfg['rmax']
                    m_ok=(ns==1 and macd_v[i]>0) or (ns==-1 and macd_v[i]<0)
                    ich_ok=True
                    if cfg['ichimoku']:
                        if ns==1: ich_ok=ich[i]>=0
                        else: ich_ok=ich[i]<=0
                    if a_ok and r_ok and m_ok and ich_ok: et=2

            if et>=0:
                if et==0: ac=-cfg['sl_v']
                elif et==1: ac=max_change-cfg['tp']
                else: ac=pc
                pnl=margin_used*lev*ac
                nom=margin_used*lev; fee=nom*fee_rate
                capital+=pnl-fee
                if capital<0: capital=0
                if capital>peak_capital: peak_capital=capital
                exit_ts=pd.Timestamp(ts_5m[i]); entry_ts=pd.Timestamp(ts_5m[entry_bar])
                trades.append({
                    'entry_time':entry_ts,'exit_time':exit_ts,
                    'direction':'long' if position==1 else 'short',
                    'entry_price':entry_price,'exit_price':price,
                    'price_change_pct':ac*100,'lev_roi_pct':ac*lev*100,
                    'pnl':pnl-fee,'exit_type':['SL','TSL','REV'][et],
                    'peak_change_pct':max_change*100,'balance':capital,
                    'year':exit_ts.year,'month':exit_ts.month,
                })
                last_exit_bar=i
                if et==2 and capital>0:
                    position=-position; entry_price=price
                    margin_used=capital*cm; fe=margin_used*lev*fee_rate; capital-=fe
                    margin_used=capital*cm; max_change=0; trail_active=False
                else:
                    position=0; entry_price=0; margin_used=0; max_change=0; trail_active=False

        if position==0 and last_cross_dir!=0 and capital>0:
            bs=i-last_cross_bar; be=i-last_exit_bar
            if 0<=bs<=cfg['ed'] and be>=cfg['mb']:
                pdiff=(price-last_cross_price)/last_cross_price*100
                eok=False
                if last_cross_dir==1 and -cfg['et']<=pdiff<=0.5: eok=True
                elif last_cross_dir==-1 and -0.5<=pdiff<=cfg['et']: eok=True
                if eok:
                    a_ok=adx_v[i]>=cfg['amin']; r_ok=rsi_v[i]>=cfg['rmin'] and rsi_v[i]<=cfg['rmax']
                    m_ok=(last_cross_dir==1 and macd_v[i]>0) or (last_cross_dir==-1 and macd_v[i]<0)
                    ich_ok=True
                    if cfg['ichimoku']:
                        if last_cross_dir==1: ich_ok=ich[i]>=0
                        else: ich_ok=ich[i]<=0
                    if a_ok and r_ok and m_ok and ich_ok:
                        position=last_cross_dir; entry_price=price; entry_bar=i
                        margin_used=capital*cm; fe=margin_used*lev*fee_rate; capital-=fe
                        margin_used=capital*cm; max_change=0; trail_active=False; last_cross_dir=0

    if position!=0 and capital>0:
        price=close[n-1]
        if position==1: pc=(price-entry_price)/entry_price
        else: pc=(entry_price-price)/entry_price
        pnl=margin_used*lev*pc; fee=margin_used*lev*fee_rate; capital+=pnl-fee
        trades.append({'entry_time':pd.Timestamp(ts_5m[entry_bar]),'exit_time':pd.Timestamp(ts_5m[n-1]),
            'direction':'long' if position==1 else 'short','entry_price':entry_price,'exit_price':price,
            'price_change_pct':pc*100,'lev_roi_pct':pc*lev*100,'pnl':pnl-fee,'exit_type':'END',
            'peak_change_pct':max_change*100,'balance':capital,'year':pd.Timestamp(ts_5m[n-1]).year,'month':pd.Timestamp(ts_5m[n-1]).month})
    return trades, capital


def print_all(trades, label, init_cap=5000):
    df=pd.DataFrame(trades)
    if len(df)==0: print(f"No trades"); return

    print(f"\n{'='*150}")
    print(f"  {label} - 전체 거래 내역 ({len(df)}건)")
    print(f"{'='*150}")
    print(f"| {'#':>3} | {'진입시간':^16} | {'청산시간':^16} | {'방향':^5} | {'진입가':>10} | {'청산가':>10} | {'가격%':>7} | {'레버ROI%':>9} | {'손익$':>14} | {'사유':^3} | {'최고%':>6} | {'잔액$':>14} |")
    print(f"|{'-'*5}|{'-'*18}|{'-'*18}|{'-'*7}|{'-'*12}|{'-'*12}|{'-'*9}|{'-'*11}|{'-'*16}|{'-'*5}|{'-'*8}|{'-'*16}|")
    for i,t in df.iterrows():
        print(f"| {i+1:>3} | {str(t['entry_time'])[:16]:^16} | {str(t['exit_time'])[:16]:^16} | "
              f"{'long' if t['direction']=='long' else 'short':^5} | {t['entry_price']:>10,.0f} | {t['exit_price']:>10,.0f} | "
              f"{t['price_change_pct']:>+6.2f}% | {t['lev_roi_pct']:>+8.1f}% | {t['pnl']:>+13,.0f} | {t['exit_type']:^3} | "
              f"{t['peak_change_pct']:>+5.1f}% | {t['balance']:>13,.0f} |")

    df['ym']=df['exit_time'].dt.to_period('M')
    all_months=pd.period_range('2020-01','2026-03',freq='M')

    print(f"\n{'='*150}")
    print(f"  {label} - 월별 상세 (75개월)")
    print(f"{'='*150}")
    print(f"| {'월':^8} | {'거래':^4} | {'승':^3} | {'패':^3} | {'승률':^6} | {'총이익$':^14} | {'총손실$':^14} | {'순손익$':^14} | {'누적$':^14} | {'PF':^6} | {'SL':^3} | {'TSL':^4} | {'REV':^4} |")
    print(f"|{'-'*10}|{'-'*6}|{'-'*5}|{'-'*5}|{'-'*8}|{'-'*16}|{'-'*16}|{'-'*16}|{'-'*16}|{'-'*8}|{'-'*5}|{'-'*6}|{'-'*6}|")

    cum=0; yearly={}
    for m in all_months:
        mt=df[df['ym']==m]; tc=len(mt); yr=m.year
        if yr not in yearly: yearly[yr]={'t':0,'w':0,'p':0,'l':0}
        if tc==0:
            print(f"| {str(m):^8} | {0:^4} | {'-':^3} | {'-':^3} | {'-':^6} | {'-':^14} | {'-':^14} | {'-':^14} | {cum:>+13,.0f} | {'-':^6} | {'-':^3} | {'-':^4} | {'-':^4} |")
        else:
            w=len(mt[mt['pnl']>0]); l=tc-w; wr=w/tc*100
            tp=mt[mt['pnl']>0]['pnl'].sum(); tl=mt[mt['pnl']<=0]['pnl'].sum()
            net=tp+tl; cum+=net; pf=tp/abs(tl) if tl!=0 else 999
            sl_c=len(mt[mt['exit_type']=='SL']); tsl_c=len(mt[mt['exit_type']=='TSL']); rev_c=len(mt[mt['exit_type']=='REV'])
            pfs=f"{pf:.2f}" if pf<100 else "INF"
            yearly[yr]['t']+=tc; yearly[yr]['w']+=w; yearly[yr]['p']+=tp; yearly[yr]['l']+=tl
            print(f"| {str(m):^8} | {tc:^4} | {w:^3} | {l:^3} | {wr:>5.1f}% | {tp:>+13,.0f} | {tl:>+13,.0f} | {net:>+13,.0f} | {cum:>+13,.0f} | {pfs:^6} | {sl_c:^3} | {tsl_c:^4} | {rev_c:^4} |")

    print(f"\n{'='*100}")
    print(f"  {label} - 연도별 요약")
    print(f"{'='*100}")
    print(f"| {'연도':^6} | {'거래':^4} | {'승':^3} | {'패':^3} | {'승률':^7} | {'총이익$':^14} | {'총손실$':^14} | {'순손익$':^14} | {'PF':^6} |")
    print(f"|{'-'*8}|{'-'*6}|{'-'*5}|{'-'*5}|{'-'*9}|{'-'*16}|{'-'*16}|{'-'*16}|{'-'*8}|")
    for yr in sorted(yearly.keys()):
        y=yearly[yr]; tc=y['t']; w=y['w']; l=tc-w; wr=w/tc*100 if tc>0 else 0
        net=y['p']+y['l']; pf=y['p']/abs(y['l']) if y['l']!=0 else 999
        pfs=f"{pf:.2f}" if pf<100 else "INF"
        print(f"| {yr:^6} | {tc:^4} | {w:^3} | {l:^3} | {wr:>6.1f}% | {y['p']:>+13,.0f} | {y['l']:>+13,.0f} | {net:>+13,.0f} | {pfs:^6} |")
    tt=len(df); tw=len(df[df['pnl']>0]); ttp=df[df['pnl']>0]['pnl'].sum()
    ttl=df[df['pnl']<=0]['pnl'].sum(); tpf=ttp/abs(ttl) if ttl!=0 else 999
    print(f"|{'─'*8}|{'─'*6}|{'─'*5}|{'─'*5}|{'─'*9}|{'─'*16}|{'─'*16}|{'─'*16}|{'─'*8}|")
    print(f"| {'합계':^5} | {tt:^4} | {tw:^3} | {tt-tw:^3} | {tw/tt*100:>6.1f}% | {ttp:>+13,.0f} | {ttl:>+13,.0f} | {ttp+ttl:>+13,.0f} | {tpf:.2f} |")
    print(f"\n  최종잔액: ${df.iloc[-1]['balance']:,.0f} | 수익률: +{(df.iloc[-1]['balance']-init_cap)/init_cap*100:,.0f}% | 거래 {tt}회")


def main():
    print("Loading...", flush=True)
    df_5m=load_5m_data()
    print("Building cache...", flush=True)
    cache=build_extended_cache(df_5m)
    tf_maps={}
    for tf in ['10m','15m','30m','1h']:
        tf_maps[tf]=map_tf_index(cache['5m']['timestamp'], cache[tf]['timestamp'])

    # Model A: 15m EMA3/EMA200, 15x, 30%, REV ON (정확한 최적화 결과)
    cfg_a={'ctf':'15m','ft':'ema','fl':3,'st':'ema','sl_len':200,
           'atf':'30m','ap':20,'amin':30,'rtf':'5m','rp':14,'rmin':30,'rmax':70,
           'macd':'8_21_5','ichimoku':False,'lev':15,
           'sl_v':0.03,'ta':0.04,'tp':0.02,'mn':0.30,'mr':0.15,
           'ed':6,'et':0.5,'rev':1,'mb':24}

    # Model B: 30m EMA7/EMA100, 15x, 30%, REV OFF (정확한 최적화 결과)
    cfg_b={'ctf':'30m','ft':'ema','fl':7,'st':'ema','sl_len':100,
           'atf':'15m','ap':20,'amin':30,'rtf':'10m','rp':14,'rmin':40,'rmax':75,
           'macd':'8_21_5','ichimoku':False,'lev':15,
           'sl_v':0.03,'ta':0.05,'tp':0.025,'mn':0.30,'mr':0.15,
           'ed':0,'et':0.5,'rev':0,'mb':1}

    print("\n\nModel A (15x, REV ON, 50,982%)...", flush=True)
    trades_a, cap_a = run_detailed(cache, cfg_a, tf_maps)
    print_all(trades_a, "v25.5 Model A: 15m EMA3/EMA200, 15x, 30%마진, REV ON")

    print("\n\nModel B (15x, REV OFF, PF 3.78)...", flush=True)
    trades_b, cap_b = run_detailed(cache, cfg_b, tf_maps)
    print_all(trades_b, "v25.5 Model B: 30m EMA7/EMA100, 15x, 30%마진, REV OFF")

    print("\n\nDONE.", flush=True)

if __name__=='__main__':
    main()
