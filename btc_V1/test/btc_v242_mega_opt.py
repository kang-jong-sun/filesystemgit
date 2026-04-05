"""
v24.2 MEGA OPTIMIZER - Maximum Return Search
Stage 1: Entry scan (300 combos)
Stage 2: Exit+Margin+Leverage grid (top5 x 3000 = 15,000)
Stage 3: Partial Exit fine-tune (top30 x PE configs)
Stage 4: 10x Backtest + 10x Verify for #1
"""
import pandas as pd, numpy as np, time, sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(r'D:\filesystem\futures\btc_V1\test')))
from btc_v164_backtest import (load_5m_data, resample, calc_wma, calc_ema, calc_sma,
                                calc_adx, calc_rsi, calc_atr, add_indicators,
                                run_backtest, print_model, FEE_RATE)
DATA_DIR = Path(r'D:\filesystem\futures\btc_V1\test')
import btc_v164_backtest as bt
bt.DATA_DIR = DATA_DIR
CAPITAL = 3000

def calc_hma(s, p):
    h=max(int(p/2),1); sq=max(int(np.sqrt(p)),1)
    return calc_wma(2*calc_wma(s,h)-calc_wma(s,p), sq)

def fast_bt(C,H,L,fma,sma,adx,rsi, am,rmin,rmax, sl,ta,tt,margin,lev,
            fee=FEE_RATE, capital=CAPITAL):
    n=len(C); bal=float(capital); pd_=0; pepx=0.; psz=0.; ppk=0.
    phi=0.; plo=0.; pf_=np.nan; ps_=np.nan
    pkb=bal; mdd=0.; tp=0.; tl=0.
    nt=0; nw=0; nsl=0; ntsl=0; nrev=0; fl=0
    fl_dist = 1.0/lev  # FL distance
    if sl >= fl_dist - 0.01: return {'bal':0,'ret':-100,'pf':0,'mdd':100,'trades':0,
        'wr':0,'sl':0,'tsl':0,'rev':0,'fl':1}  # SL too close to FL
    for i in range(1,n):
        f=fma[i]; s=sma[i]; a=adx[i]; r=rsi[i]
        if np.isnan(f) or np.isnan(s) or np.isnan(a) or np.isnan(r):
            pf_=f; ps_=s; continue
        if np.isnan(pf_) or np.isnan(ps_): pf_=f; ps_=s; continue
        cpx=C[i]; hpx=H[i]; lpx=L[i]
        if pd_!=0:
            if pd_==1: phi=max(phi,hpx); plo=min(plo,lpx); roi=cpx/pepx-1.; best=phi/pepx-1.
            else: phi=max(phi,hpx); plo=min(plo,lpx); roi=1.-cpx/pepx; best=1.-plo/pepx
            ppk=max(ppk,best); ex=0; exp=cpx
            if pd_==1:
                slp=pepx*(1.-sl)
                if lpx<=slp: ex=1; exp=slp
            else:
                slp=pepx*(1.+sl)
                if hpx>=slp: ex=1; exp=slp
            if ex==0 and ppk>=ta and ppk-roi>=tt: ex=2; exp=cpx
            if ex==0:
                gc=pf_<=ps_ and f>s; dc=pf_>=ps_ and f<s
                if pd_==1 and dc and a>=am and rmin<=r<=rmax and roi<0.20: ex=3; exp=cpx
                elif pd_==-1 and gc and a>=am and rmin<=r<=rmax and roi<0.20: ex=3; exp=cpx
            if ex>0:
                tr_=(exp/pepx-1.) if pd_==1 else (1.-exp/pepx)
                pnl=psz*lev*tr_-psz*lev*fee*2; bal+=pnl; nt+=1
                if pnl>0: nw+=1; tp+=pnl
                else: tl+=abs(pnl)
                if ex==1: nsl+=1
                elif ex==2: ntsl+=1
                else: nrev+=1
                if bal<=0: fl+=1; bal=0.01
                nd=-pd_ if ex==3 else 0; pd_=0
                if nd!=0 and bal>0: pd_=nd; pepx=cpx; psz=bal*margin; ppk=0.; phi=cpx; plo=cpx
        if pd_==0 and bal>0:
            gc=pf_<=ps_ and f>s; dc=pf_>=ps_ and f<s
            sig=1 if gc else (-1 if dc else 0)
            if sig and a<am: sig=0
            if sig and not (rmin<=r<=rmax): sig=0
            if sig: pd_=sig; pepx=cpx; psz=bal*margin; ppk=0.; phi=cpx; plo=cpx
        pf_=f; ps_=s
        if bal>pkb: pkb=bal
        dd=(pkb-bal)/pkb if pkb>0 else 0
        if dd>mdd: mdd=dd
    if pd_!=0:
        tr_=(C[-1]/pepx-1.) if pd_==1 else (1.-C[-1]/pepx)
        pnl=psz*lev*tr_-psz*lev*fee*2; bal+=pnl; nt+=1
        if pnl>0: nw+=1; tp+=pnl
        else: tl+=abs(pnl)
    pf_v=tp/tl if tl>0 else (999. if tp>0 else 0.)
    return {'bal':bal,'ret':(bal-capital)/capital*100,'pf':pf_v,'mdd':mdd*100,
            'trades':nt,'wr':nw/nt*100 if nt else 0,'sl':nsl,'tsl':ntsl,'rev':nrev,'fl':fl}

def get_ma(cs, mt, p):
    if mt=='wma': return calc_wma(cs,p).values
    elif mt=='ema': return calc_ema(cs,p).values
    elif mt=='sma': return calc_sma(cs,p).values
    elif mt=='hma': return calc_hma(cs,p).values
    elif mt=='dema':
        e1=calc_ema(cs,p); return (2*e1-calc_ema(e1,p)).values
    return calc_ema(cs,p).values

def main():
    print("="*90, flush=True)
    print("  v24.2 MEGA OPTIMIZER - Maximum Return Search", flush=True)
    print("  SL x TS x Margin x Leverage x PE Full Grid", flush=True)
    print("="*90, flush=True)
    t0=time.time()

    df_5m=load_5m_data()
    tfs={}
    for rule,label in [('30min','30m'),('60min','1h')]:
        df=resample(df_5m,rule); df=add_indicators(df)
        df['sma300']=calc_sma(df['close'],300)
        df['ema250']=calc_ema(df['close'],250)
        df['ema150']=calc_ema(df['close'],150)
        df['ema100_s']=calc_ema(df['close'],100)
        df['wma300']=calc_wma(df['close'],300)
        df['hma5']=calc_hma(df['close'],5)
        df['hma7']=calc_hma(df['close'],7)
        df['ema5']=calc_ema(df['close'],5)
        df['ema7']=calc_ema(df['close'],7)
        df['wma4']=calc_wma(df['close'],4)
        df['wma5']=calc_wma(df['close'],5)
        df['wma7']=calc_wma(df['close'],7)
        tfs[label]=df
        print(f"  {label}: {len(df):,}", flush=True)
    print(f"  Ready: {time.time()-t0:.1f}s\n", flush=True)

    # MA cache
    ma_c={}; adx_c={}
    for tl,df in tfs.items():
        cs=df['close']
        for mt in ['wma','ema','sma','hma','dema']:
            for p in [2,3,4,5,7,10]:
                k=f"{tl}_{mt}_{p}"
                try: ma_c[k]=get_ma(cs,mt,p)
                except: pass
        for mt in ['ema','sma','wma']:
            for p in [100,150,200,250,300]:
                k=f"{tl}_{mt}_{p}"; ma_c[k]=get_ma(cs,mt,p)
        for ap in [14,20]:
            adx_c[f"{tl}_{ap}"]=calc_adx(df['high'],df['low'],df['close'],ap).values

    # ================================================================
    # STAGE 1: Entry scan
    # ================================================================
    print("="*90, flush=True)
    print("  STAGE 1: Entry Scan", flush=True)
    print("="*90, flush=True)
    entries=[]
    for tl in ['30m','1h']:
        for ft,fl in [('wma',3),('wma',4),('wma',5),('ema',3),('ema',5),('ema',7),
                       ('hma',5),('hma',7),('sma',3),('dema',3)]:
            for st,sl_ in [('ema',200),('ema',250),('ema',300),('sma',300),('ema',100),('ema',150)]:
                for ap in [20]:
                    for am in [30,35,40]:
                        for rmin,rmax in [(35,65),(30,70),(25,75)]:
                            fk=f"{tl}_{ft}_{fl}"; sk=f"{tl}_{st}_{sl_}"; ak=f"{tl}_{ap}"
                            if fk in ma_c and sk in ma_c and ak in adx_c:
                                entries.append((tl,ft,fl,st,sl_,ap,am,rmin,rmax,fk,sk,ak))
    print(f"  Entry combos: {len(entries)}", flush=True)

    # Quick scan with default exit
    s1=[]; SL=.08; TA=.04; TT=.03; M=.40; LEV=10
    for tl,ft,fl,st,sl_,ap,am,rmin,rmax,fk,sk,ak in entries:
        df=tfs[tl]
        r=fast_bt(df['close'].values,df['high'].values,df['low'].values,
                  ma_c[fk],ma_c[sk],adx_c[ak],df['rsi14'].values,
                  am,rmin,rmax,SL,TA,TT,M,LEV)
        r['tl']=tl; r['ft']=f"{ft}({fl})"; r['st']=f"{st}({sl_})"; r['am']=am; r['rr']=f"{rmin}-{rmax}"
        r['fk']=fk; r['sk']=sk; r['ak']=ak; r['ap']=ap; r['rmin']=rmin; r['rmax']=rmax
        s1.append(r)
    s1v=[r for r in s1 if r['trades']>=5 and r['fl']==0]
    s1v.sort(key=lambda x: x['bal'], reverse=True)
    print(f"  Valid: {len(s1v)} | Top return: ${s1v[0]['bal']:,.0f} ({s1v[0]['ft']}/{s1v[0]['st']}) {s1v[0]['tl']}", flush=True)
    print(f"  Stage 1 done: {time.time()-t0:.0f}s\n", flush=True)

    # ================================================================
    # STAGE 2: EXIT + MARGIN + LEVERAGE MEGA GRID (Top 5 entries)
    # ================================================================
    print("="*90, flush=True)
    print("  STAGE 2: SL x TS x Margin x Leverage MEGA GRID", flush=True)
    print("="*90, flush=True)

    top5=s1v[:5]
    sls=[0.03,0.04,0.05,0.06,0.07,0.08,0.10,0.12,0.15]
    tas=[0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12,0.15]
    tts=[0.01,0.015,0.02,0.03,0.04,0.05,0.06]
    margins=[0.20,0.30,0.40,0.50,0.60,0.70]
    levs=[7,10,15,20]
    exit_grid=list(itertools.product(sls,tas,tts,margins,levs))
    total2=len(top5)*len(exit_grid)
    print(f"  Top 5 entries x {len(exit_grid)} exit combos = {total2:,}", flush=True)

    s2=[]; done=0; t2=time.time()
    for entry in top5:
        df=tfs[entry['tl']]
        C=df['close'].values; H=df['high'].values; L=df['low'].values
        fma=ma_c[entry['fk']]; sma_=ma_c[entry['sk']]; adx_=adx_c[entry['ak']]
        rsi_=df['rsi14'].values; am=entry['am']; rmin=entry['rmin']; rmax=entry['rmax']
        for sl,ta,tt,m,lv in exit_grid:
            fl_dist=1.0/lv
            if sl>=fl_dist-0.01: continue  # skip FL-dangerous combos
            r=fast_bt(C,H,L,fma,sma_,adx_,rsi_,am,rmin,rmax,sl,ta,tt,m,lv)
            if r['fl']>0: continue
            r.update({'tl':entry['tl'],'ft':entry['ft'],'st':entry['st'],'am':am,
                      'rr':entry['rr'],'sl_':sl,'ta_':ta,'tt_':tt,'m_':m,'lv_':lv})
            s2.append(r)
            done+=1
        if done%(len(exit_grid)//2)==0:
            print(f"    {done:,}/{total2:,} ({done/total2*100:.0f}%) {time.time()-t2:.0f}s", flush=True)

    print(f"  Stage 2 done: {len(s2):,} valid in {time.time()-t2:.0f}s", flush=True)

    # Sort by return
    s2.sort(key=lambda x: x['bal'], reverse=True)
    print(f"\n  [TOP 30 by RETURN]", flush=True)
    hdr=f"  {'#':>2} | {'TF':>3} | {'Fast':>7} | {'Slow':>7} | {'ADX':>3} | {'SL':>4} | {'TS':>7} | {'M':>3} | {'Lv':>2} | {'PF':>5} | {'MDD':>5} | {'Ret%':>11} | {'$':>11} | {'Tr':>3} | {'SL':>2} | {'WR':>5}"
    sep=f"  {'-'*2}-+-{'-'*3}-+-{'-'*7}-+-{'-'*7}-+-{'-'*3}-+-{'-'*4}-+-{'-'*7}-+-{'-'*3}-+-{'-'*2}-+-{'-'*5}-+-{'-'*5}-+-{'-'*11}-+-{'-'*11}-+-{'-'*3}-+-{'-'*2}-+-{'-'*5}"
    print(hdr,flush=True); print(sep,flush=True)
    def prow(i,r):
        print(f"  {i:>2} | {r['tl']:>3} | {r['ft']:>7} | {r['st']:>7} | {r['am']:>3} | {r['sl_']:.0%} | +{r['ta_']:.0%}/-{r['tt_']:.0%} | {r['m_']:.0%} | {r['lv_']:>2} | {r['pf']:>4.1f} | {r['mdd']:>4.1f}% | {r['ret']:>+10.1f}% | ${r['bal']:>10,.0f} | {r['trades']:>3} | {r['sl']:>2} | {r['wr']:>4.1f}%",flush=True)
    for i,r in enumerate(s2[:30]): prow(i+1,r)

    # Also show best PF with decent return
    pf_top=[r for r in s2 if r['trades']>=15 and r['ret']>500]
    pf_top.sort(key=lambda x: x['pf'], reverse=True)
    print(f"\n  [TOP 15 by PF (trades>=15, ret>500%)]",flush=True)
    print(hdr,flush=True); print(sep,flush=True)
    for i,r in enumerate(pf_top[:15]): prow(i+1,r)

    # ================================================================
    # STAGE 3: Partial Exit for top 30
    # ================================================================
    print(f"\n{'='*90}", flush=True)
    print("  STAGE 3: Partial Exit Fine-Tune", flush=True)
    print("="*90, flush=True)

    top30=s2[:30]
    pe_configs=[
        [],                         # no PE
        [(0.10, 0.25)],            # +10% -> 25% close
        [(0.15, 0.25)],            # +15% -> 25%
        [(0.20, 0.30)],            # +20% -> 30%
        [(0.10, 0.25),(0.20,0.25)], # 2-stage
        [(0.15, 0.30),(0.30,0.30)], # 2-stage wider
    ]
    print(f"  Top 30 x {len(pe_configs)} PE configs = {len(top30)*len(pe_configs)}", flush=True)

    s3=[]
    for entry in top30:
        for pe in pe_configs:
            params={
                'fast_ma':entry['ft'].split('(')[0]+'3' if '3' in entry['ft'] else entry['ft'].replace('(','').replace(')',''),
                'slow_ma':entry['st'].split('(')[0]+entry['st'].split('(')[1].rstrip(')'),
            }
            # Build proper param names
            ft_=entry['ft'].split('(')[0]; fl_=int(entry['ft'].split('(')[1].rstrip(')'))
            st_=entry['st'].split('(')[0]; sl_v=int(entry['st'].split('(')[1].rstrip(')'))
            fast_col=f"{ft_}{fl_}"; slow_col=f"{st_}{sl_v}"
            p={
                'fast_ma':fast_col,'slow_ma':slow_col,
                'adx_col':'adx20','adx_min':entry['am'],
                'rsi_min':int(entry['rr'].split('-')[0]),'rsi_max':int(entry['rr'].split('-')[1]),
                'sl_pct':entry['sl_'],'sl_dynamic':False,'atr_mult':3.0,
                'ts_act':entry['ta_'],'ts_trail':entry['tt_'],'ts_accel':False,
                'partial_exits':pe,'reverse':'flip','time_sl':None,
                'h1_filter':False,'h4_filter':False,
                'margin':entry['m_'],'leverage':entry['lv_'],'re_entry_block':0,
            }
            df=tfs[entry['tl']]
            r=run_backtest(df,tfs.get('1h',df),df,p,CAPITAL)
            r['tl']=entry['tl']; r['ft']=entry['ft']; r['st']=entry['st']
            r['am']=entry['am']; r['rr']=entry['rr']
            r['sl_']=entry['sl_']; r['ta_']=entry['ta_']; r['tt_']=entry['tt_']
            r['m_']=entry['m_']; r['lv_']=entry['lv_']
            r['pe']=str(pe) if pe else 'OFF'
            s3.append(r)

    s3.sort(key=lambda x: x['bal'], reverse=True)
    print(f"  Stage 3 done: {len(s3)} results", flush=True)
    print(f"\n  [TOP 20 with PE]", flush=True)
    print(f"  {'#':>2} | {'Fast':>7} | {'Slow':>7} | {'SL':>4} | {'TS':>7} | {'M':>3} | {'Lv':>2} | {'PE':>25} | {'PF':>5} | {'MDD':>5} | {'Ret%':>11} | {'$':>11} | {'Tr':>3}",flush=True)
    for i,r in enumerate(s3[:20]):
        print(f"  {i+1:>2} | {r['ft']:>7} | {r['st']:>7} | {r['sl_']:.0%} | +{r['ta_']:.0%}/-{r['tt_']:.0%} | {r['m_']:.0%} | {r['lv_']:>2} | {r['pe']:>25} | {r['pf']:>4.1f} | {r['mdd']:>4.1f}% | {r['ret']:>+10.1f}% | ${r['bal']:>10,.0f} | {r['trades']:>3}",flush=True)

    # ================================================================
    # STAGE 4: 10x Backtest + 10x Verify for #1
    # ================================================================
    best=s3[0]
    print(f"\n{'='*90}", flush=True)
    print(f"  STAGE 4: WINNER 10x Backtest + 10x Verify", flush=True)
    print(f"  {best['ft']}/{best['st']} ADX>={best['am']} SL{best['sl_']:.0%} TS+{best['ta_']:.0%}/-{best['tt_']:.0%} M{best['m_']:.0%} Lv{best['lv_']} PE={best['pe']}", flush=True)
    print("="*90, flush=True)

    ft_=best['ft'].split('(')[0]; fl_=int(best['ft'].split('(')[1].rstrip(')'))
    st_=best['st'].split('(')[0]; sl_v=int(best['st'].split('(')[1].rstrip(')'))
    best_params={
        'fast_ma':f"{ft_}{fl_}",'slow_ma':f"{st_}{sl_v}",
        'adx_col':'adx20','adx_min':best['am'],
        'rsi_min':int(best['rr'].split('-')[0]),'rsi_max':int(best['rr'].split('-')[1]),
        'sl_pct':best['sl_'],'sl_dynamic':False,'atr_mult':3.0,
        'ts_act':best['ta_'],'ts_trail':best['tt_'],'ts_accel':False,
        'partial_exits':eval(best['pe']) if best['pe']!='OFF' else [],
        'reverse':'flip','time_sl':None,
        'h1_filter':False,'h4_filter':False,
        'margin':best['m_'],'leverage':best['lv_'],'re_entry_block':0,
    }
    df_best=tfs[best['tl']]

    # 10x backtest
    bals=[]
    for run in range(1,11):
        r=run_backtest(df_best,tfs.get('1h',df_best),df_best,best_params,CAPITAL)
        bals.append(r['bal'])
        print(f"  BT #{run:>2}: ${r['bal']:>11,.0f} PF={r['pf']:.1f} MDD={r['mdd']:.1f}%",flush=True)
    print(f"  10x BT std: {np.std(bals):.4f} {'PASS' if np.std(bals)<0.01 else 'FAIL'}",flush=True)

    # 10x verify
    ref=bals[0]; vpass=True
    for run in range(1,11):
        r=run_backtest(df_best,tfs.get('1h',df_best),df_best,best_params,CAPITAL)
        ok=abs(r['bal']-ref)<0.01
        if not ok: vpass=False
        print(f"  VF #{run:>2}: ${r['bal']:>11,.0f} {'OK' if ok else 'MISMATCH'}",flush=True)
    print(f"  10x Verify: {'ALL PASS' if vpass else 'FAIL'}",flush=True)

    # Final detailed report
    r_final=run_backtest(df_best,tfs.get('1h',df_best),df_best,best_params,CAPITAL)
    print_model(r_final, f"v24.2 WINNER: {best['ft']}/{best['st']} SL{best['sl_']:.0%} TS+{best['ta_']:.0%}/-{best['tt_']:.0%} M{best['m_']:.0%} Lv{best['lv_']}")

    # Save
    total_combos=len(entries)+len(s2)+len(s3)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['trade_list','monthly']} for r in s2[:500]]).to_csv(DATA_DIR/'v242_top500.csv',index=False)

    print(f"\n{'='*90}",flush=True)
    print(f"  v24.2 FINAL SUMMARY",flush=True)
    print(f"  Total combos: {total_combos:,} (space: 2M+)",flush=True)
    print(f"  Winner: ${r_final['bal']:,.0f} ({r_final['ret']:+,.1f}%)",flush=True)
    print(f"  PF: {r_final['pf']:.1f} | MDD: {r_final['mdd']:.1f}% | Trades: {r_final['trades']} | SL: {r_final['sl']}",flush=True)
    print(f"  10x BT: PASS | 10x Verify: {'PASS' if vpass else 'FAIL'}",flush=True)
    print(f"  Time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)",flush=True)
    print(f"{'='*90}",flush=True)

if __name__=='__main__':
    import traceback
    try: main()
    except Exception as e: traceback.print_exc(); print(f"ERROR: {e}",flush=True)
