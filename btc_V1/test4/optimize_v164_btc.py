"""
v16.4 BTC 전용 1,000,000+ 조합 최적화
목표: PF>=8 + MDD 최소 + 거래30+ + 높은 수익
"""
import numpy as np, json, os, time
from bt_fast import load_5m_data, build_mtf, IndicatorCache, run_backtest

BASE = r'D:\filesystem\futures\btc_V1\test4'

def score164(r):
    if r is None or r.get('ret',0)<=0 or r.get('liq',0)>0: return 0
    pf=r['pf'];mdd=r['mdd'];ret=r['ret'];tr=r['trades']
    if tr<25: return 0
    # PF^2 (강한 가중) x MDD 보너스 x 수익 x 거래수 x 최근연도
    pf_b=pf**2
    mdd_b=((100-mdd)/40)**1.5  # MDD 낮을수록 큰 보너스
    ret_b=np.log1p(ret/100)
    tr_b=min(tr/30,2.0)
    yr=r.get('yr',{})
    rec=[yr.get(str(y),0) for y in range(2023,2027) if str(y) in yr]
    yr_b=2.0 if rec and min(rec)>0 else(1.5 if rec and np.mean(rec)>30 else 1.0)
    return pf_b*mdd_b*ret_b*tr_b*yr_b

def main():
    print("="*80)
    print("  v16.4 BTC 전용 1,000,000+ 조합 최적화")
    print("="*80)

    df5=load_5m_data(BASE)
    mtf=build_mtf(df5)
    cache=IndicatorCache(mtf)

    # ============================================================
    # Phase 1: 대탐색 (500,000 샘플)
    # ============================================================
    print(f"\n{'='*80}")
    print("  Phase 1: 진입 대탐색 (500,000)")
    print(f"{'='*80}")

    tfs=['5m','10m','15m','30m','1h']
    mfts=['ema','hma','dema','wma','sma']
    msts=['ema','wma','hma','sma']
    fls=[3,5,7,10,14,21]
    sls_l=[50,100,150,200,250]
    aps=[14,20]; ams=[25,30,35,40,45]
    rrs=[(25,55),(25,60),(30,58),(30,65),(35,65),(35,70),(40,75)]
    slps=[0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
    tas=[0.04,0.05,0.06,0.07,0.08,0.10,0.12,0.15,0.20]
    tps=[0.02,0.03,0.04,0.05,0.06]

    configs=[]
    for tf in tfs:
        for mft in mfts:
            for mst in msts:
                for fl in fls:
                    for sl in sls_l:
                        if fl>=sl: continue
                        for ap in aps:
                            for am in ams:
                                for rlo,rhi in rrs:
                                    for slp in slps:
                                        for ta in tas:
                                            for tp in tps:
                                                if tp>=ta: continue
                                                configs.append({
                                                    'timeframe':tf,'ma_fast_type':mft,'ma_slow_type':mst,
                                                    'ma_fast':fl,'ma_slow':sl,'adx_period':ap,'adx_min':am,
                                                    'rsi_period':14,'rsi_min':rlo,'rsi_max':rhi,
                                                    'sl_pct':slp,'trail_activate':ta,'trail_pct':tp,
                                                    'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
                                                    'fee_rate':0.0004,'initial_capital':3000.0,
                                                })
    total=len(configs)
    N=min(total,500000)
    np.random.seed(164)
    if total>N: configs=[configs[i] for i in np.random.choice(total,N,replace=False)]
    print(f"  전체: {total:,} -> 샘플: {N:,}")

    p1=[]; t0=time.time()
    for i,cfg in enumerate(configs):
        r=run_backtest(cache,cfg['timeframe'],cfg)
        if r and r['trades']>=20 and r.get('liq',0)==0 and r.get('ret',0)>0:
            r['_s']=score164(r); r['_cfg']=cfg; p1.append(r)
        if (i+1)%100000==0: print(f"    {i+1:,}/{N:,} ({time.time()-t0:.0f}s) 유효:{len(p1):,}")
    print(f"  P1: {len(p1):,}개 ({time.time()-t0:.0f}s)")
    p1.sort(key=lambda x:x['_s'],reverse=True)

    print(f"\n  P1 Top 10:")
    for i,r in enumerate(p1[:10]):
        print(f"  {i+1:>3} ${r['bal']:>8,.0f} PF:{r['pf']:>6.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} {r['cfg']}")

    # ============================================================
    # Phase 2: 청산 최적화 (Top 40 x 청산조합)
    # ============================================================
    print(f"\n{'='*80}")
    print("  Phase 2: 청산 최적화")
    print(f"{'='*80}")

    slps2=[0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
    tas2=[0.04,0.05,0.06,0.07,0.08,0.10,0.12,0.15,0.20]
    tps2=[0.02,0.03,0.04,0.05,0.06]
    atr_sl=[{'use_atr_sl':False},
            {'use_atr_sl':True,'atr_sl_mult':2.0,'atr_sl_min':0.03,'atr_sl_max':0.12}]
    atr_tr=[{'use_atr_trail':False},
            {'use_atr_trail':True,'atr_trail_mult':1.5}]

    p2=[]; combo=0; t0=time.time()
    for br in p1[:40]:
        base=br['_cfg']
        for slp in slps2:
            for ta in tas2:
                for tp in tps2:
                    if tp>=ta: continue
                    for asl in atr_sl:
                        for atr in atr_tr:
                            cfg=dict(base); cfg['sl_pct']=slp; cfg['trail_activate']=ta; cfg['trail_pct']=tp
                            cfg.update(asl); cfg.update(atr)
                            r=run_backtest(cache,cfg['timeframe'],cfg)
                            if r and r['trades']>=20 and r.get('liq',0)==0 and r.get('ret',0)>0:
                                r['_s']=score164(r); r['_cfg']=cfg; p2.append(r)
                            combo+=1
        if combo%50000==0: print(f"    {combo:,} ({time.time()-t0:.0f}s)")
    print(f"  P2: {combo:,}개 ({time.time()-t0:.0f}s), 유효:{len(p2):,}")
    p2.sort(key=lambda x:x['_s'],reverse=True)

    # ============================================================
    # Phase 3: 지연진입 (Top 30)
    # ============================================================
    print(f"\n{'='*80}")
    print("  Phase 3: 지연진입")
    print(f"{'='*80}")
    delay_opts=[
        {},
        {'delayed_entry':True,'delay_max_candles':3,'delay_price_min':-0.001,'delay_price_max':-0.010},
        {'delayed_entry':True,'delay_max_candles':3,'delay_price_min':-0.001,'delay_price_max':-0.025},
        {'delayed_entry':True,'delay_max_candles':6,'delay_price_min':-0.001,'delay_price_max':-0.015},
        {'delayed_entry':True,'delay_max_candles':6,'delay_price_min':-0.001,'delay_price_max':-0.025},
        {'delayed_entry':True,'delay_max_candles':6,'delay_price_min':-0.002,'delay_price_max':-0.020},
        {'delayed_entry':True,'delay_max_candles':9,'delay_price_min':-0.001,'delay_price_max':-0.025},
        {'delayed_entry':True,'delay_max_candles':12,'delay_price_min':-0.001,'delay_price_max':-0.025},
    ]
    p3=[]; combo3=0
    for br in p2[:30]:
        base=br['_cfg']
        for dopt in delay_opts:
            cfg=dict(base); cfg.update(dopt)
            r=run_backtest(cache,cfg['timeframe'],cfg)
            if r and r['trades']>=15 and r.get('liq',0)==0 and r.get('ret',0)>0:
                r['_s']=score164(r); r['_cfg']=cfg; p3.append(r)
            combo3+=1
    print(f"  P3: {combo3}개, 유효:{len(p3)}")
    p3.sort(key=lambda x:x['_s'],reverse=True)

    # ============================================================
    # Phase 4: 사이징 (Top 30)
    # ============================================================
    print(f"\n{'='*80}")
    print("  Phase 4: 사이징 + 보호")
    print(f"{'='*80}")
    mgs=[0.15,0.20,0.25,0.30,0.35,0.40,0.50]
    mls=[0,-0.10,-0.15,-0.20,-0.25,-0.30]
    dds=[0,-0.25,-0.30,-0.50]

    p4=[]; combo4=0; t0=time.time()
    for br in p3[:30]:
        base=br['_cfg']
        for mg in mgs:
            for ml in mls:
                for dd in dds:
                    cfg=dict(base); cfg['margin_normal']=mg; cfg['margin_reduced']=mg/2
                    cfg['monthly_loss_limit']=ml; cfg['dd_threshold']=dd
                    r=run_backtest(cache,cfg['timeframe'],cfg)
                    if r and r['trades']>=10 and r.get('liq',0)==0 and r.get('ret',0)>0:
                        r['_s']=score164(r); r['_cfg']=cfg; p4.append(r)
                    combo4+=1
    print(f"  P4: {combo4:,}개 ({time.time()-t0:.0f}s), 유효:{len(p4):,}")
    p4.sort(key=lambda x:x['_s'],reverse=True)

    grand_total=N+combo+combo3+combo4

    # ============================================================
    # 결과 출력
    # ============================================================
    # 1) 스코어 Top 10
    print(f"\n{'='*80}")
    print(f"  === 스코어 Top 10 (PF^2 x MDD^1.5 x 수익) ===")
    print(f"{'='*80}")
    for i,r in enumerate(p4[:10]):
        print(f"  {i+1} ${r['bal']:>12,.0f} PF:{r['pf']:>6.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} WR:{r['wr']:.1f}% {r['cfg']}")

    # 2) PF Top 10 (30+ trades, MDD<=50%)
    pf30=sorted([r for r in p4 if r['trades']>=30 and r['mdd']<=50],key=lambda x:x['pf'],reverse=True)
    print(f"\n  === PF Top 10 (거래30+, MDD<=50%) ===")
    for i,r in enumerate(pf30[:10]):
        print(f"  {i+1} ${r['bal']:>12,.0f} PF:{r['pf']:>6.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} WR:{r['wr']:.1f}% {r['cfg']}")

    # 3) MDD Top 10 (PF>=5, 30+ trades)
    mdd30=sorted([r for r in p4 if r['pf']>=5 and r['trades']>=30],key=lambda x:x['mdd'])
    print(f"\n  === MDD Top 10 (PF>=5, 거래30+) ===")
    for i,r in enumerate(mdd30[:10]):
        print(f"  {i+1} ${r['bal']:>12,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} WR:{r['wr']:.1f}% {r['cfg']}")

    # 4) 수익 Top 10 (PF>=3, 30+ trades, MDD<=50%)
    ret30=sorted([r for r in p4 if r['pf']>=3 and r['trades']>=30 and r['mdd']<=50],key=lambda x:x['bal'],reverse=True)
    print(f"\n  === 수익 Top 10 (PF>=3, 거래30+, MDD<=50%) ===")
    for i,r in enumerate(ret30[:10]):
        print(f"  {i+1} ${r['bal']:>12,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} WR:{r['wr']:.1f}% {r['cfg']}")

    # 5) 균형 추천 (PF>=8, MDD<=40%, 거래30+)
    balanced=sorted([r for r in p4 if r['pf']>=8 and r['mdd']<=40 and r['trades']>=30],key=lambda x:x['bal'],reverse=True)
    print(f"\n  === 균형 추천 (PF>=8, MDD<=40%, 거래30+): {len(balanced)}개 ===")
    for i,r in enumerate(balanced[:10]):
        print(f"  {i+1} ${r['bal']:>12,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} WR:{r['wr']:.1f}% {r['cfg']}")

    # Top 1 상세
    best=p4[0]
    print(f"\n{'='*80}")
    print(f"  Top 1 상세")
    print(f"{'='*80}")
    print(f"  설정: {best['cfg']}")
    print(f"  잔액: ${best['bal']:,.0f} ({best['ret']:+,.1f}%)")
    print(f"  PF: {best['pf']} | MDD: {best['mdd']}% | FL: {best['liq']}")
    print(f"  거래: {best['trades']} (SL:{best['sl']} TSL:{best['tsl']} REV:{best['sig']})")
    print(f"  승률: {best['wr']}% | 평균승: {best.get('avg_win',0):.2f}% | 평균패: {best.get('avg_loss',0):.2f}%")
    if best.get('avg_loss',0)!=0: print(f"  손익비: {abs(best['avg_win']/best['avg_loss']):.2f}:1")
    print(f"  연도별:")
    for y in sorted(best.get('yr',{}).keys()):
        print(f"    {y}: {best['yr'][y]:>+8.1f}%")

    # 30회 검증
    print(f"\n  30회 검증:")
    bals=[run_backtest(cache,best['_cfg']['timeframe'],best['_cfg'])['bal'] for _ in range(30)]
    print(f"  ${np.mean(bals):>12,.0f} std={np.std(bals):.4f} {'PASS' if np.std(bals)<0.01 else 'FAIL'}")

    # 저장
    save={'grand_total':grand_total}
    for k,lst in [('score_top10',p4[:10]),('pf_mdd50_30tr',pf30[:10]),
                   ('mdd_pf5_30tr',mdd30[:10]),('ret_pf3_mdd50',ret30[:10]),('balanced',balanced[:10])]:
        save[k]=[{'bal':r['bal'],'ret':r['ret'],'pf':r['pf'],'mdd':r['mdd'],'fl':r['liq'],
                   'tr':r['trades'],'wr':r['wr'],'cfg':r['cfg'],'yr':r.get('yr',{}),
                   'sl':r.get('sl',0),'tsl':r.get('tsl',0),'rev':r.get('sig',0),
                   'avg_win':r.get('avg_win',0),'avg_loss':r.get('avg_loss',0)} for r in lst]
    with open(f'{BASE}/v164_btc_results.json','w') as f:
        json.dump(save,f,indent=2,ensure_ascii=False)
    print(f"\n저장: v164_btc_results.json")
    print(f"총 조합: {grand_total:,}")
    print("완료!")

if __name__=='__main__':
    main()
