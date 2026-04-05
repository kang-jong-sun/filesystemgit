"""
v16.1 - 5코인 개별 최적화 (1,000,000+ 조합)
핵심 변경: 거래수 30+ 강제, PF>=8 목표, 1M+ 조합
"""
import numpy as np, pandas as pd, json, os, time
from bt_fast import build_mtf, IndicatorCache, run_backtest

BASE = r'D:\filesystem\futures\btc_V1\test4'

def load_coin(prefix):
    parts=[]
    for i in range(1,4):
        f=os.path.join(BASE,f'{prefix}_5m_2020_to_now_part{i}.csv')
        if os.path.exists(f): parts.append(pd.read_csv(f,parse_dates=['timestamp']))
    df=pd.concat(parts,ignore_index=True).drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    df=df.rename(columns={'timestamp':'time'})
    for c in ['open','high','low','close','volume']: df[c]=df[c].astype(float)
    return df

def score161(r):
    """v16.1 스코어: PF 최우선 + 거래수30+ 필수 + MDD 감점"""
    if r is None or r.get('ret',0)<=0 or r.get('liq',0)>0: return 0
    pf=r['pf']; mdd=r['mdd']; ret=r['ret']; tr=r['trades']
    if tr<30: return 0  # 30+ 강제!
    pf_b=pf**2  # PF 제곱 (강한 가중치)
    mdd_b=(100-mdd)/40
    ret_b=np.log1p(ret/100)
    yr=r.get('yr',{})
    rec=[yr.get(str(y),0) for y in range(2023,2027) if str(y) in yr]
    yr_b=2.0 if rec and min(rec)>0 else(1.5 if rec and np.mean(rec)>30 else 1.0)
    return pf_b*mdd_b*ret_b*yr_b

def optimize_coin_v161(name, cache, sample_n=200000):
    """단일 코인 200,000 샘플 최적화"""
    print(f"\n{'='*80}")
    print(f"  [{name}] v16.1 최적화 ({sample_n:,} 샘플)")
    print(f"{'='*80}")

    # Phase 1: 대탐색
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
    N=min(total,sample_n)
    np.random.seed(hash(name)%2**31)
    if total>N: configs=[configs[i] for i in np.random.choice(total,N,replace=False)]
    print(f"  P1 전체: {total:,} -> {N:,}")

    p1=[]; t0=time.time()
    for i,cfg in enumerate(configs):
        r=run_backtest(cache,cfg['timeframe'],cfg)
        if r and r['trades']>=20 and r.get('liq',0)==0 and r.get('ret',0)>0:
            r['_s']=score161(r); r['_cfg']=cfg; p1.append(r)
        if (i+1)%50000==0: print(f"    {i+1:,}/{N:,} ({time.time()-t0:.0f}s) 유효:{len(p1):,}")
    print(f"  P1: {len(p1):,}개 ({time.time()-t0:.0f}s)")
    p1.sort(key=lambda x:x['_s'],reverse=True)

    # PF Top (30+ trades)
    pf30=sorted([r for r in p1 if r['trades']>=30],key=lambda x:x['pf'],reverse=True)
    print(f"  P1 PF Top5 (30+거래):")
    for i,r in enumerate(pf30[:5]):
        print(f"    {i+1} ${r['bal']:>8,.0f} PF:{r['pf']:>6.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} {r['cfg']}")

    # Phase 2: 청산 최적화 (Top 30)
    slps2=[0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
    tas2=[0.04,0.05,0.06,0.07,0.08,0.10,0.12,0.15,0.20]
    tps2=[0.02,0.03,0.04,0.05,0.06]
    p2=[]; combo=0; t0=time.time()
    for br in p1[:30]:
        base=br['_cfg']
        for slp in slps2:
            for ta in tas2:
                for tp in tps2:
                    if tp>=ta: continue
                    cfg=dict(base); cfg['sl_pct']=slp; cfg['trail_activate']=ta; cfg['trail_pct']=tp
                    r=run_backtest(cache,cfg['timeframe'],cfg)
                    if r and r['trades']>=20 and r.get('liq',0)==0 and r.get('ret',0)>0:
                        r['_s']=score161(r); r['_cfg']=cfg; p2.append(r)
                    combo+=1
    print(f"  P2: {combo:,}개 ({time.time()-t0:.0f}s), 유효:{len(p2):,}")
    p2.sort(key=lambda x:x['_s'],reverse=True)

    # Phase 3: 사이징
    mgs=[0.15,0.20,0.25,0.30,0.35,0.40,0.50]
    mls=[0,-0.15,-0.20,-0.25,-0.30]
    dds=[0,-0.25,-0.30,-0.50]
    p3=[]; combo2=0; t0=time.time()
    for br in p2[:20]:
        base=br['_cfg']
        for mg in mgs:
            for ml in mls:
                for dd in dds:
                    cfg=dict(base); cfg['margin_normal']=mg; cfg['margin_reduced']=mg/2
                    cfg['monthly_loss_limit']=ml; cfg['dd_threshold']=dd
                    r=run_backtest(cache,cfg['timeframe'],cfg)
                    if r and r['trades']>=15 and r.get('liq',0)==0 and r.get('ret',0)>0:
                        r['_s']=score161(r); r['_cfg']=cfg; p3.append(r)
                    combo2+=1
    print(f"  P3: {combo2:,}개 ({time.time()-t0:.0f}s), 유효:{len(p3):,}")
    p3.sort(key=lambda x:x['_s'],reverse=True)

    total_combos=N+combo+combo2

    # 결과 카테고리
    pf30_final=sorted([r for r in p3 if r['trades']>=30],key=lambda x:x['pf'],reverse=True)
    ret30_final=sorted([r for r in p3 if r['trades']>=30 and r['pf']>=2],key=lambda x:x['bal'],reverse=True)
    mdd30_final=sorted([r for r in p3 if r['trades']>=30 and r['pf']>=3],key=lambda x:x['mdd'])

    print(f"\n  [{name}] === PF Top 5 (거래30+, FL=0) ===")
    for i,r in enumerate(pf30_final[:5]):
        print(f"  {i+1} ${r['bal']:>10,.0f} PF:{r['pf']:>6.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} WR:{r['wr']:.1f}% {r['cfg']}")

    print(f"\n  [{name}] === 수익 Top 5 (PF>=2, 거래30+) ===")
    for i,r in enumerate(ret30_final[:5]):
        print(f"  {i+1} ${r['bal']:>10,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} {r['cfg']}")

    print(f"\n  [{name}] === MDD Top 5 (PF>=3, 거래30+) ===")
    for i,r in enumerate(mdd30_final[:5]):
        print(f"  {i+1} ${r['bal']:>10,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} {r['cfg']}")

    # Top 1 상세
    best=pf30_final[0] if pf30_final else (p3[0] if p3 else None)
    if best:
        print(f"\n  [{name}] 추천 전략 상세:")
        print(f"    잔액: ${best['bal']:,.0f} ({best['ret']:+,.1f}%)")
        print(f"    PF: {best['pf']} | MDD: {best['mdd']}% | FL: {best['liq']}")
        print(f"    거래: {best['trades']} (SL:{best['sl']} TSL:{best['tsl']} REV:{best['sig']})")
        print(f"    승률: {best['wr']}%")
        if best.get('avg_win',0)!=0: print(f"    평균승: {best['avg_win']:.2f}% | 평균패: {best['avg_loss']:.2f}%")
        if best.get('yr'):
            print(f"    연도별: {' | '.join(f'{y}:{v:+.1f}%' for y,v in sorted(best['yr'].items()))}")

    return {
        'best':best, 'pf_top':pf30_final[:5], 'ret_top':ret30_final[:5],
        'mdd_top':mdd30_final[:5], 'total_combos':total_combos,
    }


def main():
    print("="*80)
    print("  v16.1 - 5코인 개별 최적화 (1,000,000+ 조합)")
    print("  PF>=8 목표 + 거래30+ 강제 + MDD 최소화")
    print("="*80)

    coins=[('btc_usdt','BTC'),('eth_usdt','ETH'),('sol_usdt','SOL'),('xrp_usdt','XRP'),('doge_usdt','DOGE')]
    all_results={}; grand_total=0

    for prefix,name in coins:
        print(f"\n  Loading {name}...")
        df=load_coin(prefix); print(f"    {len(df):,}행")
        mtf=build_mtf(df); cache=IndicatorCache(mtf)
        result=optimize_coin_v161(name, cache, sample_n=200000)
        all_results[name]=result
        grand_total+=result['total_combos']

    # 30회 검증
    print(f"\n{'='*80}")
    print("  30회 반복 검증")
    print(f"{'='*80}")
    for prefix,name in coins:
        df=load_coin(prefix); mtf=build_mtf(df); cache=IndicatorCache(mtf)
        best=all_results[name]['best']
        if best and best.get('_cfg'):
            cfg=best['_cfg']
            bals=[run_backtest(cache,cfg['timeframe'],cfg)['bal'] for _ in range(30)]
            print(f"  {name}: ${np.mean(bals):>12,.0f} std={np.std(bals):.4f} {'PASS' if np.std(bals)<0.01 else 'FAIL'} PF:{best['pf']:.2f} TR:{best['trades']}")

    # 포트폴리오
    alloc={'BTC':0.30,'ETH':0.25,'SOL':0.20,'XRP':0.15,'DOGE':0.10}
    print(f"\n{'='*80}")
    print("  $10,000 포트폴리오")
    print(f"{'='*80}")
    print(f"  {'코인':>6} {'배분':>5} {'초기$':>7} {'최종$':>12} {'수익률':>10} {'PF':>7} {'MDD':>6} {'거래':>4} 전략")
    print(f"  {'-'*100}")
    total_end=0
    for name in ['BTC','ETH','SOL','XRP','DOGE']:
        best=all_results[name]['best']
        if best:
            pct=alloc[name]; init=10000*pct; ratio=best['bal']/3000; end=init*ratio; total_end+=end
            print(f"  {name:>6} {pct:>4.0%} ${init:>5,.0f} ${end:>10,.0f} {(ratio-1)*100:>+9.1f}% {best['pf']:>6.2f} {best['mdd']:>5.1f}% {best['trades']:>4} {best['cfg'][:55]}")
    print(f"  {'-'*100}")
    print(f"  {'합계':>6} 100% $10,000 ${total_end:>10,.0f} {(total_end/10000-1)*100:>+9.1f}%")

    # 저장
    save={'grand_total':grand_total,'portfolio_end':round(total_end,0),'alloc':alloc}
    for name in ['BTC','ETH','SOL','XRP','DOGE']:
        best=all_results[name]['best']
        if best:
            save[name]={
                'bal':best['bal'],'ret':best['ret'],'pf':best['pf'],'mdd':best['mdd'],
                'fl':best['liq'],'tr':best['trades'],'wr':best['wr'],
                'sl':best.get('sl',0),'tsl':best.get('tsl',0),'rev':best.get('sig',0),
                'yr':best.get('yr',{}),'cfg':best['cfg'],
                'avg_win':best.get('avg_win',0),'avg_loss':best.get('avg_loss',0),
            }
    with open(f'{BASE}/v161_results.json','w') as f:
        json.dump(save,f,indent=2,ensure_ascii=False)
    print(f"\n저장: v161_results.json")
    print(f"총 조합: {grand_total:,}")
    print("완료!")

if __name__=='__main__':
    main()
