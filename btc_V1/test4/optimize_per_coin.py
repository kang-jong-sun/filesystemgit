"""
v16.0 코인별 개별 최적화
BTC/ETH/SOL/XRP/DOGE 각각 100,000+ 조합 -> 총 500,000+
Phase 1: 진입 탐색 -> Phase 2: 청산 -> Phase 3: 사이징
각 코인 특성에 맞는 최적 전략 도출
"""
import numpy as np, pandas as pd, json, os, time
from bt_fast import build_mtf, IndicatorCache, run_backtest

BASE = r'D:\filesystem\futures\btc_V1\test4'

def load_coin(prefix):
    parts = []
    for i in range(1,4):
        f = os.path.join(BASE, f'{prefix}_5m_2020_to_now_part{i}.csv')
        if os.path.exists(f): parts.append(pd.read_csv(f, parse_dates=['timestamp']))
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    df = df.rename(columns={'timestamp':'time'})
    for c in ['open','high','low','close','volume']: df[c]=df[c].astype(float)
    return df

def score(r):
    if r is None or r.get('ret',0)<=0 or r.get('liq',0)>0: return 0
    pf=r['pf']; mdd=r['mdd']; ret=r['ret']; tr=r['trades']
    if tr<15: return 0
    return (pf**1.5)*((100-mdd)/50)*np.log1p(ret/100)*min(tr/30,2.0)

def optimize_coin(name, cache):
    """단일 코인 3단계 최적화"""
    print(f"\n{'='*80}")
    print(f"  [{name}] 개별 최적화")
    print(f"{'='*80}")

    # Phase 1: 진입 (80,000 샘플)
    tfs=['5m','10m','15m','30m','1h']
    mfts=['ema','hma','dema','wma']
    msts=['ema','wma','sma']
    fls=[3,5,7,10,14,21]
    sls_l=[50,100,150,200,250]
    aps=[14,20]; ams=[25,30,35,40]
    rrs=[(25,60),(30,58),(30,65),(35,65),(35,70),(40,75)]
    slps=[0.04,0.05,0.06,0.07,0.08,0.09]

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
                                        configs.append({
                                            'timeframe':tf,'ma_fast_type':mft,'ma_slow_type':mst,
                                            'ma_fast':fl,'ma_slow':sl,'adx_period':ap,'adx_min':am,
                                            'rsi_period':14,'rsi_min':rlo,'rsi_max':rhi,
                                            'sl_pct':slp,'trail_activate':0.06,'trail_pct':0.03,
                                            'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
                                            'fee_rate':0.0004,'initial_capital':3000.0,
                                        })
    total=len(configs)
    N=min(total,80000)
    np.random.seed(hash(name)%2**31)
    if total>N: configs=[configs[i] for i in np.random.choice(total,N,replace=False)]
    print(f"  P1: {total:,} -> {N:,} 샘플")

    p1=[]; t0=time.time()
    for i,cfg in enumerate(configs):
        r=run_backtest(cache,cfg['timeframe'],cfg)
        if r and r['trades']>=15 and r.get('liq',0)==0 and r.get('ret',0)>0:
            r['_s']=score(r); r['_cfg']=cfg; p1.append(r)
        if (i+1)%20000==0: print(f"    {i+1:,}/{N:,} ({time.time()-t0:.0f}s) 유효:{len(p1):,}")
    print(f"  P1 완료: {len(p1):,}개 ({time.time()-t0:.0f}s)")
    p1.sort(key=lambda x:x['_s'],reverse=True)

    # Phase 2: 청산 (Top 20 x SL/Trail)
    slps2=[0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
    tas=[0.04,0.05,0.06,0.07,0.08,0.10,0.12,0.15]
    tps=[0.02,0.03,0.04,0.05,0.06]

    p2=[]; combo=0; t0=time.time()
    for br in p1[:20]:
        base=br['_cfg']
        for slp in slps2:
            for ta in tas:
                for tp in tps:
                    if tp>=ta: continue
                    cfg=dict(base); cfg['sl_pct']=slp; cfg['trail_activate']=ta; cfg['trail_pct']=tp
                    r=run_backtest(cache,cfg['timeframe'],cfg)
                    if r and r['trades']>=10 and r.get('liq',0)==0 and r.get('ret',0)>0:
                        r['_s']=score(r); r['_cfg']=cfg; p2.append(r)
                    combo+=1
    print(f"  P2: {combo:,}개 ({time.time()-t0:.0f}s), 유효:{len(p2):,}")
    p2.sort(key=lambda x:x['_s'],reverse=True)

    # Phase 3: 사이징 (Top 15 x 마진/보호)
    mgs=[0.15,0.20,0.25,0.30,0.35,0.40,0.50]
    mls=[0,-0.15,-0.20,-0.25,-0.30]
    dds=[0,-0.30,-0.50]

    p3=[]; combo2=0; t0=time.time()
    for br in p2[:15]:
        base=br['_cfg']
        for mg in mgs:
            for ml in mls:
                for dd in dds:
                    cfg=dict(base); cfg['margin_normal']=mg; cfg['margin_reduced']=mg/2
                    cfg['monthly_loss_limit']=ml; cfg['dd_threshold']=dd
                    r=run_backtest(cache,cfg['timeframe'],cfg)
                    if r and r['trades']>=5 and r.get('liq',0)==0 and r.get('ret',0)>0:
                        r['_s']=score(r); r['_cfg']=cfg; p3.append(r)
                    combo2+=1
    print(f"  P3: {combo2:,}개 ({time.time()-t0:.0f}s), 유효:{len(p3):,}")
    p3.sort(key=lambda x:x['_s'],reverse=True)

    total_combos=N+combo+combo2

    # 결과 출력
    # 스코어 Top 5
    print(f"\n  [{name}] 스코어 Top 5:")
    print(f"  {'#':>3} {'잔액':>12} {'PF':>7} {'MDD':>6} {'TR':>4} {'WR':>5} 설정")
    for i,r in enumerate(p3[:5]):
        print(f"  {i+1:>3} ${r['bal']:>10,.0f} {r['pf']:>6.2f} {r['mdd']:>5.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['cfg']}")

    # PF Top 5 (30+ trades)
    pf30=sorted([r for r in p3 if r['trades']>=30],key=lambda x:x['pf'],reverse=True)
    print(f"\n  [{name}] PF Top 5 (거래30+):")
    for i,r in enumerate(pf30[:5]):
        print(f"  {i+1:>3} ${r['bal']:>10,.0f} PF:{r['pf']:>6.2f} MDD:{r['mdd']:>5.1f}% TR:{r['trades']} {r['cfg']}")

    # 수익 Top 5 (PF>=1.5, 20+ trades)
    ret20=sorted([r for r in p3 if r['pf']>=1.5 and r['trades']>=20],key=lambda x:x['bal'],reverse=True)
    print(f"\n  [{name}] 수익 Top 5 (PF>=1.5, 거래20+):")
    for i,r in enumerate(ret20[:5]):
        print(f"  {i+1:>3} ${r['bal']:>10,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>5.1f}% TR:{r['trades']} {r['cfg']}")

    # Top 1 상세
    best=p3[0] if p3 else None
    if best:
        print(f"\n  [{name}] Top 1 상세:")
        print(f"    잔액: ${best['bal']:,.0f} ({best['ret']:+,.1f}%)")
        print(f"    PF: {best['pf']} | MDD: {best['mdd']}% | FL: {best['liq']}")
        print(f"    거래: {best['trades']} (SL:{best['sl']} TSL:{best['tsl']} REV:{best['sig']})")
        print(f"    승률: {best['wr']}% | 평균승: {best.get('avg_win',0):.2f}% | 평균패: {best.get('avg_loss',0):.2f}%")
        if best.get('yr'):
            print(f"    연도별: {' | '.join(f'{y}:{v:+.1f}%' for y,v in sorted(best['yr'].items()))}")

    return {'best':best, 'pf_top':pf30[:5] if pf30 else [],
            'ret_top':ret20[:5] if ret20 else [],
            'score_top':p3[:5], 'total_combos':total_combos}


def main():
    print("="*80)
    print("  v16.0 코인별 개별 최적화 (5코인 x 100,000+)")
    print("="*80)

    coin_info = [
        ('btc_usdt','BTC'), ('eth_usdt','ETH'), ('sol_usdt','SOL'),
        ('xrp_usdt','XRP'), ('doge_usdt','DOGE'),
    ]

    all_results={}
    grand_total=0

    for prefix, name in coin_info:
        print(f"\n  Loading {name}...")
        df=load_coin(prefix)
        print(f"    {len(df):,}행")
        mtf=build_mtf(df)
        cache=IndicatorCache(mtf)
        result=optimize_coin(name, cache)
        all_results[name]=result
        grand_total+=result['total_combos']

    # 30회 검증
    print(f"\n{'='*80}")
    print("  30회 반복 검증 (각 코인 Top 1)")
    print(f"{'='*80}")
    for prefix, name in coin_info:
        df=load_coin(prefix); mtf=build_mtf(df); cache=IndicatorCache(mtf)
        best=all_results[name]['best']
        if best:
            cfg=best['_cfg']
            bals=[run_backtest(cache,cfg['timeframe'],cfg)['bal'] for _ in range(30)]
            print(f"  {name}: ${np.mean(bals):>12,.0f} std={np.std(bals):.4f} {'PASS' if np.std(bals)<0.01 else 'FAIL'}")

    # 최종 포트폴리오
    print(f"\n{'='*80}")
    print("  $10,000 포트폴리오 시뮬레이션 (각 코인 최적 전략)")
    print(f"{'='*80}")
    alloc = {'BTC':0.30,'ETH':0.25,'SOL':0.20,'XRP':0.15,'DOGE':0.10}
    total_end=0
    print(f"  {'코인':>6} {'배분':>6} {'초기$':>8} {'최종$':>12} {'수익률':>10} {'PF':>7} {'MDD':>6} {'거래':>4} 전략")
    print(f"  {'-'*95}")
    for name in ['BTC','ETH','SOL','XRP','DOGE']:
        best=all_results[name]['best']
        if best:
            pct=alloc[name]; init=10000*pct
            ratio=best['bal']/3000  # $3000 기준 배수
            end=init*ratio
            total_end+=end
            print(f"  {name:>6} {pct:>5.0%} ${init:>6,.0f} ${end:>10,.0f} {(ratio-1)*100:>+9.1f}% {best['pf']:>6.2f} {best['mdd']:>5.1f}% {best['trades']:>4} {best['cfg'][:60]}")
    print(f"  {'-'*95}")
    print(f"  {'합계':>6} {'100%':>6} ${'10,000':>5} ${total_end:>10,.0f} {(total_end/10000-1)*100:>+9.1f}%")

    # 저장
    save={'grand_total':grand_total,'portfolio_alloc':alloc,'portfolio_end':round(total_end,0)}
    for name in ['BTC','ETH','SOL','XRP','DOGE']:
        best=all_results[name]['best']
        if best:
            save[name]={'bal':best['bal'],'ret':best['ret'],'pf':best['pf'],'mdd':best['mdd'],
                        'fl':best['liq'],'tr':best['trades'],'wr':best['wr'],
                        'sl':best.get('sl',0),'tsl':best.get('tsl',0),'rev':best.get('sig',0),
                        'yr':best.get('yr',{}), 'cfg':best['cfg'],
                        'avg_win':best.get('avg_win',0),'avg_loss':best.get('avg_loss',0)}
    with open(f'{BASE}/v16_percoin_results.json','w') as f:
        json.dump(save,f,indent=2,ensure_ascii=False)
    print(f"\n저장: v16_percoin_results.json")
    print(f"총 조합: {grand_total:,}")
    print("완료!")

if __name__=='__main__':
    main()
