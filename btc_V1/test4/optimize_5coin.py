"""
v16.0 5코인 공통 전략 최적화
BTC + ETH + SOL + XRP + DOGE
$10,000 / 5코인 포트폴리오
"""
import numpy as np, pandas as pd, json, os, time
from bt_fast import build_mtf, IndicatorCache, run_backtest

BASE = r'D:\filesystem\futures\btc_V1\test4'

def load_coin(prefix):
    parts = []
    for i in range(1, 4):
        f = os.path.join(BASE, f'{prefix}_5m_2020_to_now_part{i}.csv')
        if os.path.exists(f):
            parts.append(pd.read_csv(f, parse_dates=['timestamp']))
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    df = df.rename(columns={'timestamp':'time'})
    for c in ['open','high','low','close','volume']: df[c]=df[c].astype(float)
    print(f"  {prefix}: {len(df):,}행 ({df['time'].iloc[0]} ~ {df['time'].iloc[-1]})")
    return df

def score5(results):
    """5코인 공통 스코어: 5개 모두 양수 필수, 최약 코인 기준"""
    if any(r is None or r.get('ret',0)<=0 or r.get('liq',0)>0 for r in results): return 0
    pfs=[r['pf'] for r in results]; mdds=[r['mdd'] for r in results]; trs=[r['trades'] for r in results]
    if any(t<5 for t in trs): return 0
    min_pf=min(pfs); avg_pf=np.mean(pfs); max_mdd=max(mdds); min_tr=min(trs)
    return (min_pf**1.2)*(avg_pf**0.3)*((100-max_mdd)/50)*np.log1p(np.mean([r['ret'] for r in results])/100)*min(min_tr/20,2.0)

def main():
    print("="*80)
    print("  v16.0 - 5코인 공통 전략 최적화")
    print("  BTC + ETH + SOL + XRP + DOGE | $10,000 포트폴리오")
    print("="*80)

    # 데이터 로드
    print("\n[1] 데이터 로드")
    coin_names = ['BTC','ETH','SOL','XRP','DOGE']
    prefixes = ['btc_usdt','eth_usdt','sol_usdt','xrp_usdt','doge_usdt']
    caches = {}
    for prefix, name in zip(prefixes, coin_names):
        df = load_coin(prefix)
        mtf = build_mtf(df)
        caches[name] = IndicatorCache(mtf)
        print(f"    {name} MTF OK")

    # Phase 1: 대탐색
    print(f"\n{'='*80}")
    print("  Phase 1: 5코인 공통 진입 탐색 (250,000 샘플)")
    print(f"{'='*80}")

    tfs=['5m','10m','15m','30m','1h']
    mfts=['ema','hma','dema','wma']
    msts=['ema','wma','sma']
    fls=[3,5,7,10,14,21]
    sls_len=[50,100,150,200,250]
    aps=[14,20]; ams=[25,30,35,40]
    rrs=[(25,60),(30,58),(30,65),(35,65),(35,70),(40,75)]
    slps=[0.04,0.05,0.06,0.07,0.08]
    tas=[0.04,0.05,0.06,0.07,0.08,0.10]
    tps=[0.03,0.04,0.05]

    configs=[]
    for tf in tfs:
        for mft in mfts:
            for mst in msts:
                for fl in fls:
                    for sl in sls_len:
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
    N=min(total,250000)
    np.random.seed(42)
    if total>N: configs=[configs[i] for i in np.random.choice(total,N,replace=False)]
    print(f"  전체: {total:,} -> 샘플: {N:,}")

    results=[]; t0=time.time()
    for i,cfg in enumerate(configs):
        tf=cfg['timeframe']
        rs=[]; ok=True
        for name in coin_names:
            r=run_backtest(caches[name],tf,cfg)
            if r is None or r.get('liq',0)>0 or r.get('ret',0)<=0 or r['trades']<5:
                ok=False; break
            rs.append(r)
        if ok and len(rs)==5:
            sc=score5(rs)
            if sc>0:
                entry={'cfg':rs[0]['cfg'],'_s':sc,
                       'min_pf':min(r['pf'] for r in rs),'avg_pf':np.mean([r['pf'] for r in rs]),
                       'max_mdd':max(r['mdd'] for r in rs),'sum_bal':sum(r['bal'] for r in rs)}
                for j,name in enumerate(coin_names):
                    entry[name]={'bal':rs[j]['bal'],'ret':rs[j]['ret'],'pf':rs[j]['pf'],
                                 'mdd':rs[j]['mdd'],'tr':rs[j]['trades'],'wr':rs[j]['wr'],
                                 'fl':rs[j]['liq'],'yr':rs[j].get('yr',{}),
                                 'sl':rs[j].get('sl',0),'tsl':rs[j].get('tsl',0),'rev':rs[j].get('sig',0)}
                results.append(entry)
        if (i+1)%50000==0:
            print(f"    {i+1:,}/{N:,} ({time.time()-t0:.0f}s) 5코인공통: {len(results):,}")

    print(f"  완료: {len(results):,}개 ({time.time()-t0:.0f}s)")
    results.sort(key=lambda x:x['_s'],reverse=True)

    print(f"\n  === 5코인 공통 스코어 Top 20 ===")
    print(f"  {'#':>3} {'합계$':>10} {'minPF':>6} {'avgPF':>6} {'maxMDD':>6} {'BTC':>7} {'ETH':>7} {'SOL':>7} {'XRP':>7} {'DOGE':>7} 설정")
    for i,r in enumerate(results[:20]):
        print(f"  {i+1:>3} ${r['sum_bal']:>8,.0f} {r['min_pf']:>5.2f} {r['avg_pf']:>5.2f} {r['max_mdd']:>5.1f}% ${r['BTC']['bal']:>5,.0f} ${r['ETH']['bal']:>5,.0f} ${r['SOL']['bal']:>5,.0f} ${r['XRP']['bal']:>5,.0f} ${r['DOGE']['bal']:>5,.0f} {r['cfg']}")

    # Phase 2: 사이징 최적화
    print(f"\n{'='*80}")
    print("  Phase 2: 사이징 최적화 (Top 20 x 마진/보호)")
    print(f"{'='*80}")

    mgs=[0.15,0.20,0.25,0.30,0.35,0.40]
    ml_opts=[0,-0.15,-0.20,-0.25]
    dd_opts=[0,-0.30,-0.50]

    p2=[]; combo=0; t0=time.time()
    for br in results[:20]:
        s=br['cfg']; p=[x.strip() for x in s.split('|')]
        tf=p[0]; ma=p[1]; mt=ma.split('(')[0]; lens=ma.split('(')[1].rstrip(')').split('/')
        mf=int(lens[0]); ms=int(lens[1])
        ax=p[2]; ap=int(ax.split('>=')[0].replace('A','')); am=float(ax.split('>=')[1])
        rv=p[3].split(':')[1].split('-'); rmin=float(rv[0]); rmax=float(rv[1])
        slp=float(p[4].replace('SL','')); tv=p[5].replace('TA','').split('/'); ta=float(tv[0]); tp=float(tv[1])

        for mg in mgs:
            for ml in ml_opts:
                for dd in dd_opts:
                    cfg={'timeframe':tf,'ma_fast_type':mt,'ma_slow_type':'ema',
                         'ma_fast':mf,'ma_slow':ms,'adx_period':ap,'adx_min':am,
                         'rsi_period':14,'rsi_min':rmin,'rsi_max':rmax,
                         'sl_pct':slp,'trail_activate':ta,'trail_pct':tp,
                         'leverage':10,'margin_normal':mg,'margin_reduced':mg/2,
                         'monthly_loss_limit':ml,'dd_threshold':dd,
                         'fee_rate':0.0004,'initial_capital':3000.0}
                    rs=[]; ok=True
                    for name in coin_names:
                        r=run_backtest(caches[name],tf,cfg)
                        if r is None or r.get('liq',0)>0 or r.get('ret',0)<=0:
                            ok=False; break
                        rs.append(r)
                    if ok and len(rs)==5:
                        sc=score5(rs)
                        if sc>0:
                            entry={'cfg':rs[0]['cfg'],'_s':sc,
                                   'min_pf':min(r['pf'] for r in rs),'avg_pf':np.mean([r['pf'] for r in rs]),
                                   'max_mdd':max(r['mdd'] for r in rs),'sum_bal':sum(r['bal'] for r in rs)}
                            for j,name in enumerate(coin_names):
                                entry[name]={'bal':rs[j]['bal'],'ret':rs[j]['ret'],'pf':rs[j]['pf'],
                                             'mdd':rs[j]['mdd'],'tr':rs[j]['trades'],'wr':rs[j]['wr'],
                                             'fl':rs[j]['liq'],'yr':rs[j].get('yr',{}),
                                             'sl':rs[j].get('sl',0),'tsl':rs[j].get('tsl',0),'rev':rs[j].get('sig',0)}
                            p2.append(entry)
                    combo+=1

    print(f"  {combo}개 ({time.time()-t0:.0f}s), 유효 {len(p2)}")
    p2.sort(key=lambda x:x['_s'],reverse=True)

    # 최종 결과
    print(f"\n{'='*80}")
    print("  === 최종 Top 15 ===")
    print(f"{'='*80}")
    print(f"  {'#':>3} {'합계$':>10} {'minPF':>6} {'avgPF':>6} {'maxMDD':>6} {'BTC$':>8} {'ETH$':>8} {'SOL$':>8} {'XRP$':>8} {'DOGE$':>8} 설정")
    for i,r in enumerate(p2[:15]):
        print(f"  {i+1:>3} ${r['sum_bal']:>8,.0f} {r['min_pf']:>5.2f} {r['avg_pf']:>5.2f} {r['max_mdd']:>5.1f}% ${r['BTC']['bal']:>6,.0f} ${r['ETH']['bal']:>6,.0f} ${r['SOL']['bal']:>6,.0f} ${r['XRP']['bal']:>6,.0f} ${r['DOGE']['bal']:>6,.0f} {r['cfg']}")

    # Top 1 상세
    if p2:
        best=p2[0]
        print(f"\n{'='*80}")
        print("  Top 1 상세")
        print(f"{'='*80}")
        print(f"  설정: {best['cfg']}")
        print(f"  합계 잔액: ${best['sum_bal']:,.0f}")
        print(f"  최소PF: {best['min_pf']:.2f} | 평균PF: {best['avg_pf']:.2f} | 최대MDD: {best['max_mdd']:.1f}%")
        for name in coin_names:
            c=best[name]
            print(f"\n  [{name}] ${c['bal']:,.0f} ({c['ret']:+,.1f}%) PF:{c['pf']:.2f} MDD:{c['mdd']:.1f}% TR:{c['tr']} WR:{c['wr']:.1f}% FL:{c['fl']}")
            print(f"    SL:{c['sl']} TSL:{c['tsl']} REV:{c['rev']}")
            if c.get('yr'):
                print(f"    연도: {' | '.join(f'{y}:{v:+.1f}%' for y,v in sorted(c['yr'].items()))}")

    # 30회 검증
    print(f"\n{'='*80}")
    print("  30회 반복 검증 (Top 1)")
    print(f"{'='*80}")
    if p2:
        best=p2[0]
        s=best['cfg']; p=[x.strip() for x in s.split('|')]
        tf=p[0]; ma=p[1]; mt=ma.split('(')[0]; lens=ma.split('(')[1].rstrip(')').split('/')
        mf=int(lens[0]); ms=int(lens[1])
        ax=p[2]; ap=int(ax.split('>=')[0].replace('A','')); am=float(ax.split('>=')[1])
        rv=p[3].split(':')[1].split('-'); rmin=float(rv[0]); rmax=float(rv[1])
        slp=float(p[4].replace('SL','')); tv=p[5].replace('TA','').split('/'); ta=float(tv[0]); tp=float(tv[1])
        # margin/ML/DD 파싱
        mg_val=0.25; ml_val=0; dd_val=0
        for pp in p:
            pp=pp.strip()
            if pp.startswith('L') and 'M' in pp:
                try: mg_val=float(pp.split('M')[1].replace('%',''))/100
                except: pass
            if pp.startswith('ML'):
                try: ml_val=float(pp.replace('ML','').replace('%',''))/100
                except: pass
            if pp.startswith('DD'):
                try: dd_val=float(pp.replace('DD','').replace('%',''))/100
                except: pass

        cfg={'timeframe':tf,'ma_fast_type':mt,'ma_slow_type':'ema',
             'ma_fast':mf,'ma_slow':ms,'adx_period':ap,'adx_min':am,
             'rsi_period':14,'rsi_min':rmin,'rsi_max':rmax,
             'sl_pct':slp,'trail_activate':ta,'trail_pct':tp,
             'leverage':10,'margin_normal':mg_val,'margin_reduced':mg_val/2,
             'monthly_loss_limit':ml_val,'dd_threshold':dd_val,
             'fee_rate':0.0004,'initial_capital':3000.0}

        for name in coin_names:
            bals=[run_backtest(caches[name],tf,cfg)['bal'] for _ in range(30)]
            print(f"  {name}: ${np.mean(bals):>10,.0f} std={np.std(bals):.4f} {'PASS' if np.std(bals)<0.01 else 'FAIL'}")

    # 저장
    save={'total_combos':N+combo,'top15':[]}
    for r in p2[:15]:
        entry={'cfg':r['cfg'],'sum_bal':r['sum_bal'],'min_pf':round(r['min_pf'],2),
               'avg_pf':round(r['avg_pf'],2),'max_mdd':round(r['max_mdd'],1)}
        for name in coin_names: entry[name]=r[name]
        save['top15'].append(entry)
    with open(f'{BASE}/v16_5coin_results.json','w') as f:
        json.dump(save,f,indent=2,ensure_ascii=False)
    print(f"\n저장: v16_5coin_results.json")
    print(f"총 조합: {N+combo:,}")
    print("완료!")

if __name__=='__main__':
    main()
