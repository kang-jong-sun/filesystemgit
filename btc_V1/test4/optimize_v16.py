"""
BTC/USDT v16.0 전략 최적화 파이프라인
507,720+ 조합 4단계 최적화

핵심 혁신: Slow MA 타입 확장 (EMA 고정 → EMA/HMA/WMA/SMA)
"""
import numpy as np, json, os, time, sys
from bt_fast import load_5m_data, build_mtf, IndicatorCache, run_backtest

BASE = r'D:\filesystem\futures\btc_V1\test4'

def pf_score(r):
    if r is None or r.get('ret',0)<=0 or r.get('liq',0)>0: return 0
    pf=r['pf']; mdd=r['mdd']; ret=r['ret']; tr=r['trades']
    if tr<8: return 0
    pf_b = pf**1.5
    mdd_b = (100-mdd)/50
    ret_b = np.log1p(ret/100)
    tr_b = min(tr/30, 2.0)
    yr=r.get('yr',{})
    rec=[yr.get(str(y),0) for y in range(2023,2027) if str(y) in yr]
    yr_b=1.5 if rec and np.mean(rec)>50 else(1.2 if rec and np.mean(rec)>0 else 1.0)
    return pf_b*mdd_b*ret_b*tr_b*yr_b

def parse_cfg(r):
    """결과의 cfg 문자열에서 설정 추출"""
    s=r['cfg']; p=s.split('|')
    tf=p[0].strip()
    ma=p[1].strip(); mt=ma.split('(')[0].strip(); lens=ma.split('(')[1].rstrip(')').split('/')
    mf=int(lens[0]); ms=int(lens[1])
    ax=p[2].strip(); ap=int(ax.split('>=')[0].replace('A','')); am=float(ax.split('>=')[1])
    rx=p[3].strip(); rv=rx.split(':')[1].split('-'); rmin=float(rv[0]); rmax=float(rv[1])
    sx=p[4].strip(); slp=float(sx.replace('SL',''))
    tx=p[5].strip(); tv=tx.replace('TA','').split('/'); ta=float(tv[0]); tp=float(tv[1])
    lev=10
    for pp in p:
        pp=pp.strip()
        if 'L' in pp and 'x' in pp and 'M' in pp and not pp.startswith('ATR') and not pp.startswith('DE'):
            try: lev=int(pp.split('x')[0].replace('L','').strip())
            except: pass
    smt='ema'
    for pp in p:
        pp=pp.strip()
        if pp.startswith('sma(') or pp.startswith('wma(') or pp.startswith('hma('):
            smt=pp.split('(')[0]
    return {'tf':tf,'mt':mt,'smt':smt,'mf':mf,'ms':ms,'ap':ap,'am':am,
            'rmin':rmin,'rmax':rmax,'slp':slp,'ta':ta,'tp':tp,'lev':lev}

# ============================================================
# Phase 1: 진입 대탐색 (300,000)
# ============================================================
def phase1(cache):
    print(f"\n{'='*80}")
    print("  Phase 1: 진입 대탐색 (Slow MA 타입 확장)")
    print(f"{'='*80}")

    tfs=['5m','10m','15m','30m','1h']
    ma_f_types=['ema','hma','dema','wma']  # fast
    ma_s_types=['ema','hma','wma','sma']   # slow ← 핵심 확장!
    f_lens=[3,5,7,10,14,21]
    s_lens=[50,100,150,200,250]
    adx_ps=[14,20]
    adx_ms=[25,30,35,40,45]
    rsi_rs=[(25,60),(30,58),(30,65),(35,65),(35,70),(40,75)]
    sls=[0.03,0.04,0.05,0.06,0.07,0.08,0.09]

    configs=[]
    for tf in tfs:
        for mft in ma_f_types:
            for mst in ma_s_types:
                for fl in f_lens:
                    for sl in s_lens:
                        if fl>=sl: continue
                        for ap in adx_ps:
                            for am in adx_ms:
                                for rlo,rhi in rsi_rs:
                                    for slp in sls:
                                        configs.append({
                                            'timeframe':tf,
                                            'ma_fast_type':mft,'ma_slow_type':mst,
                                            'ma_fast':fl,'ma_slow':sl,
                                            'adx_period':ap,'adx_min':am,
                                            'rsi_period':14,'rsi_min':rlo,'rsi_max':rhi,
                                            'sl_pct':slp,
                                            'trail_activate':0.06,'trail_pct':0.03,
                                            'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
                                            'fee_rate':0.0004,'initial_capital':3000.0,
                                        })
    total=len(configs)
    N=min(total,300000)
    np.random.seed(42)
    if total>N:
        idx=np.random.choice(total,N,replace=False)
        configs=[configs[i] for i in idx]
    print(f"  전체 공간: {total:,} → 샘플: {N:,}")

    results=[]
    t0=time.time()
    for i,cfg in enumerate(configs):
        r=run_backtest(cache,cfg['timeframe'],cfg)
        if r and r['trades']>=8 and r.get('liq',0)==0 and r.get('ret',0)>0:
            r['_s']=pf_score(r); results.append(r)
        if (i+1)%50000==0:
            el=time.time()-t0
            print(f"    {i+1:,}/{N:,} ({el:.0f}s) 유효:{len(results):,}")

    print(f"  완료: {len(results):,}개 ({time.time()-t0:.0f}s)")
    results.sort(key=lambda x:x['_s'],reverse=True)

    print(f"\n  Top 30 (PF×MDD×수익 스코어):")
    print(f"  {'#':>3} {'잔액':>10} {'PF':>6} {'MDD':>5} {'TR':>4} {'WR':>5} {'점수':>6} 설정")
    for i,r in enumerate(results[:30]):
        print(f"  {i+1:>3} ${r['bal']:>8,.0f} {r['pf']:>5.2f} {r['mdd']:>4.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['_s']:>5.0f} {r['cfg']}")

    # PF Top 10 (30+ trades)
    pf_top=[r for r in results if r['trades']>=30]
    pf_top.sort(key=lambda x:x['pf'],reverse=True)
    print(f"\n  PF Top 10 (거래30+):")
    for i,r in enumerate(pf_top[:10]):
        print(f"  {i+1:>3} ${r['bal']:>8,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} {r['cfg']}")

    return results[:50]

# ============================================================
# Phase 2: 청산 최적화 (192,000)
# ============================================================
def phase2(cache,p1):
    print(f"\n{'='*80}")
    print("  Phase 2: 청산 최적화")
    print(f"{'='*80}")

    sls=[0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
    tas=[0.04,0.05,0.06,0.07,0.08,0.10,0.12,0.15]
    tps=[0.02,0.03,0.04,0.05,0.06]
    atr_sl_opts=[
        {'use_atr_sl':False},
        {'use_atr_sl':True,'atr_sl_mult':1.5,'atr_sl_min':0.02,'atr_sl_max':0.10},
        {'use_atr_sl':True,'atr_sl_mult':2.0,'atr_sl_min':0.03,'atr_sl_max':0.12},
        {'use_atr_sl':True,'atr_sl_mult':2.5,'atr_sl_min':0.03,'atr_sl_max':0.15},
    ]
    atr_tr_opts=[
        {'use_atr_trail':False},
        {'use_atr_trail':True,'atr_trail_mult':1.0},
        {'use_atr_trail':True,'atr_trail_mult':1.5},
    ]

    results=[]
    combo=0
    t0=time.time()
    for br in p1[:50]:
        c=parse_cfg(br)
        for slp in sls:
            for ta in tas:
                for tp in tps:
                    if tp>=ta: continue
                    for asl in atr_sl_opts:
                        for atr in atr_tr_opts:
                            cfg={
                                'timeframe':c['tf'],
                                'ma_fast_type':c['mt'],'ma_slow_type':c['smt'],
                                'ma_fast':c['mf'],'ma_slow':c['ms'],
                                'adx_period':c['ap'],'adx_min':c['am'],
                                'rsi_period':14,'rsi_min':c['rmin'],'rsi_max':c['rmax'],
                                'sl_pct':slp,'trail_activate':ta,'trail_pct':tp,
                                'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
                                'fee_rate':0.0004,'initial_capital':3000.0,
                            }
                            cfg.update(asl); cfg.update(atr)
                            r=run_backtest(cache,c['tf'],cfg)
                            if r and r['trades']>=8 and r.get('liq',0)==0 and r.get('ret',0)>0:
                                r['_s']=pf_score(r); results.append(r)
                            combo+=1
        if combo%50000==0:
            print(f"    {combo:,} ({time.time()-t0:.0f}s) 유효:{len(results):,}")

    print(f"  {combo:,}개 조합, {len(results):,}개 유효 ({time.time()-t0:.0f}s)")
    results.sort(key=lambda x:x['_s'],reverse=True)

    print(f"\n  Top 20:")
    print(f"  {'#':>3} {'잔액':>10} {'PF':>6} {'MDD':>5} {'TR':>4} {'WR':>5} {'점수':>6} 설정")
    for i,r in enumerate(results[:20]):
        print(f"  {i+1:>3} ${r['bal']:>8,.0f} {r['pf']:>5.2f} {r['mdd']:>4.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['_s']:>5.0f} {r['cfg']}")

    return results[:50]

# ============================================================
# Phase 3: 지연진입 (600)
# ============================================================
def phase3(cache,p2):
    print(f"\n{'='*80}")
    print("  Phase 3: 지연진입")
    print(f"{'='*80}")
    delay_opts=[
        {},
        {'delayed_entry':True,'delay_max_candles':3,'delay_price_min':-0.001,'delay_price_max':-0.010},
        {'delayed_entry':True,'delay_max_candles':3,'delay_price_min':-0.001,'delay_price_max':-0.015},
        {'delayed_entry':True,'delay_max_candles':3,'delay_price_min':-0.001,'delay_price_max':-0.025},
        {'delayed_entry':True,'delay_max_candles':3,'delay_price_min':-0.002,'delay_price_max':-0.020},
        {'delayed_entry':True,'delay_max_candles':6,'delay_price_min':-0.001,'delay_price_max':-0.010},
        {'delayed_entry':True,'delay_max_candles':6,'delay_price_min':-0.001,'delay_price_max':-0.015},
        {'delayed_entry':True,'delay_max_candles':6,'delay_price_min':-0.001,'delay_price_max':-0.025},
        {'delayed_entry':True,'delay_max_candles':6,'delay_price_min':-0.002,'delay_price_max':-0.020},
        {'delayed_entry':True,'delay_max_candles':9,'delay_price_min':-0.001,'delay_price_max':-0.015},
        {'delayed_entry':True,'delay_max_candles':9,'delay_price_min':-0.001,'delay_price_max':-0.025},
        {'delayed_entry':True,'delay_max_candles':12,'delay_price_min':-0.001,'delay_price_max':-0.020},
        {'delayed_entry':True,'delay_max_candles':12,'delay_price_min':-0.001,'delay_price_max':-0.025},
        {'delayed_entry':True,'delay_max_candles':3,'delay_price_min':-0.001,'delay_price_max':-0.005},
        {'delayed_entry':True,'delay_max_candles':6,'delay_price_min':-0.001,'delay_price_max':-0.005},
        {'delayed_entry':True,'delay_max_candles':3,'delay_price_min':-0.003,'delay_price_max':-0.015},
        {'delayed_entry':True,'delay_max_candles':6,'delay_price_min':-0.003,'delay_price_max':-0.020},
        {'delayed_entry':True,'delay_max_candles':9,'delay_price_min':-0.002,'delay_price_max':-0.015},
        {'delayed_entry':True,'delay_max_candles':6,'delay_price_min':-0.002,'delay_price_max':-0.010},
        {'delayed_entry':True,'delay_max_candles':12,'delay_price_min':-0.002,'delay_price_max':-0.030},
    ]

    results=[]
    combo=0
    t0=time.time()
    for br in p2[:30]:
        c=parse_cfg(br)
        # ATR 관련 파싱
        has_atr_sl='ATR_SL' in br['cfg']
        has_atr_tr='ATR_TS' in br['cfg']
        atr_cfg={}
        if has_atr_sl:
            for part in br['cfg'].split('|'):
                part=part.strip()
                if part.startswith('ATR_SL'):
                    mult=float(part.replace('ATR_SL',''))
                    atr_cfg.update({'use_atr_sl':True,'atr_sl_mult':mult,'atr_sl_min':0.02,'atr_sl_max':0.12})
        if has_atr_tr:
            for part in br['cfg'].split('|'):
                part=part.strip()
                if part.startswith('ATR_TS'):
                    mult=float(part.replace('ATR_TS',''))
                    atr_cfg.update({'use_atr_trail':True,'atr_trail_mult':mult})

        for dopt in delay_opts:
            cfg={
                'timeframe':c['tf'],
                'ma_fast_type':c['mt'],'ma_slow_type':c['smt'],
                'ma_fast':c['mf'],'ma_slow':c['ms'],
                'adx_period':c['ap'],'adx_min':c['am'],
                'rsi_period':14,'rsi_min':c['rmin'],'rsi_max':c['rmax'],
                'sl_pct':c['slp'],'trail_activate':c['ta'],'trail_pct':c['tp'],
                'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
                'fee_rate':0.0004,'initial_capital':3000.0,
            }
            cfg.update(atr_cfg)
            cfg.update(dopt)
            r=run_backtest(cache,c['tf'],cfg)
            if r and r['trades']>=5 and r.get('liq',0)==0 and r.get('ret',0)>0:
                r['_s']=pf_score(r); results.append(r)
            combo+=1

    print(f"  {combo}개 조합, {len(results)}개 유효 ({time.time()-t0:.0f}s)")
    results.sort(key=lambda x:x['_s'],reverse=True)
    print(f"\n  Top 10:")
    for i,r in enumerate(results[:10]):
        print(f"  {i+1:>3} ${r['bal']:>8,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} {r['cfg']}")
    return results[:30]

# ============================================================
# Phase 4: 보호+사이징 (15,120)
# ============================================================
def phase4(cache,p3):
    print(f"\n{'='*80}")
    print("  Phase 4: 보호+사이징")
    print(f"{'='*80}")
    mgs=[0.15,0.20,0.25,0.30,0.35,0.40,0.50]
    mls=[0,-0.10,-0.15,-0.20,-0.25,-0.30]
    cps=[{'consec_loss_pause':0},{'consec_loss_pause':3,'pause_candles':288},{'consec_loss_pause':5,'pause_candles':576}]
    dds=[0,-0.25,-0.30,-0.50]

    results=[]
    combo=0
    t0=time.time()
    for br in p3[:30]:
        c=parse_cfg(br)
        has_delay='DE' in br['cfg']
        dcfg={}
        if has_delay:
            for part in br['cfg'].split('|'):
                part=part.strip()
                if part.startswith('DE'):
                    dcfg={'delayed_entry':True,'delay_max_candles':int(part.replace('DE','').replace('c','')),'delay_price_min':-0.001,'delay_price_max':-0.025}
        atr_cfg={}
        if 'ATR_SL' in br['cfg']:
            for part in br['cfg'].split('|'):
                part=part.strip()
                if part.startswith('ATR_SL'):
                    atr_cfg.update({'use_atr_sl':True,'atr_sl_mult':float(part.replace('ATR_SL','')),'atr_sl_min':0.02,'atr_sl_max':0.12})
        if 'ATR_TS' in br['cfg']:
            for part in br['cfg'].split('|'):
                part=part.strip()
                if part.startswith('ATR_TS'):
                    atr_cfg.update({'use_atr_trail':True,'atr_trail_mult':float(part.replace('ATR_TS',''))})

        for mg in mgs:
            for ml in mls:
                for cp in cps:
                    for dd in dds:
                        cfg={
                            'timeframe':c['tf'],
                            'ma_fast_type':c['mt'],'ma_slow_type':c['smt'],
                            'ma_fast':c['mf'],'ma_slow':c['ms'],
                            'adx_period':c['ap'],'adx_min':c['am'],
                            'rsi_period':14,'rsi_min':c['rmin'],'rsi_max':c['rmax'],
                            'sl_pct':c['slp'],'trail_activate':c['ta'],'trail_pct':c['tp'],
                            'leverage':10,'margin_normal':mg,'margin_reduced':mg/2,
                            'monthly_loss_limit':ml,'dd_threshold':dd,
                            'fee_rate':0.0004,'initial_capital':3000.0,
                        }
                        cfg.update(cp); cfg.update(dcfg); cfg.update(atr_cfg)
                        r=run_backtest(cache,c['tf'],cfg)
                        if r and r['trades']>=5 and r.get('liq',0)==0 and r.get('ret',0)>0:
                            r['_s']=pf_score(r); results.append(r)
                        combo+=1
        if combo%5000==0:
            print(f"    {combo:,} ({time.time()-t0:.0f}s)")

    print(f"  {combo:,}개 조합, {len(results):,}개 유효 ({time.time()-t0:.0f}s)")
    results.sort(key=lambda x:x['_s'],reverse=True)

    print(f"\n  === 최종 Top 30 ===")
    print(f"  {'#':>3} {'잔액':>12} {'PF':>6} {'MDD':>5} {'TR':>4} {'WR':>5} {'점수':>6} 설정")
    for i,r in enumerate(results[:30]):
        print(f"  {i+1:>3} ${r['bal']:>10,.0f} {r['pf']:>5.2f} {r['mdd']:>4.1f}% {r['trades']:>4} {r['wr']:>4.1f}% {r['_s']:>5.0f} {r['cfg']}")

    # PF Top 10 (30+ trades)
    pf30=[r for r in results if r['trades']>=30]
    pf30.sort(key=lambda x:x['pf'],reverse=True)
    print(f"\n  === PF Top 10 (거래30+, FL=0) ===")
    for i,r in enumerate(pf30[:10]):
        print(f"  {i+1:>3} ${r['bal']:>10,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} WR:{r['wr']:.1f}% {r['cfg']}")

    # MDD Top 10 (PF>=3, 30+ trades)
    mdd30=[r for r in results if r['pf']>=3 and r['trades']>=30]
    mdd30.sort(key=lambda x:x['mdd'])
    print(f"\n  === MDD Top 10 (PF>=3, 거래30+) ===")
    for i,r in enumerate(mdd30[:10]):
        print(f"  {i+1:>3} ${r['bal']:>10,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} {r['cfg']}")

    # 수익 Top 10 (PF>=2, 30+ trades)
    ret30=[r for r in results if r['pf']>=2 and r['trades']>=30]
    ret30.sort(key=lambda x:x['bal'],reverse=True)
    print(f"\n  === 수익 Top 10 (PF>=2, 거래30+) ===")
    for i,r in enumerate(ret30[:10]):
        print(f"  {i+1:>3} ${r['bal']:>10,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} {r['cfg']}")

    return results

# ============================================================
# 메인
# ============================================================
def main():
    print("="*80)
    print("  BTC/USDT v16.0 전략 최적화 - 507,720+ 조합")
    print("="*80)

    df5=load_5m_data(BASE)
    mtf=build_mtf(df5)
    cache=IndicatorCache(mtf)
    total=0

    p1=phase1(cache); total+=300000
    p2=phase2(cache,p1); total+=192000
    p3=phase3(cache,p2); total+=600
    p4=phase4(cache,p3); total+=15120

    print(f"\n{'='*80}")
    print(f"  총 조합 수: {total:,}")
    print(f"{'='*80}")

    # 저장
    save={'total':total}
    cats=[('score_top30',sorted(p4,key=lambda x:x.get('_s',0),reverse=True)[:30]),
          ('pf_30tr_top10',sorted([r for r in p4 if r['trades']>=30],key=lambda x:x['pf'],reverse=True)[:10]),
          ('mdd_pf3_30tr',sorted([r for r in p4 if r['pf']>=3 and r['trades']>=30],key=lambda x:x['mdd'])[:10]),
          ('ret_pf2_30tr',sorted([r for r in p4 if r['pf']>=2 and r['trades']>=30],key=lambda x:x['bal'],reverse=True)[:10])]
    for k,lst in cats:
        save[k]=[{'bal':r['bal'],'ret':r['ret'],'pf':r['pf'],'mdd':r['mdd'],'fl':r['liq'],'tr':r['trades'],
                   'wr':r['wr'],'cfg':r['cfg'],'yr':r.get('yr',{}),'sl':r.get('sl',0),
                   'tsl':r.get('tsl',0),'rev':r.get('sig',0)} for r in lst]
    with open(f'{BASE}/v16_results.json','w') as f:
        json.dump(save,f,indent=2,ensure_ascii=False)
    print(f"\n저장: v16_results.json")
    print("완료!")

if __name__=='__main__':
    main()
