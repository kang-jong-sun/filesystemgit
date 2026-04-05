"""Phase 1&2 결과 로드 후 Phase 3&4 이어서 실행 + 30회 검증 + 기획서 데이터"""
import numpy as np, json, time
from bt_fast import load_5m_data, build_mtf, IndicatorCache, run_backtest

BASE = r'D:\filesystem\futures\btc_V1\test4'

def pf_score(r):
    if r is None or r.get('ret',0)<=0 or r.get('liq',0)>0: return 0
    pf=r['pf'];mdd=r['mdd'];ret=r['ret'];tr=r['trades']
    if tr<8: return 0
    return (pf**1.5)*((100-mdd)/50)*np.log1p(ret/100)*min(tr/30,2.0)*(1.5 if any(r.get('yr',{}).get(str(y),0)>50 for y in range(2023,2027)) else 1.0)

def parse_cfg_safe(cfg_str):
    """안전한 cfg 파서"""
    p = [x.strip() for x in cfg_str.split('|')]
    d = {'tf':'30m','mt':'ema','smt':'ema','mf':3,'ms':200,'ap':14,'am':35,
         'rmin':30,'rmax':65,'slp':0.07,'ta':0.06,'tp':0.03,'lev':10}
    try:
        d['tf'] = p[0]
        ma = p[1]; d['mt'] = ma.split('(')[0]; lens = ma.split('(')[1].rstrip(')').split('/')
        d['mf'] = int(lens[0]); d['ms'] = int(lens[1])
        ax = p[2]; d['ap'] = int(ax.split('>=')[0].replace('A','')); d['am'] = float(ax.split('>=')[1])
        rv = p[3].split(':')[1].split('-'); d['rmin'] = float(rv[0]); d['rmax'] = float(rv[1])
        d['slp'] = float(p[4].replace('SL',''))
        tv = p[5].replace('TA','').split('/'); d['ta'] = float(tv[0]); d['tp'] = float(tv[1])
    except: pass
    for pp in p:
        if pp.startswith('sma(') or pp.startswith('wma(') or pp.startswith('hma('): d['smt']=pp.split('(')[0]
        if 'L' in pp and 'x' in pp and 'M' in pp and not pp.startswith('ATR'):
            try: d['lev']=int(pp.split('x')[0].replace('L','').strip())
            except: pass
    return d

def make_cfg(c, extra={}):
    cfg = {
        'timeframe':c['tf'],'ma_fast_type':c['mt'],'ma_slow_type':c['smt'],
        'ma_fast':c['mf'],'ma_slow':c['ms'],'adx_period':c['ap'],'adx_min':c['am'],
        'rsi_period':14,'rsi_min':c['rmin'],'rsi_max':c['rmax'],
        'sl_pct':c['slp'],'trail_activate':c['ta'],'trail_pct':c['tp'],
        'leverage':c['lev'],'margin_normal':0.25,'margin_reduced':0.125,
        'fee_rate':0.0004,'initial_capital':3000.0,
    }
    cfg.update(extra)
    return cfg

def main():
    print("="*80)
    print("  v16.0 Phase 3&4 + 30회 검증")
    print("="*80)

    df5 = load_5m_data(BASE)
    mtf = build_mtf(df5)
    cache = IndicatorCache(mtf)

    # Phase 1 PF Top (30+ trades) 직접 재실행
    print("\n[Phase 1 PF Top 재실행]")
    p1_winners = [
        {'timeframe':'30m','ma_fast_type':'wma','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,'adx_period':20,'adx_min':35,'rsi_period':14,'rsi_min':35,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,'fee_rate':0.0004,'initial_capital':3000.0},
        {'timeframe':'30m','ma_fast_type':'wma','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,'adx_period':20,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,'sl_pct':0.06,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,'fee_rate':0.0004,'initial_capital':3000.0},
        {'timeframe':'30m','ma_fast_type':'wma','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,'adx_period':20,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,'sl_pct':0.08,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,'fee_rate':0.0004,'initial_capital':3000.0},
        {'timeframe':'15m','ma_fast_type':'wma','ma_slow_type':'ema','ma_fast':5,'ma_slow':150,'adx_period':14,'adx_min':40,'rsi_period':14,'rsi_min':35,'rsi_max':65,'sl_pct':0.06,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,'fee_rate':0.0004,'initial_capital':3000.0},
        {'timeframe':'15m','ma_fast_type':'dema','ma_slow_type':'ema','ma_fast':7,'ma_slow':200,'adx_period':20,'adx_min':35,'rsi_period':14,'rsi_min':35,'rsi_max':70,'sl_pct':0.09,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,'fee_rate':0.0004,'initial_capital':3000.0},
        {'timeframe':'30m','ma_fast_type':'wma','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,'adx_period':20,'adx_min':35,'rsi_period':14,'rsi_min':25,'rsi_max':60,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,'fee_rate':0.0004,'initial_capital':3000.0},
        {'timeframe':'1h','ma_fast_type':'hma','ma_slow_type':'ema','ma_fast':10,'ma_slow':200,'adx_period':20,'adx_min':30,'rsi_period':14,'rsi_min':35,'rsi_max':70,'sl_pct':0.08,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,'fee_rate':0.0004,'initial_capital':3000.0},
        {'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':35,'rsi_max':65,'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,'fee_rate':0.0004,'initial_capital':3000.0},
    ]

    p1_results = []
    for cfg in p1_winners:
        r = run_backtest(cache, cfg['timeframe'], cfg)
        if r:
            r['_s'] = pf_score(r)
            r['_cfg'] = cfg
            p1_results.append(r)
            print(f"  ${r['bal']:>10,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:.1f}% TR:{r['trades']} {r['cfg']}")

    # Phase 2: 청산 최적화 (이 위너들에 대해)
    print(f"\n[Phase 2: 청산 최적화]")
    sls=[0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
    tas=[0.04,0.05,0.06,0.07,0.08,0.10,0.12,0.15]
    tps=[0.02,0.03,0.04,0.05,0.06]

    p2_results = []
    combo = 0
    t0 = time.time()
    for base_cfg in p1_winners:
        for slp in sls:
            for ta in tas:
                for tp in tps:
                    if tp >= ta: continue
                    cfg = dict(base_cfg)
                    cfg['sl_pct'] = slp; cfg['trail_activate'] = ta; cfg['trail_pct'] = tp
                    r = run_backtest(cache, cfg['timeframe'], cfg)
                    if r and r['trades']>=8 and r.get('liq',0)==0 and r.get('ret',0)>0:
                        r['_s']=pf_score(r); r['_cfg']=cfg; p2_results.append(r)
                    combo += 1
    print(f"  {combo:,}개 ({time.time()-t0:.0f}s), 유효 {len(p2_results):,}")

    p2_results.sort(key=lambda x:x['_s'], reverse=True)
    print(f"\n  Phase 2 Top 10:")
    for i,r in enumerate(p2_results[:10]):
        print(f"  {i+1:>3} ${r['bal']:>8,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} {r['cfg']}")

    # Phase 4: 보호+사이징 (Phase 3 스킵 - 즉시진입이 우수)
    print(f"\n[Phase 4: 보호+사이징]")
    mgs=[0.15,0.20,0.25,0.30,0.35,0.40,0.50]
    mls=[0,-0.10,-0.15,-0.20,-0.25,-0.30]
    dds=[0,-0.25,-0.30,-0.50]

    p4_results = []
    combo = 0
    t0 = time.time()
    for br in p2_results[:30]:
        cfg_base = br['_cfg']
        for mg in mgs:
            for ml in mls:
                for dd in dds:
                    cfg = dict(cfg_base)
                    cfg['margin_normal'] = mg; cfg['margin_reduced'] = mg/2
                    cfg['monthly_loss_limit'] = ml; cfg['dd_threshold'] = dd
                    r = run_backtest(cache, cfg['timeframe'], cfg)
                    if r and r['trades']>=5 and r.get('liq',0)==0 and r.get('ret',0)>0:
                        r['_s']=pf_score(r); r['_cfg']=cfg; p4_results.append(r)
                    combo += 1
    print(f"  {combo:,}개 ({time.time()-t0:.0f}s), 유효 {len(p4_results):,}")

    p4_results.sort(key=lambda x:x['_s'], reverse=True)

    # === 최종 결과 ===
    print(f"\n{'='*80}")
    print("  === 최종 결과 ===")
    print(f"{'='*80}")

    # 1) 스코어 Top 10
    print(f"\n  [스코어 Top 10]")
    for i,r in enumerate(p4_results[:10]):
        print(f"  {i+1} ${r['bal']:>12,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} WR:{r['wr']:.1f}% {r['cfg']}")

    # 2) PF Top 10 (30+ trades)
    pf30 = sorted([r for r in p4_results if r['trades']>=30], key=lambda x:x['pf'], reverse=True)
    print(f"\n  [PF Top 10, 거래30+, FL=0]")
    for i,r in enumerate(pf30[:10]):
        print(f"  {i+1} ${r['bal']:>12,.0f} PF:{r['pf']:>6.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} WR:{r['wr']:.1f}% {r['cfg']}")

    # 3) MDD Top 10 (PF>=3, 30+ trades)
    mdd30 = sorted([r for r in p4_results if r['pf']>=3 and r['trades']>=30], key=lambda x:x['mdd'])
    print(f"\n  [MDD Top 10, PF>=3, 거래30+]")
    for i,r in enumerate(mdd30[:10]):
        print(f"  {i+1} ${r['bal']:>12,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} WR:{r['wr']:.1f}% {r['cfg']}")

    # 4) 수익 Top 10 (PF>=2, 30+ trades)
    ret30 = sorted([r for r in p4_results if r['pf']>=2 and r['trades']>=30], key=lambda x:x['bal'], reverse=True)
    print(f"\n  [수익 Top 10, PF>=2, 거래30+]")
    for i,r in enumerate(ret30[:10]):
        print(f"  {i+1} ${r['bal']:>12,.0f} PF:{r['pf']:>5.2f} MDD:{r['mdd']:>4.1f}% TR:{r['trades']} WR:{r['wr']:.1f}% {r['cfg']}")

    # 30회 검증 (Top 3)
    print(f"\n{'='*80}")
    print("  30회 반복 검증")
    print(f"{'='*80}")
    for i, br in enumerate(p4_results[:3]):
        cfg = br['_cfg']
        bals = [run_backtest(cache, cfg['timeframe'], cfg)['bal'] for _ in range(30)]
        print(f"  #{i+1}: ${np.mean(bals):>12,.0f} std={np.std(bals):.4f} {'PASS' if np.std(bals)<0.01 else 'FAIL'}")
        print(f"       {br['cfg']}")

    # 상세 (Top 1)
    if p4_results:
        best = p4_results[0]
        print(f"\n{'='*80}")
        print(f"  Top 1 상세")
        print(f"{'='*80}")
        r = best
        print(f"  잔액: ${r['bal']:,.0f} ({r['ret']:+,.1f}%)")
        print(f"  PF: {r['pf']} | MDD: {r['mdd']}% | FL: {r['liq']}")
        print(f"  거래: {r['trades']} (SL:{r['sl']} TSL:{r['tsl']} REV:{r['sig']})")
        print(f"  승률: {r['wr']}%")
        print(f"  평균승: {r['avg_win']:.2f}% | 평균패: {r['avg_loss']:.2f}%")
        if r['avg_loss']!=0: print(f"  손익비: {abs(r['avg_win']/r['avg_loss']):.2f}:1")
        print(f"  연도별:")
        for y in sorted(r.get('yr',{}).keys()):
            print(f"    {y}: {r['yr'][y]:>+.1f}%")

    # 저장
    save = {'total_combos': 300000+combo}
    for k, lst in [('score_top', p4_results[:20]), ('pf30_top', pf30[:10]),
                    ('mdd_pf3_30', mdd30[:10]), ('ret_pf2_30', ret30[:10])]:
        save[k] = [{'bal':r['bal'],'ret':r['ret'],'pf':r['pf'],'mdd':r['mdd'],
                     'fl':r['liq'],'tr':r['trades'],'wr':r['wr'],'cfg':r['cfg'],
                     'yr':r.get('yr',{}),'sl':r.get('sl',0),'tsl':r.get('tsl',0),
                     'rev':r.get('sig',0),'avg_win':r.get('avg_win',0),'avg_loss':r.get('avg_loss',0)}
                    for r in lst]
    with open(f'{BASE}/v16_final_results.json','w') as f:
        json.dump(save, f, indent=2, ensure_ascii=False)
    print(f"\n저장: v16_final_results.json")
    print("완료!")

if __name__=='__main__':
    main()
