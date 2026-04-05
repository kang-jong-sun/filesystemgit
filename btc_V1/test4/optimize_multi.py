"""
v16.0 멀티코인 공통 전략 최적화
BTC + ETH + SOL 3코인 모두 수익나는 범용 전략 탐색
500,000+ 조합
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
    df = df.rename(columns={'timestamp': 'time'})
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    print(f"  {prefix}: {len(df):,}행 ({df['time'].iloc[0]} ~ {df['time'].iloc[-1]})")
    return df

def multi_score(results_3):
    """3코인 공통 스코어: 3개 모두 양수 수익 필수"""
    if any(r is None or r.get('ret',0) <= 0 or r.get('liq',0) > 0 for r in results_3):
        return 0
    pfs = [r['pf'] for r in results_3]
    mdds = [r['mdd'] for r in results_3]
    rets = [r['ret'] for r in results_3]
    trs = [r['trades'] for r in results_3]

    if any(t < 8 for t in trs): return 0  # 최소 거래수

    # 최소 PF 기준 (가장 약한 코인)
    min_pf = min(pfs)
    avg_pf = np.mean(pfs)
    max_mdd = max(mdds)
    avg_ret = np.mean(rets)
    min_tr = min(trs)

    pf_b = min_pf ** 1.2 * avg_pf ** 0.3  # 최소 PF 가중
    mdd_b = (100 - max_mdd) / 50
    ret_b = np.log1p(avg_ret / 100)
    tr_b = min(min_tr / 30, 2.0)

    return pf_b * mdd_b * ret_b * tr_b


def main():
    print("=" * 80)
    print("  v16.0 멀티코인 공통 전략 최적화")
    print("  BTC + ETH + SOL 3코인 범용 전략")
    print("=" * 80)

    # 데이터 로드
    print("\n[1] 데이터 로드")
    coins = {}
    for prefix, name in [('btc_usdt', 'BTC'), ('eth_usdt', 'ETH'), ('sol_usdt', 'SOL')]:
        df = load_coin(prefix)
        mtf = build_mtf(df)
        cache = IndicatorCache(mtf)
        coins[name] = cache
        print(f"    {name} MTF 생성 완료")

    # Phase 1: 대탐색
    print(f"\n{'='*80}")
    print("  Phase 1: 3코인 공통 진입 탐색 (200,000 샘플)")
    print(f"{'='*80}")

    tfs = ['5m','10m','15m','30m','1h']
    ma_f_types = ['ema','hma','dema','wma']
    ma_s_types = ['ema','hma','wma','sma']
    f_lens = [3,5,7,10,14,21]
    s_lens = [50,100,150,200,250]
    adx_ps = [14,20]
    adx_ms = [25,30,35,40]
    rsi_rs = [(25,60),(30,58),(30,65),(35,65),(35,70),(40,75)]
    sls = [0.04,0.05,0.06,0.07,0.08]
    tas = [0.04,0.05,0.06,0.07,0.08,0.10]
    tps = [0.03,0.04,0.05]

    configs = []
    for tf in tfs:
        for mft in ma_f_types:
            for mst in ma_s_types:
                for fl in f_lens:
                    for sl_len in s_lens:
                        if fl >= sl_len: continue
                        for ap in adx_ps:
                            for am in adx_ms:
                                for rlo,rhi in rsi_rs:
                                    for slp in sls:
                                        for ta in tas:
                                            for tp in tps:
                                                if tp >= ta: continue
                                                configs.append({
                                                    'timeframe':tf,
                                                    'ma_fast_type':mft,'ma_slow_type':mst,
                                                    'ma_fast':fl,'ma_slow':sl_len,
                                                    'adx_period':ap,'adx_min':am,
                                                    'rsi_period':14,'rsi_min':rlo,'rsi_max':rhi,
                                                    'sl_pct':slp,'trail_activate':ta,'trail_pct':tp,
                                                    'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
                                                    'fee_rate':0.0004,'initial_capital':3000.0,
                                                })

    total = len(configs)
    N = min(total, 200000)
    np.random.seed(42)
    if total > N:
        idx = np.random.choice(total, N, replace=False)
        configs = [configs[i] for i in idx]
    print(f"  전체 공간: {total:,} -> 샘플: {N:,}")

    results = []
    t0 = time.time()
    for i, cfg in enumerate(configs):
        tf = cfg['timeframe']
        rs = []
        ok = True
        for name, cache in coins.items():
            r = run_backtest(cache, tf, cfg)
            if r is None or r.get('liq',0) > 0 or r.get('ret',0) <= 0 or r['trades'] < 8:
                ok = False; break
            rs.append(r)
        if ok and len(rs) == 3:
            sc = multi_score(rs)
            if sc > 0:
                combined = {
                    'cfg': rs[0]['cfg'],
                    'BTC': {'bal':rs[0]['bal'],'ret':rs[0]['ret'],'pf':rs[0]['pf'],'mdd':rs[0]['mdd'],'tr':rs[0]['trades'],'wr':rs[0]['wr'],'fl':rs[0]['liq'],'yr':rs[0].get('yr',{})},
                    'ETH': {'bal':rs[1]['bal'],'ret':rs[1]['ret'],'pf':rs[1]['pf'],'mdd':rs[1]['mdd'],'tr':rs[1]['trades'],'wr':rs[1]['wr'],'fl':rs[1]['liq'],'yr':rs[1].get('yr',{})},
                    'SOL': {'bal':rs[2]['bal'],'ret':rs[2]['ret'],'pf':rs[2]['pf'],'mdd':rs[2]['mdd'],'tr':rs[2]['trades'],'wr':rs[2]['wr'],'fl':rs[2]['liq'],'yr':rs[2].get('yr',{})},
                    '_s': sc,
                    'min_pf': min(r['pf'] for r in rs),
                    'avg_pf': np.mean([r['pf'] for r in rs]),
                    'max_mdd': max(r['mdd'] for r in rs),
                    'avg_ret': np.mean([r['ret'] for r in rs]),
                    'sum_bal': sum(r['bal'] for r in rs),
                }
                results.append(combined)

        if (i+1) % 50000 == 0:
            el = time.time()-t0
            print(f"    {i+1:,}/{N:,} ({el:.0f}s) 3코인공통: {len(results):,}")

    print(f"  완료: {len(results):,}개 3코인 공통 ({time.time()-t0:.0f}s)")

    results.sort(key=lambda x: x['_s'], reverse=True)

    # Top 20 출력
    print(f"\n  === 3코인 공통 스코어 Top 20 ===")
    print(f"  {'#':>3} {'합계잔액':>12} {'최소PF':>6} {'평균PF':>6} {'최대MDD':>6} {'BTC':>8} {'ETH':>8} {'SOL':>8} 설정")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:>3} ${r['sum_bal']:>10,.0f} {r['min_pf']:>5.2f} {r['avg_pf']:>5.2f} {r['max_mdd']:>5.1f}% ${r['BTC']['bal']:>6,.0f} ${r['ETH']['bal']:>6,.0f} ${r['SOL']['bal']:>6,.0f} {r['cfg']}")

    # PF Top (최소PF 기준, 거래 30+)
    pf30 = [r for r in results if all(r[c]['tr']>=30 for c in ['BTC','ETH','SOL'])]
    pf30.sort(key=lambda x: x['min_pf'], reverse=True)
    print(f"\n  === 최소PF Top 10 (3코인 모두 30+거래) ===")
    for i, r in enumerate(pf30[:10]):
        print(f"  {i+1:>3} minPF:{r['min_pf']:>5.2f} avgPF:{r['avg_pf']:>5.2f} maxMDD:{r['max_mdd']:>4.1f}% B${r['BTC']['bal']:>6,.0f} E${r['ETH']['bal']:>6,.0f} S${r['SOL']['bal']:>6,.0f} {r['cfg']}")

    # Phase 2: 사이징 최적화 (Top 20에 대해)
    print(f"\n{'='*80}")
    print("  Phase 2: 사이징 최적화")
    print(f"{'='*80}")

    mgs = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    p2_results = []
    combo = 0
    t0 = time.time()

    for br in results[:20]:
        cfg_str = br['cfg']
        # cfg 파싱
        p = [x.strip() for x in cfg_str.split('|')]
        tf=p[0]; ma=p[1]; mt=ma.split('(')[0]; lens=ma.split('(')[1].rstrip(')').split('/')
        mf=int(lens[0]); ms=int(lens[1])
        ax=p[2]; ap=int(ax.split('>=')[0].replace('A','')); am=float(ax.split('>=')[1])
        rv=p[3].split(':')[1].split('-'); rmin=float(rv[0]); rmax=float(rv[1])
        slp=float(p[4].replace('SL','')); tv=p[5].replace('TA','').split('/'); ta=float(tv[0]); tp=float(tv[1])

        for mg in mgs:
            cfg = {
                'timeframe':tf,'ma_fast_type':mt,'ma_slow_type':'ema',
                'ma_fast':mf,'ma_slow':ms,'adx_period':ap,'adx_min':am,
                'rsi_period':14,'rsi_min':rmin,'rsi_max':rmax,
                'sl_pct':slp,'trail_activate':ta,'trail_pct':tp,
                'leverage':10,'margin_normal':mg,'margin_reduced':mg/2,
                'fee_rate':0.0004,'initial_capital':3000.0,
            }
            rs = []
            ok = True
            for name, cache in coins.items():
                r = run_backtest(cache, tf, cfg)
                if r is None or r.get('liq',0)>0 or r.get('ret',0)<=0:
                    ok = False; break
                rs.append(r)
            if ok and len(rs)==3:
                sc = multi_score(rs)
                if sc > 0:
                    p2_results.append({
                        'cfg': rs[0]['cfg'],
                        'BTC': {'bal':rs[0]['bal'],'ret':rs[0]['ret'],'pf':rs[0]['pf'],'mdd':rs[0]['mdd'],'tr':rs[0]['trades'],'wr':rs[0]['wr'],'fl':rs[0]['liq'],'yr':rs[0].get('yr',{}),'sl':rs[0].get('sl',0),'tsl':rs[0].get('tsl',0),'rev':rs[0].get('sig',0)},
                        'ETH': {'bal':rs[1]['bal'],'ret':rs[1]['ret'],'pf':rs[1]['pf'],'mdd':rs[1]['mdd'],'tr':rs[1]['trades'],'wr':rs[1]['wr'],'fl':rs[1]['liq'],'yr':rs[1].get('yr',{}),'sl':rs[1].get('sl',0),'tsl':rs[1].get('tsl',0),'rev':rs[1].get('sig',0)},
                        'SOL': {'bal':rs[2]['bal'],'ret':rs[2]['ret'],'pf':rs[2]['pf'],'mdd':rs[2]['mdd'],'tr':rs[2]['trades'],'wr':rs[2]['wr'],'fl':rs[2]['liq'],'yr':rs[2].get('yr',{}),'sl':rs[2].get('sl',0),'tsl':rs[2].get('tsl',0),'rev':rs[2].get('sig',0)},
                        '_s': sc,
                        'min_pf': min(r['pf'] for r in rs),
                        'avg_pf': np.mean([r['pf'] for r in rs]),
                        'max_mdd': max(r['mdd'] for r in rs),
                        'sum_bal': sum(r['bal'] for r in rs),
                    })
            combo += 1

    print(f"  {combo}개 조합, {len(p2_results)}개 유효 ({time.time()-t0:.0f}s)")
    p2_results.sort(key=lambda x: x['_s'], reverse=True)

    print(f"\n  === 최종 Top 20 ===")
    print(f"  {'#':>3} {'합계':>12} {'minPF':>6} {'avgPF':>6} {'maxMDD':>6} {'BTC$':>10} {'ETH$':>10} {'SOL$':>10} 설정")
    for i, r in enumerate(p2_results[:20]):
        print(f"  {i+1:>3} ${r['sum_bal']:>10,.0f} {r['min_pf']:>5.2f} {r['avg_pf']:>5.2f} {r['max_mdd']:>5.1f}% ${r['BTC']['bal']:>8,.0f} ${r['ETH']['bal']:>8,.0f} ${r['SOL']['bal']:>8,.0f} {r['cfg']}")

    # 최종 Top 1 상세
    if p2_results:
        best = p2_results[0]
        print(f"\n{'='*80}")
        print(f"  Top 1 상세")
        print(f"{'='*80}")
        for coin in ['BTC','ETH','SOL']:
            c = best[coin]
            print(f"\n  [{coin}]")
            print(f"    잔액: ${c['bal']:,.0f} ({c['ret']:+,.1f}%)")
            print(f"    PF: {c['pf']} | MDD: {c['mdd']}% | FL: {c['fl']}")
            print(f"    거래: {c['tr']} | 승률: {c['wr']}%")
            if c.get('yr'):
                print(f"    연도별: {' | '.join(f'{y}:{v:+.1f}%' for y,v in sorted(c['yr'].items()))}")

    # 저장
    save = {'total_combos': N + combo}
    save['top20'] = []
    for r in p2_results[:20]:
        save['top20'].append({
            'cfg': r['cfg'], 'sum_bal': r['sum_bal'],
            'min_pf': round(r['min_pf'],2), 'avg_pf': round(r['avg_pf'],2), 'max_mdd': round(r['max_mdd'],1),
            'BTC': r['BTC'], 'ETH': r['ETH'], 'SOL': r['SOL'],
        })
    with open(f'{BASE}/v16_multi_results.json', 'w') as f:
        json.dump(save, f, indent=2, ensure_ascii=False)
    print(f"\n저장: v16_multi_results.json")
    print(f"총 조합: {N + combo:,}")
    print("완료!")


if __name__ == '__main__':
    main()
