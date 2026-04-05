"""
전체 기획서 100회 검증 + 과장/거짓 판별 + 베스트3 + 수익률 베스트3 + 폐기10 + 보고서
"""
import sys,os,time,numpy as np,json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from bt_fast import load_5m_data,build_mtf,IndicatorCache,run_backtest
DATA_DIR=os.path.dirname(os.path.abspath(__file__))

# 기획서 기재 예상값 (문서에서 추출)
CLAIMED={
    'v10.1':{'bal':6144,'pf':1.36,'mdd':98.1},
    'v11.1':{'bal':518679,'pf':19.17,'mdd':51.2},
    'v12.0':{'bal':80691,'pf':5.28,'mdd':40},
    'v12.2':{'bal':31014,'pf':6.34,'mdd':14.4},
    'v12.3':{'bal':252892,'pf':13.34,'mdd':39.9},
    'v12.5':{'bal':233881,'pf':0,'mdd':41},
    'v13.0':{'bal':254105,'pf':9.0,'mdd':68.3},
    'v13.3':{'bal':254105,'pf':9.02,'mdd':27.1},
    'v13.4A':{'bal':161046,'pf':2.54,'mdd':29.9},
    'v13.5':{'bal':468530,'pf':6.87,'mdd':74.3},
    'v14.1':{'bal':124882,'pf':7.05,'mdd':77.9},
    'v14.2':{'bal':798358,'pf':1.29,'mdd':57.8},
    'v14.3':{'bal':3210275,'pf':71.60,'mdd':73.7},
    'v14.4':{'bal':837212,'pf':2.04,'mdd':36.9},
    'v15.1':{'bal':206282,'pf':5.60,'mdd':79.9},
    'v15.2':{'bal':243482,'pf':2.48,'mdd':27.6},
    'v15.3B':{'bal':837212,'pf':2.04,'mdd':36.9},
    'v15.4':{'bal':8717659,'pf':1.65,'mdd':54.2},
    'v13.2A':{'bal':75044,'pf':3.9,'mdd':93.2},
    'v13.2B':{'bal':17091,'pf':1.4,'mdd':37.4},
    'v14.2F':{'bal':798358,'pf':1.29,'mdd':57.8},
    'v15.5':{'bal':192281,'pf':22.50,'mdd':20.1},
}

CONFIGS={
    'v10.1':{'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
             'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
             'sl_pct':0.09,'trail_activate':0.10,'trail_pct':0.05,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10},
    'v11.1':{'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
             'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
             'sl_pct':0.06,'trail_activate':0.08,'trail_pct':0.06,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10},
    'v12.0':{'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
             'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
             'sl_pct':0.06,'trail_activate':0.10,'trail_pct':0.03,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10},
    'v12.2':{'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
             'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
             'sl_pct':0.09,'trail_activate':0.08,'trail_pct':0.01,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10},
    'v12.3':{'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
             'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
             'sl_pct':0.09,'trail_activate':0.08,'trail_pct':0.06,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10},
    'v12.5':{'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
             'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
             'sl_pct':0.08,'trail_activate':0.09,'trail_pct':0.05,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10},
    'v13.0':{'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
             'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
             'sl_pct':0.09,'trail_activate':0.08,'trail_pct':0.06,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,
             'monthly_loss_limit':-0.15,'consec_loss_pause':3,'pause_candles':288,'dd_threshold':-0.30},
    'v13.3':{'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
             'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
             'sl_pct':0.12,'trail_activate':0.15,'trail_pct':0.01,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,
             'monthly_loss_limit':-0.10,'consec_loss_pause':3,'pause_candles':288,'dd_threshold':-0.30},
    'v13.4A':{'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':5,'ma_slow':100,
              'adx_period':20,'adx_min':25,'rsi_period':14,'rsi_min':30,'rsi_max':58,
              'sl_pct':0.06,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
              'monthly_loss_limit':-0.15,'dd_threshold':-0.30},
    'v13.5':{'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
             'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
             'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.06,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,
             'monthly_loss_limit':-0.20,'consec_loss_pause':3,'pause_candles':288,'dd_threshold':-0.50},
    'v14.1':{'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
             'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
             'sl_pct':0.09,'trail_activate':0.10,'trail_pct':0.06,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,
             'monthly_loss_limit':-0.25,'dd_threshold':-0.30},
    'v14.2':{'timeframe':'30m','ma_fast_type':'hma','ma_slow_type':'hma','ma_fast':7,'ma_slow':200,
             'adx_period':20,'adx_min':25,'rsi_period':14,'rsi_min':25,'rsi_max':65,
             'sl_pct':0.07,'trail_activate':0.10,'trail_pct':0.01,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,
             'monthly_loss_limit':-0.15,'dd_threshold':-0.40},
    'v14.3':{'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
             'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
             'sl_pct':0.10,'trail_activate':0.08,'trail_pct':0.06,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,
             'monthly_loss_limit':-0.25,'consec_loss_pause':3,'pause_candles':288,'dd_threshold':-0.30},
    'v14.4':{'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
             'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,
             'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
             'monthly_loss_limit':-0.20},
    'v15.1':{'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
             'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
             'sl_pct':0.07,'trail_activate':0.08,'trail_pct':0.06,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10,
             'monthly_loss_limit':-0.20,'consec_loss_pause':3,'pause_candles':288},
    'v15.2':{'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
             'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,
             'sl_pct':0.05,'trail_activate':0.06,'trail_pct':0.05,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,
             'monthly_loss_limit':-0.15},
    'v15.3B':{'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
              'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,
              'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.25,'margin_reduced':0.125,
              'monthly_loss_limit':-0.20},
    'v15.4':{'timeframe':'30m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':3,'ma_slow':200,
             'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':30,'rsi_max':65,
             'sl_pct':0.07,'trail_activate':0.06,'trail_pct':0.03,'leverage':10,'margin_normal':0.40,'margin_reduced':0.20,
             'monthly_loss_limit':-0.30},
    'v13.2A':{'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':7,'ma_slow':100,
              'adx_period':14,'adx_min':30,'rsi_period':14,'rsi_min':30,'rsi_max':58,
              'sl_pct':0.06,'trail_activate':0.10,'trail_pct':0.05,'leverage':10,'margin_normal':0.20,'margin_reduced':0.10},
    'v13.2B':{'timeframe':'5m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':30,'ma_slow':200,
              'adx_period':14,'adx_min':35,'rsi_period':14,'rsi_min':35,'rsi_max':65,
              'sl_pct':0.04,'trail_activate':0.08,'trail_pct':0.03,'leverage':10,'margin_normal':0.10,'margin_reduced':0.05},
    'v14.2F':{'timeframe':'30m','ma_fast_type':'hma','ma_slow_type':'ema','ma_fast':7,'ma_slow':200,
              'adx_period':20,'adx_min':25,'rsi_period':14,'rsi_min':25,'rsi_max':65,
              'sl_pct':0.07,'trail_activate':0.10,'trail_pct':0.01,'leverage':10,'margin_normal':0.30,'margin_reduced':0.15,
              'monthly_loss_limit':-0.15,'dd_threshold':-0.40},
    'v15.5':{'timeframe':'15m','ma_fast_type':'ema','ma_slow_type':'ema','ma_fast':21,'ma_slow':250,
             'adx_period':20,'adx_min':40,'rsi_period':14,'rsi_min':40,'rsi_max':75,
             'sl_pct':0.04,'trail_activate':0.20,'trail_pct':0.05,'leverage':15,'margin_normal':0.30,'margin_reduced':0.15},
}
DEFAULTS={'fee_rate':0.0004,'initial_capital':3000.0,'atr_period':14,
          'monthly_loss_limit':0,'consec_loss_pause':0,'pause_candles':0,'dd_threshold':0}

def main():
    t0=time.time()
    print("="*120)
    print("  BTC/USDT 전체 기획서 100회 검증 + 과장/거짓 판별 + 종합 보고서")
    print("="*120,flush=True)

    df_5m=load_5m_data(DATA_DIR); mtf=build_mtf(df_5m); cache=IndicatorCache(mtf)
    for tf in ['5m','15m','30m']:
        w={**DEFAULTS,**CONFIGS['v14.4'],'timeframe':tf}
        run_backtest(cache,tf,w)

    # 100회 검증
    print(f"\n[1/5] 100회 반복 검증\n",flush=True)
    verified={}
    for ver,cfg in CONFIGS.items():
        full={**DEFAULTS,**cfg}; tf=full['timeframe']
        r=run_backtest(cache,tf,full)
        if not r: print(f"  {ver:>8}: 결과 없음"); continue
        bals=[]
        for _ in range(100):
            rx=run_backtest(cache,tf,full)
            if rx: bals.append(rx['bal'])
        consistent=np.std(bals)==0 if bals else False
        verified[ver]={
            'bal':r['bal'],'ret':r['ret'],'pf':r['pf'],'mdd':r['mdd'],
            'trades':r['trades'],'sl':r['sl'],'tsl':r['tsl'],'rev':r['sig'],
            'fl':r['liq'],'wr':r['wr'],'yr':r.get('yr',{}),
            'consistent':consistent,'cfg':r['cfg'],
        }
        ck="PASS" if consistent else "FAIL"
        print(f"  {ver:>8} | ${r['bal']:>12,.0f} | {r['ret']:>+10,.1f}% | PF:{r['pf']:>6.2f} | "
              f"MDD:{r['mdd']:>5.1f}% | FL:{r['liq']:>2} | TR:{r['trades']:>4} | WR:{r['wr']:>5.1f}% | 100x {ck}",flush=True)

    # 과장/거짓 판별
    print(f"\n[2/5] 기획서 기재값 vs 실제값 대조 (과장/거짓 판별)\n",flush=True)
    print(f"  {'버전':>8} | {'기재 잔액':>12} | {'실제 잔액':>12} | {'차이':>8} | {'기재 PF':>8} | {'실제 PF':>8} | {'기재 MDD':>8} | {'실제 MDD':>8} | 판정")
    print(f"  {'-'*110}")
    fraud_list=[]
    for ver in verified:
        d=verified[ver]; cl=CLAIMED.get(ver,{})
        cb=cl.get('bal',0); ab=d['bal']; cpf=cl.get('pf',0); apf=d['pf']
        cmdd=cl.get('mdd',0); amdd=d['mdd']
        if cb>0:
            diff_pct=abs(ab-cb)/cb*100
        else:
            diff_pct=0
        # 판정
        if cb==0:
            verdict="기재값 없음"
        elif diff_pct<5:
            verdict="정확"
        elif diff_pct<20:
            verdict="근사"
        elif diff_pct<50:
            verdict="과장 의심"
            fraud_list.append((ver,"잔액",cb,ab,diff_pct))
        else:
            verdict="허위/다른엔진"
            fraud_list.append((ver,"잔액",cb,ab,diff_pct))

        cb_s=f"${cb:>10,.0f}" if cb>0 else "미기재"
        cpf_s=f"{cpf:>7.2f}" if cpf>0 else "미기재"
        cmdd_s=f"{cmdd:>6.1f}%" if cmdd>0 else "미기재"
        print(f"  {ver:>8} | {cb_s:>12} | ${ab:>10,.0f} | {diff_pct:>6.1f}% | {cpf_s:>8} | {apf:>7.2f} | {cmdd_s:>8} | {amdd:>6.1f}% | {verdict}")

    # 종합 스코어링
    print(f"\n[3/5] 종합 스코어 + 순위\n",flush=True)
    scored=[]
    for ver,d in verified.items():
        ret=max(d['ret'],0.01); pf=max(d['pf'],0.01); mdd=d['mdd']; fl=d['fl']; tr=d['trades']
        s_ret=np.log1p(ret/100)
        s_pf=min(pf,30)/5
        s_mdd=(100-mdd)/100
        s_fl=max(0.1,1-fl*0.03)
        s_tr=min(tr/20,2.0)
        score=s_ret*s_pf*s_mdd*s_fl*s_tr*100
        scored.append((ver,score,d))
    scored.sort(key=lambda x:x[1],reverse=True)

    print(f"  {'#':>3} {'버전':>8} | {'잔액':>12} | {'수익률':>10} | {'PF':>6} | {'MDD':>6} | {'FL':>3} | {'TR':>4} | {'WR':>5} | {'점수':>6}")
    print(f"  {'-'*90}")
    for i,(ver,sc,d) in enumerate(scored):
        print(f"  {i+1:>3} {ver:>8} | ${d['bal']:>10,.0f} | {d['ret']:>+9.1f}% | {d['pf']:>5.2f} | {d['mdd']:>5.1f}% | {d['fl']:>3} | {d['trades']:>4} | {d['wr']:>4.1f}% | {sc:>5.1f}")

    # 추천 베스트 3
    print(f"\n[4/5] 추천 BEST 3 + 수익률 BEST 3 + 폐기 10\n",flush=True)
    print(f"  === 추천 BEST 3 (종합 점수 기준) ===")
    for i,(ver,sc,d) in enumerate(scored[:3]):
        print(f"\n  [{i+1}위] {ver} (점수:{sc:.1f})")
        print(f"  ${d['bal']:,.0f} ({d['ret']:+,.1f}%) PF:{d['pf']:.2f} MDD:{d['mdd']:.1f}% FL:{d['fl']} TR:{d['trades']} WR:{d['wr']:.1f}%")
        print(f"  {d['cfg']}")
        if d.get('yr'):
            print(f"  연도별: {' | '.join(f'{k}:{v:+.1f}%' for k,v in sorted(d['yr'].items()))}")
        reasons=[]
        if d['pf']>=8: reasons.append(f"PF {d['pf']:.1f} (목표 PF>=8 달성)")
        if d['mdd']<40: reasons.append(f"MDD {d['mdd']:.1f}% (낮은 리스크)")
        if d['fl']==0: reasons.append("강제청산 0회 (FL 완전 제거)")
        if d['wr']>=50: reasons.append(f"승률 {d['wr']:.1f}% (50% 이상)")
        if d['ret']>10000: reasons.append(f"수익률 {d['ret']:+,.1f}% (10,000% 초과)")
        if d['trades']>=50: reasons.append(f"거래 {d['trades']}회 (충분한 통계 신뢰도)")
        if not reasons: reasons.append("종합 점수 기준 상위")
        print(f"  [선정 이유] {' / '.join(reasons)}")

    # 수익률 베스트 3
    by_bal=sorted(scored,key=lambda x:x[2]['bal'],reverse=True)
    print(f"\n  === 수익률 BEST 3 (절대 수익 기준) ===")
    for i,(ver,sc,d) in enumerate(by_bal[:3]):
        print(f"\n  [{i+1}위] {ver}")
        print(f"  ${d['bal']:,.0f} ({d['ret']:+,.1f}%) PF:{d['pf']:.2f} MDD:{d['mdd']:.1f}% FL:{d['fl']} TR:{d['trades']}")
        print(f"  {d['cfg']}")
        reasons=[]
        if d['bal']>=1000000: reasons.append(f"잔액 ${d['bal']:,.0f} (100만불 이상)")
        elif d['bal']>=500000: reasons.append(f"잔액 ${d['bal']:,.0f} (50만불 이상)")
        else: reasons.append(f"잔액 ${d['bal']:,.0f}")
        if d['fl']==0: reasons.append("FL 0 (안전한 고수익)")
        elif d['fl']>10: reasons.append(f"FL {d['fl']} (고위험 고수익)")
        if d['mdd']>70: reasons.append(f"주의: MDD {d['mdd']:.1f}%")
        print(f"  [이유] {' / '.join(reasons)}")

    # 폐기 10
    discard=scored[-10:] if len(scored)>=10 else scored[3:]
    discard_sorted=sorted(discard,key=lambda x:x[1])
    print(f"\n  === 폐기 권고 10개 (하위 점수) ===")
    for i,(ver,sc,d) in enumerate(discard_sorted):
        rank=[j+1 for j,(v,_,_) in enumerate(scored) if v==ver][0]
        print(f"\n  [폐기 {i+1}] {ver} (점수:{sc:.1f}, 순위:{rank}/{len(scored)})")
        print(f"  ${d['bal']:,.0f} ({d['ret']:+,.1f}%) PF:{d['pf']:.2f} MDD:{d['mdd']:.1f}% FL:{d['fl']}")
        reasons=[]
        if d['mdd']>90: reasons.append(f"MDD {d['mdd']:.1f}% → 실전 파산 확정")
        elif d['mdd']>70: reasons.append(f"MDD {d['mdd']:.1f}% → 70% 초과, 실전 위험")
        if d['fl']>10: reasons.append(f"FL {d['fl']}회 → 반복 강제청산, 구조적 결함")
        elif d['fl']>0: reasons.append(f"FL {d['fl']}회 → FL 0 버전으로 대체 가능")
        if d['pf']<1.5: reasons.append(f"PF {d['pf']:.2f} → 수익/손실 비율 불량")
        if d['ret']<500: reasons.append(f"수익률 {d['ret']:+,.1f}% → 500% 미만")
        best_d=scored[0][2]
        if d['bal']<best_d['bal']*0.05: reasons.append("1위 대비 잔액 5% 미만")
        # 과장 여부
        for fv,ft,fc,fa,fp in fraud_list:
            if fv==ver: reasons.append(f"기획서 기재값과 {fp:.0f}% 차이 (과장/허위)")
        if not reasons: reasons.append("상위 버전에 완전 대체됨")
        print(f"  [폐기 사유] {' | '.join(reasons)}")

    # 과장/거짓 요약
    print(f"\n[5/5] 과장/거짓 판별 요약\n",flush=True)
    if fraud_list:
        print(f"  {'버전':>8} | {'항목':>6} | {'기재값':>12} | {'실제값':>12} | {'차이':>6} | 판정")
        print(f"  {'-'*70}")
        for ver,item,claimed,actual,diff in fraud_list:
            verdict="과장" if diff<50 else "허위/다른엔진"
            print(f"  {ver:>8} | {item:>6} | ${claimed:>10,.0f} | ${actual:>10,.0f} | {diff:>5.1f}% | {verdict}")
    else:
        print("  과장/거짓 없음")

    elapsed=time.time()-t0
    print(f"\n{'='*120}")
    print(f"  검증 완료: {len(verified)}개 버전 x 100회 = {len(verified)*100:,}회 | {elapsed:.1f}초")
    print(f"{'='*120}")

    with open(os.path.join(DATA_DIR,'full_audit_results.json'),'w',encoding='utf-8') as f:
        json.dump({'verified':{k:{kk:vv for kk,vv in v.items() if kk!='yr'} for k,v in verified.items()},
                   'ranking':[(v,s) for v,s,_ in scored],'fraud':fraud_list},f,indent=2,ensure_ascii=False,default=str)
    print(f"  저장: full_audit_results.json")

if __name__=='__main__':
    main()
