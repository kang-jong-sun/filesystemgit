"""
AI 자유재량 v2 - 지표 추이 50캔들 제공
2023~2026 | ADX>=20 | GPT-5.4 + Claude Sonnet 4.6
"""
import pandas as pd, numpy as np, json, os, sys, time
from datetime import datetime
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
np.seterr(all='ignore')
load_dotenv("D:/filesystem/futures/btc_V1/.env"); load_dotenv()

IC=3000.0; FEE=0.0004; LEV=10; MPCT=0.25
SL_PCT=7.0; TSL_ACT=6.0; TSL_W=3.0; ML=-0.20
WARMUP=300; FL_PCT=10.0; AI_ADX=20
BASE="D:/filesystem/futures/btc_V1/test2"

# ═══ 지표 ═══
def ema(s,p): return s.ewm(span=p,adjust=False).mean()
def rsi_calc(s,p=14):
    d=s.diff();g=d.where(d>0,0.0);l=(-d).where(d<0,0.0)
    return 100-100/(1+g.ewm(alpha=1/p,min_periods=p,adjust=False).mean()/l.ewm(alpha=1/p,min_periods=p,adjust=False).mean().replace(0,1e-10))
def adx_calc(h,l,c,p=14):
    pdm=h.diff();mdm=-l.diff();pdm=pdm.where((pdm>mdm)&(pdm>0),0.0);mdm=mdm.where((mdm>pdm)&(mdm>0),0.0)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/p,min_periods=p,adjust=False).mean()
    pdi=100*(pdm.ewm(alpha=1/p,min_periods=p,adjust=False).mean()/atr)
    mdi=100*(mdm.ewm(alpha=1/p,min_periods=p,adjust=False).mean()/atr)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,1e-10)
    return dx.ewm(alpha=1/p,min_periods=p,adjust=False).mean()
def macd_calc(c):
    m=ema(c,12)-ema(c,26); s=m.ewm(span=9,adjust=False).mean(); return m,s,m-s

# ═══ AI ═══
def init_ai():
    g=c=None
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI; g=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic; c=anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return g,c

SYS_PROMPT="""당신은 15년 경력의 BTC/USDT 선물 퀀트 트레이더입니다.

[역할]
EMA(3/200) 크로스 발생 시 50캔들 분량의 지표 추이 데이터를 분석하여 진입 여부를 판단합니다.
고정 규칙 없이 데이터 기반으로 독자 판단하세요.

[핵심 분석 포인트]
1. ADX 추이: 상승 중이면 추세 강화, 하락 중이면 추세 약화
2. RSI 추이: 방향성과 과매수/과매도 탈출 여부
3. MACD 히스토그램 추이: 양/음 전환, 크기 변화로 모멘텀 판단
4. EMA 배열: 정배열(3>21>50>200) vs 역배열, 수렴/확산
5. 가격과 EMA 간격: 크로스 직후 추세의 신뢰도

[판단 기준]
- 지표 추이가 크로스 방향을 지지하면 진입
- 추이가 엇갈리거나 추세가 약화 중이면 HOLD
- 이미 같은 방향 포지션이면 재진입 불필요
- 의심스러우면 HOLD

[출력]
반드시 JSON만. {"action":"LONG/SHORT/HOLD","confidence":1~10,"reason":"한 줄"}"""

def ask_gpt(cli, prompt, retries=5):
    for a in range(retries):
        try:
            r=cli.chat.completions.create(model="gpt-5.4",temperature=0,
                response_format={"type":"json_object"},
                messages=[{"role":"system","content":SYS_PROMPT},{"role":"user","content":prompt}])
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            w=min(30,5*(a+1)); print(f"    [GPT retry {a+1}] {type(e).__name__} {w}s")
            if a<retries-1: time.sleep(w)
    return None

def ask_claude(cli, prompt, retries=5):
    for a in range(retries):
        try:
            r=cli.messages.create(model="claude-sonnet-4-6",max_tokens=300,temperature=0,
                system=SYS_PROMPT,messages=[{"role":"user","content":prompt}])
            t=r.content[0].text; s=t.find('{'); e=t.rfind('}')+1
            return json.loads(t[s:e])
        except Exception as e:
            w=min(60,10*(a+1)); print(f"    [Claude retry {a+1}] {type(e).__name__} {w}s")
            if a<retries-1: time.sleep(w)
    return None

def fmt_series(vals, n=50):
    """최근 n개 값을 간결하게 포맷"""
    v = vals[-n:]
    # 5캔들 간격으로 샘플링 + 최근 5개는 전부
    samples = list(v[::5]) + list(v[-5:])
    # 중복 제거하면서 순서 유지
    seen = set(); result = []
    for x in samples:
        if id(x) not in seen: seen.add(id(x)); result.append(x)
    return ', '.join(f'{x:.1f}' for x in v[-50:])

def build_prompt(ts, cross, candle, i, df30, ema3_s, ema21_s, ema50_s, ema200_s, adx_s, rsi_s, mhist_s, bal, pos_str, mpnl):
    s = max(0, i-49)
    # 50캔들 지표 추이 (10개 간격 요약 + 최근 5개)
    def trend_str(arr, fmt='.1f'):
        v = arr[s:i+1]
        if len(v) <= 10:
            return ', '.join(f'{x:{fmt}}' for x in v)
        # -50, -40, -30, -20, -10, -5, -4, -3, -2, -1, 현재
        picks = []
        for idx in [0, len(v)//5, 2*len(v)//5, 3*len(v)//5, 4*len(v)//5]:
            if idx < len(v): picks.append(v[idx])
        picks.extend(v[-5:])
        return ' → '.join(f'{x:{fmt}}' for x in picks)

    def trend_detail(arr, fmt='.1f'):
        v = arr[s:i+1]
        lines = []
        for j in range(0, len(v), 5):
            chunk = v[j:j+5]
            line = ', '.join(f'{x:{fmt}}' for x in chunk)
            lines.append(line)
        return '\n    '.join(lines)

    adx_v = adx_s.values; rsi_v = rsi_s.values; mh_v = mhist_s.values
    e3_v = ema3_s.values; e21_v = ema21_s.values; e50_v = ema50_s.values; e200_v = ema200_s.values

    # 현재값
    adx_now = adx_v[i]; rsi_now = rsi_v[i]; mh_now = mh_v[i]
    e3_now = e3_v[i]; e200_now = e200_v[i]

    # 추이 방향 자동 계산
    def direction(arr):
        v = arr[s:i+1]
        if len(v) < 10: return "데이터부족"
        recent = np.mean(v[-5:]); prev = np.mean(v[-15:-10])
        diff = recent - prev
        if abs(diff) < 0.5: return "횡보"
        return "상승중" if diff > 0 else "하락중"

    adx_dir = direction(adx_v); rsi_dir = direction(rsi_v); mh_dir = direction(mh_v)

    # 최근 10캔들 OHLCV
    r10 = df30.iloc[max(0,i-9):i+1]
    r10_str = r10[['open','high','low','close','volume']].to_string()

    return f"""시각: {ts}
BTC/USDT 30분봉 **{cross}** 발생

=== 현재 ===
가격: ${candle['close']:,.0f} | EMA3: {e3_now:.0f} | EMA200: {e200_now:.0f}
ADX: {adx_now:.1f} ({adx_dir}) | RSI: {rsi_now:.1f} ({rsi_dir}) | MACD Hist: {mh_now:.0f} ({mh_dir})

=== ADX 추이 (최근 50캔들, 5개씩) ===
    {trend_detail(adx_v)}

=== RSI 추이 (최근 50캔들, 5개씩) ===
    {trend_detail(rsi_v)}

=== MACD 히스토그램 추이 (최근 50캔들, 5개씩) ===
    {trend_detail(mh_v, fmt='.0f')}

=== EMA 배열 추이 (최근 50캔들: EMA3, EMA21, EMA50, EMA200) ===
    현재: EMA3={e3_now:.0f} EMA21={e21_v[i]:.0f} EMA50={e50_v[i]:.0f} EMA200={e200_now:.0f}
    10캔들전: EMA3={e3_v[max(0,i-10)]:.0f} EMA21={e21_v[max(0,i-10)]:.0f} EMA50={e50_v[max(0,i-10)]:.0f} EMA200={e200_v[max(0,i-10)]:.0f}
    30캔들전: EMA3={e3_v[max(0,i-30)]:.0f} EMA21={e21_v[max(0,i-30)]:.0f} EMA50={e50_v[max(0,i-30)]:.0f} EMA200={e200_v[max(0,i-30)]:.0f}
    50캔들전: EMA3={e3_v[max(0,i-49)]:.0f} EMA21={e21_v[max(0,i-49)]:.0f} EMA50={e50_v[max(0,i-49)]:.0f} EMA200={e200_v[max(0,i-49)]:.0f}

=== 최근 10캔들 OHLCV ===
{r10_str}

=== 계좌 ===
잔액: ${bal:,.0f} | 포지션: {pos_str} | 이번달: {mpnl:+.1f}%

{{"action":"LONG/SHORT/HOLD","confidence":1~10,"reason":"한 줄"}}"""

# ═══ Bot ═══
class Bot:
    def __init__(s,name):
        s.name=name;s.bal=IC;s.pos=0;s.epx=0;s.psz=0;s.margin=0
        s.slp=0;s.tsl_on=False;s.peak=0;s.trough=999999
        s.m_start=IC;s.cur_m="";s.sl_c=0;s.tsl_c=0;s.rev_c=0;s.fl_c=0
        s.trades=[];s.monthly={};s.pk_bal=IC;s.mdd=0;s.yr_bal={}
    def mpnl(s): return (s.bal-s.m_start)/s.m_start*100 if s.m_start>0 else 0
    def update_m(s,mk):
        if mk!=s.cur_m:
            if s.cur_m and s.cur_m in s.monthly: s.monthly[s.cur_m]['eq']=s.bal
            s.cur_m=mk;s.m_start=s.bal
            if mk not in s.monthly: s.monthly[mk]={'pnl':0,'ent':0,'w':0,'l':0,'sl':0,'tsl':0,'rev':0,'eq_s':s.bal,'eq':s.bal}
    def update_dd(s):
        s.pk_bal=max(s.pk_bal,s.bal)
        if s.pk_bal>0: s.mdd=max(s.mdd,(s.pk_bal-s.bal)/s.pk_bal)
    def enter(s,d,px,ts,chk_ml=True):
        if s.bal<=0 or s.bal*MPCT<1: return False
        if chk_ml and s.m_start>0 and (s.bal-s.m_start)/s.m_start<=ML: return False
        s.pos=d;s.epx=px;s.margin=s.bal*MPCT;s.psz=s.margin*LEV
        s.bal-=s.psz*FEE;s.tsl_on=False;s.peak=px;s.trough=px
        s.slp=px*(1-SL_PCT/100) if d==1 else px*(1+SL_PCT/100)
        if s.cur_m in s.monthly: s.monthly[s.cur_m]['ent']+=1
        return True
    def close(s,px,reason,ts):
        if s.pos==0: return 0
        if reason=='fl': pnl=-(s.margin+s.psz*FEE)
        else: pnl=(px-s.epx)/s.epx*s.psz*s.pos-s.psz*FEE
        s.bal+=pnl
        if s.bal<0: s.bal=0
        mk=s.cur_m
        if mk in s.monthly:
            s.monthly[mk]['pnl']+=pnl
            if pnl>0: s.monthly[mk]['w']+=1
            else: s.monthly[mk]['l']+=1
            if reason in('sl','tsl','rev'): s.monthly[mk][reason]+=1
        if reason=='sl':s.sl_c+=1
        elif reason=='tsl':s.tsl_c+=1
        elif reason=='rev':s.rev_c+=1
        elif reason=='fl':s.fl_c+=1
        s.trades.append({'ts':str(ts),'dir':'L' if s.pos==1 else 'S','entry':s.epx,'exit':px,'pnl':round(pnl,2),'reason':reason,'bal':round(s.bal,2)})
        s.pos=0;s.epx=0;s.psz=0;s.margin=0;s.update_dd()
        return pnl
    def check_exit(s,hi,lo,cl):
        if s.pos==0: return None
        w=(lo-s.epx)/s.epx*100 if s.pos==1 else (s.epx-hi)/s.epx*100
        if w<=-FL_PCT: return('fl',None)
        if not s.tsl_on:
            if s.pos==1 and lo<=s.slp: return('sl',s.slp)
            if s.pos==-1 and hi>=s.slp: return('sl',s.slp)
        b=(hi-s.epx)/s.epx*100 if s.pos==1 else (s.epx-lo)/s.epx*100
        if b>=TSL_ACT: s.tsl_on=True
        if s.tsl_on:
            if s.pos==1:
                s.peak=max(s.peak,hi);tl=s.peak*(1-TSL_W/100);s.slp=max(s.slp,tl)
                if cl<=tl: return('tsl',cl)
            else:
                s.trough=min(s.trough,lo);tl=s.trough*(1+TSL_W/100);s.slp=min(s.slp,tl)
                if cl>=tl: return('tsl',cl)
        return None
    def pos_str(s):
        if s.pos==0: return "없음"
        return f"{'LONG' if s.pos==1 else 'SHORT'} @ ${s.epx:,.0f}"
    def stats(s):
        n=len(s.trades);w=sum(1 for t in s.trades if t['pnl']>0)
        gp=sum(t['pnl'] for t in s.trades if t['pnl']>0)
        gl=abs(sum(t['pnl'] for t in s.trades if t['pnl']<=0))
        return{'bal':s.bal,'ret':(s.bal-IC)/IC*100,'pf':gp/max(gl,.001),'mdd':s.mdd*100,
               'n':n,'wr':w/max(n,1)*100,'sl':s.sl_c,'tsl':s.tsl_c,'rev':s.rev_c,'fl':s.fl_c}

# ═══ Main ═══
def run():
    print("="*70)
    print("  AI 자유재량 v2 - 50캔들 지표 추이 제공")
    print(f"  2023~2026 | ADX>=20 | {datetime.now()}")
    print("="*70)

    print("\n[1/4] 데이터...")
    dfs=[pd.read_csv(f"{BASE}/btc_usdt_5m_2020_to_now_part{i}.csv",parse_dates=['timestamp']) for i in[1,2,3]]
    df=pd.concat(dfs).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp',inplace=True)
    df=df[['open','high','low','close','volume']].astype(float)
    df30=df.resample('30min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    print(f"  30m: {len(df30):,}")

    print("[2/4] 지표...")
    c=df30['close'];h=df30['high'];l=df30['low']
    ema3_s=ema(c,3);ema21_s=ema(c,21);ema50_s=ema(c,50);ema200_s=ema(c,200)
    adx_s=adx_calc(h,l,c,14);rsi_s=rsi_calc(c,14)
    _,_,mhist_s=macd_calc(c)

    bull=ema3_s>ema200_s;prev=bull.shift(1).fillna(False)
    golden=(bull&~prev).values;dead=(~bull&prev).values
    v14f=((adx_s>=35)&(rsi_s>=30)&(rsi_s<=65)).values
    ts_arr=df30.index;cl=c.values;hi=h.values;lo=l.values;n=len(df30)

    start_i=max(WARMUP,np.searchsorted(ts_arr,pd.Timestamp('2023-01-01')))
    total=n-start_i
    print(f"  범위: {ts_arr[start_i]} ~ {ts_arr[-1]} ({total:,} 캔들)")

    print("[3/4] AI...")
    gpt_cli,cld_cli=init_ai()
    print(f"  GPT-5.4: {'OK' if gpt_cli else 'X'} | Claude: {'OK' if cld_cli else 'X'}")

    code=Bot("코드봇");gpt=Bot("GPT-5.4");cla=Bot("Claude")
    bots=[code,gpt,cla]
    cross_log=[];errs={'gpt':0,'claude':0}
    cx=0;api=0;proc=0

    print(f"\n[4/4] 실행...\n")

    for i in range(start_i,n):
        t=ts_arr[i];h_=hi[i];l_=lo[i];c_=cl[i]
        mk=f"{t.year}-{t.month:02d}"
        for b in bots: b.update_m(mk)

        for b in bots:
            ex=b.check_exit(h_,l_,c_)
            if ex:
                r,px=ex
                if r=='fl': b.close(c_,'fl',t)
                elif r=='sl': b.close(px,'sl',t)
                elif r=='tsl': b.close(px,'tsl',t)

        is_gx=bool(golden[i]);is_dx=bool(dead[i])
        if not(is_gx or is_dx):
            proc+=1
            if proc>0 and proc%2000==0:
                print(f"[{proc/total*100:4.0f}%] {t.strftime('%Y-%m')} | 코드:${code.bal:,.0f} GPT:${gpt.bal:,.0f} Claude:${cla.bal:,.0f} | cx:{cx}")
            for b in bots: b.yr_bal[t.year]=b.bal
            continue

        cx+=1
        cross="골든크로스" if is_gx else "데드크로스"
        nd=1 if is_gx else -1
        av=adx_s.values[i];rv=rsi_s.values[i]
        filt=bool(v14f[i])

        # 코드봇
        if code.pos!=0 and code.pos!=nd and filt: code.close(c_,'rev',t)
        ca="HOLD"
        if filt and code.pos!=nd:
            if code.enter(nd,c_,t): ca="LONG" if nd==1 else "SHORT"

        # AI
        ga=gr=cla_a=cla_r="";gc=cc=0

        if av>=AI_ADX:
            candle={'open':df30['open'].values[i],'high':h_,'low':l_,'close':c_,'volume':df30['volume'].values[i]}
            prompt=build_prompt(str(t),cross,candle,i,df30,ema3_s,ema21_s,ema50_s,ema200_s,adx_s,rsi_s,mhist_s,
                                gpt.bal,gpt.pos_str(),gpt.mpnl())
            if gpt_cli:
                res=ask_gpt(gpt_cli,prompt);api+=1
                if res: ga=res.get('action','HOLD');gr=res.get('reason','');gc=res.get('confidence',0)
                else: errs['gpt']+=1;ga="HOLD";gr="API오류"
            else: ga="HOLD";gr="API없음"

            prompt_c=build_prompt(str(t),cross,candle,i,df30,ema3_s,ema21_s,ema50_s,ema200_s,adx_s,rsi_s,mhist_s,
                                  cla.bal,cla.pos_str(),cla.mpnl())
            if cld_cli:
                res=ask_claude(cld_cli,prompt_c);api+=1
                if res: cla_a=res.get('action','HOLD');cla_r=res.get('reason','');cc=res.get('confidence',0)
                else: errs['claude']+=1;cla_a="HOLD";cla_r="API오류"
            else: cla_a="HOLD";cla_r="API없음"
        else:
            ga="HOLD";gr=f"ADX{av:.0f}<20";cla_a="HOLD";cla_r=f"ADX{av:.0f}<20"

        # AI 실행
        for bot,action in[(gpt,ga),(cla,cla_a)]:
            if action in("LONG","SHORT"):
                ad=1 if action=="LONG" else -1
                if bot.pos!=0 and bot.pos!=ad: bot.close(c_,'rev',t)
                if bot.pos==0: bot.enter(ad,c_,t)

        cross_log.append({'ts':str(t),'cross':cross,'price':c_,'adx':round(av,1),'rsi':round(rv,1),'filt':filt,
            'code':ca,'gpt':ga,'gpt_c':gc,'gpt_r':gr,'claude':cla_a,'claude_c':cc,'claude_r':cla_r,
            'code_b':round(code.bal,0),'gpt_b':round(gpt.bal,0),'cla_b':round(cla.bal,0)})

        if av>=AI_ADX:
            fm="V14" if filt else "NEW"
            print(f"\n{'━'*65}")
            print(f"[{t}] {cross} #{cx} [{fm}] ADX={av:.1f} RSI={rv:.1f} ${c_:,.0f}")
            print(f"코드:{ca:>5} | GPT:{ga:>5} c={gc} {gr[:40]} | Claude:{cla_a:>5} c={cc} {cla_r[:40]}")
            print(f"잔액 | 코드:${code.bal:,.0f} GPT:${gpt.bal:,.0f} Claude:${cla.bal:,.0f}")

        for b in bots: b.yr_bal[t.year]=b.bal
        proc+=1

    # 잔여
    for b in bots:
        if b.pos!=0: b.close(cl[-1],'end',ts_arr[-1])
        if b.cur_m in b.monthly: b.monthly[b.cur_m]['eq']=b.bal
        b.yr_bal[ts_arr[-1].year]=b.bal

    # 결과
    cs=code.stats();gs=gpt.stats();ls=cla.stats()
    print(f"\n\n{'='*70}")
    print(f"{'':>12}{'코드봇v14.4':>14}{'GPT-5.4':>14}{'Claude':>14}")
    print(f"{'='*70}")
    for k,l in[('최종잔액','bal'),('수익률','ret'),('PF','pf'),('MDD','mdd'),('거래수','n'),('SL','sl'),('TSL','tsl'),('REV','rev')]:
        if l=='bal': print(f"{k:>12} ${cs[l]:>10,.0f} ${gs[l]:>10,.0f} ${ls[l]:>10,.0f}")
        elif l in('ret','mdd'): print(f"{k:>12} {cs[l]:>+10,.1f}% {gs[l]:>+10,.1f}% {ls[l]:>+10,.1f}%")
        elif l=='pf': print(f"{k:>12} {cs[l]:>10.2f} {gs[l]:>10.2f} {ls[l]:>10.2f}")
        elif l=='n': print(f"{k:>12} {cs[l]:>10}회 {gs[l]:>10}회 {ls[l]:>10}회")
        else: print(f"{k:>12} {cs[l]:>10} {gs[l]:>10} {ls[l]:>10}")
    print(f"{'='*70}")

    tot=len(cross_log);ai_called=[x for x in cross_log if x['adx']>=AI_ADX]
    filt_pass=[x for x in cross_log if x['filt']];new_only=[x for x in ai_called if not x['filt']]
    print(f"\n크로스: 전체{tot} AI호출{len(ai_called)} V14필터{len(filt_pass)} NEW{len(new_only)}")

    gn=[x for x in new_only if x['gpt'] in('LONG','SHORT')]
    cn=[x for x in new_only if x['claude'] in('LONG','SHORT')]
    gh=[x for x in filt_pass if x['gpt']=='HOLD']
    ch=[x for x in filt_pass if x['claude']=='HOLD']
    ge=[x for x in filt_pass if x['gpt'] in('LONG','SHORT')]
    ce=[x for x in filt_pass if x['claude'] in('LONG','SHORT')]
    print(f"NEW진입: GPT {len(gn)}/{len(new_only)} Claude {len(cn)}/{len(new_only)}")
    print(f"V14진입: GPT {len(ge)}/{len(filt_pass)} Claude {len(ce)}/{len(filt_pass)}")
    print(f"V14거부: GPT {len(gh)}/{len(filt_pass)} Claude {len(ch)}/{len(filt_pass)}")

    all_yrs=sorted(set(list(code.yr_bal)+list(gpt.yr_bal)+list(cla.yr_bal)))
    print(f"\n연도별:")
    for yr in all_yrs:
        print(f"  {yr} | 코드:${code.yr_bal.get(yr,0):>10,.0f} GPT:${gpt.yr_bal.get(yr,0):>10,.0f} Claude:${cla.yr_bal.get(yr,0):>10,.0f}")

    ranking=sorted([(cs['bal'],'코드봇'),(gs['bal'],'GPT'),(ls['bal'],'Claude')],reverse=True)
    print(f"\n순위: 1등 {ranking[0][1]} ${ranking[0][0]:,.0f} | 2등 {ranking[1][1]} ${ranking[1][0]:,.0f} | 3등 {ranking[2][1]} ${ranking[2][0]:,.0f}")
    print(f"API: {api}회 | 오류: GPT {errs['gpt']} Claude {errs['claude']}")

    # CSV
    pd.DataFrame(cross_log).to_csv(f"{BASE}/ai_freehand_v2_trades.csv",index=False,encoding='utf-8-sig')
    months=sorted(set(list(code.monthly)+list(gpt.monthly)+list(cla.monthly)))
    rows=[]
    for mk in months:
        cd=code.monthly.get(mk,{});gd=gpt.monthly.get(mk,{});ld=cla.monthly.get(mk,{})
        rows.append({'month':mk,'code_bal':round(cd.get('eq',0)),'gpt_bal':round(gd.get('eq',0)),'claude_bal':round(ld.get('eq',0)),
                     'code_pnl':round(cd.get('pnl',0)),'gpt_pnl':round(gd.get('pnl',0)),'claude_pnl':round(ld.get('pnl',0))})
    pd.DataFrame(rows).to_csv(f"{BASE}/ai_freehand_v2_monthly.csv",index=False,encoding='utf-8-sig')
    print(f"저장 완료: ai_freehand_v2_trades.csv, ai_freehand_v2_monthly.csv")

if __name__=="__main__": run()
