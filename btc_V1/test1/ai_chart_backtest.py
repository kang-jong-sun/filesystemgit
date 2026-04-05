"""
AI 차트 기반 백테스트 v3
트레이딩뷰 스타일 차트 이미지를 AI에게 전달
2023~2026 | ADX>=20 | GPT-5.4 + Claude Sonnet 4.6
"""
import pandas as pd, numpy as np, json, os, sys, time, base64, io
from datetime import datetime
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import mplfinance as mpf

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
np.seterr(all='ignore')
load_dotenv("D:/filesystem/futures/btc_V1/.env"); load_dotenv()

IC=3000.0; FEE=0.0004; LEV=10; MPCT=0.25
SL_PCT=7.0; TSL_ACT=6.0; TSL_W=3.0; ML_LIM=-0.20
WARMUP=300; FL_PCT=10.0; AI_ADX=20
BASE="D:/filesystem/futures/btc_V1/test2"
CHART_DIR=f"{BASE}/charts"
os.makedirs(CHART_DIR, exist_ok=True)

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
    return dx.ewm(alpha=1/p,min_periods=p,adjust=False).mean(), pdi, mdi
def macd_calc(c):
    m=ema(c,12)-ema(c,26);s=m.ewm(span=9,adjust=False).mean();return m,s,m-s
def bb_calc(c,p=20,std=2):
    mid=c.ewm(span=p,adjust=False).mean();s=c.rolling(p).std()
    return mid+std*s, mid, mid-std*s
def stoch_calc(h,l,c,k=14,d=3):
    ll=l.rolling(k).min();hh=h.rolling(k).max()
    sk=100*(c-ll)/(hh-ll).replace(0,1e-10);sd=sk.rolling(d).mean()
    return sk,sd

# ═══ 차트 생성 ═══
def make_chart(df30, i, cross_type, ema3_s, ema5_s, ema10_s, ema21_s, ema50_s, ema100_s, ema200_s,
               adx_s, pdi_s, mdi_s, rsi_s, macd_s, msig_s, mhist_s, bb_up, bb_mid, bb_lo, stk_s, std_s):
    """트레이딩뷰 스타일 6패널 차트 생성, base64 반환"""
    n_candles = 100
    s = max(0, i - n_candles + 1)
    sl = slice(s, i+1)

    chunk = df30.iloc[sl].copy()
    if len(chunk) < 10:
        return None

    fig, axes = plt.subplots(6, 1, figsize=(16, 20), height_ratios=[4, 1.5, 1.2, 1.2, 1.2, 0.8],
                              gridspec_kw={'hspace': 0.05})
    fig.patch.set_facecolor('#1a1a2e')

    ts = chunk.index
    x = np.arange(len(chunk))

    # 색상
    C_BG = '#1a1a2e'; C_GRID = '#2d2d4e'; C_TEXT = '#e0e0e0'
    C_UP = '#26a69a'; C_DN = '#ef5350'
    C_EMA3 = '#ffeb3b'; C_EMA5 = '#ff9800'; C_EMA10 = '#ff5722'
    C_EMA21 = '#2196f3'; C_EMA50 = '#9c27b0'; C_EMA100 = '#00bcd4'; C_EMA200 = '#e91e63'
    C_BB = '#424242'; C_ADX = '#ffffff'; C_PDI = '#26a69a'; C_MDI = '#ef5350'
    C_RSI = '#ffeb3b'; C_MACD = '#2196f3'; C_SIG = '#ff9800'

    for ax in axes:
        ax.set_facecolor(C_BG)
        ax.tick_params(colors=C_TEXT, labelsize=7)
        ax.grid(True, color=C_GRID, alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(C_GRID); ax.spines['left'].set_color(C_GRID)

    # ── 패널 1: 캔들 + EMA + BB ──
    ax1 = axes[0]
    o = chunk['open'].values; h = chunk['high'].values; l = chunk['low'].values; c = chunk['close'].values

    for j in range(len(chunk)):
        color = C_UP if c[j] >= o[j] else C_DN
        ax1.plot([x[j], x[j]], [l[j], h[j]], color=color, linewidth=0.7)
        body_lo = min(o[j], c[j]); body_hi = max(o[j], c[j])
        if body_hi - body_lo < 0.001: body_hi = body_lo + 1
        ax1.bar(x[j], body_hi - body_lo, bottom=body_lo, width=0.6, color=color, edgecolor=color)

    # EMA lines
    for vals, color, label, lw in [
        (ema3_s.values[sl], C_EMA3, 'EMA3', 1.0),
        (ema5_s.values[sl], C_EMA5, 'EMA5', 0.8),
        (ema10_s.values[sl], C_EMA10, 'EMA10', 0.8),
        (ema21_s.values[sl], C_EMA21, 'EMA21', 1.2),
        (ema50_s.values[sl], C_EMA50, 'EMA50', 1.2),
        (ema100_s.values[sl], C_EMA100, 'EMA100', 1.0),
        (ema200_s.values[sl], C_EMA200, 'EMA200', 1.5),
    ]:
        ax1.plot(x, vals, color=color, linewidth=lw, label=label, alpha=0.9)

    # BB
    ax1.fill_between(x, bb_up.values[sl], bb_lo.values[sl], color=C_BB, alpha=0.15)
    ax1.plot(x, bb_up.values[sl], color=C_BB, linewidth=0.5, alpha=0.4)
    ax1.plot(x, bb_lo.values[sl], color=C_BB, linewidth=0.5, alpha=0.4)

    # 크로스 마커
    marker = 'v' if '데드' in cross_type else '^'
    marker_c = C_DN if '데드' in cross_type else C_UP
    ax1.scatter(x[-1], c[-1], marker=marker, s=200, color=marker_c, zorder=10, edgecolors='white', linewidths=1.5)

    cx_en = "Golden Cross" if "골든" in cross_type else "Dead Cross"
    ax1.set_title(f"BTC/USDT 30m | {cx_en} | {chunk.index[-1].strftime('%Y-%m-%d %H:%M')} | ${c[-1]:,.0f}",
                  color=C_TEXT, fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=6, ncol=7, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)
    ax1.set_ylabel('Price', color=C_TEXT, fontsize=8)

    # ── 패널 2: Volume ──
    ax2 = axes[1]
    vol = chunk['volume'].values
    vol_colors = [C_UP if c[j] >= o[j] else C_DN for j in range(len(chunk))]
    ax2.bar(x, vol, width=0.6, color=vol_colors, alpha=0.7)
    vol_ma = pd.Series(vol).rolling(20).mean().values
    ax2.plot(x, vol_ma, color='#ffffff', linewidth=0.8, alpha=0.5, label='Vol MA20')
    ax2.set_ylabel('Volume', color=C_TEXT, fontsize=8)
    ax2.legend(loc='upper left', fontsize=6, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

    # ── 패널 3: ADX + DI ──
    ax3 = axes[2]
    ax3.plot(x, adx_s.values[sl], color=C_ADX, linewidth=1.5, label='ADX')
    ax3.plot(x, pdi_s.values[sl], color=C_PDI, linewidth=0.8, label='+DI', alpha=0.7)
    ax3.plot(x, mdi_s.values[sl], color=C_MDI, linewidth=0.8, label='-DI', alpha=0.7)
    ax3.axhline(35, color='#ff9800', linewidth=1, linestyle='--', alpha=0.6, label='ADX=35')
    ax3.axhline(20, color='#666666', linewidth=0.5, linestyle=':', alpha=0.4)
    ax3.set_ylabel('ADX/DI', color=C_TEXT, fontsize=8)
    ax3.set_ylim(0, max(60, adx_s.values[sl].max() * 1.1))
    ax3.legend(loc='upper left', fontsize=6, ncol=4, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

    # ── 패널 4: RSI ──
    ax4 = axes[3]
    ax4.plot(x, rsi_s.values[sl], color=C_RSI, linewidth=1.2, label='RSI(14)')
    ax4.axhline(65, color=C_DN, linewidth=0.8, linestyle='--', alpha=0.5, label='65')
    ax4.axhline(30, color=C_UP, linewidth=0.8, linestyle='--', alpha=0.5, label='30')
    ax4.axhline(50, color='#666666', linewidth=0.5, linestyle=':', alpha=0.3)
    ax4.fill_between(x, 30, 65, color='#333366', alpha=0.15)
    ax4.set_ylabel('RSI', color=C_TEXT, fontsize=8)
    ax4.set_ylim(0, 100)
    ax4.legend(loc='upper left', fontsize=6, ncol=3, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

    # ── 패널 5: MACD ──
    ax5 = axes[4]
    mh = mhist_s.values[sl]
    mh_colors = [C_UP if v >= 0 else C_DN for v in mh]
    ax5.bar(x, mh, width=0.6, color=mh_colors, alpha=0.6)
    ax5.plot(x, macd_s.values[sl], color=C_MACD, linewidth=1.0, label='MACD')
    ax5.plot(x, msig_s.values[sl], color=C_SIG, linewidth=1.0, label='Signal')
    ax5.axhline(0, color='#666666', linewidth=0.5)
    ax5.set_ylabel('MACD', color=C_TEXT, fontsize=8)
    ax5.legend(loc='upper left', fontsize=6, ncol=2, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

    # ── 패널 6: 스토캐스틱 ──
    ax6 = axes[5]
    ax6.plot(x, stk_s.values[sl], color='#2196f3', linewidth=1.0, label='%K')
    ax6.plot(x, std_s.values[sl], color='#ff9800', linewidth=1.0, label='%D')
    ax6.axhline(80, color=C_DN, linewidth=0.5, linestyle='--', alpha=0.4)
    ax6.axhline(20, color=C_UP, linewidth=0.5, linestyle='--', alpha=0.4)
    ax6.set_ylim(0, 100)
    ax6.set_ylabel('Stoch', color=C_TEXT, fontsize=8)
    ax6.legend(loc='upper left', fontsize=6, ncol=2, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

    # X축 라벨 (마지막 패널만)
    tick_step = max(1, len(chunk) // 10)
    tick_pos = list(range(0, len(chunk), tick_step))
    tick_labels = [chunk.index[j].strftime('%m/%d\n%H:%M') for j in tick_pos]
    for ax in axes[:-1]:
        ax.set_xticks([])
    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels(tick_labels, fontsize=6, color=C_TEXT)

    # 저장 → base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=C_BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ═══ AI ═══
def init_ai():
    g=c=None
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI; g=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic; c=anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return g,c

SYS = """당신은 15년 경력의 BTC/USDT 선물 퀀트 트레이더입니다.
차트 이미지를 분석하여 진입 여부를 판단합니다.

[차트 구성 - 위에서 아래로]
1. 캔들차트 + EMA(3,5,10,21,50,100,200) + 볼린저밴드
2. 거래량 + 20MA
3. ADX(흰색) + +DI(초록) + -DI(빨강) + 35기준선(주황점선)
4. RSI(14) + 30/65 기준선
5. MACD + Signal + 히스토그램
6. 스토캐스틱 %K/%D

[판단 방법]
- 차트의 시각적 패턴을 종합 분석
- EMA 배열(정배열/역배열), 크로스 위치
- ADX 방향과 강도, DI 크로스
- RSI 추세와 위치
- MACD 히스토그램 방향 전환
- 거래량 동반 여부
- 이미 같은 방향 포지션이면 HOLD

[출력] JSON만: {"action":"LONG/SHORT/HOLD","confidence":1~10,"reason":"한 줄"}"""

def ask_gpt_vision(cli, img_b64, text, retries=5):
    for a in range(retries):
        try:
            r=cli.chat.completions.create(model="gpt-5.4",temperature=0,
                response_format={"type":"json_object"},
                messages=[{"role":"system","content":SYS},
                    {"role":"user","content":[
                        {"type":"text","text":text},
                        {"type":"image_url","image_url":{"url":f"data:image/png;base64,{img_b64}","detail":"high"}}
                    ]}])
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            w=min(30,5*(a+1));print(f"    [GPT retry {a+1}] {type(e).__name__} {w}s")
            if a<retries-1: time.sleep(w)
    return None

def ask_claude_vision(cli, img_b64, text, retries=5):
    for a in range(retries):
        try:
            r=cli.messages.create(model="claude-sonnet-4-6",max_tokens=300,temperature=0,
                system=SYS,
                messages=[{"role":"user","content":[
                    {"type":"text","text":text},
                    {"type":"image","source":{"type":"base64","media_type":"image/png","data":img_b64}}
                ]}])
            t=r.content[0].text;s=t.find('{');e=t.rfind('}')+1
            return json.loads(t[s:e])
        except Exception as e:
            w=min(60,10*(a+1));print(f"    [Claude retry {a+1}] {type(e).__name__} {w}s")
            if a<retries-1: time.sleep(w)
    return None

def build_text(ts, cross, price, bal, pos_str, mpnl):
    return f"""시각: {ts}
BTC/USDT 30분봉 **{cross}** 발생 (차트의 마지막 캔들에 마커 표시)
현재가: ${price:,.0f}
잔액: ${bal:,.0f} | 포지션: {pos_str} | 이번달: {mpnl:+.1f}%

위 차트를 분석하여 진입 여부를 판단하세요.
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
    def enter(s,d,px,ts):
        if s.bal<=0 or s.bal*MPCT<1: return False
        if s.m_start>0 and (s.bal-s.m_start)/s.m_start<=ML_LIM: return False
        s.pos=d;s.epx=px;s.margin=s.bal*MPCT;s.psz=s.margin*LEV
        s.bal-=s.psz*FEE;s.tsl_on=False;s.peak=px;s.trough=px
        s.slp=px*(1-SL_PCT/100) if d==1 else px*(1+SL_PCT/100)
        if s.cur_m in s.monthly: s.monthly[s.cur_m]['ent']+=1
        return True
    def close(s,px,reason,ts):
        if s.pos==0: return 0
        if reason=='fl': pnl=-(s.margin+s.psz*FEE)
        else: pnl=(px-s.epx)/s.epx*s.psz*s.pos-s.psz*FEE
        s.bal+=pnl;
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
        s.trades.append({'ts':str(ts),'dir':'L' if s.pos==1 else 'S','entry':s.epx,'exit':px,
            'pnl':round(pnl,2),'reason':reason,'bal':round(s.bal,2)})
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
    print("  AI 차트 기반 백테스트 v3 (트레이딩뷰 스타일)")
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
    ema3_s=ema(c,3);ema5_s=ema(c,5);ema10_s=ema(c,10)
    ema21_s=ema(c,21);ema50_s=ema(c,50);ema100_s=ema(c,100);ema200_s=ema(c,200)
    adx_s,pdi_s,mdi_s=adx_calc(h,l,c,14);rsi_s=rsi_calc(c,14)
    macd_s,msig_s,mhist_s=macd_calc(c)
    bb_up,bb_mid,bb_lo=bb_calc(c)
    stk_s,std_s=stoch_calc(h,l,c)

    bull=ema3_s>ema200_s;prev=bull.shift(1).fillna(False)
    golden=(bull&~prev).values;dead=(~bull&prev).values
    v14f=((adx_s>=35)&(rsi_s>=30)&(rsi_s<=65)).values
    ts_arr=df30.index;cl=c.values;hi=h.values;lo=l.values;n=len(df30)

    start_i=max(WARMUP,np.searchsorted(ts_arr,pd.Timestamp('2023-01-01')))
    total=n-start_i

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

        ga=gr=cla_a=cla_r="";gc=cc=0

        if av>=AI_ADX:
            # 차트 생성
            img_b64=make_chart(df30,i,cross,ema3_s,ema5_s,ema10_s,ema21_s,ema50_s,ema100_s,ema200_s,
                               adx_s,pdi_s,mdi_s,rsi_s,macd_s,msig_s,mhist_s,bb_up,bb_mid,bb_lo,stk_s,std_s)

            if img_b64 is None:
                ga="HOLD";gr="차트생성실패";cla_a="HOLD";cla_r="차트생성실패"
            else:
                # 10번째마다 차트 파일 저장 (샘플)
                if cx % 10 == 1:
                    with open(f"{CHART_DIR}/cross_{cx:04d}.png", "wb") as f:
                        f.write(base64.b64decode(img_b64))

                text_gpt=build_text(str(t),cross,c_,gpt.bal,gpt.pos_str(),gpt.mpnl())
                text_cla=build_text(str(t),cross,c_,cla.bal,cla.pos_str(),cla.mpnl())

                if gpt_cli:
                    res=ask_gpt_vision(gpt_cli,img_b64,text_gpt);api+=1
                    if res: ga=res.get('action','HOLD');gr=res.get('reason','');gc=res.get('confidence',0)
                    else: errs['gpt']+=1;ga="HOLD";gr="API오류"
                else: ga="HOLD";gr="API없음"

                if cld_cli:
                    res=ask_claude_vision(cld_cli,img_b64,text_cla);api+=1
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
    print(f"{'':>12}{'코드봇v14.4':>14}{'GPT-5.4차트':>14}{'Claude차트':>14}")
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
    gn=[x for x in new_only if x['gpt'] in('LONG','SHORT')]
    cn=[x for x in new_only if x['claude'] in('LONG','SHORT')]
    ge=[x for x in filt_pass if x['gpt'] in('LONG','SHORT')]
    ce=[x for x in filt_pass if x['claude'] in('LONG','SHORT')]
    print(f"\n크로스: 전체{tot} AI호출{len(ai_called)} V14필터{len(filt_pass)} NEW{len(new_only)}")
    print(f"NEW진입: GPT {len(gn)}/{len(new_only)} Claude {len(cn)}/{len(new_only)}")
    print(f"V14진입: GPT {len(ge)}/{len(filt_pass)} Claude {len(ce)}/{len(filt_pass)}")

    all_yrs=sorted(set(list(code.yr_bal)+list(gpt.yr_bal)+list(cla.yr_bal)))
    print(f"\n연도별:")
    for yr in all_yrs:
        print(f"  {yr} | 코드:${code.yr_bal.get(yr,0):>10,.0f} GPT:${gpt.yr_bal.get(yr,0):>10,.0f} Claude:${cla.yr_bal.get(yr,0):>10,.0f}")

    ranking=sorted([(cs['bal'],'코드봇'),(gs['bal'],'GPT'),(ls['bal'],'Claude')],reverse=True)
    print(f"\n순위: 1등 {ranking[0][1]} ${ranking[0][0]:,.0f} | 2등 {ranking[1][1]} ${ranking[1][0]:,.0f} | 3등 {ranking[2][1]} ${ranking[2][0]:,.0f}")
    print(f"API: {api}회 | 오류: GPT {errs['gpt']} Claude {errs['claude']}")
    print(f"차트 샘플: {CHART_DIR}/")

    pd.DataFrame(cross_log).to_csv(f"{BASE}/ai_chart_trades.csv",index=False,encoding='utf-8-sig')
    months=sorted(set(list(code.monthly)+list(gpt.monthly)+list(cla.monthly)))
    rows=[]
    for mk in months:
        cd=code.monthly.get(mk,{});gd=gpt.monthly.get(mk,{});ld=cla.monthly.get(mk,{})
        rows.append({'month':mk,'code_bal':round(cd.get('eq',0)),'gpt_bal':round(gd.get('eq',0)),'claude_bal':round(ld.get('eq',0))})
    pd.DataFrame(rows).to_csv(f"{BASE}/ai_chart_monthly.csv",index=False,encoding='utf-8-sig')
    print(f"저장: ai_chart_trades.csv, ai_chart_monthly.csv")

if __name__=="__main__": run()
