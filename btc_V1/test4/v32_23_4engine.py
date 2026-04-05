"""v32.2 & v32.3 - 4개 엔진 백테스트"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from bt_fast import load_5m_data, build_mtf, calc_ema, calc_sma, calc_rsi
from v26_wilder_bt import calc_adx_wilder
from bt_fast import calc_adx as calc_adx_ewm

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)
df30 = mtf['30m']
c = df30['close'].values.astype(np.float64)
h = df30['high'].values.astype(np.float64)
l = df30['low'].values.astype(np.float64)
t = df30['time'].values
n = len(c)

# Pre-calc indicators
ema100 = calc_ema(df30['close'], 100).values
ema600 = calc_ema(df30['close'], 600).values
ema75 = calc_ema(df30['close'], 75).values
sma750 = calc_sma(df30['close'], 750).values

adx_ewm_20 = calc_adx_ewm(df30['high'], df30['low'], df30['close'], 20).values
adx_wilder_20 = calc_adx_wilder(df30['high'], df30['low'], df30['close'], 20)

rsi10 = calc_rsi(df30['close'], 10).values
rsi11 = calc_rsi(df30['close'], 11).values
rsi14 = calc_rsi(df30['close'], 14).values


def run_engine(c, h, l, t, fast_ma, slow_ma, adx, rsi, cfg, engine_name):
    """공통 백테스트 (engine별 로직 차이 적용)"""
    n = len(c)
    cap = cfg['cap']; lev = cfg['lev']; fee = 0.0004; mn = cfg['mn']
    pos = 0; ep = 0.0; psz = 0.0; slp = 0.0
    ton = False; thi = 0.0; tlo = 999999.0
    watching = 0; ws = 0; ld = 0; pk = cap; mdd = 0.0; ms = cap
    trades = []

    tsl_disables_sl = 'TSL_SL' in engine_name  # Engine 3,4: TSL->SL비활성

    for i in range(600, n):
        px = c[i]; h_ = h[i]; l_ = l[i]
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            continue

        if i > 600 and i % 48 == 0: ms = cap

        # STEP A: Position management
        if pos != 0:
            watching = 0

            # A1: SL (TSL비활성 모드면 ton=True일때 SL 스킵)
            if not (tsl_disables_sl and ton):
                sl_hit = (pos==1 and l_<=slp) or (pos==-1 and h_>=slp)
                if sl_hit:
                    pnl = (slp-ep)/ep*psz*pos - psz*fee
                    cap += pnl
                    trades.append({'pnl':pnl,'type':'SL','bal':cap,'yr':pd.Timestamp(t[i]).year})
                    ld=pos; pos=0; pk=max(pk,cap)
                    dd=(pk-cap)/pk if pk>0 else 0
                    if dd>mdd: mdd=dd
                    continue

            # A2: TA activation
            if pos==1: br=(h_-ep)/ep*100
            else: br=(ep-l_)/ep*100
            if br >= cfg['ta']: ton=True

            # A3: TSL
            if ton:
                if pos==1:
                    if h_>thi: thi=h_
                    ns=thi*(1-cfg['tsl']/100)
                    if ns>slp: slp=ns
                    if px<=slp:
                        pnl=(px-ep)/ep*psz*pos - psz*fee
                        cap+=pnl
                        trades.append({'pnl':pnl,'type':'TSL','bal':cap,'yr':pd.Timestamp(t[i]).year})
                        ld=pos; pos=0; pk=max(pk,cap)
                        dd=(pk-cap)/pk if pk>0 else 0
                        if dd>mdd: mdd=dd
                        continue
                else:
                    if l_<tlo: tlo=l_
                    ns=tlo*(1+cfg['tsl']/100)
                    if ns<slp: slp=ns
                    if px>=slp:
                        pnl=(px-ep)/ep*psz*pos - psz*fee
                        cap+=pnl
                        trades.append({'pnl':pnl,'type':'TSL','bal':cap,'yr':pd.Timestamp(t[i]).year})
                        ld=pos; pos=0; pk=max(pk,cap)
                        dd=(pk-cap)/pk if pk>0 else 0
                        if dd>mdd: mdd=dd
                        continue

            # A4: REV
            if i > 0:
                bn = fast_ma[i]>slow_ma[i]; bp = fast_ma[i-1]>slow_ma[i-1]
                cu = bn and not bp; cd = not bn and bp
                if (pos==1 and cd) or (pos==-1 and cu):
                    pnl=(px-ep)/ep*psz*pos - psz*fee
                    cap+=pnl
                    trades.append({'pnl':pnl,'type':'REV','bal':cap,'yr':pd.Timestamp(t[i]).year})
                    ld=pos; pos=0; pk=max(pk,cap)
                    dd=(pk-cap)/pk if pk>0 else 0
                    if dd>mdd: mdd=dd

        # STEP B: Entry
        if i < 1: continue
        bn = fast_ma[i]>slow_ma[i]; bp = fast_ma[i-1]>slow_ma[i-1]
        cu = bn and not bp; cd = not bn and bp

        if pos == 0:
            if cu: watching=1; ws=i
            elif cd: watching=-1; ws=i

            if watching!=0 and i>ws:
                if i-ws>24: watching=0; continue
                if watching==1 and cd: watching=-1; ws=i; continue
                elif watching==-1 and cu: watching=1; ws=i; continue
                if cfg.get('skip',True) and watching==ld: continue
                if adx[i]<cfg['amin']: continue
                if cfg.get('adx_rise',6)>0 and i>=6:
                    if adx[i]<=adx[i-6]: continue
                if rsi[i]<cfg['rmin'] or rsi[i]>cfg['rmax']: continue
                if cfg.get('gap',0.2)>0:
                    gap=abs(fast_ma[i]-slow_ma[i])/slow_ma[i]*100 if slow_ma[i]>0 else 0
                    if gap<0.2: continue
                if ms>0 and (cap-ms)/ms<=-0.20: watching=0; continue
                if cap<=0: continue

                mg=cap*mn; psz=mg*lev; cap-=psz*fee
                pos=watching; ep=px; ton=False; thi=px; tlo=px
                if pos==1: slp=ep*(1-cfg['sl']/100)
                else: slp=ep*(1+cfg['sl']/100)
                pk=max(pk,cap); watching=0

        pk=max(pk,cap)
        dd=(pk-cap)/pk if pk>0 else 0
        if dd>mdd: mdd=dd
        if cap<=0: break

    if pos!=0 and cap>0:
        px=c[n-1]
        pnl=(px-ep)/ep*psz*pos - psz*fee
        cap+=pnl
        trades.append({'pnl':pnl,'type':'END','bal':cap,'yr':2026})

    df_t = pd.DataFrame(trades) if trades else pd.DataFrame()
    tc = len(df_t)
    if tc == 0:
        return {'trades':0,'bal':cap,'ret':(cap-cfg['cap'])/cfg['cap']*100,'pf':0,'mdd':mdd*100,'sl':0,'tsl':0,'rev':0}
    wins = (df_t['pnl']>0).sum()
    tp = df_t[df_t['pnl']>0]['pnl'].sum()
    tl = abs(df_t[df_t['pnl']<=0]['pnl'].sum()) + 1e-10
    sl_c = (df_t['type']=='SL').sum()
    tsl_c = (df_t['type']=='TSL').sum()
    rev_c = (df_t['type']=='REV').sum()

    # Year consistency
    loss_yrs = 0
    for yr in df_t['yr'].unique():
        yt = df_t[df_t['yr']==yr]
        if yt['pnl'].sum() < 0: loss_yrs += 1

    return {
        'trades':tc, 'bal':round(cap,0), 'ret':round((cap-cfg['cap'])/cfg['cap']*100,0),
        'pf':round(tp/tl,2), 'mdd':round(mdd*100,1), 'wr':round(wins/tc*100,1),
        'sl':int(sl_c), 'tsl':int(tsl_c), 'rev':int(rev_c),
        'loss_yrs':loss_yrs, 'wins':int(wins), 'losses':tc-int(wins),
    }


# ═══ CONFIGS ═══
configs = [
    ('v32.2', ema100, ema600, {'cap':5000,'lev':10,'mn':0.35,'sl':3,'ta':12,'tsl':9,'amin':30,'rmin':40,'rmax':80,'adx_rise':6,'gap':0.2,'skip':True}, rsi10),
    ('v32.3', ema75, sma750, {'cap':5000,'lev':10,'mn':0.35,'sl':3,'ta':12,'tsl':9,'amin':30,'rmin':40,'rmax':80,'adx_rise':6,'gap':0.2,'skip':True}, rsi11),
]

engines = [
    ('E1_bt_fast(ewm)', adx_ewm_20, False),
    ('E2_Wilder_ADX', adx_wilder_20, False),
    ('E3_TSL_SL_ewm', adx_ewm_20, True),
    ('E4_TSL_SL_Wilder', adx_wilder_20, True),
]

print()
print('='*140)
print('  v32.2 & v32.3 - 4 Engine Backtest')
print('='*140)

for ver, fast_ma, slow_ma, cfg, rsi_v in configs:
    print(f'\n  [{ver}] {ver.replace("v32.2","EMA100/EMA600").replace("v32.3","EMA75/SMA750")}')
    print(f'  {"─"*130}')
    print(f'  | {"엔진":^22} | {"거래":^4} | {"잔액$":^14} | {"수익률":^12} | {"PF":^6} | {"MDD":^6} | {"승률":^5} | {"SL":^3} | {"TSL":^4} | {"REV":^4} | {"손실연":^4} | {"기획서일치":^8} |')
    print(f'  |{"-"*24}|{"-"*6}|{"-"*16}|{"-"*14}|{"-"*8}|{"-"*8}|{"-"*7}|{"-"*5}|{"-"*6}|{"-"*6}|{"-"*6}|{"-"*10}|')

    for eng_name, adx_v, tsl_sl in engines:
        ename = eng_name + ('_TSL_SL' if tsl_sl else '')
        r = run_engine(c, h, l, t, fast_ma, slow_ma, adx_v, rsi_v, cfg, ename)

        # Check match with planning doc
        if ver == 'v32.2':
            match = 'O' if abs(r['trades']-70)<=2 and r['pf']>=5.0 else 'X'
        else:
            match = 'O' if abs(r['trades']-69)<=2 and r['pf']>=2.5 else 'X'

        print(f'  | {eng_name:^22} | {r["trades"]:>4} | ${r["bal"]:>12,.0f} | {r["ret"]:>+10,.0f}% | {r["pf"]:>5.2f} | {r["mdd"]:>5.1f}% | {r["wr"]:>4.1f}% | {r["sl"]:>3} | {r["tsl"]:>4} | {r["rev"]:>4} | {r["loss_yrs"]:>4} | {match:^8} |')

print('\n\nDONE.')
