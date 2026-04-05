"""v32.2 & v32.3 - 6개 엔진 교차 검증 (의사코드 정밀 재현)"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from bt_fast import load_5m_data, build_mtf, calc_ema, calc_sma

df5 = load_5m_data(r'D:\filesystem\futures\btc_V1\test4')
mtf = build_mtf(df5)
df30 = mtf['30m']
c = df30['close'].values.astype(np.float64)
h = df30['high'].values.astype(np.float64)
l = df30['low'].values.astype(np.float64)
t = df30['time'].values
n = len(c)

# ═══ 지표 계산 (기획서 공식 정확 적용) ═══
# EMA: ewm(span=N, adjust=False)
ema100 = df30['close'].ewm(span=100, adjust=False).mean().values
ema600 = df30['close'].ewm(span=600, adjust=False).mean().values
ema75 = df30['close'].ewm(span=75, adjust=False).mean().values
sma750 = df30['close'].rolling(750).mean().values

# ADX(20): ewm(alpha=1/20) — 기획서 명시
def calc_adx_alpha(high, low, close, period=20):
    h=high; l=low; c=close
    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-10)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return adx.values

adx20 = calc_adx_alpha(df30['high'], df30['low'], df30['close'], 20)

# RSI: ewm(alpha=1/period) — 기획서 명시
def calc_rsi_alpha(close, period):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return (100 - 100 / (1 + avg_gain / avg_loss.replace(0, 1e-10))).values

rsi10 = calc_rsi_alpha(df30['close'], 10)
rsi11 = calc_rsi_alpha(df30['close'], 11)

print(f'ADX(20) ewm(alpha=1/20): mean={np.nanmean(adx20):.1f}, >=30: {np.sum(adx20>=30):,}', flush=True)


def run_v32_precise(fast_ma, slow_ma, adx, rsi, cfg, engine_label):
    """v32.2/v32.3 의사코드 100% 재현"""
    cap = cfg['cap']; pos = 0; epx = 0.0; psz = 0.0; slp = 0.0
    ton = False; thi = 0.0; tlo = 999999.0
    watching = 0; ws = 0; le = 0; ld = 0
    pk = cap; mdd = 0.0; ms = cap
    sl_c=0; tsl_c=0; rev_c=0; wins=0; losses=0

    for i in range(600, n):
        px = c[i]; h_ = h[i]; l_ = l[i]
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            continue

        # 일일 리셋 (30분봉 1440봉 = 1일) — 기획서 명시
        if i > 600 and i % 1440 == 0:
            ms = cap

        # STEP A
        if pos != 0:
            watching = 0

            # A1: SL (TSL 미활성 시에만)
            if not ton:
                sl_hit = (pos==1 and l_<=slp) or (pos==-1 and h_>=slp)
                if sl_hit:
                    pnl = (slp - epx) / epx * psz * pos - psz * 0.0004
                    cap += pnl
                    sl_c += 1
                    if pnl > 0: wins += 1
                    else: losses += 1
                    ld = pos; le = i; pos = 0
                    pk = max(pk, cap)
                    dd = (pk-cap)/pk if pk>0 else 0
                    if dd > mdd: mdd = dd
                    continue

            # A2: TA
            if pos == 1: br = (h_ - epx) / epx * 100
            else: br = (epx - l_) / epx * 100
            if br >= cfg['ta']: ton = True

            # A3: TSL
            if ton:
                if pos == 1:
                    if h_ > thi: thi = h_
                    ns = thi * (1 - cfg['tsl']/100)
                    if ns > slp: slp = ns
                    if px <= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * 0.0004
                        cap += pnl; tsl_c += 1
                        if pnl > 0: wins += 1
                        else: losses += 1
                        ld = pos; le = i; pos = 0
                        pk = max(pk, cap)
                        dd = (pk-cap)/pk if pk>0 else 0
                        if dd > mdd: mdd = dd
                        continue
                else:
                    if l_ < tlo: tlo = l_
                    ns = tlo * (1 + cfg['tsl']/100)
                    if ns < slp: slp = ns
                    if px >= slp:
                        pnl = (px - epx) / epx * psz * pos - psz * 0.0004
                        cap += pnl; tsl_c += 1
                        if pnl > 0: wins += 1
                        else: losses += 1
                        ld = pos; le = i; pos = 0
                        pk = max(pk, cap)
                        dd = (pk-cap)/pk if pk>0 else 0
                        if dd > mdd: mdd = dd
                        continue

            # A4: REV
            if i > 0:
                bn = fast_ma[i]>slow_ma[i]; bp = fast_ma[i-1]>slow_ma[i-1]
                cu = bn and not bp; cd = not bn and bp
                if (pos==1 and cd) or (pos==-1 and cu):
                    pnl = (px - epx) / epx * psz * pos - psz * 0.0004
                    cap += pnl; rev_c += 1
                    if pnl > 0: wins += 1
                    else: losses += 1
                    ld = pos; le = i; pos = 0
                    pk = max(pk, cap)
                    dd = (pk-cap)/pk if pk>0 else 0
                    if dd > mdd: mdd = dd
                    # NO CONTINUE

        # STEP B
        if i < 1: continue
        bn = fast_ma[i]>slow_ma[i]; bp = fast_ma[i-1]>slow_ma[i-1]
        cu = bn and not bp; cd = not bn and bp

        if pos == 0:
            if cu: watching = 1; ws = i
            elif cd: watching = -1; ws = i

            if watching != 0 and i > ws:
                if i - ws > 24: watching = 0; continue
                if watching == 1 and cd: watching = -1; ws = i; continue
                elif watching == -1 and cu: watching = 1; ws = i; continue
                if watching == ld: continue
                if adx[i] < 30.0: continue
                if i >= 6 and adx[i] <= adx[i-6]: continue
                if rsi[i] < cfg['rsi_min'] or rsi[i] > cfg['rsi_max']: continue
                if slow_ma[i] > 0:
                    gap = abs(fast_ma[i] - slow_ma[i]) / slow_ma[i] * 100
                    if gap < 0.2: continue
                if ms > 0 and (cap - ms) / ms <= -0.20: watching = 0; continue
                if cap <= 0: continue

                mg = cap * 0.35; psz = mg * 10
                cap -= psz * 0.0004
                pos = watching; epx = px; ton = False; thi = px; tlo = px
                if pos == 1: slp = epx * (1 - 3.0/100)
                else: slp = epx * (1 + 3.0/100)
                pk = max(pk, cap); watching = 0

        pk = max(pk, cap)
        dd = (pk-cap)/pk if pk>0 else 0
        if dd > mdd: mdd = dd
        if cap <= 0: break

    if pos != 0 and cap > 0:
        px = c[n-1]
        pnl = (px - epx) / epx * psz * pos - psz * 0.0004
        cap += pnl
        if pnl > 0: wins += 1
        else: losses += 1

    tc = sl_c + tsl_c + rev_c + (1 if pos != 0 else 0)
    return {
        'engine': engine_label, 'bal': round(cap, 2), 'ret': round((cap-cfg['cap'])/cfg['cap']*100, 0),
        'trades': tc, 'wins': wins, 'losses': losses,
        'pf_approx': round(cap/cfg['cap'], 1),
        'mdd': round(mdd*100, 1), 'sl': sl_c, 'tsl': tsl_c, 'rev': rev_c,
    }


# ═══ 6 ENGINE VARIANTS ═══
def make_engines(fast_ma, slow_ma, rsi_v, cfg):
    results = []

    # E1: 기획서 정확 재현 (ewm alpha=1/20 ADX, ewm alpha RSI)
    results.append(run_v32_precise(fast_ma, slow_ma, adx20, rsi_v, cfg, 'E1: Precise (ewm alpha)'))

    # E2: ADX ewm(span=20) — bt_fast 방식
    from bt_fast import calc_adx as calc_adx_span
    adx_span = calc_adx_span(df30['high'], df30['low'], df30['close'], 20).values
    results.append(run_v32_precise(fast_ma, slow_ma, adx_span, rsi_v, cfg, 'E2: ADX ewm(span=20)'))

    # E3: RSI ewm(span) 대신 ewm(alpha) 차이 테스트
    from bt_fast import calc_rsi as calc_rsi_span
    if cfg['rsi_p'] == 10:
        rsi_span = calc_rsi_span(df30['close'], 10).values
    else:
        rsi_span = calc_rsi_span(df30['close'], 11).values
    results.append(run_v32_precise(fast_ma, slow_ma, adx20, rsi_span, cfg, 'E3: RSI ewm(span)'))

    # E4: 일일리셋 i%48 (24시간=48봉)
    # Modify: change daily reset
    cap2 = cfg['cap']; pos2 = 0; epx2 = 0.0; psz2 = 0.0; slp2 = 0.0
    ton2 = False; thi2 = 0.0; tlo2 = 999999.0
    watching2 = 0; ws2 = 0; ld2 = 0; pk2 = cap2; mdd2 = 0.0; ms2 = cap2
    sl2=0; tsl2=0; rev2=0; w2=0; l2=0
    for i in range(600, n):
        px=c[i]; h_=h[i]; l_=l[i]
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(adx20[i]) or np.isnan(rsi_v[i]): continue
        if i>600 and i%48==0: ms2=cap2  # 48봉 리셋
        if pos2!=0:
            watching2=0
            if not ton2:
                if (pos2==1 and l_<=slp2) or (pos2==-1 and h_>=slp2):
                    pnl=(slp2-epx2)/epx2*psz2*pos2-psz2*0.0004; cap2+=pnl; sl2+=1
                    if pnl>0: w2+=1
                    else: l2+=1
                    ld2=pos2; pos2=0; pk2=max(pk2,cap2); dd=(pk2-cap2)/pk2 if pk2>0 else 0
                    if dd>mdd2: mdd2=dd
                    continue
            if pos2==1: br=(h_-epx2)/epx2*100
            else: br=(epx2-l_)/epx2*100
            if br>=cfg['ta']: ton2=True
            if ton2:
                if pos2==1:
                    if h_>thi2: thi2=h_
                    ns=thi2*(1-cfg['tsl']/100)
                    if ns>slp2: slp2=ns
                    if px<=slp2:
                        pnl=(px-epx2)/epx2*psz2*pos2-psz2*0.0004; cap2+=pnl; tsl2+=1
                        if pnl>0: w2+=1
                        else: l2+=1
                        ld2=pos2; pos2=0; pk2=max(pk2,cap2); dd=(pk2-cap2)/pk2 if pk2>0 else 0
                        if dd>mdd2: mdd2=dd
                        continue
                else:
                    if l_<tlo2: tlo2=l_
                    ns=tlo2*(1+cfg['tsl']/100)
                    if ns<slp2: slp2=ns
                    if px>=slp2:
                        pnl=(px-epx2)/epx2*psz2*pos2-psz2*0.0004; cap2+=pnl; tsl2+=1
                        if pnl>0: w2+=1
                        else: l2+=1
                        ld2=pos2; pos2=0; pk2=max(pk2,cap2); dd=(pk2-cap2)/pk2 if pk2>0 else 0
                        if dd>mdd2: mdd2=dd
                        continue
            if i>0:
                bn=fast_ma[i]>slow_ma[i]; bp=fast_ma[i-1]>slow_ma[i-1]
                cu=bn and not bp; cd=not bn and bp
                if (pos2==1 and cd) or (pos2==-1 and cu):
                    pnl=(px-epx2)/epx2*psz2*pos2-psz2*0.0004; cap2+=pnl; rev2+=1
                    if pnl>0: w2+=1
                    else: l2+=1
                    ld2=pos2; pos2=0; pk2=max(pk2,cap2); dd=(pk2-cap2)/pk2 if pk2>0 else 0
                    if dd>mdd2: mdd2=dd
        if i<1: continue
        bn=fast_ma[i]>slow_ma[i]; bp=fast_ma[i-1]>slow_ma[i-1]
        cu=bn and not bp; cd=not bn and bp
        if pos2==0:
            if cu: watching2=1; ws2=i
            elif cd: watching2=-1; ws2=i
            if watching2!=0 and i>ws2:
                if i-ws2>24: watching2=0; continue
                if watching2==1 and cd: watching2=-1; ws2=i; continue
                elif watching2==-1 and cu: watching2=1; ws2=i; continue
                if watching2==ld2: continue
                if adx20[i]<30: continue
                if i>=6 and adx20[i]<=adx20[i-6]: continue
                if rsi_v[i]<cfg['rsi_min'] or rsi_v[i]>cfg['rsi_max']: continue
                if slow_ma[i]>0:
                    gap=abs(fast_ma[i]-slow_ma[i])/slow_ma[i]*100
                    if gap<0.2: continue
                if ms2>0 and (cap2-ms2)/ms2<=-0.20: watching2=0; continue
                if cap2<=0: continue
                mg=cap2*0.35; psz2=mg*10; cap2-=psz2*0.0004
                pos2=watching2; epx2=px; ton2=False; thi2=px; tlo2=px
                if pos2==1: slp2=epx2*0.97
                else: slp2=epx2*1.03
                pk2=max(pk2,cap2); watching2=0
        pk2=max(pk2,cap2); dd=(pk2-cap2)/pk2 if pk2>0 else 0
        if dd>mdd2: mdd2=dd
        if cap2<=0: break
    if pos2!=0 and cap2>0:
        px=c[n-1]; pnl=(px-epx2)/epx2*psz2*pos2-psz2*0.0004; cap2+=pnl
        if pnl>0: w2+=1
        else: l2+=1
    tc4 = sl2+tsl2+rev2+(1 if pos2!=0 else 0)
    results.append({'engine':'E4: Daily 48bar reset','bal':round(cap2,2),'ret':round((cap2-cfg['cap'])/cfg['cap']*100,0),
                    'trades':tc4,'wins':w2,'losses':l2,'mdd':round(mdd2*100,1),'sl':sl2,'tsl':tsl2,'rev':rev2})

    # E5: ADX rise OFF (no adx[i]>adx[i-6] check)
    # Quick variant
    r5 = run_v32_norise(fast_ma, slow_ma, adx20, rsi_v, cfg)
    results.append(r5)

    # E6: Monitor window 12 (instead of 24)
    r6 = run_v32_window(fast_ma, slow_ma, adx20, rsi_v, cfg, 12)
    results.append(r6)

    return results


def run_v32_norise(fast_ma, slow_ma, adx, rsi, cfg):
    """E5: ADX rise check 제거"""
    cap=cfg['cap']; pos=0; epx=0; psz=0; slp=0; ton=False; thi=0; tlo=999999
    watching=0; ws=0; ld=0; pk=cap; mdd=0; ms=cap; sl_c=0; tsl_c=0; rev_c=0; wins=0; losses=0
    for i in range(600, n):
        px=c[i]; h_=h[i]; l_=l[i]
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]): continue
        if i>600 and i%1440==0: ms=cap
        if pos!=0:
            watching=0
            if not ton:
                if (pos==1 and l_<=slp) or (pos==-1 and h_>=slp):
                    pnl=(slp-epx)/epx*psz*pos-psz*0.0004; cap+=pnl; sl_c+=1
                    if pnl>0: wins+=1
                    else: losses+=1
                    ld=pos; pos=0; pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0
                    if dd>mdd: mdd=dd; continue
            if pos==1: br=(h_-epx)/epx*100
            else: br=(epx-l_)/epx*100
            if br>=cfg['ta']: ton=True
            if ton:
                if pos==1:
                    if h_>thi: thi=h_
                    ns=thi*(1-cfg['tsl']/100)
                    if ns>slp: slp=ns
                    if px<=slp: pnl=(px-epx)/epx*psz*pos-psz*0.0004; cap+=pnl; tsl_c+=1; (wins:=wins+1) if pnl>0 else (losses:=losses+1); ld=pos; pos=0; pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd); continue
                else:
                    if l_<tlo: tlo=l_
                    ns=tlo*(1+cfg['tsl']/100)
                    if ns<slp: slp=ns
                    if px>=slp: pnl=(px-epx)/epx*psz*pos-psz*0.0004; cap+=pnl; tsl_c+=1; (wins:=wins+1) if pnl>0 else (losses:=losses+1); ld=pos; pos=0; pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd); continue
            if i>0:
                bn=fast_ma[i]>slow_ma[i]; bp=fast_ma[i-1]>slow_ma[i-1]; cu=bn and not bp; cd=not bn and bp
                if (pos==1 and cd) or (pos==-1 and cu):
                    pnl=(px-epx)/epx*psz*pos-psz*0.0004; cap+=pnl; rev_c+=1
                    if pnl>0: wins+=1
                    else: losses+=1
                    ld=pos; pos=0; pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd)
        if i<1: continue
        bn=fast_ma[i]>slow_ma[i]; bp=fast_ma[i-1]>slow_ma[i-1]; cu=bn and not bp; cd=not bn and bp
        if pos==0:
            if cu: watching=1; ws=i
            elif cd: watching=-1; ws=i
            if watching!=0 and i>ws:
                if i-ws>24: watching=0; continue
                if watching==1 and cd: watching=-1; ws=i; continue
                elif watching==-1 and cu: watching=1; ws=i; continue
                if watching==ld: continue
                if adx[i]<30: continue
                # NO ADX RISE CHECK (E5 difference)
                if rsi[i]<cfg['rsi_min'] or rsi[i]>cfg['rsi_max']: continue
                if slow_ma[i]>0:
                    gap=abs(fast_ma[i]-slow_ma[i])/slow_ma[i]*100
                    if gap<0.2: continue
                if ms>0 and (cap-ms)/ms<=-0.20: watching=0; continue
                if cap<=0: continue
                mg=cap*0.35; psz=mg*10; cap-=psz*0.0004; pos=watching; epx=px; ton=False; thi=px; tlo=px
                if pos==1: slp=epx*0.97
                else: slp=epx*1.03
                pk=max(pk,cap); watching=0
        pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd)
        if cap<=0: break
    if pos!=0 and cap>0:
        px=c[n-1]; pnl=(px-epx)/epx*psz*pos-psz*0.0004; cap+=pnl
    tc=sl_c+tsl_c+rev_c+(1 if pos!=0 else 0)
    return {'engine':'E5: No ADX rise','bal':round(cap,2),'ret':round((cap-cfg['cap'])/cfg['cap']*100,0),'trades':tc,'wins':wins,'losses':losses,'mdd':round(mdd*100,1),'sl':sl_c,'tsl':tsl_c,'rev':rev_c}


def run_v32_window(fast_ma, slow_ma, adx, rsi, cfg, window):
    """E6: Monitor window 변경"""
    cap=cfg['cap']; pos=0; epx=0; psz=0; slp=0; ton=False; thi=0; tlo=999999
    watching=0; ws=0; ld=0; pk=cap; mdd=0; ms=cap; sl_c=0; tsl_c=0; rev_c=0; wins=0; losses=0
    for i in range(600, n):
        px=c[i]; h_=h[i]; l_=l[i]
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]): continue
        if i>600 and i%1440==0: ms=cap
        if pos!=0:
            watching=0
            if not ton:
                if (pos==1 and l_<=slp) or (pos==-1 and h_>=slp):
                    pnl=(slp-epx)/epx*psz*pos-psz*0.0004; cap+=pnl; sl_c+=1
                    if pnl>0: wins+=1
                    else: losses+=1
                    ld=pos; pos=0; pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd); continue
            if pos==1: br=(h_-epx)/epx*100
            else: br=(epx-l_)/epx*100
            if br>=cfg['ta']: ton=True
            if ton:
                if pos==1:
                    if h_>thi: thi=h_
                    ns=thi*(1-cfg['tsl']/100)
                    if ns>slp: slp=ns
                    if px<=slp: pnl=(px-epx)/epx*psz*pos-psz*0.0004; cap+=pnl; tsl_c+=1; (wins:=wins+1) if pnl>0 else (losses:=losses+1); ld=pos; pos=0; pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd); continue
                else:
                    if l_<tlo: tlo=l_
                    ns=tlo*(1+cfg['tsl']/100)
                    if ns<slp: slp=ns
                    if px>=slp: pnl=(px-epx)/epx*psz*pos-psz*0.0004; cap+=pnl; tsl_c+=1; (wins:=wins+1) if pnl>0 else (losses:=losses+1); ld=pos; pos=0; pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd); continue
            if i>0:
                bn=fast_ma[i]>slow_ma[i]; bp=fast_ma[i-1]>slow_ma[i-1]; cu=bn and not bp; cd=not bn and bp
                if (pos==1 and cd) or (pos==-1 and cu):
                    pnl=(px-epx)/epx*psz*pos-psz*0.0004; cap+=pnl; rev_c+=1
                    if pnl>0: wins+=1
                    else: losses+=1
                    ld=pos; pos=0; pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd)
        if i<1: continue
        bn=fast_ma[i]>slow_ma[i]; bp=fast_ma[i-1]>slow_ma[i-1]; cu=bn and not bp; cd=not bn and bp
        if pos==0:
            if cu: watching=1; ws=i
            elif cd: watching=-1; ws=i
            if watching!=0 and i>ws:
                if i-ws>window: watching=0; continue  # WINDOW CHANGE
                if watching==1 and cd: watching=-1; ws=i; continue
                elif watching==-1 and cu: watching=1; ws=i; continue
                if watching==ld: continue
                if adx[i]<30: continue
                if i>=6 and adx[i]<=adx[i-6]: continue
                if rsi[i]<cfg['rsi_min'] or rsi[i]>cfg['rsi_max']: continue
                if slow_ma[i]>0:
                    gap=abs(fast_ma[i]-slow_ma[i])/slow_ma[i]*100
                    if gap<0.2: continue
                if ms>0 and (cap-ms)/ms<=-0.20: watching=0; continue
                if cap<=0: continue
                mg=cap*0.35; psz=mg*10; cap-=psz*0.0004; pos=watching; epx=px; ton=False; thi=px; tlo=px
                if pos==1: slp=epx*0.97
                else: slp=epx*1.03
                pk=max(pk,cap); watching=0
        pk=max(pk,cap); dd=(pk-cap)/pk if pk>0 else 0; mdd=max(mdd,dd)
        if cap<=0: break
    if pos!=0 and cap>0:
        px=c[n-1]; pnl=(px-epx)/epx*psz*pos-psz*0.0004; cap+=pnl
    tc=sl_c+tsl_c+rev_c+(1 if pos!=0 else 0)
    return {'engine':f'E6: Window {window}','bal':round(cap,2),'ret':round((cap-cfg['cap'])/cfg['cap']*100,0),'trades':tc,'wins':wins,'losses':losses,'mdd':round(mdd*100,1),'sl':sl_c,'tsl':tsl_c,'rev':rev_c}


# ═══ RUN ═══
configs = [
    ('v32.2', ema100, ema600, {'cap':5000,'ta':12,'tsl':9,'rsi_min':40,'rsi_max':80,'rsi_p':10}, rsi10),
    ('v32.3', ema75, sma750, {'cap':5000,'ta':12,'tsl':9,'rsi_min':40,'rsi_max':80,'rsi_p':11}, rsi11),
]

for ver, fast, slow, cfg, rsi_v in configs:
    print(f'\n{"="*130}', flush=True)
    print(f'  [{ver}] 6 Engine Cross Verification', flush=True)
    print(f'  Target: {ver}', flush=True)
    if ver == 'v32.2':
        print(f'  Expected: $24,073,329 | PF 5.8 | MDD 43.5% | 70T | 33W/37L | SL:30 TSL:17 REV:23', flush=True)
    else:
        print(f'  Expected: $13,236,537 | PF 3.2 | MDD 43.5% | 69T | 30W/39L | SL:33 TSL:19 REV:17', flush=True)
    print(f'{"="*130}', flush=True)
    print(f'| {"Engine":^26} | {"Bal":^16} | {"Ret%":^10} | {"T":^3} | {"W/L":^7} | {"MDD":^6} | {"SL":^3} | {"TSL":^3} | {"REV":^3} | {"Match":^5} |', flush=True)
    print(f'|{"-"*28}|{"-"*18}|{"-"*12}|{"-"*5}|{"-"*9}|{"-"*8}|{"-"*5}|{"-"*5}|{"-"*5}|{"-"*7}|', flush=True)

    results = make_engines(fast, slow, rsi_v, cfg)
    for r in results:
        if ver == 'v32.2':
            match = 'O' if abs(r['trades']-70)<=1 and abs(r['mdd']-43.5)<=2 else 'X'
        else:
            match = 'O' if abs(r['trades']-69)<=1 and abs(r['mdd']-43.5)<=2 else 'X'
        wl = f'{r.get("wins",0)}/{r.get("losses",0)}'
        print(f'| {r["engine"]:^26} | ${r["bal"]:>14,.2f} | {r["ret"]:>+9,.0f}% | {r["trades"]:>3} | {wl:^7} | {r["mdd"]:>5.1f}% | {r["sl"]:>3} | {r["tsl"]:>3} | {r["rev"]:>3} | {match:^5} |', flush=True)

print('\nDONE.', flush=True)
