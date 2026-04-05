"""
BTC/USDT 선물 자동매매 백테스트 엔진 v3
bt_fast.py _bt_core() 로직 1:1 재현

v2 대비 수정사항:
1. ML 검사: equity 대신 실현잔액(bal) 사용, pos==0일 때만 검사
2. v15.2 지연진입: "6캔들 내 -0.1%~-2.5% 풀백" 조건 (bt_fast 동일)
3. MDD: 실현잔액 기준 (거래 청산 시점에만 업데이트)
4. peak_bal 업데이트 타이밍: 진입 시 + 루프 끝
5. 역신호 시 ML 재검사 안 함 (mp 플래그만 사용)
"""
import numpy as np, pandas as pd, json, os

# ============================================================
# 데이터 & 지표 (v2와 동일)
# ============================================================
def load_5m(d):
    parts = []
    for i in range(1, 4):
        f = os.path.join(d, f'btc_usdt_5m_2020_to_now_part{i}.csv')
        print(f"  Loading {os.path.basename(f)} ...")
        parts.append(pd.read_csv(f, parse_dates=['timestamp']))
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    print(f"  5m: {len(df):,}행")
    return df

def resample_30m(df):
    r = df.set_index('timestamp').resample('30min').agg(
        {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
    print(f"  30m: {len(r):,}행")
    return r

def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()
def calc_adx(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    up = h - h.shift(1); dn = l.shift(1) - l
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = 1.0 / p
    atr = pd.Series(tr, index=c.index).ewm(alpha=a, min_periods=p, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=c.index).ewm(alpha=a, min_periods=p, adjust=False).mean() / atr
    mdi = 100 * pd.Series(mdm, index=c.index).ewm(alpha=a, min_periods=p, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=a, min_periods=p, adjust=False).mean()
def calc_rsi(c, p=14):
    d = c.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    a = 1.0 / p
    return 100 - 100/(1 + g.ewm(alpha=a, min_periods=p, adjust=False).mean() /
                        l.ewm(alpha=a, min_periods=p, adjust=False).mean().replace(0, np.nan))

def prep_df(df, ef, es):
    df = df.copy()
    df['ef'] = calc_ema(df['close'], ef)
    df['es'] = calc_ema(df['close'], es)
    df['adx'] = calc_adx(df['high'], df['low'], df['close'])
    df['rsi'] = calc_rsi(df['close'])
    df['ym'] = df['timestamp'].dt.year * 100 + df['timestamp'].dt.month
    return df

# ============================================================
# 백테스트 엔진 v3 — bt_fast._bt_core() 1:1 재현
# ============================================================
def bt_core(cfg, df):
    N = len(df)
    closes = df['close'].values.astype(np.float64)
    highs  = df['high'].values.astype(np.float64)
    lows   = df['low'].values.astype(np.float64)
    ma_f   = df['ef'].values.astype(np.float64)
    ma_s   = df['es'].values.astype(np.float64)
    rsi_v  = df['rsi'].values.astype(np.float64)
    adx_v  = df['adx'].values.astype(np.float64)
    months = df['ym'].values.astype(np.int64)
    ts     = df['timestamp'].values

    # 설정
    adx_min  = cfg['adx_min'];  rsi_min = cfg['rsi_min'];  rsi_max = cfg['rsi_max']
    sl_pct   = cfg['sl_pct'];   trail_act = cfg['trail_activate']; trail_pct = cfg['trail_pct']
    lev      = cfg['leverage'];  mg_n = cfg['margin'];  mg_r = cfg.get('margin_reduced', mg_n)
    fee      = cfg['fee_rate']
    ml_limit = cfg.get('monthly_loss_limit', 0.0)
    cp_pause = cfg.get('consec_loss_pause', 0)
    cp_cndl  = cfg.get('pause_duration_candles', 0)
    dd_thr   = cfg.get('dd_threshold', 0.0)
    delayed  = cfg.get('delayed_entry', False)
    dly_max  = cfg.get('delay_max_candles', 6)
    dly_pmin = cfg.get('delay_price_min', -0.001)
    dly_pmax = cfg.get('delay_price_max', -0.025)
    liq_d    = 1.0 / lev
    init_cap = 3000.0

    # 상태 (bt_fast 변수명 동일)
    bal = init_cap; peak_bal = bal; pos = 0; ep = 0.0; su = 0.0; ppnl = 0.0
    trail = False; rem = 1.0
    msb = bal; cm = 0; mp = False; cl = 0; pu_idx = 0
    psig = 0; sprice = 0.0; sidx = 0
    rpeak = init_cap; mdd = 0.0
    tot = 0; wc = 0; lc = 0; gp = 0.0; gl = 0.0
    slc = 0; tslc = 0; revc = 0; flc = 0
    lm = 0  # 손실 월 수

    # 연도별 추적
    yr_data = {}

    # 거래 로그
    trades_log = []

    def log_trade(i, reason, pnl_val, side):
        trades_log.append({
            'ts': str(ts[i])[:19], 'side': 'L' if side == 1 else 'S',
            'ep': round(ep, 1), 'xp': round(closes[i], 1),
            'reason': reason, 'pnl': round(pnl_val, 2), 'bal': round(bal + pnl_val, 2),
        })

    for i in range(1, N):
        cp_ = closes[i]; hp = highs[i]; lp = lows[i]
        if np.isnan(ma_f[i]) or np.isnan(ma_s[i]) or np.isnan(rsi_v[i]) or np.isnan(adx_v[i]):
            continue

        mk = months[i]

        # --- 월 초기화 ---
        if mk != cm:
            if cm != 0 and bal < msb:
                lm += 1
            cm = mk; msb = bal; mp = False

        # --- 연도 추적 ---
        yk = mk // 100
        if yk not in yr_data:
            yr_data[yk] = {'sb': bal, 'eb': bal}
        yr_data[yk]['eb'] = bal

        # ========== 포지션 보유 중 ==========
        if pos != 0:
            if pos == 1:
                pnl = (cp_ - ep) / ep
                pkc = (hp - ep) / ep
                lwc = (lp - ep) / ep
            else:
                pnl = (ep - cp_) / ep
                pkc = (ep - lp) / ep
                lwc = (ep - hp) / ep
            if pkc > ppnl:
                ppnl = pkc

            # (1) 강제청산
            if lwc <= -liq_d:
                pu2 = su * rem * (-liq_d) - su * rem * fee
                log_trade(i, 'FL', pu2, pos)
                bal += pu2
                if bal < 0: bal = 0
                tot += 1; lc += 1; gl += abs(pu2); flc += 1
                pos = 0; cl += 1
                if cp_pause > 0 and cl >= cp_pause:
                    pu_idx = i + cp_cndl
                if bal > rpeak: rpeak = bal
                dd = (rpeak - bal) / rpeak if rpeak > 0 else 0
                if dd > mdd: mdd = dd
                continue

            # (2) SL
            sth = sl_pct
            if lwc <= -sth:
                pu2 = su * rem * (-sth) - su * rem * fee
                log_trade(i, 'SL', pu2, pos)
                bal += pu2
                if bal < 0: bal = 0
                tot += 1; lc += 1; gl += abs(pu2); slc += 1
                pos = 0; cl += 1
                if cp_pause > 0 and cl >= cp_pause:
                    pu_idx = i + cp_cndl
                if bal > rpeak: rpeak = bal
                dd = (rpeak - bal) / rpeak if rpeak > 0 else 0
                if dd > mdd: mdd = dd
                continue

            # (3) 트레일링
            if ppnl >= trail_act:
                trail = True
            if trail:
                tw = trail_pct
                tl = ppnl - tw
                if pnl <= tl:
                    pu2 = su * rem * tl - su * rem * fee
                    log_trade(i, 'TSL', pu2, pos)
                    bal += pu2; tot += 1; tslc += 1
                    if tl > 0:
                        wc += 1; gp += pu2; cl = 0
                    else:
                        lc += 1; gl += abs(pu2); cl += 1
                    if cl > 0 and cp_pause > 0 and cl >= cp_pause:
                        pu_idx = i + cp_cndl
                    pos = 0
                    if bal > rpeak: rpeak = bal
                    dd = (rpeak - bal) / rpeak if rpeak > 0 else 0
                    if dd > mdd: mdd = dd
                    continue

            # (4) 역신호
            cu = ma_f[i] > ma_s[i] and ma_f[i-1] <= ma_s[i-1]
            cd = ma_f[i] < ma_s[i] and ma_f[i-1] >= ma_s[i-1]
            ao = adx_v[i] >= adx_min
            ro = rsi_min <= rsi_v[i] <= rsi_max
            rv = False; nd = 0
            if pos == 1 and cd and ao and ro:
                rv = True; nd = -1
            elif pos == -1 and cu and ao and ro:
                rv = True; nd = 1
            if rv:
                pu2 = su * rem * pnl - su * rem * fee
                log_trade(i, 'REV', pu2, pos)
                bal += pu2; tot += 1; revc += 1
                if pnl > 0:
                    wc += 1; gp += pu2; cl = 0
                else:
                    lc += 1; gl += abs(pu2); cl += 1
                if cl > 0 and cp_pause > 0 and cl >= cp_pause:
                    pu_idx = i + cp_cndl
                pos = 0
                if bal > rpeak: rpeak = bal
                dd = (rpeak - bal) / rpeak if rpeak > 0 else 0
                if dd > mdd: mdd = dd
                # 역신호 즉시 재진입 (bt_fast 동일)
                can = (not mp) and (i >= pu_idx)
                if bal > 10 and can:
                    mg = mg_n
                    if peak_bal > 0 and dd_thr < 0:
                        dn = (peak_bal - bal) / peak_bal
                        if dn > abs(dd_thr):
                            mg = mg_r
                    mu = bal * mg; s2 = mu * lev
                    bal -= s2 * fee
                    pos = nd; ep = cp_; su = s2; ppnl = 0.0; trail = False; rem = 1.0
                    if bal > peak_bal: peak_bal = bal
                continue

        # ========== 포지션 없음 ==========
        if pos == 0 and bal > 10:
            # ML 검사: 실현잔액 기준 (bt_fast 동일)
            if ml_limit < 0 and msb > 0:
                mr = (bal - msb) / msb
                if mr < ml_limit:
                    mp = True
            can = (not mp) and (i >= pu_idx)
            if not can:
                psig = 0
                continue

            cu = ma_f[i] > ma_s[i] and ma_f[i-1] <= ma_s[i-1]
            cd = ma_f[i] < ma_s[i] and ma_f[i-1] >= ma_s[i-1]
            ao = adx_v[i] >= adx_min
            ro = rsi_min <= rsi_v[i] <= rsi_max
            sig = 0
            if cu and ao and ro: sig = 1
            elif cd and ao and ro: sig = -1

            do = False; ed = 0
            if sig != 0:
                if delayed:
                    psig = sig; sprice = cp_; sidx = i
                else:
                    do = True; ed = sig
            elif psig != 0 and delayed:
                el = i - sidx
                if el > dly_max:
                    psig = 0
                else:
                    pc = (cp_ - sprice) / sprice
                    if psig == 1:
                        if dly_pmax <= pc <= dly_pmin:
                            do = True; ed = 1; psig = 0
                    elif psig == -1:
                        iv = -pc
                        if dly_pmax <= iv <= dly_pmin:
                            do = True; ed = -1; psig = 0

            if do:
                mg = mg_n
                if peak_bal > 0 and dd_thr < 0:
                    dn = (peak_bal - bal) / peak_bal
                    if dn > abs(dd_thr):
                        mg = mg_r
                mu = bal * mg; s2 = mu * lev
                bal -= s2 * fee
                pos = ed; ep = cp_; su = s2; ppnl = 0.0; trail = False; rem = 1.0
                if bal > peak_bal: peak_bal = bal

        # 루프 끝 peak_bal/rpeak 업데이트 (bt_fast 동일)
        if bal > peak_bal: peak_bal = bal
        if bal > rpeak: rpeak = bal

    # 마지막 월
    if cm != 0 and bal < msb:
        lm += 1
    # 잔여 포지션 청산
    if pos != 0 and N > 0:
        if pos == 1: pf_ = (closes[N-1] - ep) / ep
        else: pf_ = (ep - closes[N-1]) / ep
        pu2 = su * rem * pf_ - su * rem * fee
        bal += pu2; tot += 1
        if pf_ > 0: wc += 1; gp += pu2
        else: lc += 1; gl += abs(pu2)
    # 마지막 연도
    if yr_data:
        last_yk = max(yr_data.keys())
        yr_data[last_yk]['eb'] = bal

    pf = gp / gl if gl > 0 else (gp if gp > 0 else 0)
    wr = wc / tot * 100 if tot > 0 else 0

    yearly = {}
    for yk, yd in sorted(yr_data.items()):
        if yd['sb'] > 0:
            yearly[str(yk)] = round((yd['eb'] - yd['sb']) / yd['sb'] * 100, 1)

    return {
        'bal': round(bal, 0), 'ret': round((bal - init_cap) / init_cap * 100, 1),
        'trades': tot, 'wr': round(wr, 1), 'pf': round(pf, 2),
        'mdd': round(mdd * 100, 1), 'sl': slc, 'tsl': tslc, 'rev': revc, 'fl': flc,
        'lm': lm, 'yr': yearly, 'trades_log': trades_log,
    }


# ============================================================
# 설정 (bt_fast 파라미터 정확히 매핑)
# ============================================================
CONFIGS = {
    'v13.5': {
        'name': 'v13.5', 'timeframe': '5m',
        'ema_fast': 7, 'ema_slow': 100,
        'adx_min': 30, 'rsi_min': 30, 'rsi_max': 58,
        'sl_pct': 0.07, 'trail_activate': 0.08, 'trail_pct': 0.06,
        'leverage': 10, 'margin': 0.20, 'margin_reduced': 0.10,
        'fee_rate': 0.0004,
        'monthly_loss_limit': -0.20,
        'consec_loss_pause': 3, 'pause_duration_candles': 288,
        'dd_threshold': -0.50,
        'delayed_entry': False,
    },
    'v14.4': {
        'name': 'v14.4', 'timeframe': '30m',
        'ema_fast': 3, 'ema_slow': 200,
        'adx_min': 35, 'rsi_min': 30, 'rsi_max': 65,
        'sl_pct': 0.07, 'trail_activate': 0.06, 'trail_pct': 0.03,
        'leverage': 10, 'margin': 0.25,
        'fee_rate': 0.0004,
        'monthly_loss_limit': -0.20,
    },
    'v15.2': {
        'name': 'v15.2', 'timeframe': '30m',
        'ema_fast': 3, 'ema_slow': 200,
        'adx_min': 35, 'rsi_min': 30, 'rsi_max': 65,
        'sl_pct': 0.05, 'trail_activate': 0.06, 'trail_pct': 0.05,
        'leverage': 10, 'margin': 0.30,
        'fee_rate': 0.0004,
        'monthly_loss_limit': -0.15,
        'delayed_entry': True,
        'delay_max_candles': 6,
        'delay_price_min': -0.001,  # -0.1% (풀백 최소)
        'delay_price_max': -0.025,  # -2.5% (풀백 최대)
    },
    'v15.4': {
        'name': 'v15.4', 'timeframe': '30m',
        'ema_fast': 3, 'ema_slow': 200,
        'adx_min': 35, 'rsi_min': 30, 'rsi_max': 65,
        'sl_pct': 0.07, 'trail_activate': 0.06, 'trail_pct': 0.03,
        'leverage': 10, 'margin': 0.40,
        'fee_rate': 0.0004,
        'monthly_loss_limit': -0.30,
    },
}

EXPECTED = {
    'v13.5': {'bal':468530,'ret':15518,'tr':313,'pf':6.87,'mdd':74.3,'fl':1,'wr':44.1,'sl':None,'tsl':None,'rev':None},
    'v14.4': {'bal':837212,'ret':27807,'tr':105,'pf':2.04,'mdd':36.9,'fl':0,'wr':45.7,'sl':11,'tsl':47,'rev':47},
    'v15.2': {'bal':243482,'ret':8016,'tr':66,'pf':2.48,'mdd':27.6,'fl':0,'wr':62.1,'sl':None,'tsl':None,'rev':None},
    'v15.4': {'bal':8717659,'ret':290489,'tr':105,'pf':1.65,'mdd':54.2,'fl':0,'wr':45.7,'sl':11,'tsl':47,'rev':47},
}


# ============================================================
# 출력
# ============================================================
def show(name, r, e):
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")
    def row(label, rv, ev, fmt, unit=''):
        rs = f"{rv:{fmt}}{unit}"
        es = f"{ev:{fmt}}{unit}" if ev is not None else '-'
        if ev and ev != 0:
            err = (rv / ev - 1) * 100 if isinstance(ev, (int,float)) else 0
            errs = f"{err:+.1f}%"
        else:
            errs = '-'
        print(f"  {label:<16} {rs:>18} {es:>18} {errs:>10}")

    print(f"  {'항목':<16} {'백테스트':>18} {'기획서':>18} {'오차':>10}")
    print(f"  {'-'*58}")
    row('최종 잔액($)', r['bal'], e['bal'], ',.0f')
    row('수익률(%)', r['ret'], e['ret'], ',.1f')
    row('거래 수', r['trades'], e['tr'], 'd')
    row('승률(%)', r['wr'], e['wr'], '.1f')
    row('PF', r['pf'], e['pf'], '.2f')
    row('MDD(%)', r['mdd'], e['mdd'], '.1f')
    row('FL', r['fl'], e['fl'], 'd')
    print(f"  {'SL/TSL/REV':<16} {r['sl']:>5}/{r['tsl']:>5}/{r['rev']:>5}", end='')
    if e['sl'] is not None:
        print(f"     {e['sl']:>5}/{e['tsl']:>5}/{e['rev']:>5}")
    else:
        print()

    print(f"\n  연도별:")
    for y in sorted(r['yr'].keys()):
        print(f"    {y}: {r['yr'][y]:>+8.1f}%")


def show_comparison(results):
    names = ['v13.5', 'v14.4', 'v15.2', 'v15.4']
    print(f"\n{'='*90}")
    print(f"  4버전 비교 (v3 vs 기획서)")
    print(f"{'='*90}")
    print(f"\n  {'버전':<8} {'잔액 오차':>12} {'거래수':>10} {'PF':>8} {'MDD':>10} {'FL':>6}")
    print(f"  {'-'*56}")
    for n in names:
        r = results[n]; e = EXPECTED[n]
        berr = (r['bal'] / e['bal'] - 1) * 100
        terr = r['trades'] - e['tr']
        perr = r['pf'] - e['pf']
        merr = r['mdd'] - e['mdd']
        ferr = r['fl'] - e['fl']
        print(f"  {n:<8} {berr:>+11.1f}% {terr:>+9d} {perr:>+7.2f} {merr:>+9.1f}%p {ferr:>+5d}")


# ============================================================
# 메인
# ============================================================
def main():
    base = r'D:\filesystem\futures\btc_V1\test4'
    print("=" * 72)
    print("  BTC/USDT 4버전 백테스트 v3 (bt_fast.py 1:1 재현)")
    print("=" * 72)

    print("\n[1] 데이터 로드...")
    df5 = load_5m(base)
    print("[2] 30m 리샘플링...")
    df30 = resample_30m(df5)

    print("\n[3] 백테스트 실행...")
    results = {}
    for name in ['v13.5', 'v14.4', 'v15.2', 'v15.4']:
        cfg = CONFIGS[name]
        df = (df5 if cfg['timeframe'] == '5m' else df30).copy()
        df = prep_df(df, cfg['ema_fast'], cfg['ema_slow'])
        # NaN 제거 (bt_fast는 NaN skip)
        r = bt_core(cfg, df)
        results[name] = r
        show(name, r, EXPECTED[name])

    show_comparison(results)

    # 저장
    out = {}
    for n, r in results.items():
        out[n] = {k: v for k, v in r.items() if k != 'trades_log'}
    outf = os.path.join(base, 'backtest_v3_results.json')
    with open(outf, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {outf}")
    print("완료!")


if __name__ == '__main__':
    main()
