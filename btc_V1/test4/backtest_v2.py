"""
BTC/USDT 선물 자동매매 백테스트 엔진 v2
bt_fast.py 엔진 로직과 정확히 동일하게 재구현

핵심 수정사항 (v1 대비):
1. TSL 청산: close가 아닌 trailing threshold PnL (ppnl - trail_pct) 사용
2. SL 청산: SL 가격 기준 PnL 사용 (변경 없음)
3. REV 청산: close 기준 PnL 사용 (변경 없음)
4. FL 청산: 증거금 전액 손실 (변경 없음)
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


# ============================================================
# 1. 데이터 로드 & 리샘플링
# ============================================================
def load_5m_data(base_dir):
    files = [os.path.join(base_dir, f'btc_usdt_5m_2020_to_now_part{i}.csv') for i in range(1, 4)]
    dfs = []
    for f in files:
        print(f"  Loading {os.path.basename(f)} ...")
        dfs.append(pd.read_csv(f, parse_dates=['timestamp']))
    df = pd.concat(dfs, ignore_index=True)
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  총 {len(df):,}행 ({df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]})")
    return df


def resample_30m(df5):
    df = df5.set_index('timestamp')
    r = df.resample('30min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    r.reset_index(inplace=True)
    print(f"  30분봉 리샘플링: {len(r):,}행")
    return r


# ============================================================
# 2. 지표 계산
# ============================================================
def calc_ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def calc_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    a = 1.0 / period
    atr = tr.ewm(alpha=a, min_periods=period, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=a, min_periods=period, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=a, min_periods=period, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    dx = dx.fillna(0)
    return dx.ewm(alpha=a, min_periods=period, adjust=False).mean()

def calc_rsi(s, period=14):
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = (-d).where(d < 0, 0.0)
    a = 1.0 / period
    ag = g.ewm(alpha=a, min_periods=period, adjust=False).mean()
    al = l.ewm(alpha=a, min_periods=period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def add_indicators(df, ema_f, ema_s):
    df = df.copy()
    df['ef'] = calc_ema(df['close'], ema_f)
    df['es'] = calc_ema(df['close'], ema_s)
    df['adx'] = calc_adx(df['high'], df['low'], df['close'])
    df['rsi'] = calc_rsi(df['close'])
    return df


# ============================================================
# 3. 백테스트 엔진 v2 (bt_fast.py 로직 충실 재현)
# ============================================================
def run_backtest(cfg, df):
    """
    bt_fast.py와 동일한 로직:
    - SL/FL: 캔들 high/low로 체크, SL 가격에서 청산
    - TSL: close로 PnL 체크, trailing threshold PnL에서 청산
    - REV: close로 크로스 체크, close에서 청산
    """
    N = len(df)
    ts = df['timestamp'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    ef = df['ef'].values
    es = df['es'].values
    adx = df['adx'].values
    rsi = df['rsi'].values

    # 설정
    adx_min = cfg['adx_min']
    rsi_min = cfg['rsi_min']
    rsi_max = cfg['rsi_max']
    sl_pct = cfg['sl_pct']
    trail_act = cfg['trail_activate']
    trail_w = cfg['trail_pct']
    lev = cfg['leverage']
    mg = cfg['margin']
    mg_red = cfg.get('margin_reduced', mg)
    fee = cfg['fee_rate']
    ml_limit = cfg.get('monthly_loss_limit', None)
    cp_limit = cfg.get('consec_loss_pause', 0)
    cp_candles = cfg.get('pause_duration_candles', 288)
    dd_thresh = cfg.get('dd_threshold', None)
    delay = cfg.get('entry_delay_candles', 0)
    liq_d = 1.0 / lev  # 강제청산 거리 (10% for 10x)

    # 상태
    bal = 3000.0
    pos = 0  # 0=없음, 1=long, -1=short
    ep = 0.0   # entry price
    su = 0.0   # position size (value)
    margin_used = 0.0
    ppnl = 0.0  # peak PnL (%)
    tsl_active = False

    peak_bal = 3000.0
    mdd = 0.0

    # 보호 상태
    month_key = ''
    month_sb = 3000.0  # month start balance
    ml_paused = False
    consec_loss = 0
    pause_until = -1
    dd_active = False

    # 지연진입
    pending = 0      # 0=없음, 1=long대기, -1=short대기
    pending_idx = -1  # 신호 발생 캔들

    # 결과
    trades = []
    monthly = {}
    fl_count = 0
    sl_count = 0
    tsl_count = 0
    rev_count = 0

    def record_trade(exit_idx, reason, pnl_val, side):
        nonlocal bal, fl_count, sl_count, tsl_count, rev_count, consec_loss, pause_until
        bal += pnl_val
        if bal < 1.0:
            bal = 1.0
        trades.append({
            'exit_ts': str(ts[exit_idx])[:19],
            'side': 'long' if side == 1 else 'short',
            'entry_price': round(ep, 2),
            'reason': reason,
            'pnl': round(pnl_val, 2),
            'balance': round(bal, 2),
        })
        if reason == 'FL':
            fl_count += 1
        elif reason == 'SL':
            sl_count += 1
        elif reason == 'TSL':
            tsl_count += 1
        elif reason == 'REV':
            rev_count += 1

        # 연패 추적
        if pnl_val < 0:
            consec_loss += 1
            if cp_limit > 0 and consec_loss >= cp_limit:
                pause_until = exit_idx + cp_candles
                consec_loss = 0
        else:
            consec_loss = 0

    def get_margin():
        """현재 마진율 (DD 축소 포함)"""
        nonlocal dd_active
        if dd_thresh is not None and peak_bal > 0:
            dd_ratio = (peak_bal - bal) / peak_bal
            if dd_ratio >= abs(dd_thresh):
                dd_active = True
                return mg_red
            else:
                dd_active = False
        return mg

    def can_enter(i):
        """보호 메커니즘 체크"""
        if ml_paused:
            return False
        if cp_limit > 0 and i < pause_until:
            return False
        return True

    def enter(i, side):
        """진입"""
        nonlocal pos, ep, su, margin_used, ppnl, tsl_active, bal, pending, pending_idx
        m = get_margin()
        ep = closes[i]
        su = bal * m * lev
        margin_used = bal * m
        entry_fee = su * fee
        bal -= entry_fee
        pos = side
        ppnl = 0.0
        tsl_active = False
        pending = 0
        pending_idx = -1

    # ========== 메인 루프 ==========
    for i in range(1, N):
        t = ts[i]
        mk = str(t)[:7]  # YYYY-MM

        # 월 초기화
        if mk != month_key:
            if month_key and month_key in monthly:
                monthly[month_key]['eb'] = bal
            month_key = mk
            month_sb = bal
            ml_paused = False
            monthly[mk] = {'sb': bal, 'eb': bal}

        # 월간 손실 체크 (매 캔들)
        if ml_limit is not None and month_sb > 0:
            # 실현 잔액 기준 (포지션 없을 때) + 미실현 포함
            if pos == 0:
                cur_eq = bal
            elif pos == 1:
                cur_eq = bal + (closes[i] - ep) / ep * su
            else:
                cur_eq = bal + (ep - closes[i]) / ep * su
            mr = (cur_eq - month_sb) / month_sb
            if mr <= ml_limit:
                ml_paused = True

        # ---- 포지션 보유 중 ----
        if pos != 0:
            h = highs[i]
            l = lows[i]
            c = closes[i]

            if pos == 1:  # LONG
                # 캔들 high/low 기반 PnL
                hwc = (h - ep) / ep   # high worst case (최고 수익)
                lwc = (l - ep) / ep   # low worst case (최저 수익)
                cpnl = (c - ep) / ep  # close PnL

                # (1) 강제청산 (low 기준)
                if lwc <= -liq_d:
                    record_trade(i, 'FL', -margin_used, pos)
                    pos = 0
                    continue

                # (2) SL (low 기준)
                if lwc <= -sl_pct:
                    sl_pnl = su * (-sl_pct) - su * fee
                    record_trade(i, 'SL', sl_pnl, pos)
                    pos = 0
                    continue

                # peak PnL 업데이트 (high 기준)
                if hwc > ppnl:
                    ppnl = hwc

                # (3) 트레일링 (close 기준 체크, threshold PnL로 청산)
                if ppnl >= trail_act:
                    tsl_active = True
                if tsl_active:
                    tl = ppnl - trail_w  # trailing threshold
                    if cpnl <= tl:
                        tsl_pnl = su * tl - su * fee
                        record_trade(i, 'TSL', tsl_pnl, pos)
                        pos = 0
                        continue

                # (4) 역신호 (close 기준)
                if i > 0:
                    cross_dn = (ef[i] < es[i]) and (ef[i-1] >= es[i-1])
                    if cross_dn and adx[i] >= adx_min and rsi_min <= rsi[i] <= rsi_max:
                        rev_pnl = su * cpnl - su * fee
                        record_trade(i, 'REV', rev_pnl, 1)
                        pos = 0
                        # 반대 진입
                        if can_enter(i):
                            if delay > 0:
                                pending = -1
                                pending_idx = i
                            else:
                                enter(i, -1)
                        continue

            else:  # SHORT (pos == -1)
                hwc = (ep - l) / ep   # high PnL (short에서 low가 유리)
                lwc = (ep - h) / ep   # low PnL (short에서 high가 불리)
                cpnl = (ep - c) / ep  # close PnL

                # (1) 강제청산 (high 기준)
                if lwc <= -liq_d:  # lwc = (ep - h) / ep, 음수면 h > ep*(1+liq_d)
                    record_trade(i, 'FL', -margin_used, pos)
                    pos = 0
                    continue

                # (2) SL (high 기준)
                if lwc <= -sl_pct:
                    sl_pnl = su * (-sl_pct) - su * fee
                    record_trade(i, 'SL', sl_pnl, pos)
                    pos = 0
                    continue

                # peak PnL 업데이트 (low가 유리한 방향)
                if hwc > ppnl:
                    ppnl = hwc

                # (3) 트레일링
                if ppnl >= trail_act:
                    tsl_active = True
                if tsl_active:
                    tl = ppnl - trail_w
                    if cpnl <= tl:
                        tsl_pnl = su * tl - su * fee
                        record_trade(i, 'TSL', tsl_pnl, pos)
                        pos = 0
                        continue

                # (4) 역신호
                if i > 0:
                    cross_up = (ef[i] > es[i]) and (ef[i-1] <= es[i-1])
                    if cross_up and adx[i] >= adx_min and rsi_min <= rsi[i] <= rsi_max:
                        rev_pnl = su * cpnl - su * fee
                        record_trade(i, 'REV', rev_pnl, -1)
                        pos = 0
                        if can_enter(i):
                            if delay > 0:
                                pending = 1
                                pending_idx = i
                            else:
                                enter(i, 1)
                        continue

        # ---- 포지션 없을 때 진입 체크 ----
        if pos == 0 and can_enter(i) and i > 0:
            # 지연진입 처리
            if pending != 0 and delay > 0:
                if i >= pending_idx + delay:
                    # 추세 유지 확인
                    if pending == 1 and ef[i] > es[i]:
                        enter(i, 1)
                    elif pending == -1 and ef[i] < es[i]:
                        enter(i, -1)
                    else:
                        pending = 0
                        pending_idx = -1
            elif pending == 0:
                # 새 크로스 체크
                cross_up = (ef[i] > es[i]) and (ef[i-1] <= es[i-1])
                cross_dn = (ef[i] < es[i]) and (ef[i-1] >= es[i-1])
                if cross_up and adx[i] >= adx_min and rsi_min <= rsi[i] <= rsi_max:
                    if delay > 0:
                        pending = 1
                        pending_idx = i
                    else:
                        enter(i, 1)
                elif cross_dn and adx[i] >= adx_min and rsi_min <= rsi[i] <= rsi_max:
                    if delay > 0:
                        pending = -1
                        pending_idx = i
                    else:
                        enter(i, -1)

        # peak balance / MDD
        eq = bal
        if pos == 1:
            eq = bal + (closes[i] - ep) / ep * su
        elif pos == -1:
            eq = bal + (ep - closes[i]) / ep * su

        if eq > peak_bal:
            peak_bal = eq
        dd = (peak_bal - eq) / peak_bal if peak_bal > 0 else 0
        if dd > mdd:
            mdd = dd

        monthly[mk]['eb'] = bal

    # 마지막 포지션 (존재 시 close 기준 청산)
    if pos != 0:
        c = closes[N-1]
        if pos == 1:
            pnl_final = su * ((c - ep)/ep) - su * fee
        else:
            pnl_final = su * ((ep - c)/ep) - su * fee
        record_trade(N-1, 'END', pnl_final, pos)
        pos = 0

    # ========== 결과 집계 ==========
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    total_win = sum(t['pnl'] for t in wins)
    total_loss = abs(sum(t['pnl'] for t in losses))
    pf = total_win / total_loss if total_loss > 0 else float('inf')

    # 연도별
    yearly = {}
    for t in trades:
        y = t['exit_ts'][:4]
        yearly.setdefault(y, 0.0)
        yearly[y] += t['pnl']

    # 월별 수익/손실 월 수
    profit_months = sum(1 for v in monthly.values() if v['eb'] > v['sb'])
    loss_months = sum(1 for v in monthly.values() if v['eb'] < v['sb'])

    return {
        'final_balance': round(bal, 2),
        'return_pct': round((bal / 3000 - 1) * 100, 1),
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(len(wins)/len(trades)*100, 1) if trades else 0,
        'pf': round(pf, 2),
        'mdd': round(mdd * 100, 1),
        'fl_count': fl_count,
        'sl_count': sl_count,
        'tsl_count': tsl_count,
        'rev_count': rev_count,
        'profit_months': profit_months,
        'loss_months': loss_months,
        'yearly': yearly,
        'monthly': monthly,
        'trades': trades,
    }


# ============================================================
# 4. 버전별 설정
# ============================================================
CONFIGS = {
    'v13.5': {
        'name': 'v13.5',
        'timeframe': '5m',
        'ema_fast': 7, 'ema_slow': 100,
        'adx_min': 30, 'rsi_min': 30, 'rsi_max': 58,
        'sl_pct': 0.07,
        'trail_activate': 0.08, 'trail_pct': 0.06,
        'leverage': 10,
        'margin': 0.20, 'margin_reduced': 0.10,
        'fee_rate': 0.0004,
        'monthly_loss_limit': -0.20,
        'consec_loss_pause': 3, 'pause_duration_candles': 288,
        'dd_threshold': -0.50,
    },
    'v14.4': {
        'name': 'v14.4',
        'timeframe': '30m',
        'ema_fast': 3, 'ema_slow': 200,
        'adx_min': 35, 'rsi_min': 30, 'rsi_max': 65,
        'sl_pct': 0.07,
        'trail_activate': 0.06, 'trail_pct': 0.03,
        'leverage': 10,
        'margin': 0.25,
        'fee_rate': 0.0004,
        'monthly_loss_limit': -0.20,
    },
    'v15.2': {
        'name': 'v15.2',
        'timeframe': '30m',
        'ema_fast': 3, 'ema_slow': 200,
        'adx_min': 35, 'rsi_min': 30, 'rsi_max': 65,
        'sl_pct': 0.05,
        'trail_activate': 0.06, 'trail_pct': 0.05,
        'leverage': 10,
        'margin': 0.30,
        'fee_rate': 0.0004,
        'monthly_loss_limit': -0.15,
        'entry_delay_candles': 6,
    },
    'v15.4': {
        'name': 'v15.4',
        'timeframe': '30m',
        'ema_fast': 3, 'ema_slow': 200,
        'adx_min': 35, 'rsi_min': 30, 'rsi_max': 65,
        'sl_pct': 0.07,
        'trail_activate': 0.06, 'trail_pct': 0.03,
        'leverage': 10,
        'margin': 0.40,
        'fee_rate': 0.0004,
        'monthly_loss_limit': -0.30,
    },
}

# 기획서 기대값
EXPECTED = {
    'v13.5': {'balance': 468530, 'return': 15518, 'trades': 313, 'pf': 6.87, 'mdd': 74.3, 'fl': 1, 'wr': 44.1},
    'v14.4': {'balance': 837212, 'return': 27807, 'trades': 105, 'pf': 2.04, 'mdd': 36.9, 'fl': 0, 'wr': 45.7},
    'v15.2': {'balance': 243482, 'return': 8016, 'trades': 66, 'pf': 2.48, 'mdd': 27.6, 'fl': 0, 'wr': 62.1},
    'v15.4': {'balance': 8717659, 'return': 290489, 'trades': 105, 'pf': 1.65, 'mdd': 54.2, 'fl': 0, 'wr': 45.7},
}


# ============================================================
# 5. 출력
# ============================================================
def print_result(name, r, exp):
    print(f"\n{'='*70}")
    print(f"  {name} 백테스트 결과")
    print(f"{'='*70}")

    def row(label, val, ex_val, fmt_v, fmt_e, unit=''):
        val_s = f"{val:{fmt_v}}{unit}"
        ex_s = f"{ex_val:{fmt_e}}{unit}" if ex_val is not None else '-'
        if ex_val and ex_val != 0:
            err = (val / ex_val - 1) * 100
            err_s = f"{err:+.1f}%"
        else:
            err_s = '-'
        print(f"  {label:<20} {val_s:>20} {ex_s:>20} {err_s:>12}")

    print(f"  {'항목':<20} {'백테스트':>20} {'기획서':>20} {'오차':>12}")
    print(f"  {'-'*64}")
    row('최종 잔액', r['final_balance'], exp['balance'], ',.0f', ',', ' $')
    row('수익률', r['return_pct'], exp['return'], ',.1f', ',', '%')
    row('거래 수', r['total_trades'], exp['trades'], 'd', 'd', '')
    row('승률', r['win_rate'], exp.get('wr'), '.1f', '.1f', '%')
    row('PF', r['pf'], exp['pf'], '.2f', '.2f', '')
    row('MDD', r['mdd'], exp['mdd'], '.1f', '.1f', '%')
    row('강제청산', r['fl_count'], exp['fl'], 'd', 'd', '')
    print(f"  {'SL/TSL/REV':<20} {r['sl_count']:>5}/{r['tsl_count']:>5}/{r['rev_count']:>5}")
    print(f"  {'수익월/손실월':<20} {r['profit_months']:>5}/{r['loss_months']:>5}")

    # 연도별
    print(f"\n  연도별:")
    cum = 3000.0
    for y in sorted(r['yearly'].keys()):
        pnl = r['yearly'][y]
        ret = pnl / cum * 100 if cum > 0 else 0
        cum += pnl
        print(f"    {y}: ${pnl:>+14,.0f} ({ret:>+8.1f}%)  잔액 ${cum:>14,.0f}")


def print_comparison(results):
    print(f"\n{'='*90}")
    print(f"  4개 버전 비교 (백테스트 v2 vs 기획서)")
    print(f"{'='*90}")

    names = ['v13.5', 'v14.4', 'v15.2', 'v15.4']
    print(f"\n  {'':.<14} {'v13.5':>16} {'v14.4':>16} {'v15.2':>16} {'v15.4':>16}")
    print(f"  {'-'*78}")

    def row(label, key, fmt=',.0f', prefix='', suffix=''):
        vals = []
        for n in names:
            v = results[n].get(key, 0)
            vals.append(f"{prefix}{v:{fmt}}{suffix}")
        print(f"  {label:<14} {vals[0]:>16} {vals[1]:>16} {vals[2]:>16} {vals[3]:>16}")

    row('잔액', 'final_balance', ',.0f', '$')
    row('수익률', 'return_pct', ',.1f', '', '%')
    row('거래 수', 'total_trades', 'd')
    row('승률', 'win_rate', '.1f', '', '%')
    row('PF', 'pf', '.2f')
    row('MDD', 'mdd', '.1f', '', '%')
    row('FL', 'fl_count', 'd')

    print(f"\n  기획서 대비 오차율:")
    print(f"  {'버전':<10} {'잔액':>14} {'거래수':>10} {'PF':>10} {'MDD':>12} {'FL':>8}")
    for n in names:
        r = results[n]
        e = EXPECTED[n]
        b_err = (r['final_balance'] / e['balance'] - 1) * 100
        t_err = r['total_trades'] - e['trades']
        pf_err = r['pf'] - e['pf']
        mdd_err = r['mdd'] - e['mdd']
        fl_err = r['fl_count'] - e['fl']
        print(f"  {n:<10} {b_err:>+13.1f}% {t_err:>+9d} {pf_err:>+9.2f} {mdd_err:>+11.1f}%p {fl_err:>+7d}")


# ============================================================
# 6. 메인
# ============================================================
def main():
    base_dir = r'D:\filesystem\futures\btc_V1\test4'
    print("=" * 70)
    print("  BTC/USDT 4버전 백테스트 v2 (bt_fast.py 로직 동기화)")
    print("=" * 70)

    print("\n[1] 데이터 로드...")
    df5 = load_5m_data(base_dir)
    print("\n[2] 30분봉 리샘플링...")
    df30 = resample_30m(df5)

    print("\n[3] 백테스트 실행...")
    results = {}
    for name in ['v13.5', 'v14.4', 'v15.2', 'v15.4']:
        cfg = CONFIGS[name]
        print(f"\n  --- {name} ---")

        if cfg['timeframe'] == '5m':
            df = df5.copy()
        else:
            df = df30.copy()

        df = add_indicators(df, cfg['ema_fast'], cfg['ema_slow'])
        warmup = max(cfg['ema_slow'] * 2, 300)
        df = df.iloc[warmup:].reset_index(drop=True)
        print(f"  워밍업 {warmup} 제거 → {len(df):,}행")

        r = run_backtest(cfg, df)
        results[name] = r
        print_result(name, r, EXPECTED[name])

    print_comparison(results)

    # JSON 저장
    out = {}
    for n, r in results.items():
        out[n] = {
            'final_balance': r['final_balance'],
            'return_pct': r['return_pct'],
            'total_trades': r['total_trades'],
            'wins': r['wins'], 'losses': r['losses'],
            'win_rate': r['win_rate'],
            'pf': r['pf'], 'mdd': r['mdd'],
            'fl': r['fl_count'], 'sl': r['sl_count'],
            'tsl': r['tsl_count'], 'rev': r['rev_count'],
            'yearly': {str(k): round(v,2) for k,v in r['yearly'].items()},
            'monthly': {k: {'sb': round(v['sb'],2), 'eb': round(v['eb'],2)}
                       for k,v in r['monthly'].items()},
        }
    outf = os.path.join(base_dir, 'backtest_v2_results.json')
    with open(outf, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {outf}")
    print("\n백테스트 v2 완료!")


if __name__ == '__main__':
    main()
