"""
v28 정밀 검증: Top 3 Tier 전략의 모든 거래를 날짜별로 기록
- 매 거래: 진입일시, 청산일시, 방향, 진입가, 청산가, ROI, PnL, 잔액
- 월별/연도별 정확한 수익 분배
- 2023~2026 활동 검증
"""

import numpy as np
import pandas as pd
import time
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v28_backtest_engine import (
    load_5m_data, resample_ohlcv,
    calc_ma, calc_adx_wilder, calc_rsi_wilder,
    calc_wma, calc_hma, calc_vwma, calc_ema
)

BASE = r"D:\filesystem\futures\btc_V1\test4"


def detailed_backtest(timestamps, close, high, low, volume,
                      fast_ma, slow_ma, adx, rsi,
                      adx_thresh, rsi_lo, rsi_hi,
                      sl_pct, trail_act_pct, trail_pct,
                      margin_pct, leverage,
                      entry_delay_bars, entry_offset_pct,
                      fee_rate, skip_same_dir):
    """
    모든 거래를 날짜별로 기록하는 상세 백테스트
    """
    n = len(close)
    balance = 3000.0
    peak_balance = 3000.0
    max_dd = 0.0

    position = 0
    entry_price = 0.0
    entry_idx = 0
    position_size = 0.0
    position_margin = 0.0
    highest_roi = 0.0
    trail_active = False
    last_closed_dir = 0

    pending_signal = 0
    pending_bar = 0
    pending_price = 0.0

    # 월간 손실 제한
    month_key = ""
    month_start_balance = 3000.0

    trades_log = []
    balance_history = []  # (timestamp, balance) 매 거래 후

    # 월별/연도별 추적
    monthly_pnl = {}
    yearly_balances = {}

    for i in range(1, n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            continue

        cur_price = close[i]
        ts = timestamps[i]
        cur_month = f"{ts.year}-{ts.month:02d}"
        cur_year = str(ts.year)

        # 연도 시작 잔액 기록
        if cur_year not in yearly_balances:
            yearly_balances[cur_year] = {'start': balance, 'end': balance}
        yearly_balances[cur_year]['end'] = balance

        # 월간 리셋
        if cur_month != month_key:
            month_key = cur_month
            month_start_balance = balance
            if cur_month not in monthly_pnl:
                monthly_pnl[cur_month] = {'trades': 0, 'pnl': 0.0, 'start_bal': balance, 'end_bal': balance}

        # 월간 손실 -20% 체크
        if balance < month_start_balance * 0.80:
            if position != 0:
                if position == 1:
                    roi = (cur_price - entry_price) / entry_price
                else:
                    roi = (entry_price - cur_price) / entry_price
                pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
                balance += pnl
                close_type = "ML_STOP"
                trades_log.append({
                    'entry_time': str(timestamps[entry_idx]),
                    'exit_time': str(ts),
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(cur_price, 2),
                    'roi_pct': round(roi * 100, 2),
                    'pnl': round(pnl, 2),
                    'balance_after': round(balance, 2),
                    'close_type': close_type,
                    'margin_used': round(position_margin, 2),
                    'highest_roi_pct': round(highest_roi * 100, 2)
                })
                monthly_pnl[cur_month]['trades'] += 1
                monthly_pnl[cur_month]['pnl'] += pnl
                position = 0
                trail_active = False
            continue

        # 포지션 관리
        if position != 0:
            if position == 1:
                roi = (cur_price - entry_price) / entry_price
            else:
                roi = (entry_price - cur_price) / entry_price

            # SL
            if roi <= sl_pct / 100.0:
                pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
                balance += pnl
                trades_log.append({
                    'entry_time': str(timestamps[entry_idx]),
                    'exit_time': str(ts),
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(cur_price, 2),
                    'roi_pct': round(roi * 100, 2),
                    'pnl': round(pnl, 2),
                    'balance_after': round(balance, 2),
                    'close_type': 'SL',
                    'margin_used': round(position_margin, 2),
                    'highest_roi_pct': round(highest_roi * 100, 2)
                })
                if cur_month in monthly_pnl:
                    monthly_pnl[cur_month]['trades'] += 1
                    monthly_pnl[cur_month]['pnl'] += pnl
                last_closed_dir = position
                position = 0
                trail_active = False
                balance_history.append((str(ts), round(balance, 2)))
                continue

            # 트레일링 활성화
            if roi >= trail_act_pct / 100.0:
                trail_active = True
            if roi > highest_roi:
                highest_roi = roi

            # TSL
            if trail_active and highest_roi > 0:
                drop = highest_roi - roi
                if drop >= trail_pct / 100.0:
                    pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
                    balance += pnl
                    trades_log.append({
                        'entry_time': str(timestamps[entry_idx]),
                        'exit_time': str(ts),
                        'direction': 'LONG' if position == 1 else 'SHORT',
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(cur_price, 2),
                        'roi_pct': round(roi * 100, 2),
                        'pnl': round(pnl, 2),
                        'balance_after': round(balance, 2),
                        'close_type': 'TSL',
                        'margin_used': round(position_margin, 2),
                        'highest_roi_pct': round(highest_roi * 100, 2)
                    })
                    if cur_month in monthly_pnl:
                        monthly_pnl[cur_month]['trades'] += 1
                        monthly_pnl[cur_month]['pnl'] += pnl
                    last_closed_dir = position
                    position = 0
                    trail_active = False
                    balance_history.append((str(ts), round(balance, 2)))
                    continue

        # 크로스 감지
        cross_signal = 0
        if fast_ma[i - 1] <= slow_ma[i - 1] and fast_ma[i] > slow_ma[i]:
            cross_signal = 1
        elif fast_ma[i - 1] >= slow_ma[i - 1] and fast_ma[i] < slow_ma[i]:
            cross_signal = -1

        if cross_signal != 0:
            # REV
            if position != 0 and position != cross_signal:
                if position == 1:
                    roi = (cur_price - entry_price) / entry_price
                else:
                    roi = (entry_price - cur_price) / entry_price
                pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
                balance += pnl
                trades_log.append({
                    'entry_time': str(timestamps[entry_idx]),
                    'exit_time': str(ts),
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(cur_price, 2),
                    'roi_pct': round(roi * 100, 2),
                    'pnl': round(pnl, 2),
                    'balance_after': round(balance, 2),
                    'close_type': 'REV',
                    'margin_used': round(position_margin, 2),
                    'highest_roi_pct': round(highest_roi * 100, 2)
                })
                if cur_month in monthly_pnl:
                    monthly_pnl[cur_month]['trades'] += 1
                    monthly_pnl[cur_month]['pnl'] += pnl
                last_closed_dir = position
                position = 0
                trail_active = False
                balance_history.append((str(ts), round(balance, 2)))

            # 필터
            if adx[i] >= adx_thresh and rsi_lo <= rsi[i] <= rsi_hi:
                if skip_same_dir and cross_signal == last_closed_dir:
                    pass
                else:
                    pending_signal = cross_signal
                    pending_bar = i
                    pending_price = cur_price

        # 지연 진입
        if pending_signal != 0 and position == 0:
            bars_elapsed = i - pending_bar
            if bars_elapsed >= entry_delay_bars:
                if pending_signal == 1:
                    price_change = (cur_price - pending_price) / pending_price * 100.0
                    ok = price_change >= entry_offset_pct
                else:
                    price_change = (pending_price - cur_price) / pending_price * 100.0
                    ok = price_change >= entry_offset_pct

                if ok and adx[i] >= adx_thresh and rsi_lo <= rsi[i] <= rsi_hi:
                    entry_price = cur_price
                    entry_idx = i
                    position_margin = balance * margin_pct / 100.0
                    position_size = position_margin * leverage / cur_price
                    position = pending_signal
                    highest_roi = 0.0
                    trail_active = False
                    pending_signal = 0
                elif bars_elapsed > entry_delay_bars + 12:
                    pending_signal = 0

        # MDD
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100.0
        if dd > max_dd:
            max_dd = dd

        # 월별 잔액 업데이트
        if cur_month in monthly_pnl:
            monthly_pnl[cur_month]['end_bal'] = balance

        # 연도별 잔액 업데이트
        if cur_year in yearly_balances:
            yearly_balances[cur_year]['end'] = balance

    # 잔여 포지션 청산
    if position != 0:
        if position == 1:
            roi = (close[-1] - entry_price) / entry_price
        else:
            roi = (entry_price - close[-1]) / entry_price
        pnl = position_margin * leverage * roi - position_margin * leverage * fee_rate * 2
        balance += pnl
        ts = timestamps[-1]
        trades_log.append({
            'entry_time': str(timestamps[entry_idx]),
            'exit_time': str(ts),
            'direction': 'LONG' if position == 1 else 'SHORT',
            'entry_price': round(entry_price, 2),
            'exit_price': round(close[-1], 2),
            'roi_pct': round(roi * 100, 2),
            'pnl': round(pnl, 2),
            'balance_after': round(balance, 2),
            'close_type': 'END',
            'margin_used': round(position_margin, 2),
            'highest_roi_pct': round(highest_roi * 100, 2)
        })

    return {
        'balance': round(balance, 2),
        'mdd': round(max_dd, 2),
        'trades': trades_log,
        'monthly_pnl': monthly_pnl,
        'yearly_balances': yearly_balances,
        'total_trades': len(trades_log)
    }


def run_tier_verification(tier_name, df, fast_ma_type, fast_period, slow_ma_type, slow_period,
                          adx_period, adx_thresh, rsi_lo, rsi_hi,
                          sl_pct, trail_act, trail_pct,
                          margin_pct, leverage, delay, offset, skip):
    """단일 Tier 정밀 검증"""
    timestamps = df['timestamp'].values
    ts_pd = pd.to_datetime(df['timestamp'])
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)

    # 지표 계산
    fast_ma = calc_ma(close, volume, fast_ma_type, fast_period)
    slow_ma = calc_ma(close, volume, slow_ma_type, slow_period)
    adx_arr = calc_adx_wilder(high, low, close, adx_period)
    rsi_arr = calc_rsi_wilder(close, 14)

    print(f"\n{'='*70}")
    print(f"  {tier_name} 정밀 검증")
    print(f"{'='*70}")

    result = detailed_backtest(
        ts_pd, close, high, low, volume,
        fast_ma, slow_ma, adx_arr, rsi_arr,
        adx_thresh, rsi_lo, rsi_hi,
        sl_pct, trail_act, trail_pct,
        margin_pct, leverage, delay, offset,
        0.0004, skip
    )

    # 거래 내역 출력
    trades = result['trades']
    print(f"\n  총 거래: {len(trades)}회")
    print(f"  최종 잔액: ${result['balance']:,.2f}")
    print(f"  MDD: {result['mdd']:.2f}%")

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    total_profit = sum(t['pnl'] for t in wins)
    total_loss = sum(abs(t['pnl']) for t in losses)
    pf = total_profit / total_loss if total_loss > 0 else 999

    print(f"  승: {len(wins)} | 패: {len(losses)} | 승률: {len(wins)/len(trades)*100:.1f}%")
    print(f"  PF: {pf:.2f}")
    print(f"  총이익: ${total_profit:,.2f} | 총손실: ${total_loss:,.2f}")
    if len(wins) > 0:
        print(f"  평균 승: ${total_profit/len(wins):,.2f}")
    if len(losses) > 0:
        print(f"  평균 패: ${total_loss/len(losses):,.2f}")

    # 청산 유형별
    sl_count = len([t for t in trades if t['close_type'] == 'SL'])
    tsl_count = len([t for t in trades if t['close_type'] == 'TSL'])
    rev_count = len([t for t in trades if t['close_type'] == 'REV'])
    print(f"  SL: {sl_count} | TSL: {tsl_count} | REV: {rev_count}")

    # 모든 거래 상세
    print(f"\n  --- 전체 거래 내역 ---")
    print(f"  {'#':>3} {'진입일시':>20} {'청산일시':>20} {'방향':>5} {'진입가':>10} {'청산가':>10} {'ROI%':>7} {'PnL':>10} {'잔액':>12} {'유형':>4} {'최고ROI%':>8}")
    print(f"  {'-'*120}")
    for idx, t in enumerate(trades):
        print(f"  {idx+1:>3} {t['entry_time']:>20} {t['exit_time']:>20} {t['direction']:>5} "
              f"${t['entry_price']:>9,.1f} ${t['exit_price']:>9,.1f} {t['roi_pct']:>6.2f}% "
              f"${t['pnl']:>9,.2f} ${t['balance_after']:>11,.2f} {t['close_type']:>4} {t['highest_roi_pct']:>7.2f}%")

    # 연도별 요약
    print(f"\n  --- 연도별 수익 ---")
    print(f"  {'연도':>6} {'시작잔액':>12} {'종료잔액':>12} {'수익률':>10} {'거래수':>6}")
    print(f"  {'-'*50}")
    yearly = result['yearly_balances']
    for year in sorted(yearly.keys()):
        y = yearly[year]
        ret = (y['end'] - y['start']) / y['start'] * 100 if y['start'] > 0 else 0
        year_trades = len([t for t in trades if t['entry_time'][:4] == year])
        print(f"  {year:>6} ${y['start']:>11,.2f} ${y['end']:>11,.2f} {ret:>+9.1f}% {year_trades:>5}")

    # 월별 요약 (거래 있는 월만)
    print(f"\n  --- 월별 수익 (거래 있는 월) ---")
    print(f"  {'월':>7} {'거래':>4} {'PnL':>10} {'시작잔액':>12} {'종료잔액':>12}")
    print(f"  {'-'*50}")
    monthly = result['monthly_pnl']
    for month in sorted(monthly.keys()):
        m = monthly[month]
        if m['trades'] > 0:
            print(f"  {month:>7} {m['trades']:>4} ${m['pnl']:>9,.2f} ${m['start_bal']:>11,.2f} ${m['end_bal']:>11,.2f}")

    return result


def main():
    print("=" * 70)
    print("  v28 정밀 검증: 3-Tier 전체 거래 내역")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] Loading data...")
    df_5m = load_5m_data(BASE)
    df_15m = resample_ohlcv(df_5m, 15)
    print(f"  15m: {len(df_15m)} bars, {df_15m['timestamp'].iloc[0]} ~ {df_15m['timestamp'].iloc[-1]}")

    results = {}

    # Tier 1: WMA(5)/VWMA(300) ADX(20)>=35 RSI35-75 SL-8% Trail+10/-5 M15% Lev15x Delay2
    results['tier1'] = run_tier_verification(
        "Tier 1 Sniper: WMA(5)/VWMA(300) 15m",
        df_15m,
        fast_ma_type=1, fast_period=5,   # WMA(5)
        slow_ma_type=4, slow_period=300, # VWMA(300)
        adx_period=20, adx_thresh=35.0,
        rsi_lo=35.0, rsi_hi=75.0,
        sl_pct=-8.0, trail_act=10.0, trail_pct=5.0,
        margin_pct=15.0, leverage=15.0,
        delay=2, offset=0.0, skip=0
    )

    # Tier 2: HMA(14)/VWMA(300) ADX(20)>=35 RSI35-70 SL-7% Trail+10/-3 M50% Lev10x Delay3 Offset-1.5
    results['tier2'] = run_tier_verification(
        "Tier 2 Core: HMA(14)/VWMA(300) 15m",
        df_15m,
        fast_ma_type=3, fast_period=14,  # HMA(14)
        slow_ma_type=4, slow_period=300, # VWMA(300)
        adx_period=20, adx_thresh=35.0,
        rsi_lo=35.0, rsi_hi=70.0,
        sl_pct=-7.0, trail_act=10.0, trail_pct=3.0,
        margin_pct=50.0, leverage=10.0,
        delay=3, offset=-1.5, skip=0
    )

    # Tier 3: VWMA(2)/VWMA(300) ADX(14)>=45 RSI30-75 SL-9% Trail+5/-5 M40% Lev10x Delay0
    results['tier3'] = run_tier_verification(
        "Tier 3 Compounder: VWMA(2)/VWMA(300) 15m",
        df_15m,
        fast_ma_type=4, fast_period=2,   # VWMA(2)
        slow_ma_type=4, slow_period=300, # VWMA(300)
        adx_period=14, adx_thresh=45.0,
        rsi_lo=30.0, rsi_hi=75.0,
        sl_pct=-9.0, trail_act=5.0, trail_pct=5.0,
        margin_pct=40.0, leverage=10.0,
        delay=0, offset=0.0, skip=0
    )

    # 포트폴리오 합산
    print(f"\n{'='*70}")
    print(f"  포트폴리오 합산 (40% / 35% / 25%)")
    print(f"{'='*70}")

    weights = {'tier1': 0.40, 'tier2': 0.35, 'tier3': 0.25}
    total_balance = 0
    total_trades = 0
    print(f"\n  {'Tier':>15} {'배분':>6} {'초기':>8} {'최종':>12} {'수익률':>10} {'PF':>7} {'MDD':>6} {'거래':>5}")
    print(f"  {'-'*75}")
    for tier, w in weights.items():
        r = results[tier]
        init = 3000.0 * w
        final = r['balance'] * w
        ret = (r['balance'] - 3000) / 3000 * 100
        trades = r['trades']
        wins_t = [t for t in trades if t['pnl'] > 0]
        losses_t = [t for t in trades if t['pnl'] <= 0]
        tp = sum(t['pnl'] for t in wins_t)
        tl = sum(abs(t['pnl']) for t in losses_t)
        pf = tp / tl if tl > 0 else 999
        print(f"  {tier:>15} {w*100:>5.0f}% ${init:>7,.0f} ${final:>11,.2f} {ret:>+9.1f}% {pf:>6.2f} {r['mdd']:>5.1f}% {len(trades):>4}")
        total_balance += final
        total_trades += len(trades)

    port_ret = (total_balance - 3000) / 3000 * 100
    print(f"  {'-'*75}")
    print(f"  {'TOTAL':>15} {'100':>5}% ${3000:>7,.0f} ${total_balance:>11,.2f} {port_ret:>+9.1f}%{'':>7} {'':>6} {total_trades:>4}")

    # 결과 저장
    save_path = os.path.join(BASE, "v28_detailed_verification.json")
    save_data = {}
    for tier in ['tier1', 'tier2', 'tier3']:
        r = results[tier]
        save_data[tier] = {
            'balance': r['balance'],
            'mdd': r['mdd'],
            'total_trades': r['total_trades'],
            'trades': r['trades'],
            'yearly_balances': r['yearly_balances'],
            'monthly_summary': {k: v for k, v in r['monthly_pnl'].items() if v['trades'] > 0}
        }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
