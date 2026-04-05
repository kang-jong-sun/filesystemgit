"""
v27 월별 상세 데이터 추출 - 75개월 전체
"""
import pandas as pd
import numpy as np
import time, os, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

from v27_backtest_engine import (
    load_5m_data, build_indicator_cache, map_tf_index, run_backtest_core
)

def run_detailed_backtest(cache, cfg, label):
    """상세 거래 내역 포함 백테스트"""
    ts_5m = cache['5m']['timestamp']
    close_5m = cache['5m']['close']
    high_5m = cache['5m']['high']
    low_5m = cache['5m']['low']
    n = len(close_5m)
    ts_i64 = ts_5m.astype('int64')

    tf_maps = {}
    for tf in ['10m', '15m', '30m', '1h']:
        tf_maps[tf] = map_tf_index(ts_5m, cache[tf]['timestamp'])

    # Build signals
    ctf = cfg['cross_tf']
    c_data = cache[ctf]

    ft = cfg['fast_type']; fl = cfg['fast_len']
    st = cfg['slow_type']; sl_len = cfg['slow_len']

    fm = c_data[ft][fl]
    sm = c_data[st][sl_len]

    sig = np.zeros(len(fm), dtype=np.int64)
    for j in range(1, len(fm)):
        if fm[j] > sm[j]: sig[j] = 1
        elif fm[j] < sm[j]: sig[j] = -1
    csig = sig if ctf == '5m' else sig[tf_maps[ctf]]

    atf = cfg['adx_tf']
    adx_v = cache[atf]['adx'][cfg['adx_p']]
    if atf != '5m': adx_v = adx_v[tf_maps[atf]]

    rtf = cfg['rsi_tf']
    rsi_v = cache[rtf]['rsi'][cfg['rsi_p']]
    if rtf != '5m': rsi_v = rsi_v[tf_maps[rtf]]

    macd_v = cache['5m']['macd'][cfg['macd']]['hist']
    atr_v = cache['5m']['atr'][14]

    # Run full backtest - we need trade-level detail
    # Re-implement with trade tracking
    leverage = 10
    fee_rate = 0.0004
    initial_capital = 3000.0
    capital = initial_capital
    position = 0
    entry_price = 0.0
    position_size = 0.0
    peak_roi = 0.0
    trail_active = False
    peak_capital = initial_capital

    last_cross_bar = -9999
    last_cross_dir = 0
    last_cross_price = 0.0

    entry_bar = 0

    trades = []

    # Daily equity for MDD
    daily_equity = []

    for i in range(300, n):
        price = close_5m[i]
        hi = high_5m[i]
        lo = low_5m[i]

        dd = (capital - peak_capital) / peak_capital if peak_capital > 0 else 0
        current_margin = cfg['margin_dd'] if dd < -0.20 else cfg['margin']

        # Detect cross
        if csig[i] != 0 and csig[i] != csig[i-1]:
            last_cross_bar = i
            last_cross_dir = csig[i]
            last_cross_price = price

        # Position management
        if position != 0:
            if position == 1:
                current_roi = (price - entry_price) / entry_price * leverage
                max_roi_bar = (hi - entry_price) / entry_price * leverage
                min_roi_bar = (lo - entry_price) / entry_price * leverage
            else:
                current_roi = (entry_price - price) / entry_price * leverage
                max_roi_bar = (entry_price - lo) / entry_price * leverage
                min_roi_bar = (entry_price - hi) / entry_price * leverage

            if max_roi_bar > peak_roi:
                peak_roi = max_roi_bar

            actual_sl = -cfg['sl'] * leverage
            trail_act_v = cfg['trail_act'] * leverage
            trail_pct_v = cfg['trail_pct'] * leverage

            exit_type = -1
            exit_price = price

            # SL
            if min_roi_bar <= actual_sl:
                exit_type = 0
                if position == 1:
                    exit_price = entry_price * (1 + actual_sl / leverage)
                else:
                    exit_price = entry_price * (1 - actual_sl / leverage)

            # TSL
            if exit_type < 0 and peak_roi >= trail_act_v:
                trail_active = True
                if current_roi <= peak_roi - trail_pct_v:
                    exit_type = 1

            # REV
            if exit_type < 0:
                new_sig = csig[i]
                if new_sig != 0 and new_sig != position:
                    adx_ok = adx_v[i] >= cfg['adx_min']
                    rsi_ok = rsi_v[i] >= cfg['rsi_min'] and rsi_v[i] <= cfg['rsi_max']
                    if adx_ok and rsi_ok:
                        exit_type = 2

            if exit_type >= 0:
                if exit_type == 0:
                    pnl_pct = actual_sl
                elif exit_type == 1:
                    pnl_pct = peak_roi - trail_pct_v
                else:
                    pnl_pct = current_roi

                fee = abs(position_size * exit_price * fee_rate)
                pnl_dollar = position_size * exit_price * pnl_pct / leverage - fee
                capital += pnl_dollar
                if capital > peak_capital:
                    peak_capital = capital

                exit_ts = pd.Timestamp(ts_5m[i])
                entry_ts = pd.Timestamp(ts_5m[entry_bar])

                exit_names = {0: 'SL', 1: 'TSL', 2: 'REV'}
                trades.append({
                    'entry_time': entry_ts,
                    'exit_time': exit_ts,
                    'direction': 'long' if position == 1 else 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'roi_pct': pnl_pct / leverage * 100,
                    'pnl': pnl_dollar,
                    'exit_type': exit_names.get(exit_type, '?'),
                    'peak_roi_pct': peak_roi / leverage * 100,
                    'balance': capital,
                    'year': exit_ts.year,
                    'month': exit_ts.month,
                })

                if exit_type == 2:
                    position = -position
                    entry_price = price
                    entry_bar = i
                    pos_value = capital * current_margin
                    position_size = pos_value / price
                    fee_e = position_size * price * fee_rate
                    capital -= fee_e
                    peak_roi = 0.0
                    trail_active = False
                else:
                    position = 0
                    entry_price = 0.0
                    peak_roi = 0.0
                    trail_active = False

        # Entry
        if position == 0 and last_cross_dir != 0:
            bars_since = i - last_cross_bar
            if 0 <= bars_since <= cfg['entry_delay']:
                price_diff = (price - last_cross_price) / last_cross_price * 100
                entry_ok = False
                if last_cross_dir == 1 and -cfg['entry_tol'] <= price_diff <= 0.5:
                    entry_ok = True
                elif last_cross_dir == -1 and -0.5 <= price_diff <= cfg['entry_tol']:
                    entry_ok = True

                adx_ok = adx_v[i] >= cfg['adx_min']
                rsi_ok = rsi_v[i] >= cfg['rsi_min'] and rsi_v[i] <= cfg['rsi_max']
                macd_ok = (last_cross_dir == 1 and macd_v[i] > 0) or (last_cross_dir == -1 and macd_v[i] < 0)

                if entry_ok and adx_ok and rsi_ok and macd_ok:
                    position = last_cross_dir
                    entry_price = price
                    entry_bar = i
                    pos_value = capital * current_margin
                    position_size = pos_value / price
                    fee_e = position_size * price * fee_rate
                    capital -= fee_e
                    peak_roi = 0.0
                    trail_active = False
                    last_cross_dir = 0

        # Daily equity
        if i % 288 == 0:
            daily_equity.append({'date': pd.Timestamp(ts_5m[i]), 'equity': capital})

    # Close open position
    if position != 0:
        price = close_5m[n-1]
        if position == 1:
            cr = (price - entry_price) / entry_price * leverage
        else:
            cr = (entry_price - price) / entry_price * leverage
        fee = abs(position_size * price * fee_rate)
        pnl = position_size * price * cr / leverage - fee
        capital += pnl
        trades.append({
            'entry_time': pd.Timestamp(ts_5m[entry_bar]),
            'exit_time': pd.Timestamp(ts_5m[n-1]),
            'direction': 'long' if position == 1 else 'short',
            'entry_price': entry_price, 'exit_price': price,
            'roi_pct': cr / leverage * 100, 'pnl': pnl,
            'exit_type': 'END', 'peak_roi_pct': peak_roi / leverage * 100,
            'balance': capital,
            'year': pd.Timestamp(ts_5m[n-1]).year,
            'month': pd.Timestamp(ts_5m[n-1]).month,
        })

    return trades, daily_equity, capital


def print_monthly_summary(trades, label):
    """월별 상세 요약 출력"""
    df = pd.DataFrame(trades)
    if len(df) == 0:
        print(f"  No trades for {label}")
        return

    df['ym'] = df['exit_time'].dt.to_period('M')

    # Generate all months from 2020-01 to 2026-03
    all_months = pd.period_range('2020-01', '2026-03', freq='M')

    print(f"\n{'='*120}")
    print(f"  {label} - 월별 상세 데이터 (75개월)")
    print(f"{'='*120}")
    print(f"| {'월':^8} | {'거래':^4} | {'승':^3} | {'패':^3} | {'승률':^6} | {'총이익$':^12} | {'총손실$':^12} | {'순손익$':^12} | {'누적$':^14} | {'PF':^6} | {'MDD%':^6} | {'SL':^3} | {'TSL':^4} | {'REV':^4} | {'Avg Win%':^9} | {'Avg Loss%':^9} |")
    print(f"|{'-'*10}|{'-'*6}|{'-'*5}|{'-'*5}|{'-'*8}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*16}|{'-'*8}|{'-'*8}|{'-'*5}|{'-'*6}|{'-'*6}|{'-'*11}|{'-'*11}|")

    cumulative = 0
    peak_cum = 0
    max_dd = 0

    yearly_data = {}

    for month in all_months:
        month_trades = df[df['ym'] == month]
        tc = len(month_trades)

        if tc == 0:
            cumulative_display = cumulative
            print(f"| {str(month):^8} | {0:^4} | {'-':^3} | {'-':^3} | {'-':^6} | {'-':^12} | {'-':^12} | {'-':^12} | {cumulative_display:>+13,.0f} | {'-':^6} | {'-':^6} | {'-':^3} | {'-':^4} | {'-':^4} | {'-':^9} | {'-':^9} |")
        else:
            wins = len(month_trades[month_trades['pnl'] > 0])
            losses = tc - wins
            wr = wins / tc * 100

            total_profit = month_trades[month_trades['pnl'] > 0]['pnl'].sum()
            total_loss = month_trades[month_trades['pnl'] <= 0]['pnl'].sum()
            net = total_profit + total_loss
            cumulative += net

            if cumulative > peak_cum:
                peak_cum = cumulative
            dd = (cumulative - peak_cum) / (peak_cum + 3000) * 100 if peak_cum + 3000 > 0 else 0
            if abs(dd) > max_dd:
                max_dd = abs(dd)

            pf = total_profit / abs(total_loss) if total_loss != 0 else 999.0

            sl_c = len(month_trades[month_trades['exit_type'] == 'SL'])
            tsl_c = len(month_trades[month_trades['exit_type'] == 'TSL'])
            rev_c = len(month_trades[month_trades['exit_type'] == 'REV'])

            win_trades = month_trades[month_trades['pnl'] > 0]
            loss_trades = month_trades[month_trades['pnl'] <= 0]
            avg_win = win_trades['roi_pct'].mean() if len(win_trades) > 0 else 0
            avg_loss = loss_trades['roi_pct'].mean() if len(loss_trades) > 0 else 0

            pf_str = f"{pf:.2f}" if pf < 100 else "INF"

            print(f"| {str(month):^8} | {tc:^4} | {wins:^3} | {losses:^3} | {wr:>5.1f}% | {total_profit:>+12,.0f} | {total_loss:>+12,.0f} | {net:>+12,.0f} | {cumulative:>+13,.0f} | {pf_str:^6} | {abs(dd):>5.1f}% | {sl_c:^3} | {tsl_c:^4} | {rev_c:^4} | {avg_win:>+8.1f}% | {avg_loss:>+8.1f}% |")

        # Yearly aggregation
        yr = month.year
        if yr not in yearly_data:
            yearly_data[yr] = {'trades': 0, 'wins': 0, 'profit': 0, 'loss': 0}
        if tc > 0:
            yearly_data[yr]['trades'] += tc
            yearly_data[yr]['wins'] += len(month_trades[month_trades['pnl'] > 0])
            yearly_data[yr]['profit'] += month_trades[month_trades['pnl'] > 0]['pnl'].sum()
            yearly_data[yr]['loss'] += month_trades[month_trades['pnl'] <= 0]['pnl'].sum()

    # Yearly summary
    print(f"\n{'='*80}")
    print(f"  {label} - 연도별 요약")
    print(f"{'='*80}")
    print(f"| {'연도':^6} | {'거래':^5} | {'승':^4} | {'패':^4} | {'승률':^7} | {'총이익$':^12} | {'총손실$':^12} | {'순손익$':^12} | {'PF':^6} |")
    print(f"|{'-'*8}|{'-'*7}|{'-'*6}|{'-'*6}|{'-'*9}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*8}|")

    for yr in sorted(yearly_data.keys()):
        yd = yearly_data[yr]
        tc = yd['trades']
        wins = yd['wins']
        losses = tc - wins
        wr = wins / tc * 100 if tc > 0 else 0
        net = yd['profit'] + yd['loss']
        pf = yd['profit'] / abs(yd['loss']) if yd['loss'] != 0 else 999.0
        pf_str = f"{pf:.2f}" if pf < 100 else "INF"
        print(f"| {yr:^6} | {tc:^5} | {wins:^4} | {losses:^4} | {wr:>6.1f}% | {yd['profit']:>+12,.0f} | {yd['loss']:>+12,.0f} | {net:>+12,.0f} | {pf_str:^6} |")

    # Total
    total_trades = len(df)
    total_wins = len(df[df['pnl'] > 0])
    total_profit = df[df['pnl'] > 0]['pnl'].sum()
    total_loss_v = df[df['pnl'] <= 0]['pnl'].sum()
    total_pf = total_profit / abs(total_loss_v) if total_loss_v != 0 else 999

    print(f"|{'─'*8}|{'─'*7}|{'─'*6}|{'─'*6}|{'─'*9}|{'─'*14}|{'─'*14}|{'─'*14}|{'─'*8}|")
    print(f"| {'합계':^6} | {total_trades:^5} | {total_wins:^4} | {total_trades-total_wins:^4} | {total_wins/total_trades*100:>6.1f}% | {total_profit:>+12,.0f} | {total_loss_v:>+12,.0f} | {total_profit+total_loss_v:>+12,.0f} | {total_pf:.2f} |")
    print(f"\n  최종 잔액: ${df.iloc[-1]['balance']:,.0f} | 수익률: {(df.iloc[-1]['balance']-3000)/3000*100:,.1f}% | Max DD: {max_dd:.1f}%")


def main():
    print("Loading data...", flush=True)
    df_5m = load_5m_data()
    print("Building indicators...", flush=True)
    cache = build_indicator_cache(df_5m)

    # JIT warmup
    ts_5m = cache['5m']['timestamp']
    nw = 1000
    _ = run_backtest_core(
        cache['5m']['close'][:nw], cache['5m']['high'][:nw], cache['5m']['low'][:nw],
        ts_5m[:nw].astype('int64'),
        np.zeros(nw, dtype=np.int64), np.zeros(nw), np.zeros(nw),
        np.zeros(nw), np.zeros(nw),
        25, 30, 70, 0.07, 0.07, 0.03, 0.0, 0.0,
        10, 0.20, 0.10, -0.20, 0.0004, 3000.0, 6, 1.0,
        0, 2.0, 0, 2.0, np.array([1.]*7)
    )

    # Model A config
    cfg_a = {
        'cross_tf': '30m', 'fast_type': 'ema', 'fast_len': 5, 'slow_type': 'sma', 'slow_len': 50,
        'adx_tf': '5m', 'adx_p': 14, 'adx_min': 35,
        'rsi_tf': '5m', 'rsi_p': 14, 'rsi_min': 40, 'rsi_max': 80,
        'macd': '12_26_9',
        'entry_delay': 12, 'entry_tol': 0.5,
        'sl': 0.12, 'trail_act': 0.05, 'trail_pct': 0.03,
        'margin': 0.40, 'margin_dd': 0.20,
    }

    # Model B config
    cfg_b = {
        'cross_tf': '30m', 'fast_type': 'ema', 'fast_len': 5, 'slow_type': 'sma', 'slow_len': 50,
        'adx_tf': '5m', 'adx_p': 14, 'adx_min': 35,
        'rsi_tf': '5m', 'rsi_p': 14, 'rsi_min': 40, 'rsi_max': 80,
        'macd': '12_26_9',
        'entry_delay': 12, 'entry_tol': 0.5,
        'sl': 0.10, 'trail_act': 0.06, 'trail_pct': 0.03,
        'margin': 0.30, 'margin_dd': 0.15,
    }

    # Phase 1B Best config
    cfg_p1b = {
        'cross_tf': '5m', 'fast_type': 'ema', 'fast_len': 5, 'slow_type': 'ema', 'slow_len': 150,
        'adx_tf': '30m', 'adx_p': 20, 'adx_min': 40,
        'rsi_tf': '10m', 'rsi_p': 14, 'rsi_min': 30, 'rsi_max': 70,
        'macd': '5_35_5',
        'entry_delay': 6, 'entry_tol': 1.5,
        'sl': 0.07, 'trail_act': 0.07, 'trail_pct': 0.03,
        'margin': 0.40, 'margin_dd': 0.20,
    }

    print("\n\nRunning Model A backtest...", flush=True)
    trades_a, eq_a, final_a = run_detailed_backtest(cache, cfg_a, "Model A")
    print_monthly_summary(trades_a, "v27 Model A (수익극대형)")

    print("\n\nRunning Model B backtest...", flush=True)
    trades_b, eq_b, final_b = run_detailed_backtest(cache, cfg_b, "Model B")
    print_monthly_summary(trades_b, "v27 Model B (안정형)")

    print("\n\nRunning Phase1B Best backtest...", flush=True)
    trades_p, eq_p, final_p = run_detailed_backtest(cache, cfg_p1b, "Phase1B Best")
    print_monthly_summary(trades_p, "v27 Phase1B Best (EMA5/EMA150, PF 1.89)")

    print("\n\nDONE.", flush=True)


if __name__ == '__main__':
    main()
