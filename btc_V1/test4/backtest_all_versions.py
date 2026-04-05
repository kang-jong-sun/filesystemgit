"""
BTC/USDT 선물 자동매매 백테스트 엔진
4개 버전(v13.5, v14.4, v15.2, v15.4) 동시 검증
"""
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

# ============================================================
# 1. 데이터 로드
# ============================================================
def load_data(base_dir):
    """5분봉 CSV 3개 파트 로드 후 병합"""
    files = [
        os.path.join(base_dir, f'btc_usdt_5m_2020_to_now_part{i}.csv')
        for i in range(1, 4)
    ]
    dfs = []
    for f in files:
        print(f"  Loading {os.path.basename(f)} ...")
        df = pd.read_csv(f, parse_dates=['timestamp'])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  총 {len(df):,}행 로드 완료 ({df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]})")
    return df


def resample_to_30m(df_5m):
    """5분봉 -> 30분봉 리샘플링"""
    df = df_5m.set_index('timestamp')
    ohlcv = df.resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()
    ohlcv.reset_index(inplace=True)
    print(f"  30분봉 리샘플링: {len(ohlcv):,}행")
    return ohlcv


# ============================================================
# 2. 지표 계산
# ============================================================
def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def calc_adx(high, low, close, period=14):
    """Wilder's Smoothing ADX"""
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    dx = dx.fillna(0)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx


def calc_rsi(series, period=14):
    """Wilder's Smoothing RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)


def add_indicators(df, ema_fast, ema_slow, adx_period=14, rsi_period=14):
    """EMA, ADX, RSI 지표 추가"""
    df = df.copy()
    df['ema_fast'] = calc_ema(df['close'], ema_fast)
    df['ema_slow'] = calc_ema(df['close'], ema_slow)
    df['adx'] = calc_adx(df['high'], df['low'], df['close'], adx_period)
    df['rsi'] = calc_rsi(df['close'], rsi_period)
    return df


# ============================================================
# 3. 백테스트 엔진
# ============================================================
class BacktestEngine:
    def __init__(self, config, df):
        self.cfg = config
        self.df = df
        self.balance = 3000.0
        self.initial_balance = 3000.0
        self.position = None  # {'side': 'long'/'short', 'entry_price', 'size', 'margin', 'peak_price', 'trailing_active'}
        self.trades = []
        self.monthly_pnl = {}
        self.peak_balance = 3000.0
        self.mdd = 0.0
        self.fl_count = 0  # 강제청산 횟수
        self.equity_curve = []

        # 보호 상태
        self.month_start_balance = 3000.0
        self.current_month = None
        self.monthly_loss_triggered = False

        # 연패 관련 (v13.5 only)
        self.consec_losses = 0
        self.pause_until_idx = -1

        # 낙폭 관련 (v13.5 only)
        self.dd_active = False

        # 지연진입 (v15.2)
        self.pending_signal = None  # {'type': 'long'/'short', 'trigger_idx': int, 'wait_candles': int}

    def run(self):
        """메인 백테스트 루프"""
        df = self.df
        cfg = self.cfg

        for i in range(1, len(df)):
            ts = df['timestamp'].iloc[i]
            month_key = ts.strftime('%Y-%m')

            # 월 초기화
            if month_key != self.current_month:
                if self.current_month is not None:
                    self._record_month(self.current_month)
                self.current_month = month_key
                self.month_start_balance = self.balance
                self.monthly_loss_triggered = False

            # 포지션 보유 중 처리
            if self.position is not None:
                self._check_exit(i)

            # 포지션이 없으면 진입 체크
            if self.position is None:
                # 보호 조건 확인
                if self._can_enter(i):
                    self._check_entry(i)

            # 에쿼티 기록
            equity = self._calc_equity(i)
            self.equity_curve.append((ts, equity))

            if equity > self.peak_balance:
                self.peak_balance = equity
            dd = (self.peak_balance - equity) / self.peak_balance
            if dd > self.mdd:
                self.mdd = dd

        # 마지막 월 기록
        if self.current_month:
            self._record_month(self.current_month)

        # 마지막 포지션 청산
        if self.position is not None:
            self._close_position(len(df)-1, 'END')

        return self._summary()

    def _calc_equity(self, i):
        """현재 순자산 계산"""
        if self.position is None:
            return self.balance
        price = self.df['close'].iloc[i]
        pos = self.position
        if pos['side'] == 'long':
            pnl = (price - pos['entry_price']) / pos['entry_price'] * pos['size']
        else:
            pnl = (pos['entry_price'] - price) / pos['entry_price'] * pos['size']
        return self.balance + pnl

    def _can_enter(self, i):
        """보호 메커니즘 체크"""
        cfg = self.cfg

        # 월간 손실 한도
        if cfg.get('monthly_loss_limit'):
            ml = cfg['monthly_loss_limit']
            equity = self._calc_equity(i)
            if self.month_start_balance > 0:
                month_return = (equity - self.month_start_balance) / self.month_start_balance
                if month_return <= ml:
                    self.monthly_loss_triggered = True
                    return False
            if self.monthly_loss_triggered:
                return False

        # 연패 정지 (v13.5)
        if cfg.get('consec_loss_pause'):
            if i < self.pause_until_idx:
                return False

        return True

    def _get_margin_pct(self):
        """현재 마진율 (낙폭 시 축소 포함)"""
        cfg = self.cfg
        margin = cfg['margin']

        # DD 축소 (v13.5)
        if cfg.get('dd_threshold'):
            if self.peak_balance > 0:
                dd = (self.peak_balance - self.balance) / self.peak_balance
                if dd >= abs(cfg['dd_threshold']):
                    self.dd_active = True
                    return cfg.get('margin_reduced', margin / 2)
                else:
                    self.dd_active = False

        return margin

    def _check_entry(self, i):
        """진입 신호 체크"""
        df = self.df
        cfg = self.cfg

        ema_fast_curr = df['ema_fast'].iloc[i]
        ema_slow_curr = df['ema_slow'].iloc[i]
        ema_fast_prev = df['ema_fast'].iloc[i-1]
        ema_slow_prev = df['ema_slow'].iloc[i-1]
        adx_val = df['adx'].iloc[i]
        rsi_val = df['rsi'].iloc[i]

        adx_min = cfg['adx_min']
        rsi_min = cfg['rsi_min']
        rsi_max = cfg['rsi_max']

        cross_up = (ema_fast_curr > ema_slow_curr) and (ema_fast_prev <= ema_slow_prev)
        cross_dn = (ema_fast_curr < ema_slow_curr) and (ema_fast_prev >= ema_slow_prev)

        signal = None
        if cross_up and adx_val >= adx_min and rsi_min <= rsi_val <= rsi_max:
            signal = 'long'
        elif cross_dn and adx_val >= adx_min and rsi_min <= rsi_val <= rsi_max:
            signal = 'short'

        if signal is None:
            # 지연진입 대기 체크 (v15.2)
            if self.pending_signal is not None:
                wait = cfg.get('entry_delay_candles', 0)
                if wait > 0 and i >= self.pending_signal['trigger_idx'] + wait:
                    # 지연 후 진입 (추세 유지 확인)
                    if self.pending_signal['type'] == 'long' and ema_fast_curr > ema_slow_curr:
                        self._enter_position(i, 'long')
                    elif self.pending_signal['type'] == 'short' and ema_fast_curr < ema_slow_curr:
                        self._enter_position(i, 'short')
                    self.pending_signal = None
            return

        # 지연진입 처리
        delay = cfg.get('entry_delay_candles', 0)
        if delay > 0:
            self.pending_signal = {'type': signal, 'trigger_idx': i, 'wait_candles': delay}
            return

        self._enter_position(i, signal)

    def _enter_position(self, i, side):
        """포지션 진입"""
        cfg = self.cfg
        price = self.df['close'].iloc[i]
        margin_pct = self._get_margin_pct()
        margin = self.balance * margin_pct
        size = margin * cfg['leverage']

        # 수수료
        fee = size * cfg['fee_rate']
        self.balance -= fee

        self.position = {
            'side': side,
            'entry_price': price,
            'size': size,
            'margin': margin,
            'peak_price': price,
            'low_price': price,
            'trailing_active': False,
            'entry_idx': i,
            'entry_time': self.df['timestamp'].iloc[i],
        }
        self.pending_signal = None

    def _check_exit(self, i):
        """청산 조건 체크"""
        df = self.df
        cfg = self.cfg
        pos = self.position

        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        close = df['close'].iloc[i]

        # 1. 캔들 내 고점/저점으로 강제청산 및 SL 체크
        if pos['side'] == 'long':
            # 최고가 업데이트 (고점 기준)
            if high > pos['peak_price']:
                pos['peak_price'] = high
            if low < pos['low_price']:
                pos['low_price'] = low

            # 강제청산 체크 (격리마진: 약 10% 하락 시)
            fl_price = pos['entry_price'] * (1 - 1.0 / cfg['leverage'])  # ~10% 하락
            if low <= fl_price:
                self._forced_liquidation(i)
                return

            # SL 체크
            sl_price = pos['entry_price'] * (1 - cfg['sl_pct'])
            if low <= sl_price:
                self._close_position(i, 'SL', exit_price=sl_price)
                return

            # 수익률 계산
            pnl_pct = (close - pos['entry_price']) / pos['entry_price']
            peak_pnl = (pos['peak_price'] - pos['entry_price']) / pos['entry_price']

        else:  # short
            if low < pos['low_price']:
                pos['low_price'] = low
            if high > pos['peak_price']:
                pos['peak_price'] = high
            # short에서는 peak=최저가(유리), low=최고가(불리)
            # 편의를 위해 재정의
            best_price = pos['low_price']  # short에서 최저가 = 최유리

            # 강제청산 (10% 상승)
            fl_price = pos['entry_price'] * (1 + 1.0 / cfg['leverage'])
            if high >= fl_price:
                self._forced_liquidation(i)
                return

            # SL
            sl_price = pos['entry_price'] * (1 + cfg['sl_pct'])
            if high >= sl_price:
                self._close_position(i, 'SL', exit_price=sl_price)
                return

            pnl_pct = (pos['entry_price'] - close) / pos['entry_price']
            peak_pnl = (pos['entry_price'] - best_price) / pos['entry_price']

        # 2. 트레일링 SL (종가 기준)
        trail_act = cfg['trail_activate']
        trail_pct = cfg['trail_pct']

        if peak_pnl >= trail_act:
            pos['trailing_active'] = True

        if pos['trailing_active']:
            if pos['side'] == 'long':
                trail_sl = pos['peak_price'] * (1 - trail_pct)
                if close <= trail_sl:
                    self._close_position(i, 'TSL')
                    return
            else:
                trail_sl = pos['low_price'] * (1 + trail_pct)
                if close >= trail_sl:
                    self._close_position(i, 'TSL')
                    return

        # 3. 역신호 (종가 기준)
        ema_fast_curr = df['ema_fast'].iloc[i]
        ema_slow_curr = df['ema_slow'].iloc[i]
        ema_fast_prev = df['ema_fast'].iloc[i-1]
        ema_slow_prev = df['ema_slow'].iloc[i-1]
        adx_val = df['adx'].iloc[i]
        rsi_val = df['rsi'].iloc[i]

        adx_min = cfg['adx_min']
        rsi_min = cfg['rsi_min']
        rsi_max = cfg['rsi_max']

        if pos['side'] == 'long':
            cross_dn = (ema_fast_curr < ema_slow_curr) and (ema_fast_prev >= ema_slow_prev)
            if cross_dn and adx_val >= adx_min and rsi_min <= rsi_val <= rsi_max:
                self._close_position(i, 'REV')
                # 반대 진입
                if self._can_enter(i):
                    delay = cfg.get('entry_delay_candles', 0)
                    if delay > 0:
                        self.pending_signal = {'type': 'short', 'trigger_idx': i, 'wait_candles': delay}
                    else:
                        self._enter_position(i, 'short')
                return
        else:
            cross_up = (ema_fast_curr > ema_slow_curr) and (ema_fast_prev <= ema_slow_prev)
            if cross_up and adx_val >= adx_min and rsi_min <= rsi_val <= rsi_max:
                self._close_position(i, 'REV')
                if self._can_enter(i):
                    delay = cfg.get('entry_delay_candles', 0)
                    if delay > 0:
                        self.pending_signal = {'type': 'long', 'trigger_idx': i, 'wait_candles': delay}
                    else:
                        self._enter_position(i, 'long')
                return

    def _close_position(self, i, reason, exit_price=None):
        """포지션 청산"""
        cfg = self.cfg
        pos = self.position
        if exit_price is None:
            exit_price = self.df['close'].iloc[i]

        if pos['side'] == 'long':
            pnl = (exit_price - pos['entry_price']) / pos['entry_price'] * pos['size']
        else:
            pnl = (pos['entry_price'] - exit_price) / pos['entry_price'] * pos['size']

        # 청산 수수료
        fee = pos['size'] * cfg['fee_rate']
        pnl -= fee

        self.balance += pnl
        if self.balance < 0:
            self.balance = 0

        pnl_pct = pnl / pos['margin'] if pos['margin'] > 0 else 0

        trade = {
            'entry_time': pos['entry_time'],
            'exit_time': self.df['timestamp'].iloc[i],
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'size': pos['size'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'balance_after': self.balance,
        }
        self.trades.append(trade)

        # 연패 추적
        if pnl < 0:
            self.consec_losses += 1
            if self.cfg.get('consec_loss_pause') and self.consec_losses >= self.cfg['consec_loss_pause']:
                self.pause_until_idx = i + self.cfg.get('pause_duration_candles', 288)
                self.consec_losses = 0
        else:
            self.consec_losses = 0

        self.position = None

    def _forced_liquidation(self, i):
        """강제청산 (격리마진: 증거금만 손실)"""
        pos = self.position
        self.balance -= pos['margin']
        if self.balance < 0:
            self.balance = 0
        self.fl_count += 1

        trade = {
            'entry_time': pos['entry_time'],
            'exit_time': self.df['timestamp'].iloc[i],
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'exit_price': self.df['close'].iloc[i],
            'size': pos['size'],
            'pnl': -pos['margin'],
            'pnl_pct': -1.0,
            'reason': 'FL',
            'balance_after': self.balance,
        }
        self.trades.append(trade)

        self.consec_losses += 1
        if self.cfg.get('consec_loss_pause') and self.consec_losses >= self.cfg['consec_loss_pause']:
            self.pause_until_idx = i + self.cfg.get('pause_duration_candles', 288)
            self.consec_losses = 0

        self.position = None

    def _record_month(self, month_key):
        """월별 손익 기록"""
        pnl = self.balance - self.month_start_balance
        pnl_pct = pnl / self.month_start_balance if self.month_start_balance > 0 else 0
        self.monthly_pnl[month_key] = {
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'balance': self.balance,
        }

    def _summary(self):
        """결과 요약"""
        trades = self.trades
        if not trades:
            return {'error': 'No trades'}

        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]

        total_profit = sum(t['pnl'] for t in wins)
        total_loss = abs(sum(t['pnl'] for t in losses))
        pf = total_profit / total_loss if total_loss > 0 else float('inf')

        sl_count = len([t for t in trades if t['reason'] == 'SL'])
        tsl_count = len([t for t in trades if t['reason'] == 'TSL'])
        rev_count = len([t for t in trades if t['reason'] == 'REV'])
        fl_count = len([t for t in trades if t['reason'] == 'FL'])

        # 연도별 수익
        yearly = {}
        for t in trades:
            year = t['exit_time'].year
            if year not in yearly:
                yearly[year] = 0
            yearly[year] += t['pnl']

        # 월별 수익/손실 월 수
        profit_months = sum(1 for v in self.monthly_pnl.values() if v['pnl'] > 0)
        loss_months = sum(1 for v in self.monthly_pnl.values() if v['pnl'] < 0)

        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0

        return {
            'final_balance': round(self.balance, 2),
            'return_pct': round((self.balance / self.initial_balance - 1) * 100, 1),
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0,
            'pf': round(pf, 2),
            'mdd': round(self.mdd * 100, 1),
            'fl_count': fl_count,
            'sl_count': sl_count,
            'tsl_count': tsl_count,
            'rev_count': rev_count,
            'avg_win_pct': round(avg_win * 100, 2),
            'avg_loss_pct': round(avg_loss * 100, 2),
            'profit_months': profit_months,
            'loss_months': loss_months,
            'monthly_pnl': self.monthly_pnl,
            'yearly_pnl': yearly,
            'trades': trades,
        }


# ============================================================
# 4. 버전별 설정
# ============================================================

CONFIG_V13_5 = {
    'name': 'v13.5',
    'timeframe': '5m',
    'ema_fast': 7,
    'ema_slow': 100,
    'adx_min': 30,
    'rsi_min': 30,
    'rsi_max': 58,
    'sl_pct': 0.07,
    'trail_activate': 0.08,
    'trail_pct': 0.06,
    'leverage': 10,
    'margin': 0.20,
    'margin_reduced': 0.10,
    'fee_rate': 0.0004,
    'monthly_loss_limit': -0.20,
    'consec_loss_pause': 3,
    'pause_duration_candles': 288,
    'dd_threshold': -0.50,
}

CONFIG_V14_4 = {
    'name': 'v14.4',
    'timeframe': '30m',
    'ema_fast': 3,
    'ema_slow': 200,
    'adx_min': 35,
    'rsi_min': 30,
    'rsi_max': 65,
    'sl_pct': 0.07,
    'trail_activate': 0.06,
    'trail_pct': 0.03,
    'leverage': 10,
    'margin': 0.25,
    'fee_rate': 0.0004,
    'monthly_loss_limit': -0.20,
}

CONFIG_V15_2 = {
    'name': 'v15.2',
    'timeframe': '30m',
    'ema_fast': 3,
    'ema_slow': 200,
    'adx_min': 35,
    'rsi_min': 30,
    'rsi_max': 65,
    'sl_pct': 0.05,
    'trail_activate': 0.06,
    'trail_pct': 0.05,
    'leverage': 10,
    'margin': 0.30,
    'fee_rate': 0.0004,
    'monthly_loss_limit': -0.15,
    'entry_delay_candles': 6,
}

CONFIG_V15_4 = {
    'name': 'v15.4',
    'timeframe': '30m',
    'ema_fast': 3,
    'ema_slow': 200,
    'adx_min': 35,
    'rsi_min': 30,
    'rsi_max': 65,
    'sl_pct': 0.07,
    'trail_activate': 0.06,
    'trail_pct': 0.03,
    'leverage': 10,
    'margin': 0.40,
    'fee_rate': 0.0004,
    'monthly_loss_limit': -0.30,
}


# ============================================================
# 5. 실행
# ============================================================
def run_backtest(config, df_5m, df_30m):
    """단일 버전 백테스트 실행"""
    name = config['name']
    print(f"\n{'='*60}")
    print(f"  {name} 백테스트 시작")
    print(f"{'='*60}")

    # 타임프레임 선택
    if config['timeframe'] == '5m':
        df = df_5m.copy()
    else:
        df = df_30m.copy()

    # 지표 추가
    df = add_indicators(df, config['ema_fast'], config['ema_slow'])

    # 워밍업 (EMA(200) 안정화를 위해 최소 300캔들)
    warmup = max(config['ema_slow'] * 2, 300)
    df = df.iloc[warmup:].reset_index(drop=True)
    print(f"  워밍업 {warmup}캔들 제거 후 {len(df):,}행")

    # 백테스트 실행
    engine = BacktestEngine(config, df)
    result = engine.run()

    return result


def print_result(config, result):
    """결과 출력"""
    name = config['name']
    print(f"\n{'='*60}")
    print(f"  {name} 결과")
    print(f"{'='*60}")

    if 'error' in result:
        print(f"  ERROR: {result['error']}")
        return

    # 기획서 예상값
    expected = {
        'v13.5': {'balance': 468530, 'return': 15518, 'trades': 313, 'pf': 6.87, 'mdd': 74.3, 'fl': 1},
        'v14.4': {'balance': 837212, 'return': 27807, 'trades': 105, 'pf': 2.04, 'mdd': 36.9, 'fl': 0},
        'v15.2': {'balance': 243482, 'return': 8016, 'trades': 66, 'pf': 2.48, 'mdd': 27.6, 'fl': 0},
        'v15.4': {'balance': 8717659, 'return': 290489, 'trades': 105, 'pf': 1.65, 'mdd': 54.2, 'fl': 0},
    }
    exp = expected.get(name, {})

    print(f"  {'항목':<20} {'백테스트 결과':>18} {'기획서 예상':>18} {'차이':>12}")
    print(f"  {'-'*68}")
    print(f"  {'최종 잔액':.<20} ${result['final_balance']:>16,.0f}  ${exp.get('balance',0):>16,}  {result['final_balance']/exp.get('balance',1)*100-100:>+10.1f}%")
    print(f"  {'수익률':.<20} {result['return_pct']:>16,.1f}%  {exp.get('return',0):>16,}%  {result['return_pct']-exp.get('return',0):>+10.1f}%p")
    print(f"  {'거래 수':.<20} {result['total_trades']:>16}  {exp.get('trades',0):>16}  {result['total_trades']-exp.get('trades',0):>+10}")
    print(f"  {'승률':.<20} {result['win_rate']:>16.1f}%")
    print(f"  {'PF':.<20} {result['pf']:>16.2f}  {exp.get('pf',0):>16.2f}  {result['pf']-exp.get('pf',0):>+10.2f}")
    print(f"  {'MDD':.<20} {result['mdd']:>16.1f}%  {exp.get('mdd',0):>16.1f}%  {result['mdd']-exp.get('mdd',0):>+10.1f}%p")
    print(f"  {'강제청산':.<20} {result['fl_count']:>16}  {exp.get('fl',0):>16}  {result['fl_count']-exp.get('fl',0):>+10}")
    print(f"  {'SL 횟수':.<20} {result['sl_count']:>16}")
    print(f"  {'TSL 횟수':.<20} {result['tsl_count']:>16}")
    print(f"  {'REV 횟수':.<20} {result['rev_count']:>16}")
    print(f"  {'평균 승':.<20} {result['avg_win_pct']:>+16.2f}%")
    print(f"  {'평균 패':.<20} {result['avg_loss_pct']:>+16.2f}%")
    print(f"  {'수익 월 / 손실 월':.<20} {result['profit_months']:>7} / {result['loss_months']:<7}")

    # 연도별 누적 잔액
    print(f"\n  연도별 성과:")
    yearly_balance = 3000
    for year in sorted(result['yearly_pnl'].keys()):
        pnl = result['yearly_pnl'][year]
        ret = pnl / yearly_balance * 100 if yearly_balance > 0 else 0
        yearly_balance += pnl
        print(f"    {year}: PnL ${pnl:>+14,.0f}  ({ret:>+8.1f}%)  잔액 ${yearly_balance:>14,.0f}")


def print_comparison(results):
    """4개 버전 비교표"""
    print(f"\n{'='*80}")
    print(f"  4개 버전 비교 검증 결과")
    print(f"{'='*80}")

    header = f"  {'항목':<14} {'v13.5':>16} {'v14.4':>16} {'v15.2':>16} {'v15.4':>16}"
    print(header)
    print(f"  {'-'*76}")

    names = ['v13.5', 'v14.4', 'v15.2', 'v15.4']

    def row(label, key, fmt=',.0f', prefix='', suffix=''):
        vals = []
        for n in names:
            r = results.get(n, {})
            v = r.get(key, 0)
            vals.append(f"{prefix}{v:{fmt}}{suffix}")
        print(f"  {label:<14} {vals[0]:>16} {vals[1]:>16} {vals[2]:>16} {vals[3]:>16}")

    row('최종 잔액', 'final_balance', ',.0f', '$')
    row('수익률(%)', 'return_pct', ',.1f', '', '%')
    row('거래 수', 'total_trades', 'd')
    row('승률(%)', 'win_rate', '.1f', '', '%')
    row('PF', 'pf', '.2f')
    row('MDD(%)', 'mdd', '.1f', '', '%')
    row('강제청산', 'fl_count', 'd')
    row('SL', 'sl_count', 'd')
    row('TSL', 'tsl_count', 'd')
    row('REV', 'rev_count', 'd')

    # 기획서 대비 오차율
    print(f"\n  기획서 대비 오차율:")
    expected = {
        'v13.5': {'balance': 468530, 'trades': 313, 'mdd': 74.3, 'fl': 1},
        'v14.4': {'balance': 837212, 'trades': 105, 'mdd': 36.9, 'fl': 0},
        'v15.2': {'balance': 243482, 'trades': 66, 'mdd': 27.6, 'fl': 0},
        'v15.4': {'balance': 8717659, 'trades': 105, 'mdd': 54.2, 'fl': 0},
    }

    print(f"  {'버전':<10} {'잔액 오차':>14} {'거래수 오차':>14} {'MDD 오차':>14} {'FL 오차':>10}")
    for n in names:
        r = results.get(n, {})
        e = expected[n]
        bal_err = (r.get('final_balance',0) / e['balance'] - 1) * 100
        trade_err = r.get('total_trades',0) - e['trades']
        mdd_err = r.get('mdd',0) - e['mdd']
        fl_err = r.get('fl_count',0) - e['fl']
        print(f"  {n:<10} {bal_err:>+13.1f}% {trade_err:>+13d} {mdd_err:>+13.1f}%p {fl_err:>+9d}")


def main():
    base_dir = r'D:\filesystem\futures\btc_V1\test4'

    print("=" * 60)
    print("  BTC/USDT 선물 자동매매 4버전 백테스트")
    print("  v13.5 / v14.4 / v15.2 / v15.4")
    print("=" * 60)

    # 데이터 로드
    print("\n[1/3] 데이터 로드...")
    df_5m = load_data(base_dir)

    print("\n[2/3] 30분봉 리샘플링...")
    df_30m = resample_to_30m(df_5m)

    # 백테스트 실행
    print("\n[3/3] 백테스트 실행...")
    configs = [CONFIG_V13_5, CONFIG_V14_4, CONFIG_V15_2, CONFIG_V15_4]
    results = {}

    for cfg in configs:
        result = run_backtest(cfg, df_5m, df_30m)
        results[cfg['name']] = result
        print_result(cfg, result)

    # 비교표
    print_comparison(results)

    # JSON 저장
    output_file = os.path.join(base_dir, 'backtest_results.json')
    save_results = {}
    for name, r in results.items():
        save_r = {k: v for k, v in r.items() if k not in ('trades', 'monthly_pnl', 'yearly_pnl')}
        # monthly_pnl 변환
        monthly = {}
        for mk, mv in r.get('monthly_pnl', {}).items():
            monthly[mk] = {'pnl': round(mv['pnl'], 2), 'pnl_pct': round(mv['pnl_pct']*100, 2), 'balance': round(mv['balance'], 2)}
        save_r['monthly_pnl'] = monthly
        # yearly 변환
        yearly = {}
        for yk, yv in r.get('yearly_pnl', {}).items():
            yearly[str(yk)] = round(yv, 2)
        save_r['yearly_pnl'] = yearly
        save_r['trade_count_by_reason'] = {
            'SL': r.get('sl_count', 0),
            'TSL': r.get('tsl_count', 0),
            'REV': r.get('rev_count', 0),
            'FL': r.get('fl_count', 0),
        }
        save_results[name] = save_r

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_file}")

    print("\n" + "=" * 60)
    print("  백테스트 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
