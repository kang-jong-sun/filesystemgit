"""
BTC/USDT 선물 백테스트 엔진 v14.4
- 30분봉 EMA(3/200) 크로스 + ADX(14)>=35 + RSI(14) 30~65
- SL -7% | 트레일링 +6%/-3% | 역신호 청산
- 10x 격리마진 25% | 월간 손실 한도 -20%
- 수수료 0.04% (테이커)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple


# ============================================================
# 설정값 (기획서 v14.4 Section 9)
# ============================================================
CONFIG = {
    # 타임프레임
    'TIMEFRAME': '30min',

    # 진입 지표
    'EMA_FAST': 3,
    'EMA_SLOW': 200,
    'ADX_PERIOD': 14,
    'ADX_MIN': 35,
    'RSI_PERIOD': 14,
    'RSI_MIN': 30,
    'RSI_MAX': 65,

    # 청산
    'SL_PCT': 0.07,            # -7%
    'TRAIL_ACTIVATE': 0.06,    # +6%
    'TRAIL_PCT': 0.03,         # -3% (고점 대비)

    # 포지션
    'LEVERAGE': 10,
    'MARGIN_PCT': 0.25,        # 잔액의 25%

    # 보호
    'MONTHLY_LOSS_LIMIT': -0.20,

    # 수수료
    'FEE_RATE': 0.0004,        # 0.04% (바이낸스 VIP0 테이커)

    # 초기 자본
    'INITIAL_BALANCE': 3000.0,
}


# ============================================================
# 지표 계산 (기획서 v14.4 Section 10)
# ============================================================
def calc_ema(series: pd.Series, span: int) -> pd.Series:
    """EMA 계산"""
    return series.ewm(span=span, adjust=False).mean()


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI 계산 (Wilder's Smoothing)"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    gain_avg = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss_avg = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = gain_avg / loss_avg
    rsi = 100 - 100 / (1 + rs)
    return rsi


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ADX 계산 (Wilder's Smoothing)"""
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    # Wilder's Smoothing (alpha = 1/period)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr

    # DX -> ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    return adx


def prepare_data(filepath: str) -> pd.DataFrame:
    """5분봉 CSV를 30분봉으로 리샘플링 + 지표 계산"""
    print(f"데이터 로딩: {filepath}")
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"  5분봉: {len(df):,}행 ({df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]})")

    # 30분봉 리샘플링
    df = df.set_index('timestamp')
    df_30m = df.resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trades': 'sum'
    }).dropna()

    print(f"  30분봉: {len(df_30m):,}행 ({df_30m.index[0]} ~ {df_30m.index[-1]})")

    # 지표 계산
    df_30m['ema_fast'] = calc_ema(df_30m['close'], CONFIG['EMA_FAST'])
    df_30m['ema_slow'] = calc_ema(df_30m['close'], CONFIG['EMA_SLOW'])
    df_30m['adx'] = calc_adx(df_30m['high'], df_30m['low'], df_30m['close'], CONFIG['ADX_PERIOD'])
    df_30m['rsi'] = calc_rsi(df_30m['close'], CONFIG['RSI_PERIOD'])

    # 이전 EMA 값 (크로스 감지용)
    df_30m['prev_ema_fast'] = df_30m['ema_fast'].shift(1)
    df_30m['prev_ema_slow'] = df_30m['ema_slow'].shift(1)

    # 크로스 감지 (기획서 Section 4)
    df_30m['cross_up'] = (df_30m['ema_fast'] > df_30m['ema_slow']) & (df_30m['prev_ema_fast'] <= df_30m['prev_ema_slow'])
    df_30m['cross_dn'] = (df_30m['ema_fast'] < df_30m['ema_slow']) & (df_30m['prev_ema_fast'] >= df_30m['prev_ema_slow'])

    # EMA 200 워밍업 제거
    df_30m = df_30m.iloc[CONFIG['EMA_SLOW']:]

    print(f"  지표 계산 완료 (워밍업 {CONFIG['EMA_SLOW']}봉 제거 → {len(df_30m):,}행)")
    return df_30m.reset_index()


# ============================================================
# 백테스트 엔진
# ============================================================
class Position:
    """포지션 정보"""
    def __init__(self, direction: str, entry_price: float, size_usd: float,
                 margin: float, entry_time: pd.Timestamp, entry_idx: int):
        self.direction = direction       # 'LONG' or 'SHORT'
        self.entry_price = entry_price
        self.size_usd = size_usd         # 포지션 크기 (USD)
        self.margin = margin             # 증거금
        self.entry_time = entry_time
        self.entry_idx = entry_idx

        # SL 설정 (기획서 Section 6)
        if direction == 'LONG':
            self.sl_price = entry_price * (1 - CONFIG['SL_PCT'])
        else:
            self.sl_price = entry_price * (1 + CONFIG['SL_PCT'])

        # 트레일링 상태
        self.trail_active = False
        self.highest_price = entry_price   # 롱용
        self.lowest_price = entry_price    # 숏용
        self.trail_sl = None

    def calc_roi(self, current_price: float) -> float:
        """현재 ROI 계산 (레버리지 포함)"""
        if self.direction == 'LONG':
            return (current_price - self.entry_price) / self.entry_price * CONFIG['LEVERAGE']
        else:
            return (self.entry_price - current_price) / self.entry_price * CONFIG['LEVERAGE']

    def calc_pnl(self, exit_price: float) -> float:
        """손익 계산 (USD)"""
        if self.direction == 'LONG':
            pnl = self.size_usd * (exit_price - self.entry_price) / self.entry_price
        else:
            pnl = self.size_usd * (self.entry_price - exit_price) / self.entry_price
        return pnl


class BacktestEngine:
    """v14.4 백테스트 엔진"""

    def __init__(self, config: dict = None):
        self.cfg = config or CONFIG
        self.balance = self.cfg['INITIAL_BALANCE']
        self.position: Optional[Position] = None
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

        # 월간 손실 추적
        self.month_start_balance = self.balance
        self.current_month = None
        self.monthly_locked = False

        # 통계
        self.peak_balance = self.balance
        self.max_drawdown = 0.0

    def _fee(self, size_usd: float) -> float:
        """수수료 계산 (진입 + 청산 각 0.04%)"""
        return size_usd * self.cfg['FEE_RATE']

    def _check_monthly_reset(self, timestamp: pd.Timestamp):
        """월간 손실 한도 체크 및 월초 리셋"""
        month_key = (timestamp.year, timestamp.month)
        if self.current_month != month_key:
            self.current_month = month_key
            self.month_start_balance = self.balance
            self.monthly_locked = False

    def _is_monthly_locked(self) -> bool:
        """월간 손실 한도 초과 여부"""
        if self.monthly_locked:
            return True
        if self.month_start_balance > 0:
            monthly_return = (self.balance - self.month_start_balance) / self.month_start_balance
            if monthly_return <= self.cfg['MONTHLY_LOSS_LIMIT']:
                self.monthly_locked = True
                return True
        return False

    def _open_position(self, direction: str, price: float, timestamp: pd.Timestamp, idx: int):
        """포지션 오픈"""
        margin = self.balance * self.cfg['MARGIN_PCT']
        size_usd = margin * self.cfg['LEVERAGE']

        # 진입 수수료
        fee = self._fee(size_usd)
        self.balance -= fee

        self.position = Position(
            direction=direction,
            entry_price=price,
            size_usd=size_usd,
            margin=margin,
            entry_time=timestamp,
            entry_idx=idx
        )

    def _close_position(self, price: float, reason: str, timestamp: pd.Timestamp, idx: int) -> Dict:
        """포지션 클로즈"""
        pos = self.position
        pnl = pos.calc_pnl(price)
        roi = pos.calc_roi(price)

        # 청산 수수료
        fee = self._fee(pos.size_usd)
        net_pnl = pnl - fee

        self.balance += net_pnl

        trade = {
            'entry_time': pos.entry_time,
            'exit_time': timestamp,
            'direction': pos.direction,
            'entry_price': pos.entry_price,
            'exit_price': price,
            'size_usd': pos.size_usd,
            'margin': pos.margin,
            'pnl': net_pnl,
            'roi': roi,
            'reason': reason,
            'balance_after': self.balance,
            'entry_idx': pos.entry_idx,
            'exit_idx': idx,
        }
        self.trades.append(trade)
        self.position = None

        return trade

    def _update_trailing(self, candle_high: float, candle_low: float, close: float):
        """트레일링 SL 업데이트 (기획서 Section 6)"""
        pos = self.position
        if pos is None:
            return

        if pos.direction == 'LONG':
            # 최고가 갱신
            pos.highest_price = max(pos.highest_price, candle_high)
            roi_at_high = (pos.highest_price - pos.entry_price) / pos.entry_price * self.cfg['LEVERAGE']

            # +6% 도달 시 트레일링 활성화
            if roi_at_high >= self.cfg['TRAIL_ACTIVATE'] * self.cfg['LEVERAGE']:
                pos.trail_active = True
                pos.trail_sl = pos.highest_price * (1 - self.cfg['TRAIL_PCT'])
        else:
            # 최저가 갱신
            pos.lowest_price = min(pos.lowest_price, candle_low)
            roi_at_low = (pos.entry_price - pos.lowest_price) / pos.entry_price * self.cfg['LEVERAGE']

            # +6% 도달 시 트레일링 활성화
            if roi_at_low >= self.cfg['TRAIL_ACTIVATE'] * self.cfg['LEVERAGE']:
                pos.trail_active = True
                pos.trail_sl = pos.lowest_price * (1 + self.cfg['TRAIL_PCT'])

    def _check_sl(self, candle_high: float, candle_low: float) -> Optional[Tuple[float, str]]:
        """SL / 트레일링 SL 체크 - 캔들 high/low 기준"""
        pos = self.position
        if pos is None:
            return None

        if pos.direction == 'LONG':
            # 기본 SL
            if candle_low <= pos.sl_price:
                return (pos.sl_price, 'SL')
            # 트레일링 SL (종가 기준 - 기획서 Section 6: "판단: 30분봉 종가")
            # 하지만 캔들 high/low로 추적, 종가로 판단
        else:
            if candle_high >= pos.sl_price:
                return (pos.sl_price, 'SL')

        return None

    def _check_trail_sl_close(self, close: float) -> Optional[Tuple[float, str]]:
        """트레일링 SL 종가 기준 판단 (기획서: "판단: 30분봉 종가")"""
        pos = self.position
        if pos is None or not pos.trail_active:
            return None

        if pos.direction == 'LONG':
            if close <= pos.trail_sl:
                return (close, 'TSL')
        else:
            if close >= pos.trail_sl:
                return (close, 'TSL')

        return None

    def run(self, df: pd.DataFrame, skip_same_direction: bool = False) -> Dict:
        """
        백테스트 실행

        Args:
            df: 30분봉 DataFrame (prepare_data 출력)
            skip_same_direction: True면 동일방향 크로스 시 재진입 스킵 (Claude 로직)

        Returns:
            결과 딕셔너리
        """
        self.balance = self.cfg['INITIAL_BALANCE']
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.peak_balance = self.cfg['INITIAL_BALANCE']
        self.max_drawdown = 0.0
        self.current_month = None
        self.monthly_locked = False

        cross_events = []  # AI 비교용 크로스 이벤트 기록

        for i in range(len(df)):
            row = df.iloc[i]
            ts = row['timestamp']
            close = row['close']
            high = row['high']
            low = row['low']

            # 월간 리셋
            self._check_monthly_reset(ts)

            # ---- 포지션 보유 중 처리 ----
            if self.position is not None:
                # 1순위: 기본 SL 체크 (캔들 high/low)
                sl_result = self._check_sl(high, low)
                if sl_result:
                    self._close_position(sl_result[0], sl_result[1], ts, i)
                    self._update_equity(ts)
                    # SL 후에도 같은 캔들에서 신호 체크 가능 (아래로 계속)
                else:
                    # 트레일링 업데이트
                    self._update_trailing(high, low, close)

                    # 2순위: 트레일링 SL 종가 판단
                    tsl_result = self._check_trail_sl_close(close)
                    if tsl_result:
                        self._close_position(tsl_result[0], tsl_result[1], ts, i)
                        self._update_equity(ts)

            # ---- 신호 감지 ----
            cross_up = row.get('cross_up', False)
            cross_dn = row.get('cross_dn', False)
            adx = row.get('adx', 0)
            rsi = row.get('rsi', 50)

            if cross_up or cross_dn:
                signal = None
                if cross_up and adx >= self.cfg['ADX_MIN'] and self.cfg['RSI_MIN'] <= rsi <= self.cfg['RSI_MAX']:
                    signal = 'LONG'
                elif cross_dn and adx >= self.cfg['ADX_MIN'] and self.cfg['RSI_MIN'] <= rsi <= self.cfg['RSI_MAX']:
                    signal = 'SHORT'

                if signal:
                    # 크로스 이벤트 기록
                    cross_event = {
                        'idx': i,
                        'timestamp': ts,
                        'cross_type': 'golden' if cross_up else 'dead',
                        'signal': signal,
                        'close': close,
                        'adx': adx,
                        'rsi': rsi,
                        'ema_fast': row['ema_fast'],
                        'ema_slow': row['ema_slow'],
                        'current_position': self.position.direction if self.position else None,
                    }
                    cross_events.append(cross_event)

                    # 포지션 처리
                    if self.position is not None:
                        if self.position.direction != signal:
                            # 3순위: 역신호 청산
                            self._close_position(close, 'REV', ts, i)
                        elif skip_same_direction:
                            # 동일방향 크로스 → 스킵 (Claude 로직)
                            self._update_equity(ts)
                            continue
                        else:
                            # 동일방향 크로스 → 기존 청산 후 재진입
                            self._close_position(close, 'REV', ts, i)

                    # 신규 진입 (월간 손실 한도 체크)
                    if self.position is None and not self._is_monthly_locked():
                        self._open_position(signal, close, ts, i)

            self._update_equity(ts)

        # 미청산 포지션 처리
        if self.position is not None:
            last_close = df.iloc[-1]['close']
            last_ts = df.iloc[-1]['timestamp']
            self._close_position(last_close, 'END', last_ts, len(df)-1)

        return self._compile_results(cross_events)

    def _update_equity(self, ts: pd.Timestamp):
        """자산 곡선 업데이트 및 MDD 계산"""
        equity = self.balance
        if self.position:
            # 미실현 손익은 equity에 포함하지 않음 (기획서 기준: 실현 손익 기준)
            pass

        self.peak_balance = max(self.peak_balance, equity)
        if self.peak_balance > 0:
            dd = (self.peak_balance - equity) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, dd)

        self.equity_curve.append({
            'timestamp': ts,
            'balance': self.balance,
            'equity': equity,
            'peak': self.peak_balance,
            'drawdown': dd if self.peak_balance > 0 else 0,
        })

    def _compile_results(self, cross_events: List[Dict]) -> Dict:
        """결과 종합"""
        if not self.trades:
            return {
                'balance': self.balance,
                'return_pct': 0,
                'trades': 0,
                'pf': 0,
                'mdd': 0,
                'win_rate': 0,
                'trade_list': [],
                'cross_events': cross_events,
                'equity_curve': self.equity_curve,
                'monthly_stats': {},
            }

        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]

        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0

        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

        avg_win = np.mean([t['roi'] for t in wins]) * 100 if wins else 0
        avg_loss = np.mean([t['roi'] for t in losses]) * 100 if losses else 0

        # 청산 유형별 카운트
        sl_count = sum(1 for t in self.trades if t['reason'] == 'SL')
        tsl_count = sum(1 for t in self.trades if t['reason'] == 'TSL')
        rev_count = sum(1 for t in self.trades if t['reason'] == 'REV')

        # 월별 통계
        monthly_stats = self._calc_monthly_stats()

        # 연도별 통계
        yearly_stats = self._calc_yearly_stats()

        return {
            'balance': self.balance,
            'return_pct': (self.balance - self.cfg['INITIAL_BALANCE']) / self.cfg['INITIAL_BALANCE'] * 100,
            'trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'pf': pf,
            'mdd': self.max_drawdown * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'payoff_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'sl_count': sl_count,
            'tsl_count': tsl_count,
            'rev_count': rev_count,
            'fl_count': 0,  # 강제청산 (격리마진이므로 SL이 먼저 발동)
            'trade_list': self.trades,
            'cross_events': cross_events,
            'equity_curve': self.equity_curve,
            'monthly_stats': monthly_stats,
            'yearly_stats': yearly_stats,
        }

    def _calc_monthly_stats(self) -> Dict:
        """월별 성과 계산"""
        monthly = {}
        running_balance = self.cfg['INITIAL_BALANCE']

        for t in self.trades:
            month_key = t['entry_time'].strftime('%Y-%m')
            if month_key not in monthly:
                monthly[month_key] = {
                    'start_balance': running_balance,
                    'trades': 0, 'pnl': 0, 'sl': 0, 'tsl': 0, 'rev': 0,
                }
            monthly[month_key]['trades'] += 1
            monthly[month_key]['pnl'] += t['pnl']
            if t['reason'] == 'SL':
                monthly[month_key]['sl'] += 1
            elif t['reason'] == 'TSL':
                monthly[month_key]['tsl'] += 1
            elif t['reason'] == 'REV':
                monthly[month_key]['rev'] += 1
            monthly[month_key]['end_balance'] = t['balance_after']
            running_balance = t['balance_after']

        for k, v in monthly.items():
            if v['start_balance'] > 0:
                v['return_pct'] = (v.get('end_balance', v['start_balance']) - v['start_balance']) / v['start_balance'] * 100
            else:
                v['return_pct'] = 0

        return monthly

    def _calc_yearly_stats(self) -> Dict:
        """연도별 성과 계산"""
        yearly = {}

        # 연도 시작 잔액 추적
        year_start = {}
        balance = self.cfg['INITIAL_BALANCE']

        for t in self.trades:
            year = t['entry_time'].year
            if year not in year_start:
                year_start[year] = balance
                yearly[year] = {
                    'start_balance': balance,
                    'trades': 0, 'pnl': 0, 'sl': 0, 'tsl': 0, 'rev': 0,
                }
            yearly[year]['trades'] += 1
            yearly[year]['pnl'] += t['pnl']
            if t['reason'] == 'SL':
                yearly[year]['sl'] += 1
            elif t['reason'] == 'TSL':
                yearly[year]['tsl'] += 1
            elif t['reason'] == 'REV':
                yearly[year]['rev'] += 1
            yearly[year]['end_balance'] = t['balance_after']
            balance = t['balance_after']

        for k, v in yearly.items():
            if v['start_balance'] > 0:
                v['return_pct'] = (v.get('end_balance', v['start_balance']) - v['start_balance']) / v['start_balance'] * 100

        return yearly


# ============================================================
# 리포트 출력
# ============================================================
def print_report(result: Dict, title: str = "v14.4 백테스트"):
    """결과 리포트 출력"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    print(f"\n[성과 요약]")
    print(f"  최종 잔액:     ${result['balance']:,.0f}")
    print(f"  수익률:        +{result['return_pct']:,.1f}%")
    print(f"  거래 수:       {result['trades']}회")
    print(f"  승률:          {result['win_rate']:.1f}% ({result.get('wins',0)}승 / {result.get('losses',0)}패)")
    print(f"  PF:            {result['pf']:.2f}")
    print(f"  MDD:           {result['mdd']:.1f}%")
    print(f"  평균 승:       +{result.get('avg_win',0):.2f}%")
    print(f"  평균 패:       {result.get('avg_loss',0):.2f}%")
    print(f"  손익비:        {result.get('payoff_ratio',0):.2f}:1")
    print(f"  SL: {result.get('sl_count',0)} | TSL: {result.get('tsl_count',0)} | REV: {result.get('rev_count',0)} | FL: {result.get('fl_count',0)}")

    # 연도별
    if result.get('yearly_stats'):
        print(f"\n[연도별 성과]")
        print(f"  {'연도':<6} {'시작':>12} {'종료':>12} {'수익률':>10} {'거래':>5}")
        print(f"  {'-'*50}")
        for year in sorted(result['yearly_stats'].keys()):
            ys = result['yearly_stats'][year]
            print(f"  {year:<6} ${ys['start_balance']:>10,.0f} ${ys.get('end_balance', ys['start_balance']):>10,.0f} {ys.get('return_pct',0):>+9.1f}% {ys['trades']:>4}")

    # 월별 (요약)
    if result.get('monthly_stats'):
        print(f"\n[월별 성과 (거래 있는 월만)]")
        print(f"  {'월':<8} {'손익률':>8} {'거래':>4} {'SL':>3} {'TSL':>4} {'REV':>4} {'잔액':>12}")
        print(f"  {'-'*55}")
        for month in sorted(result['monthly_stats'].keys()):
            ms = result['monthly_stats'][month]
            if ms['trades'] > 0:
                print(f"  {month:<8} {ms.get('return_pct',0):>+7.1f}% {ms['trades']:>4} {ms['sl']:>3} {ms['tsl']:>4} {ms['rev']:>4} ${ms.get('end_balance',0):>10,.0f}")

    print(f"\n{'='*70}\n")


# ============================================================
# 메인 실행
# ============================================================
def run_backtest(data_path: str = None, skip_same_direction: bool = False) -> Dict:
    """백테스트 실행 (외부 호출용)"""
    if data_path is None:
        data_path = r"D:\filesystem\futures\btc_V1\test1\btc_usdt_5m_merged.csv"

    df = prepare_data(data_path)
    engine = BacktestEngine()
    result = engine.run(df, skip_same_direction=skip_same_direction)
    return result


if __name__ == '__main__':
    import time

    print("=" * 70)
    print("  BTC/USDT v14.4 백테스트 시작")
    print("=" * 70)

    start = time.time()

    # 1. 코드봇 (기본)
    result_code = run_backtest(skip_same_direction=False)
    print_report(result_code, "코드봇 v14.4 (기본)")

    # 2. 동일방향 스킵 (Claude 로직)
    result_skip = run_backtest(skip_same_direction=True)
    print_report(result_skip, "코드봇 v14.4 (동일방향 스킵 = Claude 로직)")

    elapsed = time.time() - start
    print(f"소요시간: {elapsed:.1f}초")

    # 비교
    print(f"\n{'='*70}")
    print(f"  비교 결과")
    print(f"{'='*70}")
    print(f"  {'항목':<15} {'코드봇':>15} {'Claude로직':>15} {'차이':>12}")
    print(f"  {'-'*60}")
    print(f"  {'최종잔액':<15} ${result_code['balance']:>13,.0f} ${result_skip['balance']:>13,.0f} ${result_skip['balance']-result_code['balance']:>+10,.0f}")
    print(f"  {'수익률':<15} {result_code['return_pct']:>+14.1f}% {result_skip['return_pct']:>+14.1f}% {result_skip['return_pct']-result_code['return_pct']:>+11.1f}%p")
    print(f"  {'PF':<15} {result_code['pf']:>15.2f} {result_skip['pf']:>15.2f} {result_skip['pf']-result_code['pf']:>+12.2f}")
    print(f"  {'MDD':<15} {result_code['mdd']:>14.1f}% {result_skip['mdd']:>14.1f}% {result_skip['mdd']-result_code['mdd']:>+11.1f}%p")
    print(f"  {'거래수':<15} {result_code['trades']:>15} {result_skip['trades']:>15} {result_skip['trades']-result_code['trades']:>+12}")
    print(f"  {'SL':<15} {result_code.get('sl_count',0):>15} {result_skip.get('sl_count',0):>15}")
    print(f"  {'TSL':<15} {result_code.get('tsl_count',0):>15} {result_skip.get('tsl_count',0):>15}")
    print(f"  {'REV':<15} {result_code.get('rev_count',0):>15} {result_skip.get('rev_count',0):>15}")
