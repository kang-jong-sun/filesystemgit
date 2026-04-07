"""
ETH/USDT 선물 자동매매 - 핵심 트레이딩 로직
V8.16: EMA(250)/EMA(1575) 10m Trend System — 계정 20% 복리 마진

═══════════════════════════════════════════════════
  가격 기준 (V8.16 기획서 정확 구현)
═══════════════════════════════════════════════════
  진입가       : 종가 (close)
  SL 발동      : 저가/고가 (intrabar) + 실시간 가격 (30초)
  SL 청산가    : SL가 (slp) — 종가가 아닌 설정된 SL 가격
  TA 활성화    : 고가/저가 (intrabar) + 실시간 (30초)
  TSL 고점추적 : 고가/저가 + 실시간 가격
  TSL 청산     : 종가 (close) + 실시간 가격 (30초)
  REV 판단     : 10분봉 종가 (EMA cross 확인)

  SL/TSL 우선순위:
    TSL 미활성 → SL만 작동 (SL가에 청산)
    TSL 활성   → SL 비활성, TSL만 작동 (종가 기준)
    TSL 미발동 → REV 체크

  실시간 체크: check_realtime_exit() — 30초마다 SL/TSL 확인
  봉마감 체크: _check_exit() — 10분봉 완성 시 전체 로직

  필터: ALL OFF (ADX/RSI/EMA Gap 없음)
  마진: 계정 잔액의 20% × 레버리지 10x
  Trade Statistics: state/trade_stats.json (영구 저장)
═══════════════════════════════════════════════════

- EMA 크로스 감지 및 감시 메커니즘
- 진입/청산 조건 판단
- SL/TSL/REV 상태 머신
- 동일방향 재진입 스킵
- 필터 ALL OFF (ADX/RSI/EMA Gap 없음)
- Trade Statistics 영구 저장 (state/trade_stats.json)
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
from typing import Optional

logger = logging.getLogger('eth_core')

# ═══════════════════════════════════════════════════════════
# 전략A 파라미터 — EMA(250)/EMA(1575) 10분봉 (추세 잡기)
# ═══════════════════════════════════════════════════════════
FAST_MA_PERIOD = 250
SLOW_MA_PERIOD = 1575
MONITOR_WINDOW = 18          # 크로스 후 18봉 (3시간)
SKIP_SAME_DIR = True
SL_PCT = 2.0                 # 가격 -2%
TA_PCT = 54.0                # 가격 +54% (TSL 활성화)
TSL_PCT = 2.75               # 고점 대비 -2.75%
MARGIN_PCT = 0.20            # 잔액의 20% (복리)
LEVERAGE = 10
FEE_RATE = 0.0005            # 0.05%
# 필터 ALL OFF

# ═══════════════════════════════════════════════════════════
# 전략B 파라미터 — EMA(9)10m × EMA(100)15m (보완 전략)
# ═══════════════════════════════════════════════════════════
B_SL_PCT = 2.5               # SL -2.5%
B_TA_PCT = 5.0               # TSL 활성화 +5%
B_TSL_PCT = 0.3              # 고점 대비 -0.3% 트레일
B_GAP_MIN = 0.5              # EMA 갭 ≥ 0.5% 필터
# REV 없음 — SL/TSL만 사용


class ExitType(IntEnum):
    NONE = 0
    SL = 1
    TSL = 2
    REV = 3


class Direction(IntEnum):
    NONE = 0
    LONG = 1
    SHORT = -1


@dataclass
class PositionState:
    direction: int = 0
    entry_price: float = 0.0
    position_size: float = 0.0
    margin_used: float = 0.0
    sl_price: float = 0.0
    tsl_active: bool = False
    track_high: float = 0.0
    track_low: float = float('inf')
    entry_time: float = 0.0
    entry_bar: int = 0
    peak_roi: float = 0.0
    entry_mode: str = 'A'     # 'A' = 전략A(추세), 'B' = 전략B(보완)


@dataclass
class WatchState:
    direction: int = 0
    start_bar: int = 0
    start_time: float = 0.0


@dataclass
class TradeRecord:
    direction: int
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    exit_type: str
    entry_time: str
    exit_time: str
    roi_pct: float
    peak_roi: float
    tsl_active: bool
    source: str = 'BOT'


@dataclass
class Signal:
    action: str
    direction: int = 0
    exit_type: str = ''
    exit_price: float = 0.0
    reason: str = ''
    entry_mode: str = 'A'     # 'A' or 'B'


class TradingCore:
    """핵심 트레이딩 로직 — V8.16 기획서 정확 구현"""

    def __init__(self):
        self.position = PositionState()
        self.watch = WatchState()
        self.last_exit_dir: int = 0
        self.last_exit_bar: int = 0
        self.trade_history: list[TradeRecord] = []

        # 통계
        self.sl_count = 0
        self.tsl_count = 0
        self.rev_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.peak_capital = 0.0
        self.max_drawdown = 0.0

    @property
    def has_position(self) -> bool:
        return self.position.direction != 0

    @property
    def total_trades(self) -> int:
        return self.sl_count + self.tsl_count + self.rev_count

    @property
    def profit_factor(self) -> float:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else float('inf')

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total * 100 if total > 0 else 0.0

    def update_peak(self, capital: float):
        if capital > self.peak_capital:
            self.peak_capital = capital
        if self.peak_capital > 0:
            dd = (self.peak_capital - capital) / self.peak_capital
            if dd > self.max_drawdown:
                self.max_drawdown = dd

    # ═══════════════════════════════════════════════════════════
    # 메인 신호 판단
    # ═══════════════════════════════════════════════════════════
    def evaluate(self, bar: dict, capital: float, bar_index: int) -> Signal:
        if bar is None:
            return Signal(action='NONE')

        if self.has_position:
            return self._check_exit(bar, capital, bar_index)

        return self._check_entry(bar, capital, bar_index)

    # ═══════════════════════════════════════════════════════════
    # ★ 실시간 가격 SL/TSL 체크 (30초마다, 봉 완성 안 기다림)
    # ═══════════════════════════════════════════════════════════
    def check_realtime_exit(self, price: float, high: float, low: float) -> Signal:
        """실시간 가격으로 SL/TSL만 체크. REV/진입은 봉마감 기준 유지."""
        if not self.has_position:
            return None

        pos = self.position
        # 전략별 파라미터 선택
        ta_pct = TA_PCT if pos.entry_mode == 'A' else B_TA_PCT
        tsl_pct = TSL_PCT if pos.entry_mode == 'A' else B_TSL_PCT

        # SL 체크 (TSL 미활성 시)
        if not pos.tsl_active:
            if pos.direction == 1 and low <= pos.sl_price:
                return Signal(action='EXIT', direction=pos.direction, exit_type='SL',
                              exit_price=pos.sl_price, reason=f"RT-SL [{pos.entry_mode}] (저가 ${low:,.2f} <= SL ${pos.sl_price:,.2f})")
            elif pos.direction == -1 and high >= pos.sl_price:
                return Signal(action='EXIT', direction=pos.direction, exit_type='SL',
                              exit_price=pos.sl_price, reason=f"RT-SL [{pos.entry_mode}] (고가 ${high:,.2f} >= SL ${pos.sl_price:,.2f})")

        # TA 활성화 체크
        if pos.direction == 1:
            br = (high - pos.entry_price) / pos.entry_price * 100
        else:
            br = (pos.entry_price - low) / pos.entry_price * 100
        if br >= ta_pct and not pos.tsl_active:
            pos.tsl_active = True
            logger.info(f"RT-TSL [{pos.entry_mode}] 활성화! (ROI {br:.1f}% >= {ta_pct}%)")
        if br > pos.peak_roi:
            pos.peak_roi = br

        # TSL 체크
        if pos.tsl_active:
            if pos.direction == 1:
                if high > pos.track_high:
                    pos.track_high = high
                ns = pos.track_high * (1 - tsl_pct / 100)
                if ns > pos.sl_price:
                    pos.sl_price = ns
                if price <= pos.sl_price:
                    return Signal(action='EXIT', direction=pos.direction, exit_type='TSL',
                                  exit_price=price, reason=f"RT-TSL [{pos.entry_mode}] (${price:,.2f} <= TSL ${pos.sl_price:,.2f})")
            else:
                if low < pos.track_low:
                    pos.track_low = low
                ns = pos.track_low * (1 + tsl_pct / 100)
                if ns < pos.sl_price:
                    pos.sl_price = ns
                if price >= pos.sl_price:
                    return Signal(action='EXIT', direction=pos.direction, exit_type='TSL',
                                  exit_price=price, reason=f"RT-TSL [{pos.entry_mode}] (${price:,.2f} >= TSL ${pos.sl_price:,.2f})")

        return None

    # ═══════════════════════════════════════════════════════════
    # 청산 로직 — SL → TA → TSL → REV (봉마감 기준)
    # ═══════════════════════════════════════════════════════════
    def _check_exit(self, bar: dict, capital: float, bar_index: int) -> Signal:
        pos = self.position
        px = bar['close']
        h_ = bar['high']
        l_ = bar['low']

        self.watch.direction = 0

        # 전략별 파라미터
        ta_pct = TA_PCT if pos.entry_mode == 'A' else B_TA_PCT
        tsl_pct = TSL_PCT if pos.entry_mode == 'A' else B_TSL_PCT

        # SL (TSL 미활성 시에만)
        if not pos.tsl_active:
            if pos.direction == 1 and l_ <= pos.sl_price:
                return Signal(action='EXIT', direction=pos.direction, exit_type='SL',
                              exit_price=pos.sl_price, reason=f"SL [{pos.entry_mode}] (SL가={pos.sl_price:.2f})")
            elif pos.direction == -1 and h_ >= pos.sl_price:
                return Signal(action='EXIT', direction=pos.direction, exit_type='SL',
                              exit_price=pos.sl_price, reason=f"SL [{pos.entry_mode}] (SL가={pos.sl_price:.2f})")

        # TA 활성화
        if pos.direction == 1:
            br = (h_ - pos.entry_price) / pos.entry_price * 100
        else:
            br = (pos.entry_price - l_) / pos.entry_price * 100

        if br >= ta_pct and not pos.tsl_active:
            pos.tsl_active = True
            logger.info(f"TSL [{pos.entry_mode}] 활성화! (ROI {br:.1f}% >= {ta_pct}%)")

        if br > pos.peak_roi:
            pos.peak_roi = br

        # TSL
        if pos.tsl_active:
            if pos.direction == 1:
                if h_ > pos.track_high:
                    pos.track_high = h_
                ns = pos.track_high * (1 - tsl_pct / 100)
                if ns > pos.sl_price:
                    pos.sl_price = ns
                if px <= pos.sl_price:
                    return Signal(action='EXIT', direction=pos.direction, exit_type='TSL',
                                  exit_price=px, reason=f"TSL [{pos.entry_mode}] (고점={pos.track_high:.2f})")
            else:
                if l_ < pos.track_low:
                    pos.track_low = l_
                ns = pos.track_low * (1 + tsl_pct / 100)
                if ns < pos.sl_price:
                    pos.sl_price = ns
                if px >= pos.sl_price:
                    return Signal(action='EXIT', direction=pos.direction, exit_type='TSL',
                                  exit_price=px, reason=f"TSL [{pos.entry_mode}] (저점={pos.track_low:.2f})")

        # REV — 전략A만 (전략B는 REV 없음)
        if pos.entry_mode == 'A':
            bull_now = bar['fast_ma'] > bar['slow_ma']
            bull_prev = bar['fast_ma_prev'] > bar['slow_ma_prev']
            cross_down = not bull_now and bull_prev
            cross_up = bull_now and not bull_prev

            if (pos.direction == 1 and cross_down) or (pos.direction == -1 and cross_up):
                return Signal(action='EXIT', direction=pos.direction, exit_type='REV',
                              exit_price=px, reason="REV [A] (EMA 크로스)")

        return Signal(action='NONE')

    # ═══════════════════════════════════════════════════════════
    # 진입 로직 — 필터 ALL OFF
    # ═══════════════════════════════════════════════════════════
    def _check_entry(self, bar: dict, capital: float, bar_index: int) -> Signal:
        bull_now = bar['fast_ma'] > bar['slow_ma']
        bull_prev = bar['fast_ma_prev'] > bar['slow_ma_prev']
        cross_up = bull_now and not bull_prev
        cross_down = not bull_now and bull_prev

        # 크로스 감시
        if cross_up:
            self.watch.direction = 1
            self.watch.start_bar = bar_index
            self.watch.start_time = time.time()
            logger.info(f"[WATCH] LONG 감시 시작 (bar={bar_index})")
        elif cross_down:
            self.watch.direction = -1
            self.watch.start_bar = bar_index
            self.watch.start_time = time.time()
            logger.info(f"[WATCH] SHORT 감시 시작 (bar={bar_index})")

        if self.watch.direction == 0:
            return Signal(action='NONE')

        if bar_index <= self.watch.start_bar:
            return Signal(action='NONE')

        # 모니터 윈도우 초과
        if bar_index - self.watch.start_bar > MONITOR_WINDOW:
            self.watch.direction = 0
            return Signal(action='NONE')

        # 감시 중 반대 크로스 → 방향 전환
        if self.watch.direction == 1 and cross_down:
            self.watch.direction = -1; self.watch.start_bar = bar_index
            return Signal(action='NONE')
        elif self.watch.direction == -1 and cross_up:
            self.watch.direction = 1; self.watch.start_bar = bar_index
            return Signal(action='NONE')

        # Skip same dir
        if SKIP_SAME_DIR and self.watch.direction == self.last_exit_dir:
            return Signal(action='NONE', reason='동일방향 스킵')

        # 잔액 확인
        if capital < 500:
            return Signal(action='NONE', reason='잔액 부족')

        # ═══ 진입! (필터 없음) ═══
        direction = self.watch.direction
        dir_str = "LONG" if direction == 1 else "SHORT"
        logger.info(f"[SIGNAL-A] {dir_str} 진입 신호!")

        return Signal(action='ENTER', direction=direction,
                      reason=f"{dir_str} 진입 [A] (EMA cross)", entry_mode='A')

    # ═══════════════════════════════════════════════════════════
    # 전략B 진입 체크 — EMA(9) 10m × EMA(100) 15m + 갭 필터
    # ═══════════════════════════════════════════════════════════
    def check_entry_b(self, bar: dict, capital: float) -> Signal:
        """전략B 진입 체크 — 봉마감 기준, 포지션 없을 때만 호출"""
        if self.has_position or capital < 500:
            return Signal(action='NONE')

        b_ema9 = bar.get('b_ema9')
        b_ema100 = bar.get('b_ema100_15m')
        b_ema9_prev = bar.get('b_ema9_prev')
        b_ema100_prev = bar.get('b_ema100_15m_prev')

        if b_ema9 is None or b_ema100 is None or b_ema9_prev is None or b_ema100_prev is None:
            return Signal(action='NONE')
        if np.isnan(b_ema9) or np.isnan(b_ema100) or np.isnan(b_ema9_prev) or np.isnan(b_ema100_prev):
            return Signal(action='NONE')

        # 봉마감 기준 크로스 (이전봉에서 확정)
        bull_now = b_ema9_prev > b_ema100_prev
        bull_prev = bar.get('b_ema9_prev2', b_ema9_prev) > bar.get('b_ema100_15m_prev2', b_ema100_prev)

        # prev2 없으면 스킵
        if bar.get('b_ema9_prev2') is None:
            return Signal(action='NONE')

        bull_prev = bar['b_ema9_prev2'] > bar['b_ema100_15m_prev2']
        cross_up = bull_now and not bull_prev
        cross_down = not bull_now and bull_prev

        if not cross_up and not cross_down:
            return Signal(action='NONE')

        # 갭 필터
        gap = abs(b_ema9_prev - b_ema100_prev) / max(b_ema100_prev, 1e-10) * 100
        if gap < B_GAP_MIN:
            return Signal(action='NONE')

        direction = 1 if cross_up else -1
        dir_str = "LONG" if direction == 1 else "SHORT"
        logger.info(f"[SIGNAL-B] {dir_str} 진입 신호! (갭 {gap:.2f}%)")

        return Signal(action='ENTER', direction=direction,
                      reason=f"{dir_str} 진입 [B] (EMA9/100 cross, 갭={gap:.2f}%)",
                      entry_mode='B')

    # ═══════════════════════════════════════════════════════════
    # 포지션 관리
    # ═══════════════════════════════════════════════════════════
    def open_position(self, direction: int, entry_price: float,
                      capital: float, bar_index: int, entry_mode: str = 'A') -> dict:
        margin = capital * MARGIN_PCT
        position_size = margin * LEVERAGE

        # 전략별 SL
        sl_pct = SL_PCT if entry_mode == 'A' else B_SL_PCT

        self.position = PositionState(
            direction=direction, entry_price=entry_price,
            position_size=position_size, margin_used=margin,
            sl_price=entry_price * (1 - sl_pct / 100) if direction == 1
                     else entry_price * (1 + sl_pct / 100),
            tsl_active=False, track_high=entry_price, track_low=entry_price,
            entry_time=time.time(), entry_bar=bar_index, peak_roi=0.0,
            entry_mode=entry_mode,
        )
        self.watch.direction = 0

        dir_str = "LONG" if direction == 1 else "SHORT"
        logger.info(f"포지션 오픈 [{entry_mode}]: {dir_str} @{entry_price:.2f}, "
                    f"크기=${position_size:,.0f}, 마진=${margin:,.0f}, SL={self.position.sl_price:.2f}")

        return {
            'margin': margin,
            'position_size': position_size,
            'sl_price': self.position.sl_price,
            'entry_fee': position_size * FEE_RATE,
            'entry_mode': entry_mode,
        }

    def close_position(self, exit_price: float, exit_type: str,
                       bar_index: int) -> dict:
        pos = self.position
        pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.position_size * pos.direction \
              - pos.position_size * FEE_RATE
        roi_pct = (exit_price - pos.entry_price) / pos.entry_price * pos.direction * 100

        if exit_type == 'SL': self.sl_count += 1
        elif exit_type == 'TSL': self.tsl_count += 1
        elif exit_type == 'REV': self.rev_count += 1

        if pnl > 0:
            self.win_count += 1; self.gross_profit += pnl
        else:
            self.loss_count += 1; self.gross_loss += abs(pnl)

        self.last_exit_dir = pos.direction
        self.last_exit_bar = bar_index

        record = TradeRecord(
            direction=pos.direction, entry_price=pos.entry_price,
            exit_price=exit_price, position_size=pos.position_size,
            pnl=pnl, exit_type=exit_type,
            entry_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pos.entry_time)),
            exit_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            roi_pct=roi_pct, peak_roi=pos.peak_roi, tsl_active=pos.tsl_active,
        )
        self.trade_history.append(record)
        if len(self.trade_history) > 500:
            self.trade_history = self.trade_history[-500:]

        result = {
            'pnl': pnl, 'direction': pos.direction, 'entry_price': pos.entry_price,
            'exit_price': exit_price, 'position_size': pos.position_size,
            'exit_type': exit_type, 'roi_pct': roi_pct, 'peak_roi': pos.peak_roi,
            'tsl_active': pos.tsl_active, 'hold_time': time.time() - pos.entry_time,
        }

        self.position = PositionState()
        return result

    def track_manual_position(self, direction: int, entry_price: float,
                              size: float, bar_index: int):
        self.position = PositionState(
            direction=direction, entry_price=entry_price,
            position_size=size, margin_used=size / LEVERAGE,
            sl_price=entry_price * (1 - SL_PCT / 100) if direction == 1
                     else entry_price * (1 + SL_PCT / 100),
            tsl_active=False, track_high=entry_price, track_low=entry_price,
            entry_time=time.time(), entry_bar=bar_index, peak_roi=0.0,
            entry_mode='A',
        )

    # ─── 상태 저장/복원 ───
    STATE_FILE = 'state/position_state.json'
    STATS_FILE = 'state/trade_stats.json'      # ★ 통계 영구 저장 (리셋 후에도 유지)

    def save_state(self):
        os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)

        # ★ 통계는 항상 저장 (포지션 유무와 무관)
        self._save_stats()

        if not self.has_position:
            if os.path.exists(self.STATE_FILE):
                os.remove(self.STATE_FILE)
            return

        pos = self.position
        state = {
            'direction': pos.direction, 'entry_price': pos.entry_price,
            'position_size': pos.position_size, 'margin_used': pos.margin_used,
            'sl_price': pos.sl_price, 'tsl_active': pos.tsl_active,
            'track_high': pos.track_high, 'track_low': pos.track_low,
            'entry_time': pos.entry_time, 'entry_bar': pos.entry_bar,
            'peak_roi': pos.peak_roi, 'entry_mode': pos.entry_mode,
            'last_exit_dir': self.last_exit_dir, 'saved_at': time.time(),
        }
        try:
            with open(self.STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"상태 저장 오류: {e}")

    def _save_stats(self):
        """★ Trade Statistics 영구 저장 — 리셋되어도 유지"""
        stats = {
            'sl_count': self.sl_count, 'tsl_count': self.tsl_count,
            'rev_count': self.rev_count, 'win_count': self.win_count,
            'loss_count': self.loss_count, 'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss, 'peak_capital': self.peak_capital,
            'max_drawdown': self.max_drawdown, 'last_exit_dir': self.last_exit_dir,
            'saved_at': time.time(),
        }
        try:
            with open(self.STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.error(f"통계 저장 오류: {e}")

    def load_state(self) -> bool:
        # ★ 통계는 항상 먼저 복원 (포지션 없어도)
        self._load_stats()

        if not os.path.exists(self.STATE_FILE):
            return False
        try:
            with open(self.STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            self.position = PositionState(
                direction=state['direction'], entry_price=state['entry_price'],
                position_size=state['position_size'], margin_used=state.get('margin_used', 0),
                sl_price=state['sl_price'], tsl_active=state['tsl_active'],
                track_high=state['track_high'], track_low=state['track_low'],
                entry_time=state['entry_time'], entry_bar=state.get('entry_bar', 0),
                peak_roi=state.get('peak_roi', 0),
                entry_mode=state.get('entry_mode', 'A'),
            )
            self.last_exit_dir = state.get('last_exit_dir', 0)
            return True
        except Exception as e:
            logger.error(f"상태 복원 오류: {e}")
            return False

    def _load_stats(self):
        """★ Trade Statistics 복원 — 포지션 없어도 통계는 유지"""
        if not os.path.exists(self.STATS_FILE):
            return
        try:
            with open(self.STATS_FILE, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            self.sl_count = stats.get('sl_count', 0)
            self.tsl_count = stats.get('tsl_count', 0)
            self.rev_count = stats.get('rev_count', 0)
            self.win_count = stats.get('win_count', 0)
            self.loss_count = stats.get('loss_count', 0)
            self.gross_profit = stats.get('gross_profit', 0)
            self.gross_loss = stats.get('gross_loss', 0)
            self.peak_capital = stats.get('peak_capital', 0)
            self.max_drawdown = stats.get('max_drawdown', 0)
            self.last_exit_dir = stats.get('last_exit_dir', 0)
            logger.info(f"통계 복원: {self.total_trades}거래 "
                       f"({self.win_count}W/{self.loss_count}L) "
                       f"PF={self.profit_factor:.2f}")
        except Exception as e:
            logger.error(f"통계 복원 오류: {e}")

    def get_status(self) -> dict:
        pos = self.position
        return {
            'has_position': self.has_position,
            'direction': pos.direction,
            'direction_str': 'LONG' if pos.direction == 1 else 'SHORT' if pos.direction == -1 else 'NONE',
            'entry_price': pos.entry_price, 'position_size': pos.position_size,
            'sl_price': pos.sl_price, 'tsl_active': pos.tsl_active,
            'track_high': pos.track_high, 'track_low': pos.track_low,
            'peak_roi': pos.peak_roi,
            'watching': self.watch.direction,
            'watching_str': 'LONG' if self.watch.direction == 1 else 'SHORT' if self.watch.direction == -1 else 'NONE',
            'total_trades': self.total_trades,
            'sl_count': self.sl_count, 'tsl_count': self.tsl_count, 'rev_count': self.rev_count,
            'win_count': self.win_count, 'loss_count': self.loss_count,
            'win_rate': self.win_rate, 'profit_factor': self.profit_factor,
            'gross_profit': self.gross_profit, 'gross_loss': self.gross_loss,
            'max_drawdown': self.max_drawdown * 100,
        }
