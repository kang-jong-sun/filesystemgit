"""
BTC/USDT 선물 자동매매 - 핵심 트레이딩 로직
v32.2: EMA(100)/EMA(600) Tight-SL Trend System

═══════════════════════════════════════════════════
  가격 기준 (기획서 v32.2 정확 구현)
═══════════════════════════════════════════════════
  진입가       : 종가 (close)
  SL 발동      : 저가/고가 (intrabar) — l_ <= slp 또는 h_ >= slp
  SL 청산가    : SL가 (slp) — 종가가 아닌 설정된 SL 가격
  TA 활성화    : 고가/저가 (intrabar) — 봉 중간에 +12% 도달 시
  TSL 고점추적 : 고가/저가 (TSL 활성 후에만)
  TSL 청산     : 종가 (close) — px <= slp
  REV 판단     : 종가 (close) — EMA cross 확인

  SL/TSL 우선순위:
    TSL 미활성 → SL만 작동 (SL 발동 시 CONTINUE)
    TSL 활성   → SL 비활성, TSL만 작동 (종가 기준)
    TSL 미발동 → REV 체크

  REV 후: CONTINUE 안 함 — 같은 봉에서 재진입 가능
  (실시간 봇에서는 30초 후 다음 루프에서 재진입)
═══════════════════════════════════════════════════

- EMA 크로스 감지 및 감시 메커니즘
- 진입/청산 조건 판단
- SL/TSL/REV 상태 머신
- ADX, RSI, EMA Gap 필터
- 동일방향 재진입 스킵
- 일일 손실 제한
- Trade Statistics 영구 저장 (state/trade_stats.json)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

logger = logging.getLogger('btc_core')

# ═══════════════════════════════════════════════════════════
# 전략 파라미터
# ═══════════════════════════════════════════════════════════
ADX_MIN = 30.0
ADX_RISE_BARS = 6
RSI_MIN = 40.0
RSI_MAX = 80.0
EMA_GAP_MIN = 0.2       # %
MONITOR_WINDOW = 24      # 크로스 후 24봉
SKIP_SAME_DIR = True
SL_PCT = 3.0             # 가격 -3%
TA_PCT = 12.0            # 가격 +12% (TSL 활성화)
TSL_PCT = 9.0            # 고점 대비 -9%
MARGIN_PCT = 0.35        # 잔액의 35%
LEVERAGE = 10
FEE_RATE = 0.0004        # 0.04%
DAILY_LOSS_LIMIT = -0.20 # 일일 -20%


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
    """포지션 상태"""
    direction: int = 0          # 1=LONG, -1=SHORT, 0=없음
    entry_price: float = 0.0
    position_size: float = 0.0  # 금액 단위 (마진 * 레버리지)
    margin_used: float = 0.0    # 사용 마진
    sl_price: float = 0.0
    tsl_active: bool = False
    track_high: float = 0.0     # TSL용 고점 추적
    track_low: float = 999999.0 # TSL용 저점 추적
    entry_time: float = 0.0     # 진입 시각 (timestamp)
    entry_bar: int = 0          # 진입 봉 인덱스
    peak_roi: float = 0.0       # 최고 ROI 기록


@dataclass
class WatchState:
    """크로스 감시 상태"""
    direction: int = 0      # 감시 방향 (1=LONG, -1=SHORT, 0=없음)
    start_bar: int = 0      # 크로스 발생 봉 인덱스
    start_time: float = 0.0 # 크로스 발생 시각


@dataclass
class TradeRecord:
    """거래 기록"""
    direction: int
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    exit_type: str          # 'SL', 'TSL', 'REV', 'MANUAL'
    entry_time: str
    exit_time: str
    roi_pct: float
    peak_roi: float
    tsl_active: bool
    source: str = 'BOT'     # 'BOT' or 'USER'


@dataclass
class Signal:
    """트레이딩 신호"""
    action: str             # 'ENTER', 'EXIT', 'NONE'
    direction: int = 0      # 진입 방향 or 청산 방향
    exit_type: str = ''     # 'SL', 'TSL', 'REV'
    exit_price: float = 0.0 # SL 청산가
    reason: str = ''


class TradingCore:
    """핵심 트레이딩 로직 — 기획서 v32.2 정확 구현"""

    def __init__(self):
        self.position = PositionState()
        self.watch = WatchState()
        self.last_exit_dir: int = 0
        self.last_exit_bar: int = 0
        self.daily_start_capital: float = 0.0
        self.daily_start_bar: int = 0
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

    def set_daily_start(self, capital: float, bar_index: int):
        """일일 시작 잔액 설정"""
        self.daily_start_capital = capital
        self.daily_start_bar = bar_index

    def update_peak(self, capital: float):
        """고점 잔액 및 MDD 갱신"""
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
        """
        매 봉(또는 틱)마다 호출하여 신호 판단.
        bar: get_current_bar()에서 반환된 데이터
        capital: 현재 잔액
        bar_index: 현재 봉 인덱스
        """
        if bar is None:
            return Signal(action='NONE')

        # ── STEP A: 포지션 보유 시 — 청산 체크 ──
        if self.has_position:
            return self._check_exit(bar, capital, bar_index)

        # ── STEP B: 포지션 없음 — 진입 체크 ──
        return self._check_entry(bar, capital, bar_index)

    # ═══════════════════════════════════════════════════════════
    # 청산 로직
    # ═══════════════════════════════════════════════════════════
    def _check_exit(self, bar: dict, capital: float, bar_index: int) -> Signal:
        """포지션 보유 중 청산 조건 체크"""
        pos = self.position
        px = bar['close']
        h_ = bar['high']
        l_ = bar['low']

        # 포지션 보유 중 감시 리셋
        self.watch.direction = 0

        # A1: SL 체크 (TSL 미활성 시에만)
        # ★ 발동: 저가/고가 기준 (intrabar) | 청산가: SL가 (slp)
        if not pos.tsl_active:
            sl_hit = False
            if pos.direction == 1 and l_ <= pos.sl_price:    # 저가가 SL 이하
                sl_hit = True
            elif pos.direction == -1 and h_ >= pos.sl_price:  # 고가가 SL 이상
                sl_hit = True

            if sl_hit:
                return Signal(
                    action='EXIT',
                    direction=pos.direction,
                    exit_type='SL',
                    exit_price=pos.sl_price,  # ★ SL가에 청산 (종가 아님)
                    reason=f"SL 발동 (SL가={pos.sl_price:.2f})"
                )

        # A2: TA 활성화 체크
        # ★ 고가/저가 기준 (intrabar) — 봉 중간에 도달해도 활성화
        if pos.direction == 1:
            br = (h_ - pos.entry_price) / pos.entry_price * 100   # 고가 기준
        else:
            br = (pos.entry_price - l_) / pos.entry_price * 100   # 저가 기준

        if br >= TA_PCT and not pos.tsl_active:
            pos.tsl_active = True
            logger.info(f"TSL 활성화! (수익률 {br:.1f}% >= {TA_PCT}%)")

        # ROI 추적 (고가/저가 기준)
        if br > pos.peak_roi:
            pos.peak_roi = br

        # A3: TSL 체크 (TSL 활성 시)
        # ★ 고점추적: 고가/저가 (intrabar) | 청산: 종가 기준 (px)
        if pos.tsl_active:
            if pos.direction == 1:
                if h_ > pos.track_high:
                    pos.track_high = h_            # ★ 고가로 고점 갱신
                ns = pos.track_high * (1 - TSL_PCT / 100)
                if ns > pos.sl_price:
                    pos.sl_price = ns
                if px <= pos.sl_price:             # ★ 종가 기준 TSL 체크
                    return Signal(
                        action='EXIT',
                        direction=pos.direction,
                        exit_type='TSL',
                        exit_price=px,
                        reason=f"TSL 발동 (고점={pos.track_high:.2f}, TSL가={pos.sl_price:.2f})"
                    )
            else:
                if l_ < pos.track_low:
                    pos.track_low = l_             # ★ 저가로 저점 갱신
                ns = pos.track_low * (1 + TSL_PCT / 100)
                if ns < pos.sl_price:
                    pos.sl_price = ns
                if px >= pos.sl_price:             # ★ 종가 기준 TSL 체크
                    return Signal(
                        action='EXIT',
                        direction=pos.direction,
                        exit_type='TSL',
                        exit_price=px,
                        reason=f"TSL 발동 (저점={pos.track_low:.2f}, TSL가={pos.sl_price:.2f})"
                    )

        # A4: REV 체크 (EMA 교차 반전)
        # ★ 종가 기준 EMA cross | REV 후 CONTINUE 안 함 (같은 봉 재진입 가능)
        bull_now = bar['fast_ma'] > bar['slow_ma']
        bull_prev = bar['fast_ma_prev'] > bar['slow_ma_prev']
        cross_down = not bull_now and bull_prev
        cross_up = bull_now and not bull_prev

        if (pos.direction == 1 and cross_down) or (pos.direction == -1 and cross_up):
            return Signal(
                action='EXIT',
                direction=pos.direction,
                exit_type='REV',
                exit_price=px,
                reason=f"REV 반전 신호 (EMA 크로스)"
            )

        return Signal(action='NONE')

    # ═══════════════════════════════════════════════════════════
    # 진입 로직
    # ═══════════════════════════════════════════════════════════
    def _check_entry(self, bar: dict, capital: float, bar_index: int) -> Signal:
        """포지션 없을 때 진입 조건 체크"""
        bull_now = bar['fast_ma'] > bar['slow_ma']
        bull_prev = bar['fast_ma_prev'] > bar['slow_ma_prev']
        cross_up = bull_now and not bull_prev
        cross_down = not bull_now and bull_prev

        # B2: 새 크로스 → 감시 시작
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

        # 감시 중이 아니면 패스
        if self.watch.direction == 0:
            return Signal(action='NONE')

        # 크로스 발생 봉에서는 진입 안함
        if bar_index <= self.watch.start_bar:
            return Signal(action='NONE')

        # B3: 모니터 윈도우 초과
        if bar_index - self.watch.start_bar > MONITOR_WINDOW:
            logger.debug(f"[WATCH] 윈도우 초과 ({bar_index - self.watch.start_bar} > {MONITOR_WINDOW})")
            self.watch.direction = 0
            return Signal(action='NONE')

        # B4: 감시 중 반대 크로스 → 방향 전환
        if self.watch.direction == 1 and cross_down:
            self.watch.direction = -1
            self.watch.start_bar = bar_index
            logger.info(f"[WATCH] SHORT로 전환 (bar={bar_index})")
            return Signal(action='NONE')
        elif self.watch.direction == -1 and cross_up:
            self.watch.direction = 1
            self.watch.start_bar = bar_index
            logger.info(f"[WATCH] LONG으로 전환 (bar={bar_index})")
            return Signal(action='NONE')

        # B5: 동일방향 스킵
        if SKIP_SAME_DIR and self.watch.direction == self.last_exit_dir:
            return Signal(action='NONE', reason='동일방향 스킵')

        # B6: ADX >= 30
        if bar['adx'] < ADX_MIN:
            return Signal(action='NONE', reason=f"ADX {bar['adx']:.1f} < {ADX_MIN}")

        # B7: ADX 상승 (현재 > 6봉 전)
        if ADX_RISE_BARS > 0 and bar['adx_prev6'] > 0:
            if bar['adx'] <= bar['adx_prev6']:
                return Signal(action='NONE', reason=f"ADX 하락 ({bar['adx']:.1f} <= {bar['adx_prev6']:.1f})")

        # B8: RSI 40~80
        if bar['rsi'] < RSI_MIN or bar['rsi'] > RSI_MAX:
            return Signal(action='NONE', reason=f"RSI {bar['rsi']:.1f} 범위 이탈")

        # B9: EMA 갭 >= 0.2%
        if bar['ema_gap'] < EMA_GAP_MIN:
            return Signal(action='NONE', reason=f"EMA Gap {bar['ema_gap']:.3f}% < {EMA_GAP_MIN}%")

        # B10: 일일 손실 제한
        if self.daily_start_capital > 0:
            daily_return = (capital - self.daily_start_capital) / self.daily_start_capital
            if daily_return <= DAILY_LOSS_LIMIT:
                self.watch.direction = 0
                return Signal(action='NONE', reason=f"일일 손실 제한 ({daily_return*100:.1f}%)")

        # B11: 잔액 확인
        if capital <= 0:
            return Signal(action='NONE', reason='잔액 부족')

        # ═══ 진입 신호! ═══
        direction = self.watch.direction
        dir_str = "LONG" if direction == 1 else "SHORT"
        logger.info(f"[SIGNAL] {dir_str} 진입 신호! "
                    f"ADX={bar['adx']:.1f} RSI={bar['rsi']:.1f} Gap={bar['ema_gap']:.3f}%")

        return Signal(
            action='ENTER',
            direction=direction,
            reason=f"{dir_str} 진입 (ADX={bar['adx']:.1f}, RSI={bar['rsi']:.1f}, Gap={bar['ema_gap']:.3f}%)"
        )

    # ═══════════════════════════════════════════════════════════
    # 포지션 상태 관리
    # ═══════════════════════════════════════════════════════════
    def open_position(self, direction: int, entry_price: float,
                      capital: float, bar_index: int) -> dict:
        """
        포지션 진입 처리. 마진/포지션크기 계산 및 SL 설정.
        Returns: {margin, position_size, sl_price, entry_fee}
        """
        margin = capital * MARGIN_PCT
        position_size = margin * LEVERAGE
        entry_fee = position_size * FEE_RATE

        self.position = PositionState(
            direction=direction,
            entry_price=entry_price,
            position_size=position_size,
            margin_used=margin,
            sl_price=entry_price * (1 - SL_PCT / 100) if direction == 1
                     else entry_price * (1 + SL_PCT / 100),
            tsl_active=False,
            track_high=entry_price,
            track_low=entry_price,
            entry_time=time.time(),
            entry_bar=bar_index,
            peak_roi=0.0,
        )
        self.watch.direction = 0

        dir_str = "LONG" if direction == 1 else "SHORT"
        logger.info(f"포지션 오픈: {dir_str} @{entry_price:.2f}, "
                    f"크기=${position_size:,.0f}, SL={self.position.sl_price:.2f}")

        return {
            'margin': margin,
            'position_size': position_size,
            'sl_price': self.position.sl_price,
            'entry_fee': entry_fee,
        }

    def close_position(self, exit_price: float, exit_type: str,
                       bar_index: int) -> dict:
        """
        포지션 청산 처리. PnL 계산 및 통계 갱신.
        Returns: {pnl, exit_fee, direction, entry_price, position_size, ...}
        """
        pos = self.position
        pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.position_size * pos.direction \
              - pos.position_size * FEE_RATE
        exit_fee = pos.position_size * FEE_RATE

        # ROI 계산
        roi_pct = (exit_price - pos.entry_price) / pos.entry_price * pos.direction * 100

        # 통계 갱신
        if exit_type == 'SL':
            self.sl_count += 1
        elif exit_type == 'TSL':
            self.tsl_count += 1
        elif exit_type == 'REV':
            self.rev_count += 1
        # MANUAL은 카운트에 포함하지 않음

        if pnl > 0:
            self.win_count += 1
            self.gross_profit += pnl
        else:
            self.loss_count += 1
            self.gross_loss += abs(pnl)

        self.last_exit_dir = pos.direction
        self.last_exit_bar = bar_index

        # 거래 기록
        record = TradeRecord(
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            position_size=pos.position_size,
            pnl=pnl,
            exit_type=exit_type,
            entry_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pos.entry_time)),
            exit_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            roi_pct=roi_pct,
            peak_roi=pos.peak_roi,
            tsl_active=pos.tsl_active,
        )
        self.trade_history.append(record)

        dir_str = "LONG" if pos.direction == 1 else "SHORT"
        logger.info(f"포지션 청산: {dir_str} {exit_type} @{exit_price:.2f}, "
                    f"PnL=${pnl:,.2f} ({roi_pct:+.2f}%)")

        result = {
            'pnl': pnl,
            'exit_fee': exit_fee,
            'direction': pos.direction,
            'entry_price': pos.entry_price,
            'exit_price': exit_price,
            'position_size': pos.position_size,
            'exit_type': exit_type,
            'roi_pct': roi_pct,
            'peak_roi': pos.peak_roi,
            'tsl_active': pos.tsl_active,
            'hold_time': time.time() - pos.entry_time,
        }

        # 포지션 리셋
        self.position = PositionState()

        return result

    # ─── 수동 포지션 추적 ───
    def track_manual_position(self, direction: int, entry_price: float,
                              size: float, bar_index: int):
        """수동 진입 포지션을 시스템에 등록"""
        self.position = PositionState(
            direction=direction,
            entry_price=entry_price,
            position_size=size,
            margin_used=size / LEVERAGE,
            sl_price=entry_price * (1 - SL_PCT / 100) if direction == 1
                     else entry_price * (1 + SL_PCT / 100),
            tsl_active=False,
            track_high=entry_price,
            track_low=entry_price,
            entry_time=time.time(),
            entry_bar=bar_index,
            peak_roi=0.0,
        )
        dir_str = "LONG" if direction == 1 else "SHORT"
        logger.info(f"수동 포지션 추적 시작: {dir_str} @{entry_price:.2f}, 크기=${size:,.0f}")

    # ─── 상태 저장/복원 (재시작 시 TSL/SL 유지) ───
    STATE_FILE = 'state/position_state.json'
    STATS_FILE = 'state/trade_stats.json'      # ★ 통계 영구 저장 (리셋 후에도 유지)

    def save_state(self):
        """현재 포지션 상태를 파일에 저장"""
        os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)

        # ★ 통계는 항상 저장 (포지션 유무와 무관)
        self._save_stats()

        if not self.has_position:
            if os.path.exists(self.STATE_FILE):
                os.remove(self.STATE_FILE)
                logger.debug("상태 파일 삭제 (포지션 없음)")
            return

        pos = self.position
        state = {
            'direction': pos.direction,
            'entry_price': pos.entry_price,
            'position_size': pos.position_size,
            'margin_used': pos.margin_used,
            'sl_price': pos.sl_price,
            'tsl_active': pos.tsl_active,
            'track_high': pos.track_high,
            'track_low': pos.track_low,
            'entry_time': pos.entry_time,
            'entry_bar': pos.entry_bar,
            'peak_roi': pos.peak_roi,
            'last_exit_dir': self.last_exit_dir,
            'last_exit_bar': self.last_exit_bar,
            'saved_at': time.time(),
        }

        try:
            with open(self.STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"상태 저장: TSL={pos.tsl_active}, SL={pos.sl_price:.2f}")
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
            'last_exit_bar': self.last_exit_bar, 'saved_at': time.time(),
        }
        try:
            with open(self.STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.error(f"통계 저장 오류: {e}")

    def load_state(self) -> bool:
        """저장된 상태를 복원. 성공 시 True 반환."""
        # ★ 통계는 항상 먼저 복원 (포지션 없어도)
        self._load_stats()

        if not os.path.exists(self.STATE_FILE):
            return False

        try:
            with open(self.STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)

            self.position = PositionState(
                direction=state['direction'],
                entry_price=state['entry_price'],
                position_size=state['position_size'],
                margin_used=state.get('margin_used', 0),
                sl_price=state['sl_price'],
                tsl_active=state['tsl_active'],
                track_high=state['track_high'],
                track_low=state['track_low'],
                entry_time=state['entry_time'],
                entry_bar=state.get('entry_bar', 0),
                peak_roi=state.get('peak_roi', 0),
            )

            self.last_exit_dir = state.get('last_exit_dir', 0)
            self.last_exit_bar = state.get('last_exit_bar', 0)

            dir_str = "LONG" if self.position.direction == 1 else "SHORT"
            logger.info(f"상태 복원: {dir_str} @{self.position.entry_price:.2f}, "
                       f"TSL={'ON' if self.position.tsl_active else 'OFF'}, "
                       f"SL={self.position.sl_price:.2f}")
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
            self.last_exit_bar = stats.get('last_exit_bar', 0)
            logger.info(f"통계 복원: {self.total_trades}거래 "
                       f"({self.win_count}W/{self.loss_count}L) "
                       f"PF={self.profit_factor:.2f}")
        except Exception as e:
            logger.error(f"통계 복원 오류: {e}")

    # ─── 상태 조회 ───
    def get_status(self) -> dict:
        """현재 상태 요약"""
        pos = self.position
        return {
            'has_position': self.has_position,
            'direction': pos.direction,
            'direction_str': 'LONG' if pos.direction == 1 else 'SHORT' if pos.direction == -1 else 'NONE',
            'entry_price': pos.entry_price,
            'position_size': pos.position_size,
            'sl_price': pos.sl_price,
            'tsl_active': pos.tsl_active,
            'track_high': pos.track_high,
            'track_low': pos.track_low,
            'peak_roi': pos.peak_roi,
            'watching': self.watch.direction,
            'watching_str': 'LONG' if self.watch.direction == 1 else 'SHORT' if self.watch.direction == -1 else 'NONE',
            'watch_bars': 0,  # 실시간에서 계산 필요
            'total_trades': self.total_trades,
            'sl_count': self.sl_count,
            'tsl_count': self.tsl_count,
            'rev_count': self.rev_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'max_drawdown': self.max_drawdown * 100,
        }
