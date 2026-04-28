"""
SOL/USDT 선물 자동매매 - 핵심 트레이딩 로직
V1: V12:Mass 75:25 Mutex + Skip2@4loss + 12.5% Compound

═══════════════════════════════════════════════════
  전략 구성
═══════════════════════════════════════════════════
  Strategy A (75% alloc): V12.1 추세추종
    - EMA(9) / SMA(400) 15m cross
    - 5중 필터: ADX≥22 rising + RSI 30-65 + LR±0.5 + Skip Same + ADX rising 6
    - Entry Delay 5 bars, Monitor 24 bars
    - Pyramiding 3-Leg (Leg2 +3%, Leg3 +7.5%)
    - Confidence Sigmoid k=5, base 0.5, range 1.85
    - Session Split (US aggressive)
    - ATR Expansion Defense

  Strategy B (25% alloc): Mass Index Reversal
    - Mass Index 25-bar sum > 27 (bulge forms)
    - Mass < 26.5 (bulge ends → reversal trigger)

  Mutex: 동시 진입 금지 (한쪽 활성 시 타쪽 무시)
  Skip2@4loss: 4연패 후 다음 2거래 스킵
  Margin: balance × 12.5% × allocation (복리)

═══════════════════════════════════════════════════
  PDF 제약
═══════════════════════════════════════════════════
  Init: $5,000
  Leverage: 10x
  Margin: isolated
  Fee + Slip: 0.05% × 2 (taker)
  Margin Scaling: Bull 1.5× / Bear 0.5× (BTC 1H EMA200)

═══════════════════════════════════════════════════
  검증 성과 (51개월 백테스트)
═══════════════════════════════════════════════════
  Return: +175,456% ($5k → $8.77M)
  MDD: 40.6%
  PF: 2.07
  WR: 30.2%
  Calmar: 4,323

  ⚠ 12.5% 복리는 10%보다 공격적. MDD 40%+ 감당 가능 시 사용.
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Optional, List
import numpy as np

logger = logging.getLogger('sol_core')
signal_log = logging.getLogger('sol.signals')       # 신호 통과/실패 이유
summary_log = logging.getLogger('sol.summary')      # 일간 요약
trade_log = logging.getLogger('sol.trades')         # 거래 이벤트 (apply_entry/exit용)

# ═══════════════════════════════════════════════════════════
# Strategy A: V12.1 파라미터 (EMA9/SMA400 15m)
# ═══════════════════════════════════════════════════════════
FAST_MA_PERIOD = 9               # EMA(9)
SLOW_MA_PERIOD = 400             # SMA(400)
TIMEFRAME = '15min'              # 15분봉

# Entry Filters
ADX_MIN = 22.0                   # ADX ≥ 22
ADX_RISING_BARS = 6              # ADX rising 6 bars
RSI_MIN = 30.0                   # RSI 30-65
RSI_MAX = 65.0
LR_MIN = -0.5                    # LR slope ±0.5
LR_MAX = 0.5
SKIP_SAME_DIR = True             # 동일 방향 재진입 스킵
MONITOR_WIN = 24                 # 크로스 후 24바 관찰
ENTRY_DELAY = 5                  # 5바 delay

# Exit Parameters (Asia/EU session default)
SL_PCT_BASE = 3.8                # SL -3.8%
TA_PCT_BASE = 5.6                # TA +5.6% (TSL 활성화)
TSL_PCT_BASE = 10.0              # TSL -10% from peak
USE_REV = True                   # EMA9 reverse cross → 반전 청산

# US Session (UTC 16-24) Aggressive
SL_PCT_US = 4.0
TA_PCT_US = 5.6
TSL_PCT_US = 11.0

# Pyramiding 3-Leg
USE_PYRAMIDING = True
LEG2_PCT_BASE = 3.0              # Leg2 @ +3%
LEG3_PCT_BASE = 7.5              # Leg3 @ +7.5%
LEG2_PCT_US = 3.2
LEG3_PCT_US = 8.0
LEG_MARGIN_RATIO = 0.5           # Leg2/Leg3 margin = base × 0.5

# Confidence Sigmoid
CONF_BASE = 0.5
CONF_RANGE = 1.85                # multiplier range 0.5~2.35
CONF_K = 5.0                     # Sigmoid steepness
CONF_WEIGHTS = {'adx': 0.4, 'rsi': 0.3, 'lr': 0.3}

# Margin Scaling (BTC 1H EMA200)
BULL_MULT = 1.5                  # Bull regime: margin × 1.5
BEAR_MULT = 0.5                  # Bear regime: margin × 0.5

# ATR Expansion Defense
ATR_EXPANSION_THRESHOLD = 1.5    # ATR(14) > 1.5 × ATR(50) → margin flat

# ═══════════════════════════════════════════════════════════
# Strategy B: Mass Index 파라미터
# ═══════════════════════════════════════════════════════════
MASS_EMA_PERIOD = 9              # EMA(H-L, 9)
MASS_SUM_PERIOD = 25             # Sum 25 bars
MASS_BULGE_THRESHOLD = 27.0      # Bulge forms > 27
MASS_REVERSAL_THRESHOLD = 26.5   # Bulge ends < 26.5

MASS_SL_PCT = 3.0                # Mass SL -3%
MASS_TA_PCT = 5.0                # Mass TA +5%
MASS_TSL_PCT = 8.0               # Mass TSL -8%

# ═══════════════════════════════════════════════════════════
# Global Parameters
# ═══════════════════════════════════════════════════════════
LEVERAGE = 2                     # 🧪 TEST MODE: 2x (원래 10x)
FEE_RATE = 0.0005                # 0.05% taker
SLIPPAGE = 0.0005                # 0.05% slippage

# Compound Config (V33 최적)
COMPOUND_PCT = 0.125             # 12.5% of balance
ALLOC_V12 = 0.75                 # Strategy A 75%
ALLOC_MASS = 0.25                # Strategy B 25%
MAX_CAPITAL = 1_000_000          # $1M cap (바이낸스 SOL 유동성 0.5% 깊이 $11M 대비)
                                 # balance가 $1M 초과 시 margin 계산에 $1M만 사용

# Skip2@4loss
SKIP_LOSS_THRESHOLD = 4          # 4연패 → trigger
SKIP_TRADE_COUNT = 2             # 다음 2거래 스킵

# Daily Loss Limit (emergency)
DAILY_LOSS_LIMIT = -0.025        # -2.5% 일일 손실 한도

STATE_PATH = 'state/sol_state.json'


# ═══════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════
def is_us_session(timestamp_utc: float) -> bool:
    """UTC 16:00-24:00 = US Session"""
    import datetime as dt
    hour = dt.datetime.utcfromtimestamp(timestamp_utc).hour
    return 16 <= hour < 24


def compute_confidence_score(adx: float, rsi: float, lr_slope: float) -> float:
    """V12.1 Confidence Score (0~1)"""
    if math.isnan(adx) or math.isnan(rsi) or math.isnan(lr_slope):
        return 0.5
    adx_s = min(max((adx - 22) / 18, 0), 1)       # 22~40 → 0~1
    rsi_s = max(0, min(1 - abs(rsi - 50) / 20, 1)) # 50이 최고
    lr_s = max(0, min(1 - abs(lr_slope) / 0.5, 1)) # 낮을수록 좋음
    w = CONF_WEIGHTS
    score = w['adx'] * adx_s + w['rsi'] * rsi_s + w['lr'] * lr_s
    return max(0, min(score, 1))


def confidence_multiplier(score: float) -> float:
    """Sigmoid k=5, base 0.5, range 1.85 → [0.5, 2.35]"""
    sig = 1.0 / (1.0 + math.exp(-CONF_K * (score - 0.5)))
    return CONF_BASE + CONF_RANGE * sig


class ExitType(IntEnum):
    NONE = 0
    SL = 1
    TSL = 2
    REV = 3
    FC = 4  # Force Close


class Direction(IntEnum):
    NONE = 0
    LONG = 1
    SHORT = -1


class EntryMode(IntEnum):
    NONE = 0
    V12 = 1   # Strategy A
    MASS = 2  # Strategy B


@dataclass
class PyramidLeg:
    done: bool = False
    price: float = 0.0
    margin: float = 0.0


@dataclass
class PositionState:
    direction: int = 0               # 1=LONG, -1=SHORT, 0=NONE
    entry_mode: int = 0              # 1=V12, 2=MASS
    entry_price: float = 0.0         # VWAP (Pyramiding 반영)
    entry_price_leg1: float = 0.0    # Leg1 원본 가격
    position_size: float = 0.0       # notional USD
    margin_used: float = 0.0         # 실 사용 마진 USD
    sl_price: float = 0.0
    tsl_active: bool = False
    track_high: float = 0.0
    track_low: float = float('inf')
    entry_time: float = 0.0
    entry_bar: int = 0
    peak_roi: float = 0.0
    # 파라미터 (진입 시점 스냅샷)
    sl_pct: float = 0.0
    ta_pct: float = 0.0
    tsl_pct: float = 0.0
    leg2_pct: float = 0.0
    leg3_pct: float = 0.0
    leg2_margin: float = 0.0
    leg3_margin: float = 0.0
    use_pyr: bool = False
    leg2: PyramidLeg = field(default_factory=PyramidLeg)
    leg3: PyramidLeg = field(default_factory=PyramidLeg)
    leg_count: int = 1
    # Sizing
    conf_score: float = 0.0
    conf_mult: float = 1.0
    margin_mult: float = 1.0


@dataclass
class WatchState:
    direction: int = 0
    start_bar: int = 0
    start_time: float = 0.0
    entry_mode: int = 0


@dataclass
class Signal:
    action: str                     # 'NONE', 'ENTRY', 'EXIT', 'PYRAMID'
    direction: int = 0
    entry_mode: int = 0             # 1=V12, 2=MASS
    exit_type: str = ''
    exit_price: float = 0.0
    entry_price: float = 0.0
    sl_price: float = 0.0
    margin_usd: float = 0.0
    position_size_usd: float = 0.0
    reason: str = ''
    # Entry metadata
    conf_score: float = 0.0
    conf_mult: float = 1.0
    margin_mult: float = 1.0


@dataclass
class TradeRecord:
    direction: int
    entry_mode: int
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    exit_type: str
    entry_time: str
    exit_time: str
    roi_pct: float
    peak_roi: float
    leg_count: int = 1


class TradingCore:
    """V12:Mass 75:25 Mutex + Skip2@4loss + 12.5% 복리 트레이딩 로직"""

    def __init__(self):
        self.position = PositionState()
        self.watch_v12 = WatchState()
        self.last_exit_dir: int = 0
        self.last_exit_bar: int = 0

        # Skip2@4loss 상태
        self.consec_losses: int = 0
        self.skip_remaining: int = 0

        # Mass Index 상태 (bulge tracking)
        self.mass_bulge_active: bool = False

        # 통계
        self.sl_count = 0
        self.tsl_count = 0
        self.rev_count = 0
        self.fc_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.peak_capital = 0.0
        self.max_drawdown = 0.0
        self.trade_history: List[TradeRecord] = []

        # Daily loss tracking (SOL 전용 PnL 기준 — ETH 공용 계정에서도 독립)
        self.day_start_balance: float = 0.0        # (호환성, 참고용)
        self.day_start_date: str = ''
        self.cumulative_sol_pnl: float = 0.0       # ★ 봇 시작 이후 SOL 누적 PnL
        self.day_start_sol_pnl: float = 0.0        # ★ 당일 시작 시점의 cumulative_sol_pnl
        self.daily_trade_count: int = 0            # ★ 당일 거래 수 (자정마다 리셋)
        # Daily summary 누적 카운터 (전일 요약용)
        self.day_start_win_count: int = 0
        self.day_start_loss_count: int = 0

    @property
    def has_position(self) -> bool:
        return self.position.direction != 0

    @property
    def total_trades(self) -> int:
        return self.sl_count + self.tsl_count + self.rev_count + self.fc_count

    @property
    def profit_factor(self) -> float:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else float('inf')

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total * 100 if total > 0 else 0.0

    @property
    def is_skipping(self) -> bool:
        return self.skip_remaining > 0

    def update_peak(self, capital: float):
        if capital > self.peak_capital:
            self.peak_capital = capital
        if self.peak_capital > 0:
            dd = (self.peak_capital - capital) / self.peak_capital
            if dd > self.max_drawdown:
                self.max_drawdown = dd

    def _check_daily_reset(self, capital: float):
        import datetime as dt
        today = dt.datetime.utcnow().strftime('%Y-%m-%d')
        if today != self.day_start_date:
            # ★ 전날 daily_summary 기록 (첫 실행 제외)
            if self.day_start_date:
                yesterday_pnl = self.cumulative_sol_pnl - self.day_start_sol_pnl
                yesterday_wins = self.win_count - self.day_start_win_count
                yesterday_losses = self.loss_count - self.day_start_loss_count
                yesterday_trades = self.daily_trade_count
                pct_change = ((capital - self.day_start_balance) / self.day_start_balance * 100
                              if self.day_start_balance > 0 else 0.0)
                wr_day = (yesterday_wins / yesterday_trades * 100
                          if yesterday_trades > 0 else 0.0)
                summary_log.info(
                    f"{self.day_start_date} | StartBal ${self.day_start_balance:,.2f} | "
                    f"EndBal ${capital:,.2f} ({pct_change:+.2f}%) | "
                    f"Trades {yesterday_trades} (W{yesterday_wins}/L{yesterday_losses}, WR {wr_day:.0f}%) | "
                    f"SOL_PnL ${yesterday_pnl:+,.2f} | "
                    f"ConsecLoss {self.consec_losses} | Skip {self.skip_remaining}"
                )
            # 리셋
            self.day_start_date = today
            self.day_start_balance = capital                       # 호환성 유지
            self.day_start_sol_pnl = self.cumulative_sol_pnl       # ★ SOL 전용 스냅샷
            self.day_start_win_count = self.win_count
            self.day_start_loss_count = self.loss_count
            self.daily_trade_count = 0

    def _daily_loss_exceeded(self, current_capital: float) -> bool:
        """SOL 봇의 당일 손실만 체크 (ETH 등 공용 계정 영향 배제)
        - 누적 SOL PnL의 당일 변화분만 계산
        - 기준자본: min(current_capital, MAX_CAPITAL) — 유동성 cap 반영
        """
        if current_capital <= 0:
            return False
        sol_daily_pnl = self.cumulative_sol_pnl - self.day_start_sol_pnl
        ref_capital = min(current_capital, MAX_CAPITAL)
        if ref_capital <= 0:
            return False
        pct = sol_daily_pnl / ref_capital
        exceeded = pct <= DAILY_LOSS_LIMIT
        if exceeded:
            logger.warning(f"⚠ Daily Loss Limit HIT | SOL일일PnL ${sol_daily_pnl:+,.2f} / ref ${ref_capital:,.0f} = {pct*100:+.2f}% ≤ {DAILY_LOSS_LIMIT*100:.1f}%")
        return exceeded

    # ═══════════════════════════════════════════════════════════
    # 메인 신호 판단
    # ═══════════════════════════════════════════════════════════
    def evaluate(self, bar: dict, bar_idx: int, capital: float,
                 btc_regime_bull: bool,
                 atr14: float, atr50: float,
                 indicators: dict,
                 mass_index: float) -> Signal:
        """
        bar: {timestamp, open, high, low, close}
        indicators: {adx, rsi, lr_slope, fast_ma, slow_ma, prev_fast_ma, prev_slow_ma}
        """
        if bar is None:
            return Signal(action='NONE')

        self._check_daily_reset(capital)

        # 포지션 보유 중 → 청산 로직
        if self.has_position:
            return self._check_exit(bar, bar_idx, indicators)

        # 진입 로직
        return self._check_entry(bar, bar_idx, capital, btc_regime_bull,
                                  atr14, atr50, indicators, mass_index)

    # ═══════════════════════════════════════════════════════════
    # 청산 로직 (실시간 가격 포함)
    # ═══════════════════════════════════════════════════════════
    def check_realtime_exit(self, price: float, high: float, low: float) -> Optional[Signal]:
        """실시간 가격 SL/TSL 체크 (봉 마감 안 기다림)"""
        if not self.has_position:
            return None

        pos = self.position

        # SL check (TSL 미활성 시만)
        if not pos.tsl_active:
            if pos.direction == 1 and low <= pos.sl_price:
                return Signal(action='EXIT', direction=pos.direction, exit_type='SL',
                              exit_price=pos.sl_price,
                              reason=f"RT-SL [{'V12' if pos.entry_mode == 1 else 'MASS'}] low ${low:.3f} <= SL ${pos.sl_price:.3f}")
            elif pos.direction == -1 and high >= pos.sl_price:
                return Signal(action='EXIT', direction=pos.direction, exit_type='SL',
                              exit_price=pos.sl_price,
                              reason=f"RT-SL [{'V12' if pos.entry_mode == 1 else 'MASS'}] high ${high:.3f} >= SL ${pos.sl_price:.3f}")

        # TA 활성화 check
        if pos.direction == 1:
            br = (high - pos.entry_price) / pos.entry_price * 100
        else:
            br = (pos.entry_price - low) / pos.entry_price * 100

        if br >= pos.ta_pct and not pos.tsl_active:
            pos.tsl_active = True
            logger.info(f"RT-TSL 활성화 [{'V12' if pos.entry_mode == 1 else 'MASS'}] ROI {br:.1f}% >= {pos.ta_pct}%")

        if br > pos.peak_roi:
            pos.peak_roi = br

        # TSL check
        if pos.tsl_active:
            if pos.direction == 1:
                if high > pos.track_high:
                    pos.track_high = high
                ns = pos.track_high * (1 - pos.tsl_pct / 100)
                if ns > pos.sl_price:
                    pos.sl_price = ns
                if price <= pos.sl_price:
                    return Signal(action='EXIT', direction=pos.direction, exit_type='TSL',
                                  exit_price=price,
                                  reason=f"RT-TSL ${price:.3f} <= TSL ${pos.sl_price:.3f}")
            else:
                if low < pos.track_low:
                    pos.track_low = low
                ns = pos.track_low * (1 + pos.tsl_pct / 100)
                if ns < pos.sl_price:
                    pos.sl_price = ns
                if price >= pos.sl_price:
                    return Signal(action='EXIT', direction=pos.direction, exit_type='TSL',
                                  exit_price=price,
                                  reason=f"RT-TSL ${price:.3f} >= TSL ${pos.sl_price:.3f}")

        return None

    # ═══════════════════════════════════════════════════════════
    # 봉마감 청산 + Pyramiding + REV 판단
    # ═══════════════════════════════════════════════════════════
    def _check_exit(self, bar: dict, bar_idx: int, indicators: dict) -> Signal:
        pos = self.position
        px = bar['close']
        h_ = bar['high']
        l_ = bar['low']

        # Pyramiding (V12만)
        if pos.use_pyr and pos.entry_mode == EntryMode.V12 and bar_idx > pos.entry_bar:
            if pos.leg_count < 3:
                target2 = pos.entry_price_leg1 * (1 + pos.leg2_pct/100.0) if pos.direction == 1 else pos.entry_price_leg1 * (1 - pos.leg2_pct/100.0)
                target3 = pos.entry_price_leg1 * (1 + pos.leg3_pct/100.0) if pos.direction == 1 else pos.entry_price_leg1 * (1 - pos.leg3_pct/100.0)

                if not pos.leg2.done:
                    hit2 = (pos.direction == 1 and h_ >= target2) or (pos.direction == -1 and l_ <= target2)
                    if hit2:
                        return Signal(action='PYRAMID', direction=pos.direction,
                                      entry_price=target2, margin_usd=pos.leg2_margin,
                                      reason=f"Pyramid Leg2 @ ${target2:.3f}")

                if pos.leg2.done and not pos.leg3.done:
                    hit3 = (pos.direction == 1 and h_ >= target3) or (pos.direction == -1 and l_ <= target3)
                    if hit3:
                        return Signal(action='PYRAMID', direction=pos.direction,
                                      entry_price=target3, margin_usd=pos.leg3_margin,
                                      reason=f"Pyramid Leg3 @ ${target3:.3f}")

        # SL check (TSL 미활성)
        if not pos.tsl_active:
            if pos.direction == 1 and l_ <= pos.sl_price:
                return Signal(action='EXIT', direction=pos.direction, exit_type='SL',
                              exit_price=pos.sl_price,
                              reason=f"SL hit low ${l_:.3f} <= ${pos.sl_price:.3f}")
            elif pos.direction == -1 and h_ >= pos.sl_price:
                return Signal(action='EXIT', direction=pos.direction, exit_type='SL',
                              exit_price=pos.sl_price,
                              reason=f"SL hit high ${h_:.3f} >= ${pos.sl_price:.3f}")

        # TA activation
        if pos.direction == 1:
            br = (h_ - pos.entry_price) / pos.entry_price * 100
        else:
            br = (pos.entry_price - l_) / pos.entry_price * 100
        if br >= pos.ta_pct and not pos.tsl_active:
            pos.tsl_active = True
            logger.info(f"TSL 활성화 [{'V12' if pos.entry_mode == 1 else 'MASS'}] ROI {br:.1f}%")
        if br > pos.peak_roi:
            pos.peak_roi = br

        # TSL check
        if pos.tsl_active:
            if pos.direction == 1:
                if h_ > pos.track_high:
                    pos.track_high = h_
                ns = pos.track_high * (1 - pos.tsl_pct / 100)
                if ns > pos.sl_price:
                    pos.sl_price = ns
                if l_ <= pos.sl_price:
                    return Signal(action='EXIT', direction=pos.direction, exit_type='TSL',
                                  exit_price=pos.sl_price,
                                  reason=f"TSL hit ${l_:.3f} <= ${pos.sl_price:.3f}")
            else:
                if l_ < pos.track_low:
                    pos.track_low = l_
                ns = pos.track_low * (1 + pos.tsl_pct / 100)
                if ns < pos.sl_price:
                    pos.sl_price = ns
                if h_ >= pos.sl_price:
                    return Signal(action='EXIT', direction=pos.direction, exit_type='TSL',
                                  exit_price=pos.sl_price,
                                  reason=f"TSL hit ${h_:.3f} >= ${pos.sl_price:.3f}")

        # REV (V12만, 봉마감 기준) — 진단 로그 강화 (2026-04-28)
        if USE_REV and pos.entry_mode == EntryMode.V12 and not pos.tsl_active:
            fm = indicators.get('fast_ma')
            sm = indicators.get('slow_ma')
            pfm = indicators.get('prev_fast_ma')
            psm = indicators.get('prev_slow_ma')
            pos_dir = 'LONG' if pos.direction == 1 else 'SHORT'

            # 데이터 누락 진단
            if fm is None or sm is None or pfm is None or psm is None:
                signal_log.info(f"V12 REV-SKIP @ bar#{bar_idx} | indicators 누락 "
                                f"(fm={fm} sm={sm} pfm={pfm} psm={psm})")
            elif math.isnan(fm) or math.isnan(sm) or math.isnan(pfm) or math.isnan(psm):
                signal_log.info(f"V12 REV-SKIP @ bar#{bar_idx} | indicators NaN "
                                f"(fm={fm} sm={sm} pfm={pfm} psm={psm})")
            else:
                up = fm > sm and pfm <= psm
                dn = fm < sm and pfm >= psm
                # ★ Cross 감지 시 진단 로그 (REV 발동 여부와 무관하게 항상 기록)
                if up or dn:
                    cross_str = 'CROSS_UP' if up else 'CROSS_DN'
                    signal_log.info(f"V12 REV-CHECK {cross_str} @ bar#{bar_idx} | "
                                    f"EMA9 ${fm:.3f} vs SMA400 ${sm:.3f} | "
                                    f"prev EMA9 ${pfm:.3f} vs SMA400 ${psm:.3f} | "
                                    f"position={pos_dir} tsl_active={pos.tsl_active}")
                # REV 발동
                if (pos.direction == 1 and dn) or (pos.direction == -1 and up):
                    signal_log.info(f"V12 REV TRIGGERED @ bar#{bar_idx} price ${px:.3f} | "
                                    f"position={pos_dir} → EXIT "
                                    f"(EMA9 ${fm:.3f} {'<' if dn else '>'} SMA400 ${sm:.3f}, 반대 방향)")
                    return Signal(action='EXIT', direction=pos.direction, exit_type='REV',
                                  exit_price=px, reason=f"REV (EMA9 cross {pos_dir} → {'SHORT' if dn else 'LONG'})")
        elif USE_REV and pos.entry_mode == EntryMode.V12 and pos.tsl_active:
            # TSL 활성 시 REV 비활성 — 1회만 로깅 (peak_roi 도달 시점 직후)
            pass  # TSL 우선이라 정상

        return Signal(action='NONE')

    # ═══════════════════════════════════════════════════════════
    # 진입 로직
    # ═══════════════════════════════════════════════════════════
    def _check_entry(self, bar: dict, bar_idx: int, capital: float,
                     btc_regime_bull: bool, atr14: float, atr50: float,
                     indicators: dict, mass_index: float) -> Signal:

        # Daily loss limit
        if self._daily_loss_exceeded(capital):
            # 하루 중 여러 번 호출되므로 첫 감지에만 로깅 (1일 1회)
            if not getattr(self, '_daily_loss_logged', False):
                sol_daily_pnl = self.cumulative_sol_pnl - self.day_start_sol_pnl
                signal_log.info(f"GATE daily_loss_limit @ bar#{bar_idx} | SOL일일PnL ${sol_daily_pnl:+,.2f} ≤ {DAILY_LOSS_LIMIT*100:.1f}% | 거래 중지")
                self._daily_loss_logged = True
            return Signal(action='NONE', reason='Daily loss limit exceeded')
        else:
            self._daily_loss_logged = False

        # Skip2@4loss active?
        if self.skip_remaining > 0:
            # skip_remaining 변할 때만 로깅 (tick마다 아님)
            if getattr(self, '_last_skip_logged', -1) != self.skip_remaining:
                signal_log.info(f"GATE skip2@4loss @ bar#{bar_idx} | {self.skip_remaining} trades remaining to skip")
                self._last_skip_logged = self.skip_remaining
            return Signal(action='NONE', reason=f'Skip2@4loss active ({self.skip_remaining} remaining)')
        else:
            self._last_skip_logged = -1

        # Mutex: 이미 포지션 있으면 진입 불가
        if self.has_position:
            return Signal(action='NONE')

        # === Strategy A: V12.1 Entry Check ===
        v12_sig = self._check_v12_entry(bar, bar_idx, capital, btc_regime_bull,
                                         atr14, atr50, indicators)
        if v12_sig.action == 'ENTRY':
            return v12_sig

        # === Strategy B: Mass Index Entry Check ===
        mass_sig = self._check_mass_entry(bar, bar_idx, capital, mass_index)
        return mass_sig

    def _check_v12_entry(self, bar: dict, bar_idx: int, capital: float,
                          btc_regime_bull: bool, atr14: float, atr50: float,
                          indicators: dict) -> Signal:
        """V12.1 진입 로직"""
        fm = indicators.get('fast_ma')
        sm = indicators.get('slow_ma')
        pfm = indicators.get('prev_fast_ma')
        psm = indicators.get('prev_slow_ma')

        if any(v is None or math.isnan(v) for v in [fm, sm, pfm, psm]):
            return Signal(action='NONE')

        # EMA9/SMA400 cross 감지
        cross_up = fm > sm and pfm <= psm
        cross_dn = fm < sm and pfm >= psm
        if cross_up:
            self.watch_v12 = WatchState(direction=1, start_bar=bar_idx,
                                         start_time=bar['timestamp'], entry_mode=1)
            signal_log.info(f"V12 CROSS_UP @ bar#{bar_idx} price ${bar['close']:.3f} | EMA9 ${fm:.3f} > SMA400 ${sm:.3f} | watch started (LONG)")
            self.save_state()  # ★ watch 변경 즉시 디스크 저장 (재시작 대비)
        elif cross_dn:
            self.watch_v12 = WatchState(direction=-1, start_bar=bar_idx,
                                         start_time=bar['timestamp'], entry_mode=1)
            signal_log.info(f"V12 CROSS_DN @ bar#{bar_idx} price ${bar['close']:.3f} | EMA9 ${fm:.3f} < SMA400 ${sm:.3f} | watch started (SHORT)")
            self.save_state()  # ★ watch 변경 즉시 디스크 저장

        # Watch 만료 체크
        if self.watch_v12.direction == 0:
            return Signal(action='NONE')

        # ★ TIME 기반 체크 (재시작 후 bar_idx 리셋 문제 해결)
        # MONITOR_WIN = 24봉 × 15분 = 21,600초 (6시간)
        # ENTRY_DELAY = 5봉 × 15분 = 4,500초 (75분)
        MONITOR_WIN_SEC = MONITOR_WIN * 15 * 60
        ENTRY_DELAY_SEC = ENTRY_DELAY * 15 * 60
        elapsed_sec = bar['timestamp'] - self.watch_v12.start_time

        if elapsed_sec > MONITOR_WIN_SEC:
            dir_str = 'LONG' if self.watch_v12.direction == 1 else 'SHORT'
            signal_log.info(f"V12 WATCH_EXPIRED {dir_str} | {elapsed_sec/60:.0f}min (>{MONITOR_WIN_SEC/60:.0f}min) passed without entry")
            self.watch_v12 = WatchState()
            self.save_state()  # ★ 만료 후 저장
            return Signal(action='NONE')

        # Entry delay 대기 (timestamp 기반, 재시작 후에도 정확)
        if elapsed_sec <= ENTRY_DELAY_SEC:
            return Signal(action='NONE')

        dir_str = 'LONG' if self.watch_v12.direction == 1 else 'SHORT'

        # Skip Same Direction
        if SKIP_SAME_DIR and self.watch_v12.direction == self.last_exit_dir:
            signal_log.info(f"V12 SKIP {dir_str} @ bar#{bar_idx} | Skip Same Direction (last_exit_dir={self.last_exit_dir})")
            return Signal(action='NONE', reason='Skip same direction')

        # Filters
        adx_v = indicators.get('adx')
        rsi_v = indicators.get('rsi')
        lr_v = indicators.get('lr_slope')
        adx_prev = indicators.get('adx_prev_6')  # ADX 6 bars ago

        if any(v is None or math.isnan(v) for v in [adx_v, rsi_v, lr_v]):
            return Signal(action='NONE')
        if adx_v < ADX_MIN:
            signal_log.info(f"V12 SKIP {dir_str} @ bar#{bar_idx} | ADX {adx_v:.1f} < {ADX_MIN}")
            return Signal(action='NONE', reason=f'ADX {adx_v:.1f} < {ADX_MIN}')
        if not (RSI_MIN <= rsi_v <= RSI_MAX):
            signal_log.info(f"V12 SKIP {dir_str} @ bar#{bar_idx} | RSI {rsi_v:.1f} out of range ({RSI_MIN}-{RSI_MAX})")
            return Signal(action='NONE', reason=f'RSI {rsi_v:.1f} out of range')
        if not (LR_MIN <= lr_v <= LR_MAX):
            signal_log.info(f"V12 SKIP {dir_str} @ bar#{bar_idx} | LR {lr_v:+.3f} out of range ({LR_MIN:+.1f}/{LR_MAX:+.1f})")
            return Signal(action='NONE', reason=f'LR {lr_v:.3f} out of range')

        # ADX rising check (6 bars)
        if adx_prev is not None and not math.isnan(adx_prev):
            if adx_v <= adx_prev:
                signal_log.info(f"V12 SKIP {dir_str} @ bar#{bar_idx} | ADX not rising ({adx_v:.1f} ≤ prev6 {adx_prev:.1f})")
                return Signal(action='NONE', reason='ADX not rising')

        # === Position Size Calculation ===
        # Session-based parameters
        us = is_us_session(bar['timestamp'])
        sl_pct = SL_PCT_US if us else SL_PCT_BASE
        ta_pct = TA_PCT_US if us else TA_PCT_BASE
        tsl_pct = TSL_PCT_US if us else TSL_PCT_BASE
        leg2_pct = LEG2_PCT_US if us else LEG2_PCT_BASE
        leg3_pct = LEG3_PCT_US if us else LEG3_PCT_BASE

        # Margin Scaling
        margin_mult = BULL_MULT if btc_regime_bull else BEAR_MULT

        # ATR Expansion Defense
        if not math.isnan(atr14) and not math.isnan(atr50) and atr50 > 0:
            if atr14 > ATR_EXPANSION_THRESHOLD * atr50:
                margin_mult = 1.0  # Flat
                logger.info(f"ATR Expansion Defense 활성 (ATR14 {atr14:.4f} > 1.5x ATR50 {atr50:.4f})")

        # Confidence Sigmoid
        score = compute_confidence_score(adx_v, rsi_v, lr_v)
        conf_mult = confidence_multiplier(score)

        # ★ MAX_CAPITAL cap 적용 (바이낸스 유동성 대비)
        effective_capital = min(capital, MAX_CAPITAL)

        # Final margin = effective × compound_pct × alloc_v12 × margin_mult × conf_mult
        base_margin = effective_capital * COMPOUND_PCT * ALLOC_V12
        margin = base_margin * margin_mult * conf_mult

        # Safety: min/max bounds
        if margin < 10.0 or margin > capital * 0.80:
            return Signal(action='NONE', reason=f'Margin ${margin:.0f} out of bounds')

        # Position size = margin × leverage
        position_size = margin * LEVERAGE
        entry_price = bar['close']
        direction = self.watch_v12.direction

        # Apply slippage
        exec_price = entry_price * (1 + SLIPPAGE) if direction == 1 else entry_price * (1 - SLIPPAGE)
        sl_price = exec_price * (1 - sl_pct/100) if direction == 1 else exec_price * (1 + sl_pct/100)

        # Leg margins
        leg2_margin = margin * LEG_MARGIN_RATIO
        leg3_margin = margin * LEG_MARGIN_RATIO

        # Build Signal
        sig = Signal(
            action='ENTRY', direction=direction, entry_mode=int(EntryMode.V12),
            entry_price=exec_price, sl_price=sl_price,
            margin_usd=margin, position_size_usd=position_size,
            conf_score=score, conf_mult=conf_mult, margin_mult=margin_mult,
            reason=f"V12 {'LONG' if direction == 1 else 'SHORT'} | ADX {adx_v:.1f} | RSI {rsi_v:.1f} | Score {score:.2f} | Mult {conf_mult:.2f}",
        )
        # Stash session params in signal for apply_entry
        sig.__dict__['sl_pct'] = sl_pct
        sig.__dict__['ta_pct'] = ta_pct
        sig.__dict__['tsl_pct'] = tsl_pct
        sig.__dict__['leg2_pct'] = leg2_pct
        sig.__dict__['leg3_pct'] = leg3_pct
        sig.__dict__['leg2_margin'] = leg2_margin
        sig.__dict__['leg3_margin'] = leg3_margin

        # Reset watch (진입 후 저장)
        self.watch_v12 = WatchState()
        self.save_state()  # ★ 진입 후 watch 리셋도 즉시 저장
        return sig

    def _check_mass_entry(self, bar: dict, bar_idx: int, capital: float, mass_index: float) -> Signal:
        """Mass Index Reversal 진입"""
        if math.isnan(mass_index):
            return Signal(action='NONE')

        # Bulge 형성 (Mass > 27)
        if mass_index > MASS_BULGE_THRESHOLD:
            if not self.mass_bulge_active:  # 처음 형성되는 순간만 로깅
                signal_log.info(f"Mass BULGE_FORMED @ bar#{bar_idx} | Mass {mass_index:.2f} > {MASS_BULGE_THRESHOLD} | watching for reversal")
            self.mass_bulge_active = True
            return Signal(action='NONE')

        # Bulge 종료 (Mass < 26.5) → 반전 진입
        if self.mass_bulge_active and mass_index < MASS_REVERSAL_THRESHOLD:
            self.mass_bulge_active = False
            signal_log.info(f"Mass REVERSAL_TRIGGER @ bar#{bar_idx} | Mass {mass_index:.2f} < {MASS_REVERSAL_THRESHOLD}")

            # 반전 방향: 최근 추세 반대
            # 단순화: 이전 last_exit_dir의 반대 (없으면 SHORT 디폴트)
            direction = -1 if self.last_exit_dir == 1 else 1
            dir_str = 'LONG' if direction == 1 else 'SHORT'

            # Skip Same
            if direction == self.last_exit_dir:
                signal_log.info(f"Mass SKIP {dir_str} @ bar#{bar_idx} | Skip Same Direction (last_exit_dir={self.last_exit_dir})")
                return Signal(action='NONE', reason='Mass: skip same direction')

            # ★ MAX_CAPITAL cap 적용 (바이낸스 유동성 대비)
            effective_capital = min(capital, MAX_CAPITAL)

            # Position size
            base_margin = effective_capital * COMPOUND_PCT * ALLOC_MASS
            margin = base_margin  # Mass는 Confidence Sizing 미적용

            if margin < 10.0 or margin > capital * 0.30:
                return Signal(action='NONE', reason=f'Mass margin out of bounds')

            position_size = margin * LEVERAGE
            entry_price = bar['close']
            exec_price = entry_price * (1 + SLIPPAGE) if direction == 1 else entry_price * (1 - SLIPPAGE)
            sl_price = exec_price * (1 - MASS_SL_PCT/100) if direction == 1 else exec_price * (1 + MASS_SL_PCT/100)

            sig = Signal(
                action='ENTRY', direction=direction, entry_mode=int(EntryMode.MASS),
                entry_price=exec_price, sl_price=sl_price,
                margin_usd=margin, position_size_usd=position_size,
                reason=f"Mass Reversal {'LONG' if direction == 1 else 'SHORT'} | Mass {mass_index:.2f}",
            )
            sig.__dict__['sl_pct'] = MASS_SL_PCT
            sig.__dict__['ta_pct'] = MASS_TA_PCT
            sig.__dict__['tsl_pct'] = MASS_TSL_PCT
            return sig

        return Signal(action='NONE')

    # ═══════════════════════════════════════════════════════════
    # Entry/Exit 적용
    # ═══════════════════════════════════════════════════════════
    def apply_entry(self, sig: Signal, bar_idx: int, timestamp: float):
        """진입 완료 후 상태 업데이트"""
        p = PositionState(
            direction=sig.direction,
            entry_mode=sig.entry_mode,
            entry_price=sig.entry_price,
            entry_price_leg1=sig.entry_price,
            position_size=sig.position_size_usd,
            margin_used=sig.margin_usd,
            sl_price=sig.sl_price,
            tsl_active=False,
            track_high=sig.entry_price,
            track_low=sig.entry_price,
            entry_time=timestamp,
            entry_bar=bar_idx,
            peak_roi=0.0,
            sl_pct=sig.__dict__.get('sl_pct', SL_PCT_BASE),
            ta_pct=sig.__dict__.get('ta_pct', TA_PCT_BASE),
            tsl_pct=sig.__dict__.get('tsl_pct', TSL_PCT_BASE),
            leg2_pct=sig.__dict__.get('leg2_pct', LEG2_PCT_BASE),
            leg3_pct=sig.__dict__.get('leg3_pct', LEG3_PCT_BASE),
            leg2_margin=sig.__dict__.get('leg2_margin', 0.0),
            leg3_margin=sig.__dict__.get('leg3_margin', 0.0),
            use_pyr=(sig.entry_mode == int(EntryMode.V12)) and USE_PYRAMIDING,
            conf_score=sig.conf_score,
            conf_mult=sig.conf_mult,
            margin_mult=sig.margin_mult,
        )
        self.position = p
        logger.info(f"✅ ENTRY [{'V12' if sig.entry_mode==1 else 'MASS'}] "
                    f"{'LONG' if sig.direction==1 else 'SHORT'} @ ${sig.entry_price:.3f} | "
                    f"Size ${sig.position_size_usd:,.0f} | Margin ${sig.margin_usd:,.0f} | SL ${sig.sl_price:.3f}")

    def apply_pyramid(self, sig: Signal, add_price: float, add_margin: float):
        """Pyramiding Leg 추가 (VWAP 재계산)"""
        pos = self.position
        add_size = add_margin * LEVERAGE
        # VWAP 재계산
        new_vwap = (pos.entry_price * pos.position_size + add_price * add_size) / (pos.position_size + add_size)
        pos.position_size += add_size
        pos.margin_used += add_margin
        pos.entry_price = new_vwap
        if pos.leg_count == 1:
            pos.leg2.done = True
            pos.leg2.price = add_price
            pos.leg2.margin = add_margin
        elif pos.leg_count == 2:
            pos.leg3.done = True
            pos.leg3.price = add_price
            pos.leg3.margin = add_margin
        pos.leg_count += 1
        logger.info(f"⤴ PYRAMID Leg{pos.leg_count} @ ${add_price:.3f} | VWAP now ${new_vwap:.3f}")

    def apply_exit(self, sig: Signal, timestamp: float) -> TradeRecord:
        """청산 완료 후 통계 업데이트"""
        pos = self.position
        exit_px = sig.exit_price
        # PnL 계산 (수수료 차감)
        if pos.direction == 1:
            raw_pnl = (exit_px - pos.entry_price) / pos.entry_price * pos.position_size
        else:
            raw_pnl = (pos.entry_price - exit_px) / pos.entry_price * pos.position_size
        fee = pos.position_size * FEE_RATE * 2  # 진입+청산
        pnl = raw_pnl - fee
        roi = pnl / pos.margin_used * 100 if pos.margin_used > 0 else 0

        # ★ SOL 전용 누적 PnL 추적 (Daily Loss 독립 계산용)
        self.cumulative_sol_pnl += pnl
        self.daily_trade_count += 1   # ★ 당일 거래수 (daily_summary용)

        # 통계 업데이트
        if sig.exit_type == 'SL': self.sl_count += 1
        elif sig.exit_type == 'TSL': self.tsl_count += 1
        elif sig.exit_type == 'REV': self.rev_count += 1
        else: self.fc_count += 1

        if pnl > 0:
            self.win_count += 1
            self.gross_profit += pnl
            self.consec_losses = 0  # 연패 리셋
        else:
            self.loss_count += 1
            self.gross_loss += abs(pnl)
            self.consec_losses += 1
            # Skip2@4loss trigger
            if self.consec_losses >= SKIP_LOSS_THRESHOLD:
                self.skip_remaining = SKIP_TRADE_COUNT
                self.consec_losses = 0
                logger.warning(f"⚠ Skip2@4loss TRIGGERED! 다음 {SKIP_TRADE_COUNT}개 거래 스킵")

        # 기록 저장
        import datetime as dt
        rec = TradeRecord(
            direction=pos.direction,
            entry_mode=pos.entry_mode,
            entry_price=pos.entry_price,
            exit_price=exit_px,
            position_size=pos.position_size,
            pnl=pnl,
            exit_type=sig.exit_type,
            entry_time=dt.datetime.utcfromtimestamp(pos.entry_time).isoformat(),
            exit_time=dt.datetime.utcfromtimestamp(timestamp).isoformat(),
            roi_pct=roi,
            peak_roi=pos.peak_roi,
            leg_count=pos.leg_count,
        )
        self.trade_history.append(rec)
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]

        self.last_exit_dir = pos.direction
        self.last_exit_bar = 0  # 실제 bar_idx 외부에서 설정

        logger.info(f"🔚 EXIT [{'V12' if pos.entry_mode==1 else 'MASS'}] "
                    f"{sig.exit_type} {'WIN' if pnl > 0 else 'LOSS'} | "
                    f"${pos.entry_price:.3f}→${exit_px:.3f} | PnL ${pnl:+,.2f} ({roi:+.2f}%) | "
                    f"Consec {self.consec_losses}/{SKIP_LOSS_THRESHOLD}")

        # 상태 리셋
        self.position = PositionState()

        if self.skip_remaining > 0:
            self.skip_remaining -= 1 if sig.action == 'EXIT' else 0  # Only decrement on new entries

        return rec

    def tick_skip_counter(self):
        """진입 시도 카운트 (skip counter decrement)"""
        if self.skip_remaining > 0:
            self.skip_remaining -= 1

    # ═══════════════════════════════════════════════════════════
    # 상태 저장/로드
    # ═══════════════════════════════════════════════════════════
    def save_state(self):
        try:
            os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
            state = {
                'position': asdict(self.position),
                'watch_v12': asdict(self.watch_v12),
                'last_exit_dir': self.last_exit_dir,
                'consec_losses': self.consec_losses,
                'skip_remaining': self.skip_remaining,
                'mass_bulge_active': self.mass_bulge_active,
                'sl_count': self.sl_count,
                'tsl_count': self.tsl_count,
                'rev_count': self.rev_count,
                'fc_count': self.fc_count,
                'win_count': self.win_count,
                'loss_count': self.loss_count,
                'gross_profit': self.gross_profit,
                'gross_loss': self.gross_loss,
                'peak_capital': self.peak_capital,
                'max_drawdown': self.max_drawdown,
                'day_start_balance': self.day_start_balance,
                'day_start_date': self.day_start_date,
                'cumulative_sol_pnl': self.cumulative_sol_pnl,
                'day_start_sol_pnl': self.day_start_sol_pnl,
                'daily_trade_count': self.daily_trade_count,
                'day_start_win_count': self.day_start_win_count,
                'day_start_loss_count': self.day_start_loss_count,
            }
            with open(STATE_PATH, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"State save error: {e}")

    def load_state(self):
        if not os.path.exists(STATE_PATH):
            return
        try:
            with open(STATE_PATH, 'r', encoding='utf-8') as f:
                state = json.load(f)
            if state.get('position'):
                pd = state['position']
                if 'leg2' in pd and isinstance(pd['leg2'], dict):
                    pd['leg2'] = PyramidLeg(**pd['leg2'])
                if 'leg3' in pd and isinstance(pd['leg3'], dict):
                    pd['leg3'] = PyramidLeg(**pd['leg3'])
                self.position = PositionState(**pd)
            if state.get('watch_v12'):
                self.watch_v12 = WatchState(**state['watch_v12'])
            for k in ['last_exit_dir', 'consec_losses', 'skip_remaining', 'mass_bulge_active',
                      'sl_count', 'tsl_count', 'rev_count', 'fc_count',
                      'win_count', 'loss_count', 'gross_profit', 'gross_loss',
                      'peak_capital', 'max_drawdown', 'day_start_balance', 'day_start_date',
                      'cumulative_sol_pnl', 'day_start_sol_pnl',
                      'daily_trade_count', 'day_start_win_count', 'day_start_loss_count']:
                if k in state:
                    setattr(self, k, state[k])
            logger.info(f"State loaded: trades={self.total_trades}, WR {self.win_rate:.1f}%, consec_losses={self.consec_losses}, skip={self.skip_remaining}")
        except Exception as e:
            logger.error(f"State load error: {e}")
