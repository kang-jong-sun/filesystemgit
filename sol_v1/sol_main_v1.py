"""
SOL/USDT 선물 자동매매 메인 V1
V12:Mass 75:25 Mutex + Skip2@4loss + 12.5% Compound

사용법:
  python sol_main_v1.py

실행 순서:
  1. .env 로드 (API 키)
  2. DataCollector 초기화 (SOL 15m + BTC 1h 로드, WS 시작)
  3. TradingCore 초기화 (상태 복원)
  4. OrderExecutor 초기화 (잔액 조회, DB)
  5. Telegram 시작
  6. 메인 루프: 실시간 가격 체크 + 봉마감 신호 평가
"""

import asyncio
import logging
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────
# aiohttp DNS resolver: aiodns 강제 비활성화 (Windows DNS 이슈 방지)
# ─────────────────────────────────────────────────────────
try:
    import aiohttp.resolver as _aresolver
    if hasattr(_aresolver, 'AsyncResolver'):
        # aiohttp 기본이 aiodns → ThreadedResolver로 강제
        _aresolver.DefaultResolver = _aresolver.ThreadedResolver
except Exception:
    pass

from sol_data_v1 import DataCollector
from sol_core_v1 import TradingCore, LEVERAGE
from sol_executor_v1 import OrderExecutor
from sol_telegram_v1 import TelegramNotifier
from sol_web_v1 import create_dashboard

VERSION = "1.0"
LOOP_INTERVAL = 10         # 메인 루프 (초) - 실시간 체크 주기
CANDLE_CHECK = 30          # 봉 업데이트 체크 (초)
BALANCE_CHECK = 300        # 잔액 조회 주기 (초)
POSITION_SYNC = 10         # 포지션 동기화 주기 (초) — ETH V8 패턴: 수동 포지션 즉각 감지
STATUS_REPORT = 10800      # 텔레그램 상태 리포트 (3시간)
STATE_SAVE_INTERVAL = 60   # 상태 저장 주기 (초)
HEARTBEAT_INTERVAL = 300   # 로그 heartbeat (5분) - 봇 살아있음 표시


def _cleanup_old_logs(log_dir: Path, pattern: str, retention_days: int):
    """패턴 매칭된 로그 파일 중 retention_days 이전 것 삭제"""
    import time as _t
    cutoff = _t.time() - retention_days * 86400
    for old in log_dir.glob(pattern):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except Exception:
            pass


def setup_logging():
    """
    5개 전용 로거 + 통합 백업 (총 6개 파일)
    ──────────────────────────────────────────────────────────
    | 파일                         | 내용                  | 보관 |
    |-----------------------------|---------------------|-----|
    | error_YYYYMMDD.log          | WARNING+ only       | 365일 |
    | trades.log                  | 진입/청산/피라미드    | 영구  |
    | signals_YYYYMMDD.log        | 필터 통과/실패 이유   | 90일  |
    | heartbeat_YYYYMMDD.log      | 5분 상태 스냅샷      | 30일  |
    | daily_summary.log           | 매일 1줄 요약        | 영구  |
    | sol_trading_YYYYMMDD.log    | 통합 (백업용)        | 30일  |
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 보관 정책 (차등)
    _cleanup_old_logs(log_dir, "error_*.log",        365)
    _cleanup_old_logs(log_dir, "signals_*.log",      90)
    _cleanup_old_logs(log_dir, "heartbeat_*.log",    30)
    _cleanup_old_logs(log_dir, "sol_trading_*.log",  30)
    # trades.log, daily_summary.log는 영구 (삭제 안 함)

    today = datetime.now().strftime('%Y%m%d')
    fmt = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                             datefmt='%Y-%m-%d %H:%M:%S')
    fmt_trade = logging.Formatter('%(asctime)s %(message)s',
                                   datefmt='%Y-%m-%d %H:%M:%S')

    # ═══════════════════════════════════════════════════════
    # Root: 통합 로그 + console + error (WARNING+)
    # ═══════════════════════════════════════════════════════
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # 1) 통합 백업 (INFO+)
    combined_h = logging.FileHandler(log_dir / f"sol_trading_{today}.log", encoding='utf-8')
    combined_h.setLevel(logging.INFO)
    combined_h.setFormatter(fmt)
    # heartbeat logger 출력은 combined에 중복 방지 (너무 큼)
    combined_h.addFilter(lambda rec: not rec.name.startswith('sol.heartbeat'))
    root.addHandler(combined_h)

    # 2) 에러 전용 (WARNING+)
    error_h = logging.FileHandler(log_dir / f"error_{today}.log", encoding='utf-8')
    error_h.setLevel(logging.WARNING)
    error_h.setFormatter(fmt)
    root.addHandler(error_h)

    # 3) 콘솔 (INFO+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # ═══════════════════════════════════════════════════════
    # 전용 로거 (propagate 차등)
    # ═══════════════════════════════════════════════════════

    # 📈 sol.trades — trades.log 전용 + 통합에도 전파
    trade_logger = logging.getLogger('sol.trades')
    trade_logger.setLevel(logging.INFO)
    trade_h = logging.FileHandler(log_dir / "trades.log", 'a', encoding='utf-8')
    trade_h.setFormatter(fmt_trade)
    trade_logger.addHandler(trade_h)
    trade_logger.propagate = True  # 통합 로그에도 남김

    # 🔍 sol.signals — signals_*.log + 통합 전파
    signal_logger = logging.getLogger('sol.signals')
    signal_logger.setLevel(logging.INFO)
    signal_h = logging.FileHandler(log_dir / f"signals_{today}.log", encoding='utf-8')
    signal_h.setFormatter(fmt_trade)
    signal_logger.addHandler(signal_h)
    signal_logger.propagate = True

    # 💓 sol.heartbeat — heartbeat_*.log 전용 (통합 전파 X, 용량 방지)
    hb_logger = logging.getLogger('sol.heartbeat')
    hb_logger.setLevel(logging.INFO)
    hb_h = logging.FileHandler(log_dir / f"heartbeat_{today}.log", encoding='utf-8')
    hb_h.setFormatter(fmt_trade)
    hb_logger.addHandler(hb_h)
    hb_logger.propagate = False  # 통합에 안 남김 (5분마다 → 용량 큼)

    # 📅 sol.summary — daily_summary.log 전용 (통합 전파 X)
    summary_logger = logging.getLogger('sol.summary')
    summary_logger.setLevel(logging.INFO)
    summary_h = logging.FileHandler(log_dir / "daily_summary.log", 'a', encoding='utf-8')
    summary_h.setFormatter(fmt_trade)
    summary_logger.addHandler(summary_h)
    summary_logger.propagate = False  # 한 줄 요약, 통합에 안 남김

    # 시끄러운 외부 라이브러리 억제
    for noisy in ('websockets', 'ccxt', 'aiosqlite', 'asyncio',
                  'ccxt.base.exchange', 'websockets.client', 'aiohttp'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger('sol_main')


class SOLTradingBot:
    def __init__(self):
        self.logger = logging.getLogger('sol_main')
        # 전용 로거 (trades.log / signals.log / heartbeat.log / daily_summary.log)
        self.trade_log = logging.getLogger('sol.trades')
        self.signal_log = logging.getLogger('sol.signals')
        self.hb_log = logging.getLogger('sol.heartbeat')
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._shutdown_done = False

        self.data: DataCollector = None
        self.core: TradingCore = None
        self.executor: OrderExecutor = None
        self.telegram: TelegramNotifier = None
        self.web = None

        self._last_candle_check = 0.0
        self._last_balance_check = 0.0
        self._last_position_sync = 0.0
        self._last_status_report = 0.0
        self._last_state_save = 0.0
        self._last_heartbeat = 0.0
        self._last_processed_bar_idx = -1
        self._start_time = time.time()  # uptime 계산용

    async def initialize(self):
        load_dotenv()
        binance_key = os.getenv('BINANCE_API_KEY', '')
        binance_secret = os.getenv('BINANCE_API_SECRET', '')
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        telegram_chat = os.getenv('TELEGRAM_CHAT_ID', '')
        ws_url = os.getenv('BINANCE_WS_FUTURES_URL', 'wss://fstream.binance.com/ws')

        if not binance_key or not binance_secret:
            self.logger.error("BINANCE_API_KEY/SECRET 필수!")
            sys.exit(1)

        self.data = DataCollector(binance_key, binance_secret, ws_url)
        self.core = TradingCore()
        self.executor = OrderExecutor(binance_key, binance_secret)
        self.telegram = TelegramNotifier(telegram_token, telegram_chat)

        self.logger.info("=" * 60)
        self.logger.info(f"  SOL/USDT Futures V{VERSION}")
        self.logger.info(f"  V12:Mass 75:25 Mutex + Skip2@4loss")
        self.logger.info(f"  Compound 12.5% | Leverage {LEVERAGE}x | Isolated")
        self.logger.info("=" * 60)

        await self.data.initialize()
        await self.executor.initialize()
        await self.telegram.start()

        # 상태 복원
        self.core.load_state()
        if self.core.peak_capital == 0:
            self.core.peak_capital = self.executor.balance
        if self.core.day_start_balance == 0:
            self.core.day_start_balance = self.executor.balance

        self.telegram.set_command_handler(self._handle_command)

        # 거래소 수동 포지션 체크
        await self._check_manual_position()

        # 웹 대시보드 시작 (port 8081, ETH_V8와 충돌 회피)
        try:
            web_port = int(os.getenv('SOL_WEB_PORT', '8081'))
            self.web = create_dashboard(self, port=web_port)
            await self.web.start()
        except Exception as e:
            self.logger.warning(f"웹 대시보드 실행 실패 (계속 진행): {e}")

        await self.telegram.notify_start(self.executor.balance)
        self.logger.info(f"✅ 초기화 완료 | 잔액 ${self.executor.balance:,.2f}")

    async def _check_manual_position(self):
        ex_pos = await self.executor.get_exchange_position()
        if not ex_pos:
            # 거래소에 포지션 없음 — 봇 정지 중 사용자가 수동 청산했을 가능성
            # state에 포지션 흔적 + 거래소에 SL orphan이 남아있을 수 있으므로 정리
            if self.core.has_position:
                self.logger.info("거래소 포지션 없음 — state 청산 처리 (사용자 수동 청산 추정)")
                exit_price = self.data.current_price if self.data and self.data.current_price > 0 else self.core.position.entry_price
                from sol_core_v1 import Signal
                rec_sig = Signal(action='EXIT', direction=self.core.position.direction,
                                 exit_type='MANUAL', exit_price=exit_price)
                self.core.apply_exit(rec_sig, timestamp=time.time())
            await self.executor._cancel_all_sl_orders()  # orphan SL 청소
            self.core.save_state()
            return

        # 봇 내부 상태와 비교
        eff_lev = ex_pos.get('leverage') or self.executor.leverage or LEVERAGE
        if self.core.has_position and self.core.position.direction == ex_pos['direction']:
            # 기존 상태 유지, 사이즈/margin은 거래소 실제값으로 동기화
            self.core.position.position_size = ex_pos['notional']
            # 거래소 initialMargin 우선, 없으면 notional/leverage 계산
            self.core.position.margin_used = ex_pos.get('margin') or (ex_pos['notional'] / eff_lev)
            self.logger.info(f"기존 포지션 상태 복원 (Leverage {eff_lev}x)")
            # SL이 거래소에 있는지 확인하고 없으면 다시 등록
            try:
                sl_price = self.core.position.sl_price
                await self.executor.update_stop_loss(ex_pos['direction'], ex_pos['size'], sl_price)
            except Exception as e:
                self.logger.warning(f"기존 포지션 SL 재등록 실패: {e}")
        else:
            # 수동 포지션 감지 → 트래킹 시작
            self.logger.warning("수동 포지션 감지! 트래킹 시작.")
            # V12 모드로 가정, SL 3.8% 적용
            from sol_core_v1 import PositionState, EntryMode, SL_PCT_BASE, TA_PCT_BASE, TSL_PCT_BASE
            direction = ex_pos['direction']
            entry_price = ex_pos['entry_price']
            sl_price = entry_price * (1 - SL_PCT_BASE/100) if direction == 1 else entry_price * (1 + SL_PCT_BASE/100)

            # 📈 trades.log 기록 (수동 진입)
            dir_str = 'LONG' if direction == 1 else 'SHORT'
            self.trade_log.info(
                f"[MANUAL ENTRY] {dir_str} @${entry_price:.3f} | "
                f"Notional ${ex_pos['notional']:,.0f} | Auto-SL ${sl_price:.3f} (3.8%) | "
                f"Pyramiding DISABLED"
            )
            # 사용자 실제 margin/leverage로 초기화 (ETH V8 패턴)
            user_margin = ex_pos.get('margin') or (ex_pos['notional'] / eff_lev)
            self.logger.info(f"수동 포지션 인지: {dir_str} @${entry_price:.3f} | "
                             f"Notional ${ex_pos['notional']:,.0f} | Margin ${user_margin:,.0f} | "
                             f"Leverage {eff_lev}x (사용자 설정)")
            self.core.position = PositionState(
                direction=direction, entry_mode=int(EntryMode.V12),
                entry_price=entry_price, entry_price_leg1=entry_price,
                position_size=ex_pos['notional'],
                margin_used=user_margin,  # ★ 거래소 실제값 사용 (5x/10x 등 사용자 설정 반영)
                sl_price=sl_price, tsl_active=False,
                track_high=entry_price, track_low=entry_price,
                entry_time=time.time(), entry_bar=self.data.get_latest_index(),
                sl_pct=SL_PCT_BASE, ta_pct=TA_PCT_BASE, tsl_pct=TSL_PCT_BASE,
                use_pyr=False,  # 수동 포지션은 pyramiding 비활성
            )
            try:
                await self.executor.update_stop_loss(direction, ex_pos['size'], sl_price)
            except Exception as e:
                self.logger.warning(f"SL 설정 실패: {e}")
            # ★ DB entries 테이블에 USER 진입 기록 (이전엔 누락됨)
            try:
                await self.executor.save_entry({
                    'source': 'USER',
                    'direction': direction,
                    'entry_mode': int(EntryMode.V12),
                    'entry_price': entry_price,
                    'position_size': ex_pos['notional'],
                    'margin': user_margin,
                    'leverage': eff_lev,            # 사용자 실제 레버리지 보존
                    'sl_price': sl_price,
                    'balance_after': self.executor.balance,
                    'conf_score': 0,
                    'conf_mult': 1.0,
                    'margin_mult': 1.0,
                })
            except Exception as e:
                self.logger.warning(f"수동 포지션 DB 저장 실패: {e}")
            self.telegram.notify_manual_position(direction, entry_price, ex_pos['notional'])

        self.core.save_state()

    async def run(self):
        self._running = True
        await self.data.start_websocket()
        self.logger.info("🔁 메인 루프 시작 (10초 tick / 5분 heartbeat / 3시간 status)")
        # 첫 heartbeat는 2분 후 (webscoket 안정 대기)
        self._last_heartbeat = time.time() - (HEARTBEAT_INTERVAL - 120)

        try:
            while self._running:
                try:
                    await self._tick()
                except Exception as e:
                    self.logger.error(f"Loop error: {e}\n{traceback.format_exc()[:500]}")
                    self.telegram.notify_error(f"Loop error: {str(e)[:200]}")
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=LOOP_INTERVAL)
                    break
                except asyncio.TimeoutError:
                    pass
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            if not self._shutdown_done:
                await self._shutdown()

    async def _tick(self):
        now = time.time()

        # 캔들 업데이트
        if now - self._last_candle_check >= CANDLE_CHECK:
            await self.data.update_candles()
            self._last_candle_check = now

        # 잔액 업데이트
        if now - self._last_balance_check >= BALANCE_CHECK:
            await self.executor.update_balance()
            self.core.update_peak(self.executor.balance)
            self._last_balance_check = now

        # 포지션 동기화
        if now - self._last_position_sync >= POSITION_SYNC:
            await self._sync_position()
            self._last_position_sync = now

        # 상태 리포트
        if now - self._last_status_report >= STATUS_REPORT:
            self._send_status_report()
            self._last_status_report = now

        # 상태 저장
        if now - self._last_state_save >= STATE_SAVE_INTERVAL:
            self.core.save_state()
            self._last_state_save = now

        # Heartbeat (5분마다 상태 로그 - 봇 alive 표시)
        if now - self._last_heartbeat >= HEARTBEAT_INTERVAL:
            await self._log_heartbeat()
            self._last_heartbeat = now

        # ═══════════════════════════════════════════════════════════
        # 실시간 SL/TSL 체크 (포지션 있을 때)
        # ═══════════════════════════════════════════════════════════
        if self.core.has_position and self.data.current_price > 0:
            px = self.data.current_price
            high = max(self.data.current_high, px)
            low = min(self.data.current_low, px)
            rt_sig = self.core.check_realtime_exit(px, high, low)
            if rt_sig and rt_sig.action == 'EXIT':
                await self._handle_exit(rt_sig)
                return

        # ═══════════════════════════════════════════════════════════
        # 봉마감 신호 평가 (새 마감봉 발견 시만)
        # ═══════════════════════════════════════════════════════════
        last_bar = self.data.get_last_closed_bar()
        if last_bar is None:
            return

        bar_idx = last_bar['bar_idx']
        if bar_idx <= self._last_processed_bar_idx:
            return  # 이미 처리한 봉

        # ★ 신선도 체크: bar.timestamp가 현재 시간과 30분 이내여야 함
        bar_age = time.time() - last_bar['timestamp']
        if bar_age > 1800:  # 30분 초과 → 과거 봉 (catch-up 중)
            self.logger.debug(f"Skipping stale bar (age {bar_age/60:.1f}min) - catch-up in progress")
            self._last_processed_bar_idx = bar_idx  # mark as processed
            return

        # ★ Signal-Market price gap 체크: bar.close와 current price 차이가 2% 초과면 무시
        if self.data.current_price > 0 and last_bar['close'] > 0:
            price_gap = abs(self.data.current_price - last_bar['close']) / last_bar['close']
            if price_gap > 0.02:  # 2% gap → 데이터 불일치
                self.logger.warning(f"Bar-Market price gap {price_gap*100:.1f}% (bar ${last_bar['close']:.3f} vs market ${self.data.current_price:.3f}) - skipping signal")
                self._last_processed_bar_idx = bar_idx
                return

        self._last_processed_bar_idx = bar_idx

        # 지표 추출
        ind = self.data.get_indicators_at(bar_idx)
        if not ind or ind.get('slow_ma') is None:
            return

        # BTC regime
        btc_bull = self.data.get_btc_regime_bull(last_bar['timestamp'])

        # 핵심 신호 평가
        sig = self.core.evaluate(
            bar=last_bar, bar_idx=bar_idx,
            capital=self.executor.balance,
            btc_regime_bull=btc_bull,
            atr14=ind.get('atr14', float('nan')),
            atr50=ind.get('atr50', float('nan')),
            indicators=ind,
            mass_index=ind.get('mass', float('nan')),
        )

        if sig.action == 'ENTRY':
            await self._handle_entry(sig)
        elif sig.action == 'EXIT':
            await self._handle_exit(sig)
        elif sig.action == 'PYRAMID':
            await self._handle_pyramid(sig)

    async def _handle_entry(self, sig):
        try:
            # Skip counter 감소 (진입 시도 발생)
            if self.core.skip_remaining > 0:
                self.core.tick_skip_counter()
                self.signal_log.info(f"GATE Skip2@4loss | {self.core.skip_remaining} remaining")
                self.logger.info(f"⏭ Skip ({self.core.skip_remaining} remaining)")
                return

            # ★ BOT 자동 진입은 항상 LEVERAGE 상수로 강제 (백테스트 무결성)
            #   사용자가 다른 레버리지로 변경했어도 봇 진입 시 다시 LEVERAGE로
            try:
                from sol_executor_v1 import SYMBOL as EX_SYMBOL
                await self.executor.exchange.set_leverage(LEVERAGE, EX_SYMBOL)
                self.executor.leverage = LEVERAGE
                self.executor.leverage_warned = False
            except Exception as e:
                self.logger.warning(f"BOT 진입 전 레버리지 {LEVERAGE}x 설정 실패: {e}")

            result = await self.executor.market_entry(
                direction=sig.direction,
                position_size_usd=sig.position_size_usd,
                sl_price=sig.sl_price,
            )
            # 실제 체결가로 signal 업데이트
            sig.entry_price = result['filled_price']
            sig.position_size_usd = result['position_size_usd']
            sig.sl_price = sig.entry_price * (1 - sig.__dict__['sl_pct']/100) if sig.direction == 1 else sig.entry_price * (1 + sig.__dict__['sl_pct']/100)

            # Core state 업데이트
            self.core.apply_entry(sig, bar_idx=self._last_processed_bar_idx, timestamp=time.time())

            # 📈 trades.log 기록
            mode_str = 'V12' if sig.entry_mode == 1 else 'MASS'
            dir_str = 'LONG' if sig.direction == 1 else 'SHORT'
            self.trade_log.info(
                f"[BOT ENTRY {mode_str}] {dir_str} @${sig.entry_price:.3f} | "
                f"Size ${sig.position_size_usd:,.0f} | Margin ${sig.margin_usd:,.0f} | "
                f"SL ${sig.sl_price:.3f} | Conf {sig.conf_score:.2f}x{sig.conf_mult:.2f} "
                f"MS x{sig.margin_mult:.2f} | Balance ${self.executor.balance:,.2f}"
            )

            # DB 저장
            await self.executor.save_entry({
                'direction': sig.direction,
                'entry_mode': sig.entry_mode,
                'entry_price': sig.entry_price,
                'position_size': sig.position_size_usd,
                'margin': sig.margin_usd,
                'sl_price': sig.sl_price,
                'balance_after': self.executor.balance,
                'conf_score': sig.conf_score,
                'conf_mult': sig.conf_mult,
                'margin_mult': sig.margin_mult,
                'source': 'BOT',
            })

            # 텔레그램
            self.telegram.notify_entry({
                'direction': sig.direction, 'entry_mode': sig.entry_mode,
                'entry_price': sig.entry_price, 'position_size': sig.position_size_usd,
                'margin': sig.margin_usd, 'sl_price': sig.sl_price,
                'conf_score': sig.conf_score, 'conf_mult': sig.conf_mult,
                'margin_mult': sig.margin_mult, 'balance': self.executor.balance,
            })
            self.core.save_state()
        except Exception as e:
            self.logger.error(f"Entry 처리 오류: {e}\n{traceback.format_exc()[:500]}")
            self.telegram.notify_error(f"Entry error: {str(e)[:200]}")

    async def _handle_exit(self, sig):
        try:
            pos = self.core.position
            result = await self.executor.market_exit(direction=pos.direction)
            actual_exit_price = result['filled_price'] if result else sig.exit_price
            sig.exit_price = actual_exit_price

            # Core state 업데이트 (통계, skip 등 포함)
            rec = self.core.apply_exit(sig, timestamp=time.time())

            # 📈 trades.log 기록
            mode_str = 'V12' if rec.entry_mode == 1 else 'MASS'
            dir_str = 'LONG' if rec.direction == 1 else 'SHORT'
            verdict = 'WIN' if rec.pnl > 0 else 'LOSS'
            self.trade_log.info(
                f"[BOT EXIT {mode_str}] {dir_str} {sig.exit_type} {verdict} | "
                f"${rec.entry_price:.3f}→${rec.exit_price:.3f} | "
                f"PnL ${rec.pnl:+,.2f} ({rec.roi_pct:+.2f}%) | PeakROI {rec.peak_roi:+.2f}% | "
                f"Legs {rec.leg_count} | Balance ${self.executor.balance:,.2f} | "
                f"Consec {self.core.consec_losses}/4"
            )

            # DB 저장
            await self.executor.save_trade({
                'direction': rec.direction, 'entry_mode': rec.entry_mode,
                'entry_price': rec.entry_price, 'exit_price': rec.exit_price,
                'position_size': rec.position_size, 'pnl': rec.pnl,
                'exit_type': rec.exit_type, 'roi_pct': rec.roi_pct,
                'peak_roi': rec.peak_roi, 'leg_count': rec.leg_count,
                'source': 'BOT',
            })

            # 텔레그램
            self.telegram.notify_exit({
                'direction': rec.direction, 'entry_mode': rec.entry_mode,
                'entry_price': rec.entry_price, 'exit_price': rec.exit_price,
                'pnl': rec.pnl, 'roi_pct': rec.roi_pct, 'peak_roi': rec.peak_roi,
                'exit_type': rec.exit_type, 'balance': self.executor.balance,
            })

            # Skip 트리거 알림
            if self.core.skip_remaining > 0 and rec.pnl <= 0:
                # 연패 누적으로 방금 skip 트리거됐을 수 있음
                if self.core.consec_losses == 0:  # 리셋됐으면 trigger 발생
                    self.telegram.notify_skip_trigger(4)

            self.core.save_state()
        except Exception as e:
            self.logger.error(f"Exit 처리 오류: {e}\n{traceback.format_exc()[:500]}")
            self.telegram.notify_error(f"Exit error: {str(e)[:200]}")

    async def _handle_pyramid(self, sig):
        try:
            pos = self.core.position
            add_size_usd = sig.margin_usd * LEVERAGE
            result = await self.executor.market_pyramid_add(pos.direction, add_size_usd)
            actual_price = result['filled_price']

            # Core state (VWAP 재계산)
            self.core.apply_pyramid(sig, add_price=actual_price, add_margin=sig.margin_usd)

            # 📈 trades.log 기록 (피라미드)
            dir_str = 'LONG' if pos.direction == 1 else 'SHORT'
            self.trade_log.info(
                f"[BOT PYRAMID Leg{pos.leg_count}] {dir_str} @${actual_price:.3f} | "
                f"AddMargin ${sig.margin_usd:,.0f} | VWAP ${pos.entry_price:.3f}"
            )

            # SL 업데이트 (새 VWAP 기준)
            new_size = pos.position_size / actual_price  # approximate qty
            await self.executor.update_stop_loss(pos.direction, new_size, pos.sl_price)

            self.telegram.notify_pyramid(pos.leg_count, actual_price, sig.margin_usd)
            self.core.save_state()
        except Exception as e:
            self.logger.error(f"Pyramid 오류: {e}")
            self.telegram.notify_error(f"Pyramid error: {str(e)[:200]}")

    async def _sync_position(self):
        """봇 상태 vs 거래소 포지션 동기화 — ETH V8 패턴 (3가지 케이스)"""
        try:
            ex_pos = await self.executor.get_exchange_position()
            bot_has = self.core.has_position

            # ★ 사용자 레버리지 변경 감지 → 1회 알림
            eff_lev = self.executor.leverage or LEVERAGE
            if eff_lev != LEVERAGE and not self.executor.leverage_warned:
                msg = (f"ℹ️ 사용자 레버리지 {eff_lev}x 인지 (봇 기본 {LEVERAGE}x). "
                       f"USER 포지션은 {eff_lev}x로, BOT 자동진입은 {LEVERAGE}x로 처리됩니다.")
                self.logger.info(msg)
                try:
                    self.telegram.notify_error(msg)
                except Exception:
                    pass
                self.executor.leverage_warned = True

            # 케이스 A: 봇 없는데 거래소에 있음 → 수동 진입 감지
            if ex_pos and not bot_has:
                self.logger.warning("수동 포지션 감지!")
                await self._check_manual_position()

            # 케이스 B: 봇 있는데 거래소에 없음 → 외부 청산 감지
            elif not ex_pos and bot_has:
                # 거래소 SL 주문 청소 (orphan)
                await self.executor._cancel_all_sl_orders()
                # 실제 체결가 조회 (외부 SL 발동 / 수동 청산)
                actual_exit_price = await self.executor.get_last_exit_price()
                if actual_exit_price <= 0:
                    actual_exit_price = self.data.current_price or self.core.position.entry_price
                pos_before = self.core.position
                dir_str = 'LONG' if pos_before.direction == 1 else 'SHORT'
                # 📈 trades.log 기록 (외부 청산)
                self.trade_log.info(
                    f"[EXTERNAL EXIT] {dir_str} @${actual_exit_price:.3f} | "
                    f"Entry ${pos_before.entry_price:.3f} | 수동/SL 외부 체결"
                )
                from sol_core_v1 import Signal
                rec_sig = Signal(action='EXIT', direction=pos_before.direction,
                                 exit_type='EXT', exit_price=actual_exit_price)
                rec = self.core.apply_exit(rec_sig, timestamp=time.time())
                # DB 저장
                try:
                    await self.executor.save_trade({
                        'direction': rec.direction, 'entry_mode': rec.entry_mode,
                        'entry_price': rec.entry_price, 'exit_price': rec.exit_price,
                        'position_size': rec.position_size, 'pnl': rec.pnl,
                        'exit_type': rec.exit_type, 'roi_pct': rec.roi_pct,
                        'peak_roi': rec.peak_roi, 'leg_count': rec.leg_count,
                        'source': 'USER',
                    })
                except Exception as e:
                    self.logger.warning(f"외부 청산 DB 저장 실패: {e}")
                # 텔레그램 청산 알림 (error 알림 대신 정상 exit 알림)
                self.telegram.notify_exit({
                    'direction': rec.direction, 'entry_mode': rec.entry_mode,
                    'entry_price': rec.entry_price, 'exit_price': rec.exit_price,
                    'pnl': rec.pnl, 'roi_pct': rec.roi_pct, 'peak_roi': rec.peak_roi,
                    'exit_type': 'EXT', 'balance': self.executor.balance,
                })
                self.core.save_state()

            # 케이스 C: 봇/거래소 모두 있는데 방향 다름 → 사용자가 수동으로 반대 포지션 잡음
            elif ex_pos and bot_has and ex_pos['direction'] != self.core.position.direction:
                self.logger.warning(f"포지션 방향 불일치! "
                                    f"봇={self.core.position.direction} vs 거래소={ex_pos['direction']} → "
                                    f"봇 포지션 청산 후 새 포지션 추적")
                # 옛 SL 청소
                await self.executor._cancel_all_sl_orders()
                # 옛 포지션 close (실체결가)
                actual_exit_price = await self.executor.get_last_exit_price()
                if actual_exit_price <= 0:
                    actual_exit_price = self.data.current_price or self.core.position.entry_price
                pos_before = self.core.position
                from sol_core_v1 import Signal
                rec_sig = Signal(action='EXIT', direction=pos_before.direction,
                                 exit_type='EXT', exit_price=actual_exit_price)
                rec = self.core.apply_exit(rec_sig, timestamp=time.time())
                try:
                    await self.executor.save_trade({
                        'direction': rec.direction, 'entry_mode': rec.entry_mode,
                        'entry_price': rec.entry_price, 'exit_price': rec.exit_price,
                        'position_size': rec.position_size, 'pnl': rec.pnl,
                        'exit_type': 'EXT', 'roi_pct': rec.roi_pct,
                        'peak_roi': rec.peak_roi, 'leg_count': rec.leg_count,
                        'source': 'USER',
                    })
                except Exception as e:
                    self.logger.warning(f"방향전환 DB 저장 실패: {e}")
                # 새 포지션 track + 새 SL
                await self._check_manual_position()
                self.core.save_state()
        except Exception as e:
            self.logger.error(f"포지션 동기화 오류: {e}\n{traceback.format_exc()[:300]}")

    async def _log_heartbeat(self):
        """5분마다 현재 상태 간략 로그 (가시성)"""
        try:
            bar_idx = self.data.get_latest_index() if self.data else 0
            sol_bars = len(self.data.df_sol) if self.data and self.data.df_sol is not None else 0
            btc_bars = len(self.data.df_btc) if self.data and self.data.df_btc is not None else 0
            price = self.data.current_price if self.data else 0
            ind = self.data.get_indicators_at(bar_idx) if self.data else {}

            ready = 'READY' if (not np.isnan(ind.get('slow_ma', float('nan')))) else 'WAITING-SMA400'

            if self.core.has_position:
                pos = self.core.position
                pos_str = f"POS[{'V12' if pos.entry_mode==1 else 'MASS'} {'LONG' if pos.direction==1 else 'SHORT'} @${pos.entry_price:.3f}, SL ${pos.sl_price:.3f}]"
            else:
                pos_str = "NO_POSITION"

            # 💓 heartbeat.log 전용 (통합 로그 X, 용량 방지)
            self.hb_log.info(
                f"Price ${price:.3f} | Bars SOL {sol_bars}/BTC {btc_bars} | "
                f"ADX {ind.get('adx', 0):.1f} RSI {ind.get('rsi', 0):.1f} Mass {ind.get('mass', 0):.2f} | "
                f"Balance ${self.executor.balance:,.2f} | Trades {self.core.total_trades} WR {self.core.win_rate:.1f}% | "
                f"Consec {self.core.consec_losses}/4 Skip {self.core.skip_remaining} | "
                f"{ready} | {pos_str}"
            )
        except Exception as e:
            self.logger.warning(f"Heartbeat error: {e}")

    def _send_status_report(self):
        from sol_core_v1 import LEVERAGE
        # 가격 폴백
        price = 0.0
        if self.data:
            if self.data.current_price > 0:
                price = float(self.data.current_price)
            elif self.data.df_sol is not None and len(self.data.df_sol) > 0:
                price = float(self.data.df_sol['close'].iloc[-1])
        eff_lev = self.executor.leverage if hasattr(self.executor, 'leverage') and self.executor.leverage else LEVERAGE

        stats = {
            'balance': self.executor.balance,
            'peak': self.core.peak_capital,
            'mdd': self.core.max_drawdown,
            'total_trades': self.core.total_trades,
            'wins': self.core.win_count, 'losses': self.core.loss_count,
            'wr': self.core.win_rate, 'pf': self.core.profit_factor,
            'sl': self.core.sl_count, 'tsl': self.core.tsl_count,
            'rev': self.core.rev_count,
            'consec': self.core.consec_losses,
            'skip': self.core.skip_remaining,
            'price': price,
            'leverage': eff_lev,
            'leverage_default': LEVERAGE,
            'position': None,
        }
        if self.core.has_position:
            p = self.core.position
            if price > 0 and p.entry_price > 0:
                roi = (price - p.entry_price) / p.entry_price * p.direction * 100
                pnl = (price - p.entry_price) / p.entry_price * p.position_size * p.direction
            else:
                roi = 0; pnl = 0
            from datetime import datetime as _dt
            entry_dt_str = _dt.fromtimestamp(p.entry_time).strftime('%Y-%m-%d %H:%M:%S') if p.entry_time else '-'
            hold = max(int(time.time() - p.entry_time), 0) if p.entry_time else 0
            d_, rem = divmod(hold, 86400)
            hh, rem = divmod(rem, 3600)
            mm, _ss = divmod(rem, 60)
            hold_str = (f"{d_}d {hh}h {mm}m" if d_ > 0 else (f"{hh}h {mm}m" if hh > 0 else f"{mm}m"))
            stats['position'] = {
                'direction_str': 'LONG' if p.direction == 1 else 'SHORT',
                'entry_mode_str': 'V12' if p.entry_mode == 1 else 'MASS',
                'entry_price': p.entry_price,
                'sl_price': p.sl_price,
                'tsl_active': p.tsl_active,
                'roi': roi, 'pnl': pnl, 'peak_roi': p.peak_roi,
                'entry_time_str': entry_dt_str,
                'hold_str': hold_str,
                'has_price': price > 0,
            }
        self.telegram.notify_status(stats)

    async def _handle_command(self, cmd: str):
        """텔레그램 명령어 처리 (ETH V8 패턴).
        구조:
          /help — 공용 (모든 봇 응답)
          /sol <한글> — SOL 봇 전용 (대소문자 / 공백 유무 무관)
          그 외 명령은 무시 (ETH 봇 등과 충돌 방지)
        """
        try:
            text = (cmd or '').strip()
            if not text:
                return
            words = text.split()
            first_word_raw = words[0].split('@')[0]
            first_lower = first_word_raw.lower()

            # === /help (공용) ===
            if first_lower == '/help':
                self.telegram.send(
                    "<b>[SOL V1] 명령어</b>\n"
                    "/sol 상태 — 봇 전체 상태\n"
                    "/sol 포지션 — 포지션 상세\n"
                    "/sol 잔액 — 잔액\n"
                    "/sol 최근거래 — 최근 5건 거래\n"
                    "/sol 봇정지 — 봇 정지 (systemd 자동 재시작)\n"
                    "/sol 봇시작 — 상태 확인 후 재시작\n"
                    "/sol 청산 — 포지션 수동 청산\n"
                    "(/sol, /SOL, /Sol + 공백 유무 모두 가능)"
                )
                return

            # === /sol 분기 (대소문자 + 공백 유무 모두 허용) ===
            if not first_lower.startswith('/sol'):
                return  # 다른 명령은 무시 (ETH 봇 등에 양보)

            # /sol 다음 모든 문자 결합 (공백 제거)
            remainder = first_word_raw[4:]  # /sol(4글자) 이후
            extras = ''.join(words[1:]) if len(words) > 1 else ''
            sub = (remainder + extras).strip()

            from sol_core_v1 import LEVERAGE
            # ★ price 폴백: WebSocket 가격이 0이면 df_sol 마지막 close 사용
            price = 0.0
            if self.data:
                if self.data.current_price > 0:
                    price = float(self.data.current_price)
                elif self.data.df_sol is not None and len(self.data.df_sol) > 0:
                    price = float(self.data.df_sol['close'].iloc[-1])
            # 거래소 동적 레버리지 (사용자 설정 추적)
            eff_lev = self.executor.leverage if hasattr(self.executor, 'leverage') and self.executor.leverage else LEVERAGE
            uptime = time.time() - self._start_time
            h, rem = divmod(int(max(uptime, 0)), 3600)
            m, _s = divmod(rem, 60)

            # === /sol 상태 ===
            if sub in ('상태', '전체상태', ''):
                pos_str = "NO POSITION"
                if self.core.has_position:
                    p = self.core.position
                    # 가격이 유효할 때만 ROI/PnL 계산 (0이면 N/A)
                    if price > 0 and p.entry_price > 0:
                        roi = (price - p.entry_price) / p.entry_price * p.direction * 100
                        pnl = (price - p.entry_price) / p.entry_price * p.position_size * p.direction
                        roi_str = f"{roi:+.2f}%"
                        pnl_str = f"${pnl:+,.2f}"
                    else:
                        roi_str = "N/A (가격 대기)"
                        pnl_str = "N/A"
                    d = "LONG" if p.direction == 1 else "SHORT"
                    mode = 'V12' if p.entry_mode == 1 else 'MASS'
                    pos_str = (f"{d} [{mode}] @${p.entry_price:.3f}\n"
                               f"  ROI: {roi_str} | PnL: {pnl_str}\n"
                               f"  SL: ${p.sl_price:.3f} | TSL: {'ON' if p.tsl_active else 'OFF'}")
                self.telegram.send(
                    f"<b>[SOL V1] 상태</b>\n"
                    f"Uptime: {h}h {m}m\n"
                    f"Price: ${price:.3f}\n"
                    f"Leverage: {eff_lev}x{' (사용자 설정)' if eff_lev != LEVERAGE else ''}\n\n"
                    f"<b>Position:</b> {pos_str}\n\n"
                    f"<b>Balance:</b> ${self.executor.balance:,.2f}\n"
                    f"Trades: {self.core.total_trades} | WR: {self.core.win_rate:.0f}%\n"
                    f"PF: {self.core.profit_factor:.2f}\n"
                    f"Consec: {self.core.consec_losses}/4 | Skip: {self.core.skip_remaining}"
                )

            # === /sol 포지션 ===
            elif sub == '포지션':
                if not self.core.has_position:
                    self.telegram.send("<b>[SOL V1]</b> 보유 포지션 없음")
                    return
                p = self.core.position
                if price > 0 and p.entry_price > 0:
                    roi = (price - p.entry_price) / p.entry_price * p.direction * 100
                    pnl = (price - p.entry_price) / p.entry_price * p.position_size * p.direction
                    roi_str = f"{roi:+.2f}%"
                    pnl_str = f"${pnl:+,.2f}"
                else:
                    roi_str = "N/A (가격 대기)"
                    pnl_str = "N/A"
                d = "LONG" if p.direction == 1 else "SHORT"
                mode = 'V12' if p.entry_mode == 1 else 'MASS'
                # 진입 시각
                from datetime import datetime as _dt
                entry_dt_str = _dt.fromtimestamp(p.entry_time).strftime('%Y-%m-%d %H:%M:%S') if p.entry_time else '-'
                # 보유 시간
                hold = max(int(time.time() - p.entry_time), 0) if p.entry_time else 0
                d_, rem = divmod(hold, 86400)
                hh, rem = divmod(rem, 3600)
                mm, ss = divmod(rem, 60)
                if d_ > 0:
                    hold_str = f"{d_}d {hh}h {mm}m"
                elif hh > 0:
                    hold_str = f"{hh}h {mm}m"
                else:
                    hold_str = f"{mm}m {ss}s"
                self.telegram.send(
                    f"<b>[SOL V1] 포지션</b>\n"
                    f"Direction: {d} [{mode}]\n"
                    f"Entry: ${p.entry_price:.3f}\n"
                    f"Current: ${price:.3f}\n"
                    f"Leverage: {eff_lev}x{' (사용자)' if eff_lev != LEVERAGE else ''}\n"
                    f"Size: ${p.position_size:,.0f}\n"
                    f"Margin: ${p.margin_used:,.0f}\n"
                    f"ROI: {roi_str} | PnL: {pnl_str}\n"
                    f"SL: ${p.sl_price:.3f}\n"
                    f"TSL: {'ON' if p.tsl_active else 'OFF'}\n"
                    f"Peak ROI: {p.peak_roi:+.2f}%\n"
                    f"Leg: {p.leg_count}/3\n\n"
                    f"📅 진입 시각: {entry_dt_str}\n"
                    f"⏱ 보유 시간: {hold_str}"
                )

            # === /sol 잔액 ===
            elif sub == '잔액':
                await self.executor.update_balance()
                self.telegram.send(
                    f"<b>[SOL V1] 잔액</b>\n"
                    f"Wallet: ${self.executor.balance:,.2f}\n"
                    f"Available: ${self.executor.available_balance:,.2f}\n"
                    f"Peak: ${self.core.peak_capital:,.2f}\n"
                    f"MDD: {self.core.max_drawdown*100:.1f}%"
                )

            # === /sol 최근거래 (DB에서 직접 조회 — 봇 재시작에도 영속) ===
            elif sub in ('최근거래', '거래'):
                import sqlite3
                from datetime import datetime, timedelta
                try:
                    conn = sqlite3.connect('sol_trading_bot.db')
                    rows = conn.execute(
                        "SELECT timestamp, source, direction, entry_mode, entry_price, exit_price, "
                        "pnl, roi_pct, exit_type, hold_time FROM trades ORDER BY id DESC LIMIT 5"
                    ).fetchall()
                    if not rows:
                        # 청산 거래 없으면 진입 기록(entries)이라도 표시
                        ent_rows = conn.execute(
                            "SELECT timestamp, source, direction, entry_mode, entry_price, "
                            "position_size, margin, leverage, sl_price FROM entries "
                            "ORDER BY id DESC LIMIT 5"
                        ).fetchall()
                        conn.close()
                        if not ent_rows:
                            # ★ DB 모두 비어있어도 메모리상 추적 중인 포지션이 있으면 표시
                            if self.core.has_position:
                                p = self.core.position
                                d_str = 'LONG' if p.direction == 1 else 'SHORT'
                                mode = 'V12' if p.entry_mode == 1 else 'MASS'
                                dir_emoji = '🟢' if p.direction == 1 else '🔴'
                                from datetime import datetime as _dt
                                ent_str = _dt.fromtimestamp(p.entry_time).strftime('%m-%d %H:%M') if p.entry_time else '-'
                                if price > 0 and p.entry_price > 0:
                                    roi = (price - p.entry_price) / p.entry_price * p.direction * 100
                                    pnl = (price - p.entry_price) / p.entry_price * p.position_size * p.direction
                                    pnl_str = f"${pnl:+,.2f} ({roi:+.2f}%)"
                                else:
                                    pnl_str = "N/A"
                                msg = ("<b>[SOL V1] 📥 추적 중인 포지션</b>\n"
                                       "(DB 거래/진입 기록 없음 — 메모리 state 표시)\n\n"
                                       f"{dir_emoji} <b>{d_str}</b> [{mode}] | 진입 {ent_str}\n"
                                       f"  Entry ${p.entry_price:.3f} | Current ${price:.3f}\n"
                                       f"  Size ${p.position_size:,.0f} | Margin ${p.margin_used:,.0f} | Lev {eff_lev}x\n"
                                       f"  SL ${p.sl_price:.3f} | 미실현 PnL: {pnl_str}")
                                self.telegram.send(msg)
                                return
                            self.telegram.send("<b>[SOL V1]</b> 거래/진입 내역 없음")
                            return
                        lines = ["<b>[SOL V1] 📥 최근 진입 (청산 전)</b>",
                                 "(아직 청산 거래 없음 — 진입 기록만 표시)"]
                        for ts, src, d_str, mode, ep, sz, mg, lev, sl in reversed(ent_rows):
                            try:
                                entry_dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                                entry_str = entry_dt.strftime('%m-%d %H:%M')
                            except Exception:
                                entry_str = ts[5:16] if len(ts) >= 16 else ts
                            dir_emoji = '🟢' if d_str == 'LONG' else '🔴'
                            src_str = (src or 'BOT').upper()
                            if 'TG' in src_str: src_tag = 'TG'
                            elif 'USER' in src_str: src_tag = 'USER'
                            else: src_tag = 'BOT'
                            lines.append("")
                            lines.append(f"{dir_emoji} <b>{d_str}</b> [{mode}/{src_tag}] | {entry_str}")
                            lines.append(f"  Entry ${ep:.3f} | Size ${sz:,.0f} | "
                                         f"Margin ${mg:,.0f} | Lev {lev}x | SL ${sl:.3f}")
                        self.telegram.send("\n".join(lines))
                        return

                    def src_tag(s):
                        s = (s or 'BOT').upper()
                        if 'TG' in s: return 'TG'
                        if 'USER' in s: return 'USER'
                        return 'BOT'

                    lines = ["<b>[SOL V1] 📊 최근 거래 (5건)</b>"]
                    for ts, src, d_str, mode, ep, xp, pnl, roi, et, ht in reversed(rows):
                        try:
                            close_dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                            entry_dt = close_dt - timedelta(seconds=ht or 0)
                            entry_str = entry_dt.strftime('%m-%d %H:%M')
                            close_str = close_dt.strftime('%m-%d %H:%M')
                        except Exception:
                            entry_str = '?'
                            close_str = ts[5:16] if len(ts) >= 16 else ts

                        exit_src = src_tag(src)
                        dir_emoji = '🟢' if d_str == 'LONG' else '🔴'
                        pnl_emoji = '✅' if pnl > 0 else '❌'

                        lines.append("")
                        lines.append(f"{dir_emoji} <b>진입</b> | {entry_str} | "
                                     f"<b>{d_str}</b> [{mode}] | ${ep:.3f}")
                        lines.append(f"{pnl_emoji} <b>청산</b> | {close_str} | "
                                     f"<b>{et}</b> [{exit_src}] | ${xp:.3f} | "
                                     f"<b>${pnl:+,.2f}</b> ({roi:+.2f}%)")
                    conn.close()
                    self.telegram.send("\n".join(lines))
                except Exception as e:
                    self.logger.error(f"거래 내역 조회 실패: {e}")
                    self.telegram.send(f"<b>[SOL V1]</b> 거래 조회 실패: {str(e)[:100]}")

            # === /sol 봇정지 ===
            elif sub in ('봇정지', '정지'):
                self.logger.info("텔레그램 /sol 봇정지 명령 수신")
                self.telegram.send("<b>[SOL V1]</b> 봇 정지 요청 (systemd 자동 재시작)")
                self._running = False
                self._shutdown_event.set()

            # === /sol 봇시작 (상태 확인 후 재시작) ===
            elif sub in ('봇시작', '시작'):
                self.logger.info("텔레그램 /sol 봇시작 명령 수신")
                pos_count = 1 if self.core.has_position else 0
                self.telegram.send(
                    f"<b>[SOL V1] 봇 재시작</b>\n"
                    f"잔액: ${self.executor.balance:,.2f} | 포지션: {pos_count}\n"
                    f"Leverage: {LEVERAGE}x\n"
                    f"→ systemd 자동 재시작 진행..."
                )
                self._running = False
                self._shutdown_event.set()

            # === /sol 청산 (수동 포지션 청산) ===
            elif sub == '청산':
                if not self.core.has_position:
                    self.telegram.send("<b>[SOL V1]</b> 청산할 포지션 없음")
                    return
                p = self.core.position
                d = "LONG" if p.direction == 1 else "SHORT"
                self.telegram.send(
                    f"<b>[SOL V1] 수동 청산 처리 중</b>\n"
                    f"{d} @${p.entry_price:.3f} → 시장가 청산 요청"
                )
                # 비동기 실제 청산
                try:
                    await self.executor._cancel_all_sl_orders()
                    result = await self.executor.market_exit(direction=p.direction)
                    if not result:
                        self.telegram.send("<b>[SOL V1]</b> 수동 청산 실패: 거래소 응답 없음")
                        return
                    actual_price = result.get('filled_price') or price or p.entry_price
                    from sol_core_v1 import Signal
                    sig = Signal(action='EXIT', direction=p.direction,
                                 exit_type='TG-MANUAL', exit_price=actual_price)
                    rec = self.core.apply_exit(sig, timestamp=time.time())
                    await self.executor.update_balance()
                    self.core.update_peak(self.executor.balance)
                    await self.executor.save_trade({
                        'direction': rec.direction, 'entry_mode': rec.entry_mode,
                        'entry_price': rec.entry_price, 'exit_price': rec.exit_price,
                        'position_size': rec.position_size, 'pnl': rec.pnl,
                        'exit_type': 'TG-MANUAL', 'roi_pct': rec.roi_pct,
                        'peak_roi': rec.peak_roi, 'leg_count': rec.leg_count,
                        'source': 'TG-USER',
                    })
                    self.telegram.notify_exit({
                        'direction': rec.direction, 'entry_mode': rec.entry_mode,
                        'entry_price': rec.entry_price, 'exit_price': rec.exit_price,
                        'pnl': rec.pnl, 'roi_pct': rec.roi_pct, 'peak_roi': rec.peak_roi,
                        'exit_type': 'TG-MANUAL', 'balance': self.executor.balance,
                    })
                    self.core.save_state()
                except Exception as e:
                    self.logger.error(f"텔레그램 수동 청산 오류: {e}")
                    self.telegram.send(f"<b>[SOL V1]</b> 수동 청산 오류: {str(e)[:150]}")

            # === 알 수 없는 서브커맨드 ===
            else:
                self.telegram.send(f"<b>[SOL V1]</b> 알 수 없는 명령: '/sol {sub}'\n/help 참조")

        except Exception as e:
            self.logger.error(f"명령어 처리 오류: {e}\n{traceback.format_exc()[:300]}")
            self.telegram.send(f"<b>[SOL V1]</b> Error: {str(e)[:200]}")

    async def _shutdown(self):
        self._shutdown_done = True
        self.logger.info("🛑 Shutdown 진행...")
        try:
            self.core.save_state()
            if self.telegram:
                self.telegram.send("🛑 SOL Bot 종료됨")
                await asyncio.sleep(1)
                await self.telegram.stop()
            if self.web:
                await self.web.stop()
            if self.data:
                await self.data.close()
            if self.executor:
                await self.executor.close()
        except Exception as e:
            self.logger.error(f"Shutdown 오류: {e}")
        self.logger.info("✅ Shutdown 완료")


def _install_signal_handlers(bot: SOLTradingBot, loop):
    def _handler(signum, frame):
        bot.logger.info(f"Signal {signum} 수신 → shutdown")
        bot._running = False
        loop.call_soon_threadsafe(bot._shutdown_event.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


async def main():
    logger = setup_logging()
    bot = SOLTradingBot()
    loop = asyncio.get_running_loop()
    _install_signal_handlers(bot, loop)

    try:
        await bot.initialize()
        await bot.run()
    except Exception as e:
        logger.error(f"치명 오류: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
