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
POSITION_SYNC = 60         # 포지션 동기화 주기 (초)
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
            self.core.save_state()
            return

        # 봇 내부 상태와 비교
        if self.core.has_position and self.core.position.direction == ex_pos['direction']:
            # 기존 상태 유지, 사이즈만 동기화
            self.core.position.position_size = ex_pos['notional']
            self.core.position.margin_used = ex_pos['notional'] / LEVERAGE
            self.logger.info("기존 포지션 상태 복원")
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
            self.core.position = PositionState(
                direction=direction, entry_mode=int(EntryMode.V12),
                entry_price=entry_price, entry_price_leg1=entry_price,
                position_size=ex_pos['notional'],
                margin_used=ex_pos['notional'] / LEVERAGE,
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
        """봇 상태 vs 거래소 포지션 동기화"""
        try:
            ex_pos = await self.executor.get_exchange_position()
            if self.core.has_position and not ex_pos:
                # 봇은 포지션 있다고 생각 but 거래소에 없음 → 외부 청산 감지
                self.logger.warning("외부 청산 감지! 상태 리셋.")
                pos_before = self.core.position
                dir_str = 'LONG' if pos_before.direction == 1 else 'SHORT'
                # 📈 trades.log 기록 (외부 청산)
                self.trade_log.info(
                    f"[EXTERNAL EXIT] {dir_str} @${self.data.current_price:.3f} | "
                    f"Entry ${pos_before.entry_price:.3f} | 수동/SL 외부 체결"
                )
                from sol_core_v1 import Signal
                rec_sig = Signal(action='EXIT', direction=self.core.position.direction,
                                 exit_type='EXT', exit_price=self.data.current_price)
                self.core.apply_exit(rec_sig, timestamp=time.time())
                self.telegram.notify_error(f"외부 청산 감지. 봇 상태 동기화.")
                self.core.save_state()
            elif not self.core.has_position and ex_pos:
                # 봇은 없다고 하는데 거래소에 포지션 → 수동 진입 감지
                self.logger.warning("수동 포지션 감지!")
                await self._check_manual_position()
        except Exception as e:
            self.logger.error(f"포지션 동기화 오류: {e}")

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
        }
        self.telegram.notify_status(stats)

    async def _handle_command(self, cmd: str):
        cmd = cmd.strip().lower()
        if cmd == '/status':
            self._send_status_report()
        elif cmd == '/balance':
            await self.executor.update_balance()
            self.telegram.send(f"💰 잔액: ${self.executor.balance:,.2f}")
        elif cmd == '/stop':
            self.telegram.send("🛑 봇 종료 요청 접수. 현재 포지션 유지 후 종료.")
            self._running = False
            self._shutdown_event.set()
        elif cmd == '/close':
            if self.core.has_position:
                pos = self.core.position
                result = await self.executor.market_exit(direction=pos.direction)
                if result:
                    from sol_core_v1 import Signal
                    sig = Signal(action='EXIT', direction=pos.direction,
                                 exit_type='MANUAL', exit_price=result['filled_price'])
                    await self._handle_exit(sig)
                    self.telegram.send("✅ 수동 청산 완료")
            else:
                self.telegram.send("포지션 없음")
        elif cmd == '/help':
            self.telegram.send(
                "📋 <b>사용 가능 명령</b>\n\n"
                "/status - 상태 리포트\n"
                "/balance - 잔액 조회\n"
                "/close - 수동 청산\n"
                "/stop - 봇 종료\n"
                "/help - 이 메시지"
            )

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
