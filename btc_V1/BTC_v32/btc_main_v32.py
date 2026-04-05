"""
BTC/USDT 선물 자동매매 시스템 v32.2
EMA(100)/EMA(600) Tight-SL Trend System

메인 실행 파일:
- 환경 변수 로드 (.env)
- 모든 컴포넌트 초기화
- 메인 트레이딩 루프 (30초 간격)
- 수동 포지션 자동 추적
- 안전한 종료 처리

사용법:
  python btc_main_v32.py
"""

import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from btc_data_v32 import DataCollector
from btc_core_v32 import TradingCore, SL_PCT, TSL_PCT, TA_PCT, MARGIN_PCT, LEVERAGE
from btc_executor_v32 import OrderExecutor
from btc_monitor_v32 import SystemMonitor
from btc_telegram_v32 import TelegramNotifier

# ═══════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════
VERSION = "32.2"
LOOP_INTERVAL = 30       # 메인 루프 간격 (초)
CANDLE_CHECK = 60        # 캔들 업데이트 체크 간격 (초)
BALANCE_CHECK = 300      # 잔액 갱신 간격 (초)
POSITION_SYNC = 30       # 포지션 동기화 간격 (초)

# ═══════════════════════════════════════════════════════════
# 로깅 설정
# ═══════════════════════════════════════════════════════════
def setup_logging():
    """로그 파일 및 콘솔 출력 설정"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"btc_trading_{datetime.now().strftime('%Y%m%d')}.log"

    formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 파일 핸들러 (INFO 이상만 — DEBUG 제외)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # 콘솔 핸들러 (WARNING 이상만)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(ch)

    # 외부 라이브러리 로그 억제
    for noisy in ('websockets', 'ccxt', 'aiosqlite', 'asyncio',
                  'ccxt.base.exchange', 'websockets.client'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger('btc_main')


# ═══════════════════════════════════════════════════════════
# 메인 트레이딩 봇
# ═══════════════════════════════════════════════════════════
class BTCTradingBot:
    """BTC/USDT 선물 자동매매 시스템"""

    def __init__(self):
        self.logger = logging.getLogger('btc_main')
        self._running = False
        self._shutdown_event = asyncio.Event()

        # 컴포넌트
        self.data: DataCollector = None
        self.core: TradingCore = None
        self.executor: OrderExecutor = None
        self.monitor: SystemMonitor = None
        self.telegram: TelegramNotifier = None

        # 타이머
        self._last_candle_check = 0.0
        self._last_balance_check = 0.0
        self._last_tsl_update_price = 0.0  # TSL SL 주문 업데이트 추적
        self._last_position_sync = 0.0
        self._tsl_notified = False  # TSL 활성화 알림 1회
        self._shutdown_done = False

    async def initialize(self):
        """모든 컴포넌트 초기화"""
        load_dotenv()

        # API 키 로드
        binance_key = os.getenv('BINANCE_API_KEY', '')
        binance_secret = os.getenv('BINANCE_API_SECRET', '')
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        telegram_chat = os.getenv('TELEGRAM_CHAT_ID', '')
        ws_url = os.getenv('BINANCE_WS_FUTURES_URL', 'wss://fstream.binance.com/ws')

        if not binance_key or not binance_secret:
            self.logger.error("BINANCE_API_KEY/SECRET 필수!")
            sys.exit(1)

        # 컴포넌트 생성
        self.data = DataCollector(binance_key, binance_secret, ws_url)
        self.core = TradingCore()
        self.executor = OrderExecutor(binance_key, binance_secret)
        self.monitor = SystemMonitor()
        self.telegram = TelegramNotifier(telegram_token, telegram_chat)

        # 초기화
        self.logger.info("=" * 60)
        self.logger.info(f"  BTC/USDT Futures v{VERSION}")
        self.logger.info(f"  EMA(100)/EMA(600) Tight-SL Trend System")
        self.logger.info("=" * 60)

        await self.data.initialize()
        await self.executor.initialize()
        await self.telegram.start()

        # 모니터 설정
        self.monitor.set_components(self.data, self.core, self.executor)

        # 잔액으로 peak/daily 초기화
        self.core.peak_capital = self.executor.balance
        self.core.set_daily_start(self.executor.balance, self.data.get_latest_index())

        # 수동 포지션 체크
        await self._check_manual_position()

        # 시작 알림 (즉시 전송 대기)
        await self.telegram.notify_start(self.executor.balance)

        self.logger.info(f"초기화 완료 | 잔액: ${self.executor.balance:,.2f}")

    async def _check_manual_position(self):
        """시작 시 포지션 복원: 저장된 상태 → 거래소 확인"""
        ex_pos = await self.executor.get_exchange_position()

        if not ex_pos:
            # 거래소에 포지션 없으면 상태 파일도 정리
            self.core.save_state()
            return

        # 1순위: 저장된 상태 복원 (TSL/SL/TrackHigh 등 유지)
        restored = self.core.load_state()

        if restored and self.core.position.direction == ex_pos['direction']:
            # 저장된 상태와 거래소 방향 일치 → 복원 성공
            # 포지션 크기만 거래소 값으로 동기화
            self.core.position.position_size = ex_pos['notional']
            self.core.position.margin_used = ex_pos['notional'] / LEVERAGE

            # SL 주문 설정 (복원된 SL가 사용)
            sl_price = self.core.position.sl_price
            await self.executor.update_stop_loss(ex_pos['direction'], ex_pos['size'], sl_price)

            dir_str = "LONG" if ex_pos['direction'] == 1 else "SHORT"
            tsl_str = "TSL=ON" if self.core.position.tsl_active else "TSL=OFF"
            self.logger.info(f"상태 복원: {dir_str} @{self.core.position.entry_price:.2f}, "
                           f"{tsl_str}, SL={sl_price:.2f}")
        else:
            # 저장된 상태 없거나 방향 불일치 → 새로 추적
            direction = ex_pos['direction']
            entry_price = ex_pos['entry_price']
            notional = ex_pos['notional']

            self.core.track_manual_position(
                direction=direction,
                entry_price=entry_price,
                size=notional,
                bar_index=self.data.get_latest_index()
            )

            sl_price = self.core.position.sl_price
            await self.executor.update_stop_loss(direction, ex_pos['size'], sl_price)

            self.telegram.notify_manual_position(direction, entry_price, notional)
            self.logger.info(f"수동 포지션 추적: {'LONG' if direction == 1 else 'SHORT'} "
                           f"@{entry_price:.2f}, ${notional:,.0f}")

        self.core.save_state()

    # ═══════════════════════════════════════════════════════════
    # 메인 루프
    # ═══════════════════════════════════════════════════════════
    async def run(self):
        """메인 트레이딩 루프"""
        self._running = True

        # WebSocket 시작
        await self.data.start_websocket()

        # 모니터링 시작
        await self.monitor.start()

        self.logger.info("메인 루프 시작")

        try:
            while self._running:
                try:
                    await self._tick()
                except Exception as e:
                    self.logger.error(f"루프 오류: {e}", exc_info=True)
                    self.telegram.notify_error(f"Loop error: {str(e)[:200]}")

                # 종료 확인
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=LOOP_INTERVAL
                    )
                    break
                except asyncio.TimeoutError:
                    pass

        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            if not self._shutdown_done:
                await self._shutdown()

    async def _tick(self):
        """매 루프 사이클 — 핵심 로직"""
        now = time.time()

        # ─── 캔들 업데이트 ───
        if now - self._last_candle_check >= CANDLE_CHECK:
            await self.data.update_candles()
            self._last_candle_check = now

        # ─── 잔액 갱신 ───
        if now - self._last_balance_check >= BALANCE_CHECK:
            await self.executor.update_balance()
            self.core.update_peak(self.executor.balance)
            self._last_balance_check = now

        # ─── 포지션 동기화 (거래소 ↔ 봇) ───
        if now - self._last_position_sync >= POSITION_SYNC:
            await self._sync_position()
            self._last_position_sync = now

        # ─── 현재 봉 데이터 ───
        bar = self.data.get_current_bar()
        if bar is None:
            return

        bar_index = bar['index']
        capital = self.executor.balance

        # ─── 신호 판단 ───
        signal = self.core.evaluate(bar, capital, bar_index)

        # ─── 신호 처리 ───
        if signal.action == 'EXIT':
            await self._handle_exit(signal, bar)

        elif signal.action == 'ENTER':
            await self._handle_entry(signal, bar, capital, bar_index)

        # ─── 포지션 보유 중 TSL 관리 ───
        if self.core.has_position:
            await self._manage_tsl(bar)

        # ─── 상태 저장 (재시작 시 TSL/SL 복원용) ───
        self.core.save_state()

    async def _handle_entry(self, signal, bar: dict, capital: float, bar_index: int):
        """진입 신호 처리"""
        try:
            # Core에서 마진/포지션크기 계산
            entry_info = self.core.open_position(
                direction=signal.direction,
                entry_price=bar['close'],
                capital=capital,
                bar_index=bar_index,
            )

            # 거래소 주문 실행
            order = await self.executor.market_entry(
                direction=signal.direction,
                position_size_usd=entry_info['position_size'],
                sl_price=entry_info['sl_price'],
            )

            # 잔액 차감 (수수료)
            await self.executor.update_balance()
            self.core.update_peak(self.executor.balance)

            # 진입 기록 저장 (DB + TXT)
            await self.executor.save_entry({
                'source': 'BOT',
                'direction': signal.direction,
                'entry_price': order['filled_price'],
                'position_size': entry_info['position_size'],
                'margin': entry_info['margin'],
                'sl_price': entry_info['sl_price'],
                'balance_before': capital,
                'balance_after': self.executor.balance,
                'adx': bar.get('adx', 0),
                'rsi': bar.get('rsi', 0),
                'ema_gap': bar.get('ema_gap', 0),
                'ema100': bar.get('fast_ma', 0),
                'ema600': bar.get('slow_ma', 0),
            })

            # TSL 알림 플래그 리셋
            self._tsl_notified = False
            self._last_tsl_update_price = 0.0

            # 텔레그램 알림
            self.telegram.notify_entry(
                direction=signal.direction,
                entry_price=order['filled_price'],
                position_size=entry_info['position_size'],
                sl_price=entry_info['sl_price'],
                balance=self.executor.balance,
                bar_info=bar,
            )

            self.logger.info(f"진입 완료: {signal.reason}")

        except Exception as e:
            self.logger.error(f"진입 처리 오류: {e}", exc_info=True)
            self.telegram.notify_error(f"Entry failed: {str(e)[:200]}")
            # 진입 실패 시 Core 상태 롤백
            self.core.position.__init__()

    async def _handle_exit(self, signal, bar: dict):
        """청산 신호 처리"""
        try:
            # 거래소 청산
            order = await self.executor.market_exit(
                direction=signal.direction
            )

            if not order:
                self.logger.warning("청산 주문 없음 (이미 청산됨)")
                self.core.position.__init__()
                return

            # Core 청산 처리
            exit_price = order['filled_price'] if signal.exit_type != 'SL' else signal.exit_price
            trade_info = self.core.close_position(
                exit_price=exit_price,
                exit_type=signal.exit_type,
                bar_index=bar['index'],
            )

            # 잔액 갱신
            await self.executor.update_balance()
            self.core.update_peak(self.executor.balance)

            # DB 저장
            trade_info['source'] = 'BOT'
            await self.executor.save_trade(trade_info)

            # 텔레그램 알림
            self.telegram.notify_exit(trade_info, self.executor.balance)

            self.logger.info(f"청산 완료: {signal.reason} | PnL=${trade_info['pnl']:+,.2f}")

        except Exception as e:
            self.logger.error(f"청산 처리 오류: {e}", exc_info=True)
            self.telegram.notify_error(f"Exit failed: {str(e)[:200]}")

    async def _sync_position(self):
        """거래소 포지션과 봇 상태 동기화 (30초마다)"""
        try:
            ex_pos = await self.executor.get_exchange_position()
            bot_has = self.core.has_position

            # Case 1: 사용자가 수동 진입 (거래소에 포지션 있음, 봇은 없음)
            if ex_pos and not bot_has:
                direction = ex_pos['direction']
                entry_price = ex_pos['entry_price']
                notional = ex_pos['notional']

                self.core.track_manual_position(
                    direction=direction,
                    entry_price=entry_price,
                    size=notional,
                    bar_index=self.data.get_latest_index()
                )

                # SL 주문 설정
                sl_price = self.core.position.sl_price
                qty = ex_pos['size']
                await self.executor.update_stop_loss(direction, qty, sl_price)

                # 진입 기록 저장 (source: USER)
                bar = self.data.get_current_bar()
                await self.executor.save_entry({
                    'source': 'USER',
                    'direction': direction,
                    'entry_price': entry_price,
                    'position_size': notional,
                    'margin': notional / LEVERAGE,
                    'sl_price': sl_price,
                    'balance_before': self.executor.balance,
                    'balance_after': self.executor.balance,
                    'adx': bar.get('adx', 0) if bar else 0,
                    'rsi': bar.get('rsi', 0) if bar else 0,
                    'ema_gap': bar.get('ema_gap', 0) if bar else 0,
                    'ema100': bar.get('fast_ma', 0) if bar else 0,
                    'ema600': bar.get('slow_ma', 0) if bar else 0,
                })

                self.telegram.notify_manual_position(direction, entry_price, notional)
                self._tsl_notified = False
                self._last_tsl_update_price = 0.0

                dir_str = "LONG" if direction == 1 else "SHORT"
                self.logger.info(f"[SYNC] USER 진입 감지: {dir_str} @{entry_price:.2f}, ${notional:,.0f}")

            # Case 2: 사용자가 수동 청산 (거래소에 포지션 없음, 봇은 있음)
            elif not ex_pos and bot_has:
                pos = self.core.position
                dir_str = "LONG" if pos.direction == 1 else "SHORT"

                # 실제 청산가 조회 (최근 체결 내역에서)
                actual_exit_price = await self.executor.get_last_exit_price()
                if actual_exit_price <= 0:
                    actual_exit_price = self.data.current_price if self.data.current_price > 0 else pos.entry_price

                trade_info = self.core.close_position(
                    exit_price=actual_exit_price,
                    exit_type='MANUAL',
                    bar_index=self.data.get_latest_index(),
                )

                await self.executor.update_balance()
                self.core.update_peak(self.executor.balance)

                # 청산 기록 저장 (source: USER)
                trade_info['source'] = 'USER'
                await self.executor.save_trade(trade_info)

                self.telegram.notify_exit(trade_info, self.executor.balance)
                self.logger.info(f"[SYNC] USER 청산 감지: {dir_str} @{actual_exit_price:.2f} PnL=${trade_info['pnl']:+,.2f}")

            # Case 3: 방향 불일치 (봇은 LONG인데 거래소는 SHORT 등)
            elif ex_pos and bot_has:
                if ex_pos['direction'] != self.core.position.direction:
                    # 실제 청산가 조회
                    actual_exit_price = await self.executor.get_last_exit_price()
                    if actual_exit_price <= 0:
                        actual_exit_price = self.data.current_price if self.data.current_price > 0 else self.core.position.entry_price

                    # 기존 포지션 청산 처리
                    trade_info = self.core.close_position(
                        exit_price=actual_exit_price,
                        exit_type='MANUAL',
                        bar_index=self.data.get_latest_index(),
                    )
                    trade_info['source'] = 'USER'
                    await self.executor.save_trade(trade_info)

                    # 새 포지션 등록
                    self.core.track_manual_position(
                        direction=ex_pos['direction'],
                        entry_price=ex_pos['entry_price'],
                        size=ex_pos['notional'],
                        bar_index=self.data.get_latest_index()
                    )
                    self._tsl_notified = False
                    self._last_tsl_update_price = 0.0

                    self.telegram.notify_manual_position(
                        ex_pos['direction'], ex_pos['entry_price'], ex_pos['notional']
                    )
                    self.logger.info(f"[SYNC] USER 포지션 방향 전환 감지")

        except Exception as e:
            self.logger.error(f"포지션 동기화 오류: {e}")

    async def _manage_tsl(self, bar: dict):
        """TSL 활성 시 거래소 SL 주문 업데이트"""
        pos = self.core.position
        if not pos.tsl_active:
            return

        # TSL 활성화 알림 (1회)
        if not self._tsl_notified:
            self._tsl_notified = True
            price = bar['close']
            roi = (price - pos.entry_price) / pos.entry_price * pos.direction * 100
            self.telegram.notify_tsl_activated(pos.direction, pos.entry_price, price, roi)

        # SL가가 변경되었으면 거래소 주문 업데이트
        current_sl = pos.sl_price
        if abs(current_sl - self._last_tsl_update_price) > 1.0:
            try:
                ex_pos = await self.executor.get_exchange_position()
                if ex_pos:
                    await self.executor.update_stop_loss(
                        pos.direction, ex_pos['size'], current_sl
                    )
                    self._last_tsl_update_price = current_sl

                    extreme = pos.track_high if pos.direction == 1 else pos.track_low
                    self.telegram.notify_tsl_update(current_sl, extreme, pos.direction)
            except Exception as e:
                self.logger.error(f"TSL 업데이트 오류: {e}")

    # ═══════════════════════════════════════════════════════════
    # 종료
    # ═══════════════════════════════════════════════════════════
    def request_shutdown(self):
        """안전한 종료 요청"""
        self.logger.info("종료 요청 수신")
        self._running = False
        self._shutdown_event.set()

    async def _shutdown(self):
        """컴포넌트 순차 정리 (1회만 실행)"""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._running = False

        print("\n  Shutting down...")
        self.logger.info("시스템 종료 중...")

        # 모니터링 먼저 중지 (화면 클리어 방지)
        if self.monitor:
            await self.monitor.stop()

        # 포지션 경고
        if self.core and self.core.has_position:
            pos = self.core.position
            dir_str = "LONG" if pos.direction == 1 else "SHORT"
            self.logger.warning(f"열린 포지션 있음! {dir_str} @{pos.entry_price:.2f}")
            print(f"  [WARNING] Open position: {dir_str} @${pos.entry_price:,.2f}")
            print(f"  SL order remains active on exchange.")

        # 종료 알림 (5초 타임아웃)
        if self.telegram and self.telegram.enabled and self.telegram._session and not self.telegram._session.closed:
            try:
                total_trades = self.core.total_trades if self.core else 0
                balance = self.executor.balance if self.executor else 0
                await asyncio.wait_for(
                    self.telegram.notify_stop(balance, total_trades),
                    timeout=5
                )
            except (asyncio.TimeoutError, Exception) as e:
                self.logger.warning(f"종료 알림 전송 실패: {e}")
            try:
                await asyncio.wait_for(self.telegram.stop(), timeout=3)
            except (asyncio.TimeoutError, Exception):
                pass

        # 나머지 컴포넌트 순차 종료
        import warnings
        warnings.filterwarnings('ignore')
        logging.getLogger('ccxt.base.exchange').setLevel(logging.CRITICAL)
        logging.getLogger('asyncio').setLevel(logging.CRITICAL)

        try:
            if self.data:
                await self.data.stop_websocket()
                await asyncio.wait_for(self.data.exchange.close(), timeout=5)
        except Exception:
            pass

        try:
            if self.executor:
                if self.executor.db:
                    await self.executor.db.close()
                await asyncio.wait_for(self.executor.exchange.close(), timeout=5)
        except Exception:
            pass

        print("  System stopped.")


# ═══════════════════════════════════════════════════════════
# 엔트리 포인트
# ═══════════════════════════════════════════════════════════
async def main():
    logger = setup_logging()
    bot = BTCTradingBot()

    # Windows용 Ctrl+C 처리: SIGINT를 shutdown으로 연결
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, bot.request_shutdown)
        except NotImplementedError:
            pass  # Windows — KeyboardInterrupt로 처리

    try:
        await bot.initialize()
        await bot.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.critical(f"치명적 오류: {e}", exc_info=True)
        if bot.telegram and bot.telegram.enabled:
            try:
                bot.telegram.notify_error(f"FATAL: {str(e)[:300]}")
                await asyncio.sleep(2)
            except Exception:
                pass
    finally:
        await bot._shutdown()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
