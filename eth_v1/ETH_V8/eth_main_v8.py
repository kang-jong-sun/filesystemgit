"""
ETH/USDT 선물 자동매매 시스템 V8.16
EMA(250)/EMA(1575) 10m Trend System — 계정 20% 복리 마진

사용법:
  python eth_main_v8.py
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

from eth_data_v8 import DataCollector
from eth_core_v8 import TradingCore, SL_PCT, TSL_PCT, TA_PCT, MARGIN_PCT, LEVERAGE
from eth_executor_v8 import OrderExecutor
from eth_monitor_v8 import SystemMonitor
from eth_telegram_v8 import TelegramNotifier

VERSION = "8.16"
LOOP_INTERVAL = 30
CANDLE_CHECK = 60
BALANCE_CHECK = 300
POSITION_SYNC = 30


def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"eth_trading_{datetime.now().strftime('%Y%m%d')}.log"
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(ch)

    for noisy in ('websockets', 'ccxt', 'aiosqlite', 'asyncio', 'ccxt.base.exchange', 'websockets.client'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger('eth_main')


class ETHTradingBot:
    def __init__(self):
        self.logger = logging.getLogger('eth_main')
        self._running = False
        self._shutdown_event = asyncio.Event()
        self.data: DataCollector = None
        self.core: TradingCore = None
        self.executor: OrderExecutor = None
        self.monitor: SystemMonitor = None
        self.telegram: TelegramNotifier = None
        self._last_candle_check = 0.0
        self._last_balance_check = 0.0
        self._last_tsl_update_price = 0.0
        self._last_position_sync = 0.0
        self._tsl_notified = False
        self._shutdown_done = False

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
        self.monitor = SystemMonitor()
        self.telegram = TelegramNotifier(telegram_token, telegram_chat)

        self.logger.info("=" * 60)
        self.logger.info(f"  ETH/USDT Futures V{VERSION}")
        self.logger.info(f"  EMA(250)/EMA(1575) 10m | Margin 20% | 10x")
        self.logger.info("=" * 60)

        await self.data.initialize()
        await self.executor.initialize()
        await self.telegram.start()

        self.monitor.set_components(self.data, self.core, self.executor)
        self.core.peak_capital = self.executor.balance
        await self._check_manual_position()
        await self.telegram.notify_start(self.executor.balance)
        self.logger.info(f"초기화 완료 | 잔액: ${self.executor.balance:,.2f}")

    async def _check_manual_position(self):
        # ★ 통계 먼저 복원 (포지션 유무와 무관하게 항상)
        self.core.load_state()

        ex_pos = await self.executor.get_exchange_position()
        if not ex_pos:
            self.core.save_state()
            return

        restored = self.core.has_position
        if restored and self.core.position.direction == ex_pos['direction']:
            self.core.position.position_size = ex_pos['notional']
            self.core.position.margin_used = ex_pos['notional'] / LEVERAGE
            sl_price = self.core.position.sl_price
            await self.executor.update_stop_loss(ex_pos['direction'], ex_pos['size'], sl_price)
        else:
            self.core.track_manual_position(
                direction=ex_pos['direction'], entry_price=ex_pos['entry_price'],
                size=ex_pos['notional'], bar_index=self.data.get_latest_index())
            sl_price = self.core.position.sl_price
            await self.executor.update_stop_loss(ex_pos['direction'], ex_pos['size'], sl_price)
            self.telegram.notify_manual_position(ex_pos['direction'], ex_pos['entry_price'], ex_pos['notional'])

        self.core.save_state()

    async def run(self):
        self._running = True
        await self.data.start_websocket()
        await self.monitor.start()
        self.logger.info("메인 루프 시작")

        try:
            while self._running:
                try:
                    await self._tick()
                except Exception as e:
                    self.logger.error(f"루프 오류: {e}", exc_info=True)
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

        if now - self._last_candle_check >= CANDLE_CHECK:
            await self.data.update_candles()
            self._last_candle_check = now

        if now - self._last_balance_check >= BALANCE_CHECK:
            await self.executor.update_balance()
            self.core.update_peak(self.executor.balance)
            self._last_balance_check = now

        if now - self._last_position_sync >= POSITION_SYNC:
            await self._sync_position()
            self._last_position_sync = now

        bar = self.data.get_current_bar()
        if bar is None: return

        # ★ 실시간 가격으로 SL/TSL 체크 (30초마다)
        if self.core.has_position and self.data.current_price > 0:
            rt_signal = self.core.check_realtime_exit(
                self.data.current_price, self.data.current_high, self.data.current_low)
            if rt_signal and rt_signal.action == 'EXIT':
                self.logger.info(f"실시간 SL/TSL 발동: {rt_signal.reason}")
                await self._handle_exit(rt_signal, bar)
                self.core.save_state()
                return

        bar_index = bar['index']
        capital = self.executor.balance
        signal = self.core.evaluate(bar, capital, bar_index)

        if signal.action == 'EXIT':
            await self._handle_exit(signal, bar)
        elif signal.action == 'ENTER':
            await self._handle_entry(signal, bar, capital, bar_index)

        # 전략B 체크 (전략A가 진입 안 했을 때만)
        if not self.core.has_position:
            signal_b = self.core.check_entry_b(bar, capital)
            if signal_b.action == 'ENTER':
                await self._handle_entry(signal_b, bar, capital, bar_index)

        if self.core.has_position:
            await self._manage_tsl(bar)

        self.core.save_state()

    async def _handle_entry(self, signal, bar, capital, bar_index):
        try:
            entry_info = self.core.open_position(
                direction=signal.direction, entry_price=bar['close'],
                capital=capital, bar_index=bar_index,
                entry_mode=signal.entry_mode)

            order = await self.executor.market_entry(
                direction=signal.direction, position_size_usd=entry_info['position_size'],
                sl_price=entry_info['sl_price'])

            await self.executor.update_balance()
            self.core.update_peak(self.executor.balance)

            await self.executor.save_entry({
                'source': 'BOT', 'direction': signal.direction,
                'entry_price': order['filled_price'],
                'position_size': entry_info['position_size'],
                'margin': entry_info['margin'], 'sl_price': entry_info['sl_price'],
                'balance_before': capital, 'balance_after': self.executor.balance,
                'ema250': bar.get('fast_ma', 0), 'ema1575': bar.get('slow_ma', 0),
            })

            self._tsl_notified = False
            self._last_tsl_update_price = 0.0
            self.telegram.notify_entry(
                direction=signal.direction, entry_price=order['filled_price'],
                position_size=entry_info['position_size'], sl_price=entry_info['sl_price'],
                balance=self.executor.balance, bar_info=bar)

        except Exception as e:
            self.logger.error(f"진입 처리 오류: {e}", exc_info=True)
            self.telegram.notify_error(f"Entry failed: {str(e)[:200]}")
            self.core.position.__init__()

    async def _handle_exit(self, signal, bar):
        try:
            order = await self.executor.market_exit(direction=signal.direction)
            if not order:
                self.core.position.__init__()
                return

            exit_price = order['filled_price'] if signal.exit_type != 'SL' else signal.exit_price
            trade_info = self.core.close_position(
                exit_price=exit_price, exit_type=signal.exit_type, bar_index=bar['index'])

            await self.executor.update_balance()
            self.core.update_peak(self.executor.balance)
            trade_info['source'] = 'BOT'
            await self.executor.save_trade(trade_info)
            self.telegram.notify_exit(trade_info, self.executor.balance)

        except Exception as e:
            self.logger.error(f"청산 처리 오류: {e}", exc_info=True)
            self.telegram.notify_error(f"Exit failed: {str(e)[:200]}")

    async def _sync_position(self):
        try:
            ex_pos = await self.executor.get_exchange_position()
            bot_has = self.core.has_position

            if ex_pos and not bot_has:
                self.core.track_manual_position(
                    direction=ex_pos['direction'], entry_price=ex_pos['entry_price'],
                    size=ex_pos['notional'], bar_index=self.data.get_latest_index())
                sl_price = self.core.position.sl_price
                await self.executor.update_stop_loss(ex_pos['direction'], ex_pos['size'], sl_price)
                await self.executor.save_entry({
                    'source': 'USER', 'direction': ex_pos['direction'],
                    'entry_price': ex_pos['entry_price'], 'position_size': ex_pos['notional'],
                    'margin': ex_pos['notional'] / LEVERAGE, 'sl_price': sl_price,
                    'balance_before': self.executor.balance, 'balance_after': self.executor.balance,
                    'ema250': 0, 'ema1575': 0})
                self.telegram.notify_manual_position(ex_pos['direction'], ex_pos['entry_price'], ex_pos['notional'])
                self._tsl_notified = False

            elif not ex_pos and bot_has:
                actual_exit_price = await self.executor.get_last_exit_price()
                if actual_exit_price <= 0:
                    actual_exit_price = self.data.current_price or self.core.position.entry_price
                trade_info = self.core.close_position(
                    exit_price=actual_exit_price, exit_type='MANUAL',
                    bar_index=self.data.get_latest_index())
                await self.executor.update_balance()
                self.core.update_peak(self.executor.balance)
                trade_info['source'] = 'USER'
                await self.executor.save_trade(trade_info)
                self.telegram.notify_exit(trade_info, self.executor.balance)

            elif ex_pos and bot_has and ex_pos['direction'] != self.core.position.direction:
                actual_exit_price = await self.executor.get_last_exit_price()
                if actual_exit_price <= 0:
                    actual_exit_price = self.data.current_price or self.core.position.entry_price
                trade_info = self.core.close_position(
                    exit_price=actual_exit_price, exit_type='MANUAL',
                    bar_index=self.data.get_latest_index())
                trade_info['source'] = 'USER'
                await self.executor.save_trade(trade_info)
                self.core.track_manual_position(
                    direction=ex_pos['direction'], entry_price=ex_pos['entry_price'],
                    size=ex_pos['notional'], bar_index=self.data.get_latest_index())
                self._tsl_notified = False
                self.telegram.notify_manual_position(ex_pos['direction'], ex_pos['entry_price'], ex_pos['notional'])

        except Exception as e:
            self.logger.error(f"포지션 동기화 오류: {e}")

    async def _manage_tsl(self, bar):
        pos = self.core.position
        if not pos.tsl_active: return

        if not self._tsl_notified:
            self._tsl_notified = True
            price = bar['close']
            roi = (price - pos.entry_price) / pos.entry_price * pos.direction * 100
            self.telegram.notify_tsl_activated(pos.direction, pos.entry_price, price, roi)

        current_sl = pos.sl_price
        if abs(current_sl - self._last_tsl_update_price) > 0.5:
            try:
                ex_pos = await self.executor.get_exchange_position()
                if ex_pos:
                    await self.executor.update_stop_loss(pos.direction, ex_pos['size'], current_sl)
                    self._last_tsl_update_price = current_sl
                    extreme = pos.track_high if pos.direction == 1 else pos.track_low
                    self.telegram.notify_tsl_update(current_sl, extreme, pos.direction)
            except Exception as e:
                self.logger.error(f"TSL 업데이트 오류: {e}")

    def request_shutdown(self):
        self._running = False
        self._shutdown_event.set()

    async def _shutdown(self):
        if self._shutdown_done: return
        self._shutdown_done = True
        self._running = False
        print("\n  Shutting down...")

        if self.monitor: await self.monitor.stop()

        if self.core and self.core.has_position:
            pos = self.core.position
            dir_str = "LONG" if pos.direction == 1 else "SHORT"
            print(f"  [WARNING] Open position: {dir_str} @${pos.entry_price:,.2f}")

        if self.telegram and self.telegram.enabled and self.telegram._session and not self.telegram._session.closed:
            try:
                total_trades = self.core.total_trades if self.core else 0
                balance = self.executor.balance if self.executor else 0
                await asyncio.wait_for(self.telegram.notify_stop(balance, total_trades), timeout=5)
            except Exception: pass
            try: await asyncio.wait_for(self.telegram.stop(), timeout=3)
            except Exception: pass

        import warnings; warnings.filterwarnings('ignore')
        logging.getLogger('ccxt.base.exchange').setLevel(logging.CRITICAL)
        logging.getLogger('asyncio').setLevel(logging.CRITICAL)

        try:
            if self.data:
                await self.data.stop_websocket()
                await asyncio.wait_for(self.data.exchange.close(), timeout=5)
        except Exception: pass
        try:
            if self.executor:
                if self.executor.db: await self.executor.db.close()
                await asyncio.wait_for(self.executor.exchange.close(), timeout=5)
        except Exception: pass

        print("  System stopped.")


async def main():
    logger = setup_logging()
    bot = ETHTradingBot()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try: loop.add_signal_handler(sig, bot.request_shutdown)
        except NotImplementedError: pass

    try:
        await bot.initialize()
        await bot.run()
    except KeyboardInterrupt: pass
    except Exception as e:
        logger.critical(f"치명적 오류: {e}", exc_info=True)
        if bot.telegram and bot.telegram.enabled:
            try:
                bot.telegram.notify_error(f"FATAL: {str(e)[:300]}")
                await asyncio.sleep(2)
            except Exception: pass
    finally:
        await bot._shutdown()


if __name__ == '__main__':
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
