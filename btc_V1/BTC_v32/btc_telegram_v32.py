"""
BTC/USDT 선물 자동매매 - 텔레그램 알림
v32.2: EMA(100)/EMA(600) Tight-SL Trend System

- 비동기 텔레그램 메시지 전송
- 진입/청산/에러/시스템 이벤트 알림
- 일일 요약 리포트
- 메시지 큐 및 재시도
"""

import asyncio
import logging
from datetime import datetime, timedelta

import aiohttp

logger = logging.getLogger('btc_telegram')

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """비동기 텔레그램 알림 전송"""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
        self._session: aiohttp.ClientSession = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task = None
        self._running = False

        if not self.enabled:
            logger.warning("텔레그램 알림 비활성화 (토큰/채팅ID 없음)")

    async def start(self):
        """알림 전송 루프 시작"""
        if not self.enabled:
            return
        self._session = aiohttp.ClientSession()
        self._running = True
        self._task = asyncio.create_task(self._send_loop())
        logger.info("텔레그램 알림 시작")

    async def stop(self):
        """알림 전송 중지 (큐 잔여 메시지 flush 후 종료)"""
        self._running = False
        # 큐에 남은 메시지 모두 전송
        if self._session:
            while not self._queue.empty():
                try:
                    msg = self._queue.get_nowait()
                    await self._send(msg)
                except Exception:
                    break
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()

    async def _send_loop(self):
        """메시지 큐에서 꺼내 전송"""
        while self._running:
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._send(msg)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"텔레그램 전송 루프 오류: {e}")
                await asyncio.sleep(5)

    async def _send(self, text: str, retry: int = 3):
        """텔레그램 메시지 전송 (재시도)"""
        url = TELEGRAM_API.format(token=self.bot_token)
        payload = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True,
        }
        for attempt in range(retry):
            try:
                async with self._session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        return True
                    else:
                        body = await resp.text()
                        logger.warning(f"텔레그램 응답 {resp.status}: {body}")
            except Exception as e:
                logger.warning(f"텔레그램 전송 실패 (시도 {attempt+1}/{retry}): {e}")
                await asyncio.sleep(2)
        return False

    def _enqueue(self, text: str):
        """메시지 큐에 추가"""
        if self.enabled:
            self._queue.put_nowait(text)

    # ═══════════════════════════════════════════════════════════
    # 알림 메서드
    # ═══════════════════════════════════════════════════════════
    async def notify_start(self, balance: float):
        """시스템 시작 알림 (즉시 전송, await 필수)"""
        msg = (
            "<b>[BTC v32.2] System Started</b>\n"
            f"Strategy: EMA(100)/EMA(600) Tight-SL\n"
            f"Balance: ${balance:,.2f}\n"
            f"Leverage: 10x | Margin: 35% | SL: 3%\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if self.enabled and self._session:
            result = await self._send(msg)
            if result:
                logger.info("시작 알림 전송 완료")
            else:
                logger.error("시작 알림 전송 실패")

    async def notify_stop(self, balance: float, total_trades: int):
        """시스템 종료 알림 (즉시 전송, await 필수)"""
        msg = (
            "<b>[BTC v32.2] System Stopped</b>\n"
            f"Balance: ${balance:,.2f}\n"
            f"Total Trades: {total_trades}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if self.enabled and self._session:
            await self._send(msg)

    def notify_entry(self, direction: int, entry_price: float,
                     position_size: float, sl_price: float,
                     balance: float, bar_info: dict = None):
        """포지션 진입 알림"""
        dir_str = "LONG" if direction == 1 else "SHORT"
        dir_emoji = "UP" if direction == 1 else "DN"
        msg = f"<b>[BTC v32.2] {dir_emoji} {dir_str} Entry</b>\n"
        msg += f"Price: ${entry_price:,.2f}\n"
        msg += f"Size: ${position_size:,.0f}\n"
        msg += f"SL: ${sl_price:,.2f} (-{3.0}%)\n"
        msg += f"Balance: ${balance:,.2f}\n"
        if bar_info:
            msg += f"ADX: {bar_info.get('adx', 0):.1f} | "
            msg += f"RSI: {bar_info.get('rsi', 0):.1f} | "
            msg += f"Gap: {bar_info.get('ema_gap', 0):.3f}%\n"
        msg += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self._enqueue(msg)

    def notify_exit(self, trade_info: dict, balance: float):
        """포지션 청산 알림"""
        dir_str = "LONG" if trade_info['direction'] == 1 else "SHORT"
        exit_type = trade_info['exit_type']
        pnl = trade_info['pnl']
        roi = trade_info['roi_pct']
        pnl_str = "PROFIT" if pnl > 0 else "LOSS"

        msg = f"<b>[BTC v32.2] {dir_str} {exit_type} Exit ({pnl_str})</b>\n"
        msg += f"Entry: ${trade_info['entry_price']:,.2f}\n"
        msg += f"Exit: ${trade_info['exit_price']:,.2f}\n"
        msg += f"PnL: ${pnl:+,.2f} ({roi:+.2f}%)\n"
        msg += f"Peak ROI: {trade_info.get('peak_roi', 0):+.2f}%\n"

        if trade_info.get('hold_time'):
            hold = str(timedelta(seconds=int(trade_info['hold_time'])))
            msg += f"Hold: {hold}\n"

        if trade_info.get('tsl_active'):
            msg += f"TSL: Active\n"

        msg += f"Balance: ${balance:,.2f}\n"
        msg += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self._enqueue(msg)

    def notify_tsl_activated(self, direction: int, entry_price: float,
                             current_price: float, roi: float):
        """TSL 활성화 알림"""
        dir_str = "LONG" if direction == 1 else "SHORT"
        msg = (
            f"<b>[BTC v32.2] TSL Activated ({dir_str})</b>\n"
            f"Entry: ${entry_price:,.2f}\n"
            f"Current: ${current_price:,.2f}\n"
            f"ROI: {roi:+.2f}%\n"
            f"TSL will trail at -9% from peak\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._enqueue(msg)

    def notify_tsl_update(self, new_sl: float, track_extreme: float,
                          direction: int):
        """TSL 업데이트 알림"""
        dir_str = "LONG" if direction == 1 else "SHORT"
        extreme_str = "High" if direction == 1 else "Low"
        msg = (
            f"<b>[BTC v32.2] TSL Updated ({dir_str})</b>\n"
            f"Track {extreme_str}: ${track_extreme:,.2f}\n"
            f"New SL: ${new_sl:,.2f}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._enqueue(msg)

    def notify_watch(self, direction: int, bar_index: int):
        """감시 시작 알림"""
        dir_str = "LONG" if direction == 1 else "SHORT"
        msg = (
            f"<b>[BTC v32.2] Watch Started: {dir_str}</b>\n"
            f"EMA Cross detected at bar {bar_index}\n"
            f"Monitoring for 24 bars (12h)\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._enqueue(msg)

    def notify_manual_position(self, direction: int, entry_price: float,
                               size: float):
        """수동 포지션 추적 알림"""
        dir_str = "LONG" if direction == 1 else "SHORT"
        msg = (
            f"<b>[BTC v32.2] Manual Position Detected</b>\n"
            f"Direction: {dir_str}\n"
            f"Entry: ${entry_price:,.2f}\n"
            f"Size: ${size:,.0f}\n"
            f"Auto-management enabled (SL/TSL/REV)\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._enqueue(msg)

    def notify_daily_summary(self, stats: dict):
        """일일 요약 알림"""
        msg = (
            f"<b>[BTC v32.2] Daily Summary</b>\n"
            f"Date: {stats.get('date', 'N/A')}\n"
            f"Balance: ${stats.get('balance', 0):,.2f}\n"
            f"Daily PnL: ${stats.get('daily_pnl', 0):+,.2f}\n"
            f"Trades: {stats.get('trades', 0)}\n"
            f"W/L: {stats.get('wins', 0)}W / {stats.get('losses', 0)}L\n"
            f"Total Trades: {stats.get('total_trades', 0)}\n"
            f"MDD: {stats.get('mdd', 0):.1f}%"
        )
        self._enqueue(msg)

    def notify_error(self, error_msg: str):
        """시스템 오류 알림"""
        msg = (
            f"<b>[BTC v32.2] ERROR</b>\n"
            f"{error_msg}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._enqueue(msg)

    def notify_daily_loss_limit(self, daily_return: float, balance: float):
        """일일 손실 제한 알림"""
        msg = (
            f"<b>[BTC v32.2] Daily Loss Limit Hit</b>\n"
            f"Daily Return: {daily_return*100:.1f}%\n"
            f"No new entries until next day reset\n"
            f"Balance: ${balance:,.2f}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._enqueue(msg)
