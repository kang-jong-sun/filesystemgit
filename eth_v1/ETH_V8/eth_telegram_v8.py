"""
ETH/USDT 선물 자동매매 - 텔레그램 알림
V8.16: EMA(250)/EMA(1575) 10m Trend System
"""

import asyncio
import logging
from datetime import datetime, timedelta
import aiohttp

logger = logging.getLogger('eth_telegram')
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
        self._session: aiohttp.ClientSession = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task = None
        self._running = False
        if not self.enabled:
            logger.warning("텔레그램 알림 비활성화")

    async def start(self):
        if not self.enabled: return
        self._session = aiohttp.ClientSession()
        self._running = True
        self._task = asyncio.create_task(self._send_loop())

    async def stop(self):
        self._running = False
        if self._session:
            while not self._queue.empty():
                try:
                    msg = self._queue.get_nowait()
                    await self._send(msg)
                except Exception: break
        if self._task:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
        if self._session: await self._session.close()

    async def _send_loop(self):
        while self._running:
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._send(msg)
            except asyncio.TimeoutError: continue
            except Exception as e:
                logger.error(f"텔레그램 전송 오류: {e}")
                await asyncio.sleep(5)

    async def _send(self, text: str, retry: int = 3):
        url = TELEGRAM_API.format(token=self.bot_token)
        payload = {'chat_id': self.chat_id, 'text': text, 'parse_mode': 'HTML', 'disable_web_page_preview': True}
        for attempt in range(retry):
            try:
                async with self._session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200: return True
            except Exception:
                await asyncio.sleep(2)
        return False

    def _enqueue(self, text: str):
        if self.enabled: self._queue.put_nowait(text)

    async def notify_start(self, balance: float):
        msg = (f"<b>[ETH V8.16] System Started</b>\n"
               f"Strategy: EMA(250)/EMA(1575) 10m\n"
               f"Balance: ${balance:,.2f}\n"
               f"Leverage: 10x | Margin: 20% | SL: 2%\n"
               f"TA: 54% | TSL: 8% | Filters: OFF\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.enabled and self._session:
            await self._send(msg)

    async def notify_stop(self, balance: float, total_trades: int):
        msg = (f"<b>[ETH V8.16] System Stopped</b>\n"
               f"Balance: ${balance:,.2f} | Trades: {total_trades}\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.enabled and self._session:
            await self._send(msg)

    def notify_entry(self, direction: int, entry_price: float, position_size: float,
                     sl_price: float, balance: float, bar_info: dict = None):
        dir_str = "LONG" if direction == 1 else "SHORT"
        msg = (f"<b>[ETH V8.16] {dir_str} Entry</b>\n"
               f"Price: ${entry_price:,.2f} | Size: ${position_size:,.0f}\n"
               f"SL: ${sl_price:,.2f} (-2.0%)\n"
               f"Balance: ${balance:,.2f}\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._enqueue(msg)

    def notify_exit(self, trade_info: dict, balance: float):
        dir_str = "LONG" if trade_info['direction'] == 1 else "SHORT"
        pnl = trade_info['pnl']
        msg = (f"<b>[ETH V8.16] {dir_str} {trade_info['exit_type']} ({'PROFIT' if pnl > 0 else 'LOSS'})</b>\n"
               f"Entry: ${trade_info['entry_price']:,.2f} → Exit: ${trade_info['exit_price']:,.2f}\n"
               f"PnL: ${pnl:+,.2f} ({trade_info['roi_pct']:+.2f}%)\n"
               f"Peak ROI: {trade_info.get('peak_roi', 0):+.2f}%\n"
               f"Balance: ${balance:,.2f}\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._enqueue(msg)

    def notify_tsl_activated(self, direction: int, entry_price: float, current_price: float, roi: float):
        dir_str = "LONG" if direction == 1 else "SHORT"
        msg = (f"<b>[ETH V8.16] TSL Activated ({dir_str})</b>\n"
               f"Entry: ${entry_price:,.2f} | Current: ${current_price:,.2f}\n"
               f"ROI: {roi:+.2f}% | Trail: -8% from peak\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._enqueue(msg)

    def notify_tsl_update(self, new_sl: float, track_extreme: float, direction: int):
        extreme_str = "High" if direction == 1 else "Low"
        msg = (f"<b>[ETH V8.16] TSL Updated</b>\n"
               f"Track {extreme_str}: ${track_extreme:,.2f}\n"
               f"New SL: ${new_sl:,.2f}\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._enqueue(msg)

    def notify_manual_position(self, direction: int, entry_price: float, size: float):
        dir_str = "LONG" if direction == 1 else "SHORT"
        msg = (f"<b>[ETH V8.16] Manual Position Detected</b>\n"
               f"{dir_str} @${entry_price:,.2f} | Size: ${size:,.0f}\n"
               f"Auto-management enabled\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._enqueue(msg)

    def notify_error(self, error_msg: str):
        msg = f"<b>[ETH V8.16] ERROR</b>\n{error_msg}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self._enqueue(msg)
