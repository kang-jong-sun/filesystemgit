"""
ETH/USDT 선물 자동매매 - 텔레그램 알림 + 명령어
V8: EMA(250)/EMA(1575) 10m Trend System
V16 Balanced Sizing: 전략A에 Confidence Score 기반 Tier 사이징 (FULL 1.5x / HALF 1.0x / LOW 0.5x)
"""

import asyncio
import logging
from datetime import datetime, timedelta
import aiohttp

logger = logging.getLogger('eth_telegram')
TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
POLL_INTERVAL = 5  # 명령어 폴링 주기 (초)


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
        self._session: aiohttp.ClientSession = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task = None
        self._poll_task: asyncio.Task = None
        self._running = False
        self._last_update_id = 0
        self._command_handler = None  # 콜백: 명령어 처리
        if not self.enabled:
            logger.warning("텔레그램 알림 비활성화")

    def set_command_handler(self, handler):
        """명령어 콜백 등록: handler(command) -> str"""
        self._command_handler = handler

    async def start(self):
        if not self.enabled: return
        self._session = aiohttp.ClientSession()
        self._running = True
        self._task = asyncio.create_task(self._send_loop())
        self._poll_task = asyncio.create_task(self._poll_loop())

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
        if self._poll_task:
            self._poll_task.cancel()
            try: await self._poll_task
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
        url = TELEGRAM_API.format(token=self.bot_token, method='sendMessage')
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

    # ═══════════════════════════════════════════════════════════
    # 명령어 수신 (폴링)
    # ═══════════════════════════════════════════════════════════
    async def _poll_loop(self):
        """텔레그램 명령어 폴링"""
        # 시작 시 기존 메시지 무시 (최신 update_id 가져오기)
        await self._skip_old_updates()

        while self._running:
            try:
                await self._check_commands()
            except Exception as e:
                logger.error(f"명령어 폴링 오류: {e}")
            await asyncio.sleep(POLL_INTERVAL)

    async def _skip_old_updates(self):
        """봇 시작 전 쌓인 메시지 건너뛰기"""
        try:
            url = TELEGRAM_API.format(token=self.bot_token, method='getUpdates')
            params = {'offset': -1, 'limit': 1}
            async with self._session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get('result', [])
                    if results:
                        self._last_update_id = results[-1]['update_id'] + 1
                        logger.info(f"기존 메시지 스킵 (update_id: {self._last_update_id})")
        except Exception as e:
            logger.warning(f"기존 메시지 스킵 실패: {e}")

    async def _check_commands(self):
        """새 메시지 확인 및 명령어 처리"""
        url = TELEGRAM_API.format(token=self.bot_token, method='getUpdates')
        params = {'offset': self._last_update_id, 'limit': 10, 'timeout': 0}
        try:
            async with self._session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                for update in data.get('result', []):
                    self._last_update_id = update['update_id'] + 1
                    msg = update.get('message', {})
                    chat_id = str(msg.get('chat', {}).get('id', ''))
                    text = msg.get('text', '').strip()

                    # 허가된 chat_id만 처리
                    if chat_id != self.chat_id:
                        continue
                    if not text.startswith('/'):
                        continue

                    command = text.split()[0].lower().split('@')[0]  # /status@botname → /status
                    logger.info(f"명령어 수신: {command}")

                    if self._command_handler:
                        try:
                            response = self._command_handler(command)
                            if response:
                                await self._send(response)
                        except Exception as e:
                            logger.error(f"명령어 처리 오류: {e}")
                            await self._send(f"Error: {str(e)[:200]}")
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.error(f"getUpdates 오류: {e}")

    async def notify_start(self, balance: float):
        msg = (f"<b>[ETH V8 + V16 Balanced] System Started</b>\n"
               f"Strategy: EMA(250)/EMA(1575) 10m\n"
               f"Balance: ${balance:,.2f}\n"
               f"Leverage: 10x | Margin 20% | SL 2%\n"
               f"A: SL2%/TA54%/TSL2.75% | B: SL2.5%/TA5%/TSL0.3%\n"
               f"Sizing (A only): FULL 1.5x / HALF 1.0x / LOW 0.5x\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.enabled and self._session:
            await self._send(msg)

    async def notify_stop(self, balance: float, total_trades: int):
        msg = (f"<b>[ETH V8] System Stopped</b>\n"
               f"Balance: ${balance:,.2f} | Trades: {total_trades}\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.enabled and self._session:
            await self._send(msg)

    def notify_entry(self, direction: int, entry_price: float, position_size: float,
                     sl_price: float, balance: float, bar_info: dict = None,
                     entry_mode: str = 'A', score: float = 0.0,
                     tier: str = 'N/A', mult: float = 1.0):
        dir_str = "LONG" if direction == 1 else "SHORT"
        # 전략A: V16 Balanced 사이징 정보 표시 / B: 기본
        sizing_line = (f"Sizing: {tier} | Score: {score:.0f} | Mult: {mult:.1f}x\n"
                       if entry_mode == 'A' else "")
        msg = (f"<b>[ETH V8] {dir_str} Entry [{entry_mode}]</b>\n"
               f"{sizing_line}"
               f"Price: ${entry_price:,.2f} | Size: ${position_size:,.0f}\n"
               f"SL: ${sl_price:,.2f} (-2.0%)\n"
               f"Balance: ${balance:,.2f}\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._enqueue(msg)

    def notify_exit(self, trade_info: dict, balance: float):
        dir_str = "LONG" if trade_info['direction'] == 1 else "SHORT"
        pnl = trade_info['pnl']
        msg = (f"<b>[ETH V8] {dir_str} {trade_info['exit_type']} ({'PROFIT' if pnl > 0 else 'LOSS'})</b>\n"
               f"Entry: ${trade_info['entry_price']:,.2f} → Exit: ${trade_info['exit_price']:,.2f}\n"
               f"PnL: ${pnl:+,.2f} ({trade_info['roi_pct']:+.2f}%)\n"
               f"Peak ROI: {trade_info.get('peak_roi', 0):+.2f}%\n"
               f"Balance: ${balance:,.2f}\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._enqueue(msg)

    def notify_tsl_activated(self, direction: int, entry_price: float, current_price: float, roi: float):
        dir_str = "LONG" if direction == 1 else "SHORT"
        msg = (f"<b>[ETH V8] TSL Activated ({dir_str})</b>\n"
               f"Entry: ${entry_price:,.2f} | Current: ${current_price:,.2f}\n"
               f"ROI: {roi:+.2f}% | Trail: -8% from peak\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._enqueue(msg)

    def notify_tsl_update(self, new_sl: float, track_extreme: float, direction: int):
        extreme_str = "High" if direction == 1 else "Low"
        msg = (f"<b>[ETH V8] TSL Updated</b>\n"
               f"Track {extreme_str}: ${track_extreme:,.2f}\n"
               f"New SL: ${new_sl:,.2f}\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._enqueue(msg)

    def notify_manual_position(self, direction: int, entry_price: float, size: float):
        dir_str = "LONG" if direction == 1 else "SHORT"
        msg = (f"<b>[ETH V8] Manual Position Detected</b>\n"
               f"{dir_str} @${entry_price:,.2f} | Size: ${size:,.0f}\n"
               f"Auto-management enabled\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._enqueue(msg)

    def notify_status_report(self, price: float, balance: float,
                              ema_status: str, ema_gap: float,
                              position_str: str, status: dict,
                              uptime_str: str, total_transferred: float = 0):
        net = status['gross_profit'] - status['gross_loss']
        lines = [
            f"<b>[ETH V8] Hourly Report</b>",
            f"Uptime: {uptime_str}",
            f"",
            f"<b>Price:</b> ${price:,.2f}",
            f"<b>EMA:</b> {ema_status} (Gap: {ema_gap:.3f}%)",
            f"",
            f"<b>Position:</b> {position_str}",
            f"",
            f"<b>Account:</b> ${balance:,.2f}",
            f"<b>Trades:</b> {status['total_trades']} (W:{status['win_count']} L:{status['loss_count']} WR:{status['win_rate']:.0f}%)",
            f"<b>PF:</b> {status['profit_factor']:.2f} | Net: ${net:+,.2f}",
        ]
        if total_transferred > 0:
            lines.append(f"<b>Transferred:</b> ${total_transferred:,.2f}")
        lines.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._enqueue("\n".join(lines))

    def notify_transfer(self, amount: float, balance_before: float, balance_after: float, tran_id: str):
        msg = (f"<b>[ETH V8] Futures → Spot Transfer</b>\n"
               f"Amount: ${amount:,.2f}\n"
               f"Futures Before: ${balance_before:,.2f}\n"
               f"Futures After: ${balance_after:,.2f}\n"
               f"Tran ID: {tran_id}\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._enqueue(msg)

    def notify_transfer_failed(self, amount: float, error: str):
        msg = (f"<b>[ETH V8] Transfer FAILED</b>\n"
               f"Amount: ${amount:,.2f}\n"
               f"Error: {error[:200]}\n"
               f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._enqueue(msg)

    def notify_error(self, error_msg: str):
        msg = f"<b>[ETH V8] ERROR</b>\n{error_msg}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self._enqueue(msg)
