"""
SOL/USDT 선물 텔레그램 알림
V1: V12:Mass 75:25 + Skip2@4loss + 12.5% Compound
"""

import asyncio
import logging
import aiohttp
from datetime import datetime
from typing import Optional

logger = logging.getLogger('sol_telegram')


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.session: Optional[aiohttp.ClientSession] = None
        self.enabled = bool(token and chat_id)
        self._command_handler = None
        self._poll_task: Optional[asyncio.Task] = None
        self._last_update_id = 0

    async def start(self):
        if not self.enabled:
            logger.warning("Telegram 비활성화 (token/chat_id 없음)")
            return
        self.session = aiohttp.ClientSession()
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("Telegram 알림 활성")

    async def stop(self):
        if self._poll_task:
            self._poll_task.cancel()
            try: await self._poll_task
            except asyncio.CancelledError: pass
        if self.session:
            await self.session.close()

    def set_command_handler(self, handler):
        self._command_handler = handler

    async def _send_raw(self, text: str, parse_mode: str = 'HTML'):
        if not self.enabled or not self.session: return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            async with self.session.post(url, json={
                'chat_id': self.chat_id, 'text': text, 'parse_mode': parse_mode,
                'disable_web_page_preview': True,
            }, timeout=10) as r:
                if r.status != 200:
                    logger.warning(f"Telegram send failed: {r.status}")
        except Exception as e:
            logger.warning(f"Telegram 전송 오류: {e}")

    def send(self, text: str):
        """동기 호출용 (asyncio.create_task로 비동기 실행)"""
        if not self.enabled: return
        try:
            asyncio.create_task(self._send_raw(text))
        except Exception as e:
            logger.warning(f"Telegram send: {e}")

    async def _poll_loop(self):
        """Telegram 명령 polling"""
        while True:
            try:
                await asyncio.sleep(3)
                url = f"https://api.telegram.org/bot{self.token}/getUpdates"
                params = {'timeout': 25, 'offset': self._last_update_id + 1}
                async with self.session.get(url, params=params, timeout=30) as r:
                    data = await r.json()
                    for upd in data.get('result', []):
                        self._last_update_id = upd['update_id']
                        msg = upd.get('message', {})
                        if str(msg.get('chat', {}).get('id')) != str(self.chat_id):
                            continue
                        text = msg.get('text', '').strip()
                        if text.startswith('/') and self._command_handler:
                            asyncio.create_task(self._command_handler(text))
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(5)

    # ═══════════════════════════════════════════════════════════
    # 알림 이벤트
    # ═══════════════════════════════════════════════════════════
    async def notify_start(self, balance: float):
        text = (
            f"🚀 <b>SOL Bot V1 시작</b>\n\n"
            f"전략: V12:Mass 75:25 Mutex + Skip2@4loss\n"
            f"Compound: 12.5% of balance\n"
            f"Leverage: 10x | Isolated\n\n"
            f"💰 잔액: <code>${balance:,.2f}</code>\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        await self._send_raw(text)

    def notify_entry(self, info: dict):
        mode = 'V12' if info.get('entry_mode') == 1 else 'MASS'
        dir_emoji = '🟢 LONG' if info['direction'] == 1 else '🔴 SHORT'
        text = (
            f"📥 <b>ENTRY [{mode}]</b>\n"
            f"{dir_emoji}\n\n"
            f"Price: <code>${info['entry_price']:.3f}</code>\n"
            f"Size: <code>${info['position_size']:,.0f}</code>\n"
            f"Margin: <code>${info['margin']:,.0f}</code>\n"
            f"SL: <code>${info['sl_price']:.3f}</code>\n"
        )
        if info.get('conf_score'):
            text += f"Conf: {info['conf_score']:.2f} ×{info['conf_mult']:.2f} | MS ×{info['margin_mult']:.2f}\n"
        text += f"\n💰 Balance: ${info.get('balance', 0):,.2f}"
        self.send(text)

    def notify_exit(self, info: dict):
        mode = 'V12' if info.get('entry_mode') == 1 else 'MASS'
        dir_str = 'LONG' if info['direction'] == 1 else 'SHORT'
        pnl = info['pnl']
        emoji = '✅ WIN' if pnl > 0 else '❌ LOSS'
        text = (
            f"📤 <b>EXIT [{mode}] {emoji}</b>\n"
            f"{info['exit_type']} {dir_str}\n\n"
            f"${info['entry_price']:.3f} → ${info['exit_price']:.3f}\n"
            f"PnL: <code>${pnl:+,.2f}</code> ({info['roi_pct']:+.2f}%)\n"
            f"Peak ROI: {info.get('peak_roi', 0):+.2f}%\n\n"
            f"💰 Balance: ${info.get('balance', 0):,.2f}"
        )
        self.send(text)

    def notify_pyramid(self, leg: int, price: float, margin: float):
        text = f"⤴ <b>Pyramid Leg{leg}</b> @ ${price:.3f} | +Margin ${margin:,.0f}"
        self.send(text)

    def notify_skip_trigger(self, consec: int):
        text = (
            f"⚠ <b>Skip2@4loss TRIGGERED!</b>\n"
            f"{consec}연패 감지 → 다음 2거래 스킵"
        )
        self.send(text)

    def notify_error(self, msg: str):
        text = f"❗ <b>ERROR</b>\n<code>{msg[:500]}</code>"
        self.send(text)

    def notify_status(self, stats: dict):
        text = (
            f"📊 <b>SOL Bot 상태</b>\n\n"
            f"💰 Balance: ${stats['balance']:,.2f}\n"
            f"📈 Peak: ${stats['peak']:,.2f}\n"
            f"📉 MDD: {stats['mdd']*100:.1f}%\n\n"
            f"🎯 Trades: {stats['total_trades']}\n"
            f"✅ Win: {stats['wins']} ({stats['wr']:.1f}%)\n"
            f"❌ Loss: {stats['losses']}\n"
            f"PF: {stats['pf']:.2f}\n"
            f"SL: {stats['sl']} / TSL: {stats['tsl']} / REV: {stats['rev']}\n\n"
            f"🔁 Consec losses: {stats['consec']}/4\n"
            f"⏭ Skip remaining: {stats['skip']}"
        )
        self.send(text)

    def notify_manual_position(self, direction: int, entry_price: float, notional: float):
        dir_emoji = '🟢 LONG' if direction == 1 else '🔴 SHORT'
        text = (
            f"⚠ <b>수동 포지션 감지</b>\n\n"
            f"{dir_emoji}\n"
            f"Entry: ${entry_price:.3f}\n"
            f"Size: ${notional:,.0f}\n\n"
            f"봇이 트래킹을 시작합니다."
        )
        self.send(text)
