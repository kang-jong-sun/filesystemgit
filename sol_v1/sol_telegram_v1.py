"""
SOL/USDT 선물 텔레그램 알림
V1: V12:Mass 75:25 + Skip2@4loss + 12.5% Compound

ETH V8 패턴 적용 (2026-04-28):
- 메시지 큐 (_send_loop) — 전송 누락 방지
- _skip_old_updates — 봇 재시작 시 묵은 메시지 폭격 방지
- /sol 한글 명령어 (대소문자 / 공백 무관, sol_main_v1.py 핸들러)
"""

import asyncio
import logging
import aiohttp
from datetime import datetime
from typing import Optional

logger = logging.getLogger('sol_telegram')

POLL_INTERVAL = 5  # 명령어 폴링 주기 (초)


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.session: Optional[aiohttp.ClientSession] = None
        self.enabled = bool(token and chat_id)
        self._command_handler = None
        self._send_task: Optional[asyncio.Task] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._last_update_id = 0

    async def start(self):
        if not self.enabled:
            logger.warning("Telegram 비활성화 (token/chat_id 없음)")
            return
        self.session = aiohttp.ClientSession()
        self._running = True
        self._send_task = asyncio.create_task(self._send_loop())
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("Telegram 알림 활성 (큐 + 폴링)")

    async def stop(self):
        self._running = False
        # 큐에 남은 메시지 마무리 발송
        if self.session:
            while not self._queue.empty():
                try:
                    msg = self._queue.get_nowait()
                    await self._send_raw(msg)
                except Exception:
                    break
        if self._send_task:
            self._send_task.cancel()
            try: await self._send_task
            except asyncio.CancelledError: pass
        if self._poll_task:
            self._poll_task.cancel()
            try: await self._poll_task
            except asyncio.CancelledError: pass
        if self.session:
            await self.session.close()

    def set_command_handler(self, handler):
        self._command_handler = handler

    async def _send_loop(self):
        """큐에서 메시지 꺼내서 발송 (전송 실패 시 누락 방지)"""
        while self._running:
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._send_raw(msg)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning(f"_send_loop 오류: {e}")
                await asyncio.sleep(2)

    async def _send_raw(self, text: str, parse_mode: str = 'HTML', max_retries: int = 3):
        """Telegram 전송 (재시도 포함). 네트워크 일시 끊김은 재시도로 복구."""
        if not self.enabled or not self.session: return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        last_err = None
        for attempt in range(max_retries):
            try:
                async with self.session.post(url, json={
                    'chat_id': self.chat_id, 'text': text, 'parse_mode': parse_mode,
                    'disable_web_page_preview': True,
                }, timeout=10) as r:
                    if r.status == 200:
                        return  # 성공
                    # 400/401/403 등은 재시도해도 소용 없음
                    if r.status in (400, 401, 403, 404):
                        logger.warning(f"Telegram send failed (재시도 불가): HTTP {r.status}")
                        return
                    last_err = f"HTTP {r.status}"
            except (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
                last_err = e
                # 네트워크 에러는 재시도
            except Exception as e:
                # 기타 에러는 즉시 중단
                logger.warning(f"Telegram 전송 오류: {e}")
                return
            # 재시도 대기 (1초, 2초)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        # 모두 실패
        logger.warning(f"Telegram 전송 {max_retries}회 실패: {last_err}")

    def send(self, text: str):
        """동기 호출용 (큐에 enqueue, _send_loop가 처리)"""
        if not self.enabled: return
        try:
            self._queue.put_nowait(text)
        except Exception as e:
            logger.warning(f"Telegram send: {e}")

    async def _skip_old_updates(self):
        """★ 봇 시작 전 쌓인 메시지 건너뛰기 (재시작 시 묵은 명령어 폭격 방지)"""
        if not self.enabled or not self.session: return
        try:
            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            params = {'offset': -1, 'limit': 1}
            async with self.session.get(url, params=params, timeout=10) as r:
                if r.status == 200:
                    data = await r.json()
                    results = data.get('result', [])
                    if results:
                        self._last_update_id = results[-1]['update_id']
                        logger.info(f"기존 메시지 스킵 (last_update_id={self._last_update_id})")
        except Exception as e:
            logger.warning(f"기존 메시지 스킵 실패: {e}")

    async def _poll_loop(self):
        """Telegram 명령 polling — 시작 시 묵은 메시지 스킵 후 진입"""
        await self._skip_old_updates()
        while self._running:
            try:
                await asyncio.sleep(POLL_INTERVAL)
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
                            logger.info(f"명령어 수신: {text}")
                            asyncio.create_task(self._command_handler(text))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"poll_loop: {e}")
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
        # 포지션 섹션
        pos = stats.get('position')
        price = stats.get('price', 0)
        lev = stats.get('leverage', 0)
        lev_default = stats.get('leverage_default', lev)
        lev_str = f"{lev}x{' (사용자)' if lev != lev_default else ''}" if lev else ""
        if pos:
            roi_str = f"{pos['roi']:+.2f}%" if pos.get('has_price') else "N/A"
            pnl_str = f"${pos['pnl']:+,.2f}" if pos.get('has_price') else "N/A"
            pos_block = (
                f"📍 <b>Position:</b> {pos['direction_str']} [{pos['entry_mode_str']}]\n"
                f"  Entry ${pos['entry_price']:.3f} → Current ${price:.3f}\n"
                f"  ROI: {roi_str} | PnL: {pnl_str}\n"
                f"  Peak ROI: {pos['peak_roi']:+.2f}%\n"
                f"  SL ${pos['sl_price']:.3f} | TSL: {'ON' if pos['tsl_active'] else 'OFF'}\n"
                f"  📅 진입: {pos['entry_time_str']}\n"
                f"  ⏱ 보유: {pos['hold_str']}\n\n"
            )
        else:
            pos_block = "📭 <b>Position:</b> 없음 (신호 대기)\n\n"

        text = (
            f"📊 <b>SOL Bot 상태</b>\n"
            f"🎯 Price: ${price:.3f} | Lev {lev_str}\n\n"
            f"{pos_block}"
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
