"""
ETH/USDT 선물 자동매매 - 실시간 모니터링
V8.16: EMA(250)/EMA(1575) 10m Trend System
"""

import asyncio
import logging
import time
import os
from datetime import datetime, timedelta

logger = logging.getLogger('eth_monitor')
MONITOR_INTERVAL = 30


class SystemMonitor:
    def __init__(self):
        self._running = False
        self._task: asyncio.Task = None
        self.data_collector = None
        self.trading_core = None
        self.executor = None
        self.start_time: float = 0.0

    def set_components(self, data_collector, trading_core, executor):
        self.data_collector = data_collector
        self.trading_core = trading_core
        self.executor = executor

    async def start(self):
        self._running = True
        self.start_time = time.time()
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass

    async def _monitor_loop(self):
        while self._running:
            try:
                if self.executor:
                    await self.executor.update_balance()
                self._display()
            except Exception as e:
                logger.error(f"모니터링 오류: {e}")
            await asyncio.sleep(MONITOR_INTERVAL)

    def _display(self):
        dc = self.data_collector
        tc = self.trading_core
        ex = self.executor
        if not dc or not tc or not ex: return

        bar = dc.get_current_bar()
        if bar is None: return

        status = tc.get_status()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        uptime = str(timedelta(seconds=int(time.time() - self.start_time)))
        price = dc.current_price if dc.current_price > 0 else bar['close']

        os.system('cls' if os.name == 'nt' else 'clear')

        lines = []
        lines.append("=" * 70)
        lines.append("  ETH/USDT V8.16 Dual | A:EMA250/1575 + B:EMA9/100 | 20% | 10x")
        lines.append("=" * 70)
        lines.append(f"  {now}  |  Uptime: {uptime}")
        lines.append("")

        # 가격 & EMA
        ema_status = "BULLISH" if bar['fast_ma'] > bar['slow_ma'] else "BEARISH"
        ema_gap = abs(bar['fast_ma'] - bar['slow_ma']) / bar['slow_ma'] * 100 if bar['slow_ma'] > 0 else 0
        lines.append("  [Price & EMA]")
        lines.append(f"    ETH Price     : ${price:,.2f}")
        lines.append(f"    EMA(250)      : ${bar['fast_ma']:,.2f}")
        lines.append(f"    EMA(1575)     : ${bar['slow_ma']:,.2f}")
        lines.append(f"    EMA Status    : {ema_status} (Gap: {ema_gap:.3f}%)")
        # 전략B EMA
        b_ema9 = bar.get('b_ema9', 0)
        b_ema100 = bar.get('b_ema100_15m', 0)
        b_gap = abs(b_ema9 - b_ema100) / max(b_ema100, 1) * 100 if b_ema100 else 0
        b_status = "BULL" if b_ema9 > b_ema100 else "BEAR"
        lines.append(f"    [B] EMA(9)    : ${b_ema9:,.2f}")
        lines.append(f"    [B] EMA(100)  : ${b_ema100:,.2f}")
        lines.append(f"    [B] Status    : {b_status} (Gap: {b_gap:.3f}%)")
        lines.append("")

        # 잔액
        lines.append("  [Binance Account]")
        lines.append(f"    Wallet Balance: ${ex.balance:,.2f}")
        lines.append(f"    Available     : ${ex.available_balance:,.2f}")
        if status['has_position']:
            pos = tc.position
            unrealized = (price - pos.entry_price) / pos.entry_price * pos.position_size * pos.direction
            lines.append(f"    Unrealized PnL: ${unrealized:+,.2f}")
            lines.append(f"    Total Equity  : ${ex.balance + unrealized:,.2f}")
        lines.append(f"    Peak Capital  : ${tc.peak_capital:,.2f}")
        lines.append(f"    Max Drawdown  : {tc.max_drawdown * 100:.1f}%")
        lines.append("")

        # 포지션
        lines.append("  [Position]")
        if status['has_position']:
            pos = tc.position
            roi = (price - pos.entry_price) / pos.entry_price * pos.direction * 100
            pnl = (price - pos.entry_price) / pos.entry_price * pos.position_size * pos.direction
            hold_str = str(timedelta(seconds=int(time.time() - pos.entry_time)))
            mode_str = f" [{pos.entry_mode}]" if hasattr(pos, 'entry_mode') else ""
            lines.append(f"    Direction     : {status['direction_str']}{mode_str}")
            lines.append(f"    Entry Price   : ${pos.entry_price:,.2f}")
            lines.append(f"    Position Size : ${pos.position_size:,.0f} (Margin: ${pos.margin_used:,.0f})")
            lines.append(f"    Current ROI   : {roi:+.2f}%")
            lines.append(f"    Unrealized PnL: ${pnl:+,.2f}")
            lines.append(f"    Peak ROI      : {pos.peak_roi:+.2f}%")
            lines.append(f"    Hold Time     : {hold_str}")
            lines.append(f"    SL Price      : ${pos.sl_price:,.2f}")
            lines.append(f"    TSL Active    : {'YES' if pos.tsl_active else 'NO'}")
            if pos.tsl_active:
                extreme = pos.track_high if pos.direction == 1 else pos.track_low
                lines.append(f"    Track {'High' if pos.direction == 1 else 'Low'}     : ${extreme:,.2f}")
            if ex.exchange_position and ex.exchange_position.get('liquidation_price', 0) > 0:
                liq = ex.exchange_position['liquidation_price']
                lines.append(f"    Liq. Price    : ${liq:,.2f} ({abs(price - liq) / price * 100:.1f}% away)")
        else:
            lines.append(f"    Status        : NO POSITION")
            if status['watching'] != 0:
                lines.append(f"    Watching      : {status['watching_str']}")
        lines.append("")

        # 통계
        lines.append("  [Trade Statistics]")
        lines.append(f"    Total Trades  : {status['total_trades']}")
        lines.append(f"    SL / TSL / REV: {status['sl_count']} / {status['tsl_count']} / {status['rev_count']}")
        lines.append(f"    Win / Loss    : {status['win_count']}W / {status['loss_count']}L")
        lines.append(f"    Win Rate      : {status['win_rate']:.1f}%")
        lines.append(f"    Profit Factor : {status['profit_factor']:.2f}")
        lines.append(f"    Net Profit    : ${status['gross_profit'] - status['gross_loss']:,.2f}")
        lines.append("")

        # 최근 거래
        if tc.trade_history:
            lines.append("  [Recent Trades]")
            for tr in tc.trade_history[-5:]:
                dir_s = "L" if tr.direction == 1 else "S"
                lines.append(f"    {tr.exit_time} {dir_s} {tr.exit_type:>4} "
                             f"${tr.entry_price:,.0f}->${tr.exit_price:,.0f} "
                             f"PnL=${tr.pnl:+,.0f} ({tr.roi_pct:+.1f}%)")
            lines.append("")

        lines.append("=" * 70)
        lines.append("  Ctrl+C to stop")
        lines.append("=" * 70)
        print("\n".join(lines))
