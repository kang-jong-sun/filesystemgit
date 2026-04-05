"""
BTC/USDT 선물 자동매매 - 실시간 모니터링
v32.2: EMA(100)/EMA(600) Tight-SL Trend System

- Rich 콘솔 출력 (30초 간격)
- 현재가, EMA, ADX, RSI 표시
- 포지션 정보, ROI, SL/TSL 상태
- 거래 통계 (승률, PF, MDD)
- 감시 상태 표시
"""

import asyncio
import logging
import time
import os
from datetime import datetime, timedelta

logger = logging.getLogger('btc_monitor')

MONITOR_INTERVAL = 30  # 30초


class SystemMonitor:
    """실시간 모니터링 화면 출력"""

    def __init__(self):
        self._running = False
        self._task: asyncio.Task = None

        # 외부 참조
        self.data_collector = None
        self.trading_core = None
        self.executor = None
        self.start_time: float = 0.0

    def set_components(self, data_collector, trading_core, executor):
        """컴포넌트 참조 설정"""
        self.data_collector = data_collector
        self.trading_core = trading_core
        self.executor = executor

    async def start(self):
        """모니터링 시작"""
        self._running = True
        self.start_time = time.time()
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("모니터링 시작 (30초 간격)")

    async def stop(self):
        """모니터링 중지"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        """30초마다 상태 출력"""
        while self._running:
            try:
                # 최신 바이낸스 잔액 갱신 후 출력
                if self.executor:
                    await self.executor.update_balance()
                self._display()
            except Exception as e:
                logger.error(f"모니터링 출력 오류: {e}")
            await asyncio.sleep(MONITOR_INTERVAL)

    def _display(self):
        """콘솔에 상태 정보 출력 (30초 간격, 화면 클리어)"""
        dc = self.data_collector
        tc = self.trading_core
        ex = self.executor

        if not dc or not tc or not ex:
            return

        bar = dc.get_current_bar()
        if bar is None:
            return

        status = tc.get_status()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        uptime = str(timedelta(seconds=int(time.time() - self.start_time)))

        # 화면 클리어
        os.system('cls' if os.name == 'nt' else 'clear')

        price = dc.current_price if dc.current_price > 0 else bar['close']

        lines = []
        lines.append("=" * 70)
        lines.append("  BTC/USDT Futures v32.2 | EMA(100)/EMA(600) Tight-SL Trend System")
        lines.append("=" * 70)
        lines.append(f"  {now}  |  Uptime: {uptime}")
        lines.append("")

        # ─── 가격 및 지표 ───
        if bar['fast_ma'] > bar['slow_ma']:
            ema_status = "BULLISH (EMA100 > EMA600)"
        else:
            ema_status = "BEARISH (EMA100 < EMA600)"
        adx_trend = "RISING" if bar['adx'] > bar['adx_prev6'] else "FALLING"
        adx_ok = "OK" if bar['adx'] >= 30.0 else "LOW"

        lines.append("  [Price & Indicators]")
        lines.append(f"    BTC Price     : ${price:,.2f}")
        lines.append(f"    EMA(100)      : ${bar['fast_ma']:,.2f}")
        lines.append(f"    EMA(600)      : ${bar['slow_ma']:,.2f}")
        lines.append(f"    EMA Status    : {ema_status}")
        lines.append(f"    EMA Gap       : {bar['ema_gap']:.3f}%")
        lines.append(f"    ADX(20)       : {bar['adx']:.1f}  (6bar ago: {bar['adx_prev6']:.1f})  [{adx_ok}/{adx_trend}]")
        lines.append(f"    RSI(10)       : {bar['rsi']:.1f}")
        lines.append("")

        # ─── 바이낸스 잔액 ───
        lines.append("  [Binance Account]")
        lines.append(f"    Wallet Balance: ${ex.balance:,.2f}")
        lines.append(f"    Available     : ${ex.available_balance:,.2f}")
        if status['has_position']:
            pos = tc.position
            unrealized = (price - pos.entry_price) / pos.entry_price * pos.position_size * pos.direction
            total_equity = ex.balance + unrealized
            lines.append(f"    Unrealized PnL: ${unrealized:+,.2f}")
            lines.append(f"    Total Equity  : ${total_equity:,.2f}")
        lines.append(f"    Peak Capital  : ${tc.peak_capital:,.2f}")
        lines.append(f"    Max Drawdown  : {tc.max_drawdown * 100:.1f}%")
        lines.append("")

        # ─── 포지션 ───
        lines.append("  [Position]")
        if status['has_position']:
            pos = tc.position
            dir_str = status['direction_str']
            roi = (price - pos.entry_price) / pos.entry_price * pos.direction * 100
            unrealized_pnl = (price - pos.entry_price) / pos.entry_price * pos.position_size * pos.direction
            hold_str = str(timedelta(seconds=int(time.time() - pos.entry_time)))

            lines.append(f"    Direction     : {dir_str}")
            lines.append(f"    Entry Price   : ${pos.entry_price:,.2f}")
            lines.append(f"    Position Size : ${pos.position_size:,.0f}")
            lines.append(f"    Current ROI   : {roi:+.2f}%")
            lines.append(f"    Unrealized PnL: ${unrealized_pnl:+,.2f}")
            lines.append(f"    Peak ROI      : {pos.peak_roi:+.2f}%")
            lines.append(f"    Hold Time     : {hold_str}")
            lines.append(f"    SL Price      : ${pos.sl_price:,.2f}")
            lines.append(f"    TSL Active    : {'YES' if pos.tsl_active else 'NO'}")
            if pos.tsl_active:
                if pos.direction == 1:
                    lines.append(f"    Track High    : ${pos.track_high:,.2f}")
                else:
                    lines.append(f"    Track Low     : ${pos.track_low:,.2f}")
            # 강제청산가
            if ex.exchange_position and ex.exchange_position.get('liquidation_price', 0) > 0:
                liq = ex.exchange_position['liquidation_price']
                liq_dist = abs(price - liq) / price * 100
                lines.append(f"    Liq. Price    : ${liq:,.2f} ({liq_dist:.1f}% away)")
        else:
            lines.append(f"    Status        : NO POSITION")
            if status['watching'] != 0:
                lines.append(f"    Watching      : {status['watching_str']}")
            else:
                lines.append(f"    Watching      : IDLE")
        lines.append("")

        # ─── 거래 통계 ───
        lines.append("  [Trade Statistics]")
        lines.append(f"    Total Trades  : {status['total_trades']}")
        lines.append(f"    SL / TSL / REV: {status['sl_count']} / {status['tsl_count']} / {status['rev_count']}")
        lines.append(f"    Win / Loss    : {status['win_count']}W / {status['loss_count']}L")
        lines.append(f"    Win Rate      : {status['win_rate']:.1f}%")
        lines.append(f"    Profit Factor : {status['profit_factor']:.2f}")
        lines.append(f"    Gross Profit  : ${status['gross_profit']:,.2f}")
        lines.append(f"    Gross Loss    : ${status['gross_loss']:,.2f}")
        lines.append(f"    Net Profit    : ${status['gross_profit'] - status['gross_loss']:,.2f}")
        lines.append("")

        # ─── 최근 거래 ───
        if tc.trade_history:
            lines.append("  [Recent Trades]")
            for tr in tc.trade_history[-5:]:
                dir_s = "L" if tr.direction == 1 else "S"
                src = tr.source if hasattr(tr, 'source') else "BOT"
                lines.append(f"    {tr.exit_time} [{src:>4}] {dir_s} {tr.exit_type:>6} "
                             f"${tr.entry_price:,.0f}->${tr.exit_price:,.0f} "
                             f"PnL=${tr.pnl:+,.0f} ({tr.roi_pct:+.1f}%)")
            lines.append("")

        lines.append("=" * 70)
        lines.append("  Ctrl+C to stop")
        lines.append("=" * 70)

        print("\n".join(lines))

    def display_once(self):
        """즉시 1회 출력"""
        try:
            self._display()
        except Exception as e:
            logger.error(f"모니터링 출력 오류: {e}")
