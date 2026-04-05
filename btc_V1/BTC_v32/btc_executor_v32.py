"""
BTC/USDT 선물 자동매매 - 주문 실행 및 포지션 관리
v32.2: EMA(100)/EMA(600) Tight-SL Trend System

- Binance Futures 주문 실행 (시장가, SL)
- 격리마진 / 레버리지 설정
- 포지션 조회 및 동기화
- SL → TSL 전환 관리
- 거래 통계 DB 저장
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import ccxt.async_support as ccxt
import aiosqlite

logger = logging.getLogger('btc_executor')

SYMBOL = 'BTC/USDT'
BINANCE_SYMBOL = 'BTCUSDT'
LEVERAGE = 10
MARGIN_MODE = 'isolated'
FEE_RATE = 0.0004
DB_PATH = 'btc_trading_bot.db'
TRADE_LOG_PATH = 'logs/btc_trades.txt'


class OrderExecutor:
    """Binance Futures 주문 실행 및 포지션 관리"""

    def __init__(self, api_key: str, api_secret: str):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
        self.db: aiosqlite.Connection = None

        # 잔액 추적
        self.balance: float = 0.0
        self.available_balance: float = 0.0

        # 현재 거래소 포지션
        self.exchange_position: dict = None

        # SL 주문 ID 추적
        self.sl_order_id: str = None

        # 헤지모드 여부
        self.hedge_mode: bool = False

    # ─── 초기화 ───
    async def initialize(self):
        """레버리지, 마진모드 설정 및 DB 초기화"""
        try:
            # 레버리지 설정
            await self.exchange.set_leverage(LEVERAGE, BINANCE_SYMBOL)
            logger.info(f"레버리지 설정: {LEVERAGE}x")
        except Exception as e:
            logger.warning(f"레버리지 설정 (이미 설정됨 가능): {e}")

        try:
            # 격리마진 설정
            await self.exchange.set_margin_mode(MARGIN_MODE, BINANCE_SYMBOL)
            logger.info(f"마진모드 설정: {MARGIN_MODE}")
        except Exception as e:
            logger.warning(f"마진모드 설정 (이미 설정됨 가능): {e}")

        # 헤지모드 감지
        try:
            resp = await self.exchange.fapiPrivateGetPositionSideDual()
            self.hedge_mode = resp.get('dualSidePosition', False)
            logger.info(f"포지션 모드: {'Hedge' if self.hedge_mode else 'One-Way'}")
        except Exception as e:
            logger.warning(f"포지션 모드 확인 실패: {e}")

        # 잔액 조회
        await self.update_balance()

        # DB 초기화
        await self._init_db()

        logger.info(f"Executor 초기화 완료: 잔액=${self.balance:,.2f}")

    async def _init_db(self):
        """SQLite DB 초기화"""
        self.db = await aiosqlite.connect(DB_PATH)
        await self.db.execute('''
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                source TEXT,
                direction TEXT,
                entry_price REAL,
                position_size REAL,
                margin REAL,
                leverage INTEGER,
                sl_price REAL,
                balance_before REAL,
                balance_after REAL,
                adx REAL,
                rsi REAL,
                ema_gap REAL,
                ema100 REAL,
                ema600 REAL
            )
        ''')
        await self.db.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                source TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                position_size REAL,
                pnl REAL,
                exit_type TEXT,
                roi_pct REAL,
                peak_roi REAL,
                hold_time REAL,
                balance_after REAL,
                tsl_active INTEGER
            )
        ''')
        await self.db.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                start_balance REAL,
                end_balance REAL,
                trades INTEGER,
                pnl REAL,
                wins INTEGER,
                losses INTEGER
            )
        ''')
        await self.db.commit()

        # 기존 DB 마이그레이션: source 컬럼 추가
        for table in ('entries', 'trades'):
            try:
                await self.db.execute(f"ALTER TABLE {table} ADD COLUMN source TEXT DEFAULT 'BOT'")
                await self.db.commit()
                logger.info(f"{table} 테이블에 source 컬럼 추가")
            except Exception:
                pass  # 이미 존재하면 무시

        # TXT 로그 디렉토리 생성
        Path(TRADE_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

    # ─── 잔액 ───
    async def update_balance(self) -> float:
        """잔액 조회 및 업데이트"""
        try:
            balance = await self.exchange.fetch_balance()
            usdt = balance.get('USDT', {})
            self.balance = float(usdt.get('total', 0))
            self.available_balance = float(usdt.get('free', 0))
            return self.balance
        except Exception as e:
            logger.error(f"잔액 조회 오류: {e}")
            return self.balance

    # ─── 포지션 조회 ───
    async def get_exchange_position(self) -> dict:
        """거래소의 현재 BTCUSDT 포지션 조회"""
        try:
            positions = await self.exchange.fetch_positions([BINANCE_SYMBOL])
            for p in positions:
                # ccxt 선물 심볼: 'BTC/USDT:USDT' 또는 'BTC/USDT'
                sym = p.get('symbol', '')
                if BINANCE_SYMBOL not in sym.replace('/', '').replace(':USDT', '').replace(':usdt', ''):
                    continue
                contracts = abs(float(p.get('contracts', 0) or 0))
                if contracts <= 0:
                    continue

                self.exchange_position = {
                    'direction': 1 if p['side'] == 'long' else -1,
                    'entry_price': float(p.get('entryPrice', 0) or 0),
                    'size': contracts,
                    'notional': abs(float(p.get('notional', 0) or 0)),
                    'unrealized_pnl': float(p.get('unrealizedPnl', 0) or 0),
                    'margin': float(p.get('initialMargin', 0) or 0),
                    'liquidation_price': float(p.get('liquidationPrice', 0) or 0),
                }
                logger.debug(f"포지션 감지: {p['side']} {contracts} BTC @{self.exchange_position['entry_price']:.2f}")
                return self.exchange_position

            self.exchange_position = None
            return None
        except Exception as e:
            logger.error(f"포지션 조회 오류: {e}")
            return self.exchange_position

    # ─── 주문 실행 ───
    async def market_entry(self, direction: int, position_size_usd: float,
                           sl_price: float) -> dict:
        """
        시장가 진입 + SL 주문 설정
        direction: 1=LONG, -1=SHORT
        position_size_usd: 포지션 금액 (USDT)
        sl_price: 손절가
        Returns: {order_id, filled_price, filled_qty, sl_order_id}
        """
        side = 'buy' if direction == 1 else 'sell'
        dir_str = "LONG" if direction == 1 else "SHORT"

        try:
            # 현재가 조회
            ticker = await self.exchange.fetch_ticker(SYMBOL)
            current_price = ticker['last']

            # BTC 수량 계산
            qty = position_size_usd / current_price
            # Binance BTC 최소 단위: 0.001
            qty = round(qty, 3)
            if qty < 0.001:
                qty = 0.001

            logger.info(f"시장가 {dir_str} 주문: {qty} BTC (${position_size_usd:,.0f})")

            # 시장가 주문
            params = {}
            if self.hedge_mode:
                params['positionSide'] = 'LONG' if direction == 1 else 'SHORT'
            order = await self.exchange.create_order(
                symbol=SYMBOL,
                type='market',
                side=side,
                amount=qty,
                params=params,
            )

            filled_price = float(order.get('average', current_price))
            filled_qty = float(order.get('filled', qty))

            logger.info(f"주문 체결: {dir_str} {filled_qty} BTC @{filled_price:.2f}")

            # SL 주문 설정
            sl_order_id = await self._set_stop_loss(direction, filled_qty, sl_price)

            await self.update_balance()

            return {
                'order_id': order['id'],
                'filled_price': filled_price,
                'filled_qty': filled_qty,
                'sl_order_id': sl_order_id,
                'position_size_usd': filled_qty * filled_price,
            }

        except Exception as e:
            logger.error(f"진입 주문 오류: {e}")
            raise

    async def market_exit(self, direction: int, qty: float = None) -> dict:
        """
        시장가 청산
        direction: 현재 포지션 방향 (1=LONG → sell, -1=SHORT → buy)
        """
        # 기존 SL 주문 취소
        await self._cancel_sl_order()

        side = 'sell' if direction == 1 else 'buy'

        try:
            if qty is None:
                # 전체 청산 — 포지션에서 수량 조회
                pos = await self.get_exchange_position()
                if pos:
                    qty = pos['size']
                else:
                    logger.warning("청산할 포지션 없음")
                    return None

            params = {'reduceOnly': True}
            if self.hedge_mode:
                params = {'positionSide': 'LONG' if direction == 1 else 'SHORT'}
            order = await self.exchange.create_order(
                symbol=SYMBOL,
                type='market',
                side=side,
                amount=qty,
                params=params,
            )

            filled_price = float(order.get('average', 0))
            logger.info(f"청산 체결: {qty} BTC @{filled_price:.2f}")

            await self.update_balance()

            return {
                'order_id': order['id'],
                'filled_price': filled_price,
                'filled_qty': float(order.get('filled', qty)),
            }

        except Exception as e:
            logger.error(f"청산 주문 오류: {e}")
            raise

    # ─── SL/TSL 관리 ───
    async def _set_stop_loss(self, direction: int, qty: float,
                             sl_price: float) -> str:
        """SL 주문 설정 — 거래소 실패 시 봇 자체 모니터링으로 대체"""
        side = 'sell' if direction == 1 else 'buy'
        sl_price_rounded = round(sl_price, 2)

        params = {'reduceOnly': True}
        if self.hedge_mode:
            params = {'positionSide': 'LONG' if direction == 1 else 'SHORT'}

        # 거래소 SL 주문 시도
        methods = [
            ('create_stop_loss_order', lambda: self.exchange.create_stop_loss_order(
                SYMBOL, 'market', side, qty, stopLossPrice=sl_price_rounded, params=params)),
            ('create_trigger_order', lambda: self.exchange.create_trigger_order(
                SYMBOL, 'market', side, qty, triggerPrice=sl_price_rounded, params=params)),
        ]

        for method_name, method_fn in methods:
            try:
                order = await method_fn()
                self.sl_order_id = order['id']
                logger.info(f"SL 거래소 설정 ({method_name}): {sl_price_rounded:.2f}")
                return self.sl_order_id
            except Exception:
                continue

        # 거래소 SL 실패 → 봇 자체 모니터링 (core._check_exit에서 처리)
        logger.info(f"SL 봇 자체 관리: {sl_price_rounded:.2f} (거래소 SL 미지원)")
        self.sl_order_id = None
        return 'BOT_MANAGED'

    async def _cancel_sl_order(self):
        """기존 SL 주문 취소"""
        if self.sl_order_id and self.sl_order_id != 'BOT_MANAGED':
            try:
                await self.exchange.cancel_order(self.sl_order_id, SYMBOL)
                logger.info(f"SL 주문 취소: {self.sl_order_id}")
            except Exception as e:
                logger.warning(f"SL 취소 오류 (이미 체결/취소됨 가능): {e}")
        self.sl_order_id = None

    async def update_stop_loss(self, direction: int, qty: float,
                               new_sl_price: float) -> str:
        """
        SL가 업데이트 (TSL 활성 시 SL → TSL 전환).
        기존 SL 취소 후 새 SL 설정.
        """
        await self._cancel_sl_order()
        return await self._set_stop_loss(direction, qty, new_sl_price)

    async def cancel_all_orders(self):
        """BTCUSDT 모든 미체결 주문 취소"""
        try:
            orders = await self.exchange.fetch_open_orders(SYMBOL)
            for o in orders:
                await self.exchange.cancel_order(o['id'], SYMBOL)
                logger.info(f"주문 취소: {o['id']} ({o['type']})")
            self.sl_order_id = None
        except Exception as e:
            logger.error(f"주문 일괄 취소 오류: {e}")

    # ─── 최근 거래 내역 조회 (실제 청산가 확인용) ───
    async def get_recent_trades(self, limit: int = 10) -> list:
        """최근 체결 내역 조회 (실제 청산가 확인)"""
        try:
            trades = await self.exchange.fetch_my_trades(SYMBOL, limit=limit)
            return trades
        except Exception as e:
            logger.error(f"체결 내역 조회 오류: {e}")
            return []

    async def get_last_exit_price(self) -> float:
        """최근 청산(체결) 가격 조회"""
        trades = await self.get_recent_trades(limit=5)
        if trades:
            # 가장 최근 체결가
            return float(trades[-1].get('price', 0))
        return 0.0

    # ─── 거래 기록 ───
    async def save_entry(self, entry_info: dict):
        """진입 기록 — DB + TXT 저장"""
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dir_str = 'LONG' if entry_info['direction'] == 1 else 'SHORT'
        source = entry_info.get('source', 'BOT')  # 'BOT' or 'USER'

        # DB 저장
        if self.db:
            try:
                await self.db.execute('''
                    INSERT INTO entries (timestamp, source, direction, entry_price,
                        position_size, margin, leverage, sl_price,
                        balance_before, balance_after,
                        adx, rsi, ema_gap, ema100, ema600)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    now_str,
                    source,
                    dir_str,
                    entry_info['entry_price'],
                    entry_info['position_size'],
                    entry_info['margin'],
                    LEVERAGE,
                    entry_info['sl_price'],
                    entry_info.get('balance_before', 0),
                    entry_info.get('balance_after', 0),
                    entry_info.get('adx', 0),
                    entry_info.get('rsi', 0),
                    entry_info.get('ema_gap', 0),
                    entry_info.get('ema100', 0),
                    entry_info.get('ema600', 0),
                ))
                await self.db.commit()
            except Exception as e:
                logger.error(f"진입 기록 DB 저장 오류: {e}")

        # TXT 저장
        try:
            source_tag = "[BOT ENTRY]" if source == 'BOT' else "[USER ENTRY]"
            with open(TRADE_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"{source_tag} {now_str}\n")
                f.write(f"  Source       : {source}\n")
                f.write(f"  Direction    : {dir_str}\n")
                f.write(f"  Entry Price  : ${entry_info['entry_price']:,.2f}\n")
                f.write(f"  Position Size: ${entry_info['position_size']:,.0f}\n")
                f.write(f"  Margin       : ${entry_info['margin']:,.0f}\n")
                f.write(f"  Leverage     : {LEVERAGE}x\n")
                f.write(f"  SL Price     : ${entry_info['sl_price']:,.2f}\n")
                f.write(f"  Balance      : ${entry_info.get('balance_after', 0):,.2f}\n")
                f.write(f"  EMA(100)     : ${entry_info.get('ema100', 0):,.2f}\n")
                f.write(f"  EMA(600)     : ${entry_info.get('ema600', 0):,.2f}\n")
                f.write(f"  EMA Gap      : {entry_info.get('ema_gap', 0):.3f}%\n")
                f.write(f"  ADX(20)      : {entry_info.get('adx', 0):.1f}\n")
                f.write(f"  RSI(10)      : {entry_info.get('rsi', 0):.1f}\n")
        except Exception as e:
            logger.error(f"진입 기록 TXT 저장 오류: {e}")

        logger.info(f"[{source}] 진입 기록: {dir_str} @${entry_info['entry_price']:,.2f}")

    async def save_trade(self, trade_info: dict):
        """청산 기록 — DB + TXT 저장"""
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dir_str = 'LONG' if trade_info['direction'] == 1 else 'SHORT'
        source = trade_info.get('source', 'BOT')  # 'BOT' or 'USER'

        # DB 저장
        if self.db:
            try:
                await self.db.execute('''
                    INSERT INTO trades (timestamp, source, direction, entry_price, exit_price,
                        position_size, pnl, exit_type, roi_pct, peak_roi, hold_time,
                        balance_after, tsl_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    now_str,
                    source,
                    dir_str,
                    trade_info['entry_price'],
                    trade_info['exit_price'],
                    trade_info['position_size'],
                    trade_info['pnl'],
                    trade_info['exit_type'],
                    trade_info['roi_pct'],
                    trade_info.get('peak_roi', 0),
                    trade_info.get('hold_time', 0),
                    self.balance,
                    1 if trade_info.get('tsl_active') else 0,
                ))
                await self.db.commit()
            except Exception as e:
                logger.error(f"청산 기록 DB 저장 오류: {e}")

        # TXT 저장
        try:
            pnl = trade_info['pnl']
            result_str = "PROFIT" if pnl > 0 else "LOSS"
            hold_secs = trade_info.get('hold_time', 0)
            hours = int(hold_secs // 3600)
            mins = int((hold_secs % 3600) // 60)

            source_tag = "[BOT EXIT]" if source == 'BOT' else "[USER EXIT]"
            with open(TRADE_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(f"\n{source_tag}  {now_str}  ({result_str})\n")
                f.write(f"  Source       : {source}\n")
                f.write(f"  Direction    : {dir_str}\n")
                f.write(f"  Exit Type    : {trade_info['exit_type']}\n")
                f.write(f"  Entry Price  : ${trade_info['entry_price']:,.2f}\n")
                f.write(f"  Exit Price   : ${trade_info['exit_price']:,.2f}\n")
                f.write(f"  Position Size: ${trade_info['position_size']:,.0f}\n")
                f.write(f"  PnL          : ${pnl:+,.2f}\n")
                f.write(f"  ROI          : {trade_info['roi_pct']:+.2f}%\n")
                f.write(f"  Peak ROI     : {trade_info.get('peak_roi', 0):+.2f}%\n")
                f.write(f"  Hold Time    : {hours}h {mins}m\n")
                f.write(f"  TSL Active   : {'YES' if trade_info.get('tsl_active') else 'NO'}\n")
                f.write(f"  Balance After: ${self.balance:,.2f}\n")
                f.write(f"{'='*60}\n")
        except Exception as e:
            logger.error(f"청산 기록 TXT 저장 오류: {e}")

        logger.info(f"[{source}] 청산 기록: {dir_str} {trade_info['exit_type']} PnL=${pnl:+,.2f}")

    async def save_daily_stats(self, date: str, start_bal: float,
                               end_bal: float, trades: int, pnl: float,
                               wins: int, losses: int):
        """일일 통계 저장"""
        if not self.db:
            return
        try:
            await self.db.execute('''
                INSERT OR REPLACE INTO daily_stats
                (date, start_balance, end_balance, trades, pnl, wins, losses)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (date, start_bal, end_bal, trades, pnl, wins, losses))
            await self.db.commit()
        except Exception as e:
            logger.error(f"일일 통계 저장 오류: {e}")

    # ─── 정리 ───
    async def close(self):
        """리소스 정리"""
        if self.db:
            await self.db.close()
        await self.exchange.close()
        logger.info("Executor 종료")
