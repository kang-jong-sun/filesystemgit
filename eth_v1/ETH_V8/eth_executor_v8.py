"""
ETH/USDT 선물 자동매매 - 주문 실행 및 포지션 관리
V8.16: EMA(250)/EMA(1575) 10m Trend System

- Binance Futures 주문 실행 (시장가)
- 격리마진 / 레버리지 설정
- 포지션 조회 및 동기화
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

logger = logging.getLogger('eth_executor')

SYMBOL = 'ETH/USDT'
BINANCE_SYMBOL = 'ETHUSDT'
LEVERAGE = 10
MARGIN_MODE = 'isolated'
FEE_RATE = 0.0004
DB_PATH = 'eth_trading_bot.db'
TRADE_LOG_PATH = 'logs/eth_trades.txt'


class OrderExecutor:
    def __init__(self, api_key: str, api_secret: str):
        self.exchange = ccxt.binance({
            'apiKey': api_key, 'secret': api_secret,
            'options': {'defaultType': 'future'}, 'enableRateLimit': True,
        })
        self.db: aiosqlite.Connection = None
        self.balance: float = 0.0
        self.available_balance: float = 0.0
        self.exchange_position: dict = None
        self.sl_order_id: str = None
        self.hedge_mode: bool = False

    async def initialize(self):
        try:
            await self.exchange.set_leverage(LEVERAGE, BINANCE_SYMBOL)
            logger.info(f"레버리지 설정: {LEVERAGE}x")
        except Exception as e:
            logger.warning(f"레버리지 설정: {e}")

        try:
            await self.exchange.set_margin_mode(MARGIN_MODE, BINANCE_SYMBOL)
            logger.info(f"마진모드 설정: {MARGIN_MODE}")
        except Exception as e:
            logger.warning(f"마진모드 설정: {e}")

        try:
            resp = await self.exchange.fapiPrivateGetPositionSideDual()
            self.hedge_mode = resp.get('dualSidePosition', False)
            logger.info(f"포지션 모드: {'Hedge' if self.hedge_mode else 'One-Way'}")
        except Exception as e:
            logger.warning(f"포지션 모드 확인 실패: {e}")

        await self.update_balance()
        await self._init_db()
        logger.info(f"Executor 초기화 완료: 잔액=${self.balance:,.2f}")

    async def _init_db(self):
        self.db = await aiosqlite.connect(DB_PATH)
        await self.db.execute('''CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, source TEXT,
            direction TEXT, entry_price REAL, position_size REAL, margin REAL,
            leverage INTEGER, sl_price REAL, balance_before REAL, balance_after REAL,
            ema250 REAL, ema1575 REAL)''')
        await self.db.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, source TEXT,
            direction TEXT, entry_price REAL, exit_price REAL, position_size REAL,
            pnl REAL, exit_type TEXT, roi_pct REAL, peak_roi REAL,
            hold_time REAL, balance_after REAL, tsl_active INTEGER)''')
        await self.db.execute('''CREATE TABLE IF NOT EXISTS daily_stats (
            date TEXT PRIMARY KEY, start_balance REAL, end_balance REAL,
            trades INTEGER, pnl REAL, wins INTEGER, losses INTEGER)''')
        await self.db.commit()
        Path(TRADE_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

    async def update_balance(self) -> float:
        try:
            balance = await self.exchange.fetch_balance()
            usdt = balance.get('USDT', {})
            self.balance = float(usdt.get('total', 0))
            self.available_balance = float(usdt.get('free', 0))
            return self.balance
        except Exception as e:
            logger.error(f"잔액 조회 오류: {e}")
            return self.balance

    async def get_exchange_position(self) -> dict:
        try:
            positions = await self.exchange.fetch_positions([BINANCE_SYMBOL])
            for p in positions:
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
                return self.exchange_position
            self.exchange_position = None
            return None
        except Exception as e:
            logger.error(f"포지션 조회 오류: {e}")
            return self.exchange_position

    async def market_entry(self, direction: int, position_size_usd: float,
                           sl_price: float) -> dict:
        side = 'buy' if direction == 1 else 'sell'
        try:
            ticker = await self.exchange.fetch_ticker(SYMBOL)
            current_price = ticker['last']
            qty = position_size_usd / current_price
            qty = round(qty, 2)  # ETH 최소 단위 0.01
            if qty < 0.01: qty = 0.01

            params = {}
            if self.hedge_mode:
                params['positionSide'] = 'LONG' if direction == 1 else 'SHORT'
            order = await self.exchange.create_order(
                symbol=SYMBOL, type='market', side=side, amount=qty, params=params)

            filled_price = float(order.get('average', current_price))
            filled_qty = float(order.get('filled', qty))
            logger.info(f"주문 체결: {'LONG' if direction == 1 else 'SHORT'} {filled_qty} ETH @{filled_price:.2f}")

            sl_order_id = await self._set_stop_loss(direction, filled_qty, sl_price)
            await self.update_balance()

            return {'order_id': order['id'], 'filled_price': filled_price,
                    'filled_qty': filled_qty, 'sl_order_id': sl_order_id,
                    'position_size_usd': filled_qty * filled_price}
        except Exception as e:
            logger.error(f"진입 주문 오류: {e}")
            raise

    async def market_exit(self, direction: int, qty: float = None) -> dict:
        await self._cancel_sl_order()
        side = 'sell' if direction == 1 else 'buy'
        try:
            if qty is None:
                pos = await self.get_exchange_position()
                if pos: qty = pos['size']
                else: return None

            params = {'reduceOnly': True}
            if self.hedge_mode:
                params = {'positionSide': 'LONG' if direction == 1 else 'SHORT'}
            order = await self.exchange.create_order(
                symbol=SYMBOL, type='market', side=side, amount=qty, params=params)

            filled_price = float(order.get('average', 0))
            await self.update_balance()
            return {'order_id': order['id'], 'filled_price': filled_price,
                    'filled_qty': float(order.get('filled', qty))}
        except Exception as e:
            logger.error(f"청산 주문 오류: {e}")
            raise

    async def _set_stop_loss(self, direction: int, qty: float, sl_price: float) -> str:
        side = 'sell' if direction == 1 else 'buy'
        sl_price_rounded = round(sl_price, 2)
        params = {'reduceOnly': True}
        if self.hedge_mode:
            params = {'positionSide': 'LONG' if direction == 1 else 'SHORT'}

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
                return self.sl_order_id
            except Exception:
                continue

        self.sl_order_id = None
        return 'BOT_MANAGED'

    async def _cancel_sl_order(self):
        if self.sl_order_id and self.sl_order_id != 'BOT_MANAGED':
            try:
                await self.exchange.cancel_order(self.sl_order_id, SYMBOL)
            except Exception:
                pass
        self.sl_order_id = None

    async def update_stop_loss(self, direction: int, qty: float, new_sl_price: float) -> str:
        await self._cancel_sl_order()
        return await self._set_stop_loss(direction, qty, new_sl_price)

    async def get_last_exit_price(self) -> float:
        try:
            trades = await self.exchange.fetch_my_trades(SYMBOL, limit=5)
            if trades: return float(trades[-1].get('price', 0))
        except Exception:
            pass
        return 0.0

    async def save_entry(self, entry_info: dict):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dir_str = 'LONG' if entry_info['direction'] == 1 else 'SHORT'
        source = entry_info.get('source', 'BOT')
        if self.db:
            try:
                await self.db.execute('''INSERT INTO entries
                    (timestamp, source, direction, entry_price, position_size, margin,
                     leverage, sl_price, balance_before, balance_after, ema250, ema1575)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (now_str, source, dir_str, entry_info['entry_price'],
                     entry_info['position_size'], entry_info['margin'], LEVERAGE,
                     entry_info['sl_price'], entry_info.get('balance_before', 0),
                     entry_info.get('balance_after', 0),
                     entry_info.get('ema250', 0), entry_info.get('ema1575', 0)))
                await self.db.commit()
            except Exception as e:
                logger.error(f"진입 기록 DB 저장 오류: {e}")

        try:
            with open(TRADE_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n[{source} ENTRY] {now_str}\n")
                f.write(f"  Direction: {dir_str} | Price: ${entry_info['entry_price']:,.2f}\n")
                f.write(f"  Size: ${entry_info['position_size']:,.0f} | Margin: ${entry_info['margin']:,.0f}\n")
                f.write(f"  SL: ${entry_info['sl_price']:,.2f} | Balance: ${entry_info.get('balance_after', 0):,.2f}\n")
        except Exception as e:
            logger.error(f"TXT 저장 오류: {e}")

    async def save_trade(self, trade_info: dict):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dir_str = 'LONG' if trade_info['direction'] == 1 else 'SHORT'
        source = trade_info.get('source', 'BOT')
        if self.db:
            try:
                await self.db.execute('''INSERT INTO trades
                    (timestamp, source, direction, entry_price, exit_price, position_size,
                     pnl, exit_type, roi_pct, peak_roi, hold_time, balance_after, tsl_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (now_str, source, dir_str, trade_info['entry_price'],
                     trade_info['exit_price'], trade_info['position_size'],
                     trade_info['pnl'], trade_info['exit_type'],
                     trade_info['roi_pct'], trade_info.get('peak_roi', 0),
                     trade_info.get('hold_time', 0), self.balance,
                     1 if trade_info.get('tsl_active') else 0))
                await self.db.commit()
            except Exception as e:
                logger.error(f"청산 기록 DB 저장 오류: {e}")

        try:
            pnl = trade_info['pnl']
            with open(TRADE_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(f"\n[{source} EXIT] {now_str} ({'PROFIT' if pnl > 0 else 'LOSS'})\n")
                f.write(f"  {dir_str} {trade_info['exit_type']} | PnL: ${pnl:+,.2f} ({trade_info['roi_pct']:+.2f}%)\n")
                f.write(f"  Entry: ${trade_info['entry_price']:,.2f} → Exit: ${trade_info['exit_price']:,.2f}\n")
                f.write(f"  Balance: ${self.balance:,.2f}\n{'='*60}\n")
        except Exception as e:
            logger.error(f"TXT 저장 오류: {e}")

    async def close(self):
        if self.db: await self.db.close()
        await self.exchange.close()
