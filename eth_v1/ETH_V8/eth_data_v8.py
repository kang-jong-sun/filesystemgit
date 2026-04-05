"""
ETH/USDT 선물 자동매매 - 데이터 수집 및 지표 계산
V8.16: EMA(250)/EMA(1575) 10m Trend System

- Binance Futures API 연동 (ccxt async)
- 10분봉 캔들 데이터 수집 및 유지
- EMA(250), EMA(1575) 계산 (ADX/RSI 불필요)
- WebSocket 실시간 가격 업데이트
"""

import asyncio
import logging
import time
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
import websockets
import json

logger = logging.getLogger('eth_data')

SYMBOL = 'ETH/USDT'
BINANCE_SYMBOL = 'ETHUSDT'
TIMEFRAME_API = '5m'        # 바이낸스 API: 5분봉 수집
TIMEFRAME_RESAMPLE = '10min' # 내부: 10분봉으로 리샘플링
FAST_MA_PERIOD = 250
SLOW_MA_PERIOD = 1575
CANDLE_LIMIT = 4000  # 5분봉 4000개 → 10분봉 2000개 (EMA1575 워밍업)


class DataCollector:
    """Binance Futures 데이터 수집 및 지표 계산"""

    def __init__(self, api_key: str, api_secret: str, ws_url: str = None):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
        self.ws_url = ws_url or 'wss://fstream.binance.com/ws'

        self.df: pd.DataFrame = None
        self.last_candle_time: int = 0

        self.close: np.ndarray = None
        self.high: np.ndarray = None
        self.low: np.ndarray = None
        self.fast_ma: np.ndarray = None
        self.slow_ma: np.ndarray = None

        self.current_price: float = 0.0
        self.current_high: float = 0.0
        self.current_low: float = 0.0
        self.price_update_time: float = 0.0

        self._ws_task: asyncio.Task = None
        self._ws_running = False
        self._ws = None

    async def initialize(self):
        logger.info("초기 캔들 데이터 로딩...")
        await self._load_historical_candles()
        self._calculate_indicators()
        logger.info(f"데이터 초기화 완료: {len(self.df)}봉, "
                    f"EMA250={self.fast_ma[-1]:.2f}, EMA1575={self.slow_ma[-1]:.2f}")

    async def _load_historical_candles(self):
        """5분봉 로드 → 10분봉 리샘플링"""
        all_candles = []
        since = None
        total_needed = CANDLE_LIMIT

        while len(all_candles) < total_needed:
            limit = min(1500, total_needed - len(all_candles))
            candles = await self.exchange.fetch_ohlcv(
                SYMBOL, TIMEFRAME_API, since=since, limit=limit
            )
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            if len(candles) < limit:
                break
            await asyncio.sleep(0.1)

        df5 = pd.DataFrame(all_candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])
        df5['timestamp'] = pd.to_datetime(df5['timestamp'], unit='ms', utc=True)
        df5.set_index('timestamp', inplace=True)
        df5 = df5.astype(float)
        df5 = df5[~df5.index.duplicated(keep='last')]
        df5.sort_index(inplace=True)

        logger.info(f"  5분봉 로드: {len(df5)}봉")

        # 10분봉 리샘플링
        self.df = df5.resample(TIMEFRAME_RESAMPLE).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()

        self.last_candle_time = int(self.df.index[-1].timestamp() * 1000)
        self.current_price = float(self.df['close'].iloc[-1])
        logger.info(f"  10분봉 리샘플: {len(self.df)}봉 "
                    f"({self.df.index[0]} ~ {self.df.index[-1]})")

        # 5분봉 원본 보관 (업데이트용)
        self._df5 = df5

    def _calculate_indicators(self):
        close_s = self.df['close']
        self.close = close_s.values.astype(np.float64)
        self.high = self.df['high'].values.astype(np.float64)
        self.low = self.df['low'].values.astype(np.float64)

        # EMA만 계산 (ADX/RSI 불필요 — 필터 OFF)
        self.fast_ma = close_s.ewm(span=FAST_MA_PERIOD, adjust=False).mean().values.astype(np.float64)
        self.slow_ma = close_s.ewm(span=SLOW_MA_PERIOD, adjust=False).mean().values.astype(np.float64)

    async def update_candles(self):
        """5분봉 최신 가져와서 10분봉 재생성"""
        try:
            candles = await self.exchange.fetch_ohlcv(SYMBOL, TIMEFRAME_API, limit=10)
            if not candles:
                return False

            new_count = 0
            for c in candles:
                ts = pd.Timestamp(c[0], unit='ms', tz='UTC')
                if ts in self._df5.index:
                    self._df5.loc[ts] = [c[1], c[2], c[3], c[4], c[5]]
                else:
                    self._df5.loc[ts] = [c[1], c[2], c[3], c[4], c[5]]
                    new_count += 1

            # 10분봉 리샘플링
            self._df5.sort_index(inplace=True)
            self.df = self._df5.resample(TIMEFRAME_RESAMPLE).agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna()

            self._calculate_indicators()
            self.last_candle_time = int(self.df.index[-1].timestamp() * 1000)

            if new_count > 0:
                logger.info(f"새 5분봉 {new_count}개, 10분봉 총 {len(self.df)}봉")
                return True
            return False

        except Exception as e:
            logger.error(f"캔들 업데이트 오류: {e}")
            return False

    async def start_websocket(self):
        self._ws_running = True
        self._ws_task = asyncio.create_task(self._ws_loop())
        logger.info("WebSocket 시작")

    async def stop_websocket(self):
        self._ws_running = False
        if self._ws:
            await self._ws.close()
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        logger.info("WebSocket 중지")

    async def _ws_loop(self):
        stream_url = f"{self.ws_url}/{BINANCE_SYMBOL.lower()}@kline_5m"

        while self._ws_running:
            try:
                async with websockets.connect(stream_url, ping_interval=20) as ws:
                    self._ws = ws
                    logger.info(f"WebSocket 연결: {stream_url}")
                    async for msg in ws:
                        if not self._ws_running:
                            break
                        self._process_ws_message(msg)
            except websockets.ConnectionClosed:
                logger.warning("WebSocket 연결 끊김, 5초 후 재연결...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"WebSocket 오류: {e}, 5초 후 재연결...")
                await asyncio.sleep(5)

    def _process_ws_message(self, msg: str):
        try:
            data = json.loads(msg)
            if 'k' not in data:
                return
            k = data['k']
            self.current_price = float(k['c'])
            self.current_high = float(k['h'])
            self.current_low = float(k['l'])
            self.price_update_time = time.time()

            if k['x']:  # 5분봉 완성 → 5분봉 DF 업데이트 → 10분봉 리샘플
                ts = pd.Timestamp(k['t'], unit='ms', tz='UTC')
                self._df5.loc[ts] = [
                    float(k['o']), float(k['h']),
                    float(k['l']), float(k['c']),
                    float(k['v'])
                ]
                self._df5.sort_index(inplace=True)
                # 10분봉 리샘플링
                self.df = self._df5.resample(TIMEFRAME_RESAMPLE).agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum'
                }).dropna()
                self._calculate_indicators()
                self.last_candle_time = k['t']
                logger.debug(f"5분봉 완성→10분봉 갱신: {ts} close={k['c']}")

        except Exception as e:
            logger.error(f"WS 메시지 처리 오류: {e}")

    def get_latest_index(self) -> int:
        return len(self.close) - 1

    def get_current_bar(self) -> dict:
        i = self.get_latest_index()
        if i < 1:
            return None
        return {
            'index': i,
            'close': self.close[i],
            'high': self.high[i],
            'low': self.low[i],
            'fast_ma': self.fast_ma[i],
            'slow_ma': self.slow_ma[i],
            'fast_ma_prev': self.fast_ma[i - 1],
            'slow_ma_prev': self.slow_ma[i - 1],
            'timestamp': str(self.df.index[i]),
            'realtime_price': self.current_price,
            'realtime_high': self.current_high,
            'realtime_low': self.current_low,
        }

    async def close_exchange(self):
        await self.stop_websocket()
        await self.exchange.close()
        logger.info("거래소 연결 종료")
