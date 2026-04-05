"""
BTC/USDT 선물 자동매매 - 데이터 수집 및 지표 계산
v32.2: EMA(100)/EMA(600) Tight-SL Trend System

- Binance Futures API 연동 (ccxt async)
- 30분봉 캔들 데이터 수집 및 유지
- EMA(100), EMA(600), ADX(20), RSI(10) 계산
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

logger = logging.getLogger('btc_data')

# ═══════════════════════════════════════════════════════════
# 파라미터
# ═══════════════════════════════════════════════════════════
SYMBOL = 'BTC/USDT'
BINANCE_SYMBOL = 'BTCUSDT'
TIMEFRAME = '30m'
FAST_MA_PERIOD = 100
SLOW_MA_PERIOD = 600
ADX_PERIOD = 20
RSI_PERIOD = 10
CANDLE_LIMIT = 1000  # 초기 로드 캔들 수 (EMA600 워밍업)


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

        # 캔들 데이터
        self.df: pd.DataFrame = None
        self.last_candle_time: int = 0

        # 계산된 지표 (numpy arrays)
        self.close: np.ndarray = None
        self.high: np.ndarray = None
        self.low: np.ndarray = None
        self.fast_ma: np.ndarray = None
        self.slow_ma: np.ndarray = None
        self.adx: np.ndarray = None
        self.rsi: np.ndarray = None

        # 실시간 가격
        self.current_price: float = 0.0
        self.current_high: float = 0.0
        self.current_low: float = 0.0
        self.price_update_time: float = 0.0

        # WebSocket 상태
        self._ws_task: asyncio.Task = None
        self._ws_running = False
        self._ws = None

    # ─── 초기화 ───
    async def initialize(self):
        """초기 캔들 데이터 로드 및 지표 계산"""
        logger.info("초기 캔들 데이터 로딩...")
        await self._load_historical_candles()
        self._calculate_indicators()
        logger.info(f"데이터 초기화 완료: {len(self.df)}봉, "
                    f"EMA100={self.fast_ma[-1]:.2f}, EMA600={self.slow_ma[-1]:.2f}")

    async def _load_historical_candles(self):
        """Binance에서 과거 캔들 로드 (최소 1000봉)"""
        all_candles = []
        since = None
        total_needed = CANDLE_LIMIT

        while len(all_candles) < total_needed:
            limit = min(1500, total_needed - len(all_candles))
            candles = await self.exchange.fetch_ohlcv(
                SYMBOL, TIMEFRAME, since=since, limit=limit
            )
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            if len(candles) < limit:
                break
            await asyncio.sleep(0.1)

        # DataFrame 생성
        self.df = pd.DataFrame(all_candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms', utc=True)
        self.df.set_index('timestamp', inplace=True)
        self.df = self.df.astype(float)

        # 중복 제거
        self.df = self.df[~self.df.index.duplicated(keep='last')]
        self.df.sort_index(inplace=True)

        self.last_candle_time = int(self.df.index[-1].timestamp() * 1000)
        self.current_price = float(self.df['close'].iloc[-1])
        logger.info(f"  로드 완료: {len(self.df)}봉 "
                    f"({self.df.index[0]} ~ {self.df.index[-1]})")

    # ─── 지표 계산 ───
    def _calculate_indicators(self):
        """모든 지표 재계산"""
        close_s = self.df['close']
        high_s = self.df['high']
        low_s = self.df['low']

        self.close = close_s.values.astype(np.float64)
        self.high = high_s.values.astype(np.float64)
        self.low = low_s.values.astype(np.float64)

        # EMA (pandas ewm span, adjust=False)
        self.fast_ma = close_s.ewm(span=FAST_MA_PERIOD, adjust=False).mean().values.astype(np.float64)
        self.slow_ma = close_s.ewm(span=SLOW_MA_PERIOD, adjust=False).mean().values.astype(np.float64)

        # ADX(20)
        self.adx = self._calc_adx(self.high, self.low, self.close, ADX_PERIOD)

        # RSI(10)
        self.rsi = self._calc_rsi(self.close, RSI_PERIOD)

    @staticmethod
    def _calc_adx(high, low, close, period):
        """ADX 계산 — ewm(alpha=1/period)"""
        n = len(close)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        tr = np.zeros(n)

        for i in range(1, n):
            h_diff = high[i] - high[i - 1]
            l_diff = low[i - 1] - low[i]
            plus_dm[i] = h_diff if (h_diff > l_diff and h_diff > 0) else 0.0
            minus_dm[i] = l_diff if (l_diff > h_diff and l_diff > 0) else 0.0
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))

        alpha = 1.0 / period
        atr = pd.Series(tr).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        sm_p = pd.Series(plus_dm).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        sm_m = pd.Series(minus_dm).ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        plus_di = 100.0 * sm_p / atr.replace(0, 1e-10)
        minus_di = 100.0 * sm_m / atr.replace(0, 1e-10)
        dx_denom = (plus_di + minus_di).replace(0, 1e-10)
        dx = 100.0 * (plus_di - minus_di).abs() / dx_denom
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        return adx.values.astype(np.float64)

    @staticmethod
    def _calc_rsi(close, period):
        """RSI 계산 — ewm(alpha=1/period)"""
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        alpha = 1.0 / period
        avg_gain = pd.Series(gain).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        avg_loss = pd.Series(loss).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return rsi.values.astype(np.float64)

    # ─── 캔들 업데이트 ───
    async def update_candles(self):
        """새 캔들 확인 및 추가"""
        try:
            candles = await self.exchange.fetch_ohlcv(
                SYMBOL, TIMEFRAME, limit=5
            )
            if not candles:
                return False

            new_count = 0
            for c in candles:
                ts = pd.Timestamp(c[0], unit='ms', tz='UTC')
                if ts in self.df.index:
                    # 기존 캔들 업데이트 (현재 형성 중인 캔들)
                    self.df.loc[ts] = [c[1], c[2], c[3], c[4], c[5]]
                else:
                    # 새 캔들 추가
                    self.df.loc[ts] = [c[1], c[2], c[3], c[4], c[5]]
                    new_count += 1

            if new_count > 0:
                self.df.sort_index(inplace=True)
                self._calculate_indicators()
                self.last_candle_time = int(self.df.index[-1].timestamp() * 1000)
                logger.info(f"새 캔들 {new_count}개 추가, 총 {len(self.df)}봉")
                return True
            else:
                # 현재 캔들만 업데이트된 경우에도 지표 재계산
                self._calculate_indicators()
                return False

        except Exception as e:
            logger.error(f"캔들 업데이트 오류: {e}")
            return False

    # ─── WebSocket 실시간 가격 ───
    async def start_websocket(self):
        """WebSocket으로 실시간 가격 수신 시작"""
        self._ws_running = True
        self._ws_task = asyncio.create_task(self._ws_loop())
        logger.info("WebSocket 시작")

    async def stop_websocket(self):
        """WebSocket 중지"""
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
        """WebSocket 연결 루프 (자동 재연결)"""
        stream_url = f"{self.ws_url}/{BINANCE_SYMBOL.lower()}@kline_30m"

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
        """WebSocket 메시지 처리"""
        try:
            data = json.loads(msg)
            if 'k' not in data:
                return
            k = data['k']
            self.current_price = float(k['c'])  # 종가
            self.current_high = float(k['h'])    # 고가
            self.current_low = float(k['l'])     # 저가
            self.price_update_time = time.time()

            # 캔들 완성 시 DataFrame 업데이트
            if k['x']:  # is_closed
                ts = pd.Timestamp(k['t'], unit='ms', tz='UTC')
                self.df.loc[ts] = [
                    float(k['o']), float(k['h']),
                    float(k['l']), float(k['c']),
                    float(k['v'])
                ]
                self.df.sort_index(inplace=True)
                self._calculate_indicators()
                self.last_candle_time = k['t']
                logger.debug(f"캔들 완성: {ts} close={k['c']}")

        except Exception as e:
            logger.error(f"WS 메시지 처리 오류: {e}")

    # ─── 데이터 접근 ───
    def get_latest_index(self) -> int:
        """최신 봉 인덱스"""
        return len(self.close) - 1

    def get_current_bar(self) -> dict:
        """현재 봉 데이터 및 지표"""
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
            'adx': self.adx[i],
            'adx_prev6': self.adx[i - 6] if i >= 6 else 0.0,
            'rsi': self.rsi[i],
            'ema_gap': abs(self.fast_ma[i] - self.slow_ma[i]) / self.slow_ma[i] * 100
                       if self.slow_ma[i] > 0 else 0.0,
            'timestamp': str(self.df.index[i]),
            'realtime_price': self.current_price,
            'realtime_high': self.current_high,
            'realtime_low': self.current_low,
        }

    def get_bar_at(self, index: int) -> dict:
        """특정 인덱스의 봉 데이터"""
        if index < 1 or index >= len(self.close):
            return None
        return {
            'index': index,
            'close': self.close[index],
            'high': self.high[index],
            'low': self.low[index],
            'fast_ma': self.fast_ma[index],
            'slow_ma': self.slow_ma[index],
            'fast_ma_prev': self.fast_ma[index - 1],
            'slow_ma_prev': self.slow_ma[index - 1],
            'adx': self.adx[index],
            'adx_prev6': self.adx[index - 6] if index >= 6 else 0.0,
            'rsi': self.rsi[index],
            'ema_gap': abs(self.fast_ma[index] - self.slow_ma[index]) /
                       self.slow_ma[index] * 100 if self.slow_ma[index] > 0 else 0.0,
        }

    # ─── 정리 ───
    async def close_exchange(self):
        """거래소 연결 종료"""
        await self.stop_websocket()
        await self.exchange.close()
        logger.info("거래소 연결 종료")
