"""
ETH/USDT 선물 자동매매 - 데이터 수집 및 지표 계산
V8.16: EMA(250)/EMA(1575) 10m Trend System

- Binance Futures API 연동 (ccxt async)
- 10분봉 캔들 데이터 수집 및 유지
- EMA(250), EMA(1575) 계산 (ADX/RSI 불필요)
- WebSocket 실시간 가격 업데이트
"""

import asyncio
import glob
import logging
import os
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
CANDLE_LIMIT = 4000  # CSV 없을 때 API에서 로드할 5분봉 수
CSV_PATH = 'eth_usdt_5m_2020_TO_NOW_merged.csv'  # 기존 merged CSV (호환용)
CSV_SAVE_INTERVAL = 300  # CSV 저장 주기 (초) — 5분


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
        self._last_csv_save: float = 0.0

    async def initialize(self):
        logger.info("초기 캔들 데이터 로딩...")
        await self._load_historical_candles()
        self._calculate_indicators()
        logger.info(f"데이터 초기화 완료: {len(self.df)}봉, "
                    f"EMA250={self.fast_ma[-1]:.2f}, EMA1575={self.slow_ma[-1]:.2f}")

    async def _load_historical_candles(self):
        """5분봉 로드 → 10분봉 리샘플링
        1순위: 연도별 CSV(eth_5m_YYYY.csv) → API 최신 보충
        2순위: 기존 merged CSV → API 최신 보충
        3순위: API에서 CANDLE_LIMIT만큼 로드
        """
        df5 = None

        # ── 1단계: CSV에서 히스토리 로드 ──
        df_csv = self._load_yearly_csvs()
        if df_csv is None and os.path.exists(CSV_PATH):
            try:
                df_csv = pd.read_csv(CSV_PATH, parse_dates=['timestamp'])
                df_csv.set_index('timestamp', inplace=True)
                df_csv = df_csv[['open', 'high', 'low', 'close', 'volume']].astype(float)
                logger.info(f"  기존 CSV 로드: {len(df_csv):,}봉")
            except Exception as e:
                logger.warning(f"  기존 CSV 로드 실패: {e}")
                df_csv = None

        if df_csv is not None:
            try:
                if df_csv.index.tz is None:
                    df_csv.index = df_csv.index.tz_localize('UTC')
                df_csv = df_csv[~df_csv.index.duplicated(keep='last')]
                df_csv.sort_index(inplace=True)
                logger.info(f"  CSV 총: {len(df_csv):,}봉 "
                            f"({df_csv.index[0]} ~ {df_csv.index[-1]})")

                # API로 CSV 이후 최신 봉 보충
                last_ts_ms = int(df_csv.index[-1].timestamp() * 1000) + 1
                api_candles = []
                since = last_ts_ms
                for _ in range(10):
                    candles = await self.exchange.fetch_ohlcv(
                        SYMBOL, TIMEFRAME_API, since=since, limit=1500)
                    if not candles:
                        break
                    api_candles.extend(candles)
                    since = candles[-1][0] + 1
                    if len(candles) < 1500:
                        break
                    await asyncio.sleep(0.1)

                if api_candles:
                    df_api = pd.DataFrame(api_candles, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_api['timestamp'] = pd.to_datetime(df_api['timestamp'], unit='ms', utc=True)
                    df_api.set_index('timestamp', inplace=True)
                    df_api = df_api.astype(float)
                    logger.info(f"  API 보충: {len(df_api)}봉 "
                                f"({df_api.index[0]} ~ {df_api.index[-1]})")
                    df5 = pd.concat([df_csv, df_api])
                else:
                    logger.info("  API 보충: 최신 데이터 없음 (CSV가 최신)")
                    df5 = df_csv
            except Exception as e:
                logger.warning(f"  CSV 로드 실패: {e}, API로 전환")
                df5 = None

        # ── 2단계: CSV 없으면 API에서 로드 (기존 방식) ──
        if df5 is None:
            logger.info("  CSV 없음 — API에서 로드")
            all_candles = []
            since = None
            total_needed = CANDLE_LIMIT
            while len(all_candles) < total_needed:
                limit = min(1500, total_needed - len(all_candles))
                candles = await self.exchange.fetch_ohlcv(
                    SYMBOL, TIMEFRAME_API, since=since, limit=limit)
                if not candles:
                    break
                all_candles.extend(candles)
                since = candles[-1][0] + 1
                if len(candles) < limit:
                    break
                await asyncio.sleep(0.1)
            df5 = pd.DataFrame(all_candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df5['timestamp'] = pd.to_datetime(df5['timestamp'], unit='ms', utc=True)
            df5.set_index('timestamp', inplace=True)
            df5 = df5.astype(float)

        # ── 공통: 정리 및 리샘플링 ──
        df5 = df5[~df5.index.duplicated(keep='last')]
        df5.sort_index(inplace=True)
        logger.info(f"  5분봉 총: {len(df5):,}봉")

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

        # CSV 저장 (API 보충분 포함 — 초기 로드 시 전체 저장)
        self._save_candles(full=True)
        self._last_csv_save = time.time()

    @staticmethod
    def _load_yearly_csvs() -> pd.DataFrame:
        """연도별 CSV 파일들을 로드하여 합치기"""
        files = sorted(glob.glob('eth_5m_*.csv'))
        if not files:
            return None
        dfs = []
        for f in files:
            df = pd.read_csv(f, parse_dates=['timestamp'], index_col='timestamp')
            dfs.append(df)
            logger.info(f"    {f}: {len(df):,}봉")
        combined = pd.concat(dfs)
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.sort_index(inplace=True)
        return combined

    def _save_candles(self, full=False):
        """5분봉 데이터를 연도별 CSV로 분할 저장
        full=True: 전체 연도 저장 (초기 로드 시)
        full=False: 현재 연도만 저장 (주기적 저장 시)
        """
        try:
            df_save = self._df5.copy()
            if df_save.index.tz is not None:
                df_save.index = df_save.index.tz_localize(None)
            df_save.index.name = 'timestamp'
            df_save = df_save[['open', 'high', 'low', 'close', 'volume']]

            if full:
                years = df_save.index.year.unique()
                for year in years:
                    df_year = df_save[df_save.index.year == year]
                    df_year.to_csv(f'eth_5m_{year}.csv')
                logger.info(f"  CSV 전체 저장: {len(df_save):,}봉, {len(years)}개 파일")
            else:
                current_year = df_save.index[-1].year
                df_year = df_save[df_save.index.year == current_year]
                df_year.to_csv(f'eth_5m_{current_year}.csv')
                logger.info(f"  CSV 저장: eth_5m_{current_year}.csv ({len(df_year):,}봉)")
        except Exception as e:
            logger.error(f"  CSV 저장 실패: {e}")

    def _calculate_indicators(self):
        close_s = self.df['close']
        self.close = close_s.values.astype(np.float64)
        self.high = self.df['high'].values.astype(np.float64)
        self.low = self.df['low'].values.astype(np.float64)

        # 전략A: EMA(250)/EMA(1575) on 10분봉
        self.fast_ma = close_s.ewm(span=FAST_MA_PERIOD, adjust=False).mean().values.astype(np.float64)
        self.slow_ma = close_s.ewm(span=SLOW_MA_PERIOD, adjust=False).mean().values.astype(np.float64)

        # 전략B: EMA(9) on 10분봉
        self.b_ema9 = close_s.ewm(span=9, adjust=False).mean().values.astype(np.float64)

        # 전략B: EMA(100) on 15분봉 → 10분봉 매핑
        df15 = self._df5.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
        ema100_15m = df15['close'].ewm(span=100, adjust=False).mean()

        # 15분봉 EMA(100) → 10분봉 매핑
        ts15 = df15.index.values.astype(np.int64) // 10**9
        ts10 = self.df.index.values.astype(np.int64) // 10**9
        ema100_vals = ema100_15m.values
        self.b_ema100_15m = np.full(len(self.df), np.nan)
        j = 0
        for i in range(len(self.df)):
            while j < len(df15) - 1 and ts15[j + 1] <= ts10[i]:
                j += 1
            self.b_ema100_15m[i] = ema100_vals[j]

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

            # 주기적 CSV 저장 (5분마다, 현재 연도만)
            if time.time() - self._last_csv_save >= CSV_SAVE_INTERVAL:
                self._save_candles(full=False)
                self._last_csv_save = time.time()

            return new_count > 0

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
            # 전략A
            'fast_ma': self.fast_ma[i],
            'slow_ma': self.slow_ma[i],
            'fast_ma_prev': self.fast_ma[i - 1],
            'slow_ma_prev': self.slow_ma[i - 1],
            # 전략B
            'b_ema9': self.b_ema9[i],
            'b_ema100_15m': self.b_ema100_15m[i],
            'b_ema9_prev': self.b_ema9[i - 1],
            'b_ema100_15m_prev': self.b_ema100_15m[i - 1],
            'b_ema9_prev2': self.b_ema9[i - 2] if i > 1 else None,
            'b_ema100_15m_prev2': self.b_ema100_15m[i - 2] if i > 1 else None,
            'timestamp': str(self.df.index[i]),
            'realtime_price': self.current_price,
            'realtime_high': self.current_high,
            'realtime_low': self.current_low,
        }

    async def close_exchange(self):
        await self.stop_websocket()
        await self.exchange.close()
        logger.info("거래소 연결 종료")
