"""
ETH/USDT 선물 자동매매 - 데이터 수집 및 지표 계산
V8: EMA(250)/EMA(1575) 10m Trend System
V16 Balanced: Confidence Score Sizing (ADX/RSI/LR/MACD_SIG)

- Binance Futures API 연동 (ccxt async)
- 10분봉 캔들 데이터 수집 및 유지
- EMA(250), EMA(1575) 계산
- V16 Balanced Sizing용 지표 (검증된 backtest와 동일 슬롯 매핑):
    [0]ADX(20) Wilder, [1]RSI(10) Wilder,
    [9]MACD_SIG(12,26,9), [11]LR_slope(20) 정규화
  → 이 4개로 0~100 점수 계산 → FULL 1.5x / HALF 1.0x / LOW 0.5x
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

# V16 Balanced Sizing 지표 파라미터 (backtest engine_data.py와 동일)
ADX_PERIOD = 20       # Wilder's smoothing
RSI_PERIOD = 10       # Wilder's smoothing
LR_PERIOD = 20        # Linear regression slope
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


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

        # V16 Balanced Sizing용 지표 (10분봉)
        self.adx: np.ndarray = None         # ADX(20)
        self.rsi: np.ndarray = None         # RSI(10)
        self.lr_slope: np.ndarray = None    # LR slope(20) 가격 대비 %
        self.mom: np.ndarray = None         # MACD signal line (momentum)

        self.current_price: float = 0.0
        self.current_high: float = 0.0
        self.current_low: float = 0.0
        self.price_update_time: float = 0.0

        self._ws_task: asyncio.Task = None
        self._ws_running = False
        self._ws = None
        self._last_csv_save: float = 0.0
        self._last_csv_cleanup: str = ''  # 마지막 cleanup 실행 월 (YYYY-MM)

    async def initialize(self):
        logger.info("초기 캔들 데이터 로딩...")
        await self._load_historical_candles()
        self._calculate_indicators()
        logger.info(f"데이터 초기화 완료: {len(self.df)}봉, "
                    f"EMA250={self.fast_ma[-1]:.2f}, EMA1575={self.slow_ma[-1]:.2f}")
        logger.info(f"V16 Sizing 지표: ADX={self.adx[-1]:.1f}, RSI={self.rsi[-1]:.1f}, "
                    f"LR={self.lr_slope[-1]:+.3f}%, MOM={self.mom[-1]:+.2f}")

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

                # 최근 6개월만 메모리 로드 (AWS t3.micro 메모리 절약)
                # EMA(1575) warmup은 10분봉 1600봉 ≈ 11일로 6개월은 충분
                cutoff_6m = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=180)
                before_cnt = len(df_csv)
                df_csv = df_csv[df_csv.index >= cutoff_6m]
                logger.info(f"  6개월 필터: {before_cnt:,} → {len(df_csv):,}봉")

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

        # CSV 저장 (API 보충분 포함 — 현재 연도만 저장)
        self._save_candles(full=False)
        self._last_csv_save = time.time()

    @staticmethod
    def _load_yearly_csvs() -> pd.DataFrame:
        """연도별 CSV 파일들을 로드 (최근 1년만 — 메모리 절약)"""
        files = sorted(glob.glob('eth_5m_*.csv'))
        if not files:
            return None

        current_year = datetime.now().year
        min_year = current_year - 1  # 이전 연도 + 현재 연도만 로드

        dfs = []
        for f in files:
            try:
                year = int(os.path.basename(f).replace('eth_5m_', '').replace('.csv', ''))
                if year < min_year:
                    os.remove(f)
                    logger.info(f"    {f}: 삭제 (1년 이전)")
                    continue
            except ValueError:
                continue
            df = pd.read_csv(f, parse_dates=['timestamp'], index_col='timestamp')
            dfs.append(df)
            logger.info(f"    {f}: {len(df):,}봉")

        if not dfs:
            return None

        combined = pd.concat(dfs)
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.sort_index(inplace=True)
        return combined

    def cleanup_old_csv(self):
        """매월 1일: 6개월 이전 5분봉 CSV 데이터 삭제/축소"""
        now = datetime.now(timezone.utc)
        if now.day != 1:
            return
        month_key = now.strftime('%Y-%m')
        if self._last_csv_cleanup == month_key:
            return  # 이번 달 이미 실행됨

        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=180)
        files = sorted(glob.glob('eth_5m_*.csv'))
        total_removed = 0
        files_deleted = 0
        files_shrunk = 0

        for f in files:
            try:
                df = pd.read_csv(f, parse_dates=['timestamp'], index_col='timestamp')
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                before = len(df)
                df_new = df[df.index >= cutoff]
                after = len(df_new)

                if after == 0:
                    os.remove(f)
                    files_deleted += 1
                    total_removed += before
                    logger.info(f"  CSV 삭제: {f} ({before:,}봉 전체 6개월 이전)")
                elif after < before:
                    df_new = df_new.copy()
                    df_new.index = df_new.index.tz_localize(None)
                    df_new.to_csv(f)
                    files_shrunk += 1
                    total_removed += (before - after)
                    logger.info(f"  CSV 축소: {f} ({before:,} → {after:,}봉)")
            except Exception as e:
                logger.error(f"  CSV cleanup 실패: {f} — {e}")

        self._last_csv_cleanup = month_key
        if total_removed > 0:
            logger.info(f"[CSV Cleanup {month_key}] 삭제 {files_deleted}개, 축소 {files_shrunk}개, "
                        f"총 {total_removed:,}봉 제거")
        else:
            logger.info(f"[CSV Cleanup {month_key}] 제거 대상 없음 (모두 6개월 이내)")

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
        high_s = self.df['high']
        low_s = self.df['low']
        self.close = close_s.values.astype(np.float64)
        self.high = high_s.values.astype(np.float64)
        self.low = low_s.values.astype(np.float64)

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

        # V16 Balanced: Confidence Score용 4개 지표 (검증된 backtest와 동일 계산)
        self._calc_score_indicators(close_s, high_s, low_s)

    def _calc_score_indicators(self, close_s: pd.Series, high_s: pd.Series, low_s: pd.Series):
        """V16 Balanced Sizing용 4개 지표 — engine_data.py와 완벽 동일 계산"""
        # ── ADX(20) — Wilder's smoothing (alpha=1/20) ──
        h = high_s; l = low_s; c = close_s
        a = 1.0 / ADX_PERIOD
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        up = h - h.shift(1)
        dn = l.shift(1) - l
        pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
        mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)
        atr = tr.ewm(alpha=a, min_periods=ADX_PERIOD, adjust=False).mean()
        pdi = 100 * pdm.ewm(alpha=a, min_periods=ADX_PERIOD, adjust=False).mean() / atr
        mdi = 100 * mdm.ewm(alpha=a, min_periods=ADX_PERIOD, adjust=False).mean() / atr
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-10)
        self.adx = dx.ewm(alpha=a, min_periods=ADX_PERIOD, adjust=False).mean().fillna(0).values.astype(np.float64)

        # ── RSI(10) — Wilder's smoothing (alpha=1/10) ──
        d = c.diff()
        g = d.where(d > 0, 0.0)
        lo = (-d).where(d < 0, 0.0)
        ar = 1.0 / RSI_PERIOD
        ag = g.ewm(alpha=ar, min_periods=RSI_PERIOD, adjust=False).mean()
        al = lo.ewm(alpha=ar, min_periods=RSI_PERIOD, adjust=False).mean()
        self.rsi = (100 - 100 / (1 + ag / al.replace(0, 1e-10))).fillna(50).values.astype(np.float64)

        # ── LR slope(20) — 가격 대비 % 정규화 ──
        n = len(c)
        lr_arr = np.zeros(n, dtype=np.float64)
        x = np.arange(LR_PERIOD, dtype=np.float64)
        x_mean = x.mean()
        x_var = float(np.sum((x - x_mean) ** 2))
        close_arr = self.close
        for i in range(LR_PERIOD - 1, n):
            y = close_arr[i - LR_PERIOD + 1:i + 1]
            y_mean = y.mean()
            cov = float(np.sum((x - x_mean) * (y - y_mean)))
            slope = cov / x_var if x_var != 0 else 0.0
            lr_arr[i] = slope / close_arr[i] * 100.0 if close_arr[i] != 0 else 0.0
        self.lr_slope = lr_arr

        # ── MACD signal line (12,26,9) — "momentum" 점수용 (검증된 backtest와 동일 슬롯 [9]) ──
        ema_fast = c.ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = c.ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        self.mom = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean().fillna(0).values.astype(np.float64)

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
            # V16 Balanced: Confidence Score용 (전략A 진입 시점 sizing)
            'adx': float(self.adx[i]) if self.adx is not None else 0.0,
            'rsi': float(self.rsi[i]) if self.rsi is not None else 50.0,
            'lr_slope': float(self.lr_slope[i]) if self.lr_slope is not None else 0.0,
            'mom': float(self.mom[i]) if self.mom is not None else 0.0,
            'timestamp': str(self.df.index[i]),
            'realtime_price': self.current_price,
            'realtime_high': self.current_high,
            'realtime_low': self.current_low,
        }

    async def close_exchange(self):
        await self.stop_websocket()
        await self.exchange.close()
        logger.info("거래소 연결 종료")
