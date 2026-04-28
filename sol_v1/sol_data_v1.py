"""
SOL/USDT 선물 데이터 수집 및 지표 계산
V1: V12:Mass 75:25 Mutex

- Binance Futures API (ccxt async)
- SOL 5분봉 수집 → 15분봉 리샘플링 (V12/Mass 공통)
- BTC 5분봉 → 1시간봉 리샘플링 (Margin Scaling용)
- WebSocket 실시간 가격
- 지표: EMA(9), SMA(400), ADX(20) Wilder, RSI(10) Wilder, LR slope(14), ATR(14/50), Mass Index
"""

import asyncio
import glob
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
import websockets

logger = logging.getLogger('sol_data')

SYMBOL = 'SOL/USDT:USDT'          # ccxt.binanceusdm perpetual swap 포맷
BINANCE_SYMBOL = 'SOLUSDT'         # WebSocket stream ID
BTC_SYMBOL = 'BTC/USDT:USDT'       # ccxt.binanceusdm perpetual swap 포맷
BTC_BINANCE_SYMBOL = 'BTCUSDT'

TIMEFRAME_API = '5m'
TIMEFRAME_PRIMARY = '15min'
TIMEFRAME_BTC = '1h'

FAST_MA = 9            # EMA(9)
SLOW_MA = 400          # SMA(400)
ADX_PERIOD = 20        # Wilder (TradingView 기준 + V12.1 백테스트 일치)
RSI_PERIOD = 10        # Wilder (TradingView 기준 + V12.1 백테스트 일치)
LR_PERIOD = 14
ATR_PERIOD = 14
ATR50_PERIOD = 50
MASS_EMA_PERIOD = 9
MASS_SUM_PERIOD = 25
BTC_EMA_PERIOD = 200

CANDLE_LIMIT_LOAD = 51840  # 5분봉 51840개 = 180일 (6개월) = 15m봉 17280개
CSV_SAVE_INTERVAL = 300    # 5분마다 CSV 저장
CSV_CACHE_DIR = 'cache'    # CSV 저장 디렉토리
CSV_FILTER_DAYS = 180      # 메모리 + CSV 보관 기간: 최근 180일 (6개월)
                           # AWS 디스크 용량 대비 경량화 + SMA400(100시간)에 충분
CSV_CLEANUP_CHECK_HOURS = 6  # 6시간마다 오래된 파일 정리 체크


class DataCollector:
    def __init__(self, api_key: str, api_secret: str, ws_url: str = None):
        # ccxt.binanceusdm: USD-M Futures 전용 (Spot API 호출 완전 회피)
        self.exchange = ccxt.binanceusdm({
            'apiKey': api_key, 'secret': api_secret,
            'enableRateLimit': True,
            'timeout': 30000,  # 30초 (exchangeInfo 큰 응답용)
            'options': {
                'fetchCurrencies': False,  # /sapi/v1/capital/config/getall 차단
            },
        })
        self.ws_url = ws_url or 'wss://fstream.binance.com/ws'

        # SOL 15m
        self.df_sol: Optional[pd.DataFrame] = None
        self.last_candle_time: int = 0

        # 지표 배열 (close/open 이름 충돌 방지: _arr suffix)
        self.close_arr: Optional[np.ndarray] = None
        self.high_arr: Optional[np.ndarray] = None
        self.low_arr: Optional[np.ndarray] = None
        self.open_arr: Optional[np.ndarray] = None
        self.fast_ma: Optional[np.ndarray] = None
        self.slow_ma: Optional[np.ndarray] = None
        self.adx: Optional[np.ndarray] = None
        self.rsi: Optional[np.ndarray] = None
        self.lr_slope: Optional[np.ndarray] = None
        self.atr14: Optional[np.ndarray] = None
        self.atr50: Optional[np.ndarray] = None
        self.mass_idx: Optional[np.ndarray] = None

        # BTC 1h EMA200
        self.df_btc: Optional[pd.DataFrame] = None
        self.btc_ema200: Optional[np.ndarray] = None
        self.btc_close: Optional[np.ndarray] = None
        self.btc_last_time: int = 0

        # WebSocket real-time
        self.current_price: float = 0.0
        self.current_high: float = 0.0
        self.current_low: float = 0.0
        self.price_update_time: float = 0.0

        self._ws_task: Optional[asyncio.Task] = None
        self._ws_running = False
        self._last_csv_save_sol: float = 0.0
        self._last_csv_save_btc: float = 0.0
        # CSV 원본 5분봉 캐시 (리샘플 전 원본 저장)
        self._raw_5m_sol: Optional[pd.DataFrame] = None
        self._raw_5m_btc: Optional[pd.DataFrame] = None
        # 캐시 디렉토리 확보
        os.makedirs(CSV_CACHE_DIR, exist_ok=True)

    async def initialize(self):
        logger.info("SOL 15m + BTC 1h 초기 데이터 로딩...")
        # 1. Markets 명시 로드 (재시도 포함)
        for attempt in range(3):
            try:
                await self.exchange.load_markets()
                logger.info(f"  Markets 로드 성공 ({len(self.exchange.markets)} symbols)")
                break
            except Exception as e:
                logger.warning(f"  Markets 로드 시도 {attempt+1}/3 실패: {type(e).__name__} {e}")
                if attempt == 2:
                    raise
                await asyncio.sleep(3)

        await self._load_historical_sol()
        await self._load_historical_btc()
        self._calculate_indicators()
        logger.info(f"SOL {len(self.df_sol)}봉 | BTC 1h {len(self.df_btc)}봉")
        logger.info(f"  EMA9={self.fast_ma[-1]:.3f} SMA400={self.slow_ma[-1]:.3f}")
        logger.info(f"  ADX={self.adx[-1]:.1f} RSI={self.rsi[-1]:.1f} LR={self.lr_slope[-1]:+.3f}")
        logger.info(f"  ATR14={self.atr14[-1]:.4f} ATR50={self.atr50[-1]:.4f} Mass={self.mass_idx[-1]:.2f}")
        logger.info(f"  BTC EMA200={self.btc_ema200[-1]:.2f} close={self.btc_close[-1]:.2f} | "
                    f"regime={'BULL' if self.btc_close[-1] > self.btc_ema200[-1] else 'BEAR'}")

    @staticmethod
    def _csv_path(symbol_short: str, year: int) -> str:
        """CSV 파일 경로 생성 (cache/sol_5m_2026.csv)"""
        return os.path.join(CSV_CACHE_DIR, f"{symbol_short}_5m_{year}.csv")

    def _load_yearly_csvs(self, symbol_short: str) -> Optional[pd.DataFrame]:
        """cache/ 디렉토리의 연도별 CSV 전부 로드 후 병합"""
        import glob as _glob
        pattern = os.path.join(CSV_CACHE_DIR, f"{symbol_short}_5m_*.csv")
        files = sorted(_glob.glob(pattern))
        if not files:
            logger.info(f"  CSV 캐시 없음 ({symbol_short}) - API에서 초기 로드")
            return None
        dfs = []
        for fp in files:
            try:
                df = pd.read_csv(fp, parse_dates=['timestamp'])
                df.set_index('timestamp', inplace=True)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                dfs.append(df[['open', 'high', 'low', 'close', 'volume']].astype(float))
            except Exception as e:
                logger.warning(f"  CSV 로드 실패 {fp}: {e}")
        if not dfs:
            return None
        out = pd.concat(dfs)
        out = out[~out.index.duplicated(keep='last')].sort_index()
        logger.info(f"  CSV {symbol_short} 총 {len(out):,}봉 로드 ({len(files)}개 파일)")
        return out

    def _save_csv_yearly(self, symbol_short: str, df5m: pd.DataFrame):
        """5분봉 DataFrame을 연도별 CSV로 분리 저장 + 6개월 초과 데이터 삭제"""
        if df5m is None or len(df5m) == 0:
            return

        # 1) 6개월 cutoff 적용 (메모리 DataFrame 자체도 경량화)
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=CSV_FILTER_DAYS)
        df5m_recent = df5m[df5m.index >= cutoff]

        # 2) 연도별 분리 저장 (최근 6개월 범위만)
        df_reset = df5m_recent.reset_index()
        if 'timestamp' not in df_reset.columns:
            df_reset.rename(columns={'index': 'timestamp'}, inplace=True)
        df_reset['year'] = pd.to_datetime(df_reset['timestamp']).dt.year
        saved_years = set()
        for yr, grp in df_reset.groupby('year'):
            path = self._csv_path(symbol_short, yr)
            grp = grp.drop(columns=['year'])
            grp.to_csv(path, index=False)
            saved_years.add(int(yr))

        # 3) 오래된 연도 CSV 파일 삭제 (6개월 범위 밖)
        self._cleanup_old_csv(symbol_short, cutoff, saved_years)

        logger.debug(f"  CSV {symbol_short} 저장 ({len(df5m_recent):,}봉, "
                     f"{df5m_recent.index[0]} ~ {df5m_recent.index[-1]})")

    def _cleanup_old_csv(self, symbol_short: str, cutoff: pd.Timestamp, saved_years: set):
        """6개월 초과 CSV 파일 삭제.
        saved_years 에 없는 연도 파일 = 저장된 데이터 범위 밖 → 삭제."""
        import glob as _glob
        pattern = os.path.join(CSV_CACHE_DIR, f"{symbol_short}_5m_*.csv")
        files = _glob.glob(pattern)
        deleted = []
        for fp in files:
            try:
                fname = os.path.basename(fp)
                # sol_5m_2025.csv 에서 2025 추출
                yr_str = fname.replace(f'{symbol_short}_5m_', '').replace('.csv', '')
                yr = int(yr_str)
                # 저장된 연도 집합에 없으면 = 오래된 파일 → 삭제
                if yr not in saved_years and yr < cutoff.year:
                    os.remove(fp)
                    deleted.append(fname)
            except (ValueError, OSError):
                continue
        if deleted:
            logger.info(f"  CSV cleanup ({symbol_short}): {len(deleted)}개 오래된 파일 삭제 - {deleted}")

    def _trim_raw_to_retention(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw 5m DataFrame을 보관 기간(180일)으로 자름"""
        if df is None or len(df) == 0:
            return df
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=CSV_FILTER_DAYS)
        return df[df.index >= cutoff]

    async def _load_5m_with_csv_fallback(self, symbol: str, symbol_short: str, total: int) -> Optional[pd.DataFrame]:
        """CSV-first 로드 전략:
        1. 기존 CSV 전체 로드
        2. CSV 최신 시점 이후 API 보충
        3. 최근 CSV_FILTER_DAYS 일만 유지 (메모리 절약)
        4. CSV 재저장 (전체 histroy 유지)"""
        df_csv = self._load_yearly_csvs(symbol_short)

        # CSV 로드 직후 6개월 trim (오래된 데이터는 저장하지 않음)
        if df_csv is not None and len(df_csv) > 0:
            df_csv = self._trim_raw_to_retention(df_csv)
            if len(df_csv) == 0:
                df_csv = None  # 6개월 내 데이터 없으면 API 전체 로드

        # API로 최신 보충
        if df_csv is not None and len(df_csv) > 0:
            last_ts_ms = int(df_csv.index[-1].timestamp() * 1000) + 1
            logger.info(f"  {symbol} CSV → API 최신 보충 (since {df_csv.index[-1]})")
            df_api = await self._fetch_5m_from(symbol, since_ms=last_ts_ms, max_batches=20)
            if df_api is not None and len(df_api) > 0:
                df_csv = pd.concat([df_csv, df_api])
                df_csv = df_csv[~df_csv.index.duplicated(keep='last')].sort_index()
                df_csv = self._trim_raw_to_retention(df_csv)
        else:
            # CSV 없음 or 6개월 이내 데이터 없음 → API 로드
            days_back = CSV_FILTER_DAYS   # ★ 항상 180일 전체 로드
            logger.info(f"  {symbol} API 로드 (최근 {days_back}일, 최대 {total}개 5분봉)")
            since_ms = int((time.time() - days_back * 86400) * 1000)
            df_csv = await self._fetch_5m_from(symbol, since_ms=since_ms, max_batches=40)
            if df_csv is None or len(df_csv) < 500:
                return None

        # ★ CSV 데이터가 부족하면 (보관기간 절반 미만) 과거 데이터 역행 fetch
        if df_csv is not None and len(df_csv) > 0:
            csv_span_days = (df_csv.index.max() - df_csv.index.min()).total_seconds() / 86400
            target_days = CSV_FILTER_DAYS
            if csv_span_days < target_days * 0.9:  # 162일 미만이면 부족
                logger.info(f"  {symbol} CSV 부족 ({csv_span_days:.1f}일 < {target_days}일) → 과거 데이터 역행 fetch")
                earliest_ms = int(df_csv.index.min().timestamp() * 1000)
                target_earliest_ms = int((time.time() - target_days * 86400) * 1000)
                if earliest_ms > target_earliest_ms:
                    df_past = await self._fetch_5m_from(symbol,
                                                        since_ms=target_earliest_ms,
                                                        max_batches=40)
                    if df_past is not None and len(df_past) > 0:
                        df_csv = pd.concat([df_past, df_csv])
                        df_csv = df_csv[~df_csv.index.duplicated(keep='last')].sort_index()
                        df_csv = self._trim_raw_to_retention(df_csv)
                        logger.info(f"  {symbol} 역행 fetch 완료: 총 {len(df_csv):,}봉")

        # CSV에 저장 (6개월 내 데이터만, 연도별 분리) + 오래된 파일 자동 삭제
        self._save_csv_yearly(symbol_short, df_csv)

        logger.info(f"  {symbol} 메모리 {len(df_csv):,}봉 유지 (최근 {CSV_FILTER_DAYS}일)")
        return df_csv

    async def _fetch_5m_from(self, symbol: str, since_ms: int, max_batches: int = 20) -> Optional[pd.DataFrame]:
        """since_ms 부터 현재까지 5분봉 페이지네이션 수집"""
        all_candles = []
        since = since_ms
        for batch in range(max_batches):
            try:
                candles = await self.exchange.fetch_ohlcv(symbol, TIMEFRAME_API, since=since, limit=1500)
            except Exception as e:
                logger.warning(f"  {symbol} batch {batch+1} 실패: {e}")
                await asyncio.sleep(3)
                continue
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            if len(candles) < 1500:
                break
            await asyncio.sleep(0.2)
        if not all_candles:
            return None
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        return df.astype(float)

    async def _load_historical_sol(self):
        df5 = await self._load_5m_with_csv_fallback(SYMBOL, 'sol', CANDLE_LIMIT_LOAD)
        if df5 is None or len(df5) < 500:
            raise RuntimeError("SOL 5m 데이터 로드 실패")
        self._raw_5m_sol = df5  # 원본 5m 보존 (나중에 CSV 저장용)
        df15 = self._resample_to(df5, TIMEFRAME_PRIMARY)
        self.df_sol = df15
        self.last_candle_time = int(df15.index[-1].timestamp() * 1000)
        self._last_csv_save_sol = time.time()

    async def _load_historical_btc(self):
        df5 = await self._load_5m_with_csv_fallback(BTC_SYMBOL, 'btc', CANDLE_LIMIT_LOAD)
        if df5 is None or len(df5) < 500:
            raise RuntimeError("BTC 5m 데이터 로드 실패")
        self._raw_5m_btc = df5
        df1h = self._resample_to(df5, TIMEFRAME_BTC)
        self.df_btc = df1h
        self.btc_last_time = int(df1h.index[-1].timestamp() * 1000)
        self._last_csv_save_btc = time.time()

    async def _fetch_5m_candles(self, symbol: str, total: int) -> Optional[pd.DataFrame]:
        """Fetch with retry + detailed error logging.
        total = 원하는 5분봉 수. since 를 과거로 설정하여 페이지네이션으로 충분히 수집."""
        all_candles = []
        # 과거 시점부터 시작하여 현재까지 수집 (total/288 일 전부터)
        days_back = max(30, total // 288 + 5)  # 최소 30일, 필요시 확장
        since = int((time.time() - days_back * 86400) * 1000)
        batch_count = 0
        max_retries = 3

        while len(all_candles) < total:
            limit = min(1500, total - len(all_candles))
            candles = None
            for attempt in range(max_retries):
                try:
                    candles = await self.exchange.fetch_ohlcv(symbol, TIMEFRAME_API, since=since, limit=limit)
                    break
                except Exception as e:
                    err_type = type(e).__name__
                    logger.warning(f"{symbol} fetch batch {batch_count+1} 시도 {attempt+1}/{max_retries} {err_type}: {str(e)[:200]}")
                    if attempt == max_retries - 1:
                        if all_candles:
                            # 부분 성공 → 있는 것이라도 반환
                            logger.warning(f"  부분 데이터 사용: {len(all_candles)}개")
                            break
                        else:
                            logger.error(f"{symbol} 5m fetch 완전 실패: {err_type} {e}")
                            return None
                    await asyncio.sleep(3)
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            batch_count += 1
            if len(candles) < 1500:
                break
            await asyncio.sleep(0.2)

        if not all_candles:
            return None

        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        logger.info(f"  {symbol} 5m: {len(df)}봉 로드")
        return df.astype(float)

    @staticmethod
    def _resample_to(df5: pd.DataFrame, tf: str) -> pd.DataFrame:
        # pandas 최신 표기: '15min', '1h' 직접 사용 (deprecated 'T'/'H' 대신)
        rule = tf
        out = df5.resample(rule, label='left', closed='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
        return out

    # ═══════════════════════════════════════════════════════════
    # 지표 계산
    # ═══════════════════════════════════════════════════════════
    def _calculate_indicators(self):
        c = self.df_sol['close'].values.astype(np.float64)
        h = self.df_sol['high'].values.astype(np.float64)
        l = self.df_sol['low'].values.astype(np.float64)
        o = self.df_sol['open'].values.astype(np.float64)
        self.close_arr = c; self.high_arr = h; self.low_arr = l; self.open_arr = o

        # EMA(9)
        self.fast_ma = self._ema(c, FAST_MA)
        # SMA(400)
        self.slow_ma = pd.Series(c).rolling(SLOW_MA).mean().values
        # ADX
        self.adx = self._adx(h, l, c, ADX_PERIOD)
        # RSI
        self.rsi = self._rsi(c, RSI_PERIOD)
        # LR slope (14)
        self.lr_slope = self._lr_slope(c, LR_PERIOD)
        # ATR
        self.atr14 = self._atr(h, l, c, ATR_PERIOD)
        self.atr50 = self._atr(h, l, c, ATR50_PERIOD)
        # Mass Index
        self.mass_idx = self._mass_index(h, l, MASS_EMA_PERIOD, MASS_SUM_PERIOD)

        # BTC EMA200
        btc_c = self.df_btc['close'].values.astype(np.float64)
        self.btc_close = btc_c
        self.btc_ema200 = self._ema(btc_c, BTC_EMA_PERIOD)

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(arr).ewm(span=period, adjust=False).mean().values

    @staticmethod
    def _rsi(c: np.ndarray, period: int = 10) -> np.ndarray:
        """Wilder's smoothing RSI (TradingView 기본값 + V12.1 백테스트 일치).
        alpha = 1/period, min_periods=period 로 pandas ewm 사용.
        """
        series = pd.Series(c)
        d = series.diff()
        g = d.where(d > 0, 0.0)
        l = (-d).where(d < 0, 0.0)
        a = 1.0 / period
        ag = g.ewm(alpha=a, min_periods=period, adjust=False).mean()
        al = l.ewm(alpha=a, min_periods=period, adjust=False).mean()
        rsi = 100 - 100 / (1 + ag / al.replace(0, 1e-10))
        return rsi.fillna(50).values.astype(np.float64)

    @staticmethod
    def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        return pd.Series(tr).rolling(period).mean().values

    @staticmethod
    def _adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 20) -> np.ndarray:
        """Wilder's smoothing ADX (TradingView 기본값 + V12.1 백테스트 일치).
        alpha = 1/period, ATR/+DI/-DI/ADX 모두 Wilder ewm 사용.
        """
        h_s = pd.Series(h); l_s = pd.Series(l); c_s = pd.Series(c)
        a = 1.0 / period

        # True Range
        tr = pd.concat([
            h_s - l_s,
            (h_s - c_s.shift(1)).abs(),
            (l_s - c_s.shift(1)).abs()
        ], axis=1).max(axis=1)

        # Directional Movement
        up = h_s - h_s.shift(1)
        dn = l_s.shift(1) - l_s
        pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c_s.index)
        mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c_s.index)

        # Wilder 평활 (ewm alpha=1/period)
        atr_w = tr.ewm(alpha=a, min_periods=period, adjust=False).mean()
        pdi = 100 * pdm.ewm(alpha=a, min_periods=period, adjust=False).mean() / atr_w
        mdi = 100 * mdm.ewm(alpha=a, min_periods=period, adjust=False).mean() / atr_w
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-10)
        adx = dx.ewm(alpha=a, min_periods=period, adjust=False).mean()
        return adx.fillna(0).values.astype(np.float64)

    @staticmethod
    def _lr_slope(c: np.ndarray, period: int = 14) -> np.ndarray:
        n = len(c)
        slope = np.full(n, np.nan)
        x = np.arange(period)
        for i in range(period, n):
            y = c[i-period+1:i+1]
            m = np.polyfit(x, y, 1)[0]
            slope[i] = m / (y.mean() + 1e-10) * 100
        return slope

    @staticmethod
    def _mass_index(h: np.ndarray, l: np.ndarray, ema_period: int, sum_period: int) -> np.ndarray:
        hl = h - l
        ema1 = pd.Series(hl).ewm(span=ema_period, adjust=False).mean().values
        ema2 = pd.Series(ema1).ewm(span=ema_period, adjust=False).mean().values
        ratio = ema1 / (ema2 + 1e-10)
        mass = pd.Series(ratio).rolling(sum_period).sum().values
        return mass

    # ═══════════════════════════════════════════════════════════
    # Live candle update
    # ═══════════════════════════════════════════════════════════
    async def _fetch_ohlcv_retry(self, symbol: str, since: int, limit: int = 20, max_retries: int = 3):
        """네트워크 에러 시 재시도 (지수 백오프). 모두 실패 시 None 반환."""
        last_err = None
        for attempt in range(max_retries):
            try:
                return await self.exchange.fetch_ohlcv(symbol, TIMEFRAME_API, since=since, limit=limit)
            except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as e:
                last_err = e
                wait = 2 ** attempt  # 1, 2, 4초
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait)
                    continue
            except Exception as e:
                last_err = e
                break  # 다른 에러는 재시도 안 함
        # 모두 실패
        logger.warning(f"{symbol} fetch_ohlcv {max_retries}회 실패: {last_err}")
        return None

    async def update_candles(self):
        """최신 SOL + BTC 캔들 가져와서 추가 + 주기적 CSV 저장"""
        now = time.time()

        # SOL
        try:
            since = self.last_candle_time + 1
            candles = await self._fetch_ohlcv_retry(SYMBOL, since, 20)
            if candles:
                df_new5 = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_new5['timestamp'] = pd.to_datetime(df_new5['timestamp'], unit='ms', utc=True)
                df_new5.set_index('timestamp', inplace=True)
                df_new5 = df_new5.astype(float)

                # 원본 5분봉 누적 (CSV 저장용) + 6개월 trim
                if self._raw_5m_sol is not None:
                    self._raw_5m_sol = pd.concat([self._raw_5m_sol, df_new5])
                    self._raw_5m_sol = self._raw_5m_sol[~self._raw_5m_sol.index.duplicated(keep='last')].sort_index()
                    self._raw_5m_sol = self._trim_raw_to_retention(self._raw_5m_sol)

                # ★ 리샘플링은 df_new5(새 fetch)가 아닌 _raw_5m_sol 최근 부분으로
                # 이유: df_new5에 진행 중 15m 봉의 모든 5m 봉이 없을 수 있음 (since 필터로 일부 제외)
                # 최근 30개 5m 봉(150분) = 최근 10개 15m 봉이 정확히 리샘플됨
                recent_5m = self._raw_5m_sol.tail(30) if self._raw_5m_sol is not None and len(self._raw_5m_sol) >= 30 else df_new5
                df_new15 = self._resample_to(recent_5m, TIMEFRAME_PRIMARY)
                if len(df_new15) > 0:
                    # ★ 기존 봉도 덮어쓰기: 진행 중 15m 봉의 OHLC 갱신 (좁은 봉 고정 버그 해결)
                    # 5m 봉이 추가될 때마다 기존 15m 봉의 H/L/C가 확장되어야 함
                    self.df_sol = pd.concat([self.df_sol, df_new15])
                    self.df_sol = self.df_sol[~self.df_sol.index.duplicated(keep='last')].sort_index()
                    # 6개월(180일) 기준으로 15m 봉 제한: 17,280봉 = 180*24*4
                    MAX_15M_BARS = CSV_FILTER_DAYS * 24 * 4  # 17,280
                    if len(self.df_sol) > MAX_15M_BARS:
                        self.df_sol = self.df_sol.iloc[-MAX_15M_BARS:]
                    self.last_candle_time = int(df_new15.index[-1].timestamp() * 1000)
                    self._calculate_indicators()
                    logger.debug(f"SOL 15m 갱신, 총 {len(self.df_sol)}봉 (df_new15 {len(df_new15)}봉)")

            # 주기적 CSV 저장 (5분마다)
            if now - self._last_csv_save_sol >= CSV_SAVE_INTERVAL:
                if self._raw_5m_sol is not None and len(self._raw_5m_sol) > 0:
                    self._save_csv_yearly('sol', self._raw_5m_sol)
                    self._last_csv_save_sol = now
        except Exception as e:
            # 네트워크 에러는 이미 _fetch_ohlcv_retry 에서 처리. 여기까지 온 건 다른 에러.
            logger.warning(f"SOL candle update 오류 (무시, 다음 tick에 재시도): {e}")

        # BTC 1h
        try:
            since = self.btc_last_time + 1
            candles = await self._fetch_ohlcv_retry(BTC_SYMBOL, since, 20)
            if candles:
                df_new5 = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_new5['timestamp'] = pd.to_datetime(df_new5['timestamp'], unit='ms', utc=True)
                df_new5.set_index('timestamp', inplace=True)
                df_new5 = df_new5.astype(float)

                # 원본 5분봉 누적 + 6개월 trim
                if self._raw_5m_btc is not None:
                    self._raw_5m_btc = pd.concat([self._raw_5m_btc, df_new5])
                    self._raw_5m_btc = self._raw_5m_btc[~self._raw_5m_btc.index.duplicated(keep='last')].sort_index()
                    self._raw_5m_btc = self._trim_raw_to_retention(self._raw_5m_btc)

                # ★ BTC도 동일: 최근 24개 5m 봉 (=2시간) 사용해서 1h 봉 정확 리샘플
                recent_btc_5m = self._raw_5m_btc.tail(30) if self._raw_5m_btc is not None and len(self._raw_5m_btc) >= 30 else df_new5
                df_new1h = self._resample_to(recent_btc_5m, TIMEFRAME_BTC)
                if len(df_new1h) > 0:
                    # ★ 기존 봉 덮어쓰기 (진행 중 1h 봉 OHLC 갱신)
                    self.df_btc = pd.concat([self.df_btc, df_new1h])
                    self.df_btc = self.df_btc[~self.df_btc.index.duplicated(keep='last')].sort_index()
                    # 6개월(180일) 기준 1h 봉 제한: 4,320봉 = 180*24
                    MAX_1H_BARS = CSV_FILTER_DAYS * 24  # 4,320
                    if len(self.df_btc) > MAX_1H_BARS:
                        self.df_btc = self.df_btc.iloc[-MAX_1H_BARS:]
                    self.btc_last_time = int(df_new1h.index[-1].timestamp() * 1000)
                    btc_c = self.df_btc['close'].values.astype(np.float64)
                    self.btc_close = btc_c
                    self.btc_ema200 = self._ema(btc_c, BTC_EMA_PERIOD)

            # 주기적 CSV 저장
            if now - self._last_csv_save_btc >= CSV_SAVE_INTERVAL:
                if self._raw_5m_btc is not None and len(self._raw_5m_btc) > 0:
                    self._save_csv_yearly('btc', self._raw_5m_btc)
                    self._last_csv_save_btc = now
        except Exception as e:
            logger.warning(f"BTC candle update 오류 (무시, 다음 tick에 재시도): {e}")

    # ═══════════════════════════════════════════════════════════
    # WebSocket (real-time price)
    # ═══════════════════════════════════════════════════════════
    async def start_websocket(self):
        self._ws_task = asyncio.create_task(self._ws_loop())

    async def _ws_loop(self):
        self._ws_running = True
        stream = f"{BINANCE_SYMBOL.lower()}@kline_1m"
        url = f"{self.ws_url}/{stream}"
        while self._ws_running:
            try:
                async with websockets.connect(url, ping_interval=30, ping_timeout=10) as ws:
                    logger.info(f"WebSocket 연결: {stream}")
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                            k = data.get('k', {})
                            self.current_price = float(k.get('c', self.current_price))
                            self.current_high = float(k.get('h', self.current_price))
                            self.current_low = float(k.get('l', self.current_price))
                            self.price_update_time = time.time()
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"WebSocket 재연결 예정: {e}")
                await asyncio.sleep(5)

    async def stop_websocket(self):
        self._ws_running = False
        if self._ws_task:
            self._ws_task.cancel()
            try: await self._ws_task
            except asyncio.CancelledError: pass

    # ═══════════════════════════════════════════════════════════
    # 편의 메서드
    # ═══════════════════════════════════════════════════════════
    def get_latest_index(self) -> int:
        return len(self.df_sol) - 1 if self.df_sol is not None else 0

    def get_last_closed_bar(self) -> Optional[dict]:
        """마지막 마감된 15m 봉"""
        if self.df_sol is None or len(self.df_sol) < 2:
            return None
        idx = len(self.df_sol) - 2  # -2: 마지막이 현재 진행중 봉이면, -2가 마감봉
        row = self.df_sol.iloc[idx]
        return {
            'timestamp': self.df_sol.index[idx].timestamp(),
            'open': float(row['open']), 'high': float(row['high']),
            'low': float(row['low']), 'close': float(row['close']),
            'volume': float(row['volume']), 'bar_idx': idx,
        }

    def get_indicators_at(self, idx: int) -> dict:
        """특정 bar_idx 의 지표들"""
        if idx < 0 or idx >= len(self.close_arr):
            return {}
        ind = {
            'fast_ma': self.fast_ma[idx] if idx < len(self.fast_ma) else np.nan,
            'slow_ma': self.slow_ma[idx] if idx < len(self.slow_ma) else np.nan,
            'prev_fast_ma': self.fast_ma[idx-1] if idx >= 1 else np.nan,
            'prev_slow_ma': self.slow_ma[idx-1] if idx >= 1 else np.nan,
            'adx': self.adx[idx] if idx < len(self.adx) else np.nan,
            'adx_prev_6': self.adx[idx-6] if idx >= 6 else np.nan,
            'rsi': self.rsi[idx] if idx < len(self.rsi) else np.nan,
            'lr_slope': self.lr_slope[idx] if idx < len(self.lr_slope) else np.nan,
            'atr14': self.atr14[idx] if idx < len(self.atr14) else np.nan,
            'atr50': self.atr50[idx] if idx < len(self.atr50) else np.nan,
            'mass': self.mass_idx[idx] if idx < len(self.mass_idx) else np.nan,
        }
        return ind

    def get_btc_regime_bull(self, sol_bar_time: float) -> bool:
        """SOL bar_time 기준 BTC 1H EMA200 regime (이전 마감봉 기준)"""
        if self.df_btc is None or self.btc_ema200 is None:
            return True  # default Bull
        ts = pd.Timestamp(sol_bar_time, unit='s', tz='UTC')
        # 이전 1h 마감봉: floor(1h) - 1h
        floor_h = ts.floor('1h')
        last_closed = floor_h - pd.Timedelta(hours=1)
        btc_ts = self.df_btc.index
        idx = btc_ts.searchsorted(last_closed, side='right') - 1
        if idx < 0 or idx >= len(self.btc_close) or np.isnan(self.btc_ema200[idx]):
            return True
        return bool(self.btc_close[idx] > self.btc_ema200[idx])

    async def close(self):
        # 종료 시 CSV 최종 저장 (데이터 유실 방지)
        try:
            if self._raw_5m_sol is not None and len(self._raw_5m_sol) > 0:
                self._save_csv_yearly('sol', self._raw_5m_sol)
                logger.info(f"종료 시 SOL CSV 저장: {len(self._raw_5m_sol):,}봉")
            if self._raw_5m_btc is not None and len(self._raw_5m_btc) > 0:
                self._save_csv_yearly('btc', self._raw_5m_btc)
                logger.info(f"종료 시 BTC CSV 저장: {len(self._raw_5m_btc):,}봉")
        except Exception as e:
            logger.warning(f"종료 CSV 저장 오류: {e}")
        await self.stop_websocket()
        await self.exchange.close()
