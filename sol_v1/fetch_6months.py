"""
SOL/BTC 6개월치 5m 봉 완전 수집 → cache/*.csv
Binance Futures API (ccxt.binanceusdm) 페이지네이션 강건 버전

사용법:
  python fetch_6months.py
  → cache/sol_5m_YYYY.csv, cache/btc_5m_YYYY.csv 생성
  → AWS로 scp 전송 후 봇 재시작
"""
import asyncio
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import ccxt.async_support as ccxt
from dotenv import load_dotenv

load_dotenv()

DAYS = 180
BATCH_LIMIT = 1500
BATCH_INTERVAL_MS = BATCH_LIMIT * 5 * 60 * 1000  # 5분봉 × 1500개 = 7,500분 = 125시간 = 5.2일


async def fetch_full(exchange, symbol: str, days: int) -> pd.DataFrame:
    """symbol의 최근 days일치 5분봉을 since 포인터 진행 방식으로 완전 수집"""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000

    all_candles = []
    since = start_ms
    batch_num = 0

    print(f"\n[{symbol}] Target: {days}일 (since {datetime.utcfromtimestamp(start_ms/1000)} → now)")

    while since < end_ms:
        batch_num += 1
        try:
            # endTime 명시해서 배치당 범위 제한 (안전)
            params = {'endTime': min(since + BATCH_INTERVAL_MS, end_ms)}
            candles = await exchange.fetch_ohlcv(
                symbol, timeframe='5m', since=since, limit=BATCH_LIMIT, params=params
            )
        except Exception as e:
            print(f"  batch {batch_num} FAIL: {e}, 재시도...")
            await asyncio.sleep(3)
            continue

        if not candles:
            print(f"  batch {batch_num}: empty → 종료")
            break

        all_candles.extend(candles)
        first_ts = datetime.utcfromtimestamp(candles[0][0] / 1000)
        last_ts = datetime.utcfromtimestamp(candles[-1][0] / 1000)
        print(f"  batch {batch_num}: +{len(candles)} → 누적 {len(all_candles):,}  "
              f"({first_ts.strftime('%m-%d %H:%M')} ~ {last_ts.strftime('%m-%d %H:%M')})")

        # since 포인터: 다음 배치는 마지막 봉 + 1분
        next_since = candles[-1][0] + 1
        if next_since <= since:
            # 진행 안 됨 → 강제 진행 (1 batch interval)
            next_since = since + BATCH_INTERVAL_MS
        since = next_since

        # 너무 많은 배치 방지
        if batch_num > 100:
            print(f"  max batches 100 도달 → 종료")
            break

        await asyncio.sleep(0.15)  # rate limit

    if not all_candles:
        return None

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    span = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
    print(f"  최종: {len(df):,}봉, 기간 {span.days}일, "
          f"{df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")
    return df


def save_yearly(df: pd.DataFrame, symbol_short: str, cache_dir='cache'):
    """연도별 CSV 저장 (sol_data_v1.py 호환 포맷)"""
    os.makedirs(cache_dir, exist_ok=True)
    # timestamp 열을 index로 만들고 다시 column으로 (ISO 포맷 유지)
    df_save = df.copy()
    df_save['timestamp'] = df_save['timestamp'].dt.tz_convert('UTC')

    years = df_save['timestamp'].dt.year.unique()
    for year in sorted(years):
        mask = df_save['timestamp'].dt.year == year
        df_year = df_save[mask]
        path = os.path.join(cache_dir, f"{symbol_short}_5m_{year}.csv")
        df_year.to_csv(path, index=False)
        size_kb = os.path.getsize(path) / 1024
        print(f"  저장: {path} ({len(df_year):,}봉, {size_kb:.0f} KB)")


async def main():
    print("=" * 60)
    print(f"SOL + BTC 6개월치 5m 봉 수집 (Binance USDM Futures)")
    print(f"목표: {DAYS}일 × 288봉/일 = {DAYS * 288:,}봉")
    print("=" * 60)

    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    if not api_key or not api_secret:
        print("❌ BINANCE_API_KEY/SECRET 없음")
        return

    exchange = ccxt.binanceusdm({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'timeout': 30000,
        'options': {'fetchCurrencies': False, 'defaultType': 'future'},
    })

    try:
        await exchange.load_markets()
        print(f"Markets loaded: {len(exchange.symbols)} symbols")

        # SOL
        sol_df = await fetch_full(exchange, 'SOL/USDT:USDT', DAYS)
        if sol_df is not None:
            save_yearly(sol_df, 'sol')

        # BTC
        btc_df = await fetch_full(exchange, 'BTC/USDT:USDT', DAYS)
        if btc_df is not None:
            save_yearly(btc_df, 'btc')

        print("\n" + "=" * 60)
        print("✅ 6개월치 CSV 생성 완료")
        print("다음 단계: scp로 AWS 전송 + 봇 재시작")
        print("=" * 60)

    finally:
        await exchange.close()


if __name__ == '__main__':
    asyncio.run(main())
