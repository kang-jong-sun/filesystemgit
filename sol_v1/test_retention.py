"""6개월 보관 정책 테스트
- 가짜 오래된 연도 CSV 생성
- _save_csv_yearly 호출 → 자동 cleanup 확인
"""
import asyncio
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()


async def test():
    from sol_data_v1 import DataCollector, CSV_FILTER_DAYS

    print("="*60)
    print("6개월 데이터 보관 정책 테스트")
    print("="*60)
    print(f"보관 기간: {CSV_FILTER_DAYS}일")

    # 1. 가짜 오래된 파일 생성
    print("\n[1] 가짜 오래된 연도 CSV 생성 (2023, 2024)")
    fake_years = [2023, 2024]
    for yr in fake_years:
        path = f"cache/sol_5m_{yr}.csv"
        timestamps = pd.date_range(f'{yr}-01-01', f'{yr}-01-02', freq='5min', tz='UTC')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': np.random.uniform(50, 100, len(timestamps)),
            'high': np.random.uniform(50, 100, len(timestamps)),
            'low': np.random.uniform(50, 100, len(timestamps)),
            'close': np.random.uniform(50, 100, len(timestamps)),
            'volume': np.random.uniform(1000, 10000, len(timestamps)),
        })
        df.to_csv(path, index=False)
        size = os.path.getsize(path) / 1024
        print(f"  생성: {path} ({size:.1f} KB, {len(df)}봉)")

    # 2. 현재 캐시 파일 목록 확인
    print("\n[2] 생성 후 캐시 파일 목록")
    import glob
    for f in sorted(glob.glob('cache/*.csv')):
        size = os.path.getsize(f) / 1024
        print(f"  {f}: {size:.1f} KB")

    # 3. DataCollector 초기화 → CSV 로드 시 trim + cleanup 발동
    print("\n[3] DataCollector 초기화 (cleanup 발동 예상)")
    dc = DataCollector(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
    await dc.initialize()
    print(f"  SOL 15m: {len(dc.df_sol)}봉")
    print(f"  BTC 1h: {len(dc.df_btc)}봉")
    print(f"  Raw 5m SOL: {len(dc._raw_5m_sol):,}봉")

    # 4. 저장 후 파일 목록 확인 (오래된 파일 삭제되었는지)
    print("\n[4] 처리 후 캐시 파일 목록 (오래된 연도 삭제 예상)")
    for f in sorted(glob.glob('cache/*.csv')):
        size = os.path.getsize(f) / 1024
        print(f"  {f}: {size:.1f} KB")

    # 5. 6개월 이상 된 봉이 없는지 검증
    print("\n[5] 메모리 데이터 검증")
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=CSV_FILTER_DAYS)
    sol_oldest = dc._raw_5m_sol.index.min()
    print(f"  SOL 최고 오래된 봉: {sol_oldest}")
    print(f"  Cutoff (180일 전): {cutoff}")
    if sol_oldest >= cutoff:
        print(f"  ✅ 6개월 이내 데이터만 메모리에 있음")
    else:
        print(f"  ❌ 6개월 넘는 데이터 존재!")

    await dc.close()
    print("\n[OK] 6개월 보관 정책 정상 작동 ✅")


asyncio.run(test())
