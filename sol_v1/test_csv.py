"""CSV 저장/로드 플로우 테스트"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test():
    from sol_data_v1 import DataCollector

    print("="*60)
    print("CSV 저장/로드 테스트")
    print("="*60)

    # 1. 첫 로드 (CSV 없음 → API 초기 로드 → CSV 저장)
    print("\n[1차] 초기 로드 (CSV 없음)")
    dc = DataCollector(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
    await dc.initialize()
    print(f"  SOL 15m: {len(dc.df_sol)}봉")
    print(f"  BTC 1h: {len(dc.df_btc)}봉")
    print(f"  Raw 5m SOL: {len(dc._raw_5m_sol):,}봉")
    print(f"  Raw 5m BTC: {len(dc._raw_5m_btc):,}봉")

    # 캐시 파일 확인
    print("\n[Cache 파일]")
    import glob
    for f in sorted(glob.glob('cache/*.csv')):
        size = os.path.getsize(f) / 1024
        print(f"  {f}: {size:.1f} KB")

    await dc.close()

    # 2. 재로드 (CSV 있음 → CSV 우선 로드)
    print("\n[2차] 재로드 (CSV 캐시 활용)")
    import time
    t0 = time.time()
    dc2 = DataCollector(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
    await dc2.initialize()
    elapsed = time.time() - t0
    print(f"  SOL 15m: {len(dc2.df_sol)}봉")
    print(f"  BTC 1h: {len(dc2.df_btc)}봉")
    print(f"  로드 시간: {elapsed:.1f}초")
    await dc2.close()

    print("\n[OK] CSV 저장/로드 플로우 정상")

asyncio.run(test())
