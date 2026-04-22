"""
연결 테스트: binanceusdm (USD-M Futures 전용) 작동 검증
"""
import asyncio
import os
from dotenv import load_dotenv
import ccxt.async_support as ccxt


async def test():
    load_dotenv()
    key = os.getenv('BINANCE_API_KEY')
    sec = os.getenv('BINANCE_API_SECRET')

    print("="*60)
    print("Binance USD-M Futures 연결 테스트")
    print("="*60)

    exchange = ccxt.binanceusdm({
        'apiKey': key, 'secret': sec,
        'enableRateLimit': True,
        'options': {'fetchCurrencies': False},
    })

    try:
        # 1. 마켓 로드 테스트
        print("\n[1] 마켓 로드...")
        markets = await exchange.load_markets()
        sol_market = markets.get('SOL/USDT')
        btc_market = markets.get('BTC/USDT')
        print(f"  SOL/USDT 활성: {sol_market is not None}")
        print(f"  BTC/USDT 활성: {btc_market is not None}")
        if sol_market:
            print(f"  SOL 최소수량: {sol_market['limits']['amount']['min']}")

        # 2. 잔액 조회
        print("\n[2] 잔액 조회...")
        balance = await exchange.fetch_balance()
        usdt = balance.get('USDT', {})
        print(f"  USDT total: ${float(usdt.get('total', 0)):.2f}")
        print(f"  USDT free:  ${float(usdt.get('free', 0)):.2f}")

        # 3. SOL 캔들 조회
        print("\n[3] SOL 5m 캔들 조회 (최근 10개)...")
        candles = await exchange.fetch_ohlcv('SOL/USDT:USDT', '5m', limit=10)
        print(f"  받은 캔들: {len(candles)}")
        if candles:
            last = candles[-1]
            import datetime as dt
            ts = dt.datetime.utcfromtimestamp(last[0]/1000)
            print(f"  마지막 봉: {ts} O={last[1]:.3f} H={last[2]:.3f} L={last[3]:.3f} C={last[4]:.3f}")

        # 4. BTC 캔들 조회
        print("\n[4] BTC 5m 캔들 조회 (최근 5개)...")
        candles = await exchange.fetch_ohlcv('BTC/USDT:USDT', '5m', limit=5)
        print(f"  받은 캔들: {len(candles)}")
        if candles:
            last = candles[-1]
            print(f"  마지막 가격: ${last[4]:,.2f}")

        # 5. 포지션 조회
        print("\n[5] SOL 포지션 조회...")
        positions = await exchange.fetch_positions(['SOL/USDT:USDT'])
        for p in positions:
            contracts = abs(float(p.get('contracts', 0) or 0))
            if contracts > 0:
                print(f"  활성 포지션: {p['side']} {contracts} SOL @ ${p.get('entryPrice')}")
            else:
                print(f"  {p.get('symbol', '?')}: 포지션 없음")

        print("\n✅ 모든 연결 테스트 성공!")

    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await exchange.close()


if __name__ == '__main__':
    asyncio.run(test())
