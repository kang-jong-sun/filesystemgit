"""사용 가능한 SOL/BTC 심볼 확인"""
import asyncio
import os
from dotenv import load_dotenv
import ccxt.async_support as ccxt

async def test():
    load_dotenv()
    exchange = ccxt.binanceusdm({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_API_SECRET'),
        'enableRateLimit': True,
        'options': {'fetchCurrencies': False},
    })
    try:
        markets = await exchange.load_markets()
        # SOL 관련 심볼 찾기
        sol_syms = [s for s in markets.keys() if 'SOL' in s]
        btc_syms = [s for s in markets.keys() if s.startswith('BTC')]
        print("SOL 심볼들:")
        for s in sol_syms[:10]:
            print(f"  {s}")
        print("\nBTC 심볼들:")
        for s in btc_syms[:5]:
            print(f"  {s}")
        # SOL/USDT 관련
        for s in markets.keys():
            if 'SOL' in s and 'USDT' in s:
                m = markets[s]
                print(f"\n{s}: type={m.get('type')} linear={m.get('linear')} swap={m.get('swap')} active={m.get('active')}")
    finally:
        await exchange.close()

asyncio.run(test())
