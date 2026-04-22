"""전체 시스템 통합 테스트"""
import asyncio
import sys, os
from dotenv import load_dotenv
load_dotenv()

async def test():
    from sol_data_v1 import DataCollector
    from sol_core_v1 import TradingCore
    from sol_executor_v1 import OrderExecutor

    print('[1] DataCollector 초기화...')
    dc = DataCollector(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
    await dc.initialize()
    print(f'   SOL 15m bars: {len(dc.df_sol)}')
    print(f'   BTC 1h bars: {len(dc.df_btc)}')
    print(f'   BTC EMA200 {dc.btc_ema200[-1]:.2f} | close {dc.btc_close[-1]:.2f} | regime {"BULL" if dc.btc_close[-1] > dc.btc_ema200[-1] else "BEAR"}')

    print('\n[2] OrderExecutor 초기화...')
    ex = OrderExecutor(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
    await ex.initialize()
    print(f'   Balance: ${ex.balance:,.2f}')

    print('\n[3] Signal 평가 테스트...')
    core = TradingCore()
    core.load_state()
    print(f'   Has position: {core.has_position} | Consec losses: {core.consec_losses}')

    bar = dc.get_last_closed_bar()
    if bar:
        ind = dc.get_indicators_at(bar['bar_idx'])
        btc_bull = dc.get_btc_regime_bull(bar['timestamp'])
        sig = core.evaluate(
            bar=bar, bar_idx=bar['bar_idx'], capital=ex.balance,
            btc_regime_bull=btc_bull,
            atr14=ind.get('atr14', 0), atr50=ind.get('atr50', 0),
            indicators=ind, mass_index=ind.get('mass', 0),
        )
        print(f'   Action: {sig.action}')
        print(f'   Reason: {sig.reason}')
        print(f'   Last bar close: ${bar["close"]:.3f}')
        print(f'   ADX: {ind.get("adx", 0):.1f} | RSI: {ind.get("rsi", 0):.1f}')
        print(f'   LR slope: {ind.get("lr_slope", 0):+.4f}')
        print(f'   EMA9: {ind.get("fast_ma", 0):.3f} | SMA400: {ind.get("slow_ma", 0):.3f}')
        print(f'   Mass Index: {ind.get("mass", 0):.2f}')
        print(f'   ATR14: {ind.get("atr14", 0):.4f} | ATR50: {ind.get("atr50", 0):.4f}')
        print(f'   BTC regime (Bull): {btc_bull}')

    await dc.close()
    await ex.close()
    print('\n[OK] 전체 시스템 정상 작동 ✅')

asyncio.run(test())
