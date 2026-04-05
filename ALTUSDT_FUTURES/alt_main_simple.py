#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALT/USDT Simplified Futures Trading Program
MACD Signal Line Based Trading
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Module import test
try:
    import ccxt
    import pandas as pd
    import numpy as np
    from alt_core_simple import SimpleTradingBot, TradingConfig
    print("All required modules imported successfully")
except ImportError as e:
    print(f"Module import failed: {e}")
    print("Run: pip install ccxt pandas numpy python-dotenv")
    sys.exit(1)

def setup_logging() -> logging.Logger:
    """Setup logging"""
    # Create log directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Log file name (12 hour intervals)
    current_time = datetime.now()
    hour_suffix = "AM" if current_time.hour < 12 else "PM"
    log_file = log_dir / f"alt_trading_{current_time.strftime('%Y%m%d')}_{hour_suffix}.log"
    
    # Setup logger
    logger = logging.getLogger("ALTTrading")
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Format setting
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler
    if not logger.handlers:
        logger.addHandler(file_handler)
    
    return logger

async def main():
    """Main function"""
    # Setup logging
    logger = setup_logging()
    
    # Create config
    config = TradingConfig()
    
    # API key settings
    config.binance_api_key = os.getenv("BINANCE_API_KEY", "")
    config.binance_api_secret = os.getenv("BINANCE_API_SECRET", "")
    
    if not config.binance_api_key or not config.binance_api_secret:
        logger.error("Binance API keys not configured")
        print("Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        return
    
    # Header output
    print("=" * 100)
    print("ALT/USDT Futures Trading System (MACD Signal Line Based)")
    print("=" * 100)
    print(f"Symbol: {config.symbol}")
    print(f"Timeframe: {config.timeframe}")
    print(f"Position size: {config.position_size}% of account balance")
    print(f"Leverage: {config.leverage}x")
    print(f"Base stop loss: -{config.base_stop_loss}%")
    print(f"Monitoring interval: {config.monitoring_interval}s")
    print("Strategy: MACD signal line rising/falling for 3+ minutes")
    print("=" * 100)
    
    # Create trading bot
    bot = SimpleTradingBot(config)
    
    # Initialize
    if not await bot.initialize():
        logger.error("System initialization failed")
        return
    
    # Run bot
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Program terminated")
        print("\nTerminating program...")
    except Exception as e:
        logger.error(f"Execution error: {e}")
        print(f"Error: {e}")
    finally:
        if bot.db:
            bot.db.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTerminating program...")
    except Exception as e:
        print(f"Error: {e}")