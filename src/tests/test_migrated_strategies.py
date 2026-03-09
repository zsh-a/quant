import sys
import os
import logging
from loguru import logger

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.engine import TradingEngine
from src.core.backtest_broker import BacktestBroker
from src.core.data_stream import DBDataStream
from src.strategies.jsg_strategy import JSGStrategy
from src.strategies.rotation_strategy import RotationStrategy
from src.market_data.db import DB
import global_var

# Setup loguru
logger.remove()
logger.add(sys.stderr, level="INFO")

def test_jsg_migration():
    print("\n--- Testing JSG Strategy Migration ---")
    db_client = DB()
    # Test symbols from api/backtest.py example
    symbol = "sh.000300" 
    global_var.SYMBOLS = [symbol]
    
    # We use a short window for verification
    start_date = "2024-01-01"
    end_date = "2024-03-01"
    
    # DBDataStream needs a list of symbols
    # Ideally for JSG we need all stocks in index 000985 etc. 
    # But backtest.py seems to only pass the 'main' symbol to MultiMarketEnv 
    # and use DB for other calculations.
    stream = DBDataStream(db_client, [symbol], start_date=start_date, end_date=end_date)
    broker = BacktestBroker(db_client=db_client)
    strategy = JSGStrategy(db_client)
    
    engine = TradingEngine(strategy, broker, stream)
    engine.run()
    
    print("\nJSG Backtest Result:")
    print(broker.get_report())

def test_rotation_migration():
    print("\n--- Testing Rotation Strategy Migration ---")
    db_client = DB()
    symbol = "sh.000300"
    
    start_date = "2025-01-01"
    end_date = "2026-03-01"
    
    stream = DBDataStream(db_client, [symbol], start_date=start_date, end_date=end_date)
    broker = BacktestBroker(db_client=db_client)
    strategy = RotationStrategy(db_client)
    
    engine = TradingEngine(strategy, broker, stream)
    engine.run()
    
    print("\nRotation Backtest Result:")
    print(broker.get_report())

if __name__ == "__main__":
    # Test JSG
    try:
        test_jsg_migration()
    except Exception as e:
        logger.exception("JSG test failed")
        
    # Test Rotation
    try:
        test_rotation_migration()
    except Exception as e:
        logger.exception("Rotation test failed")
