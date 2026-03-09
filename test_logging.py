#!/usr/bin/env python3
"""
Test script for unified logging system.
Tests log levels, rotation, structured logging, and file output.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logging_config import (
    setup_logging, 
    get_logger, 
    log_performance,
    log_trade,
    log_error_with_context
)

def test_basic_logging():
    """Test basic logging at different levels"""
    logger = get_logger("test_basic")
    
    print("\n" + "="*60)
    print("TEST: Basic Logging Levels")
    print("="*60)
    
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    print("✓ Basic logging test completed")

def test_structured_logging():
    """Test structured logging helpers"""
    print("\n" + "="*60)
    print("TEST: Structured Logging")
    print("="*60)
    
    # Performance logging
    log_performance("backtest_execution", 12.345, 
                   symbols=100, 
                   bars_processed=24200,
                   memory_mb=450.5)
    
    # Trade logging
    log_trade("BUY", "sh.600000", 1000, 15.23,
             commission=1.52,
             slippage=0.01,
             strategy="JSG")
    
    log_trade("SELL", "sh.510880", 500, 3.45,
             commission=0.17,
             reason="stop_loss")
    
    print("✓ Structured logging test completed")

def test_error_logging():
    """Test error logging with context"""
    print("\n" + "="*60)
    print("TEST: Error Logging with Context")
    print("="*60)
    
    try:
        # Simulate an error
        result = 1 / 0
    except Exception as e:
        log_error_with_context(e, {
            'operation': 'calculate_returns',
            'symbol': 'sh.600000',
            'timestamp': '2024-01-01 10:30:00',
            'portfolio_value': 1000000.0
        })
    
    print("✓ Error logging test completed")

def test_logger_binding():
    """Test logger name binding"""
    print("\n" + "="*60)
    print("TEST: Logger Name Binding")
    print("="*60)
    
    strategy_logger = get_logger("JSGStrategy")
    broker_logger = get_logger("BacktestBroker")
    engine_logger = get_logger("TradingEngine")
    
    strategy_logger.info("Strategy initialized with parameters")
    broker_logger.info("Broker ready, initial cash: $1,000,000")
    engine_logger.info("Engine starting backtest run")
    
    print("✓ Logger binding test completed")

def test_log_file_creation():
    """Test that log files are created"""
    print("\n" + "="*60)
    print("TEST: Log File Creation")
    print("="*60)
    
    log_dir = Path("logs")
    
    if log_dir.exists():
        log_files = list(log_dir.glob("quant*.log"))
        print(f"Found {len(log_files)} log file(s):")
        for log_file in log_files:
            size_kb = log_file.stat().st_size / 1024
            print(f"  - {log_file.name} ({size_kb:.2f} KB)")
        
        if log_files:
            print("✓ Log files created successfully")
        else:
            print("⚠ No log files found (file logging may be disabled)")
    else:
        print("⚠ Logs directory does not exist")

def test_performance_logging():
    """Test logging performance with many messages"""
    print("\n" + "="*60)
    print("TEST: Logging Performance")
    print("="*60)
    
    logger = get_logger("performance_test")
    
    num_messages = 1000
    start_time = time.time()
    
    for i in range(num_messages):
        logger.info(f"Performance test message {i}", 
                   iteration=i,
                   timestamp=time.time())
    
    duration = time.time() - start_time
    rate = num_messages / duration
    
    print(f"Logged {num_messages} messages in {duration:.3f}s")
    print(f"Rate: {rate:.0f} messages/second")
    print("✓ Performance test completed")

def main():
    """Run all logging tests"""
    print("\n" + "#"*60)
    print("# Unified Logging System Test Suite")
    print("#"*60)
    
    try:
        # Test 1: Basic logging
        test_basic_logging()
        
        # Test 2: Structured logging
        test_structured_logging()
        
        # Test 3: Error logging
        test_error_logging()
        
        # Test 4: Logger binding
        test_logger_binding()
        
        # Test 5: Log file creation
        test_log_file_creation()
        
        # Test 6: Performance
        test_performance_logging()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
