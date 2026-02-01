#!/usr/bin/env python3
"""
Test script for RealtimeDataStream
Tests trading hours check, error handling, and data fetching
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.data_stream import RealtimeDataStream
from loguru import logger

def test_trading_hours_check():
    """Test trading hours validation"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Trading Hours Check")
    logger.info("="*60)
    
    stream = RealtimeDataStream(
        symbols=['sh.510880'],
        interval_seconds=5,
        data_source='mock',  # Use mock to avoid API calls
        enable_trading_hours_check=True
    )
    
    # Check current status
    is_trading = stream._is_trading_hours()
    now = datetime.now()
    
    logger.info(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %A')}")
    logger.info(f"Is trading hours: {is_trading}")
    
    if is_trading:
        logger.info("✓ Currently in trading hours")
    else:
        logger.info("✗ Currently outside trading hours")
    
    return is_trading

def test_mock_data_fetch():
    """Test mock data generation"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Mock Data Fetch")
    logger.info("="*60)
    
    stream = RealtimeDataStream(
        symbols=['sh.510880', 'sh.510300', 'sh.510500'],
        interval_seconds=1,
        data_source='mock',
        enable_trading_hours_check=False  # Disable for testing
    )
    
    logger.info("Fetching 3 bars of mock data...")
    
    for i in range(3):
        bars = stream.next_bar()
        logger.info(f"\nBar {i+1}:")
        for symbol, bar in bars.items():
            logger.info(f"  {symbol}: O={bar.open:.2f}, H={bar.high:.2f}, "
                       f"L={bar.low:.2f}, C={bar.close:.2f}, V={bar.volume:,}")
    
    logger.info("\n✓ Mock data fetch successful")

def test_error_recovery():
    """Test error handling and retry logic"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Error Recovery")
    logger.info("="*60)
    
    # This will fail if akshare is not available or symbols don't exist
    stream = RealtimeDataStream(
        symbols=['INVALID_SYMBOL_123'],
        interval_seconds=1,
        data_source='akshare',
        enable_trading_hours_check=False
    )
    
    logger.info("Attempting to fetch invalid symbol (should retry and fail gracefully)...")
    bars = stream.next_bar()
    
    if not bars:
        logger.info("✓ Error handled gracefully, returned empty dict")
    else:
        logger.warning("Unexpected: got data for invalid symbol")
    
    logger.info(f"Consecutive errors: {stream.consecutive_errors}")

def test_data_source_fallback():
    """Test data source initialization and fallback"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Data Source Fallback")
    logger.info("="*60)
    
    # Test with invalid data source
    stream = RealtimeDataStream(
        symbols=['sh.510880'],
        interval_seconds=1,
        data_source='invalid_source',
        enable_trading_hours_check=False
    )
    
    logger.info(f"Requested source: 'invalid_source'")
    logger.info(f"Actual source: '{stream.data_source}'")
    
    if stream.data_source == 'mock':
        logger.info("✓ Correctly fell back to mock data source")
    else:
        logger.warning(f"Unexpected fallback to: {stream.data_source}")

def main():
    """Run all tests"""
    logger.info("\n" + "#"*60)
    logger.info("# RealtimeDataStream Test Suite")
    logger.info("#"*60)
    
    try:
        # Test 1: Trading hours check
        test_trading_hours_check()
        
        # Test 2: Mock data fetch
        test_mock_data_fetch()
        
        # Test 3: Data source fallback
        test_data_source_fallback()
        
        # Test 4: Error recovery (may take a few seconds due to retries)
        test_error_recovery()
        
        logger.info("\n" + "="*60)
        logger.info("ALL TESTS COMPLETED")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
