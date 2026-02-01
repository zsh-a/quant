#!/usr/bin/env python3
"""
End-to-end integration test for the quantitative trading platform.
Tests the complete flow from data stream to API to frontend.
"""

import sys
import os
import time
import requests
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logging_config import get_logger

logger = get_logger("e2e_test")

# Configuration
API_BASE = "http://localhost:8000"
TEST_TIMEOUT = 60  # seconds

def check_api_server():
    """Check if API server is running"""
    logger.info("Checking API server status...")
    try:
        response = requests.get(f"{API_BASE}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ API server is up: {data}")
            return True
        else:
            logger.error(f"API server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("✗ API server is not running")
        logger.info("Please start the server with: uvicorn src.api.server:app --reload")
        return False
    except Exception as e:
        logger.error(f"Error checking API server: {e}")
        return False

def test_get_strategies():
    """Test getting available strategies"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Get Available Strategies")
    logger.info("="*60)
    
    try:
        response = requests.get(f"{API_BASE}/strategies", timeout=5)
        response.raise_for_status()
        
        strategies = response.json()
        logger.info(f"Found {len(strategies)} strategies:")
        for strategy in strategies:
            logger.info(f"  - {strategy['name']}: {strategy['label']}")
            logger.info(f"    Parameters: {len(strategy.get('params', []))}")
        
        assert len(strategies) > 0, "No strategies found"
        logger.info("✓ Strategy list retrieved successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to get strategies: {e}")
        return False

def test_create_backtest_session():
    """Test creating a backtest session"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Create Backtest Session")
    logger.info("="*60)
    
    try:
        # Create a short backtest session
        payload = {
            "strategy": "jsg",
            "symbol": "sh.600000",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",  # Just 1 month for quick test
            "mode": "backtest",
            "params": {}
        }
        
        logger.info(f"Creating session: {payload['strategy']} on {payload['symbol']}")
        response = requests.post(f"{API_BASE}/session/run", json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        session_id = data.get('session_id')
        
        assert session_id, "No session_id returned"
        logger.info(f"✓ Session created: {session_id}")
        
        return session_id
        
    except Exception as e:
        logger.error(f"✗ Failed to create session: {e}")
        return None

def test_session_status_incremental(session_id):
    """Test incremental session status updates"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Incremental Session Status Updates")
    logger.info("="*60)
    
    try:
        # First request - full data
        logger.info("Request 1: Full data (no since parameter)")
        response1 = requests.get(f"{API_BASE}/session/{session_id}/status", timeout=10)
        response1.raise_for_status()
        data1 = response1.json()
        
        size1 = len(json.dumps(data1))
        equity_count1 = len(data1.get('equity_history', []))
        trades_count1 = len(data1.get('trades', []))
        
        logger.info(f"  Response size: {size1} bytes")
        logger.info(f"  Equity points: {equity_count1}")
        logger.info(f"  Trades: {trades_count1}")
        logger.info(f"  Status: {data1.get('status')}")
        logger.info(f"  Progress: {data1.get('progress', 0):.1f}%")
        
        # Wait a bit for more data
        time.sleep(2)
        
        # Second request - incremental (with since parameter)
        if equity_count1 > 0:
            last_timestamp = data1['equity_history'][-1]['timestamp']
            logger.info(f"\nRequest 2: Incremental data (since={last_timestamp})")
            
            response2 = requests.get(
                f"{API_BASE}/session/{session_id}/status",
                params={'since': last_timestamp},
                timeout=10
            )
            response2.raise_for_status()
            data2 = response2.json()
            
            size2 = len(json.dumps(data2))
            equity_count2 = len(data2.get('equity_history', []))
            trades_count2 = len(data2.get('trades', []))
            
            logger.info(f"  Response size: {size2} bytes")
            logger.info(f"  Equity points: {equity_count2}")
            logger.info(f"  Trades: {trades_count2}")
            
            # Calculate bandwidth savings
            if size1 > 0:
                savings = ((size1 - size2) / size1) * 100
                logger.info(f"\n  Bandwidth savings: {savings:.1f}% ({size1} → {size2} bytes)")
                
                if savings > 50:
                    logger.info("✓ Incremental updates working efficiently")
                else:
                    logger.warning(f"⚠ Bandwidth savings only {savings:.1f}% (expected >50%)")
            
        else:
            logger.info("No equity data yet, skipping incremental test")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to test incremental updates: {e}")
        return False

def test_wait_for_completion(session_id, timeout=TEST_TIMEOUT):
    """Wait for session to complete"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Wait for Session Completion")
    logger.info("="*60)
    
    start_time = time.time()
    last_progress = 0
    
    try:
        while time.time() - start_time < timeout:
            response = requests.get(f"{API_BASE}/session/{session_id}/status", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            status = data.get('status')
            progress = data.get('progress', 0)
            
            if progress != last_progress:
                logger.info(f"Progress: {progress:.1f}% - Status: {status}")
                last_progress = progress
            
            if status in ['completed', 'failed', 'stopped']:
                logger.info(f"\n✓ Session {status}")
                
                # Show final metrics
                equity_count = len(data.get('equity_history', []))
                trades_count = len(data.get('trades', []))
                logger.info(f"  Total equity points: {equity_count}")
                logger.info(f"  Total trades: {trades_count}")
                
                if data.get('error'):
                    logger.error(f"  Error: {data['error']}")
                    return False
                
                return status == 'completed'
            
            time.sleep(1)
        
        logger.error(f"✗ Timeout after {timeout}s")
        return False
        
    except Exception as e:
        logger.error(f"✗ Error waiting for completion: {e}")
        return False

def test_get_all_sessions():
    """Test getting all sessions"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Get All Sessions")
    logger.info("="*60)
    
    try:
        response = requests.get(f"{API_BASE}/sessions", timeout=10)
        response.raise_for_status()
        
        sessions = response.json()
        logger.info(f"Found {len(sessions)} session(s):")
        for session in sessions[:5]:  # Show first 5
            logger.info(f"  - {session.get('session_id', 'N/A')[:8]}... "
                       f"({session.get('strategy')}/{session.get('symbol')}) "
                       f"- {session.get('status')}")
        
        logger.info("✓ Session list retrieved successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to get sessions: {e}")
        return False

def test_log_files():
    """Test that log files are being created"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Log File Creation")
    logger.info("="*60)
    
    log_dir = Path("logs")
    
    if not log_dir.exists():
        logger.warning("⚠ Logs directory does not exist")
        return False
    
    log_files = list(log_dir.glob("quant_*.log"))
    
    if log_files:
        logger.info(f"Found {len(log_files)} log file(s):")
        for log_file in log_files:
            size_kb = log_file.stat().st_size / 1024
            logger.info(f"  - {log_file.name} ({size_kb:.2f} KB)")
        logger.info("✓ Log files created successfully")
        return True
    else:
        logger.warning("⚠ No log files found")
        return False

def main():
    """Run all integration tests"""
    logger.info("\n" + "#"*60)
    logger.info("# End-to-End Integration Test Suite")
    logger.info("#"*60)
    
    results = {}
    
    # Check if server is running
    if not check_api_server():
        logger.error("\n❌ API server is not running. Please start it first:")
        logger.error("   cd /home/zs/workspace/exp/quent")
        logger.error("   uvicorn src.api.server:app --reload")
        return 1
    
    # Test 1: Get strategies
    results['strategies'] = test_get_strategies()
    
    # Test 2: Create backtest session
    session_id = test_create_backtest_session()
    results['create_session'] = session_id is not None
    
    if session_id:
        # Test 3: Incremental updates
        results['incremental_updates'] = test_session_status_incremental(session_id)
        
        # Test 4: Wait for completion
        results['completion'] = test_wait_for_completion(session_id)
    
    # Test 5: Get all sessions
    results['get_sessions'] = test_get_all_sessions()
    
    # Test 6: Log files
    results['log_files'] = test_log_files()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    exit(main())
