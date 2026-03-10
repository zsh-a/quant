#!/usr/bin/env python3
"""
Test script for Celery task system.
Tests task submission, progress tracking, and result retrieval.
"""

import os
import requests
import time
import sys
import pytest

BASE_URL = "http://localhost:8000"

def _skip_if_not_integration():
    if os.getenv("RUN_INTEGRATION") != "1" and os.getenv("PYTEST_CURRENT_TEST"):
        pytest.skip("Integration tests disabled (set RUN_INTEGRATION=1 to enable).")


def _require_api():
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=2)
    except requests.exceptions.RequestException:
        pytest.skip("API server not reachable")
    if response.status_code != 200:
        pytest.skip("API server not healthy")


def _require_workers():
    response = requests.get(f"{BASE_URL}/tasks/workers")
    if response.status_code != 200:
        pytest.skip("Celery worker status not available")
    data = response.json()
    if not data.get('workers'):
        pytest.skip("No Celery workers online")


def test_task_system():
    _skip_if_not_integration()
    _require_api()
    _require_workers()
    print("="*60)
    print("Testing Celery Task System")
    print("="*60)
    
    # Test 1: Submit a backtest task
    print("\n1. Submitting backtest task...")
    task_request = {
        "session_id": "test_celery_001",
        "symbol": "000001.SZ",
        "strategy": "jsg",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "params": {},
        "initial_cash": 1000000,
        "commission": 0.0001,
        "enable_risk_management": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/tasks/backtest", json=task_request)
        assert response.status_code == 200, f"Failed to submit task: {response.text}"
        
        task_data = response.json()
        task_id = task_data['task_id']
        print(f"✓ Task submitted: {task_id}")
        print(f"  Session: {task_data['session_id']}")
        print(f"  Status: {task_data['status']}")
        
    except Exception as e:
        pytest.fail(f"Error submitting task: {e}")
    
    # Test 2: Monitor task progress
    print(f"\n2. Monitoring task progress...")
    max_wait = 120  # 2 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{BASE_URL}/tasks/backtest/{task_id}")
            assert response.status_code == 200, f"Failed to get task status: {response.text}"
            
            status_data = response.json()
            status = status_data['status']
            progress = status_data.get('progress', 0)
            message = status_data.get('message', '')
            
            print(f"  Status: {status} | Progress: {progress:.1f}% | {message}")
            
            if status == 'SUCCESS':
                print(f"\n✓ Task completed successfully!")
                result = status_data.get('result', {})
                print(f"  Final equity: ${result.get('final_equity', 0):,.2f}")
                print(f"  Total trades: {result.get('total_trades', 0)}")
                return None
            
            elif status == 'FAILURE':
                print(f"\n❌ Task failed!")
                print(f"  Error: {status_data.get('error', 'Unknown error')}")
                pytest.fail("Task failed")
            
            time.sleep(2)
            
        except Exception as e:
            pytest.fail(f"Error checking status: {e}")
    
    pytest.fail(f"Task did not complete within {max_wait} seconds")


def test_worker_status():
    """Test worker status endpoint"""
    _skip_if_not_integration()
    _require_api()
    print("\n3. Checking worker status...")
    
    try:
        response = requests.get(f"{BASE_URL}/tasks/workers")
        assert response.status_code == 200, f"Failed to get workers: {response.text}"
        
        data = response.json()
        workers = data.get('workers', [])
        
        if not workers:
            pytest.fail("No workers online. Start one with: ./start_worker.sh")
        
        print(f"✓ Found {len(workers)} worker(s):")
        for worker in workers:
            print(f"  - {worker['name']}")
            print(f"    Concurrency: {worker['concurrency']}")
            print(f"    Active tasks: {worker['active_tasks']}")
        
        return None
        
    except Exception as e:
        pytest.fail(f"Error: {e}")


def main():
    print("\n" + "#"*60)
    print("# Celery Task System Test Suite")
    print("#"*60)
    
    # Check worker status first
    if not _run_test("worker_status", test_worker_status):
        print("\n⚠️  Please start a Celery worker before running tasks:")
        print("   ./start_worker.sh")
        print("\nOr start Redis if not running:")
        print("   redis-server")
        return 1
    
    # Run task test
    success = _run_test("task_system", test_task_system)
    
    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED!")
        return 0
    else:
        print("❌ TESTS FAILED")
        return 1


def _run_test(name, fn):
    try:
        fn()
        return True
    except AssertionError as exc:
        print(f"❌ {name} assertion failed: {exc}")
        return False
    except Exception as exc:
        print(f"❌ {name} error: {exc}")
        return False


if __name__ == '__main__':
    sys.exit(main())
