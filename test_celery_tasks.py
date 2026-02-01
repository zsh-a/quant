#!/usr/bin/env python3
"""
Test script for Celery task system.
Tests task submission, progress tracking, and result retrieval.
"""

import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def test_task_system():
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
        if response.status_code != 200:
            print(f"❌ Failed to submit task: {response.text}")
            return False
        
        task_data = response.json()
        task_id = task_data['task_id']
        print(f"✓ Task submitted: {task_id}")
        print(f"  Session: {task_data['session_id']}")
        print(f"  Status: {task_data['status']}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("   Make sure the server is running: uvicorn src.api.server:app --reload")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: Monitor task progress
    print(f"\n2. Monitoring task progress...")
    max_wait = 120  # 2 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{BASE_URL}/tasks/backtest/{task_id}")
            if response.status_code != 200:
                print(f"❌ Failed to get task status: {response.text}")
                break
            
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
                return True
            
            elif status == 'FAILURE':
                print(f"\n❌ Task failed!")
                print(f"  Error: {status_data.get('error', 'Unknown error')}")
                return False
            
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Error checking status: {e}")
            break
    
    print(f"\n⚠️  Task did not complete within {max_wait} seconds")
    return False


def test_worker_status():
    """Test worker status endpoint"""
    print("\n3. Checking worker status...")
    
    try:
        response = requests.get(f"{BASE_URL}/tasks/workers")
        if response.status_code != 200:
            print(f"❌ Failed to get workers: {response.text}")
            return False
        
        data = response.json()
        workers = data.get('workers', [])
        
        if not workers:
            print("⚠️  No workers online!")
            print("   Start a worker with: ./start_worker.sh")
            return False
        
        print(f"✓ Found {len(workers)} worker(s):")
        for worker in workers:
            print(f"  - {worker['name']}")
            print(f"    Concurrency: {worker['concurrency']}")
            print(f"    Active tasks: {worker['active_tasks']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("\n" + "#"*60)
    print("# Celery Task System Test Suite")
    print("#"*60)
    
    # Check worker status first
    if not test_worker_status():
        print("\n⚠️  Please start a Celery worker before running tasks:")
        print("   ./start_worker.sh")
        print("\nOr start Redis if not running:")
        print("   redis-server")
        return 1
    
    # Run task test
    success = test_task_system()
    
    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED!")
        return 0
    else:
        print("❌ TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
