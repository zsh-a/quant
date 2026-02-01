#!/usr/bin/env python3
"""
Quick test script to verify risk management API endpoints.
"""

import requests
import time

BASE_URL = "http://localhost:8000"

def test_risk_api():
    print("Testing Risk Management API Endpoints")
    print("="*60)
    
    # First, create a test session
    print("\n1. Creating test session...")
    response = requests.post(f"{BASE_URL}/session", json={
        "symbol": "000001.SZ",
        "strategy": "jsg",
        "mode": "backtest",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "params": {}
    })
    
    if response.status_code != 200:
        print(f"❌ Failed to create session: {response.text}")
        return
    
    session_id = response.json()["session_id"]
    print(f"✓ Session created: {session_id}")
    
    # Wait a bit for session to start
    time.sleep(2)
    
    # Test risk status endpoint
    print("\n2. Getting risk status...")
    response = requests.get(f"{BASE_URL}/session/{session_id}/risk")
    
    if response.status_code == 200:
        risk_data = response.json()
        print(f"✓ Risk status retrieved:")
        print(f"  Enabled: {risk_data.get('enabled', False)}")
        if risk_data.get('enabled'):
            metrics = risk_data.get('metrics', {})
            print(f"  Position count: {metrics.get('position_count', 0)}")
            print(f"  Total exposure: ${metrics.get('total_exposure', 0):,.2f}")
            print(f"  Current capital: ${metrics.get('current_capital', 0):,.2f}")
    else:
        print(f"❌ Failed to get risk status: {response.text}")
    
    # Test session status
    print("\n3. Getting session status...")
    response = requests.get(f"{BASE_URL}/session/{session_id}/status")
    
    if response.status_code == 200:
        status_data = response.json()
        print(f"✓ Session status: {status_data.get('status')}")
        print(f"  Progress: {status_data.get('progress', 0):.1f}%")
    else:
        print(f"❌ Failed to get session status: {response.text}")
    
    print("\n" + "="*60)
    print("Test completed!")

if __name__ == '__main__':
    try:
        test_risk_api()
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure it's running:")
        print("   uvicorn src.api.server:app --reload")
    except Exception as e:
        print(f"❌ Test failed: {e}")
