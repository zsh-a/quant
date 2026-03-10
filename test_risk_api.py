#!/usr/bin/env python3
"""
Quick test script to verify risk management API endpoints.
"""

import os
import requests
import time
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


def test_risk_api():
    _skip_if_not_integration()
    _require_api()
    print("Testing Risk Management API Endpoints")
    print("="*60)
    
    # First, create a test session
    print("\n1. Creating test session...")
    response = requests.post(f"{BASE_URL}/session/run", json={
        "symbol": "000001.SZ",
        "strategy": "jsg",
        "mode": "backtest",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "params": {}
    })
    
    assert response.status_code == 200, f"Failed to create session: {response.text}"
    
    session_id = response.json()["session_id"]
    print(f"✓ Session created: {session_id}")
    
    # Wait a bit for session to start
    time.sleep(2)
    
    # Test risk status endpoint
    print("\n2. Getting risk status...")
    response = requests.get(f"{BASE_URL}/session/{session_id}/risk")
    
    assert response.status_code == 200, f"Failed to get risk status: {response.text}"
    risk_data = response.json()
    print(f"✓ Risk status retrieved:")
    print(f"  Enabled: {risk_data.get('enabled', False)}")
    if risk_data.get('enabled'):
        metrics = risk_data.get('metrics', {})
        print(f"  Position count: {metrics.get('position_count', 0)}")
        print(f"  Total exposure: ${metrics.get('total_exposure', 0):,.2f}")
        print(f"  Current capital: ${metrics.get('current_capital', 0):,.2f}")
    
    # Test session status
    print("\n3. Getting session status...")
    response = requests.get(f"{BASE_URL}/session/{session_id}/status")
    
    assert response.status_code == 200, f"Failed to get session status: {response.text}"
    status_data = response.json()
    print(f"✓ Session status: {status_data.get('status')}")
    print(f"  Progress: {status_data.get('progress', 0):.1f}%")
    
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
