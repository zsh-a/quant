#!/usr/bin/env python3
"""
Test script for monitoring and alerting system.
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


def test_health_check():
    """Test health check endpoint"""
    _skip_if_not_integration()
    _require_api()
    print("\n" + "="*60)
    print("1. Testing Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/monitoring/health")
        assert response.status_code == 200, f"Health check failed: {response.text}"
        
        data = response.json()
        status = data.get('status', 'unknown')
        
        print(f"✓ Overall Status: {status.upper()}")
        print(f"  Timestamp: {data.get('timestamp')}")
        
        # Show individual checks
        checks = data.get('checks', [])
        for check in checks:
            component = check.get('component')
            check_status = check.get('status')
            message = check.get('message', '')
            
            icon = "✓" if check_status == "healthy" else "⚠️" if check_status == "degraded" else "❌"
            print(f"  {icon} {component}: {message}")
        
        assert status in ['healthy', 'degraded'], f"Unexpected health status: {status}"
        return None
        
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not reachable")
    except Exception as e:
        pytest.fail(f"Error: {e}")


def test_metrics():
    """Test Prometheus metrics endpoint"""
    _skip_if_not_integration()
    _require_api()
    print("\n" + "="*60)
    print("2. Testing Prometheus Metrics")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/monitoring/metrics")
        assert response.status_code == 200, f"Metrics endpoint failed: {response.text}"
        
        metrics = response.text
        
        # Count metrics
        metric_lines = [line for line in metrics.split('\n') if line and not line.startswith('#')]
        
        print(f"✓ Metrics endpoint working")
        print(f"  Total metrics: {len(metric_lines)}")
        
        # Show sample metrics
        print("\n  Sample metrics:")
        for line in metric_lines[:5]:
            print(f"    {line}")
        
        assert len(metric_lines) > 0, "Expected metrics output"
        return None
        
    except Exception as e:
        pytest.fail(f"Error: {e}")


def test_system_status():
    """Test system status endpoint"""
    _skip_if_not_integration()
    _require_api()
    print("\n" + "="*60)
    print("3. Testing System Status")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/monitoring/status")
        assert response.status_code == 200, f"Status endpoint failed: {response.text}"
        
        data = response.json()
        
        print(f"✓ System Status: {data.get('status', 'unknown').upper()}")
        
        # Show system metrics
        system = data.get('system', {})
        
        cpu = system.get('cpu', {})
        print(f"\n  CPU:")
        print(f"    Usage: {cpu.get('percent', 0):.1f}%")
        print(f"    Cores: {cpu.get('count', 0)}")
        
        memory = system.get('memory', {})
        print(f"\n  Memory:")
        print(f"    Usage: {memory.get('percent', 0):.1f}%")
        print(f"    Used: {memory.get('used_mb', 0):.0f} MB")
        print(f"    Total: {memory.get('total_mb', 0):.0f} MB")
        
        disk = system.get('disk', {})
        print(f"\n  Disk:")
        print(f"    Usage: {disk.get('percent', 0):.1f}%")
        print(f"    Used: {disk.get('used_gb', 0):.1f} GB")
        print(f"    Total: {disk.get('total_gb', 0):.1f} GB")
        
        assert 'system' in data, "Missing system field"
        return None
        
    except Exception as e:
        pytest.fail(f"Error: {e}")


def test_alert_notification():
    """Test alert notification"""
    _skip_if_not_integration()
    _require_api()
    print("\n" + "="*60)
    print("4. Testing Alert Notification")
    print("="*60)
    
    try:
        response = requests.post(
            f"{BASE_URL}/monitoring/alert/test",
            params={
                "title": "Test Alert from Monitoring System",
                "message": "This is a test alert to verify Feishu integration",
                "severity": "info"
            }
        )
        
        if response.status_code != 200:
            pytest.skip(f"Alert test unavailable: {response.text}")
        
        data = response.json()
        
        if data.get('status') == 'success':
            print("✓ Test alert sent successfully")
            print("  Check your Feishu channel for the notification")
            return None
        else:
            pytest.fail(f"Alert sent but may have issues: {data.get('error', 'Unknown')}")
        
    except Exception as e:
        pytest.fail(f"Error: {e}")


def main():
    print("\n" + "#"*60)
    print("# Monitoring & Alerting System Test Suite")
    print("#"*60)
    
    results = []
    
    # Run tests
    results.append(("Health Check", _run_test("Health Check", test_health_check)))
    results.append(("Prometheus Metrics", _run_test("Prometheus Metrics", test_metrics)))
    results.append(("System Status", _run_test("System Status", test_system_status)))
    results.append(("Alert Notification", _run_test("Alert Notification", test_alert_notification)))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        icon = "✅" if result else "❌"
        print(f"{icon} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
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
