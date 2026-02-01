#!/usr/bin/env python3
"""
Test script for monitoring and alerting system.
"""

import requests
import time
import sys

BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("1. Testing Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/monitoring/health")
        if response.status_code != 200:
            print(f"❌ Health check failed: {response.text}")
            return False
        
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
        
        return status in ['healthy', 'degraded']
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("   Make sure the server is running: uvicorn src.api.server:app --reload")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_metrics():
    """Test Prometheus metrics endpoint"""
    print("\n" + "="*60)
    print("2. Testing Prometheus Metrics")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/monitoring/metrics")
        if response.status_code != 200:
            print(f"❌ Metrics endpoint failed: {response.text}")
            return False
        
        metrics = response.text
        
        # Count metrics
        metric_lines = [line for line in metrics.split('\n') if line and not line.startswith('#')]
        
        print(f"✓ Metrics endpoint working")
        print(f"  Total metrics: {len(metric_lines)}")
        
        # Show sample metrics
        print("\n  Sample metrics:")
        for line in metric_lines[:5]:
            print(f"    {line}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_system_status():
    """Test system status endpoint"""
    print("\n" + "="*60)
    print("3. Testing System Status")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/monitoring/status")
        if response.status_code != 200:
            print(f"❌ Status endpoint failed: {response.text}")
            return False
        
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
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_alert_notification():
    """Test alert notification"""
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
            print(f"❌ Alert test failed: {response.text}")
            return False
        
        data = response.json()
        
        if data.get('status') == 'success':
            print("✓ Test alert sent successfully")
            print("  Check your Feishu channel for the notification")
            return True
        else:
            print(f"⚠️  Alert sent but may have issues: {data.get('error', 'Unknown')}")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("\n" + "#"*60)
    print("# Monitoring & Alerting System Test Suite")
    print("#"*60)
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health_check()))
    results.append(("Prometheus Metrics", test_metrics()))
    results.append(("System Status", test_system_status()))
    results.append(("Alert Notification", test_alert_notification()))
    
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


if __name__ == '__main__':
    sys.exit(main())
