#!/usr/bin/env python3
"""
Risk management system test script.
Tests position limits, stop-loss, take-profit, and risk metrics.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.risk_manager import RiskManager, RiskLimits


def test_position_limits():
    """Test position limit checks"""
    print(f"\n{'='*60}")
    print(f"TEST: Position Limits")
    print(f"{'='*60}")
    
    rm = RiskManager(initial_capital=1000000)
    
    # Test 1: Normal position (should pass)
    allowed, reason = rm.check_position_limit('AAPL', 1000, 100)
    assert allowed, "Normal position should be allowed"
    print(f"✓ Normal position (10% of capital): PASS")
    
    # Test 2: Oversized position (should fail)
    allowed, reason = rm.check_position_limit('AAPL', 2000, 100)
    assert not allowed, "Oversized position should be rejected"
    print(f"✓ Oversized position (20% of capital): REJECTED - {reason}")
    
    # Test 3: Total exposure limit
    rm.add_position('AAPL', 3000, 100)  # 30% exposure
    rm.add_position('GOOGL', 3000, 100)  # 30% exposure
    rm.add_position('MSFT', 3000, 100)  # 30% exposure
    # Total: 90% exposure
    
    # Try to add another position that would exceed 95%
    allowed, reason = rm.check_position_limit('TSLA', 1000, 100)  # Would be 100% total
    assert not allowed, "Should exceed total exposure limit"
    print(f"✓ Total exposure limit (would be 100%): REJECTED - {reason}")
    
    return None


def test_stop_loss():
    """Test stop loss triggers"""
    print(f"\n{'='*60}")
    print(f"TEST: Stop Loss")
    print(f"{'='*60}")
    
    rm = RiskManager(initial_capital=1000000)
    rm.add_position('AAPL', 1000, 100)  # Entry at $100
    
    # Test 1: Price drop within limit (should not trigger)
    rm.update_position('AAPL', 96)  # 4% loss
    triggered, reason = rm.check_stop_loss('AAPL', 96)
    assert not triggered, "Stop loss should not trigger at 4% loss"
    print(f"✓ 4% loss: No stop loss")
    
    # Test 2: Price drop exceeds limit (should trigger)
    rm.update_position('AAPL', 94)  # 6% loss
    triggered, reason = rm.check_stop_loss('AAPL', 94)
    assert triggered, "Stop loss should trigger at 6% loss"
    print(f"✓ 6% loss: Stop loss triggered - {reason}")
    
    return None


def test_take_profit():
    """Test take profit triggers"""
    print(f"\n{'='*60}")
    print(f"TEST: Take Profit")
    print(f"{'='*60}")
    
    rm = RiskManager(initial_capital=1000000)
    rm.add_position('AAPL', 1000, 100)  # Entry at $100
    
    # Test 1: Price gain within target (should not trigger)
    rm.update_position('AAPL', 110)  # 10% gain
    triggered, reason = rm.check_take_profit('AAPL', 110)
    assert not triggered, "Take profit should not trigger at 10% gain"
    print(f"✓ 10% gain: No take profit")
    
    # Test 2: Price gain exceeds target (should trigger)
    rm.update_position('AAPL', 120)  # 20% gain
    triggered, reason = rm.check_take_profit('AAPL', 120)
    assert triggered, "Take profit should trigger at 20% gain"
    print(f"✓ 20% gain: Take profit triggered - {reason}")
    
    return None


def test_daily_loss_limit():
    """Test daily loss limit"""
    print(f"\n{'='*60}")
    print(f"TEST: Daily Loss Limit")
    print(f"{'='*60}")
    
    rm = RiskManager(initial_capital=1000000)
    
    # Simulate a large losing day - need to lose more than 10%
    rm.add_position('AAPL', 11000, 100)  # $1.1M position (110% of capital - will be rejected)
    # Use smaller position
    rm.add_position('AAPL', 5000, 100)  # $500K position
    rm.update_position('AAPL', 80)  # 20% loss on position = $100K loss = 10% of capital
    result = rm.close_position('AAPL', 80)
    
    print(f"Position closed: P&L = ${result['pnl']:.2f} ({result['pnl_pct']:.2%})")
    print(f"Current capital: ${rm.current_capital:.2f}")
    print(f"Daily start: ${rm.daily_start_capital:.2f}")
    print(f"Loss: ${rm.daily_start_capital - rm.current_capital:.2f}")
    
    # Check daily loss limit
    exceeded, reason = rm.check_daily_loss_limit()
    assert exceeded, "Daily loss limit should be exceeded"
    print(f"✓ Daily loss limit exceeded: {reason}")
    
    return None


def test_max_drawdown():
    """Test maximum drawdown"""
    print(f"\n{'='*60}")
    print(f"TEST: Maximum Drawdown")
    print(f"{'='*60}")
    
    rm = RiskManager(initial_capital=1000000)
    
    # Simulate profitable trade (new peak)
    rm.add_position('AAPL', 1000, 100)
    rm.close_position('AAPL', 110)  # 10% profit
    print(f"After profit: capital=${rm.current_capital:.2f}, peak=${rm.peak_capital:.2f}")
    
    # Simulate large loss
    rm.add_position('GOOGL', 2000, 100)
    rm.close_position('GOOGL', 90)  # 10% loss on larger position
    print(f"After loss: capital=${rm.current_capital:.2f}, peak=${rm.peak_capital:.2f}")
    
    # Check drawdown
    exceeded, reason = rm.check_max_drawdown()
    print(f"Drawdown check: exceeded={exceeded}")
    if exceeded:
        print(f"  Reason: {reason}")
    
    return None


def test_risk_metrics():
    """Test risk metrics calculation"""
    print(f"\n{'='*60}")
    print(f"TEST: Risk Metrics")
    print(f"{'='*60}")
    
    rm = RiskManager(initial_capital=1000000)
    
    # Add multiple positions
    rm.add_position('AAPL', 500, 150)   # $75,000
    rm.add_position('GOOGL', 300, 200)  # $60,000
    rm.add_position('MSFT', 400, 250)   # $100,000
    
    # Update prices
    rm.update_position('AAPL', 160)  # +6.7%
    rm.update_position('GOOGL', 190)  # -5%
    rm.update_position('MSFT', 260)  # +4%
    
    # Calculate metrics
    metrics = rm.calculate_metrics()
    
    print(f"Risk Metrics:")
    print(f"  Total exposure: ${metrics.total_exposure:,.2f}")
    print(f"  Position count: {metrics.position_count}")
    print(f"  Largest position: {metrics.largest_position_pct:.2%}")
    print(f"  Daily P&L: ${metrics.daily_pnl:,.2f} ({metrics.daily_pnl_pct:.2%})")
    print(f"  Max drawdown: ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.2%})")
    
    assert metrics.position_count == 3, "Should have 3 positions"
    assert metrics.total_exposure > 0, "Should have positive exposure"
    
    print(f"✓ Risk metrics calculated correctly")
    
    return None


def test_alerts():
    """Test alert system"""
    print(f"\n{'='*60}")
    print(f"TEST: Alert System")
    print(f"{'='*60}")
    
    rm = RiskManager(initial_capital=1000000)
    
    # Trigger stop loss alert
    rm.add_position('AAPL', 1000, 100)
    rm.check_stop_loss('AAPL', 94)
    
    # Trigger take profit alert
    rm.add_position('GOOGL', 1000, 100)
    rm.check_take_profit('GOOGL', 120)
    
    # Get alerts
    alerts = rm.get_alerts()
    
    print(f"Generated {len(alerts)} alert(s):")
    for alert in alerts:
        print(f"  - [{alert['type']}] {alert['symbol']}: {alert['message']}")
    
    assert len(alerts) >= 2, "Should have at least 2 alerts"
    print(f"✓ Alert system working")
    
    return None


def test_status():
    """Test status reporting"""
    print(f"\n{'='*60}")
    print(f"TEST: Status Reporting")
    print(f"{'='*60}")
    
    rm = RiskManager(initial_capital=1000000)
    rm.add_position('AAPL', 1000, 100)
    rm.add_position('GOOGL', 500, 200)
    
    status = rm.get_status()
    
    print(f"Risk Manager Status:")
    print(f"  Enabled: {status['enabled']}")
    print(f"  Current capital: ${status['current_capital']:,.2f}")
    print(f"  Peak capital: ${status['peak_capital']:,.2f}")
    print(f"  Positions: {status['metrics']['position_count']}")
    print(f"  Total exposure: ${status['metrics']['total_exposure']:,.2f}")
    
    assert 'limits' in status, "Should include limits"
    assert 'metrics' in status, "Should include metrics"
    assert 'positions' in status, "Should include positions"
    
    print(f"✓ Status reporting working")
    
    return None


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


def main():
    """Run all risk management tests"""
    print("\n" + "#"*60)
    print("# Risk Management Test Suite")
    print("#"*60)
    
    results = {}
    
    # Run tests
    results['position_limits'] = _run_test("position_limits", test_position_limits)
    results['stop_loss'] = _run_test("stop_loss", test_stop_loss)
    results['take_profit'] = _run_test("take_profit", test_take_profit)
    results['daily_loss_limit'] = _run_test("daily_loss_limit", test_daily_loss_limit)
    results['max_drawdown'] = _run_test("max_drawdown", test_max_drawdown)
    results['risk_metrics'] = _run_test("risk_metrics", test_risk_metrics)
    results['alerts'] = _run_test("alerts", test_alerts)
    results['status'] = _run_test("status", test_status)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
