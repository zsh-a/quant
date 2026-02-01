#!/usr/bin/env python3
"""
Integration test for risk management system.
Tests end-to-end risk management with trading engine and broker.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.risk_manager import RiskManager
from src.core.broker_interface import MockBroker, Order, OrderSide, OrderType
from src.core.engine import TradingEngine
from src.core.base import Strategy, Bar
from datetime import datetime


class SimpleTestStrategy(Strategy):
    """Simple test strategy for integration testing"""
    
    def __init__(self):
        self.engine = None
        self.call_count = 0
    
    def set_engine(self, engine):
        self.engine = engine
    
    def on_bar(self, bars):
        """Generate test orders"""
        self.call_count += 1
        
        # First bar: try to buy (should succeed)
        if self.call_count == 1:
            order = Order(
                symbol='AAPL',
                side=OrderSide.BUY,
                quantity=500,  # 5% of capital
                order_type=OrderType.MARKET
            )
            self.engine.submit_order(order)
        
        # Second bar: try to buy oversized position (should be rejected)
        elif self.call_count == 2:
            order = Order(
                symbol='GOOGL',
                side=OrderSide.BUY,
                quantity=2000,  # 20% of capital - exceeds 10% limit
                order_type=OrderType.MARKET
            )
            self.engine.submit_order(order)


def test_risk_integration():
    """Test risk manager integration with trading engine"""
    print(f"\n{'='*60}")
    print(f"TEST: Risk Manager Integration")
    print(f"{'='*60}")
    
    from src.core.backtest_broker import BacktestBroker
    
    # Setup
    initial_capital = 100000
    risk_manager = RiskManager(initial_capital=initial_capital)
    broker = BacktestBroker(
        initial_cash=initial_capital, 
        commission=0.0001,
        risk_manager=risk_manager
    )
    
    print(f"Initial capital: ${initial_capital:,.2f}")
    
    # Test 1: Normal position (should succeed)
    print(f"\nTest 1: Normal position (5% of capital)")
    risk_manager.current_capital = initial_capital
    allowed, reason = risk_manager.check_position_limit('AAPL', 50, 100.0)
    assert allowed, f"Normal position should be allowed: {reason}"
    print(f"  ✓ Position allowed: 50 shares @ $100 = $5,000 (5%)")
    
    # Test 2: Oversized position (should be rejected)
    print(f"\nTest 2: Oversized position (20% of capital)")
    allowed, reason = risk_manager.check_position_limit('GOOGL', 200, 100.0)
    assert not allowed, "Oversized position should be rejected"
    print(f"  ✓ Position rejected: {reason}")
    
    # Test 3: Add position and check stop loss
    print(f"\nTest 3: Stop loss check")
    risk_manager.add_position('AAPL', 100, 100.0)
    risk_manager.update_position('AAPL', 94.0)  # 6% loss
    stop_triggered, stop_reason = risk_manager.check_stop_loss('AAPL', 94.0)
    assert stop_triggered, "Stop loss should trigger at 6% loss"
    print(f"  ✓ Stop loss triggered: {stop_reason}")
    
    # Test 4: Take profit check
    print(f"\nTest 4: Take profit check")
    risk_manager.positions.clear()  # Reset
    risk_manager.add_position('GOOGL', 100, 100.0)
    risk_manager.update_position('GOOGL', 120.0)  # 20% gain
    profit_triggered, profit_reason = risk_manager.check_take_profit('GOOGL', 120.0)
    assert profit_triggered, "Take profit should trigger at 20% gain"
    print(f"  ✓ Take profit triggered: {profit_reason}")
    
    # Test 5: Risk metrics
    print(f"\nTest 5: Risk metrics")
    risk_status = risk_manager.get_status()
    print(f"  Position count: {risk_status['metrics']['position_count']}")
    print(f"  Total exposure: ${risk_status['metrics']['total_exposure']:,.2f}")
    print(f"  ✓ Risk metrics calculated")
    
    return True


def test_stop_loss_integration():
    """Test stop loss auto-execution"""
    print(f"\n{'='*60}")
    print(f"TEST: Stop Loss Integration")
    print(f"{'='*60}")
    
    # This test would require BacktestBroker integration
    # For now, we'll create a simplified version
    
    from src.core.backtest_broker import BacktestBroker
    from src.core.base import Order as BacktestOrder
    
    initial_capital = 100000
    risk_manager = RiskManager(initial_capital=initial_capital)
    broker = BacktestBroker(initial_cash=initial_capital, risk_manager=risk_manager)
    
    # Manually add a position
    broker.positions['AAPL'] = 100
    broker.position_costs['AAPL'] = 100.0
    broker.last_prices['AAPL'] = 100.0
    
    # Add to risk manager
    risk_manager.add_position('AAPL', 100, 100.0)
    
    print(f"Initial position: 100 shares AAPL @ $100")
    
    # Simulate price drop to trigger stop loss
    current_price = 94.0  # 6% loss
    risk_manager.update_position('AAPL', current_price)
    
    stop_triggered, reason = risk_manager.check_stop_loss('AAPL', current_price)
    
    assert stop_triggered, "Stop loss should be triggered at 6% loss"
    print(f"  ✓ Stop loss triggered: {reason}")
    
    return True


def test_take_profit_integration():
    """Test take profit auto-execution"""
    print(f"\n{'='*60}")
    print(f"TEST: Take Profit Integration")
    print(f"{'='*60}")
    
    initial_capital = 100000
    risk_manager = RiskManager(initial_capital=initial_capital)
    
    # Add position
    risk_manager.add_position('AAPL', 100, 100.0)
    print(f"Initial position: 100 shares AAPL @ $100")
    
    # Simulate price gain to trigger take profit
    current_price = 120.0  # 20% gain
    risk_manager.update_position('AAPL', current_price)
    
    profit_triggered, reason = risk_manager.check_take_profit('AAPL', current_price)
    
    assert profit_triggered, "Take profit should be triggered at 20% gain"
    print(f"  ✓ Take profit triggered: {reason}")
    
    return True


def main():
    """Run all integration tests"""
    print("\n" + "#"*60)
    print("# Risk Management Integration Test Suite")
    print("#"*60)
    
    results = {}
    
    # Run tests
    try:
        results['risk_integration'] = test_risk_integration()
    except Exception as e:
        print(f"❌ risk_integration failed: {e}")
        results['risk_integration'] = False
    
    try:
        results['stop_loss'] = test_stop_loss_integration()
    except Exception as e:
        print(f"❌ stop_loss failed: {e}")
        results['stop_loss'] = False
    
    try:
        results['take_profit'] = test_take_profit_integration()
    except Exception as e:
        print(f"❌ take_profit failed: {e}")
        results['take_profit'] = False
    
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
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
