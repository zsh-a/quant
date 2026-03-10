#!/usr/bin/env python3
"""
Broker integration test script.
Tests broker interface, order execution, and position management.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.broker_interface import (
    MockBroker, Order, OrderType, OrderSide, OrderStatus
)


def test_connection():
    """Test broker connection"""
    print(f"\n{'='*60}")
    print(f"TEST: Broker Connection")
    print(f"{'='*60}")
    
    broker = MockBroker(initial_cash=1000000)
    
    # Test connection
    assert not broker.is_connected(), "Should not be connected initially"
    print(f"✓ Initial state: not connected")
    
    connected = broker.connect()
    assert connected, "Connection should succeed"
    assert broker.is_connected(), "Should be connected"
    print(f"✓ Connected successfully")
    
    disconnected = broker.disconnect()
    assert disconnected, "Disconnection should succeed"
    assert not broker.is_connected(), "Should not be connected"
    print(f"✓ Disconnected successfully")
    
    return None


def test_market_order():
    """Test market order execution"""
    print(f"\n{'='*60}")
    print(f"TEST: Market Order Execution")
    print(f"{'='*60}")
    
    broker = MockBroker(initial_cash=1000000, commission_rate=0.0001)
    broker.connect()
    
    # Set market price
    broker.set_market_price('AAPL', 150.0)
    
    # Submit buy order
    order = Order(
        symbol='AAPL',
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET
    )
    
    order_id = broker.submit_order(order)
    print(f"Buy order submitted: {order_id}")
    
    # Check order status
    filled_order = broker.get_order_status(order_id)
    assert filled_order.status == OrderStatus.FILLED, "Order should be filled"
    assert filled_order.filled_quantity == 100, "Should fill 100 shares"
    assert filled_order.avg_fill_price == 150.0, "Fill price should be $150"
    
    commission = 100 * 150.0 * 0.0001
    print(f"✓ Buy order filled: 100 shares @ $150.00, commission=${commission:.2f}")
    
    # Check account
    account = broker.get_account_info()
    expected_cash = 1000000 - (100 * 150.0 + commission)
    assert abs(account.cash - expected_cash) < 0.01, f"Cash should be ${expected_cash:.2f}"
    print(f"✓ Cash updated: ${account.cash:.2f}")
    
    # Check position
    position = broker.get_position('AAPL')
    assert position is not None, "Should have AAPL position"
    assert position.quantity == 100, "Position should be 100 shares"
    assert position.avg_cost == 150.0, "Avg cost should be $150"
    print(f"✓ Position created: {position.quantity} shares @ ${position.avg_cost:.2f}")
    
    return None


def test_sell_order():
    """Test sell order execution"""
    print(f"\n{'='*60}")
    print(f"TEST: Sell Order Execution")
    print(f"{'='*60}")
    
    broker = MockBroker(initial_cash=1000000, commission_rate=0.0001)
    broker.connect()
    
    # Buy first
    broker.set_market_price('AAPL', 150.0)
    buy_order = Order(
        symbol='AAPL',
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET
    )
    broker.submit_order(buy_order)
    
    # Update price and sell
    broker.set_market_price('AAPL', 160.0)
    sell_order = Order(
        symbol='AAPL',
        side=OrderSide.SELL,
        quantity=50,
        order_type=OrderType.MARKET
    )
    
    order_id = broker.submit_order(sell_order)
    print(f"Sell order submitted: {order_id}")
    
    # Check order status
    filled_order = broker.get_order_status(order_id)
    assert filled_order.status == OrderStatus.FILLED, "Order should be filled"
    print(f"✓ Sell order filled: 50 shares @ $160.00")
    
    # Check position
    position = broker.get_position('AAPL')
    assert position.quantity == 50, "Position should be 50 shares remaining"
    print(f"✓ Position updated: {position.quantity} shares remaining")
    
    # Check P&L
    account = broker.get_account_info()
    profit = 50 * (160.0 - 150.0)  # $500 profit
    print(f"✓ Realized profit: ${profit:.2f}")
    
    return None


def test_insufficient_cash():
    """Test order rejection due to insufficient cash"""
    print(f"\n{'='*60}")
    print(f"TEST: Insufficient Cash Rejection")
    print(f"{'='*60}")
    
    broker = MockBroker(initial_cash=10000, commission_rate=0.0001)
    broker.connect()
    
    # Try to buy more than we can afford
    broker.set_market_price('AAPL', 150.0)
    order = Order(
        symbol='AAPL',
        side=OrderSide.BUY,
        quantity=100,  # Would cost $15,000 + commission
        order_type=OrderType.MARKET
    )
    
    order_id = broker.submit_order(order)
    
    # Check order status
    rejected_order = broker.get_order_status(order_id)
    assert rejected_order.status == OrderStatus.REJECTED, "Order should be rejected"
    print(f"✓ Order rejected due to insufficient cash")
    
    # Check cash unchanged
    account = broker.get_account_info()
    assert account.cash == 10000, "Cash should be unchanged"
    print(f"✓ Cash unchanged: ${account.cash:.2f}")
    
    return None


def test_insufficient_position():
    """Test order rejection due to insufficient position"""
    print(f"\n{'='*60}")
    print(f"TEST: Insufficient Position Rejection")
    print(f"{'='*60}")
    
    broker = MockBroker(initial_cash=1000000, commission_rate=0.0001)
    broker.connect()
    
    # Buy 100 shares
    broker.set_market_price('AAPL', 150.0)
    buy_order = Order(
        symbol='AAPL',
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET
    )
    broker.submit_order(buy_order)
    
    # Try to sell more than we have
    sell_order = Order(
        symbol='AAPL',
        side=OrderSide.SELL,
        quantity=200,  # We only have 100
        order_type=OrderType.MARKET
    )
    
    order_id = broker.submit_order(sell_order)
    
    # Check order status
    rejected_order = broker.get_order_status(order_id)
    assert rejected_order.status == OrderStatus.REJECTED, "Order should be rejected"
    print(f"✓ Order rejected due to insufficient position")
    
    # Check position unchanged
    position = broker.get_position('AAPL')
    assert position.quantity == 100, "Position should be unchanged"
    print(f"✓ Position unchanged: {position.quantity} shares")
    
    return None


def test_position_pnl():
    """Test position P&L calculation"""
    print(f"\n{'='*60}")
    print(f"TEST: Position P&L Calculation")
    print(f"{'='*60}")
    
    broker = MockBroker(initial_cash=1000000, commission_rate=0.0001)
    broker.connect()
    
    # Buy at $150
    broker.set_market_price('AAPL', 150.0)
    order = Order(
        symbol='AAPL',
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET
    )
    broker.submit_order(order)
    
    # Update price to $160
    broker.set_market_price('AAPL', 160.0)
    
    # Get account info (updates position values)
    account = broker.get_account_info()
    position = account.positions['AAPL']
    
    expected_pnl = 100 * (160.0 - 150.0)  # $1000
    expected_pnl_pct = (160.0 - 150.0) / 150.0  # 6.67%
    
    assert abs(position.unrealized_pnl - expected_pnl) < 0.01, "P&L should be $1000"
    assert abs(position.unrealized_pnl_pct - expected_pnl_pct) < 0.0001, "P&L% should be 6.67%"
    
    print(f"Position P&L:")
    print(f"  Quantity: {position.quantity}")
    print(f"  Avg cost: ${position.avg_cost:.2f}")
    print(f"  Current price: ${position.current_price:.2f}")
    print(f"  Market value: ${position.market_value:.2f}")
    print(f"  Unrealized P&L: ${position.unrealized_pnl:.2f} ({position.unrealized_pnl_pct:.2%})")
    
    print(f"✓ P&L calculated correctly")
    
    return None


def test_account_equity():
    """Test total account equity calculation"""
    print(f"\n{'='*60}")
    print(f"TEST: Account Equity Calculation")
    print(f"{'='*60}")
    
    broker = MockBroker(initial_cash=1000000, commission_rate=0.0001)
    broker.connect()
    
    # Buy multiple positions
    broker.set_market_price('AAPL', 150.0)
    broker.set_market_price('GOOGL', 200.0)
    
    broker.submit_order(Order(symbol='AAPL', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET))
    broker.submit_order(Order(symbol='GOOGL', side=OrderSide.BUY, quantity=50, order_type=OrderType.MARKET))
    
    # Update prices
    broker.set_market_price('AAPL', 160.0)  # +$10
    broker.set_market_price('GOOGL', 190.0)  # -$10
    
    # Get account info
    account = broker.get_account_info()
    
    # Calculate expected equity
    aapl_value = 100 * 160.0  # $16,000
    googl_value = 50 * 190.0  # $9,500
    expected_equity = account.cash + aapl_value + googl_value
    
    assert abs(account.total_equity - expected_equity) < 0.01, "Total equity mismatch"
    
    print(f"Account Summary:")
    print(f"  Cash: ${account.cash:,.2f}")
    print(f"  Position value: ${aapl_value + googl_value:,.2f}")
    print(f"  Total equity: ${account.total_equity:,.2f}")
    print(f"  Initial capital: $1,000,000.00")
    print(f"  Total P&L: ${account.total_equity - 1000000:,.2f}")
    
    print(f"✓ Account equity calculated correctly")
    
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
    """Run all broker integration tests"""
    print("\n" + "#"*60)
    print("# Broker Integration Test Suite")
    print("#"*60)
    
    results = {}
    
    # Run tests
    results['connection'] = _run_test("connection", test_connection)
    results['market_order'] = _run_test("market_order", test_market_order)
    results['sell_order'] = _run_test("sell_order", test_sell_order)
    results['insufficient_cash'] = _run_test("insufficient_cash", test_insufficient_cash)
    results['insufficient_position'] = _run_test("insufficient_position", test_insufficient_position)
    results['position_pnl'] = _run_test("position_pnl", test_position_pnl)
    results['account_equity'] = _run_test("account_equity", test_account_equity)
    
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
