"""
Trading engine + BacktestBroker unit tests.
"""

from datetime import datetime
from typing import Dict, Optional

import pytest

from src.core.backtest_broker import BacktestBroker
from src.core.base import Bar, DataStream, Order, Strategy
from src.core.engine import TradingEngine
from src.core.risk_manager import RiskManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(symbol: str, ts: str, open_: float, high: float, low: float, close: float, volume: float = 1000) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=datetime.fromisoformat(ts),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        amount=close * volume,
    )


class ListDataStream(DataStream):
    """Simple data stream that iterates over a list of bar dicts."""

    def __init__(self, bars_list: list[Dict[str, Bar]]):
        self._bars = bars_list
        self._idx = 0

    def next_bar(self) -> Optional[Dict[str, Bar]]:
        if self._idx >= len(self._bars):
            return None
        b = self._bars[self._idx]
        self._idx += 1
        return b

    def reset(self):
        self._idx = 0


class BuyAndHoldStrategy(Strategy):
    """Buy on first bar, hold until end."""

    def __init__(self, symbol: str, qty: float = 100, execution_type: str = "NEXT_OPEN"):
        super().__init__()
        self.symbol = symbol
        self.qty = qty
        self.execution_type = execution_type
        self._bought = False

    def on_bar(self, bars):
        if not self._bought:
            self.buy(self.symbol, self.qty, execution_type=self.execution_type)
            self._bought = True


class SellStrategy(Strategy):
    """Sell everything on the first bar."""

    def __init__(self, symbol: str, qty: float = 100, execution_type: str = "IMMEDIATE_CLOSE"):
        super().__init__()
        self.symbol = symbol
        self.qty = qty
        self.execution_type = execution_type

    def on_bar(self, bars):
        self.sell(self.symbol, self.qty, execution_type=self.execution_type)


# ---------------------------------------------------------------------------
# BacktestBroker tests
# ---------------------------------------------------------------------------

class TestBacktestBroker:
    def test_initial_state(self):
        broker = BacktestBroker(initial_cash=500_000, commission=0.001, slippage=0.002)
        assert broker.cash == 500_000
        assert broker.initial_cash == 500_000
        assert broker.commission == 0.001
        assert broker.slippage == 0.002
        assert broker.positions == {}
        assert broker.trades == []
        assert broker.equity_history == []

    def test_submit_order_assigns_id(self):
        broker = BacktestBroker()
        order = Order("SH.600000", "buy", 100, execution_type="NEXT_OPEN")
        oid = broker.submit_order(order)
        assert oid is not None
        assert order.id == oid
        assert order.status == "SUBMITTED"
        assert oid in broker.orders

    def test_cancel_order(self):
        broker = BacktestBroker()
        order = Order("SH.600000", "buy", 100)
        oid = broker.submit_order(order)
        broker.cancel_order(oid)
        assert oid not in broker.orders
        assert order.status == "CANCELLED"
        assert order in broker.history

    def test_buy_execution_next_open(self):
        broker = BacktestBroker(initial_cash=100_000, commission=0.001, slippage=0.0)
        order = Order("SH.600000", "buy", 100, execution_type="NEXT_OPEN")
        broker.submit_order(order)

        bars = {"SH.600000": _bar("SH.600000", "2024-01-02", 10.0, 11.0, 9.5, 10.5)}
        broker.step(bars)

        # Order filled at open (10.0), no slippage
        assert order.status == "FILLED"
        assert order.avg_fill_price == 10.0
        assert broker.positions["SH.600000"] == 100
        expected_cost = 10.0 * 100 * (1 + 0.001)
        assert broker.cash == pytest.approx(100_000 - expected_cost, rel=1e-6)

    def test_buy_execution_immediate_close(self):
        broker = BacktestBroker(initial_cash=100_000, commission=0.001, slippage=0.0)
        order = Order("SH.600000", "buy", 100, execution_type="IMMEDIATE_CLOSE")
        broker.submit_order(order)

        bars = {"SH.600000": _bar("SH.600000", "2024-01-02", 10.0, 11.0, 9.5, 10.5)}
        broker.process_same_bar_orders(bars, "IMMEDIATE_CLOSE")

        assert order.status == "FILLED"
        assert order.avg_fill_price == 10.5  # close price

    def test_sell_reduces_position(self):
        broker = BacktestBroker(initial_cash=100_000, commission=0.001, slippage=0.0)
        broker.positions["SH.600000"] = 200
        broker.position_costs["SH.600000"] = 10.0
        broker.last_prices["SH.600000"] = 10.0

        order = Order("SH.600000", "sell", 200, execution_type="IMMEDIATE_CLOSE")
        broker.submit_order(order)

        bars = {"SH.600000": _bar("SH.600000", "2024-01-03", 11.0, 12.0, 10.5, 11.5)}
        broker.process_same_bar_orders(bars, "IMMEDIATE_CLOSE")

        assert order.status == "FILLED"
        assert "SH.600000" not in broker.positions
        # Sold at close 11.5, 200 shares, minus commission
        assert broker.cash > 100_000

    def test_buy_rejected_insufficient_cash(self):
        broker = BacktestBroker(initial_cash=100, commission=0.001, slippage=0.0)
        order = Order("SH.600000", "buy", 100, execution_type="NEXT_OPEN")
        broker.submit_order(order)

        bars = {"SH.600000": _bar("SH.600000", "2024-01-02", 10.0, 11.0, 9.5, 10.5)}
        broker.step(bars)

        assert order.status == "REJECTED"
        assert broker.positions.get("SH.600000", 0) == 0

    def test_sell_rejected_insufficient_quantity(self):
        broker = BacktestBroker(initial_cash=100_000, commission=0.001, slippage=0.0)
        order = Order("SH.600000", "sell", 100, execution_type="IMMEDIATE_CLOSE")
        broker.submit_order(order)

        bars = {"SH.600000": _bar("SH.600000", "2024-01-02", 10.0, 11.0, 9.5, 10.5)}
        broker.process_same_bar_orders(bars, "IMMEDIATE_CLOSE")

        assert order.status == "REJECTED"

    def test_slippage_applied(self):
        broker = BacktestBroker(initial_cash=100_000, commission=0.0, slippage=0.01)
        order = Order("SH.600000", "buy", 100, execution_type="NEXT_OPEN")
        broker.submit_order(order)

        bars = {"SH.600000": _bar("SH.600000", "2024-01-02", 10.0, 11.0, 9.5, 10.5)}
        broker.step(bars)

        # Buy slippage: price * (1 + 0.01) = 10.1
        assert order.avg_fill_price == pytest.approx(10.1, rel=1e-6)

    def test_equity_history_recorded(self):
        broker = BacktestBroker(initial_cash=100_000, commission=0.0, slippage=0.0)

        bars = {"SH.600000": _bar("SH.600000", "2024-01-02", 10.0, 11.0, 9.5, 10.5)}
        broker.step(bars)

        assert len(broker.equity_history) == 1
        assert broker.equity_history[0]["total_equity"] == 100_000
        assert broker.equity_history[0]["cash"] == 100_000

    def test_get_account_info(self):
        broker = BacktestBroker(initial_cash=100_000, commission=0.0, slippage=0.0)
        broker.positions["SH.600000"] = 100
        broker.position_costs["SH.600000"] = 10.0
        broker.current_bars = {"SH.600000": _bar("SH.600000", "2024-01-02", 10.0, 11.0, 9.5, 11.0)}

        info = broker.get_account_info()
        assert info["cash"] == 100_000
        assert info["total_equity"] == 100_000 + 100 * 11.0
        assert "SH.600000" in info["detailed_positions"]
        assert info["detailed_positions"]["SH.600000"]["unrealized_pnl"] == pytest.approx(100.0)

    def test_snapshot_and_restore(self):
        broker = BacktestBroker(initial_cash=100_000, commission=0.001, slippage=0.002)
        broker.positions = {"SH.600000": 200}
        broker.position_costs = {"SH.600000": 10.0}
        broker.last_prices = {"SH.600000": 11.0}
        broker._last_equity = 102_000

        snapshot = broker.get_state_snapshot()

        broker2 = BacktestBroker()
        broker2.restore_from_snapshot(snapshot)

        assert broker2.cash == 100_000
        assert broker2.positions == {"SH.600000": 200.0}
        assert broker2.position_costs == {"SH.600000": 10.0}
        assert broker2._last_equity == 102_000


# ---------------------------------------------------------------------------
# TradingEngine tests
# ---------------------------------------------------------------------------

class TestTradingEngine:
    def test_engine_runs_to_completion(self):
        bars_list = [
            {"SH.600000": _bar("SH.600000", "2024-01-02", 10.0, 11.0, 9.5, 10.5)},
            {"SH.600000": _bar("SH.600000", "2024-01-03", 10.5, 12.0, 10.0, 11.0)},
            {"SH.600000": _bar("SH.600000", "2024-01-04", 11.0, 11.5, 10.5, 11.2)},
        ]
        ds = ListDataStream(bars_list)
        broker = BacktestBroker(initial_cash=100_000, commission=0.0, slippage=0.0)
        strategy = BuyAndHoldStrategy("SH.600000", qty=100, execution_type="IMMEDIATE_CLOSE")
        engine = TradingEngine(strategy, broker, ds)

        engine.run()

        assert len(broker.equity_history) == 3
        assert broker.positions.get("SH.600000") == 100
        assert len(broker.trades) == 1

    def test_next_open_fills_on_next_bar(self):
        bars_list = [
            {"SH.600000": _bar("SH.600000", "2024-01-02", 10.0, 11.0, 9.5, 10.5)},
            {"SH.600000": _bar("SH.600000", "2024-01-03", 10.5, 12.0, 10.0, 11.0)},
        ]
        ds = ListDataStream(bars_list)
        broker = BacktestBroker(initial_cash=100_000, commission=0.0, slippage=0.0)
        strategy = BuyAndHoldStrategy("SH.600000", qty=100, execution_type="NEXT_OPEN")
        engine = TradingEngine(strategy, broker, ds)

        engine.run()

        # Order submitted on bar1, filled at bar2's open (10.5)
        assert len(broker.trades) == 1
        assert broker.trades[0]["price"] == 10.5

    def test_on_step_callback(self):
        bars_list = [
            {"SH.600000": _bar("SH.600000", "2024-01-02", 10.0, 11.0, 9.5, 10.5)},
            {"SH.600000": _bar("SH.600000", "2024-01-03", 10.5, 12.0, 10.0, 11.0)},
        ]
        steps = []
        ds = ListDataStream(bars_list)
        broker = BacktestBroker(initial_cash=100_000)
        strategy = BuyAndHoldStrategy("SH.600000")
        engine = TradingEngine(strategy, broker, ds, on_step=lambda b: steps.append(b))

        engine.run()
        assert len(steps) == 2

    def test_engine_stop(self):
        bars_list = [
            {"SH.600000": _bar("SH.600000", f"2024-01-{d:02d}", 10.0, 11.0, 9.5, 10.5)}
            for d in range(2, 20)
        ]

        class StopAfterThree(Strategy):
            def __init__(self):
                super().__init__()
                self.count = 0

            def on_bar(self, bars):
                self.count += 1
                if self.count >= 3:
                    self.engine.stop()

        ds = ListDataStream(bars_list)
        broker = BacktestBroker()
        strategy = StopAfterThree()
        engine = TradingEngine(strategy, broker, ds)

        engine.run()
        assert strategy.count == 3
        assert not engine.running

    def test_risk_manager_rejects_order(self):
        bars_list = [
            {"SH.600000": _bar("SH.600000", "2024-01-02", 10.0, 11.0, 9.5, 10.5)},
            {"SH.600000": _bar("SH.600000", "2024-01-03", 10.5, 12.0, 10.0, 11.0)},
        ]
        ds = ListDataStream(bars_list)
        rm = RiskManager(enabled=True)
        rm.limits.max_position_pct = 0.001  # very small — will reject
        broker = BacktestBroker(initial_cash=100_000, commission=0.0, slippage=0.0, risk_manager=rm)
        strategy = BuyAndHoldStrategy("SH.600000", qty=10_000, execution_type="IMMEDIATE_CLOSE")
        engine = TradingEngine(strategy, broker, ds, risk_manager=rm)

        engine.run()

        # Order should be rejected by risk manager (position too large)
        assert broker.positions.get("SH.600000", 0) == 0

    def test_daily_reset_called_on_date_change(self):
        """Risk manager reset_daily() is called when trading date changes."""
        bars_list = [
            {"SH.600000": _bar("SH.600000", "2024-01-02", 10.0, 11.0, 9.5, 10.5)},
            {"SH.600000": _bar("SH.600000", "2024-01-03", 10.5, 12.0, 10.0, 11.0)},
            {"SH.600000": _bar("SH.600000", "2024-01-04", 11.0, 11.5, 10.5, 11.2)},
        ]
        ds = ListDataStream(bars_list)
        rm = RiskManager(enabled=True)
        rm.current_capital = 100_000
        rm.daily_start_capital = 100_000
        broker = BacktestBroker(initial_cash=100_000, commission=0.0, slippage=0.0, risk_manager=rm)
        strategy = BuyAndHoldStrategy("SH.600000", qty=100, execution_type="IMMEDIATE_CLOSE")
        engine = TradingEngine(strategy, broker, ds, risk_manager=rm)

        engine.run()

        # daily_start_capital should have been updated (reset) on day 2 and day 3
        # After 3 bars, daily_start_capital should reflect the equity at start of last day
        assert rm.daily_start_capital != 100_000  # was reset at least once

    def test_daily_loss_limit_check(self):
        """check_daily_loss_limit() is invoked during step()."""
        rm = RiskManager(enabled=True)
        rm.limits.max_daily_loss_pct = 0.01  # 1% — very tight
        rm.current_capital = 99_000
        rm.daily_start_capital = 100_000
        halt, reason = rm.check_daily_loss_limit()
        assert halt is True
        assert "Daily loss limit" in reason

    def test_max_drawdown_check(self):
        """check_max_drawdown() triggers when drawdown exceeds limit."""
        rm = RiskManager(enabled=True)
        rm.limits.max_drawdown_pct = 0.05  # 5%
        rm.peak_capital = 100_000
        rm.current_capital = 94_000  # 6% drawdown
        halt, reason = rm.check_max_drawdown()
        assert halt is True
        assert "Max drawdown" in reason


# ---------------------------------------------------------------------------
# Short selling & stop order tests
# ---------------------------------------------------------------------------

class TestAdvancedOrders:
    def test_short_sell_disabled_by_default(self):
        broker = BacktestBroker(initial_cash=100_000, commission=0.0, slippage=0.0)
        order = Order("SH.600000", "sell_short", 100, execution_type="IMMEDIATE_CLOSE")
        broker.submit_order(order)
        bars = {"SH.600000": _bar("SH.600000", "2024-01-02", 10.0, 11.0, 9.5, 10.5)}
        broker.process_same_bar_orders(bars, "IMMEDIATE_CLOSE")
        assert order.status == "REJECTED"

    def test_short_sell_enabled(self):
        broker = BacktestBroker(initial_cash=100_000, commission=0.0, slippage=0.0, allow_short=True)
        order = Order("SH.600000", "sell_short", 100, execution_type="IMMEDIATE_CLOSE")
        broker.submit_order(order)
        bars = {"SH.600000": _bar("SH.600000", "2024-01-02", 10.0, 11.0, 9.5, 10.5)}
        broker.process_same_bar_orders(bars, "IMMEDIATE_CLOSE")

        assert order.status == "FILLED"
        assert broker.positions["SH.600000"] == -100  # negative position
        assert broker.cash == 100_000 + 10.5 * 100  # received proceeds

    def test_buy_to_cover(self):
        broker = BacktestBroker(initial_cash=200_000, commission=0.0, slippage=0.0, allow_short=True)
        broker.positions["SH.600000"] = -100
        broker.position_costs["SH.600000"] = 10.0

        order = Order("SH.600000", "buy_to_cover", 100, execution_type="IMMEDIATE_CLOSE")
        broker.submit_order(order)
        bars = {"SH.600000": _bar("SH.600000", "2024-01-03", 9.0, 10.0, 8.5, 9.5)}
        broker.process_same_bar_orders(bars, "IMMEDIATE_CLOSE")

        assert order.status == "FILLED"
        assert "SH.600000" not in broker.positions  # position closed

    def test_stop_order_triggers(self):
        broker = BacktestBroker(initial_cash=100_000, commission=0.0, slippage=0.0)
        broker.positions["SH.600000"] = 100
        broker.position_costs["SH.600000"] = 10.0

        # Stop-sell at 9.0
        order = Order("SH.600000", "sell", 100, execution_type="NEXT_OPEN", stop_price=9.0)
        broker.submit_order(order)

        # Day 1: price stays above stop — should NOT trigger
        bars1 = {"SH.600000": _bar("SH.600000", "2024-01-02", 10.0, 11.0, 9.5, 10.5)}
        broker.step(bars1)
        assert order.status == "SUBMITTED"  # not triggered

        # Day 2: low touches stop price — should trigger and fill
        bars2 = {"SH.600000": _bar("SH.600000", "2024-01-03", 9.5, 10.0, 8.8, 9.0)}
        broker.step(bars2)
        assert order.status == "FILLED"
        assert "SH.600000" not in broker.positions

    def test_order_stop_price_field(self):
        order = Order("SH.600000", "sell", 100, stop_price=9.5)
        assert order.stop_price == 9.5
        assert order.execution_type == "NEXT_OPEN"


# ---------------------------------------------------------------------------
# Validators tests
# ---------------------------------------------------------------------------

class TestValidators:
    def test_date_str_valid(self):
        from src.api.validators import _validate_date
        assert _validate_date("2024-01-15") == "2024-01-15"

    def test_date_str_invalid(self):
        from src.api.validators import _validate_date
        with pytest.raises(ValueError):
            _validate_date("01-15-2024")
        with pytest.raises(ValueError):
            _validate_date("2024/01/15")
        with pytest.raises(ValueError):
            _validate_date("not-a-date")

    def test_symbol_valid(self):
        from src.api.validators import _validate_symbol
        assert _validate_symbol("sh.600000") == "sh.600000"
        assert _validate_symbol("BTCUSDT") == "BTCUSDT"

    def test_symbol_invalid(self):
        from src.api.validators import _validate_symbol
        with pytest.raises(ValueError):
            _validate_symbol("'; DROP TABLE --")

    def test_mode_valid(self):
        from src.api.validators import _validate_mode
        assert _validate_mode("backtest") == "backtest"
        assert _validate_mode("simulation") == "simulation"
        assert _validate_mode("live") == "live"

    def test_mode_invalid(self):
        from src.api.validators import _validate_mode
        with pytest.raises(ValueError):
            _validate_mode("invalid")


# ---------------------------------------------------------------------------
# Auth module tests
# ---------------------------------------------------------------------------

class TestAuth:
    def test_create_and_verify_token(self):
        from src.api.auth import _create_access_token, _verify_token
        token = _create_access_token("testuser")
        assert _verify_token(token) == "testuser"

    def test_invalid_token_raises(self):
        from src.api.auth import _verify_token
        with pytest.raises(Exception):
            _verify_token("invalid.token.here")
