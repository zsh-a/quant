"""Tests for src.brooks.risk.time_stop."""

from __future__ import annotations

import pytest

from src.brooks.risk.state import PositionState
from src.brooks.risk.time_stop import TimeStop


def _pos() -> PositionState:
    return PositionState(
        symbol="BTCUSDT",
        side="long",
        entry_px=100.0,
        stop_px=99.0,
        qty_initial=1.0,
        qty_open=1.0,
        one_r=1.0,
    )


class TestTimeStop:
    def test_exit_when_no_progress(self):
        stop = TimeStop(max_bars_to_1r=10)
        assert stop.should_exit(_pos(), bars_since_entry=10, max_unrealized_r=0.3) is True

    def test_exit_exactly_at_boundary(self):
        """spec: bars_since_entry >= max_bars AND max_r < 1.0."""
        stop = TimeStop(max_bars_to_1r=10)
        # 10 bars, 0.9R max → exit.
        assert stop.should_exit(_pos(), 10, 0.9) is True

    def test_no_exit_before_boundary(self):
        stop = TimeStop(max_bars_to_1r=10)
        assert stop.should_exit(_pos(), 9, 0.0) is False

    def test_no_exit_if_reached_1r(self):
        stop = TimeStop(max_bars_to_1r=10)
        assert stop.should_exit(_pos(), 50, 1.0) is False
        assert stop.should_exit(_pos(), 50, 2.5) is False

    def test_no_exit_if_briefly_touched_1r(self):
        """Trade that hit 1R and pulled back should NOT be time-stopped."""
        stop = TimeStop(max_bars_to_1r=10)
        assert stop.should_exit(_pos(), 15, 1.2) is False

    def test_construction_validation(self):
        with pytest.raises(ValueError):
            TimeStop(max_bars_to_1r=0)
