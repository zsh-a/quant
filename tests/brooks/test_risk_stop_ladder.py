"""Tests for src.brooks.risk.stop_ladder.

Each test uses a hand-crafted bar sequence to step a position through the
four ladder stages in order, then verifies the stop never retreats when
price moves back against us.
"""

from __future__ import annotations

import pytest

from src.brooks.features import SwingPoint
from src.brooks.risk.state import PositionState
from src.brooks.risk.stop_ladder import StopLadder


def _make_long_pos(entry: float = 100.0, stop: float = 99.0) -> PositionState:
    return PositionState(
        symbol="BTCUSDT",
        side="long",
        entry_px=entry,
        stop_px=stop,
        qty_initial=1.0,
        qty_open=1.0,
        one_r=abs(entry - stop),
        entry_bar_idx=0,
    )


def _make_short_pos(entry: float = 100.0, stop: float = 101.0) -> PositionState:
    return PositionState(
        symbol="BTCUSDT",
        side="short",
        entry_px=entry,
        stop_px=stop,
        qty_initial=1.0,
        qty_open=1.0,
        one_r=abs(entry - stop),
        entry_bar_idx=0,
    )


class TestLongLadder:
    def test_all_four_stages_in_order(self):
        ladder = StopLadder(
            partial_trail_trigger_r=1.5,
            partial_trail_lookback=5,
            use_swing_trail=True,
        )
        pos = _make_long_pos(entry=100.0, stop=99.0)  # 1R = 1

        # Stage 0 (initial): price holds near entry → no move.
        assert ladder.update(pos, 100.2, 99.8, 100.1, []) is None
        assert pos.ladder_stage == "initial"
        assert pos.stop_px == pytest.approx(99.0)

        # Stage 1 → 2: push close to 101 (1R exactly) → break_even.
        new_stop = ladder.update(pos, 101.0, 100.0, 101.0, [])
        assert new_stop == pytest.approx(100.0)
        assert pos.ladder_stage == "break_even"
        assert pos.stop_px == pytest.approx(100.0)

        # Stage 2 → 3: push close to 101.5 (1.5R) → partial_trail.
        # With no new bar-lows below 100 recorded, the trail stop equals min
        # of recent_lows which is still > 100 → improvement picked.
        new_stop = ladder.update(pos, 101.6, 101.3, 101.5, [])
        assert pos.ladder_stage == "partial_trail"
        assert pos.stop_px >= 100.0  # monotonic
        partial_stop = pos.stop_px

        # Push harder — partial_trail should lift the stop to min of recent lows.
        ladder.update(pos, 102.0, 101.5, 101.9, [])
        assert pos.stop_px >= partial_stop

        # Stage 3 → 4: a confirmed swing_low above current stop → swing_trail.
        swings = [
            SwingPoint(bar_idx=3, price=101.0, kind="low", confirmed_at_idx=6),
        ]
        ladder.update(pos, 102.5, 102.0, 102.4, swings)
        # swing price 101 should beat partial-trail min (which tracked lows ~100-ish).
        assert pos.ladder_stage == "swing_trail"
        assert pos.stop_px == pytest.approx(101.0)

    def test_stop_never_retreats(self):
        ladder = StopLadder(partial_trail_trigger_r=1.5, partial_trail_lookback=5)
        pos = _make_long_pos(entry=100.0, stop=99.0)

        # Promote to break_even.
        ladder.update(pos, 101.2, 100.5, 101.1, [])
        assert pos.stop_px == pytest.approx(100.0)

        # Push into partial_trail with a bar whose low drags recent_lows down.
        ladder.update(pos, 101.6, 101.2, 101.5, [])
        high_water_stop = pos.stop_px
        assert high_water_stop >= 100.0

        # Now a pullback bar with a much lower low — partial_trail math would
        # suggest a lower stop, but the ladder must not retreat.
        ladder.update(pos, 101.4, 100.3, 101.0, [])
        assert pos.stop_px == pytest.approx(high_water_stop)

    def test_swing_below_current_stop_ignored(self):
        ladder = StopLadder(partial_trail_trigger_r=1.5, partial_trail_lookback=5)
        pos = _make_long_pos(entry=100.0, stop=99.0)

        # Walk to partial_trail, pin stop around 100.5.
        ladder.update(pos, 101.2, 100.5, 101.1, [])  # → break_even
        ladder.update(pos, 101.6, 100.7, 101.5, [])  # → partial_trail
        before = pos.stop_px

        swings = [
            SwingPoint(bar_idx=2, price=99.5, kind="low", confirmed_at_idx=5),
        ]
        ladder.update(pos, 101.8, 101.0, 101.7, swings)
        assert pos.stop_px >= before

    def test_swing_before_entry_ignored(self):
        ladder = StopLadder(partial_trail_trigger_r=1.5, partial_trail_lookback=5)
        pos = _make_long_pos(entry=100.0, stop=99.0)
        pos.entry_bar_idx = 10

        ladder.update(pos, 101.2, 100.5, 101.1, [])
        ladder.update(pos, 101.6, 101.3, 101.5, [])
        before = pos.stop_px

        swings = [
            SwingPoint(bar_idx=5, price=101.0, kind="low", confirmed_at_idx=8),
        ]
        ladder.update(pos, 101.8, 101.4, 101.7, swings)
        assert pos.stop_px == pytest.approx(before)
        assert pos.ladder_stage != "swing_trail"

    def test_max_unrealized_r_tracked(self):
        ladder = StopLadder()
        pos = _make_long_pos(entry=100.0, stop=99.0)
        ladder.update(pos, 102.0, 101.0, 101.8, [])
        assert pos.max_unrealized_r == pytest.approx(1.8)

        # A pullback bar should NOT reduce max_unrealized_r.
        ladder.update(pos, 101.5, 100.5, 101.0, [])
        assert pos.max_unrealized_r == pytest.approx(1.8)


class TestShortLadder:
    def test_short_stages_in_order(self):
        ladder = StopLadder(partial_trail_trigger_r=1.5, partial_trail_lookback=5)
        pos = _make_short_pos(entry=100.0, stop=101.0)  # 1R = 1

        # 1R: close at 99 → break_even at 100.
        new_stop = ladder.update(pos, 100.0, 99.0, 99.0, [])
        assert new_stop == pytest.approx(100.0)
        assert pos.ladder_stage == "break_even"

        # 1.5R: close at 98.5 → partial_trail.
        ladder.update(pos, 99.0, 98.3, 98.5, [])
        assert pos.ladder_stage == "partial_trail"
        assert pos.stop_px <= 100.0  # monotonic (lower is better for short)

        # Confirmed swing high → swing_trail.
        swings = [
            SwingPoint(bar_idx=3, price=99.2, kind="high", confirmed_at_idx=6),
        ]
        ladder.update(pos, 99.0, 98.2, 98.5, swings)
        assert pos.ladder_stage == "swing_trail"
        assert pos.stop_px == pytest.approx(99.2)

    def test_short_stop_never_retreats(self):
        ladder = StopLadder(partial_trail_trigger_r=1.5, partial_trail_lookback=5)
        pos = _make_short_pos(entry=100.0, stop=101.0)

        ladder.update(pos, 100.0, 99.0, 99.0, [])
        assert pos.stop_px == pytest.approx(100.0)
        low_water_stop = pos.stop_px

        # A bar that spikes above high_water would normally ratchet partial_trail
        # up. Ensure stop does not move back up (unfavourable for short).
        ladder.update(pos, 99.9, 99.0, 99.5, [])
        assert pos.stop_px <= low_water_stop


class TestLadderEdgeCases:
    def test_flat_position_no_update(self):
        ladder = StopLadder()
        pos = _make_long_pos()
        pos.qty_open = 0
        assert ladder.update(pos, 101.0, 100.0, 101.0, []) is None

    def test_swing_trail_disabled(self):
        ladder = StopLadder(partial_trail_trigger_r=1.5, use_swing_trail=False)
        pos = _make_long_pos(entry=100.0, stop=99.0)
        ladder.update(pos, 101.2, 100.5, 101.1, [])  # → break_even
        ladder.update(pos, 101.8, 101.3, 101.7, [])  # → partial_trail

        swings = [SwingPoint(bar_idx=3, price=101.2, kind="low", confirmed_at_idx=6)]
        ladder.update(pos, 102.0, 101.6, 101.9, swings)
        assert pos.ladder_stage != "swing_trail"

    def test_construction_validation(self):
        with pytest.raises(ValueError):
            StopLadder(partial_trail_trigger_r=0.9)  # must be > 1.0
        with pytest.raises(ValueError):
            StopLadder(partial_trail_lookback=0)
