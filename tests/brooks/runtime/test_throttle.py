"""Throttle is the time-source-agnostic min-gap rate limiter.

Live mode pairs it with a wall-clock; replay pairs it with bar timestamps.
The behaviour is identical — only the ``now_seconds`` value differs.
"""

from src.brooks.runtime.throttle import Throttle


def test_first_call_passes_through():
    throttle = Throttle(min_gap_seconds=10.0)
    assert throttle.should_skip("k", 100.0) is False
    throttle.mark_called("k", 100.0)


def test_within_gap_skips():
    throttle = Throttle(min_gap_seconds=10.0)
    throttle.mark_called("k", 100.0)
    assert throttle.should_skip("k", 105.0) is True


def test_beyond_gap_passes():
    throttle = Throttle(min_gap_seconds=10.0)
    throttle.mark_called("k", 100.0)
    assert throttle.should_skip("k", 111.0) is False


def test_separate_keys_independent():
    throttle = Throttle(min_gap_seconds=10.0)
    throttle.mark_called("a", 100.0)
    assert throttle.should_skip("a", 105.0) is True
    assert throttle.should_skip("b", 105.0) is False


def test_zero_gap_disables():
    throttle = Throttle(min_gap_seconds=0)
    throttle.mark_called("k", 100.0)
    assert throttle.should_skip("k", 100.0) is False


def test_reset_clears_state():
    throttle = Throttle(min_gap_seconds=10.0)
    throttle.mark_called("k", 100.0)
    throttle.reset()
    assert throttle.should_skip("k", 105.0) is False
