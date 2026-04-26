"""WallClock + BarClock are the two clock implementations the runtime ships."""

import time

from src.brooks.runtime.clock import BarClock, WallClock


def test_wall_clock_advances():
    clock = WallClock()
    a = clock.now_seconds()
    time.sleep(0.01)
    b = clock.now_seconds()
    assert b > a


def test_bar_clock_starts_at_zero():
    clock = BarClock()
    assert clock.now_seconds() == 0.0


def test_bar_clock_advance_to_sets_value():
    clock = BarClock()
    clock.advance_to(1234.5)
    assert clock.now_seconds() == 1234.5
    clock.advance_to(2000.0)
    assert clock.now_seconds() == 2000.0


def test_bar_clock_initial_override():
    clock = BarClock(initial=42.0)
    assert clock.now_seconds() == 42.0
