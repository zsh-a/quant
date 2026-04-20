"""TimeframeResampler tests."""

from __future__ import annotations

from datetime import datetime, timedelta

from src.core.base import Bar
from src.core.timeframe_resampler import TimeframeResampler


def mk(i: int, c: float) -> Bar:
    return Bar(
        symbol="BTCUSDT",
        timestamp=datetime(2025, 1, 1, 0, 0) + timedelta(minutes=5 * i),
        open=c - 0.1,
        high=c + 0.2,
        low=c - 0.2,
        close=c,
        volume=1.0,
        amount=c,
    )


def test_15m_bar_closes_every_three_5m():
    r = TimeframeResampler(base="5m", higher=["15m"])
    closed = []
    for i in range(10):
        out = r.update(mk(i, 100.0 + i))
        if out["15m"] is not None:
            closed.append((i, out["15m"]))
    # Bar i=2 is the closing 5m bar of the 15m bucket [0,15m) — the resampler
    # emits the completed bar at that moment. Closes should therefore land at
    # i=2, 5, 8, ... (one *completed* 15m bar every three 5m bars).
    indices = [c[0] for c in closed]
    assert indices == [2, 5, 8]
    first_bar = closed[0][1]
    assert first_bar.open == 100.0 - 0.1  # open of first 5m bar
    # high should be max of close+0.2 over i=0..2 = 100.2 max = 102.2
    assert abs(first_bar.high - 102.2) < 1e-9


def test_no_look_ahead_partial_bar_never_returned():
    r = TimeframeResampler(base="5m", higher=["1h"])
    # 1h = 12 5m bars. Before bar 12, never emit anything.
    for i in range(11):
        out = r.update(mk(i, 100.0 + i))
        assert out["1h"] is None
