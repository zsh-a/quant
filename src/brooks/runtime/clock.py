"""Pluggable time source for the Brooks runtime.

Live mode uses ``WallClock`` (monotonic seconds). Replay mode uses
``BarClock``, which the engine advances to each bar's timestamp before
running per-bar logic. The throttle and any other time-sensitive code
ask the clock — they don't read ``time.monotonic()`` directly.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    """Returns the current time in seconds, in whatever frame the caller chose."""

    def now_seconds(self) -> float: ...  # noqa: D401, E704


class WallClock:
    """Real wall-clock time. Used by live trading."""

    def now_seconds(self) -> float:
        return time.monotonic()


class BarClock:
    """Clock pinned to the most recently advanced bar's timestamp.

    Engine drives this via :meth:`advance_to` once per bar; everything else
    treats it like any other ``Clock``. Default value before the first
    ``advance_to`` is ``0.0`` so the throttle correctly treats the first
    call as "no prior call".
    """

    def __init__(self, initial: float = 0.0):
        self._t = float(initial)

    def advance_to(self, seconds: float) -> None:
        self._t = float(seconds)

    def now_seconds(self) -> float:
        return self._t
