"""Min-gap rate limiter, decoupled from any specific time source.

The caller passes ``now_seconds`` from a :class:`~src.brooks.runtime.clock.Clock`,
which lets the same throttle work for live (wall-clock) and replay
(bar-timestamp) without per-call branching.
"""

from __future__ import annotations

from typing import Dict


class Throttle:
    """Skip a key when the last call landed within ``min_gap_seconds``.

    Key example: ``f"{symbol}:{interval}"`` so each bucket throttles
    independently. ``min_gap <= 0`` disables the throttle entirely.
    """

    def __init__(self, min_gap_seconds: float):
        self.min_gap = float(min_gap_seconds)
        self._last: Dict[str, float] = {}

    def should_skip(self, key: str, now_seconds: float) -> bool:
        if self.min_gap <= 0:
            return False
        last = self._last.get(key)
        if last is None:
            return False
        return (now_seconds - last) < self.min_gap

    def mark_called(self, key: str, now_seconds: float) -> None:
        self._last[key] = now_seconds

    def reset(self) -> None:
        self._last.clear()
