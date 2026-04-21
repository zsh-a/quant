"""3-push wedge detectors.

Bullish wedge (``wedge_long``): 3 sequential *lower* swing lows with
contracting gaps between pushes; the third-push recovery forms the
signal bar. ``wedge_short`` mirrors on swing highs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from src.brooks.patterns.base import DetectorContext, PatternDetector, PatternSignal
from src.brooks.patterns.registry import PatternRegistry

_TICK = 1e-6


@dataclass
class _WedgeState:
    last_signal_bar_idx: int = -1


class _WedgeBase(PatternDetector):
    side: Literal["long", "short"]

    def __init__(self, min_separation_bars: int = 2, max_total_bars: int = 60):
        self.min_separation_bars = min_separation_bars
        self.max_total_bars = max_total_bars
        self._state = _WedgeState()

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "last_signal_bar_idx": self._state.last_signal_bar_idx,
        }

    def reset(self) -> None:
        self._state = _WedgeState()


@PatternRegistry.register("wedge_long")
class WedgeLongDetector(_WedgeBase):
    side: Literal["long", "short"] = "long"

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        pivots = [
            s for s in ctx.structure.confirmed_swing_lows if s.confirmed_at_idx <= ctx.feat.bar_idx
        ]
        if len(pivots) < 3:
            return None
        p1, p2, p3 = pivots[-3:]
        if p3.bar_idx - p1.bar_idx > self.max_total_bars:
            return None
        if p2.bar_idx - p1.bar_idx < self.min_separation_bars:
            return None
        if p3.bar_idx - p2.bar_idx < self.min_separation_bars:
            return None
        if not (p1.price > p2.price > p3.price):
            return None
        if (p2.price - p3.price) >= (p1.price - p2.price):
            return None
        if not (ctx.feat.is_bull and ctx.feat.close > p3.price):
            return None
        if p3.bar_idx == self._state.last_signal_bar_idx:
            return None
        self._state.last_signal_bar_idx = p3.bar_idx
        entry = ctx.feat.high + _TICK
        stop = p3.price - _TICK
        return PatternSignal(
            detector=self.name,
            side="long",
            signal_bar_idx=ctx.feat.bar_idx,
            entry_px=entry,
            stop_px=stop,
            timestamp_ns=ctx.feat.timestamp_ns,
            reason=f"3-push wedge bottom: {p1.price:.4f} → {p2.price:.4f} → {p3.price:.4f}",
            metadata={"p1_idx": p1.bar_idx, "p2_idx": p2.bar_idx, "p3_idx": p3.bar_idx},
        )


@PatternRegistry.register("wedge_short")
class WedgeShortDetector(_WedgeBase):
    side: Literal["long", "short"] = "short"

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        pivots = [
            s for s in ctx.structure.confirmed_swing_highs if s.confirmed_at_idx <= ctx.feat.bar_idx
        ]
        if len(pivots) < 3:
            return None
        p1, p2, p3 = pivots[-3:]
        if p3.bar_idx - p1.bar_idx > self.max_total_bars:
            return None
        if p2.bar_idx - p1.bar_idx < self.min_separation_bars:
            return None
        if p3.bar_idx - p2.bar_idx < self.min_separation_bars:
            return None
        if not (p1.price < p2.price < p3.price):
            return None
        if (p3.price - p2.price) >= (p2.price - p1.price):
            return None
        if not (not ctx.feat.is_bull and ctx.feat.close < p3.price):
            return None
        if p3.bar_idx == self._state.last_signal_bar_idx:
            return None
        self._state.last_signal_bar_idx = p3.bar_idx
        entry = ctx.feat.low - _TICK
        stop = p3.price + _TICK
        return PatternSignal(
            detector=self.name,
            side="short",
            signal_bar_idx=ctx.feat.bar_idx,
            entry_px=entry,
            stop_px=stop,
            timestamp_ns=ctx.feat.timestamp_ns,
            reason=f"3-push wedge top: {p1.price:.4f} → {p2.price:.4f} → {p3.price:.4f}",
            metadata={"p1_idx": p1.bar_idx, "p2_idx": p2.bar_idx, "p3_idx": p3.bar_idx},
        )
