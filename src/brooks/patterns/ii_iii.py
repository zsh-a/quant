"""ii / iii (inside-bar compression) breakout detectors.

* **ii**: two consecutive inside bars; the second's high/low brackets
  define the breakout level.
* **iii**: three consecutive inside bars; tighter compression — the
  expected post-breakout range is larger.

We do not wait for the actual range break in price — the moment the
inside-bar stack is complete we emit a stop-entry signal in the
*always_in* trend direction. ``entry = inside_high + 1 tick`` (long) or
``inside_low - 1 tick`` (short); the opposite extreme is the stop.

When ``always_in == neutral`` no signal is emitted — counter-trend
inside-bar breakouts are noisy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.brooks.features import ExtendedBarFeatures
from src.brooks.patterns.base import DetectorContext, PatternDetector, PatternSignal
from src.brooks.patterns.registry import PatternRegistry

_TICK = 1e-6


class _InsideBarBreakoutBase(PatternDetector):
    """Shared logic for ii / iii detection."""

    n_inside: int = 2

    def __init__(self):
        self._last_signal_bar_idx = -1

    def reset(self) -> None:
        self._last_signal_bar_idx = -1

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_inside": self.n_inside,
            "last_signal_bar_idx": self._last_signal_bar_idx,
        }

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        recent: List[ExtendedBarFeatures] = ctx.recent_features
        if len(recent) < self.n_inside:
            return None
        tail = recent[-self.n_inside :]
        if not all(b.is_inside_bar for b in tail):
            return None

        ai = ctx.structure.always_in
        if ai not in ("long", "short"):
            return None

        feat = ctx.feat
        if feat.bar_idx == self._last_signal_bar_idx:
            return None
        self._last_signal_bar_idx = feat.bar_idx

        ii_high = max(b.high for b in tail)
        ii_low = min(b.low for b in tail)

        if ai == "long":
            entry = ii_high + _TICK
            stop = ii_low - _TICK
            return PatternSignal(
                detector=self.name,
                side="long",
                signal_bar_idx=feat.bar_idx,
                entry_px=entry,
                stop_px=stop,
                timestamp_ns=feat.timestamp_ns,
                reason=f"{self.n_inside}-inside-bar compression in always_in=long",
                metadata={
                    "n_inside": self.n_inside,
                    "ii_high": ii_high,
                    "ii_low": ii_low,
                    "first_inside_idx": tail[0].bar_idx,
                },
            )

        entry = ii_low - _TICK
        stop = ii_high + _TICK
        return PatternSignal(
            detector=self.name,
            side="short",
            signal_bar_idx=feat.bar_idx,
            entry_px=entry,
            stop_px=stop,
            timestamp_ns=feat.timestamp_ns,
            reason=f"{self.n_inside}-inside-bar compression in always_in=short",
            metadata={
                "n_inside": self.n_inside,
                "ii_high": ii_high,
                "ii_low": ii_low,
                "first_inside_idx": tail[0].bar_idx,
            },
        )


@PatternRegistry.register("ii_breakout")
class IIBreakoutDetector(_InsideBarBreakoutBase):
    n_inside = 2


@PatternRegistry.register("iii_breakout")
class IIIBreakoutDetector(_InsideBarBreakoutBase):
    n_inside = 3
