"""L1 / L3 / L4 pullback detectors — bear-side mirrors of H-series.

In an ``always_in == short`` run, each successive bull pullback is
numbered 1, 2, 3 ... ; the Nth pullback's bear recovery bar is the
signal bar for an LN entry.

* **L1** — first pullback in the bear leg.
* **L3** — third pullback; often a wedge bear flag.
* **L4** — fourth pullback; trend exhaustion candidate, marked
  ``metadata.quality == "low"``.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from src.brooks.patterns.base import DetectorContext, PatternDetector, PatternSignal
from src.brooks.patterns.registry import PatternRegistry

_TICK = 1e-6


class _LnDetector(PatternDetector):
    """Shared FSM for L1/L3/L4. Subclasses set :attr:`target_pullback_count`."""

    side: Literal["long", "short"] = "short"
    target_pullback_count: int = 1
    quality: Literal["normal", "low"] = "normal"

    def __init__(self, max_leg_bars: int = 8):
        self.max_leg_bars = max_leg_bars
        self._reset_run()
        self._last_always_in: Literal["long", "short", "neutral"] = "neutral"

    def _reset_run(self) -> None:
        self._pb_count = 0
        self._in_pullback = False
        self._pb_start_idx = -1
        self._fired = False

    def reset(self) -> None:
        self._reset_run()
        self._last_always_in = "neutral"

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pb_count": self._pb_count,
            "in_pullback": self._in_pullback,
            "fired": self._fired,
        }

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        ai = ctx.structure.always_in
        if ai != self._last_always_in:
            self._reset_run()
            self._last_always_in = ai

        if ai != "short":
            return None
        if self._fired:
            return None

        feat = ctx.feat
        if not self._in_pullback:
            if self._is_against(feat):
                self._in_pullback = True
                self._pb_start_idx = feat.bar_idx
            return None

        if self._is_with(feat):
            self._pb_count += 1
            self._in_pullback = False
            if self._pb_count == self.target_pullback_count:
                self._fired = True
                return self._make_signal(ctx)
            return None

        if self._pb_start_idx >= 0 and feat.bar_idx - self._pb_start_idx > self.max_leg_bars:
            self._in_pullback = False
        return None

    def _is_against(self, feat) -> bool:
        return feat.is_bull and feat.body_pct >= 30

    def _is_with(self, feat) -> bool:
        return not feat.is_bull and feat.body_pct >= 30

    def _make_signal(self, ctx: DetectorContext) -> PatternSignal:
        feat = ctx.feat
        entry = feat.low - _TICK
        stop = feat.high + _TICK
        return PatternSignal(
            detector=self.name,
            side="short",
            signal_bar_idx=feat.bar_idx,
            entry_px=entry,
            stop_px=stop,
            timestamp_ns=feat.timestamp_ns,
            reason=(f"{self.name.upper()} formed: pullback #{self.target_pullback_count} completed in always_in=short"),
            metadata={
                "pullback_count": self.target_pullback_count,
                "quality": self.quality,
            },
        )


@PatternRegistry.register("l1")
class L1Detector(_LnDetector):
    target_pullback_count = 1


@PatternRegistry.register("l3")
class L3Detector(_LnDetector):
    target_pullback_count = 3


@PatternRegistry.register("l4")
class L4Detector(_LnDetector):
    target_pullback_count = 4
    quality = "low"
