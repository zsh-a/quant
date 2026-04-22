"""H1 / H3 / H4 pullback detectors.

In a sustained ``always_in == long`` run, Brooks numbers each successive
pullback (a bear leg ended by a bull recovery bar) starting at 1. The
*Nth* pullback's recovery bar is the **signal bar** for an HN entry:

* **H1** — first pullback; lighter probability than H2 because it lacks
  a confirmed second touch, but valid in tight bull channels.
* **H3** — third pullback; often forms a wedge bull flag, lower
  probability because the leg is older.
* **H4** — fourth pullback; usually marks trend exhaustion. The signal
  is still emitted (with ``metadata.quality == "low"``) so downstream
  rules can either downweight it or flip to L1 short on failure.

Each detector fires *at most once per always_in run* — when ``always_in``
flips away from long the internal counter resets, so the next bull
breakout can start counting from H1 again.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from src.brooks.patterns.base import DetectorContext, PatternDetector, PatternSignal
from src.brooks.patterns.registry import PatternRegistry

_TICK = 1e-6


class _HnDetector(PatternDetector):
    """Shared FSM for H1/H3/H4. Subclasses set :attr:`target_pullback_count`."""

    side: Literal["long", "short"] = "long"
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

        if ai != "long":
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
        return not feat.is_bull and feat.body_pct >= 30

    def _is_with(self, feat) -> bool:
        return feat.is_bull and feat.body_pct >= 30

    def _make_signal(self, ctx: DetectorContext) -> PatternSignal:
        feat = ctx.feat
        entry = feat.high + _TICK
        stop = feat.low - _TICK
        return PatternSignal(
            detector=self.name,
            side="long",
            signal_bar_idx=feat.bar_idx,
            entry_px=entry,
            stop_px=stop,
            timestamp_ns=feat.timestamp_ns,
            reason=(f"{self.name.upper()} formed: pullback #{self.target_pullback_count} completed in always_in=long"),
            metadata={
                "pullback_count": self.target_pullback_count,
                "quality": self.quality,
            },
        )


@PatternRegistry.register("h1")
class H1Detector(_HnDetector):
    target_pullback_count = 1


@PatternRegistry.register("h3")
class H3Detector(_HnDetector):
    target_pullback_count = 3


@PatternRegistry.register("h4")
class H4Detector(_HnDetector):
    target_pullback_count = 4
    quality = "low"
