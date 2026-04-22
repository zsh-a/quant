"""Major Trend Reversal (MTR) detectors.

Brooks' two-step reversal: a *lower-high after a higher-low* (bull→bear)
or *higher-low after a lower-high* (bear→bull), confirmed by a strong
trend bar in the new direction that breaks the most recent intervening
swing extreme.

We do **not** verify the wedge/climax precondition here — that grading
is left to the rule-analyst layer. Detection is structural:

* ``mtr_short``: confirmed swings show ``H[-1] < H[-2]`` (lower-high)
  and the current bar is a bear trend bar that closes below ``L[-1]``
  (the most recent swing low).
* ``mtr_long``: mirror — ``L[-1] > L[-2]`` (higher-low) and the current
  bar is a bull trend bar closing above ``H[-1]``.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from src.brooks.patterns.base import DetectorContext, PatternDetector, PatternSignal
from src.brooks.patterns.registry import PatternRegistry

_TICK = 1e-6


class _MTRBase(PatternDetector):
    side: Literal["long", "short"]

    def __init__(self, max_swing_age: int = 20):
        self.max_swing_age = max_swing_age
        self._last_signal_bar_idx = -1

    def reset(self) -> None:
        self._last_signal_bar_idx = -1

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "last_signal_bar_idx": self._last_signal_bar_idx,
        }


@PatternRegistry.register("mtr_short")
class MTRShortDetector(_MTRBase):
    """Bull → bear MTR: lower-high + strong bear bar breaks recent swing low."""

    side: Literal["long", "short"] = "short"

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        feat = ctx.feat
        if feat.is_bull or not feat.is_trend:
            return None
        highs = [s for s in ctx.structure.confirmed_swing_highs if s.confirmed_at_idx <= feat.bar_idx]
        lows = [s for s in ctx.structure.confirmed_swing_lows if s.confirmed_at_idx <= feat.bar_idx]
        if len(highs) < 2 or len(lows) < 1:
            return None

        h_prev, h_recent = highs[-2], highs[-1]
        l_recent = lows[-1]

        if h_recent.price >= h_prev.price:
            return None
        if h_recent.bar_idx <= l_recent.bar_idx:
            return None
        if feat.close >= l_recent.price:
            return None
        if feat.bar_idx - h_recent.bar_idx > self.max_swing_age:
            return None
        if feat.bar_idx == self._last_signal_bar_idx:
            return None

        self._last_signal_bar_idx = feat.bar_idx
        entry = feat.low - _TICK
        stop = h_recent.price + _TICK
        return PatternSignal(
            detector=self.name,
            side="short",
            signal_bar_idx=feat.bar_idx,
            entry_px=entry,
            stop_px=stop,
            timestamp_ns=feat.timestamp_ns,
            reason=(
                f"MTR short: lower-high {h_recent.price:.4f}<{h_prev.price:.4f} + "
                f"break below swing-low {l_recent.price:.4f}"
            ),
            metadata={
                "lower_high_idx": h_recent.bar_idx,
                "lower_high_px": h_recent.price,
                "prior_high_idx": h_prev.bar_idx,
                "prior_high_px": h_prev.price,
                "broken_low_idx": l_recent.bar_idx,
                "broken_low_px": l_recent.price,
            },
        )


@PatternRegistry.register("mtr_long")
class MTRLongDetector(_MTRBase):
    """Bear → bull MTR: higher-low + strong bull bar breaks recent swing high."""

    side: Literal["long", "short"] = "long"

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        feat = ctx.feat
        if not feat.is_bull or not feat.is_trend:
            return None
        highs = [s for s in ctx.structure.confirmed_swing_highs if s.confirmed_at_idx <= feat.bar_idx]
        lows = [s for s in ctx.structure.confirmed_swing_lows if s.confirmed_at_idx <= feat.bar_idx]
        if len(lows) < 2 or len(highs) < 1:
            return None

        l_prev, l_recent = lows[-2], lows[-1]
        h_recent = highs[-1]

        if l_recent.price <= l_prev.price:
            return None
        if l_recent.bar_idx <= h_recent.bar_idx:
            return None
        if feat.close <= h_recent.price:
            return None
        if feat.bar_idx - l_recent.bar_idx > self.max_swing_age:
            return None
        if feat.bar_idx == self._last_signal_bar_idx:
            return None

        self._last_signal_bar_idx = feat.bar_idx
        entry = feat.high + _TICK
        stop = l_recent.price - _TICK
        return PatternSignal(
            detector=self.name,
            side="long",
            signal_bar_idx=feat.bar_idx,
            entry_px=entry,
            stop_px=stop,
            timestamp_ns=feat.timestamp_ns,
            reason=(
                f"MTR long: higher-low {l_recent.price:.4f}>{l_prev.price:.4f} + "
                f"break above swing-high {h_recent.price:.4f}"
            ),
            metadata={
                "higher_low_idx": l_recent.bar_idx,
                "higher_low_px": l_recent.price,
                "prior_low_idx": l_prev.bar_idx,
                "prior_low_px": l_prev.price,
                "broken_high_idx": h_recent.bar_idx,
                "broken_high_px": h_recent.price,
            },
        )
