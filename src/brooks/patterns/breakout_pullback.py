"""Breakout-Pullback (BP) and Failed Breakout (FBO) detectors.

A *breakout* is the bar whose close pierces the prior N-bar extreme
(``MarketStructureTracker`` flips ``breakout_state`` to
``bull_breakout`` / ``bear_breakout`` on those bars).

* **bp_long / bp_short**: after a bull/bear breakout, the *first*
  with-trend reversal bar that follows a counter-trend pullback (and
  whose close has not slipped back through the breakout level) is the
  signal bar.
* **failed_breakout**: after a breakout, the *first* bar whose close
  re-enters the prior range within ``max_lookback`` bars triggers a
  reversal signal in the opposite direction.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from src.brooks.patterns.base import DetectorContext, PatternDetector, PatternSignal
from src.brooks.patterns.registry import PatternRegistry

_TICK = 1e-6


class _BreakoutPullbackBase(PatternDetector):
    """Shared logic for bp_long / bp_short."""

    side: Literal["long", "short"]
    breakout_state_key: str  # "bull_breakout" or "bear_breakout"

    def __init__(self, max_lookback: int = 10):
        self.max_lookback = max_lookback
        self._reset()

    def _reset(self) -> None:
        self._breakout_idx = -1
        self._breakout_level = 0.0
        self._saw_pullback = False
        self._last_signal_bar_idx = -1

    def reset(self) -> None:
        self._reset()

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "breakout_idx": self._breakout_idx,
            "breakout_level": round(self._breakout_level, 6),
            "saw_pullback": self._saw_pullback,
        }


@PatternRegistry.register("bp_long")
class BreakoutPullbackLongDetector(_BreakoutPullbackBase):
    side: Literal["long", "short"] = "long"
    breakout_state_key = "bull_breakout"

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        feat = ctx.feat
        bs = ctx.structure.breakout_state

        if bs == "bull_breakout" and not self._saw_pullback:
            self._breakout_idx = feat.bar_idx
            self._breakout_level = ctx.structure.last_breakout_lookback_high
            return None

        if self._breakout_idx < 0:
            return None

        if feat.bar_idx - self._breakout_idx > self.max_lookback:
            self._reset()
            return None

        if not feat.is_bull and feat.body_pct >= 30:
            self._saw_pullback = True
            if feat.close < self._breakout_level:
                self._reset()
            return None

        if self._saw_pullback and feat.is_bull and feat.body_pct >= 30 and feat.close > self._breakout_level:
            self._last_signal_bar_idx = feat.bar_idx
            entry = feat.high + _TICK
            stop = feat.low - _TICK
            sig = PatternSignal(
                detector=self.name,
                side="long",
                signal_bar_idx=feat.bar_idx,
                entry_px=entry,
                stop_px=stop,
                timestamp_ns=feat.timestamp_ns,
                reason=(
                    f"BP long: bull breakout at bar {self._breakout_idx} "
                    f"(level={self._breakout_level:.4f}) + pullback + with-trend bar"
                ),
                metadata={
                    "breakout_idx": self._breakout_idx,
                    "breakout_level": self._breakout_level,
                },
            )
            self._reset()
            return sig

        return None


@PatternRegistry.register("bp_short")
class BreakoutPullbackShortDetector(_BreakoutPullbackBase):
    side: Literal["long", "short"] = "short"
    breakout_state_key = "bear_breakout"

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        feat = ctx.feat
        bs = ctx.structure.breakout_state

        if bs == "bear_breakout" and not self._saw_pullback:
            self._breakout_idx = feat.bar_idx
            self._breakout_level = ctx.structure.last_breakout_lookback_low
            return None

        if self._breakout_idx < 0:
            return None

        if feat.bar_idx - self._breakout_idx > self.max_lookback:
            self._reset()
            return None

        if feat.is_bull and feat.body_pct >= 30:
            self._saw_pullback = True
            if feat.close > self._breakout_level:
                self._reset()
            return None

        if self._saw_pullback and not feat.is_bull and feat.body_pct >= 30 and feat.close < self._breakout_level:
            self._last_signal_bar_idx = feat.bar_idx
            entry = feat.low - _TICK
            stop = feat.high + _TICK
            sig = PatternSignal(
                detector=self.name,
                side="short",
                signal_bar_idx=feat.bar_idx,
                entry_px=entry,
                stop_px=stop,
                timestamp_ns=feat.timestamp_ns,
                reason=(
                    f"BP short: bear breakout at bar {self._breakout_idx} "
                    f"(level={self._breakout_level:.4f}) + pullback + with-trend bar"
                ),
                metadata={
                    "breakout_idx": self._breakout_idx,
                    "breakout_level": self._breakout_level,
                },
            )
            self._reset()
            return sig

        return None


@PatternRegistry.register("failed_breakout")
class FailedBreakoutDetector(PatternDetector):
    """A breakout that reverses back into the prior range within ``max_lookback`` bars."""

    def __init__(self, max_lookback: int = 5):
        self.max_lookback = max_lookback
        self._breakout_dir: Optional[Literal["long", "short"]] = None
        self._breakout_idx = -1
        self._breakout_level = 0.0
        self._last_signal_bar_idx = -1

    def reset(self) -> None:
        self._breakout_dir = None
        self._breakout_idx = -1
        self._breakout_level = 0.0

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "breakout_dir": self._breakout_dir,
            "breakout_idx": self._breakout_idx,
            "breakout_level": round(self._breakout_level, 6),
        }

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        feat = ctx.feat
        bs = ctx.structure.breakout_state

        if self._breakout_dir is not None:
            bars_since = feat.bar_idx - self._breakout_idx
            if bars_since > self.max_lookback:
                self.reset()
            elif bars_since > 0:
                if self._breakout_dir == "long" and feat.close < self._breakout_level:
                    sig = PatternSignal(
                        detector=self.name,
                        side="short",
                        signal_bar_idx=feat.bar_idx,
                        entry_px=feat.low - _TICK,
                        stop_px=feat.high + _TICK,
                        timestamp_ns=feat.timestamp_ns,
                        reason=(
                            f"failed bull breakout: close {feat.close:.4f} re-entered range "
                            f"below {self._breakout_level:.4f} ({bars_since} bars after breakout)"
                        ),
                        metadata={
                            "breakout_idx": self._breakout_idx,
                            "breakout_level": self._breakout_level,
                            "bars_since": bars_since,
                        },
                    )
                    self._last_signal_bar_idx = feat.bar_idx
                    self.reset()
                    return sig
                if self._breakout_dir == "short" and feat.close > self._breakout_level:
                    sig = PatternSignal(
                        detector=self.name,
                        side="long",
                        signal_bar_idx=feat.bar_idx,
                        entry_px=feat.high + _TICK,
                        stop_px=feat.low - _TICK,
                        timestamp_ns=feat.timestamp_ns,
                        reason=(
                            f"failed bear breakout: close {feat.close:.4f} re-entered range "
                            f"above {self._breakout_level:.4f} ({bars_since} bars after breakout)"
                        ),
                        metadata={
                            "breakout_idx": self._breakout_idx,
                            "breakout_level": self._breakout_level,
                            "bars_since": bars_since,
                        },
                    )
                    self._last_signal_bar_idx = feat.bar_idx
                    self.reset()
                    return sig

        if bs == "bull_breakout" and self._breakout_dir != "long":
            self._breakout_dir = "long"
            self._breakout_idx = feat.bar_idx
            self._breakout_level = ctx.structure.last_breakout_lookback_high
        elif bs == "bear_breakout" and self._breakout_dir != "short":
            self._breakout_dir = "short"
            self._breakout_idx = feat.bar_idx
            self._breakout_level = ctx.structure.last_breakout_lookback_low

        return None
