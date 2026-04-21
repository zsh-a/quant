"""L2 — Market structure built on top of confirmed swings.

Tracks the Brooks-style *structural* state of the market:

- Always-in long/short/neutral (bull/bear breakout of prior N-bar extremes)
- Current leg direction and start index
- Latest confirmed swing-high and swing-low (for stop-placement, trailing)
- Micro-channel fit on the most recent leg (least-squares on extremes)

All state is *confirmed-as-of now*: only swings whose ``confirmed_at_idx <= now``
are consulted, eliminating look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from .features import BarFeatureExtractor, ExtendedBarFeatures, SwingPoint


@dataclass
class ChannelFit:
    """Linear channel slope/intercept on a leg."""

    slope: float
    intercept: float
    start_idx: int
    end_idx: int
    kind: Literal["top", "bottom"]

    def project(self, idx: int) -> float:
        return self.slope * idx + self.intercept

    def to_dict(self) -> dict:
        return {
            "slope": round(self.slope, 8),
            "intercept": round(self.intercept, 6),
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "kind": self.kind,
        }


@dataclass
class MarketStructure:
    """Structural view updated once per bar."""

    current_idx: int = -1
    ema20: float = 0.0
    atr14: float = 0.0

    confirmed_swing_highs: List[SwingPoint] = field(default_factory=list)
    confirmed_swing_lows: List[SwingPoint] = field(default_factory=list)

    always_in: Literal["long", "short", "neutral"] = "neutral"
    breakout_state: Literal["bull_breakout", "bear_breakout", "none"] = "none"

    current_leg_dir: Literal["up", "down", "flat"] = "flat"
    current_leg_start_idx: int = 0
    current_leg_length: int = 0

    micro_channel_top: Optional[ChannelFit] = None
    micro_channel_bot: Optional[ChannelFit] = None

    last_close: float = 0.0
    last_high: float = 0.0
    last_low: float = 0.0
    last_breakout_lookback_high: float = 0.0
    last_breakout_lookback_low: float = 0.0

    def to_dict(self) -> dict:
        return {
            "current_idx": self.current_idx,
            "ema20": round(self.ema20, 6),
            "atr14": round(self.atr14, 6),
            "always_in": self.always_in,
            "breakout_state": self.breakout_state,
            "current_leg_dir": self.current_leg_dir,
            "current_leg_start_idx": self.current_leg_start_idx,
            "current_leg_length": self.current_leg_length,
            "last_swing_high": self.confirmed_swing_highs[-1].to_dict() if self.confirmed_swing_highs else None,
            "last_swing_low": self.confirmed_swing_lows[-1].to_dict() if self.confirmed_swing_lows else None,
            "micro_channel_top": self.micro_channel_top.to_dict() if self.micro_channel_top else None,
            "micro_channel_bot": self.micro_channel_bot.to_dict() if self.micro_channel_bot else None,
        }


class MarketStructureTracker:
    """Consumes ``ExtendedBarFeatures`` output from ``BarFeatureExtractor``.

    Responsibilities:
      * maintain ``MarketStructure`` snapshot
      * compute always-in via N-bar breakout (:attr:`breakout_lookback`)
      * fit micro-channels on the current leg
    """

    def __init__(self, extractor: BarFeatureExtractor, breakout_lookback: int = 20):
        self._ext = extractor
        self.breakout_lookback = breakout_lookback
        self.state = MarketStructure()
        self._high_history: List[float] = []
        self._low_history: List[float] = []

    def on_features(self, feat: ExtendedBarFeatures) -> MarketStructure:
        self.state.current_idx = feat.bar_idx
        self.state.ema20 = feat.ema20
        self.state.atr14 = feat.atr14
        self.state.last_close = feat.close
        self.state.last_high = feat.high
        self.state.last_low = feat.low

        self._high_history.append(feat.high)
        self._low_history.append(feat.low)
        if len(self._high_history) > self.breakout_lookback * 4:
            # keep the working window a bit larger than lookback
            self._high_history = self._high_history[-self.breakout_lookback * 2 :]
            self._low_history = self._low_history[-self.breakout_lookback * 2 :]

        self.state.current_leg_dir = feat.leg_dir
        self.state.current_leg_length = feat.leg_length
        self.state.current_leg_start_idx = feat.bar_idx - feat.leg_length + 1

        # Snapshot confirmed swings visible as of this bar.
        current = feat.bar_idx
        confirmed = [s for s in self._ext.confirmed_swings if s.confirmed_at_idx <= current]
        self.state.confirmed_swing_highs = [s for s in confirmed if s.kind == "high"]
        self.state.confirmed_swing_lows = [s for s in confirmed if s.kind == "low"]

        self._update_always_in(feat)
        self._fit_micro_channels(feat)
        return self.state

    # ---- always-in resolver --------------------------------------------

    def _update_always_in(self, feat: ExtendedBarFeatures) -> None:
        N = self.breakout_lookback
        if len(self._high_history) <= N:
            return
        prior_high = max(self._high_history[-N - 1 : -1])
        prior_low = min(self._low_history[-N - 1 : -1])
        self.state.last_breakout_lookback_high = prior_high
        self.state.last_breakout_lookback_low = prior_low

        if feat.close > prior_high:
            self.state.always_in = "long"
            self.state.breakout_state = "bull_breakout"
        elif feat.close < prior_low:
            self.state.always_in = "short"
            self.state.breakout_state = "bear_breakout"
        else:
            # Don't flip on chop — hold prior always_in decision.
            self.state.breakout_state = "none"

    # ---- channel fit ---------------------------------------------------

    def _fit_micro_channels(self, feat: ExtendedBarFeatures) -> None:
        start = self.state.current_leg_start_idx
        end = feat.bar_idx
        if end - start < 3:
            self.state.micro_channel_top = None
            self.state.micro_channel_bot = None
            return

        n = end - start + 1
        xs = list(range(start, end + 1))
        tail = n
        highs = self._high_history[-tail:]
        lows = self._low_history[-tail:]

        self.state.micro_channel_top = _least_squares(xs, highs, start, end, "top")
        self.state.micro_channel_bot = _least_squares(xs, lows, start, end, "bottom")


def _least_squares(
    xs: List[int], ys: List[float], start: int, end: int, kind: Literal["top", "bottom"]
) -> Optional[ChannelFit]:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return ChannelFit(slope=slope, intercept=intercept, start_idx=start, end_idx=end, kind=kind)
