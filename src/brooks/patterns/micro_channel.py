"""Micro-channel failure detectors.

A *micro channel* is a tight one-sided sequence where every bar's
extreme advances the prior bar's extreme in the trend direction. Brooks
treats the *first* break of the channel as a high-probability reversal
setup ("failed micro channel").

Channel state is sourced from
:attr:`MarketStructure.micro_channel_top` /
:attr:`MarketStructure.micro_channel_bot`, which are least-squares fits
on the current leg's highs/lows.

Because a leg flip (the very break bar we want to detect) **resets** the
channel fit to ``None``, each detector keeps a one-bar snapshot of the
last known channel — the break is evaluated against that snapshot
*before* the snapshot is refreshed from the current bar.

* ``micro_channel_short`` — bull micro channel
  (``micro_channel_top.slope > 0``) of length ≥ ``min_channel_bars``
  fails when the next bar closes below the channel's projected lower
  bound. Emits a short signal.
* ``micro_channel_long`` — mirror for bear micro channels.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from src.brooks.patterns.base import DetectorContext, PatternDetector, PatternSignal
from src.brooks.patterns.registry import PatternRegistry
from src.brooks.structure import ChannelFit

_TICK = 1e-6


class _MicroChannelBase(PatternDetector):
    side: Literal["long", "short"]

    def __init__(self, min_channel_bars: int = 5):
        self.min_channel_bars = min_channel_bars
        self._snap_top: Optional[ChannelFit] = None
        self._snap_bot: Optional[ChannelFit] = None
        self._snap_len: int = 0
        self._last_signal_bar_idx = -1

    def reset(self) -> None:
        self._snap_top = None
        self._snap_bot = None
        self._snap_len = 0
        self._last_signal_bar_idx = -1

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "snap_len": self._snap_len,
            "has_snapshot": self._snap_top is not None and self._snap_bot is not None,
            "last_signal_bar_idx": self._last_signal_bar_idx,
        }


@PatternRegistry.register("micro_channel_short")
class MicroChannelShortDetector(_MicroChannelBase):
    """Bull micro-channel break to the downside → short setup."""

    side: Literal["long", "short"] = "short"

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        feat = ctx.feat

        if (
            self._snap_top is not None
            and self._snap_bot is not None
            and self._snap_len >= self.min_channel_bars
        ):
            projected_bot = self._snap_bot.project(feat.bar_idx)
            if (
                feat.close < projected_bot
                and not feat.is_bull
                and feat.body_pct >= 30
                and feat.bar_idx != self._last_signal_bar_idx
            ):
                self._last_signal_bar_idx = feat.bar_idx
                snap_len = self._snap_len
                top_slope = self._snap_top.slope
                bot_slope = self._snap_bot.slope
                self._snap_top = None
                self._snap_bot = None
                self._snap_len = 0
                return PatternSignal(
                    detector=self.name,
                    side="short",
                    signal_bar_idx=feat.bar_idx,
                    entry_px=feat.low - _TICK,
                    stop_px=feat.high + _TICK,
                    timestamp_ns=feat.timestamp_ns,
                    reason=(
                        f"failed bull micro-channel ({snap_len} bars, slope={top_slope:.4f}): "
                        f"close {feat.close:.4f} broke below projected bot {projected_bot:.4f}"
                    ),
                    metadata={
                        "channel_len": snap_len,
                        "channel_top_slope": top_slope,
                        "channel_bot_slope": bot_slope,
                        "projected_bot": projected_bot,
                    },
                )

        top = ctx.structure.micro_channel_top
        bot = ctx.structure.micro_channel_bot
        if top is not None and bot is not None and top.slope > 0:
            self._snap_top = top
            self._snap_bot = bot
            self._snap_len = top.end_idx - top.start_idx + 1
        else:
            self._snap_top = None
            self._snap_bot = None
            self._snap_len = 0
        return None


@PatternRegistry.register("micro_channel_long")
class MicroChannelLongDetector(_MicroChannelBase):
    """Bear micro-channel break to the upside → long setup."""

    side: Literal["long", "short"] = "long"

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        feat = ctx.feat

        if (
            self._snap_top is not None
            and self._snap_bot is not None
            and self._snap_len >= self.min_channel_bars
        ):
            projected_top = self._snap_top.project(feat.bar_idx)
            if (
                feat.close > projected_top
                and feat.is_bull
                and feat.body_pct >= 30
                and feat.bar_idx != self._last_signal_bar_idx
            ):
                self._last_signal_bar_idx = feat.bar_idx
                snap_len = self._snap_len
                top_slope = self._snap_top.slope
                bot_slope = self._snap_bot.slope
                self._snap_top = None
                self._snap_bot = None
                self._snap_len = 0
                return PatternSignal(
                    detector=self.name,
                    side="long",
                    signal_bar_idx=feat.bar_idx,
                    entry_px=feat.high + _TICK,
                    stop_px=feat.low - _TICK,
                    timestamp_ns=feat.timestamp_ns,
                    reason=(
                        f"failed bear micro-channel ({snap_len} bars, slope={bot_slope:.4f}): "
                        f"close {feat.close:.4f} broke above projected top {projected_top:.4f}"
                    ),
                    metadata={
                        "channel_len": snap_len,
                        "channel_top_slope": top_slope,
                        "channel_bot_slope": bot_slope,
                        "projected_top": projected_top,
                    },
                )

        top = ctx.structure.micro_channel_top
        bot = ctx.structure.micro_channel_bot
        if top is not None and bot is not None and bot.slope < 0:
            self._snap_top = top
            self._snap_bot = bot
            self._snap_len = bot.end_idx - bot.start_idx + 1
        else:
            self._snap_top = None
            self._snap_bot = None
            self._snap_len = 0
        return None
