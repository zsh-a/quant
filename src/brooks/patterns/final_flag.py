"""Final Flag detector — tight range after a strong trend leg, then break
the *opposite* direction (trend exhaustion).

Rules (simplified):
  * Preceding strong leg: trend has ``leg_length >= leg_min`` bars
  * Consolidation: ``flag_min_bars`` bars of tight range (range <
    ``flag_atr_mult`` × ATR)
  * Breakout: close breaks opposite extreme of the flag range
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.brooks.patterns.base import DetectorContext, PatternDetector, PatternSignal
from src.brooks.patterns.registry import PatternRegistry

_TICK = 1e-6


@dataclass
class _FlagState:
    last_signal_bar_idx: int = -1


@PatternRegistry.register("final_flag")
class FinalFlagDetector(PatternDetector):
    def __init__(
        self,
        leg_min: int = 6,
        flag_min_bars: int = 5,
        flag_max_bars: int = 15,
        flag_atr_mult: float = 0.8,
    ):
        self.leg_min = leg_min
        self.flag_min_bars = flag_min_bars
        self.flag_max_bars = flag_max_bars
        self.flag_atr_mult = flag_atr_mult
        self._state = _FlagState()

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "last_signal_bar_idx": self._state.last_signal_bar_idx,
        }

    def reset(self) -> None:
        self._state = _FlagState()

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        feat = ctx.feat
        atr = feat.atr14
        if atr <= 0:
            return None
        recent = ctx.recent_features
        if len(recent) < self.flag_min_bars + self.leg_min + 1:
            return None

        for flag_len in range(
            self.flag_min_bars,
            min(self.flag_max_bars, len(recent) - self.leg_min - 1) + 1,
        ):
            flag_slice: List = recent[-flag_len - 1 : -1]
            if len(flag_slice) < flag_len:
                continue
            flag_high = max(b.high for b in flag_slice)
            flag_low = min(b.low for b in flag_slice)
            flag_range = flag_high - flag_low
            if flag_range > self.flag_atr_mult * atr * flag_len / self.flag_min_bars:
                continue
            if flag_range <= 0:
                continue

            pre_slice: List = recent[-flag_len - 1 - self.leg_min : -flag_len - 1]
            if len(pre_slice) < self.leg_min:
                continue
            pre_high = max(b.high for b in pre_slice)
            pre_low = min(b.low for b in pre_slice)
            pre_range = pre_high - pre_low
            if pre_range < 2 * flag_range:
                continue

            pre_trend_up = pre_slice[-1].close > pre_slice[0].close
            if pre_trend_up and feat.close < flag_low:
                if feat.bar_idx == self._state.last_signal_bar_idx:
                    return None
                self._state.last_signal_bar_idx = feat.bar_idx
                entry = feat.low - _TICK
                stop = flag_high + _TICK
                return PatternSignal(
                    detector=self.name,
                    side="short",
                    signal_bar_idx=feat.bar_idx,
                    entry_px=entry,
                    stop_px=stop,
                    timestamp_ns=feat.timestamp_ns,
                    reason=f"final flag top (len={flag_len}, range/ATR={flag_range / atr:.2f})",
                    metadata={
                        "flag_high": flag_high,
                        "flag_low": flag_low,
                        "flag_len": flag_len,
                    },
                )
            if (not pre_trend_up) and feat.close > flag_high:
                if feat.bar_idx == self._state.last_signal_bar_idx:
                    return None
                self._state.last_signal_bar_idx = feat.bar_idx
                entry = feat.high + _TICK
                stop = flag_low - _TICK
                return PatternSignal(
                    detector=self.name,
                    side="long",
                    signal_bar_idx=feat.bar_idx,
                    entry_px=entry,
                    stop_px=stop,
                    timestamp_ns=feat.timestamp_ns,
                    reason=f"final flag bottom (len={flag_len}, range/ATR={flag_range / atr:.2f})",
                    metadata={
                        "flag_high": flag_high,
                        "flag_low": flag_low,
                        "flag_len": flag_len,
                    },
                )
        return None
