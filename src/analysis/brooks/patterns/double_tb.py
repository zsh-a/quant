"""Double Top / Double Bottom Twin detectors.

A "double bottom" in Brooks terms is a pair of confirmed swing lows close
enough in price that the second retest forms a signal bar. Entry = second
bottom's bar_high + 1 tick, stop = min(lows) - 1 tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from .base import DetectorContext, PatternSignal

_TICK = 1e-6


@dataclass
class _DoubleState:
    last_signal_bar_idx: int = -1


class _DoubleTwinBase:
    name: str
    side: Literal["long", "short"]

    def __init__(self, max_separation_atr: float = 0.5, min_separation_bars: int = 3, max_separation_bars: int = 30):
        self.max_separation_atr = max_separation_atr
        self.min_separation_bars = min_separation_bars
        self.max_separation_bars = max_separation_bars
        self._state = _DoubleState()

    def state_snapshot(self) -> Dict[str, Any]:
        return {"detector": self.name, "last_signal_bar_idx": self._state.last_signal_bar_idx}

    def reset(self) -> None:
        self._state = _DoubleState()


class DoubleBottomDetector(_DoubleTwinBase):
    name: str = "double_bottom"
    side: Literal["long", "short"] = "long"

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        lows = [s for s in ctx.structure.confirmed_swing_lows if s.confirmed_at_idx <= ctx.feat.bar_idx]
        if len(lows) < 2:
            return None
        a, b = lows[-2], lows[-1]
        sep = b.bar_idx - a.bar_idx
        if sep < self.min_separation_bars or sep > self.max_separation_bars:
            return None
        atr = ctx.feat.atr14
        if atr <= 0:
            return None
        if abs(a.price - b.price) > self.max_separation_atr * atr:
            return None
        # Current bar must be a bullish reaction bar after confirmed second low.
        if not (ctx.feat.is_bull and ctx.feat.close > ctx.feat.open):
            return None
        if b.bar_idx == self._state.last_signal_bar_idx:
            return None
        self._state.last_signal_bar_idx = b.bar_idx
        entry = ctx.feat.high + _TICK
        stop = min(a.price, b.price) - _TICK
        return PatternSignal(
            detector=self.name,
            side="long",
            signal_bar_idx=ctx.feat.bar_idx,
            entry_px=entry,
            stop_px=stop,
            timestamp_ns=ctx.feat.timestamp_ns,
            reason=f"double-bottom: swings at {a.bar_idx}={a.price:.4f} & {b.bar_idx}={b.price:.4f}",
            metadata={"low1_idx": a.bar_idx, "low1_px": a.price, "low2_idx": b.bar_idx, "low2_px": b.price},
        )


class DoubleTopDetector(_DoubleTwinBase):
    name: str = "double_top"
    side: Literal["long", "short"] = "short"

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        highs = [s for s in ctx.structure.confirmed_swing_highs if s.confirmed_at_idx <= ctx.feat.bar_idx]
        if len(highs) < 2:
            return None
        a, b = highs[-2], highs[-1]
        sep = b.bar_idx - a.bar_idx
        if sep < self.min_separation_bars or sep > self.max_separation_bars:
            return None
        atr = ctx.feat.atr14
        if atr <= 0:
            return None
        if abs(a.price - b.price) > self.max_separation_atr * atr:
            return None
        if not (not ctx.feat.is_bull and ctx.feat.close < ctx.feat.open):
            return None
        if b.bar_idx == self._state.last_signal_bar_idx:
            return None
        self._state.last_signal_bar_idx = b.bar_idx
        entry = ctx.feat.low - _TICK
        stop = max(a.price, b.price) + _TICK
        return PatternSignal(
            detector=self.name,
            side="short",
            signal_bar_idx=ctx.feat.bar_idx,
            entry_px=entry,
            stop_px=stop,
            timestamp_ns=ctx.feat.timestamp_ns,
            reason=f"double-top: swings at {a.bar_idx}={a.price:.4f} & {b.bar_idx}={b.price:.4f}",
            metadata={"high1_idx": a.bar_idx, "high1_px": a.price, "high2_idx": b.bar_idx, "high2_px": b.price},
        )
