"""3-push wedge detector.

Buy side (bullish wedge reversal): 3 sequential *lower* swing lows whose line
is flattening (decreasing gap between successive pushes), with swing highs
between them forming a downward channel. The third-push low forms the signal.
Sell side mirrors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from .base import DetectorContext, PatternSignal

_TICK = 1e-6


@dataclass
class _WedgeState:
    last_signal_bar_idx: int = -1


class _WedgeBase:
    name: str
    side: Literal["long", "short"]

    def __init__(self, min_separation_bars: int = 2, max_total_bars: int = 60):
        self.min_separation_bars = min_separation_bars
        self.max_total_bars = max_total_bars
        self._state = _WedgeState()

    def state_snapshot(self) -> Dict[str, Any]:
        return {"detector": self.name, "last_signal_bar_idx": self._state.last_signal_bar_idx}

    def reset(self) -> None:
        self._state = _WedgeState()


class WedgeDetector(_WedgeBase):
    """Buy-side 3-push wedge bottom. Use ``side='short'`` via constructor flag."""

    name: str = "wedge"
    side: Literal["long", "short"] = "long"

    def __init__(self, direction: Literal["long", "short"] = "long", **kwargs):
        super().__init__(**kwargs)
        self.side = direction
        self.name = "wedge_bottom" if direction == "long" else "wedge_top"

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        if self.side == "long":
            pivots = [s for s in ctx.structure.confirmed_swing_lows if s.confirmed_at_idx <= ctx.feat.bar_idx]
        else:
            pivots = [s for s in ctx.structure.confirmed_swing_highs if s.confirmed_at_idx <= ctx.feat.bar_idx]

        if len(pivots) < 3:
            return None
        p1, p2, p3 = pivots[-3:]
        if p3.bar_idx - p1.bar_idx > self.max_total_bars:
            return None
        if p2.bar_idx - p1.bar_idx < self.min_separation_bars:
            return None
        if p3.bar_idx - p2.bar_idx < self.min_separation_bars:
            return None

        if self.side == "long":
            # 3 successively lower lows with contracting gaps
            if not (p1.price > p2.price > p3.price):
                return None
            if (p2.price - p3.price) >= (p1.price - p2.price):
                # Not contracting — normal downtrend, not a wedge
                return None
            # Signal bar: current bar must close bullish and above p3
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
        else:
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
