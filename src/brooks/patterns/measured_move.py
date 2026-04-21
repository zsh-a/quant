"""Measured Move target calculator.

Not a signal generator — a helper that projects a target price from the
most recent impulse leg, used by the risk model for optional TP
placement. ``on_bar`` always returns ``None``; call :meth:`target`
directly for a projected TP.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from src.brooks.patterns.base import DetectorContext, PatternDetector, PatternSignal
from src.brooks.patterns.registry import PatternRegistry


@PatternRegistry.register("measured_move")
class MeasuredMoveTargeter(PatternDetector):
    def state_snapshot(self) -> Dict[str, Any]:
        return {"name": self.name}

    def reset(self) -> None:  # noqa: D401 — no state
        pass

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        return None

    def target(
        self,
        ctx: DetectorContext,
        side: Literal["long", "short"],
        entry_px: float,
    ) -> Optional[float]:
        """Return measured-move TP for the given side, or ``None`` if no impulse visible."""
        recent = ctx.recent_features
        if len(recent) < 6:
            return None
        if side == "long":
            lows = [(b.bar_idx, b.low) for b in recent]
            if not lows:
                return None
            low_idx = min(range(len(lows)), key=lambda i: lows[i][1])
            high_after = max(b.high for b in recent[low_idx:])
            leg = high_after - lows[low_idx][1]
            if leg <= 0:
                return None
            return entry_px + leg
        else:
            highs = [(b.bar_idx, b.high) for b in recent]
            if not highs:
                return None
            high_idx = max(range(len(highs)), key=lambda i: highs[i][1])
            low_after = min(b.low for b in recent[high_idx:])
            leg = highs[high_idx][1] - low_after
            if leg <= 0:
                return None
            return entry_px - leg
