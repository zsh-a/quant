"""Measured Move target calculator.

Not a signal generator — a helper that projects a target price from the most
recent impulse leg, used by the risk model for optional TP placement.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from .base import DetectorContext


class MeasuredMoveTargeter:
    name: str = "measured_move"

    def state_snapshot(self) -> Dict[str, Any]:
        return {"detector": self.name}

    def reset(self) -> None:  # noqa: D401 — no state
        pass

    def on_bar(self, ctx: DetectorContext):  # pragma: no cover — not a signal detector
        return None

    def target(self, ctx: DetectorContext, side: Literal["long", "short"], entry_px: float) -> Optional[float]:
        """Return measured-move TP for the given side, or ``None`` if no impulse visible."""
        recent = ctx.recent_features
        if len(recent) < 6:
            return None
        # Simple heuristic: take the most recent leg (same direction as entry)
        # and project its length from entry_px.
        if side == "long":
            # Find the last local low in recent_features
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
