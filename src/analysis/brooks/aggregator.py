"""Signal aggregation with optional confluence gating."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from .patterns.base import PatternSignal


@dataclass
class AggregatedDecision:
    side: Literal["long", "short"]
    signal_bar_idx: int
    entry_px: float
    stop_px: float
    timestamp_ns: int
    hit_detectors: List[str]
    reasons: List[str]
    raw_signals: List[PatternSignal] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "signal_bar_idx": self.signal_bar_idx,
            "entry_px": round(self.entry_px, 6),
            "stop_px": round(self.stop_px, 6),
            "timestamp_ns": self.timestamp_ns,
            "hit_detectors": self.hit_detectors,
            "reasons": self.reasons,
            "raw_signals": [s.to_dict() for s in self.raw_signals],
        }


class SignalAggregator:
    """Combine multiple detectors' signals.

    Parameters
    ----------
    confluence_n:
        Minimum # of detectors agreeing on the same side for a bar to produce
        a decision. 1 = any-of, ≥2 = confluence.
    """

    def __init__(self, confluence_n: int = 1):
        if confluence_n < 1:
            raise ValueError("confluence_n must be >= 1")
        self.confluence_n = confluence_n

    def resolve(self, signals: List[PatternSignal]) -> Optional[AggregatedDecision]:
        signals = [s for s in signals if s is not None]
        if not signals:
            return None
        longs = [s for s in signals if s.side == "long"]
        shorts = [s for s in signals if s.side == "short"]

        pick: Optional[List[PatternSignal]] = None
        if len(longs) >= self.confluence_n and len(longs) >= len(shorts):
            pick = longs
        elif len(shorts) >= self.confluence_n:
            pick = shorts
        if pick is None:
            return None

        side: Literal["long", "short"] = pick[0].side
        # Entry: most conservative (highest for long, lowest for short)
        if side == "long":
            entry = max(s.entry_px for s in pick)
            stop = min(s.stop_px for s in pick)
        else:
            entry = min(s.entry_px for s in pick)
            stop = max(s.stop_px for s in pick)

        return AggregatedDecision(
            side=side,
            signal_bar_idx=max(s.signal_bar_idx for s in pick),
            entry_px=entry,
            stop_px=stop,
            timestamp_ns=pick[-1].timestamp_ns,
            hit_detectors=[s.detector for s in pick],
            reasons=[s.reason for s in pick],
            raw_signals=list(pick),
        )
