"""Signal aggregation with optional confluence gating.

Operates on the unified :class:`~src.brooks.schema.Signal`. The Trader's
Equation evaluator and risk model layer downstream are responsible for
turning the aggregated output into a full :class:`~src.brooks.schema.Decision`;
this layer's contract is only "combine signals → conservative entry/stop".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from src.brooks.schema import Signal


@dataclass
class AggregatedDecision:
    """Output of :class:`SignalAggregator` — a confluence-gated combination."""

    side: Literal["long", "short"]
    signal_bar_idx: int
    entry_px: float
    stop_px: float
    timestamp_ns: int
    hit_detectors: List[str]
    reasons: List[str]
    raw_signals: List[Signal] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "signal_bar_idx": self.signal_bar_idx,
            "entry_px": round(self.entry_px, 6),
            "stop_px": round(self.stop_px, 6),
            "timestamp_ns": self.timestamp_ns,
            "hit_detectors": list(self.hit_detectors),
            "reasons": list(self.reasons),
            "raw_signals": [s.model_dump() for s in self.raw_signals],
        }


class SignalAggregator:
    """Combine multiple analysts'/detectors' signals.

    Parameters
    ----------
    confluence_n:
        Minimum # of signals agreeing on the same side for a bar to
        produce a decision. ``1`` = any-of, ``≥2`` = confluence.
    """

    def __init__(self, confluence_n: int = 1):
        if confluence_n < 1:
            raise ValueError("confluence_n must be >= 1")
        self.confluence_n = confluence_n

    def resolve(self, signals: List[Signal]) -> Optional[AggregatedDecision]:
        signals = [s for s in signals if s is not None]
        if not signals:
            return None
        longs = [s for s in signals if s.side == "long"]
        shorts = [s for s in signals if s.side == "short"]

        pick: Optional[List[Signal]] = None
        if len(longs) >= self.confluence_n and len(longs) >= len(shorts):
            pick = longs
        elif len(shorts) >= self.confluence_n:
            pick = shorts
        if pick is None:
            return None

        side: Literal["long", "short"] = pick[0].side
        # Conservative entry: highest stop-price for long (latest breakout level),
        # lowest for short. Conservative stop: tightest distance from entry.
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
            timestamp_ns=_latest_timestamp(pick),
            hit_detectors=[s.pattern for s in pick],
            reasons=[s.reasoning for s in pick],
            raw_signals=list(pick),
        )


def _latest_timestamp(signals: List[Signal]) -> int:
    """Pull ``timestamp_ns`` from ``Signal.meta`` (where the rule analyst
    stores it); fall back to ``0`` if no signal carries one."""
    last = 0
    for s in signals:
        ts = s.meta.get("timestamp_ns") if s.meta else None
        if isinstance(ts, int) and ts > last:
            last = ts
    return last
