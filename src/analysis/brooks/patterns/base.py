"""Base Protocol and shared dataclasses for L3 pattern detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from ..features import ExtendedBarFeatures
from ..structure import MarketStructure


@dataclass
class DetectorContext:
    """Everything a detector needs from the outer layers."""

    feat: ExtendedBarFeatures
    structure: MarketStructure
    recent_features: List[ExtendedBarFeatures]  # most recent last; typically last ~60 bars
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternSignal:
    """A single-bar buy/sell setup.

    ``signal_bar_idx`` is the bar that *triggered* the setup. Entry actually
    happens on the next bar via a stop-at-``entry_px`` working order (Brooks'
    standard "1 tick above/below signal bar" convention).
    """

    detector: str
    side: Literal["long", "short"]
    signal_bar_idx: int
    entry_px: float
    stop_px: float
    timestamp_ns: int
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "detector": self.detector,
            "side": self.side,
            "signal_bar_idx": self.signal_bar_idx,
            "entry_px": round(self.entry_px, 6),
            "stop_px": round(self.stop_px, 6),
            "timestamp_ns": self.timestamp_ns,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@runtime_checkable
class PatternDetector(Protocol):
    """Stateful per-bar detector."""

    name: str

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        """Ingest one bar; return a signal on the bar it forms, else None."""
        ...

    def state_snapshot(self) -> Dict[str, Any]:
        """Return a JSON-safe snapshot of internal state for decision logs."""
        ...
