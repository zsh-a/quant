"""Base ABC and shared dataclasses for Brooks pattern detectors.

``PatternSignal`` is the **internal** detector output type. The
``src/brooks/analyst/rule.py`` layer (Phase 2.5) is responsible for
promoting it to the unified :class:`src.brooks.schema.Signal`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from src.brooks.features import ExtendedBarFeatures
from src.brooks.structure import MarketStructure

if TYPE_CHECKING:  # pragma: no cover — imported only for typing to avoid cycles
    from src.brooks.context import TFSnapshot


@dataclass
class DetectorContext:
    """Everything a detector needs from the outer layers.

    ``htf`` carries the most recent higher-timeframe snapshots keyed by
    interval (e.g. ``"1h"``). Detectors may consult it to annotate or
    gate signals, but the default implementations only use the primary
    LTF features and structure — HTF awareness is opt-in.
    """

    feat: ExtendedBarFeatures
    structure: MarketStructure
    recent_features: List[ExtendedBarFeatures]
    params: Dict[str, Any] = field(default_factory=dict)
    htf: Dict[str, "TFSnapshot"] = field(default_factory=dict)


@dataclass
class PatternSignal:
    """A single-bar buy/sell setup.

    ``signal_bar_idx`` is the bar that *triggered* the setup. Entry
    actually happens on the next bar via a stop-at-``entry_px`` working
    order (Brooks' standard "1 tick above/below signal bar" convention).
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


class PatternDetector(ABC):
    """Stateful per-bar detector.

    Subclasses register themselves via
    :meth:`src.brooks.patterns.registry.PatternRegistry.register`;
    the decorator assigns :attr:`name` to the registry key.
    """

    name: str = ""

    @abstractmethod
    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        """Ingest one bar; return a signal on the bar it forms, else None."""

    def state_snapshot(self) -> Dict[str, Any]:
        return {"name": self.name}
