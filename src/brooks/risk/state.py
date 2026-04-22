"""Shared position state for the Brooks risk layer.

``PositionState`` holds everything the risk components (sizer, stop ladder,
time stop, portfolio guard) need to decide what to do with an open position.
The stop ladder walks a position through four stages:

    initial → break_even → partial_trail → swing_trail

Stage transitions are one-way; the stored ``stop_px`` never retreats (monotonic
up for longs, monotonic down for shorts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

LadderStage = Literal["initial", "break_even", "partial_trail", "swing_trail"]


@dataclass
class PositionState:
    """Mutable state for one open position, threaded through the risk layer."""

    symbol: str
    side: Literal["long", "short"]
    entry_px: float
    stop_px: float
    qty_initial: float
    qty_open: float
    one_r: float  # |entry - initial_stop|

    # Risk contribution (fraction of equity at risk when the trade opened).
    risk_pct: float = 0.0

    # Entry bookkeeping.
    entry_bar_idx: int = -1
    entry_timestamp_ns: int = 0

    # Stop ladder state.
    ladder_stage: LadderStage = "initial"
    partial_taken: bool = False
    breakeven_moved: bool = False
    max_unrealized_r: float = 0.0
    highest_since_entry: Optional[float] = None
    lowest_since_entry: Optional[float] = None

    # Ring buffers for the partial_trail stage (bar-level extrema).
    recent_lows: List[float] = field(default_factory=list)
    recent_highs: List[float] = field(default_factory=list)

    # Swing-point de-duplication for the swing_trail stage.
    used_swing_lows: List[int] = field(default_factory=list)
    used_swing_highs: List[int] = field(default_factory=list)

    def unrealized_r(self, bar_close: float) -> float:
        if self.one_r <= 0:
            return 0.0
        if self.side == "long":
            return (bar_close - self.entry_px) / self.one_r
        return (self.entry_px - bar_close) / self.one_r
