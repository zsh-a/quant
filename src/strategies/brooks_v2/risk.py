"""Risk model for BrooksStrategyV2.

Responsibilities:
  * Position sizing: ``qty = equity * risk_pct / |entry - stop|``
  * After fill: track current position + stop + 1R-partial flag
  * On each bar: (a) partial-close 50% when unrealized ≥ 1R and move stop to
    breakeven; (b) trail stop behind latest confirmed swing point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from src.analysis.brooks.features import SwingPoint


@dataclass
class PositionState:
    symbol: str
    side: Literal["long", "short"]
    entry_px: float
    stop_px: float
    qty_initial: float
    qty_open: float
    one_r: float  # |entry - initial_stop|
    partial_taken: bool = False
    breakeven_moved: bool = False
    trailing_stop_px: Optional[float] = None
    highest_since_entry: Optional[float] = None
    lowest_since_entry: Optional[float] = None
    entry_bar_idx: int = -1
    entry_timestamp_ns: int = 0
    used_swing_lows: List[int] = field(default_factory=list)
    used_swing_highs: List[int] = field(default_factory=list)


@dataclass
class ManagementAction:
    """A single management order the strategy should submit."""

    symbol: str
    order_type: Literal["buy", "sell", "buy_to_cover", "sell_short"]
    quantity: float
    reason: str
    new_stop_px: Optional[float] = None
    is_partial: bool = False


class BrooksRiskModel:
    def __init__(
        self,
        risk_pct: float = 0.005,
        partial_close_1r: bool = True,
        partial_fraction: float = 0.5,
        swing_trail: bool = True,
        max_concurrent: int = 1,
    ):
        self.risk_pct = risk_pct
        self.partial_close_1r = partial_close_1r
        self.partial_fraction = partial_fraction
        self.swing_trail = swing_trail
        self.max_concurrent = max_concurrent

    # ---- sizing --------------------------------------------------------

    def sizing(self, equity: float, entry_px: float, stop_px: float, available_cash: Optional[float] = None) -> float:
        risk_usd = equity * self.risk_pct
        per_unit = abs(entry_px - stop_px)
        if per_unit <= 0 or risk_usd <= 0 or entry_px <= 0:
            return 0.0
        qty = risk_usd / per_unit
        # Cap by available cash (no leverage in spot backtest)
        cash_cap = (available_cash if available_cash is not None else equity) / entry_px
        return min(qty, cash_cap)

    def can_open(self, open_positions: int) -> bool:
        return open_positions < self.max_concurrent

    # ---- management per bar -------------------------------------------

    def manage(
        self,
        pos: PositionState,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        confirmed_swing_highs: List[SwingPoint],
        confirmed_swing_lows: List[SwingPoint],
    ) -> List[ManagementAction]:
        actions: List[ManagementAction] = []
        if pos.side == "long":
            pos.highest_since_entry = max(pos.highest_since_entry or bar_high, bar_high)
            unrealized_r = (bar_close - pos.entry_px) / pos.one_r if pos.one_r > 0 else 0.0
            if self.partial_close_1r and not pos.partial_taken and unrealized_r >= 1.0:
                partial_qty = pos.qty_initial * self.partial_fraction
                actions.append(
                    ManagementAction(
                        symbol=pos.symbol,
                        order_type="sell",
                        quantity=partial_qty,
                        reason=f"1R partial close ({unrealized_r:.2f}R)",
                        is_partial=True,
                    )
                )
                pos.partial_taken = True
                pos.qty_open -= partial_qty
                pos.stop_px = pos.entry_px  # move to breakeven
                pos.breakeven_moved = True
                actions.append(
                    ManagementAction(
                        symbol=pos.symbol,
                        order_type="sell",  # not real — marker
                        quantity=0.0,
                        reason="move stop to breakeven",
                        new_stop_px=pos.entry_px,
                    )
                )

            if self.swing_trail:
                for s in confirmed_swing_lows:
                    if s.bar_idx <= pos.entry_bar_idx:
                        continue
                    if s.bar_idx in pos.used_swing_lows:
                        continue
                    if s.price > pos.stop_px:
                        pos.used_swing_lows.append(s.bar_idx)
                        pos.stop_px = s.price
                        actions.append(
                            ManagementAction(
                                symbol=pos.symbol,
                                order_type="sell",
                                quantity=0.0,
                                reason=f"trail stop to swing-low @ bar {s.bar_idx} ({s.price:.4f})",
                                new_stop_px=s.price,
                            )
                        )
        else:  # short
            pos.lowest_since_entry = min(pos.lowest_since_entry or bar_low, bar_low)
            unrealized_r = (pos.entry_px - bar_close) / pos.one_r if pos.one_r > 0 else 0.0
            if self.partial_close_1r and not pos.partial_taken and unrealized_r >= 1.0:
                partial_qty = pos.qty_initial * self.partial_fraction
                actions.append(
                    ManagementAction(
                        symbol=pos.symbol,
                        order_type="buy_to_cover",
                        quantity=partial_qty,
                        reason=f"1R partial close ({unrealized_r:.2f}R)",
                        is_partial=True,
                    )
                )
                pos.partial_taken = True
                pos.qty_open -= partial_qty
                pos.stop_px = pos.entry_px
                pos.breakeven_moved = True
                actions.append(
                    ManagementAction(
                        symbol=pos.symbol,
                        order_type="buy_to_cover",
                        quantity=0.0,
                        reason="move stop to breakeven",
                        new_stop_px=pos.entry_px,
                    )
                )

            if self.swing_trail:
                for s in confirmed_swing_highs:
                    if s.bar_idx <= pos.entry_bar_idx:
                        continue
                    if s.bar_idx in pos.used_swing_highs:
                        continue
                    if s.price < pos.stop_px:
                        pos.used_swing_highs.append(s.bar_idx)
                        pos.stop_px = s.price
                        actions.append(
                            ManagementAction(
                                symbol=pos.symbol,
                                order_type="buy_to_cover",
                                quantity=0.0,
                                reason=f"trail stop to swing-high @ bar {s.bar_idx} ({s.price:.4f})",
                                new_stop_px=s.price,
                            )
                        )
        return actions

    def check_stop_hit(self, pos: PositionState, bar_high: float, bar_low: float) -> Optional[ManagementAction]:
        """Close the full remaining qty if the bar has crossed the stop."""
        if pos.side == "long" and bar_low <= pos.stop_px and pos.qty_open > 0:
            return ManagementAction(
                symbol=pos.symbol,
                order_type="sell",
                quantity=pos.qty_open,
                reason=f"stop hit @ {pos.stop_px:.4f}",
            )
        if pos.side == "short" and bar_high >= pos.stop_px and pos.qty_open > 0:
            return ManagementAction(
                symbol=pos.symbol,
                order_type="buy_to_cover",
                quantity=pos.qty_open,
                reason=f"stop hit @ {pos.stop_px:.4f}",
            )
        return None
