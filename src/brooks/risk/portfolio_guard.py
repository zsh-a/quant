"""Portfolio-level entry gate.

Three limits applied before a new position opens:

1. Max aggregate daily risk — sum of ``risk_pct`` across open positions plus
   the candidate's ``new_risk_pct`` must not exceed ``max_daily_risk_pct``.
2. Max concurrent positions per symbol.
3. Correlation groups — each group names a set of symbols that share a risk
   factor (e.g. same sector, stable/coin basket); only one open position per
   group is allowed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.brooks.risk.state import PositionState


class PortfolioGuard:
    def __init__(
        self,
        max_daily_risk_pct: float = 0.03,
        max_symbol_positions: int = 1,
        correlation_groups: Optional[Dict[str, List[str]]] = None,
    ):
        if max_daily_risk_pct < 0:
            raise ValueError("max_daily_risk_pct must be non-negative")
        if max_symbol_positions < 1:
            raise ValueError("max_symbol_positions must be >= 1")
        self.max_daily_risk_pct = max_daily_risk_pct
        self.max_symbol_positions = max_symbol_positions
        self.correlation_groups: Dict[str, List[str]] = correlation_groups or {}

    def can_open(
        self,
        new_symbol: str,
        new_risk_pct: float,
        open_positions: Dict[str, PositionState],
    ) -> Tuple[bool, str]:
        """Check all gates. Returns (ok, reason)."""
        if new_risk_pct < 0:
            return False, "new_risk_pct must be non-negative"

        active = [p for p in open_positions.values() if p.qty_open > 0]

        symbol_count = sum(1 for p in active if p.symbol == new_symbol)
        if symbol_count >= self.max_symbol_positions:
            return (
                False,
                f"symbol {new_symbol} already has {symbol_count} open position(s) "
                f"(max {self.max_symbol_positions})",
            )

        daily_risk = sum(p.risk_pct for p in active) + new_risk_pct
        if daily_risk > self.max_daily_risk_pct:
            return (
                False,
                f"daily risk {daily_risk:.4f} would exceed cap {self.max_daily_risk_pct:.4f}",
            )

        conflict_group = self._find_correlation_conflict(new_symbol, active)
        if conflict_group is not None:
            return (
                False,
                f"correlation group {conflict_group!r} already has an open position",
            )

        return True, "ok"

    def _find_correlation_conflict(
        self, new_symbol: str, active: List[PositionState]
    ) -> Optional[str]:
        for group_name, members in self.correlation_groups.items():
            if new_symbol not in members:
                continue
            if any(p.symbol in members for p in active):
                return group_name
        return None
