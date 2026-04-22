"""Position sizing.

Two concrete implementations plus a ``Sizer`` protocol:

* :class:`FixedPercentSizer` — risk a fixed fraction of equity per trade
  (default 0.5%). This matches the legacy ``BrooksRiskModel`` behaviour.
* :class:`KellySizer` — derive the risk fraction from Kelly's criterion,
  ``f* = p - (1 - p) / b`` where ``b = reward / risk``. A ``fraction``
  multiplier (default 0.5 → half-Kelly) tempers full Kelly, and the final
  value is clipped to ``[0, max_risk_pct]``.

Both compute ``qty = (equity * risk_pct) / |entry - stop|`` and cap by
``available_cash / entry_px`` (no leverage on the spot side).
"""

from __future__ import annotations

from typing import Protocol


class Sizer(Protocol):
    def size(
        self,
        equity: float,
        entry_px: float,
        stop_px: float,
        probability: float,
        expected_r: float,
        available_cash: float,
    ) -> float: ...


def _apply_cap(qty: float, entry_px: float, available_cash: float) -> float:
    if qty <= 0 or entry_px <= 0:
        return 0.0
    cash_cap = available_cash / entry_px if available_cash > 0 else 0.0
    return max(0.0, min(qty, cash_cap))


class FixedPercentSizer:
    """Risk ``risk_pct`` of equity per trade."""

    def __init__(self, risk_pct: float = 0.005):
        if risk_pct < 0:
            raise ValueError("risk_pct must be non-negative")
        self.risk_pct = risk_pct

    def size(
        self,
        equity: float,
        entry_px: float,
        stop_px: float,
        probability: float = 0.0,
        expected_r: float = 0.0,
        available_cash: float = 0.0,
    ) -> float:
        per_unit = abs(entry_px - stop_px)
        if per_unit <= 0 or equity <= 0 or self.risk_pct <= 0:
            return 0.0
        qty = (equity * self.risk_pct) / per_unit
        cash = available_cash if available_cash > 0 else equity
        return _apply_cap(qty, entry_px, cash)


class KellySizer:
    """Kelly-based sizer.

    ``f* = p - (1 - p) / b`` where ``b = reward / risk`` (``expected_r``).
    The effective risk fraction is ``clip(f* * fraction, 0, max_risk_pct)``.
    """

    def __init__(self, max_risk_pct: float = 0.02, fraction: float = 0.5):
        if max_risk_pct < 0:
            raise ValueError("max_risk_pct must be non-negative")
        if fraction <= 0:
            raise ValueError("fraction must be positive (use 0.5 for half-Kelly)")
        self.max_risk_pct = max_risk_pct
        self.fraction = fraction

    @staticmethod
    def kelly_fraction(probability: float, expected_r: float) -> float:
        """Raw Kelly ``f*`` for this (p, b)."""
        if expected_r <= 0:
            return 0.0
        return probability - (1.0 - probability) / expected_r

    def effective_risk_pct(self, probability: float, expected_r: float) -> float:
        f_star = self.kelly_fraction(probability, expected_r)
        adj = f_star * self.fraction
        if adj <= 0:
            return 0.0
        return min(adj, self.max_risk_pct)

    def size(
        self,
        equity: float,
        entry_px: float,
        stop_px: float,
        probability: float,
        expected_r: float,
        available_cash: float = 0.0,
    ) -> float:
        per_unit = abs(entry_px - stop_px)
        if per_unit <= 0 or equity <= 0:
            return 0.0
        risk_pct = self.effective_risk_pct(probability, expected_r)
        if risk_pct <= 0:
            return 0.0
        qty = (equity * risk_pct) / per_unit
        cash = available_cash if available_cash > 0 else equity
        return _apply_cap(qty, entry_px, cash)
