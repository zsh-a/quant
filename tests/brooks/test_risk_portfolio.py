"""Tests for src.brooks.risk.portfolio_guard."""

from __future__ import annotations

import pytest

from src.brooks.risk.portfolio_guard import PortfolioGuard
from src.brooks.risk.state import PositionState


def _open_pos(symbol: str, risk_pct: float = 0.01) -> PositionState:
    return PositionState(
        symbol=symbol,
        side="long",
        entry_px=100.0,
        stop_px=99.0,
        qty_initial=1.0,
        qty_open=1.0,
        one_r=1.0,
        risk_pct=risk_pct,
    )


class TestDailyRisk:
    def test_allows_when_under_cap(self):
        guard = PortfolioGuard(max_daily_risk_pct=0.03)
        ok, _ = guard.can_open(
            new_symbol="ETHUSDT",
            new_risk_pct=0.01,
            open_positions={"BTCUSDT": _open_pos("BTCUSDT", 0.01)},
        )
        assert ok is True

    def test_rejects_when_over_cap(self):
        guard = PortfolioGuard(max_daily_risk_pct=0.03)
        ok, reason = guard.can_open(
            new_symbol="ETHUSDT",
            new_risk_pct=0.02,
            open_positions={
                "BTCUSDT": _open_pos("BTCUSDT", 0.01),
                "SOLUSDT": _open_pos("SOLUSDT", 0.015),
            },
        )
        assert ok is False
        assert "daily risk" in reason

    def test_accepts_at_exact_cap(self):
        guard = PortfolioGuard(max_daily_risk_pct=0.03)
        ok, _ = guard.can_open(
            new_symbol="ETHUSDT",
            new_risk_pct=0.015,
            open_positions={"BTCUSDT": _open_pos("BTCUSDT", 0.015)},
        )
        assert ok is True

    def test_closed_positions_dont_count(self):
        guard = PortfolioGuard(max_daily_risk_pct=0.02)
        closed = _open_pos("BTCUSDT", 0.02)
        closed.qty_open = 0.0
        ok, _ = guard.can_open(
            new_symbol="ETHUSDT",
            new_risk_pct=0.015,
            open_positions={"BTCUSDT": closed},
        )
        assert ok is True


class TestSymbolCap:
    def test_default_rejects_second_position_in_same_symbol(self):
        guard = PortfolioGuard(max_daily_risk_pct=0.1, max_symbol_positions=1)
        ok, reason = guard.can_open(
            new_symbol="BTCUSDT",
            new_risk_pct=0.01,
            open_positions={"BTCUSDT": _open_pos("BTCUSDT", 0.01)},
        )
        assert ok is False
        assert "BTCUSDT" in reason

    def test_larger_cap_allows_multiple(self):
        guard = PortfolioGuard(max_daily_risk_pct=0.1, max_symbol_positions=2)
        ok, _ = guard.can_open(
            new_symbol="BTCUSDT",
            new_risk_pct=0.01,
            open_positions={"BTCUSDT": _open_pos("BTCUSDT", 0.01)},
        )
        assert ok is True


class TestCorrelationGroups:
    def test_rejects_conflicting_group(self):
        guard = PortfolioGuard(
            max_daily_risk_pct=0.1,
            max_symbol_positions=5,
            correlation_groups={"layer1": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]},
        )
        ok, reason = guard.can_open(
            new_symbol="ETHUSDT",
            new_risk_pct=0.01,
            open_positions={"BTCUSDT": _open_pos("BTCUSDT", 0.01)},
        )
        assert ok is False
        assert "layer1" in reason

    def test_allows_when_no_group_conflict(self):
        guard = PortfolioGuard(
            max_daily_risk_pct=0.1,
            correlation_groups={
                "layer1": ["BTCUSDT", "ETHUSDT"],
                "defi": ["UNIUSDT", "AAVEUSDT"],
            },
        )
        ok, _ = guard.can_open(
            new_symbol="UNIUSDT",
            new_risk_pct=0.01,
            open_positions={"BTCUSDT": _open_pos("BTCUSDT", 0.01)},
        )
        assert ok is True

    def test_new_symbol_not_in_any_group_passes_group_check(self):
        guard = PortfolioGuard(
            max_daily_risk_pct=0.1,
            correlation_groups={"layer1": ["BTCUSDT", "ETHUSDT"]},
        )
        ok, _ = guard.can_open(
            new_symbol="DOGEUSDT",
            new_risk_pct=0.01,
            open_positions={"BTCUSDT": _open_pos("BTCUSDT", 0.01)},
        )
        assert ok is True


class TestValidation:
    def test_rejects_negative_new_risk(self):
        guard = PortfolioGuard()
        ok, _ = guard.can_open("ETHUSDT", -0.01, {})
        assert ok is False

    def test_rejects_invalid_construction(self):
        with pytest.raises(ValueError):
            PortfolioGuard(max_daily_risk_pct=-0.01)
        with pytest.raises(ValueError):
            PortfolioGuard(max_symbol_positions=0)
