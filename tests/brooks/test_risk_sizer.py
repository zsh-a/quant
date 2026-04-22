"""Tests for src.brooks.risk.sizer."""

from __future__ import annotations

import math

import pytest

from src.brooks.risk.sizer import FixedPercentSizer, KellySizer


class TestFixedPercentSizer:
    def test_basic_sizing(self):
        sizer = FixedPercentSizer(risk_pct=0.01)
        qty = sizer.size(
            equity=100_000, entry_px=100.0, stop_px=99.0,
            probability=0.6, expected_r=2.0, available_cash=100_000,
        )
        # risk_usd = 1000, per_unit = 1, qty = 1000, cash_cap = 1000 → 1000
        assert qty == pytest.approx(1000.0)

    def test_zero_risk_returns_zero(self):
        sizer = FixedPercentSizer(risk_pct=0.0)
        assert sizer.size(100_000, 100.0, 99.0, 0.5, 2.0, 100_000) == 0.0

    def test_stop_equals_entry_returns_zero(self):
        sizer = FixedPercentSizer(risk_pct=0.01)
        assert sizer.size(100_000, 100.0, 100.0, 0.5, 2.0, 100_000) == 0.0

    def test_cash_cap_binds(self):
        sizer = FixedPercentSizer(risk_pct=0.5)  # want to risk 50k on a $1 stop
        qty = sizer.size(100_000, 100.0, 99.0, 0.5, 2.0, 10_000)
        # Without cap: 50_000 / 1 = 50_000. Cash cap = 10_000 / 100 = 100.
        assert qty == pytest.approx(100.0)

    def test_rejects_negative_risk_pct(self):
        with pytest.raises(ValueError):
            FixedPercentSizer(risk_pct=-0.01)


class TestKellySizer:
    def test_kelly_fraction_formula(self):
        """f* = p - (1 - p) / b.   p=0.6, b=2 → 0.4."""
        assert KellySizer.kelly_fraction(0.6, 2.0) == pytest.approx(0.4)
        assert KellySizer.kelly_fraction(0.5, 1.0) == pytest.approx(0.0)
        assert KellySizer.kelly_fraction(0.7, 1.5) == pytest.approx(0.7 - 0.3 / 1.5)

    def test_kelly_fraction_negative_when_edge_negative(self):
        # p=0.4, b=1: f* = 0.4 - 0.6 = -0.2
        assert KellySizer.kelly_fraction(0.4, 1.0) == pytest.approx(-0.2)

    def test_kelly_fraction_zero_b_returns_zero(self):
        assert KellySizer.kelly_fraction(0.6, 0.0) == 0.0

    def test_half_kelly_uncapped(self):
        """Half-Kelly of f*=0.4 is 0.2 — larger than any sane max, so cap binds."""
        sizer = KellySizer(max_risk_pct=1.0, fraction=0.5)
        assert sizer.effective_risk_pct(0.6, 2.0) == pytest.approx(0.2)

    def test_half_kelly_capped_to_max_risk_pct(self):
        """Spec acceptance: cap to max_risk_pct=0.02."""
        sizer = KellySizer(max_risk_pct=0.02, fraction=0.5)
        assert sizer.effective_risk_pct(0.6, 2.0) == pytest.approx(0.02)

    def test_fraction_one_equals_full_kelly(self):
        sizer = KellySizer(max_risk_pct=1.0, fraction=1.0)
        assert sizer.effective_risk_pct(0.6, 2.0) == pytest.approx(0.4)

    def test_negative_edge_returns_zero_size(self):
        sizer = KellySizer(max_risk_pct=0.02, fraction=0.5)
        qty = sizer.size(
            equity=100_000, entry_px=100.0, stop_px=99.0,
            probability=0.4, expected_r=1.0, available_cash=100_000,
        )
        assert qty == 0.0

    def test_size_uses_capped_risk_pct(self):
        sizer = KellySizer(max_risk_pct=0.02, fraction=0.5)
        qty = sizer.size(
            equity=100_000, entry_px=100.0, stop_px=99.0,
            probability=0.6, expected_r=2.0, available_cash=100_000,
        )
        # effective risk_pct = 0.02 → risk_usd = 2000, per_unit = 1 → qty 2000
        # cash cap = 100_000 / 100 = 1000 → 1000 wins.
        assert qty == pytest.approx(1000.0)

    def test_size_no_cash_cap_when_unconstrained(self):
        sizer = KellySizer(max_risk_pct=0.02, fraction=0.5)
        qty = sizer.size(
            equity=100_000, entry_px=10.0, stop_px=9.0,
            probability=0.6, expected_r=2.0, available_cash=100_000,
        )
        # risk 2000 / 1 = 2000; cash cap 100_000 / 10 = 10_000 → 2000 wins
        assert qty == pytest.approx(2000.0)

    def test_zero_entry_rejected(self):
        sizer = KellySizer(max_risk_pct=0.02, fraction=0.5)
        qty = sizer.size(100_000, 0.0, -1.0, 0.6, 2.0, 100_000)
        assert qty == 0.0

    def test_rejects_invalid_construction(self):
        with pytest.raises(ValueError):
            KellySizer(max_risk_pct=-0.01)
        with pytest.raises(ValueError):
            KellySizer(fraction=0.0)

    def test_finite_qty(self):
        sizer = KellySizer(max_risk_pct=0.02, fraction=0.5)
        qty = sizer.size(100_000, 100.0, 99.0, 0.6, 2.0, 100_000)
        assert math.isfinite(qty)
