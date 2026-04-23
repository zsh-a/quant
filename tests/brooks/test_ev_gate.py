"""EVGate tests — filters signals by expected_r and promotes to Decision."""

from __future__ import annotations

import pytest

from src.brooks.decision.ev_gate import EVGate
from src.brooks.decision.hit_rate import HitRateTable
from src.brooks.decision.trader_equation import TraderEquation
from src.brooks.schema import Decision, Signal


def _sig(
    *,
    pattern: str = "h2",
    side: str = "long",
    entry: float = 100.0,
    stop: float = 99.0,
    target: float | None = 102.0,
    probability: float = 0.5,
    source: str | None = None,
) -> Signal:
    return Signal(
        pattern=pattern,
        side=side,  # type: ignore[arg-type]
        signal_bar_idx=10,
        entry_px=entry,
        stop_px=stop,
        target_px=target,
        probability=probability,
        quality=0.5,
        source=source or f"rule:{pattern}",
    )


def _gate(min_expected_r: float = 0.1) -> EVGate:
    return EVGate(TraderEquation(HitRateTable.empty(), cost_r=0.05), min_expected_r=min_expected_r)


def test_filters_out_low_ev_signals():
    gate = _gate(min_expected_r=0.1)
    # p=0.3, reward=2R, cost=0.05 → E = 0.3*2 - 0.7 - 0.05 = -0.15 < 0.1 → dropped
    low = _sig(probability=0.3, target=102.0)
    # p=0.7, reward=2R, cost=0.05 → E = 1.4 - 0.3 - 0.05 = 1.05 >= 0.1 → kept
    high = _sig(pattern="wedge_long", probability=0.7, target=102.0, source="rule:wedge_long")
    out = gate.filter([low, high], regime="x", htf_aligned=True, symbol="BTCUSDT")
    assert len(out) == 1
    assert isinstance(out[0], Decision)
    assert out[0].source == "rule:wedge_long"


def test_boundary_signal_equal_to_min_is_kept():
    """E == min_expected_r is accepted (only strictly-less signals are dropped)."""
    # p=0.5, reward=2R, cost=0.0 → E = 1.0 - 0.5 = 0.5; min=0.5 → keep.
    sig = _sig(probability=0.5, target=102.0)
    gate_no_cost = EVGate(TraderEquation(HitRateTable.empty(), cost_r=0.0), min_expected_r=0.5)
    out = gate_no_cost.filter([sig], regime="x", htf_aligned=False, symbol="BTCUSDT")
    assert len(out) == 1
    assert out[0].expected_r == pytest.approx(0.5)


def test_decision_carries_probability_expected_r_regime():
    gate = _gate(min_expected_r=-1.0)  # accept everything
    sig = _sig(probability=0.6, target=102.0)
    [dec] = gate.filter([sig], regime="strong_bull_trend", htf_aligned=True, symbol="ETHUSDT")
    assert dec.symbol == "ETHUSDT"
    assert dec.side == "long"
    assert dec.entry_px == 100.0
    assert dec.stop_px == 99.0
    assert dec.target_px == 102.0
    assert dec.probability == pytest.approx(0.6)
    # 0.6*2 - 0.4 - 0.05 = 0.75
    assert dec.expected_r == pytest.approx(0.75)
    assert dec.regime == "strong_bull_trend"
    assert dec.htf_aligned is True
    assert dec.signals == [sig]
    assert "p=0.60" in dec.reasoning and "E=0.75" in dec.reasoning


def test_missing_target_uses_default_reward_r():
    gate = EVGate(
        TraderEquation(HitRateTable.empty(), cost_r=0.0, default_reward_r=2.0),
        min_expected_r=-1.0,
    )
    sig = _sig(probability=0.5, target=None, entry=100.0, stop=99.0)
    [dec] = gate.filter([sig], regime="x", htf_aligned=False, symbol="X")
    # synthesized target = entry + 2R = 100 + 2*1 = 102
    assert dec.target_px == pytest.approx(102.0)


def test_missing_target_short_side_synthesizes_downside_target():
    gate = EVGate(
        TraderEquation(HitRateTable.empty(), cost_r=0.0, default_reward_r=2.0),
        min_expected_r=-1.0,
    )
    sig = _sig(side="short", probability=0.5, target=None, entry=100.0, stop=101.0)
    [dec] = gate.filter([sig], regime="x", htf_aligned=False, symbol="X")
    assert dec.target_px == pytest.approx(98.0)


def test_empty_signal_list_returns_empty():
    gate = _gate()
    assert gate.filter([], regime="x", htf_aligned=True, symbol="S") == []


def test_none_entries_are_skipped():
    gate = _gate(min_expected_r=-1.0)
    sig = _sig(probability=0.6, target=102.0)
    out = gate.filter([None, sig, None], regime="x", htf_aligned=True, symbol="S")  # type: ignore[list-item]
    assert len(out) == 1


def test_gate_replaces_min_rr_hard_threshold():
    """An E-based gate keeps high-p low-RR trades that a pure min_rr=2 would reject."""
    # High win rate (0.80), reward=1.5R → legacy min_rr=2 gate would reject.
    # E = 0.80*1.5 - 0.20 - 0.05 = 0.95 → EV gate accepts.
    gate = _gate(min_expected_r=0.1)
    sig = _sig(probability=0.80, entry=100.0, stop=99.0, target=101.5)
    out = gate.filter([sig], regime="x", htf_aligned=False, symbol="S")
    assert len(out) == 1 and out[0].expected_r == pytest.approx(0.95)
