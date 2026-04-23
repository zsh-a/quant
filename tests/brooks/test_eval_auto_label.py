"""Tests for :class:`AutoLabeler`.

The auto-labeler couples a deterministic rule analyst with stochastic LLM
analysts and emits silver samples via majority vote. We replace both
sides with controllable :class:`MockAnalyst` instances so the tests can
assert the consensus mechanics without spinning up real detectors or
networking.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List

import pytest

from src.brooks.context import Bar, BrooksContext
from src.brooks.eval.auto_label import AutoLabeler
from src.brooks.schema import Signal

# ---------------------------------------------------------------------------
# Test analyst implementation
# ---------------------------------------------------------------------------


@dataclass
class MockAnalyst:
    """Programmable analyst used by the auto-label tests."""

    name: str
    fire_at_indices: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)

    async def analyze(self, ctx: BrooksContext) -> List[Signal]:
        idx = len(ctx.primary.bars) - 1
        self.calls.append(idx)
        spec = self.fire_at_indices.get(idx)
        if spec is None:
            return []
        return [
            Signal(
                pattern=spec["pattern"],
                side=spec["side"],
                signal_bar_idx=idx,
                entry_px=spec["entry"],
                stop_px=spec["stop"],
                target_px=spec.get("target"),
                probability=spec.get("probability", 0.5),
                quality=spec.get("quality", 0.5),
                reasoning=spec.get("reasoning", f"{self.name}@{idx}"),
                source=f"mock:{self.name}",
            )
        ]


def _bars(n: int = 30, base: float = 100.0) -> List[Bar]:
    return [
        Bar(
            timestamp_ns=1_700_000_000_000_000_000 + i * 60_000_000_000,
            open=base + i,
            high=base + i + 1,
            low=base + i - 0.5,
            close=base + i + 0.5,
            volume=1.0,
        )
        for i in range(n)
    ]


def _setup(idx: int, **overrides) -> dict:
    base = dict(pattern="h2", side="long", entry=110.0, stop=109.0, target=112.0)
    base.update(overrides)
    return base


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_requires_rule_analyst():
    with pytest.raises(ValueError):
        AutoLabeler(rule_analyst=None)  # type: ignore[arg-type]


def test_min_agreement_lower_bound():
    with pytest.raises(ValueError):
        AutoLabeler(rule_analyst=MockAnalyst(name="rule"), min_agreement=0)


# ---------------------------------------------------------------------------
# Consensus mechanics
# ---------------------------------------------------------------------------


def test_two_of_three_agreement_emits_silver_sample():
    rule = MockAnalyst(name="rule", fire_at_indices={25: _setup(25)})
    llm_a = MockAnalyst(name="llm_a", fire_at_indices={25: _setup(25)})
    llm_b = MockAnalyst(name="llm_b", fire_at_indices={25: _setup(25, side="short", entry=99.0, stop=100.0)})

    labeler = AutoLabeler(
        rule_analyst=rule,
        llm_analysts=[llm_a, llm_b],
        min_agreement=2,
        min_bars_for_label=20,
    )
    samples = _run(labeler.label(bars=_bars(28), symbol="BTC", interval="5m"))

    # Only the (h2, long) bucket has 2 voters; the lone short bucket is dropped.
    assert len(samples) == 1
    sample = samples[0]
    assert sample.expected_pattern == "h2"
    assert sample.expected_side == "long"
    assert sample.target_bar_idx == 25
    assert sample.source == "silver"
    assert sample.meta["agreement"] == 2
    assert "rule" in sample.meta["voters"]
    assert "llm_a" in sample.meta["voters"]


def test_below_min_agreement_emits_nothing():
    rule = MockAnalyst(name="rule", fire_at_indices={25: _setup(25)})
    llm_a = MockAnalyst(name="llm_a")  # silent
    llm_b = MockAnalyst(name="llm_b")  # silent
    labeler = AutoLabeler(
        rule_analyst=rule,
        llm_analysts=[llm_a, llm_b],
        min_agreement=2,
        min_bars_for_label=20,
    )
    samples = _run(labeler.label(bars=_bars(28), symbol="BTC", interval="5m"))
    assert samples == []


def test_min_agreement_one_emits_rule_only():
    rule = MockAnalyst(name="rule", fire_at_indices={25: _setup(25)})
    labeler = AutoLabeler(
        rule_analyst=rule,
        llm_analysts=[],
        min_agreement=1,
        min_bars_for_label=20,
    )
    samples = _run(labeler.label(bars=_bars(28), symbol="BTC", interval="5m"))
    assert len(samples) == 1
    assert samples[0].meta["agreement"] == 1


def test_entry_tolerance_rejects_wide_spreads():
    """Two analysts agree on (h2, long) but with very different entries → reject."""
    rule = MockAnalyst(name="rule", fire_at_indices={25: _setup(25, entry=100.0, stop=99.0)})
    llm_a = MockAnalyst(name="llm_a", fire_at_indices={25: _setup(25, entry=120.0, stop=119.0)})
    labeler = AutoLabeler(
        rule_analyst=rule,
        llm_analysts=[llm_a],
        min_agreement=2,
        entry_tolerance=0.01,
        min_bars_for_label=20,
    )
    samples = _run(labeler.label(bars=_bars(28), symbol="BTC", interval="5m"))
    assert samples == []


def test_entry_tolerance_zero_allows_any_spread():
    rule = MockAnalyst(name="rule", fire_at_indices={25: _setup(25, entry=100.0, stop=99.0)})
    llm_a = MockAnalyst(name="llm_a", fire_at_indices={25: _setup(25, entry=120.0, stop=119.0)})
    labeler = AutoLabeler(
        rule_analyst=rule,
        llm_analysts=[llm_a],
        min_agreement=2,
        entry_tolerance=0.0,
        min_bars_for_label=20,
    )
    samples = _run(labeler.label(bars=_bars(28), symbol="BTC", interval="5m"))
    assert len(samples) == 1


def test_skips_bars_before_warmup():
    rule = MockAnalyst(name="rule", fire_at_indices={i: _setup(i) for i in range(0, 10)})
    labeler = AutoLabeler(
        rule_analyst=rule,
        llm_analysts=[],
        min_agreement=1,
        min_bars_for_label=20,
    )
    samples = _run(labeler.label(bars=_bars(15), symbol="BTC", interval="5m"))
    assert samples == []
    # The labeler must not even invoke the analyst before the warmup boundary.
    assert rule.calls == []


def test_strongest_bucket_wins_when_multiple_agree():
    """When two distinct (pattern, side) groups both reach the threshold,
    the larger group is chosen for the silver sample."""
    setup_long = _setup(25, pattern="h2", side="long", entry=110.0, stop=109.0)
    setup_short = _setup(25, pattern="l2", side="short", entry=109.0, stop=110.0)

    rule = MockAnalyst(name="rule", fire_at_indices={25: setup_long})
    a = MockAnalyst(name="a", fire_at_indices={25: setup_long})
    b = MockAnalyst(name="b", fire_at_indices={25: setup_long})
    c = MockAnalyst(name="c", fire_at_indices={25: setup_short})
    d = MockAnalyst(name="d", fire_at_indices={25: setup_short})

    labeler = AutoLabeler(
        rule_analyst=rule,
        llm_analysts=[a, b, c, d],
        min_agreement=2,
        min_bars_for_label=20,
    )
    samples = _run(labeler.label(bars=_bars(28), symbol="BTC", interval="5m"))
    assert len(samples) == 1
    # The 3-vote (h2, long) bucket beats the 2-vote (l2, short) bucket.
    assert samples[0].expected_pattern == "h2"
    assert samples[0].meta["agreement"] == 3


def test_consensus_uses_median_entry_and_stop():
    rule = MockAnalyst(name="rule", fire_at_indices={25: _setup(25, entry=100.0, stop=99.0)})
    a = MockAnalyst(name="a", fire_at_indices={25: _setup(25, entry=100.4, stop=99.2)})
    b = MockAnalyst(name="b", fire_at_indices={25: _setup(25, entry=100.2, stop=99.1)})
    labeler = AutoLabeler(
        rule_analyst=rule,
        llm_analysts=[a, b],
        min_agreement=2,
        entry_tolerance=0.05,
        min_bars_for_label=20,
    )
    samples = _run(labeler.label(bars=_bars(28), symbol="BTC", interval="5m"))
    assert len(samples) == 1
    s = samples[0]
    assert s.expected_entry == pytest.approx(100.2)
    assert s.expected_stop == pytest.approx(99.1)


def test_regime_is_populated_from_classifier():
    rule = MockAnalyst(name="rule", fire_at_indices={25: _setup(25)})
    a = MockAnalyst(name="a", fire_at_indices={25: _setup(25)})
    labeler = AutoLabeler(
        rule_analyst=rule,
        llm_analysts=[a],
        min_agreement=2,
        min_bars_for_label=20,
    )
    samples = _run(labeler.label(bars=_bars(28), symbol="BTC", interval="5m"))
    assert len(samples) == 1
    # The synthetic strictly-up bars resolve to one of the bull regimes.
    assert "bull" in samples[0].regime or samples[0].regime in {
        "breakout_mode",
        "climax",
        "weak_bull_trend",
        "strong_bull_trend",
        "unknown",
    }
