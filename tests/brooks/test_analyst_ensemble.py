"""Behavioral tests for ensemble analysts (Phase 4.3).

Three suites match the three plug-in analysts:

* ``test_vote``    — concurrent fan-out + consensus filter
* ``test_router``  — regime-based dispatch
* ``test_critic``  — producer/critic confirmation pipeline
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from src.brooks.analyst import (
    AnalystRegistry,
    CriticAnalyst,
    RouterAnalyst,
    VoteAnalyst,
)
from src.brooks.analyst.base import Analyst
from src.brooks.context import Bar, BrooksContext, TFSnapshot
from src.brooks.regime import BrooksRegime, RegimeSnapshot
from src.brooks.schema import Signal

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _ctx(regime: Optional[BrooksRegime] = None) -> BrooksContext:
    bars = [
        Bar(
            timestamp_ns=1_000_000_000 * (i + 1),
            open=100.0 + i,
            high=100.5 + i,
            low=99.5 + i,
            close=100.2 + i,
            volume=1.0,
        )
        for i in range(5)
    ]
    snap = TFSnapshot(interval="5m", bars=bars)
    if regime is not None:
        snap.regime = RegimeSnapshot(regime=regime, confidence=0.8)
    return BrooksContext(symbol="BTCUSDT", primary=snap)


def _signal(
    *,
    pattern: str = "h2",
    side: str = "long",
    signal_bar_idx: int = 4,
    entry_px: float = 101.0,
    stop_px: float = 99.5,
    target_px: Optional[float] = 103.0,
    probability: float = 0.6,
    quality: float = 0.7,
    reasoning: str = "",
    source: str = "x",
    meta: Optional[dict] = None,
) -> Signal:
    return Signal(
        pattern=pattern,
        side=side,
        signal_bar_idx=signal_bar_idx,
        entry_px=entry_px,
        stop_px=stop_px,
        target_px=target_px,
        probability=probability,
        quality=quality,
        reasoning=reasoning,
        source=source,
        meta=dict(meta or {}),
    )


@dataclass
class MockAnalyst:
    """In-process Analyst double — returns a fixed signal list and counts calls."""

    name: str
    signals: List[Signal] = field(default_factory=list)
    delay_s: float = 0.0
    call_count: int = 0
    last_ctx: Optional[BrooksContext] = None
    raise_exc: Optional[Exception] = None

    async def analyze(self, ctx: BrooksContext) -> List[Signal]:
        self.call_count += 1
        self.last_ctx = ctx
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.raise_exc is not None:
            raise self.raise_exc
        return [s.model_copy(deep=True) for s in self.signals]


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Registry / Protocol conformance
# ---------------------------------------------------------------------------


def test_registry_knows_all_three_ensemble_analysts() -> None:
    assert "ensemble.vote" in AnalystRegistry.all()
    assert "ensemble.router" in AnalystRegistry.all()
    assert "ensemble.critic" in AnalystRegistry.all()
    assert AnalystRegistry.get("ensemble.vote") is VoteAnalyst
    assert AnalystRegistry.get("ensemble.router") is RouterAnalyst
    assert AnalystRegistry.get("ensemble.critic") is CriticAnalyst


def test_ensemble_analysts_satisfy_analyst_protocol() -> None:
    a = MockAnalyst(name="a", signals=[])
    b = MockAnalyst(name="b", signals=[])
    assert isinstance(VoteAnalyst([a, b]), Analyst)
    assert isinstance(RouterAnalyst({}, default=a), Analyst)
    assert isinstance(CriticAnalyst(producer=a, critic=b), Analyst)


# ---------------------------------------------------------------------------
# test_vote
# ---------------------------------------------------------------------------


class TestVote:
    def test_two_analysts_agree_one_signal_kept(self) -> None:
        a1 = MockAnalyst("a1", [_signal(probability=0.6, quality=0.7, source="a1")])
        a2 = MockAnalyst("a2", [_signal(probability=0.8, quality=0.9, source="a2")])
        vote = VoteAnalyst([a1, a2], min_agree_count=2)
        out = _run(vote.analyze(_ctx()))
        assert len(out) == 1
        sig = out[0]
        assert sig.pattern == "h2"
        assert sig.side == "long"
        assert sig.source == "ensemble.vote"
        # equal weights → simple averages
        assert sig.probability == pytest.approx(0.7)
        assert sig.quality == pytest.approx(0.8)
        assert sig.meta["votes"] == 2
        assert sorted(sig.meta["voters"]) == ["a1", "a2"]

    def test_disagreement_drops_signal(self) -> None:
        # Two analysts each emit a signal but on different patterns —
        # neither group meets the 2-vote consensus floor.
        a1 = MockAnalyst("a1", [_signal(pattern="h2", source="a1")])
        a2 = MockAnalyst("a2", [_signal(pattern="l2", side="short", entry_px=99.0, stop_px=101.0, source="a2")])
        vote = VoteAnalyst([a1, a2], min_agree_count=2)
        out = _run(vote.analyze(_ctx()))
        assert out == []

    def test_signals_within_one_bar_idx_are_grouped(self) -> None:
        a1 = MockAnalyst("a1", [_signal(signal_bar_idx=4, source="a1")])
        a2 = MockAnalyst("a2", [_signal(signal_bar_idx=5, source="a2")])  # +1 bar
        vote = VoteAnalyst([a1, a2], min_agree_count=2)
        out = _run(vote.analyze(_ctx()))
        assert len(out) == 1
        assert out[0].signal_bar_idx in (4, 5)

    def test_signals_more_than_one_bar_apart_are_separate_groups(self) -> None:
        a1 = MockAnalyst("a1", [_signal(signal_bar_idx=2, source="a1")])
        a2 = MockAnalyst("a2", [_signal(signal_bar_idx=4, source="a2")])  # gap = 2
        vote = VoteAnalyst([a1, a2], min_agree_count=2)
        out = _run(vote.analyze(_ctx()))
        assert out == []

    def test_weights_are_applied_to_probability_average(self) -> None:
        a1 = MockAnalyst("a1", [_signal(probability=0.4, quality=0.5, source="a1")])
        a2 = MockAnalyst("a2", [_signal(probability=0.9, quality=0.5, source="a2")])
        vote = VoteAnalyst([a1, a2], weights={"a1": 1.0, "a2": 3.0}, min_agree_count=2)
        out = _run(vote.analyze(_ctx()))
        assert len(out) == 1
        # (1*0.4 + 3*0.9) / 4 = 3.1/4 = 0.775
        assert out[0].probability == pytest.approx(0.775)

    def test_min_agree_three_requires_three_distinct_voters(self) -> None:
        a1 = MockAnalyst("a1", [_signal(source="a1")])
        a2 = MockAnalyst("a2", [_signal(source="a2")])
        vote = VoteAnalyst([a1, a2], min_agree_count=3)
        out = _run(vote.analyze(_ctx()))
        assert out == []

    def test_concurrent_dispatch_runs_analysts_in_parallel(self) -> None:
        # Two analysts each sleep 100ms — gather should keep total < 200ms.
        # A serial implementation would take ~200ms.
        a1 = MockAnalyst("a1", [_signal(source="a1")], delay_s=0.1)
        a2 = MockAnalyst("a2", [_signal(source="a2")], delay_s=0.1)
        vote = VoteAnalyst([a1, a2], min_agree_count=2)

        import time

        t0 = time.perf_counter()
        out = _run(vote.analyze(_ctx()))
        elapsed = time.perf_counter() - t0
        assert len(out) == 1
        assert elapsed < 0.18, f"expected concurrent dispatch (<180ms); got {elapsed:.3f}s"

    def test_empty_input_returns_empty(self) -> None:
        a1 = MockAnalyst("a1", signals=[])
        a2 = MockAnalyst("a2", signals=[])
        vote = VoteAnalyst([a1, a2], min_agree_count=2)
        assert _run(vote.analyze(_ctx())) == []

    def test_min_agree_one_keeps_singletons(self) -> None:
        a1 = MockAnalyst("a1", [_signal(source="a1")])
        a2 = MockAnalyst("a2", signals=[])
        vote = VoteAnalyst([a1, a2], min_agree_count=1)
        out = _run(vote.analyze(_ctx()))
        assert len(out) == 1
        assert out[0].meta["votes"] == 1


# ---------------------------------------------------------------------------
# test_router
# ---------------------------------------------------------------------------


class TestRouter:
    def test_routes_by_regime(self) -> None:
        rule_a = MockAnalyst("rule", [_signal(source="rule")])
        llm_a = MockAnalyst("llm:x", [_signal(source="llm:x")])
        default = MockAnalyst("default", [_signal(source="default")])
        router = RouterAnalyst(
            routes={
                BrooksRegime.STRONG_BULL_TREND: rule_a,
                BrooksRegime.TIGHT_TRADING_RANGE: llm_a,
            },
            default=default,
        )

        out_bull = _run(router.analyze(_ctx(BrooksRegime.STRONG_BULL_TREND)))
        assert rule_a.call_count == 1
        assert llm_a.call_count == 0
        assert default.call_count == 0
        assert out_bull[0].meta["routed_by"] == "strong_bull_trend"

        out_tr = _run(router.analyze(_ctx(BrooksRegime.TIGHT_TRADING_RANGE)))
        assert rule_a.call_count == 1
        assert llm_a.call_count == 1
        assert default.call_count == 0
        assert out_tr[0].meta["routed_by"] == "tight_trading_range"

    def test_unknown_regime_falls_back_to_default(self) -> None:
        rule_a = MockAnalyst("rule", [_signal(source="rule")])
        default = MockAnalyst("default", [_signal(source="default")])
        router = RouterAnalyst(
            routes={BrooksRegime.STRONG_BULL_TREND: rule_a},
            default=default,
        )
        # CLIMAX is not in routes → default takes over.
        out = _run(router.analyze(_ctx(BrooksRegime.CLIMAX)))
        assert rule_a.call_count == 0
        assert default.call_count == 1
        assert out[0].meta["routed_by"] == "climax"

    def test_missing_regime_uses_default(self) -> None:
        rule_a = MockAnalyst("rule", signals=[])
        default = MockAnalyst("default", [_signal(source="default")])
        router = RouterAnalyst(
            routes={BrooksRegime.STRONG_BULL_TREND: rule_a},
            default=default,
        )
        out = _run(router.analyze(_ctx(regime=None)))
        assert default.call_count == 1
        assert out[0].meta["routed_by"] == BrooksRegime.UNKNOWN.value

    def test_router_passes_context_through_unchanged(self) -> None:
        captured = MockAnalyst("captured", [_signal(source="captured")])
        router = RouterAnalyst(routes={}, default=captured)
        ctx = _ctx(BrooksRegime.BROAD_TRADING_RANGE)
        _run(router.analyze(ctx))
        assert captured.last_ctx is ctx


# ---------------------------------------------------------------------------
# test_critic
# ---------------------------------------------------------------------------


class TestCritic:
    def test_producer_three_signals_critic_keeps_two(self) -> None:
        candidates = [
            _signal(pattern="h2", source="producer", probability=0.6),
            _signal(pattern="l2", side="short", entry_px=99.0, stop_px=101.0, source="producer", probability=0.55),
            _signal(pattern="ii", source="producer", probability=0.5),
        ]
        critique = [
            _signal(source="critic", probability=0.85, reasoning="strong h2", meta={"confirms": 0}),
            _signal(source="critic", probability=0.40, reasoning="weak ii", meta={"confirms": 2}),
        ]
        producer = MockAnalyst("producer", candidates)
        critic = MockAnalyst("critic", critique)
        ca = CriticAnalyst(producer=producer, critic=critic)

        out = _run(ca.analyze(_ctx()))
        assert len(out) == 2
        # First kept came from candidate 0 (h2) with probability lifted to 0.85.
        assert out[0].pattern == "h2"
        assert out[0].probability == pytest.approx(0.85)
        assert out[0].source == "ensemble.critic"
        assert out[0].meta["producer_source"] == "producer"
        assert out[0].meta["critic_source"] == "critic"
        # Second kept came from candidate 2 (ii) with probability cut to 0.40.
        assert out[1].pattern == "ii"
        assert out[1].probability == pytest.approx(0.40)
        # Candidate 1 (l2 short) was rejected — never appears in output.
        assert all(o.pattern != "l2" for o in out)

    def test_no_candidates_skips_critic_call(self) -> None:
        producer = MockAnalyst("producer", signals=[])
        critic = MockAnalyst("critic", signals=[])
        ca = CriticAnalyst(producer=producer, critic=critic)
        out = _run(ca.analyze(_ctx()))
        assert out == []
        assert producer.call_count == 1
        assert critic.call_count == 0

    def test_critic_receives_candidates_on_context(self) -> None:
        candidates = [_signal(source="producer")]
        producer = MockAnalyst("producer", candidates)
        critic = MockAnalyst(
            "critic",
            [_signal(source="critic", probability=0.9, meta={"confirms": 0})],
        )
        ca = CriticAnalyst(producer=producer, critic=critic, critic_prompt_overlay="be strict")
        _run(ca.analyze(_ctx()))
        injected = critic.last_ctx
        assert injected is not None
        # Producer's candidates land on the context for the critic to read.
        assert getattr(injected, "candidates")[0].pattern == candidates[0].pattern
        assert getattr(injected, "critic_overlay") == "be strict"

    def test_invalid_confirms_index_is_dropped(self) -> None:
        producer = MockAnalyst("producer", [_signal(source="producer")])
        critic = MockAnalyst(
            "critic",
            [
                _signal(source="critic", probability=0.9, meta={"confirms": -1}),  # rejected sentinel
                _signal(source="critic", probability=0.9, meta={"confirms": 7}),  # out of range
            ],
        )
        ca = CriticAnalyst(producer=producer, critic=critic)
        assert _run(ca.analyze(_ctx())) == []

    def test_kept_signal_inherits_candidate_geometry(self) -> None:
        cand = _signal(
            entry_px=101.5,
            stop_px=99.0,
            target_px=104.0,
            source="producer",
            reasoning="bull pullback",
        )
        producer = MockAnalyst("producer", [cand])
        critic = MockAnalyst(
            "critic",
            [_signal(source="critic", probability=0.8, reasoning="confirmed", meta={"confirms": 0})],
        )
        ca = CriticAnalyst(producer=producer, critic=critic)
        out = _run(ca.analyze(_ctx()))
        assert len(out) == 1
        kept = out[0]
        # entry/stop/target taken from the producer; probability from the critic.
        assert kept.entry_px == 101.5
        assert kept.stop_px == 99.0
        assert kept.target_px == 104.0
        assert kept.probability == pytest.approx(0.8)
        assert "bull pullback" in kept.reasoning
        assert "confirmed" in kept.reasoning
