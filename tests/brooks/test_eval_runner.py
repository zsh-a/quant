"""Tests for :class:`EvalRunner` — pattern matching, hit-rate scoring,
and Trader's-Equation integration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List

import pandas as pd
import pytest

from src.brooks.context import Bar, BrooksContext
from src.brooks.decision.hit_rate import HitRateTable
from src.brooks.decision.trader_equation import TraderEquation
from src.brooks.eval.golden import GoldenDataset, GoldenSample
from src.brooks.eval.runner import EvalRunner, _realized_r
from src.brooks.schema import Signal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bar(i: int, o: float, h: float, l: float, c: float) -> Bar:
    return Bar(
        timestamp_ns=1_700_000_000_000_000_000 + i * 60_000_000_000,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=1.0,
    )


def _flat_bars(n: int = 10, base: float = 100.0) -> List[Bar]:
    return [_bar(i, base, base + 0.1, base - 0.1, base) for i in range(n)]


def _sample_with_future(
    *,
    sample_id: str = "s",
    symbol: str = "BTC",
    expected_pattern: str = "h2",
    expected_side: str = "long",
    entry: float = 100.0,
    stop: float = 99.0,
    target: float = 102.0,
    future: List[Bar] = None,
    regime: str = "weak_bull_trend",
    htf_aligned: bool = True,
    source: str = "human",
) -> GoldenSample:
    if future is None:
        future = []
    bars = _flat_bars(5) + future
    return GoldenSample(
        id=sample_id,
        symbol=symbol,
        interval="5m",
        bars=bars,
        target_bar_idx=4,
        expected_pattern=expected_pattern,
        expected_side=expected_side,  # type: ignore[arg-type]
        expected_entry=entry,
        expected_stop=stop,
        expected_target=target,
        regime=regime,
        htf_aligned=htf_aligned,
        source=source,
    )


@dataclass
class FixedAnalyst:
    """Analyst that returns a fixed signal list keyed by ``sample_id``."""

    name: str = "fixed"
    by_id: dict = field(default_factory=dict)

    async def analyze(self, ctx: BrooksContext) -> List[Signal]:
        return list(self.by_id.get(ctx.symbol, []))


def _sig(
    pattern: str = "h2",
    side: str = "long",
    entry: float = 100.0,
    stop: float = 99.0,
    target: float = 102.0,
) -> Signal:
    return Signal(
        pattern=pattern,
        side=side,  # type: ignore[arg-type]
        signal_bar_idx=4,
        entry_px=entry,
        stop_px=stop,
        target_px=target,
        probability=0.55,
        quality=0.6,
        source=f"rule:{pattern}",
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Pattern / side matching
# ---------------------------------------------------------------------------


def test_runner_pattern_match_picks_exact_signal():
    sample = _sample_with_future(sample_id="ok", symbol="ok")
    analyst = FixedAnalyst(by_id={"ok": [_sig(pattern="l2", side="short", entry=100.0, stop=101.0), _sig()]})
    ds = GoldenDataset.from_samples([sample])
    report = _run(EvalRunner(analyst=analyst, dataset=ds).run())
    assert len(report.results) == 1
    r = report.results[0]
    assert r.pattern_match is True
    assert r.side_match is True
    assert r.predicted_signal.pattern == "h2"
    assert "h2" in r.pattern_emitted
    assert "l2" in r.pattern_emitted


def test_runner_pattern_miss_when_only_other_pattern_emits():
    sample = _sample_with_future(sample_id="miss", symbol="miss")
    analyst = FixedAnalyst(by_id={"miss": [_sig(pattern="wedge")]})
    ds = GoldenDataset.from_samples([sample])
    report = _run(EvalRunner(analyst=analyst, dataset=ds).run())
    r = report.results[0]
    assert r.pattern_match is False
    # Side still matches because the wedge signal is also long.
    assert r.side_match is True


def test_runner_no_signal_marks_neither_match():
    sample = _sample_with_future(sample_id="empty", symbol="empty")
    analyst = FixedAnalyst()
    ds = GoldenDataset.from_samples([sample])
    report = _run(EvalRunner(analyst=analyst, dataset=ds).run())
    r = report.results[0]
    assert r.pattern_match is False
    assert r.side_match is False
    assert r.predicted_signal is None


# ---------------------------------------------------------------------------
# Hit-rate / realized R
# ---------------------------------------------------------------------------


def test_realized_r_long_target_2r_returns_2():
    sig = _sig(entry=100.0, stop=99.0)
    future = [
        _bar(5, 100.0, 100.5, 100.0, 100.4),
        _bar(6, 100.4, 102.5, 100.0, 102.5),  # touches 2R target (102.0)
    ]
    realized, h1, h2 = _realized_r(sig, future)
    assert realized == 2.0
    assert h1 == "hit"
    assert h2 == "hit"


def test_realized_r_long_stop_returns_minus_1():
    sig = _sig(entry=100.0, stop=99.0)
    future = [_bar(5, 100.0, 100.2, 98.9, 99.0)]  # touches stop
    realized, h1, h2 = _realized_r(sig, future)
    assert realized == -1.0
    assert h1 == "miss"
    assert h2 == "miss"


def test_realized_r_long_1r_only_then_window_ends():
    sig = _sig(entry=100.0, stop=99.0)
    future = [
        _bar(5, 100.0, 101.2, 100.0, 100.5),  # touches 1R = 101.0
        _bar(6, 100.5, 100.7, 100.3, 100.4),
    ]
    realized, h1, h2 = _realized_r(sig, future)
    assert h1 == "hit"
    assert h2 == "miss"
    # Final mark to market is the last close minus entry in R units = 0.4
    assert realized == pytest.approx(0.4)


def test_realized_r_short_target_2r_returns_2():
    sig = _sig(side="short", entry=100.0, stop=101.0, target=98.0)
    future = [
        _bar(5, 100.0, 100.0, 99.5, 99.6),
        _bar(6, 99.6, 99.7, 97.5, 97.5),  # touches 2R target (98.0)
    ]
    realized, h1, h2 = _realized_r(sig, future)
    assert realized == 2.0
    assert h1 == "hit"
    assert h2 == "hit"


def test_realized_r_no_future_returns_unresolved():
    sig = _sig()
    realized, h1, h2 = _realized_r(sig, [])
    assert realized is None
    assert h1 == "unresolved"
    assert h2 == "unresolved"


def test_runner_records_realized_r_for_predicted_signal():
    future = [
        _bar(5, 100.0, 100.2, 100.0, 100.1),
        _bar(6, 100.1, 102.5, 100.0, 102.5),  # 2R hit
    ]
    sample = _sample_with_future(sample_id="realized", symbol="realized", future=future)
    analyst = FixedAnalyst(by_id={"realized": [_sig()]})
    report = _run(EvalRunner(analyst=analyst, dataset=GoldenDataset.from_samples([sample])).run())
    r = report.results[0]
    assert r.realized_r == 2.0
    assert r.hit_1r == "hit"
    assert r.hit_2r == "hit"


# ---------------------------------------------------------------------------
# TraderEquation integration
# ---------------------------------------------------------------------------


def test_runner_scores_with_trader_equation():
    df = pd.DataFrame(
        [
            {
                "pattern": "h2",
                "regime": "weak_bull_trend",
                "htf_aligned": True,
                "side": "long",
                "samples": 100,
                "hit_rate_1r": 0.7,
                "hit_rate_2r": 0.4,
                "avg_realized_r": 0.8,
            }
        ]
    )
    te = TraderEquation(HitRateTable(df))

    sample = _sample_with_future(sample_id="te", symbol="te", future=[_bar(5, 100.0, 100.2, 100.0, 100.1)])
    analyst = FixedAnalyst(by_id={"te": [_sig()]})

    report = _run(EvalRunner(analyst=analyst, dataset=GoldenDataset.from_samples([sample]), te=te).run())
    r = report.results[0]
    # Probability should come from the table (0.7), not the prior.
    assert r.probability == pytest.approx(0.7)
    # E[R] = 0.7 * 2 - 0.3 - 0.05 = 1.05
    assert r.expected_r == pytest.approx(1.05, abs=1e-6)


# ---------------------------------------------------------------------------
# EvalReport plumbing exposed by the runner
# ---------------------------------------------------------------------------


def test_runner_validates_inputs():
    with pytest.raises(ValueError):
        EvalRunner(analyst=None, dataset=GoldenDataset())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        EvalRunner(analyst=FixedAnalyst(), dataset=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        EvalRunner(analyst=FixedAnalyst(), dataset=GoldenDataset(), max_concurrent=0)


def test_runner_runs_empty_dataset():
    analyst = FixedAnalyst()
    report = _run(EvalRunner(analyst=analyst, dataset=GoldenDataset()).run())
    assert report.results == []
    assert report.dataset_size == 0
