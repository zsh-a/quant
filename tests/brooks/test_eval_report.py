"""Tests for :class:`EvalReport` — bucketing, Wilson CI, and HTML
serialisation. Includes the spec acceptance test that wires
:class:`RuleAnalyst` end-to-end against a 30-sample mock dataset.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List

import pytest

from src.brooks.analyst import AnalystRegistry
from src.brooks.context import Bar
from src.brooks.eval.golden import GoldenDataset, GoldenSample
from src.brooks.eval.report import EvalReport, _wilson
from src.brooks.eval.runner import EvalRunner, SampleResult
from src.brooks.schema import Signal

# ---------------------------------------------------------------------------
# SampleResult fixtures
# ---------------------------------------------------------------------------


def _result(
    *,
    sample_id: str = "s",
    pattern: str = "h2",
    side: str = "long",
    regime: str = "weak_bull_trend",
    htf_aligned: bool = True,
    source: str = "human",
    pattern_match: bool = True,
    side_match: bool = True,
    realized_r: float = 1.0,
    hit_1r: str = "hit",
    hit_2r: str = "miss",
    predicted: bool = True,
    latency_ms: float = 12.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_hit: bool = False,
) -> SampleResult:
    sig = (
        Signal(
            pattern=pattern,
            side=side,  # type: ignore[arg-type]
            signal_bar_idx=4,
            entry_px=100.0,
            stop_px=99.0,
            target_px=102.0,
            probability=0.55,
            quality=0.6,
            source=f"rule:{pattern}",
        )
        if predicted
        else None
    )
    return SampleResult(
        sample_id=sample_id,
        symbol="BTC",
        interval="5m",
        expected_pattern=pattern,
        expected_side=side,
        regime=regime,
        htf_aligned=htf_aligned,
        source=source,
        pattern_match=pattern_match,
        side_match=side_match,
        pattern_emitted=[pattern] if predicted else [],
        predicted_signal=sig,
        realized_r=realized_r,
        hit_1r=hit_1r,
        hit_2r=hit_2r,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit=cache_hit,
    )


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------


def test_wilson_centered_around_proportion():
    lo, hi = _wilson(50, 100)
    assert 0.39 < lo < 0.41
    assert 0.59 < hi < 0.61


def test_wilson_zero_trials_returns_zero_interval():
    assert _wilson(0, 0) == (0.0, 0.0)


def test_wilson_perfect_score_caps_at_one():
    lo, hi = _wilson(10, 10)
    assert hi == 1.0
    assert lo > 0.5


# ---------------------------------------------------------------------------
# Bucket metrics
# ---------------------------------------------------------------------------


def test_overall_metrics_blend_pattern_and_hit_rate():
    rows = [
        _result(sample_id="a", pattern_match=True, hit_1r="hit", hit_2r="miss", realized_r=1.0),
        _result(sample_id="b", pattern_match=False, hit_1r="miss", hit_2r="miss", realized_r=-1.0),
        _result(sample_id="c", pattern_match=True, hit_1r="hit", hit_2r="hit", realized_r=2.0),
        _result(sample_id="d", pattern_match=True, hit_1r="hit", hit_2r="hit", realized_r=2.0),
    ]
    report = EvalReport(results=rows, dataset_size=4, analyst_name="test")
    overall = report.overall()
    assert overall.samples == 4
    assert overall.pattern_precision == 0.75
    assert overall.pattern_recall == 0.75
    assert overall.hit_rate_1r == 0.75
    assert overall.hit_rate_2r == 0.5
    assert overall.avg_realized_r == pytest.approx(1.0)


def test_bucket_by_pattern_groups_correctly():
    rows = [
        _result(sample_id="a", pattern="h2"),
        _result(sample_id="b", pattern="h2"),
        _result(sample_id="c", pattern="l2", side="short", side_match=True),
    ]
    buckets = EvalReport(results=rows).bucket_metrics("pattern")
    keys = [b.value for b in buckets]
    assert keys == ["h2", "l2"]


def test_bucket_by_htf_alignment():
    rows = [
        _result(sample_id="a", htf_aligned=True),
        _result(sample_id="b", htf_aligned=False),
        _result(sample_id="c", htf_aligned=False),
    ]
    by_htf = {b.value: b for b in EvalReport(results=rows).bucket_metrics("htf_aligned")}
    assert by_htf["aligned"].samples == 1
    assert by_htf["unaligned"].samples == 2


def test_bucket_by_source():
    rows = [
        _result(sample_id="a", source="human"),
        _result(sample_id="b", source="silver"),
    ]
    by_source = {b.value: b for b in EvalReport(results=rows).bucket_metrics("source")}
    assert by_source["human"].samples == 1
    assert by_source["silver"].samples == 1


def test_bucket_unknown_dimension_raises():
    with pytest.raises(ValueError):
        EvalReport(results=[_result()]).bucket_metrics("nonsense")


def test_buckets_include_confidence_intervals():
    rows = [_result(sample_id=f"r{i}", hit_1r="hit") for i in range(5)] + [
        _result(sample_id=f"m{i}", hit_1r="miss") for i in range(5)
    ]
    overall = EvalReport(results=rows).overall()
    lo, hi = overall.hit_rate_1r_ci
    assert lo < 0.5 < hi
    assert lo >= 0.0 and hi <= 1.0


# ---------------------------------------------------------------------------
# Cost summary
# ---------------------------------------------------------------------------


def test_cost_summary_aggregates_tokens_and_cache():
    rows = [
        _result(sample_id="a", latency_ms=10.0, input_tokens=100, output_tokens=20, cache_hit=True),
        _result(sample_id="b", latency_ms=30.0, input_tokens=80, output_tokens=15, cache_hit=False),
        _result(sample_id="c", latency_ms=20.0),  # no LLM tokens
    ]
    cost = EvalReport(results=rows).cost_summary()
    assert cost["samples"] == 3
    assert cost["total_input_tokens"] == 180
    assert cost["total_output_tokens"] == 35
    # Two rows had token usage; one was a cache hit → 0.5
    assert cost["cache_hit_rate"] == 0.5


def test_cost_summary_empty():
    cost = EvalReport(results=[]).cost_summary()
    assert cost["samples"] == 0
    assert cost["cache_hit_rate"] == 0.0


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_to_dataframe_has_one_row_per_result():
    rows = [_result(sample_id=f"r{i}") for i in range(3)]
    df = EvalReport(results=rows).to_dataframe()
    assert len(df) == 3
    assert {"pattern_match", "side_match", "realized_r"}.issubset(set(df.columns))


def test_to_html_writes_self_contained_document(tmp_path: Path):
    rows = [_result(sample_id=f"r{i}") for i in range(3)]
    out = tmp_path / "report.html"
    EvalReport(results=rows, dataset_size=3, analyst_name="rule").to_html(out)
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "<!doctype html>" in body
    assert "Brooks Eval" in body
    # Each bucket dimension must produce a <table> and the headline KPIs are present.
    assert body.count("<table>") >= 4
    assert "pattern P" in body
    assert "hit-1R" in body


def test_to_json_includes_buckets_and_rows():
    import json as _json

    rows = [_result(sample_id=f"r{i}") for i in range(2)]
    payload = _json.loads(EvalReport(results=rows, dataset_size=2, analyst_name="rule").to_json())
    assert set(payload["buckets"]) == {"pattern", "regime", "htf_aligned", "source"}
    assert payload["overall"]["samples"] == 2
    assert len(payload["rows"]) == 2


# ---------------------------------------------------------------------------
# Spec acceptance — RuleAnalyst on a 30-sample dataset
# ---------------------------------------------------------------------------


def _bar(i: int, base: float, future: List[Bar] = None) -> Bar:
    return Bar(
        timestamp_ns=1_700_000_000_000_000_000 + i * 60_000_000_000,
        open=base,
        high=base + 0.1,
        low=base - 0.1,
        close=base,
        volume=1.0,
    )


def _build_30_sample_fixture() -> GoldenDataset:
    """Build a 30-sample dataset of mostly-flat bars labeled with synthetic
    expected setups. The rule analyst will not match these (no real
    Brooks setup is in the bars) so this test exercises the runner +
    report plumbing rather than detector accuracy."""
    samples = []
    for i in range(30):
        bars = [_bar(j, 100.0 + 0.01 * j) for j in range(40)]
        samples.append(
            GoldenSample(
                id=f"silver-{i}",
                symbol=f"SYM{i % 3}",
                interval="5m",
                bars=bars,
                target_bar_idx=29,
                expected_pattern=("h2" if i % 2 == 0 else "l2"),
                expected_side=("long" if i % 2 == 0 else "short"),
                expected_entry=100.5,
                expected_stop=99.5 if i % 2 == 0 else 101.5,
                expected_target=102.0 if i % 2 == 0 else 98.0,
                regime=("weak_bull_trend" if i % 2 == 0 else "weak_bear_trend"),
                htf_aligned=(i % 3 == 0),
                source=("human" if i < 5 else "silver"),
            )
        )
    return GoldenDataset.from_samples(samples)


def test_acceptance_rule_analyst_produces_html_report(tmp_path: Path):
    dataset = _build_30_sample_fixture()
    runner = EvalRunner(analyst=AnalystRegistry.build("rule"), dataset=dataset)
    report = asyncio.run(runner.run())

    assert report.dataset_size == 30
    assert len(report.results) == 30

    out = tmp_path / "rule_report.html"
    body = report.to_html(out)
    assert out.exists()
    assert "<!doctype html>" in body
    # Bucket dimensions all rendered.
    assert "By pattern" in body
    assert "By regime" in body
    assert "By htf_aligned" in body
    assert "By source" in body

    # DataFrame round-trip works.
    df = report.to_dataframe()
    assert len(df) == 30
