"""SignalAggregator tests covering confluence_n boundaries (1 / 2 / 3)."""

from __future__ import annotations

import pytest

from src.brooks.decision import AggregatedDecision, SignalAggregator
from src.brooks.schema import Signal


def _sig(
    pattern: str,
    side: str,
    entry: float,
    stop: float,
    *,
    idx: int = 10,
    timestamp_ns: int = 0,
) -> Signal:
    return Signal(
        pattern=pattern,
        side=side,
        signal_bar_idx=idx,
        entry_px=entry,
        stop_px=stop,
        probability=0.5,
        quality=0.5,
        source=f"rule:{pattern}",
        meta={"timestamp_ns": timestamp_ns} if timestamp_ns else {},
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_invalid_confluence_n_rejected():
    with pytest.raises(ValueError):
        SignalAggregator(confluence_n=0)


# ---------------------------------------------------------------------------
# confluence_n = 1 (any-of)
# ---------------------------------------------------------------------------


def test_confluence_1_single_signal_passes():
    agg = SignalAggregator(confluence_n=1)
    d = agg.resolve([_sig("h2", "long", 101.0, 99.0)])
    assert isinstance(d, AggregatedDecision)
    assert d.side == "long"
    assert d.hit_detectors == ["h2"]
    assert d.entry_px == 101.0
    assert d.stop_px == 99.0


def test_confluence_1_empty_returns_none():
    assert SignalAggregator(confluence_n=1).resolve([]) is None


def test_confluence_1_filters_none_inside_list():
    agg = SignalAggregator(confluence_n=1)
    d = agg.resolve([None, _sig("l2", "short", 99.0, 101.0)])  # type: ignore[list-item]
    assert d is not None
    assert d.side == "short"


def test_confluence_1_long_wins_tie_against_short():
    agg = SignalAggregator(confluence_n=1)
    d = agg.resolve([_sig("h2", "long", 101.0, 99.0), _sig("l2", "short", 99.0, 101.0)])
    # tie 1-vs-1: long preferred (>=)
    assert d is not None and d.side == "long"


# ---------------------------------------------------------------------------
# confluence_n = 2
# ---------------------------------------------------------------------------


def test_confluence_2_split_sides_returns_none():
    agg = SignalAggregator(confluence_n=2)
    d = agg.resolve([_sig("h2", "long", 101.0, 99.0), _sig("l2", "short", 99.0, 101.0)])
    assert d is None


def test_confluence_2_two_longs_pass_with_conservative_levels():
    agg = SignalAggregator(confluence_n=2)
    d = agg.resolve(
        [
            _sig("h2", "long", 101.0, 99.0),
            _sig("wedge_long", "long", 101.5, 98.5),
        ]
    )
    assert d is not None
    assert d.side == "long"
    assert set(d.hit_detectors) == {"h2", "wedge_long"}
    assert d.entry_px == 101.5  # conservative: highest entry for long
    assert d.stop_px == 98.5  # conservative: lowest stop for long


def test_confluence_2_one_long_two_shorts_picks_shorts():
    agg = SignalAggregator(confluence_n=2)
    d = agg.resolve(
        [
            _sig("h2", "long", 101.0, 99.0),
            _sig("l2", "short", 99.5, 101.5),
            _sig("double_top", "short", 99.0, 101.0),
        ]
    )
    assert d is not None
    assert d.side == "short"
    # conservative entry for short = lowest entry; conservative stop = highest stop
    assert d.entry_px == 99.0
    assert d.stop_px == 101.5


def test_confluence_2_single_signal_rejected():
    agg = SignalAggregator(confluence_n=2)
    assert agg.resolve([_sig("h2", "long", 101.0, 99.0)]) is None


# ---------------------------------------------------------------------------
# confluence_n = 3
# ---------------------------------------------------------------------------


def test_confluence_3_two_signals_rejected():
    agg = SignalAggregator(confluence_n=3)
    assert agg.resolve([_sig("h2", "long", 101.0, 99.0), _sig("wedge_long", "long", 101.2, 98.8)]) is None


def test_confluence_3_three_longs_pass():
    agg = SignalAggregator(confluence_n=3)
    d = agg.resolve(
        [
            _sig("h2", "long", 101.0, 99.0, idx=10),
            _sig("wedge_long", "long", 101.5, 98.5, idx=11),
            _sig("double_bottom", "long", 100.8, 99.2, idx=12),
        ]
    )
    assert d is not None
    assert d.side == "long"
    assert set(d.hit_detectors) == {"h2", "wedge_long", "double_bottom"}
    assert d.signal_bar_idx == 12  # latest
    assert d.entry_px == 101.5
    assert d.stop_px == 98.5


def test_confluence_3_mixed_sides_with_two_long_one_short_rejected():
    agg = SignalAggregator(confluence_n=3)
    d = agg.resolve(
        [
            _sig("h2", "long", 101.0, 99.0),
            _sig("wedge_long", "long", 101.5, 98.5),
            _sig("l2", "short", 99.0, 101.0),
        ]
    )
    assert d is None


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_aggregated_decision_to_dict_roundtrips_signals():
    agg = SignalAggregator(confluence_n=1)
    d = agg.resolve([_sig("h2", "long", 101.0, 99.0, timestamp_ns=12345)])
    assert d is not None
    blob = d.to_dict()
    assert blob["side"] == "long"
    assert blob["hit_detectors"] == ["h2"]
    assert blob["timestamp_ns"] == 12345
    assert blob["raw_signals"][0]["pattern"] == "h2"
    assert blob["raw_signals"][0]["source"] == "rule:h2"


def test_timestamp_uses_latest_meta():
    agg = SignalAggregator(confluence_n=2)
    d = agg.resolve(
        [
            _sig("h2", "long", 101.0, 99.0, timestamp_ns=100),
            _sig("wedge_long", "long", 101.5, 98.5, timestamp_ns=200),
        ]
    )
    assert d is not None
    assert d.timestamp_ns == 200
