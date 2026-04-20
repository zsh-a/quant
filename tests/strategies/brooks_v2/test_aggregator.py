"""SignalAggregator tests."""

from __future__ import annotations

from src.analysis.brooks.aggregator import SignalAggregator
from src.analysis.brooks.patterns.base import PatternSignal


def _sig(name: str, side: str, entry: float, stop: float, idx: int = 10) -> PatternSignal:
    return PatternSignal(detector=name, side=side, signal_bar_idx=idx, entry_px=entry, stop_px=stop, timestamp_ns=idx)


def test_any_of_mode_returns_single_signal():
    agg = SignalAggregator(confluence_n=1)
    d = agg.resolve([_sig("h2", "long", 101.0, 99.0)])
    assert d is not None
    assert d.side == "long"
    assert d.hit_detectors == ["h2"]


def test_confluence_requires_n_same_side():
    agg = SignalAggregator(confluence_n=2)
    # one long + one short → no confluence
    d = agg.resolve([_sig("h2", "long", 101.0, 99.0), _sig("l2", "short", 99.0, 101.0)])
    assert d is None
    # two longs → confluence
    d = agg.resolve([_sig("h2", "long", 101.0, 99.0), _sig("wedge", "long", 101.5, 98.5)])
    assert d is not None
    assert d.side == "long"
    assert set(d.hit_detectors) == {"h2", "wedge"}
    # conservative entry = max(101.0, 101.5) = 101.5, conservative stop = min(99.0, 98.5) = 98.5
    assert abs(d.entry_px - 101.5) < 1e-9
    assert abs(d.stop_px - 98.5) < 1e-9


def test_empty_signals_returns_none():
    agg = SignalAggregator()
    assert agg.resolve([]) is None
