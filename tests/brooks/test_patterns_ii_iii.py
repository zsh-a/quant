"""ii / iii (inside-bar compression) breakout detector tests."""

from __future__ import annotations

from typing import List, Tuple

from src.brooks.features import BarFeatureExtractor, ExtendedBarFeatures
from src.brooks.patterns import (
    DetectorContext,
    IIBreakoutDetector,
    IIIBreakoutDetector,
    PatternRegistry,
)
from src.brooks.structure import MarketStructureTracker
from src.core.base import Bar

from .conftest import make_series


def _bull(price: float, body: float = 1.0, wick: float = 0.05) -> Tuple[float, float, float, float]:
    return (price, price + body + wick, price - wick, price + body)


def _bear(price: float, body: float = 1.0, wick: float = 0.05) -> Tuple[float, float, float, float]:
    return (price, price + wick, price - body - wick, price - body)


def _run(bars: List[Bar], detector, *, breakout_lookback=8, swing_k=2):
    ext = BarFeatureExtractor(swing_k=swing_k, breakout_lookback=breakout_lookback, atr_period=5)
    tr = MarketStructureTracker(ext, breakout_lookback=breakout_lookback)
    hist: List[ExtendedBarFeatures] = []
    signals = []
    for b in bars:
        ts_ns = int(b.timestamp.timestamp() * 1_000_000_000)
        f = ext.on_bar(ts_ns, b.open, b.high, b.low, b.close)
        s = tr.on_features(f)
        hist.append(f)
        sig = detector.on_bar(DetectorContext(feat=f, structure=s, recent_features=hist))
        if sig is not None:
            signals.append(sig)
    return signals


def _ii_after_bull_rally(n_inside: int) -> List[Bar]:
    """Bull rally to flip always_in long, then a wide bar followed by ``n_inside`` inside bars."""
    vals: list[tuple] = []
    price = 100.0
    for _ in range(12):
        v = _bull(price, body=1.0)
        vals.append(v)
        price = v[3]
    # Wide reference bar (the parent of the first inside)
    o, c = price, price + 0.5
    parent_h, parent_l = c + 1.0, o - 1.0
    vals.append((o, parent_h, parent_l, c))
    # n_inside successively-tightening inside bars
    h, l = parent_h, parent_l
    for _ in range(n_inside):
        new_h = h - 0.1
        new_l = l + 0.1
        mid = (new_h + new_l) / 2
        vals.append((mid - 0.05, new_h, new_l, mid + 0.05))
        h, l = new_h, new_l
    return make_series(vals)


def _ii_after_bear_rally(n_inside: int) -> List[Bar]:
    vals: list[tuple] = []
    price = 100.0
    for _ in range(12):
        v = _bear(price, body=1.0)
        vals.append(v)
        price = v[3]
    o, c = price, price - 0.5
    parent_h, parent_l = o + 1.0, c - 1.0
    vals.append((o, parent_h, parent_l, c))
    h, l = parent_h, parent_l
    for _ in range(n_inside):
        new_h = h - 0.1
        new_l = l + 0.1
        mid = (new_h + new_l) / 2
        vals.append((mid + 0.05, new_h, new_l, mid - 0.05))
        h, l = new_h, new_l
    return make_series(vals)


# ---- positives ------------------------------------------------------------


def test_ii_breakout_long_in_bull_trend():
    sigs = _run(_ii_after_bull_rally(2), IIBreakoutDetector())
    assert len(sigs) >= 1
    s = sigs[-1]
    assert s.detector == "ii_breakout"
    assert s.side == "long"
    assert s.entry_px > s.stop_px
    assert s.metadata["n_inside"] == 2


def test_ii_breakout_short_in_bear_trend():
    sigs = _run(_ii_after_bear_rally(2), IIBreakoutDetector())
    assert len(sigs) >= 1
    s = sigs[-1]
    assert s.detector == "ii_breakout"
    assert s.side == "short"
    assert s.entry_px < s.stop_px


def test_iii_breakout_long_in_bull_trend():
    sigs = _run(_ii_after_bull_rally(3), IIIBreakoutDetector())
    assert len(sigs) >= 1
    s = sigs[-1]
    assert s.detector == "iii_breakout"
    assert s.side == "long"
    assert s.metadata["n_inside"] == 3


# ---- negatives ------------------------------------------------------------


def test_ii_no_signal_without_two_inside_bars():
    """A pure trend with no inside-bar pair → no signal."""
    vals = []
    p = 100.0
    for _ in range(20):
        v = _bull(p, body=1.0)
        vals.append(v)
        p = v[3]
    sigs = _run(make_series(vals), IIBreakoutDetector())
    assert sigs == []


def test_iii_no_signal_when_only_two_inside_bars():
    sigs = _run(_ii_after_bull_rally(2), IIIBreakoutDetector())
    assert sigs == []


# ---- registry -------------------------------------------------------------


def test_ii_iii_registered():
    assert "ii_breakout" in PatternRegistry.all()
    assert "iii_breakout" in PatternRegistry.all()


def test_ii_iii_class_attrs():
    assert IIBreakoutDetector.name == "ii_breakout"
    assert IIBreakoutDetector.n_inside == 2
    assert IIIBreakoutDetector.name == "iii_breakout"
    assert IIIBreakoutDetector.n_inside == 3
