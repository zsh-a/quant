"""Double top / double bottom detector smoke tests."""

from __future__ import annotations

from typing import List

from src.brooks.features import BarFeatureExtractor, ExtendedBarFeatures
from src.brooks.patterns import (
    DetectorContext,
    DoubleBottomDetector,
    DoubleTopDetector,
)
from src.brooks.structure import MarketStructureTracker
from src.core.base import Bar

from .conftest import make_series


def _run(bars: List[Bar], detector, *, swing_k=2, breakout_lookback=6):
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


def test_double_bottom_registered_name_and_side():
    det = DoubleBottomDetector()
    assert det.name == "double_bottom"
    assert det.side == "long"


def test_double_top_registered_name_and_side():
    det = DoubleTopDetector()
    assert det.name == "double_top"
    assert det.side == "short"


def test_double_bottom_no_signal_without_two_lows():
    vals = [(100.0, 100.5, 99.5, 100.0) for _ in range(10)]
    signals = _run(make_series(vals), DoubleBottomDetector())
    assert signals == []


def test_double_top_no_signal_without_two_highs():
    vals = [(100.0, 100.5, 99.5, 100.0) for _ in range(10)]
    signals = _run(make_series(vals), DoubleTopDetector())
    assert signals == []
