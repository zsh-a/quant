"""Wedge (long / short) detector smoke tests."""

from __future__ import annotations

from typing import List

from src.brooks.features import BarFeatureExtractor, ExtendedBarFeatures
from src.brooks.patterns import DetectorContext, WedgeLongDetector, WedgeShortDetector
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


def test_wedge_long_no_signal_on_flat_series():
    """No 3-push wedge in a chop series → no signal."""
    vals = [(100.0, 100.5, 99.5, 100.0) for _ in range(30)]
    signals = _run(make_series(vals), WedgeLongDetector(min_separation_bars=1))
    assert signals == []


def test_wedge_short_no_signal_on_flat_series():
    vals = [(100.0, 100.5, 99.5, 100.0) for _ in range(30)]
    signals = _run(make_series(vals), WedgeShortDetector(min_separation_bars=1))
    assert signals == []


def test_wedge_long_detector_name_and_side():
    det = WedgeLongDetector()
    assert det.name == "wedge_long"
    assert det.side == "long"


def test_wedge_short_detector_name_and_side():
    det = WedgeShortDetector()
    assert det.name == "wedge_short"
    assert det.side == "short"
