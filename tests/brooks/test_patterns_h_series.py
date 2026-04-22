"""H1 / H3 / H4 detector behavioural tests."""

from __future__ import annotations

from typing import List, Tuple

from src.brooks.features import BarFeatureExtractor, ExtendedBarFeatures
from src.brooks.patterns import (
    DetectorContext,
    H1Detector,
    H3Detector,
    H4Detector,
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


def _h_pattern_bars(n_pullbacks: int) -> List[Bar]:
    """Steady bull rally, then ``n_pullbacks`` × (bear bars + bull recovery)."""
    vals: list[tuple] = []
    price = 100.0
    for _ in range(12):
        v = _bull(price, body=1.0)
        vals.append(v)
        price = v[3]
    for _ in range(n_pullbacks):
        v = _bear(price, body=0.6)
        vals.append(v)
        price = v[3]
        v = _bull(price, body=0.7)
        vals.append(v)
        price = v[3]
    return make_series(vals)


# ---- positives ------------------------------------------------------------


def test_h1_fires_on_first_pullback():
    sigs = _run(_h_pattern_bars(1), H1Detector())
    assert len(sigs) == 1
    s = sigs[0]
    assert s.detector == "h1"
    assert s.side == "long"
    assert s.entry_px > s.stop_px
    assert s.metadata["pullback_count"] == 1


def test_h3_fires_on_third_pullback():
    sigs = _run(_h_pattern_bars(3), H3Detector())
    assert len(sigs) == 1
    s = sigs[0]
    assert s.detector == "h3"
    assert s.side == "long"
    assert s.metadata["pullback_count"] == 3


def test_h4_fires_on_fourth_pullback():
    sigs = _run(_h_pattern_bars(4), H4Detector())
    assert len(sigs) == 1
    s = sigs[0]
    assert s.detector == "h4"
    assert s.side == "long"
    assert s.metadata["pullback_count"] == 4
    assert s.metadata["quality"] == "low"  # H4 is exhaustion-flagged


# ---- negatives ------------------------------------------------------------


def test_h1_no_signal_in_persistent_downtrend():
    vals: list[tuple] = []
    p = 100.0
    for _ in range(25):
        v = _bear(p, body=0.6)
        vals.append(v)
        p = v[3]
    sigs = _run(make_series(vals), H1Detector())
    assert sigs == []


def test_h3_no_signal_when_only_two_pullbacks_present():
    sigs = _run(_h_pattern_bars(2), H3Detector())
    assert sigs == []


def test_h4_no_signal_when_three_pullbacks_present():
    sigs = _run(_h_pattern_bars(3), H4Detector())
    assert sigs == []


# ---- registry / metadata --------------------------------------------------


def test_h_series_registered():
    for name in ("h1", "h3", "h4"):
        assert name in PatternRegistry.all()


def test_h_series_class_names():
    assert H1Detector.name == "h1"
    assert H3Detector.name == "h3"
    assert H4Detector.name == "h4"
    assert H1Detector.target_pullback_count == 1
    assert H3Detector.target_pullback_count == 3
    assert H4Detector.target_pullback_count == 4
