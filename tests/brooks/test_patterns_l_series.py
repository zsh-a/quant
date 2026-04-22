"""L1 / L3 / L4 detector behavioural tests."""

from __future__ import annotations

from typing import List, Tuple

from src.brooks.features import BarFeatureExtractor, ExtendedBarFeatures
from src.brooks.patterns import (
    DetectorContext,
    L1Detector,
    L3Detector,
    L4Detector,
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


def _l_pattern_bars(n_pullbacks: int) -> List[Bar]:
    """Steady bear rally, then ``n_pullbacks`` × (bull pullback + bear recovery)."""
    vals: list[tuple] = []
    price = 100.0
    for _ in range(12):
        v = _bear(price, body=1.0)
        vals.append(v)
        price = v[3]
    for _ in range(n_pullbacks):
        v = _bull(price, body=0.6)
        vals.append(v)
        price = v[3]
        v = _bear(price, body=0.7)
        vals.append(v)
        price = v[3]
    return make_series(vals)


# ---- positives ------------------------------------------------------------


def test_l1_fires_on_first_pullback():
    sigs = _run(_l_pattern_bars(1), L1Detector())
    assert len(sigs) == 1
    s = sigs[0]
    assert s.detector == "l1"
    assert s.side == "short"
    assert s.entry_px < s.stop_px
    assert s.metadata["pullback_count"] == 1


def test_l3_fires_on_third_pullback():
    sigs = _run(_l_pattern_bars(3), L3Detector())
    assert len(sigs) == 1
    s = sigs[0]
    assert s.detector == "l3"
    assert s.side == "short"
    assert s.metadata["pullback_count"] == 3


def test_l4_fires_on_fourth_pullback():
    sigs = _run(_l_pattern_bars(4), L4Detector())
    assert len(sigs) == 1
    s = sigs[0]
    assert s.detector == "l4"
    assert s.side == "short"
    assert s.metadata["pullback_count"] == 4
    assert s.metadata["quality"] == "low"


# ---- negatives ------------------------------------------------------------


def test_l1_no_signal_in_persistent_uptrend():
    vals: list[tuple] = []
    p = 100.0
    for _ in range(25):
        v = _bull(p, body=0.6)
        vals.append(v)
        p = v[3]
    sigs = _run(make_series(vals), L1Detector())
    assert sigs == []


def test_l3_no_signal_when_only_two_pullbacks_present():
    sigs = _run(_l_pattern_bars(2), L3Detector())
    assert sigs == []


def test_l4_no_signal_when_three_pullbacks_present():
    sigs = _run(_l_pattern_bars(3), L4Detector())
    assert sigs == []


# ---- registry -------------------------------------------------------------


def test_l_series_registered():
    for name in ("l1", "l3", "l4"):
        assert name in PatternRegistry.all()


def test_l_series_class_names():
    assert L1Detector.name == "l1"
    assert L3Detector.name == "l3"
    assert L4Detector.name == "l4"
    assert L1Detector.target_pullback_count == 1
    assert L3Detector.target_pullback_count == 3
    assert L4Detector.target_pullback_count == 4
