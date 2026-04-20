"""L2 MarketStructure tests."""

from __future__ import annotations

from src.analysis.brooks.features import BarFeatureExtractor
from src.analysis.brooks.structure import MarketStructureTracker


def test_always_in_long_on_breakout():
    ext = BarFeatureExtractor(swing_k=2, breakout_lookback=5)
    tr = MarketStructureTracker(ext, breakout_lookback=5)
    # 5 bars hugging 100
    for i in range(5):
        f = ext.on_bar(i, 100.0, 100.5, 99.5, 100.0)
        tr.on_features(f)
    # break out above
    f = ext.on_bar(5, 100.5, 102.5, 100.4, 102.3)
    s = tr.on_features(f)
    assert s.always_in == "long"
    assert s.breakout_state == "bull_breakout"


def test_always_in_short_on_breakdown():
    ext = BarFeatureExtractor(swing_k=2, breakout_lookback=5)
    tr = MarketStructureTracker(ext, breakout_lookback=5)
    for i in range(5):
        f = ext.on_bar(i, 100.0, 100.5, 99.5, 100.0)
        tr.on_features(f)
    f = ext.on_bar(5, 99.5, 99.6, 97.5, 97.7)
    s = tr.on_features(f)
    assert s.always_in == "short"
    assert s.breakout_state == "bear_breakout"


def test_always_in_sticky_during_chop():
    """Once always-in is set, chop should not flip it."""
    ext = BarFeatureExtractor(swing_k=2, breakout_lookback=5)
    tr = MarketStructureTracker(ext, breakout_lookback=5)
    for i in range(5):
        f = ext.on_bar(i, 100.0, 100.5, 99.5, 100.0)
        tr.on_features(f)
    f = ext.on_bar(5, 100.5, 102.5, 100.4, 102.3)
    tr.on_features(f)  # long
    # a neutral bar
    f = ext.on_bar(6, 102.0, 102.3, 101.5, 101.8)
    s = tr.on_features(f)
    assert s.always_in == "long"  # unchanged
