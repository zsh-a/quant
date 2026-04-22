"""Measured-move TP helper tests."""

from __future__ import annotations

from typing import List

from src.brooks.features import BarFeatureExtractor, ExtendedBarFeatures
from src.brooks.patterns import DetectorContext, MeasuredMoveTargeter
from src.brooks.structure import MarketStructureTracker

from .conftest import make_series


def _build_ctx(bars):
    ext = BarFeatureExtractor(swing_k=2, breakout_lookback=5, atr_period=5)
    tr = MarketStructureTracker(ext, breakout_lookback=5)
    hist: List[ExtendedBarFeatures] = []
    s = None
    f = None
    for b in bars:
        ts_ns = int(b.timestamp.timestamp() * 1_000_000_000)
        f = ext.on_bar(ts_ns, b.open, b.high, b.low, b.close)
        s = tr.on_features(f)
        hist.append(f)
    return DetectorContext(feat=f, structure=s, recent_features=hist)


def test_measured_move_registered_name():
    det = MeasuredMoveTargeter()
    assert det.name == "measured_move"


def test_measured_move_on_bar_always_none():
    ctx = _build_ctx(make_series([(100.0, 100.5, 99.5, 100.0) for _ in range(10)]))
    assert MeasuredMoveTargeter().on_bar(ctx) is None


def test_measured_move_target_long_projects_leg():
    # Simple impulse: low at bar 0, high at bar 5 → leg = 5 → target = entry + 5
    vals = [
        (100.0, 100.2, 99.9, 100.1),
        (100.1, 101.2, 100.0, 101.1),
        (101.1, 102.2, 101.0, 102.1),
        (102.1, 103.2, 102.0, 103.1),
        (103.1, 104.2, 103.0, 104.1),
        (104.1, 105.0, 104.0, 104.9),
    ]
    ctx = _build_ctx(make_series(vals))
    det = MeasuredMoveTargeter()
    tp = det.target(ctx, side="long", entry_px=105.0)
    assert tp is not None
    # Leg = high_after(105.0) − low_at_start(99.9) ≈ 5.1 → tp ≈ 110.1
    assert 109.0 < tp < 111.0


def test_measured_move_target_returns_none_on_short_history():
    ctx = _build_ctx(make_series([(100.0, 100.5, 99.5, 100.0) for _ in range(3)]))
    det = MeasuredMoveTargeter()
    assert det.target(ctx, side="long", entry_px=100.0) is None
