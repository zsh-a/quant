"""H2 / L2 FSM behavioral tests."""

from __future__ import annotations

from typing import List

from src.analysis.brooks.features import BarFeatureExtractor, ExtendedBarFeatures
from src.analysis.brooks.patterns import DetectorContext, H2Detector, L2Detector
from src.analysis.brooks.structure import MarketStructureTracker
from src.core.base import Bar


def _run(bars: List[Bar], detector, *, breakout_lookback=8, swing_k=2):
    ext = BarFeatureExtractor(swing_k=swing_k, breakout_lookback=breakout_lookback, atr_period=5)
    tr = MarketStructureTracker(ext, breakout_lookback=breakout_lookback)
    hist: List[ExtendedBarFeatures] = []
    states = []
    signals = []
    for b in bars:
        ts_ns = int(b.timestamp.timestamp() * 1_000_000_000)
        f = ext.on_bar(ts_ns, b.open, b.high, b.low, b.close)
        s = tr.on_features(f)
        hist.append(f)
        ctx = DetectorContext(feat=f, structure=s, recent_features=hist)
        sig = detector.on_bar(ctx)
        states.append(detector.state_snapshot()["state"])
        if sig is not None:
            signals.append(sig)
    return states, signals


def test_h2_forms_and_emits_signal(synthetic_h2_bars):
    det = H2Detector()
    states, signals = _run(synthetic_h2_bars, det, breakout_lookback=8, swing_k=2)
    assert "H2_FORMED" in states, f"state trace: {states}"
    assert len(signals) >= 1
    sig = signals[0]
    assert sig.side == "long"
    assert sig.entry_px > sig.stop_px  # long: entry above stop


def test_l2_forms_and_emits_signal(synthetic_l2_bars):
    det = L2Detector()
    states, signals = _run(synthetic_l2_bars, det, breakout_lookback=8, swing_k=2)
    assert "H2_FORMED" in states  # shared state machine naming
    assert len(signals) >= 1
    sig = signals[0]
    assert sig.side == "short"
    assert sig.entry_px < sig.stop_px  # short: entry below stop


def test_h2_no_signal_when_always_in_short():
    """If always_in is short, H2 (long-side) should never fire."""
    from .conftest import make_series

    vals = []
    # Strictly declining bars → always_in short
    p = 100.0
    for _ in range(25):
        o = p
        c = p - 0.5
        h = o + 0.05
        l = c - 0.05
        vals.append((o, h, l, c))
        p = c
    bars = make_series(vals)
    det = H2Detector()
    _, signals = _run(bars, det, breakout_lookback=8, swing_k=2)
    assert signals == []
