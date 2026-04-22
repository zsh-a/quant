"""Major Trend Reversal (MTR) detector tests."""

from __future__ import annotations

from typing import List, Tuple

from src.brooks.features import BarFeatureExtractor, ExtendedBarFeatures
from src.brooks.patterns import (
    DetectorContext,
    MTRLongDetector,
    MTRShortDetector,
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


def _mtr_short_bars() -> List[Bar]:
    """Bull rally → high1 → pullback → high2 (lower than high1) → strong bear break of pullback low."""
    vals: list[tuple] = []
    p = 100.0
    # bull rally up to high1 ~ 110
    for _ in range(8):
        v = _bull(p, body=1.2)
        vals.append(v)
        p = v[3]
    # explicit higher pivot bar (creates a confirmed swing high after K bars)
    high1 = p + 0.5
    vals.append((p, high1, p - 0.05, p + 0.3))
    p = p + 0.3
    # pullback (3 bear bars) -> swing low
    for _ in range(3):
        v = _bear(p, body=0.8)
        vals.append(v)
        p = v[3]
    pb_low = p
    # rally back to a LOWER high
    for _ in range(3):
        v = _bull(p, body=0.6)
        vals.append(v)
        p = v[3]
    high2 = p + 0.3
    assert high2 < high1, f"high2 {high2} should be < high1 {high1}"
    vals.append((p, high2, p - 0.05, p + 0.1))
    p = p + 0.1
    # filler bears so swing high2 confirms
    for _ in range(3):
        v = _bear(p, body=0.5)
        vals.append(v)
        p = v[3]
    # strong bear break of pb_low
    target = pb_low - 0.5
    body = p - target
    vals.append((p, p + 0.05, target - 0.05, target))
    return make_series(vals)


def _mtr_long_bars() -> List[Bar]:
    """Mirror: bear leg → low1 → pullback up → low2 (higher than low1) → strong bull break of recent high."""
    vals: list[tuple] = []
    p = 100.0
    for _ in range(8):
        v = _bear(p, body=1.2)
        vals.append(v)
        p = v[3]
    # explicit lower pivot bar
    low1 = p - 0.5
    vals.append((p, p + 0.05, low1, p - 0.3))
    p = p - 0.3
    # pullback up
    for _ in range(3):
        v = _bull(p, body=0.8)
        vals.append(v)
        p = v[3]
    pb_high = p
    # back down to a HIGHER low
    for _ in range(3):
        v = _bear(p, body=0.6)
        vals.append(v)
        p = v[3]
    low2 = p - 0.3
    assert low2 > low1, f"low2 {low2} should be > low1 {low1}"
    vals.append((p, p + 0.05, low2, p - 0.1))
    p = p - 0.1
    for _ in range(3):
        v = _bull(p, body=0.5)
        vals.append(v)
        p = v[3]
    target = pb_high + 0.5
    vals.append((p, target + 0.05, p - 0.05, target))
    return make_series(vals)


# ---- positives ------------------------------------------------------------


def test_mtr_short_fires_on_lower_high_break():
    sigs = _run(_mtr_short_bars(), MTRShortDetector())
    assert len(sigs) >= 1
    s = sigs[0]
    assert s.detector == "mtr_short"
    assert s.side == "short"
    assert s.entry_px < s.stop_px
    assert s.metadata["lower_high_px"] < s.metadata["prior_high_px"]


def test_mtr_long_fires_on_higher_low_break():
    sigs = _run(_mtr_long_bars(), MTRLongDetector())
    assert len(sigs) >= 1
    s = sigs[0]
    assert s.detector == "mtr_long"
    assert s.side == "long"
    assert s.entry_px > s.stop_px
    assert s.metadata["higher_low_px"] > s.metadata["prior_low_px"]


def test_mtr_short_emits_metadata_with_swing_indices():
    sigs = _run(_mtr_short_bars(), MTRShortDetector())
    assert sigs
    md = sigs[0].metadata
    for key in ("lower_high_idx", "prior_high_idx", "broken_low_idx"):
        assert key in md and md[key] >= 0


# ---- negatives ------------------------------------------------------------


def test_mtr_short_no_signal_in_pure_uptrend():
    """Strict bull rally with no lower-high → no MTR short."""
    vals = []
    p = 100.0
    for _ in range(30):
        v = _bull(p, body=1.0)
        vals.append(v)
        p = v[3]
    sigs = _run(make_series(vals), MTRShortDetector())
    assert sigs == []


def test_mtr_long_no_signal_in_pure_downtrend():
    vals = []
    p = 100.0
    for _ in range(30):
        v = _bear(p, body=1.0)
        vals.append(v)
        p = v[3]
    sigs = _run(make_series(vals), MTRLongDetector())
    assert sigs == []


# ---- registry -------------------------------------------------------------


def test_mtr_registered():
    assert "mtr_long" in PatternRegistry.all()
    assert "mtr_short" in PatternRegistry.all()
