"""Breakout-pullback (BP) and failed-breakout (FBO) detector tests."""

from __future__ import annotations

from typing import List, Tuple

from src.brooks.features import BarFeatureExtractor, ExtendedBarFeatures
from src.brooks.patterns import (
    BreakoutPullbackLongDetector,
    BreakoutPullbackShortDetector,
    DetectorContext,
    FailedBreakoutDetector,
    PatternRegistry,
)
from src.brooks.structure import MarketStructureTracker
from src.core.base import Bar

from .conftest import make_series


def _bull(price: float, body: float = 1.0, wick: float = 0.05) -> Tuple[float, float, float, float]:
    return (price, price + body + wick, price - wick, price + body)


def _bear(price: float, body: float = 1.0, wick: float = 0.05) -> Tuple[float, float, float, float]:
    return (price, price + wick, price - body - wick, price - body)


def _flat(price: float, w: float = 0.3) -> Tuple[float, float, float, float]:
    return (price, price + w, price - w, price)


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


def _bp_long_bars() -> List[Bar]:
    """Range, then bull breakout, then small pullback, then with-trend bull bar."""
    vals: list[tuple] = []
    p = 100.0
    # tight range to give a defined breakout level
    for _ in range(10):
        vals.append(_flat(p, w=0.3))
    # explicit bull breakout (close > prior_high)
    bo_open = p
    bo_close = p + 1.5
    vals.append((bo_open, bo_close + 0.05, bo_open - 0.05, bo_close))
    p = bo_close
    # small bear pullback (still above breakout level)
    pb_open = p
    pb_close = p - 0.5
    vals.append((pb_open, pb_open + 0.05, pb_close - 0.05, pb_close))
    p = pb_close
    # with-trend bull bar (close > breakout level which is ~100.3)
    sig_open = p
    sig_close = p + 0.7
    vals.append((sig_open, sig_close + 0.05, sig_open - 0.05, sig_close))
    return make_series(vals)


def _bp_short_bars() -> List[Bar]:
    vals: list[tuple] = []
    p = 100.0
    for _ in range(10):
        vals.append(_flat(p, w=0.3))
    bo_open = p
    bo_close = p - 1.5
    vals.append((bo_open, bo_open + 0.05, bo_close - 0.05, bo_close))
    p = bo_close
    pb_open = p
    pb_close = p + 0.5
    vals.append((pb_open, pb_close + 0.05, pb_open - 0.05, pb_close))
    p = pb_close
    sig_open = p
    sig_close = p - 0.7
    vals.append((sig_open, sig_open + 0.05, sig_close - 0.05, sig_close))
    return make_series(vals)


def _fbo_short_bars() -> List[Bar]:
    """Range, bull breakout, then close back inside range → FBO short."""
    vals: list[tuple] = []
    p = 100.0
    for _ in range(10):
        vals.append(_flat(p, w=0.3))
    bo_open = p
    bo_close = p + 1.0
    vals.append((bo_open, bo_close + 0.05, bo_open - 0.05, bo_close))
    p = bo_close
    # Strong bear bar that closes back inside the prior range
    fail_open = p
    fail_close = 100.0 - 0.5  # decisively back inside
    vals.append((fail_open, fail_open + 0.05, fail_close - 0.05, fail_close))
    return make_series(vals)


def _fbo_long_bars() -> List[Bar]:
    vals: list[tuple] = []
    p = 100.0
    for _ in range(10):
        vals.append(_flat(p, w=0.3))
    bo_open = p
    bo_close = p - 1.0
    vals.append((bo_open, bo_open + 0.05, bo_close - 0.05, bo_close))
    p = bo_close
    fail_open = p
    fail_close = 100.0 + 0.5
    vals.append((fail_open, fail_close + 0.05, fail_open - 0.05, fail_close))
    return make_series(vals)


# ---- BP positives ---------------------------------------------------------


def test_bp_long_fires_after_bull_breakout_pullback():
    sigs = _run(_bp_long_bars(), BreakoutPullbackLongDetector())
    assert len(sigs) == 1
    s = sigs[0]
    assert s.detector == "bp_long"
    assert s.side == "long"
    assert s.entry_px > s.stop_px


def test_bp_short_fires_after_bear_breakout_pullback():
    sigs = _run(_bp_short_bars(), BreakoutPullbackShortDetector())
    assert len(sigs) == 1
    s = sigs[0]
    assert s.detector == "bp_short"
    assert s.side == "short"
    assert s.entry_px < s.stop_px


# ---- FBO positives --------------------------------------------------------


def test_fbo_short_fires_after_failed_bull_breakout():
    sigs = _run(_fbo_short_bars(), FailedBreakoutDetector())
    assert len(sigs) >= 1
    s = sigs[0]
    assert s.detector == "failed_breakout"
    assert s.side == "short"


def test_fbo_long_fires_after_failed_bear_breakout():
    sigs = _run(_fbo_long_bars(), FailedBreakoutDetector())
    assert len(sigs) >= 1
    s = sigs[0]
    assert s.detector == "failed_breakout"
    assert s.side == "long"


# ---- negatives ------------------------------------------------------------


def test_bp_long_no_signal_without_breakout():
    """Quiet range → no breakout → no BP."""
    vals = [_flat(100.0, w=0.3) for _ in range(20)]
    sigs = _run(make_series(vals), BreakoutPullbackLongDetector())
    assert sigs == []


def test_fbo_no_signal_when_breakout_holds():
    """Breakout that continues higher → no FBO."""
    vals: list[tuple] = []
    p = 100.0
    for _ in range(10):
        vals.append(_flat(p, w=0.3))
    # Sustained bull rally — breakout never fails
    for _ in range(5):
        v = _bull(p, body=1.5)
        vals.append(v)
        p = v[3]
    sigs = _run(make_series(vals), FailedBreakoutDetector())
    assert sigs == []


# ---- registry -------------------------------------------------------------


def test_bp_fbo_registered():
    for name in ("bp_long", "bp_short", "failed_breakout"):
        assert name in PatternRegistry.all()
