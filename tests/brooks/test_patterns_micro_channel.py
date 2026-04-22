"""Micro-channel failure detector tests."""

from __future__ import annotations

from typing import List, Tuple

from src.brooks.features import BarFeatureExtractor, ExtendedBarFeatures
from src.brooks.patterns import (
    DetectorContext,
    MicroChannelLongDetector,
    MicroChannelShortDetector,
    PatternRegistry,
)
from src.brooks.structure import MarketStructureTracker
from src.core.base import Bar

from .conftest import make_series


def _bull(price: float, body: float = 0.6, wick: float = 0.05) -> Tuple[float, float, float, float]:
    return (price, price + body + wick, price - wick, price + body)


def _bear(price: float, body: float = 0.6, wick: float = 0.05) -> Tuple[float, float, float, float]:
    return (price, price + wick, price - body - wick, price - body)


def _run(bars: List[Bar], detector, *, breakout_lookback=20, swing_k=2):
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


def _bull_micro_channel_then_break() -> List[Bar]:
    """8 bull bars (each higher high & higher low → leg up + positive top slope), then one strong bear bar."""
    vals: list[tuple] = []
    p = 100.0
    for _ in range(8):
        v = _bull(p, body=1.0)
        vals.append(v)
        p = v[3]
    # break bar: large bear that closes well below the projected channel bot
    target = p - 4.0
    vals.append((p, p + 0.05, target - 0.05, target))
    return make_series(vals)


def _bear_micro_channel_then_break() -> List[Bar]:
    vals: list[tuple] = []
    p = 100.0
    for _ in range(8):
        v = _bear(p, body=1.0)
        vals.append(v)
        p = v[3]
    target = p + 4.0
    vals.append((p, target + 0.05, p - 0.05, target))
    return make_series(vals)


# ---- positives ------------------------------------------------------------


def test_micro_channel_short_fires_on_bull_channel_break():
    sigs = _run(_bull_micro_channel_then_break(), MicroChannelShortDetector(min_channel_bars=5))
    assert len(sigs) >= 1
    s = sigs[-1]
    assert s.detector == "micro_channel_short"
    assert s.side == "short"
    assert s.entry_px < s.stop_px
    assert s.metadata["channel_top_slope"] > 0
    assert s.metadata["channel_len"] >= 5


def test_micro_channel_long_fires_on_bear_channel_break():
    sigs = _run(_bear_micro_channel_then_break(), MicroChannelLongDetector(min_channel_bars=5))
    assert len(sigs) >= 1
    s = sigs[-1]
    assert s.detector == "micro_channel_long"
    assert s.side == "long"
    assert s.entry_px > s.stop_px
    assert s.metadata["channel_bot_slope"] < 0


def test_micro_channel_short_includes_projected_bot_in_metadata():
    sigs = _run(_bull_micro_channel_then_break(), MicroChannelShortDetector(min_channel_bars=5))
    assert sigs
    md = sigs[-1].metadata
    assert "projected_bot" in md
    assert "channel_len" in md and md["channel_len"] >= 5


# ---- negatives ------------------------------------------------------------


def test_micro_channel_short_no_signal_in_pure_bull_run():
    """No bear break bar → no failure signal."""
    vals = []
    p = 100.0
    for _ in range(15):
        v = _bull(p, body=1.0)
        vals.append(v)
        p = v[3]
    sigs = _run(make_series(vals), MicroChannelShortDetector(min_channel_bars=5))
    assert sigs == []


def test_micro_channel_long_no_signal_when_channel_too_short():
    """min_channel_bars=10 but only 8 bars in channel → no signal."""
    sigs = _run(_bear_micro_channel_then_break(), MicroChannelLongDetector(min_channel_bars=10))
    assert sigs == []


# ---- registry -------------------------------------------------------------


def test_micro_channel_registered():
    assert "micro_channel_long" in PatternRegistry.all()
    assert "micro_channel_short" in PatternRegistry.all()
