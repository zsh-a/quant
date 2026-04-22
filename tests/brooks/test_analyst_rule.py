"""Behavioral tests for :class:`RuleAnalyst`.

Three fixture flavors per the Phase 2.5 spec:
  * ``bull_strong_trend_h2``   — uptrend then H2 setup → expect ``rule:h2`` long signal
  * ``bear_weak_trend_l2``     — downtrend then L2 setup → expect ``rule:l2`` short signal
  * ``range_no_signal``        — sideways chop → no signals
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from src.brooks.analyst import AnalystRegistry, RuleAnalyst
from src.brooks.context import Bar as BrooksBar
from src.brooks.context import BrooksContext, TFSnapshot
from src.brooks.features import BarFeatureExtractor
from src.brooks.patterns import DetectorContext, H2Detector, L2Detector
from src.brooks.patterns.registry import PatternRegistry
from src.brooks.schema import Signal
from src.brooks.structure import MarketStructureTracker
from src.core.base import Bar as CoreBar

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_brooks_bars(core_bars: List[CoreBar]) -> List[BrooksBar]:
    return [
        BrooksBar(
            timestamp_ns=int(b.timestamp.timestamp() * 1_000_000_000),
            open=b.open,
            high=b.high,
            low=b.low,
            close=b.close,
            volume=b.volume,
        )
        for b in core_bars
    ]


def _ctx_from(core_bars: List[CoreBar], symbol: str = "BTCUSDT") -> BrooksContext:
    snap = TFSnapshot(interval="5m", bars=_to_brooks_bars(core_bars))
    return BrooksContext(symbol=symbol, primary=snap)


def _detector_only_signals(core_bars, detector_factory, **extractor_kwargs):
    """Run a single detector standalone — establishes the parity baseline."""
    ext = BarFeatureExtractor(**extractor_kwargs)
    tr = MarketStructureTracker(ext, breakout_lookback=extractor_kwargs.get("breakout_lookback", 20))
    hist = []
    det = detector_factory()
    out = []
    for b in core_bars:
        ts_ns = int(b.timestamp.timestamp() * 1_000_000_000)
        f = ext.on_bar(ts_ns, b.open, b.high, b.low, b.close)
        s = tr.on_features(f)
        hist.append(f)
        sig = det.on_bar(DetectorContext(feat=f, structure=s, recent_features=hist))
        if sig is not None:
            out.append(sig)
    return out


def _run(analyst, ctx) -> List[Signal]:
    return asyncio.run(analyst.analyze(ctx))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def range_no_signal_bars() -> List[CoreBar]:
    """Tight chop — alternating tiny doji-ish bars near 100.0."""
    from .conftest import make_series

    vals = []
    p = 100.0
    for i in range(40):
        o = p
        c = p + (0.05 if i % 2 == 0 else -0.05)
        h = max(o, c) + 0.02
        l = min(o, c) - 0.02
        vals.append((o, h, l, c))
        p = c
    return make_series(vals)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_builds_rule_analyst():
    a = AnalystRegistry.build("rule")
    assert isinstance(a, RuleAnalyst)
    assert a.name == "rule"
    assert "rule" in AnalystRegistry.all()


def test_registry_unknown_name_raises():
    with pytest.raises(KeyError):
        AnalystRegistry.build("does_not_exist")


# ---------------------------------------------------------------------------
# bull_strong_trend_h2 fixture
# ---------------------------------------------------------------------------


def test_bull_strong_trend_h2_emits_h2_long(synthetic_h2_bars):
    ctx = _ctx_from(synthetic_h2_bars)
    analyst = AnalystRegistry.build(
        "rule",
        detector_names=["h2"],
        extractor_kwargs={"swing_k": 2, "breakout_lookback": 8, "atr_period": 5},
        structure_kwargs={"breakout_lookback": 8},
    )
    signals = _run(analyst, ctx)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.pattern == "h2"
    assert sig.side == "long"
    assert sig.source == "rule:h2"
    assert sig.entry_px > sig.stop_px
    assert 0.0 <= sig.probability <= 1.0
    assert 0.0 <= sig.quality <= 1.0


def test_bull_strong_trend_h2_matches_standalone_detector(synthetic_h2_bars):
    """RuleAnalyst output must equal H2Detector output bar-for-bar."""
    baseline = _detector_only_signals(
        synthetic_h2_bars,
        H2Detector,
        swing_k=2,
        breakout_lookback=8,
        atr_period=5,
    )
    ctx = _ctx_from(synthetic_h2_bars)
    analyst = AnalystRegistry.build(
        "rule",
        detector_names=["h2"],
        extractor_kwargs={"swing_k": 2, "breakout_lookback": 8, "atr_period": 5},
        structure_kwargs={"breakout_lookback": 8},
    )
    signals = _run(analyst, ctx)
    # Baseline runs every bar; analyst sees only the last bar — so analyst's
    # signal corresponds to the baseline signal for that final bar (if any).
    last_idx = len(synthetic_h2_bars) - 1
    expected = [s for s in baseline if s.signal_bar_idx == last_idx]
    assert len(signals) == len(expected) == 1
    assert signals[0].signal_bar_idx == expected[0].signal_bar_idx
    assert signals[0].entry_px == expected[0].entry_px
    assert signals[0].stop_px == expected[0].stop_px


# ---------------------------------------------------------------------------
# bear_weak_trend_l2 fixture
# ---------------------------------------------------------------------------


def test_bear_weak_trend_l2_emits_l2_short(synthetic_l2_bars):
    ctx = _ctx_from(synthetic_l2_bars)
    analyst = AnalystRegistry.build(
        "rule",
        detector_names=["l2"],
        extractor_kwargs={"swing_k": 2, "breakout_lookback": 8, "atr_period": 5},
        structure_kwargs={"breakout_lookback": 8},
    )
    signals = _run(analyst, ctx)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.pattern == "l2"
    assert sig.side == "short"
    assert sig.source == "rule:l2"
    assert sig.entry_px < sig.stop_px


def test_bear_weak_trend_l2_matches_standalone_detector(synthetic_l2_bars):
    baseline = _detector_only_signals(
        synthetic_l2_bars,
        L2Detector,
        swing_k=2,
        breakout_lookback=8,
        atr_period=5,
    )
    ctx = _ctx_from(synthetic_l2_bars)
    analyst = AnalystRegistry.build(
        "rule",
        detector_names=["l2"],
        extractor_kwargs={"swing_k": 2, "breakout_lookback": 8, "atr_period": 5},
        structure_kwargs={"breakout_lookback": 8},
    )
    signals = _run(analyst, ctx)
    last_idx = len(synthetic_l2_bars) - 1
    expected = [s for s in baseline if s.signal_bar_idx == last_idx]
    assert len(signals) == len(expected) == 1
    assert signals[0].entry_px == expected[0].entry_px
    assert signals[0].stop_px == expected[0].stop_px


# ---------------------------------------------------------------------------
# range_no_signal fixture
# ---------------------------------------------------------------------------


def test_range_no_signal_emits_nothing(range_no_signal_bars):
    ctx = _ctx_from(range_no_signal_bars)
    analyst = AnalystRegistry.build("rule")  # all registered detectors
    signals = _run(analyst, ctx)
    assert signals == []


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_empty_context_returns_empty_signals():
    ctx = BrooksContext(symbol="BTCUSDT", primary=TFSnapshot(interval="5m", bars=[]))
    analyst = AnalystRegistry.build("rule")
    assert _run(analyst, ctx) == []


def test_default_detector_set_is_full_registry():
    analyst = AnalystRegistry.build("rule")
    # Cover every currently-registered detector.
    assert set(analyst._detector_names) == set(PatternRegistry.all())


def test_signal_meta_carries_timestamp(synthetic_h2_bars):
    ctx = _ctx_from(synthetic_h2_bars)
    analyst = AnalystRegistry.build(
        "rule",
        detector_names=["h2"],
        extractor_kwargs={"swing_k": 2, "breakout_lookback": 8, "atr_period": 5},
        structure_kwargs={"breakout_lookback": 8},
    )
    signals = _run(analyst, ctx)
    assert signals
    assert signals[0].meta.get("timestamp_ns")
