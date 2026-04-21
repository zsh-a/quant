"""L3 BrooksRegimeClassifier tests.

Each test constructs a fixture bar sequence that isolates a single target
regime, feeds it through the L1 extractor + L2 tracker + L3 classifier, and
asserts the final classification.  The state-transition test verifies that
``strong_bull → climax → weak_bull`` is reached with latency < 3 bars on
both transitions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.brooks.features import BarFeatureExtractor, ExtendedBarFeatures
from src.brooks.regime import BrooksRegime, BrooksRegimeClassifier, RegimeSnapshot
from src.brooks.structure import MarketStructureTracker

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _classify_series(
    bars: List[Tuple[int, float, float, float, float]],
    clf: Optional[BrooksRegimeClassifier] = None,
    *,
    ema_period: int = 10,
    atr_period: int = 5,
    breakout_lookback: int = 10,
    swing_k: int = 2,
) -> Tuple[List[ExtendedBarFeatures], List[RegimeSnapshot], BrooksRegimeClassifier]:
    if clf is None:
        clf = BrooksRegimeClassifier(tr_lookback=10, breakout_decay_bars=5)
    ext = BarFeatureExtractor(
        swing_k=swing_k,
        ema_period=ema_period,
        atr_period=atr_period,
        breakout_lookback=breakout_lookback,
    )
    tr = MarketStructureTracker(ext, breakout_lookback=breakout_lookback)
    features: List[ExtendedBarFeatures] = []
    snaps: List[RegimeSnapshot] = []
    for b in bars:
        f = ext.on_bar(*b)
        s = tr.on_features(f)
        features.append(f)
        snaps.append(clf.classify(features, s))
    return features, snaps, clf


def _warmup_sideways(n: int, start_idx: int = 0) -> List[Tuple[int, float, float, float, float]]:
    return [(start_idx + i, 100.0, 100.2, 99.8, 100.0) for i in range(n)]


# ---------------------------------------------------------------------------
# basic cases
# ---------------------------------------------------------------------------


def test_unknown_when_insufficient_data():
    clf = BrooksRegimeClassifier(tr_lookback=10)
    bars = _warmup_sideways(5)
    _, snaps, _ = _classify_series(bars, clf=clf)
    assert snaps[-1].regime == BrooksRegime.UNKNOWN
    assert snaps[-1].confidence == 0.0
    assert any("insufficient_data" in r for r in snaps[-1].reasons)


def test_snapshot_to_dict_roundtrip():
    clf = BrooksRegimeClassifier(tr_lookback=10)
    _, snaps, _ = _classify_series(_warmup_sideways(5), clf=clf)
    d = snaps[-1].to_dict()
    assert d["regime"] == "unknown"
    assert d["confidence"] == 0.0
    assert isinstance(d["reasons"], list)


# ---------------------------------------------------------------------------
# climax
# ---------------------------------------------------------------------------


def test_bull_climax_after_consecutive_trend_bars():
    clf = BrooksRegimeClassifier(
        tr_lookback=10, climax_consecutive_bars=3, climax_ema_dist_atr=2.0
    )
    bars = _warmup_sideways(10)
    # 3 huge bull trend bars — body_pct=100, ema_atr_distance ≫ 2.
    bars.append((10, 100.0, 106.0, 100.0, 106.0))
    bars.append((11, 106.0, 112.0, 106.0, 112.0))
    bars.append((12, 112.0, 118.0, 112.0, 118.0))
    _, snaps, _ = _classify_series(bars, clf=clf)
    assert snaps[-1].regime == BrooksRegime.CLIMAX
    assert snaps[-1].confidence > 0.5
    assert snaps[-1].consecutive_trend_bars >= 3


def test_bear_climax_after_consecutive_trend_bars():
    clf = BrooksRegimeClassifier(
        tr_lookback=10, climax_consecutive_bars=3, climax_ema_dist_atr=2.0
    )
    bars = _warmup_sideways(10)
    bars.append((10, 100.0, 100.0, 94.0, 94.0))
    bars.append((11, 94.0, 94.0, 88.0, 88.0))
    bars.append((12, 88.0, 88.0, 82.0, 82.0))
    _, snaps, _ = _classify_series(bars, clf=clf)
    assert snaps[-1].regime == BrooksRegime.CLIMAX
    assert snaps[-1].confidence > 0.5


def test_climax_confidence_monotonic_with_margin():
    """Farther past the ema_dist threshold ⇒ higher confidence."""
    low = BrooksRegimeClassifier(tr_lookback=10, climax_ema_dist_atr=2.0)
    high = BrooksRegimeClassifier(tr_lookback=10, climax_ema_dist_atr=2.0)

    bars_low = _warmup_sideways(10)
    bars_low.append((10, 100.0, 103.0, 100.0, 103.0))
    bars_low.append((11, 103.0, 105.5, 103.0, 105.5))
    bars_low.append((12, 105.5, 107.5, 105.5, 107.5))

    bars_high = _warmup_sideways(10)
    bars_high.append((10, 100.0, 110.0, 100.0, 110.0))
    bars_high.append((11, 110.0, 122.0, 110.0, 122.0))
    bars_high.append((12, 122.0, 140.0, 122.0, 140.0))

    _, snaps_low, _ = _classify_series(bars_low, clf=low)
    _, snaps_high, _ = _classify_series(bars_high, clf=high)

    assert snaps_low[-1].regime == BrooksRegime.CLIMAX
    assert snaps_high[-1].regime == BrooksRegime.CLIMAX
    assert snaps_high[-1].confidence > snaps_low[-1].confidence
    assert snaps_low[-1].confidence < 1.0  # near threshold → < 1


# ---------------------------------------------------------------------------
# breakout mode
# ---------------------------------------------------------------------------


def test_breakout_mode_for_fresh_breakout():
    clf = BrooksRegimeClassifier(tr_lookback=10, breakout_decay_bars=5)
    bars = _warmup_sideways(10)
    # Single breakout bar with small body (body_pct < 60 ⇒ not climax).
    bars.append((10, 100.0, 101.0, 99.9, 100.8))
    _, snaps, _ = _classify_series(bars, clf=clf)
    assert snaps[-1].regime == BrooksRegime.BREAKOUT_MODE
    assert snaps[-1].confidence > 0.5


def test_breakout_mode_decays():
    clf = BrooksRegimeClassifier(tr_lookback=10, breakout_decay_bars=5)
    bars = _warmup_sideways(10)
    bars.append((10, 100.0, 101.0, 99.9, 100.8))
    # Price holds above prior range but bars don't make new closes (no new
    # always_in flips) → breakout mode confidence should decay.
    for i in range(4):
        bars.append((11 + i, 100.8, 100.95, 100.6, 100.75))
    _, snaps, _ = _classify_series(bars, clf=clf)
    confs_in_breakout = [s.confidence for s in snaps if s.regime == BrooksRegime.BREAKOUT_MODE]
    assert len(confs_in_breakout) >= 2
    # First breakout bar has higher confidence than the last one in the run.
    assert confs_in_breakout[0] >= confs_in_breakout[-1]


# ---------------------------------------------------------------------------
# trading ranges
# ---------------------------------------------------------------------------


def test_tight_trading_range_neutral_and_overlapping():
    clf = BrooksRegimeClassifier(tr_lookback=10, tight_tr_overlap_threshold=0.5)
    bars = [(i, 100.0, 100.2, 99.8, 100.0) for i in range(15)]
    _, snaps, _ = _classify_series(bars, clf=clf)
    assert snaps[-1].regime == BrooksRegime.TIGHT_TRADING_RANGE
    assert snaps[-1].bar_overlap_ratio > 0.5


def test_broad_trading_range_wide_swing_low_overlap():
    clf = BrooksRegimeClassifier(tr_lookback=10, tight_tr_overlap_threshold=0.5)
    bars = _warmup_sideways(10)
    # Alternate narrow bars at 100 with wide bars spanning [97, 103].
    for i in range(10):
        if i % 2 == 0:
            bars.append((10 + i, 100.0, 103.0, 97.0, 100.0))  # wide
        else:
            bars.append((10 + i, 100.0, 100.3, 99.7, 100.0))  # narrow
    _, snaps, _ = _classify_series(bars, clf=clf)
    assert snaps[-1].regime == BrooksRegime.BROAD_TRADING_RANGE
    assert snaps[-1].swing_range_atr > 2.0
    assert snaps[-1].bar_overlap_ratio < 0.5


# ---------------------------------------------------------------------------
# trend regimes
# ---------------------------------------------------------------------------


def test_strong_bull_trend():
    clf = BrooksRegimeClassifier(
        tr_lookback=10,
        breakout_decay_bars=5,
        climax_ema_dist_atr=5.0,  # relax to avoid climaxing the quiet continuation
    )
    bars = _warmup_sideways(10)
    # One decisive breakout bar then a smooth continuation that stays below
    # bar 10's high (so no repeated breakouts re-arm BREAKOUT_MODE).
    bars.append((10, 100.0, 110.0, 99.9, 109.5))
    for i in range(10):
        base = 107.5 + i * 0.05
        # range 2.0, body 1.1 → body_pct ≈ 55%, close near high
        bars.append((11 + i, base, base + 1.5, base - 0.5, base + 1.1))
    _, snaps, _ = _classify_series(bars, clf=clf)
    assert snaps[-1].regime == BrooksRegime.STRONG_BULL_TREND
    assert snaps[-1].confidence > 0.5
    assert "pullback_ratio" in "".join(snaps[-1].reasons)


def test_weak_bull_trend():
    clf = BrooksRegimeClassifier(tr_lookback=10, breakout_decay_bars=5)
    bars = _warmup_sideways(10)
    bars.append((10, 100.0, 102.0, 99.9, 101.8))
    # Alternating bull/bear bars — always_in stays long but trend_bar_ratio
    # is low and pullbacks are frequent.
    pattern = [
        (101.8, 102.3, 101.3, 101.9),
        (101.9, 102.1, 101.0, 101.2),
        (101.2, 102.2, 101.1, 102.0),
        (102.0, 102.3, 101.3, 101.5),
        (101.5, 102.2, 101.2, 102.0),
        (102.0, 102.3, 101.3, 101.5),
        (101.5, 102.2, 101.2, 102.0),
        (102.0, 102.3, 101.3, 101.5),
        (101.5, 102.2, 101.2, 102.0),
        (102.0, 102.3, 101.3, 101.5),
    ]
    for i, (o, h, l, c) in enumerate(pattern):
        bars.append((11 + i, o, h, l, c))
    _, snaps, _ = _classify_series(bars, clf=clf)
    assert snaps[-1].regime == BrooksRegime.WEAK_BULL_TREND


def test_strong_bear_trend():
    clf = BrooksRegimeClassifier(
        tr_lookback=10, breakout_decay_bars=5, climax_ema_dist_atr=5.0
    )
    bars = _warmup_sideways(10)
    bars.append((10, 100.0, 100.1, 90.0, 90.5))  # big breakdown
    for i in range(10):
        base = 92.5 - i * 0.05
        # close near low, body ≈ 55%
        bars.append((11 + i, base, base + 0.5, base - 1.5, base - 1.1))
    _, snaps, _ = _classify_series(bars, clf=clf)
    assert snaps[-1].regime == BrooksRegime.STRONG_BEAR_TREND
    assert snaps[-1].confidence > 0.5


def test_weak_bear_trend():
    clf = BrooksRegimeClassifier(tr_lookback=10, breakout_decay_bars=5)
    bars = _warmup_sideways(10)
    bars.append((10, 100.0, 100.1, 98.0, 98.2))
    pattern = [
        (98.2, 98.7, 97.7, 98.1),
        (98.1, 99.0, 97.9, 98.8),
        (98.8, 98.8, 97.8, 98.0),
        (98.0, 98.7, 97.7, 98.5),
        (98.5, 98.7, 97.8, 98.0),
        (98.0, 98.8, 97.8, 98.5),
        (98.5, 98.7, 97.7, 98.0),
        (98.0, 98.8, 97.8, 98.5),
        (98.5, 98.7, 97.7, 98.0),
        (98.0, 98.8, 97.8, 98.5),
    ]
    for i, (o, h, l, c) in enumerate(pattern):
        bars.append((11 + i, o, h, l, c))
    _, snaps, _ = _classify_series(bars, clf=clf)
    assert snaps[-1].regime == BrooksRegime.WEAK_BEAR_TREND


# ---------------------------------------------------------------------------
# state transitions
# ---------------------------------------------------------------------------


def test_transition_strong_bull_to_climax_to_weak_bull_within_3_bars():
    """Verify the path strong_bull → climax → weak_bull with latency < 3 bars.

    The fixture:
      * 10 warmup sideways bars
      * 1 breakout bar (always_in flips to long → fresh BREAKOUT_MODE)
      * 10 smooth continuation bars (strong_bull kicks in once breakout decays)
      * 3 huge bull trend bars (climax)
      * 1 decisive bear pullback bar (weak_bull — not all-bull, deeper retrace)
    """
    clf = BrooksRegimeClassifier(
        tr_lookback=10,
        breakout_decay_bars=5,
        climax_ema_dist_atr=2.0,
        climax_consecutive_bars=3,
    )
    bars = _warmup_sideways(10)
    bars.append((10, 100.0, 110.0, 99.9, 109.5))  # breakout
    # smooth continuation — keep bars below 110 so no new breakouts
    for i in range(10):
        base = 107.5 + i * 0.05
        bars.append((11 + i, base, base + 1.5, base - 0.5, base + 1.1))
    # climax — 3 huge bull trend bars
    bars.append((21, 109.5, 118.0, 109.5, 118.0))
    bars.append((22, 118.0, 128.0, 118.0, 128.0))
    bars.append((23, 128.0, 140.0, 128.0, 140.0))
    # pullback bar → weak_bull (trend_bull_ratio still ≥ 0.6 but pullback deep)
    bars.append((24, 140.0, 140.0, 120.0, 121.0))

    _, snaps, _ = _classify_series(bars, clf=clf)

    # Find first occurrence of each regime in the expected order.
    def first_idx(regime: BrooksRegime, start: int = 0) -> Optional[int]:
        for i in range(start, len(snaps)):
            if snaps[i].regime == regime:
                return i
        return None

    strong_idx = first_idx(BrooksRegime.STRONG_BULL_TREND)
    assert strong_idx is not None, f"never saw STRONG_BULL — path was {[s.regime for s in snaps]}"

    climax_idx = first_idx(BrooksRegime.CLIMAX, start=strong_idx)
    assert climax_idx is not None, "never saw CLIMAX after STRONG_BULL"

    weak_idx = first_idx(BrooksRegime.WEAK_BULL_TREND, start=climax_idx)
    assert weak_idx is not None, "never saw WEAK_BULL after CLIMAX"

    # Transition latencies — bars from when the new conditions first apply
    # on the tape to when the classifier picks them up.  Climax starts
    # forming at bar 21 (the first big trend bar); the 3-bar confirmation
    # means detection can't happen before bar 23.
    climax_condition_start = 21  # first big bull trend bar in the fixture
    weak_condition_start = 24  # pullback bar after climax completes

    assert climax_idx - climax_condition_start < 3, (
        f"climax detection latency {climax_idx - climax_condition_start} ≥ 3"
    )
    assert weak_idx - weak_condition_start < 3, (
        f"weak-bull detection latency {weak_idx - weak_condition_start} ≥ 3"
    )
    # The regime must reach STRONG_BULL before the climax bars arrive.
    assert strong_idx < climax_condition_start, (
        f"STRONG_BULL must precede CLIMAX (strong_idx={strong_idx})"
    )


# ---------------------------------------------------------------------------
# construction guards
# ---------------------------------------------------------------------------


def test_invalid_constructor_args():
    import pytest

    with pytest.raises(ValueError):
        BrooksRegimeClassifier(climax_consecutive_bars=0)
    with pytest.raises(ValueError):
        BrooksRegimeClassifier(climax_ema_dist_atr=0)
    with pytest.raises(ValueError):
        BrooksRegimeClassifier(strong_trend_retracement_pct=0)
    with pytest.raises(ValueError):
        BrooksRegimeClassifier(strong_trend_retracement_pct=1.0)
    with pytest.raises(ValueError):
        BrooksRegimeClassifier(tight_tr_overlap_threshold=0)
    with pytest.raises(ValueError):
        BrooksRegimeClassifier(tr_lookback=1)
    with pytest.raises(ValueError):
        BrooksRegimeClassifier(breakout_decay_bars=0)
