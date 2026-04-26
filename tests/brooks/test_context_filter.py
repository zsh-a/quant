"""ContextFilter — Brooks "background → signal" gate.

Each regime is exercised at least once with both a passing and a
rejecting fixture. The structure-stale guard and micro-channel
steepness check are tested separately.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pytest

from src.brooks.decision.context_filter import (
    CONTINUATION_TYPES,
    ContextDecision,
    ContextFilter,
    PATTERN_TYPE_MAP,
    REVERSAL_TYPES,
    pattern_type_for,
)
from src.brooks.features import SwingPoint
from src.brooks.regime import BrooksRegime
from src.brooks.schema import Signal
from src.brooks.structure import ChannelFit, MarketStructure


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _signal(pattern: str, side: str, *, idx: int = 10) -> Signal:
    return Signal(
        pattern=pattern,
        side=side,  # type: ignore[arg-type]
        signal_bar_idx=idx,
        entry_px=100.0,
        stop_px=99.0 if side == "long" else 101.0,
        probability=0.5,
        quality=0.5,
        source=f"rule:{pattern}",
    )


def _structure(
    *,
    always_in: str = "long",
    leg_dir: str = "up",
    current_idx: int = 50,
    swing_high_at: Optional[int] = 45,
    swing_low_at: Optional[int] = 40,
    atr14: float = 1.0,
    micro_channel_top_slope: Optional[float] = None,
    micro_channel_bot_slope: Optional[float] = None,
) -> MarketStructure:
    s = MarketStructure()
    s.always_in = always_in  # type: ignore[assignment]
    s.current_leg_dir = leg_dir  # type: ignore[assignment]
    s.current_idx = current_idx
    s.atr14 = atr14
    if swing_high_at is not None:
        s.confirmed_swing_highs = [
            SwingPoint(bar_idx=swing_high_at, price=110.0, kind="high", confirmed_at_idx=swing_high_at + 2)
        ]
    if swing_low_at is not None:
        s.confirmed_swing_lows = [
            SwingPoint(bar_idx=swing_low_at, price=90.0, kind="low", confirmed_at_idx=swing_low_at + 2)
        ]
    if micro_channel_top_slope is not None:
        s.micro_channel_top = ChannelFit(
            slope=micro_channel_top_slope,
            intercept=0.0,
            start_idx=current_idx - 5,
            end_idx=current_idx,
            kind="top",
        )
    if micro_channel_bot_slope is not None:
        s.micro_channel_bot = ChannelFit(
            slope=micro_channel_bot_slope,
            intercept=0.0,
            start_idx=current_idx - 5,
            end_idx=current_idx,
            kind="bottom",
        )
    return s


def _check(
    cf: ContextFilter,
    *,
    side: str,
    patterns: Iterable[str],
    regime: BrooksRegime,
    structure: Optional[MarketStructure] = None,
) -> ContextDecision:
    structure = structure or _structure()
    sigs = [_signal(p, side) for p in patterns]
    return cf.check(side=side, signals=sigs, regime=regime, structure=structure)


# ---------------------------------------------------------------------------
# pattern_type_for / map
# ---------------------------------------------------------------------------


def test_pattern_type_for_known_and_unknown():
    assert pattern_type_for("h2") == "pullback"
    assert pattern_type_for("wedge_long") == "wedge"
    assert pattern_type_for("ii_breakout") == "breakout"
    assert pattern_type_for("not_a_real_detector") == "unknown"


def test_continuation_and_reversal_disjoint():
    assert CONTINUATION_TYPES.isdisjoint(REVERSAL_TYPES)
    # Every value in PATTERN_TYPE_MAP is either continuation, reversal, or
    # explicitly bridged (failed_breakout is in reversal).
    classified = CONTINUATION_TYPES | REVERSAL_TYPES
    for cat in PATTERN_TYPE_MAP.values():
        assert cat in classified, f"category {cat!r} not classified"


# ---------------------------------------------------------------------------
# constructor guards
# ---------------------------------------------------------------------------


def test_constructor_rejects_invalid_args():
    with pytest.raises(ValueError):
        ContextFilter(require_reversal_package_for_countertrend=0)
    with pytest.raises(ValueError):
        ContextFilter(micro_channel_steepness_atr=0)
    with pytest.raises(ValueError):
        ContextFilter(max_swing_age_bars=0)


# ---------------------------------------------------------------------------
# UNKNOWN — reject everything
# ---------------------------------------------------------------------------


def test_unknown_regime_rejects_all():
    cf = ContextFilter()
    res = _check(cf, side="long", patterns=["h2"], regime=BrooksRegime.UNKNOWN)
    assert res.allow is False
    assert "regime_unknown" in res.reason
    assert res.regime == "unknown"
    assert res.pattern_types == ["pullback"]


# ---------------------------------------------------------------------------
# STRONG_BULL_TREND
# ---------------------------------------------------------------------------


def test_strong_bull_allows_with_trend_pullback():
    cf = ContextFilter(micro_channel_steepness_atr=None)
    # Pullback signal is only valid when leg is currently *against* the
    # trend (i.e. price is in a pullback, not extending).
    s = _structure(always_in="long", leg_dir="down")
    res = _check(
        cf,
        side="long",
        patterns=["h2"],
        regime=BrooksRegime.STRONG_BULL_TREND,
        structure=s,
    )
    assert res.allow, res.reason
    assert "with_trend_continuation" in res.reason


def test_strong_bull_rejects_pullback_during_extending_leg():
    cf = ContextFilter(micro_channel_steepness_atr=None)
    s = _structure(always_in="long", leg_dir="up")
    res = _check(
        cf,
        side="long",
        patterns=["h2"],
        regime=BrooksRegime.STRONG_BULL_TREND,
        structure=s,
    )
    assert not res.allow
    assert "with_trend_pullback_needs_counter_leg" in res.reason


def test_strong_bull_rejects_pullback_when_leg_flat():
    cf = ContextFilter(micro_channel_steepness_atr=None)
    s = _structure(always_in="long", leg_dir="flat")
    res = _check(
        cf,
        side="long",
        patterns=["h2"],
        regime=BrooksRegime.STRONG_BULL_TREND,
        structure=s,
    )
    assert not res.allow
    assert "with_trend_pullback_needs_counter_leg" in res.reason


def test_strong_bull_rejects_lone_counter_short():
    cf = ContextFilter()
    res = _check(
        cf,
        side="short",
        patterns=["wedge_short"],
        regime=BrooksRegime.STRONG_BULL_TREND,
    )
    assert not res.allow
    assert "counter_strong_trend_needs_reversal_package" in res.reason


def test_strong_bull_allows_counter_with_full_reversal_package():
    cf = ContextFilter()
    res = _check(
        cf,
        side="short",
        patterns=["wedge_short", "double_top"],
        regime=BrooksRegime.STRONG_BULL_TREND,
    )
    assert res.allow, res.reason
    assert "counter_strong_trend_with_reversal_package" in res.reason


def test_strong_bull_blocks_pullback_when_micro_channel_too_steep():
    cf = ContextFilter(micro_channel_steepness_atr=1.0)
    # slope = 1.5 ATR/bar — too steep. leg_dir=down so the
    # leg-extending guard does not pre-empt the steepness check.
    s = _structure(
        atr14=1.0, micro_channel_top_slope=1.5, leg_dir="down", always_in="long"
    )
    res = _check(
        cf,
        side="long",
        patterns=["h2"],
        regime=BrooksRegime.STRONG_BULL_TREND,
        structure=s,
    )
    assert not res.allow
    assert "micro_channel_too_steep" in res.reason


# ---------------------------------------------------------------------------
# WEAK_BULL_TREND
# ---------------------------------------------------------------------------


def test_weak_bull_allows_single_reversal_short():
    cf = ContextFilter()
    res = _check(
        cf,
        side="short",
        patterns=["wedge_short"],
        regime=BrooksRegime.WEAK_BULL_TREND,
    )
    assert res.allow, res.reason
    assert "counter_weak_trend_with_reversal" in res.reason


def test_weak_bull_rejects_counter_without_reversal():
    cf = ContextFilter()
    res = _check(
        cf,
        side="short",
        patterns=["l2"],
        regime=BrooksRegime.WEAK_BULL_TREND,
    )
    assert not res.allow
    assert "counter_weak_trend_needs_reversal" in res.reason


# ---------------------------------------------------------------------------
# STRONG_BEAR_TREND / WEAK_BEAR_TREND
# ---------------------------------------------------------------------------


def test_strong_bear_allows_with_trend_pullback():
    cf = ContextFilter(micro_channel_steepness_atr=None)
    # Pullback in a bear trend = a *bull* leg (against trend), then L2.
    s = _structure(always_in="short", leg_dir="up")
    res = _check(
        cf,
        side="short",
        patterns=["l2"],
        regime=BrooksRegime.STRONG_BEAR_TREND,
        structure=s,
    )
    assert res.allow, res.reason


def test_strong_bear_rejects_lone_counter_long():
    cf = ContextFilter()
    s = _structure(always_in="short", leg_dir="down")
    res = _check(
        cf,
        side="long",
        patterns=["wedge_long"],
        regime=BrooksRegime.STRONG_BEAR_TREND,
        structure=s,
    )
    assert not res.allow


def test_weak_bear_allows_single_reversal_long():
    cf = ContextFilter()
    s = _structure(always_in="short", leg_dir="down")
    res = _check(
        cf,
        side="long",
        patterns=["wedge_long"],
        regime=BrooksRegime.WEAK_BEAR_TREND,
        structure=s,
    )
    assert res.allow, res.reason


# ---------------------------------------------------------------------------
# TIGHT_TRADING_RANGE
# ---------------------------------------------------------------------------


def test_tight_range_blocks_breakout_continuation():
    cf = ContextFilter()
    s = _structure(always_in="neutral", leg_dir="flat")
    res = _check(
        cf,
        side="long",
        patterns=["bp_long"],
        regime=BrooksRegime.TIGHT_TRADING_RANGE,
        structure=s,
    )
    assert not res.allow
    assert "tight_range_blocks_continuation" in res.reason


def test_tight_range_requires_reversal_confluence():
    cf = ContextFilter()
    s = _structure(always_in="neutral", leg_dir="flat")
    # Single reversal pattern is no longer enough — tight TRs are
    # "no-trade" regimes by default.
    lone = _check(
        cf,
        side="short",
        patterns=["double_top"],
        regime=BrooksRegime.TIGHT_TRADING_RANGE,
        structure=s,
    )
    assert not lone.allow
    assert "tight_range_needs_strong_reversal" in lone.reason

    confluence = _check(
        cf,
        side="short",
        patterns=["double_top", "wedge_short"],
        regime=BrooksRegime.TIGHT_TRADING_RANGE,
        structure=s,
    )
    assert confluence.allow
    assert "tight_range_reversal_confluence" in confluence.reason


def test_tight_range_allows_failed_breakout_solo():
    cf = ContextFilter()
    s = _structure(always_in="neutral", leg_dir="flat")
    res = _check(
        cf,
        side="short",
        patterns=["failed_breakout"],
        regime=BrooksRegime.TIGHT_TRADING_RANGE,
        structure=s,
    )
    assert res.allow
    assert "tight_range_failed_breakout" in res.reason


# ---------------------------------------------------------------------------
# BROAD_TRADING_RANGE
# ---------------------------------------------------------------------------


def test_broad_range_allows_reversal_and_breakout_pullback():
    cf = ContextFilter()
    s = _structure(always_in="neutral", leg_dir="flat")
    rev = _check(
        cf,
        side="short",
        patterns=["double_top"],
        regime=BrooksRegime.BROAD_TRADING_RANGE,
        structure=s,
    )
    assert rev.allow
    assert "broad_range_reversal" in rev.reason

    bp = _check(
        cf,
        side="long",
        patterns=["bp_long"],
        regime=BrooksRegime.BROAD_TRADING_RANGE,
        structure=s,
    )
    assert bp.allow
    assert "broad_range_breakout_pullback" in bp.reason


def test_broad_range_rejects_lone_pullback():
    cf = ContextFilter()
    s = _structure(always_in="neutral", leg_dir="flat")
    res = _check(
        cf,
        side="long",
        patterns=["h2"],
        regime=BrooksRegime.BROAD_TRADING_RANGE,
        structure=s,
    )
    assert not res.allow
    assert "broad_range_needs_reversal_or_bp" in res.reason


def test_broad_range_blocks_naked_breakout():
    cf = ContextFilter()
    s = _structure(always_in="neutral", leg_dir="flat")
    res = _check(
        cf,
        side="long",
        patterns=["ii_breakout"],
        regime=BrooksRegime.BROAD_TRADING_RANGE,
        structure=s,
    )
    assert not res.allow
    assert "broad_range_blocks_naked_breakout" in res.reason


# ---------------------------------------------------------------------------
# BREAKOUT_MODE
# ---------------------------------------------------------------------------


def test_breakout_with_direction_allows_bp_and_continuation():
    """First-leg follow-through patterns ride a fresh breakout.

    The earlier "BO + BP only" gate dropped same-direction H1/H2/H3 and
    ii/iii_breakout signals on real strong-breakout sessions (QUA-71:
    2025-01-20 lost the 8k spike). With the relaxed rule any
    with-breakout continuation pattern passes, while structurally-
    aligned reversals (double_bottom in a long breakout) also go
    through.
    """
    cf = ContextFilter()
    s = _structure(always_in="long", leg_dir="up")

    bp = _check(
        cf,
        side="long",
        patterns=["bp_long"],
        regime=BrooksRegime.BREAKOUT_MODE,
        structure=s,
    )
    assert bp.allow, bp.reason
    assert "breakout_with_direction_continuation" in bp.reason

    h2 = _check(
        cf,
        side="long",
        patterns=["h2"],
        regime=BrooksRegime.BREAKOUT_MODE,
        structure=s,
    )
    assert h2.allow, h2.reason
    assert "breakout_with_direction_continuation" in h2.reason
    assert "pullback" in h2.reason

    naked = _check(
        cf,
        side="long",
        patterns=["ii_breakout"],
        regime=BrooksRegime.BREAKOUT_MODE,
        structure=s,
    )
    assert naked.allow, naked.reason
    assert "breakout_with_direction_continuation" in naked.reason
    assert "breakout" in naked.reason

    aligned_reversal = _check(
        cf,
        side="long",
        patterns=["double_bottom"],
        regime=BrooksRegime.BREAKOUT_MODE,
        structure=s,
    )
    assert aligned_reversal.allow, aligned_reversal.reason
    assert "double_bottom" in aligned_reversal.reason


def test_breakout_with_direction_short_continuation():
    """Mirror of the long-side check: bear breakout + L2 short rides."""
    cf = ContextFilter()
    s = _structure(always_in="short", leg_dir="down")
    res = _check(
        cf,
        side="short",
        patterns=["l2"],
        regime=BrooksRegime.BREAKOUT_MODE,
        structure=s,
    )
    assert res.allow, res.reason
    assert "breakout_with_direction_continuation" in res.reason


def test_breakout_with_direction_unknown_pattern_rejected():
    """Same-direction signal whose pattern_type is neither continuation
    nor an aligned reversal still fails — we should not paper over an
    unknown detector by treating breakout mode as a free pass."""
    cf = ContextFilter()
    s = _structure(always_in="long", leg_dir="up")
    res = cf.check(
        side="long",
        signals=[_signal("not_a_real_detector", "long")],
        regime=BrooksRegime.BREAKOUT_MODE,
        structure=s,
    )
    assert not res.allow
    assert "breakout_with_direction_no_continuation" in res.reason


def test_breakout_counter_pure_pullback_rejected():
    """Counter-direction L2 against a fresh long breakout — no reversal
    package present, so default-reject."""
    cf = ContextFilter()
    s = _structure(always_in="long", leg_dir="up")
    res = _check(
        cf,
        side="short",
        patterns=["l2"],
        regime=BrooksRegime.BREAKOUT_MODE,
        structure=s,
    )
    assert not res.allow
    assert "counter_breakout_needs_failure" in res.reason


def test_breakout_counter_with_climax_exhaustion_allowed():
    """Counter-direction signal stacked with a climax-exhaustion marker
    (final_flag / mtr) is allowed — it's the failed-breakout-style fade
    that reversed QUA-70's 2025-01-20 morning spike. Without this carve-
    out the ``REVERSAL_TYPES`` final_flag / mtr would still be dropped
    by the old ``{failed_breakout, wedge, double_*}`` allow-list."""
    cf = ContextFilter()
    s = _structure(always_in="long", leg_dir="up")

    final_flag_combo = _check(
        cf,
        side="short",
        patterns=["l2", "final_flag"],
        regime=BrooksRegime.BREAKOUT_MODE,
        structure=s,
    )
    assert final_flag_combo.allow, final_flag_combo.reason
    assert "counter_breakout_reversal_package" in final_flag_combo.reason
    assert "final_flag" in final_flag_combo.reason

    mtr_combo = _check(
        cf,
        side="short",
        patterns=["l2", "mtr_short"],
        regime=BrooksRegime.BREAKOUT_MODE,
        structure=s,
    )
    assert mtr_combo.allow, mtr_combo.reason
    assert "counter_breakout_reversal_package" in mtr_combo.reason


def test_breakout_counter_failure_allowed():
    cf = ContextFilter()
    s = _structure(always_in="long", leg_dir="up")
    res = _check(
        cf,
        side="short",
        patterns=["failed_breakout"],
        regime=BrooksRegime.BREAKOUT_MODE,
        structure=s,
    )
    assert res.allow
    assert "breakout_failure_reversal" in res.reason


# ---------------------------------------------------------------------------
# CLIMAX
# ---------------------------------------------------------------------------


def test_climax_blocks_with_trend_continuation():
    cf = ContextFilter()
    s = _structure(always_in="long", leg_dir="up")
    res = _check(
        cf,
        side="long",
        patterns=["h2"],
        regime=BrooksRegime.CLIMAX,
        structure=s,
    )
    assert not res.allow
    assert "climax_blocks_continuation" in res.reason


def test_climax_allows_counter_reversal():
    cf = ContextFilter()
    s = _structure(always_in="long", leg_dir="up")
    res = _check(
        cf,
        side="short",
        patterns=["final_flag"],
        regime=BrooksRegime.CLIMAX,
        structure=s,
    )
    assert res.allow
    assert "climax_reversal" in res.reason


# ---------------------------------------------------------------------------
# Structure guards
# ---------------------------------------------------------------------------


def test_stale_swing_age_blocks_signal():
    cf = ContextFilter(max_swing_age_bars=20)
    # current_idx=200, last swing at 50 → age=150 ≫ 20
    s = _structure(current_idx=200, swing_high_at=50, swing_low_at=50)
    res = _check(
        cf,
        side="long",
        patterns=["h2"],
        regime=BrooksRegime.STRONG_BULL_TREND,
        structure=s,
    )
    assert not res.allow
    assert "structure_stale" in res.reason


def test_swing_age_unknown_does_not_block():
    cf = ContextFilter(max_swing_age_bars=20)
    # leg_dir="down" so the with-trend pullback is permitted on its own;
    # otherwise the leg-extending guard would mask the swing-age check.
    s = _structure(
        current_idx=200, swing_high_at=None, swing_low_at=None, leg_dir="down"
    )
    res = _check(
        cf,
        side="long",
        patterns=["h2"],
        regime=BrooksRegime.STRONG_BULL_TREND,
        structure=s,
    )
    # swing_age=None → guard skipped
    assert res.allow
    assert res.swing_age_bars is None


# ---------------------------------------------------------------------------
# 9-regime coverage matrix sanity check
# ---------------------------------------------------------------------------


def test_filter_handles_every_brooks_regime():
    """Ensure no regime crashes the filter and every value of
    :class:`BrooksRegime` is reachable end-to-end. Coverage of the
    actual rule output is validated by the regime-specific tests
    above; this one is a smoke-test for the dispatch itself."""
    cf = ContextFilter()
    # leg_dir="down" so the "with-trend pullback in a real pullback leg"
    # path is exercised; the trend regimes need that to allow the signal.
    s = _structure(leg_dir="down")
    seen: dict[BrooksRegime, ContextDecision] = {}
    for regime in BrooksRegime:
        res = cf.check(
            side="long",
            signals=[_signal("h2", "long"), _signal("bp_long", "long")],
            regime=regime,
            structure=s,
        )
        assert isinstance(res, ContextDecision)
        seen[regime] = res
    assert set(seen.keys()) == set(BrooksRegime)
    # Sanity: at least one regime allows the with-trend pullback and at
    # least one rejects it — proves the dispatch isn't a no-op.
    assert any(r.allow for r in seen.values())
    assert any(not r.allow for r in seen.values())


# ---------------------------------------------------------------------------
# ContextDecision serialization
# ---------------------------------------------------------------------------


def test_context_decision_to_dict_roundtrip():
    cf = ContextFilter()
    res = _check(cf, side="long", patterns=["h2"], regime=BrooksRegime.STRONG_BULL_TREND)
    blob = res.to_dict()
    assert blob["allow"] is res.allow
    assert blob["reason"] == res.reason
    assert blob["regime"] == "strong_bull_trend"
    assert blob["pattern_types"] == ["pullback"]


# ---------------------------------------------------------------------------
# Signal.pattern_type auto-population
# ---------------------------------------------------------------------------


def test_signal_pattern_type_auto_populated():
    sig = _signal("wedge_long", "long")
    assert sig.pattern_type == "wedge"
    sig2 = _signal("not_a_known_detector", "long")
    assert sig2.pattern_type == "unknown"
