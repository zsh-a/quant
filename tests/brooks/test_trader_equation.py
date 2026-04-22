"""TraderEquation tests — EV math, probability source, reward scaling."""

from __future__ import annotations

import pandas as pd
import pytest

from src.brooks.decision.hit_rate import HitRateKey, HitRateTable
from src.brooks.decision.trader_equation import TraderEquation
from src.brooks.schema import Signal


def _sig(
    *,
    pattern: str = "h2",
    side: str = "long",
    entry: float = 100.0,
    stop: float = 99.0,
    target: float | None = 102.0,
    probability: float = 0.5,
) -> Signal:
    return Signal(
        pattern=pattern,
        side=side,  # type: ignore[arg-type]
        signal_bar_idx=10,
        entry_px=entry,
        stop_px=stop,
        target_px=target,
        probability=probability,
        quality=0.5,
        source=f"rule:{pattern}",
    )


def _table_with(pattern: str, regime: str, htf_aligned: bool, side: str, hit_rate_1r: float, samples: int = 100):
    df = pd.DataFrame(
        [
            {
                "pattern": pattern,
                "regime": regime,
                "htf_aligned": htf_aligned,
                "side": side,
                "samples": samples,
                "hit_rate_1r": hit_rate_1r,
                "hit_rate_2r": hit_rate_1r * 0.6,
                "avg_realized_r": 0.8,
            }
        ]
    )
    return HitRateTable(df)


def test_spec_fixture_long_hits_expected_ev():
    """Spec fixture: p=0.6, reward=2R, risk=1R ⇒ E = 0.55 at cost_r=0.25.

    (The spec quotes ``E ≈ 0.55`` for the stated (p, reward, cost); the
    closed form ``p·reward − (1−p)·1 − cost`` only reaches 0.55 when
    ``cost_r=0.25`` — i.e. a meaningful round-trip execution cost. The
    default ``cost_r=0.05`` produces ``E=0.75`` for the same (p, reward),
    verified separately below.)
    """
    ht = _table_with("h2", "strong_bull_trend", True, "long", hit_rate_1r=0.6)
    te = TraderEquation(ht, cost_r=0.25)
    sig = _sig(entry=100.0, stop=99.0, target=102.0)  # reward = 2R
    p, e = te.score(sig, regime="strong_bull_trend", htf_aligned=True)
    assert p == pytest.approx(0.6)
    assert e == pytest.approx(0.55, abs=1e-9)


def test_formula_at_default_cost():
    """Same fixture at default cost_r=0.05: E = 1.2 - 0.4 - 0.05 = 0.75."""
    ht = _table_with("h2", "strong_bull_trend", True, "long", hit_rate_1r=0.6)
    te = TraderEquation(ht, cost_r=0.05)
    sig = _sig(entry=100.0, stop=99.0, target=102.0)
    _, e = te.score(sig, regime="strong_bull_trend", htf_aligned=True)
    assert e == pytest.approx(0.75, abs=1e-9)


def test_probability_uses_signal_prior_when_bucket_insufficient():
    ht = _table_with("h2", "strong_bull_trend", True, "long", hit_rate_1r=0.9, samples=5)
    te = TraderEquation(ht, cost_r=0.05)
    sig = _sig(probability=0.4, entry=100.0, stop=99.0, target=102.0)
    p, _ = te.score(sig, regime="strong_bull_trend", htf_aligned=True)
    assert p == pytest.approx(0.4)


def test_probability_uses_historical_when_sufficient():
    ht = _table_with("h2", "strong_bull_trend", True, "long", hit_rate_1r=0.75, samples=50)
    te = TraderEquation(ht, cost_r=0.05)
    sig = _sig(probability=0.4, entry=100.0, stop=99.0, target=102.0)
    p, _ = te.score(sig, regime="strong_bull_trend", htf_aligned=True)
    assert p == pytest.approx(0.75)


def test_probability_prior_when_key_missing():
    te = TraderEquation(HitRateTable.empty(), cost_r=0.05)
    sig = _sig(probability=0.3, entry=100.0, stop=99.0, target=102.0)
    p, _ = te.score(sig, regime="weak_bear_trend", htf_aligned=False)
    assert p == pytest.approx(0.3)


def test_reward_scales_with_target_distance():
    te = TraderEquation(HitRateTable.empty(), cost_r=0.0)
    # reward = 3R
    sig = _sig(probability=0.5, entry=100.0, stop=99.0, target=103.0)
    assert te.reward_r(sig) == pytest.approx(3.0)
    # E = 0.5 * 3 - 0.5 * 1 - 0 = 1.0
    _, e = te.score(sig, regime="x", htf_aligned=False)
    assert e == pytest.approx(1.0)


def test_short_side_reward_inverted():
    te = TraderEquation(HitRateTable.empty(), cost_r=0.0)
    sig = _sig(side="short", probability=0.5, entry=100.0, stop=101.0, target=98.0)
    assert te.reward_r(sig) == pytest.approx(2.0)


def test_missing_target_uses_default_reward_r():
    te = TraderEquation(HitRateTable.empty(), cost_r=0.05, default_reward_r=2.0)
    sig = _sig(probability=0.6, entry=100.0, stop=99.0, target=None)
    assert te.reward_r(sig) == pytest.approx(2.0)
    _, e = te.score(sig, regime="x", htf_aligned=False)
    # 0.6*2 - 0.4*1 - 0.05 = 0.75
    assert e == pytest.approx(0.75)


def test_negative_target_clamped_to_zero_reward():
    """A long signal with target below entry is degenerate; reward saturates at 0."""
    te = TraderEquation(HitRateTable.empty(), cost_r=0.0)
    sig = _sig(probability=0.5, entry=100.0, stop=99.0, target=99.5)
    assert te.reward_r(sig) == pytest.approx(0.0)


def test_cost_r_reduces_ev():
    ht = _table_with("h2", "x", True, "long", 0.6)
    sig = _sig(entry=100.0, stop=99.0, target=102.0)
    lo = TraderEquation(ht, cost_r=0.0).score(sig, "x", True)[1]
    hi = TraderEquation(ht, cost_r=0.10).score(sig, "x", True)[1]
    assert lo - hi == pytest.approx(0.10)


def test_constructor_validates_cost_r():
    with pytest.raises(ValueError):
        TraderEquation(HitRateTable.empty(), cost_r=-0.01)


def test_constructor_validates_default_reward():
    with pytest.raises(ValueError):
        TraderEquation(HitRateTable.empty(), default_reward_r=0.0)
