"""HitRateTable tests — parquet round-trip, lookup, and prior fallback."""

from __future__ import annotations

import pandas as pd
import pytest

from src.brooks.decision.hit_rate import (
    DEFAULT_PRIOR_HIT_RATE,
    HitRateKey,
    HitRateTable,
)


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pattern": "h2",
                "regime": "strong_bull_trend",
                "htf_aligned": True,
                "side": "long",
                "samples": 120,
                "hit_rate_1r": 0.62,
                "hit_rate_2r": 0.41,
                "avg_realized_r": 0.85,
            },
            {
                "pattern": "l2",
                "regime": "weak_bear_trend",
                "htf_aligned": False,
                "side": "short",
                "samples": 12,  # < default min_samples
                "hit_rate_1r": 0.50,
                "hit_rate_2r": 0.30,
                "avg_realized_r": 0.10,
            },
        ]
    )


def test_lookup_hit_returns_bucket_stats():
    table = HitRateTable(_make_df())
    stats = table.lookup(HitRateKey("h2", "strong_bull_trend", True, "long"))
    assert stats["samples"] == 120
    assert stats["hit_rate_1r"] == pytest.approx(0.62)
    assert stats["hit_rate_2r"] == pytest.approx(0.41)
    assert stats["avg_r"] == pytest.approx(0.85)


def test_lookup_miss_returns_prior():
    table = HitRateTable(_make_df())
    stats = table.lookup(HitRateKey("wedge_long", "tight_trading_range", True, "long"))
    assert stats["samples"] == 0
    assert stats["hit_rate_1r"] == pytest.approx(DEFAULT_PRIOR_HIT_RATE)


def test_lookup_custom_default_prior():
    table = HitRateTable.empty()
    stats = table.lookup(HitRateKey("h2", "strong_bull_trend", True, "long"), default=0.7)
    assert stats["hit_rate_1r"] == pytest.approx(0.7)


def test_is_sufficient_threshold():
    table = HitRateTable(_make_df())
    assert table.is_sufficient(HitRateKey("h2", "strong_bull_trend", True, "long"))
    # Bucket has only 12 samples — below default 30.
    assert not table.is_sufficient(HitRateKey("l2", "weak_bear_trend", False, "short"))
    # Custom min_samples
    assert table.is_sufficient(HitRateKey("l2", "weak_bear_trend", False, "short"), min_samples=10)
    # Missing key always insufficient
    assert not table.is_sufficient(HitRateKey("missing", "x", True, "long"))


def test_pandas_round_trip(tmp_path):
    table = HitRateTable(_make_df())
    out = tmp_path / "hit_rate.parquet"
    table.save(out)
    assert out.exists()
    reloaded = HitRateTable.load(out)
    assert len(reloaded) == 2
    assert reloaded.lookup(HitRateKey("h2", "strong_bull_trend", True, "long"))["samples"] == 120
    assert reloaded.lookup(HitRateKey("l2", "weak_bear_trend", False, "short"))["samples"] == 12


def test_load_missing_path_returns_empty(tmp_path):
    table = HitRateTable.load(tmp_path / "does_not_exist.parquet")
    assert len(table) == 0
    # Empty table → prior fallback for any lookup
    stats = table.lookup(HitRateKey("h2", "strong_bull_trend", True, "long"))
    assert stats["samples"] == 0


def test_save_empty_writes_canonical_schema(tmp_path):
    out = tmp_path / "empty.parquet"
    HitRateTable.empty().save(out)
    df = pd.read_parquet(out)
    assert df.empty
    assert list(df.columns) == [
        "pattern",
        "regime",
        "htf_aligned",
        "side",
        "samples",
        "hit_rate_1r",
        "hit_rate_2r",
        "avg_realized_r",
    ]


def test_missing_required_column_rejected():
    bad = _make_df().drop(columns=["hit_rate_2r"])
    with pytest.raises(ValueError, match="missing columns"):
        HitRateTable(bad)


def test_contains_membership():
    table = HitRateTable(_make_df())
    assert HitRateKey("h2", "strong_bull_trend", True, "long") in table
    assert HitRateKey("h2", "strong_bull_trend", False, "long") not in table
    assert "h2" not in table  # type: ignore[operator]
