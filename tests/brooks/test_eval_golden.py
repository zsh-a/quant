"""Tests for :mod:`src.brooks.eval.golden`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from src.brooks.context import Bar
from src.brooks.eval.golden import GoldenDataset, GoldenSample


def _bars(n: int, base: float = 100.0) -> List[Bar]:
    out = []
    for i in range(n):
        c = base + i
        out.append(
            Bar(
                timestamp_ns=1_700_000_000_000_000_000 + i * 60_000_000_000,
                open=c - 0.5,
                high=c + 0.5,
                low=c - 0.6,
                close=c,
                volume=1.0,
            )
        )
    return out


def _sample(idx: int = 5, **overrides) -> GoldenSample:
    base = dict(
        id=f"sample-{idx}",
        symbol="BTCUSDT",
        interval="5m",
        bars=_bars(10),
        target_bar_idx=idx,
        expected_pattern="h2",
        expected_side="long",
        expected_entry=110.0,
        expected_stop=109.0,
        expected_target=112.0,
        regime="weak_bull_trend",
        htf_aligned=True,
        source="human",
        reasoning="leg-1 / leg-2 setup",
    )
    base.update(overrides)
    return GoldenSample(**base)


# ---------------------------------------------------------------------------
# GoldenSample
# ---------------------------------------------------------------------------


def test_sample_round_trip_dict():
    s = _sample()
    raw = s.to_dict()
    rebuilt = GoldenSample.from_dict(raw)
    assert rebuilt.id == s.id
    assert rebuilt.bars[0].close == s.bars[0].close
    assert rebuilt.expected_target == s.expected_target


def test_sample_validates_target_idx():
    with pytest.raises(ValueError):
        _sample(target_bar_idx=99)


def test_sample_validates_entry_neq_stop():
    with pytest.raises(ValueError):
        _sample(expected_entry=100.0, expected_stop=100.0)


def test_sample_validates_side():
    with pytest.raises(ValueError):
        _sample(expected_side="upward")  # type: ignore[arg-type]


def test_sample_context_and_future_split():
    s = _sample(idx=4)
    assert len(s.context_bars) == 5
    assert len(s.future_bars) == 5
    assert s.context_bars[-1] == s.bars[4]
    assert s.future_bars[0] == s.bars[5]


def test_sample_one_r():
    s = _sample(expected_entry=110.0, expected_stop=109.5)
    assert s.one_r == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# GoldenDataset persistence
# ---------------------------------------------------------------------------


def test_dataset_roundtrip_jsonl(tmp_path: Path):
    ds = GoldenDataset.from_samples(
        [_sample(idx=4), _sample(idx=5, expected_side="short", expected_entry=99.0, expected_stop=100.0)]
    )
    path = tmp_path / "ds.jsonl"
    ds.save(path)
    reloaded = GoldenDataset.load(path)
    assert len(reloaded) == 2
    assert reloaded[1].expected_side == "short"
    assert reloaded[0].bars[-1].timestamp_ns == ds[0].bars[-1].timestamp_ns


def test_dataset_roundtrip_parquet(tmp_path: Path):
    pytest.importorskip("pyarrow")
    ds = GoldenDataset.from_samples([_sample(idx=4), _sample(idx=5)])
    path = tmp_path / "ds.parquet"
    ds.save(path)
    reloaded = GoldenDataset.load(path)
    assert len(reloaded) == 2
    assert reloaded[0].expected_pattern == "h2"
    assert reloaded[0].bars[3].open == ds[0].bars[3].open


def test_dataset_roundtrip_directory(tmp_path: Path):
    GoldenDataset.from_samples([_sample(idx=4)]).save(tmp_path / "a.jsonl")
    GoldenDataset.from_samples(
        [_sample(idx=5, expected_pattern="l2", expected_side="short", expected_entry=99.0, expected_stop=100.0)]
    ).save(tmp_path / "b.jsonl")
    loaded = GoldenDataset.load(tmp_path)
    assert len(loaded) == 2
    patterns = {s.expected_pattern for s in loaded}
    assert patterns == {"h2", "l2"}


def test_dataset_save_unsupported_suffix(tmp_path: Path):
    ds = GoldenDataset.from_samples([_sample()])
    with pytest.raises(ValueError):
        ds.save(tmp_path / "ds.csv")


def test_dataset_load_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        GoldenDataset.load(tmp_path / "missing.jsonl")


def test_dataset_load_empty_directory(tmp_path: Path):
    assert len(GoldenDataset.load(tmp_path)) == 0


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def test_filter_by_value():
    ds = GoldenDataset.from_samples(
        [
            _sample(idx=4, expected_pattern="h2"),
            _sample(idx=5, expected_pattern="l2", expected_side="short", expected_entry=99.0, expected_stop=100.0),
            _sample(idx=6, expected_pattern="h2"),
        ]
    )
    h2 = ds.filter(expected_pattern="h2")
    assert len(h2) == 2
    assert all(s.expected_pattern == "h2" for s in h2)


def test_filter_by_membership():
    ds = GoldenDataset.from_samples(
        [
            _sample(idx=4, regime="weak_bull_trend"),
            _sample(idx=5, regime="strong_bull_trend"),
            _sample(idx=6, regime="tight_trading_range"),
        ]
    )
    bulls = ds.filter(regime={"weak_bull_trend", "strong_bull_trend"})
    assert len(bulls) == 2


def test_filter_unknown_attr_raises():
    ds = GoldenDataset.from_samples([_sample()])
    with pytest.raises(AttributeError):
        ds.filter(does_not_exist="x")


def test_iteration_and_indexing():
    samples = [_sample(idx=i + 1) for i in range(3)]
    ds = GoldenDataset.from_samples(samples)
    assert list(ds) == samples
    assert ds[1] == samples[1]
    assert len(ds) == 3


# ---------------------------------------------------------------------------
# Manual JSON ingestion
# ---------------------------------------------------------------------------


def test_load_json_array_file(tmp_path: Path):
    path = tmp_path / "ds.json"
    payload = [_sample(idx=2).to_dict(), _sample(idx=3).to_dict()]
    path.write_text(json.dumps(payload))
    loaded = GoldenDataset.load(path)
    assert len(loaded) == 2


def test_load_json_single_object(tmp_path: Path):
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(_sample(idx=2).to_dict()))
    loaded = GoldenDataset.load(path)
    assert len(loaded) == 1
