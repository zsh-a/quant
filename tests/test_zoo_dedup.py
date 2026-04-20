"""Zoo canonical + signature dedup semantics."""

from __future__ import annotations

from pathlib import Path

from src.alpha.infra.persistence import (
    AlphaPersistence,
    canonical_hash,
    full_expr_hash,
    signature_hash,
)


def test_canonical_hash_is_commutative_stable():
    assert canonical_hash("a + b") == canonical_hash("b + a")
    assert canonical_hash("a * b") == canonical_hash("b * a")
    # Non-commutative: argument order of a call matters
    assert canonical_hash("ts_corr(close, volume, 5)") != canonical_hash("ts_corr(volume, close, 5)")


def test_full_expr_hash_is_64_hex():
    h = full_expr_hash("a + b")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_upsert_merges_same_canonical_as_aliases(tmp_path: Path):
    p = AlphaPersistence(root_dir=str(tmp_path))
    e1 = p.upsert_zoo_entry({"formula": "a + b", "fitness": 1.0, "expr_hash": "x" * 64})
    e2 = p.upsert_zoo_entry({"formula": "b + a", "fitness": 1.5})
    assert e1["canonical_hash"] == e2["canonical_hash"]
    # Same canonical → same file on disk
    assert e1["path"] == e2["path"]
    # The earlier expr_hash shows up in aliases of the merged record
    assert "x" * 64 in e2["aliases"]
    # Fitness takes the max, not the last write
    assert e2["fitness"] == 1.5


def test_upsert_signature_near_dup_writes_similar_to_edge(tmp_path, monkeypatch):
    # Use an isolated SessionDB so lineage writes don't pollute dev data.
    from session_db import SessionDB

    db = SessionDB(db_path=str(tmp_path / "test.db"))
    from src.alpha.infra import persistence as mod

    original = mod.AlphaPersistence._write_lineage_edge

    def _local_edge(self, **kw):
        db.add_lineage_edge(**kw)

    monkeypatch.setattr(mod.AlphaPersistence, "_write_lineage_edge", _local_edge)
    try:
        p = AlphaPersistence(root_dir=str(tmp_path / "zoo"))
        sig = [0.1, 0.3, 0.2, 0.5, 0.4]
        a = p.upsert_zoo_entry({"formula": "cs_rank(close)", "fitness": 1.0, "alpha_signature": sig})
        b = p.upsert_zoo_entry({"formula": "cs_rank(open)", "fitness": 0.9, "alpha_signature": sig})
        # Different canonicals
        assert a["canonical_hash"] != b["canonical_hash"]
        # A similar_to edge must exist pointing from a → b
        parents = db.list_lineage_parents("zoo_factor", b["canonical_hash"])
        relations = [p_["relation"] for p_ in parents]
        assert "similar_to" in relations
    finally:
        monkeypatch.setattr(mod.AlphaPersistence, "_write_lineage_edge", original)


def test_live_metrics_filters_none_values(tmp_path: Path):
    p = AlphaPersistence(root_dir=str(tmp_path))
    entry = p.upsert_zoo_entry({"formula": "cs_rank(close)", "fitness": 1.0})
    merged = p.append_live_metrics(
        entry["canonical_hash"],
        {"sharpe": None, "max_drawdown": -0.1, "total_return": None},
    )
    assert merged is not None
    last = merged["live_metrics"]
    assert "sharpe" not in last
    assert "total_return" not in last
    assert last["max_drawdown"] == -0.1


def test_signature_hash_stable_for_identical_arrays():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [1.0, 2.0, 3.0, 4.0]
    assert signature_hash(a) == signature_hash(b)
    # Zero-std arrays return the sentinel rather than an opaque hash
    assert signature_hash([1.0, 1.0, 1.0]) == "zero_std"
