"""Cross-cycle persistent search state round-trip + merge semantics."""

from __future__ import annotations

from pathlib import Path

from src.alpha.search.persistent_context import (
    PERSISTENT_STATE_SCHEMA_VERSION,
    ArchiveSummary,
    PersistentSearchState,
)


def test_schema_version_round_trip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.config.paths.ALPHA_STATE_DIR", tmp_path)
    state = PersistentSearchState(market="crypto")
    state.merge_cycle_result(
        new_hashes={"aaa", "bbb"},
        new_archive=[
            {
                "formula": "f1",
                "expr_hash": "aaa",
                "fitness": 1.1,
                "metrics": {"sharpe": 1.0},
                "strategy": "seed",
                "round_idx": 0,
            },
            {"formula": "f2", "expr_hash": "bbb", "fitness": 0.5, "strategy": "llm", "round_idx": 1},
        ],
        window={"start": "t0", "end": "t1"},
        run_id="r1",
        feedback="keep pushing rank_ic",
    )

    # Serialise → deserialise via the module helpers so we exercise _state_path.
    from src.alpha.search.persistent_context import (
        load_persistent_state,
        save_persistent_state,
    )

    save_persistent_state(state)
    reloaded = load_persistent_state("crypto")

    assert reloaded.schema_version == PERSISTENT_STATE_SCHEMA_VERSION
    assert reloaded.seen_set() == {"aaa", "bbb"}
    assert reloaded.total_cycles == 1
    assert reloaded.last_cycle_run_id == "r1"
    assert reloaded.last_cycle_feedback == "keep pushing rank_ic"
    assert reloaded.top_archive and reloaded.top_archive[0].fitness >= 1.0


def test_merge_keeps_best_fitness_per_hash(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.config.paths.ALPHA_STATE_DIR", tmp_path)
    state = PersistentSearchState(market="test")
    state.merge_cycle_result(
        new_hashes={"h1"},
        new_archive=[{"formula": "f", "expr_hash": "h1", "fitness": 0.5}],
    )
    state.merge_cycle_result(
        new_hashes={"h1"},
        new_archive=[{"formula": "f", "expr_hash": "h1", "fitness": 2.0}],
    )
    assert len(state.top_archive) == 1
    assert state.top_archive[0].fitness == 2.0


def test_hash_cap_enforced(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.config.paths.ALPHA_STATE_DIR", tmp_path)
    state = PersistentSearchState(market="test")
    for i in range(200):
        state.merge_cycle_result(
            new_hashes={f"h{i}"},
            new_archive=[],
            hash_cap=50,
        )
    assert len(state.seen_hashes) == 50
    # Newest N must survive
    assert "h199" in state.seen_hashes
    assert "h0" not in state.seen_hashes


def test_malformed_state_file_returns_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.config.paths.ALPHA_STATE_DIR", tmp_path)
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "state.json").write_text("{ not json")
    from src.alpha.search.persistent_context import load_persistent_state

    state = load_persistent_state("broken")
    assert state.seen_set() == set()
    assert state.total_cycles == 0


def test_archive_summary_from_dict_tolerates_missing_fields():
    # Orchestrator sometimes passes partial dicts; parser must not raise.
    s = ArchiveSummary.from_dict({"formula": "f", "expr_hash": "h"})
    assert s.fitness == 0.0
    assert s.metrics == {}
