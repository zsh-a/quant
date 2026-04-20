"""End-to-end lineage DAG: search_job → zoo_factor → simulation_job → run."""

from __future__ import annotations

import pytest

from session_db import SessionDB


def test_add_and_query_edges(tmp_path):
    db = SessionDB(db_path=str(tmp_path / "t.db"))
    assert db.add_lineage_edge("search_job", "j", "zoo_factor", "z", "produced")
    # Duplicate insert is a no-op (UNIQUE constraint)
    assert not db.add_lineage_edge("search_job", "j", "zoo_factor", "z", "produced")

    parents = db.list_lineage_parents("zoo_factor", "z")
    assert len(parents) == 1
    assert parents[0]["parent_id"] == "j"
    assert parents[0]["relation"] == "produced"


def test_unknown_kind_raises(tmp_path):
    db = SessionDB(db_path=str(tmp_path / "t.db"))
    with pytest.raises(ValueError):
        db.add_lineage_edge("bogus_kind", "x", "zoo_factor", "z", "produced")
    with pytest.raises(ValueError):
        db.add_lineage_edge("search_job", "x", "zoo_factor", "z", "bogus_relation")


def test_graph_traversal_multi_hop(tmp_path):
    db = SessionDB(db_path=str(tmp_path / "t.db"))
    db.add_lineage_edge("search_job", "j1", "zoo_factor", "z1", "produced")
    db.add_lineage_edge("zoo_factor", "z1", "simulation_job", "s1", "promoted_to")
    db.add_lineage_edge("simulation_run", "r1", "zoo_factor", "z1", "backtests", meta={"sharpe": 1.5})
    db.add_lineage_edge("zoo_factor", "z0", "zoo_factor", "z1", "derived_from")

    graph = db.get_lineage_graph("zoo_factor", "z1", max_depth=4)
    node_ids = {(n["kind"], n["id"]) for n in graph["nodes"]}
    assert ("search_job", "j1") in node_ids
    assert ("simulation_job", "s1") in node_ids
    assert ("simulation_run", "r1") in node_ids
    assert ("zoo_factor", "z0") in node_ids

    relations = {e["relation"] for e in graph["edges"]}
    assert {"produced", "promoted_to", "backtests", "derived_from"} <= relations


def test_edge_meta_roundtrip(tmp_path):
    db = SessionDB(db_path=str(tmp_path / "t.db"))
    db.add_lineage_edge(
        "simulation_run",
        "r",
        "zoo_factor",
        "z",
        "backtests",
        meta={"sharpe": 1.2, "win_rate": 0.6},
    )
    edges = db.list_lineage_parents("zoo_factor", "z")
    assert edges[0]["meta"]["sharpe"] == 1.2
    assert edges[0]["meta"]["win_rate"] == 0.6
