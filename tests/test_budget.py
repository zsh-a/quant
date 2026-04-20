"""BudgetTracker: caps, exhaustion, snapshot, ContextVar scoping."""

from __future__ import annotations

import time

from src.alpha.search.context import BudgetTracker, budget_scope, current_budget


def test_unlimited_budget_never_exhausts():
    b = BudgetTracker()  # all caps 0 = unlimited
    b.record_eval(10_000)
    b.record_llm(tokens=1_000_000, cost_usd=99.9)
    assert not b.exhausted()
    snap = b.snapshot()
    assert snap["exhausted"] is False


def test_eval_cap_triggers_exhausted():
    b = BudgetTracker(max_full_eval=5)
    b.record_eval(3)
    assert not b.exhausted()
    b.record_eval(2)
    assert b.exhausted()
    assert "full_eval" in b.exhausted_reasons


def test_token_cap_triggers_exhausted():
    b = BudgetTracker(max_llm_tokens=100)
    b.record_llm(tokens=60)
    assert not b.exhausted()
    b.record_llm(tokens=45)
    assert b.exhausted()
    assert "llm_tokens" in b.exhausted_reasons


def test_cost_cap_triggers_exhausted():
    b = BudgetTracker(max_cost_usd=0.10)
    b.record_llm(tokens=0, cost_usd=0.04)
    assert not b.exhausted()
    b.record_llm(tokens=0, cost_usd=0.08)
    assert b.exhausted()
    assert "cost_usd" in b.exhausted_reasons


def test_wall_time_cap_triggers_exhausted():
    b = BudgetTracker(max_wall_time_sec=0.05)
    time.sleep(0.06)
    assert b.exhausted()
    assert "wall_time" in b.exhausted_reasons


def test_snapshot_keys_present():
    b = BudgetTracker(max_full_eval=10, max_wall_time_sec=60)
    b.record_eval(3)
    snap = b.snapshot()
    assert snap["used_full_eval"] == 3
    assert snap["max_wall_time_sec"] == 60
    assert snap["remaining_wall_sec"] is not None


def test_budget_scope_binds_and_unbinds():
    b = BudgetTracker(max_full_eval=10)
    assert current_budget() is None
    with budget_scope(b):
        assert current_budget() is b
    assert current_budget() is None


def test_nested_scopes_restore():
    outer = BudgetTracker(max_full_eval=10)
    inner = BudgetTracker(max_full_eval=5)
    with budget_scope(outer):
        assert current_budget() is outer
        with budget_scope(inner):
            assert current_budget() is inner
        assert current_budget() is outer
    assert current_budget() is None
