"""Cancel semantics: orchestrator exits cooperatively, pipeline marked aborted."""

from __future__ import annotations

import pytest

from src.alpha import AlphaService


@pytest.fixture
def minimal_service():
    # Keep the build cheap — we only need the compiler & search_engine wiring.
    return AlphaService(market="crypto", strategy="evolution")


def test_pipeline_record_has_aborted_field():
    from src.alpha.search.pipeline import PipelineRecord

    pr = PipelineRecord(job_id="x")
    assert pr.aborted is False
    assert pr.budget_exhausted is False
    d = pr.to_dict()
    assert "aborted" in d and "budget_exhausted" in d


def test_should_abort_returns_true_exits_before_rounds(minimal_service: AlphaService):
    """When should_abort() is truthy from the outset, the search exits cleanly.

    We don't need a real dataset: calling the orchestrator directly with a
    stubbed eval_fn is enough to prove the abort path is wired.
    """
    from src.alpha.search.evolution import EvalResult

    def fake_eval(_inds):
        return EvalResult(metrics_by_hash={}, signatures_by_hash={}, details_by_hash={}, timing={})

    flag = {"tripped": True}
    result = minimal_service.search_engine.run(
        seeds=["cs_rank(close)"],
        rounds=3,
        batch_size=2,
        top_k=3,
        novelty_threshold=0.995,
        evaluate_fn=fake_eval,
        quick_evaluate_fn=None,
        should_abort=lambda: flag["tripped"],
    )
    assert result.pipeline is not None
    assert result.pipeline.aborted is True
    # 0 full rounds should have executed — init might have run, which is ok.
    assert len(result.pipeline.rounds) == 0


def test_api_cancel_endpoint_toggles_status():
    """POST /cancel flips an in-memory job to the ``cancelling`` state."""
    import asyncio

    from src.api.alpha_lab_router import _SEARCH_CANCEL, _SEARCH_JOBS, cancel_search_job

    _SEARCH_JOBS["cancel_test"] = {
        "status": "running",
        "params": {},
        "result": None,
        "error": None,
        "created_at": "2026-01-01T00:00:00",
    }
    try:
        resp = asyncio.run(cancel_search_job("cancel_test"))
        assert resp["status"] == "cancelling"
        assert _SEARCH_CANCEL["cancel_test"].is_set()
        # Second call is idempotent-ish (job still cancelling)
        resp2 = asyncio.run(cancel_search_job("cancel_test"))
        assert resp2["status"] in {"cancelling", "already_settled"} or resp2.get("already_settled")
    finally:
        _SEARCH_JOBS.pop("cancel_test", None)
        _SEARCH_CANCEL.pop("cancel_test", None)


def test_cancel_missing_job_raises_404():
    import asyncio

    from fastapi import HTTPException

    from src.api.alpha_lab_router import cancel_search_job

    with pytest.raises(HTTPException) as exc:
        asyncio.run(cancel_search_job("nonexistent"))
    assert exc.value.status_code == 404
