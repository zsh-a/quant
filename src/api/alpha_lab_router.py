"""Alpha Lab API endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, Field

from src.alpha import AlphaService
from src.alpha.tracing import InMemoryCollector, tracer
from src.config.settings import get_alpha_lab_config, get_bitget_config, get_crypto_market_config

router = APIRouter(prefix="/alpha-lab", tags=["alpha-lab"])
service = AlphaService()

_memory_collector = InMemoryCollector()
tracer.add_collector(_memory_collector)

# ---------------------------------------------------------------------------
# In-memory job store for async search tasks
# ---------------------------------------------------------------------------

_SEARCH_JOBS: dict[str, dict[str, Any]] = {}


def _run_search_job(job_id: str, params: dict[str, Any]) -> None:
    """Execute search in background thread — updates _SEARCH_JOBS in-place."""
    try:
        _SEARCH_JOBS[job_id]["status"] = "running"
        strategy = params.pop("strategy", "evolution")
        neural_batch = params.pop("neural_batch", 4096)
        svc = AlphaService(strategy=strategy, neural_sample_batch=neural_batch) if strategy != "evolution" else service
        result = svc.search_formulas_on_db(**params)
        _SEARCH_JOBS[job_id].update(status="completed", result=result)
        logger.info("alpha.search job={} completed", job_id)
    except Exception as exc:
        _SEARCH_JOBS[job_id].update(status="failed", error=str(exc))
        logger.error("alpha.search job={} failed: {}", job_id, exc)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CompileRequest(BaseModel):
    formula: str


class EvaluateDbRequest(BaseModel):
    formula: str
    symbols: list[str] = Field(default_factory=lambda: get_crypto_market_config().default_symbols)
    start_time: datetime
    end_time: datetime
    interval: str = Field(default_factory=lambda: get_crypto_market_config().default_interval)
    min_quote_volume: float = 0.0
    blocked_utc_hours: list[int] = Field(default_factory=list)
    summary_only: bool = True


class SearchDbRequest(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: get_crypto_market_config().default_symbols)
    start_time: datetime
    end_time: datetime
    interval: str = "5m"
    min_quote_volume: float = 0.0
    blocked_utc_hours: list[int] = Field(default_factory=list)
    seeds: list[str] = Field(default_factory=list)
    population_size: int = 8
    offspring_count: int = 4
    top_k: int = 5
    generations: int = 3
    run_name: str | None = None
    persist: bool = True
    novelty_threshold: float = 0.995
    n_splits: int = 5
    purge_window: int = 0
    embargo_window: int = 0
    strategy: str = "evolution"  # "evolution" | "neural" | "full"
    neural_batch: int = 4096


class CombineZooRequest(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: get_crypto_market_config().default_symbols)
    start_time: datetime
    end_time: datetime
    interval: str = "5m"
    min_quote_volume: float = 0.0
    blocked_utc_hours: list[int] = Field(default_factory=list)
    method: str = "ic_weighted"
    max_factors: int = 10
    min_abs_ic: float = 0.01
    max_correlation: float = 0.70
    ic_lookback: int = 60
    zoo_limit: int = 50
    summary_only: bool = True


class SaveZooRequest(BaseModel):
    formula: str
    fitness: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    lineage: dict[str, str] = Field(default_factory=dict)
    note: str | None = None
    tags: list[str] = Field(default_factory=list)
    source: str = "manual"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_blocked_hours(values: list[int] | None) -> list[int] | None:
    if not values:
        return None
    return sorted({hour for hour in values if 0 <= hour <= 23})


def _workspace_defaults() -> dict[str, object]:
    crypto_market = get_crypto_market_config()
    return {
        "alpha_lab": get_alpha_lab_config().model_dump(),
        "bitget": get_bitget_config().model_dump(),
        "crypto_market": crypto_market.model_dump(),
        "intervals": ["5m", "15m", "1h", "4h"],
        "sample_formulas": [
            "cs_rank(ts_mean(close, 5) - close)",
            "cs_rank(ts_std(close, 10))",
            "cs_rank(ts_zscore(funding_rate, 20))",
        ],
        "strategy_modes": ["evolution", "neural", "full"],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/workspace")
async def get_workspace(
    run_limit: int = Query(default=8, ge=1, le=50),
    zoo_limit: int = Query(default=50, ge=1, le=200),
):
    return {
        "operators": service.list_operators(),
        "defaults": _workspace_defaults(),
        "strategy_modes": ["evolution", "neural", "full"],
        "runs": service.list_runs(limit=run_limit),
        "zoo": service.list_zoo(limit=zoo_limit),
        "engine": {
            "backend": service.vm.backend,
            "device": str(service.vm.device) if service.vm.device is not None else "cpu",
            "triton": getattr(service.vm, "use_triton", False),
        },
    }


@router.post("/validate")
async def validate_formula(request: CompileRequest):
    return service.validate_formula(request.formula)


@router.post("/compile")
async def compile_formula(request: CompileRequest):
    try:
        return service.compile_formula(request.formula)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/evaluate-db")
async def evaluate_formula_from_db(request: EvaluateDbRequest):
    try:
        return service.evaluate_formula_from_db(
            formula=request.formula,
            symbols=request.symbols,
            start_time=request.start_time,
            end_time=request.end_time,
            interval=request.interval,
            min_quote_volume=request.min_quote_volume,
            blocked_utc_hours=_normalize_blocked_hours(request.blocked_utc_hours),
            summary_only=request.summary_only,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --- Async search: submit → poll → result ---


@router.post("/search-db")
async def submit_search(request: SearchDbRequest, background_tasks: BackgroundTasks):
    """Submit a GA search job. Returns immediately with a job_id for polling."""
    job_id = str(uuid.uuid4())[:12]
    params = {
        "symbols": request.symbols,
        "start_time": request.start_time,
        "end_time": request.end_time,
        "interval": request.interval,
        "min_quote_volume": request.min_quote_volume,
        "seeds": request.seeds,
        "population_size": request.population_size,
        "offspring_count": request.offspring_count,
        "top_k": request.top_k,
        "generations": request.generations,
        "run_name": request.run_name,
        "persist": request.persist,
        "novelty_threshold": request.novelty_threshold,
        "n_splits": request.n_splits,
        "purge_window": request.purge_window,
        "embargo_window": request.embargo_window,
        "blocked_utc_hours": _normalize_blocked_hours(request.blocked_utc_hours),
        "strategy": request.strategy,
        "neural_batch": request.neural_batch,
    }
    _SEARCH_JOBS[job_id] = {
        "status": "pending",
        "params": {k: str(v) if isinstance(v, datetime) else v for k, v in params.items()},
        "result": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
    }
    background_tasks.add_task(_run_search_job, job_id, params)
    return {"job_id": job_id, "status": "pending"}


@router.get("/search-jobs/{job_id}")
async def get_search_job(job_id: str):
    """Poll search job status."""
    job = _SEARCH_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    response: dict[str, Any] = {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job.get("created_at"),
    }
    if job["status"] == "completed" and job["result"]:
        result = job["result"]
        response["top_results"] = result.get("top_results", [])
        response["search_stats"] = result.get("search_stats", {})
        response["timing"] = result.get("timing", {})
        response["run_id"] = result.get("persistence", {}).get("run_id")
    if job["error"]:
        response["error"] = job["error"]
    return response


@router.get("/search-jobs")
async def list_search_jobs():
    """List all search jobs."""
    jobs = []
    for job_id, job in sorted(_SEARCH_JOBS.items(), key=lambda x: x[1].get("created_at", ""), reverse=True):
        entry: dict[str, Any] = {"job_id": job_id, "status": job["status"], "created_at": job.get("created_at")}
        if job["error"]:
            entry["error"] = job["error"]
        if job["status"] == "completed" and job["result"]:
            entry["run_id"] = job["result"].get("persistence", {}).get("run_id")
            top = job["result"].get("top_results", [])
            entry["top_count"] = len(top)
        jobs.append(entry)
    return {"jobs": jobs}


# --- Runs & Zoo ---


@router.get("/runs")
async def list_runs(limit: int = Query(default=20, ge=1, le=100)):
    return {"runs": service.list_runs(limit=limit)}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    try:
        return service.load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/zoo")
async def list_zoo(limit: int = Query(default=50, ge=1, le=200)):
    return {"entries": service.list_zoo(limit=limit)}


@router.post("/zoo")
async def save_formula_to_zoo(request: SaveZooRequest):
    try:
        return service.save_formula_to_zoo(
            formula=request.formula,
            fitness=request.fitness,
            metrics=request.metrics,
            lineage=request.lineage,
            note=request.note,
            tags=request.tags,
            source=request.source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --- Combination & Benchmark ---


@router.post("/combine-zoo")
async def combine_factors_from_zoo(request: CombineZooRequest):
    try:
        return service.combine_factors_from_db(
            symbols=request.symbols,
            start_time=request.start_time,
            end_time=request.end_time,
            interval=request.interval,
            min_quote_volume=request.min_quote_volume,
            blocked_utc_hours=_normalize_blocked_hours(request.blocked_utc_hours),
            method=request.method,
            max_factors=request.max_factors,
            min_abs_ic=request.min_abs_ic,
            max_correlation=request.max_correlation,
            ic_lookback=request.ic_lookback,
            zoo_limit=request.zoo_limit,
            summary_only=request.summary_only,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --- Tracing ---


@router.get("/tracing/summary")
async def get_tracing_summary():
    return _memory_collector.summary()


@router.get("/tracing/spans")
async def get_tracing_spans(
    kind: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    spans = _memory_collector.find(kind=kind) if kind else _memory_collector.spans
    recent = spans[-limit:]
    return {"spans": [s.to_dict() for s in reversed(recent)], "total": len(spans)}


# --- Neural strategy ---


@router.get("/neural/history")
async def get_neural_history():
    """Get neural strategy training history if available."""
    from pathlib import Path
    import json
    p = Path("data/alpha_lab/neural/training_history.json")
    if not p.exists():
        return {"history": [], "plot_available": False}
    history = json.loads(p.read_text())
    plot_available = Path("data/alpha_lab/neural/training_curves.png").exists()
    return {"history": history, "plot_available": plot_available}


@router.get("/neural/plot")
async def get_neural_plot():
    from pathlib import Path
    p = Path("data/alpha_lab/neural/training_curves.png")
    if not p.exists():
        raise HTTPException(status_code=404, detail="No training plot available")
    return FileResponse(p, media_type="image/png")
