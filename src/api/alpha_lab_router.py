"""Alpha Lab API endpoints."""

from __future__ import annotations

import asyncio
import json as _json
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from src.alpha import AlphaService
from src.alpha.core.market import list_market_types
from src.alpha.search.pipeline import RoundRecord, StageRecord
from src.alpha.infra.tracing import InMemoryCollector, tracer
from src.config.settings import get_alpha_lab_config, get_bitget_config, get_crypto_market_config

router = APIRouter(prefix="/alpha-lab", tags=["alpha-lab"])

# Lazy per-market service instances
_services: dict[str, AlphaService] = {}


def _get_service(market: str = "crypto") -> AlphaService:
    if market not in _services:
        _services[market] = AlphaService(market=market)
    return _services[market]

_memory_collector = InMemoryCollector()
tracer.add_collector(_memory_collector)

# ---------------------------------------------------------------------------
# In-memory job store for async search tasks
# ---------------------------------------------------------------------------

_SEARCH_JOBS: dict[str, dict[str, Any]] = {}
_SEARCH_EVENTS: dict[str, asyncio.Queue[dict[str, Any]]] = {}


def _run_search_job(
    job_id: str,
    params: dict[str, Any],
    loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    """Execute search in background thread — updates _SEARCH_JOBS in-place."""
    queue = _SEARCH_EVENTS.get(job_id)

    def _push(event: dict[str, Any]) -> None:
        if queue and loop:
            loop.call_soon_threadsafe(queue.put_nowait, event)

    def _on_stage(stage: StageRecord) -> None:
        _push({"type": "stage", "data": stage.to_dict()})

    def _on_round(round_rec: RoundRecord) -> None:
        _push({"type": "round", "data": round_rec.to_dict()})

    try:
        _SEARCH_JOBS[job_id]["status"] = "running"
        market = params.pop("market", "crypto")
        strategy = params.pop("strategy", "")
        neural_batch = params.pop("neural_batch", 4096)
        enum_max = params.pop("enum_max", 500)
        enum_top_k = params.pop("enum_top_k", 30)
        logger.info("alpha.search creating service strategy={!r} market={!r}", strategy, market)
        svc = AlphaService(
            market=market, strategy=strategy, neural_sample_batch=neural_batch,
            enum_max=enum_max, enum_top_k=enum_top_k,
        )
        logger.info("alpha.search strategies={}", [s.name for s in svc.search_engine.strategies])
        result = svc.search_formulas_on_db(
            **params,
            job_id=job_id,
            on_stage_complete=_on_stage,
            on_round_complete=_on_round,
        )
        _SEARCH_JOBS[job_id].update(status="completed", result=result)
        _push({"type": "complete", "data": {"status": "completed"}})
        logger.info("alpha.search job={} completed", job_id)
    except Exception as exc:
        _SEARCH_JOBS[job_id].update(status="failed", error=str(exc))
        _push({"type": "complete", "data": {"status": "failed", "error": str(exc)}})
        logger.error("alpha.search job={} failed: {}", job_id, exc)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CompileRequest(BaseModel):
    formula: str


class EvaluateDbRequest(BaseModel):
    market: str = "crypto"
    formula: str
    symbols: list[str] = Field(default_factory=lambda: get_crypto_market_config().default_symbols)
    universe: str | None = None  # A-share: 指数代码，如 "000300" (沪深300)
    start_time: datetime
    end_time: datetime
    interval: str = Field(default_factory=lambda: get_crypto_market_config().default_interval)
    min_quote_volume: float = 0.0
    blocked_utc_hours: list[int] = Field(default_factory=list)
    exclude_st: bool = False
    summary_only: bool = True


class SearchDbRequest(BaseModel):
    market: str = "crypto"
    symbols: list[str] = Field(default_factory=lambda: get_crypto_market_config().default_symbols)
    universe: str | None = None  # A-share: 指数代码，如 "000300" (沪深300)
    start_time: datetime
    end_time: datetime
    interval: str = "5m"
    min_quote_volume: float = 0.0
    blocked_utc_hours: list[int] = Field(default_factory=list)
    exclude_st: bool = False
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
    strategy: str = ""  # extra strategies: "mcts", "neural", "mcts,neural"
    neural_batch: int = 4096
    enum_max: int = 500       # max formulas to enumerate (round 0)
    enum_top_k: int = 30      # top-K from enumeration to keep


class CombineZooRequest(BaseModel):
    market: str = "crypto"
    symbols: list[str] = Field(default_factory=lambda: get_crypto_market_config().default_symbols)
    universe: str | None = None  # A-share: 指数代码
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
    from src.market_data.binance_vision import EXPANDED_SYMBOLS
    from src.alpha.core.market import get_market_profile

    crypto_market = get_crypto_market_config()
    crypto_profile = get_market_profile("crypto")
    a_share_profile = get_market_profile("a_share")
    return {
        "alpha_lab": get_alpha_lab_config().model_dump(),
        "bitget": get_bitget_config().model_dump(),
        "crypto_market": crypto_market.model_dump(),
        "available_markets": list_market_types(),
        "market_presets": {
            "crypto": {
                "intervals": list(crypto_profile.supported_intervals),
                "sample_formulas": list(crypto_profile.seeds[:3]),
                "symbol_presets": list(crypto_profile.symbol_presets) if crypto_profile.symbol_presets else [],
                "eval_methods": list(crypto_profile.eval_methods),
                "default_eval_method": crypto_profile.default_eval_method,
            },
            "a_share": {
                "intervals": list(a_share_profile.supported_intervals),
                "sample_formulas": list(a_share_profile.seeds[:3]),
                "symbol_presets": list(a_share_profile.symbol_presets),
                "eval_methods": list(a_share_profile.eval_methods),
                "default_eval_method": a_share_profile.default_eval_method,
            },
        },
        "intervals": ["5m", "15m", "1h", "4h"],
        "sample_formulas": [
            "cs_rank(ts_mean(close, 5) - close)",
            "cs_rank(ts_std(close, 10))",
            "cs_rank(ts_zscore(funding_rate, 20))",
        ],
        "extra_strategies": ["mcts", "neural"],
        "symbol_presets": [
            {
                "key": "top5",
                "label": "Top 5 (BTC/ETH/SOL...)",
                "brief": "流动性最强的 5 个，适合单因子验证",
                "symbols": EXPANDED_SYMBOLS[:5],
            },
            {
                "key": "mega_cap",
                "label": "Mega Cap (Top 10)",
                "brief": "前 10 大市值，定价有效，alpha 薄",
                "symbols": EXPANDED_SYMBOLS[:10],
            },
            {
                "key": "mid_cap",
                "label": "Mid Cap (11-60)",
                "brief": "小资金黄金区间，散户多，错误定价多",
                "symbols": EXPANDED_SYMBOLS[10:60],
            },
            {
                "key": "small_fund",
                "label": "小资金推荐 (30-80)",
                "brief": "跳过 mega-cap，专注中小盘高波动",
                "symbols": EXPANDED_SYMBOLS[10:],
            },
            {
                "key": "expanded",
                "label": "全市场 (80 symbols)",
                "brief": "80 个流动永续合约，截面宽度最大",
                "symbols": EXPANDED_SYMBOLS,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/workspace")
async def get_workspace(
    run_limit: int = Query(default=8, ge=1, le=50),
    zoo_limit: int = Query(default=50, ge=1, le=200),
):
    svc = _get_service()
    strategies_info = svc.get_strategy_modes_info()
    return {
        "operators": svc.list_operators(),
        "defaults": _workspace_defaults(),
        "strategy_modes_info": strategies_info,
        "runs": svc.list_runs(limit=run_limit),
        "zoo": svc.list_zoo(limit=zoo_limit),
        "engine": {
            "backend": svc.vm.backend,
            "device": str(svc.vm.device) if svc.vm.device is not None else "cpu",
            "triton": getattr(svc.vm, "use_triton", False),
        },
        "available_markets": list_market_types(),
    }


@router.post("/validate")
async def validate_formula(request: CompileRequest):
    return _get_service().validate_formula(request.formula)


@router.post("/compile")
async def compile_formula(request: CompileRequest):
    try:
        return _get_service().compile_formula(request.formula)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/evaluate-db")
async def evaluate_formula_from_db(request: EvaluateDbRequest):
    try:
        return _get_service(request.market).evaluate_formula_from_db(
            formula=request.formula,
            symbols=request.symbols,
            start_time=request.start_time,
            end_time=request.end_time,
            interval=request.interval,
            min_quote_volume=request.min_quote_volume,
            blocked_utc_hours=_normalize_blocked_hours(request.blocked_utc_hours),
            summary_only=request.summary_only,
            exclude_st=request.exclude_st,
            universe=request.universe,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --- Async search: submit → poll → result ---


@router.post("/search-db")
async def submit_search(request: SearchDbRequest, background_tasks: BackgroundTasks):
    """Submit a GA search job. Returns immediately with a job_id for polling."""
    job_id = str(uuid.uuid4())[:12]
    params = {
        "market": request.market,
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
        "exclude_st": request.exclude_st,
        "universe": request.universe,
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
    loop = asyncio.get_running_loop()
    _SEARCH_EVENTS[job_id] = asyncio.Queue()
    background_tasks.add_task(_run_search_job, job_id, params, loop)
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
        response["pipeline"] = result.get("pipeline")
        response["dataset"] = result.get("dataset")
        # Per-factor split metrics (train/valid/test) for top results
        evaluations = result.get("evaluations", {})
        for tr in response.get("top_results", []):
            h = tr.get("expr_hash")
            if h and h in evaluations:
                tr["split_metrics"] = evaluations[h].get("split_metrics", {})
    if job["error"]:
        response["error"] = job["error"]
    return response


@router.get("/search-jobs")
async def list_search_jobs():
    """List all search jobs (newest first)."""
    jobs = []
    for job_id, job in sorted(_SEARCH_JOBS.items(), key=lambda x: x[1].get("created_at", ""), reverse=True):
        params = job.get("params", {})
        entry: dict[str, Any] = {
            "job_id": job_id,
            "status": job["status"],
            "created_at": job.get("created_at"),
            "strategy": params.get("strategy", "evolution"),
        }
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
    return {"runs": _get_service().list_runs(limit=limit)}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    try:
        return _get_service().load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/zoo")
async def list_zoo(limit: int = Query(default=50, ge=1, le=200)):
    return {"entries": _get_service().list_zoo(limit=limit)}


@router.post("/zoo")
async def save_formula_to_zoo(request: SaveZooRequest):
    try:
        return _get_service().save_formula_to_zoo(
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
        return _get_service(request.market).combine_factors_from_db(
            symbols=request.symbols,
            start_time=request.start_time,
            end_time=request.end_time,
            interval=request.interval,
            min_quote_volume=request.min_quote_volume,
            blocked_utc_hours=_normalize_blocked_hours(request.blocked_utc_hours),
            universe=request.universe,
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


# --- SSE: real-time search progress ---


@router.get("/search-jobs/{job_id}/events")
async def stream_search_events(job_id: str):
    """SSE stream of pipeline stage/round events for a running search job."""
    job = _SEARCH_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    queue = _SEARCH_EVENTS.get(job_id)
    if not queue:
        raise HTTPException(status_code=404, detail="No event stream for this job")

    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield f"data: {_json.dumps(event)}\n\n"
                if event.get("type") == "complete":
                    break
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/search-jobs/{job_id}/pipeline")
async def get_search_pipeline(job_id: str):
    """Get the full pipeline record for a completed search job."""
    job = _SEARCH_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed" or not job.get("result"):
        raise HTTPException(status_code=400, detail="Job not completed yet")
    pipeline = job["result"].get("pipeline")
    if not pipeline:
        return {"pipeline": None}
    return {"pipeline": pipeline}


class AnalyzeSearchRequest(BaseModel):
    instruction: str = "Analyze the search results and suggest improvements"


@router.post("/search-jobs/{job_id}/analyze")
async def analyze_search(job_id: str, request: AnalyzeSearchRequest):
    """Send pipeline state to LLM for analysis."""
    from src.alpha.llm.context import build_analysis_prompt, build_pipeline_summary
    from src.alpha.search.pipeline import ArchiveEntry

    job = _SEARCH_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed" or not job.get("result"):
        raise HTTPException(status_code=400, detail="Job not completed yet")

    result = job["result"]
    pipeline_data = result.get("pipeline")
    if not pipeline_data:
        raise HTTPException(status_code=400, detail="No pipeline data available")

    # Build archive entries from top_results
    archive_entries: list[ArchiveEntry] = []
    for tr in result.get("top_results", []):
        metrics = tr.get("metrics", {})
        lineage = tr.get("lineage", {})
        archive_entries.append(ArchiveEntry(
            formula=tr.get("formula", ""),
            expr_hash=tr.get("expr_hash", ""),
            fitness=tr.get("fitness", 0),
            rank_ic=float(metrics.get("rank_ic", 0) or 0),
            sharpe=float(metrics.get("sharpe", 0) or 0),
            turnover=float(metrics.get("avg_turnover", 0) or 0),
            origin=lineage.get("origin", "unknown") if isinstance(lineage, dict) else "unknown",
        ))

    # Build pipeline record from serialized data
    from src.alpha.search.pipeline import PipelineRecord, RoundRecord as RR, StageRecord as SR, StageKind
    pr = PipelineRecord(
        job_id=pipeline_data.get("job_id", job_id),
        total_evaluations=pipeline_data.get("total_evaluations", 0),
        total_rejected=pipeline_data.get("total_rejected", 0),
    )
    for rd in pipeline_data.get("rounds", []):
        rr = RR(
            round_idx=rd.get("round", 0),
            strategies_activated=rd.get("strategies", []),
            archive_size=rd.get("archive_size", 0),
            population_size=rd.get("population_size", 0),
            best_fitness=rd.get("best_fitness", 0),
            duration_ms=rd.get("duration_ms", 0),
        )
        for sd in rd.get("stages", []):
            try:
                kind = StageKind(sd.get("kind", "generate"))
            except ValueError:
                kind = StageKind.GENERATE
            rr.stages.append(SR(
                kind=kind,
                strategy=sd.get("strategy", ""),
                round_idx=sd.get("round", 0),
                input_count=sd.get("input", 0),
                output_count=sd.get("output", 0),
                duration_ms=sd.get("duration_ms", 0),
                best_fitness=sd.get("best_fitness"),
            ))
        pr.rounds.append(rr)

    summary = build_pipeline_summary(
        pipeline=pr,
        archive=archive_entries,
        strategy_memory=getattr(_get_service(), "strategy_memory", None),
    )
    prompt = build_analysis_prompt(summary, request.instruction)

    # Call LLM for analysis (use the same backend as formula generation)
    analysis = ""
    try:
        llm = getattr(_get_service(), "_llm_backend", None)
        if llm and hasattr(llm, "_call_llm"):
            analysis = llm._call_llm(prompt, temperature=0.3, tag="analysis")
        else:
            analysis = "(LLM backend not available — returning summary only)"
    except Exception as e:
        analysis = f"(LLM analysis failed: {e})"

    return {
        "summary": summary,
        "analysis": analysis,
        "prompt_length": len(prompt),
    }


# --- Tracing ---


@router.get("/tracing/summary")
async def get_tracing_summary(trace_id: str | None = None):
    return _memory_collector.summary(trace_id=trace_id)


@router.get("/tracing/spans")
async def get_tracing_spans(
    kind: str | None = None,
    trace_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    spans = _memory_collector.find(kind=kind, trace_id=trace_id)
    recent = spans[-limit:]
    return {
        "spans": [s.to_dict() for s in reversed(recent)],
        "total": len(spans),
        "trace_id": trace_id or _memory_collector.latest_trace_id,
        "trace_ids": _memory_collector.trace_ids,
    }


@router.get("/tracing/traces")
async def get_tracing_traces():
    """List all available traces with summary info."""
    traces = []
    for tid in _memory_collector.trace_ids:
        spans = _memory_collector.find(trace_id=tid)
        root = next((s for s in spans if s.parent_id is None), None)
        traces.append({
            "trace_id": tid,
            "operation": root.operation if root else "unknown",
            "start_time": root.start_time if root else None,
            "duration_ms": root.duration_ms if root else 0,
            "span_count": len(spans),
            "status": root.status if root else "unknown",
        })
    return {"traces": list(reversed(traces))}


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


# --- Strategy State Management ---


@router.get("/strategy-state")
async def get_strategy_state():
    """Get current state of all registered strategies."""
    from src.alpha.search.context import StatefulStrategy

    strategies_info: list[dict[str, Any]] = []
    for strategy in _get_service().search_engine.strategies:
        info: dict[str, Any] = {
            "name": strategy.name,
            "stateful": isinstance(strategy, StatefulStrategy),
        }
        if hasattr(strategy, "get_stats"):
            stats = strategy.get_stats()
            # Exclude large fields like full history
            info["stats"] = {
                k: v for k, v in stats.items()
                if k != "history" and not isinstance(v, (list, bytes))
            }
        strategies_info.append(info)

    # Strategy memory summary
    memory_summary = None
    svc = _get_service()
    if svc.strategy_memory:
        memory_summary = {
            "themes": svc.strategy_memory.get_theme_summary(),
            "operators": svc.strategy_memory.get_operator_summary(),
        }

    return {
        "strategies": strategies_info,
        "strategy_memory": memory_summary,
    }


@router.get("/checkpoints")
async def list_checkpoints():
    """List all available checkpoints across jobs."""
    from pathlib import Path

    ckpt_dir = Path(_get_service().checkpoint_manager._dir)
    if not ckpt_dir.exists():
        return {"checkpoints": []}

    checkpoints: list[dict[str, Any]] = []
    for job_dir in sorted(ckpt_dir.iterdir(), reverse=True):
        if not job_dir.is_dir():
            continue
        for ckpt in sorted(job_dir.glob("checkpoint_r*"), reverse=True):
            manifest_path = ckpt / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = _json.loads(manifest_path.read_text())
                strategies_saved = manifest.get("strategy_names", [])
                checkpoints.append({
                    "job_id": manifest.get("job_id", job_dir.name),
                    "round_idx": manifest.get("round_idx", 0),
                    "timestamp": manifest.get("timestamp", 0),
                    "path": str(ckpt),
                    "strategies": strategies_saved,
                    "archive_count": len(manifest.get("archive_formulas", [])),
                    "context_state": manifest.get("context_state", {}),
                })
            except Exception:
                continue

    return {"checkpoints": checkpoints}


@router.get("/checkpoints/{job_id}")
async def get_job_checkpoints(job_id: str):
    """List checkpoints for a specific job."""
    from pathlib import Path

    job_dir = Path(_get_service().checkpoint_manager._dir) / job_id
    if not job_dir.exists():
        return {"checkpoints": []}

    checkpoints: list[dict[str, Any]] = []
    for ckpt in sorted(job_dir.glob("checkpoint_r*"), reverse=True):
        manifest_path = ckpt / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = _json.loads(manifest_path.read_text())
            # Read strategy metadata
            strategy_details = []
            for name in manifest.get("strategy_names", []):
                safe = name.replace("/", "_")
                meta_path = ckpt / f"strategy_{safe}.meta.json"
                if meta_path.exists():
                    meta = _json.loads(meta_path.read_text())
                    strategy_details.append({
                        "name": name,
                        "format": meta.get("format"),
                        "metadata": meta.get("metadata", {}),
                    })
            checkpoints.append({
                "round_idx": manifest.get("round_idx", 0),
                "timestamp": manifest.get("timestamp", 0),
                "path": str(ckpt),
                "strategies": strategy_details,
                "archive_formulas": manifest.get("archive_formulas", [])[:5],
                "context_state": manifest.get("context_state", {}),
            })
        except Exception:
            continue

    return {"job_id": job_id, "checkpoints": checkpoints}


@router.get("/factor-catalog")
async def get_factor_catalog(
    strategy: str | None = None,
    round_idx: int | None = None,
    min_ic: float | None = None,
    evaluated_only: bool = False,
    limit: int = Query(default=100, ge=1, le=500),
):
    """Query the factor catalog from the latest search context."""
    # Find the most recent completed job's factor catalog
    catalog_data: list[dict[str, Any]] = []
    catalog_stats: dict[str, Any] = {}

    # Check active search jobs for factor catalog
    for job_id, job in sorted(
        _SEARCH_JOBS.items(),
        key=lambda x: x[1].get("created_at", ""),
        reverse=True,
    ):
        result = job.get("result")
        if not result:
            continue
        pipeline = result.get("pipeline")
        if not pipeline:
            continue
        # Rebuild catalog from pipeline rounds
        from src.alpha.search.context import FactorCatalog, FactorCatalogEntry
        catalog = FactorCatalog()
        # Try to load from checkpoint if available
        ckpt = _get_service().checkpoint_manager.latest_checkpoint(job_id)
        if ckpt:
            try:
                from src.alpha.search.checkpoint import SearchCheckpoint
                saved = SearchCheckpoint.load(ckpt)
                catalog = FactorCatalog.from_json_list(saved.factor_catalog_json)
            except Exception:
                pass
        if len(catalog) > 0:
            entries = catalog.query(
                strategy=strategy,
                round_idx=round_idx,
                min_ic=min_ic,
                evaluated_only=evaluated_only,
                limit=limit,
            )
            catalog_data = [
                {
                    "formula": e.formula,
                    "expr_hash": e.expr_hash,
                    "strategy": e.strategy,
                    "round_idx": e.round_idx,
                    "rank_ic": e.rank_ic,
                    "sharpe": e.sharpe,
                    "turnover": e.turnover,
                    "fitness": e.fitness,
                    "evaluated": e.evaluated,
                }
                for e in entries
            ]
            catalog_stats = catalog.stats_by_strategy()
            break  # Use the first job with catalog data

    return {
        "entries": catalog_data,
        "stats": catalog_stats,
        "total": len(catalog_data),
    }


@router.get("/factor-catalog/stats")
async def get_factor_catalog_stats():
    """Aggregate statistics across all strategies."""
    stats: dict[str, Any] = {"strategies": {}, "total_factors": 0}

    for job_id, job in sorted(
        _SEARCH_JOBS.items(),
        key=lambda x: x[1].get("created_at", ""),
        reverse=True,
    ):
        result = job.get("result")
        if not result:
            continue
        ckpt = _get_service().checkpoint_manager.latest_checkpoint(job_id)
        if ckpt:
            try:
                from src.alpha.search.checkpoint import SearchCheckpoint
                saved = SearchCheckpoint.load(ckpt)
                from src.alpha.search.context import FactorCatalog
                catalog = FactorCatalog.from_json_list(saved.factor_catalog_json)
                stats["strategies"] = catalog.stats_by_strategy()
                stats["total_factors"] = len(catalog)
                stats["job_id"] = job_id
                break
            except Exception:
                pass

    return stats
