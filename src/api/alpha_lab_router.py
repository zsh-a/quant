"""
Alpha Lab API endpoints.
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.alpha import AlphaService
from src.alpha.tracing import InMemoryCollector, tracer
from src.config.settings import (
    get_alpha_lab_config,
    get_bitget_config,
    get_crypto_market_config,
)


router = APIRouter(prefix="/alpha-lab", tags=["alpha-lab"])
service = AlphaService()

_memory_collector = InMemoryCollector()
tracer.add_collector(_memory_collector)


class CompileRequest(BaseModel):
    formula: str


class EvaluateRequest(BaseModel):
    formula: str
    fields: dict[str, list[list[float]]]
    liquidity_mask: list[list[bool]] | None = None
    session_mask: list[list[bool]] | None = None


class SeedPopulationRequest(BaseModel):
    seeds: list[str] = Field(default_factory=list)
    population_size: int = 8


class BreedRequest(BaseModel):
    formulas: list[str] = Field(default_factory=list)
    offspring_count: int = 4


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


class SaveZooRequest(BaseModel):
    formula: str
    fitness: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    lineage: dict[str, str] = Field(default_factory=dict)
    note: str | None = None
    tags: list[str] = Field(default_factory=list)
    source: str = "manual"


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
        "data_source": "crypto_data.futures_5m",
        "intervals": ["1m", "5m", "15m", "1h", "4h"],
        "sample_formulas": [
            "CSRank(ts_mean(close, 5) - close)",
            "CSRank(ts_std(close, 10))",
            "CSRank(oi_delta(open_interest, 3) - spread_ratio(bid_ask_spread, close))",
        ],
    }


@router.get("/operators")
async def list_operators():
    return {
        "operators": service.list_operators(),
        "defaults": _workspace_defaults(),
    }


@router.get("/workspace")
async def get_workspace(
    run_limit: int = Query(default=8, ge=1, le=50),
    zoo_limit: int = Query(default=16, ge=1, le=100),
):
    return {
        "operators": service.list_operators(),
        "defaults": _workspace_defaults(),
        "runs": service.list_runs(limit=run_limit),
        "zoo": service.list_zoo(limit=zoo_limit),
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


@router.post("/evaluate")
async def evaluate_formula(request: EvaluateRequest):
    try:
        return service.evaluate_formula(
            formula=request.formula,
            fields=request.fields,
            liquidity_mask=request.liquidity_mask,
            session_mask=request.session_mask,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/population/seed")
async def seed_population(request: SeedPopulationRequest):
    return {
        "population": service.seed_population(request.seeds, request.population_size),
    }


@router.post("/population/breed")
async def breed_population(request: BreedRequest):
    return {
        "offspring": service.breed_population(request.formulas, request.offspring_count),
    }


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


@router.post("/search-db")
async def search_formulas_on_db(request: SearchDbRequest):
    try:
        return service.search_formulas_on_db(
            symbols=request.symbols,
            start_time=request.start_time,
            end_time=request.end_time,
            interval=request.interval,
            min_quote_volume=request.min_quote_volume,
            seeds=request.seeds,
            population_size=request.population_size,
            offspring_count=request.offspring_count,
            top_k=request.top_k,
            generations=request.generations,
            run_name=request.run_name,
            persist=request.persist,
            novelty_threshold=request.novelty_threshold,
            n_splits=request.n_splits,
            purge_window=request.purge_window,
            embargo_window=request.embargo_window,
            blocked_utc_hours=_normalize_blocked_hours(request.blocked_utc_hours),
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/runs")
async def list_runs(limit: int = Query(default=20, ge=1, le=100)):
    return {"runs": service.list_runs(limit=limit)}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    try:
        return service.load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/lineage/{run_id}")
async def get_lineage(run_id: str):
    try:
        return service.get_lineage(run_id)
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


@router.get("/bitget/config")
async def get_bitget_defaults():
    return get_bitget_config().model_dump()
