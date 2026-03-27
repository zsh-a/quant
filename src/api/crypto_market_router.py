"""
Crypto market data API endpoints.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.config.settings import get_crypto_market_config
from src.market_data.crypto_pipeline import CryptoMinuteSyncService
from src.tasks.crypto_tasks import (
    backfill_crypto_market_data_task,
    bootstrap_crypto_market_data_task,
    init_crypto_market_db_task,
    sync_default_minute_bars_task,
    sync_minute_bars_task,
)

router = APIRouter(prefix="/crypto-market", tags=["crypto-market"])
service = CryptoMinuteSyncService()
config = get_crypto_market_config()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class CryptoMinuteSyncRequest(BaseModel):
    provider: str = config.default_provider
    symbols: list[str] = Field(default_factory=list)
    interval: str = config.default_interval
    start_time: str | None = None
    end_time: str | None = None
    async_mode: bool = False


class CryptoInitRequest(BaseModel):
    provider: str | None = None
    symbols: list[str] = Field(default_factory=list)
    async_mode: bool = False


class CryptoBootstrapRequest(BaseModel):
    provider: str | None = None
    symbols: list[str] = Field(default_factory=list)
    interval: str | None = None
    async_mode: bool = False


class CryptoBackfillRequest(BaseModel):
    provider: str | None = None
    symbols: list[str] = Field(default_factory=list)
    interval: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    async_mode: bool = False


@router.get("/providers")
async def list_crypto_providers():
    return {
        "providers": [
            {"provider": "bitget", "market_type": "perpetual", "intervals": ["1m"]},
            {"provider": "binance", "market_type": "spot", "intervals": ["1m"]},
        ],
        "defaults": config.model_dump(),
    }


@router.get("/overview")
async def get_crypto_overview():
    return service.get_overview()


@router.get("/coverage")
async def get_crypto_coverage(interval: str = "1m", limit: int = 100):
    return {"coverage": service.get_coverage(interval=interval, limit=limit)}


@router.post("/init-db")
async def init_crypto_market_db(request: CryptoInitRequest):
    try:
        symbols = request.symbols or None
        if request.async_mode:
            task = init_crypto_market_db_task.apply_async(
                kwargs={"provider": request.provider, "symbols": symbols},
                queue="automation",
            )
            return {"status": "submitted", "task_id": task.id}
        return service.initialize_database(provider=request.provider, symbols=symbols)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/bootstrap")
async def bootstrap_crypto_market_data(request: CryptoBootstrapRequest):
    try:
        symbols = request.symbols or None
        if request.async_mode:
            task = bootstrap_crypto_market_data_task.apply_async(
                kwargs={
                    "provider": request.provider,
                    "symbols": symbols,
                    "interval": request.interval,
                },
                queue="automation",
            )
            return {"status": "submitted", "task_id": task.id}
        return service.bootstrap_default_dataset(
            provider=request.provider,
            symbols=symbols,
            interval=request.interval,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/backfill")
async def backfill_crypto_market_data(request: CryptoBackfillRequest):
    try:
        symbols = request.symbols or None
        if request.async_mode:
            task = backfill_crypto_market_data_task.apply_async(
                kwargs={
                    "provider": request.provider,
                    "symbols": symbols,
                    "interval": request.interval,
                    "start_time": request.start_time,
                    "end_time": request.end_time,
                },
                queue="automation",
            )
            return {"status": "submitted", "task_id": task.id}
        return service.backfill_history(
            provider=request.provider,
            symbols=symbols,
            interval=request.interval,
            start_time=_parse_iso(request.start_time),
            end_time=_parse_iso(request.end_time),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/sync")
async def sync_crypto_minute_bars(request: CryptoMinuteSyncRequest):
    try:
        if request.async_mode:
            task = sync_minute_bars_task.apply_async(
                kwargs={
                    "provider": request.provider,
                    "symbols": request.symbols,
                    "interval": request.interval,
                    "start_time": request.start_time,
                    "end_time": request.end_time,
                },
                queue="automation",
            )
            return {"status": "submitted", "task_id": task.id}

        results = service.sync_minute_bars(
            provider=request.provider,
            symbols=request.symbols,
            interval=request.interval,
            start_time=_parse_iso(request.start_time),
            end_time=_parse_iso(request.end_time),
        )
        return {"status": "success", "results": results}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/sync-default")
async def sync_default_crypto_minute_bars(async_mode: bool = False):
    if async_mode:
        task = sync_default_minute_bars_task.apply_async(queue="automation")
        return {"status": "submitted", "task_id": task.id}

    results = service.sync_default_minute_bars()
    return {"status": "success", "results": results}


@router.get("/bars")
async def get_crypto_bars(
    provider: str,
    symbol: str,
    start_time: str,
    end_time: str,
    interval: str = "1m",
):
    try:
        bars = service.query_bars(
            provider=provider,
            symbol=symbol.upper(),
            start_time=_parse_iso(start_time),
            end_time=_parse_iso(end_time),
            interval=interval,
        )
        return {"provider": provider, "symbol": symbol.upper(), "interval": interval, "bars": bars}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
