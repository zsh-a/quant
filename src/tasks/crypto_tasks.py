"""
Crypto market data synchronization tasks.
"""

from __future__ import annotations

from datetime import UTC, datetime

from loguru import logger

from src.config.settings import get_crypto_market_config
from src.market_data.crypto_pipeline import CryptoMinuteSyncService
from src.tasks.celery_app import app

crypto_market_config = get_crypto_market_config()


def _parse_iso_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@app.task(name="src.tasks.crypto_tasks.sync_minute_bars")
def sync_minute_bars_task(
    provider: str,
    symbols: list[str] | None = None,
    interval: str = "1m",
    start_time: str | None = None,
    end_time: str | None = None,
):
    logger.info(
        f"Starting crypto minute-bar sync provider={provider} interval={interval} symbols={symbols}"
    )
    service = CryptoMinuteSyncService()
    result = service.sync_minute_bars(
        provider=provider,
        symbols=symbols,
        interval=interval,
        start_time=_parse_iso_dt(start_time),
        end_time=_parse_iso_dt(end_time),
    )
    return {
        "status": "success",
        "provider": provider,
        "interval": interval,
        "results": result,
    }


@app.task(name="src.tasks.crypto_tasks.sync_default_minute_bars")
def sync_default_minute_bars_task():
    provider = crypto_market_config.default_provider
    logger.info(
        f"Starting default crypto minute-bar sync provider={provider} interval={crypto_market_config.default_interval}"
    )
    service = CryptoMinuteSyncService()
    result = service.sync_default_minute_bars(
        provider=provider,
        symbols=crypto_market_config.default_symbols,
        interval=crypto_market_config.default_interval,
    )
    return {
        "status": "success",
        "provider": provider,
        "interval": crypto_market_config.default_interval,
        "results": result,
    }


@app.task(name="src.tasks.crypto_tasks.init_crypto_market_db")
def init_crypto_market_db_task(
    provider: str | None = None,
    symbols: list[str] | None = None,
):
    logger.info(f"Initializing crypto market database provider={provider} symbols={symbols}")
    service = CryptoMinuteSyncService()
    result = service.initialize_database(provider=provider, symbols=symbols)
    return result


@app.task(name="src.tasks.crypto_tasks.bootstrap_crypto_market_data")
def bootstrap_crypto_market_data_task(
    provider: str | None = None,
    symbols: list[str] | None = None,
    interval: str | None = None,
):
    logger.info(
        f"Bootstrapping crypto market database provider={provider} interval={interval} symbols={symbols}"
    )
    service = CryptoMinuteSyncService()
    return service.bootstrap_default_dataset(
        provider=provider,
        symbols=symbols,
        interval=interval,
    )


@app.task(name="src.tasks.crypto_tasks.backfill_crypto_market_data")
def backfill_crypto_market_data_task(
    provider: str | None = None,
    symbols: list[str] | None = None,
    interval: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
):
    logger.info(
        f"Backfilling crypto market data provider={provider} interval={interval} symbols={symbols} start={start_time} end={end_time}"
    )
    service = CryptoMinuteSyncService()
    return service.backfill_history(
        provider=provider,
        symbols=symbols,
        interval=interval,
        start_time=_parse_iso_dt(start_time),
        end_time=_parse_iso_dt(end_time),
    )
