from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable

from loguru import logger

from src.config.settings import get_crypto_market_config
from src.market_data.ccxt_adapter import DEFAULT_CRYPTO_SYMBOLS, build_default_providers
from src.market_data.crypto_store import CryptoMinuteBarStore
from src.market_data.crypto_sync_state import CryptoSyncStateStore


@dataclass
class SyncSummary:
    provider: str
    market_type: str
    symbol: str
    fetched: int
    inserted: int
    start_time: str
    end_time: str


class CryptoMinuteSyncService:
    def __init__(self, store: CryptoMinuteBarStore | None = None):
        self.store = store
        self.config = get_crypto_market_config()
        self.state_store = CryptoSyncStateStore(self.config.state_file)
        self.providers = build_default_providers()

    def _store(self) -> CryptoMinuteBarStore:
        if self.store is None:
            self.store = CryptoMinuteBarStore()
        return self.store

    def _get_provider(self, provider: str):
        adapter = self.providers.get(provider.lower())
        if adapter is None:
            raise ValueError(f"Unsupported provider: {provider}")
        return adapter

    def _normalize_time(self, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def _interval_delta(self, interval: str) -> timedelta:
        normalized = str(interval).strip().lower()
        if not normalized.endswith("m"):
            raise ValueError(f"Unsupported interval: {interval}")
        return timedelta(minutes=max(int(normalized[:-1]), 1))

    def _get_state_latest(self, provider: str, market_type: str, symbol: str, interval: str) -> datetime | None:
        state_point = self.state_store.get_sync_point(
            provider=provider,
            market_type=market_type,
            symbol=symbol,
            interval=interval,
        )
        if not state_point or not state_point.get("last_open_time"):
            return None
        return datetime.fromisoformat(state_point["last_open_time"]).astimezone(UTC)

    def _resolve_sync_start(
        self,
        provider: str,
        market_type: str,
        symbol: str,
        interval: str,
        requested_start: datetime | None,
        end_time: datetime,
    ) -> datetime:
        normalized_start = self._normalize_time(requested_start)
        if normalized_start is not None:
            return normalized_start

        step = self._interval_delta(interval)
        latest = self._store().get_latest_open_time(
            provider=provider,
            symbol=symbol,
            interval=interval,
            market_type=market_type,
        )
        if latest is not None:
            return latest + step

        state_latest = self._get_state_latest(provider, market_type, symbol, interval)
        if state_latest is not None:
            return state_latest + step

        return end_time - timedelta(days=1)

    def ensure_schema(self) -> None:
        self._store().ensure_schema()

    def list_providers(self) -> list[dict[str, Any]]:
        return [adapter.describe() for adapter in self.providers.values()]

    def initialize_database(
        self,
        provider: str | None = None,
        symbols: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        self.ensure_schema()
        providers = [provider] if provider else list(self.providers.keys())
        instrument_count = 0
        initialized = []
        provider_symbols = list(symbols or self.config.default_symbols or DEFAULT_CRYPTO_SYMBOLS)

        for provider_name in providers:
            adapter = self._get_provider(provider_name)
            rows = adapter.build_instruments(provider_symbols)
            instrument_count += self._store().upsert_instruments(rows)
            initialized.append(
                {
                    "provider": adapter.provider_name,
                    "market_type": adapter.market_type,
                    "symbols": [row["symbol"] for row in rows],
                }
            )

        return {
            "status": "initialized",
            "providers": initialized,
            "instrument_rows_written": instrument_count,
            "overview": self.get_overview(),
        }

    def sync_minute_bars(
        self,
        provider: str,
        symbols: Iterable[str] | None = None,
        interval: str = "1m",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        progress_callback=None,
        use_historical_endpoint: bool = False,
    ) -> list[dict[str, Any]]:
        del use_historical_endpoint

        adapter = self._get_provider(provider)
        self.ensure_schema()
        resolved_end_time = self._normalize_time(end_time) or datetime.now(UTC).replace(second=0, microsecond=0)
        resolved_symbols = list(symbols or self.config.default_symbols or DEFAULT_CRYPTO_SYMBOLS)
        self._store().upsert_instruments(adapter.build_instruments(resolved_symbols))
        summaries: list[dict[str, Any]] = []

        for requested_symbol in resolved_symbols:
            normalized_symbol = adapter.normalize_symbol(requested_symbol)
            effective_start = self._resolve_sync_start(
                provider=adapter.provider_name,
                market_type=adapter.market_type,
                symbol=normalized_symbol,
                interval=interval,
                requested_start=start_time,
                end_time=resolved_end_time,
            )
            latest = self._store().get_latest_open_time(
                provider=adapter.provider_name,
                symbol=normalized_symbol,
                interval=interval,
                market_type=adapter.market_type,
            )
            state_latest = self._get_state_latest(
                provider=adapter.provider_name,
                market_type=adapter.market_type,
                symbol=normalized_symbol,
                interval=interval,
            )
            last_open_time = latest or state_latest

            self.state_store.mark_started(
                provider=adapter.provider_name,
                market_type=adapter.market_type,
                symbol=normalized_symbol,
                interval=interval,
                start_time=effective_start,
            )

            if effective_start > resolved_end_time:
                self.state_store.mark_completed(
                    provider=adapter.provider_name,
                    market_type=adapter.market_type,
                    symbol=normalized_symbol,
                    interval=interval,
                    last_open_time=last_open_time,
                    fetched=0,
                    inserted=0,
                )
                summaries.append(
                    SyncSummary(
                        provider=adapter.provider_name,
                        market_type=adapter.market_type,
                        symbol=normalized_symbol,
                        fetched=0,
                        inserted=0,
                        start_time=effective_start.isoformat(),
                        end_time=resolved_end_time.isoformat(),
                    ).__dict__
                )
                continue

            windows = list(
                self._store().iter_missing_windows(
                    provider=adapter.provider_name,
                    symbol=normalized_symbol,
                    start_time=effective_start,
                    end_time=resolved_end_time,
                    interval=interval,
                    market_type=adapter.market_type,
                    batch_size=adapter.batch_limit,
                )
            )
            fetched_total = 0
            inserted_total = 0

            try:
                for window in windows:
                    bars = adapter.fetch_bars(
                        symbol=requested_symbol,
                        interval=interval,
                        start_time=window.start_time,
                        end_time=window.end_time,
                        limit=window.expected_points,
                    )
                    if bars and adapter.is_futures:
                        adapter.enrich_bars(bars, requested_symbol, window.start_time, window.end_time)
                    fetched_total += len(bars)
                    inserted_total += self._store().insert_bars(bars)
                    window_last = bars[-1].open_time if bars else window.end_time
                    if last_open_time is None or window_last > last_open_time:
                        last_open_time = window_last
                    self.state_store.mark_progress(
                        provider=adapter.provider_name,
                        market_type=adapter.market_type,
                        symbol=normalized_symbol,
                        interval=interval,
                        last_open_time=last_open_time,
                        fetched=fetched_total,
                        inserted=inserted_total,
                    )
                    if progress_callback:
                        progress_callback(
                            {
                                "provider": adapter.provider_name,
                                "market_type": adapter.market_type,
                                "symbol": normalized_symbol,
                                "interval": interval,
                                "cursor": window.start_time.isoformat(),
                                "batch_end": window.end_time.isoformat(),
                                "last_open_time": last_open_time.isoformat(),
                                "fetched": fetched_total,
                                "inserted": inserted_total,
                                "note": "window_synced" if bars else "window_empty",
                            }
                        )
            except Exception as exc:
                self.state_store.mark_failed(
                    provider=adapter.provider_name,
                    market_type=adapter.market_type,
                    symbol=normalized_symbol,
                    interval=interval,
                    error=str(exc),
                )
                raise

            self.state_store.mark_completed(
                provider=adapter.provider_name,
                market_type=adapter.market_type,
                symbol=normalized_symbol,
                interval=interval,
                last_open_time=last_open_time,
                fetched=fetched_total,
                inserted=inserted_total,
            )

            summary = SyncSummary(
                provider=adapter.provider_name,
                market_type=adapter.market_type,
                symbol=normalized_symbol,
                fetched=fetched_total,
                inserted=inserted_total,
                start_time=effective_start.isoformat(),
                end_time=resolved_end_time.isoformat(),
            )
            summaries.append(summary.__dict__)
            logger.info(
                "Synced {}:{} {} fetched={} inserted={} windows={}",
                adapter.provider_name,
                normalized_symbol,
                interval,
                fetched_total,
                inserted_total,
                len(windows),
            )

        return summaries

    def sync_default_minute_bars(
        self,
        provider: str | None = None,
        symbols: Iterable[str] | None = None,
        interval: str | None = None,
        end_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        resolved_provider = provider or self.config.default_provider
        resolved_interval = interval or self.config.default_interval
        lookback = timedelta(hours=max(self.config.default_lookback_hours, 1))
        resolved_end_time = self._normalize_time(end_time) or datetime.now(UTC).replace(second=0, microsecond=0)
        return self.sync_minute_bars(
            provider=resolved_provider,
            symbols=list(symbols or self.config.default_symbols or DEFAULT_CRYPTO_SYMBOLS),
            interval=resolved_interval,
            start_time=resolved_end_time - lookback,
            end_time=resolved_end_time,
        )

    def bootstrap_default_dataset(
        self,
        provider: str | None = None,
        symbols: Iterable[str] | None = None,
        interval: str | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        init_result = self.initialize_database(provider=provider, symbols=symbols)
        sync_result = self.sync_default_minute_bars(
            provider=provider,
            symbols=symbols,
            interval=interval,
            end_time=end_time,
        )
        return {
            "status": "bootstrapped",
            "initialization": init_result,
            "sync_results": sync_result,
            "overview": self.get_overview(),
        }

    def backfill_history(
        self,
        provider: str | None = None,
        symbols: Iterable[str] | None = None,
        interval: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        progress_callback=None,
    ) -> dict[str, Any]:
        resolved_provider = provider or self.config.default_provider
        resolved_interval = interval or self.config.default_interval
        configured_start = self._normalize_time(start_time)
        if configured_start is None:
            configured_start = datetime.fromisoformat(self.config.full_history_start)
            configured_start = self._normalize_time(configured_start)

        init_result = self.initialize_database(provider=resolved_provider, symbols=symbols)
        sync_result = self.sync_minute_bars(
            provider=resolved_provider,
            symbols=list(symbols or self.config.default_symbols or DEFAULT_CRYPTO_SYMBOLS),
            interval=resolved_interval,
            start_time=configured_start,
            end_time=end_time,
            progress_callback=progress_callback,
        )
        resolved_end_time = self._normalize_time(end_time) or datetime.now(UTC).replace(second=0, microsecond=0)
        return {
            "status": "backfilled",
            "start_time": configured_start.isoformat(),
            "end_time": resolved_end_time.isoformat(),
            "initialization": init_result,
            "sync_results": sync_result,
            "overview": self.get_overview(),
        }

    def query_bars(
        self,
        provider: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
    ) -> list[dict[str, Any]]:
        self.ensure_schema()
        return self._store().query_bars(
            provider=provider,
            symbol=symbol.upper(),
            start_time=start_time,
            end_time=end_time,
            interval=interval,
        )

    def get_overview(self) -> dict[str, Any]:
        self.ensure_schema()
        return self._store().get_overview()

    def get_coverage(self, interval: str = "1m", limit: int = 100) -> list[dict[str, Any]]:
        self.ensure_schema()
        return self._store().get_coverage(interval=interval, limit=limit)
