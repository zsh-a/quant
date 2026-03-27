from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Iterable

from loguru import logger

from src.datahub.binance import BinanceSpotDataAdapter
from src.datahub.bitget import BitgetDataAdapter
from src.config.settings import get_crypto_market_config
from src.market_data.crypto_store import CryptoMinuteBarStore, UnifiedMinuteBar
from src.market_data.crypto_sync_state import CryptoSyncStateStore


DEFAULT_CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
MINUTE_MS = 60_000
MAX_BATCH_LIMIT = 1000
BITGET_HISTORY_BATCH_LIMIT = 200
EMPTY_WINDOW_ADVANCE_DAYS = 30


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
        self.providers = {
            "bitget": BitgetDataAdapter(),
            "binance": BinanceSpotDataAdapter(),
        }

    def _store(self) -> CryptoMinuteBarStore:
        if self.store is None:
            self.store = CryptoMinuteBarStore()
        return self.store

    def ensure_schema(self) -> None:
        self._store().ensure_schema()

    def initialize_database(
        self,
        provider: str | None = None,
        symbols: Iterable[str] | None = None,
    ) -> dict:
        self.ensure_schema()
        providers = [provider] if provider else list(self.providers.keys())
        instrument_count = 0
        initialized = []
        for provider_name in providers:
            adapter = self.providers.get(provider_name.lower())
            if adapter is None:
                raise ValueError(f"Unsupported provider: {provider_name}")
            provider_symbols = list(symbols or self.config.default_symbols or DEFAULT_CRYPTO_SYMBOLS)
            rows = self._build_instruments(adapter, provider_symbols)
            instrument_count += self._store().upsert_instruments(rows)
            initialized.append(
                {
                    "provider": adapter.provider_name,
                    "market_type": adapter.market_type,
                    "symbols": provider_symbols,
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
    ) -> list[dict]:
        adapter = self.providers.get(provider.lower())
        if adapter is None:
            raise ValueError(f"Unsupported provider: {provider}")

        self.ensure_schema()
        end_time = end_time or datetime.now(UTC).replace(second=0, microsecond=0)
        symbols = list(symbols or self.config.default_symbols or DEFAULT_CRYPTO_SYMBOLS)
        self._store().upsert_instruments(self._build_instruments(adapter, symbols))
        summaries: list[dict] = []

        for symbol in symbols:
            latest = self._store().get_latest_open_time(
                provider=adapter.provider_name,
                symbol=symbol.upper(),
                interval=interval,
                market_type=adapter.market_type,
            )
            state_point = self.state_store.get_sync_point(
                provider=adapter.provider_name,
                market_type=adapter.market_type,
                symbol=symbol.upper(),
                interval=interval,
            )
            state_latest = None
            if state_point and state_point.get("last_open_time"):
                state_latest = datetime.fromisoformat(state_point["last_open_time"]).astimezone(UTC)
            effective_start = start_time or (
                latest + timedelta(minutes=1)
                if latest
                else state_latest + timedelta(minutes=1)
                if state_latest
                else end_time - timedelta(days=1)
            )
            self.state_store.mark_started(
                provider=adapter.provider_name,
                market_type=adapter.market_type,
                symbol=symbol.upper(),
                interval=interval,
                start_time=effective_start,
            )
            if effective_start >= end_time:
                self.state_store.mark_completed(
                    provider=adapter.provider_name,
                    market_type=adapter.market_type,
                    symbol=symbol.upper(),
                    interval=interval,
                    last_open_time=latest or state_latest,
                    fetched=0,
                    inserted=0,
                )
                summaries.append(
                    SyncSummary(
                        provider=adapter.provider_name,
                        market_type=adapter.market_type,
                        symbol=symbol.upper(),
                        fetched=0,
                        inserted=0,
                        start_time=effective_start.isoformat(),
                        end_time=end_time.isoformat(),
                    ).__dict__
                )
                continue

            fetched_total = 0
            inserted_total = 0
            cursor = effective_start
            last_open_time = latest or state_latest
            batch_limit = (
                BITGET_HISTORY_BATCH_LIMIT
                if use_historical_endpoint and adapter.provider_name == "bitget"
                else MAX_BATCH_LIMIT
            )
            while cursor < end_time:
                batch_end = min(
                    cursor + timedelta(minutes=batch_limit - 1),
                    end_time,
                )
                try:
                    if use_historical_endpoint and hasattr(adapter, "fetch_history_candles"):
                        records = adapter.fetch_history_candles(
                            symbol=symbol.upper(),
                            interval=interval,
                            start_time_ms=int(cursor.timestamp() * 1000),
                            end_time_ms=int(batch_end.timestamp() * 1000),
                            limit=batch_limit,
                        )
                    else:
                        records = adapter.fetch_candles(
                            symbol=symbol.upper(),
                            interval=interval,
                            start_time_ms=int(cursor.timestamp() * 1000),
                            end_time_ms=int(batch_end.timestamp() * 1000),
                            limit=batch_limit,
                        )
                except Exception as exc:
                    self.state_store.mark_failed(
                        provider=adapter.provider_name,
                        market_type=adapter.market_type,
                        symbol=symbol.upper(),
                        interval=interval,
                        error=str(exc),
                    )
                    raise
                if not records:
                    if use_historical_endpoint:
                        cursor = min(cursor + timedelta(days=EMPTY_WINDOW_ADVANCE_DAYS), end_time)
                        if progress_callback:
                            progress_callback(
                                {
                                    "provider": adapter.provider_name,
                                    "market_type": adapter.market_type,
                                    "symbol": symbol.upper(),
                                    "interval": interval,
                                    "cursor": cursor.isoformat(),
                                    "batch_end": batch_end.isoformat(),
                                    "last_open_time": last_open_time.isoformat() if last_open_time else "",
                                    "fetched": fetched_total,
                                    "inserted": inserted_total,
                                    "note": "empty_window_advanced",
                                }
                            )
                        continue
                    break
                normalized = self._normalize_records(adapter, symbol.upper(), interval, records)
                fetched_total += len(normalized)
                inserted_total += self._store().insert_bars(normalized)

                last_open_time = normalized[-1].open_time
                self.state_store.mark_progress(
                    provider=adapter.provider_name,
                    market_type=adapter.market_type,
                    symbol=symbol.upper(),
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
                            "symbol": symbol.upper(),
                            "interval": interval,
                            "cursor": cursor.isoformat(),
                            "batch_end": batch_end.isoformat(),
                            "last_open_time": last_open_time.isoformat(),
                            "fetched": fetched_total,
                            "inserted": inserted_total,
                        }
                    )
                next_cursor = last_open_time + timedelta(minutes=1)
                if next_cursor <= cursor:
                    break
                cursor = next_cursor

            self.state_store.mark_completed(
                provider=adapter.provider_name,
                market_type=adapter.market_type,
                symbol=symbol.upper(),
                interval=interval,
                last_open_time=last_open_time,
                fetched=fetched_total,
                inserted=inserted_total,
            )

            summary = SyncSummary(
                provider=adapter.provider_name,
                market_type=adapter.market_type,
                symbol=symbol.upper(),
                fetched=fetched_total,
                inserted=inserted_total,
                start_time=effective_start.isoformat(),
                end_time=end_time.isoformat(),
            )
            summaries.append(summary.__dict__)
            logger.info(
                f"Synced {provider}:{symbol} {interval} fetched={fetched_total} inserted={inserted_total}"
            )

        return summaries

    def sync_default_minute_bars(
        self,
        provider: str | None = None,
        symbols: Iterable[str] | None = None,
        interval: str | None = None,
        end_time: datetime | None = None,
    ) -> list[dict]:
        provider = provider or self.config.default_provider
        interval = interval or self.config.default_interval
        lookback = timedelta(hours=max(self.config.default_lookback_hours, 1))
        end_time = end_time or datetime.now(UTC).replace(second=0, microsecond=0)
        start_time = end_time - lookback
        return self.sync_minute_bars(
            provider=provider,
            symbols=list(symbols or self.config.default_symbols or DEFAULT_CRYPTO_SYMBOLS),
            interval=interval,
            start_time=start_time,
            end_time=end_time,
        )

    def bootstrap_default_dataset(
        self,
        provider: str | None = None,
        symbols: Iterable[str] | None = None,
        interval: str | None = None,
        end_time: datetime | None = None,
    ) -> dict:
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
    ) -> dict:
        provider = provider or self.config.default_provider
        interval = interval or self.config.default_interval
        configured_start = start_time or datetime.fromisoformat(self.config.full_history_start)
        if configured_start.tzinfo is None:
            configured_start = configured_start.replace(tzinfo=UTC)
        else:
            configured_start = configured_start.astimezone(UTC)

        init_result = self.initialize_database(provider=provider, symbols=symbols)
        sync_result = self.sync_minute_bars(
            provider=provider,
            symbols=list(symbols or self.config.default_symbols or DEFAULT_CRYPTO_SYMBOLS),
            interval=interval,
            start_time=configured_start,
            end_time=end_time,
            progress_callback=progress_callback,
            use_historical_endpoint=True,
        )
        return {
            "status": "backfilled",
            "start_time": configured_start.isoformat(),
            "end_time": (end_time or datetime.now(UTC).replace(second=0, microsecond=0)).isoformat(),
            "initialization": init_result,
            "sync_results": sync_result,
            "overview": self.get_overview(),
        }

    def _normalize_records(self, adapter, symbol: str, interval: str, records) -> list[UnifiedMinuteBar]:
        rows = []
        for record in records:
            open_time_ms = getattr(record, "open_time_ms", None) or getattr(record, "timestamp_ms", None)
            close_time_ms = getattr(record, "close_time_ms", None) or (
                int(open_time_ms) + MINUTE_MS - 1
            )
            exchange_symbol = getattr(record, "exchange_symbol", None) or symbol
            rows.append(
                UnifiedMinuteBar(
                    provider=adapter.provider_name,
                    market_type=adapter.market_type,
                    symbol=symbol,
                    exchange_symbol=exchange_symbol,
                    interval=interval,
                    open_time=datetime.fromtimestamp(int(open_time_ms) / 1000, tz=UTC),
                    close_time=datetime.fromtimestamp(int(close_time_ms) / 1000, tz=UTC),
                    open=float(record.open),
                    high=float(record.high),
                    low=float(record.low),
                    close=float(record.close),
                    volume_base=float(record.volume_base),
                    volume_quote=float(record.volume_quote),
                    trade_count=int(getattr(record, "trade_count", 0) or 0),
                )
            )
        return rows

    def query_bars(
        self,
        provider: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
    ) -> list[dict]:
        self.ensure_schema()
        return self._store().query_bars(
            provider=provider,
            symbol=symbol.upper(),
            start_time=start_time,
            end_time=end_time,
            interval=interval,
        )

    def get_overview(self) -> dict:
        self.ensure_schema()
        return self._store().get_overview()

    def get_coverage(self, interval: str = "1m", limit: int = 100) -> list[dict]:
        self.ensure_schema()
        return self._store().get_coverage(interval=interval, limit=limit)

    def _build_instruments(self, adapter, symbols: Iterable[str]) -> list[dict]:
        now = datetime.now(UTC)
        instruments = []
        for symbol in symbols:
            base_asset, quote_asset = self._split_symbol(symbol.upper())
            instruments.append(
                {
                    "provider": adapter.provider_name,
                    "market_type": adapter.market_type,
                    "symbol": symbol.upper(),
                    "exchange_symbol": symbol.upper(),
                    "base_asset": base_asset,
                    "quote_asset": quote_asset,
                    "is_active": 1,
                    "updated_at": now,
                }
            )
        return instruments

    def _split_symbol(self, symbol: str) -> tuple[str, str]:
        for quote in ("USDT", "USDC", "BUSD", "USD"):
            if symbol.endswith(quote) and len(symbol) > len(quote):
                return symbol[: -len(quote)], quote
        return symbol, ""
