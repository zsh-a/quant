from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Iterator

import pandas as pd

from src.market_data.clickhouse import create_clickhouse_client


@dataclass
class UnifiedMinuteBar:
    provider: str
    market_type: str
    symbol: str
    exchange_symbol: str
    interval: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume_base: float
    volume_quote: float
    trade_count: int = 0
    funding_rate: float = 0.0
    open_interest: float = 0.0
    ingest_source: str = "api_sync"
    ingested_at: datetime | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["ingested_at"] = row["ingested_at"] or datetime.now(timezone.utc)
        return row


@dataclass(frozen=True)
class SyncWindow:
    start_time: datetime
    end_time: datetime
    expected_points: int


class CryptoMinuteBarStore:
    def __init__(self, client=None):
        self.client = client or create_clickhouse_client()

    def _ensure_utc(self, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _interval_delta(self, interval: str) -> timedelta:
        normalized = str(interval).strip().lower()
        if not normalized.endswith("m"):
            raise ValueError(f"Unsupported interval: {interval}")
        minutes = int(normalized[:-1])
        return timedelta(minutes=max(minutes, 1))

    def _sql_quote(self, value: str) -> str:
        return str(value).replace("'", "''")

    def _sql_datetime(self, value: datetime) -> str:
        normalized = self._ensure_utc(value)
        return normalized.strftime("%Y-%m-%d %H:%M:%S")

    def _build_common_filters(
        self,
        provider: str,
        symbol: str,
        interval: str,
        market_type: str | None = None,
    ) -> list[str]:
        filters = [
            f"provider = '{self._sql_quote(provider)}'",
            f"symbol = '{self._sql_quote(symbol)}'",
            f"interval = '{self._sql_quote(interval)}'",
        ]
        if market_type:
            filters.append(f"market_type = '{self._sql_quote(market_type)}'")
        return filters

    def ensure_schema(self) -> None:
        self.client.command("CREATE DATABASE IF NOT EXISTS crypto_data")
        self.client.command(
            """
            CREATE TABLE IF NOT EXISTS crypto_data.minute_bars
            (
                provider LowCardinality(String),
                market_type LowCardinality(String),
                symbol LowCardinality(String),
                exchange_symbol LowCardinality(String),
                interval LowCardinality(String),
                open_time DateTime64(3, 'UTC'),
                close_time DateTime64(3, 'UTC'),
                open Float64,
                high Float64,
                low Float64,
                close Float64,
                volume_base Float64,
                volume_quote Float64,
                trade_count UInt32,
                funding_rate Float64 DEFAULT 0,
                open_interest Float64 DEFAULT 0,
                ingest_source LowCardinality(String),
                ingested_at DateTime64(3, 'UTC')
            )
            ENGINE = ReplacingMergeTree(ingested_at)
            PARTITION BY toYYYYMM(open_time)
            ORDER BY (provider, market_type, symbol, interval, open_time)
            """
        )
        self.client.command(
            """
            CREATE TABLE IF NOT EXISTS crypto_data.instruments
            (
                provider LowCardinality(String),
                market_type LowCardinality(String),
                symbol LowCardinality(String),
                exchange_symbol LowCardinality(String),
                base_asset LowCardinality(String),
                quote_asset LowCardinality(String),
                is_active UInt8,
                updated_at DateTime64(3, 'UTC')
            )
            ENGINE = ReplacingMergeTree(updated_at)
            ORDER BY (provider, market_type, symbol)
            """
        )

    def insert_bars(self, bars: Iterable[UnifiedMinuteBar]) -> int:
        grouped_rows: dict[tuple[str, str, str, str], dict[datetime, dict[str, Any]]] = defaultdict(dict)
        for bar in bars:
            row = bar.to_row()
            key = (
                row["provider"],
                row["market_type"],
                row["symbol"],
                row["interval"],
            )
            grouped_rows[key][self._ensure_utc(row["open_time"])] = row

        rows_to_insert: list[dict[str, Any]] = []
        for key, rows_by_open_time in grouped_rows.items():
            provider, market_type, symbol, interval = key
            ordered_times = sorted(rows_by_open_time)
            if not ordered_times:
                continue
            existing_times = self.get_existing_open_times(
                provider=provider,
                symbol=symbol,
                start_time=ordered_times[0],
                end_time=ordered_times[-1],
                interval=interval,
                market_type=market_type,
            )
            for open_time in ordered_times:
                if open_time not in existing_times:
                    rows_to_insert.append(rows_by_open_time[open_time])

        if not rows_to_insert:
            return 0

        df = pd.DataFrame(rows_to_insert)
        self.client.insert_df("crypto_data.minute_bars", df)
        return len(df)

    def upsert_instruments(self, instruments: list[dict[str, Any]]) -> int:
        if not instruments:
            return 0
        df = pd.DataFrame(instruments)
        self.client.insert_df("crypto_data.instruments", df)
        return len(df)

    def get_latest_open_time(
        self,
        provider: str,
        symbol: str,
        interval: str = "1m",
        market_type: str | None = None,
    ) -> datetime | None:
        filters = self._build_common_filters(provider, symbol, interval, market_type)
        sql = "SELECT max(open_time) FROM crypto_data.minute_bars WHERE " + " AND ".join(filters)
        result = self.client.query(sql)
        if not result.result_rows or result.result_rows[0][0] is None:
            return None
        return self._ensure_utc(result.result_rows[0][0])

    def get_existing_open_times(
        self,
        provider: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
        market_type: str | None = None,
    ) -> set[datetime]:
        filters = self._build_common_filters(provider, symbol, interval, market_type)
        filters.extend(
            [
                f"open_time >= toDateTime64('{self._sql_datetime(start_time)}', 3, 'UTC')",
                f"open_time <= toDateTime64('{self._sql_datetime(end_time)}', 3, 'UTC')",
            ]
        )
        sql = f"""
        SELECT DISTINCT open_time
        FROM crypto_data.minute_bars
        WHERE {' AND '.join(filters)}
        ORDER BY open_time
        """
        result = self.client.query(sql)
        return {self._ensure_utc(row[0]) for row in result.result_rows}

    def iter_missing_windows(
        self,
        provider: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
        market_type: str | None = None,
        batch_size: int = 1000,
    ) -> Iterator[SyncWindow]:
        normalized_start = self._ensure_utc(start_time)
        normalized_end = self._ensure_utc(end_time)
        if normalized_start is None or normalized_end is None or normalized_start > normalized_end:
            return iter(())

        step = self._interval_delta(interval)
        existing_times = sorted(
            self.get_existing_open_times(
                provider=provider,
                symbol=symbol,
                start_time=normalized_start,
                end_time=normalized_end,
                interval=interval,
                market_type=market_type,
            )
        )
        missing_ranges = self._find_missing_ranges(
            start_time=normalized_start,
            end_time=normalized_end,
            existing_times=existing_times,
            step=step,
        )
        return self._split_missing_ranges(missing_ranges, step=step, batch_size=batch_size)

    def query_bars(
        self,
        provider: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
    ) -> list[dict[str, Any]]:
        sql = f"""
        SELECT provider, market_type, symbol, exchange_symbol, interval, open_time, close_time,
               open, high, low, close, volume_base, volume_quote, trade_count,
               funding_rate, open_interest
        FROM crypto_data.minute_bars
        WHERE provider = '{self._sql_quote(provider)}'
          AND symbol = '{self._sql_quote(symbol)}'
          AND interval = '{self._sql_quote(interval)}'
          AND open_time >= toDateTime64('{self._sql_datetime(start_time)}', 3, 'UTC')
          AND open_time <= toDateTime64('{self._sql_datetime(end_time)}', 3, 'UTC')
        ORDER BY open_time
        """
        result = self.client.query(sql)
        columns = result.column_names
        return [dict(zip(columns, row)) for row in result.result_rows]

    def get_overview(self) -> dict[str, Any]:
        def scalar(sql: str):
            result = self.client.query(sql)
            if not result.result_rows:
                return None
            return result.result_rows[0][0]

        return {
            "row_count": int(scalar("SELECT count() FROM crypto_data.minute_bars") or 0),
            "provider_count": int(scalar("SELECT uniqExact(provider) FROM crypto_data.minute_bars") or 0),
            "symbol_count": int(scalar("SELECT uniqExact(symbol) FROM crypto_data.minute_bars") or 0),
            "instrument_count": int(scalar("SELECT count() FROM crypto_data.instruments") or 0),
            "earliest_open_time": scalar("SELECT min(open_time) FROM crypto_data.minute_bars"),
            "latest_open_time": scalar("SELECT max(open_time) FROM crypto_data.minute_bars"),
        }

    def get_max_open_time_in_range(
        self,
        provider: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
        market_type: str | None = None,
    ) -> datetime | None:
        filters = self._build_common_filters(provider, symbol, interval, market_type)
        filters.extend(
            [
                f"open_time >= toDateTime64('{self._sql_datetime(start_time)}', 3, 'UTC')",
                f"open_time <= toDateTime64('{self._sql_datetime(end_time)}', 3, 'UTC')",
            ]
        )
        sql = "SELECT max(open_time) FROM crypto_data.minute_bars WHERE " + " AND ".join(filters)
        result = self.client.query(sql)
        if not result.result_rows or result.result_rows[0][0] is None:
            return None
        return self._ensure_utc(result.result_rows[0][0])

    def get_range_coverage_stats(
        self,
        provider: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
        market_type: str | None = None,
    ) -> dict[str, Any]:
        filters = self._build_common_filters(provider, symbol, interval, market_type)
        filters.extend(
            [
                f"open_time >= toDateTime64('{self._sql_datetime(start_time)}', 3, 'UTC')",
                f"open_time <= toDateTime64('{self._sql_datetime(end_time)}', 3, 'UTC')",
            ]
        )
        sql = f"""
        SELECT
            uniqExact(open_time) AS distinct_open_time_count,
            min(open_time) AS earliest_open_time,
            max(open_time) AS latest_open_time
        FROM crypto_data.minute_bars
        WHERE {' AND '.join(filters)}
        """
        result = self.client.query(sql)
        row = result.result_rows[0] if result.result_rows else (0, None, None)
        return {
            "distinct_open_time_count": int(row[0] or 0),
            "earliest_open_time": self._ensure_utc(row[1]),
            "latest_open_time": self._ensure_utc(row[2]),
        }

    def get_coverage(self, interval: str = "1m", limit: int = 100) -> list[dict[str, Any]]:
        sql = f"""
        SELECT provider, market_type, symbol, interval,
               min(open_time) AS earliest_open_time,
               max(open_time) AS latest_open_time,
               count() AS row_count
        FROM crypto_data.minute_bars
        WHERE interval = '{self._sql_quote(interval)}'
        GROUP BY provider, market_type, symbol, interval
        ORDER BY provider, market_type, symbol
        LIMIT {int(limit)}
        """
        result = self.client.query(sql)
        return [dict(zip(result.column_names, row)) for row in result.result_rows]

    @staticmethod
    def _find_missing_ranges(
        start_time: datetime,
        end_time: datetime,
        existing_times: list[datetime],
        step: timedelta,
    ) -> list[tuple[datetime, datetime]]:
        cursor = start_time
        missing_ranges: list[tuple[datetime, datetime]] = []
        for existing_time in existing_times:
            if existing_time < cursor:
                continue
            if existing_time > end_time:
                break
            if existing_time > cursor:
                missing_ranges.append((cursor, existing_time - step))
            cursor = max(cursor, existing_time + step)
        if cursor <= end_time:
            missing_ranges.append((cursor, end_time))
        return [(start, end) for start, end in missing_ranges if start <= end]

    @staticmethod
    def _split_missing_ranges(
        missing_ranges: list[tuple[datetime, datetime]],
        step: timedelta,
        batch_size: int,
    ) -> Iterator[SyncWindow]:
        for range_start, range_end in missing_ranges:
            cursor = range_start
            while cursor <= range_end:
                batch_end = min(cursor + step * (max(batch_size, 1) - 1), range_end)
                expected_points = int(((batch_end - cursor) / step)) + 1
                yield SyncWindow(
                    start_time=cursor,
                    end_time=batch_end,
                    expected_points=expected_points,
                )
                cursor = batch_end + step
