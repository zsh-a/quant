from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

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
    ingest_source: str = "api_sync"
    ingested_at: datetime | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["ingested_at"] = row["ingested_at"] or datetime.now(timezone.utc)
        return row


class CryptoMinuteBarStore:
    def __init__(self, client=None):
        self.client = client or create_clickhouse_client()

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
        rows = [bar.to_row() for bar in bars]
        if not rows:
            return 0
        df = pd.DataFrame(rows)
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
        filters = [
            f"provider = '{provider}'",
            f"symbol = '{symbol}'",
            f"interval = '{interval}'",
        ]
        if market_type:
            filters.append(f"market_type = '{market_type}'")
        sql = (
            "SELECT max(open_time) FROM crypto_data.minute_bars WHERE "
            + " AND ".join(filters)
        )
        result = self.client.query(sql)
        if not result.result_rows or result.result_rows[0][0] is None:
            return None
        return result.result_rows[0][0]

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
               open, high, low, close, volume_base, volume_quote, trade_count
        FROM crypto_data.minute_bars
        WHERE provider = '{provider}'
          AND symbol = '{symbol}'
          AND interval = '{interval}'
          AND open_time >= toDateTime64('{start_time.strftime("%Y-%m-%d %H:%M:%S")}', 3, 'UTC')
          AND open_time <= toDateTime64('{end_time.strftime("%Y-%m-%d %H:%M:%S")}', 3, 'UTC')
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

    def get_coverage(self, interval: str = "1m", limit: int = 100) -> list[dict[str, Any]]:
        sql = f"""
        SELECT provider, market_type, symbol, interval,
               min(open_time) AS earliest_open_time,
               max(open_time) AS latest_open_time,
               count() AS row_count
        FROM crypto_data.minute_bars
        WHERE interval = '{interval}'
        GROUP BY provider, market_type, symbol, interval
        ORDER BY provider, market_type, symbol
        LIMIT {int(limit)}
        """
        result = self.client.query(sql)
        return [dict(zip(result.column_names, row)) for row in result.result_rows]
