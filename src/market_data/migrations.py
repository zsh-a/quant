"""
ClickHouse schema 版本化迁移系统。

使用方式:
    from src.market_data.migrations import run_migrations
    run_migrations()  # 在应用启动时调用

迁移记录存储在 ClickHouse 的 system_meta.schema_migrations 表中。
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from loguru import logger

from src.market_data.clickhouse import create_clickhouse_client

# ---------------------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------------------

Migration = tuple[str, str, Callable]  # (version, description, callable)
_MIGRATIONS: list[Migration] = []


def migration(version: str, description: str):
    """Decorator to register a schema migration."""
    def decorator(fn: Callable):
        _MIGRATIONS.append((version, description, fn))
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Migration tracking table
# ---------------------------------------------------------------------------

_TRACKING_DDL = """
CREATE TABLE IF NOT EXISTS system_meta.schema_migrations (
    version     String,
    description String,
    applied_at  DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY version
"""


def _ensure_tracking_table(client):
    client.command("CREATE DATABASE IF NOT EXISTS system_meta")
    client.command(_TRACKING_DDL)


def _applied_versions(client) -> set[str]:
    result = client.query("SELECT version FROM system_meta.schema_migrations")
    return {row[0] for row in result.result_rows}


def _mark_applied(client, version: str, description: str):
    client.command(
        "INSERT INTO system_meta.schema_migrations (version, description) VALUES "
        "({version:String}, {desc:String})",
        parameters={"version": version, "desc": description},
    )


# ---------------------------------------------------------------------------
# Registered migrations
# ---------------------------------------------------------------------------

@migration("001", "Create stock_data database and core tables")
def _m001(client):
    client.command("CREATE DATABASE IF NOT EXISTS stock_data")
    client.command("""
        CREATE TABLE IF NOT EXISTS stock_data.stock_daily (
            code        String,
            date        Date,
            open        Float64,
            high        Float64,
            low         Float64,
            close       Float64,
            volume      Float64,
            amount      Float64,
            adjfactor   Float64 DEFAULT 1.0,
            turn        Float64 DEFAULT 0,
            tradestatus Float64 DEFAULT 1,
            pctChg      Float64 DEFAULT 0,
            isST        UInt8   DEFAULT 0,
            peTTM       Float64 DEFAULT 0,
            pbMRQ       Float64 DEFAULT 0,
            psTTM       Float64 DEFAULT 0,
            pcfNcfTTM   Float64 DEFAULT 0
        ) ENGINE = ReplacingMergeTree()
        ORDER BY (code, date)
        PARTITION BY toYYYYMM(date)
    """)
    client.command("""
        CREATE TABLE IF NOT EXISTS stock_data.stock_daily_meta (
            code               String,
            last_update_date   Date,
            last_adjfactor     Float64 DEFAULT 1.0,
            error_update_count UInt32  DEFAULT 0,
            name               String  DEFAULT ''
        ) ENGINE = ReplacingMergeTree()
        ORDER BY code
    """)


@migration("002", "Create crypto_data database and futures table")
def _m002(client):
    client.command("CREATE DATABASE IF NOT EXISTS crypto_data")
    client.command("""
        CREATE TABLE IF NOT EXISTS crypto_data.futures_5m (
            provider   LowCardinality(String),
            symbol     LowCardinality(String),
            interval   LowCardinality(String) DEFAULT '5m',
            open_time  DateTime64(3, 'UTC'),
            open       Float64,
            high       Float64,
            low        Float64,
            close      Float64,
            volume     Float64,
            quote_volume    Float64 DEFAULT 0,
            trade_count     UInt32  DEFAULT 0,
            taker_buy_vol   Float64 DEFAULT 0,
            taker_buy_quote Float64 DEFAULT 0
        ) ENGINE = ReplacingMergeTree()
        ORDER BY (provider, symbol, interval, open_time)
        PARTITION BY toYYYYMM(open_time)
    """)


@migration("003", "Create index_stocks and industry tables")
def _m003(client):
    client.command("""
        CREATE TABLE IF NOT EXISTS stock_data.index_stocks (
            `index` String,
            code    String
        ) ENGINE = ReplacingMergeTree()
        ORDER BY (`index`, code)
    """)
    client.command("""
        CREATE TABLE IF NOT EXISTS stock_data.industry_info (
            code           String,
            industry       String DEFAULT '',
            industry_code  String DEFAULT '',
            enter_date     Date
        ) ENGINE = ReplacingMergeTree()
        ORDER BY (code, enter_date)
    """)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_migrations() -> list[str]:
    """Execute pending migrations. Returns list of newly applied versions."""
    client = create_clickhouse_client()
    _ensure_tracking_table(client)
    applied = _applied_versions(client)
    newly_applied = []

    for version, description, fn in sorted(_MIGRATIONS, key=lambda m: m[0]):
        if version in applied:
            continue
        logger.info("Running migration {} — {}", version, description)
        try:
            fn(client)
            _mark_applied(client, version, description)
            newly_applied.append(version)
            logger.info("Migration {} applied successfully", version)
        except Exception as exc:
            logger.error("Migration {} failed: {}", version, exc)
            raise

    if not newly_applied:
        logger.info("Database schema is up to date (no pending migrations)")
    return newly_applied
