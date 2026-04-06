# Data Pipeline

## Overview

```
Data Sources                    Processing              Storage
─────────────                  ──────────              ───────
Akshare (A-share daily)   ─┐
Baostock (A-share hist)   ─┤   Processors             ClickHouse
TDX/Tongdaxin (domestic)  ─┼─→ (normalize, validate) ──→ (columnar, partitioned)
CCXT (multi-exchange)     ─┤                               │
Binance Vision (crypto)   ─┘                          DataStream
                                                      (chunked loading)
                                                           │
                                                      TradingEngine / AlphaService
```

## Data Sources

### A-Share Market

| Source | Module | Coverage |
|--------|--------|----------|
| Akshare | `src/market_data/processors/akshare.py` | Chinese market daily data, ETFs, indices |
| Baostock | `src/market_data/processors/baostock.py` | Historical stock data, fundamentals |
| TDX | `src/market_data/processors/tdx.py` | Tongdaxin domestic real-time data |

### Cryptocurrency

| Source | Module | Coverage |
|--------|--------|----------|
| CCXT | `src/market_data/ccxt_adapter.py` | Multi-exchange (100+ exchanges) |
| Binance Vision | `src/market_data/binance_vision.py` | Binance historical klines |

**Crypto pipeline**: `src/market_data/crypto_pipeline.py`
- Bootstrap: Initialize DB schema + download historical data
- Backfill: Fill gaps in historical data
- Sync: Incremental minute bar updates
- State tracking: `crypto_sync_state.json`

**Crypto store**: `src/market_data/crypto_store.py`
- ClickHouse table management for crypto bars
- Intervals: 1m, 5m, 15m, 1h, 4h, 1d
- Fields: open, high, low, close, volume, quote_volume, trades_count

## ClickHouse Storage

**Config**: `src/market_data/clickhouse.py`, `src/market_data/db.py`

### Schema

```sql
-- A-share daily bars
CREATE TABLE market_data (
    stock_code String,
    trade_date Date,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64,
    amount Float64,
    adjfactor Float64,
    ...
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (stock_code, trade_date)

-- Crypto bars
CREATE TABLE crypto_bars_{interval} (
    symbol String,
    timestamp DateTime,
    open Float64, high Float64, low Float64, close Float64,
    volume Float64, quote_volume Float64, trades_count UInt64
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp)
```

### Connection

Default config (override via `QUANT_DATABASE__*`):
- Host: `localhost`
- Port: `8123`
- Username: `default`
- Password: (empty)

## DataStream Implementations

**File**: `src/core/data_stream.py`

### CSVDataStream
- Loads from CSV files, maps symbols to file paths
- Date range filtering
- Assumes aligned timestamps across symbols

### DBDataStream
- Queries ClickHouse with chunked loading for memory efficiency
- Dynamic chunk sizing:
  - \>1000 symbols: 1 month chunks
  - 100-1000 symbols: 3 month chunks
  - <100 symbols: 12 month chunks
- Tracks memory usage per chunk
- Supports `adjfactor` column for adjusted prices

### RealtimeDataStream
- Event-driven live data fetching
- Sources: AkShare (primary), TuShare (fallback)
- Trading hours detection: 09:30-11:30, 13:00-15:00 CST weekdays
- Auto retry: 3 attempts, 5s delay
- Mock data mode for testing

## Factor Storage (jointdata/)

Independent ClickHouse-based factor data storage system.

**Key files**:
- `jointdata/main.py` - Unified API
- `jointdata/factor_manager.py` - Factor metadata & CRUD
- `jointdata/csv_importer.py` - Batch CSV import with chunking
- `jointdata/wide_table_manager.py` - Wide-format table management

**Features**:
- Monthly partitioning by `trade_date`
- Composite indices (`stock_code`, `trade_date`)
- Factor metadata management
- Incremental update mechanism
- See `jointdata/README.md` for detailed usage

## Update Pipeline

**File**: `src/market_data/update_pipeline.py`

Orchestrates automated data updates:
1. Fetch latest data from sources
2. Validate and normalize
3. Upsert into ClickHouse
4. Update sync state

Triggered via:
- API: `POST /market-admin/update-runs`
- Automation: `POST /data-update/run`
- Celery task: `src/tasks/data_tasks.py`

## Crypto CLI

**File**: `src/market_data/crypto_cli.py`

Command-line interface for crypto data management:
```bash
python -m src.market_data.crypto_cli init-db --provider binance
python -m src.market_data.crypto_cli bootstrap --symbols BTCUSDT,ETHUSDT
python -m src.market_data.crypto_cli backfill --start 2024-01-01
python -m src.market_data.crypto_cli sync --interval 1h
```
