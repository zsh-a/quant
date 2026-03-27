CREATE TABLE stock_data.stock_daily
(
    `date` Date CODEC(Delta, LZ4),
    `code` LowCardinality(String),
    `open` Float64 CODEC(Gorilla, LZ4),
    `high` Float64 CODEC(Gorilla, LZ4),
    `low` Float64 CODEC(Gorilla, LZ4),
    `close` Float64 CODEC(Gorilla, LZ4),
    `preclose` Float64 CODEC(Gorilla, LZ4),
    `volume` UInt64 CODEC(T64, LZ4),
    `amount` Float64 CODEC(Gorilla, LZ4),
    `turn` Float64 CODEC(Gorilla, LZ4),
    `pctChg` Float64 CODEC(Gorilla, LZ4),
    `peTTM` Float64 CODEC(Gorilla, LZ4),
    `pbMRQ` Float64 CODEC(Gorilla, LZ4),
    `tradestatus` Int16,
    `isST` Int16,
    `adjfactor` Float64 CODEC(Gorilla, LZ4)
) ENGINE = ReplacingMergeTree()
ORDER BY (code, date);

-- OPTIMIZE TABLE stock_data.stock_daily FINAL;

CREATE TABLE stock_data.stock_daily_meta
(
    `code` LowCardinality(String),
    `name` LowCardinality(String),
    `last_update_date` Date,
    `last_adjfactor` Float64 CODEC(Gorilla, LZ4),
    `error_update_count` UInt32
) ENGINE = ReplacingMergeTree()
ORDER BY (code);

-- OPTIMIZE TABLE stock_data.stock_daily_meta FINAL;

CREATE TABLE stock_data.trade_dates
(   `calendar_date` Date CODEC(Delta, LZ4),
    `is_trading_day` UInt8
) ENGINE = ReplacingMergeTree()
ORDER BY (calendar_date);

CREATE TABLE stock_data.all_stock
(
    `day` Date CODEC(Delta, LZ4),
    `code` LowCardinality(String),
    `tradeStatus` UInt8, -- 0: suspended, 1: trading
    `code_name` LowCardinality(String)
) ENGINE = ReplacingMergeTree()
ORDER BY (day, code);

-- View to track stock name history (Slowly Changing Dimension)
-- It extracts unique name periods from the daily all_stock snapshots.
CREATE VIEW IF NOT EXISTS stock_data.v_stock_names_history AS
SELECT 
    code,
    code_name as name,
    min(day) as start_date,
    max(day) as end_date
FROM stock_data.all_stock
GROUP BY code, code_name
ORDER BY code, start_date;


CREATE TABLE stock_data.finicial_data
(   `date` Date,
    `code` String,
    `industry` String,
    `industryClassification` String,
    `total_shares` Float64,
) ENGINE = ReplacingMergeTree() 
ORDER BY (code, date)


CREATE TABLE stock_data.index_stocks
(   
    `index` String,
    `code` String,
    `enter_date` Date,
) ENGINE = ReplacingMergeTree() 
ORDER BY (index, code)


CREATE TABLE stock_data.finicial_report
(   
    `report_date` Date,
    `code` String,
    `publish_date` Date,
    `net_profit` Float64,
    `adjusted_profit` Float64,
    `total_operating_revenue` Float64,
    `subtotal_operate_cash_inflow` Float64,
    `roe` Float64,
    `roa` Float64,
    `inc_net_profit_year_on_year` Float64,
    `total_shares` Float64,
    `circulating_a` Float64,
    `nav_per_share` Float64,
    `market_cap` Float64,
    `circulating_market_cap` Float64,
    `pe_ratio` Float64,
    `pb_ratio` Float64,
    `adjusted_profit_diff` Float64,
) ENGINE = ReplacingMergeTree() 
ORDER BY (report_date, code)



CREATE TABLE stock_data.industry_info
(   
    `code` String,
    `enter_date` Date,
    `industry_code` String,
    `industry_name` String,

) ENGINE = ReplacingMergeTree() 
ORDER BY (code,enter_date)

CREATE TABLE stock_data.etf_daily
(   `date` Date,
    `code` String,
    `open` Float64,
    `high` Float64,
    `low` Float64,
    `close` Float64,
    `volume` UInt64,
    `amount` Float64,
    `turn` Float64,
) ENGINE = ReplacingMergeTree() 
ORDER BY (code, date)

-- 若表已存在，添加 nav_per_share 列（用于单季度 ROE/ROA 计算）:
-- ALTER TABLE stock_data.finicial_report ADD COLUMN IF NOT EXISTS nav_per_share Float64 DEFAULT 0 AFTER circulating_a;

-- OPTIMIZE TABLE stock_data.finicial_report FINAL;


CREATE TABLE stock_data.shares_info
(   `publish_date` Date,
    `change_date` Date,
    `code` String,
    `total_shares` Float64,
    `circulating_a` Float64,
) ENGINE = ReplacingMergeTree() 
ORDER BY (publish_date, code)


-- BACKUP DATABASE stock_data TO Disk('backups', '1.zip')

-- RESTORE DATABASE stock_data FROM Disk('backups', '1.zip')


CREATE DATABASE IF NOT EXISTS crypto_data;

CREATE TABLE IF NOT EXISTS crypto_data.minute_bars
(
    `provider` LowCardinality(String),
    `market_type` LowCardinality(String),
    `symbol` LowCardinality(String),
    `exchange_symbol` LowCardinality(String),
    `interval` LowCardinality(String),
    `open_time` DateTime64(3, 'UTC'),
    `close_time` DateTime64(3, 'UTC'),
    `open` Float64,
    `high` Float64,
    `low` Float64,
    `close` Float64,
    `volume_base` Float64,
    `volume_quote` Float64,
    `trade_count` UInt32,
    `ingest_source` LowCardinality(String),
    `ingested_at` DateTime64(3, 'UTC')
) ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMM(open_time)
ORDER BY (provider, market_type, symbol, interval, open_time);

CREATE TABLE IF NOT EXISTS crypto_data.instruments
(
    `provider` LowCardinality(String),
    `market_type` LowCardinality(String),
    `symbol` LowCardinality(String),
    `exchange_symbol` LowCardinality(String),
    `base_asset` LowCardinality(String),
    `quote_asset` LowCardinality(String),
    `is_active` UInt8,
    `updated_at` DateTime64(3, 'UTC')
) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (provider, market_type, symbol);
