CREATE TABLE stock_data.stock_daily
(   `date` Date,
    `code` String,
    `open` Float64,
    `high` Float64,
    `low` Float64,
    `close` Float64,
    `preclose` Float64,
    `volume` UInt64,
    `amount` Float64,
    `turn` Float64,
    `pctChg` Float64,
    `peTTM` Float64,
    `pbMRQ` Float64,
    `tradestatus` Int16,
    `isST` Int16,
    `adjfactor` Float64,
) ENGINE = ReplacingMergeTree() 
ORDER BY (code, date)

-- OPTIMIZE TABLE stock_data.stock_daily FINAL;

CREATE TABLE stock_data.stock_daily_meta
(  `code` String,
   `name` String,
   `last_update_date` Date,
   `last_adjfactor` Float64,
   `error_update_count` UInt32
) ENGINE = ReplacingMergeTree()
ORDER BY (code)

-- OPTIMIZE TABLE stock_data.stock_daily_meta FINAL;


-- CREATE TABLE stock_daily_data (
--     stock_code String,
--     trade_date Date,
--     open Float64,
--     high Float64,
--     low Float64,
--     close Float64,
--     volume UInt64,
--     turnover Float64
-- ) 
-- ENGINE = MergeTree
-- ORDER BY (stock_code, trade_date)
-- PARTITION BY toYYYYMM(trade_date)
-- SETTINGS index_granularity = 8192;

