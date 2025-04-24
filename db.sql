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
    `roa` Float64,
    `total_shares` Float64,
    `circulating_a` Float64,
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