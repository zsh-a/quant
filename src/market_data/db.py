import re

import pandas as pd
from loguru import logger

from src.market_data.clickhouse import create_clickhouse_client

# ---------------------------------------------------------------------------
# Column / table allowlists — prevent injection via dynamic identifiers
# ---------------------------------------------------------------------------
_VALID_STOCK_COLUMNS = frozenset(
    {
        "code",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "adjfactor",
        "turn",
        "tradestatus",
        "pctChg",
        "isST",
        "peTTM",
        "pbMRQ",
        "psTTM",
        "pcfNcfTTM",
    }
)

_VALID_FINANCIAL_COLUMNS = frozenset(
    {
        "code",
        "publish_date",
        "report_date",
        "circulating_a",
        "total_shares",
        "total_share",
        "revenue",
        "net_profit",
        "roe",
        "roa",
        "adjusted_profit",
        "adjusted_profit_diff",
        "total_operating_revenue",
        "subtotal_operate_cash_inflow",
        "inc_net_profit_year_on_year",
        "nav_per_share",
        "eps",
        "market_cap",
        "circulating_market_cap",
        "pe_ratio",
        "pb_ratio",
        "gross_profit_margin",
        "net_profit_margin",
        "bps",
        "operating_cash_flow",
        "total_assets",
        "total_liabilities",
        "equity",
        "debt_to_assets",
        "current_ratio",
        "quick_ratio",
    }
)

_VALID_TABLES = frozenset(
    {
        "stock_data.stock_daily",
        "stock_data.stock_daily_meta",
        "stock_data.all_stock",
        "stock_data.finicial_data",
        "stock_data.finicial_report",
        "stock_data.industry_info",
        "stock_data.shares_info",
        "stock_data.index_stocks",
    }
)

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")


def _validate_identifier(name: str) -> str:
    """Raise if *name* is not a safe SQL identifier."""
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"非法标识符: {name!r}")
    return name


def _validate_columns(fields: list[str], allowed: frozenset[str]) -> list[str]:
    """Return *fields* after checking every element is in *allowed*."""
    bad = set(fields) - allowed
    if bad:
        raise ValueError(f"不允许的列名: {bad}")
    return fields


def _validate_table(table: str) -> str:
    if table not in _VALID_TABLES:
        raise ValueError(f"不允许的表名: {table!r}")
    return table


class DB:
    def __init__(self):
        self.client = create_clickhouse_client()
        self._cache = {}

    # ------------------------------------------------------------------
    # K-line data
    # ------------------------------------------------------------------
    def get_kline(self, code, start_date, end_date):
        params = {"code": code}
        query = "SELECT * FROM stock_data.stock_daily WHERE code = {code:String}"
        if start_date:
            query += " AND date >= {start_date:String}"
            params["start_date"] = start_date
        if end_date:
            query += " AND date <= {end_date:String}"
            params["end_date"] = end_date
        query += " ORDER BY date"

        data = self.client.query(query, parameters=params)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        try:
            df.rename(columns={"date": "datetime"}, inplace=True)
            df.set_index("datetime", inplace=True)
            df.index = pd.to_datetime(df.index)
        except Exception as exc:
            logger.warning("get_kline 索引处理异常: {}", exc)
        return df

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------
    def update_meta(self):
        """Update stock names in meta table from all_stock table in DB."""
        data = self.client.query(
            "SELECT code, code_name FROM stock_data.all_stock WHERE code_name != '' ORDER BY day DESC LIMIT 1 BY code"
        )
        for code, name in data.result_rows:
            if not name:
                continue
            self.client.command(
                "INSERT INTO stock_data.stock_daily_meta "
                "    (code, last_update_date, last_adjfactor, error_update_count, name) "
                "SELECT code, last_update_date, last_adjfactor, error_update_count, "
                "    {name:String} "
                "FROM stock_data.stock_daily_meta "
                "WHERE code = {code:String} "
                "ORDER BY last_update_date DESC "
                "LIMIT 1",
                parameters={"name": name, "code": code},
            )

    def get_meta(self, code):
        data = self.client.query(
            "SELECT * FROM stock_data.stock_daily_meta WHERE code = {code:String}",
            parameters={"code": code},
        )
        assert len(data.result_rows) == 1
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        return df

    # ------------------------------------------------------------------
    # Table maintenance
    # ------------------------------------------------------------------
    def opt_table(self, table_name):
        _validate_table(table_name)
        self.client.command(f"OPTIMIZE TABLE {table_name} FINAL")

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------
    def get_price(self, stocks, end_date, fields, count, price_adj=True, start_date=None):
        if not isinstance(stocks, list):
            stocks = [stocks]
        if not isinstance(fields, list):
            fields = [fields]

        cache_key = (
            f"get_price_{hash(tuple(sorted(stocks)))}_{end_date}_"
            f"{hash(tuple(sorted(fields)))}_{count}_{price_adj}_{start_date}"
        )
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        if len(stocks) == 0 or len(fields) == 0:
            return pd.DataFrame()

        db_fields = fields.copy()
        if "adjfactor" not in db_fields and price_adj:
            db_fields.append("adjfactor")

        _validate_columns(db_fields, _VALID_STOCK_COLUMNS)
        fields_str = ", ".join(db_fields)

        params: dict = {
            "stocks": stocks,
            "end_date": end_date,
            "count": count,
        }

        query = (
            "SELECT * FROM ("
            "    SELECT"
            f"        code, date, {fields_str}, tradestatus,"
            "        ROW_NUMBER() OVER(PARTITION BY code ORDER BY date DESC) AS rn"
            "    FROM stock_data.stock_daily FINAL"
            "    WHERE code IN {stocks:Array(String)}"
            "    AND date <= {end_date:String}"
        )
        if start_date:
            query += " AND date >= {start_date:String}"
            params["start_date"] = start_date
        query += ") t WHERE rn <= {count:UInt32} ORDER BY code, date"

        data = self.client.query(query, parameters=params)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)

        if len(df) == 0:
            if "code" in df.columns and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index(["code", "date"], inplace=True)
            return df

        df["date"] = pd.to_datetime(df["date"])

        if len(df) != len(df.drop_duplicates(subset=["code", "date"])):
            logger.warning("发现重复索引，将保留第一个出现的值")
            df = df.drop_duplicates(subset=["code", "date"], keep="first")

        df.set_index(["code", "date"], inplace=True)

        if price_adj and "adjfactor" in df.columns:
            price_fields = set(["close", "open", "high", "low"]) & set(df.columns)
            for field in price_fields:
                df[field] = df[field] * df["adjfactor"]

        self._cache[cache_key] = df
        return df

    # ------------------------------------------------------------------
    # Industry
    # ------------------------------------------------------------------
    def get_stock_industry(self, stocks, date=None):
        if not isinstance(stocks, list):
            stocks = [stocks]

        params: dict = {"stocks": stocks}
        query = (
            "SELECT code, industry FROM ("
            "    SELECT"
            "        code, industry,"
            "        ROW_NUMBER() OVER(PARTITION BY code ORDER BY date DESC) AS rn"
            "    FROM stock_data.finicial_data"
            "    WHERE code IN {stocks:Array(String)}"
        )
        if date:
            query += " AND date <= {date:String}"
            params["date"] = date
        query += ") t WHERE rn = 1 ORDER BY code"

        data = self.client.query(query, parameters=params)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        df.set_index("code", inplace=True)
        return df

    # ------------------------------------------------------------------
    # Index stocks
    # ------------------------------------------------------------------
    def get_index_stocks(self, index_code, date=None):
        if not isinstance(index_code, list):
            index_code = [index_code]

        cache_key = f"index_stocks_{'_'.join(sorted(index_code))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        data = self.client.query(
            "SELECT code FROM stock_data.index_stocks WHERE index IN {codes:Array(String)}",
            parameters={"codes": index_code},
        )
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        if df.empty or "code" not in df.columns:
            return []
        res = df["code"].tolist()
        self._cache[cache_key] = res
        return res

    # ------------------------------------------------------------------
    # Financial data
    # ------------------------------------------------------------------
    def get_stock_fincial(self, stocks, fields, date=None):
        if not isinstance(stocks, list):
            stocks = [stocks]
        if not isinstance(fields, list):
            fields = [fields]

        _validate_columns(fields, _VALID_FINANCIAL_COLUMNS)
        fields_str = ", ".join(fields)

        params: dict = {"stocks": stocks}
        query = (
            "SELECT * FROM ("
            "    SELECT"
            f"        code, {fields_str},"
            "        ROW_NUMBER() OVER(PARTITION BY code ORDER BY (publish_date,report_date) DESC) AS rn"
            "    FROM stock_data.finicial_report"
            "    WHERE code IN {stocks:Array(String)}"
            "    AND circulating_a > 0"
        )
        if date:
            query += " AND publish_date <= {date:String}"
            params["date"] = date
        query += ") t WHERE rn = 1 ORDER BY code"

        data = self.client.query(query, parameters=params)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        df.set_index("code", inplace=True)
        return df

    # ------------------------------------------------------------------
    # SW industry
    # ------------------------------------------------------------------
    def get_stock_industry_sw(self, stocks, date=None):
        cache_key = f"industry_sw_{len(stocks)}_{stocks[0] if stocks else ''}_{date}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        params: dict = {"stocks": stocks}
        query = (
            "SELECT * FROM ("
            "    SELECT *, "
            "        ROW_NUMBER() OVER(PARTITION BY code ORDER BY enter_date DESC) AS rn"
            "    FROM stock_data.industry_info"
            "    WHERE code IN {stocks:Array(String)}"
        )
        if date:
            query += " AND enter_date <= {date:String}"
            params["date"] = date
        query += ") t WHERE rn = 1"

        data = self.client.query(query, parameters=params)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        df.set_index(keys="code", inplace=True)

        self._cache[cache_key] = df
        return df

    # ------------------------------------------------------------------
    # Shares info
    # ------------------------------------------------------------------
    def get_stock_shares_info(self, stocks, date=None):
        params: dict = {"stocks": stocks}
        query = (
            "SELECT * FROM ("
            "    SELECT *, "
            "        ROW_NUMBER() OVER(PARTITION BY code ORDER BY change_date DESC) AS rn"
            "    FROM stock_data.shares_info"
            "    WHERE code IN {stocks:Array(String)}"
        )
        if date:
            query += " AND change_date <= {date:String}"
            params["date"] = date
        query += ") t WHERE rn = 1"

        data = self.client.query(query, parameters=params)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        df.set_index(keys="code", inplace=True)
        return df

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_all_etf_code(self):
        from src.market_data.static_data import get_etf_codes

        return get_etf_codes()

    def get_all_stock_code(self):
        data = self.client.query("SELECT code FROM stock_data.stock_daily_meta")
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        stocks = df["code"].tolist()
        return [
            stock
            for stock in stocks
            if stock.startswith("sz.00") or stock.startswith("sh.60") or stock.startswith("sz.30")
        ]

    def get_trading_calendar(self):
        """获取交易日历 (返回 TradingCalendar 实例)。"""
        from src.core.trading_calendar import TradingCalendar

        return TradingCalendar(self)

    def get_swindustry_stocks(self, industry_code, date=None):
        params: dict = {"industry_code": industry_code}
        query = "SELECT code FROM stock_data.industry_info WHERE industry_code = {industry_code:String}"
        if date:
            query += " AND enter_date <= {date:String}"
            params["date"] = date

        data = self.client.query(query, parameters=params)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        return df["code"].tolist()
