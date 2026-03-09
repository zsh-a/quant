import pandas as pd

from loguru import logger

from src.market_data.clickhouse import create_clickhouse_client


class DB:
    def __init__(self):
        self.client = create_clickhouse_client()
        self._cache = {}

    def get_kline(self, code, start_date, end_date):
        query = f"""
        SELECT *
        FROM stock_data.stock_daily
        WHERE code = '{code}'"""
        if start_date:
            query += f" AND date >= '{start_date}'"
        if end_date:
            query += f" AND date <= '{end_date}'"

        query += "ORDER BY date"

        data = self.client.query(query)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        try:
            df.rename(columns={"date": "datetime"}, inplace=True)
            df.set_index("datetime", inplace=True)
            df.index = pd.to_datetime(df.index)
        except Exception as exc:
            print(exc)
        return df

    def update_meta(self):
        df = pd.read_csv("all_stock.csv", index_col="code")

        for code, data in df.iterrows():
            name = data["code_name"]
            if pd.isna(name):
                continue

            update_query = f"""
            INSERT INTO stock_data.stock_daily_meta (code, last_update_date, last_adjfactor, error_update_count, name)
            SELECT
                code,
                last_update_date,
                last_adjfactor,
                error_update_count,
                '{name}'
            FROM stock_data.stock_daily_meta
            WHERE code = '{code}'
            ORDER BY last_update_date DESC
            LIMIT 1;
            """
            self.client.command(update_query)

    def get_meta(self, code):
        query = f"""
        SELECT *
        FROM stock_data.stock_daily_meta
        WHERE code = '{code}'
        """

        data = self.client.query(query)
        assert len(data.result_rows) == 1
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        return df

    def opt_table(self, table_name):
        query = f"""
        OPTIMIZE TABLE {table_name} FINAL;
        """
        self.client.command(query)

    def get_price(
        self, stocks, end_date, fields, count, price_adj=True, start_date=None
    ):
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

        fields_str = ", ".join(db_fields)
        stocks_str = ", ".join([f"'{code}'" for code in stocks])

        query = f"""
        SELECT * FROM (
            SELECT
                code,
                date,
                {fields_str},
                tradestatus,
                ROW_NUMBER() OVER(PARTITION BY code ORDER BY date DESC) AS rn
            FROM stock_data.stock_daily FINAL
            WHERE code IN ({stocks_str})
            AND date <= '{end_date}'
            """
        if start_date:
            query += f" AND date >= '{start_date}'"
        query += f"""
        ) t
        WHERE rn <= {count}
        ORDER BY code, date
        """

        data = self.client.query(query)
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

    def get_stock_industry(self, stocks, date=None):
        if not isinstance(stocks, list):
            stocks = [stocks]

        stocks_str = ", ".join([f"'{code}'" for code in stocks])

        query = f"""
        SELECT code, industry
        FROM (
            SELECT
                code,
                industry,
                ROW_NUMBER() OVER(PARTITION BY code ORDER BY date DESC) AS rn
            FROM stock_data.finicial_data
            WHERE code IN ({stocks_str})"""

        if date:
            query += f" AND date <= '{date}'"

        query += """
        ) t
        WHERE rn = 1
        ORDER BY code
        """

        data = self.client.query(query)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        df.set_index("code", inplace=True)
        return df

    def get_index_stocks(self, index_code, date=None):
        if not isinstance(index_code, list):
            index_code = [index_code]

        cache_key = f"index_stocks_{'_'.join(sorted(index_code))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        index_code_str = ", ".join([f"'{code}'" for code in index_code])

        sql = f"""
        SELECT code
        FROM stock_data.index_stocks
        WHERE index in ({index_code_str})
        """
        data = self.client.query(sql)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        if df.empty or "code" not in df.columns:
            return []
        res = df["code"].tolist()
        self._cache[cache_key] = res
        return res

    def get_stock_fincial(self, stocks, fields, date=None):
        if not isinstance(stocks, list):
            stocks = [stocks]

        if not isinstance(fields, list):
            fields = [fields]
        fields_str = ", ".join(fields)
        stocks_str = ", ".join([f"'{code}'" for code in stocks])
        query = f"""
        SELECT * FROM (
            SELECT
                code,{fields_str},
                ROW_NUMBER() OVER(PARTITION BY code ORDER BY (publish_date,report_date) DESC) AS rn
            FROM stock_data.finicial_report
            WHERE code IN ({stocks_str})
            AND circulating_a > 0
            """
        if date:
            query += f" AND publish_date <= '{date}'"
        query += """
        ) t
        WHERE rn = 1
        ORDER BY code
        """
        data = self.client.query(query)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        df.set_index("code", inplace=True)
        return df

    def get_stock_industry_sw(self, stocks, date=None):
        cache_key = f"industry_sw_{len(stocks)}_{stocks[0] if stocks else ''}_{date}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        stocks_str = ", ".join([f"'{code}'" for code in stocks])

        sql = f"""
        SELECT * FROM (
            SELECT
                *,
                ROW_NUMBER() OVER(PARTITION BY code ORDER BY enter_date DESC) AS rn
            FROM stock_data.industry_info
            WHERE code IN ({stocks_str})
        """
        if date:
            sql += f" AND enter_date <= '{date}'"
        sql += """
        ) t
        WHERE rn = 1
        """
        data = self.client.query(sql)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        df.set_index(keys="code", inplace=True)

        self._cache[cache_key] = df
        return df

    def get_stock_shares_info(self, stocks, date=None):
        stocks_str = ", ".join([f"'{code}'" for code in stocks])

        sql = f"""
        SELECT * FROM (
            SELECT
                *,
                ROW_NUMBER() OVER(PARTITION BY code ORDER BY change_date DESC) AS rn
            FROM stock_data.shares_info
            WHERE code IN ({stocks_str})
        """
        if date:
            sql += f" AND change_date <= '{date}'"
        sql += """
        ) t
        WHERE rn = 1
        """
        data = self.client.query(sql)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        df.set_index(keys="code", inplace=True)

        return df

    def get_all_etf_code(self):
        all_etfs = pd.read_csv("all_etf.csv", names=["基金代码", "类别", "名称"])
        all_etfs = all_etfs["基金代码"].astype(str).to_list()
        return all_etfs

    def get_all_stock_code(self):
        sql = """
        SELECT code FROM stock_data.stock_daily_meta
        """
        data = self.client.query(sql)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        stocks = df["code"].tolist()
        return [
            stock
            for stock in stocks
            if stock.startswith("sz.00")
            or stock.startswith("sh.60")
            or stock.startswith("sz.30")
        ]

    def get_swindustry_stocks(self, industry_code, date=None):
        sql = f"""
        SELECT code FROM stock_data.industry_info
        WHERE industry_code = '{industry_code}'
        """
        if date:
            sql += f" AND enter_date <= '{date}'"
        data = self.client.query(sql)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        return df["code"].tolist()
