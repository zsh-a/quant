import numpy as np
import pandas as pd
import clickhouse_connect

from loguru import logger
# import talib as ta


class DB:
    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host="localhost", username="default", password=""
        )
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

        # logger.debug(f"exec query : {query}")

        data = self.client.query(query)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        try:
            df.rename(columns={"date": "datetime"}, inplace=True)
            df.set_index("datetime", inplace=True)
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            print(e)
        return df

    def update_meta(self):
        df = pd.read_csv("all_stock.csv", index_col="code")

        for code, data in df.iterrows():
            # pass
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

        # logger.debug(f"exec query : {query}")
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
        """
        查询多个股票的指定字段数据
        :param stocks: 股票代码列表
        :param end_date: 结束日期
        :param fields: 需要查询的字段列表
        :param count: 查询的数据条数
        :param price_adj: 是否进行复权调整
        :param start_date: 开始日期
        :return: 多维索引DataFrame (股票代码, 日期)
        """
        if not isinstance(stocks, list):
            stocks = [stocks]
        
        if not isinstance(fields, list):
            fields = [fields]

        # 缓存键：股票列表(排序)、结束日期、字段、条数
        # 注意：这里仅针对 count 较小的查询进行简单缓存
        cache_key = f"get_price_{hash(tuple(sorted(stocks)))}_{end_date}_{hash(tuple(sorted(fields)))}_{count}_{price_adj}_{start_date}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        if len(stocks) == 0 or len(fields) == 0:
            return pd.DataFrame()

        # 确保查询字段中包含adjfactor
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
            FROM stock_data.stock_daily
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

        # logger.debug(f"exec query: {query}")
        data = self.client.query(query)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)

        if len(df) == 0:
            # 即使为空也尝试建立索引结构以保证调用方逻辑一致
            if "code" in df.columns and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index(["code", "date"], inplace=True)
            return df

        # 转换为多层索引DataFrame
        df["date"] = pd.to_datetime(df["date"])

        # 检查并删除重复索引
        if len(df) != len(df.drop_duplicates(subset=["code", "date"])):
            logger.warning("发现重复索引，将保留第一个出现的值")
            df = df.drop_duplicates(subset=["code", "date"], keep="first")

        df.set_index(["code", "date"], inplace=True)

        if price_adj and "adjfactor" in df.columns:
            # 计算复权价
            price_fields = set(["close", "open", "high", "low"]) & set(df.columns)
            for field in price_fields:
                df[field] = df[field] * df["adjfactor"]

        self._cache[cache_key] = df
        return df

    def get_stock_industry(self, stocks, date=None):
        """
        查询多个股票的行业分类
        :param stocks: 股票代码列表
        :param date: 查询日期(可选)
        :return: 包含股票代码和行业分类的DataFrame
        """
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

        # logger.debug(f"exec query: {query}")
        data = self.client.query(query)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        df.set_index("code", inplace=True)
        return df

    def get_index_stocks(self, index_code, date=None):
        if not isinstance(index_code, list):
            index_code = [index_code]

        # 缓存逻辑：按索引代码列表和大致日期区间（或仅索引代码，如果enter_date未启用）
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
        # 仅获取距离date最近的一条数据
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
        # logger.debug(f"exec query: {query}")
        data = self.client.query(query)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        df.set_index("code", inplace=True)
        return df

    def get_stock_industry_sw(self, stocks, date=None):
        # 缓存逻辑：由于行业数据相对稳定，且回测中会频繁查询
        # 这里使用大致的缓存策略，或者如果是全量查询则缓存
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
        # logger.debug(f"{sql}")
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
        # logger.debug(f"{sql}")
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


if __name__ == "__main__":
    # df = get_kline("sz.300059", "20220101", "20221231")

    # print(df)
    # print(get_meta("sz.000001"))
    # update_meta()
    # opt_table("stock_data.stock_daily_meta")
    # print(get_stock_industry("sz.000001", "20221231"))
    # print(get_zz500_stocks())
    # print(get_price(['sh.000001'],"20200101",['close',"open"],10))
    # print(get_stock_industry('sh.601228', "20210101"))
    db_client = DB()
    db_client.update_meta()
    # db_client.get_stock_industry_sw(["sz.002193"], date="2022-01-01")
    # print(db_client.get_stock_shares_info(["sz.002166","sz.002193"], date="2022-01-01"))
    # print(db_client.get_stock_fincial(["sz.002193"], "20220301")['adjusted_profit'].iloc[0])
    # print(len(db_client.get_index_stocks("000852","20240101")))
