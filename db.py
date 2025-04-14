import numpy as np
import pandas as pd
import clickhouse_connect

from loguru import logger
import talib as ta


class DB:
    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host="localhost", username="default", password=""
        )

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

        logger.debug(f"exec query : {query}")

        data = self.client.query(query)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)

        df.rename(columns={"date": "datetime"}, inplace=True)
        df.set_index("datetime", inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    def update_meta(self):
        df = pd.read_csv("all_stock.csv", index_col="code")

        for code, data in df.iterrows():
            # pass
            name = data["code_name"]
            update_query = f"""
            INSERT INTO stock_data.stock_daily_meta (code, last_update_date, last_adjfactor, error_update_count, name)
            SELECT 
                code,
                last_update_date,
                last_adjfactor,
                error_update_count,
                '{name}'
            FROM stock_data.stock_daily_meta
            WHERE code = '{code}';
            """
            self.client.command(update_query)
            print(code)

    def get_meta(self, code):
        query = f"""
        SELECT *
        FROM stock_data.stock_daily_meta
        WHERE code = '{code}'
        """

        logger.debug(f"exec query : {query}")
        data = self.client.query(query)
        assert len(data.result_rows) == 1
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        return df

    def opt_table(self, table_name):
        query = f"""
        OPTIMIZE TABLE {table_name} FINAL;
        """
        self.client.command(query)

    # def get_stock_industry(code, date=None):
    #     # 查找距离date最新的行业分类
    #     query = f"""
    #     SELECT *
    #     FROM stock_data.finicial_data
    #     WHERE code = '{code}'
    #     """
    #     if date:
    #         query += f" AND date <= '{date}'"
    #     query += " ORDER BY date DESC LIMIT 1"
    #     logger.debug(f"exec query : {query}")
    #     data = self.client.query(query)
    #     assert len(data.result_rows) == 1
    #     return data.result_rows[0][2]

    def get_price(self, stocks, end_date, fields, count, price_adj=True, start_date=None):
        """
        查询多个股票的指定字段数据
        :param stocks: 股票代码列表
        :param end_date: 结束日期
        :param fields: 需要查询的字段列表
        :param count: 查询的数据条数
        :return: 多维索引DataFrame (股票代码, 日期)
        """
        if not isinstance(stocks, list):
            stocks = [stocks]

        if not isinstance(fields, list):
            fields = [fields]

        if len(stocks) == 0 or len(fields) == 0:
            return pd.DataFrame()

        fields_str = ", ".join(fields)
        stocks_str = ", ".join([f"'{code}'" for code in stocks])

        query = f"""
        SELECT * FROM (
            SELECT 
                code, 
                date, 
                {fields_str},
                adjfactor,
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

        logger.debug(f"exec query: {query}")
        data = self.client.query(query)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)

        if len(df) == 0:
            return df

        if price_adj:
            # 计算复权价
            for field in set(fields) & set(["close", "open", "high", "low"]):
                df[field] = df[field] * df["adjfactor"]
        # 转换为多层索引DataFrame
        df["date"] = pd.to_datetime(df["date"])
        df.set_index(["code", "date"], inplace=True)

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

        logger.debug(f"exec query: {query}")
        data = self.client.query(query)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        df.set_index("code", inplace=True)
        return df

    def get_index_stocks(self, index_code, date=None):
        if not isinstance(index_code, list):
            index_code = [index_code]

        index_code_str = ", ".join([f"'{code}'" for code in index_code])

        sql = f"""
        SELECT *
        FROM stock_data.index_stocks
        WHERE index in ({index_code_str})
        """
        # if date:
        #     sql += f" AND enter_date <= '{date}'"
        logger.debug(f"exec query: {sql}")
        data = self.client.query(sql)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        return df["code"].tolist()

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
        logger.debug(f"exec query: {query}")
        data = self.client.query(query)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        df.set_index("code", inplace=True)
        return df

    def get_stock_industry_sw(self, stocks, date=None):
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
        logger.debug(f"{sql}")
        data = self.client.query(sql)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        df.set_index(keys="code",inplace=True)
        return df
    
    def get_stock_shares_info(self,stocks,date=None):
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
        logger.debug(f"{sql}")
        data = self.client.query(sql)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        df.set_index(keys="code",inplace=True)

        return df



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
    # db_client.get_stock_industry_sw(["sz.002193"], date="2022-01-01")
    print(db_client.get_stock_shares_info(["sz.002166","sz.002193"], date="2022-01-01"))
    # print(db_client.get_stock_fincial(["sz.002193"], "20220301")['adjusted_profit'].iloc[0])
    # print(get_index_stocks("399101","20240101"))
