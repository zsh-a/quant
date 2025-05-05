import akshare as ak

import pandas as pd
import clickhouse_connect
from loguru import logger
import os, sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 将外层目录添加到 sys.path
sys.path.append(parent_dir)
from config_manager import cm
from db import DB


from utils.utils import get_sw_comoment


class AKDataProcessor:
    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host=cm.get("db.host"),
            username=cm.get("db.username"),
            password=cm.get("db.password"),
        )

    # def __del__(self):
    #     self.client.command("OPTIMIZE TABLE stock_data.index_stocks FINAL")
    #     self.client.command("OPTIMIZE TABLE stock_data.industry_info FINAL")

    def insert_index_stocks(self, index_code):
        logger.info(f"fetch index stocks : {index_code}")
        df = ak.index_stock_cons_csindex(symbol=index_code)
        logger.info(f"insert index stocks : {index_code}")
        for index, row in df.iterrows():
            code = row["成分券代码"]
            if code.startswith("6"):
                code = "sh." + code
            else:
                code = "sz." + code

            sql = f"""
            INSERT INTO stock_data.index_stocks (index,code,enter_date)
            VALUES ('{index_code}','{code}','{row["日期"]}')
            """
            self.client.command(sql)

    def insert_sw_index(self):
        df = pd.read_csv("sw_industry.csv", index_col="代码")
        print(df)

        for index, row in df.iterrows():
            index = index.split(".")[0]
            index_component_df = ak.index_component_sw(symbol=f"{index}")

            print(index_component_df)
            for idx, stock in index_component_df.iterrows():
                code = stock["证券代码"]

                if code.startswith("6"):
                    code = "sh." + code
                else:
                    code = "sz." + code
                sql = f"""
                INSERT INTO stock_data.index_stocks (index,code,enter_date)
                VALUES ('{index}','{code}','{stock["计入日期"]}')
                """
                self.client.command(sql)

    # def insert_sw_industry(self):
    #     df = pd.read_csv("sw_industry.csv", index_col="index")
    #     for index, row in df.iterrows():
    #         index = index.split(".")[0]
    #         index_component_df = ak.index_component_sw(symbol=f"{index}")
    #         logger.info(f"process industry : {index} {row['name']}")
    #         for idx, stock in index_component_df.iterrows():
    #             code = stock["证券代码"]

    #             if code.startswith("6"):
    #                 code = "sh." + code
    #             else:
    #                 code = "sz." + code
    #             sql = f"""
    #             INSERT INTO stock_data.industry_info (code,enter_date,industry_code,industry_name)
    #             VALUES ('{code}','{stock["计入日期"]}','{index}','{row["name"]}')
    #             """
    #             self.client.command(sql)

    def insert_sw_industry(self):
        df = pd.read_csv("sw_industry.csv", index_col="index")
        for index, row in df.iterrows():
            index = index.split(".")[0]
            index_component_df = get_sw_comoment(index)
            logger.info(f"process industry : {index} {row['name']}")
            for idx, stock in index_component_df.iterrows():
                code = stock["股票代码"].split(".")[0]

                if code.startswith("6"):
                    code = "sh." + code
                else:
                    code = "sz." + code
                sql = f"""
                INSERT INTO stock_data.industry_info (code,enter_date,industry_code,industry_name)
                VALUES ('{code}','{stock["纳入时间"]}','{index}','{row["name"]}')
                """
                self.client.command(sql)

    def update_shares(self, start_date="20100101", end_date="20260101"):
        logger.info("update shares index : 399101")
        db_client = DB()
        stocks = db_client.get_index_stocks("399101")
        for code in stocks:
            try:
                stock_share_change_cninfo_df = ak.stock_share_change_cninfo(
                    symbol=code.split(".")[1], start_date=start_date, end_date=end_date
                )
            except Exception as e:
                logger.error(f"fetch {code} shares {start_date} {end_date} error : {e}")
                continue

            for index, row in stock_share_change_cninfo_df.iterrows():
                sql = f"""
                INSERT INTO stock_data.shares_info (publish_date,change_date,code,total_shares,circulating_a)
                VALUES ('{row["公告日期"]}','{row["变动日期"]}','{code}',{float(row["总股本"]) * 10000},{float(row["已流通股份"]) * 10000})
                """

                self.client.command(sql)

    def update_etf_data(self, code):
        logger.info(f"update etf data : {code}")
        df = ak.fund_etf_hist_em(
            symbol=code,
            period="daily",
            start_date="20100101",
            end_date="20260201",
            adjust="hfq",
        )
        df = df.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "换手率": "turn",
            }
        )
        df["date"] = pd.to_datetime(df["date"])  # 确保 timestamp 是 datetime 类型
        # df.set_index("date", inplace=True)
        df["tradestatus"] = 1.0
        df["adjfactor"] = 1.0

        df["code"] = f"{code}"
        df = df.astype(
            {
                "open": float,
                "high": float,
                "code": str,
                "low": float,
                "close": float,
                "volume": float,
                "amount": float,
                "turn": float,
                "tradestatus": float,
            }
        )
        df = df[
            [
                "date",
                "code",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "turn",
                "tradestatus",
                "adjfactor",
            ]
        ]
        self.client.insert(
            table="stock_data.stock_daily",
            column_names=[
                "date",
                "code",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "turn",
                "tradestatus",
                "adjfactor",
            ],
            data=df,
        )
        self.client.command("OPTIMIZE TABLE stock_data.stock_daily FINAL")

    def create_etf_meta(self):
        # 查询最新的 K 线数据

        all_etfs = pd.read_csv("all_etf.csv", names=["code", "type", "name"], dtype=str)
        all_etfs = all_etfs.set_index("code")

        code_str = ",".join([f"'{code}'" for code in all_etfs.index])

        sql = f"""
            SELECT
        code,
        date,
        adjfactor
    FROM (
        SELECT
            code,
            date,
            adjfactor,
            ROW_NUMBER() OVER (PARTITION BY code ORDER BY date DESC) AS rn 
        FROM stock_data.stock_daily
        WHERE code in ({code_str})
    ) AS t
    WHERE rn = 1
    """

        kline_data = self.client.query(sql)

        all_etfs = pd.read_csv("all_etf.csv", names=["code", "type", "name"], dtype=str)
        all_etfs = all_etfs.set_index("code")

        df = kline_data.result_rows
        # 构建更新语句
        for code, last_update_date, adjfactor in df:
            try:
                # 插入或更新数据
                update_query = f"""
                INSERT INTO stock_data.stock_daily_meta (code,name, last_update_date, last_adjfactor, error_update_count)
                VALUES ('{code}','{all_etfs.loc[code]["name"]}', '{last_update_date}', '{adjfactor}', 0)
                """
                self.client.command(update_query)
            except Exception as e:
                # 处理失败的情况，更新 error_update_count
                # error_query = f"""
                # INSERT INTO stock_data.stock_daily_meta (code, last_update_date, last_adjfactor, error_update_count)
                # VALUES ('{code}','{all_etfs.loc[code]['name']}', '{last_update_date}', '{adjfactor}',error_update_count + 1)
                # """
                # self.client.command(error_query)
                print(f"Error updating code {code}: {e}")
        self.client.command("OPTIMIZE TABLE stock_data.stock_daily_meta FINAL")


if __name__ == "__main__":
    # insert_index_stocks("399101")
    dp = AKDataProcessor()
    # etfs = ["511260","518880","513100","159980","162411","159985"]
    # all_etfs = pd.read_csv("all_etf.csv", names=["基金代码", "类别", "名称"])
    # all_etfs = all_etfs["基金代码"].astype(str).to_list()
    # for code in all_etfs:
    #     dp.update_etf_data(code)
    # dp.create_etf_meta()
    # dp.update_shares()
    dp.insert_index_stocks("000852")
    # print(ak.index_stock_cons_csindex(symbol="000852"))
    # dp.insert_sw_industry()

    # print(ak.stock_zh_a_hist())
