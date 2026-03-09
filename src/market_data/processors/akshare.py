import akshare as ak
import pandas as pd

from loguru import logger

from src.market_data.clickhouse import create_clickhouse_client
from src.market_data.db import DB
from utils.utils import get_sw_comoment


class AKDataProcessor:
    def __init__(self):
        self.client = create_clickhouse_client()

    def insert_index_stocks(self, index_code):
        logger.info(f"fetch index stocks : {index_code}")
        try:
            df = ak.index_stock_cons_csindex(symbol=index_code)
            logger.info(f"insert index stocks : {index_code}")
            for _, row in df.iterrows():
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
        except Exception:
            df = ak.index_stock_cons(symbol=index_code)
            for _, row in df.iterrows():
                code = row["品种代码"]
                if code.startswith("6"):
                    code = "sh." + code
                else:
                    code = "sz." + code
                sql = f"""
                INSERT INTO stock_data.index_stocks (index,code,enter_date)
                VALUES ('{index_code}','{code}','{row["纳入日期"]}')
                """
                self.client.command(sql)

    def insert_sw_index(self):
        df = pd.read_csv("sw_industry.csv", index_col="代码")

        for index, _ in df.iterrows():
            index = index.split(".")[0]
            index_component_df = ak.index_component_sw(symbol=f"{index}")

            for _, stock in index_component_df.iterrows():
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

    def insert_sw_industry(self):
        df = pd.read_csv("sw_industry.csv", index_col="index")
        for index, row in df.iterrows():
            index = index.split(".")[0]
            index_component_df = get_sw_comoment(index)
            logger.info(f"process industry : {index} {row['name']}")
            for _, stock in index_component_df.iterrows():
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

    def update_shares(self, start_date="20100101", end_date="20270101"):
        logger.info("update shares index : 399101")
        db_client = DB()
        stocks = db_client.get_index_stocks("399101")
        for code in stocks:
            try:
                stock_share_change_cninfo_df = ak.stock_share_change_cninfo(
                    symbol=code.split(".")[1], start_date=start_date, end_date=end_date
                )
            except Exception as exc:
                logger.error(f"fetch {code} shares {start_date} {end_date} error : {exc}")
                continue

            for _, row in stock_share_change_cninfo_df.iterrows():
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
            end_date="20270201",
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
        df["date"] = pd.to_datetime(df["date"])
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
        for code, last_update_date, adjfactor in df:
            try:
                update_query = f"""
                INSERT INTO stock_data.stock_daily_meta (code,name, last_update_date, last_adjfactor, error_update_count)
                VALUES ('{code}','{all_etfs.loc[code]["name"]}', '{last_update_date}', '{adjfactor}', 0)
                """
                self.client.command(update_query)
            except Exception as exc:
                print(f"Error updating code {code}: {exc}")
        self.client.command("OPTIMIZE TABLE stock_data.stock_daily_meta FINAL")
