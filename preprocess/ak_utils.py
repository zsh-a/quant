import akshare as ak
from db import DB
import pandas as pd
import clickhouse_connect
from config_manager import cm
from loguru import logger


class AKDataProcessor:
    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host=cm.get("db.host"),
            username=cm.get("db.username"),
            password=cm.get("db.password"),
        )

    def __del__(self):
        db = DB()

        db.opt_table("stock_data.index_stocks")
        db.opt_table("stock_data.industry_info")

    def insert_index_stocks(self, index_code):
        df = ak.index_stock_cons(symbol=index_code)
        for index, row in df.iterrows():
            code = row["品种代码"]
            name = row["品种名称"]
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

    def insert_sw_industry(self):
        df = pd.read_csv("sw_industry.csv", index_col="index")
        for index, row in df.iterrows():
            index = index.split(".")[0]
            index_component_df = ak.index_component_sw(symbol=f"{index}")
            logger.info(f"process industry : {index} {row['name']}")
            for idx, stock in index_component_df.iterrows():
                code = stock["证券代码"]

                if code.startswith("6"):
                    code = "sh." + code
                else:
                    code = "sz." + code
                sql = f"""
                INSERT INTO stock_data.industry_info (code,enter_date,industry_code,industry_name)
                VALUES ('{code}','{stock["计入日期"]}','{index}','{row["name"]}')
                """
                self.client.command(sql)


if __name__ == "__main__":
    # insert_index_stocks("399101")
    dp = AKDataProcessor()
    dp.insert_index_stocks("399101")
