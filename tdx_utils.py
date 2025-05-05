from mootdx.affair import Affair
from datetime import datetime
import pandas as pd
import clickhouse_connect
from config_manager import cm
from loguru import logger
import os
import hashlib
from db import DB


def convert_to_date(num):
    try:
        date_str = f"{int(num):06d}"  # 去除小数点并补零至6位（如250315）
        return datetime.strptime(date_str, "%y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return "无效日期"


class TDXProcess:
    table_name = "stock_data.finicial_report"

    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host=cm.get("db.host"),
            username=cm.get("db.username"),
            password=cm.get("db.password"),
        )
        self.fin_path = cm.get("fin_data.path")

    def __del__(self):
        db = DB()
        db.opt_table(TDXProcess.table_name)

    def fetch_tdx(self):
        local_hash = {}

        if os.path.exists(self.fin_path):
            files = os.listdir(self.fin_path)
            for file in files:
                file_path = os.path.join(self.fin_path, file)

                with open(file_path, "rb") as f:
                    local_md5 = hashlib.md5(f.read()).hexdigest()
                    local_hash[file] = local_md5

        updated_files = []
        files = Affair.files()
        for item in files:
            filename = item["filename"]
            md5 = item["hash"]
            if local_hash[filename] != md5:
                logger.info(f"update fin date from tdx : {filename}")
                Affair.fetch(downdir=self.fin_path, filename=filename)
                updated_files.append(filename)

        return updated_files

    def update_fincial_db(self, start_year=2024):
        files = self.fetch_tdx()
        for filename in files:
            df = Affair.parse(downdir=self.fin_path, filename=filename)

            df["report_date"] = df["report_date"].apply(
                lambda x: datetime.strptime(str(int(x)), "%Y%m%d").strftime("%Y-%m-%d")
            )
            df["publish_date"] = df["财报公告日期"].apply(convert_to_date)
            df["net_profit"] = df["五、净利润"]
            df["roa"] = df["净资产收益率"].iloc[:, [0]]  # 保留第一列
            df["adjusted_profit"] = df["扣除非经常性损益后的净利润"].iloc[:, [0]]
            df["total_shares"] = df["总股本"]
            df["circulating_a"] = df["已上市流通A股"]
            # df['circulating_b'] = df["已上市流通B股"]
            # df['circulating_h'] = df["已上市流通H股"]

            # 计算更多财务指标
            df["gross_profit_margin"] = df["销售毛利率(%)(非金融类指标)"]

            for code, row in df.iterrows():
                if (
                    not code.startswith("6")
                    and not code.startswith("0")
                    and not code.startswith("3")
                ):
                    continue
                if code.startswith("6"):
                    code = "sh." + code
                else:
                    code = "sz." + code
                adjusted_profit_diff = (
                    row["adjusted_profit"] if filename[8:12] == "0331" else 0
                )
                sql = f"""
                INSERT INTO stock_data.finicial_report
                (
                    report_date,
                    code,
                    publish_date,
                    net_profit,
                    adjusted_profit,
                    roa,
                    total_shares,
                    circulating_a,
                    adjusted_profit_diff
                )
                VALUES
                (
                    '{row["report_date"]}',
                    '{code}',
                    '{row["publish_date"]}',
                    {row["net_profit"]},
                    {row["adjusted_profit"]},
                    {row["roa"]},
                    {row["total_shares"]},
                    {row["circulating_a"]},
                    {adjusted_profit_diff}
                )"""
                logger.debug(f"{sql}")
                self.client.command(sql)
        self.update(start_year)

    def init_fincial_db(self, start_year="2024"):
        dates_def = ["0331", "0630", "0930", "1231"]
        now_year = datetime.now().year
        years = [str(year) for year in range(int(start_year), now_year + 1)]
        dates = [year + date for year in years for date in dates_def]

        for date in dates:
            filename = f"gpcw{date}.zip"

            try:
                df = Affair.parse(downdir=self.fin_path, filename=filename)
            except Exception as e:
                logger.error(f"parse {filename} error : {e}")
                continue
            if len(df) == 0:
                continue

            df["report_date"] = df["report_date"].apply(
                lambda x: datetime.strptime(str(int(x)), "%Y%m%d").strftime("%Y-%m-%d")
            )
            df["publish_date"] = df["财报公告日期"].apply(convert_to_date)
            df["net_profit"] = df["五、净利润"]
            df["roa"] = df["净资产收益率"].iloc[:, [0]]  # 保留第一列
            df["adjusted_profit"] = df["扣除非经常性损益后的净利润"].iloc[:, [0]]
            df["total_shares"] = df["总股本"]
            df["circulating_a"] = df["已上市流通A股"]
            # df['circulating_b'] = df["已上市流通B股"]
            # df['circulating_h'] = df["已上市流通H股"]

            # 计算更多财务指标
            df["gross_profit_margin"] = df["销售毛利率(%)(非金融类指标)"]

            for code, row in df.iterrows():
                if (
                    not code.startswith("6")
                    and not code.startswith("0")
                    and not code.startswith("3")
                ):
                    continue
                if code.startswith("6"):
                    code = "sh." + code
                else:
                    code = "sz." + code
                adjusted_profit_diff = (
                    row["adjusted_profit"] if date[4:] == "0331" else 0
                )
                sql = f"""
                INSERT INTO stock_data.finicial_report
                (
                    report_date,
                    code,
                    publish_date,
                    net_profit,
                    adjusted_profit,
                    roa,
                    total_shares,
                    circulating_a,
                    adjusted_profit_diff
                )
                VALUES
                (
                    '{row["report_date"]}',
                    '{code}',
                    '{row["publish_date"]}',
                    {row["net_profit"]},
                    {row["adjusted_profit"]},
                    {row["roa"]},
                    {row["total_shares"]},
                    {row["circulating_a"]},
                    {adjusted_profit_diff}
                )"""
                logger.debug(f"{sql}")
                self.client.command(sql)
            # except Exception as e:
            #     logger.error(f"{e}")
        self.client.command("OPTIMIZE TABLE stock_data.finicial_report FINAL")

        self.update(start_year)

    def calc_profit_diff(self):
        query = """
        SELECT
            code,
            report_date,
            net_profit,
            adjusted_profit
        FROM stock_data.finicial_report
        WHERE report_date = '2023-06-30'
        """
        df = self.client.query_df(query)
        df["profit_diff"] = df["net_profit"] - df["adjusted_profit"]
        df["profit_diff_percent"] = df["profit_diff"] / df["adjusted_profit"] * 100
        print(df)

    def update(self, start_year):
        now_year = datetime.now().year + 1
        sql = """
        SELECT code from stock_data.finicial_report
        """
        df = self.client.query_df(sql)
        for code in df["code"].unique():
            # try:
            dates_def = ["-03-31", "-06-30", "-09-30", "-12-31"]
            years = [str(year) for year in range(int(start_year), now_year)]
            dates = [year + date for year in years for date in dates_def[1:]]
            prev_dates = [year + date for year in years for date in dates_def[:-1]]
            for date, prev_date in zip(reversed(dates), reversed(prev_dates)):
                sql = f"""
                INSERT INTO stock_data.finicial_report 
                (report_date, code, publish_date, net_profit, adjusted_profit, roa, total_shares, circulating_a, adjusted_profit_diff)
                SELECT
                    report_date,
                    code,
                    publish_date,
                    net_profit,
                    adjusted_profit,
                    roa,
                    total_shares,
                    circulating_a,
                    adjusted_profit - (
                        SELECT adjusted_profit
                        FROM stock_data.finicial_report
                        WHERE report_date = '{prev_date}'
                        AND code = '{code}'
                    )
                FROM
                    stock_data.finicial_report
                WHERE 
                    report_date = '{date}'
                    AND code = '{code}'
                """
                logger.info(sql)
                self.client.command(sql)
            # except Exception as e:
            #     logger.error(f"update adjusted_profit_diff error {e}")
            #     continue


if __name__ == "__main__":
    # CREATE TABLE stock_data.finicial_report
    # (
    #     `report_date` Date,
    #     `code` String,
    #     `publish_date` Date,
    #     `net_profit` Float64,
    #     `adjusted_profit` Float64,
    #     `roa` Float64,
    #     `total_shares` Float64,
    #     `circulating_a` Float64,
    #     `adjusted_profit_diff` Float64,

    # ) ENGINE = ReplacingMergeTree()
    # ORDER BY (report_date, code)
    # code = "sz.002193"

    # df = Affair.parse(downdir="fin_data", filename="gpcw20210630.zip")
    # df.to_csv("test.csv")

    # from mootdx.quotes import Quotes

    # client = Quotes.factory(market="std")
    # client.finance(symbol="600300").to_csv("tmp.csv")

    proc = TDXProcess()
    proc.update_fincial_db()
    # proc.update_fincial_db()
