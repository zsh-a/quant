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
            try:
                if filename not in local_hash or local_hash[filename] != md5:
                    logger.info(f"update fin date from tdx : {filename}")
                    Affair.fetch(downdir=self.fin_path, filename=filename)
                    updated_files.append(filename)
            except Exception as e:
                logger.error(f"update fin date from tdx : {filename} error : {e}")
                continue

        return updated_files

    def _process_df(self, df, filename=None, date=None):
        if len(df) == 0:
            return pd.DataFrame()

        # 核心映射逻辑
        mapping = {
            "五、净利润": "net_profit",
            "扣除非经常性损益后的净利润": "adjusted_profit",
            "营业总收入": "total_operating_revenue",
            "经营活动现金流入小计": "subtotal_operate_cash_inflow",
            "净资产收益率": "roe",
            "总资产报酬率": "roa",
            "净利润增长率(%)": "inc_net_profit_year_on_year",
            "总股本": "total_shares",
            "已上市流通A股": "circulating_a",
            "每股净资产": "nav_per_share", # 用于计算PB
            "每股收益": "eps", # 用于计算PE
        }

        new_df = pd.DataFrame(index=df.index)
        
        # 转换 report_date
        new_df["report_date"] = pd.to_datetime(df["report_date"].astype(str), format="%Y%m%d", errors='coerce').dt.date
        
        # 转换 publish_date
        new_df["publish_date"] = pd.to_datetime(df["财报公告日期"].apply(convert_to_date), errors='coerce').dt.date

        # 映射字段（带多列处理）
        for tdx_col, internal_col in mapping.items():
            if tdx_col in df.columns:
                val = df[tdx_col]
            elif internal_col == "total_operating_revenue" and "营业总收入(万元)" in df.columns:
                val = df["营业总收入(万元)"] * 10000 # 转换为元
            elif internal_col == "total_operating_revenue" and "营业收入" in df.columns:
                val = df["营业收入"]
            else:
                new_df[internal_col] = 0.0
                continue

            if isinstance(val, pd.DataFrame):
                new_df[internal_col] = val.iloc[:, 0]
            else:
                new_df[internal_col] = val

        # 获取代码前缀
        def format_code(code):
            code_str = str(code).zfill(6)
            if code_str.startswith("6"):
                return "sh." + code_str
            elif code_str.startswith("0") or code_str.startswith("3"):
                return "sz." + code_str
            return None

        new_df["code"] = new_df.index.to_series().apply(format_code)
        
        # 清理数据：必须有 code 和有效的日期
        new_df = new_df.dropna(subset=["code", "report_date", "publish_date"])

        # 计算增量利润 (adjusted_profit_diff)
        # 注意：这里需要根据季度逻辑，但为了简化，我们先填入 adjusted_profit
        # 后续 update 方法会根据上一季度数据重写这个值
        new_df["adjusted_profit_diff"] = 0.0
        
        # 市值相关字段 (需要结合股价，但目前仅从报表中拿财务数据)
        # 实际市值计算可能需要同步行情数据，这里先保留字段为0或从报表拿(如果报备里有的话，通常没有)
        new_df["market_cap"] = 0.0
        new_df["circulating_market_cap"] = 0.0
        new_df["pe_ratio"] = 0.0
        new_df["pb_ratio"] = 0.0 # 稍后可以在 SQL 中或此处通过 (股价 / 每股净资产) 处理

        return new_df

    def update_fincial_db(self, start_year=2024):
        files = self.fetch_tdx()
        for filename in files:
            logger.info(f"Processing {filename}...")
            df = Affair.parse(downdir=self.fin_path, filename=filename)
            processed_df = self._process_df(df, filename=filename)
            
            if not processed_df.empty:
                # 批量插入
                cols = [
                    "report_date", "code", "publish_date", "net_profit", "adjusted_profit",
                    "total_operating_revenue", "subtotal_operate_cash_inflow", "roe", "roa",
                    "inc_net_profit_year_on_year", "total_shares", "circulating_a",
                    "market_cap", "circulating_market_cap", "pe_ratio", "pb_ratio", "adjusted_profit_diff"
                ]
                self.client.insert_df(self.table_name, processed_df[cols])
        
        self.update(start_year)

    def init_fincial_db(self, start_year="2024"):
        dates_def = ["0331", "0630", "0930", "1231"]
        now_year = datetime.now().year
        years = [str(year) for year in range(int(start_year), now_year + 1)]
        dates = [year + date for year in years for date in dates_def]

        for date in dates:
            filename = f"gpcw{date}.zip"
            try:
                logger.info(f"Parsing {filename}...")
                df = Affair.parse(downdir=self.fin_path, filename=filename)
                processed_df = self._process_df(df, date=date)
                
                if not processed_df.empty:
                    cols = [
                        "report_date", "code", "publish_date", "net_profit", "adjusted_profit",
                        "total_operating_revenue", "subtotal_operate_cash_inflow", "roe", "roa",
                        "inc_net_profit_year_on_year", "total_shares", "circulating_a",
                        "market_cap", "circulating_market_cap", "pe_ratio", "pb_ratio", "adjusted_profit_diff"
                    ]
                    self.client.insert_df(self.table_name, processed_df[cols])
            except Exception as e:
                logger.error(f"parse {filename} error : {e}")
                continue

        self.client.command(f"OPTIMIZE TABLE {self.table_name} FINAL")
        self.update(start_year)

    def update(self, start_year):
        """
        优化后的 update 方法：使用单次 SQL INSERT ... SELECT 计算所有派生指标（增量利润、市值、PE/PB）。
        使用 FINAL 确保读取最新行，防止多步更新时的 eclipsing 效应。
        """
        logger.info(f"Computing derived financial indicators starting from {start_year}")
        
        sql = f"""
        INSERT INTO {self.table_name}
        (report_date, code, publish_date, net_profit, adjusted_profit, total_operating_revenue, subtotal_operate_cash_inflow, roe, roa, inc_net_profit_year_on_year, total_shares, circulating_a, market_cap, circulating_market_cap, pe_ratio, pb_ratio, adjusted_profit_diff)
        SELECT
            curr.report_date, curr.code, curr.publish_date, curr.net_profit, curr.adjusted_profit, curr.total_operating_revenue, curr.subtotal_operate_cash_inflow, curr.roe, curr.roa, curr.inc_net_profit_year_on_year, curr.total_shares, curr.circulating_a,
            coalesce(daily.close * curr.total_shares, 0) AS market_cap,
            coalesce(daily.close * curr.circulating_a, 0) AS circulating_market_cap,
            CASE WHEN curr.net_profit > 0 THEN (coalesce(daily.close, 0) * curr.total_shares) / curr.net_profit ELSE 0 END AS pe_ratio,
            CASE 
                WHEN curr.roe != 0 AND (curr.net_profit / (curr.roe/100)) > 0 
                THEN (coalesce(daily.close, 0) * curr.total_shares) / (curr.net_profit / (curr.roe/100)) 
                ELSE 0 
            END AS pb_ratio,
            CASE 
                WHEN formatDateTime(curr.report_date, '%m-%d') = '03-31' THEN curr.adjusted_profit
                ELSE curr.adjusted_profit - coalesce(prev.adjusted_profit, 0)
            END AS adjusted_profit_diff
        FROM (SELECT * FROM {self.table_name} FINAL WHERE report_date >= '{start_year}-01-01') AS curr
        LEFT JOIN (
            SELECT code, report_date, adjusted_profit FROM {self.table_name} FINAL 
            WHERE formatDateTime(report_date, '%m-%d') IN ('03-31', '06-30', '09-30')
        ) AS prev 
            ON curr.code = prev.code 
            AND prev.report_date = CASE
                WHEN formatDateTime(curr.report_date, '%m-%d') = '06-30' THEN toDate(concat(toString(toYear(curr.report_date)), '-03-31'))
                WHEN formatDateTime(curr.report_date, '%m-%d') = '09-30' THEN toDate(concat(toString(toYear(curr.report_date)), '-06-30'))
                WHEN formatDateTime(curr.report_date, '%m-%d') = '12-31' THEN toDate(concat(toString(toYear(curr.report_date)), '-09-30'))
                ELSE toDate('1900-01-01')
            END
        LEFT JOIN stock_data.stock_daily AS daily ON curr.code = daily.code AND curr.publish_date = daily.date
        """
        self.client.command(sql)
        self.client.command(f"OPTIMIZE TABLE {self.table_name} FINAL")
        logger.info("Financial indicators update completed.")
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
    # proc.init_fincial_db(start_year="2010")
    # proc.update_fincial_db(start_year=2010)
    proc.update(start_year="2010")
    # proc.update_fincial_db()
    # proc.update_fincial_db()

    
