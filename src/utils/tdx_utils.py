from mootdx.affair import Affair
from datetime import datetime
import pandas as pd
import clickhouse_connect
import os
import hashlib
import json
from loguru import logger
from db import DB
# Use config_manager for now to keep compatibility with existing config files
from config_manager import cm

def convert_to_date(num):
    try:
        date_str = f"{int(num):06d}"  # 去除小数点并补零至6位（如250315）
        return datetime.strptime(date_str, "%y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return "无效日期"

class TDXProcess:
    table_name = "stock_data.finicial_report"
    state_file = "data/tdx_sync_state.json"

    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host=cm.get("db.host"),
            username=cm.get("db.username"),
            password=cm.get("db.password"),
        )
        self.fin_path = cm.get("fin_data.path", "fin_data")
        os.makedirs(self.fin_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state file: {e}")
        return {"file_hashes": {}}

    def _save_state(self):
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")

    def fetch_tdx(self):
        """
        Fetch updated financial data files from TDX.
        Uses cached hashes to avoid re-hashing all local files.
        """
        local_hash = self.state.get("file_hashes", {})
        
        # Verify if local files still match cached hashes (optional, but good for first run or after manual changes)
        # To optimize, we only check files that exist in local_hash
        existing_files = set(os.listdir(self.fin_path))
        keys_to_remove = [f for f in local_hash if f not in existing_files]
        for f in keys_to_remove:
            del local_hash[f]

        updated_files = []
        remote_files = Affair.files()
        
        for item in remote_files:
            filename = item["filename"]
            remote_md5 = item["hash"]
            
            # Check if we already have this file with the same hash
            if filename in local_hash and local_hash[filename] == remote_md5:
                continue
                
            try:
                logger.info(f"Updating financial data from TDX: {filename}")
                Affair.fetch(downdir=self.fin_path, filename=filename)
                
                # Update hash after successful fetch
                local_hash[filename] = remote_md5
                updated_files.append(filename)
            except Exception as e:
                logger.error(f"Failed to fetch {filename} from TDX: {e}")
                continue

        self.state["file_hashes"] = local_hash
        self._save_state()
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
            "每股净资产": "nav_per_share", 
            "每股收益": "eps", 
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
                val = df["营业总收入(万元)"] * 10000 
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
            if code_str.startswith("6") or code_str.startswith("68"):
                return "sh." + code_str
            elif code_str.startswith("0") or code_str.startswith("3") or code_str.startswith("00"):
                return "sz." + code_str
            return None

        new_df["code"] = new_df.index.to_series().apply(format_code)
        
        # 清理数据：必须有 code 和有效的日期
        new_df = new_df.dropna(subset=["code", "report_date", "publish_date"])
        new_df["adjusted_profit_diff"] = 0.0
        new_df["market_cap"] = 0.0
        new_df["circulating_market_cap"] = 0.0
        new_df["pe_ratio"] = 0.0
        new_df["pb_ratio"] = 0.0 

        return new_df

    def update_fincial_db(self, start_year=None):
        """
        Incremental update: only process files downloaded by fetch_tdx.
        """
        if start_year is None:
            # Try to detect start_year from DB
            try:
                res = self.client.query(f"SELECT min(toYear(report_date)) FROM {self.table_name}")
                if res.result_rows and res.result_rows[0][0]:
                    start_year = str(res.result_rows[0][0])
                else:
                    start_year = "2024"
            except Exception:
                start_year = "2024"

        files = self.fetch_tdx()
        if not files:
            logger.info("No new financial data files to process.")
            # Still run update to ensure derived fields are correct if needed
            self.update(start_year)
            return

        for filename in files:
            logger.info(f"Processing {filename}...")
            try:
                df = Affair.parse(downdir=self.fin_path, filename=filename)
                processed_df = self._process_df(df, filename=filename)
                
                if not processed_df.empty:
                    cols = [
                        "report_date", "code", "publish_date", "net_profit", "adjusted_profit",
                        "total_operating_revenue", "subtotal_operate_cash_inflow", "roe", "roa",
                        "inc_net_profit_year_on_year", "total_shares", "circulating_a",
                        "market_cap", "circulating_market_cap", "pe_ratio", "pb_ratio", "adjusted_profit_diff"
                    ]
                    self.client.insert_df(self.table_name, processed_df[cols])
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")

        self.update(start_year)

    def init_fincial_db(self, start_year="2024"):
        """
        Full initialization from a specific year.
        """
        dates_def = ["0331", "0630", "0930", "1231"]
        now_year = datetime.now().year
        years = [str(year) for year in range(int(start_year), now_year + 1)]
        dates = [year + date for year in years for date in dates_def]

        for date in dates:
            filename = f"gpcw{date}.zip"
            if not os.path.exists(os.path.join(self.fin_path, filename)):
                # If file doesn't exist locally, we can't parse it. 
                # Normally fetch_tdx should have been called first if we want remote files.
                continue
                
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
        Compute derived financial indicators.
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
        try:
            self.client.command(sql)
            self.client.command(f"OPTIMIZE TABLE {self.table_name} FINAL")
            logger.info("Financial indicators update completed.")
        except Exception as e:
            logger.error(f"Failed to update derived fields: {e}")

if __name__ == "__main__":
    proc = TDXProcess()
    proc.update_fincial_db()
