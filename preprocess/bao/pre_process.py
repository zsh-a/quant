import numpy as np
import pandas as pd
import clickhouse_connect
import os
import datetime
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

import baostock as bs
from loguru import logger


def _fetch_stock_batch(batch_tasks):
    """
    进程池任务：一组股票共用一个登录会话
    batch_tasks: List of (code, last_update_date, last_adjfactor)
    """
    if not batch_tasks:
        return [], []

    bs.login()
    all_results = []
    meta_updates = []
    batch_total = len(batch_tasks)
    
    try:
        for idx, (code, last_update_date, last_adjfactor) in enumerate(batch_tasks):
            start_str = last_update_date.strftime("%Y-%m-%d") if hasattr(last_update_date, 'strftime') else str(last_update_date)
            try:
                rs = bs.query_history_k_data_plus(
                    code,
                    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,isST",
                    start_date=start_str,
                    frequency="d",
                    adjustflag="3",
                )

                if rs.error_code != "0":
                    logger.error(
                        f"Baostock API error for {code} (start_date={start_str}): "
                        f"error_code={rs.error_code}, error_msg={rs.error_msg}"
                    )
                    continue

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())
                
                if not data_list:
                    logger.warning(f"No data returned for {code} (start_date={start_str}), may be up-to-date or no trading data")
                    continue

                df = pd.DataFrame(data_list, columns=rs.fields)
                
                # 数据清洗
                df["peTTM"] = df["peTTM"].replace("", "0")
                df["pbMRQ"] = df["pbMRQ"].replace("", "0")
                df.replace("", np.nan, inplace=True)
                df.dropna(inplace=True)
                
                if df.empty:
                    continue

                df = df.astype({
                    "open": float, "high": float, "low": float, "close": float,
                    "preclose": float, "volume": float, "amount": float, "turn": float,
                    "tradestatus": float, "pctChg": float, "peTTM": float, "pbMRQ": float, "isST": int,
                })
                df["date"] = pd.to_datetime(df["date"])
                
                # 计算复权
                df["adjfactor"] = df["close"].shift(1) / df["preclose"]
                df.loc[df.index[0], "adjfactor"] = last_adjfactor
                df["adjfactor"] = df["adjfactor"].cumprod()
                
                # 剔除已存在的首行
                df = df[1:]
                
                if not df.empty:
                    all_results.append(df)
                    last_row = df.iloc[-1]
                    meta_updates.append({
                        'code': code,
                        'last_update_date': last_row['date'],
                        'last_adjfactor': last_row['adjfactor'],
                        'error_update_count': 0
                    })
            except Exception as e:
                logger.exception(f"Error processing {code} (start_date={start_str}) in batch: {e}")
                continue

            if (idx + 1) % 200 == 0 or idx + 1 == batch_total:
                pct = 100 * (idx + 1) / batch_total
                logger.info(f"Batch progress: {idx + 1}/{batch_total} ({pct:.1f}%), fetched {len(meta_updates)} with new data")
                
    finally:
        try:
            bs.logout()
        except Exception as e:
            # logout 失败时不要影响已获取的数据返回，多进程下 logout 常失败
            logger.warning(f"Batch logout failed (data preserved): {e}")
        
    return all_results, meta_updates


class BaoStockProcessor:
    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host="localhost", username="default", password=""
        )

    def fetch_bao_data(self, code, start_date):
        bs.login()
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,isST",
            start_date=start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else str(start_date),
            frequency="d",
            adjustflag="3",
        )

        data_list = []
        while (rs.error_code == "0") & rs.next():
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        bs.logout()
        return result

    def insert_data(self):
        for file in os.listdir("data/bao"):
            code = file.split(".csv")[0]
            if not code.startswith(('sh', 'sz')): continue
            
            df = pd.read_csv(os.path.join("data", "bao", file))
            df["date"] = pd.to_datetime(df["date"])

            df["adjfactor"] = df["close"].shift(1) / df["preclose"]
            df.loc[0, "adjfactor"] = 1
            df["adjfactor"] = df["adjfactor"].cumprod()

            if "adjustflag" in df.columns:
                del df["adjustflag"]
            
            logger.info(f"Inserting {code}, rows: {len(df)}")

            self.client.insert(
                table="stock_data.stock_daily",
                column_names=[
                    "date", "code", "open", "high", "low", "close", "preclose",
                    "volume", "amount", "turn", "tradestatus", "pctChg",
                    "peTTM", "pbMRQ", "isST", "adjfactor",
                ],
                data=df,
            )

    def update_meta_batch(self, updates):
        if not updates:
            return
        
        codes = [u['code'] for u in updates]
        placeholders = ", ".join([f"'{c}'" for c in codes])
        name_rows = self.client.query(f"SELECT code, name FROM stock_data.stock_daily_meta WHERE code IN ({placeholders})").result_rows
        name_map = {r[0]: r[1] for r in name_rows}

        data = []
        for u in updates:
            data.append([
                u['code'], 
                u['last_update_date'], 
                u['last_adjfactor'], 
                u['error_update_count'],
                name_map.get(u['code'], '')
            ])
        
        self.client.insert(
            "stock_data.stock_daily_meta",
            data=data,
            column_names=["code", "last_update_date", "last_adjfactor", "error_update_count", "name"]
        )

    def update_daily_data(self, max_workers=1):
        logger.info("Starting daily K-line update with batch processing...")
        query = """
            SELECT code, last_update_date, last_adjfactor
            FROM stock_data.stock_daily_meta
        """
        rows = self.client.query(query).result_rows
        
        # 过滤任务
        tasks = [(r[0], r[1], r[2]) for r in rows if len(r[0]) == 9]
        total_tasks = len(tasks)
        logger.info(f"Total stocks to check: {total_tasks}")
        
        if total_tasks == 0:
            return

        # 分片逻辑：将所有任务平均分配给 worker
        batch_size = math.ceil(total_tasks / max_workers)
        chunks = [tasks[i : i + batch_size] for i in range(0, total_tasks, batch_size)]
        total_batches = len(chunks)
        
        all_new_kline = []
        all_meta_updates = []
        completed_batches = 0
        
        logger.info(f"Split into {total_batches} batches, ~{batch_size} stocks each")
        
        # 并行执行分片任务
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_stock_batch, chunk): i for i, chunk in enumerate(chunks)}
            
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    batch_kline, batch_meta = future.result()
                    if batch_kline:
                        all_new_kline.extend(batch_kline)
                    if batch_meta:
                        all_meta_updates.extend(batch_meta)
                    
                    completed_batches += 1
                    logger.info(f"Batch {completed_batches}/{total_batches} done, +{len(batch_meta)} new, total fetched: {len(all_meta_updates)}")
                except Exception as e:
                    completed_batches += 1
                    logger.error(f"Batch {completed_batches}/{total_batches} failed: {e}")
        
        # 批量写入
        if all_new_kline:
            combined_kline = pd.concat(all_new_kline)
            cols = [
                "date", "code", "open", "high", "low", "close", "preclose",
                "volume", "amount", "turn", "tradestatus", "pctChg",
                "peTTM", "pbMRQ", "isST", "adjfactor",
            ]
            self.client.insert_df("stock_data.stock_daily", combined_kline[cols])
            logger.info(f"Successfully inserted {len(combined_kline)} new K-line rows")
            
            self.update_meta_batch(all_meta_updates)
            logger.info(f"Updated meta for {len(all_meta_updates)} stocks")
            self.client.command("OPTIMIZE TABLE stock_data.stock_daily_meta FINAL")
        else:
            logger.info("No new data found in any batch.")

    def update_industry_data_weekly(self):
        """
        从2010年开始每周一更新股票行业信息
        """

        # 获取当前日期
        today = datetime.date.today()

        # 从2010年开始循环
        start_date = datetime.date(2023, 11, 1)

        # 每周一执行更新
        while start_date <= today:
            if start_date.weekday() == 0:  # 0代表周一
                print(f"正在更新{start_date}的行业数据...")
                try:
                    rs = bs.query_stock_industry(date=start_date.strftime("%Y-%m-%d"))
                    # 打印结果集
                    industry_list = []
                    while (rs.error_code == "0") & rs.next():
                        # 获取一条记录，将记录合并在一起
                        industry_list.append(rs.get_row_data())
                    df = pd.DataFrame(industry_list, columns=rs.fields)
                    df["date"] = df["updateDate"]
                    for index, row in df.iterrows():
                        cmd = f"""
                        INSERT INTO stock_data.finicial_data (date, code, industry, industryClassification)
                        VALUES ('{row["date"]}', '{row["code"]}', '{row["industry"]}', '{row["industryClassification"]}')
                        """
                        self.client.command(cmd)

                except Exception as e:
                    print(f"Error updating industry data for {start_date}: {e}")
                # print(type(df['date'][0]))
                # df.to_csv('tmp.csv')
            start_date += datetime.timedelta(days=1)


if __name__ == "__main__":
    processor = BaoStockProcessor()
    # processor.create_meta()
    # processor.update_daily_data()
    # processor.update_industry_data_weekly()
    # print(processor.fetch_bao_data("sh.000985",datetime.date(2020,1,1)))
