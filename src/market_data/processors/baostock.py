import datetime
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import current_process

import baostock as bs
import numpy as np
import pandas as pd
from loguru import logger

from src.market_data.clickhouse import create_clickhouse_client


def _fetch_stock_batch(batch_tasks):
    if not batch_tasks:
        return [], []

    bs.login()
    all_results = []
    meta_updates = []
    batch_total = len(batch_tasks)

    try:
        for idx, (code, last_update_date, last_adjfactor) in enumerate(batch_tasks):
            start_str = (
                last_update_date.strftime("%Y-%m-%d")
                if hasattr(last_update_date, "strftime")
                else str(last_update_date)
            )
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
                    logger.warning(
                        f"No data returned for {code} (start_date={start_str}), may be up-to-date or no trading data"
                    )
                    continue

                df = pd.DataFrame(data_list, columns=rs.fields)
                df["peTTM"] = df["peTTM"].replace("", "0")
                df["pbMRQ"] = df["pbMRQ"].replace("", "0")
                df.replace("", np.nan, inplace=True)
                df.dropna(inplace=True)

                if df.empty:
                    continue

                df = df.astype(
                    {
                        "open": float,
                        "high": float,
                        "low": float,
                        "close": float,
                        "preclose": float,
                        "volume": float,
                        "amount": float,
                        "turn": float,
                        "tradestatus": float,
                        "pctChg": float,
                        "peTTM": float,
                        "pbMRQ": float,
                        "isST": int,
                    }
                )
                df["date"] = pd.to_datetime(df["date"])

                df["adjfactor"] = df["close"].shift(1) / df["preclose"]
                df.loc[df.index[0], "adjfactor"] = last_adjfactor
                df["adjfactor"] = df["adjfactor"].cumprod()

                df = df[1:]

                if not df.empty:
                    all_results.append(df)
                    last_row = df.iloc[-1]
                    meta_updates.append(
                        {
                            "code": code,
                            "last_update_date": last_row["date"],
                            "last_adjfactor": last_row["adjfactor"],
                            "error_update_count": 0,
                        }
                    )
            except Exception as exc:
                logger.exception(f"Error processing {code} (start_date={start_str}) in batch: {exc}")
                continue

            if (idx + 1) % 50 == 0 or idx + 1 == batch_total:
                pct = 100 * (idx + 1) / batch_total
                logger.info(
                    f"Batch progress: {idx + 1}/{batch_total} ({pct:.1f}%), fetched {len(meta_updates)} with new data"
                )

    finally:
        try:
            bs.logout()
        except Exception as exc:
            logger.warning(f"Batch logout failed (data preserved): {exc}")

    return all_results, meta_updates


class BaoStockProcessor:
    def __init__(self):
        self.client = create_clickhouse_client()

    @staticmethod
    def _coerce_date(value):
        if value is None:
            return None
        if isinstance(value, datetime.datetime):
            dt = value.date()
        elif isinstance(value, datetime.date):
            dt = value
        else:
            try:
                dt = datetime.date.fromisoformat(str(value))
            except (ValueError, TypeError):
                return None

        if dt and dt.year < 1990:
            return None
        return dt

    def _query_scalar(self, sql: str, params: dict | None = None):
        try:
            result = self.client.query(sql, parameters=params)
        except Exception as exc:
            logger.warning(f"Scalar query failed: {exc}")
            return None
        if not result.result_rows:
            return None
        return result.result_rows[0][0]

    def fetch_bao_data(self, code, start_date):
        bs.login()
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,isST",
            start_date=start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else str(start_date),
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
            if not code.startswith(("sh", "sz")):
                continue

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
                    "date",
                    "code",
                    "open",
                    "high",
                    "low",
                    "close",
                    "preclose",
                    "volume",
                    "amount",
                    "turn",
                    "tradestatus",
                    "pctChg",
                    "peTTM",
                    "pbMRQ",
                    "isST",
                    "adjfactor",
                ],
                data=df,
            )

    def update_meta_batch(self, updates):
        if not updates:
            return

        codes = [u["code"] for u in updates]
        name_rows = self.client.query(
            "SELECT code, name FROM stock_data.stock_daily_meta WHERE code IN {codes:Array(String)}",
            parameters={"codes": codes},
        ).result_rows
        name_map = {r[0]: r[1] for r in name_rows}

        data = []
        for u in updates:
            data.append(
                [
                    u["code"],
                    u["last_update_date"],
                    u["last_adjfactor"],
                    u["error_update_count"],
                    name_map.get(u["code"], ""),
                ]
            )

        self.client.insert(
            "stock_data.stock_daily_meta",
            data=data,
            column_names=["code", "last_update_date", "last_adjfactor", "error_update_count", "name"],
        )

    def update_daily_data(self, max_workers=1, progress_callback=None):
        logger.info("Starting daily K-line update with batch processing...")
        query = """
            SELECT code, last_update_date, last_adjfactor
            FROM stock_data.stock_daily_meta
        """
        rows = self.client.query(query).result_rows

        tasks = [(r[0], r[1], r[2]) for r in rows if len(r[0]) == 9]
        total_tasks = len(tasks)
        logger.info(f"Total stocks to check: {total_tasks}")

        if total_tasks == 0:
            return

        worker_count = max(1, int(max_workers))
        batch_size = math.ceil(total_tasks / worker_count)
        chunks = [tasks[i : i + batch_size] for i in range(0, total_tasks, batch_size)]
        total_batches = len(chunks)

        all_new_kline = []
        all_meta_updates = []
        completed_batches = 0

        logger.info(f"Split into {total_batches} batches, ~{batch_size} stocks each")

        run_in_process_pool = worker_count > 1 and not current_process().daemon
        if run_in_process_pool:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {executor.submit(_fetch_stock_batch, chunk): i for i, chunk in enumerate(chunks)}

                for future in as_completed(futures):
                    try:
                        batch_kline, batch_meta = future.result()
                        if batch_kline:
                            all_new_kline.extend(batch_kline)
                        if batch_meta:
                            all_meta_updates.extend(batch_meta)

                        completed_batches += 1
                        logger.info(
                            f"Batch {completed_batches}/{total_batches} done, +{len(batch_meta)} new, total fetched: {len(all_meta_updates)}"
                        )
                        if progress_callback:
                            progress_callback(completed_batches, total_batches, len(all_meta_updates))
                    except Exception as exc:
                        completed_batches += 1
                        logger.error(f"Batch {completed_batches}/{total_batches} failed: {exc}")
                        if progress_callback:
                            progress_callback(completed_batches, total_batches, len(all_meta_updates))
        else:
            if worker_count > 1 and current_process().daemon:
                logger.info("Current process is daemonized; falling back to in-process BaoStock execution")
            for chunk in chunks:
                try:
                    batch_kline, batch_meta = _fetch_stock_batch(chunk)
                    if batch_kline:
                        all_new_kline.extend(batch_kline)
                    if batch_meta:
                        all_meta_updates.extend(batch_meta)

                    completed_batches += 1
                    logger.info(
                        f"Batch {completed_batches}/{total_batches} done, +{len(batch_meta)} new, total fetched: {len(all_meta_updates)}"
                    )
                    if progress_callback:
                        progress_callback(completed_batches, total_batches, len(all_meta_updates))
                except Exception as exc:
                    completed_batches += 1
                    logger.error(f"Batch {completed_batches}/{total_batches} failed: {exc}")
                    if progress_callback:
                        progress_callback(completed_batches, total_batches, len(all_meta_updates))

        if all_new_kline:
            combined_kline = pd.concat(all_new_kline)
            cols = [
                "date",
                "code",
                "open",
                "high",
                "low",
                "close",
                "preclose",
                "volume",
                "amount",
                "turn",
                "tradestatus",
                "pctChg",
                "peTTM",
                "pbMRQ",
                "isST",
                "adjfactor",
            ]
            self.client.insert_df("stock_data.stock_daily", combined_kline[cols])
            logger.info(f"Successfully inserted {len(combined_kline)} new K-line rows")

            self.update_meta_batch(all_meta_updates)
            logger.info(f"Updated meta for {len(all_meta_updates)} stocks")
            self.client.command("OPTIMIZE TABLE stock_data.stock_daily_meta FINAL")
        else:
            logger.info("No new data found in any batch.")

    def update_industry_data_weekly(self):
        today = datetime.date.today()
        start_date = datetime.date(2023, 11, 1)

        while start_date <= today:
            if start_date.weekday() == 0:
                print(f"正在更新{start_date}的行业数据...")
                try:
                    rs = bs.query_stock_industry(date=start_date.strftime("%Y-%m-%d"))
                    industry_list = []
                    while (rs.error_code == "0") & rs.next():
                        industry_list.append(rs.get_row_data())
                    df = pd.DataFrame(industry_list, columns=rs.fields)
                    df["date"] = df["updateDate"]
                    for _, row in df.iterrows():
                        cmd = f"""
                        INSERT INTO stock_data.finicial_data (date, code, industry, industryClassification)
                        VALUES ('{row["date"]}', '{row["code"]}', '{row["industry"]}', '{row["industryClassification"]}')
                        """
                        self.client.command(cmd)

                except Exception as exc:
                    print(f"Error updating industry data for {start_date}: {exc}")
            start_date += datetime.timedelta(days=1)

    def update_trade_dates(self, start_date=None, end_date=None):
        if end_date is None:
            end_date = datetime.date.today().strftime("%Y-%m-%d")

        if start_date is None:
            latest = self._coerce_date(self._query_scalar("SELECT max(calendar_date) FROM stock_data.trade_dates"))
            if latest is None:
                start_date = "1990-01-01"
            else:
                start_date = (latest + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        if start_date > end_date:
            return {"message": "trade dates already up-to-date", "rows": 0}

        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {lg.error_code} {lg.error_msg}")

        try:
            rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
            if rs.error_code != "0":
                raise RuntimeError(f"query_trade_dates failed: {rs.error_code} {rs.error_msg}")

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            df = pd.DataFrame(data_list, columns=rs.fields)
            if df.empty:
                return {"message": "no trade dates returned", "rows": 0}

            df["calendar_date"] = pd.to_datetime(df["calendar_date"])
            df["is_trading_day"] = pd.to_numeric(df["is_trading_day"], errors="coerce").fillna(0).astype(int)
            df = df[["calendar_date", "is_trading_day"]]

            self.client.insert_df("stock_data.trade_dates", df)
            self.client.command("OPTIMIZE TABLE stock_data.trade_dates FINAL")
            return {
                "start_date": start_date,
                "end_date": end_date,
                "rows": len(df),
            }
        finally:
            try:
                bs.logout()
            except Exception as exc:
                logger.warning(f"BaoStock logout failed: {exc}")

    def update_all_stock(self, day=None, force=False):
        if day is None:
            latest_trading = self._coerce_date(
                self._query_scalar("SELECT max(calendar_date) FROM stock_data.trade_dates WHERE is_trading_day = 1")
            )
            if latest_trading is None:
                latest_trading = self._coerce_date(self._query_scalar("SELECT max(date) FROM stock_data.stock_daily"))
            if latest_trading is None:
                latest_trading = datetime.date.today()
            day = latest_trading.strftime("%Y-%m-%d")

        if not force:
            existing = self._query_scalar(
                "SELECT count() FROM stock_data.all_stock WHERE day = {day:String}",
                params={"day": day},
            )
            if existing and int(existing) > 0:
                return {"day": day, "rows": 0, "message": "already ingested"}

        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {lg.error_code} {lg.error_msg}")

        try:
            rs = bs.query_all_stock(day=day)
            if rs.error_code != "0":
                raise RuntimeError(f"query_all_stock failed: {rs.error_code} {rs.error_msg}")

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            df = pd.DataFrame(data_list, columns=rs.fields)
            if df.empty:
                raise RuntimeError(f"query_all_stock returned empty for day={day}")

            df["day"] = pd.to_datetime(day)
            if "tradeStatus" in df.columns:
                df["tradeStatus"] = pd.to_numeric(df["tradeStatus"], errors="coerce").fillna(0).astype(int)
            if "code_name" in df.columns:
                df["code_name"] = df["code_name"].fillna("").astype(str)

            cols = ["day", "code", "tradeStatus", "code_name"]
            df = df[cols]

            self.client.insert_df("stock_data.all_stock", df)
            self.client.command("OPTIMIZE TABLE stock_data.all_stock FINAL")
            return {"day": day, "rows": len(df)}
        finally:
            try:
                bs.logout()
            except Exception as exc:
                logger.warning(f"BaoStock logout failed: {exc}")
