import numpy as np
import pandas as pd
import clickhouse_connect
import os
import datetime

import baostock as bs
import pandas as pd
from loguru import logger


class BaoStockProcessor:
    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host="localhost", username="default", password=""
        )

        self.lg = bs.login()

    def __del__(self):
        bs.logout()

    def fetch_bao_data(self, code, start_date):
        #### 登陆系统 ####

        #### 获取沪深A股历史K线数据 ####
        # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
        # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
        # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg

        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,isST",
            start_date=start_date.strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="3",
        )

        data_list = []
        while (rs.error_code == "0") & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)

        #### 登出系统 ####

        return result

    def insert_data(self):
        for file in os.listdir("data/bao"):
            ps = []
            code = file.split(".csv")[0]
            df = pd.read_csv(os.path.join("data", "bao", f"{code}.csv"))
            df["date"] = pd.to_datetime(df["date"])  # 确保 timestamp 是 datetime 类型

            df["adjfactor"] = df["close"].shift(1) / df["preclose"]
            # df["adj_factor"].iloc[0] = 1
            df.loc[0, "adjfactor"] = 1

            df["adjfactor"] = df["adjfactor"].cumprod()

            del df["adjustflag"]
            print(code, df)

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

    def create_meta(self):
        # 查询最新的 K 线数据
        query = """
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
    ) AS t
    WHERE rn = 1
        """
        kline_data = self.client.query(query)

        df = kline_data.result_rows
        # 构建更新语句
        for code, last_update_date, adjfactor in df:
            try:
                # 插入或更新数据
                update_query = f"""
                INSERT INTO stock_data.stock_daily_meta (code, last_update_date, last_adjfactor, error_update_count)
                VALUES ('{code}', '{last_update_date}', '{adjfactor}', 0)
                """
                self.client.command(update_query)
            except Exception as e:
                # 处理失败的情况，更新 error_update_count
                error_query = f"""
                INSERT INTO stock_data.stock_daily_meta (code, last_update_date, last_adjfactor, error_update_count)
                VALUES ('{code}', '{last_update_date}', '{adjfactor}',error_update_count + 1)
                """
                self.client.command(error_query)
                print(f"Error updating code {code}: {e}")

    def update_meta(self, code, last_update_date, adjfactor):
        update_query = f"""
        INSERT INTO stock_data.stock_daily_meta (code, last_update_date, last_adjfactor, error_update_count,name)
        SELECT 
            code,
            '{last_update_date}',
            '{adjfactor}',
            error_update_count,
            name
        FROM stock_data.stock_daily_meta
        WHERE code = '{code}';
        """
        self.client.command(update_query)

    def update_meta_error(self, code, error_update_count_delta):
        update_query = f"""
        INSERT INTO stock_data.stock_daily_meta (code, last_update_date, last_adjfactor, error_update_count)
        SELECT 
            code,
            last_update_date,
            last_adjfactor,
            error_update_count + {error_update_count_delta}
        FROM stock_data.stock_daily_meta
        WHERE code = '{code}';
        """
        self.client.command(update_query)

    def fetch_update(self, code, last_update_date, last_adjfactor):
        df = self.fetch_bao_data(code, last_update_date)
        df["peTTM"] = df["peTTM"].replace("", "0")
        df["pbMRQ"] = df["pbMRQ"].replace("", "0")
        df.replace("", np.nan, inplace=True)

        df.dropna(inplace=True)
        # print(df)
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
        df["date"] = pd.to_datetime(df["date"])  # 确保 timestamp 是 datetime 类型
        df["adjfactor"] = df["close"].shift(1) / df["preclose"]
        df.loc[df.index[0], "adjfactor"] = last_adjfactor
        df["adjfactor"] = df["adjfactor"].cumprod()
        del df["adjustflag"]
        df = df[1:]
        
        print(code, df)
        return df

    def update_daily_data(self):
        query = """
            SELECT
                code,
                last_update_date,
                last_adjfactor,
                error_update_count
            FROM  
                stock_data.stock_daily_meta
        """

        df = self.client.query(query).result_rows

        for code, last_update_date, last_adjfactor, error_update_count in df:
            if len(code) != 9:
                continue
            try:
                new_df = self.fetch_update(code, last_update_date, last_adjfactor)
                if len(new_df) > 0:
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
                        data=new_df,
                    )
                    new_data = new_df.iloc[-1]
                    self.update_meta(code, new_data["date"], new_data["adjfactor"])
                # print(new_data['adjfactor'])
            except Exception as e:
                logger.error(f"Error updating code {code}: {e}")
                self.update_meta_error(code, 1)
            # update meta
            # break

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
