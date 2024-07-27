import pandas as pd
import clickhouse_connect
import os
import pandas as pd
import talib as ta
from talib import EMA

from loguru import logger


client = clickhouse_connect.get_client(
    host="localhost", username="default", password=""
)

print(client.command("show databases"))


def get_stock_name(code):
    df = pd.read_csv("all_stock.csv", index_col="code")

    return df.loc[code]["code_name"]


start_date = "20220101"


def select_stock():
    query = """
        SELECT
            code,
            last_update_date,
            last_adjfactor,
            error_update_count
        FROM  
            stock_data.stock_daily_meta
    """

    res = []

    df = client.query(query).result_rows

    for code, last_update_date, last_adjfactor, error_update_count in df:
        # if code.startswith("00") or code.startswith("60"):
        # code = "sh.603809"
        query = f"""
        SELECT 
            code,
            MAX(isST) AS has_st
        FROM stock_data.stock_daily
        WHERE date >= '{start_date}'
            AND code = '{code}'
        GROUP BY code
        """

        data = client.query(query)
        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        if df["has_st"][0] == 1:
            continue

        query = f"""
        SELECT *,
            -- 计算复权收盘价并估算流通市值（单位：元）
            close * (volume * 100 / turn) AS float_market_cap
        FROM stock_data.stock_daily
        WHERE 
            code = '{code}'
            AND date > '{start_date}'
            AND turn > 0
        ORDER BY date
        """
        dt = code.split(".")[1]
        # print(dt)
        # breakpoint()
        if (
            not dt.startswith("00") and not dt.startswith("60")
            # and not dt.startswith("30")
        ):
            continue
        data = client.query(query)

        df = pd.DataFrame(data.result_rows, columns=data.column_names)
        # print(df['turn'])
        if len(df) < 388:
            continue
        df["date"] = pd.to_datetime(df["date"])  # 确保日期列为 datetime 类型

        df["close"] = df["close"] * df["adjfactor"]
        df["open"] = df["open"] * df["adjfactor"]
        df["high"] = df["high"] * df["adjfactor"]
        df["low"] = df["low"] * df["adjfactor"]

        # 计算ATR（真实波动幅度）
        df["ATR"] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=14)

        # 转为float
        # 计算 5 日移动平均线
        df["ma_55"] = EMA(df["close"], timeperiod=55)

        # 计算 10 日移动平均线
        df["ma_144"] = EMA(df["close"], timeperiod=144)
        # 计算 30 日移动平均线
        df["ma_169"] = EMA(df["close"], timeperiod=169)

        df["ma_288"] = EMA(df["close"], timeperiod=288)
        df["ma_338"] = EMA(df["close"], timeperiod=338)

        df["ma_30"] = EMA(df["close"], timeperiod=30)
        df["ma_60"] = EMA(df["close"], timeperiod=60)
        df["ma_120"] = EMA(df["close"], timeperiod=120)


        condition = (df['ma_30'] > df['ma_60']) & (df['ma_60'] > df['ma_120'])

        group_ids = (~condition).cumsum()  # 条件不满足时生成新组
        consecutive = condition.groupby(group_ids).cumcount() + 1  # 组内计数从1开始
        df['consecutive_days'] = consecutive.where(condition, 0)  # 不满足时设为0

        # print(df)
        df.dropna(inplace=True)

        df["fake_signal"] = (
            # ((df["high"] > df["ma_55"]) & (df["low"] <= df["ma_55"]))
            ((df["high"] > df["ma_30"]) & (df["low"] <= df["ma_30"]))
            | ((df["high"] > df["ma_60"]) & (df["low"] <= df["ma_60"]))
            | ((df["high"] > df["ma_120"]) & (df["low"] <= df["ma_120"]))
            # ((df["high"] > df["ma_288"]) & (df["low"] <= df["ma_288"]))
            # | ((df["high"] > df["ma_338"]) & (df["low"] <= df["ma_338"]))
        )
        df["up_signal"] = (
            (df["ma_30"] > df["ma_60"]) & (df["ma_60"] > df["ma_120"])
            # & (df["ma_144"] > df["ma_288"])
        )
        df["day_up"] = df["close"] > df["low"]
        df["signal"] = (
            (
                df["fake_signal"]
                | df["fake_signal"].shift(1)
                | df["fake_signal"].shift(2)
            )
            & df["up_signal"].shift(1)
            & df["up_signal"]
            # & df["day_up"]
            & (df["close"] > df["open"])
            # & (df["turn"] > 5)
            & (df["float_market_cap"] > 5 * 10**9)
            & (df['consecutive_days'] >= 40)
        )
        df.to_csv('tmp.csv')
        if len(df) == 0:
            continue
        if df.iloc[-1]["signal"]:
            res.append(code)


    for code in res:
        print(f"{code} {get_stock_name(code)}")
    return res


if __name__ == "__main__":
    res = select_stock()
    # print(f"size : {len(res)}")
    # print(res)

    # get_stock_name("sz.000001")
