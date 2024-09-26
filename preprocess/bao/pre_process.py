import numpy as np

import pandas as pd
import re, os
import matplotlib.pyplot as plt
import math


import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import datetime
import pandas as pd
import akshare as ak
import schedule
from loguru import logger

bucket = "stock"
token = os.environ.get("INFLUXDB_TOKEN")
org = "zs"
url = "http://localhost:8086"

client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)


def write_data():
    write_api = client.write_api(write_options=SYNCHRONOUS)

    for file in os.listdir("data/bao"):
        code = file.split(".")[0]
        df = pd.read_csv(os.path.join("data", "bao", f"{code}.csv"))
        # print(df.iloc[-3:-1])

        # break
        # df = df.iloc[-3:-1]
        df["ticker"] = f"{code}"

        df["date"] = pd.to_datetime(df["date"])  # 确保 timestamp 是 datetime 类型

        df["adj_factor"] = df["close"].shift(1) / df["preclose"]
        # df["adj_factor"].iloc[0] = 1
        df.loc[0, "adj_factor"] = 1

        df["adj_factor"] = df["adj_factor"].cumprod()
        print(code, df)

        # 将 pandas DataFrame 插入 InfluxDB
        for index, row in df.iterrows():
            point = (
                Point("stock_data")
                .tag("ticker", row["ticker"])
                .field("open", row["open"])
                .field("high", row["high"])
                .field("low", row["low"])
                .field("close", row["close"])
                .field("pre_close", row["preclose"])
                .field("amount", row["amount"])
                .field(
                    "volume", int(row["volume"]) if not math.isnan(row["volume"]) else 0
                )
                .time(row["date"], WritePrecision.S)
            )
            # print(point)
            write_api.write(bucket=bucket, org="zs", record=point)
            # print(row["volume"])
            # break
        # break


if __name__ == "__main__":
    # df = pd.read_csv("000001.csv")
    # print(df)
    write_data()
