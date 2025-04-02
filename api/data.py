from fastapi import FastAPI
from loguru import logger
import re, os
import numpy as np
from datetime import datetime
import pandas as pd
import talib as ta
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from talib import EMA

import sys



# 获取当前文件的绝对路径，并找到上层目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import db
import backtest
from loguru import logger

app = FastAPI()


# 配置 CORS 中间件
origins = [
    "http://localhost:5173",  # 允许跨域请求的源，可以根据需要添加更多
    "http://localhost:5174",  # 允许跨域请求的源，可以根据需要添加更多
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)


# 定义API端点，用于根据股票代码获取股票数据
@app.get("/stocks/{symbol}")
def get_stock_by_symbol(symbol: str):
    df = db.get_kline(symbol, None, None)

    # df["datetime"] = pd.to_datetime(df["datetime"])
    # df.index = df.index.strftime("%Y-%m-%d")
    # df.set_index("datetime", inplace=True)

    df["close"] = df["close"] * df["adjfactor"]
    df["open"] = df["open"] * df["adjfactor"]
    df["high"] = df["high"] * df["adjfactor"]
    df["low"] = df["low"] * df["adjfactor"]

    df = df[["open", "high", "low", "close", "volume", "amount", "adjfactor"]]

    # 计算 5 日移动平均线
    df["ma_55"] = EMA(df["close"], timeperiod=55)

    # 计算 10 日移动平均线
    df["ma_144"] = EMA(df["close"], timeperiod=144)
    # 计算 30 日移动平均线
    df["ma_169"] = EMA(df["close"], timeperiod=169)

    df["ma_30"] = EMA(df["close"], timeperiod=30)
    df["ma_60"] = EMA(df["close"], timeperiod=60)
    df["ma_120"] = EMA(df["close"], timeperiod=120)

    df["ma_288"] = EMA(df["close"], timeperiod=288)
    df["ma_338"] = EMA(df["close"], timeperiod=338)

    df["ATR"] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=14)

    df.dropna(inplace=True)
    df = df.astype(float)
    df = df.round(2)

    # ma = {"ma_55": [], "ma_144": [], "ma_169": [], "ma_288": [], "ma_338": []}
    ma = {"ma_30": [], "ma_60": [], "ma_120": []}
    atr = []
    kline = []
    for index, row in df.iterrows():
        kline.append(
            {
                "time": index,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
        )
        ma["ma_30"].append({"time": index, "value": row["ma_30"]})
        ma["ma_60"].append({"time": index, "value": row["ma_60"]})
        ma["ma_120"].append({"time": index, "value": row["ma_120"]})
        # ma["ma_55"].append({"time": index, "value": row["ma_55"]})
        # ma["ma_144"].append({"time": index, "value": row["ma_144"]})
        # ma["ma_169"].append({"time": index, "value": row["ma_169"]})
        # ma["ma_288"].append({"time": index, "value": row["ma_288"]})
        # ma["ma_338"].append({"time": index, "value": row["ma_338"]})
        atr.append({"time": index, "value": row["ATR"]})

    resp = {"kline": kline, "ma": ma, "ATR": atr}

    return resp

@app.get("/backtest/{symbol}/{start_date}/{end_date}")
def run_backtest(symbol: str, start_date: str, end_date: str):
    
    return backtest.run_policy(symbol)


@app.get("/meta/{symbol}")
def get_meta(symbol: str):
    # print(db.get_meta(symbol).iloc[0])
    return db.get_meta(symbol).iloc[0].to_dict()


if __name__ == "__main__":
    # 调用 uvicorn.run 启动应用
    uvicorn.run(app, host="0.0.0.0", port=8000)
