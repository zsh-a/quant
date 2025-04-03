from loguru import logger
import re, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import mplfinance as mpf
import indictor
import matplotlib.dates as mdates

import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime
import pandas as pd
import akshare as ak
import schedule
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.charts import Kline, Line, Grid, Scatter
import talib as ta
import clickhouse_connect

from loguru import logger
import db
import talib


def data_preprocess(file_path):
    with open(file_path, "r", errors="ignore") as file:
        lines = file.readlines()
    match = re.search(r"#(\d+)\.", file_path)
    code = match.group(1)
    lines = lines[2:-1]
    out_path = os.path.join("data", f"{code}.csv")
    with open(out_path, "w") as file:
        file.writelines(lines)


class DBDataSource:
    def __init__(
        self,
        code,
        trading_days=None,
        start_date="20150701",
        end_date=None,
        work_dir=".",
        random_start=False,
    ) -> None:
        self.code = code

        self.start_date = start_date
        self.end_date = end_date

        self.trading_days = trading_days

        self.cur_step = 0
        self.offset = 0

        self.data = self._load()

        self._preprocess()
        self.origin_data = self.data

        self.min_values = self.data.min()
        self.max_values = self.data.max()

        self._normilize()

        self.random_start = random_start
        self.seq_len = 0

        self.trading_days = len(self.data)


    def reset(self):
        self.offset = (
            np.random.randint(self.seq_len, len(self.data.index) - self.trading_days)
            if self.random_start
            else 0
        )
        self.cur_step = 0

    def __len__(self):
        return len(self.data)

    def step(self):
        obs, ori_obs = (
            self.data.iloc[self.offset + self.cur_step],
            self.origin_data.iloc[self.offset + self.cur_step],
        )
        self.cur_step += 1
        done = self.cur_step >= self.trading_days
        return obs, done, ori_obs

    def get_data(self):
        return self.origin_data

    def _normilize(self):
        # 计算均值和标准差
        df = self.data
        mean = df.mean()
        std = df.std()

        # 应用标准化公式
        df = (df - mean) / std
        df = df.astype(float)
        self.data = df

    def _preprocess(self):
        df = self.data
        # print(df)
        # df['adj_factor'] = df['close'].shift(1) / df['pre_close']
        # df['adj_factor'].iloc[0] = 1

        # df['adj_factor'] = df['adj_factor'].cumprod()

        # df['close'] = df['close'] * df['adj_factor']
        # df['open'] = df['open'] * df['adj_factor']
        # df['high'] = df['high'] * df['adj_factor']
        # df['low'] = df['low'] * df['adj_factor']
        # 定义MACD计算函数
        df = self.data.dropna()

        # 创建一个逐日生成周线MACD的函数
        def update_weekly_macd_daily(df):
            # 创建空的DataFrame来存储逐日的"周线"MACD数据
            weekly_macd_data = pd.DataFrame(index=df.index, columns=[])

            # 创建一个dataframe存储历史周线数据
            weekly_his_data = pd.DataFrame(
                columns=[
                    "open_weekly",
                    "high_weekly",
                    "low_weekly",
                    "close_weekly",
                    "volume_weekly",
                    "amount_weekly",
                ],
            )

            # 逐日循环，构建当前周的动态"周线"数据
            for current_day in df.index:
                # 找到本周的开始日期（周一）
                week_start = current_day - pd.to_timedelta(
                    current_day.weekday(), unit="d"
                )

                # 获取从本周开始日期到当前日期的日线数据
                current_week_data = df[week_start:current_day]

                # 如果当前周至少有一天数据，则生成周线数据并计算MACD
                if len(current_week_data) > 0:
                    weekly_open = current_week_data["open"].iloc[
                        0
                    ]  # 本周第一天的开盘价
                    weekly_high = current_week_data["high"].max()  # 本周内的最高价
                    weekly_low = current_week_data["low"].min()  # 本周内的最低价
                    weekly_close = current_week_data["close"].iloc[
                        -1
                    ]  # 当前日期的收盘价
                    weekly_volume = current_week_data["volume"].sum()  # 本周的总交易量
                    weekly_amount = current_week_data["amount"].sum()  # 本周的总交易额

                    # 构造一个包含当前周线数据的DataFrame
                    weekly_data = pd.DataFrame(
                        {
                            "open_weekly": [weekly_open],
                            "high_weekly": [weekly_high],
                            "low_weekly": [weekly_low],
                            "close_weekly": [weekly_close],
                            "volume_weekly": [weekly_volume],
                            "amount_weekly": [weekly_amount],
                        },
                        index=[current_day],
                    )
                    weekly_his_data.loc[current_day] = weekly_data.iloc[0]

                    # weekly_his_data = calculate_macd(weekly_his_data)
                    indictor.indictor_macd(weekly_his_data, colums=["close_weekly"])

                    weekly_macd_data.loc[current_day, "macd_close_weekly"] = (
                        weekly_his_data["macd_close_weekly"].loc[current_day]
                    )
                    weekly_macd_data.loc[current_day, "macd_close_weekly_last"] = (
                        (weekly_his_data["macd_close_weekly"].iloc[-2])
                        if len(weekly_his_data) > 1
                        else np.nan
                    )

                    # 如果不是周五
                    if current_day.weekday() != 4:
                        weekly_his_data.drop(current_day, inplace=True)

            return weekly_macd_data

        weekly_macd_result = update_weekly_macd_daily(df)

        data_with_weekly_macd = pd.concat([df, weekly_macd_result], axis=1)

        print(data_with_weekly_macd)
        df = data_with_weekly_macd

        weekly_df = (
            df.resample("W-FRI")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "amount": "sum",
                }
            )
            .dropna()
        )
        # print(self.code, weekly_df)
        weekly_df.rename(columns={"close": "close_weekly_vis"}, inplace=True)
        indictor.indictor_macd(weekly_df, colums=["close_weekly_vis"])
        df = df.join(weekly_df, rsuffix="_weekly", how="left").ffill()
        # print(self.code, df)
        indictor.indictor_force_index(df)
        indictor.indictor_KDJ(df)

        df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
        df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
        df["ema13"] = df["close"].ewm(span=13, adjust=False).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df.dropna(inplace=True)

        df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1)
        df["returns"] = np.log(df["close"] / df["close"].shift(1))

        df["price_volume"] = df["close"] * df["volume"]
        df["vwap"] = df["price_volume"].sum() / df["volume"].sum()

        # 示例：布林带
        # df["middle_band"] = df["close"].rolling(window=20).mean()
        # df["std"] = df["close"].rolling(window=20).std()
        # df["upper_band"] = df["middle_band"] + 2 * df["std"]
        # df["lower_band"] = df["middle_band"] - 2 * df["std"]

        df["high_point"] = df["high"].rolling(window=20).max()
        df["low_point"] = df["low"].rolling(window=20).min()

        df["resistance"] = df["high_point"].rolling(window=20).mean()  # 平均阻力
        df["support"] = df["low_point"].rolling(window=20).mean()  # 平均支撑

        # 参数设置
        period = 14

        # 1. 计算 +DM 和 -DM
        df["prev_high"] = df["high"].shift(1)
        df["prev_low"] = df["low"].shift(1)

        df["+DM"] = np.where(
            (df["high"] - df["prev_high"]) > (df["prev_low"] - df["low"]),
            np.maximum(df["high"] - df["prev_high"], 0),
            0,
        )
        df["-DM"] = np.where(
            (df["prev_low"] - df["low"]) > (df["high"] - df["prev_high"]),
            np.maximum(df["prev_low"] - df["low"], 0),
            0,
        )

        # 2. 计算 TR（真实波幅）
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = abs(df["high"] - df["close"].shift(1))
        df["tr3"] = abs(df["low"] - df["close"].shift(1))
        df["TR"] = df[["tr1", "tr2", "tr3"]].max(axis=1)

        # 3. 平滑 +DM、-DM 和 TR
        df["+DM_smoothed"] = df["+DM"].rolling(window=period).mean()
        df["-DM_smoothed"] = df["-DM"].rolling(window=period).mean()
        df["TR_smoothed"] = df["TR"].rolling(window=period).mean()

        # 4. 计算 +DI 和 -DI
        df["+DI"] = (df["+DM_smoothed"] / df["TR_smoothed"]) * 100
        df["-DI"] = (df["-DM_smoothed"] / df["TR_smoothed"]) * 100

        # 5. 计算 DX
        df["DX"] = (abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])) * 100

        # 6. 计算 ADX
        df["ADX"] = df["DX"].rolling(window=period).mean()

        # turtle
        df["up"] = ta.MAX(df.high, timeperiod=20).shift(1)
        # 最近N2个交易日最低价
        df["down"] = ta.MIN(df.low, timeperiod=10).shift(1)
        # 每日真实波动幅度
        df["ATR"] = ta.ATR(df.high, df.low, df.close, timeperiod=20)
        df["turtle_short"] = df["up"] - 2 * df["ATR"]

        # 动量因子: 过去5日涨跌幅
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1

        # 成交量因子: (最近5日平均成交量) / (最近10日平均成交量) - 1

        df["vol_ratio"] = (df["volume"].rolling(5).mean()) / (
            df["volume"].rolling(10).mean()
        ) - 1  #
        # 计算RSI (默认周期14)
        df["RSI_14"] = talib.RSI(df["close"], timeperiod=14)

        # 布林带
        upper, middle, lower = talib.BBANDS(
            df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        df["BB_upper"] = upper
        df["BB_middle"] = middle
        df["BB_lower"] = lower

        # 反转因子 = -动量因子
        df['reversal_5'] = -df['close'].pct_change(periods=5)


        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_20'] = df['log_ret'].rolling(20).std() * np.sqrt(252)

        df['turnover_avg_20'] = df['turn'].rolling(20).mean()

        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()

        df['turnover_5'] = df['turn'].rolling(5).sum()


        df['MACD'], df['Signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        


        df["ma_30"] = talib.EMA(df["close"], timeperiod=30)
        df["ma_60"] = talib.EMA(df["close"], timeperiod=60)

        # 计算均线斜率（20日均线）
        df['MA20_slope'] = df['ma_30'].diff().rolling(5).mean()
        # 判断支撑条件
        df['support_condition'] = (abs(df['close'] - df['ma_30']) / df['ma_30'] < 0.01) & (df['MA20_slope'] > 0) & (df['ma_30'] > df['ma_60'])


        # 计算成交量均量
        df['vol_ma10'] = df['volume'].rolling(10).mean()
        # 缩量条件
        df['low_volume'] = df['volume'] < 0.5 * df['vol_ma10'].shift(5)
        # 综合信号
        df['buy_signal'] = df['support_condition'] 



                
        # df["ma_60"] = talib.EMA(df["close"], timeperiod=60)
        # df["ma_120"] = talib.EMA(df["close"], timeperiod=120)

        df.dropna(inplace=True)
        self.data = df
        # self.data = df[['MA5','returns','MA10','MA20','MA30']]

    def _load(self):
        df = db.get_kline(self.code, self.start_date, self.end_date)

        df["close"] = df["close"] * df["adjfactor"]
        df["open"] = df["open"] * df["adjfactor"]
        df["high"] = df["high"] * df["adjfactor"]
        df["low"] = df["low"] * df["adjfactor"]

        df = df[["open", "high", "low", "close", "volume", "amount", "adjfactor","turn"]]
        df = df.astype(float)
        df = df[self.start_date :]
        print(df)
        # df['change'] = df['close'].pct_change()
        return df

        # df['returns_5d'] = (df['close'].shift(-PREDICT_KS) - df['close']) / df['close']

        # # 定义区间
        # bins = np.linspace(0,0.1,NUM_CLASS + 1)
        # labels = np.arange(0,NUM_CLASS)

        # # 使用 cut 函数生成标签
        # # df['label'] = pd.cut(df['returns_5d'], bins=bins, labels=labels)
        # df['label'] = (df['returns_5d'] > 0).astype(int)

        # df.dropna(inplace=True)

        # # df['volume_norm'] = (df['volume'] - df['volume'].min()) / (df['volume'].max() - df['volume'].min())

        # train_size = int(len(df) * 0.8)
        # train_set,test_set = df.iloc[:train_size],df.iloc[train_size:]
        # return normilize(train_set),normilize(test_set)


if __name__ == "__main__":
    ds = DBDataSource("sz.000001", 220, start_date="20230401", end_date="20240401")
    ds.reset()
    # print(ds.get_data())
    # print(df['returns'])
    print(ds.step())
    # for i in range(10):
    #     print(ds.step())
