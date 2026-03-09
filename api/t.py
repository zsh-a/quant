from fastapi import FastAPI
from loguru import logger
import re, os
import numpy as np
from datetime import datetime
import pandas as pd
import talib as ta
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
from talib import EMA
from src.market_data.clickhouse import create_clickhouse_client

from loguru import logger

# 配置 CORS 中间件
origins = [
    "http://localhost:5173",  # 允许跨域请求的源，可以根据需要添加更多
]

MA_PERIOD = 20  # 均线周期
THRESHOLD_PCT = 1.0  # 百分比阈值(1%)
THRESHOLD_ATR_MULT = 0.5  # ATR倍数阈值
START_DATE = "2023-01-01"


def get_stock_by_symbol(symbol: str):
    client = create_clickhouse_client()

    query = f"""
    SELECT *,
            close * (volume * 100 / turn) AS float_market_cap
    FROM stock_data.stock_daily
    WHERE code = '{symbol}' AND date >= '20220101'
    ORDER BY date
    """
    # logger.debug(f"exec query : {query}")

    data = client.query(query)
    df = pd.DataFrame(data.result_rows, columns=data.column_names)
    if len(df) < 388:
        return
    # print(df)

    df.rename(columns={"date": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d")
    df.set_index("datetime", inplace=True)

    df["close"] = df["close"] * df["adjfactor"]
    df["open"] = df["open"] * df["adjfactor"]
    df["high"] = df["high"] * df["adjfactor"]
    df["low"] = df["low"] * df["adjfactor"]
    df = df[["open", "high", "low", "close", "volume", "amount", "adjfactor", "turn",'float_market_cap']]

    # 计算 5 日移动平均线
    df["ma_55"] = df["close"].ewm(55).mean()
    # 计算 10 日移动平均线
    df["ma_144"] = df["close"].ewm(144).mean()
    # 计算 30 日移动平均线
    df["ma_169"] = df["close"].ewm(169).mean()

    df["ma_288"] = df["close"].ewm(288).mean()
    df["ma_338"] = df["close"].ewm(338).mean()

    df["ma_30"] = EMA(df["close"], timeperiod=30)
    df["ma_60"] = EMA(df["close"], timeperiod=60)
    df["ma_120"] = EMA(df["close"], timeperiod=120)

    df = df.astype(float)
    df = df.round(2)

    # df["fake_signal"] = (
    #     # ((df["high"] > df["ma_55"]) & (df["low"] <= df["ma_55"]))
    #     # ((df["high"] > df["ma_30"]) & (df["low"] <= df["ma_30"]))
    #     ((df["high"] > df["ma_60"]) & (df["low"] <= df["ma_60"]))
    #     | ((df["high"] > df["ma_120"]) & (df["low"] <= df["ma_120"]))
    #     # | ((df["high"] > df["ma_144"]) & (df["low"] <= df["ma_144"]))
    #     # | ((df["high"] > df["ma_169"]) & (df["low"] <= df["ma_169"]))
    #     # | ((df["high"] > df["ma_288"]) & (df["low"] <= df["ma_288"]))
    #     # | ((df["high"] > df["ma_338"]) & (df["low"] <= df["ma_338"]))
    # )
    # df["up_signal"] = (
    #     (df["ma_30"] > df["ma_60"]) & (df["ma_60"] > df["ma_120"])
    #     # (df["ma_55"] > df["ma_144"])
    #     # & (df["ma_144"] > df["ma_288"])
    # )
    # df["day_up"] = df["close"] > df["low"]
    # df["hh"] = df["high"] > df["high"].shift(1)

    # df["signal"] = (
    #     (df["fake_signal"] )
    #     & df["up_signal"].shift(1)
    #     & df["up_signal"]
    #     & df["day_up"]
    #     & (df["close"] > df["open"])
    #     & (df["turn"] > 3)
    #     & df["hh"]
    # )


    condition = (df['ma_30'] > df['ma_60']) & (df['ma_60'] > df['ma_120'])

    group_ids = (~condition).cumsum()  # 条件不满足时生成新组
    consecutive = condition.groupby(group_ids).cumcount() + 1  # 组内计数从1开始
    df['consecutive_days'] = consecutive.where(condition, 0)  # 不满足时设为0

    # print(df)
    df.dropna(inplace=True)

    df["fake_signal"] = (
        # ((df["high"] > df["ma_55"]) & (df["low"] <= df["ma_55"]))
        # ((df["high"] > df["ma_30"]) & (df["low"] <= df["ma_30"]))
        ((df["high"] > df["ma_60"]) & (df["low"] <= df["ma_60"]))
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
        & df["day_up"]
        & (df["close"] > df["open"])
        # & (df["turn"] > 5)
        & (df["float_market_cap"] > 5 * 10**9)
        & (df['consecutive_days'] >= 30)
    )
    df["Composite_Signal"] = df["signal"]

    # df["MA"] = df["ma_55"]

    # df['ATR'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    # # 计算价格与均线距离
    # df['Distance_pct'] = abs(df['close'] - df['MA']) / df['MA'] * 100
    # df['Distance_ATR'] = abs(df['close'] - df['MA']) / df['ATR']

    # # 生成信号
    # df['Signal_pct'] = df['Distance_pct'] <= THRESHOLD_PCT
    # df['Signal_ATR'] = df['Distance_ATR'] <= THRESHOLD_ATR_MULT
    # df['Composite_Signal'] = df['Signal_pct'] & df['Signal_ATR']

    # # 在calculate_technical函数中添加：
    # df['MA_Slope'] = df['MA'].diff(3)  # 3日斜率
    # df['Trend_Up'] = df['MA_Slope'] > 0
    # df['Composite_Signal'] = df['Composite_Signal'] & df['Trend_Up']

    # print(df)

    # plot_signals(df)

    analyze_signals(symbol,df)


# ====================
# 可视化
# ====================
def plot_signals(df):
    plt.figure(figsize=(12, 6))

    # 绘制价格和均线
    plt.plot(df["close"], label="Price", alpha=0.5)
    plt.plot(df["MA"], label=f"{MA_PERIOD}D MA", linestyle="--")

    # 标记接近区域
    plt.fill_between(
        df.index,
        df["MA"] * (1 - THRESHOLD_PCT / 100),
        df["MA"] * (1 + THRESHOLD_PCT / 100),
        color="orange",
        alpha=0.2,
        label="Threshold Zone",
    )

    # 标记复合信号点
    signals = df[df["Composite_Signal"]]
    plt.scatter(
        signals.index,
        signals["close"],
        marker="^",
        color="green",
        s=100,
        label="Buy Signal",
    )

    plt.title(" Price & MA Proximity Signals")
    plt.legend()
    plt.grid(True)
    plt.savefig("signals.png", dpi=300, bbox_inches="tight")
    # plt.show()

dis = []

def analyze_signals(code, df):
    df["future_return"] = df["close"].transform(lambda x: x.shift(-5) / x - 1)
    signals = df[df["Composite_Signal"]].dropna()
    if len(signals) < 5:  
        return
    print(signals)
    # print(df[df['Composite_Signal']].dropna()['future_return']))
    # signals = df[df['Composite_Signal']].copy()
    # signals['Next_5d_Return'] = signals['close'].pct_change(10).shift(-10)
    dis.append(signals['future_return'])
    print(f"\n信号统计:{code} {get_stock_name(code)}")
    print(f"总信号数量: {len(signals)}")
    print(f"信号后5日平均收益率: {signals['future_return'].mean():.2%}")
    print(f"信号出现频率: {len(signals) / len(df):.2%}")
def get_stock_name(code):
    df = pd.read_csv("all_stock.csv", index_col="code")

    return df.loc[code]["code_name"]


if __name__ == '__main__':
    client = create_clickhouse_client()

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
    # code = 'sh.600105'
    # get_stock_by_symbol("sz.000716")
    for code, last_update_date, last_adjfactor, error_update_count in df:
        dt = code.split(".")[1]
        if (
            not dt.startswith("00") and not dt.startswith("60")
            and not dt.startswith("30")
        ):
            continue
        # print(code)
        get_stock_by_symbol(code)
        # break
        if len(dis) > 1000:
            break
    counts, bins = np.histogram(dis, bins=20)
    plt.figure(figsize=(10, 6))
    plt.hist(dis, bins=bins, edgecolor='black', alpha=0.7)

    # 添加标题和标签
    plt.title("频率分布直方图 (正态分布)")
    plt.xlabel("数值区间")
    plt.ylabel("频数")

    # 显示图形
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    # plt.show()
    plt.savefig("histogram.png", dpi=300, bbox_inches='tight')

    # print(counts, bins)
# counts: 每个区间的频数
# bins: 区间的边界值
