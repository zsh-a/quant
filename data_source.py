from loguru import logger
import re, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import indictor
import matplotlib.dates as mdates


def data_preprocess(file_path):
    with open(file_path, "r", errors="ignore") as file:
        lines = file.readlines()
    match = re.search(r"#(\d+)\.", file_path)
    code = match.group(1)
    lines = lines[2:-1]
    out_path = os.path.join("data", f"{code}.csv")
    with open(out_path, "w") as file:
        file.writelines(lines)


class DataSource:
    def __init__(
        self,
        code,
        trading_days=252,
        start_date="20150701",
        end_date="20180701",
        work_dir=".",
        random_start=False,
    ) -> None:
        self.file_path = os.path.join(work_dir, f"data/{code}.csv")
        if not os.path.exists(self.file_path):
            tdx_path = f"tdx_export/{code}.csv"
        # logger.info(f'get stock code : {code}, file path : {self.file_path}')

        self.start_date = start_date
        self.end_date = end_date

        self.trading_days = trading_days

        self.cur_step = 0
        self.offset = 0

        self.data = self._load_csv()

        self._preprocess()
        self.origin_data = self.data
        self.min_values = self.data.min()
        self.max_values = self.data.max()
        # self._normilize()

        self.random_start = random_start

        # print(self.origin_data)
        # print(self.data)
        # print(len(self.data))

        self.seq_len = 0
        # self.data.set_index('trade_date').sort_index().dropna()

        # print(len(self.data.index) - self.trading_days)

    def reset(self):
        # print(len(self.data.index))
        self.offset = (
            np.random.randint(self.seq_len, len(self.data.index) - self.trading_days)
            if self.random_start
            else 0
        )
        self.cur_step = 0

    def step(self):
        obs, ori_obs = (
            self.data.iloc[self.offset + self.cur_step],
            self.origin_data.iloc[self.offset + self.cur_step],
        )
        self.cur_step += 1
        done = self.cur_step > self.trading_days
        return obs, done, ori_obs

    def get_data(self):
        return self.data

    def _normilize(self):
        def min_max_scaling(column):
            return (column - column.min()) / (column.max() - column.min())

        self.data = self.data.apply(min_max_scaling)

    def _preprocess(self):
        df = self.data
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
        weekly_df.rename(columns={"close": "close_weekly"}, inplace=True)
        indictor.indictor_macd(weekly_df, colums=["close_weekly"])
        df = df.join(weekly_df, rsuffix="_weekly", how="outer").ffill()
        indictor.indictor_force_index(df)

        df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()

        df.dropna(inplace=True)
        print(df)
        # plt.figure(figsize=(14, 7))
        # plt.plot(df["force_index_close"], label="Close Price", color="blue")
        # plt.title("Stock Price with SMA and EMA")
        # plt.xlabel("Date")
        # plt.ylabel("Price")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        # print(df)
        df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1)
        df["returns"] = np.log(df["close"] / df["close"].shift(1))
        # df["ret_2"] = np.log(df["close"] / df["close"].shift(2))
        # df["ret_5"] = np.log(df["close"] / df["close"].shift(5))
        # df["ret_21"] = np.log(df["close"] / df["close"].shift(21))

        # df['moving_average'] = df['close'].rolling(window=21).mean()
        df["MA5"] = df["close"].rolling(window=5).mean()
        df["MA10"] = df["close"].rolling(window=10).mean()
        # df['MA30'] = df['close'].rolling(window=30).mean()
        df["MA20"] = df["close"].rolling(window=20).mean()

        # 计算每日价格变化
        df["Price Change"] = df["close"].diff()

        # 计算增益和损失
        window_length = 6
        df["Gain"] = df["Price Change"].apply(lambda x: x if x > 0 else 0)
        df["Loss"] = df["Price Change"].apply(lambda x: -x if x < 0 else 0)

        # 计算平均增益和平均损失
        df["Avg Gain"] = df["Gain"].rolling(window=window_length, min_periods=1).mean()
        df["Avg Loss"] = df["Loss"].rolling(window=window_length, min_periods=1).mean()

        # 计算RS和RSI
        df["RS"] = df["Avg Gain"] / df["Avg Loss"]
        df["RSI"] = 100 - (100 / (1 + df["RS"]))

        df.dropna(inplace=True)
        self.data = df[
            [
                "amplitude",
                "returns",
                "macd_close_weekly",
                "force_index_close",
                "MA5",
                "ema12",
                "ema26",
                "volume",
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        # self.data = df[['MA5','returns','MA10','MA20','MA30']]
        # print(self.data)

    def _load_csv(self):
        df = pd.read_csv(self.file_path)

        df.columns = ["datetime", "open", "high", "low", "close", "volume", "amount"]
        df.set_index("datetime", inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[["open", "high", "low", "close", "volume", "amount"]]
        df = df.astype(float)
        df = df[self.start_date : self.end_date]
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

    def macd_bars(self, buy_sell_points):
        colors = [
            "green" if val >= 0 else "red"
            for val in self.origin_data["macd_close_weekly"]
        ]

        return [
            mpf.make_addplot(self.origin_data["ema12"], color="lime"),
            mpf.make_addplot(self.origin_data["ema26"], color="c"),
            mpf.make_addplot(
                self.origin_data["macd_close_weekly"],
                type="bar",
                width=0.7,
                color=colors,
                panel=1,
                alpha=0.5,
                secondary_y=False,
            ),
            mpf.make_addplot(
                self.origin_data["force_index_close"],
                type="bar",
                width=0.7,
                color="b",
                panel=2,
                alpha=0.5,
                secondary_y=False,
            ),
        ]

    def plot(self, buy_sell_points):
        custom_colors = mpf.make_marketcolors(
            up="red",  # 上涨的颜色
            down="green",  # 下跌的颜色
            edge="black",  # K线边缘颜色
            wick="black",  # K线上下影线颜色
            volume="blue",  # 成交量条颜色
        )
        style = mpf.make_mpf_style(marketcolors=custom_colors)
        apds = self.macd_bars(buy_sell_points)

        fig, axes = mpf.plot(
            self.origin_data, addplot=apds, type="candle", style=style, returnfig=True
        )
        ax = axes[0]
        for date, action in buy_sell_points:
            print(date, action)

            if action == "buy":
                ax.scatter(
                    self.origin_data.index.get_loc(date),
                    self.origin_data.loc[date, "low"],
                    color="red",
                    marker="^",
                    s=100,
                )
            elif action == "sell":
                ax.scatter(
                    self.origin_data.index.get_loc(date),
                    self.origin_data.loc[date, "high"],
                    color="green",
                    marker="v",
                    s=100,
                )

        # # 获取当前图表的Axes对象
        # # 设置x轴日期格式为matplotlib的日期格式
        # fig.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # ax2 = fig.twinx()
        # colors = ['green' if val >= 0 else 'red' for val in self.origin_data['macd_close_weekly']]
        # # print(self.origin_data.index)
        # ax2.bar(self.origin_data.index,self.origin_data['macd_close_weekly'], color=colors, alpha=0.3, width=0.7)
        # ax2.set_ylabel('MACD Histogram')


if __name__ == "__main__":
    ds = DataSource("510880", 220, start_date="20230401", end_date="20240401")
    ds.reset()
    # print(ds.get_data())
    # print(df['returns'])
    print(ds.step())
    # for i in range(10):
    #     print(ds.step())
