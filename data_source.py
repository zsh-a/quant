import re
import os
import numpy as np
import indictor



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
        **args
    ) -> None:
        self.code = code

        self.db_client = args["db_client"]
        self.start_date = start_date
        self.end_date = end_date

        self.trading_days = trading_days

        self.cur_step = 0
        self.offset = 0

        self.data = self._load()
        self._preprocess()

    def add_indicator(self, indicator_func):
        self.data = indicator_func(self.data)

    def reset(self):
        self.cur_step = 0

    def __len__(self):
        return len(self.data)

    def step(self):
        if self.cur_step >= len(self.data):
            return None
        current_idx = self.offset + self.cur_step

        obs = self.data.iloc[current_idx]
        self.cur_step += 1
        return obs

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

        df["close"] = df["close"] * df["adjfactor"]
        df["open"] = df["open"] * df["adjfactor"]
        df["high"] = df["high"] * df["adjfactor"]
        df["low"] = df["low"] * df["adjfactor"]

        self._init_basic_indicator()


    def clean_data(self):
        self.data.dropna(inplace=True)
        self.data = self.data[self.start_date : self.end_date]

    def _init_basic_indicator(self):
        self.add_indicator(indictor.indictor_ema())

        df = self.data
        df["returns"] = np.log(df["close"] / df["close"].shift(1))
        df["price_volume"] = df["close"] * df["volume"]
        df["vwap"] = df["price_volume"].sum() / df["volume"].sum()
        self.data = df

    def _load(self):
        df = self.db_client.get_kline(self.code, "20100101", self.end_date)

        df = df[
            ["open", "high", "low", "close", "volume", "amount", "adjfactor", "turn"]
        ]

        if self.end_date is None:
            self.end_date = df.index.iloc[-1]
        df = df.astype(float)
        return df


if __name__ == "__main__":
    ds = DBDataSource("sz.000001", 220, start_date="20220401", end_date="20240401")
    ds.reset()
    # print(ds.get_data())
    # print(df['returns'])
    print(ds.step())
    # for i in range(10):
    #     print(ds.step())
