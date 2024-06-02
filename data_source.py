from loguru import logger
import re,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def data_preprocess(file_path):
    with open(file_path, 'r',errors="ignore") as file:
        lines = file.readlines()
    match = re.search(r'#(\d+)\.', file_path)
    code = match.group(1)
    lines = lines[2:-1]
    out_path = os.path.join("data",f'{code}.csv')
    with open(out_path, 'w') as file:
        file.writelines(lines)

class DataSource:
    def __init__(self,code,trading_days=252,start_date='20150701',end_date='20180701') -> None:

        self.file_path = f'data/{code}.csv'
        if not os.path.exists(self.file_path):
            tdx_path = f'tdx_export/{code}.csv'
        logger.info(f'get stock code : {code}, file path : {self.file_path}')
        
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
        self._normilize()
        print(self.origin_data)
        print(self.data)
        print(len(self.data))

        self.seq_len = 20
        # self.data.set_index('trade_date').sort_index().dropna()
        
        # print(len(self.data.index) - self.trading_days)


    def reset(self):
        self.offset = np.random.randint(self.seq_len,len(self.data.index) - self.trading_days)
        self.cur_step = 0

        
    def step(self):
        obs,ori_obs = self.data.iloc[self.offset + self.cur_step - self.seq_len:self.offset + self.cur_step],self.origin_data.iloc[self.offset + self.cur_step]
        self.cur_step += 1
        done = self.cur_step > self.trading_days
        return obs,done,ori_obs
    
    def get_data(self):
        return self.data

    def _normilize(self):
        def min_max_scaling(column):
            return (column - column.min()) / (column.max() - column.min())
        self.data = self.data.apply(min_max_scaling)
        
    def _preprocess(self):
        df = self.data
        df['amplitude'] = ((df['high'] - df['low']) / df['close'].shift(1))
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        # df['ret_2'] = np.log(df['close'] / df['close'].shift(2))
        # df['ret_5'] = np.log(df['close'] / df['close'].shift(5))
        # df['ret_21'] = np.log(df['close'] / df['close'].shift(21))

        # df['moving_average'] = df['close'].rolling(window=21).mean()
        # df['MA5'] = df['close'].rolling(window=5).mean()
        # df['MA10'] = df['close'].rolling(window=10).mean()
        # df['MA20'] = df['close'].rolling(window=20).mean()
        # df['MA30'] = df['close'].rolling(window=30).mean()

        # plt.figure(figsize=(10, 6))
        # # plt.plot(df.index, df['close'], label='Price')
        # plt.plot(df.index, df['MA5'], label='Moving Average')
        # plt.plot(df.index, df['MA10'], label='MA21')
        # plt.plot(df.index, df['MA20'], label='MA21')
        # plt.plot(df.index, df['MA60'], label='MA21')
        # plt.title('Price and Moving Average')
        # plt.xlabel('Date')
        # plt.ylabel('Price')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        short_window = 12
        long_window = 26
        signal_window = 9

        # 计算短期EMA和长期EMA
        df["EMA_12"] = df["close"].ewm(span=short_window, adjust=False).mean()
        df["EMA_26"] = df["close"].ewm(span=long_window, adjust=False).mean()

        # 计算MACD值
        df["MACD"] = df["EMA_12"] - df["EMA_26"]

        # 计算信号线
        df["Signal_Line"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()

        # MACD Histogram，可选，代表MACD与信号线之间的差值
        df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]

        # 计算每日价格变化
        df['Price Change'] = df['close'].diff()

        # 计算增益和损失
        window_length = 6
        df['Gain'] = df['Price Change'].apply(lambda x: x if x > 0 else 0)
        df['Loss'] = df['Price Change'].apply(lambda x: -x if x < 0 else 0)

        # 计算平均增益和平均损失
        df['Avg Gain'] = df['Gain'].rolling(window=window_length, min_periods=1).mean()
        df['Avg Loss'] = df['Loss'].rolling(window=window_length, min_periods=1).mean()

        # 计算RS和RSI
        df['RS'] = df['Avg Gain'] / df['Avg Loss']
        df['RSI'] = 100 - (100 / (1 + df['RS']))


        df.dropna(inplace=True)
        self.data = df[['amplitude','returns','MACD_Histogram','RSI']]
        # self.data = df[['MA5','returns','MA10','MA20','MA30']]
        
    def _load_csv(self):
        df = pd.read_csv(self.file_path)
        df.columns = ['datetime','open','high','low','close','volume','amount']
        df.set_index('datetime', inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[['open','high','low','close','volume']]
        df = df.astype(float)
        df = df[self.start_date:self.end_date]
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


if __name__ == '__main__':
    ds = DataSource('510880')
    ds.reset()
    # print(ds.get_data())
    # print(df['returns'])
    print(ds.step())
    # for i in range(10):
    #     print(ds.step())