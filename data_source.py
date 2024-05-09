from loguru import logger
import re,os
import pandas as pd
import numpy as np

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

        self.min_values = self.data.min()
        self.max_values = self.data.max()
        print(self.data)
        # self.data.set_index('trade_date').sort_index().dropna()
        
        # print(len(self.data.index) - self.trading_days)


    def reset(self):
        self.offset = np.random.randint(0,len(self.data.index) - self.trading_days)
        self.cur_step = 0

        
    def step(self):
        obs = self.data.iloc[self.offset + self.cur_step]
        self.cur_step += 1
        done = self.cur_step > self.trading_days
        return obs,done
    
    def get_data(self):
        return self.data

    def normilize(df):
        for col in feature_columns:
            df.loc[:,col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        df.loc[:,'returns_5d'] = (df['returns_5d'] - df['returns_5d'].min()) / (df['returns_5d'].max() - df['returns_5d'].min())
        # print(df)
        X = []
        Y = []
        for i in range(len(df) - PREDICT_KS - FEATURE_KS):
            X.append(df[feature_columns].iloc[i:i + FEATURE_KS].values)
            # Y.append(df['returns_5d'].iloc[i + FEATURE_KS])
            Y.append(df['label'].iloc[i + FEATURE_KS])
        return np.array(X),np.array(Y)
        
    def _preprocess(self):
        df = self.data
        df['amplitude'] = ((df['high'] - df['low']) / df['close'].shift(1))
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['ret_2'] = np.log(df['close'] / df['close'].shift(2))
        df['ret_5'] = np.log(df['close'] / df['close'].shift(5))
        df['ret_21'] = np.log(df['close'] / df['close'].shift(21))
        df.dropna(inplace=True)
        return df
        
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
    # print(ds.get_data())
    df = ds.get_data()
    # print(df['returns'])
    print(np.exp(df['returns'].sum()))
    # for i in range(10):
    #     print(ds.step())