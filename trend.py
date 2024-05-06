import pandas as pd
import numpy as np

PREDICT_KS = 5
short_window = 12
long_window = 26
signal_window = 9


file_path = '/home/zs/workspace/exp/quent/data/510880.csv'
df = pd.read_csv(file_path)
df.columns = ['datetime','open','high','low','close','volume','amount']
df.set_index('datetime', inplace=True)
df.index = pd.to_datetime(df.index)
df = df['2010':]
df = df[['open','high','low','close','volume']]
df = df.astype(float)

# df['change'] = df['close'].pct_change()

df['amplitude'] = ((df['high'] - df['low']) / df['close'].shift(1))
df['returns'] = np.log(df['close'] / df['close'].shift(1))
df['returns_5d'] = (df['close'].shift(-PREDICT_KS) - df['close']) / df['close']

# 计算短期EMA和长期EMA
df["EMA_12"] = df["close"].ewm(span=short_window, adjust=False).mean()
df["EMA_26"] = df["close"].ewm(span=long_window, adjust=False).mean()

# 计算MACD值
df["MACD"] = df["EMA_12"] - df["EMA_26"]

# 计算信号线
df["Signal_Line"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()

# MACD Histogram，可选，代表MACD与信号线之间的差值
df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]


# 使用 cut 函数生成标签
# df['label'] = pd.cut(df['returns_5d'], bins=bins, labels=labels)

df['sig'] = (df['MACD_Histogram'].shift(-1) - df['MACD_Histogram'])

df['label'] = (df['sig'] * df['returns_5d'] > 0).astype(int)

df.dropna(inplace=True)
# print(df)

data = df["label"].to_frame()
data.reset_index(inplace=True)
train_size = int(len(df) * 0.8)
data = data.iloc[train_size:]

count_of_ones = (data['label'] == 1).sum()
print(data)

print("acc :", count_of_ones / len(data))

