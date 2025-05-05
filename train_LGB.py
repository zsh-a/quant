import os
import pickle
from db import DB
from utils import *
import torch
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from models import LeNet, TransformerModel
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import talib
import pandas as pd
from sklearn.metrics import r2_score
factors = [
    "momentum_5",
    "vol_ratio",
    "RSI_14",
    "BB_upper",
    "BB_lower",
    "volatility_20",
    "turnover_avg_20",
    "obv",
    "turnover_5",
]


def calc_factors(df):
    # 动量因子: 过去5日涨跌幅
    df["momentum_5"] = df["close"] / df["close"].shift(5) - 1

    # 成交量因子: (最近5日平均成交量) / (最近10日平均成交量) - 1

    df["vol_ratio"] = (df["volume"].rolling(5).mean()) / (
        df["volume"].rolling(10).mean()
    ) - 1
    # 计算RSI (默认周期14)
    df["RSI_14"] = talib.RSI(df["close"], timeperiod=14)

    # 布林带
    upper, middle, lower = talib.BBANDS(
        df["close"],
        timeperiod=20,
        nbdevup=2,
        nbdevdn=2,
        matype=0,
    )
    df["BB_upper"] = upper
    df["BB_middle"] = middle
    df["BB_lower"] = lower

    # 反转因子 = -动量因子
    df["reversal_5"] = -df["close"].pct_change(periods=5)

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility_20"] = df["log_ret"].rolling(20).std() * np.sqrt(252)

    df["turnover_avg_20"] = df["turn"].rolling(20).mean()

    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()

    df["turnover_5"] = df["turn"].rolling(5).sum()
    df["future_ret_1d"] = df["close"].shift(-5) / df["close"] - 1

    return df


def build_dataset(index, start_date, end_data):
    cache_file = f"dataset_cache_{index}_{start_date}_{end_data}.pkl"

    # 检查缓存文件是否存在
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    db_client = DB()
    stocks = (
        db_client.get_index_stocks(index)
        if len(index) == 6
        else db_client.get_all_etf_code()
    )
    fields = ["open", "high", "low", "close", "volume", "turn"]
    all_data = db_client.get_price(stocks, end_data, fields, 99999, True, start_date)

    all_data = (
        all_data.groupby("code", group_keys=False)
        .apply(lambda df: calc_factors(df))
        .dropna()
    )
    all_data = all_data.reset_index("date")

    train_date = "2023-01-01"

    X_train = all_data[all_data["date"] < train_date][factors]
    Y_train = all_data[all_data["date"] < train_date]["future_ret_1d"]
    X_valid = all_data[all_data["date"] >= train_date][factors]
    Y_valid = all_data[all_data["date"] >= train_date]["future_ret_1d"]

    print(X_train.shape)
    # normalize
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)

    # corr = all_data[['close']+factors].corr()  # JayBee黄量化模型
    # plt.figure(figsize=(8, 6))  # 本代码归JayBee黄所有
    # sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)  # Copyright © JayBee黄
    # plt.title('因子与目标变量相关性')  # JayBee黄版权所有，未经授权禁止复制
    # plt.savefig('correlation.png')  # JayBee黄版权所有，未经授权禁止复制

    return X_train, Y_train, X_valid, Y_valid, scaler


if __name__ == "__main__":
    X_train, Y_train, X_valid, Y_valid, scaler = build_dataset(
        "etf", "2018-01-01", "2026-01-01"
    )
    
    # save scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"X_train shape: {X_train.shape} X_valid shape: {X_valid.shape}")
    # 创建LGB数据集
    train_dataset = lgb.Dataset(X_train, label=Y_train)
    valid_dataset = lgb.Dataset(X_valid, label=Y_valid, reference=train_dataset)

    # 设置LGB参数
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'max_depth': -1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': 0,
        'early_stopping_rounds': 50
    }

    # 训练模型
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=1000,
        valid_sets=[train_dataset, valid_dataset],
    )

    y_train = model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(Y_train, y_train))
    r2 = r2_score(Y_train, y_train)
    print(f'Train R2: {r2}')
    print(f'Train RMSE: {rmse}')

    # 在测试集上评估
    y_pred = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(Y_valid, y_pred))
    r2 = r2_score(Y_valid, y_pred)
    print(f'Test R2: {r2}')
    print(f'Test RMSE: {rmse}')

    # 保存模型
    model.save_model('lgb_model.txt')
