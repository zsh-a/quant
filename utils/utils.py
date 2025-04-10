import numpy as np
import pandas as pd
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 将外层目录添加到 sys.path
sys.path.append(parent_dir)

from db import DB

def log2percent(x):
    return np.round(x * 100 - 100, 2)


name_cache = {}

def get_name(symbol):
    if symbol not in name_cache:
        db = DB()
        name_cache[symbol] = db.get_meta(symbol).iloc[0]["name"]

    return name_cache[symbol]

def get_sw_industry_code(name_list):
    df = pd.read_csv("sw_industry.csv",index_col="name")

    return [code.split('.')[0] for code in df.loc[name_list]['index'].to_list()]


if __name__ == "__main__":
    print(get_sw_industry_code(["银行"]))