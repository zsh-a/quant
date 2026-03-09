from io import StringIO
import numpy as np
import pandas as pd
import os
import sys

import requests
from bs4 import BeautifulSoup
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 将外层目录添加到 sys.path
sys.path.append(parent_dir)

from src.market_data.db import DB


def log2percent(x):
    return np.round(x * 100 - 100, 2)


name_cache = {}


def get_name(symbol):
    if symbol not in name_cache:
        db = DB()
        name_cache[symbol] = db.get_meta(symbol).iloc[0]["name"]

    return name_cache[symbol]


def get_sw_industry_code(name_list):
    df = pd.read_csv("sw_industry.csv", index_col="name")

    return [code.split(".")[0] for code in df.loc[name_list]["index"].to_list()]


def get_sw_comoment(index):
    url = f"https://legulegu.com/stockdata/index-composition?industryCode={index}.SI"

    params = {"page": "1", "page_size": "10000"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, params=params, headers=headers, verify=False)

    # 解析HTML
    soup = BeautifulSoup(resp.text, "html.parser")

    # 找到表格
    table = soup.find("table", class_="table")
    if not table:
        print("未找到表格")
        return None
    # 提取表头
    headers = [th.text.strip() for th in table.find("thead").find_all("th")]

    # 提取表格数据
    data = []
    for row in table.find("tbody").find_all("tr"):
        cols = [td.text.strip() for td in row.find_all("td")]
        data.append(cols)

    # 创建DataFrame
    df = pd.DataFrame(data, columns=headers)

    return df


def index_stock_cons(symbol: str = "399639") -> pd.DataFrame:
    """
    最新股票指数的成份股目录
    https://vip.stock.finance.sina.com.cn/corp/view/vII_NewestComponent.php?page=1&indexid=399639
    :param symbol: 指数代码, 可以通过 ak.index_stock_info() 函数获取
    :type symbol: str
    :return: 最新股票指数的成份股目录
    :rtype: pandas.DataFrame
    """
    url = f"https://vip.stock.finance.sina.com.cn/corp/go.php/vII_NewestComponent/indexid/{symbol}.phtml"
    r = requests.get(url)
    r.encoding = "gb2312"
    soup = BeautifulSoup(r.text, "lxml")
    page_num = (
        soup.find(attrs={"class": "table2"})
        .find("td")
        .find_all("a")[-1]["href"]
        .split("page=")[-1]
        .split("&")[0]
    )
    if page_num == "#":
        temp_df = pd.read_html(StringIO(r.text), header=0, skiprows=1)[3].iloc[:, :3]
        temp_df["品种代码"] = temp_df["品种代码"].astype(str).str.zfill(6)
        return temp_df
    
    print(page_num)

    temp_df = pd.DataFrame()
    for page in range(1, int(page_num) + 1):
        url = f"https://vip.stock.finance.sina.com.cn/corp/view/vII_NewestComponent.php?page={page}&indexid={symbol}"
        r = requests.get(url)
        r.encoding = "gb2312"

        pd.read_html(StringIO(r.text), header=1)[3].to_csv(f"temp_{page}.csv", index=False)

        temp_df = pd.concat(
            objs=[temp_df, pd.read_html(StringIO(r.text), header=1)[3]],
            ignore_index=True,
        )
    temp_df = temp_df.iloc[:, :3]
    temp_df["品种代码"] = temp_df["品种代码"].astype(str).str.zfill(6)
    return temp_df


if __name__ == "__main__":
    # print(get_sw_industry_code(["银行"]))
    index_stock_cons("000985").to_csv("index_stock_cons.csv")
