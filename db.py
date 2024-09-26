import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import datetime
import pandas as pd
import akshare as ak
import schedule
from loguru import logger

bucket = "stock"
token = os.environ.get("INFLUXDB_TOKEN")
org = "zs"
url = "http://localhost:8086"

client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)


def write_data():
    write_api = client.write_api(write_options=SYNCHRONOUS)

    for file in os.listdir("data/combine"):
        code = file.split(".")[0]
        df = pd.read_csv(os.path.join("data", "combine", f"{code}.csv"))
        print(df.iloc[-3:-1])

        # break
        # df = df.iloc[-3:-1]
        df["ticker"] = f"{code}"

        df["datetime"] = pd.to_datetime(
            df["datetime"]
        )  # 确保 timestamp 是 datetime 类型

        df["adj_factor"] = df["close"].shift(1) / df["pre_close"]
        df["adj_factor"].iloc[0] = 1

        df["adj_factor"] = df["adj_factor"].cumprod()
        print(code, df)

        # 将 pandas DataFrame 插入 InfluxDB
        for index, row in df.iterrows():
            point = (
                Point("stock_data")
                .tag("ticker", row["ticker"])
                .field("open", row["open"])
                .field("high", row["high"])
                .field("low", row["low"])
                .field("close", row["close"])
                .field("pre_close", row["pre_close"])
                .field("amount", row["amount"])
                .field("volume", row["volume"])
                .time(row["datetime"], WritePrecision.S)
            )
            write_api.write(bucket=bucket, org="zs", record=point)


def read_data():
    query_api = client.query_api()

    query = f"""
        from(bucket: "{bucket}")
        |> range(start: -30d)
        |> filter(fn: (r) => r._measurement == "stock_data" and r.ticker == "510880")
        |> filter(fn: (r) => r._field == "close" or r._field == "pre_close")
    """
    tables = query_api.query(query, org="zs")

    # 处理查询结果
    results = {}

    for table in tables:
        for record in table.records:
            time = record.get_time()
            field = record.get_field()
            value = record.get_value()

            if time not in results:
                results[time] = {}

            results[time][field] = value

    # 打印结果
    for time, fields in results.items():
        print(f"Time: {time}, Data: {fields}")


def update_data():
    try:
        df = ak.fund_etf_spot_em()
        now = datetime.datetime.now()
        name = f"daily/etf_day_{now}.csv"
        df.to_csv(name)
        all_etf = pd.read_csv("all_etf.csv")

        all_etf.columns = ["代码", "1", "2"]
        # print(all_etf)
        all_etf = all_etf.set_index("代码")
        df = pd.read_csv(name, index_col="代码")
        # df.set_index('代码',inplace=True)
        df = df.loc[all_etf.index.values]
        # print(df)
        df["成交量"] = df["成交量"].astype(int)
        date = pd.to_datetime(df["数据日期"].unique()[0])

        write_api = client.write_api(write_options=SYNCHRONOUS)

        for index, row in df.iterrows():
            # query = f'''
            #     from(bucket: "{bucket}")
            #     |> range(start: -30d)
            #     |> filter(fn: (r) => r._measurement == "stock_data" and r.ticker == "{index}")
            #     |> filter(fn: (r) => r._field == "close")
            #     |> last()
            # '''
            # close = 1.
            # tables = query_api.query(query)
            # # 处理查询结果
            # for table in tables:
            #     for record in table.records:
            #         close = record.get_value()
            point = (
                Point("stock_data")
                .tag("ticker", index)
                .field("open", row["开盘价"])
                .field("high", row["最高价"])
                .field("low", row["最低价"])
                .field("close", row["最新价"])
                .field("pre_close", row["昨收"])
                .field("amount", row["成交额"])
                .field("volume", row["成交量"])
                .time(date, WritePrecision.S)
            )
            # print(point)
            write_api.write(bucket=bucket, org="zs", record=point)
        print(f"{datetime.datetime.now()} update completed")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    update_data()

    schedule.every().day.at(f"{18:02d}:{00:02d}").do(update_data)
    while True:
        schedule.run_pending()
        time.sleep(1)
    # write_data()
