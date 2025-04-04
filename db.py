import numpy as np
import pandas as pd
import clickhouse_connect

from loguru import logger

client = clickhouse_connect.get_client(
    host="localhost", username="default", password=""
)


def get_kline(code, start_date, end_date):
    query = f"""
    SELECT *
    FROM stock_data.stock_daily
    WHERE code = '{code}'"""
    if start_date:
        query += f" AND date >= '{start_date}'"
    if end_date:
        query += f" AND date <= '{end_date}'"

    query += "ORDER BY date"

    logger.info(f"exec query : {query}")

    data = client.query(query)
    df = pd.DataFrame(data.result_rows, columns=data.column_names)

    df.rename(columns={"date": "datetime"}, inplace=True)
    df.set_index("datetime", inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def update_meta():
    df = pd.read_csv("all_stock.csv", index_col="code")

    for code, data in df.iterrows():
        # pass
        name = data["code_name"]
        update_query = f"""
        INSERT INTO stock_data.stock_daily_meta (code, last_update_date, last_adjfactor, error_update_count, name)
        SELECT 
            code,
            last_update_date,
            last_adjfactor,
            error_update_count,
            '{name}'
        FROM stock_data.stock_daily_meta
        WHERE code = '{code}';
        """
        client.command(update_query)
        print(code)


def get_meta(code):
    query = f"""
    SELECT *
    FROM stock_data.stock_daily_meta
    WHERE code = '{code}'
    """

    logger.info(f"exec query : {query}")
    data = client.query(query)
    assert len(data.result_rows) == 1
    df = pd.DataFrame(data.result_rows, columns=data.column_names)
    return df


def opt_table(table_name):
    query = f"""
    OPTIMIZE TABLE {table_name} FINAL;
    """
    client.command(query)


if __name__ == "__main__":
    # df = get_kline("sz.300059", "20220101", "20221231")

    # print(df)
    print(get_meta("sz.000001"))
