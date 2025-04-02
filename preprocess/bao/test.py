import clickhouse_connect
from pre_process import fetch_update

client = clickhouse_connect.get_client(
    host="localhost", username="default", password=""
)
import baostock as bs
lg = bs.login()
query = """
    SELECT
        code,
        last_update_date,
        last_adjfactor,
        error_update_count
    FROM  
        stock_data.stock_daily_meta
    WHERE code = 'sh.600718'
"""

df = client.query(query).result_rows

for code, last_update_date, last_adjfactor, error_update_count in df:
    print(fetch_update(code, last_update_date, last_adjfactor))