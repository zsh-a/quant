# import akshare as ak

# # index_stock_cons_df = ak.index_stock_cons(symbol="399101")
# # index_stock_cons_df.to_csv("tmp.csv")

# import pandas as pd
# import clickhouse_connect

# client = clickhouse_connect.get_client(
#     host="localhost", username="default", password=""
# )

# print(client.command("show databases"))


# def insert_index_stocks(index_code):
#     df = pd.read_csv("tmp.csv",dtype=str)
#     del df["Unnamed: 0"]
#     for index, row in df.iterrows():
#         print(row)
#         code = row["品种代码"]
#         name = row["品种名称"]
#         if code.startswith("6"):
#             code = "sh." + code
#         else:
#             code = "sz." + code
#         sql = f"""
#         INSERT INTO stock_data.index_stocks (index,code,enter_date)
#         VALUES ('{index_code}','{code}','{row["纳入日期"]}')
#         """
#         client.command(sql)

# if __name__ == "__main__":
#     insert_index_stocks("399101")

import akshare as ak

stock_financial_report_sina_df = ak.stock_financial_report_sina(stock="sh601898", symbol="利润表")
stock_financial_report_sina_df.to_csv("tmp_ak.csv")
print(stock_financial_report_sina_df)