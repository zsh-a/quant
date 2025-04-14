# from mootdx.affair import Affair


# df = Affair.parse(downdir="fin_data", filename="gpcw20150930.zip")


# df.to_csv('gpcw20150930.csv')


import akshare as ak

index_stock_cons_df = ak.index_stock_cons_csindex(symbol="000985")
index_stock_cons_df.to_csv("index_stock_cons.csv")