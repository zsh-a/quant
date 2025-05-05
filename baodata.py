# from mootdx.affair import Affair


# df = Affair.parse(downdir="fin_data", filename="gpcw20150930.zip")


# df.to_csv('gpcw20150930.csv')
import pandas as pd
import akshare as ak
all_etfs = pd.read_csv("all_etf.csv", names=["code", "type", "name"])
all_etfs = all_etfs.astype(str)
all_etfs = all_etfs.set_index("code")
# print(all_etfs)
print(all_etfs.loc["513050"]['name'])