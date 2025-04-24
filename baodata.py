# from mootdx.affair import Affair


# df = Affair.parse(downdir="fin_data", filename="gpcw20150930.zip")


# df.to_csv('gpcw20150930.csv')


import akshare as ak


df = ak.fund_etf_fund_daily_em()

df.to_csv("tmp.csv")