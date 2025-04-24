# from mootdx.affair import Affair


# df = Affair.parse(downdir="fin_data", filename="gpcw20150930.zip")


# df.to_csv('gpcw20150930.csv')
import pandas as pd
import akshare as ak
all_etfs = pd.read_csv("all_etf.csv",names=["基金代码","类别","名称"])
all_etfs = all_etfs['基金代码'].astype(str).to_list()
for code in all_etfs:
    df = ak.fund_etf_hist_em(code,adjust="hfq")
    df.to_csv(f"{code}.csv",index=False)
