import baostock as bs
import pandas as pd


# #### 登陆系统 ####
# lg = bs.login()
# # 显示登陆返回信息
# print("login respond error_code:" + lg.error_code)
# print("login respond  error_msg:" + lg.error_msg)

# df = pd.read_csv("hs300_stocks.csv")
# for code in df["code"]:
#     mk = code.split(".")[1][0]
#     if mk == "3":
#         continue

#     #### 获取沪深A股历史K线数据 ####
#     # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
#     # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
#     # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg

#     rs = bs.query_history_k_data_plus(
#         code,
#         "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
#         start_date="2010-01-01",
#         end_date="2024-08-20",
#         frequency="d",
#         adjustflag="3",
#     )
#     print("query_history_k_data_plus respond error_code:" + rs.error_code)
#     print("query_history_k_data_plus respond  error_msg:" + rs.error_msg)

#     #### 打印结果集 ####
#     data_list = []
#     while (rs.error_code == "0") & rs.next():
#         # 获取一条记录，将记录合并在一起
#         data_list.append(rs.get_row_data())
#     result = pd.DataFrame(data_list, columns=rs.fields)

#     #### 结果集输出到csv文件 ####
#     result.to_csv(f"data/bao/{code.split('.')[1]}.csv", index=False)
#     # break
#     # print(result)

# #### 登出系统 ####
# bs.logout()


import baostock as bs
import pandas as pd

# 登陆系统
lg = bs.login()
# 显示登陆返回信息
print("login respond error_code:" + lg.error_code)
print("login respond  error_msg:" + lg.error_msg)

# 获取沪深300成分股
rs = bs.query_(day="2024-09-30")
print("query_hs300 error_code:" + rs.error_code)
print("query_hs300  error_msg:" + rs.error_msg)

# 打印结果集
all_etf_stocks = []
while (rs.error_code == "0") & rs.next():
    if rs.get_row_data()[2].startswith("510") or rs.get_row_data()[2].startswith("159"):
        # 获取一条记录，将记录合并在一起
        all_etf_stocks.append(rs.get_row_data())
result = pd.DataFrame(all_etf_stocks, columns=rs.fields)
# 结果集输出到csv文件
result.to_csv("bao_all_etf.csv", encoding="utf-8", index=False)
print(result)

# 登出系统
bs.logout()



# #### 登陆系统 ####
# lg = bs.login()
# # 显示登陆返回信息
# print("login respond error_code:" + lg.error_code)
# print("login respond  error_msg:" + lg.error_msg)


# #### 获取沪深A股历史K线数据 ####
# # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
# # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
# # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg

# rs = bs.query_history_k_data_plus(
#     "sz.159755",
#     "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
#     start_date="2010-01-01",
#     end_date="2024-08-20",
#     frequency="d",
#     adjustflag="3",
# )
# print("query_history_k_data_plus respond error_code:" + rs.error_code)
# print("query_history_k_data_plus respond  error_msg:" + rs.error_msg)

# #### 打印结果集 ####
# data_list = []
# while (rs.error_code == "0") & rs.next():
#     # 获取一条记录，将记录合并在一起
#     data_list.append(rs.get_row_data())
# result = pd.DataFrame(data_list, columns=rs.fields)

# #### 结果集输出到csv文件 ####
# # result.to_csv(f"data/bao/{code.split('.')[1]}.csv", index=False)
# print(result)
# # break
# # print(result)

# #### 登出系统 ####
# bs.logout()
