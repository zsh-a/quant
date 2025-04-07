import baostock as bs
import pandas as pd
from datetime import datetime

# 更新行业数据
def update_industry_data(weekly=False):
    # 登陆系统
    lg = bs.login()
    print('login respond error_code:'+lg.error_code)
    print('login respond error_msg:'+lg.error_msg)

    # 获取行业分类数据
    if weekly:
        today = datetime.now().strftime('%Y-%m-%d')
        if datetime.now().weekday() != 0:  # 0代表周一
            print(f"今天是{today}，不是周一，跳过更新")
            bs.logout()
            return
        rs = bs.query_stock_industry(date=today)
    else:
        rs = bs.query_stock_industry(date="2020-01-01")
    print('query_stock_industry error_code:'+rs.error_code)
    print('query_stock_industry respond error_msg:'+rs.error_msg)

    # 处理结果
    industry_list = []
    while (rs.error_code == '0') & rs.next():
        industry_list.append(rs.get_row_data())
    result = pd.DataFrame(industry_list, columns=rs.fields)
    print(result)
    # 登出系统
    bs.logout()

if __name__ == "__main__":
    # create_industry_table()
    # 每周一自动更新
    # update_industry_data(weekly=True)
    update_industry_data()
    # 全量更新
    # update_industry_data()