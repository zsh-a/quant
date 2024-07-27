# import akshare as ak
# import pandas as pd

# # 获取交易日期数据
# tool_trade_date_hist_sina_df = ak.tool_trade_date_hist_sina()

# # 确保 trade_date 列是日期类型
# tool_trade_date_hist_sina_df['trade_date'] = pd.to_datetime(tool_trade_date_hist_sina_df['trade_date'])

# # 获取日期范围
# start_date = tool_trade_date_hist_sina_df['trade_date'].min()
# end_date = tool_trade_date_hist_sina_df['trade_date'].max()

# # 生成从开始日期到结束日期的所有日期
# all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# # 创建 DataFrame
# result_df = pd.DataFrame({
#     'calendar_date': all_dates.strftime('%Y-%m-%d'),
#     'is_trading_day': 0
# })

# # 将交易日标记为 1
# trading_dates_set = set(tool_trade_date_hist_sina_df['trade_date'].dt.strftime('%Y-%m-%d'))
# result_df.loc[result_df['calendar_date'].isin(trading_dates_set), 'is_trading_day'] = 1

# # 写入 CSV 文件
# output_file = 'trade_datas.csv'
# result_df.to_csv(output_file, index=False)

# print(f"数据已写入 {output_file}")
# print(f"共 {len(result_df)} 行数据，其中交易日 {result_df['is_trading_day'].sum()} 天")
# print("\n前10行数据：")
# print(result_df.head(10))

from mootdx.quotes import Quotes

# 1. 创建客户端连接（推荐使用 std 标准市场）
client = Quotes.factory(market='std')

# 2. 定义参数
# 示例：获取 沪深300ETF (510300) 的日线数据
etf_code = '159919' 

# frequency (K线周期): 
# 9=日线, 5=周线, 6=月线
# 0=5分钟, 1=15分钟, 2=30分钟, 3=1小时, 4=日线(旧版), 7=1分钟
# count: 获取的数据条数
data = client.bars(symbol=etf_code, frequency=9, offset=0, count=100)

# 3. 打印结果
print(f"ETF {etf_code} 历史行情:")
print(data)

# 如果想保存为 CSV
# data.to_csv('510300_daily.csv')