import pandas as pd
import os

def find_last_trading_day(csv_path):
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 将日期列转换为datetime类型
    df['calendar_date'] = pd.to_datetime(df['calendar_date'])
    # 添加月份列
    df['month'] = df['calendar_date'].dt.month
    # 按周分组，并找出每周最后一个交易日
    df['week'] = df['calendar_date'].dt.isocalendar().week
    df['year'] = df['calendar_date'].dt.isocalendar().year
    # 筛选交易日
    trading_days = df[df['is_trading_day'] == 1]

    # 找出每周最后一个交易日
    last_trading_days = trading_days.groupby(['year', 'week']).last().reset_index()

    # 找出每月最后一个交易日
    last_trading_days_monthly = trading_days.groupby(['year', 'month']).last().reset_index()

    # 标记原始数据中的最后一个交易日
    df['is_last_trading_day'] = 0
    df['is_last_trading_day_monthly'] = 0
    df.loc[df['calendar_date'].isin(last_trading_days['calendar_date']), 'is_last_trading_day'] = 1
    df.loc[df['calendar_date'].isin(last_trading_days_monthly['calendar_date']), 'is_last_trading_day_monthly'] = 1

    # 保存结果
    output_path = os.path.join(os.path.dirname(csv_path), 'marked_trade_datas.csv')
    df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}")

    return df

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), 'trade_datas.csv')
    find_last_trading_day(csv_path)
