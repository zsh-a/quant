from mootdx.affair import Affair
from datetime import datetime
import pandas as pd
import clickhouse_connect

client = clickhouse_connect.get_client(
    host="localhost", username="default", password=""
)

# files = Affair.files()

# for item in files:
#     Affair.fetch(downdir='fin_data', filename=f'{item["filename"]}')

def update_fincial_db(path="fin_data", start_year="2009"):
    dates = ["0331","0630","0930","1231"]
    now_year = datetime.now().year
    years = [str(year) for year in range(int(start_year), now_year+1)]
    dates = [year + date for year in years for date in dates]
    def convert_to_date(num):
        try:
            date_str = f"{int(num):06d}"  # 去除小数点并补零至6位（如250315）
            return datetime.strptime(date_str, "%y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            return "无效日期"
    for date in dates:
        filename = f"gpcw{date}.zip"
        print(filename)
        df = Affair.parse(downdir=path, filename=filename)

        df['report_date'] = df['report_date'].apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").strftime("%Y-%m-%d"))
        df['publish_date'] = df['财报公告日期'].apply(convert_to_date)
        # print(df['publish_date'])
        df['net_profit'] = df['五、净利润']
        df['roa'] = df['净资产收益率'].iloc[:, [0]] # 保留第一列
        df['adjusted_profit'] = df['扣除非经常性损益后的净利润'].iloc[:, [0]]
        df['total_shares'] = df['总股本']
        df['circulating_a'] = df["已上市流通A股"]
        # df['circulating_b'] = df["已上市流通B股"]
        # df['circulating_h'] = df["已上市流通H股"]
        
        # 计算更多财务指标
        df['gross_profit_margin'] = df['销售毛利率(%)(非金融类指标)']

        for code,row in df.iterrows():
            if not code.startswith('6') and not code.startswith('0') and not code.startswith("3"):
                continue
            if code.startswith("6"):
                code = "sh." + code
            else:
                code = "sz." + code

            sql = f"""
            INSERT INTO stock_data.finicial_report
            (
                report_date,
                code,
                publish_date,
                net_profit,
                adjusted_profit,
                roa,
                total_shares,
                circulating_a
            )
            VALUES
            (
                '{row['report_date']}',
                '{code}',
                '{row['publish_date']}',
                {row['net_profit']},
                {row['adjusted_profit']},
                {row['roa']},
                {row['total_shares']},
                {row['circulating_a']}
            )"""
            client.command(sql)
    

if __name__ == "__main__":
    update_fincial_db()

