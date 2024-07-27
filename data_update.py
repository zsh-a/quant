import preprocess.bao.pre_process as bao
import preprocess.ak_utils as ak
import tdx_utils as tdx
import pandas as pd
import time
INDEX_LIST = [
    "000985",
    "399673", # 创业板
    "399101" # 中小板 
]

last_update_date = "20250101"


def update_kline_daily():
    proc = bao.BaoStockProcessor()
    proc.update_daily_data()


def update_index_stocks_weekly():
    proc = ak.AKDataProcessor()
    for index in INDEX_LIST:
        proc.insert_index_stocks(index)


def update_industry_weekly():
    proc = ak.AKDataProcessor()
    proc.insert_sw_industry()


def update_fincial():
    proc = tdx.TDXProcess()
    proc.fetch_tdx()
    proc.init_fincial_db(start_year="2024")


def update_share_info():
    proc = ak.AKDataProcessor()
    proc.update_shares(start_date=last_update_date)


def update_etf_kline():
    proc = ak.AKDataProcessor()
    all_etfs = pd.read_csv("all_etf.csv", names=["基金代码", "类别", "名称"])
    all_etfs = all_etfs["基金代码"].astype(str).to_list()
    for code in all_etfs:
        try:
            proc.update_etf_data(code)
        except Exception as e:
            print(f"Error updating code {code}: {e}")
            continue
        time.sleep(1)
    proc.create_etf_meta()


if __name__ == "__main__":
    # update_fincial()
    # update_kline_daily()
    #update_industry_weekly()
    #update_etf_kline()
    # update_share_info()
    # update_index_stocks_weekly()
    update_etf_kline()