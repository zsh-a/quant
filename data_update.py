import preprocess.bao.pre_process as bao
import preprocess.ak_utils as ak
import tdx_utils as tdx

INDEX_LIST = [
    "000985",
    # "399101"
]


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


def update_fincial_daily():
    proc = tdx.TDXProcess()
    # proc.fetch_tdx()
    proc.init_fincial_db(start_year="2024")


if __name__ == "__main__":
    update_fincial_daily()
    # update_kline_daily()
    # update_index_stocks_weekly()
    # update_industry_weekly()
