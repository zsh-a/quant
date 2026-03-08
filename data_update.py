import json
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

import preprocess.ak_utils as ak
import preprocess.bao.pre_process as bao
from db import DB
from src.utils.tdx_utils import TDXProcess

INDEX_LIST = [
    "000985",
    "399673",
    "399101",
]

DEFAULT_SHARE_START_DATE = "20250101"
REFERENCE_SYMBOL = "sh.000300"


class DataUpdateError(Exception):
    pass



def get_reference_latest_date(symbol: str = REFERENCE_SYMBOL) -> Optional[str]:
    try:
        db = DB()
        query = f"SELECT max(date) AS latest_date FROM stock_data.stock_daily WHERE code = '{symbol}'"
        result = db.client.query(query)
        if not result.result_rows:
            return None
        latest = result.result_rows[0][0]
        if latest is None:
            return None
        if hasattr(latest, "strftime"):
            return latest.strftime("%Y-%m-%d")
        return str(latest)
    except Exception:
        return None



def update_kline_daily() -> Dict:
    proc = bao.BaoStockProcessor()
    proc.update_daily_data()
    return {"message": "daily kline updated"}



def update_index_stocks_weekly() -> Dict:
    proc = ak.AKDataProcessor()
    for index in INDEX_LIST:
        proc.insert_index_stocks(index)
    return {"updated_indexes": INDEX_LIST}



def update_industry_weekly() -> Dict:
    proc = ak.AKDataProcessor()
    proc.insert_sw_industry()
    return {"message": "industry mapping updated"}



def update_financial() -> Dict:
    proc = TDXProcess()
    proc.update_fincial_db()
    return {"message": "financial data updated"}



def update_share_info(start_date: str = DEFAULT_SHARE_START_DATE) -> Dict:
    proc = ak.AKDataProcessor()
    proc.update_shares(start_date=start_date)
    return {"start_date": start_date, "message": "share info updated"}



def update_etf_kline() -> Dict:
    proc = ak.AKDataProcessor()
    all_etfs = pd.read_csv("all_etf.csv", names=["基金代码", "类别", "名称"])
    all_etfs = all_etfs["基金代码"].astype(str).to_list()
    updated = 0
    errors: List[str] = []
    for code in all_etfs:
        try:
            proc.update_etf_data(code)
            updated += 1
        except Exception as exc:
            errors.append(f"{code}: {exc}")
        time.sleep(1)
    proc.create_etf_meta()
    return {"updated": updated, "errors": errors[:20]}


UPDATE_STEP_DEFINITIONS = [
    {
        "key": "financial",
        "label": "财务数据",
        "description": "更新财报与衍生财务指标",
        "default_selected": True,
        "builder": update_financial,
    },
    {
        "key": "kline_daily",
        "label": "日线行情",
        "description": "更新股票日线行情与复权数据",
        "default_selected": True,
        "builder": update_kline_daily,
    },
    {
        "key": "share_info",
        "label": "股本信息",
        "description": "更新总股本与流通股本数据",
        "default_selected": True,
        "builder": update_share_info,
    },
    {
        "key": "industry_weekly",
        "label": "行业映射",
        "description": "更新申万行业映射",
        "default_selected": False,
        "builder": update_industry_weekly,
    },
    {
        "key": "index_stocks_weekly",
        "label": "指数成分",
        "description": "更新主要指数成分股列表",
        "default_selected": False,
        "builder": update_index_stocks_weekly,
    },
    {
        "key": "etf_kline",
        "label": "ETF 行情",
        "description": "更新 ETF 日线与元数据",
        "default_selected": False,
        "builder": update_etf_kline,
    },
]

UPDATE_STEP_BUILDERS: Dict[str, Callable[..., Dict]] = {
    item["key"]: item["builder"] for item in UPDATE_STEP_DEFINITIONS
}

DEFAULT_UPDATE_STEPS = [
    item["key"] for item in UPDATE_STEP_DEFINITIONS if item["default_selected"]
]


def get_update_step_capabilities() -> Dict[str, List[Dict[str, object]]]:
    return {
        "steps": [
            {
                "key": item["key"],
                "label": item["label"],
                "description": item["description"],
                "default_selected": item["default_selected"],
            }
            for item in UPDATE_STEP_DEFINITIONS
        ],
        "default_selected_steps": DEFAULT_UPDATE_STEPS,
        "share_start_date_default": DEFAULT_SHARE_START_DATE,
        "reference_symbol": REFERENCE_SYMBOL,
    }



def run_data_update_pipeline(
    selected_steps: Optional[List[str]] = None,
    share_start_date: str = DEFAULT_SHARE_START_DATE,
) -> Dict:
    steps = selected_steps or DEFAULT_UPDATE_STEPS
    before_date = get_reference_latest_date()
    started_at = datetime.now().isoformat()
    results = []
    errors = []

    for step_name in steps:
        func = UPDATE_STEP_BUILDERS.get(step_name)
        if func is None:
            errors.append({"step": step_name, "error": "unknown step"})
            results.append(
                {"step": step_name, "status": "error", "error": "unknown step"}
            )
            continue

        step_started = time.time()
        try:
            if step_name == "share_info":
                detail = func(start_date=share_start_date)
            else:
                detail = func()
            results.append(
                {
                    "step": step_name,
                    "status": "success",
                    "duration_seconds": round(time.time() - step_started, 2),
                    "detail": detail,
                }
            )
        except Exception as exc:
            errors.append({"step": step_name, "error": str(exc)})
            results.append(
                {
                    "step": step_name,
                    "status": "error",
                    "duration_seconds": round(time.time() - step_started, 2),
                    "error": str(exc),
                }
            )

    after_date = get_reference_latest_date()
    has_new_data = bool(before_date and after_date and after_date != before_date)
    if before_date is None and after_date is not None:
        has_new_data = True

    status = "success"
    if errors and len(errors) == len(results):
        status = "failed"
    elif errors:
        status = "partial_success"

    return {
        "status": status,
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(),
        "reference_symbol": REFERENCE_SYMBOL,
        "before_latest_date": before_date,
        "after_latest_date": after_date,
        "has_new_data": has_new_data,
        "steps": results,
        "errors": errors,
    }


if __name__ == "__main__":
    result = run_data_update_pipeline()
    print(json.dumps(result, ensure_ascii=False, indent=2))
