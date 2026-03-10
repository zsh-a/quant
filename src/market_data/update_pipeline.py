import json
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.market_data.db import DB
from src.market_data.processors.akshare import AKDataProcessor
from src.market_data.processors.baostock import BaoStockProcessor
from src.market_data.processors.tdx import TDXProcess

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


def update_kline_daily(progress_callback: Optional[Callable[[Dict[str, object]], None]] = None) -> Dict:
    proc = BaoStockProcessor()

    def on_batch_progress(completed_batches: int, total_batches: int, fetched_count: int):
        if not progress_callback or total_batches <= 0:
            return
        progress = round((completed_batches / total_batches) * 100, 2)
        progress_callback(
            {
                "event": "step_progress",
                "step": "kline_daily",
                "progress": progress,
                "current": completed_batches,
                "total": total_batches,
                "fetched": fetched_count,
            }
        )

    proc.update_daily_data(progress_callback=on_batch_progress)
    return {"message": "daily kline updated"}


def update_index_stocks_weekly(
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None
) -> Dict:
    proc = AKDataProcessor()
    for index in INDEX_LIST:
        proc.insert_index_stocks(index)
    return {"updated_indexes": INDEX_LIST}


def update_industry_weekly(
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None
) -> Dict:
    proc = AKDataProcessor()
    proc.insert_sw_industry()
    return {"message": "industry mapping updated"}


def update_financial(
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None
) -> Dict:
    proc = TDXProcess()
    proc.update_fincial_db()
    return {"message": "financial data updated"}


def update_share_info(
    start_date: str = DEFAULT_SHARE_START_DATE,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict:
    proc = AKDataProcessor()
    proc.update_shares(start_date=start_date)
    return {"start_date": start_date, "message": "share info updated"}


def update_etf_kline(
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None
) -> Dict:
    proc = AKDataProcessor()
    all_etfs = pd.read_csv("all_etf.csv", names=["基金代码", "类别", "名称"])
    all_etfs = all_etfs["基金代码"].astype(str).to_list()
    total = len(all_etfs)
    updated = 0
    errors: List[str] = []
    for idx, code in enumerate(all_etfs, start=1):
        try:
            proc.update_etf_data(code)
            updated += 1
        except Exception as exc:
            errors.append(f"{code}: {exc}")
        time.sleep(1)
        if progress_callback and total > 0:
            progress = round((idx / total) * 100, 2)
            progress_callback(
                {
                    "event": "step_progress",
                    "step": "etf_kline",
                    "progress": progress,
                    "current": idx,
                    "total": total,
                    "updated": updated,
                    "errors": len(errors),
                }
            )
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
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict:
    steps = selected_steps or DEFAULT_UPDATE_STEPS
    before_date = get_reference_latest_date()
    started_at = datetime.now().isoformat()
    results = []
    errors = []
    total_steps = len(steps)
    step_label_map = {item["key"]: item.get("label", item["key"]) for item in UPDATE_STEP_DEFINITIONS}

    def emit_progress(payload: Dict[str, object]):
        if progress_callback:
            progress_callback(payload)

    for step_index, step_name in enumerate(steps, start=1):
        func = UPDATE_STEP_BUILDERS.get(step_name)
        step_label = step_label_map.get(step_name, step_name)
        if func is None:
            errors.append({"step": step_name, "error": "unknown step"})
            results.append(
                {"step": step_name, "status": "error", "error": "unknown step"}
            )
            emit_progress(
                {
                    "event": "step_failed",
                    "step": step_name,
                    "label": step_label,
                    "error": "unknown step",
                    "index": step_index,
                    "total_steps": total_steps,
                }
            )
            continue

        emit_progress(
            {
                "event": "step_started",
                "step": step_name,
                "label": step_label,
                "index": step_index,
                "total_steps": total_steps,
                "timestamp": datetime.now().isoformat(),
            }
        )
        step_started = time.time()
        try:
            if step_name == "share_info":
                detail = func(start_date=share_start_date, progress_callback=progress_callback)
            else:
                detail = func(progress_callback=progress_callback)
            results.append(
                {
                    "step": step_name,
                    "label": step_label,
                    "status": "success",
                    "duration_seconds": round(time.time() - step_started, 2),
                    "detail": detail,
                }
            )
            emit_progress(
                {
                    "event": "step_completed",
                    "step": step_name,
                    "label": step_label,
                    "status": "success",
                    "duration_seconds": round(time.time() - step_started, 2),
                    "detail": detail,
                    "index": step_index,
                    "total_steps": total_steps,
                }
            )
        except Exception as exc:
            errors.append({"step": step_name, "error": str(exc)})
            results.append(
                {
                    "step": step_name,
                    "label": step_label,
                    "status": "error",
                    "duration_seconds": round(time.time() - step_started, 2),
                    "error": str(exc),
                }
            )
            emit_progress(
                {
                    "event": "step_failed",
                    "step": step_name,
                    "label": step_label,
                    "status": "error",
                    "duration_seconds": round(time.time() - step_started, 2),
                    "error": str(exc),
                    "index": step_index,
                    "total_steps": total_steps,
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
    result = run_data_update_pipeline(["kline_daily"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
