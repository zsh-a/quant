"""Market database admin dashboard API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Optional

import anyio
from fastapi import APIRouter
from pydantic import BaseModel

from data_update import REFERENCE_SYMBOL, get_update_step_capabilities
from db import DB
from session_db import SessionDB
from src.tasks.automation import run_data_update_pipeline_task

router = APIRouter(prefix="/market-admin", tags=["market-admin"])

session_db = SessionDB()


class MarketAdminUpdateRequest(BaseModel):
    selected_steps: Optional[list[str]] = None
    share_start_date: Optional[str] = None


class MarketDbOverviewService:
    def __init__(self):
        self.db = DB()

    def _query_scalar(self, sql: str):
        result = self.db.client.query(sql)
        if not result.result_rows:
            return None
        return result.result_rows[0][0]

    def _format_date(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d")
        return str(value)

    def _table_summary(
        self,
        *,
        table: str,
        date_column: Optional[str],
        distinct_column: Optional[str] = "code",
    ) -> Dict[str, Any]:
        try:
            summary = {
                "table": table,
                "status": "ok",
                "row_count": int(self._query_scalar(f"SELECT count() FROM {table}") or 0),
                "distinct_count": None,
                "distinct_label": distinct_column,
                "earliest_date": None,
                "latest_date": None,
                "error": None,
            }
            if distinct_column:
                summary["distinct_count"] = int(
                    self._query_scalar(f"SELECT uniqExact({distinct_column}) FROM {table}") or 0
                )
            if date_column:
                summary["earliest_date"] = self._format_date(
                    self._query_scalar(f"SELECT min({date_column}) FROM {table}")
                )
                summary["latest_date"] = self._format_date(
                    self._query_scalar(f"SELECT max({date_column}) FROM {table}")
                )
            return summary
        except Exception as exc:
            return {
                "table": table,
                "status": "error",
                "row_count": None,
                "distinct_count": None,
                "distinct_label": distinct_column,
                "earliest_date": None,
                "latest_date": None,
                "error": str(exc),
            }

    def _compute_lag_days(self, latest_market_date: Optional[str]) -> Optional[int]:
        if not latest_market_date:
            return None
        try:
            latest = datetime.fromisoformat(latest_market_date).date()
        except ValueError:
            latest = datetime.strptime(latest_market_date, "%Y-%m-%d").date()
        return max((date.today() - latest).days, 0)

    def get_overview(self) -> Dict[str, Any]:
        latest_run = session_db.get_latest_data_update_run()
        running_run = session_db.get_running_data_update_run()

        table_summaries = {
            "stock_daily": self._table_summary(
                table="stock_data.stock_daily",
                date_column="date",
            ),
            "stock_daily_meta": self._table_summary(
                table="stock_data.stock_daily_meta",
                date_column="last_update_date",
            ),
            "finicial_report": self._table_summary(
                table="stock_data.finicial_report",
                date_column="publish_date",
            ),
            "shares_info": self._table_summary(
                table="stock_data.shares_info",
                date_column="change_date",
            ),
            "industry_info": self._table_summary(
                table="stock_data.industry_info",
                date_column="enter_date",
            ),
            "index_stocks": self._table_summary(
                table="stock_data.index_stocks",
                date_column=None,
                distinct_column="code",
            ),
        }

        etf_summary = {
            "table": "local_all_etf_csv",
            "status": "ok",
            "row_count": len(self.db.get_all_etf_code()),
            "distinct_count": None,
            "distinct_label": None,
            "earliest_date": None,
            "latest_date": None,
            "error": None,
        }

        stock_coverage = {
            "tracked_stock_codes": len(self.db.get_all_stock_code()),
            "tracked_etf_codes": len(self.db.get_all_etf_code()),
        }

        latest_market_date = table_summaries["stock_daily"]["latest_date"]
        return {
            "reference_symbol": REFERENCE_SYMBOL,
            "latest_market_date": latest_market_date,
            "data_lag_days": self._compute_lag_days(latest_market_date),
            "stock_coverage": stock_coverage,
            "tables": {
                **table_summaries,
                "etf_catalog": etf_summary,
            },
            "last_update_run": latest_run,
            "running_update": running_run,
            "generated_at": datetime.now().isoformat(),
        }


@router.get("/overview")
async def get_market_admin_overview():
    service = MarketDbOverviewService()
    return await anyio.to_thread.run_sync(service.get_overview)


@router.get("/update-capabilities")
async def get_market_update_capabilities():
    return get_update_step_capabilities()


@router.get("/update-runs")
async def list_market_update_runs(limit: int = 20):
    return await anyio.to_thread.run_sync(session_db.list_data_update_runs, limit)


@router.post("/update-runs")
async def create_market_update_run(req: MarketAdminUpdateRequest):
    update_run = await anyio.to_thread.run_sync(session_db.create_data_update_run, "manual")
    task = run_data_update_pipeline_task.apply_async(
        kwargs={
            "trigger_source": "manual",
            "selected_steps": req.selected_steps,
            "share_start_date": req.share_start_date,
            "update_run_id": update_run["update_run_id"],
        },
        queue="automation",
    )
    return {
        "status": "submitted",
        "task_id": task.id,
        "update_run_id": update_run["update_run_id"],
    }
