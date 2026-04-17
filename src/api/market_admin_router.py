"""Market database admin dashboard API."""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any, Dict, Optional

import anyio
from fastapi import APIRouter
from pydantic import BaseModel

from session_db import SessionDB
from src.market_data.db import DB
from src.market_data.update_pipeline import REFERENCE_SYMBOL, get_update_step_capabilities
from src.tasks.automation import run_data_update_pipeline_task

router = APIRouter(prefix="/market-admin", tags=["market-admin"])

session_db = SessionDB()


class MarketAdminUpdateRequest(BaseModel):
    selected_steps: Optional[list[str]] = None
    share_start_date: Optional[str] = None


class MarketDbOverviewService:
    _ALLOWED_TABLES = frozenset(
        {
            "stock_data.trade_dates",
            "stock_data.all_stock",
            "stock_data.stock_daily",
            "stock_data.stock_daily_meta",
            "stock_data.finicial_report",
            "stock_data.finicial_data",
            "stock_data.shares_info",
            "stock_data.industry_info",
            "stock_data.index_stocks",
        }
    )

    _ALLOWED_COLUMNS = frozenset(
        {
            "code",
            "date",
            "day",
            "calendar_date",
            "last_update_date",
            "report_date",
            "publish_date",
            "change_date",
            "enter_date",
            "index",
        }
    )

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
        if table not in self._ALLOWED_TABLES:
            raise ValueError(f"不允许的表名: {table!r}")
        if distinct_column and distinct_column not in self._ALLOWED_COLUMNS:
            raise ValueError(f"不允许的列名: {distinct_column!r}")
        if date_column and date_column not in self._ALLOWED_COLUMNS:
            raise ValueError(f"不允许的列名: {date_column!r}")
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

    def _parse_timestamp(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None

    def _mark_stale_run_if_needed(self, run: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not run or run.get("status") != "running":
            return run

        stale_seconds = int(os.getenv("DATA_UPDATE_STALE_SECONDS", "600"))
        last_heartbeat = run.get("last_heartbeat_at") or run.get("started_at") or run.get("created_at")
        last_dt = self._parse_timestamp(last_heartbeat)
        if not last_dt:
            return run

        elapsed = (datetime.now() - last_dt).total_seconds()
        if elapsed < stale_seconds:
            return run

        now = datetime.now().isoformat()
        session_db.update_data_update_run(
            run["update_run_id"],
            status="failed_timeout",
            error=f"Heartbeat timeout after {stale_seconds}s",
            completed_at=now,
        )
        return session_db.get_data_update_run(run["update_run_id"])

    def get_overview(self) -> Dict[str, Any]:
        running_run = session_db.get_running_data_update_run()
        running_run = self._mark_stale_run_if_needed(running_run)
        latest_run = session_db.get_latest_data_update_run()
        if running_run and running_run.get("status") != "running":
            running_run = None

        table_summaries = {
            "trade_dates": self._table_summary(
                table="stock_data.trade_dates",
                date_column="calendar_date",
                distinct_column="calendar_date",
            ),
            "all_stock": self._table_summary(
                table="stock_data.all_stock",
                date_column="day",
                distinct_column="code",
            ),
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


@router.get("/data-quality")
async def get_data_quality_report():
    """数据质量报告：缺失率、异常值、新鲜度。"""
    from src.market_data.data_quality import DataQualityChecker

    checker = DataQualityChecker()
    return await anyio.to_thread.run_sync(checker.full_report)


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
