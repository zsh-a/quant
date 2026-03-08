from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from data_update import get_reference_latest_date
from session_db import SessionDB


class AutomationService:
    def __init__(self, session_db: Optional[SessionDB] = None):
        self.session_db = session_db or SessionDB()

    def create_job(
        self,
        name: str,
        strategy: str,
        symbol: str,
        start_date: str,
        params: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        schedule: str = "daily",
    ) -> Dict[str, Any]:
        return self.session_db.create_simulation_job(
            name=name,
            strategy_name=strategy,
            symbol=symbol,
            start_date=start_date,
            params=params or {},
            enabled=enabled,
            schedule=schedule,
        )

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.session_db.get_simulation_job(job_id)

    def list_jobs(self, enabled_only: bool = False):
        return self.session_db.list_simulation_jobs(enabled_only=enabled_only)

    def enable_job(self, job_id: str):
        return self.session_db.set_simulation_job_enabled(job_id, True)

    def disable_job(self, job_id: str):
        return self.session_db.set_simulation_job_enabled(job_id, False)

    def get_run_window(self, job: Dict[str, Any], latest_market_date: Optional[str] = None):
        end_date = latest_market_date or get_reference_latest_date()
        if not end_date:
            return None

        last_processed_at = job.get("last_processed_at")
        if last_processed_at:
            start_dt = datetime.fromisoformat(last_processed_at[:10]) + timedelta(days=1)
            start_date = start_dt.strftime("%Y-%m-%d")
        else:
            start_date = job["start_date"]

        if start_date > end_date:
            return None

        return {"start_date": start_date, "end_date": end_date}
