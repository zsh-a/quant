from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from session_db import SessionDB
from src.market_data.update_pipeline import get_reference_latest_date


class AutomationService:
    def __init__(self, session_db: Optional[SessionDB] = None):
        self.session_db = session_db or SessionDB()

    def create_job(
        self,
        name: str,
        strategy: str,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        notification: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        schedule: str = "daily",
        source_zoo_factor_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        job = self.session_db.create_simulation_job(
            name=name,
            strategy_name=strategy,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            params=params or {},
            notification=notification or {},
            enabled=enabled,
            schedule=schedule,
        )
        if source_zoo_factor_id:
            self.session_db.update_simulation_job(
                job["job_id"],
                source_zoo_factor_id=source_zoo_factor_id,
            )
            try:
                self.session_db.add_lineage_edge(
                    parent_kind="zoo_factor",
                    parent_id=source_zoo_factor_id,
                    child_kind="simulation_job",
                    child_id=job["job_id"],
                    relation="promoted_to",
                )
            except Exception:
                pass
            job["source_zoo_factor_id"] = source_zoo_factor_id
        return job

    def get_job(self, job_id: str, include_snapshot: bool = True) -> Optional[Dict[str, Any]]:
        return self.session_db.get_simulation_job(job_id, include_snapshot=include_snapshot)

    def list_jobs(self, enabled_only: bool = False, include_snapshot: bool = False):
        return self.session_db.list_simulation_jobs(
            enabled_only=enabled_only,
            include_snapshot=include_snapshot,
        )

    def enable_job(self, job_id: str):
        return self.session_db.set_simulation_job_enabled(job_id, True)

    def disable_job(self, job_id: str):
        return self.session_db.set_simulation_job_enabled(job_id, False)

    def delete_job(self, job_id: str) -> bool:
        return self.session_db.delete_simulation_job(job_id)

    def get_run_window(
        self,
        job: Dict[str, Any],
        latest_market_date: Optional[str] = None,
        force_full_replay: bool = False,
    ):
        latest_end_date = latest_market_date or get_reference_latest_date()
        if not latest_end_date:
            return None

        configured_end_date = job.get("end_date")
        last_processed_at = job.get("last_processed_at")

        if force_full_replay:
            start_date = job["start_date"]
            end_date = (
                configured_end_date
                if configured_end_date and configured_end_date < latest_end_date
                else latest_end_date
            )
        elif last_processed_at:
            start_dt = datetime.fromisoformat(last_processed_at[:10]) + timedelta(days=1)
            start_date = start_dt.strftime("%Y-%m-%d")
            end_date = latest_end_date
        else:
            start_date = job["start_date"]
            end_date = (
                configured_end_date
                if configured_end_date and configured_end_date < latest_end_date
                else latest_end_date
            )

        if start_date > end_date:
            return None

        return {
            "start_date": start_date,
            "end_date": end_date,
            "force_full_replay": force_full_replay,
        }
