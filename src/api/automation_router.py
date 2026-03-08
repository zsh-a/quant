"""Automation and data update API endpoints."""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from session_db import SessionDB
from src.automation.service import AutomationService
from src.tasks.automation import run_data_update_pipeline_task, run_simulation_job_task

router = APIRouter(tags=["automation"])

session_db = SessionDB()
automation_service = AutomationService(session_db)


class SimulationJobRequest(BaseModel):
    name: str
    strategy: str
    symbol: str
    start_date: str
    params: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    schedule: str = "daily"


@router.post("/simulation-jobs")
async def create_simulation_job(req: SimulationJobRequest):
    return automation_service.create_job(
        name=req.name,
        strategy=req.strategy,
        symbol=req.symbol,
        start_date=req.start_date,
        params=req.params,
        enabled=req.enabled,
        schedule=req.schedule,
    )


@router.get("/simulation-jobs")
async def list_simulation_jobs(enabled_only: bool = False):
    return automation_service.list_jobs(enabled_only=enabled_only)


@router.get("/simulation-jobs/{job_id}")
async def get_simulation_job(job_id: str):
    job = automation_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Simulation job not found")
    return job


@router.post("/simulation-jobs/{job_id}/enable")
async def enable_simulation_job(job_id: str):
    job = automation_service.enable_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Simulation job not found")
    return job


@router.post("/simulation-jobs/{job_id}/disable")
async def disable_simulation_job(job_id: str):
    job = automation_service.disable_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Simulation job not found")
    return job


@router.post("/simulation-jobs/{job_id}/run")
async def run_simulation_job(job_id: str):
    job = automation_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Simulation job not found")
    task = run_simulation_job_task.apply_async(args=[job_id, None, "manual"], queue="automation")
    return {"job_id": job_id, "task_id": task.id, "status": "submitted"}


@router.get("/simulation-jobs/{job_id}/runs")
async def list_simulation_job_runs(job_id: str, limit: int = Query(20, ge=1, le=200)):
    job = automation_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Simulation job not found")
    return session_db.list_simulation_runs(job_id, limit=limit)


@router.get("/simulation-runs/{run_id}")
async def get_simulation_run(run_id: str):
    run = session_db.get_simulation_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Simulation run not found")
    return run


@router.get("/simulation-runs/{run_id}/steps")
async def list_simulation_run_steps(
    run_id: str,
    limit: int = Query(200, ge=1, le=1000),
    since_step: Optional[int] = Query(None, ge=0),
):
    run = session_db.get_simulation_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Simulation run not found")
    return session_db.list_simulation_run_steps(run_id, limit=limit, since_step=since_step)


class DataUpdateRequest(BaseModel):
    selected_steps: Optional[list[str]] = None


@router.post("/data-update/run")
async def trigger_data_update(req: DataUpdateRequest):
    task = run_data_update_pipeline_task.apply_async(
        kwargs={"trigger_source": "manual", "selected_steps": req.selected_steps},
        queue="automation",
    )
    return {"status": "submitted", "task_id": task.id}


@router.get("/data-update/history")
async def get_data_update_history(limit: int = Query(20, ge=1, le=200)):
    return session_db.list_data_update_runs(limit=limit)
