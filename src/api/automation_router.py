"""Automation and data update API endpoints."""

from typing import Any, Dict, Optional

import anyio
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
    end_date: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    notification: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    schedule: str = "daily"


class SimulationJobNotificationRequest(BaseModel):
    notification: Dict[str, Any] = Field(default_factory=dict)


@router.post("/simulation-jobs")
async def create_simulation_job(req: SimulationJobRequest):
    return await anyio.to_thread.run_sync(
        automation_service.create_job,
        req.name,
        req.strategy,
        req.symbol,
        req.start_date,
        req.end_date,
        req.params,
        req.notification,
        req.enabled,
        req.schedule,
    )


@router.get("/simulation-jobs")
async def list_simulation_jobs(enabled_only: bool = False):
    return await anyio.to_thread.run_sync(
        automation_service.list_jobs,
        enabled_only,
        False,
    )


@router.post("/simulation-jobs/run-enabled")
async def run_enabled_simulation_jobs(force: bool = Query(False)):
    jobs = await anyio.to_thread.run_sync(automation_service.list_jobs, True, False)
    tasks = []
    for job in jobs:
        task = run_simulation_job_task.apply_async(
            args=[job["job_id"], None, "manual_batch", force],
            queue="automation",
        )
        tasks.append(
            {
                "job_id": job["job_id"],
                "name": job["name"],
                "task_id": task.id,
            }
        )
    return {"status": "submitted", "count": len(tasks), "tasks": tasks, "force_full_replay": force}


@router.get("/simulation-jobs/{job_id}")
async def get_simulation_job(job_id: str):
    job = await anyio.to_thread.run_sync(automation_service.get_job, job_id, True)
    if not job:
        raise HTTPException(status_code=404, detail="Simulation job not found")
    return job


@router.delete("/simulation-jobs/{job_id}")
async def delete_simulation_job(job_id: str):
    job = await anyio.to_thread.run_sync(automation_service.get_job, job_id, False)
    if not job:
        raise HTTPException(status_code=404, detail="Simulation job not found")
    ok = await anyio.to_thread.run_sync(automation_service.delete_job, job_id)
    return {"job_id": job_id, "deleted": ok}


@router.post("/simulation-jobs/{job_id}/notification")
async def update_simulation_job_notification(job_id: str, req: SimulationJobNotificationRequest):
    job = await anyio.to_thread.run_sync(automation_service.get_job, job_id, False)
    if not job:
        raise HTTPException(status_code=404, detail="Simulation job not found")
    return await anyio.to_thread.run_sync(
        lambda: session_db.update_simulation_job(job_id, notification=req.notification)
    )


@router.post("/simulation-jobs/{job_id}/enable")
async def enable_simulation_job(job_id: str):
    job = await anyio.to_thread.run_sync(automation_service.enable_job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Simulation job not found")
    return job


@router.post("/simulation-jobs/{job_id}/disable")
async def disable_simulation_job(job_id: str):
    job = await anyio.to_thread.run_sync(automation_service.disable_job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Simulation job not found")
    return job


@router.post("/simulation-jobs/{job_id}/run")
async def run_simulation_job(job_id: str, force: bool = Query(False)):
    job = await anyio.to_thread.run_sync(automation_service.get_job, job_id, False)
    if not job:
        raise HTTPException(status_code=404, detail="Simulation job not found")
    trigger_source = "manual_force" if force else "manual"
    task = run_simulation_job_task.apply_async(
        args=[job_id, None, trigger_source, force],
        queue="automation",
    )
    return {"job_id": job_id, "task_id": task.id, "status": "submitted", "force_full_replay": force}


@router.get("/simulation-jobs/{job_id}/runs")
async def list_simulation_job_runs(job_id: str, limit: int = Query(20, ge=1, le=200)):
    job = await anyio.to_thread.run_sync(automation_service.get_job, job_id, False)
    if not job:
        raise HTTPException(status_code=404, detail="Simulation job not found")
    return await anyio.to_thread.run_sync(lambda: session_db.list_simulation_runs(job_id, limit=limit))


@router.get("/simulation-runs/{run_id}")
async def get_simulation_run(run_id: str):
    run = await anyio.to_thread.run_sync(session_db.get_simulation_run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Simulation run not found")
    return run


@router.get("/simulation-runs/{run_id}/steps")
async def list_simulation_run_steps(
    run_id: str,
    limit: int = Query(200, ge=1, le=1000),
    since_step: Optional[int] = Query(None, ge=0),
):
    run = await anyio.to_thread.run_sync(session_db.get_simulation_run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Simulation run not found")
    return await anyio.to_thread.run_sync(
        lambda: session_db.list_simulation_run_steps(run_id, limit=limit, since_step=since_step)
    )


class DataUpdateRequest(BaseModel):
    selected_steps: Optional[list[str]] = None
    share_start_date: Optional[str] = None


@router.post("/data-update/run")
async def trigger_data_update(req: DataUpdateRequest):
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


@router.get("/data-update/history")
async def get_data_update_history(limit: int = Query(20, ge=1, le=200)):
    return await anyio.to_thread.run_sync(session_db.list_data_update_runs, limit)
