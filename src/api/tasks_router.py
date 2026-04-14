"""
Task management API endpoints for Celery distributed processing.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from celery.result import AsyncResult
from src.config.settings import get_broker_config
from src.tasks.celery_app import app as celery_app
from src.tasks.backtest import run_backtest_task, cancel_backtest_task
from loguru import logger

router = APIRouter(prefix="/tasks", tags=["tasks"])
broker_config = get_broker_config()


from src.api.validators import DateStr, SymbolStr, NonNegativeFloat, Ratio


class BacktestTaskRequest(BaseModel):
    """Request model for creating a backtest task"""
    session_id: str
    symbol: SymbolStr
    strategy: str
    start_date: DateStr
    end_date: DateStr
    params: dict = {}
    initial_cash: NonNegativeFloat = broker_config.backtest.initial_cash
    commission: Ratio = broker_config.backtest.commission
    slippage: Ratio = broker_config.backtest.slippage
    enable_risk_management: bool = True
    chunk_size_months: Optional[int] = None


class TaskResponse(BaseModel):
    """Response model for task operations"""
    task_id: str
    session_id: str
    status: str
    message: Optional[str] = None


class TaskStatusResponse(BaseModel):
    """Response model for task status"""
    task_id: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


@router.post("/backtest", response_model=TaskResponse)
async def create_backtest_task(request: BacktestTaskRequest):
    """
    Submit a backtest task to the queue.
    
    Returns task_id for tracking progress.
    """
    try:
        config = {
            'symbol': request.symbol,
            'strategy': request.strategy,
            'start_date': request.start_date,
            'end_date': request.end_date,
            'params': request.params,
            'initial_cash': request.initial_cash,
            'commission': request.commission,
            'slippage': request.slippage,
            'enable_risk_management': request.enable_risk_management,
            'chunk_size_months': request.chunk_size_months
        }
        
        # Submit task to Celery
        task = run_backtest_task.apply_async(
            args=[request.session_id, config],
            queue='backtest'
        )
        
        logger.info(f"Backtest task submitted: {task.id} for session {request.session_id}")
        
        return TaskResponse(
            task_id=task.id,
            session_id=request.session_id,
            status="submitted",
            message="Backtest task submitted to queue"
        )
        
    except Exception as e:
        logger.error(f"Failed to submit backtest task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the status of a backtest task.
    
    Returns current progress and result if completed.
    """
    try:
        task = AsyncResult(task_id, app=celery_app)
        
        response = TaskStatusResponse(
            task_id=task_id,
            status=task.state
        )
        
        if task.state == 'PENDING':
            response.message = "Task is waiting in queue"
        
        elif task.state == 'PROGRESS':
            # Task is running, get progress info
            info = task.info or {}
            response.progress = info.get('progress', 0)
            response.message = info.get('message', 'Running...')
        
        elif task.state == 'SUCCESS':
            # Task completed successfully
            response.progress = 100.0
            response.result = task.result
            response.message = "Task completed successfully"
        
        elif task.state == 'FAILURE':
            # Task failed
            response.error = str(task.info)
            response.message = "Task failed"
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/backtest/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a running backtest task.
    """
    try:
        cancel_backtest_task.delay(task_id)
        
        logger.info(f"Task cancellation requested: {task_id}")
        
        return {
            "task_id": task_id,
            "status": "cancellation_requested",
            "message": "Task cancellation has been requested"
        }
        
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest")
async def list_tasks(limit: int = 50):
    """
    List recent backtest tasks.
    
    Note: This requires Celery result backend to be configured.
    """
    try:
        # Get active tasks
        inspect = celery_app.control.inspect()
        
        active_tasks = inspect.active() or {}
        scheduled_tasks = inspect.scheduled() or {}
        reserved_tasks = inspect.reserved() or {}
        
        all_tasks = []
        
        # Collect active tasks
        for worker, tasks in active_tasks.items():
            for task in tasks:
                all_tasks.append({
                    'task_id': task['id'],
                    'name': task['name'],
                    'worker': worker,
                    'status': 'active',
                    'args': task.get('args', [])
                })
        
        # Collect scheduled tasks
        for worker, tasks in scheduled_tasks.items():
            for task in tasks:
                all_tasks.append({
                    'task_id': task['request']['id'],
                    'name': task['request']['name'],
                    'worker': worker,
                    'status': 'scheduled'
                })
        
        # Collect reserved tasks
        for worker, tasks in reserved_tasks.items():
            for task in tasks:
                all_tasks.append({
                    'task_id': task['id'],
                    'name': task['name'],
                    'worker': worker,
                    'status': 'reserved'
                })
        
        return {
            'tasks': all_tasks[:limit],
            'total': len(all_tasks)
        }
        
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workers")
async def get_workers():
    """
    Get information about active Celery workers.
    """
    try:
        inspect = celery_app.control.inspect()
        
        stats = inspect.stats() or {}
        active = inspect.active() or {}
        
        workers = []
        for worker_name, worker_stats in stats.items():
            workers.append({
                'name': worker_name,
                'status': 'online',
                'concurrency': worker_stats.get('pool', {}).get('max-concurrency', 0),
                'active_tasks': len(active.get(worker_name, [])),
                'total_tasks': worker_stats.get('total', {})
            })
        
        return {
            'workers': workers,
            'total': len(workers)
        }
        
    except Exception as e:
        logger.error(f"Failed to get workers info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
