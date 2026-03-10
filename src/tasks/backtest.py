"""
Backtest tasks for Celery distributed processing.
"""

from datetime import datetime

from celery import Task
from loguru import logger

from session_db import SessionDB
from src.services import (
    SessionExecutionConfig,
    SessionExecutionHooks,
    execute_session,
)
from src.tasks.celery_app import app


class BacktestTask(Task):
    """Base task with progress tracking"""
    
    def update_progress(self, session_id: str, progress: float, message: str = ""):
        """Update task progress"""
        self.update_state(
            state='PROGRESS',
            meta={
                'session_id': session_id,
                'progress': progress,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
        )


@app.task(bind=True, base=BacktestTask, name='src.tasks.backtest.run_backtest')
def run_backtest_task(self, session_id: str, config: dict):
    """
    Run a backtest task asynchronously.
    
    Args:
        session_id: Session identifier
        config: Backtest configuration
            - symbol: Stock symbol
            - strategy: Strategy name
            - start_date: Start date
            - end_date: End date
            - params: Strategy parameters
            - initial_cash: Initial capital
            - commission: Commission rate
    
    Returns:
        dict: Backtest results
    """
    logger.info(f"Starting backtest task for session {session_id}")
    
    session_db = SessionDB()

    def handle_progress(progress: float, status: str):
        message = "Backtest completed" if status == "completed" else f"Progress {progress:.1f}%"
        self.update_progress(session_id, progress, message)

    hooks = SessionExecutionHooks(on_progress=handle_progress)

    try:
        self.update_progress(session_id, 0, "Initializing backtest...")
        result = execute_session(
            SessionExecutionConfig(
                session_id=session_id,
                strategy=config["strategy"],
                symbol=config["symbol"],
                start_date=config["start_date"],
                end_date=config.get("end_date"),
                mode=config.get("mode", "backtest"),
                params=config.get("params", {}),
                initial_cash=config.get("initial_cash"),
                commission=config.get("commission"),
                slippage=config.get("slippage"),
                enable_risk_management=config.get("enable_risk_management", False),
                chunk_size_months=config.get("chunk_size_months"),
            ),
            session_db=session_db,
            hooks=hooks,
        )
        payload = result.to_dict()
        payload["completed_at"] = datetime.now().isoformat()
        logger.info(f"Backtest completed for session {session_id}")
        return payload
    except Exception as e:
        logger.error(f"Backtest task failed for session {session_id}: {e}")
        session_db.update_session_status(session_id, "failed", error=str(e))
        raise


@app.task(name='src.tasks.backtest.cancel_backtest')
def cancel_backtest_task(task_id: str):
    """Cancel a running backtest task"""
    app.control.revoke(task_id, terminate=True)
    logger.info(f"Cancelled task: {task_id}")
    return {'status': 'cancelled', 'task_id': task_id}
