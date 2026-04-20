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
            state="PROGRESS",
            meta={
                "session_id": session_id,
                "progress": progress,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            },
        )


def resolve_market_interval(session_id: str, config: dict, session_db) -> tuple[str, str]:
    """Resolve market/interval for a backtest task.

    Prefers the Celery payload; falls back to the persisted session record so
    reruns and legacy payloads don't silently regress to A-share defaults.
    """
    market = config.get("market")
    interval = config.get("interval")
    if market is None or interval is None:
        row = session_db.get_session(session_id) if session_db is not None else None
        if row:
            market = market or row.get("market")
            interval = interval or row.get("interval")
    return market or "a_share", interval or "1d"


@app.task(
    bind=True,
    base=BacktestTask,
    name="src.tasks.backtest.run_backtest",
    autoretry_for=(ConnectionError, OSError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=300,
    max_retries=3,
    retry_jitter=True,
)
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
    # Bind correlation ID from API request for distributed tracing
    req_id = config.get("request_id")
    if req_id:
        from src.utils.logging_config import set_request_context

        set_request_context(req_id, session_id)

    logger.info(f"Starting backtest task for session {session_id}")

    session_db = SessionDB()

    def handle_progress(progress: float, status: str):
        message = "Backtest completed" if status == "completed" else f"Progress {progress:.1f}%"
        self.update_progress(session_id, progress, message)

    hooks = SessionExecutionHooks(on_progress=handle_progress)

    try:
        self.update_progress(session_id, 0, "Initializing backtest...")
        market, interval = resolve_market_interval(session_id, config, session_db)
        result = execute_session(
            SessionExecutionConfig(
                session_id=session_id,
                strategy=config["strategy"],
                symbol=config["symbol"],
                start_date=config["start_date"],
                end_date=config.get("end_date"),
                mode=config.get("mode", "backtest"),
                market=market,
                interval=interval,
                params=config.get("params", {}),
                initial_cash=config.get("initial_cash"),
                commission=config.get("commission"),
                slippage=config.get("slippage"),
                enable_risk_management=config.get("enable_risk_management", True),
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
    finally:
        if req_id:
            from src.utils.logging_config import clear_request_context

            clear_request_context()


@app.task(name="src.tasks.backtest.cancel_backtest")
def cancel_backtest_task(task_id: str):
    """Cancel a running backtest task"""
    app.control.revoke(task_id, terminate=True)
    logger.info(f"Cancelled task: {task_id}")
    return {"status": "cancelled", "task_id": task_id}
