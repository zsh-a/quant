"""Automation tasks for scheduled data updates and simulation continuation."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, Optional

from loguru import logger

from data_update import get_reference_latest_date, run_data_update_pipeline
from db import DB
from session_db import SessionDB
from src.analysis.backtest_metrics import calculate_metrics as calc_perf_metrics
from src.automation.service import AutomationService
from src.config.settings import get_broker_config, get_data_stream_config
from src.core.backtest_broker import BacktestBroker
from src.core.data_stream import DBDataStream
from src.core.engine import TradingEngine
from src.strategies.registry import StrategyRegistry
from src.tasks.celery_app import app

broker_config = get_broker_config()
data_stream_config = get_data_stream_config()


@app.task(name="src.tasks.automation.run_data_update_pipeline")
def run_data_update_pipeline_task(trigger_source: str = "manual", selected_steps=None):
    session_db = SessionDB()
    update_run = session_db.create_data_update_run(trigger_source=trigger_source)
    update_run_id = update_run["update_run_id"]

    try:
        result = run_data_update_pipeline(selected_steps=selected_steps)
        triggered_jobs = []

        if result.get("has_new_data"):
            service = AutomationService(session_db)
            jobs = service.list_jobs(enabled_only=True)
            for job in jobs:
                task = run_simulation_job_task.apply_async(
                    args=[job["job_id"], update_run_id, "data_update"],
                    queue="automation",
                )
                triggered_jobs.append(
                    {
                        "job_id": job["job_id"],
                        "name": job["name"],
                        "task_id": task.id,
                    }
                )

        details = {**result, "triggered_jobs": triggered_jobs}
        session_db.update_data_update_run(
            update_run_id,
            status=result["status"],
            has_new_data=result.get("has_new_data", False),
            details=details,
            completed_at=datetime.now().isoformat(),
        )
        return {"update_run_id": update_run_id, **details}
    except Exception as exc:
        logger.exception("Data update pipeline failed")
        session_db.update_data_update_run(
            update_run_id,
            status="failed",
            has_new_data=False,
            error=str(exc),
            details={"errors": [{"error": str(exc)}]},
            completed_at=datetime.now().isoformat(),
        )
        return {"update_run_id": update_run_id, "status": "failed", "error": str(exc)}


@app.task(name="src.tasks.automation.run_simulation_job")
def run_simulation_job_task(
    job_id: str,
    update_run_id: Optional[str] = None,
    trigger_source: str = "manual",
    force_full_replay: bool = False,
):
    session_db = SessionDB()
    service = AutomationService(session_db)
    job = service.get_job(job_id)
    if not job:
        return {"status": "error", "error": f"job {job_id} not found"}

    latest_market_date = get_reference_latest_date()
    window = service.get_run_window(
        job,
        latest_market_date=latest_market_date,
        force_full_replay=force_full_replay,
    )
    if not window:
        session_db.update_simulation_job(
            job_id,
            status="idle",
            last_update_at=datetime.now().isoformat(),
            error=None,
        )
        return {
            "status": "skipped",
            "job_id": job_id,
            "reason": "no new market data",
            "latest_market_date": latest_market_date,
            "force_full_replay": force_full_replay,
        }

    existing_session_id = job.get("latest_session_id")
    existing_session = session_db.get_session(existing_session_id) if existing_session_id else None
    session_id = existing_session_id if existing_session else str(uuid.uuid4())

    run = session_db.create_simulation_run(
        job_id=job_id,
        session_id=session_id,
        start_date=window["start_date"],
        end_date=window["end_date"],
        trigger_source=trigger_source,
        update_run_id=update_run_id,
    )
    run_id = run["run_id"]

    if existing_session:
        if force_full_replay:
            from src.utils.session_logger import clear_session_logs

            clear_session_logs(session_id)
            session_db.clear_session_runtime_data(session_id, clear_logs=False)
        session_db.update_session(
            session_id,
            strategy_name=job["strategy_name"],
            symbol=job["symbol"],
            mode="simulation",
            start_date=job["start_date"],
            end_date=window["end_date"],
            status="starting",
            progress=0.0,
            error=None,
            params=job.get("params") or {},
            source="automation",
            job_id=job_id,
            run_id=run_id,
            last_processed_at=None if force_full_replay else job.get("last_processed_at"),
        )
    else:
        session_db.create_session(
            session_id=session_id,
            strategy_name=job["strategy_name"],
            symbol=job["symbol"],
            mode="simulation",
            start_date=job["start_date"],
            end_date=window["end_date"],
            params=job.get("params") or {},
            source="automation",
            job_id=job_id,
            run_id=run_id,
            last_processed_at=None if force_full_replay else job.get("last_processed_at"),
        )

    db_client = DB()
    stream = None
    step_index = 0

    try:
        stream = DBDataStream(
            db_client,
            [job["symbol"]],
            window["start_date"],
            window["end_date"],
            chunk_size_months=data_stream_config.chunk_size_months,
        )
        total_bars = getattr(stream, "total_bars", 0)
        snapshot = {} if force_full_replay else (job.get("snapshot") or {})
        initial_cash = snapshot.get("initial_cash", broker_config.backtest.initial_cash)

        broker = BacktestBroker(
            db_client=db_client,
            initial_cash=initial_cash,
            commission=broker_config.backtest.commission,
            slippage=getattr(broker_config.backtest, "slippage", 0.001),
        )
        broker.restore_from_snapshot(snapshot)

        strategy = StrategyRegistry.create_strategy(
            job["strategy_name"],
            db_client,
            session_id=session_id,
            **(job.get("params") or {}),
        )
        if strategy is None:
            raise ValueError(f"Unknown strategy: {job['strategy_name']}")

        session_db.add_simulation_run_step(
            run_id,
            session_id,
            step_index=0,
            timestamp=datetime.now().isoformat(),
            event_type="simulation_batch_started",
            payload={
                "job_id": job_id,
                "job_name": job["name"],
                "strategy": job["strategy_name"],
                "symbol": job["symbol"],
                "start_date": window["start_date"],
                "end_date": window["end_date"],
                "trigger_source": trigger_source,
                "force_full_replay": force_full_replay,
            },
        )

        def on_step(bars):
            nonlocal step_index
            step_index += 1
            current_ts = next(iter(bars.values())).timestamp if bars else None
            progress = round((stream.idx / total_bars) * 100, 2) if total_bars else 100.0
            info = broker.get_account_info()
            new_equity_points = list(info.get("equity_history", []))
            latest_equity = (
                new_equity_points[-1]
                if new_equity_points
                else {
                    "timestamp": str(current_ts),
                    "total_equity": round(float(info.get("total_equity", 0.0)), 2),
                    "cash": round(float(info.get("cash", 0.0)), 2),
                    "positions": info.get("detailed_positions", {}),
                    "daily_pnl": 0.0,
                    "daily_return": 0.0,
                }
            )
            if new_equity_points:
                session_db.add_equity_points(session_id, new_equity_points)
                broker.equity_history.clear()

            new_trades = list(info.get("trades", []))
            if new_trades:
                session_db.add_trades(session_id, new_trades)
                broker.trades.clear()

            payload = {
                "bar_timestamp": str(current_ts) if current_ts else None,
                "progress": progress,
                "symbol": job["symbol"],
                "close_prices": {symbol: float(bar.close) for symbol, bar in bars.items()},
                "total_equity": latest_equity.get("total_equity"),
                "cash": latest_equity.get("cash"),
                "positions": latest_equity.get("positions", {}),
                "pending_orders": len(info.get("pending_orders", [])),
                "new_trades": new_trades,
            }
            session_db.add_simulation_run_step(
                run_id,
                session_id,
                step_index=step_index,
                timestamp=str(current_ts) if current_ts else datetime.now().isoformat(),
                event_type="strategy_step",
                payload=payload,
            )
            session_db.update_session_status(
                session_id,
                "running",
                progress,
                last_processed_at=str(current_ts.date()) if current_ts else None,
            )
            session_db.update_simulation_run(
                run_id,
                status="running",
                progress=progress,
                bars_processed=step_index,
                steps_recorded=step_index,
            )

        engine = TradingEngine(strategy=strategy, broker=broker, data_stream=stream, on_step=on_step)
        engine.run()

        account = broker.get_account_info()
        if broker.equity_history:
            session_db.add_equity_points(session_id, broker.equity_history)
            broker.equity_history.clear()
        if broker.trades:
            session_db.add_trades(session_id, broker.trades)
            broker.trades.clear()

        equity_history = session_db.get_equity_history(session_id)
        trades = session_db.get_trades(session_id)
        metrics = calc_perf_metrics(equity_history, trades).to_dict() if equity_history else {}
        final_equity = equity_history[-1]["total_equity"] if equity_history else account.get("total_equity", 0.0)
        snapshot = broker.get_state_snapshot(last_processed_at=window["end_date"])
        summary = {
            "final_equity": round(float(final_equity), 2),
            "total_trades": len(trades),
            "bars_processed": step_index,
            "metrics": metrics,
            "last_processed_at": window["end_date"],
            "force_full_replay": force_full_replay,
        }

        session_db.add_simulation_run_step(
            run_id,
            session_id,
            step_index=step_index + 1,
            timestamp=datetime.now().isoformat(),
            event_type="simulation_batch_completed",
            payload=summary,
        )
        session_db.update_session_status(
            session_id,
            "completed",
            100.0,
            last_processed_at=window["end_date"],
        )
        session_db.update_simulation_run(
            run_id,
            status="completed",
            progress=100.0,
            bars_processed=step_index,
            steps_recorded=step_index + 1,
            summary=summary,
            completed_at=datetime.now().isoformat(),
        )
        session_db.update_simulation_job(
            job_id,
            status="idle",
            last_processed_at=window["end_date"],
            last_update_at=datetime.now().isoformat(),
            latest_session_id=session_id,
            latest_run_id=run_id,
            snapshot=snapshot,
            error=None,
        )
        return {
            "status": "completed",
            "job_id": job_id,
            "run_id": run_id,
            "session_id": session_id,
            "force_full_replay": force_full_replay,
            **summary,
        }
    except Exception as exc:
        logger.exception("Simulation job failed")
        session_db.add_simulation_run_step(
            run_id,
            session_id,
            step_index=step_index + 1,
            timestamp=datetime.now().isoformat(),
            event_type="simulation_batch_failed",
            payload={"error": str(exc)},
        )
        session_db.update_session_status(session_id, "failed", error=str(exc))
        session_db.update_simulation_run(
            run_id,
            status="failed",
            error=str(exc),
            completed_at=datetime.now().isoformat(),
            steps_recorded=step_index + 1,
        )
        session_db.update_simulation_job(
            job_id,
            status="error",
            error=str(exc),
            latest_session_id=session_id,
            latest_run_id=run_id,
            last_update_at=datetime.now().isoformat(),
        )
        return {
            "status": "failed",
            "job_id": job_id,
            "run_id": run_id,
            "error": str(exc),
            "force_full_replay": force_full_replay,
        }


@app.task(name="src.tasks.automation.run_automation_cycle")
def run_automation_cycle_task():
    return run_data_update_pipeline_task(trigger_source="schedule")
