"""Automation tasks for scheduled data updates and simulation continuation."""

from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Optional

from loguru import logger

from src.market_data.update_pipeline import (
    DEFAULT_SHARE_START_DATE,
    DEFAULT_UPDATE_STEPS,
    UPDATE_STEP_DEFINITIONS,
    get_reference_latest_date,
    run_data_update_pipeline,
)
from src.market_data.db import DB
from session_db import SessionDB
from src.analysis.backtest_metrics import calculate_metrics as calc_perf_metrics
from src.automation.service import AutomationService
from src.config.settings import (
    get_broker_config,
    get_data_stream_config,
    get_notifications_config,
)
from src.core.backtest_broker import BacktestBroker
from src.core.data_stream import DBDataStream
from src.core.engine import TradingEngine
from src.notifications.telegram import (
    TelegramNotifier,
    build_simulation_order_message,
    get_notification_chat_id,
    is_trade_notification_enabled,
)
from src.strategies.registry import StrategyRegistry
from src.tasks.celery_app import app

broker_config = get_broker_config()
data_stream_config = get_data_stream_config()
notifications_config = get_notifications_config()

DATA_UPDATE_SOFT_TIME_LIMIT = int(os.getenv("DATA_UPDATE_SOFT_TIME_LIMIT", "7200"))
DATA_UPDATE_TIME_LIMIT = int(os.getenv("DATA_UPDATE_TIME_LIMIT", "10800"))
DATA_UPDATE_HEARTBEAT_SECONDS = float(os.getenv("DATA_UPDATE_HEARTBEAT_SECONDS", "5"))


class DataUpdateProgressTracker:
    def __init__(
        self,
        session_db: SessionDB,
        update_run_id: str,
        steps_meta: list[dict],
        heartbeat_interval: float = DATA_UPDATE_HEARTBEAT_SECONDS,
    ):
        self.session_db = session_db
        self.update_run_id = update_run_id
        self.steps_meta = steps_meta
        self.step_index_map = {step["key"]: idx for idx, step in enumerate(steps_meta)}
        self.steps_state = [
            {
                "step": step["key"],
                "label": step.get("label", step["key"]),
                "status": "pending",
                "progress": 0.0,
            }
            for step in steps_meta
        ]
        self.current_step_key: Optional[str] = None
        self.heartbeat_interval = heartbeat_interval
        self.last_persist_at = 0.0
        self._lock = threading.Lock()

    def bootstrap(self):
        self._persist(force=True)

    def _current_step_payload(self) -> Optional[dict]:
        if not self.current_step_key:
            return None
        idx = self.step_index_map.get(self.current_step_key)
        if idx is None:
            return None
        step = self.steps_state[idx]
        return {
            "key": step.get("step"),
            "label": step.get("label"),
            "progress": step.get("progress"),
            "current": step.get("current"),
            "total": step.get("total"),
            "message": step.get("message"),
        }

    def _build_details(self) -> dict:
        with self._lock:
            steps_snapshot = [dict(step) for step in self.steps_state]
            current_step = self._current_step_payload()
        total = len(steps_snapshot)
        completed = sum(1 for step in steps_snapshot if step["status"] in {"success", "error", "skipped"})
        current_progress = 0.0
        for step in steps_snapshot:
            if step["status"] == "running":
                current_progress = max(current_progress, float(step.get("progress") or 0.0))
        overall = 100.0 if total == 0 else round(((completed + current_progress / 100.0) / total) * 100, 2)
        return {
            "progress": overall,
            "current_step": current_step,
            "total_steps": total,
            "completed_steps": completed,
            "steps": steps_snapshot,
        }

    def _persist(self, force: bool = False):
        now = time.time()
        if not force and now - self.last_persist_at < self.heartbeat_interval:
            return
        self.last_persist_at = now
        heartbeat = datetime.now().isoformat()
        self.session_db.update_data_update_run(
            self.update_run_id,
            details=self._build_details(),
            last_heartbeat_at=heartbeat,
        )

    def handle_event(self, event: dict):
        step_key = event.get("step")
        if step_key not in self.step_index_map:
            return
        idx = self.step_index_map[step_key]
        event_type = event.get("event")
        with self._lock:
            step = self.steps_state[idx]
            if event_type == "step_started":
                step["status"] = "running"
                step["progress"] = 0.0
                step["started_at"] = event.get("timestamp") or datetime.now().isoformat()
                step["label"] = event.get("label") or step.get("label")
                self.current_step_key = step_key
            elif event_type == "step_progress":
                progress = event.get("progress")
                if progress is not None:
                    step["progress"] = float(progress)
                for key in ("current", "total", "message", "fetched", "updated", "errors"):
                    if key in event:
                        step[key] = event.get(key)
            elif event_type == "step_completed":
                step["status"] = "success"
                step["progress"] = 100.0
                step["label"] = event.get("label") or step.get("label")
                if event.get("duration_seconds") is not None:
                    step["duration_seconds"] = event.get("duration_seconds")
                if event.get("detail") is not None:
                    step["detail"] = event.get("detail")
                self.current_step_key = None
            elif event_type == "step_failed":
                step["status"] = "error"
                step["label"] = event.get("label") or step.get("label")
                if event.get("duration_seconds") is not None:
                    step["duration_seconds"] = event.get("duration_seconds")
                if event.get("error") is not None:
                    step["error"] = event.get("error")
                self.current_step_key = None
        if event_type in {"step_started", "step_completed", "step_failed"}:
            self._persist(force=True)
        elif event_type == "step_progress":
            self._persist()

    def finalize(self, result: dict) -> dict:
        steps_result = {item.get("step"): item for item in result.get("steps", [])}
        with self._lock:
            for step in self.steps_state:
                step_key = step.get("step")
                if step_key not in steps_result:
                    continue
                final = steps_result[step_key]
                step["status"] = final.get("status", step.get("status"))
                step["progress"] = 100.0 if step["status"] in {"success", "error"} else step.get("progress", 0.0)
                if "duration_seconds" in final:
                    step["duration_seconds"] = final["duration_seconds"]
                if "detail" in final:
                    step["detail"] = final["detail"]
                if "error" in final:
                    step["error"] = final["error"]
                if "label" in final:
                    step["label"] = final["label"]
        details = self._build_details()
        details.update({k: v for k, v in result.items() if k != "steps"})
        details["progress"] = 100.0
        return details


def _collect_simulation_order_notification(
    *,
    job: Dict[str, object],
    session_id: str,
    run_id: str,
    broker: BacktestBroker,
    order,
) -> Optional[Dict[str, str]]:
    if not is_trade_notification_enabled(job.get("notification"), notifications_config.telegram):
        return None

    chat_id = get_notification_chat_id(job.get("notification"), notifications_config.telegram)
    if not chat_id:
        logger.warning("Telegram order notifications enabled on job but chat_id is missing")
        return None

    reference_price = None
    if order.symbol in broker.current_bars:
        reference_price = broker.current_bars[order.symbol].close
    elif order.symbol in broker.last_prices:
        reference_price = broker.last_prices[order.symbol]

    order_message = build_simulation_order_message(
        job_name=str(job.get("name") or job.get("job_id") or "simulation-job"),
        strategy_name=str(job.get("strategy_name") or ""),
        session_id=session_id,
        run_id=run_id,
        order=order,
        reference_price=reference_price,
        stock_name=broker.stock_names.get(order.symbol, "Unknown"),
    )
    price_text = f"{float(order.price):.4f}" if order.price is not None else "市价"
    reference_price_text = f"{float(reference_price):.4f}" if reference_price is not None and reference_price > 0 else "N/A"
    stock_display = f"{order.symbol} {broker.stock_names.get(order.symbol, 'Unknown')}".strip()
    action = "买入" if order.type == "buy" else "卖出"
    return {
        "chat_id": chat_id,
        "job_name": str(job.get("name") or job.get("job_id") or "simulation-job"),
        "strategy_name": str(job.get("strategy_name") or ""),
        "session_id": session_id,
        "run_id": run_id,
        "bar_timestamp": order.created_at.isoformat(),
        "order_message": order_message,
        "order_summary": " | ".join(
            [
                f"{action} `{float(order.quantity):g}`",
                f"`{stock_display}`",
                f"订单价 `{price_text}`",
                f"执行 `{order.execution_type}`",
                f"参考 `{reference_price_text}`",
            ]
        ),
    }


def _send_batched_simulation_order_notifications(order_notifications: list[Dict[str, str]]) -> None:
    if not order_notifications:
        return

    notifier = TelegramNotifier(notifications_config.telegram)
    if not notifier.is_enabled():
        logger.warning("Telegram order notifications enabled on job but global Telegram config is incomplete")
        return

    notifications_by_chat: Dict[str, list[Dict[str, str]]] = {}
    for item in order_notifications:
        chat_id = str(item.get("chat_id") or "")
        if not chat_id:
            continue
        notifications_by_chat.setdefault(chat_id, []).append(item)

    max_message_length = 3500
    for chat_id, messages in notifications_by_chat.items():
        chunks: list[str] = []
        current_chunk = ""
        first = messages[0]
        header = "\n".join(
            [
                "*模拟运行订单通知*",
                f"*任务*: `{first.get('job_name', '')}`",
                f"*策略*: `{first.get('strategy_name', '')}`",
                f"*Bar 时间*: `{first.get('bar_timestamp', '')}`",
                f"*Session*: `{first.get('session_id', '')}`",
                f"*Run*: `{first.get('run_id', '')}`",
                "",
            ]
        )
        for index, message in enumerate(messages, start=1):
            entry = f"{index}. {message.get('order_summary', '')}"
            next_chunk = f"{current_chunk}\n\n{entry}" if current_chunk else entry
            if current_chunk and len(next_chunk) > max_message_length:
                chunks.append(current_chunk)
                current_chunk = entry
            else:
                current_chunk = next_chunk
        if current_chunk:
            chunks.append(current_chunk)

        for chunk_index, chunk in enumerate(chunks, start=1):
            title = (
                f"{header}*订单数*: `{len(messages)}` | *分片*: `{chunk_index}/{len(chunks)}`\n\n"
                if len(chunks) > 1
                else f"{header}*订单数*: `{len(messages)}`\n\n"
            )
            notifier.send_message(chat_id=chat_id, text=f"{title}{chunk}")


@app.task(name="src.tasks.automation.send_telegram_validation_notification")
def send_telegram_validation_notification_task(
    chat_id: Optional[str] = None,
    context: str = "manual",
):
    notifier = TelegramNotifier(notifications_config.telegram)
    if not notifier.is_enabled():
        return {
            "status": "error",
            "error": "telegram notifier is disabled or bot token is missing",
        }

    target_chat_id = str(chat_id or notifications_config.telegram.default_chat_id or "")
    if not target_chat_id:
        return {
            "status": "error",
            "error": "telegram chat_id is missing",
        }

    timestamp = datetime.now().isoformat()
    message = "\n".join(
        [
            "*Telegram 验证通知*",
            f"*Context*: `{context}`",
            f"*Timestamp*: `{timestamp}`",
            "*Status*: 配置已生效，自动任务可发送通知。",
        ]
    )
    delivered = notifier.send_message(chat_id=target_chat_id, text=message)
    return {
        "status": "success" if delivered else "error",
        "chat_id": target_chat_id,
        "context": context,
        "timestamp": timestamp,
    }


@app.task(
    name="src.tasks.automation.run_data_update_pipeline",
    soft_time_limit=DATA_UPDATE_SOFT_TIME_LIMIT,
    time_limit=DATA_UPDATE_TIME_LIMIT,
)
def run_data_update_pipeline_task(
    trigger_source: str = "manual",
    selected_steps=None,
    share_start_date: Optional[str] = None,
    update_run_id: Optional[str] = None,
):
    session_db = SessionDB()
    if update_run_id:
        update_run = session_db.update_data_update_run(
            update_run_id,
            trigger_source=trigger_source,
            status="running",
            started_at=datetime.now().isoformat(),
            error=None,
            details={},
            completed_at=None,
            last_heartbeat_at=datetime.now().isoformat(),
        )
        if not update_run:
            update_run = session_db.create_data_update_run(trigger_source=trigger_source)
            update_run_id = update_run["update_run_id"]
    else:
        update_run = session_db.create_data_update_run(trigger_source=trigger_source)
        update_run_id = update_run["update_run_id"]

    steps = selected_steps or DEFAULT_UPDATE_STEPS
    step_label_map = {item["key"]: item.get("label", item["key"]) for item in UPDATE_STEP_DEFINITIONS}
    steps_meta = [{"key": key, "label": step_label_map.get(key, key)} for key in steps]
    tracker = DataUpdateProgressTracker(session_db, update_run_id, steps_meta)
    tracker.bootstrap()
    heartbeat_stop = threading.Event()

    def heartbeat_loop():
        while not heartbeat_stop.wait(tracker.heartbeat_interval):
            tracker._persist(force=True)

    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    def stop_heartbeat():
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1)

    try:
        result = run_data_update_pipeline(
            selected_steps=steps,
            share_start_date=share_start_date or DEFAULT_SHARE_START_DATE,
            progress_callback=tracker.handle_event,
        )
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

        stop_heartbeat()
        details = tracker.finalize({**result, "triggered_jobs": triggered_jobs})
        session_db.update_data_update_run(
            update_run_id,
            status=result["status"],
            has_new_data=result.get("has_new_data", False),
            details=details,
            completed_at=datetime.now().isoformat(),
            last_heartbeat_at=datetime.now().isoformat(),
        )
        return {"update_run_id": update_run_id, **details}
    except Exception as exc:
        logger.exception("Data update pipeline failed")
        stop_heartbeat()
        details = tracker.finalize(
            {
                "status": "failed",
                "errors": [{"error": str(exc)}],
            }
        )
        session_db.update_data_update_run(
            update_run_id,
            status="failed",
            has_new_data=False,
            error=str(exc),
            details=details,
            completed_at=datetime.now().isoformat(),
            last_heartbeat_at=datetime.now().isoformat(),
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
    pending_order_notifications: list[Dict[str, str]] = []

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
            slippage=broker_config.backtest.slippage,
            on_order_submitted=lambda order: pending_order_notifications.append(item)
            if (
                item := _collect_simulation_order_notification(
                    job=job,
                    session_id=session_id,
                    run_id=run_id,
                    broker=broker,
                    order=order,
                )
            )
            else None,
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
            if pending_order_notifications:
                _send_batched_simulation_order_notifications(pending_order_notifications)
                pending_order_notifications.clear()

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

        # Flush session logs so the final strategy entries are persisted before
        # we mark the run as completed (avoids the frontend missing tail logs).
        from src.utils.session_logger import get_session_logger as _get_sl
        _sl = _get_sl(session_id, create=False)
        if _sl:
            _sl.flush(force=True)

        account = broker.get_account_info()
        if broker.equity_history:
            session_db.add_equity_points(session_id, broker.equity_history)
            broker.equity_history.clear()
        if broker.trades:
            session_db.add_trades(session_id, broker.trades)
            broker.trades.clear()
        if pending_order_notifications:
            _send_batched_simulation_order_notifications(pending_order_notifications)
            pending_order_notifications.clear()

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
    task = run_data_update_pipeline_task.apply_async(
        kwargs={"trigger_source": "schedule"},
        queue="automation",
    )
    return {"status": "submitted", "task_id": task.id}
