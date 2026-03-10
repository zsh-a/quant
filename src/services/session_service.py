from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException

from session_db import SessionDB
from src.api.state_persistence import StatePersistence


@dataclass
class SessionRuntime:
    session_id: str
    strategy_name: str
    symbol: str
    mode: str
    start_date: str
    end_date: Optional[str]
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = "starting"
    progress: float = 0.0
    positions: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    engine: Any = None
    broker: Any = None


class SessionService:
    def __init__(self, session_db: SessionDB, persistence: StatePersistence):
        self.session_db = session_db
        self.persistence = persistence
        self._active_sessions: Dict[str, SessionRuntime] = {}
        self._lock = RLock()

    def create_session(
        self,
        *,
        session_id: str,
        strategy_name: str,
        symbol: str,
        mode: str,
        start_date: str,
        end_date: Optional[str],
        params: Optional[Dict[str, Any]] = None,
        source: str = "manual",
        job_id: Optional[str] = None,
        run_id: Optional[str] = None,
        last_processed_at: Optional[str] = None,
        register_runtime: bool = True,
    ) -> SessionRuntime:
        runtime = SessionRuntime(
            session_id=session_id,
            strategy_name=strategy_name,
            symbol=symbol,
            mode=mode,
            start_date=start_date,
            end_date=end_date,
            params=dict(params or {}),
        )
        self.session_db.create_session(
            session_id,
            strategy_name,
            symbol,
            mode,
            start_date,
            end_date,
            runtime.params,
            source=source,
            job_id=job_id,
            run_id=run_id,
            last_processed_at=last_processed_at,
        )
        if register_runtime:
            with self._lock:
                self._active_sessions[session_id] = runtime
        return runtime

    def get_runtime(self, session_id: str) -> Optional[SessionRuntime]:
        with self._lock:
            return self._active_sessions.get(session_id)

    def update_runtime(
        self,
        session_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        positions: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        engine: Any = None,
        broker: Any = None,
    ) -> Optional[SessionRuntime]:
        runtime = self.get_runtime(session_id)
        if runtime is None:
            return None

        if status is not None:
            runtime.status = status
        if progress is not None:
            runtime.progress = progress
        if positions is not None:
            runtime.positions = positions
        if error is not None or status in {"running", "completed"}:
            runtime.error = error
        if engine is not None:
            runtime.engine = engine
        if broker is not None:
            runtime.broker = broker
        return runtime

    def get_runtime_or_persisted(
        self, session_id: str
    ) -> Tuple[Optional[SessionRuntime], Optional[Dict[str, Any]]]:
        runtime = self.get_runtime(session_id)
        persisted = self.session_db.get_session(session_id)
        return runtime, persisted

    def get_status_payload(self, session_id: str, since: Optional[str] = None) -> Dict[str, Any]:
        runtime, persisted = self.get_runtime_or_persisted(session_id)
        if runtime is None and persisted is None:
            raise HTTPException(status_code=404, detail="Session not found")

        if runtime is not None:
            status = runtime.status
            mode = runtime.mode
            progress = runtime.progress
            error = runtime.error
            start_date = runtime.start_date
            end_date = runtime.end_date
            positions = runtime.positions
        else:
            assert persisted is not None
            status = persisted["status"]
            mode = persisted["mode"]
            progress = persisted["progress"]
            error = persisted["error"]
            start_date = persisted["start_date"]
            end_date = persisted["end_date"]
            positions = {}

        equity_history = self.session_db.get_equity_history(session_id, since)
        trades = self.session_db.get_trades(session_id, since)
        if not positions and equity_history:
            positions = equity_history[-1].get("positions", {})

        return {
            "status": status,
            "mode": mode,
            "progress": progress,
            "equity_history": equity_history,
            "trades": trades,
            "positions": positions,
            "error": error,
            "start_date": start_date,
            "end_date": end_date,
        }

    def delete_session(self, session_id: str) -> bool:
        runtime, persisted = self.get_runtime_or_persisted(session_id)
        if runtime is None and persisted is None:
            raise HTTPException(status_code=404, detail="Session not found")

        status = runtime.status if runtime is not None else persisted["status"]
        if status == "running":
            raise HTTPException(status_code=409, detail="Running sessions cannot be deleted")

        deleted = self.session_db.delete_session(session_id)
        self.persistence.delete_checkpoints(session_id)
        with self._lock:
            self._active_sessions.pop(session_id, None)

        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return True

    def stop_session(self, session_id: str) -> Dict[str, str]:
        runtime = self.get_runtime(session_id)
        if runtime is not None:
            if runtime.engine:
                runtime.engine.stop()
            runtime.status = "stopped"
            self.session_db.update_session_status(session_id, "stopped")
            return {"status": "stopped"}

        persisted = self.session_db.get_session(session_id)
        if persisted and persisted["status"] == "running":
            self.session_db.update_session_status(session_id, "stopped")
            return {"status": "stopped (db updated)"}

        raise HTTPException(status_code=404, detail="Session not found or not running")

    def count_running_sessions(self) -> int:
        with self._lock:
            return len(
                [runtime for runtime in self._active_sessions.values() if runtime.status == "running"]
            )

    def build_checkpoint_state(self, session_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        runtime, persisted = self.get_runtime_or_persisted(session_id)
        if runtime is None and persisted is None:
            raise HTTPException(status_code=404, detail="Session not found")

        base = persisted or {
            "session_id": runtime.session_id,
            "strategy_name": runtime.strategy_name,
            "symbol": runtime.symbol,
            "mode": runtime.mode,
            "start_date": runtime.start_date,
            "end_date": runtime.end_date,
            "params": runtime.params,
            "status": runtime.status,
            "progress": runtime.progress,
            "error": runtime.error,
        }
        equity_history = self.session_db.get_equity_history(session_id)
        trades = self.session_db.get_trades(session_id)
        positions = (
            runtime.positions
            if runtime is not None and runtime.positions
            else (equity_history[-1].get("positions", {}) if equity_history else {})
        )
        state = {
            "session_id": session_id,
            "strategy": base["strategy_name"],
            "symbol": base["symbol"],
            "mode": base["mode"],
            "status": runtime.status if runtime is not None else base["status"],
            "progress": runtime.progress if runtime is not None else base["progress"],
            "equity_history": equity_history,
            "trades": trades,
            "positions": positions,
            "start_date": base["start_date"],
            "end_date": base.get("end_date"),
            "params": base.get("params", {}),
            "error": runtime.error if runtime is not None else base.get("error"),
        }
        metadata = {
            "checkpoint_type": "manual",
            "status": state["status"],
            "progress": state["progress"],
            "positions": len(positions),
        }
        return state, metadata

    def restore_session(self, state: Dict[str, Any]) -> SessionRuntime:
        runtime = SessionRuntime(
            session_id=state["session_id"],
            strategy_name=state["strategy"],
            symbol=state["symbol"],
            mode=state["mode"],
            start_date=state["start_date"],
            end_date=state.get("end_date"),
            params=dict(state.get("params", {})),
            status=state.get("status", "starting"),
            progress=state.get("progress", 0.0),
            positions=dict(state.get("positions", {})),
            error=state.get("error"),
        )
        if self.session_db.get_session(runtime.session_id) is None:
            self.session_db.create_session(
                runtime.session_id,
                runtime.strategy_name,
                runtime.symbol,
                runtime.mode,
                runtime.start_date,
                runtime.end_date,
                runtime.params,
            )
            self.session_db.update_session(
                runtime.session_id,
                status=runtime.status,
                progress=runtime.progress,
                error=runtime.error,
            )
        with self._lock:
            self._active_sessions[runtime.session_id] = runtime
        return runtime
