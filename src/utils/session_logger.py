"""
Strategy Log Collector - collects session logs in memory and persists them to SQLite.
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class LogEntry:
    """Single log entry."""

    timestamp: str
    level: str
    source: str
    message: str
    extra: Dict[str, Any] = field(default_factory=dict)


_session_logs: Dict[str, "SessionLogCollector"] = {}
_logs_lock = threading.Lock()
_persistence_db = None
_persistence_db_lock = threading.Lock()
_persistence_db_path: Optional[str] = None


def _current_db_path() -> str:
    return os.environ.get("SESSION_DB_PATH", "sessions.db")


def _get_persistence_db():
    global _persistence_db, _persistence_db_path

    db_path = _current_db_path()
    with _persistence_db_lock:
        if _persistence_db is None or _persistence_db_path != db_path:
            from session_db import SessionDB

            _persistence_db = SessionDB(db_path)
            _persistence_db_path = db_path
        return _persistence_db


class SessionLogCollector:
    """Collects logs for a single session and persists them in batches."""

    def __init__(
        self,
        session_id: str,
        max_entries: int = 5000,
        flush_batch_size: int = 20,
        flush_interval_seconds: float = 1.0,
    ):
        self.session_id = session_id
        self.max_entries = max_entries
        self.flush_batch_size = flush_batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.logs: deque[LogEntry] = deque(maxlen=max_entries)
        self.pending_logs: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.created_at = datetime.now()
        self._last_flush_at = time.monotonic()

        self._hydrate_from_db()

    def _hydrate_from_db(self):
        try:
            persisted_logs = _get_persistence_db().get_session_logs(
                self.session_id,
                limit=self.max_entries,
            )
        except Exception:
            persisted_logs = []

        if not persisted_logs:
            return

        with self.lock:
            for item in persisted_logs:
                self.logs.append(self._dict_to_entry(item))

    def _entry_to_dict(self, entry: LogEntry) -> Dict[str, Any]:
        return {
            "timestamp": entry.timestamp,
            "level": entry.level,
            "source": entry.source,
            "message": entry.message,
            "extra": entry.extra,
        }

    def _dict_to_entry(self, item: Dict[str, Any]) -> LogEntry:
        return LogEntry(
            timestamp=str(item.get("timestamp") or datetime.now().isoformat()),
            level=str(item.get("level") or "INFO").upper(),
            source=str(item.get("source") or "system"),
            message=str(item.get("message") or ""),
            extra=dict(item.get("extra") or {}),
        )

    def _drain_pending(self, force: bool = False) -> List[Dict[str, Any]]:
        with self.lock:
            if not self.pending_logs:
                return []

            should_flush = force or len(self.pending_logs) >= self.flush_batch_size
            if not should_flush:
                elapsed = time.monotonic() - self._last_flush_at
                should_flush = elapsed >= self.flush_interval_seconds
            if not should_flush:
                return []

            batch = list(self.pending_logs)
            self.pending_logs.clear()
            self._last_flush_at = time.monotonic()
            return batch

    def flush(self, force: bool = False) -> int:
        batch = self._drain_pending(force=force)
        if not batch:
            return 0

        try:
            _get_persistence_db().add_session_logs(self.session_id, batch)
            return len(batch)
        except Exception:
            with self.lock:
                self.pending_logs = batch + self.pending_logs
            return 0

    def add(self, level: str, source: str, message: str, extra: Optional[Dict] = None):
        """Add a log entry."""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.upper(),
            source=source,
            message=message,
            extra=extra or {},
        )

        with self.lock:
            self.logs.append(entry)
            self.pending_logs.append(self._entry_to_dict(entry))

        self.flush(force=False)

    def debug(self, source: str, message: str, **kwargs):
        self.add("DEBUG", source, message, kwargs)

    def info(self, source: str, message: str, **kwargs):
        self.add("INFO", source, message, kwargs)

    def warning(self, source: str, message: str, **kwargs):
        self.add("WARNING", source, message, kwargs)

    def error(self, source: str, message: str, **kwargs):
        self.add("ERROR", source, message, kwargs)

    def get_logs(
        self,
        level: Optional[str] = None,
        source: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Get logs with optional filters."""
        self.flush(force=True)
        return get_session_logs(
            self.session_id,
            level=level,
            source=source,
            since=since,
            limit=limit,
        )

    def get_formatted_logs(self, **filters) -> str:
        """Get logs as formatted text for log viewer."""
        logs = self.get_logs(**filters)
        lines = []
        for log in logs:
            color_map = {
                "DEBUG": "\033[36m",
                "INFO": "\033[32m",
                "WARNING": "\033[33m",
                "ERROR": "\033[31m",
            }
            reset = "\033[0m"
            color = color_map.get(log["level"], "")

            line = f"{log['timestamp']} {color}[{log['level']:7}]{reset} [{log['source']:10}] {log['message']}"
            if log["extra"]:
                extras = " | ".join(f"{k}={v}" for k, v in log["extra"].items())
                line += f" | {extras}"
            lines.append(line)

        return "\n".join(lines)

    def clear(self):
        """Clear logs from memory and persistence."""
        clear_session_logs(self.session_id)


def get_session_logger(session_id: str, create: bool = True) -> Optional[SessionLogCollector]:
    """Get or create a log collector for a session."""
    with _logs_lock:
        collector = _session_logs.get(session_id)
        if collector is None and create:
            collector = SessionLogCollector(session_id)
            _session_logs[session_id] = collector
        return collector


def remove_session_logger(session_id: str):
    """Remove a session's log collector after flushing pending logs."""
    with _logs_lock:
        collector = _session_logs.pop(session_id, None)
    if collector:
        collector.flush(force=True)


def list_session_loggers() -> List[str]:
    """List all session IDs that currently have or had persisted logs."""
    persisted_ids: List[str] = []
    try:
        persisted_ids = _get_persistence_db().list_sessions_with_logs()
    except Exception:
        persisted_ids = []

    with _logs_lock:
        in_memory_ids = [
            session_id for session_id, collector in _session_logs.items() if collector.logs or collector.pending_logs
        ]

    result: List[str] = []
    for session_id in [*in_memory_ids, *persisted_ids]:
        if session_id not in result:
            result.append(session_id)
    return result


def get_session_logs(
    session_id: str,
    level: Optional[str] = None,
    source: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Fetch logs for a session from persistence, flushing live buffers first."""
    collector = get_session_logger(session_id, create=False)
    if collector:
        collector.flush(force=True)

    try:
        return _get_persistence_db().get_session_logs(
            session_id,
            level=level,
            source=source,
            since=since,
            limit=limit,
        )
    except Exception:
        if not collector:
            return []
        with collector.lock:
            result = [collector._entry_to_dict(item) for item in collector.logs]

        if level:
            result = [item for item in result if item["level"] == level.upper()]
        if source:
            result = [item for item in result if item["source"] == source]
        if since:
            result = [item for item in result if item["timestamp"] >= since]
        return result[-limit:]


def clear_session_logs(session_id: str):
    """Clear a session's logs from memory and persistence."""
    collector = get_session_logger(session_id, create=False)
    if collector:
        with collector.lock:
            collector.logs.clear()
            collector.pending_logs.clear()
            collector._last_flush_at = time.monotonic()

    try:
        _get_persistence_db().clear_session_logs(session_id)
    except Exception:
        pass


def get_all_logs(session_id: str, **filters) -> str:
    """Get formatted logs for a session (API helper)."""
    logs = get_session_logs(session_id, **filters)
    if not logs:
        return f"No logs found for session {session_id}"

    lines = []
    for log in logs:
        color_map = {
            "DEBUG": "\033[36m",
            "INFO": "\033[32m",
            "WARNING": "\033[33m",
            "ERROR": "\033[31m",
        }
        reset = "\033[0m"
        color = color_map.get(log["level"], "")
        line = f"{log['timestamp']} {color}[{log['level']:7}]{reset} [{log['source']:10}] {log['message']}"
        if log["extra"]:
            extras = " | ".join(f"{k}={v}" for k, v in log["extra"].items())
            line += f" | {extras}"
        lines.append(line)
    return "\n".join(lines)


def flush_all_session_loggers():
    """Flush all live session log buffers to persistence."""
    with _logs_lock:
        collectors = list(_session_logs.values())
    for collector in collectors:
        collector.flush(force=True)


def reset_session_log_store():
    """Reset global log collector state, primarily for tests."""
    flush_all_session_loggers()

    global _persistence_db, _persistence_db_path
    with _logs_lock:
        _session_logs.clear()
    with _persistence_db_lock:
        _persistence_db = None
        _persistence_db_path = None


atexit.register(flush_all_session_loggers)
