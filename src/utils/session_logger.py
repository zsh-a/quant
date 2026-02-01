"""
Strategy Log Collector - Collects and stores strategy execution logs per session.
"""

from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
import threading
from loguru import logger


@dataclass
class LogEntry:
    """Single log entry"""
    timestamp: str
    level: str  # DEBUG, INFO, WARNING, ERROR
    source: str  # strategy, broker, engine
    message: str
    extra: Dict = field(default_factory=dict)


class SessionLogCollector:
    """Collects logs for a single session"""
    
    def __init__(self, session_id: str, max_entries: int = 5000):
        self.session_id = session_id
        self.max_entries = max_entries
        self.logs: deque = deque(maxlen=max_entries)
        self.lock = threading.Lock()
        self.created_at = datetime.now()
    
    def add(self, level: str, source: str, message: str, extra: Optional[Dict] = None):
        """Add a log entry"""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            source=source,
            message=message,
            extra=extra or {}
        )
        with self.lock:
            self.logs.append(entry)
    
    def debug(self, source: str, message: str, **kwargs):
        self.add("DEBUG", source, message, kwargs)
    
    def info(self, source: str, message: str, **kwargs):
        self.add("INFO", source, message, kwargs)
    
    def warning(self, source: str, message: str, **kwargs):
        self.add("WARNING", source, message, kwargs)
    
    def error(self, source: str, message: str, **kwargs):
        self.add("ERROR", source, message, kwargs)
    
    def get_logs(self, level: Optional[str] = None, source: Optional[str] = None,
                 since: Optional[str] = None, limit: int = 500) -> List[Dict]:
        """Get logs with optional filters"""
        with self.lock:
            result = list(self.logs)
        
        # Apply filters
        if level:
            result = [l for l in result if l.level == level.upper()]
        if source:
            result = [l for l in result if l.source == source]
        if since:
            result = [l for l in result if l.timestamp >= since]
        
        # Return most recent entries up to limit
        return [
            {
                "timestamp": l.timestamp,
                "level": l.level,
                "source": l.source,
                "message": l.message,
                "extra": l.extra
            }
            for l in result[-limit:]
        ]
    
    def get_formatted_logs(self, **filters) -> str:
        """Get logs as formatted text for log viewer"""
        logs = self.get_logs(**filters)
        lines = []
        for log in logs:
            # ANSI color codes for log levels
            color_map = {
                "DEBUG": "\033[36m",   # Cyan
                "INFO": "\033[32m",    # Green
                "WARNING": "\033[33m", # Yellow
                "ERROR": "\033[31m",   # Red
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
        """Clear all logs"""
        with self.lock:
            self.logs.clear()


# Global log store
_session_logs: Dict[str, SessionLogCollector] = {}
_logs_lock = threading.Lock()


def get_session_logger(session_id: str) -> SessionLogCollector:
    """Get or create a log collector for a session"""
    with _logs_lock:
        if session_id not in _session_logs:
            _session_logs[session_id] = SessionLogCollector(session_id)
        return _session_logs[session_id]


def remove_session_logger(session_id: str):
    """Remove a session's log collector"""
    with _logs_lock:
        _session_logs.pop(session_id, None)


def list_session_loggers() -> List[str]:
    """List all session IDs with logs"""
    with _logs_lock:
        return list(_session_logs.keys())


def get_all_logs(session_id: str, **filters) -> str:
    """Get formatted logs for a session (API helper)"""
    collector = _session_logs.get(session_id)
    if not collector:
        return f"No logs found for session {session_id}"
    return collector.get_formatted_logs(**filters)
