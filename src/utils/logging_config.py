"""
Unified logging configuration for the quantitative trading platform.
Uses loguru for structured, configurable logging with rotation and retention.
Supports JSON output for log aggregation and request correlation.
"""

import sys
import json
import contextvars
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional, Callable
from loguru import logger

# Context variables for request correlation
request_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)
session_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "session_id", default=None
)

# Configuration
LOG_LEVEL = "INFO"
LOG_ROTATION = "100 MB"
LOG_RETENTION = "30 days"
LOG_PATH = "logs/quant_{time:YYYY-MM-DD}.log"
JSON_LOG_PATH = "logs/quant_{time:YYYY-MM-DD}.json"

# Human-readable format for console
CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[name]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
    "{extra[context_str]}"
)

# File format (with full timestamp)
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{extra[name]}:{function}:{line} - {message}"
)


def json_serializer(record: Dict[str, Any]) -> str:
    """Serialize log record to JSON for structured logging."""

    # Extract context from extras
    request_id = request_id_ctx.get()
    session_id = session_id_ctx.get()

    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["extra"].get("name", "root"),
        "message": record["message"],
        "function": record["function"],
        "line": record["line"],
        "file": record["file"].name if record["file"] else None,
    }

    # Add correlation IDs if present
    if request_id:
        log_entry["request_id"] = request_id
    if session_id:
        log_entry["session_id"] = session_id

    # Add extra fields (excluding internal ones)
    extra_fields = {
        k: v
        for k, v in record["extra"].items()
        if k not in ("name", "context_str") and not k.startswith("_")
    }
    if extra_fields:
        log_entry["context"] = extra_fields

    # Add exception info if present
    if record["exception"]:
        log_entry["exception"] = {
            "type": record["exception"].type.__name__
            if record["exception"].type
            else None,
            "value": str(record["exception"].value)
            if record["exception"].value
            else None,
            "traceback": record["exception"].traceback is not None,
        }

    return json.dumps(log_entry, default=str) + "\n"


def format_context_string(record: Dict[str, Any]) -> str:
    """Format context for console display."""
    extra = record["extra"]
    context_parts = []

    request_id = request_id_ctx.get()
    session_id = session_id_ctx.get()

    if request_id:
        context_parts.append(f"req={request_id[:8]}")
    if session_id:
        context_parts.append(f"sess={session_id[:8]}")

    # Add other extra fields
    for k, v in extra.items():
        if k not in ("name", "context_str") and not k.startswith("_"):
            context_parts.append(f"{k}={v}")

    if context_parts:
        return f" | {' '.join(context_parts)}"
    return ""


def patcher(record: Dict[str, Any]) -> None:
    """Patch record with default values and computed fields."""
    if "name" not in record["extra"]:
        record["extra"]["name"] = "root"
    record["extra"]["context_str"] = format_context_string(record)


def setup_logging(
    level: str = LOG_LEVEL,
    json_output: bool = False,
    console: bool = True,
    file_output: bool = True,
) -> None:
    """
    Setup global logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: Enable JSON log file output
        console: Enable console output
        file_output: Enable file output
    """
    logger.remove()

    # Apply patcher to all handlers
    logger.configure(patcher=patcher)

    # Console handler (human-readable)
    if console:
        logger.add(
            sys.stderr,
            format=CONSOLE_FORMAT,
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    # Ensure log directory exists
    log_dir = Path(LOG_PATH).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # File handler (human-readable)
    if file_output:
        logger.add(
            LOG_PATH,
            format=FILE_FORMAT,
            level=level,
            rotation=LOG_ROTATION,
            retention=LOG_RETENTION,
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )

    # JSON file handler (for log aggregation)
    if json_output:
        logger.add(
            JSON_LOG_PATH,
            format=json_serializer,
            level=level,
            rotation=LOG_ROTATION,
            retention=LOG_RETENTION,
            compression="zip",
            enqueue=True,
            serialize=False,  # We handle serialization ourselves
        )

    logger.info(f"Logging system initialized: level={level}, json={json_output}")


def get_logger(name: str = "root"):
    """Get a logger instance with optional name binding."""
    return logger.bind(name=name)


# ============= Context Management =============


def set_request_context(request_id: str, session_id: Optional[str] = None) -> None:
    """Set request correlation context for logging."""
    request_id_ctx.set(request_id)
    if session_id:
        session_id_ctx.set(session_id)


def clear_request_context() -> None:
    """Clear request correlation context."""
    request_id_ctx.set(None)
    session_id_ctx.set(None)


def with_context(**context):
    """Decorator to add logging context to a function."""

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with logger.contextualize(**context):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with logger.contextualize(**context):
                return func(*args, **kwargs)

        if asyncio_iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def asyncio_iscoroutinefunction(func):
    """Check if function is async."""
    import asyncio

    return asyncio.iscoroutinefunction(func)


# ============= Specialized Loggers =============


def log_performance(operation: str, duration: float, **kwargs) -> None:
    """Log performance metrics in a structured format."""
    logger.info(
        f"PERF: {operation} completed in {duration:.3f}s",
        perf_operation=operation,
        perf_duration_ms=round(duration * 1000, 2),
        **kwargs,
    )


def log_trade(
    action: str, symbol: str, quantity: float, price: float, **kwargs
) -> None:
    """Log trade execution in a structured format."""
    logger.info(
        f"TRADE: {action} {quantity} {symbol} @ {price}",
        trade_action=action,
        trade_symbol=symbol,
        trade_quantity=quantity,
        trade_price=price,
        **kwargs,
    )


def log_backtest_progress(
    session_id: str, progress: float, current_date: str, trades: int = 0, **kwargs
) -> None:
    """Log backtest progress for monitoring."""
    logger.info(
        f"BACKTEST: {progress:.1f}% - {current_date}",
        backtest_session=session_id,
        backtest_progress=progress,
        backtest_date=current_date,
        backtest_trades=trades,
        **kwargs,
    )


def log_error_with_context(error: Exception, context: Dict[str, Any]) -> None:
    """Log error with full context for debugging."""
    logger.exception(
        f"Error occurred: {error}",
        error_type=type(error).__name__,
        error_message=str(error),
        **context,
    )


def log_api_request(
    method: str, path: str, status_code: int, duration_ms: float, **kwargs
) -> None:
    """Log API request for monitoring."""
    level = "INFO" if status_code < 400 else "WARNING" if status_code < 500 else "ERROR"

    getattr(logger, level.lower())(
        f"API: {method} {path} -> {status_code} ({duration_ms:.0f}ms)",
        api_method=method,
        api_path=path,
        api_status=status_code,
        api_duration_ms=duration_ms,
        **kwargs,
    )


# ============= FastAPI Middleware Integration =============


async def logging_middleware(request, call_next):
    """FastAPI middleware for request logging with correlation IDs."""
    import uuid
    import time

    # Generate request ID
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    session_id = request.headers.get("X-Session-ID")

    # Set context
    set_request_context(request_id, session_id)

    # Track timing
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Log request
        log_api_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_api_request(
            method=request.method,
            path=request.url.path,
            status_code=500,
            duration_ms=duration_ms,
            error=str(e),
        )
        raise
    finally:
        clear_request_context()


# Auto-initialize on import (can be reconfigured)
try:
    setup_logging()
except Exception:
    logger.add(sys.stderr, level="INFO")
