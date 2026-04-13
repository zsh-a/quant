"""
Unified logging configuration for the quantitative trading platform.
Routes stdlib logging, Uvicorn, and Celery through loguru so console and file
output stay consistent across processes.
"""

import contextvars
import orjson
import logging
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from loguru import logger

from src.config.settings import get_logging_config

# Context variables for request correlation
request_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)
session_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "session_id", default=None
)

# Fallback configuration
LOG_LEVEL = "INFO"
LOG_ROTATION = "100 MB"
LOG_RETENTION = "30 days"
from src.config.paths import LOG_FILE_PATH as _LOG_FILE_PATH, JSON_LOG_PATH as _JSON_LOG_PATH

LOG_PATH = str(_LOG_FILE_PATH)
JSON_LOG_PATH = str(_JSON_LOG_PATH)

# Human-readable format for console
# Compact: timestamp | LEVEL | module - message | context
CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[name]}</cyan> - "
    "<level>{message}</level>"
    "{extra[context_str]}"
)

# File format (with full timestamp)
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{extra[name]}:{function}:{line} - {message}"
)


class InterceptHandler(logging.Handler):
    """Route stdlib logging records through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.bind(name=record.name).opt(
            depth=depth,
            exception=record.exc_info,
            ansi=False,
        ).log(level, record.getMessage())


def _resolve_logging_options(
    level: Optional[str],
    json_output: Optional[bool],
    console: Optional[bool],
    file_output: Optional[bool],
) -> Dict[str, Any]:
    cfg = get_logging_config()
    return {
        "level": (level or cfg.level or LOG_LEVEL).upper(),
        "json_output": bool(json_output) if json_output is not None else False,
        "console": cfg.console.enabled if console is None else console,
        "console_colorize": cfg.console.colorize,
        "file_output": cfg.file.enabled if file_output is None else file_output,
        "file_path": cfg.file.path or LOG_PATH,
        "rotation": cfg.rotation or LOG_ROTATION,
        "retention": cfg.retention or LOG_RETENTION,
    }


def _configure_stdlib_logging(level: str) -> None:
    """Redirect stdlib loggers to loguru so runtime logs share the same sinks."""
    logging.captureWarnings(True)
    handler = InterceptHandler()

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)

    managed_loggers = (
        "uvicorn",
        "uvicorn.error",
        "fastapi",
        "celery",
        "celery.app.trace",
        "celery.worker",
        "kombu",
        "asyncio",
    )
    for logger_name in managed_loggers:
        managed_logger = logging.getLogger(logger_name)
        managed_logger.handlers = [handler]
        managed_logger.propagate = False
        managed_logger.setLevel(level)

    # Suppress uvicorn.access — our logging_middleware already logs requests
    # with richer context (duration, request_id, session_id).
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.handlers = []
    uvicorn_access.propagate = False
    uvicorn_access.setLevel(logging.CRITICAL)


def json_serializer(record: Dict[str, Any]) -> str:
    """Serialize log record to JSON for structured logging."""
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

    if request_id:
        log_entry["request_id"] = request_id
    if session_id:
        log_entry["session_id"] = session_id

    extra_fields = {
        key: value
        for key, value in record["extra"].items()
        if key not in ("name", "context_str") and not key.startswith("_")
    }
    if extra_fields:
        log_entry["context"] = extra_fields

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

    return orjson.dumps(log_entry, default=str).decode() + "\n"


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

    for key, value in extra.items():
        if key not in ("name", "context_str") and not key.startswith("_"):
            context_parts.append(f"{key}={value}")

    return f" | {' '.join(context_parts)}" if context_parts else ""


def patcher(record: Dict[str, Any]) -> None:
    """Patch record with default values and computed fields."""
    if "name" not in record["extra"]:
        record["extra"]["name"] = "root"
    record["extra"]["context_str"] = format_context_string(record)


def setup_logging(
    level: Optional[str] = None,
    json_output: Optional[bool] = None,
    console: Optional[bool] = None,
    file_output: Optional[bool] = None,
) -> None:
    """Setup unified logging sinks and stdlib interception."""
    options = _resolve_logging_options(level, json_output, console, file_output)

    logger.remove()
    logger.configure(patcher=patcher)
    _configure_stdlib_logging(options["level"])

    if options["console"]:
        logger.add(
            sys.stderr,
            format=CONSOLE_FORMAT,
            level=options["level"],
            colorize=options["console_colorize"],
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )

    log_path = Path(options["file_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if options["file_output"]:
        logger.add(
            str(log_path),
            format=FILE_FORMAT,
            level=options["level"],
            rotation=options["rotation"],
            retention=options["retention"],
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )

    if options["json_output"]:
        json_log_path = log_path.with_suffix(".jsonl")
        logger.add(
            str(json_log_path),
            format=json_serializer,
            level=options["level"],
            rotation=options["rotation"],
            retention=options["retention"],
            compression="zip",
            enqueue=True,
            serialize=False,
        )

    logger.bind(name="root").info(
        "Logging system initialized",
        log_level=options["level"],
        log_path=str(log_path),
        json_output=options["json_output"],
        console_output=options["console"],
    )


def get_logger(name: str = "root"):
    """Get a logger instance with optional name binding."""
    return logger.bind(name=name)


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


# Paths that are polled frequently — log at DEBUG to reduce noise.
_QUIET_PATHS = frozenset({"/sessions", "/monitoring/health"})
_QUIET_PREFIXES = ("/alpha-lab/search-jobs/",)


def log_api_request(
    method: str, path: str, status_code: int, duration_ms: float, **kwargs
) -> None:
    """Log API request for monitoring.

    High-frequency polling endpoints (search-job status, sessions, health)
    are logged at DEBUG to keep the console readable.
    """
    if status_code >= 500:
        level = "ERROR"
    elif status_code >= 400:
        level = "WARNING"
    elif path in _QUIET_PATHS or any(path.startswith(p) for p in _QUIET_PREFIXES):
        level = "DEBUG"
    else:
        level = "INFO"
    getattr(logger, level.lower())(
        f"API {method} {path} {status_code} ({duration_ms:.0f}ms)",
        api_method=method,
        api_path=path,
        api_status=status_code,
        api_duration_ms=duration_ms,
        **kwargs,
    )


async def logging_middleware(request, call_next):
    """FastAPI middleware for request logging with correlation IDs."""
    import time
    import uuid

    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    session_id = request.headers.get("X-Session-ID")
    set_request_context(request_id, session_id)

    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_api_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_api_request(
            method=request.method,
            path=request.url.path,
            status_code=500,
            duration_ms=duration_ms,
            error=str(exc),
        )
        raise
    finally:
        clear_request_context()


try:
    setup_logging()
except Exception:
    logger.add(sys.stderr, level="INFO")
