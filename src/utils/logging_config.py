"""
Unified logging configuration for the quantitative trading platform.
Uses loguru for structured, configurable logging with rotation and retention.
"""

import sys
from pathlib import Path
from loguru import logger

LOG_LEVEL = "INFO"
LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)
LOG_ROTATION = "100 MB"
LOG_RETENTION = "30 days"
LOG_PATH = "logs/quant_{time:YYYY-MM-DD}.log"


def setup_logging():
    """Setup global logging configuration."""
    logger.remove()

    level = LOG_LEVEL
    log_format = LOG_FORMAT

    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    log_dir = Path(LOG_PATH).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        LOG_PATH,
        format=log_format,
        level=level,
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION,
        compression="zip",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

    logger.info(f"Logging system initialized: level={level}")

    return logger


def get_logger(name: str = "root"):
    """Get a logger instance with optional name binding."""
    if name:
        return logger.bind(name=name)
    return logger


def log_performance(operation: str, duration: float, **kwargs):
    """Log performance metrics in a structured format."""
    logger.info(
        f"PERF: {operation} completed in {duration:.3f}s",
        operation=operation,
        duration=duration,
        **kwargs,
    )


def log_trade(action: str, symbol: str, quantity: float, price: float, **kwargs):
    """Log trade execution in a structured format."""
    logger.info(
        f"TRADE: {action} {quantity} {symbol} @ {price}",
        action=action,
        symbol=symbol,
        quantity=quantity,
        price=price,
        **kwargs,
    )


def log_error_with_context(error: Exception, context: dict):
    """Log error with full context for debugging."""
    logger.error(
        f"Error occurred: {error}",
        error_type=type(error).__name__,
        error_message=str(error),
        **context,
    )


try:
    setup_logging()
except Exception:
    logger.add(sys.stderr, level="INFO")
