"""
Unified logging configuration for the quantitative trading platform.
Uses loguru for structured, configurable logging with rotation and retention.
"""

import sys
from pathlib import Path
from loguru import logger
from src.utils.config import get_section

def setup_logging():
    """
    Setup global logging configuration based on config file.
    Should be called once at application startup.
    """
    # Remove default handler
    logger.remove()
    
    # Load logging configuration
    log_config = get_section('logging')
    
    level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', 
        '{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}')
    
    # Console logging
    console_config = log_config.get('console', {})
    if console_config.get('enabled', True):
        logger.add(
            sys.stderr,
            format=log_format,
            level=level,
            colorize=console_config.get('colorize', True),
            backtrace=True,
            diagnose=True
        )
    
    # File logging
    file_config = log_config.get('file', {})
    if file_config.get('enabled', True):
        log_path = file_config.get('path', 'logs/quant_{time:YYYY-MM-DD}.log')
        
        # Ensure log directory exists
        log_dir = Path(log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            format=log_format,
            level=level,
            rotation=log_config.get('rotation', '100 MB'),
            retention=log_config.get('retention', '30 days'),
            compression='zip',
            backtrace=True,
            diagnose=True,
            enqueue=True  # Thread-safe
        )
    
    logger.info(f"Logging system initialized: level={level}, "
               f"console={console_config.get('enabled', True)}, "
               f"file={file_config.get('enabled', True)}")
    
    return logger


def get_logger(name: str = None):
    """
    Get a logger instance with optional name binding.
    
    Args:
        name: Optional name to bind to the logger (e.g., module name)
    
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Convenience function for structured logging
def log_performance(operation: str, duration: float, **kwargs):
    """
    Log performance metrics in a structured format.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        **kwargs: Additional context to log
    """
    logger.info(f"PERF: {operation} completed in {duration:.3f}s", 
               operation=operation, 
               duration=duration, 
               **kwargs)


def log_trade(action: str, symbol: str, quantity: float, price: float, **kwargs):
    """
    Log trade execution in a structured format.
    
    Args:
        action: Trade action (BUY/SELL)
        symbol: Symbol traded
        quantity: Quantity traded
        price: Execution price
        **kwargs: Additional context
    """
    logger.info(f"TRADE: {action} {quantity} {symbol} @ {price}", 
               action=action,
               symbol=symbol,
               quantity=quantity,
               price=price,
               **kwargs)


def log_error_with_context(error: Exception, context: dict):
    """
    Log error with full context for debugging.
    
    Args:
        error: Exception object
        context: Dictionary of contextual information
    """
    logger.error(f"Error occurred: {error}", 
                error_type=type(error).__name__,
                error_message=str(error),
                **context)


# Initialize on module import if config is available
try:
    setup_logging()
except Exception as e:
    # Fallback to basic logging if config fails
    logger.add(sys.stderr, level="INFO")
    logger.warning(f"Failed to load logging config, using defaults: {e}")
