"""
Prometheus metrics for system monitoring.
Tracks API performance, backtest execution, and system health.
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time
from typing import Callable
from loguru import logger

# API Metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Backtest Metrics
backtest_duration = Histogram(
    'backtest_duration_seconds',
    'Backtest execution time',
    ['strategy', 'mode'],
    buckets=[10, 30, 60, 120, 300, 600, 1800, 3600]
)

backtest_total = Counter(
    'backtest_total',
    'Total backtests executed',
    ['strategy', 'status']
)

backtest_trades = Histogram(
    'backtest_trades_count',
    'Number of trades per backtest',
    ['strategy'],
    buckets=[0, 10, 50, 100, 500, 1000, 5000]
)

# Session Metrics
active_sessions = Gauge(
    'active_sessions',
    'Number of active trading sessions'
)

session_progress = Gauge(
    'session_progress_percent',
    'Session progress percentage',
    ['session_id']
)

# System Metrics
system_cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'CPU usage percentage'
)

system_memory_usage = Gauge(
    'system_memory_usage_bytes',
    'Memory usage in bytes'
)

system_disk_usage = Gauge(
    'system_disk_usage_percent',
    'Disk usage percentage'
)

# Database Metrics
db_query_duration = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['operation'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

db_connections = Gauge(
    'db_connections_active',
    'Active database connections'
)

# Task Queue Metrics
celery_tasks_active = Gauge(
    'celery_tasks_active',
    'Number of active Celery tasks',
    ['queue']
)

celery_tasks_total = Counter(
    'celery_tasks_total',
    'Total Celery tasks',
    ['queue', 'status']
)

celery_task_duration = Histogram(
    'celery_task_duration_seconds',
    'Celery task execution time',
    ['task_name'],
    buckets=[10, 30, 60, 120, 300, 600, 1800, 3600]
)

# WebSocket Metrics
websocket_connections = Gauge(
    'websocket_connections_active',
    'Active WebSocket connections'
)

websocket_messages = Counter(
    'websocket_messages_total',
    'Total WebSocket messages',
    ['type']
)

# Application Info
app_info = Info(
    'app_info',
    'Application information'
)

# Set application info
app_info.info({
    'version': '3.0.0',
    'phase': 'Phase 3',
    'environment': 'development'
})


def track_api_request(method: str, endpoint: str):
    """Decorator to track API request metrics"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                api_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status=status
                ).inc()
                api_request_duration.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
        
        return wrapper
    return decorator


def track_backtest(strategy: str, mode: str):
    """Decorator to track backtest metrics"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                
                # Track trade count if available
                if isinstance(result, dict) and 'total_trades' in result:
                    backtest_trades.labels(strategy=strategy).observe(
                        result['total_trades']
                    )
                
                return result
            except Exception as e:
                status = 'failed'
                logger.error(f"Backtest failed: {e}")
                raise
            finally:
                duration = time.time() - start_time
                backtest_duration.labels(
                    strategy=strategy,
                    mode=mode
                ).observe(duration)
                backtest_total.labels(
                    strategy=strategy,
                    status=status
                ).inc()
        
        return wrapper
    return decorator


def update_system_metrics():
    """Update system resource metrics"""
    try:
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        system_cpu_usage.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        system_memory_usage.set(memory.used)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        system_disk_usage.set(disk.percent)
        
    except ImportError:
        logger.warning("psutil not installed, system metrics unavailable")
    except Exception as e:
        logger.error(f"Failed to update system metrics: {e}")
