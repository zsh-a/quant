"""
Monitoring API router for health checks and metrics.
"""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from src.monitoring.health import health_checker
from src.monitoring.metrics import update_system_metrics
from src.monitoring.alerts import alert_manager
from loguru import logger

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns overall system health status.
    """
    try:
        health_status = health_checker.run_all_checks()
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus format.
    """
    try:
        # Update system metrics before returning
        update_system_metrics()
        
        # Generate Prometheus metrics
        return PlainTextResponse(
            content=generate_latest().decode('utf-8'),
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return PlainTextResponse(
            content=f"# Error generating metrics: {e}",
            status_code=500
        )


@router.get("/status")
async def system_status():
    """
    Detailed system status endpoint.
    Returns comprehensive system information.
    """
    try:
        import psutil
        
        # Get system info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get health status
        health = health_checker.run_all_checks()
        
        return {
            "status": health['status'],
            "timestamp": health['timestamp'],
            "system": {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "percent": memory.percent,
                    "used_mb": memory.used / (1024 * 1024),
                    "total_mb": memory.total / (1024 * 1024),
                    "available_mb": memory.available / (1024 * 1024)
                },
                "disk": {
                    "percent": disk.percent,
                    "used_gb": disk.used / (1024 ** 3),
                    "total_gb": disk.total / (1024 ** 3),
                    "free_gb": disk.free / (1024 ** 3)
                }
            },
            "health_checks": health['checks']
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/alert/test")
async def test_alert(
    title: str = "Test Alert",
    message: str = "This is a test alert",
    severity: str = "warning"
):
    """
    Test alert notification.
    Sends a test alert through configured notifiers.
    """
    try:
        alert_manager.send_alert(
            title=title,
            message=message,
            severity=severity,
            details={"test": True}
        )
        return {
            "status": "success",
            "message": "Test alert sent"
        }
    except Exception as e:
        logger.error(f"Failed to send test alert: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
