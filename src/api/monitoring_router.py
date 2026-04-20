"""
Monitoring API router for health checks and metrics.
"""

import anyio
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.monitoring.alerts import alert_manager
from src.monitoring.health import health_checker
from src.monitoring.metrics import update_alpha_lab_metrics, update_system_metrics

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/health")
async def health_check():
    try:
        return await anyio.to_thread.run_sync(health_checker.run_all_checks)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    try:
        update_system_metrics()
        update_alpha_lab_metrics()
        return PlainTextResponse(
            content=generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return PlainTextResponse(content=f"# Error: {e}", status_code=500)


@router.get("/status")
async def system_status():
    try:
        health = await anyio.to_thread.run_sync(health_checker.run_all_checks)
        return {
            "status": health["status"],
            "timestamp": health["timestamp"],
            "checks": health["checks"],
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/alert/test")
async def test_alert(
    title: str = "Test Alert",
    message: str = "This is a test alert",
    severity: str = "warning",
):
    try:
        alert_manager.send_alert(title=title, message=message, severity=severity, details={"test": True})
        return {"status": "success", "message": "Test alert sent"}
    except Exception as e:
        logger.error(f"Failed to send test alert: {e}")
        return {"status": "error", "error": str(e)}
