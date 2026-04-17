"""
Health check system for monitoring application status.
"""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime
from typing import Any, Dict

import psutil

_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="health")


class HealthStatus:
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


def _classify(value: float, warn: float, crit: float) -> str:
    if value > crit:
        return HealthStatus.UNHEALTHY
    if value > warn:
        return HealthStatus.DEGRADED
    return HealthStatus.HEALTHY


class HealthCheck:
    def __init__(self):
        self._db = None
        self._redis = None

    # ---- local checks (instant) ----

    @staticmethod
    def check_cpu() -> Dict[str, Any]:
        try:
            pct = psutil.cpu_percent(interval=None)
            return {"component": "cpu", "status": _classify(pct, 80, 90), "value": pct}
        except Exception as e:
            return {"component": "cpu", "status": HealthStatus.UNHEALTHY, "error": str(e)}

    @staticmethod
    def check_memory() -> Dict[str, Any]:
        try:
            mem = psutil.virtual_memory()
            return {
                "component": "memory",
                "status": _classify(mem.percent, 90, 95),
                "value": mem.percent,
                "used_mb": round(mem.used / (1024 * 1024)),
                "total_mb": round(mem.total / (1024 * 1024)),
            }
        except Exception as e:
            return {"component": "memory", "status": HealthStatus.UNHEALTHY, "error": str(e)}

    @staticmethod
    def check_disk() -> Dict[str, Any]:
        try:
            disk = psutil.disk_usage("/")
            return {
                "component": "disk",
                "status": _classify(disk.percent, 90, 95),
                "value": disk.percent,
                "used_gb": round(disk.used / (1024**3), 1),
                "total_gb": round(disk.total / (1024**3), 1),
            }
        except Exception as e:
            return {"component": "disk", "status": HealthStatus.UNHEALTHY, "error": str(e)}

    # ---- remote checks (may be slow) ----

    def _check_database_inner(self) -> Dict[str, Any]:
        try:
            if self._db is None:
                from src.market_data.db import DB
                self._db = DB()
            self._db.client.query("SELECT 1")
            return {"component": "database", "status": HealthStatus.HEALTHY}
        except Exception as e:
            self._db = None
            return {"component": "database", "status": HealthStatus.UNHEALTHY, "error": str(e)}

    def _check_redis_inner(self) -> Dict[str, Any]:
        try:
            if self._redis is None:
                import redis
                self._redis = redis.Redis(host="localhost", port=6379, db=0, socket_timeout=2, socket_connect_timeout=2)
            self._redis.ping()
            return {"component": "redis", "status": HealthStatus.HEALTHY}
        except Exception as e:
            self._redis = None
            return {"component": "redis", "status": HealthStatus.DEGRADED, "error": str(e)}

    @staticmethod
    def _run_with_timeout(fn, timeout: float, component: str) -> Dict[str, Any]:
        """Run a check function with a hard timeout."""
        fut = _executor.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except FuturesTimeout:
            fut.cancel()
            return {"component": component, "status": HealthStatus.DEGRADED, "error": f"timeout ({timeout}s)"}
        except Exception as e:
            return {"component": component, "status": HealthStatus.UNHEALTHY, "error": str(e)}

    # ---- aggregate ----

    def run_all_checks(self) -> Dict[str, Any]:
        # Local checks: instant
        checks = [self.check_cpu(), self.check_memory(), self.check_disk()]

        # Remote checks: hard 3s timeout each, run concurrently
        db_fut = _executor.submit(self._check_database_inner)
        redis_fut = _executor.submit(self._check_redis_inner)

        for fut, component in [(db_fut, "database"), (redis_fut, "redis")]:
            try:
                checks.append(fut.result(timeout=3))
            except FuturesTimeout:
                fut.cancel()
                checks.append({"component": component, "status": HealthStatus.DEGRADED, "error": "timeout (3s)"})
            except Exception as e:
                checks.append({"component": component, "status": HealthStatus.UNHEALTHY, "error": str(e)})

        worst = HealthStatus.HEALTHY
        for c in checks:
            s = c["status"]
            if s == HealthStatus.UNHEALTHY:
                worst = HealthStatus.UNHEALTHY
            elif s == HealthStatus.DEGRADED and worst == HealthStatus.HEALTHY:
                worst = HealthStatus.DEGRADED

        return {"status": worst, "timestamp": datetime.now().isoformat(), "checks": checks}


health_checker = HealthCheck()
