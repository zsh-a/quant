"""
Health check system for monitoring application status.
"""

from typing import Dict, Any, List
from datetime import datetime
import psutil
from loguru import logger


class HealthStatus:
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck:
    """System health checker"""
    
    def __init__(self):
        self.checks: List[callable] = []
        self.register_default_checks()
    
    def register_check(self, check_func: callable):
        """Register a health check function"""
        self.checks.append(check_func)
    
    def register_default_checks(self):
        """Register default health checks"""
        self.register_check(self.check_cpu)
        self.register_check(self.check_memory)
        self.register_check(self.check_disk)
    
    def check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"CPU usage critical: {cpu_percent}%"
            elif cpu_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"CPU usage high: {cpu_percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent}%"
            
            return {
                'component': 'cpu',
                'status': status,
                'message': message,
                'value': cpu_percent,
                'threshold': 80
            }
        except Exception as e:
            return {
                'component': 'cpu',
                'status': HealthStatus.UNHEALTHY,
                'message': f"Failed to check CPU: {e}",
                'error': str(e)
            }
    
    def check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critical: {memory.percent}%"
            elif memory.percent > 90:
                status = HealthStatus.DEGRADED
                message = f"Memory usage high: {memory.percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory.percent}%"
            
            return {
                'component': 'memory',
                'status': status,
                'message': message,
                'value': memory.percent,
                'used_mb': memory.used / (1024 * 1024),
                'total_mb': memory.total / (1024 * 1024),
                'threshold': 90
            }
        except Exception as e:
            return {
                'component': 'memory',
                'status': HealthStatus.UNHEALTHY,
                'message': f"Failed to check memory: {e}",
                'error': str(e)
            }
    
    def check_disk(self) -> Dict[str, Any]:
        """Check disk usage"""
        try:
            disk = psutil.disk_usage('/')
            
            if disk.percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Disk usage critical: {disk.percent}%"
            elif disk.percent > 90:
                status = HealthStatus.DEGRADED
                message = f"Disk usage high: {disk.percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk.percent}%"
            
            return {
                'component': 'disk',
                'status': status,
                'message': message,
                'value': disk.percent,
                'used_gb': disk.used / (1024 ** 3),
                'total_gb': disk.total / (1024 ** 3),
                'threshold': 90
            }
        except Exception as e:
            return {
                'component': 'disk',
                'status': HealthStatus.UNHEALTHY,
                'message': f"Failed to check disk: {e}",
                'error': str(e)
            }
    
    def check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            from src.market_data.db import DB

            db = DB()
            result = db.client.query("SELECT 1")

            if result.result_rows:
                return {
                    'component': 'database',
                    'status': HealthStatus.HEALTHY,
                    'message': 'Database connection OK'
                }
            else:
                return {
                    'component': 'database',
                    'status': HealthStatus.UNHEALTHY,
                    'message': 'Database query failed'
                }
        except Exception as e:
            return {
                'component': 'database',
                'status': HealthStatus.UNHEALTHY,
                'message': f"Database connection failed: {e}",
                'error': str(e)
            }
    
    def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity"""
        try:
            import redis
            
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            
            return {
                'component': 'redis',
                'status': HealthStatus.HEALTHY,
                'message': 'Redis connection OK'
            }
        except Exception as e:
            return {
                'component': 'redis',
                'status': HealthStatus.DEGRADED,
                'message': f"Redis connection failed: {e}",
                'error': str(e)
            }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = []
        overall_status = HealthStatus.HEALTHY
        
        # Run registered checks
        for check in self.checks:
            try:
                result = check()
                results.append(result)
                
                # Update overall status
                if result['status'] == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result['status'] == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                results.append({
                    'component': 'unknown',
                    'status': HealthStatus.UNHEALTHY,
                    'message': f"Check failed: {e}",
                    'error': str(e)
                })
                overall_status = HealthStatus.UNHEALTHY
        
        # Check database
        results.append(self.check_database())
        
        # Check Redis (optional)
        redis_check = self.check_redis()
        if redis_check['status'] != HealthStatus.HEALTHY:
            # Redis is optional, so only degrade if other checks are healthy
            if overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        results.append(redis_check)
        
        return {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'checks': results
        }


# Global health checker instance
health_checker = HealthCheck()
