"""
Celery application configuration for distributed task processing.
Handles concurrent backtesting and other async tasks.
"""

from celery import Celery
from kombu import Exchange, Queue
import os

# Celery app instance
app = Celery('quant_tasks')

# Configuration
app.conf.update(
    # Broker settings
    broker_url=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1'),
    
    # Task settings
    task_serializer='msgpack',
    result_serializer='msgpack',
    accept_content=['msgpack', 'json'],
    timezone='Asia/Shanghai',
    enable_utc=True,
    
    # Worker settings
    worker_concurrency=4,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,  # Prevent memory leaks
    
    # Task execution
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3000,  # 50 minutes soft limit
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Result backend
    result_expires=86400,  # 24 hours
    result_compression='gzip',
    
    # Routing
    task_routes={
        'src.tasks.backtest.*': {'queue': 'backtest'},
        'src.tasks.analysis.*': {'queue': 'analysis'},
    },
    
    # Queues
    task_queues=(
        Queue('default', Exchange('default'), routing_key='default'),
        Queue('backtest', Exchange('backtest'), routing_key='backtest'),
        Queue('analysis', Exchange('analysis'), routing_key='analysis'),
    ),
)

# Auto-discover tasks
app.autodiscover_tasks(['src.tasks'])

if __name__ == '__main__':
    app.start()
