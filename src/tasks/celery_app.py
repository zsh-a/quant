"""
Celery application configuration for distributed task processing.
Handles concurrent backtesting and other async tasks.
"""

import os

from celery import Celery
from celery.schedules import crontab
from celery.signals import setup_logging as celery_setup_logging
from kombu import Exchange, Queue

from src.config.paths import ensure_data_dirs
from src.utils.logging_config import setup_logging

ensure_data_dirs()

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
    worker_hijack_root_logger=False,

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
        'src.tasks.data_tasks.*': {'queue': 'default'},
        'src.tasks.automation.*': {'queue': 'automation'},
        'src.tasks.crypto_tasks.*': {'queue': 'automation'},
    },

    # Queues
    task_queues=(
        Queue('default', Exchange('default'), routing_key='default'),
        Queue('backtest', Exchange('backtest'), routing_key='backtest'),
        Queue('analysis', Exchange('analysis'), routing_key='analysis'),
        Queue('automation', Exchange('automation'), routing_key='automation'),
    ),

    beat_schedule={
        'scheduled-data-update-and-simulation': {
            'task': 'src.tasks.automation.run_automation_cycle',
            'schedule': crontab(hour=18, minute=0),
            'options': {'queue': 'automation'},
        }
    },

    # Explicit imports for task discovery
    imports=['src.tasks.backtest', 'src.tasks.data_tasks', 'src.tasks.automation', 'src.tasks.crypto_tasks'],
)

if __name__ == '__main__':
    app.start()


@celery_setup_logging.connect
def _configure_celery_logging(*args, **kwargs):
    """Route Celery logs through the shared logging configuration."""
    setup_logging()
