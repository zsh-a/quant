#!/bin/bash
# Start Celery worker for backtest tasks

echo "Starting Celery worker..."

celery -A src.tasks.celery_app worker \
    --loglevel=info \
    --concurrency=4 \
    --queues=backtest,default \
    --max-tasks-per-child=100 \
    --time-limit=3600 \
    --soft-time-limit=3000

