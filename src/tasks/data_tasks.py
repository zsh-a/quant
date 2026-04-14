"""
Data synchronization tasks for the quantitative trading platform.
"""

from loguru import logger

from src.market_data.processors.tdx import TDXProcess
from src.tasks.celery_app import app

@app.task(
    name='src.tasks.data_tasks.sync_financial_data',
    autoretry_for=(ConnectionError, OSError, TimeoutError),
    retry_backoff=True,
    max_retries=3,
)
def sync_financial_data(start_year=None):
    """
    Synchronize financial data from TDX incrementally.
    """
    logger.info("Starting financial data synchronization...")
    try:
        proc = TDXProcess()
        proc.update_fincial_db(start_year=start_year)
        logger.info("Financial data synchronization completed.")
        return {"status": "success", "message": "Financial data synced successfully"}
    except Exception as e:
        logger.error(f"Financial data synchronization failed: {e}")
        return {"status": "error", "message": str(e)}
