import pandas as pd
from src.alpha_mining.pipeline import AlphaMiningPipeline
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(sys.stderr, level="DEBUG")

def test_pipeline():
    # Use a small set of stocks and a short period for testing
    pipeline = AlphaMiningPipeline(
        symbols=['sh.600000', 'sh.600036', 'sz.000001', 'sz.000002'],
        start_date='2024-01-01',
        end_date='2024-06-01'
    )
    
    logger.info("Starting Real LLM Alpha Mining Test...")
    
    # Run 2 iterations to verify the LLM can generate and refine
    try:
        pipeline.run(iterations=2)
        logger.info("Pipeline completed successfully!")
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")

if __name__ == "__main__":
    test_pipeline()
