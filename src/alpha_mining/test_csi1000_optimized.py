from src.alpha_mining.pipeline import AlphaMiningPipeline
import sys
from loguru import logger

# Ensure logging is informative
logger.remove()
logger.add(sys.stderr, level="INFO")

def run_optimized_mining():
    # CSI 1000 Mining Task
    # Period: 2024 (Full Year)
    # Iterations: 5 (Demonstration)
    pipeline = AlphaMiningPipeline(
        start_date='2024-01-01',
        end_date='2025-01-01',
        index_code='000852'
    )
    
    try:
        pipeline.run(iterations=5)
    except Exception as e:
        logger.exception(f"Mining failed: {e}")

if __name__ == "__main__":
    run_optimized_mining()
