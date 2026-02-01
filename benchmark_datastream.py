#!/usr/bin/env python3
"""
Performance benchmark script for data stream optimizations.
Tests memory usage and loading speed of DBDataStream.
"""

import sys
import os
import time
import tracemalloc
import psutil
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.data_stream import DBDataStream
from db import DB
from loguru import logger

def format_bytes(bytes_val):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f}TB"

def benchmark_data_stream(symbols, start_date, end_date, chunk_size_months=None):
    """Benchmark DBDataStream with given parameters"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking DBDataStream")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Chunk size: {chunk_size_months or 'auto'} months")
    logger.info(f"{'='*60}\n")
    
    # Get process for memory monitoring
    process = psutil.Process(os.getpid())
    
    # Start memory tracking
    tracemalloc.start()
    mem_before = process.memory_info().rss
    start_time = time.time()
    
    # Initialize data stream
    db_client = DB()
    stream = DBDataStream(
        db_client, 
        symbols, 
        start_date, 
        end_date,
        chunk_size_months=chunk_size_months
    )
    
    init_time = time.time() - start_time
    mem_after_init = process.memory_info().rss
    mem_delta_init = mem_after_init - mem_before
    
    logger.info(f"✓ Initialization complete in {init_time:.2f}s")
    logger.info(f"  Memory used: {format_bytes(mem_delta_init)}")
    
    # Iterate through all bars
    bars_processed = 0
    chunks_seen = set()
    last_chunk = stream.chunks_loaded
    
    iteration_start = time.time()
    
    while True:
        bars = stream.next_bar()
        if bars is None:
            break
        
        bars_processed += 1
        
        # Track chunk changes
        if stream.chunks_loaded != last_chunk:
            chunks_seen.add(stream.chunks_loaded)
            last_chunk = stream.chunks_loaded
        
        # Log progress every 100 bars
        if bars_processed % 100 == 0:
            current_mem = process.memory_info().rss
            logger.debug(f"Processed {bars_processed} bars, "
                        f"memory: {format_bytes(current_mem)}")
    
    iteration_time = time.time() - iteration_start
    total_time = time.time() - start_time
    
    # Final memory measurement
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    mem_final = process.memory_info().rss
    mem_delta_total = mem_final - mem_before
    
    # Results
    logger.info(f"\n{'='*60}")
    logger.info(f"BENCHMARK RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Bars processed: {bars_processed}")
    logger.info(f"Chunks loaded: {stream.chunks_loaded}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"  - Initialization: {init_time:.2f}s ({init_time/total_time*100:.1f}%)")
    logger.info(f"  - Iteration: {iteration_time:.2f}s ({iteration_time/total_time*100:.1f}%)")
    logger.info(f"Throughput: {bars_processed/iteration_time:.2f} bars/sec")
    logger.info(f"\nMemory Usage:")
    logger.info(f"  - Initialization: {format_bytes(mem_delta_init)}")
    logger.info(f"  - Total increase: {format_bytes(mem_delta_total)}")
    logger.info(f"  - Peak (tracemalloc): {format_bytes(peak)}")
    logger.info(f"  - Per symbol: {format_bytes(mem_delta_total/len(symbols))}")
    logger.info(f"  - Per bar: {format_bytes(mem_delta_total/bars_processed)}")
    logger.info(f"{'='*60}\n")
    
    return {
        'bars_processed': bars_processed,
        'chunks_loaded': stream.chunks_loaded,
        'total_time': total_time,
        'init_time': init_time,
        'iteration_time': iteration_time,
        'throughput': bars_processed/iteration_time,
        'memory_init': mem_delta_init,
        'memory_total': mem_delta_total,
        'memory_peak': peak,
        'memory_per_symbol': mem_delta_total/len(symbols),
        'memory_per_bar': mem_delta_total/bars_processed
    }

def main():
    """Run benchmarks with different configurations"""
    
    logger.info("Starting DBDataStream Performance Benchmarks\n")
    
    # Test configurations
    configs = [
        {
            'name': 'Single Stock - 1 Year',
            'symbols': ['sh.600000'],
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'chunk_size_months': None
        },
        {
            'name': 'Single Stock - 5 Years',
            'symbols': ['sh.600000'],
            'start_date': '2019-01-01',
            'end_date': '2023-12-31',
            'chunk_size_months': None
        },
        {
            'name': '10 Stocks - 1 Year',
            'symbols': [f'sh.60000{i}' for i in range(10)],
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'chunk_size_months': None
        },
        {
            'name': '10 Stocks - 1 Year (3-month chunks)',
            'symbols': [f'sh.60000{i}' for i in range(10)],
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'chunk_size_months': 3
        }
    ]
    
    results = []
    
    for config in configs:
        try:
            logger.info(f"\n\n{'#'*60}")
            logger.info(f"# Test: {config['name']}")
            logger.info(f"{'#'*60}")
            
            result = benchmark_data_stream(
                symbols=config['symbols'],
                start_date=config['start_date'],
                end_date=config['end_date'],
                chunk_size_months=config['chunk_size_months']
            )
            result['config'] = config['name']
            results.append(result)
            
            # Cool down between tests
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Test '{config['name']}' failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    if results:
        logger.info(f"\n\n{'='*80}")
        logger.info("SUMMARY COMPARISON")
        logger.info(f"{'='*80}")
        logger.info(f"{'Test':<40} {'Throughput':<15} {'Memory':<15}")
        logger.info(f"{'-'*80}")
        for r in results:
            logger.info(f"{r['config']:<40} {r['throughput']:>10.2f} bars/s {format_bytes(r['memory_total']):>12}")
        logger.info(f"{'='*80}\n")

if __name__ == '__main__':
    main()
