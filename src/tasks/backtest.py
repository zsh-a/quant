"""
Backtest tasks for Celery distributed processing.
"""

from celery import Task
from src.tasks.celery_app import app
from src.core.engine import TradingEngine
from src.core.backtest_broker import BacktestBroker
from src.core.data_stream import DBDataStream
from src.strategies.registry import StrategyRegistry
from src.core.risk_manager import RiskManager
from src.market_data.db import DB
from session_db import SessionDB
from loguru import logger
import time
from datetime import datetime


class BacktestTask(Task):
    """Base task with progress tracking"""
    
    def update_progress(self, session_id: str, progress: float, message: str = ""):
        """Update task progress"""
        self.update_state(
            state='PROGRESS',
            meta={
                'session_id': session_id,
                'progress': progress,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
        )


@app.task(bind=True, base=BacktestTask, name='src.tasks.backtest.run_backtest')
def run_backtest_task(self, session_id: str, config: dict):
    """
    Run a backtest task asynchronously.
    
    Args:
        session_id: Session identifier
        config: Backtest configuration
            - symbol: Stock symbol
            - strategy: Strategy name
            - start_date: Start date
            - end_date: End date
            - params: Strategy parameters
            - initial_cash: Initial capital
            - commission: Commission rate
    
    Returns:
        dict: Backtest results
    """
    logger.info(f"Starting backtest task for session {session_id}")
    
    try:
        # Initialize database
        db_client = DB()
        session_db = SessionDB()
        
        # Update session status
        session_db.update_session_status(session_id, "running", 0.0)
        self.update_progress(session_id, 0, "Initializing backtest...")
        
        # Setup data stream
        symbols = [config['symbol']]
        stream = DBDataStream(
            db_client,
            symbols,
            config['start_date'],
            config['end_date'],
            chunk_size_months=config.get('chunk_size_months')
        )
        
        total_bars = getattr(stream, 'total_bars', 1)
        logger.info(f"Data stream initialized: {total_bars} bars")
        
        # Setup broker
        broker = BacktestBroker(
            db_client=db_client,
            initial_cash=config.get('initial_cash', 1000000.0),
            commission=config.get('commission', 0.0001)
        )
        
        # Setup risk manager if enabled
        risk_manager = None
        if config.get('enable_risk_management', False):
            risk_manager = RiskManager(
                initial_capital=config.get('initial_cash', 1000000.0)
            )
            broker.risk_manager = risk_manager
            logger.info("Risk manager enabled")
        
        # Setup strategy
        strategy_name = config["strategy"]
        strategy_params = config.get("params", {})

        strategy = StrategyRegistry.create_strategy(
            strategy_name, db_client, session_id=session_id, **strategy_params
        )

        if strategy is None:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        logger.info(f"Strategy initialized: {strategy_name}")
        
        # Progress callback
        last_update_time = time.time()
        last_db_write = time.time()
        
        def on_step(bars):
            nonlocal last_update_time, last_db_write
            
            # Update progress (throttle to every 2 seconds)
            current_time = time.time()
            if current_time - last_update_time >= 2.0:
                if total_bars > 0:
                    progress = (stream.idx / total_bars) * 100
                    self.update_progress(
                        session_id, 
                        progress,
                        f"Processing bar {stream.idx}/{total_bars}"
                    )
                    session_db.update_session_status(session_id, "running", progress)
                last_update_time = current_time
            
            # Persist equity and trades (throttle to every 2 seconds)
            if current_time - last_db_write >= 2.0:
                info = broker.get_account_info()
                
                new_equity_points = list(info.get('equity_history', []))
                if new_equity_points:
                    session_db.add_equity_points(session_id, new_equity_points)
                    broker.equity_history.clear()
                
                new_trades = list(info.get('trades', []))
                if new_trades:
                    session_db.add_trades(session_id, new_trades)
                    broker.trades.clear()
                
                last_db_write = current_time
        
        # Create and run engine
        self.update_progress(session_id, 5, "Starting backtest engine...")
        
        engine = TradingEngine(
            strategy=strategy,
            broker=broker,
            data_stream=stream,
            on_step=on_step,
            risk_manager=risk_manager
        )
        
        logger.info("Running backtest engine...")
        engine.run()
        
        # Get final results
        final_info = broker.get_account_info()
        
        # Update session status
        session_db.update_session_status(session_id, "completed", 100.0)
        self.update_progress(session_id, 100, "Backtest completed")
        
        logger.info(f"Backtest completed for session {session_id}")
        
        return {
            'session_id': session_id,
            'status': 'completed',
            'final_equity': final_info['total_equity'],
            'total_trades': len(final_info.get('trades', [])),
            'positions': final_info.get('positions', {}),
            'completed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Backtest task failed for session {session_id}: {e}")
        session_db.update_session_status(session_id, "failed", error=str(e))
        
        # Re-raise to mark task as failed
        raise


@app.task(name='src.tasks.backtest.cancel_backtest')
def cancel_backtest_task(task_id: str):
    """Cancel a running backtest task"""
    app.control.revoke(task_id, terminate=True)
    logger.info(f"Cancelled task: {task_id}")
    return {'status': 'cancelled', 'task_id': task_id}
