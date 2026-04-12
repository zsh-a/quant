"""
Optimizer API Router - API endpoints for parameter optimization.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional
from loguru import logger
import uuid

from src.optimizer.optimizer import (
    ParameterOptimizer,
    ParamSpec,
    OptimizationMethod,
    OptimizationObjective,
    OptimizationReport
)
from src.strategies.registry import StrategyRegistry


router = APIRouter(prefix="/optimize", tags=["optimizer"])

# In-memory storage for optimization tasks
OPTIMIZATION_TASKS: Dict[str, Dict] = {}


class ParamSpecRequest(BaseModel):
    """Parameter specification"""
    name: str
    param_type: str  # int, float, categorical
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List] = None


class OptimizeRequest(BaseModel):
    """Optimization request"""
    strategy: str  # jsg, rotation
    param_space: List[ParamSpecRequest]
    method: str = "grid"  # grid, random, bayesian
    objective: str = "max_sharpe"  # max_sharpe, max_return, min_drawdown, max_calmar
    n_iterations: int = 50
    backtest_config: Dict  # start_date, end_date, symbols, etc.


@router.post("")
async def submit_optimization(req: OptimizeRequest, background_tasks: BackgroundTasks):
    """Submit optimization task"""
    task_id = str(uuid.uuid4())[:8]
    
    OPTIMIZATION_TASKS[task_id] = {
        'status': 'pending',
        'progress': 0,
        'result': None,
        'error': None
    }
    
    # Run optimization in background
    background_tasks.add_task(
        _run_optimization,
        task_id,
        req
    )
    
    return {
        'task_id': task_id,
        'status': 'pending',
        'message': 'Optimization task submitted'
    }


@router.get("/{task_id}")
async def get_optimization_status(task_id: str):
    """Get optimization task status"""
    task = OPTIMIZATION_TASKS.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    
    response = {
        'task_id': task_id,
        'status': task['status'],
        'progress': task['progress']
    }
    
    if task['status'] == 'completed' and task['result']:
        report = task['result']
        response['result'] = {
            'best_params': report.best_params,
            'best_score': report.best_score,
            'param_importance': report.param_importance,
            'elapsed_time': report.elapsed_time,
            'n_iterations': report.n_iterations
        }
    
    if task['error']:
        response['error'] = task['error']
    
    return response


@router.get("/{task_id}/results")
async def get_optimization_results(task_id: str, top_n: int = 20):
    """Get top optimization results"""
    task = OPTIMIZATION_TASKS.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    
    if task['status'] != 'completed' or not task['result']:
        raise HTTPException(400, "Optimization not completed")
    
    report: OptimizationReport = task['result']
    
    # Sort by score descending
    sorted_results = sorted(report.results, key=lambda r: r.score, reverse=True)
    
    return {
        'task_id': task_id,
        'best_params': report.best_params,
        'best_score': report.best_score,
        'top_results': [
            {
                'params': r.params,
                'sharpe_ratio': r.sharpe_ratio,
                'total_return': r.total_return,
                'max_drawdown': r.max_drawdown,
                'score': r.score
            }
            for r in sorted_results[:top_n]
        ]
    }


@router.get("/{task_id}/heatmap")
async def get_param_heatmap(task_id: str, param1: str, param2: str):
    """Get parameter heatmap data"""
    task = OPTIMIZATION_TASKS.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    
    if task['status'] != 'completed' or not task.get('optimizer'):
        raise HTTPException(400, "Optimization not completed")
    
    optimizer: ParameterOptimizer = task['optimizer']
    heatmap = optimizer.get_heatmap_data(param1, param2)
    
    return {
        'task_id': task_id,
        'heatmap': heatmap
    }


@router.delete("/{task_id}")
async def cancel_optimization(task_id: str):
    """Cancel optimization task"""
    task = OPTIMIZATION_TASKS.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    
    task['status'] = 'cancelled'
    
    return {'task_id': task_id, 'status': 'cancelled'}


def _run_optimization(task_id: str, req: OptimizeRequest):
    """Run optimization task"""
    try:
        OPTIMIZATION_TASKS[task_id]['status'] = 'running'
        
        # Load strategy class
        strategy_class = StrategyRegistry.get_strategy_class(req.strategy)
        if not strategy_class:
            raise ValueError(f"Unknown strategy: {req.strategy}")
        
        # Build param space
        param_space = {}
        for spec in req.param_space:
            param_space[spec.name] = ParamSpec(
                name=spec.name,
                param_type=spec.param_type,
                low=spec.low,
                high=spec.high,
                step=spec.step,
                choices=spec.choices
            )
        
        # Parse objective
        objective_map = {
            'max_sharpe': OptimizationObjective.MAX_SHARPE,
            'max_return': OptimizationObjective.MAX_RETURN,
            'min_drawdown': OptimizationObjective.MIN_DRAWDOWN,
            'max_calmar': OptimizationObjective.MAX_CALMAR
        }
        objective = objective_map.get(req.objective, OptimizationObjective.MAX_SHARPE)
        
        # Create optimizer
        optimizer = ParameterOptimizer(
            strategy_class=strategy_class,
            param_space=param_space,
            objective=objective
        )
        
        # Create backtest function
        def backtest_fn(params: Dict) -> Dict:
            return _run_single_backtest(
                strategy_class,
                params,
                req.backtest_config
            )
        
        # Run optimization
        if req.method == 'grid':
            report = optimizer.grid_search(backtest_fn, max_combinations=req.n_iterations)
        elif req.method == 'random':
            report = optimizer.random_search(backtest_fn, n_iterations=req.n_iterations)
        elif req.method == 'bayesian':
            report = optimizer.bayesian_optimize(backtest_fn, n_iterations=req.n_iterations)
        else:
            raise ValueError(f"Unknown method: {req.method}")
        
        OPTIMIZATION_TASKS[task_id]['status'] = 'completed'
        OPTIMIZATION_TASKS[task_id]['progress'] = 100
        OPTIMIZATION_TASKS[task_id]['result'] = report
        OPTIMIZATION_TASKS[task_id]['optimizer'] = optimizer
        
        logger.info(f"Optimization {task_id} completed")
    
    except Exception as e:
        logger.error(f"Optimization {task_id} failed: {e}")
        OPTIMIZATION_TASKS[task_id]['status'] = 'failed'
        OPTIMIZATION_TASKS[task_id]['error'] = str(e)


def _run_single_backtest(strategy_class, params: Dict, config: Dict) -> Dict:
    """Run a single backtest with given params"""
    from src.core.engine import TradingEngine
    from src.core.backtest_broker import BacktestBroker
    from src.core.data_stream import DBDataStream
    
    try:
        # Initialize db_client for strategies that need it
        db_client = None
        try:
            from src.market_data.db import DB
            db_client = DB()
        except Exception as e:
            logger.debug(f"Could not initialize DB: {e}")
        
        # Some strategies require db_client
        strategy_init_kwargs = {**params}
        
        # Check if strategy requires db_client by inspecting __init__
        import inspect
        sig = inspect.signature(strategy_class.__init__)
        if 'db_client' in sig.parameters:
            if db_client is None:
                raise ValueError("Strategy requires db_client but DB not available")
            strategy_init_kwargs['db_client'] = db_client
        
        # Create components
        strategy = strategy_class(**strategy_init_kwargs)
        broker = BacktestBroker(
            initial_capital=config.get('initial_capital', 1000000),
            commission_rate=0.0003
        )
        data_stream = DBDataStream(
            symbols=config.get('symbols', []),
            start_date=config.get('start_date', '2023-01-01'),
            end_date=config.get('end_date', '2024-01-01')
        )
        
        # Run engine
        engine = TradingEngine(
            strategy=strategy,
            broker=broker,
            data_stream=data_stream
        )
        
        result = engine.run()
        
        # Calculate metrics
        equity_history = result.get('equity_history', [])
        if len(equity_history) >= 2:
            import numpy as np
            
            equities = [e.get('total_equity', 0) for e in equity_history]
            returns = [(equities[i] - equities[i-1]) / equities[i-1] 
                      for i in range(1, len(equities)) if equities[i-1] > 0]
            
            if returns:
                mean_ret = np.mean(returns)
                std_ret = np.std(returns)
                sharpe = (mean_ret * 252) / (std_ret * np.sqrt(252)) if std_ret > 0 else 0
            else:
                sharpe = 0
            
            total_return = (equities[-1] - equities[0]) / equities[0] if equities[0] > 0 else 0
            
            # Max drawdown
            peak = equities[0]
            max_dd = 0
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
        else:
            sharpe = 0
            total_return = 0
            max_dd = 0
        
        return {
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'n_trades': len(result.get('trades', []))
        }
    
    except Exception as e:
        logger.warning(f"Backtest failed: {e}")
        return {
            'sharpe_ratio': -999,
            'total_return': -999,
            'max_drawdown': 1,
            'n_trades': 0
        }
