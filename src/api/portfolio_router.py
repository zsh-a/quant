"""
Portfolio API Router - API endpoints for portfolio management.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from loguru import logger

from src.portfolio.portfolio_manager import (
    PortfolioManager,
    StrategyConfig,
    WeightMethod
)
from src.portfolio.backtest import PortfolioBacktester, PortfolioBacktestResult
from src.portfolio import persistence as portfolio_db
from src.strategies.registry import StrategyRegistry


router = APIRouter(prefix="/portfolio", tags=["portfolio"])

# In-memory cache — loaded from DB on startup
PORTFOLIOS: Dict[str, PortfolioManager] = {}
BACKTEST_RESULTS: Dict[str, PortfolioBacktestResult] = {}
_PORTFOLIO_CONFIGS: Dict[str, Dict] = {}  # raw config for persistence


class CreatePortfolioRequest(BaseModel):
    """Request to create a portfolio"""
    name: str
    strategies: List[Dict]  # [{name, strategy, params, weight}]
    weight_method: str = "equal"  # equal, vol_inverse, sharpe, custom
    rebalance_frequency: str = "weekly"


class UpdateWeightsRequest(BaseModel):
    """Request to update portfolio weights"""
    weights: Dict[str, float]


from src.api.validators import DateStr, NonNegativeFloat


class PortfolioBacktestRequest(BaseModel):
    """Request to run portfolio backtest"""
    start_date: DateStr
    end_date: DateStr
    symbols: List[str]
    initial_capital: NonNegativeFloat = 1000000.0


@router.post("")
async def create_portfolio(req: CreatePortfolioRequest):
    """Create a new portfolio"""
    try:
        # Parse weight method
        method_map = {
            "equal": WeightMethod.EQUAL,
            "vol_inverse": WeightMethod.VOLATILITY_INVERSE,
            "sharpe": WeightMethod.SHARPE_WEIGHTED,
            "custom": WeightMethod.CUSTOM
        }
        weight_method = method_map.get(req.weight_method, WeightMethod.EQUAL)
        
        # Create strategy configs
        strategy_configs = []
        for s in req.strategies:
            # Import strategy class dynamically
            strategy_class = StrategyRegistry.get_strategy_class(s.get('strategy', 'jsg'))
            if strategy_class is None:
                raise HTTPException(400, f"Unknown strategy: {s.get('strategy')}")
            
            config = StrategyConfig(
                name=s.get('name', s.get('strategy')),
                strategy_class=strategy_class,
                params=s.get('params', {}),
                initial_weight=s.get('weight', 0),
                enabled=True
            )
            strategy_configs.append(config)
        
        # Create portfolio manager
        portfolio = PortfolioManager(
            strategies=strategy_configs,
            weight_method=weight_method,
            rebalance_frequency=req.rebalance_frequency
        )
        
        portfolio_id = f"pf_{req.name}_{len(PORTFOLIOS)}"
        PORTFOLIOS[portfolio_id] = portfolio

        raw_config = {
            "strategies": req.strategies,
            "weight_method": req.weight_method,
            "rebalance_frequency": req.rebalance_frequency,
        }
        _PORTFOLIO_CONFIGS[portfolio_id] = raw_config
        portfolio_db.save_portfolio(portfolio_id, req.name, raw_config)

        logger.info(f"Created portfolio: {portfolio_id}")
        
        return {
            "portfolio_id": portfolio_id,
            "name": req.name,
            "strategies": [s.get('name') for s in req.strategies],
            "weights": portfolio.get_weights(),
            "weight_method": req.weight_method
        }
    
    except Exception as e:
        logger.error(f"Create portfolio error: {e}")
        raise HTTPException(500, str(e))


@router.get("/{portfolio_id}")
async def get_portfolio(portfolio_id: str):
    """Get portfolio details"""
    portfolio = PORTFOLIOS.get(portfolio_id)
    if not portfolio:
        raise HTTPException(404, "Portfolio not found")
    
    return {
        "portfolio_id": portfolio_id,
        "stats": portfolio.get_portfolio_stats()
    }


@router.get("")
async def list_portfolios():
    """List all portfolios"""
    portfolios = []
    for pid, p in PORTFOLIOS.items():
        portfolios.append({
            "portfolio_id": pid,
            "n_strategies": len(p.strategies),
            "weights": p.get_weights()
        })
    return {"portfolios": portfolios}


@router.put("/{portfolio_id}/weights")
async def update_weights(portfolio_id: str, req: UpdateWeightsRequest):
    """Update portfolio weights"""
    portfolio = PORTFOLIOS.get(portfolio_id)
    if not portfolio:
        raise HTTPException(404, "Portfolio not found")
    
    portfolio.set_weights(req.weights)
    
    return {
        "portfolio_id": portfolio_id,
        "weights": portfolio.get_weights()
    }


@router.post("/{portfolio_id}/backtest")
async def run_portfolio_backtest(portfolio_id: str, req: PortfolioBacktestRequest):
    """Run portfolio backtest"""
    portfolio = PORTFOLIOS.get(portfolio_id)
    if not portfolio:
        raise HTTPException(404, "Portfolio not found")
    
    try:
        backtester = PortfolioBacktester(
            portfolio_manager=portfolio,
            initial_capital=req.initial_capital
        )
        
        result = backtester.run(
            start_date=req.start_date,
            end_date=req.end_date,
            symbols=req.symbols
        )
        
        # Store result (memory + DB)
        BACKTEST_RESULTS[portfolio_id] = result
        portfolio_db.save_backtest_result(portfolio_id, {
            "portfolio_id": result.portfolio_id,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "final_equity": result.final_equity,
            "strategy_results": result.strategy_results,
            "trades": result.trades,
            "equity_history": result.equity_history,
            "weights_history": result.weights_history,
        })

        return {
            "portfolio_id": result.portfolio_id,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "final_equity": result.final_equity,
            "strategy_results": result.strategy_results,
            "n_trades": len(result.trades)
        }
    
    except Exception as e:
        logger.error(f"Portfolio backtest error: {e}")
        raise HTTPException(500, str(e))


@router.get("/{portfolio_id}/result")
async def get_backtest_result(portfolio_id: str):
    """Get portfolio backtest result"""
    result = BACKTEST_RESULTS.get(portfolio_id)
    if not result:
        raise HTTPException(404, "Backtest result not found")
    
    return {
        "portfolio_id": result.portfolio_id,
        "total_return": result.total_return,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "equity_history": result.equity_history,
        "weights_history": result.weights_history,
        "trades": result.trades[:100]  # Limit trades returned
    }


