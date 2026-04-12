"""
Parameter Optimizer - Strategy parameter optimization using grid search and Bayesian optimization.
"""

from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
import time


class OptimizationMethod(Enum):
    """Optimization methods"""
    GRID_SEARCH = "grid"
    RANDOM_SEARCH = "random"
    BAYESIAN = "bayesian"


class OptimizationObjective(Enum):
    """Optimization objectives"""
    MAX_SHARPE = "max_sharpe"
    MAX_RETURN = "max_return"
    MIN_DRAWDOWN = "min_drawdown"
    MAX_CALMAR = "max_calmar"


@dataclass
class ParamSpec:
    """Parameter specification"""
    name: str
    param_type: str  # int, float, categorical
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List] = None
    
    def generate_values(self) -> List:
        """Generate all values for grid search"""
        if self.param_type == 'categorical':
            return self.choices or []
        
        if self.low is None or self.high is None:
            return []
        
        if self.param_type == 'int':
            step = int(self.step or 1)
            return list(range(int(self.low), int(self.high) + 1, step))
        
        elif self.param_type == 'float':
            step = self.step or 0.01
            values = []
            v = self.low
            while v <= self.high:
                values.append(round(v, 6))
                v += step
            return values
        
        return []
    
    def sample_random(self) -> Any:
        """Sample a random value"""
        if self.param_type == 'categorical':
            return np.random.choice(self.choices)
        
        if self.param_type == 'int':
            return np.random.randint(int(self.low), int(self.high) + 1)
        
        elif self.param_type == 'float':
            return np.random.uniform(self.low, self.high)
        
        return None


@dataclass
class OptimizationResult:
    """Single optimization result"""
    params: Dict
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    calmar_ratio: float
    n_trades: int
    score: float  # Based on objective


@dataclass 
class OptimizationReport:
    """Full optimization report"""
    method: str
    objective: str
    best_params: Dict
    best_score: float
    results: List[OptimizationResult]
    param_importance: Dict[str, float]
    elapsed_time: float
    n_iterations: int


class ParameterOptimizer:
    """
    Strategy parameter optimizer.
    
    Supports grid search, random search, and Bayesian optimization.
    """
    
    def __init__(
        self,
        strategy_class: type,
        param_space: Dict[str, ParamSpec],
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        n_workers: int = 4
    ):
        self.strategy_class = strategy_class
        self.param_space = param_space
        self.objective = objective
        self.n_workers = n_workers
        
        self.results: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None
    
    def grid_search(
        self,
        backtest_fn: Callable[[Dict], Dict],
        max_combinations: int = 1000
    ) -> OptimizationReport:
        """
        Grid search over parameter space.
        
        Args:
            backtest_fn: Function that takes params dict and returns backtest result
            max_combinations: Maximum combinations to test
        
        Returns:
            OptimizationReport
        """
        logger.info("Starting grid search optimization")
        start_time = time.time()
        
        # Generate all parameter combinations
        param_names = list(self.param_space.keys())
        param_values = [self.param_space[name].generate_values() for name in param_names]
        
        all_combinations = list(product(*param_values))
        n_combinations = len(all_combinations)
        
        logger.info(f"Total parameter combinations: {n_combinations}")
        
        # Limit combinations
        if n_combinations > max_combinations:
            logger.warning(f"Limiting to {max_combinations} combinations")
            indices = np.random.choice(n_combinations, max_combinations, replace=False)
            all_combinations = [all_combinations[i] for i in indices]
        
        # Run backtests
        self.results = []
        for i, values in enumerate(all_combinations):
            params = dict(zip(param_names, values))
            
            try:
                backtest_result = backtest_fn(params)
                result = self._create_result(params, backtest_result)
                self.results.append(result)
                
                if i % 10 == 0:
                    logger.info(f"Progress: {i+1}/{len(all_combinations)}")
            
            except Exception as e:
                logger.warning(f"Backtest failed for params {params}: {e}")
        
        return self._create_report("grid", start_time)
    
    def random_search(
        self,
        backtest_fn: Callable[[Dict], Dict],
        n_iterations: int = 100
    ) -> OptimizationReport:
        """
        Random search over parameter space.
        
        Args:
            backtest_fn: Function that takes params dict and returns backtest result
            n_iterations: Number of random samples to test
        
        Returns:
            OptimizationReport
        """
        logger.info(f"Starting random search optimization ({n_iterations} iterations)")
        start_time = time.time()
        
        self.results = []
        for i in range(n_iterations):
            # Sample random parameters
            params = {
                name: spec.sample_random()
                for name, spec in self.param_space.items()
            }
            
            try:
                backtest_result = backtest_fn(params)
                result = self._create_result(params, backtest_result)
                self.results.append(result)
                
                if i % 10 == 0:
                    logger.info(f"Progress: {i+1}/{n_iterations}")
            
            except Exception as e:
                logger.warning(f"Backtest failed for params {params}: {e}")
        
        return self._create_report("random", start_time)
    
    def bayesian_optimize(
        self,
        backtest_fn: Callable[[Dict], Dict],
        n_iterations: int = 50,
        n_initial: int = 10
    ) -> OptimizationReport:
        """
        Bayesian optimization using Gaussian Process.
        
        Args:
            backtest_fn: Function that takes params dict and returns backtest result
            n_iterations: Total number of iterations
            n_initial: Number of initial random samples
        
        Returns:
            OptimizationReport
        """
        logger.info(f"Starting Bayesian optimization ({n_iterations} iterations)")
        start_time = time.time()
        
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
            from scipy.stats import norm
        except ImportError:
            logger.warning("sklearn not available, falling back to random search")
            return self.random_search(backtest_fn, n_iterations)
        
        # Prepare parameter bounds
        param_names = list(self.param_space.keys())
        bounds = []
        for name in param_names:
            spec = self.param_space[name]
            if spec.param_type == 'categorical':
                bounds.append((0, len(spec.choices) - 1))
            else:
                bounds.append((spec.low, spec.high))
        
        bounds = np.array(bounds)
        
        # Initial random samples
        X_samples = []
        y_samples = []
        self.results = []
        
        for i in range(n_initial):
            params = {
                name: spec.sample_random()
                for name, spec in self.param_space.items()
            }
            
            try:
                backtest_result = backtest_fn(params)
                result = self._create_result(params, backtest_result)
                self.results.append(result)
                
                # Convert params to array
                x = self._params_to_array(params, param_names)
                X_samples.append(x)
                y_samples.append(result.score)
            
            except Exception as e:
                logger.warning(f"Initial sample failed: {e}")
        
        if len(X_samples) < 3:
            logger.error("Not enough initial samples, falling back to random")
            return self.random_search(backtest_fn, n_iterations - n_initial)
        
        X = np.array(X_samples)
        y = np.array(y_samples)
        
        # Bayesian optimization loop
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=5,
            normalize_y=True
        )
        
        for i in range(n_initial, n_iterations):
            # Fit GP
            gp.fit(X, y)
            
            # Find next point using Expected Improvement
            x_next = self._acquisition_ei(gp, bounds, y.max())
            
            # Convert back to params
            params = self._array_to_params(x_next, param_names)
            
            try:
                backtest_result = backtest_fn(params)
                result = self._create_result(params, backtest_result)
                self.results.append(result)
                
                X = np.vstack([X, x_next.reshape(1, -1)])
                y = np.append(y, result.score)
                
                logger.info(f"Iteration {i+1}/{n_iterations}: score={result.score:.4f}")
            
            except Exception as e:
                logger.warning(f"Bayesian iteration failed: {e}")
        
        return self._create_report("bayesian", start_time)
    
    def _acquisition_ei(self, gp, bounds, y_best, n_samples=1000):
        """Expected Improvement acquisition function"""
        from scipy.stats import norm
        
        # Sample random points
        X_random = np.random.uniform(
            bounds[:, 0], bounds[:, 1],
            size=(n_samples, len(bounds))
        )
        
        mu, sigma = gp.predict(X_random, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        
        # EI
        z = (mu - y_best) / sigma
        ei = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
        
        return X_random[np.argmax(ei)]
    
    def _params_to_array(self, params: Dict, param_names: List[str]) -> np.ndarray:
        """Convert params dict to array"""
        arr = []
        for name in param_names:
            spec = self.param_space[name]
            value = params[name]
            
            if spec.param_type == 'categorical':
                idx = spec.choices.index(value) if value in spec.choices else 0
                arr.append(idx)
            else:
                arr.append(float(value))
        
        return np.array(arr)
    
    def _array_to_params(self, arr: np.ndarray, param_names: List[str]) -> Dict:
        """Convert array back to params dict"""
        params = {}
        for i, name in enumerate(param_names):
            spec = self.param_space[name]
            value = arr[i]
            
            if spec.param_type == 'categorical':
                idx = int(round(value))
                idx = max(0, min(idx, len(spec.choices) - 1))
                params[name] = spec.choices[idx]
            elif spec.param_type == 'int':
                params[name] = int(round(value))
            else:
                params[name] = float(value)
        
        return params
    
    def _create_result(self, params: Dict, backtest_result: Dict) -> OptimizationResult:
        """Create OptimizationResult from backtest output"""
        sharpe = backtest_result.get('sharpe_ratio', 0)
        total_return = backtest_result.get('total_return', 0)
        max_dd = backtest_result.get('max_drawdown', 1)
        n_trades = backtest_result.get('n_trades', 0)
        
        # Calculate Calmar ratio
        calmar = total_return / max_dd if max_dd > 0 else 0
        
        # Calculate score based on objective
        if self.objective == OptimizationObjective.MAX_SHARPE:
            score = sharpe
        elif self.objective == OptimizationObjective.MAX_RETURN:
            score = total_return
        elif self.objective == OptimizationObjective.MIN_DRAWDOWN:
            score = -max_dd  # Negative for minimization
        elif self.objective == OptimizationObjective.MAX_CALMAR:
            score = calmar
        else:
            score = sharpe
        
        return OptimizationResult(
            params=params,
            sharpe_ratio=sharpe,
            total_return=total_return,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            n_trades=n_trades,
            score=score
        )
    
    def _create_report(self, method: str, start_time: float) -> OptimizationReport:
        """Create optimization report"""
        elapsed = time.time() - start_time
        
        if not self.results:
            return OptimizationReport(
                method=method,
                objective=self.objective.value,
                best_params={},
                best_score=0,
                results=[],
                param_importance={},
                elapsed_time=elapsed,
                n_iterations=0
            )
        
        # Find best result
        best = max(self.results, key=lambda r: r.score)
        self.best_result = best
        
        # Calculate parameter importance (correlation with score)
        param_importance = self._calculate_importance()
        
        logger.info(f"Optimization complete in {elapsed:.1f}s")
        logger.info(f"Best params: {best.params}")
        logger.info(f"Best score: {best.score:.4f}")
        
        return OptimizationReport(
            method=method,
            objective=self.objective.value,
            best_params=best.params,
            best_score=best.score,
            results=self.results,
            param_importance=param_importance,
            elapsed_time=elapsed,
            n_iterations=len(self.results)
        )
    
    def _calculate_importance(self) -> Dict[str, float]:
        """Calculate parameter importance via correlation with score"""
        if len(self.results) < 5:
            return {}
        
        importance = {}
        scores = [r.score for r in self.results]
        
        for name, spec in self.param_space.items():
            if spec.param_type == 'categorical':
                continue
            
            values = [r.params.get(name, 0) for r in self.results]
            
            try:
                corr = np.corrcoef(values, scores)[0, 1]
                importance[name] = abs(corr) if not np.isnan(corr) else 0
            except Exception:
                importance[name] = 0
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance
    
    def get_heatmap_data(self, param1: str, param2: str) -> Dict:
        """Get data for parameter heatmap"""
        if not self.results:
            return {}
        
        data = []
        for r in self.results:
            data.append({
                param1: r.params.get(param1),
                param2: r.params.get(param2),
                'score': r.score
            })
        
        return {
            'param1': param1,
            'param2': param2,
            'data': data
        }
