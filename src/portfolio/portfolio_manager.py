"""
Portfolio Manager - Multi-strategy portfolio management system.
Supports multiple strategies with configurable weight allocation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class WeightMethod(Enum):
    """Weight allocation methods"""

    EQUAL = "equal"  # 等权重
    VOLATILITY_INVERSE = "vol_inverse"  # 波动率倒数
    SHARPE_WEIGHTED = "sharpe"  # 夏普比率加权
    CUSTOM = "custom"  # 自定义权重


@dataclass
class StrategyConfig:
    """Strategy configuration"""

    name: str
    strategy_class: type
    params: Dict
    initial_weight: float = 0.0
    enabled: bool = True


@dataclass
class PortfolioSignal:
    """Combined portfolio signal"""

    symbol: str
    direction: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1
    source_strategies: List[str]
    timestamp: str


@dataclass
class PortfolioState:
    """Portfolio state"""

    weights: Dict[str, float] = field(default_factory=dict)
    strategy_returns: Dict[str, List[float]] = field(default_factory=dict)
    strategy_positions: Dict[str, Dict] = field(default_factory=dict)
    combined_equity: List[float] = field(default_factory=list)
    last_rebalance: Optional[str] = None


class PortfolioManager:
    """
    Multi-strategy portfolio manager.

    Manages multiple trading strategies and combines their signals
    with configurable weight allocation methods.
    """

    def __init__(
        self,
        strategies: List[StrategyConfig],
        weight_method: WeightMethod = WeightMethod.EQUAL,
        rebalance_frequency: str = "weekly",  # daily, weekly, monthly
        min_weight: float = 0.05,
        max_weight: float = 0.40,
    ):
        self.strategy_configs = strategies
        self.weight_method = weight_method
        self.rebalance_frequency = rebalance_frequency
        self.min_weight = min_weight
        self.max_weight = max_weight

        self.strategies: Dict[str, object] = {}
        self.state = PortfolioState()

        # Initialize weights
        self._initialize_strategies()
        self._calculate_weights()

        logger.info(f"PortfolioManager initialized with {len(strategies)} strategies")
        logger.info(f"Weight method: {weight_method.value}")

    def _initialize_strategies(self):
        """Initialize all strategies"""
        for config in self.strategy_configs:
            if not config.enabled:
                continue
            try:
                strategy = config.strategy_class(**config.params)
                self.strategies[config.name] = strategy
                self.state.strategy_returns[config.name] = []
                self.state.strategy_positions[config.name] = {}
                logger.debug(f"Initialized strategy: {config.name}")
            except Exception as e:
                logger.error(f"Failed to initialize strategy {config.name}: {e}")

    def _calculate_weights(self):
        """Calculate strategy weights based on method"""
        n_strategies = len(self.strategies)
        if n_strategies == 0:
            return

        if self.weight_method == WeightMethod.EQUAL:
            weight = 1.0 / n_strategies
            self.state.weights = {name: weight for name in self.strategies}

        elif self.weight_method == WeightMethod.VOLATILITY_INVERSE:
            self._calculate_vol_inverse_weights()

        elif self.weight_method == WeightMethod.SHARPE_WEIGHTED:
            self._calculate_sharpe_weights()

        elif self.weight_method == WeightMethod.CUSTOM:
            # Use initial weights from config
            for config in self.strategy_configs:
                if config.name in self.strategies:
                    self.state.weights[config.name] = config.initial_weight

        # Apply min/max constraints
        self._apply_weight_constraints()

        logger.info(f"Portfolio weights: {self.state.weights}")

    def _calculate_vol_inverse_weights(self):
        """Calculate weights inversely proportional to volatility"""
        volatilities = {}

        for name, returns in self.state.strategy_returns.items():
            if len(returns) >= 20:  # Need minimum data
                vol = np.std(returns) * np.sqrt(252)  # Annualized
                volatilities[name] = vol if vol > 0 else 0.001
            else:
                volatilities[name] = 0.15  # Default 15% vol

        # Inverse volatility
        inv_vols = {name: 1 / vol for name, vol in volatilities.items()}
        total = sum(inv_vols.values())

        self.state.weights = {name: iv / total for name, iv in inv_vols.items()}

    def _calculate_sharpe_weights(self):
        """Calculate weights proportional to Sharpe ratio"""
        sharpes = {}
        risk_free = 0.03 / 252  # Daily risk-free rate

        for name, returns in self.state.strategy_returns.items():
            if len(returns) >= 20:
                mean_ret = np.mean(returns)
                std_ret = np.std(returns)
                if std_ret > 0:
                    sharpe = (mean_ret - risk_free) / std_ret * np.sqrt(252)
                else:
                    sharpe = 0
                sharpes[name] = max(sharpe, 0.01)  # Floor at 0.01
            else:
                sharpes[name] = 1.0  # Default

        total = sum(sharpes.values())
        self.state.weights = {name: s / total for name, s in sharpes.items()}

    def _apply_weight_constraints(self):
        """Apply min/max weight constraints"""
        for name in self.state.weights:
            w = self.state.weights[name]
            w = max(self.min_weight, min(self.max_weight, w))
            self.state.weights[name] = w

        # Renormalize
        total = sum(self.state.weights.values())
        if total > 0:
            self.state.weights = {n: w / total for n, w in self.state.weights.items()}

    def collect_signals(self, bar: Dict) -> Dict[str, Optional[Dict]]:
        """Collect signals from all strategies"""
        signals = {}

        for name, strategy in self.strategies.items():
            try:
                if hasattr(strategy, "on_bar"):
                    signal = strategy.on_bar(bar)
                    signals[name] = signal
            except Exception as e:
                logger.error(f"Strategy {name} signal error: {e}")
                signals[name] = None

        return signals

    def combine_signals(self, signals: Dict[str, Optional[Dict]]) -> List[PortfolioSignal]:
        """Combine signals from multiple strategies"""
        combined = {}  # symbol -> aggregated signal info

        for strategy_name, signal in signals.items():
            if signal is None:
                continue

            symbol = signal.get("symbol")
            if not symbol:
                continue

            direction = signal.get("direction", "hold")
            strength = signal.get("strength", 1.0)
            weight = self.state.weights.get(strategy_name, 0)

            if symbol not in combined:
                combined[symbol] = {"buy_score": 0, "sell_score": 0, "sources": []}

            if direction == "buy":
                combined[symbol]["buy_score"] += strength * weight
            elif direction == "sell":
                combined[symbol]["sell_score"] += strength * weight

            combined[symbol]["sources"].append(strategy_name)

        # Convert to PortfolioSignals
        portfolio_signals = []
        for symbol, info in combined.items():
            buy_score = info["buy_score"]
            sell_score = info["sell_score"]

            if buy_score > sell_score and buy_score > 0.3:
                direction = "buy"
                strength = buy_score
            elif sell_score > buy_score and sell_score > 0.3:
                direction = "sell"
                strength = sell_score
            else:
                direction = "hold"
                strength = 0

            portfolio_signals.append(
                PortfolioSignal(
                    symbol=symbol,
                    direction=direction,
                    strength=strength,
                    source_strategies=info["sources"],
                    timestamp="",
                )
            )

        return portfolio_signals

    def should_rebalance(self, current_date: str) -> bool:
        """Check if rebalancing is needed"""
        if self.state.last_rebalance is None:
            return True

        # Simple frequency check (can be enhanced)
        last = pd.Timestamp(self.state.last_rebalance)
        current = pd.Timestamp(current_date)

        if self.rebalance_frequency == "daily":
            return (current - last).days >= 1
        elif self.rebalance_frequency == "weekly":
            return (current - last).days >= 7
        elif self.rebalance_frequency == "monthly":
            return (current - last).days >= 30

        return False

    def rebalance(self, current_date: str):
        """Rebalance portfolio weights"""
        logger.info(f"Rebalancing portfolio on {current_date}")
        self._calculate_weights()
        self.state.last_rebalance = current_date

    def update_strategy_return(self, strategy_name: str, daily_return: float):
        """Update strategy return for weight calculation"""
        if strategy_name in self.state.strategy_returns:
            self.state.strategy_returns[strategy_name].append(daily_return)
            # Keep last 252 days
            if len(self.state.strategy_returns[strategy_name]) > 252:
                self.state.strategy_returns[strategy_name].pop(0)

    def get_portfolio_stats(self) -> Dict:
        """Get portfolio statistics"""
        stats = {
            "n_strategies": len(self.strategies),
            "weights": self.state.weights,
            "weight_method": self.weight_method.value,
            "rebalance_frequency": self.rebalance_frequency,
            "last_rebalance": self.state.last_rebalance,
        }

        # Calculate strategy correlations if enough data
        if all(len(r) >= 20 for r in self.state.strategy_returns.values()):
            returns_df = pd.DataFrame(self.state.strategy_returns)
            stats["correlation_matrix"] = returns_df.corr().to_dict()

        return stats

    def get_weights(self) -> Dict[str, float]:
        """Get current strategy weights"""
        return self.state.weights.copy()

    def set_weights(self, weights: Dict[str, float]):
        """Manually set strategy weights"""
        for name, weight in weights.items():
            if name in self.strategies:
                self.state.weights[name] = weight

        self._apply_weight_constraints()
        logger.info(f"Weights manually set: {self.state.weights}")
