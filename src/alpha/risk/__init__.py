"""Risk management, signal transformation, and factor combination."""

from .models import (
    BacktestResult, CostModel, EvalMethod, ExecutionSimulator, MarketContext,
    PortfolioManager, RiskConfig, RuleOverlay, SignalTransformer,
)
from .combination import FactorCombiner, FactorSignal
