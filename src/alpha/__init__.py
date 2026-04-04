"""Unified alpha factor discovery module.

Regularised Evolution + Quality-Diversity archive for alpha formula mining,
multi-factor combination, and portfolio-level risk management.
"""

from .combination import FactorCombiner, FactorSignal
from .compiler import BytecodeProgram, FormulaCompiler, Instruction
from .dataset import AlphaDataset, CryptoMinuteDatasetLoader, StockDailyDatasetLoader
from .dsl import ASTNode, FormulaParser, TensorSchema, TypeChecker, ValidationReport
from .evaluation import compute_forward_returns, compute_ic_metrics, compute_rank_ic
from .evolution import (
    BreedingSpec,
    EvalResult,
    FitnessEngine,
    FitnessPolicy,
    HeuristicLLMBackend,
    Individual,
    SearchEngine,
    SearchResult,
)
from .mcts import AlphaNode, MCTSEngine
from .operators import OperatorRegistry, OperatorSpec
from .persistence import AlphaPersistence, PersistedRun
from .risk import (
    BacktestResult,
    CostModel,
    ExecutionSimulator,
    MarketContext,
    PortfolioManager,
    RiskConfig,
    RuleOverlay,
    SignalTransformer,
)
from .service import AlphaService
from .tracing import InMemoryCollector, LangfuseCollector, Span, SpanCollector, tracer
from .validation import CPCVValidator, ValidationFold
from .vm import StackVM, TensorStore

__all__ = [
    "AlphaDataset",
    "AlphaNode",
    "AlphaPersistence",
    "AlphaService",
    "ASTNode",
    "BacktestResult",
    "BreedingSpec",
    "BytecodeProgram",
    "compute_forward_returns",
    "compute_ic_metrics",
    "compute_rank_ic",
    "CostModel",
    "CPCVValidator",
    "CryptoMinuteDatasetLoader",
    "EvalResult",
    "ExecutionSimulator",
    "FactorCombiner",
    "FactorSignal",
    "FitnessEngine",
    "FitnessPolicy",
    "FormulaCompiler",
    "FormulaParser",
    "HeuristicLLMBackend",
    "Individual",
    "InMemoryCollector",
    "Instruction",
    "LangfuseCollector",
    "MarketContext",
    "MCTSEngine",
    "OperatorRegistry",
    "OperatorSpec",
    "PersistedRun",
    "PortfolioManager",
    "RiskConfig",
    "RuleOverlay",
    "SearchEngine",
    "SearchResult",
    "SignalTransformer",
    "Span",
    "SpanCollector",
    "StackVM",
    "StockDailyDatasetLoader",
    "TensorSchema",
    "TensorStore",
    "TypeChecker",
    "tracer",
    "ValidationFold",
    "ValidationReport",
]
