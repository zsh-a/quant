"""Unified alpha factor discovery module.

Pluggable strategy framework for alpha formula mining with:
- LLM-driven evolution + RL feedback loop
- MCTS local refinement
- Financial knowledge base + derived feature catalog
- Multi-factor combination and portfolio-level risk management
"""

from .combination import FactorCombiner, FactorSignal
from .compiler import BytecodeProgram, FormulaCompiler, Instruction
from .dataset import AlphaDataset, CryptoMinuteDatasetLoader
from .enumerator import FormulaEnumerator
from .fast_screen import fast_screen_ic
from .dsl import ASTNode, FormulaParser, TensorSchema, TypeChecker, ValidationReport
from .evaluation import compute_forward_returns, compute_ic_metrics, compute_rank_ic
from .evolution import (
    BreedingSpec,
    EvalResult,
    FitnessEngine,
    FitnessPolicy,
    HeuristicLLMBackend,
    Individual,
    SearchResult,
)
from .feature_kitchen import DerivedFeature, FeatureKitchen
from .financial_knowledge import FeatureGroup, FinancialKnowledgeBase, FinancialTheme
from .mcts import AlphaNode, MCTSEngine, MCTSLLMAdapter
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
from .search_strategy import SearchContext, SearchOrchestrator, SearchStrategy
from .service import AlphaService
from .strategies import LLMEvolutionStrategy, MCTSRefinementStrategy
from .strategy_memory import StrategyMemory
from .tracing import InMemoryCollector, LangfuseCollector, Span, SpanCollector, tracer
from .validation import CPCVValidator, ValidationFold
from .gpu_evaluation import compute_ic_metrics_gpu, compute_rank_ic_batch_gpu, compute_rank_ic_gpu
from .gpu_ops import TRITON_AVAILABLE as triton_available
from .vm import StackVM, TensorStore

__all__ = [
    "triton_available",
    "compute_rank_ic_gpu",
    "compute_rank_ic_batch_gpu",
    "compute_ic_metrics_gpu",
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
    "DerivedFeature",
    "EvalResult",
    "ExecutionSimulator",
    "FactorCombiner",
    "FactorSignal",
    "FeatureGroup",
    "FeatureKitchen",
    "FinancialKnowledgeBase",
    "FinancialTheme",
    "FitnessEngine",
    "FitnessPolicy",
    "FormulaCompiler",
    "FormulaParser",
    "HeuristicLLMBackend",
    "Individual",
    "InMemoryCollector",
    "Instruction",
    "LangfuseCollector",
    "LLMEvolutionStrategy",
    "MarketContext",
    "MCTSEngine",
    "MCTSLLMAdapter",
    "MCTSRefinementStrategy",
    "OperatorRegistry",
    "OperatorSpec",
    "PersistedRun",
    "PortfolioManager",
    "RiskConfig",
    "RuleOverlay",
    "SearchContext",
    "SearchOrchestrator",
    "SearchResult",
    "SearchStrategy",
    "FormulaEnumerator",
    "fast_screen_ic",
    "SignalTransformer",
    "Span",
    "SpanCollector",
    "StackVM",
    "StrategyMemory",
    "TensorSchema",
    "TensorStore",
    "TypeChecker",
    "tracer",
    "ValidationFold",
    "ValidationReport",
]
