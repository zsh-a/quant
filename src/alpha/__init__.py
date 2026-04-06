"""Unified alpha factor discovery module.

Subpackages:
  core/       — DSL, compiler, VM, operators, dataset
  eval/       — Metrics, screening, validation, GPU acceleration
  search/     — Orchestrator, context, pipeline, evolution, checkpoints
  strategies/ — Pluggable strategy implementations (LLM, MCTS, Neural, Enum)
  llm/        — LLM backends (Heuristic, OpenAI) and context building
  knowledge/  — Financial themes, feature engineering, strategy memory
  risk/       — Risk models, signal transformation, factor combination
  infra/      — Persistence and tracing
"""

# --- Core ---
from .core import (
    ASTNode, AlphaDataset, BytecodeProgram, CryptoMinuteDatasetLoader,
    FormulaCompiler, FormulaParser, Instruction, OperatorRegistry, OperatorSpec,
    StackVM, TensorSchema, TensorStore, TypeChecker, ValidationReport,
)

# --- Eval ---
from .eval import (
    TRITON_AVAILABLE as triton_available,
    CPCVValidator, ValidationFold,
    compute_forward_returns, compute_ic_metrics, compute_rank_ic,
    compute_ic_metrics_gpu, compute_rank_ic_batch_gpu, compute_rank_ic_gpu,
    fast_screen_ic,
)

# --- Search ---
from .search import (
    ArchiveEntry, BreedingSpec, CheckpointManager, EvalResult,
    FactorCatalog, FactorCatalogEntry, FitnessEngine, FitnessPolicy,
    FormulaEnumerator, Individual, Lineage, PipelineRecord, RoundRecord,
    SearchCheckpoint, SearchContext, SearchOrchestrator, SearchResult,
    SearchStrategy, StageKind, StageRecord, StatefulStrategy, StrategySnapshot,
    build_individual,
)

# --- LLM ---
from .llm import HeuristicLLMBackend, OpenAILLMBackend

# --- Strategies ---
from .strategies import (
    EnumerationStrategy, LLMEvolutionStrategy,
    MCTSRefinementStrategy, NeuralFormulaStrategy,
)
from .strategies.mcts import AlphaNode, MCTSEngine, MCTSLLMAdapter

# --- Knowledge ---
from .knowledge import (
    DerivedFeature, FeatureGroup, FeatureKitchen,
    FinancialKnowledgeBase, FinancialTheme, StrategyMemory,
)

# --- Risk ---
from .risk import (
    BacktestResult, CostModel, ExecutionSimulator, FactorCombiner, FactorSignal,
    MarketContext, PortfolioManager, RiskConfig, RuleOverlay, SignalTransformer,
)

# --- Infra ---
from .infra import (
    AlphaPersistence, InMemoryCollector, LangfuseCollector,
    PersistedRun, Span, SpanCollector, tracer,
)

# --- Service ---
from .service import AlphaService

__all__ = [
    # Core
    "ASTNode", "AlphaDataset", "BytecodeProgram", "CryptoMinuteDatasetLoader",
    "FormulaCompiler", "FormulaParser", "Instruction", "OperatorRegistry",
    "OperatorSpec", "StackVM", "TensorSchema", "TensorStore",
    "TypeChecker", "ValidationReport",
    # Eval
    "triton_available", "CPCVValidator", "ValidationFold",
    "compute_forward_returns", "compute_ic_metrics", "compute_rank_ic",
    "compute_ic_metrics_gpu", "compute_rank_ic_batch_gpu", "compute_rank_ic_gpu",
    "fast_screen_ic",
    # Search
    "ArchiveEntry", "BreedingSpec", "CheckpointManager", "EvalResult",
    "FactorCatalog", "FactorCatalogEntry", "FitnessEngine", "FitnessPolicy",
    "FormulaEnumerator", "Individual", "Lineage", "PipelineRecord",
    "RoundRecord", "SearchCheckpoint", "SearchContext", "SearchOrchestrator",
    "SearchResult", "SearchStrategy", "StageKind", "StageRecord",
    "StatefulStrategy", "StrategySnapshot", "build_individual",
    # LLM
    "HeuristicLLMBackend", "OpenAILLMBackend",
    # Strategies
    "AlphaNode", "EnumerationStrategy", "LLMEvolutionStrategy",
    "MCTSEngine", "MCTSLLMAdapter", "MCTSRefinementStrategy",
    "NeuralFormulaStrategy",
    # Knowledge
    "DerivedFeature", "FeatureGroup", "FeatureKitchen",
    "FinancialKnowledgeBase", "FinancialTheme", "StrategyMemory",
    # Risk
    "BacktestResult", "CostModel", "ExecutionSimulator", "FactorCombiner",
    "FactorSignal", "MarketContext", "PortfolioManager", "RiskConfig",
    "RuleOverlay", "SignalTransformer",
    # Infra
    "AlphaPersistence", "InMemoryCollector", "LangfuseCollector",
    "PersistedRun", "Span", "SpanCollector", "tracer",
    # Service
    "AlphaService",
]
