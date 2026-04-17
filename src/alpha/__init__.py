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
    AlphaDataset,
    AShareDailyDatasetLoader,
    ASTNode,
    BytecodeProgram,
    CryptoMinuteDatasetLoader,
    DatasetLoader,
    FormulaCompiler,
    FormulaParser,
    Instruction,
    MarketProfile,
    MarketType,
    OperatorRegistry,
    OperatorSpec,
    StackVM,
    TensorSchema,
    TensorStore,
    TypeChecker,
    ValidationReport,
    get_market_profile,
    list_market_types,
)

# --- Eval ---
from .eval import (
    TRITON_AVAILABLE as triton_available,
)
from .eval import (
    CPCVValidator,
    ValidationFold,
    compute_forward_returns,
    compute_ic_metrics,
    compute_ic_metrics_gpu,
    compute_rank_ic,
    compute_rank_ic_batch_gpu,
    compute_rank_ic_gpu,
    fast_screen_ic,
)

# --- Infra ---
from .infra import (
    AlphaPersistence,
    InMemoryCollector,
    LangfuseCollector,
    PersistedRun,
    Span,
    SpanCollector,
    tracer,
)

# --- Knowledge ---
from .knowledge import (
    DerivedFeature,
    FeatureGroup,
    FeatureKitchen,
    FinancialKnowledgeBase,
    FinancialTheme,
    StrategyMemory,
)

# --- LLM ---
from .llm import HeuristicLLMBackend, OpenAILLMBackend

# --- Risk ---
from .risk import (
    BacktestResult,
    CostModel,
    ExecutionSimulator,
    FactorCombiner,
    FactorSignal,
    MarketContext,
    PortfolioManager,
    RiskConfig,
    RuleOverlay,
    SignalTransformer,
)

# --- Search ---
from .search import (
    ArchiveEntry,
    BreedingSpec,
    CheckpointManager,
    EvalResult,
    FactorCatalog,
    FactorCatalogEntry,
    FitnessEngine,
    FitnessPolicy,
    FormulaEnumerator,
    Individual,
    Lineage,
    PipelineRecord,
    RoundRecord,
    SearchCheckpoint,
    SearchContext,
    SearchOrchestrator,
    SearchResult,
    SearchStrategy,
    StageKind,
    StageRecord,
    StatefulStrategy,
    StrategySnapshot,
    build_individual,
)

# --- Service ---
from .service import AlphaService

# --- Strategies ---
from .strategies import (
    EnumerationStrategy,
    LLMEvolutionStrategy,
    MCTSRefinementStrategy,
    NeuralFormulaStrategy,
)
from .strategies.mcts import AlphaNode, MCTSEngine, MCTSLLMAdapter

__all__ = [
    # Core
    "ASTNode", "AlphaDataset", "AShareDailyDatasetLoader", "BytecodeProgram",
    "CryptoMinuteDatasetLoader", "DatasetLoader", "FormulaCompiler",
    "FormulaParser", "Instruction", "MarketProfile", "MarketType",
    "OperatorRegistry", "OperatorSpec", "StackVM", "TensorSchema",
    "TensorStore", "TypeChecker", "ValidationReport",
    "get_market_profile", "list_market_types",
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
