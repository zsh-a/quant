"""Unified alpha factor discovery module.

Pluggable strategy framework for alpha formula mining with:
- LLM-driven evolution + RL feedback loop
- LLM-guided MCTS refinement
- Neural formula generation (Transformer + REINFORCE)
- Programmatic enumeration with fast IC screening
- Financial knowledge base + derived feature catalog
- Multi-factor combination and portfolio-level risk management
"""

# --- Core DSL / Compiler / VM ---
from .compiler import BytecodeProgram, FormulaCompiler, Instruction
from .dsl import ASTNode, FormulaParser, TensorSchema, TypeChecker, ValidationReport
from .operators import OperatorRegistry, OperatorSpec
from .vm import StackVM, TensorStore

# --- Data ---
from .dataset import AlphaDataset, CryptoMinuteDatasetLoader

# --- Evaluation ---
from .evaluation import compute_forward_returns, compute_ic_metrics, compute_rank_ic
from .gpu_evaluation import compute_ic_metrics_gpu, compute_rank_ic_batch_gpu, compute_rank_ic_gpu
from .gpu_ops import TRITON_AVAILABLE as triton_available
from .fast_screen import fast_screen_ic

# --- Evolution primitives ---
from .evolution import BreedingSpec, EvalResult, FitnessEngine, FitnessPolicy, Individual, SearchResult
from .pipeline import ArchiveEntry, Lineage, PipelineRecord, RoundRecord, StageKind, StageRecord

# --- Strategy framework ---
from .strategy_state import (
    FactorCatalog, FactorCatalogEntry, SearchContext, SearchStrategy,
    StatefulStrategy, StrategySnapshot, build_individual,
)
from .search_strategy import SearchOrchestrator

# --- LLM backends ---
from .llm import HeuristicLLMBackend

# --- Search strategies ---
from .strategies import EnumerationStrategy, LLMEvolutionStrategy, MCTSRefinementStrategy, NeuralFormulaStrategy
from .mcts import AlphaNode, MCTSEngine, MCTSLLMAdapter

# --- Knowledge & memory ---
from .feature_kitchen import DerivedFeature, FeatureKitchen
from .financial_knowledge import FeatureGroup, FinancialKnowledgeBase, FinancialTheme
from .strategy_memory import StrategyMemory
from .enumerator import FormulaEnumerator

# --- Infrastructure ---
from .checkpoint import CheckpointManager, SearchCheckpoint
from .persistence import AlphaPersistence, PersistedRun
from .combination import FactorCombiner, FactorSignal
from .validation import CPCVValidator, ValidationFold
from .tracing import InMemoryCollector, LangfuseCollector, Span, SpanCollector, tracer

# --- Risk ---
from .risk import (
    BacktestResult, CostModel, ExecutionSimulator, MarketContext,
    PortfolioManager, RiskConfig, RuleOverlay, SignalTransformer,
)

# --- Service (top-level API) ---
from .service import AlphaService

__all__ = [
    # Core
    "ASTNode", "BytecodeProgram", "FormulaCompiler", "FormulaParser",
    "Instruction", "OperatorRegistry", "OperatorSpec", "StackVM",
    "TensorSchema", "TensorStore", "TypeChecker", "ValidationReport",
    # Data
    "AlphaDataset", "CryptoMinuteDatasetLoader",
    # Evaluation
    "compute_forward_returns", "compute_ic_metrics", "compute_rank_ic",
    "compute_ic_metrics_gpu", "compute_rank_ic_batch_gpu", "compute_rank_ic_gpu",
    "triton_available", "fast_screen_ic",
    # Evolution
    "BreedingSpec", "EvalResult", "FitnessEngine", "FitnessPolicy",
    "Individual", "SearchResult",
    # Pipeline
    "ArchiveEntry", "Lineage", "PipelineRecord", "RoundRecord",
    "StageKind", "StageRecord",
    # Strategy framework
    "build_individual", "FactorCatalog", "FactorCatalogEntry",
    "SearchContext", "SearchOrchestrator", "SearchStrategy",
    "StatefulStrategy", "StrategySnapshot",
    # LLM
    "HeuristicLLMBackend",
    # Strategies
    "EnumerationStrategy", "LLMEvolutionStrategy",
    "MCTSRefinementStrategy", "NeuralFormulaStrategy",
    "AlphaNode", "MCTSEngine", "MCTSLLMAdapter",
    # Knowledge
    "DerivedFeature", "FeatureGroup", "FeatureKitchen",
    "FinancialKnowledgeBase", "FinancialTheme",
    "FormulaEnumerator", "StrategyMemory",
    # Infrastructure
    "AlphaPersistence", "CheckpointManager", "CPCVValidator",
    "FactorCombiner", "FactorSignal", "InMemoryCollector",
    "LangfuseCollector", "PersistedRun", "SearchCheckpoint",
    "Span", "SpanCollector", "tracer", "ValidationFold",
    # Risk
    "BacktestResult", "CostModel", "ExecutionSimulator",
    "MarketContext", "PortfolioManager", "RiskConfig",
    "RuleOverlay", "SignalTransformer",
    # Service
    "AlphaService",
]
