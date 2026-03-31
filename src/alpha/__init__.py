"""Unified alpha factor discovery module.

Merges alpha_lab (crypto, GA+LLM, compiled VM) and alpha_mining (stock, MCTS+LLM)
into a single extensible framework.
"""

from .compiler import BytecodeProgram, FormulaCompiler, Instruction
from .dataset import AlphaDataset, CryptoMinuteDatasetLoader, StockDailyDatasetLoader
from .dsl import ASTNode, FormulaParser, TensorSchema, TypeChecker, ValidationReport
from .evaluation import compute_forward_returns, compute_ic_metrics, compute_rank_ic
from .evolution import (
    BreedingSpec,
    EvolutionEngine,
    FitnessEngine,
    FitnessPolicy,
    HeuristicLLMBackend,
    Individual,
)
from .mcts import AlphaNode, MCTSEngine
from .operators import OperatorRegistry, OperatorSpec
from .persistence import AlphaPersistence, PersistedRun
from .risk import (
    BacktestResult,
    CostModel,
    ExecutionSimulator,
    MarketContext,
    RuleOverlay,
    SignalTransformer,
)
from .service import AlphaService
from .validation import CPCVValidator, ValidationFold
from .vm import StackVM, TensorStore

# Backward-compatible aliases
DSLRegistry = OperatorRegistry
AlphaLabService = AlphaService
AlphaLabPersistence = AlphaPersistence

__all__ = [
    "AlphaService",
    "AlphaLabService",
    "AlphaPersistence",
    "AlphaLabPersistence",
    "AlphaDataset",
    "AlphaNode",
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
    "DSLRegistry",
    "EvolutionEngine",
    "ExecutionSimulator",
    "FitnessEngine",
    "FitnessPolicy",
    "FormulaCompiler",
    "FormulaParser",
    "HeuristicLLMBackend",
    "Individual",
    "Instruction",
    "MarketContext",
    "MCTSEngine",
    "OperatorRegistry",
    "OperatorSpec",
    "PersistedRun",
    "RuleOverlay",
    "SignalTransformer",
    "StackVM",
    "StockDailyDatasetLoader",
    "TensorSchema",
    "TensorStore",
    "TypeChecker",
    "ValidationFold",
    "ValidationReport",
]
