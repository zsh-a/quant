"""
Alpha lab package.

This package contains the first production-oriented implementation skeleton
for the LLM + ES + Stack VM alpha mining workflow.
"""

from .compiler import BytecodeProgram, FormulaCompiler, Instruction
from .dataset import AlphaDataset, CryptoMinuteDatasetLoader
from .dsl import DSLRegistry, FormulaParser, TensorSchema, TypeChecker
from .evolution import EvolutionEngine, FitnessEngine, HeuristicLLMBackend, Individual
from .persistence import AlphaLabPersistence, PersistedRun
from .risk import BacktestResult, CostModel, ExecutionSimulator, MarketContext, RuleOverlay, SignalTransformer
from .service import AlphaLabService
from .validation import CPCVValidator, ValidationFold
from .vm import StackVM, TensorStore

__all__ = [
    "AlphaLabService",
    "BacktestResult",
    "BytecodeProgram",
    "CPCVValidator",
    "CostModel",
    "CryptoMinuteDatasetLoader",
    "DSLRegistry",
    "EvolutionEngine",
    "ExecutionSimulator",
    "FitnessEngine",
    "FormulaCompiler",
    "FormulaParser",
    "HeuristicLLMBackend",
    "Individual",
    "Instruction",
    "MarketContext",
    "RuleOverlay",
    "SignalTransformer",
    "StackVM",
    "AlphaDataset",
    "AlphaLabPersistence",
    "TensorSchema",
    "TensorStore",
    "TypeChecker",
    "ValidationFold",
    "PersistedRun",
]
