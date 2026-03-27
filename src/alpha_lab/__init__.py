"""
Alpha lab package.

This package contains the first production-oriented implementation skeleton
for the LLM + ES + Stack VM alpha mining workflow.
"""

from .compiler import BytecodeProgram, FormulaCompiler, Instruction
from .dsl import DSLRegistry, FormulaParser, TensorSchema, TypeChecker
from .evolution import EvolutionEngine, FitnessEngine, HeuristicLLMBackend, Individual
from .risk import BacktestResult, CostModel, ExecutionSimulator, MarketContext, RuleOverlay, SignalTransformer
from .service import AlphaLabService
from .vm import StackVM, TensorStore

__all__ = [
    "AlphaLabService",
    "BacktestResult",
    "BytecodeProgram",
    "CostModel",
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
    "TensorSchema",
    "TensorStore",
    "TypeChecker",
]
