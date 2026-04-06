"""
Pluggable search strategies for alpha factor discovery.

All strategies inherit from ``BaseStrategy`` and implement the
``SearchStrategy`` protocol (4 methods):
    name, should_activate, generate_candidates, on_evaluation_complete

Base strategies (always active, ``always_on=True``):
    EnumerationStrategy    — Programmatic bulk formula generation + IC screening
    LLMEvolutionStrategy   — LLM-driven regularized evolution (main strategy)

Extra strategies (opt-in via strategy="mcts", "neural", etc.):
    MCTSRefinementStrategy — LLM-guided MCTS refinement of archive elites
    NeuralFormulaStrategy  — Transformer + REINFORCE on RPN token sequences

Strategy registry:
    @register_strategy(StrategyMeta(...)) decorator for adding new strategies.
    build_strategies({"mcts", "neural"}, StrategyInfra(...)) to construct them.
"""

from .base import BaseStrategy, StrategyMeta
from .registry import (
    register_strategy,
    build_strategies,
    build_extra_strategies,
    available_strategies,
    get_all_meta,
    get_strategy_meta,
    StrategyInfra,
)

# Import strategy modules to trigger @register_strategy decorators
from .enumeration import EnumerationStrategy
from .llm_evolution import LLMEvolutionStrategy
from .mcts_refinement import MCTSRefinementStrategy
from .neural_formula import NeuralFormulaStrategy

__all__ = [
    # Base class and metadata
    "BaseStrategy",
    "StrategyMeta",
    "StrategyInfra",
    # Registry API
    "register_strategy",
    "build_strategies",
    "build_extra_strategies",
    "available_strategies",
    "get_all_meta",
    "get_strategy_meta",
    # Strategy implementations
    "EnumerationStrategy",
    "LLMEvolutionStrategy",
    "MCTSRefinementStrategy",
    "NeuralFormulaStrategy",
]
