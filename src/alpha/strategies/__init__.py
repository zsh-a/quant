"""
Pluggable search strategies for alpha factor discovery.

All strategies implement the ``SearchStrategy`` protocol (4 methods):
    name, should_activate, generate_candidates, on_evaluation_complete

Search modes define which strategies run together — see ``SearchMode``
in ``registry.py``.  Each mode explicitly lists its strategies; there
are no implicit "always-on" additions.

Strategy registry:
    @register_strategy(StrategyMeta(...)) decorator for adding new strategies.
    register_mode(SearchMode(...)) for adding new modes.
    build_strategies({"mcts", "neural"}, StrategyInfra(...)) to construct them.
"""

from .alpha_forge import AlphaForgeStrategy
from .alpha_probe import AlphaPROBEStrategy
from .base import BaseStrategy, StrategyMeta

# Import strategy modules to trigger @register_strategy decorators
from .enumeration import EnumerationStrategy
from .llm_evolution import LLMEvolutionStrategy
from .mcts_refinement import MCTSRefinementStrategy
from .neural_formula import NeuralFormulaStrategy
from .registry import (
    SearchMode,
    StrategyInfra,
    available_strategies,
    build_strategies,
    get_all_meta,
    get_all_modes,
    get_mode,
    get_strategy_meta,
    register_mode,
    register_strategy,
)

__all__ = [
    # Base class and metadata
    "BaseStrategy",
    "StrategyMeta",
    "StrategyInfra",
    "SearchMode",
    # Registry API
    "register_strategy",
    "register_mode",
    "build_strategies",
    "available_strategies",
    "get_all_meta",
    "get_all_modes",
    "get_mode",
    "get_strategy_meta",
    # Strategy implementations
    "EnumerationStrategy",
    "LLMEvolutionStrategy",
    "MCTSRefinementStrategy",
    "NeuralFormulaStrategy",
    "AlphaForgeStrategy",
    "AlphaPROBEStrategy",
]
