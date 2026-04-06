"""
Pluggable search strategies for alpha factor discovery.

All strategies implement the SearchStrategy protocol (4 methods):
    name, should_activate, generate_candidates, on_evaluation_complete

Base strategies (always active):
    EnumerationStrategy    — Programmatic bulk formula generation + IC screening
    LLMEvolutionStrategy   — LLM-driven regularized evolution (main strategy)

Extra strategies (opt-in via strategy="mcts", "neural", etc.):
    MCTSRefinementStrategy — LLM-guided MCTS refinement of archive elites
    NeuralFormulaStrategy  — Transformer + REINFORCE on RPN token sequences

Strategy registry:
    @register_strategy("name") decorator for adding new strategies.
    build_extra_strategies({"mcts", "neural"}, **infra) to construct them.
"""

from .registry import register_strategy, build_extra_strategies, available_strategies

# Import strategy modules to trigger @register_strategy decorators
from .enumeration import EnumerationStrategy
from .llm_evolution import LLMEvolutionStrategy
from .mcts_refinement import MCTSRefinementStrategy
from .neural_formula import NeuralFormulaStrategy

__all__ = [
    "EnumerationStrategy",
    "LLMEvolutionStrategy",
    "MCTSRefinementStrategy",
    "NeuralFormulaStrategy",
    "register_strategy",
    "build_extra_strategies",
    "available_strategies",
]
