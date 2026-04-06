"""
Pluggable search strategies for alpha factor discovery.

All strategies implement the SearchStrategy protocol (4 methods):
    name, should_activate, generate_candidates, on_evaluation_complete

Current:
    EnumerationStrategy    — Programmatic bulk formula generation + IC screening
    LLMEvolutionStrategy   — LLM-driven regularized evolution (main strategy)
    MCTSRefinementStrategy — LLM-guided MCTS refinement of archive elites
    NeuralFormulaStrategy  — Transformer + REINFORCE on RPN token sequences
"""

from .enumeration import EnumerationStrategy
from .llm_evolution import LLMEvolutionStrategy
from .mcts_refinement import MCTSRefinementStrategy
from .neural_formula import NeuralFormulaStrategy

__all__ = [
    "EnumerationStrategy",
    "LLMEvolutionStrategy",
    "MCTSRefinementStrategy",
    "NeuralFormulaStrategy",
]
