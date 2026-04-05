"""
Pluggable search strategies for alpha factor discovery.

Current:
    LLMEvolutionStrategy   — LLM-driven regularized evolution (main strategy)
    MCTSRefinementStrategy — MCTS local refinement of archive elites
    NeuralFormulaStrategy  — AlphaGPT-style Transformer + REINFORCE generation

Future (implement SearchStrategy protocol to add):
    DAGEvolutionStrategy       — AlphaPROBE: DAG topology + Bayesian retrieval
    GrammarGuidedStrategy      — AlphaCFG: CFG constraint + Tree-LSTM + RL-MCTS
    SynergyRLStrategy          — PPO for combination-IC optimization
    NeuralGenerativeStrategy   — AlphaForge: generative-predictive networks
    DistributionalRLStrategy   — AlphaQCM: QCM variance-guided exploration
"""

from .llm_evolution import LLMEvolutionStrategy
from .mcts_refinement import MCTSRefinementStrategy
from .neural_formula import NeuralFormulaStrategy

__all__ = [
    "LLMEvolutionStrategy",
    "MCTSRefinementStrategy",
    "NeuralFormulaStrategy",
]
