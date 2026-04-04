"""
MCTS local refinement strategy.

Wraps the existing MCTSEngine as a pluggable SearchStrategy.
Periodically refines the top archive members via tree search.

Corresponds to: LLM-MCTS paper (mcts.pdf)
Future extensions:
  - Frequent Subtree Avoidance (FSA)
  - Grammar constraints (AlphaCFG)
  - Multi-dimensional reward aggregation
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ..search_strategy import SearchContext, build_individual
from ..evolution import Individual


class MCTSRefinementStrategy:
    """MCTS local refinement of archive elites.

    Activates every ``activation_frequency`` rounds.  Takes the top-k
    individuals from the shared archive, runs MCTSEngine on each to
    explore nearby formula space, and returns discovered candidates
    for the orchestrator to evaluate.
    """

    def __init__(
        self,
        mcts_engine: Any,  # MCTSEngine (avoid circular import)
        activation_frequency: int = 3,
        top_k_to_refine: int = 2,
        iterations_per_refine: int = 3,
    ) -> None:
        self.mcts_engine = mcts_engine
        self.activation_frequency = activation_frequency
        self.top_k_to_refine = top_k_to_refine
        self.iterations_per_refine = iterations_per_refine

    @property
    def name(self) -> str:
        return "mcts_refinement"

    def should_activate(self, ctx: SearchContext) -> bool:
        return (
            ctx.round_idx > 0
            and ctx.round_idx % self.activation_frequency == 0
            and len(ctx.archive) > 0
            and ctx.dataset is not None
        )

    def generate_candidates(self, ctx: SearchContext) -> list[Individual]:
        top_members = sorted(
            ctx.archive.values(), key=lambda x: x.fitness, reverse=True
        )
        top_members = top_members[: self.top_k_to_refine]

        candidates: list[Individual] = []
        for member in top_members:
            try:
                # Reset zoo for fresh refinement
                self.mcts_engine.alpha_zoo = []
                self.mcts_engine.run(
                    initial_formula=member.formula,
                    dataset=ctx.dataset,
                    iterations=self.iterations_per_refine,
                )
                for formula in self.mcts_engine.get_refined_formulas():
                    ind = build_individual(
                        ctx.compiler,
                        ctx.schema,
                        formula,
                        {
                            "origin": "mcts_refinement",
                            "parent_a": member.expr_hash,
                        },
                    )
                    if ind and ind.expr_hash not in ctx.seen_hashes:
                        candidates.append(ind)
            except Exception as e:
                logger.warning(f"MCTS refinement failed for {member.formula[:40]}: {e}")

        return candidates

    def on_evaluation_complete(
        self, ctx: SearchContext, evaluated: list[Individual]
    ) -> None:
        # Record MCTS results to strategy memory too
        if ctx.strategy_memory is not None:
            for ind in evaluated:
                ctx.strategy_memory.record(
                    formula=ind.formula,
                    theme_id="mcts_refinement",
                    metrics=ind.metrics,
                    is_novel=True,
                    round_idx=ctx.round_idx,
                    all_fields=ctx.schema.fields,
                )

    def get_stats(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "activation_frequency": self.activation_frequency,
            "top_k_to_refine": self.top_k_to_refine,
            "zoo_size": len(self.mcts_engine.alpha_zoo) if self.mcts_engine else 0,
        }
