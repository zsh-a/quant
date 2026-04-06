"""
LLM-driven regularized evolution strategy.

Tournament selection + LLM breeding with RL feedback loop
(strategy_memory) and financial-semantic prompts.
"""

from __future__ import annotations

import random
from typing import Any

from loguru import logger

from ..search.evolution import BreedingSpec, Individual
from ..search.pipeline import Lineage
from ..search.context import SearchContext, build_individual


class LLMEvolutionStrategy:
    """LLM-driven evolutionary search — the primary search strategy.

    Each round:
    1. Tournament-select parents from the shared population.
    2. Build a BreedingSpec with parent metrics for LLM context.
    3. Call llm_backend.generate_offspring() to produce formulas.
    4. Compile into Individuals and return to the orchestrator.

    On evaluation complete:
    - Records results to strategy_memory (closing the RL loop).
    """

    def __init__(
        self,
        llm_backend: Any,
        tournament_size: int = 7,
        batch_size: int | None = None,
    ) -> None:
        self.llm_backend = llm_backend
        self.tournament_size = tournament_size
        self._batch_size = batch_size  # None = use ctx.batch_size

    @property
    def name(self) -> str:
        return "llm_evolution"

    def should_activate(self, ctx: SearchContext) -> bool:
        return True  # Every round

    def generate_candidates(self, ctx: SearchContext) -> list[Individual]:
        batch_size = self._batch_size or ctx.batch_size
        if not ctx.population:
            # Genesis: use initial population from the LLM
            formulas = self.llm_backend.generate_initial_population(batch_size)
            return self._compile_formulas(ctx, formulas, Lineage(origin="llm_genesis"))

        parents = self._tournament_select(list(ctx.population))
        if not parents:
            return []

        parent_a = parents[0]
        parent_b = parents[1] if len(parents) > 1 else None

        spec = BreedingSpec(
            parent_a=parent_a.formula,
            parent_b=parent_b.formula if parent_b else None,
            objective="improve robustness and reduce turnover",
            parent_feedback=[
                {
                    "formula": parent_a.formula,
                    "metrics": parent_a.metrics,
                    "rationale": "Tournament winner.",
                },
                *(
                    [
                        {
                            "formula": parent_b.formula,
                            "metrics": parent_b.metrics,
                            "rationale": "Tournament runner-up.",
                        }
                    ]
                    if parent_b
                    else []
                ),
            ],
        )

        formulas = self.llm_backend.generate_offspring(spec, count=batch_size)
        return self._compile_formulas_with_theme(ctx, formulas, Lineage(
            origin="llm_evolution",
            parent_a=parent_a.expr_hash,
            parent_b=parent_b.expr_hash if parent_b else None,
        ))

    def on_evaluation_complete(
        self, ctx: SearchContext, evaluated: list[Individual]
    ) -> None:
        # Close the RL feedback loop
        if hasattr(self.llm_backend, "record_evaluation_result"):
            for ind in evaluated:
                theme = ind.lineage.theme if isinstance(ind.lineage, Lineage) else ind.lineage.get("theme")
                self.llm_backend.record_evaluation_result(
                    ind.formula,
                    theme,
                    ind.metrics,
                    is_novel=True,
                )

    def get_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {"strategy": self.name}
        if hasattr(self.llm_backend, "call_stats"):
            stats["llm_calls"] = self.llm_backend.call_stats
        return stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tournament_select(self, population: list[Individual]) -> list[Individual]:
        """Sample a tournament, return top 2 as parents."""
        if len(population) <= 2:
            return list(population)
        k = min(self.tournament_size, len(population))
        tournament = random.sample(population, k)
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[:2]

    def _compile_formulas_with_theme(
        self,
        ctx: SearchContext,
        formulas: list[str],
        base_lineage: Lineage,
    ) -> list[Individual]:
        """Compile formulas and attach theme info from LLM output."""
        offspring: list[Individual] = []
        for f in formulas:
            lineage = Lineage(
                origin=base_lineage.origin,
                parent_a=base_lineage.parent_a,
                parent_b=base_lineage.parent_b,
            )
            # Retrieve theme assigned by LLM during extraction
            if hasattr(self.llm_backend, "get_theme_for_formula"):
                theme = self.llm_backend.get_theme_for_formula(f)
                if theme:
                    lineage.theme = theme
            ind = build_individual(ctx.compiler, ctx.schema, f, lineage)
            if ind and ind.expr_hash not in ctx.seen_hashes:
                offspring.append(ind)
        return offspring

    @staticmethod
    def _compile_formulas(
        ctx: SearchContext,
        formulas: list[str],
        lineage: Lineage,
    ) -> list[Individual]:
        offspring: list[Individual] = []
        for f in formulas:
            lin = Lineage(origin=lineage.origin, theme=lineage.theme)
            ind = build_individual(ctx.compiler, ctx.schema, f, lin)
            if ind and ind.expr_hash not in ctx.seen_hashes:
                offspring.append(ind)
        return offspring


# --- Registry ---
from .registry import register_strategy  # noqa: E402


@register_strategy("llm_evolution")
def _build_llm_evolution(*, llm_backend, **_kw):
    return LLMEvolutionStrategy(llm_backend=llm_backend)
