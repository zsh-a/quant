"""
Enumeration strategy — bulk seeding via programmatic formula generation.

On round 0, generates hundreds of candidates via FormulaEnumerator,
screens them with a fast IC-only check (no CPCV), and returns the top-K
as pre-screened Individual objects for the orchestrator to evaluate.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ..search.evolution import Individual
from ..search.pipeline import Lineage
from ..search.context import SearchContext, build_individual


class EnumerationStrategy:
    """Flood the initial population with programmatically enumerated formulas.

    Complementary to LLM-based strategies: enumeration provides breadth,
    LLM provides depth.
    """

    def __init__(
        self,
        max_enumerate: int = 500,
        top_k: int = 30,
        min_abs_ic: float = 0.015,
    ) -> None:
        self.max_enumerate = max_enumerate
        self.top_k = top_k
        self.min_abs_ic = min_abs_ic

    @property
    def name(self) -> str:
        return "enumeration"

    def should_activate(self, ctx: SearchContext) -> bool:
        return ctx.round_idx == 0 and ctx.dataset is not None

    def generate_candidates(self, ctx: SearchContext) -> list[Individual]:
        from ..search.enumerator import FormulaEnumerator

        enumerator = FormulaEnumerator(compiler=ctx.compiler, schema=ctx.schema)
        formulas = enumerator.generate(max_count=self.max_enumerate)
        if not formulas:
            return []

        # Use shared evaluator if available, else fall back to fast_screen_ic
        if ctx.evaluator is not None:
            passed = [
                (f, ic) for f, ic in ctx.evaluator.eval_ic_batch(formulas)
                if abs(ic) >= self.min_abs_ic
            ]
            passed.sort(key=lambda x: abs(x[1]), reverse=True)
        else:
            from ..eval.fast_screen import fast_screen_ic
            from ..core.vm import StackVM
            vm = ctx.vm or StackVM()
            passed = fast_screen_ic(
                formulas, ctx.dataset, ctx.compiler, vm, ctx.schema,
                min_abs_ic=self.min_abs_ic,
            )

        candidates: list[Individual] = []
        for formula, ic in passed[: self.top_k]:
            ind = build_individual(
                ctx.compiler, ctx.schema, formula,
                Lineage(origin="enumeration", screen_ic=round(ic, 5)),
            )
            if ind and ind.expr_hash not in ctx.seen_hashes:
                candidates.append(ind)
            if len(candidates) >= self.top_k:
                break

        logger.info(
            "enumeration.generate enumerated={} ic_passed={} candidates={}",
            len(formulas), len(passed), len(candidates),
        )
        return candidates

    def on_evaluation_complete(
        self, ctx: SearchContext, evaluated: list[Individual],
    ) -> None:
        pass


# --- Registry ---
from .registry import register_strategy  # noqa: E402


@register_strategy("enumeration")
def _build_enumeration(*, enum_max: int = 500, enum_top_k: int = 30, **_kw):
    return EnumerationStrategy(max_enumerate=enum_max, top_k=enum_top_k)
