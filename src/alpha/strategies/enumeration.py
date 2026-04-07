"""
Enumeration strategy — bulk seeding via programmatic formula generation.

On round 0, generates hundreds of candidates via FormulaEnumerator,
screens them with a fast IC-only check (no CPCV), and returns the top-K
as pre-screened Individual objects for the orchestrator to evaluate.
"""

from __future__ import annotations

from typing import Any, ClassVar

from loguru import logger

from ..search.evolution import Individual
from ..search.pipeline import Lineage
from ..search.context import SearchContext
from .base import BaseStrategy, StrategyMeta


class EnumerationStrategy(BaseStrategy):
    """Flood the initial population with programmatically enumerated formulas.

    Complementary to LLM-based strategies: enumeration provides breadth,
    LLM provides depth.
    """

    meta: ClassVar[StrategyMeta] = StrategyMeta(
        registry_name="enumeration",
        label="枚举种子",
        brief="Round 0 批量枚举公式 + fast-IC 筛选 top-K",
    )

    def __init__(
        self,
        max_enumerate: int = 500,
        top_k: int = 30,
        min_abs_ic: float = 0.015,
    ) -> None:
        self.max_enumerate = max_enumerate
        self.top_k = top_k
        self.min_abs_ic = min_abs_ic

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

        screened_formulas = [f for f, _ic in passed[: self.top_k]]
        ic_by_formula = {f: ic for f, ic in passed[: self.top_k]}

        candidates = self.compile_and_dedup(
            ctx,
            screened_formulas,
            lineage_fn=lambda f: Lineage(
                origin="enumeration",
                screen_ic=round(ic_by_formula.get(f, 0.0), 5),
            ),
            limit=self.top_k,
        )

        logger.info(
            "enumeration.generate enumerated={} ic_passed={} candidates={}",
            len(formulas), len(passed), len(candidates),
        )
        return candidates

    def on_evaluation_complete(
        self, ctx: SearchContext, evaluated: list[Individual],
    ) -> None:
        pass  # Enumeration does no learning


# --- Registry ---
from .registry import register_strategy, StrategyInfra  # noqa: E402


@register_strategy(EnumerationStrategy.meta)
def _build_enumeration(infra: StrategyInfra):
    return EnumerationStrategy(max_enumerate=infra.enum_max, top_k=infra.enum_top_k)
