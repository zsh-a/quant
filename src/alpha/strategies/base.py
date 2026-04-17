"""
Strategy base class and metadata.

Provides:
  - ``StrategyMeta`` — single source of truth for strategy identity and UI metadata
  - ``BaseStrategy``  — optional ABC with shared boilerplate (compile_and_dedup,
    default on_evaluation_complete, default get_stats)

Strategies are NOT required to inherit BaseStrategy — the ``SearchStrategy``
protocol from ``context.py`` is the only contract with the orchestrator.
BaseStrategy is purely opt-in convenience.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from ..search.context import SearchContext, build_individual
from ..search.evolution import Individual


@dataclass(frozen=True)
class StrategyMeta:
    """Strategy metadata — defined once per strategy class.

    Used by the registry for construction and by the UI for display.
    """

    registry_name: str       # key in registry, also returned by ``name`` property
    label: str               # human-readable label (中文 for UI)
    brief: str               # one-line description
    detail: str = ""         # optional multi-line description


class BaseStrategy(ABC):
    """Optional base class providing shared boilerplate for search strategies.

    Subclasses must:
      - Define a class-level ``meta: ClassVar[StrategyMeta]``
      - Implement ``should_activate`` and ``generate_candidates``

    Subclasses get for free:
      - ``name`` property (from meta)
      - ``compile_and_dedup`` static helper
      - Default ``on_evaluation_complete`` (records to strategy_memory)
      - Default ``get_stats``
    """

    meta: ClassVar[StrategyMeta]

    @property
    def name(self) -> str:
        return self.meta.registry_name

    @abstractmethod
    def should_activate(self, ctx: SearchContext) -> bool: ...

    @abstractmethod
    def generate_candidates(self, ctx: SearchContext) -> list[Individual]: ...

    def on_evaluation_complete(
        self, ctx: SearchContext, evaluated: list[Individual],
    ) -> None:
        """Default: record each evaluated individual to strategy_memory."""
        if ctx.strategy_memory is not None:
            for ind in evaluated:
                ctx.strategy_memory.record(
                    formula=ind.formula,
                    theme_id=self.name,
                    metrics=ind.metrics,
                    is_novel=True,
                    round_idx=ctx.round_idx,
                    all_fields=ctx.schema.fields,
                )

    def get_stats(self) -> dict[str, Any]:
        """Override in subclass for richer stats."""
        return {"strategy": self.name}

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compile_and_dedup(
        ctx: SearchContext,
        formulas: list[str],
        lineage_fn: Any,
        limit: int | None = None,
    ) -> list[Individual]:
        """Compile formulas and deduplicate against ``ctx.seen_hashes``.

        ``lineage_fn(formula: str) -> Lineage | dict`` controls per-formula
        lineage (e.g. attaching theme info or parent hashes).
        """
        results: list[Individual] = []
        for formula in formulas:
            ind = build_individual(
                ctx.compiler, ctx.schema, formula, lineage_fn(formula),
            )
            if ind is not None and ind.expr_hash not in ctx.seen_hashes:
                results.append(ind)
                if limit is not None and len(results) >= limit:
                    break
        return results
