"""
MCTS refinement strategy — pluggable SearchStrategy wrapper.

Wraps the paper's MCTSEngine (Algorithm 1) as a pluggable SearchStrategy.
Periodically refines the top archive members via LLM-guided MCTS tree search
with multi-dimensional evaluation, FSA, and dynamic budget allocation.

Reference: "Navigating the Alpha Jungle: An LLM-Powered MCTS Framework
for Formulaic Factor Mining" (Shi et al., 2025)
"""

from __future__ import annotations

import json as _json
from typing import Any, ClassVar

from loguru import logger

from ..search.context import SearchContext, StrategySnapshot
from ..search.evolution import Individual
from ..search.pipeline import Lineage
from .base import BaseStrategy, StrategyMeta


class MCTSRefinementStrategy(BaseStrategy):
    """MCTS refinement of archive elites via LLM-guided tree search.

    Activates every ``activation_frequency`` rounds.  Takes the top-k
    individuals from the shared archive, seeds a fresh MCTS tree for each,
    and runs the full Algorithm 1 loop.  Discovered candidates are returned
    for the orchestrator to evaluate via CPCV.
    """

    meta: ClassVar[StrategyMeta] = StrategyMeta(
        registry_name="mcts",
        label="MCTS 精炼",
        brief="LLM 引导的蒙特卡洛树搜索 (Navigating the Alpha Jungle)",
        detail=(
            "从 archive 精英出发构建搜索树，UCT 选择 + 维度定向精化 + FSA 子树回避。\n"
            "LLM 生成精化建议，经验证后加入搜索树。动态预算随发现自动增加。"
        ),
        always_on=False,
    )

    def __init__(
        self,
        mcts_engine: Any,  # MCTSEngine
        activation_frequency: int = 3,
        top_k_to_refine: int = 2,
        iterations_per_refine: int = 5,
    ) -> None:
        self.mcts_engine = mcts_engine
        self.activation_frequency = activation_frequency
        self.top_k_to_refine = top_k_to_refine
        self.iterations_per_refine = iterations_per_refine
        self._total_trees_searched: int = 0

    def should_activate(self, ctx: SearchContext) -> bool:
        has_seeds = len(ctx.archive) > 0 or len(ctx.population) > 0
        active = (
            ctx.round_idx > 0
            and ctx.round_idx % self.activation_frequency == 0
            and has_seeds
            and ctx.dataset is not None
        )
        logger.info(
            "mcts.should_activate={} round={} archive={} population={} dataset={}",
            active, ctx.round_idx, len(ctx.archive), len(ctx.population),
            ctx.dataset is not None,
        )
        return active

    def generate_candidates(self, ctx: SearchContext) -> list[Individual]:
        # Prefer archive members; fall back to population if archive is empty
        if ctx.archive:
            pool = sorted(ctx.archive.values(), key=lambda x: x.fitness, reverse=True)
        else:
            pool = sorted(ctx.population, key=lambda x: x.fitness, reverse=True)
        top_members = pool[: self.top_k_to_refine]

        # Seed the MCTS engine's zoo with existing archive formulas
        self._seed_zoo_from_archive(ctx)

        # Pass shared evaluator to engine
        self.mcts_engine.evaluator = ctx.evaluator

        candidates: list[Individual] = []
        for member in top_members:
            try:
                self.mcts_engine.reset_tree()

                self.mcts_engine.run(
                    initial_formula=member.formula,
                    dataset=ctx.dataset,
                    iterations=self.iterations_per_refine,
                )
                self._total_trees_searched += 1

                refined = self.mcts_engine.get_refined_formulas()
                new = self.compile_and_dedup(
                    ctx,
                    refined,
                    lineage_fn=lambda _f, _parent=member: Lineage(
                        origin="mcts_refinement",
                        parent_a=_parent.expr_hash,
                    ),
                )
                candidates.extend(new)
            except Exception as e:
                logger.warning(
                    "MCTS refinement failed for {}: {}",
                    member.formula[:40], e,
                )

        logger.info(
            "mcts_refinement.generate trees={} zoo={} candidates={}",
            self._total_trees_searched,
            len(self.mcts_engine.alpha_zoo),
            len(candidates),
        )
        return candidates

    def get_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "strategy": self.name,
            "activation_frequency": self.activation_frequency,
            "top_k_to_refine": self.top_k_to_refine,
            "total_trees_searched": self._total_trees_searched,
        }
        stats.update(self.mcts_engine.get_search_stats())
        return stats

    # ------------------------------------------------------------------
    # Zoo seeding from archive
    # ------------------------------------------------------------------

    def _seed_zoo_from_archive(self, ctx: SearchContext) -> None:
        """Pre-populate the MCTS zoo with archive members.

        This gives the multi-dimensional evaluation meaningful percentile
        baselines and provides FSA with enough formulas to detect patterns.
        """
        entries = [
            {"formula": m.formula, "metrics": dict(m.metrics), "fitness": m.fitness}
            for m in ctx.archive.values()
        ]
        added = self.mcts_engine.seed_zoo(entries)
        if added > 0:
            logger.debug(
                "mcts_refinement.seed_zoo added={} total={}",
                added, len(self.mcts_engine.alpha_zoo),
            )

    # ------------------------------------------------------------------
    # StatefulStrategy interface — warm-start zoo
    # ------------------------------------------------------------------

    def save_state(self) -> StrategySnapshot:
        """Serialize the MCTS alpha zoo for warm-starting."""
        zoo_data = self.mcts_engine.get_zoo_snapshot()
        return StrategySnapshot(
            strategy_name=self.name,
            round_idx=0,
            format="json",
            data=_json.dumps(zoo_data).encode("utf-8"),
            metadata={
                "zoo_size": len(zoo_data),
                "total_trees": self._total_trees_searched,
            },
        )

    def load_state(self, snapshot: StrategySnapshot) -> None:
        """Restore alpha zoo from a previous checkpoint."""
        zoo_data = _json.loads(snapshot.data.decode("utf-8"))
        self.mcts_engine.restore_zoo(zoo_data)

        logger.info(
            "mcts_refinement.restored zoo_size={} forbidden={}",
            len(self.mcts_engine.alpha_zoo),
            len(self.mcts_engine.get_search_stats().get("forbidden_subtrees", [])),
        )


# --- Registry ---
from .registry import register_strategy, StrategyInfra  # noqa: E402


@register_strategy(MCTSRefinementStrategy.meta)
def _build_mcts(infra: StrategyInfra):
    from .mcts import MCTSEngine, MCTSLLMAdapter
    llm_adapter = MCTSLLMAdapter(infra.llm_backend)
    engine = MCTSEngine(
        compiler=infra.compiler, vm=infra.vm, schema=infra.schema,
        llm_agent=llm_adapter,
        c_puct=1.0, initial_budget=3, budget_increment=1,
        temperature=1.0, fsa_top_k=3,
        zoo_threshold=0.015, effectiveness_threshold=0.3,
    )
    return MCTSRefinementStrategy(
        mcts_engine=engine,
        activation_frequency=infra.mcts_frequency,
        top_k_to_refine=3,
        iterations_per_refine=5,
    )
