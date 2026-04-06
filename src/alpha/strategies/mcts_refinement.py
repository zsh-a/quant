"""
MCTS refinement strategy — pluggable SearchStrategy wrapper.

Wraps the paper's MCTSEngine (Algorithm 1) as a pluggable SearchStrategy.
Periodically refines the top archive members via LLM-guided MCTS tree search
with multi-dimensional evaluation, FSA, and dynamic budget allocation.

Reference: "Navigating the Alpha Jungle: An LLM-Powered MCTS Framework
for Formulaic Factor Mining" (Shi et al., 2025)
"""

from __future__ import annotations

from typing import Any

from loguru import logger

import json as _json

from ..strategy_state import SearchContext, build_individual
from ..evolution import Individual
from ..pipeline import Lineage
from ..strategy_state import StrategySnapshot


class MCTSRefinementStrategy:
    """MCTS refinement of archive elites via LLM-guided tree search.

    Activates every ``activation_frequency`` rounds.  Takes the top-k
    individuals from the shared archive, seeds a fresh MCTS tree for each,
    and runs the full Algorithm 1 loop.  Discovered candidates are returned
    for the orchestrator to evaluate via CPCV.

    Compared to the old wrapper, this version:
      - Passes the existing alpha zoo from SearchContext so that the
        multi-dimensional evaluation can compute percentile ranks.
      - Seeds the MCTS engine's zoo with archive members for better
        FSA and diversity scoring from the first iteration.
      - Reports richer stats including tree depth, budget usage, and
        per-dimension score distributions.
    """

    def __init__(
        self,
        mcts_engine: Any,  # MCTSEngine (avoid circular import)
        activation_frequency: int = 3,
        top_k_to_refine: int = 2,
        iterations_per_refine: int = 5,
    ) -> None:
        self.mcts_engine = mcts_engine
        self.activation_frequency = activation_frequency
        self.top_k_to_refine = top_k_to_refine
        self.iterations_per_refine = iterations_per_refine
        self._total_trees_searched: int = 0
        self._total_zoo_found: int = 0

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
            ctx.archive.values(), key=lambda x: x.fitness, reverse=True,
        )
        top_members = top_members[: self.top_k_to_refine]

        # Seed the MCTS engine's zoo with existing archive formulas for
        # better percentile ranking and FSA from the start.
        self._seed_zoo_from_archive(ctx)

        candidates: list[Individual] = []
        for member in top_members:
            try:
                # Reset tree but keep zoo (accumulated across refinements)
                self.mcts_engine.root = None
                self.mcts_engine._factor_cache.clear()

                self.mcts_engine.run(
                    initial_formula=member.formula,
                    dataset=ctx.dataset,
                    iterations=self.iterations_per_refine,
                )
                self._total_trees_searched += 1

                for formula in self.mcts_engine.get_refined_formulas():
                    ind = build_individual(
                        ctx.compiler,
                        ctx.schema,
                        formula,
                        Lineage(
                            origin="mcts_refinement",
                            parent_a=member.expr_hash,
                        ),
                    )
                    if ind and ind.expr_hash not in ctx.seen_hashes:
                        candidates.append(ind)
            except Exception as e:
                logger.warning(
                    "MCTS refinement failed for {}: {}",
                    member.formula[:40], e,
                )

        self._total_zoo_found = len(self.mcts_engine.alpha_zoo)
        logger.info(
            "mcts_refinement.generate trees={} zoo={} candidates={}",
            self._total_trees_searched,
            self._total_zoo_found,
            len(candidates),
        )
        return candidates

    def on_evaluation_complete(
        self, ctx: SearchContext, evaluated: list[Individual],
    ) -> None:
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
        engine = self.mcts_engine
        return {
            "strategy": self.name,
            "activation_frequency": self.activation_frequency,
            "top_k_to_refine": self.top_k_to_refine,
            "zoo_size": len(engine.alpha_zoo) if engine else 0,
            "total_trees_searched": self._total_trees_searched,
            "tree_depth": engine._tree_depth() if engine and engine.root else 0,
            "tree_size": engine._tree_size() if engine and engine.root else 0,
            "forbidden_subtrees": (
                engine._forbidden_subtrees[:3] if engine else []
            ),
        }

    # ------------------------------------------------------------------
    # Zoo seeding from archive
    # ------------------------------------------------------------------

    def _seed_zoo_from_archive(self, ctx: SearchContext) -> None:
        """Pre-populate the MCTS zoo with archive members.

        This gives the multi-dimensional evaluation meaningful percentile
        baselines and provides FSA with enough formulas to detect patterns.
        """
        from ..mcts import AlphaNode

        existing_formulas = {n.formula for n in self.mcts_engine.alpha_zoo}
        added = 0

        for member in ctx.archive.values():
            if member.formula in existing_formulas:
                continue
            node = AlphaNode(formula=member.formula)
            node.metrics = dict(member.metrics)
            node.alpha_score = member.fitness
            node.visits = 1
            self.mcts_engine.alpha_zoo.append(node)
            existing_formulas.add(member.formula)
            added += 1

        if added > 0:
            # Recompute FSA with the seeded zoo
            from ..mcts import compute_forbidden_subtrees

            self.mcts_engine._forbidden_subtrees = compute_forbidden_subtrees(
                [n.formula for n in self.mcts_engine.alpha_zoo],
                top_k=self.mcts_engine.fsa_top_k,
            )
            logger.debug(
                "mcts_refinement.seed_zoo added={} total={}",
                added, len(self.mcts_engine.alpha_zoo),
            )

    # ------------------------------------------------------------------
    # StatefulStrategy interface — warm-start zoo
    # ------------------------------------------------------------------

    def save_state(self) -> StrategySnapshot:
        """Serialize the MCTS alpha zoo for warm-starting."""
        zoo_data = [
            {
                "formula": node.formula,
                "metrics": node.metrics,
                "eval_scores": node.eval_scores,
                "alpha_score": node.alpha_score,
                "visits": node.visits,
                "name": node.name,
                "description": node.description,
            }
            for node in self.mcts_engine.alpha_zoo
        ]
        return StrategySnapshot(
            strategy_name=self.name,
            round_idx=0,
            format="json",
            data=_json.dumps(zoo_data).encode("utf-8"),
            metadata={
                "zoo_size": len(zoo_data),
                "total_trees": self._total_trees_searched,
                "forbidden_subtrees": self.mcts_engine._forbidden_subtrees[:5],
            },
        )

    def load_state(self, snapshot: StrategySnapshot) -> None:
        """Restore alpha zoo from a previous checkpoint."""
        from ..mcts import AlphaNode, compute_forbidden_subtrees

        zoo_data = _json.loads(snapshot.data.decode("utf-8"))
        self.mcts_engine.alpha_zoo = []
        for item in zoo_data:
            node = AlphaNode(formula=item["formula"])
            node.metrics = item.get("metrics", {})
            node.eval_scores = item.get("eval_scores", {})
            node.alpha_score = item.get("alpha_score", 0.0)
            node.visits = item.get("visits", 0)
            node.name = item.get("name", "")
            node.description = item.get("description", "")
            self.mcts_engine.alpha_zoo.append(node)

        # Recompute FSA from restored zoo
        self.mcts_engine._forbidden_subtrees = compute_forbidden_subtrees(
            [n.formula for n in self.mcts_engine.alpha_zoo],
            top_k=self.mcts_engine.fsa_top_k,
        )

        logger.info(
            "mcts_refinement.restored zoo_size={} forbidden={}",
            len(self.mcts_engine.alpha_zoo),
            len(self.mcts_engine._forbidden_subtrees),
        )
