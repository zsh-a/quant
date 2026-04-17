"""
Search orchestrator for alpha factor discovery.

Manages multiple search strategies in a unified loop with:
  - Quick-screen filtering
  - Full CPCV evaluation
  - MAP-Elites archive management
  - Checkpoint/restore
  - Pipeline tracing
"""

from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from loguru import logger

from ..core.compiler import FormulaCompiler
from ..core.dsl import TensorSchema
from ..core.operators import OperatorRegistry

if TYPE_CHECKING:
    from ..core.dataset import AlphaDataset
from .context import (
    FactorCatalog,
    FactorCatalogEntry,
    SearchContext,
    SearchStrategy,
    build_individual,
)
from .evolution import (
    BreedingSpec,
    EvalResult,
    FitnessEngine,
    Individual,
    SearchResult,
)
from .pipeline import (
    ArchiveEntry,
    Lineage,
    PipelineRecord,
    RoundRecord,
    StageKind,
    StageRecord,
)

# ---------------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------------

_IC_BINS = (0.0, 0.02, 0.05, float("inf"))
_TURNOVER_BINS = (0.0, 0.05, 0.20, float("inf"))


def _bin_index(value: float, edges: tuple[float, ...]) -> int:
    for i in range(len(edges) - 1):
        if value < edges[i + 1]:
            return i
    return len(edges) - 2


def archive_cell(ind: Individual) -> tuple[int, int]:
    ic = abs(float(ind.metrics.get("rank_ic_abs", ind.metrics.get("rank_ic", 0)) or 0))
    turnover = float(ind.metrics.get("avg_turnover", 0) or 0)
    return (_bin_index(ic, _IC_BINS), _bin_index(turnover, _TURNOVER_BINS))


def archive_update(
    archive: dict[tuple[int, int], Individual],
    ind: Individual,
    reject_score: float = -5.0,
) -> None:
    if ind.fitness < reject_score:
        return
    cell = archive_cell(ind)
    existing = archive.get(cell)
    if existing is None or ind.fitness > existing.fitness:
        archive[cell] = ind


def _sig_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Pearson correlation between two signature vectors."""
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    am, bm = a - a.mean(), b - b.mean()
    denom = np.sqrt((am * am).sum() * (bm * bm).sum())
    if denom < 1e-12:
        return 1.0
    return abs(float((am * bm).sum() / denom))


# ---------------------------------------------------------------------------
# Search Orchestrator
# ---------------------------------------------------------------------------


class SearchOrchestrator:
    """Unified orchestrator managing multiple search strategies.

    Each round, iterates over registered strategies, generates candidates,
    evaluates them through a shared pipeline, and updates the MAP-Elites archive.
    """

    def __init__(
        self,
        strategies: list[SearchStrategy],
        compiler: FormulaCompiler | None = None,
        registry: OperatorRegistry | None = None,
        schema: TensorSchema | None = None,
        fitness_engine: FitnessEngine | None = None,
        strategy_memory: Any | None = None,
        knowledge_base: Any | None = None,
        feature_kitchen: Any | None = None,
        population_cap: int = 30,
        checkpoint_manager: Any | None = None,
    ) -> None:
        self.registry = registry or OperatorRegistry()
        self.compiler = compiler or FormulaCompiler(self.registry)
        self.schema = schema or TensorSchema.default_market_schema()
        self.fitness_engine = fitness_engine or FitnessEngine()
        self.strategies = list(strategies)
        self.strategy_memory = strategy_memory
        self.knowledge_base = knowledge_base
        self.feature_kitchen = feature_kitchen
        self._population_cap = population_cap
        self._checkpoint_manager = checkpoint_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        seeds: list[str],
        rounds: int,
        batch_size: int,
        top_k: int,
        novelty_threshold: float,
        evaluate_fn: Callable[[list[Individual]], EvalResult],
        quick_evaluate_fn: Callable[[list[Individual]], EvalResult] | None = None,
        dataset: AlphaDataset | None = None,
        vm: Any | None = None,
        job_id: str = "",
        on_stage_complete: Callable[[StageRecord], None] | None = None,
        on_round_complete: Callable[[RoundRecord], None] | None = None,
        resume_from: str | None = None,
    ) -> SearchResult:
        """Execute the search loop with all registered strategies."""
        from pathlib import Path

        from ..infra.tracing import tracer

        overall_start = perf_counter()
        pipeline = PipelineRecord(job_id=job_id)

        # Restore from checkpoint if resuming
        start_round = 0
        restored_catalog: FactorCatalog | None = None
        if resume_from and self._checkpoint_manager:
            ckpt_path = Path(resume_from)
            if ckpt_path.exists():
                restored_catalog, saved_state = self._checkpoint_manager.restore_strategies(
                    ckpt_path, self.strategies,
                )
                start_round = saved_state.get("round_idx", -1) + 1
                logger.info("search.resume from round={}", start_round)

        # Build shared evaluator for strategies
        evaluator = None
        if dataset is not None:
            from ..core.vm import StackVM
            from .evaluator import FormulaEvaluator
            _eval_vm = vm or StackVM()
            evaluator = FormulaEvaluator(self.compiler, _eval_vm, self.schema, dataset)

        # Build context
        ctx = SearchContext(
            population=deque(maxlen=self._population_cap),
            archive={},
            seen_hashes=set(),
            all_evaluated=[],
            details_by_hash={},
            compiler=self.compiler,
            schema=self.schema,
            registry=self.registry,
            fitness_engine=self.fitness_engine,
            evaluate_fn=evaluate_fn,
            quick_evaluate_fn=quick_evaluate_fn,
            strategy_memory=self.strategy_memory,
            knowledge_base=self.knowledge_base,
            feature_kitchen=self.feature_kitchen,
            vm=vm,
            evaluator=evaluator,
            dataset=dataset,
            round_idx=0,
            total_rounds=rounds,
            total_evaluations=0,
            total_rejected=0,
            batch_size=batch_size,
            factor_catalog=restored_catalog or FactorCatalog(),
        )

        if start_round > 0:
            ctx.round_idx = start_round

        round_summaries: list[dict[str, Any]] = []

        # --- init ---
        with tracer.start_span(
            "init", kind="search",
            seed_count=len(seeds), batch_size=batch_size,
        ) as init_span:
            initial = self._build_initial_population(seeds, max(batch_size, len(seeds)))
            init_span.set("compiled", len(initial))

            if initial:
                with tracer.start_span("init_evaluate", kind="eval", count=len(initial)):
                    self._evaluate_and_update(ctx, initial)

            init_span.set("population", len(ctx.population))
            init_span.set("archive", len(ctx.archive))
        init_seconds = perf_counter() - overall_start

        # --- main loop (with pipeline overlap) ---
        # Pre-generated candidates from the previous round's LLM call.
        # While evaluation runs, the next round's LLM call is already in flight.
        from concurrent.futures import Future, ThreadPoolExecutor
        prefetch_future: Future | None = None
        prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm_prefetch")

        # --- warm-start: relax fitness thresholds while archive is empty ---
        _warm_start_active = False
        _WARM_MAX_ROUNDS = 3

        for round_idx in range(rounds):
            round_start = perf_counter()
            ctx.round_idx = round_idx

            # Enable warm-start during early rounds when archive is empty.
            # Once the archive has entries, warm-start is disabled permanently.
            if not ctx.archive and round_idx < _WARM_MAX_ROUNDS:
                if not _warm_start_active:
                    self.fitness_engine.set_warm(True)
                    _warm_start_active = True
                    logger.info(
                        "search.warm_start enabled (archive empty at round {})",
                        round_idx,
                    )
            elif _warm_start_active:
                self.fitness_engine.set_warm(False)
                _warm_start_active = False
                logger.info(
                    "search.warm_start disabled (archive={} round={})",
                    len(ctx.archive), round_idx,
                )

            round_ctx = tracer.start_span(
                f"round_{round_idx}", kind="search",
                round=round_idx, population=len(ctx.population),
                archive=len(ctx.archive),
            )
            round_span = round_ctx.__enter__()

            round_info: dict[str, Any] = {
                "round": round_idx,
                "strategies_activated": [],
            }
            round_rec = RoundRecord(round_idx=round_idx)

            for strategy in self.strategies:
                if not strategy.should_activate(ctx):
                    continue

                strategy_name = strategy.name
                round_info["strategies_activated"].append(strategy_name)
                round_rec.strategies_activated.append(strategy_name)

                with tracer.start_span(
                    f"strategy_{strategy_name}", kind="search",
                    strategy=strategy_name, round=round_idx,
                ) as strat_span:
                    # 1. Generate candidates (or use prefetched from previous round)
                    gen_start = perf_counter()
                    if prefetch_future is not None and strategy_name == "llm_evolution":
                        try:
                            candidates = prefetch_future.result(timeout=120)
                        except Exception:
                            candidates = strategy.generate_candidates(ctx)
                        prefetch_future = None
                    else:
                        candidates = strategy.generate_candidates(ctx)
                    strat_span.set("candidates_generated", len(candidates))

                    gen_stage = StageRecord(
                        kind=StageKind.GENERATE,
                        strategy=strategy_name,
                        round_idx=round_idx,
                        input_count=0,
                        output_count=len(candidates),
                        duration_ms=(perf_counter() - gen_start) * 1000,
                    )
                    round_rec.stages.append(gen_stage)
                    if on_stage_complete:
                        on_stage_complete(gen_stage)

                    if not candidates:
                        continue

                    # 1b. Dedup against previously evaluated formulas
                    before_dedup = len(candidates)
                    candidates = [
                        ind for ind in candidates
                        if ind.expr_hash not in ctx.seen_hashes
                    ]
                    dedup_removed = before_dedup - len(candidates)
                    if dedup_removed > 0:
                        logger.info(
                            "alpha.dedup strategy={} removed={} kept={}",
                            strategy_name, dedup_removed, len(candidates),
                        )
                    gen_stage.metadata["dedup_removed"] = dedup_removed
                    gen_stage.output_count = len(candidates)

                    # Mark as seen immediately so other strategies in this round skip them
                    for ind in candidates:
                        ctx.seen_hashes.add(ind.expr_hash)

                    if not candidates:
                        continue

                    # 2. Quick screen
                    screened = candidates
                    quick_rejected = 0
                    if quick_evaluate_fn and len(candidates) > 1:
                        qs_start = perf_counter()
                        with tracer.start_span(
                            "quick_screen", kind="eval", count=len(candidates),
                        ) as qs_span:
                            quick_result = quick_evaluate_fn(candidates)
                            screened = []
                            qs_diag_logged = 0
                            for ind in candidates:
                                qm = quick_result.metrics_by_hash.get(ind.expr_hash, {})
                                if self._passes_quick_screen(qm):
                                    screened.append(ind)
                                else:
                                    quick_rejected += 1
                                    if qs_diag_logged < 3:
                                        logger.debug(
                                            "quick_screen.reject formula={} coverage={:.3f} active={:.3f} inactive={} turnover={:.4f}",
                                            ind.formula[:50],
                                            float(qm.get("signal_coverage", -1)),
                                            float(qm.get("active_bar_ratio", -1)),
                                            qm.get("inactive", "?"),
                                            float(qm.get("avg_turnover", -1)),
                                        )
                                        qs_diag_logged += 1
                            if quick_rejected > 0 and len(screened) == 0:
                                # All rejected — log summary for diagnosis
                                sample = quick_result.metrics_by_hash
                                first_hash = next(iter(sample), None)
                                if first_hash:
                                    sm = sample[first_hash]
                                    logger.warning(
                                        "quick_screen.all_rejected count={} sample_metrics: coverage={:.3f} active={:.3f} inactive={} turnover={:.5f} sharpe={:.4f}",
                                        quick_rejected,
                                        float(sm.get("signal_coverage", -1)),
                                        float(sm.get("active_bar_ratio", -1)),
                                        sm.get("inactive", "?"),
                                        float(sm.get("avg_turnover", -1)),
                                        float(sm.get("sharpe", 0)),
                                    )
                                # Force-pass best N to avoid total stall
                                force_n = min(max(ctx.batch_size, 4), len(candidates))
                                def _qs_score(ind: Individual) -> float:
                                    qm = quick_result.metrics_by_hash.get(ind.expr_hash, {})
                                    return float(qm.get("signal_coverage", 0)) + abs(float(qm.get("rank_ic", 0)))
                                candidates_ranked = sorted(candidates, key=_qs_score, reverse=True)
                                screened = candidates_ranked[:force_n]
                                quick_rejected -= len(screened)
                                logger.warning(
                                    "quick_screen.force_pass count={} (best by coverage+IC to prevent search stall)",
                                    len(screened),
                                )
                            qs_span.set("passed", len(screened))
                            qs_span.set("rejected", quick_rejected)

                        qs_stage = StageRecord(
                            kind=StageKind.QUICK_SCREEN,
                            strategy=strategy_name,
                            round_idx=round_idx,
                            input_count=len(candidates),
                            output_count=len(screened),
                            duration_ms=(perf_counter() - qs_start) * 1000,
                        )
                        round_rec.stages.append(qs_stage)
                        if on_stage_complete:
                            on_stage_complete(qs_stage)
                    ctx.total_rejected += quick_rejected

                    # 3. Top-N gating: limit full CPCV to top candidates
                    max_full_eval = max(ctx.batch_size * 2, 8)
                    if quick_evaluate_fn and len(screened) > max_full_eval:
                        quick_result_for_rank = quick_evaluate_fn(screened)
                        if quick_result_for_rank:
                            def _qs_rank(ind: Individual) -> float:
                                qm = quick_result_for_rank.metrics_by_hash.get(ind.expr_hash, {})
                                return abs(float(qm.get("rank_ic", 0))) + float(qm.get("sharpe", 0)) * 0.1
                            screened.sort(key=_qs_rank, reverse=True)
                        screened = screened[:max_full_eval]
                        logger.info("search.top_n_gate passed={} limit={}", len(screened), max_full_eval)

                    # 4. Pipeline: pre-generate next round's LLM candidates
                    #    while this round's evaluation runs.
                    next_round_idx = round_idx + 1
                    if (
                        next_round_idx < rounds
                        and strategy_name == "llm_evolution"
                        and prefetch_future is None
                    ):
                        # Snapshot context for prefetch (population won't change
                        # until after evaluation completes below).
                        def _prefetch_gen(strat=strategy, c=ctx):
                            try:
                                return strat.generate_candidates(c)
                            except Exception:
                                return []
                        prefetch_future = prefetch_executor.submit(_prefetch_gen)

                    # 5. Full evaluate
                    if screened:
                        eval_start = perf_counter()
                        with tracer.start_span(
                            "evaluate", kind="eval", count=len(screened),
                        ) as eval_span:
                            self._evaluate_and_update(ctx, screened, strategy_name=strategy_name)
                            best = max((ind.fitness for ind in screened), default=-999)
                            eval_span.set("best_fitness", round(best, 4))

                        eval_stage = StageRecord(
                            kind=StageKind.EVALUATE,
                            strategy=strategy_name,
                            round_idx=round_idx,
                            input_count=len(screened),
                            output_count=len(screened),
                            duration_ms=(perf_counter() - eval_start) * 1000,
                            best_fitness=best,
                        )
                        round_rec.stages.append(eval_stage)
                        if on_stage_complete:
                            on_stage_complete(eval_stage)

                        strategy.on_evaluation_complete(ctx, screened)

                    strat_span.set("evaluated", len(screened))
                    strat_span.set("rejected", quick_rejected)

            round_info["archive_size"] = len(ctx.archive)
            round_info["population_size"] = len(ctx.population)
            round_info["total_seconds"] = perf_counter() - round_start

            best_in_archive = (
                max(ctx.archive.values(), key=lambda x: x.fitness) if ctx.archive else None
            )
            round_info["best_fitness"] = best_in_archive.fitness if best_in_archive else 0.0

            # Build archive snapshot for this round
            round_rec.archive_size = len(ctx.archive)
            round_rec.population_size = len(ctx.population)
            round_rec.best_fitness = round_info["best_fitness"]
            round_rec.duration_ms = (perf_counter() - round_start) * 1000
            round_rec.archive_snapshot = self._build_archive_snapshot(ctx.archive, limit=5)

            # Force-seed: if archive is still empty after warm-start rounds
            # exhausted, push the best candidate from population regardless of
            # fitness score.  This ensures downstream strategies (MCTS) have at
            # least one seed to work with.
            if (
                not ctx.archive
                and ctx.population
                and round_idx == _WARM_MAX_ROUNDS - 1
            ):
                best_pop = max(ctx.population, key=lambda x: x.fitness)
                cell = archive_cell(best_pop)
                ctx.archive[cell] = best_pop
                logger.warning(
                    "search.force_seed archive was empty after {} warm-start "
                    "rounds; injected best from population: {} fitness={:.3f}",
                    _WARM_MAX_ROUNDS,
                    best_pop.formula[:50],
                    best_pop.fitness,
                )
                round_info["archive_size"] = len(ctx.archive)
                round_rec.archive_size = len(ctx.archive)

            round_span.set("archive", len(ctx.archive))
            round_span.set("best_fitness", round(round_info["best_fitness"], 4))
            round_ctx.__exit__(None, None, None)

            round_summaries.append(round_info)
            pipeline.rounds.append(round_rec)
            if on_round_complete:
                on_round_complete(round_rec)

            # Checkpoint at round boundary (if manager configured)
            if (
                self._checkpoint_manager
                and ctx.factor_catalog is not None
                and self._checkpoint_manager.should_checkpoint(round_idx, rounds)
            ):
                self._checkpoint_manager.create_checkpoint(
                    job_id=job_id,
                    round_idx=round_idx,
                    strategies=self.strategies,
                    factor_catalog=ctx.factor_catalog,
                    ctx_state={
                        "round_idx": round_idx,
                        "total_evaluations": ctx.total_evaluations,
                        "total_rejected": ctx.total_rejected,
                    },
                    archive_snapshot=[
                        e.to_dict() for e in self._build_archive_snapshot(ctx.archive, limit=10)
                    ],
                )

        prefetch_executor.shutdown(wait=False)

        # Ensure warm-start is off after search completes
        if _warm_start_active:
            self.fitness_engine.set_warm(False)

        pipeline.total_evaluations = ctx.total_evaluations
        pipeline.total_rejected = ctx.total_rejected

        # --- result ---
        archive_list = sorted(ctx.archive.values(), key=lambda x: x.fitness, reverse=True)
        final = self._novelty_filter(archive_list, ctx.details_by_hash, novelty_threshold)[:top_k]

        timing = {
            "init_seconds": init_seconds,
            "rounds": round_summaries,
            "overall_seconds": perf_counter() - overall_start,
        }

        return SearchResult(
            archive=final,
            all_evaluated=ctx.all_evaluated,
            details_by_hash=ctx.details_by_hash,
            rounds=round_summaries,
            timing=timing,
            total_evaluations=ctx.total_evaluations,
            total_rejected=ctx.total_rejected,
            pipeline=pipeline,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _MAX_EVALUATED_HISTORY = 200  # cap to bound memory; archive keeps the best

    def _evaluate_and_update(
        self,
        ctx: SearchContext,
        individuals: list[Individual],
        strategy_name: str = "",
    ) -> None:
        """Run full evaluation, update fitness/population/archive."""
        result = ctx.evaluate_fn(individuals)
        ctx.details_by_hash.update(result.details_by_hash)

        for ind in individuals:
            m = result.metrics_by_hash.get(ind.expr_hash, {})
            ind.metrics = m
            ind.fitness = self.fitness_engine.score(m)
            ctx.population.append(ind)
            ctx.seen_hashes.add(ind.expr_hash)
            ctx.all_evaluated.append(ind)
            archive_update(ctx.archive, ind, self.fitness_engine.policy.reject_score)

            # Record to factor catalog
            if ctx.factor_catalog is not None:
                origin = ind.lineage.origin if isinstance(ind.lineage, Lineage) else "unknown"
                ctx.factor_catalog.record(FactorCatalogEntry(
                    formula=ind.formula,
                    expr_hash=ind.expr_hash,
                    strategy=strategy_name or origin,
                    round_idx=ctx.round_idx,
                    rank_ic=float(m.get("rank_ic", 0) or 0),
                    sharpe=float(m.get("sharpe", 0) or 0),
                    turnover=float(m.get("avg_turnover", 0) or 0),
                    fitness=ind.fitness,
                    evaluated=True,
                    lineage=ind.lineage.to_dict() if isinstance(ind.lineage, Lineage) else {},
                    parent_hashes=[
                        h for h in [
                            getattr(ind.lineage, "parent_a", None),
                            getattr(ind.lineage, "parent_b", None),
                        ] if h
                    ] if isinstance(ind.lineage, Lineage) else [],
                ))

        ctx.total_evaluations += len(individuals)

        # Diagnostic: log fitness distribution for rejected batches
        if individuals:
            best_ind = max(individuals, key=lambda x: x.fitness)
            reject_threshold = self.fitness_engine.policy.reject_score
            rejected = sum(1 for x in individuals if x.fitness < reject_threshold)
            if rejected == len(individuals):
                m = best_ind.metrics
                reasons = self.fitness_engine.rejection_reasons(m)
                logger.warning(
                    "evaluate.all_rejected n={} best_fitness={:.2f} formula={} "
                    "sharpe={:.3f} test_sharpe={:.3f} rank_ic={:.4f} "
                    "active={:.2f} turnover={:.4f} coverage={:.2f} inactive={} "
                    "neg_test_ratio={:.2f} reasons=[{}]",
                    len(individuals), best_ind.fitness, best_ind.formula[:60],
                    float(m.get("sharpe", 0)), float(m.get("test_sharpe", 0)),
                    float(m.get("rank_ic", 0)), float(m.get("active_bar_ratio", 0)),
                    float(m.get("avg_turnover", 0)), float(m.get("signal_coverage", 0)),
                    m.get("inactive", "?"),
                    float(m.get("negative_test_ratio", 0)),
                    ", ".join(reasons) if reasons else "none",
                )

        # Evict oldest entries to cap memory (archive retains the best)
        while len(ctx.all_evaluated) > self._MAX_EVALUATED_HISTORY:
            evicted = ctx.all_evaluated.pop(0)
            # Keep details only for archive members
            if evicted.expr_hash not in {ind.expr_hash for cell in ctx.archive.values() for ind in [cell]}:
                ctx.details_by_hash.pop(evicted.expr_hash, None)

    def _build_initial_population(self, seeds: list[str], target: int) -> list[Individual]:
        """Compile seeds + fill gap with heuristic mutations."""
        population: list[Individual] = []
        seen: set[str] = set()

        for formula in seeds:
            ind = build_individual(self.compiler, self.schema, formula, Lineage(origin="seed"))
            if ind and ind.expr_hash not in seen:
                population.append(ind)
                seen.add(ind.expr_hash)

        from ..llm import HeuristicLLMBackend
        heuristic = HeuristicLLMBackend(registry=self.registry, schema=self.schema)
        if not seeds:
            seeds = heuristic.generate_initial_population(target)
            for formula in seeds:
                ind = build_individual(self.compiler, self.schema, formula, Lineage(origin="seed"))
                if ind and ind.expr_hash not in seen:
                    population.append(ind)
                    seen.add(ind.expr_hash)

        attempt = 0
        while len(population) < target and attempt < target * 4:
            base = seeds[attempt % len(seeds)] if seeds else "cs_rank(ts_std(close, 10))"
            for f in heuristic.generate_offspring(
                BreedingSpec(parent_a=base, parent_b=None, objective="bootstrap"), count=1,
            ):
                ind = build_individual(self.compiler, self.schema, f, Lineage(origin="bootstrap"))
                if ind and ind.expr_hash not in seen:
                    population.append(ind)
                    seen.add(ind.expr_hash)
            attempt += 1

        return population[:target]

    @staticmethod
    def _build_archive_snapshot(
        archive: dict[tuple[int, int], Individual], limit: int = 5,
    ) -> list[ArchiveEntry]:
        """Build a lightweight snapshot of the top archive members."""
        top = sorted(archive.values(), key=lambda x: x.fitness, reverse=True)[:limit]
        entries: list[ArchiveEntry] = []
        for ind in top:
            entries.append(ArchiveEntry(
                formula=ind.formula,
                expr_hash=ind.expr_hash,
                fitness=ind.fitness,
                rank_ic=float(ind.metrics.get("rank_ic", 0) or 0),
                sharpe=float(ind.metrics.get("sharpe", 0) or 0),
                turnover=float(ind.metrics.get("avg_turnover", 0) or 0),
                origin=ind.lineage.origin if isinstance(ind.lineage, Lineage) else ind.lineage.get("origin", "unknown"),
            ))
        return entries

    _qs_log_count: int = 0

    @staticmethod
    def _passes_quick_screen(metrics: dict[str, float]) -> bool:
        # Empty metrics dict means evaluation failed entirely — reject
        if not metrics:
            return False
        coverage = float(metrics.get("signal_coverage", 1.0))
        active = float(metrics.get("active_bar_ratio", 1.0))
        # Only reject truly degenerate signals:
        # - signal_coverage < 15%: almost entirely NaN
        # - active_bar_ratio < 2%: almost no bars with non-zero weights
        if coverage < 0.15:
            return False
        if active < 0.02:
            return False
        return True

    @staticmethod
    def _novelty_filter(
        individuals: list[Individual],
        details: dict[str, dict[str, Any]],
        threshold: float,
    ) -> list[Individual]:
        """Remove behaviourally similar individuals from a ranked list."""
        kept: list[Individual] = []
        kept_sigs: list[np.ndarray] = []
        for ind in individuals:
            detail = details.get(ind.expr_hash, {})
            sig = detail.get("alpha_signature")
            if sig is None:
                kept.append(ind)
                continue
            sig_arr = np.asarray(sig, dtype=float)
            is_dup = False
            for existing in kept_sigs:
                if _sig_corr(sig_arr, existing) >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(ind)
                kept_sigs.append(sig_arr)
        return kept if kept else individuals[:1]
