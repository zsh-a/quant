"""
Pluggable search strategy framework for alpha factor discovery.

Defines the SearchStrategy protocol, SearchContext shared state, and
SearchOrchestrator that manages multiple strategies in a unified loop.

Current strategies:
  - LLMEvolutionStrategy (strategies/llm_evolution.py)
  - MCTSRefinementStrategy (strategies/mcts_refinement.py)
  - NeuralFormulaStrategy (strategies/neural_formula.py)

Future strategies (each just implements the same 4-method interface):
  - DAGEvolutionStrategy (AlphaPROBE)
  - GrammarGuidedStrategy (AlphaCFG)
  - SynergyRLStrategy (Synergistic RL)
  - NeuralGenerativeStrategy (AlphaForge)
  - DistributionalRLStrategy (AlphaQCM)
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
from loguru import logger

from .compiler import BytecodeProgram, FormulaCompiler
from .dataset import AlphaDataset
from .dsl import TensorSchema
from .evolution import (
    EvalResult,
    FitnessEngine,
    FitnessPolicy,
    HeuristicLLMBackend,
    Individual,
    BreedingSpec,
    SearchResult,
)
from .operators import OperatorRegistry


# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------


@dataclass
class SearchContext:
    """Shared state accessible to all strategies via a single object.

    Strategies read/write population and archive through this context,
    enabling indirect collaboration without direct coupling.
    """

    # --- shared collections ---
    population: deque[Individual]
    archive: dict[tuple[int, int], Individual]
    seen_hashes: set[str]
    all_evaluated: list[Individual]
    details_by_hash: dict[str, dict[str, Any]]

    # --- infrastructure ---
    compiler: FormulaCompiler
    schema: TensorSchema
    registry: OperatorRegistry
    fitness_engine: FitnessEngine

    # --- evaluation callbacks ---
    evaluate_fn: Callable[[list[Individual]], EvalResult]
    quick_evaluate_fn: Callable[[list[Individual]], EvalResult] | None

    # --- optional enhanced modules ---
    strategy_memory: Any | None  # StrategyMemory (avoid circular import)
    knowledge_base: Any | None  # FinancialKnowledgeBase
    feature_kitchen: Any | None  # FeatureKitchen

    # --- dataset (needed by some strategies like MCTS) ---
    dataset: AlphaDataset | None = None

    # --- search state ---
    round_idx: int = 0
    total_rounds: int = 0
    total_evaluations: int = 0
    total_rejected: int = 0
    batch_size: int = 8


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SearchStrategy(Protocol):
    """Interface that all search strategies must implement.

    To add a new strategy (e.g. from a paper), implement these 4 methods
    and register the strategy with SearchOrchestrator.
    """

    @property
    def name(self) -> str:
        """Strategy name for logging and tracing."""
        ...

    def should_activate(self, ctx: SearchContext) -> bool:
        """Whether this strategy should run in the current round."""
        ...

    def generate_candidates(self, ctx: SearchContext) -> list[Individual]:
        """Generate candidate Individuals (compiled, not yet evaluated).

        The orchestrator handles quick-screen, full evaluation, archive
        update, and population management.
        """
        ...

    def on_evaluation_complete(
        self, ctx: SearchContext, evaluated: list[Individual]
    ) -> None:
        """Callback after evaluation — for internal learning.

        Called with the evaluated individuals (metrics and fitness filled in).
        """
        ...


# ---------------------------------------------------------------------------
# Archive helpers (shared across orchestrator)
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
    if ind.fitness <= reject_score:
        return
    cell = archive_cell(ind)
    existing = archive.get(cell)
    if existing is None or ind.fitness > existing.fitness:
        archive[cell] = ind


def build_individual(
    compiler: FormulaCompiler,
    schema: TensorSchema,
    formula: str,
    lineage: dict[str, Any],
) -> Individual | None:
    """Compile a formula into an Individual, or return None on failure."""
    try:
        program = compiler.compile(formula, schema)
    except ValueError:
        return None
    return Individual(
        formula=formula,
        program=program,
        expr_hash=program.expr_hash,
        lineage=lineage,
    )


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
# Enumeration Strategy (bulk seeding via programmatic generation)
# ---------------------------------------------------------------------------


class EnumerationStrategy:
    """Flood the initial population with programmatically enumerated formulas.

    On round 0, generates hundreds of candidates via :class:`FormulaEnumerator`,
    screens them with a fast IC-only check (no CPCV), and returns the top-K as
    pre-screened :class:`Individual` objects for the orchestrator to evaluate.

    This is complementary to LLM-based strategies: enumeration provides breadth,
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
        # Only run on the genesis round for bulk seeding
        return ctx.round_idx == 0 and ctx.dataset is not None

    def generate_candidates(self, ctx: SearchContext) -> list[Individual]:
        from .enumerator import FormulaEnumerator
        from .fast_screen import fast_screen_ic

        # 1. Enumerate formulas
        enumerator = FormulaEnumerator(compiler=ctx.compiler, schema=ctx.schema)
        formulas = enumerator.generate(max_count=self.max_enumerate)
        if not formulas:
            return []

        # 2. Fast IC screen on the full dataset (no CPCV)
        from .vm import StackVM
        vm = StackVM()
        passed = fast_screen_ic(
            formulas,
            ctx.dataset,
            ctx.compiler,
            vm,
            ctx.schema,
            min_abs_ic=self.min_abs_ic,
        )

        # 3. Compile top-K into Individuals
        candidates: list[Individual] = []
        for formula, ic in passed[:self.top_k]:
            if len(candidates) >= self.top_k:
                break
            ind = build_individual(ctx.compiler, ctx.schema, formula, {"origin": "enumeration", "screen_ic": round(ic, 5)})
            if ind and ind.expr_hash not in ctx.seen_hashes:
                candidates.append(ind)

        logger.info(
            "enumeration.generate enumerated={} ic_passed={} candidates={}",
            len(formulas), len(passed), len(candidates),
        )
        return candidates

    def on_evaluation_complete(
        self, ctx: SearchContext, evaluated: list[Individual],
    ) -> None:
        pass  # No internal learning needed


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
    ) -> SearchResult:
        """Execute the search loop with all registered strategies."""
        from .tracing import tracer

        overall_start = perf_counter()

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
            dataset=dataset,
            round_idx=0,
            total_rounds=rounds,
            total_evaluations=0,
            total_rejected=0,
            batch_size=batch_size,
        )

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
        from concurrent.futures import ThreadPoolExecutor, Future
        prefetch_future: Future | None = None
        prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm_prefetch")

        for round_idx in range(rounds):
            round_start = perf_counter()
            ctx.round_idx = round_idx

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

            for strategy in self.strategies:
                if not strategy.should_activate(ctx):
                    continue

                strategy_name = strategy.name
                round_info["strategies_activated"].append(strategy_name)

                with tracer.start_span(
                    f"strategy_{strategy_name}", kind="search",
                    strategy=strategy_name, round=round_idx,
                ) as strat_span:
                    # 1. Generate candidates (or use prefetched from previous round)
                    if prefetch_future is not None and strategy_name == "llm_evolution":
                        try:
                            candidates = prefetch_future.result(timeout=120)
                        except Exception:
                            candidates = strategy.generate_candidates(ctx)
                        prefetch_future = None
                    else:
                        candidates = strategy.generate_candidates(ctx)
                    strat_span.set("candidates_generated", len(candidates))

                    if not candidates:
                        continue

                    # 2. Quick screen
                    screened = candidates
                    quick_rejected = 0
                    if quick_evaluate_fn and len(candidates) > 1:
                        with tracer.start_span(
                            "quick_screen", kind="eval", count=len(candidates),
                        ) as qs_span:
                            quick_result = quick_evaluate_fn(candidates)
                            screened = []
                            for ind in candidates:
                                qm = quick_result.metrics_by_hash.get(ind.expr_hash, {})
                                if self._passes_quick_screen(qm):
                                    screened.append(ind)
                                else:
                                    quick_rejected += 1
                            qs_span.set("passed", len(screened))
                            qs_span.set("rejected", quick_rejected)
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
                        with tracer.start_span(
                            "evaluate", kind="eval", count=len(screened),
                        ) as eval_span:
                            self._evaluate_and_update(ctx, screened)
                            best = max((ind.fitness for ind in screened), default=-999)
                            eval_span.set("best_fitness", round(best, 4))

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

            round_span.set("archive", len(ctx.archive))
            round_span.set("best_fitness", round(round_info["best_fitness"], 4))
            round_ctx.__exit__(None, None, None)

            round_summaries.append(round_info)

        prefetch_executor.shutdown(wait=False)

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
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _MAX_EVALUATED_HISTORY = 200  # cap to bound memory; archive keeps the best

    def _evaluate_and_update(self, ctx: SearchContext, individuals: list[Individual]) -> None:
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

        ctx.total_evaluations += len(individuals)

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
            ind = build_individual(self.compiler, self.schema, formula, {"origin": "seed"})
            if ind and ind.expr_hash not in seen:
                population.append(ind)
                seen.add(ind.expr_hash)

        heuristic = HeuristicLLMBackend(registry=self.registry, schema=self.schema)
        if not seeds:
            seeds = heuristic.generate_initial_population(target)
            for formula in seeds:
                ind = build_individual(self.compiler, self.schema, formula, {"origin": "seed"})
                if ind and ind.expr_hash not in seen:
                    population.append(ind)
                    seen.add(ind.expr_hash)

        attempt = 0
        while len(population) < target and attempt < target * 4:
            base = seeds[attempt % len(seeds)] if seeds else "cs_rank(ts_std(close, 10))"
            for f in heuristic.generate_offspring(
                BreedingSpec(parent_a=base, parent_b=None, objective="bootstrap"), count=1,
            ):
                ind = build_individual(self.compiler, self.schema, f, {"origin": "bootstrap"})
                if ind and ind.expr_hash not in seen:
                    population.append(ind)
                    seen.add(ind.expr_hash)
            attempt += 1

        return population[:target]

    @staticmethod
    def _passes_quick_screen(metrics: dict[str, float]) -> bool:
        if float(metrics.get("inactive", 0)) >= 1.0:
            return False
        if float(metrics.get("signal_coverage", 1.0)) < 0.30:
            return False
        if float(metrics.get("active_bar_ratio", 1.0)) < 0.05:
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
