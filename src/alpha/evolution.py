from __future__ import annotations

import hashlib
import random
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Protocol

import numpy as np
from loguru import logger

from .compiler import BytecodeProgram, FormulaCompiler
from .dsl import TensorSchema
from .operators import OperatorRegistry


@dataclass
class Individual:
    formula: str
    program: BytecodeProgram
    expr_hash: str
    lineage: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    fitness: float = 0.0


@dataclass
class BreedingSpec:
    parent_a: str
    parent_b: str | None
    objective: str
    max_nodes: int = 24
    parent_feedback: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class FitnessPolicy:
    min_active_bar_ratio: float = 0.10
    min_signal_coverage: float = 0.50
    min_avg_turnover: float = 0.005
    min_test_sharpe: float = 0.0
    max_negative_test_ratio: float = 0.50
    reject_score: float = -5.0
    sharpe_scale: float = 2.0
    test_sharpe_scale: float = 1.5
    rank_ic_scale: float = 0.05
    pnl_efficiency_cap: float = 10.0
    tail_ratio_scale: float = 2.0


class LLMBackend(Protocol):
    def generate_initial_population(self, count: int) -> list[str]:
        ...

    def generate_offspring(self, spec: BreedingSpec, count: int) -> list[str]:
        ...


class HeuristicLLMBackend:
    """
    Local structured backend.

    It is still deterministic and lightweight, but it now generates formulas via
    mutation, crossover, and objective-aware template wrapping so the closed
    loop can explore a meaningfully larger operator space.
    """

    def __init__(self, registry: OperatorRegistry | None = None, schema: TensorSchema | None = None):
        self.registry = registry or OperatorRegistry()
        self.schema = schema or TensorSchema.default_market_schema()
        self.call_stats = {
            "initial_population_calls": 0,
            "offspring_calls": 0,
        }

    @property
    def backend_name(self) -> str:
        return "heuristic"

    def generate_offspring(self, spec: BreedingSpec, count: int) -> list[str]:
        self.call_stats["offspring_calls"] += 1
        offspring: list[str] = []
        bases = [spec.parent_a]
        if spec.parent_b:
            bases.append(spec.parent_b)
        attempts = 0
        max_attempts = max(count * 8, 8)
        while len(offspring) < count and attempts < max_attempts:
            base = bases[attempts % len(bases)]
            if spec.parent_b and attempts % 3 == 0:
                candidate = self._crossover_formula(spec.parent_a, spec.parent_b, attempts)
            elif attempts % 3 == 1:
                candidate = self._mutate_formula(base, attempts)
            else:
                candidate = self._wrap_formula(base, spec.objective, attempts)
            if candidate not in offspring:
                offspring.append(candidate)
            attempts += 1
        if not offspring:
            offspring.append(self._mutate_formula(spec.parent_a, 0))
        return offspring

    def generate_initial_population(self, count: int) -> list[str]:
        self.call_stats["initial_population_calls"] += 1
        seeds = [
            "cs_rank(ts_mean(close, 5) - close)",
            "cs_rank(ts_std(close, 10))",
            "cs_rank(delta(premium_close, 5))",
            "cs_rank(ts_zscore(funding_rate, 20))",
            "cs_rank(delta(open_interest, 10) - ts_mean(delta(open_interest, 10), 20))",
            "cs_rank(ts_zscore(long_short_ratio, 20))",
            "cs_rank(div(taker_buy_volume, volume + 1e-12) - 0.5)",
            "cs_rank(ts_corr(close, taker_buy_volume, 10))",
        ]
        generated: list[str] = []
        attempts = 0
        while len(generated) < count and attempts < max(count * 4, 8):
            base = seeds[attempts % len(seeds)]
            candidate = base if attempts < len(seeds) else self._wrap_formula(base, "bootstrap", attempts)
            if candidate not in generated:
                generated.append(candidate)
            attempts += 1
        return generated[:count]

    def _mutate_formula(self, formula: str, variant: int = 0) -> str:
        replacements = [
            ("ts_mean(", "ts_std("),
            ("ts_std(", "ts_mean("),
            ("ts_max(", "ts_rank("),
            ("ts_rank(", "ts_mean("),
            ("close", "vwap"),
            ("close", "mark_close"),
            ("volume", "turnover"),
            ("volume", "taker_buy_volume"),
            ("close", "hlc3(high, low, close)"),
            ("turnover", "adv_n(turnover, 5)"),
            ("close", "ohlc4(open, high, low, close)"),
            ("volatility_n(close, 20)", "atr_n(high, low, close, 14)"),
            ("funding_rate", "ts_zscore(funding_rate, 20)"),
            ("open_interest", "delta(open_interest, 5)"),
            ("close", "premium_close"),
            ("volume", "trade_count"),
        ]
        offset = self._stable_index(formula, len(replacements), salt=f"mutate:{variant}")
        for idx in range(len(replacements)):
            source, target = replacements[(offset + idx) % len(replacements)]
            if source in formula:
                return formula.replace(source, target, 1)
        return self._wrap_formula(formula, "mutation fallback", variant)

    def _wrap_formula(self, formula: str, objective: str, variant: int = 0) -> str:
        wrappers = [
            f"cs_rank(({formula}) - amihud(close, turnover, 5))",
            f"cs_rank(({formula}) + atr_n(high, low, close, 5))",
            f"cs_zscore(decay_linear(({formula}), 3))",
            f"cs_rank(fillna(({formula}), 0) + cs_demean(vwap))",
            f"cs_zscore(clip(({formula}), -3, 3) + adv_n(turnover, 10))",
            f"cs_rank(ts_zscore(({formula}), 5) - volatility_n(close, 10))",
            f"cs_rank(({formula}) + ts_corr(close, volume, 10))",
            f"cs_rank(({formula}) - ts_rank(turnover, 20))",
            f"cs_rank(decay_linear(({formula}), 5) + ts_mean(volume, 10))",
            # New-field-aware wrappers
            f"cs_rank(({formula}) + delta(premium_close, 5))",
            f"cs_rank(({formula}) - ts_zscore(funding_rate, 20))",
            f"cs_rank(({formula}) + ts_zscore(long_short_ratio, 20))",
            f"cs_rank(({formula}) + delta(open_interest, 10))",
            f"cs_rank(({formula}) - ts_rank(taker_buy_volume, 10))",
        ]
        if "turnover" in objective.lower():
            wrappers.extend(
                [
                    f"cs_rank(decay_linear(({formula}), 5) - spread_ratio(bid_ask_spread, close))",
                    f"cs_rank(fillna(({formula}), 0) - amihud(close, turnover, 10))",
                ]
            )
        idx = self._stable_index(formula, len(wrappers), salt=f"wrap:{objective}:{variant}")
        return wrappers[idx]

    def _crossover_formula(self, parent_a: str, parent_b: str, variant: int = 0) -> str:
        templates = [
            f"cs_rank(({parent_a}) + ({parent_b}))",
            f"cs_rank(({parent_a}) - ({parent_b}))",
            f"cs_zscore(decay_linear((({parent_a}) + ({parent_b})), 3))",
            f"cs_rank(max(({parent_a}), ({parent_b})) - volatility_n(close, 10))",
            f"cs_rank(min(({parent_a}), ({parent_b})) + ts_corr(close, volume, 10))",
            f"cs_rank((({parent_a}) + atr_n(high, low, close, 5)) - (({parent_b}) + amihud(close, turnover, 5)))",
            f"cs_rank(ts_mean(({parent_a}), 3) - ts_mean(({parent_b}), 3))",
        ]
        idx = self._stable_index(
            f"{parent_a}|{parent_b}",
            len(templates),
            salt=f"cross:{variant}",
        )
        return templates[idx]

    def _stable_index(self, value: str, modulo: int, salt: str = "") -> int:
        digest = hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()
        return int(digest[:12], 16) % max(modulo, 1)


class FitnessEngine:
    def __init__(self, policy: FitnessPolicy | None = None):
        self.policy = policy or FitnessPolicy()

    def _clip(self, value: float, lo: float, hi: float) -> float:
        return float(np.clip(value, lo, hi))

    def _safe(self, metrics: dict[str, float], key: str, default: float = 0.0) -> float:
        try:
            return float(metrics.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def score(self, metrics: dict[str, float]) -> float:
        active_bar_ratio = self._safe(metrics, "active_bar_ratio")
        signal_coverage = self._safe(metrics, "signal_coverage", 1.0)
        avg_turnover = self._safe(metrics, "avg_turnover")
        test_sharpe = self._safe(metrics, "test_sharpe")
        negative_test_ratio = self._safe(metrics, "negative_test_ratio")
        inactive = self._safe(metrics, "inactive")

        reject_reasons = 0
        if inactive >= 1.0 or active_bar_ratio < self.policy.min_active_bar_ratio:
            reject_reasons += 1
        if signal_coverage < self.policy.min_signal_coverage:
            reject_reasons += 1
        if avg_turnover < self.policy.min_avg_turnover:
            reject_reasons += 1
        if test_sharpe < self.policy.min_test_sharpe:
            reject_reasons += 1
        if negative_test_ratio > self.policy.max_negative_test_ratio:
            reject_reasons += 1
        if reject_reasons:
            return self.policy.reject_score - 0.25 * float(reject_reasons - 1)

        sharpe_score = self._clip(self._safe(metrics, "sharpe") / self.policy.sharpe_scale, -1.0, 1.0)
        test_score = self._clip(test_sharpe / self.policy.test_sharpe_scale, -1.0, 1.0)
        ic_score = self._clip(
            self._safe(metrics, "rank_ic_abs") / self.policy.rank_ic_scale,
            0.0,
            1.0,
        )
        pnl_efficiency = max(self._safe(metrics, "pnl_per_turnover"), 0.0)
        efficiency_score = self._clip(
            np.log1p(pnl_efficiency) / np.log1p(self.policy.pnl_efficiency_cap),
            0.0,
            1.0,
        )
        tail_score = self._clip(
            self._safe(metrics, "tail_ratio") / self.policy.tail_ratio_scale,
            -1.0,
            1.0,
        )
        activity_score = self._clip(self._safe(metrics, "activity_score"), 0.0, 1.0)
        turnover_penalty = self._clip(self._safe(metrics, "turnover_penalty"), 0.0, 1.0)
        complexity_penalty = self._clip(self._safe(metrics, "complexity_penalty"), 0.0, 1.0)
        train_valid_gap_penalty = self._clip(self._safe(metrics, "train_valid_gap_penalty"), 0.0, 1.0)
        valid_test_gap_penalty = self._clip(self._safe(metrics, "valid_test_gap_penalty"), 0.0, 1.0)
        coverage_penalty = self._clip(self._safe(metrics, "coverage_penalty"), 0.0, 1.0)

        return (
            0.28 * sharpe_score
            + 0.20 * test_score
            + 0.17 * ic_score
            + 0.12 * efficiency_score
            + 0.10 * tail_score
            + 0.08 * activity_score
            - 0.10 * turnover_penalty
            - 0.07 * complexity_penalty
            - 0.08 * train_valid_gap_penalty
            - 0.10 * valid_test_gap_penalty
            - 0.05 * coverage_penalty
        )




# ---------------------------------------------------------------------------
# Evaluation callback types
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result from an evaluation callback."""

    metrics_by_hash: dict[str, dict[str, float]]
    signatures_by_hash: dict[str, list[float]]
    details_by_hash: dict[str, dict[str, Any]]
    timing: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Final output of SearchEngine.run()."""

    archive: list[Individual]
    all_evaluated: list[Individual]
    details_by_hash: dict[str, dict[str, Any]]
    rounds: list[dict[str, Any]]
    timing: dict[str, Any]
    total_evaluations: int
    total_rejected: int


# ---------------------------------------------------------------------------
# SearchEngine — Regularized Evolution with Quality-Diversity Archive
# ---------------------------------------------------------------------------


# Archive bin boundaries for MAP-Elites grid
_IC_BINS = (0.0, 0.02, 0.05, float("inf"))
_TURNOVER_BINS = (0.0, 0.05, 0.20, float("inf"))


def _bin_index(value: float, edges: tuple[float, ...]) -> int:
    for i in range(len(edges) - 1):
        if value < edges[i + 1]:
            return i
    return len(edges) - 2


class SearchEngine:
    """
    Regularised Evolution with Quality-Diversity archive.

    Differences from the old synchronous GA:

    * **Tournament selection + aging** — each round samples a small tournament
      from the population, picks the best parents, breeds offspring.  Old
      individuals are evicted by age, preventing premature convergence.

    * **Early stopping** — new candidates are first quick-screened on a single
      validation fold; only promising ones proceed to full CPCV evaluation.
      This typically eliminates ~40-60 % of candidates cheaply.

    * **MAP-Elites archive** — an (|IC| × turnover) grid stores the best
      individual per behavioural niche.  The archive is the primary output,
      guaranteeing diversity in the final result set.

    * **No synchronous generations** — evaluation budget is spent continuously,
      which is more efficient and easier to parallelise in the future.
    """

    def __init__(
        self,
        llm_backend: LLMBackend | None = None,
        compiler: FormulaCompiler | None = None,
        registry: OperatorRegistry | None = None,
        schema: TensorSchema | None = None,
        backend_name: str = "auto",
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        population_cap: int = 30,
        tournament_size: int = 7,
    ):
        self.registry = registry or OperatorRegistry()
        self.compiler = compiler or FormulaCompiler(self.registry)
        self.schema = schema or TensorSchema.default_market_schema()
        if llm_backend is None:
            from .llm import build_default_llm_backend

            llm_backend = build_default_llm_backend(
                registry=self.registry,
                schema=self.schema,
                backend_name=backend_name,
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
            )
        self.llm_backend = llm_backend
        self.fitness_engine = FitnessEngine()

        self._population_cap = population_cap
        self._tournament_size = tournament_size

    # -- public API ---------------------------------------------------------

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
    ) -> SearchResult:
        """
        Execute the search loop.

        Parameters
        ----------
        seeds : initial formula strings
        rounds : number of breed-evaluate rounds
        batch_size : offspring generated per round
        top_k : how many to return from archive
        novelty_threshold : signature correlation ceiling
        evaluate_fn : full CPCV evaluator (provided by service)
        quick_evaluate_fn : optional 1-fold quick screener
        """
        overall_start = perf_counter()

        # 1. Initialise population from seeds
        population: deque[Individual] = deque(maxlen=self._population_cap)
        archive: dict[tuple[int, int], Individual] = {}
        seen_hashes: set[str] = set()
        all_evaluated: list[Individual] = []
        details_by_hash: dict[str, dict[str, Any]] = {}
        round_summaries: list[dict[str, Any]] = []
        total_evaluations = 0
        total_rejected = 0

        init_start = perf_counter()
        initial = self._build_initial_population(seeds, max(batch_size, len(seeds)))
        init_seconds = perf_counter() - init_start

        # Evaluate initial batch
        if initial:
            init_eval = evaluate_fn(initial)
            details_by_hash.update(init_eval.details_by_hash)
            for ind in initial:
                m = init_eval.metrics_by_hash.get(ind.expr_hash, {})
                ind.metrics = m
                ind.fitness = self.fitness_engine.score(m)
                population.append(ind)
                seen_hashes.add(ind.expr_hash)
                all_evaluated.append(ind)
                self._archive_update(archive, ind)
            total_evaluations += len(initial)

        logger.info(
            "search.init population={} archive={} init_seconds={:.3f}",
            len(population), len(archive), init_seconds,
        )

        # 2. Main loop
        for round_idx in range(rounds):
            round_start = perf_counter()
            round_info: dict[str, Any] = {"round": round_idx}

            # Tournament selection → parents
            parents = self._tournament_select(list(population))

            # Generate offspring batch
            gen_start = perf_counter()
            offspring = self._breed_batch(parents, batch_size, seen_hashes)
            round_info["generate_seconds"] = perf_counter() - gen_start
            round_info["candidates_generated"] = len(offspring)

            if not offspring:
                round_summaries.append(round_info)
                continue

            # Quick screen (optional early stopping)
            screened = offspring
            quick_rejected = 0
            if quick_evaluate_fn and len(offspring) > 1:
                screen_start = perf_counter()
                quick_result = quick_evaluate_fn(offspring)
                screened = []
                for ind in offspring:
                    qm = quick_result.metrics_by_hash.get(ind.expr_hash, {})
                    if self._passes_quick_screen(qm):
                        screened.append(ind)
                    else:
                        quick_rejected += 1
                round_info["quick_screen_seconds"] = perf_counter() - screen_start
            round_info["quick_rejected"] = quick_rejected
            total_rejected += quick_rejected

            # Full evaluate survivors
            if screened:
                eval_start = perf_counter()
                full_result = evaluate_fn(screened)
                round_info["eval_seconds"] = perf_counter() - eval_start
                details_by_hash.update(full_result.details_by_hash)

                for ind in screened:
                    m = full_result.metrics_by_hash.get(ind.expr_hash, {})
                    ind.metrics = m
                    ind.fitness = self.fitness_engine.score(m)
                    population.append(ind)  # aging: oldest auto-evicted by deque
                    seen_hashes.add(ind.expr_hash)
                    all_evaluated.append(ind)
                    self._archive_update(archive, ind)

                total_evaluations += len(screened)

            round_info["evaluated"] = len(screened)
            round_info["archive_size"] = len(archive)
            round_info["population_size"] = len(population)
            round_info["total_seconds"] = perf_counter() - round_start

            best_in_archive = max(archive.values(), key=lambda x: x.fitness) if archive else None
            round_info["best_fitness"] = best_in_archive.fitness if best_in_archive else 0.0

            round_summaries.append(round_info)
            logger.info(
                "search.round round={} generated={} rejected={} evaluated={} "
                "archive={} best_fitness={:.4f} seconds={:.3f}",
                round_idx, len(offspring), quick_rejected, len(screened),
                len(archive),
                round_info["best_fitness"],
                round_info["total_seconds"],
            )

        # 3. Build final result from archive
        archive_list = sorted(archive.values(), key=lambda x: x.fitness, reverse=True)

        # Novelty filter on archive for the final top_k
        final = self._novelty_filter(archive_list, details_by_hash, novelty_threshold)[:top_k]

        timing = {
            "init_seconds": init_seconds,
            "rounds": round_summaries,
            "overall_seconds": perf_counter() - overall_start,
        }

        logger.info(
            "search.complete total_eval={} rejected={} archive={} final={} seconds={:.3f}",
            total_evaluations, total_rejected, len(archive), len(final), timing["overall_seconds"],
        )

        return SearchResult(
            archive=final,
            all_evaluated=all_evaluated,
            details_by_hash=details_by_hash,
            rounds=round_summaries,
            timing=timing,
            total_evaluations=total_evaluations,
            total_rejected=total_rejected,
        )

    # -- internals ----------------------------------------------------------

    def _build_initial_population(self, seeds: list[str], target: int) -> list[Individual]:
        population: list[Individual] = []
        seen: set[str] = set()
        # Compile seeds
        for formula in seeds:
            ind = self._build_individual(formula, {"origin": "seed"})
            if ind and ind.expr_hash not in seen:
                population.append(ind)
                seen.add(ind.expr_hash)
        # Fill with heuristic mutations
        heuristic = HeuristicLLMBackend(registry=self.registry, schema=self.schema)
        if not seeds:
            seeds = heuristic.generate_initial_population(target)
            for formula in seeds:
                ind = self._build_individual(formula, {"origin": "seed"})
                if ind and ind.expr_hash not in seen:
                    population.append(ind)
                    seen.add(ind.expr_hash)
        attempt = 0
        while len(population) < target and attempt < target * 4:
            base = seeds[attempt % len(seeds)] if seeds else "cs_rank(ts_std(close, 10))"
            for f in heuristic.generate_offspring(
                BreedingSpec(parent_a=base, parent_b=None, objective="bootstrap"), count=1,
            ):
                ind = self._build_individual(f, {"origin": "bootstrap"})
                if ind and ind.expr_hash not in seen:
                    population.append(ind)
                    seen.add(ind.expr_hash)
            attempt += 1
        return population[:target]

    def _tournament_select(self, population: list[Individual]) -> list[Individual]:
        """Sample a tournament, return top 2 as parents."""
        if len(population) <= 2:
            return list(population)
        k = min(self._tournament_size, len(population))
        tournament = random.sample(population, k)
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[:2]

    def _breed_batch(
        self,
        parents: list[Individual],
        count: int,
        seen: set[str],
    ) -> list[Individual]:
        """Generate a batch of offspring from parents."""
        if not parents:
            return []
        parent_a = parents[0]
        parent_b = parents[1] if len(parents) > 1 else None

        spec = BreedingSpec(
            parent_a=parent_a.formula,
            parent_b=parent_b.formula if parent_b else None,
            objective="improve robustness and reduce turnover",
            parent_feedback=[
                {"formula": parent_a.formula, "metrics": parent_a.metrics,
                 "rationale": "Tournament winner."},
                *(
                    [{"formula": parent_b.formula, "metrics": parent_b.metrics,
                      "rationale": "Tournament runner-up."}]
                    if parent_b else []
                ),
            ],
        )
        formulas = self.llm_backend.generate_offspring(spec, count=count)
        offspring: list[Individual] = []
        for f in formulas:
            ind = self._build_individual(f, {
                "parent_a": parent_a.expr_hash,
                "parent_b": parent_b.expr_hash if parent_b else None,
            })
            if ind and ind.expr_hash not in seen:
                offspring.append(ind)
        return offspring

    def _passes_quick_screen(self, metrics: dict[str, float]) -> bool:
        """Fast rejection of obviously bad candidates."""
        if float(metrics.get("inactive", 0)) >= 1.0:
            return False
        if float(metrics.get("signal_coverage", 1.0)) < 0.30:
            return False
        if float(metrics.get("active_bar_ratio", 1.0)) < 0.05:
            return False
        return True

    @staticmethod
    def _archive_cell(ind: Individual) -> tuple[int, int]:
        ic = abs(float(ind.metrics.get("rank_ic_abs", ind.metrics.get("rank_ic", 0)) or 0))
        turnover = float(ind.metrics.get("avg_turnover", 0) or 0)
        return (_bin_index(ic, _IC_BINS), _bin_index(turnover, _TURNOVER_BINS))

    def _archive_update(self, archive: dict[tuple[int, int], Individual], ind: Individual) -> None:
        if ind.fitness <= self.fitness_engine.policy.reject_score:
            return
        cell = self._archive_cell(ind)
        existing = archive.get(cell)
        if existing is None or ind.fitness > existing.fitness:
            archive[cell] = ind

    def _novelty_filter(
        self,
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

    def _build_individual(self, formula: str, lineage: dict[str, Any]) -> Individual | None:
        try:
            program = self.compiler.compile(formula, self.schema)
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
