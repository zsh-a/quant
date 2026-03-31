from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .compiler import BytecodeProgram, FormulaCompiler
from .dsl import DSLRegistry, TensorSchema


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

    def __init__(self, registry: DSLRegistry | None = None, schema: TensorSchema | None = None):
        self.registry = registry or DSLRegistry()
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
            "cs_rank(volatility_n(close, 20))",
            "cs_rank(oi_delta(open_interest, 1) - spread_ratio(bid_ask_spread, close))",
            "cs_rank(adv_n(turnover, 10) - amihud(close, turnover, 5))",
            "cs_rank(atr_n(high, low, close, 14) + funding_delta(funding_rate, 1))",
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
            ("volume", "turnover"),
            ("close", "hlc3(high, low, close)"),
            ("turnover", "adv_n(turnover, 5)"),
            ("close", "ohlc4(open, high, low, close)"),
            ("volatility_n(close, 20)", "atr_n(high, low, close, 14)"),
        ]
        offset = self._stable_index(formula, len(replacements), salt=f"mutate:{variant}")
        for idx in range(len(replacements)):
            source, target = replacements[(offset + idx) % len(replacements)]
            if source in formula:
                return formula.replace(source, target, 1)
        return self._wrap_formula(formula, "mutation fallback", variant)

    def _wrap_formula(self, formula: str, objective: str, variant: int = 0) -> str:
        wrappers = [
            f"cs_rank(({formula}) + oi_delta(open_interest, 1))",
            f"cs_rank(({formula}) - funding_delta(funding_rate, 1))",
            f"cs_rank(({formula}) - spread_ratio(bid_ask_spread, close))",
            f"cs_zscore(decay_linear(({formula}), 3))",
            f"cs_rank(({formula}) - amihud(close, turnover, 5))",
            f"cs_rank(({formula}) + atr_n(high, low, close, 5))",
            f"cs_rank(fillna(({formula}), 0) + cs_demean(vwap))",
            f"cs_zscore(clip(({formula}), -3, 3) + adv_n(turnover, 10))",
            f"cs_rank(ts_zscore(({formula}), 5) + oi_delta(open_interest, 3) - funding_delta(funding_rate, 1))",
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
            f"where(spread_ratio(bid_ask_spread, close) < 0.002, ({parent_a}), ({parent_b}))",
            f"cs_rank(max(({parent_a}), ({parent_b})) - funding_delta(funding_rate, 1))",
            f"cs_rank(min(({parent_a}), ({parent_b})) + oi_delta(open_interest, 1))",
            f"cs_rank((({parent_a}) + atr_n(high, low, close, 5)) - (({parent_b}) + amihud(close, turnover, 5)))",
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


class EvolutionEngine:
    def __init__(
        self,
        llm_backend: LLMBackend | None = None,
        compiler: FormulaCompiler | None = None,
        registry: DSLRegistry | None = None,
        schema: TensorSchema | None = None,
        backend_name: str = "auto",
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.registry = registry or DSLRegistry()
        self.compiler = compiler or FormulaCompiler(self.registry)
        self.schema = schema or TensorSchema.default_market_schema()
        if llm_backend is None:
            from .llm_backend import build_default_llm_backend

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

    def generate_feedback_seed_formulas(
        self,
        entries: list[dict[str, Any]],
        count: int,
        objective: str = "improve robustness and reduce turnover",
    ) -> list[str]:
        if count <= 0:
            return []

        ranked_entries = sorted(
            [entry for entry in entries if entry.get("formula")],
            key=lambda item: float(item.get("fitness", item.get("metrics", {}).get("sharpe", 0.0)) or 0.0),
            reverse=True,
        )
        if not ranked_entries:
            return []

        parent_a = ranked_entries[0]
        parent_b = ranked_entries[1] if len(ranked_entries) > 1 else None
        parent_feedback = [
            {
                "formula": parent_a.get("formula", ""),
                "metrics": dict(parent_a.get("metrics") or {}),
                "rationale": "Carryover elite from previous evaluated cycle.",
            }
        ]
        if parent_b:
            parent_feedback.append(
                {
                    "formula": parent_b.get("formula", ""),
                    "metrics": dict(parent_b.get("metrics") or {}),
                    "rationale": "Secondary carryover elite for crossover diversity.",
                }
            )

        generated = self.llm_backend.generate_offspring(
            BreedingSpec(
                parent_a=parent_a.get("formula", ""),
                parent_b=parent_b.get("formula", "") if parent_b else None,
                objective=objective,
                parent_feedback=parent_feedback,
            ),
            count=count,
        )

        validated: list[str] = []
        seen: set[str] = set()
        for formula in generated:
            if not formula or formula in seen:
                continue
            if self._build_individual(formula, {"origin": "feedback_seed"}) is None:
                continue
            seen.add(formula)
            validated.append(formula)
            if len(validated) >= count:
                break
        return validated

    def initialize(self, seeds: list[str], population_size: int) -> list[Individual]:
        population: list[Individual] = []
        resolved_seeds = list(seeds)
        if not resolved_seeds:
            resolved_seeds = self.llm_backend.generate_initial_population(population_size)
        for formula in resolved_seeds[:population_size]:
            individual = self._build_individual(formula, {"origin": "seed"})
            if individual:
                population.append(individual)
        while len(population) < population_size and resolved_seeds:
            random_seed = random.choice(resolved_seeds)
            generated = self.llm_backend.generate_offspring(
                BreedingSpec(parent_a=random_seed, parent_b=None, objective="bootstrap"),
                count=1,
            )
            individual = self._build_individual(generated[0], {"origin": "bootstrap"})
            if individual:
                population.append(individual)
        return population

    def select_survivors(self, pop: list[Individual], top_k: int | None = None) -> list[Individual]:
        top_k = top_k or max(1, len(pop) // 2)
        ranked = sorted(pop, key=lambda item: item.fitness, reverse=True)
        return ranked[:top_k]

    def breed(self, survivors: list[Individual], n_offspring: int) -> list[Individual]:
        if not survivors or n_offspring <= 0:
            return []
        offspring: list[Individual] = []
        jobs = self._build_breeding_jobs(survivors, n_offspring)
        for parent_a, parent_b, requested_count in jobs:
            formulas = self.llm_backend.generate_offspring(
                BreedingSpec(
                    parent_a=parent_a.formula,
                    parent_b=parent_b.formula if parent_b else None,
                    objective="improve robustness and reduce turnover",
                    parent_feedback=[
                        {
                            "formula": parent_a.formula,
                            "metrics": parent_a.metrics,
                            "rationale": "Primary parent selected from previous generation elites.",
                        },
                        *(
                            [
                                {
                                    "formula": parent_b.formula,
                                    "metrics": parent_b.metrics,
                                    "rationale": "Secondary parent selected for crossover diversity.",
                                }
                            ]
                            if parent_b
                            else []
                        ),
                    ],
                ),
                count=requested_count,
            )
            lineage = {
                "parent_a": parent_a.expr_hash,
                "parent_b": parent_b.expr_hash if parent_b else None,
            }
            for formula in formulas:
                child = self._build_individual(formula, lineage)
                if child:
                    offspring.append(child)
                if len(offspring) >= n_offspring:
                    return offspring[:n_offspring]

        if len(offspring) >= n_offspring:
            return offspring[:n_offspring]

        for idx in range(n_offspring - len(offspring)):
            parent_a = survivors[idx % len(survivors)]
            parent_b = survivors[(idx + 1) % len(survivors)] if len(survivors) > 1 else None
            formulas = self.llm_backend.generate_offspring(
                BreedingSpec(
                    parent_a=parent_a.formula,
                    parent_b=parent_b.formula if parent_b else None,
                    objective="improve robustness and reduce turnover",
                    parent_feedback=[
                        {
                            "formula": parent_a.formula,
                            "metrics": parent_a.metrics,
                            "rationale": "Fallback primary parent selected from previous generation elites.",
                        },
                        *(
                            [
                                {
                                    "formula": parent_b.formula,
                                    "metrics": parent_b.metrics,
                                    "rationale": "Fallback secondary parent selected for crossover diversity.",
                                }
                            ]
                            if parent_b
                            else []
                        ),
                    ],
                ),
                count=1,
            )
            if not formulas:
                continue
            child = self._build_individual(
                formulas[0],
                {
                    "parent_a": parent_a.expr_hash,
                    "parent_b": parent_b.expr_hash if parent_b else None,
                },
            )
            if child:
                offspring.append(child)
        return offspring

    def _build_breeding_jobs(
        self,
        survivors: list[Individual],
        n_offspring: int,
    ) -> list[tuple[Individual, Individual | None, int]]:
        if not survivors or n_offspring <= 0:
            return []
        pair_count = min(len(survivors), n_offspring)
        jobs: list[tuple[Individual, Individual | None, int]] = []
        base_count = n_offspring // pair_count
        remainder = n_offspring % pair_count
        for idx in range(pair_count):
            parent_a = survivors[idx % len(survivors)]
            parent_b = survivors[(idx + 1) % len(survivors)] if len(survivors) > 1 else None
            requested_count = base_count + (1 if idx < remainder else 0)
            if requested_count <= 0:
                continue
            jobs.append((parent_a, parent_b, requested_count))
        return jobs

    def attach_metrics(self, pop: list[Individual], metrics_by_hash: dict[str, dict[str, float]]) -> list[Individual]:
        for individual in pop:
            metrics = metrics_by_hash.get(individual.expr_hash, {})
            individual.metrics = metrics
            individual.fitness = self.fitness_engine.score(metrics)
        return pop

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
