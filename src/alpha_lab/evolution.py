from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol

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


class LLMBackend(Protocol):
    def generate_offspring(self, spec: BreedingSpec, count: int) -> list[str]:
        ...


class HeuristicLLMBackend:
    """
    Local placeholder backend.

    This keeps the interface stable while the real remote/local LLM backend is
    integrated later.
    """

    def generate_offspring(self, spec: BreedingSpec, count: int) -> list[str]:
        offspring = []
        bases = [spec.parent_a]
        if spec.parent_b:
            bases.append(spec.parent_b)
        for idx in range(count):
            base = bases[idx % len(bases)]
            offspring.append(self._mutate_formula(base))
        return offspring

    def _mutate_formula(self, formula: str) -> str:
        replacements = [
            ("ts_mean(", "ts_std("),
            ("ts_std(", "ts_mean("),
            ("ts_max(", "ts_rank("),
            ("ts_rank(", "ts_mean("),
            ("close", "vwap"),
            ("volume", "turnover"),
        ]
        for source, target in replacements:
            if source in formula:
                return formula.replace(source, target, 1)
        return f"cs_rank({formula})"


class FitnessEngine:
    def score(self, metrics: dict[str, float]) -> float:
        sharpe = metrics.get("sharpe", 0.0)
        pnl_per_turnover = metrics.get("pnl_per_turnover", 0.0)
        rank_ic = metrics.get("rank_ic", 0.0)
        stability = metrics.get("stability", 0.0)
        tail = metrics.get("tail_penalty_adjusted_return", 0.0)
        turnover_penalty = metrics.get("turnover_penalty", 0.0)
        complexity_penalty = metrics.get("complexity_penalty", 0.0)
        gap_penalty = metrics.get("train_valid_gap_penalty", 0.0)

        return (
            0.30 * sharpe
            + 0.20 * pnl_per_turnover
            + 0.15 * rank_ic
            + 0.10 * stability
            + 0.10 * tail
            - 0.10 * turnover_penalty
            - 0.10 * complexity_penalty
            - 0.15 * gap_penalty
        )


class EvolutionEngine:
    def __init__(
        self,
        llm_backend: LLMBackend | None = None,
        compiler: FormulaCompiler | None = None,
        registry: DSLRegistry | None = None,
        schema: TensorSchema | None = None,
    ):
        self.registry = registry or DSLRegistry()
        self.compiler = compiler or FormulaCompiler(self.registry)
        self.schema = schema or TensorSchema.default_market_schema()
        self.llm_backend = llm_backend or HeuristicLLMBackend()
        self.fitness_engine = FitnessEngine()

    def initialize(self, seeds: list[str], population_size: int) -> list[Individual]:
        population: list[Individual] = []
        for formula in seeds[:population_size]:
            individual = self._build_individual(formula, {"origin": "seed"})
            if individual:
                population.append(individual)
        while len(population) < population_size and seeds:
            random_seed = random.choice(seeds)
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
        if not survivors:
            return []
        offspring: list[Individual] = []
        for idx in range(n_offspring):
            parent_a = survivors[idx % len(survivors)]
            parent_b = survivors[(idx + 1) % len(survivors)] if len(survivors) > 1 else None
            formulas = self.llm_backend.generate_offspring(
                BreedingSpec(
                    parent_a=parent_a.formula,
                    parent_b=parent_b.formula if parent_b else None,
                    objective="improve robustness and reduce turnover",
                ),
                count=1,
            )
            lineage = {
                "parent_a": parent_a.expr_hash,
                "parent_b": parent_b.expr_hash if parent_b else None,
            }
            child = self._build_individual(formulas[0], lineage)
            if child:
                offspring.append(child)
        return offspring

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
