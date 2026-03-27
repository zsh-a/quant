from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .compiler import FormulaCompiler
from .dsl import DSLRegistry, TensorSchema
from .evolution import EvolutionEngine
from .risk import CostModel, ExecutionSimulator, MarketContext, RuleOverlay, SignalTransformer
from .vm import StackVM, TensorStore


class AlphaLabService:
    def __init__(self, schema: TensorSchema | None = None):
        self.registry = DSLRegistry()
        self.schema = schema or TensorSchema.default_market_schema()
        self.compiler = FormulaCompiler(self.registry)
        self.vm = StackVM()
        self.evolution = EvolutionEngine(registry=self.registry, compiler=self.compiler, schema=self.schema)
        self.signal_transformer = SignalTransformer()
        self.rule_overlay = RuleOverlay()
        self.execution = ExecutionSimulator()

    def list_operators(self) -> list[dict[str, Any]]:
        return [asdict(spec) for spec in self.registry.list_operators()]

    def validate_formula(self, formula: str) -> dict[str, Any]:
        report = self.registry.validate_formula(formula, self.schema)
        return asdict(report)

    def compile_formula(self, formula: str) -> dict[str, Any]:
        program = self.compiler.compile(formula, self.schema)
        report = self.registry.validate_formula(formula, self.schema)
        return {
            "validation": asdict(report),
            "program": program.to_dict(),
        }

    def evaluate_formula(
        self,
        formula: str,
        fields: dict[str, list[list[float]]],
        liquidity_mask: list[list[bool]] | None = None,
        session_mask: list[list[bool]] | None = None,
    ) -> dict[str, Any]:
        store = TensorStore({name: np.asarray(values, dtype=float) for name, values in fields.items()})
        program = self.compiler.compile(formula, self.schema)
        alpha = np.asarray(self.vm.run(program, store), dtype=float)

        market_ctx = MarketContext(
            liquidity_mask=np.asarray(liquidity_mask, dtype=bool) if liquidity_mask is not None else None,
            session_mask=np.asarray(session_mask, dtype=bool) if session_mask is not None else None,
        )
        target_weights = self.signal_transformer.to_target_weights(alpha, market_ctx)
        wrapped_weights = self.rule_overlay.apply(target_weights, market_ctx)
        result = self.execution.simulate(wrapped_weights, {"close": store.get_field("close")}, CostModel())
        return {
            "program": program.to_dict(),
            "alpha": alpha.tolist(),
            "weights": wrapped_weights.tolist(),
            "metrics": result.summary(),
        }

    def seed_population(self, seeds: list[str], population_size: int = 8) -> list[dict[str, Any]]:
        population = self.evolution.initialize(seeds, population_size)
        return [
            {
                "formula": individual.formula,
                "expr_hash": individual.expr_hash,
                "lineage": individual.lineage,
            }
            for individual in population
        ]

    def breed_population(self, formulas: list[str], offspring_count: int = 4) -> list[dict[str, Any]]:
        survivors = self.evolution.initialize(formulas, len(formulas))
        offspring = self.evolution.breed(survivors, offspring_count)
        return [
            {
                "formula": individual.formula,
                "expr_hash": individual.expr_hash,
                "lineage": individual.lineage,
            }
            for individual in offspring
        ]
