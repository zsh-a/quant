from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from loguru import logger

from .compiler import FormulaCompiler
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
    """Final output of SearchOrchestrator.run()."""

    archive: list[Individual]
    all_evaluated: list[Individual]
    details_by_hash: dict[str, dict[str, Any]]
    rounds: list[dict[str, Any]]
    timing: dict[str, Any]
    total_evaluations: int
    total_rejected: int


