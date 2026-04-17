from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.compiler import BytecodeProgram
from .pipeline import Lineage, PipelineRecord


@dataclass
class Individual:
    formula: str
    program: BytecodeProgram
    expr_hash: str
    lineage: Lineage = field(default_factory=lambda: Lineage(origin="unknown"))
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

    def relaxed(self) -> "FitnessPolicy":
        """Return a lenient variant for warm-start rounds.

        Relaxes rejection thresholds so that mediocre-but-non-degenerate
        formulas can enter the archive, giving MCTS and evolution useful
        seeds to refine.
        """
        return FitnessPolicy(
            min_active_bar_ratio=0.02,
            min_signal_coverage=0.15,
            min_avg_turnover=0.001,
            min_test_sharpe=-0.5,
            max_negative_test_ratio=0.80,
            reject_score=-8.0,  # much lower: allow up to 12 violations before reject
            sharpe_scale=self.sharpe_scale,
            test_sharpe_scale=self.test_sharpe_scale,
            rank_ic_scale=self.rank_ic_scale,
            pnl_efficiency_cap=self.pnl_efficiency_cap,
            tail_ratio_scale=self.tail_ratio_scale,
        )


class FitnessEngine:
    def __init__(self, policy: FitnessPolicy | None = None):
        self.policy = policy or FitnessPolicy()
        self._normal_policy = self.policy
        self._warm_policy = self.policy.relaxed()
        self._warm = False

    def set_warm(self, enabled: bool) -> None:
        """Switch between normal and warm-start (relaxed) policy."""
        self._warm = enabled
        self.policy = self._warm_policy if enabled else self._normal_policy

    def _clip(self, value: float, lo: float, hi: float) -> float:
        return float(np.clip(value, lo, hi))

    def _safe(self, metrics: dict[str, float], key: str, default: float = 0.0) -> float:
        try:
            return float(metrics.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def rejection_reasons(self, metrics: dict[str, float]) -> list[str]:
        """Return human-readable list of rejection reasons (empty if none)."""
        reasons: list[str] = []
        active_bar_ratio = self._safe(metrics, "active_bar_ratio")
        signal_coverage = self._safe(metrics, "signal_coverage", 1.0)
        avg_turnover = self._safe(metrics, "avg_turnover")
        test_sharpe = self._safe(metrics, "test_sharpe")
        negative_test_ratio = self._safe(metrics, "negative_test_ratio")
        inactive = self._safe(metrics, "inactive")

        if inactive >= 1.0 or active_bar_ratio < self.policy.min_active_bar_ratio:
            reasons.append(
                f"active_bar_ratio={active_bar_ratio:.3f}<{self.policy.min_active_bar_ratio}"
                if active_bar_ratio < self.policy.min_active_bar_ratio
                else f"inactive={inactive:.2f}>=1.0"
            )
        if signal_coverage < self.policy.min_signal_coverage:
            reasons.append(f"signal_coverage={signal_coverage:.3f}<{self.policy.min_signal_coverage}")
        if avg_turnover < self.policy.min_avg_turnover:
            reasons.append(f"avg_turnover={avg_turnover:.4f}<{self.policy.min_avg_turnover}")
        if test_sharpe < self.policy.min_test_sharpe:
            reasons.append(f"test_sharpe={test_sharpe:.3f}<{self.policy.min_test_sharpe}")
        if negative_test_ratio > self.policy.max_negative_test_ratio:
            reasons.append(f"neg_test_ratio={negative_test_ratio:.2f}>{self.policy.max_negative_test_ratio}")
        return reasons

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
    pipeline: PipelineRecord | None = None


