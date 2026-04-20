"""
Structured pipeline data models for alpha factor discovery.

Provides typed, serializable records for each stage of the search pipeline,
enabling clear observability, SSE streaming, and LLM-ready summaries.

Pipeline stages:
  Generate → QuickScreen → Evaluate → Fitness → Archive
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Stage model
# ---------------------------------------------------------------------------


class StageKind(str, Enum):
    GENERATE = "generate"
    QUICK_SCREEN = "quick_screen"
    EVALUATE = "evaluate"
    FITNESS = "fitness"
    ARCHIVE = "archive"


@dataclass
class StageRecord:
    """One execution of a pipeline stage within a round."""

    kind: StageKind
    strategy: str
    round_idx: int
    input_count: int
    output_count: int
    duration_ms: float = 0.0
    best_fitness: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": self.kind.value,
            "strategy": self.strategy,
            "round": self.round_idx,
            "input": self.input_count,
            "output": self.output_count,
            "duration_ms": round(self.duration_ms, 1),
        }
        if self.best_fitness is not None:
            d["best_fitness"] = round(self.best_fitness, 4)
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# ---------------------------------------------------------------------------
# Archive entry (lightweight snapshot for SSE / LLM context)
# ---------------------------------------------------------------------------


@dataclass
class ArchiveEntry:
    """Snapshot of one archive member — used for SSE push and LLM context."""

    formula: str
    expr_hash: str
    fitness: float
    rank_ic: float
    sharpe: float
    turnover: float
    origin: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula": self.formula,
            "expr_hash": self.expr_hash,
            "fitness": round(self.fitness, 4),
            "rank_ic": round(self.rank_ic, 5),
            "sharpe": round(self.sharpe, 4),
            "turnover": round(self.turnover, 5),
            "origin": self.origin,
        }


# ---------------------------------------------------------------------------
# Round record
# ---------------------------------------------------------------------------


@dataclass
class RoundRecord:
    """All stages executed in one search round."""

    round_idx: int
    strategies_activated: list[str] = field(default_factory=list)
    stages: list[StageRecord] = field(default_factory=list)
    archive_snapshot: list[ArchiveEntry] = field(default_factory=list)
    archive_size: int = 0
    population_size: int = 0
    best_fitness: float = 0.0
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_idx,
            "strategies": self.strategies_activated,
            "stages": [s.to_dict() for s in self.stages],
            "archive_snapshot": [a.to_dict() for a in self.archive_snapshot],
            "archive_size": self.archive_size,
            "population_size": self.population_size,
            "best_fitness": round(self.best_fitness, 4),
            "duration_ms": round(self.duration_ms, 1),
        }


# ---------------------------------------------------------------------------
# Pipeline record (full search execution log)
# ---------------------------------------------------------------------------


@dataclass
class PipelineRecord:
    """Complete execution log for one search job."""

    job_id: str
    rounds: list[RoundRecord] = field(default_factory=list)
    total_evaluations: int = 0
    total_rejected: int = 0
    aborted: bool = False
    budget_exhausted: bool = False
    budget_snapshot: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "rounds": [r.to_dict() for r in self.rounds],
            "total_evaluations": self.total_evaluations,
            "total_rejected": self.total_rejected,
            "aborted": self.aborted,
            "budget_exhausted": self.budget_exhausted,
            "budget_snapshot": self.budget_snapshot,
        }


# ---------------------------------------------------------------------------
# Lineage (structured replacement for dict[str, Any])
# ---------------------------------------------------------------------------


@dataclass
class Lineage:
    """Structured lineage for an Individual.

    Replaces the previous ``dict[str, Any]`` with a typed dataclass that
    all strategies populate consistently.
    """

    origin: str  # "seed" | "bootstrap" | "enumeration" | "llm_genesis" | "llm_evolution" | "mcts_refinement" | "neural_formula"
    parent_a: str | None = None  # expr_hash of first parent
    parent_b: str | None = None  # expr_hash of second parent
    theme: str | None = None  # financial theme id
    screen_ic: float | None = None  # IC from fast screen (enumeration / neural)
    round_idx: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"origin": self.origin}
        if self.parent_a is not None:
            d["parent_a"] = self.parent_a
        if self.parent_b is not None:
            d["parent_b"] = self.parent_b
        if self.theme is not None:
            d["theme"] = self.theme
        if self.screen_ic is not None:
            d["screen_ic"] = self.screen_ic
        if self.round_idx is not None:
            d["round_idx"] = self.round_idx
        if self.extra:
            d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Lineage:
        """Backward-compatible construction from a plain dict."""
        known = {"origin", "parent_a", "parent_b", "theme", "screen_ic", "round_idx"}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(
            origin=d.get("origin", "unknown"),
            parent_a=d.get("parent_a"),
            parent_b=d.get("parent_b"),
            theme=d.get("theme"),
            screen_ic=d.get("screen_ic"),
            round_idx=d.get("round_idx"),
            extra=extra,
        )
