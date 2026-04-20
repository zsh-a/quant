"""Cross-cycle persistent search state.

Keeps a slim summary of what the auto-runner has already seen so the
next cycle can skip previously-evaluated formulas and warm-start from
the best archive entries instead of rebuilding from scratch.

Stored under ``data/alpha/state/{market}/state.json``. The format has
an explicit ``schema_version`` so we can evolve it without silently
breaking old files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

PERSISTENT_STATE_SCHEMA_VERSION = 1


@dataclass
class ArchiveSummary:
    formula: str
    expr_hash: str
    fitness: float
    metrics: dict[str, Any] = field(default_factory=dict)
    strategy: str = ""
    round_idx: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchiveSummary":
        return cls(
            formula=str(data.get("formula", "")),
            expr_hash=str(data.get("expr_hash", "")),
            fitness=float(data.get("fitness", 0.0) or 0.0),
            metrics=dict(data.get("metrics", {}) or {}),
            strategy=str(data.get("strategy", "") or ""),
            round_idx=int(data.get("round_idx", 0) or 0),
        )


@dataclass
class PersistentSearchState:
    """Slim, human-readable summary of search progress across cycles.

    Not a full replica of ``SearchContext`` — just enough to seed the
    next cycle and prevent repeated work.
    """

    schema_version: int = PERSISTENT_STATE_SCHEMA_VERSION
    market: str = ""
    seen_hashes: list[str] = field(default_factory=list)
    top_archive: list[ArchiveSummary] = field(default_factory=list)
    theme_stats: dict[str, Any] = field(default_factory=dict)
    last_cycle_feedback: str = ""
    last_cycle_window: dict[str, str] = field(default_factory=dict)
    last_cycle_at: str = ""
    last_cycle_run_id: str = ""
    total_cycles: int = 0

    def seen_set(self) -> set[str]:
        return set(self.seen_hashes)

    def merge_cycle_result(
        self,
        *,
        new_hashes: set[str],
        new_archive: list[dict[str, Any]],
        window: dict[str, str] | None = None,
        run_id: str = "",
        feedback: str = "",
        theme_stats: dict[str, Any] | None = None,
        hash_cap: int = 20000,
        archive_cap: int = 50,
    ) -> "PersistentSearchState":
        """Fold new cycle output into the persistent state in-place."""
        # Bounded FIFO for seen hashes so the file doesn't grow without limit
        merged_hashes = list(self.seen_hashes)
        existing = set(merged_hashes)
        for h in new_hashes:
            if h and h not in existing:
                merged_hashes.append(h)
                existing.add(h)
        if len(merged_hashes) > hash_cap:
            merged_hashes = merged_hashes[-hash_cap:]
        self.seen_hashes = merged_hashes

        # Merge top archive — keep best unique entries by expr_hash
        by_hash: dict[str, ArchiveSummary] = {a.expr_hash: a for a in self.top_archive}
        for entry in new_archive:
            summary = ArchiveSummary.from_dict(entry)
            prev = by_hash.get(summary.expr_hash)
            if prev is None or summary.fitness > prev.fitness:
                by_hash[summary.expr_hash] = summary
        top = sorted(by_hash.values(), key=lambda s: s.fitness, reverse=True)[:archive_cap]
        self.top_archive = top

        if window:
            self.last_cycle_window = dict(window)
        if run_id:
            self.last_cycle_run_id = run_id
        if feedback:
            self.last_cycle_feedback = feedback
        if theme_stats:
            self.theme_stats = theme_stats
        self.last_cycle_at = datetime.now(UTC).isoformat()
        self.total_cycles += 1
        return self

    # ---------------- serialization ----------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["top_archive"] = [asdict(a) for a in self.top_archive]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersistentSearchState":
        schema = int(data.get("schema_version", 0))
        if schema > PERSISTENT_STATE_SCHEMA_VERSION:
            logger.warning(
                "persistent_state schema_version={} newer than supported {}; ignoring extra fields",
                schema,
                PERSISTENT_STATE_SCHEMA_VERSION,
            )
        top = [ArchiveSummary.from_dict(a) for a in (data.get("top_archive") or [])]
        return cls(
            schema_version=PERSISTENT_STATE_SCHEMA_VERSION,
            market=str(data.get("market", "") or ""),
            seen_hashes=list(data.get("seen_hashes", []) or []),
            top_archive=top,
            theme_stats=dict(data.get("theme_stats", {}) or {}),
            last_cycle_feedback=str(data.get("last_cycle_feedback", "") or ""),
            last_cycle_window=dict(data.get("last_cycle_window", {}) or {}),
            last_cycle_at=str(data.get("last_cycle_at", "") or ""),
            last_cycle_run_id=str(data.get("last_cycle_run_id", "") or ""),
            total_cycles=int(data.get("total_cycles", 0) or 0),
        )


def _state_path(market: str) -> Path:
    from src.config.paths import ALPHA_STATE_DIR

    base = ALPHA_STATE_DIR / (market or "default")
    base.mkdir(parents=True, exist_ok=True)
    return base / "state.json"


def load_persistent_state(market: str) -> PersistentSearchState:
    path = _state_path(market)
    if not path.exists():
        return PersistentSearchState(market=market)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("persistent_state load failed path={} exc={}", path, exc)
        return PersistentSearchState(market=market)
    state = PersistentSearchState.from_dict(data)
    state.market = state.market or market
    return state


def save_persistent_state(state: PersistentSearchState) -> Path:
    path = _state_path(state.market)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path
