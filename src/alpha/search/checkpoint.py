"""
Checkpoint manager for alpha search sessions.

Saves and restores full search state at round boundaries, including:
  - Strategy snapshots (model weights, MCTS zoo, etc.)
  - Factor catalog (all generated factors with metrics)
  - Archive state and context counters
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from .context import FactorCatalog, StatefulStrategy, StrategySnapshot

# ---------------------------------------------------------------------------
# Search checkpoint (on-disk representation)
# ---------------------------------------------------------------------------


@dataclass
class SearchCheckpoint:
    """Full snapshot of a search session at a round boundary."""

    job_id: str
    round_idx: int
    timestamp: float
    strategy_snapshots: dict[str, StrategySnapshot]
    factor_catalog_json: list[dict[str, Any]]
    archive_formulas: list[dict[str, Any]]
    context_state: dict[str, Any]

    def save(self, directory: Path) -> Path:
        """Write checkpoint to disk. Returns the checkpoint directory."""
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = directory / f"checkpoint_r{self.round_idx:04d}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Strategy snapshots — one file per strategy
        for name, snap in self.strategy_snapshots.items():
            safe_name = name.replace("/", "_")
            snap_path = checkpoint_dir / f"strategy_{safe_name}.{snap.format}"
            snap_path.write_bytes(snap.data)
            meta_path = checkpoint_dir / f"strategy_{safe_name}.meta.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "strategy_name": snap.strategy_name,
                        "round_idx": snap.round_idx,
                        "format": snap.format,
                        "metadata": snap.metadata,
                    },
                    indent=2,
                )
            )

        # Factor catalog
        (checkpoint_dir / "factor_catalog.json").write_text(json.dumps(self.factor_catalog_json, ensure_ascii=False))

        # Manifest
        (checkpoint_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "job_id": self.job_id,
                    "round_idx": self.round_idx,
                    "timestamp": self.timestamp,
                    "context_state": self.context_state,
                    "archive_formulas": self.archive_formulas,
                    "strategy_names": list(self.strategy_snapshots.keys()),
                },
                indent=2,
                default=str,
            )
        )

        return checkpoint_dir

    @classmethod
    def load(cls, checkpoint_dir: Path) -> SearchCheckpoint:
        """Read checkpoint from disk."""
        manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
        catalog_json = json.loads((checkpoint_dir / "factor_catalog.json").read_text())

        snapshots: dict[str, StrategySnapshot] = {}
        for name in manifest.get("strategy_names", []):
            safe_name = name.replace("/", "_")
            meta_path = checkpoint_dir / f"strategy_{safe_name}.meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            data_path = checkpoint_dir / f"strategy_{safe_name}.{meta['format']}"
            if not data_path.exists():
                continue
            snapshots[name] = StrategySnapshot(
                strategy_name=meta["strategy_name"],
                round_idx=meta["round_idx"],
                format=meta["format"],
                data=data_path.read_bytes(),
                metadata=meta.get("metadata", {}),
            )

        return cls(
            job_id=manifest["job_id"],
            round_idx=manifest["round_idx"],
            timestamp=manifest["timestamp"],
            strategy_snapshots=snapshots,
            factor_catalog_json=catalog_json,
            archive_formulas=manifest.get("archive_formulas", []),
            context_state=manifest.get("context_state", {}),
        )


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Manages checkpoint creation and restoration for search sessions.

    Used by the ``SearchOrchestrator`` — strategies are unaware of this class.
    The orchestrator calls ``should_checkpoint()`` after each round and
    ``create_checkpoint()`` / ``restore_strategies()`` as needed.
    """

    def __init__(
        self,
        checkpoint_dir: str = "",
        checkpoint_every_n_rounds: int = 5,
        keep_latest: int = 3,
    ) -> None:
        if not checkpoint_dir:
            from src.config.paths import ALPHA_LAB_CHECKPOINTS_DIR

            checkpoint_dir = str(ALPHA_LAB_CHECKPOINTS_DIR)
        self._dir = Path(checkpoint_dir)
        self._every_n = checkpoint_every_n_rounds
        self._keep_latest = keep_latest

    def should_checkpoint(self, round_idx: int, total_rounds: int) -> bool:
        """Whether to create a checkpoint after this round."""
        if round_idx == total_rounds - 1:
            return True  # Always on last round
        return (round_idx + 1) % self._every_n == 0

    def create_checkpoint(
        self,
        job_id: str,
        round_idx: int,
        strategies: list[Any],
        factor_catalog: FactorCatalog,
        ctx_state: dict[str, Any],
        archive_snapshot: list[dict[str, Any]],
    ) -> Path | None:
        """Snapshot all stateful strategies + factor catalog to disk."""
        snapshots: dict[str, StrategySnapshot] = {}
        for strategy in strategies:
            if isinstance(strategy, StatefulStrategy):
                try:
                    snap = strategy.save_state()
                    snap.round_idx = round_idx
                    snapshots[strategy.name] = snap
                except Exception as e:
                    logger.warning(
                        "checkpoint.save_state failed for {}: {}",
                        strategy.name,
                        e,
                    )

        checkpoint = SearchCheckpoint(
            job_id=job_id,
            round_idx=round_idx,
            timestamp=time.time(),
            strategy_snapshots=snapshots,
            factor_catalog_json=factor_catalog.to_json_list(),
            archive_formulas=archive_snapshot,
            context_state=ctx_state,
        )
        path = checkpoint.save(self._dir / job_id)
        logger.info("checkpoint.saved round={} path={}", round_idx, path)
        self._prune(job_id)
        return path

    def restore_strategies(
        self,
        checkpoint_dir: Path,
        strategies: list[Any],
    ) -> tuple[FactorCatalog, dict[str, Any]]:
        """Restore strategies from checkpoint.

        Returns ``(factor_catalog, context_state)`` so the orchestrator can
        resume from the checkpointed round.
        """
        checkpoint = SearchCheckpoint.load(checkpoint_dir)
        for strategy in strategies:
            if isinstance(strategy, StatefulStrategy):
                snap = checkpoint.strategy_snapshots.get(strategy.name)
                if snap is not None:
                    try:
                        strategy.load_state(snap)
                        logger.info(
                            "checkpoint.restored strategy={}",
                            strategy.name,
                        )
                    except Exception as e:
                        logger.warning(
                            "checkpoint.load_state failed for {}: {}",
                            strategy.name,
                            e,
                        )
        catalog = FactorCatalog.from_json_list(checkpoint.factor_catalog_json)
        return catalog, checkpoint.context_state

    def latest_checkpoint(self, job_id: str) -> Path | None:
        """Find the most recent checkpoint for a job."""
        job_dir = self._dir / job_id
        if not job_dir.exists():
            return None
        checkpoints = sorted(job_dir.glob("checkpoint_r*"), reverse=True)
        return checkpoints[0] if checkpoints else None

    def _prune(self, job_id: str) -> None:
        """Keep only the latest N checkpoints per job."""
        job_dir = self._dir / job_id
        if not job_dir.exists():
            return
        checkpoints = sorted(job_dir.glob("checkpoint_r*"), reverse=True)
        for old in checkpoints[self._keep_latest :]:
            shutil.rmtree(old, ignore_errors=True)
