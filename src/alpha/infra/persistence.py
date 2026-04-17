from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger


@dataclass
class PersistedRun:
    run_id: str
    run_path: str
    zoo_dir: str


class AlphaPersistence:
    def __init__(self, root_dir: str = ""):
        from src.config.paths import ALPHA_DIR

        self.root_dir = Path(root_dir) if root_dir else ALPHA_DIR
        self.runs_dir = self.root_dir / "runs"
        self.zoo_dir = self.root_dir / "zoo"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.zoo_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Run management
    # -----------------------------------------------------------------------

    def save_run(self, payload: dict[str, Any], run_name: str | None = None) -> PersistedRun:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        slug = run_name or "alpha_lab"
        run_id = f"{slug}_{timestamp}"
        run_path = self.runs_dir / f"{run_id}.json"
        run_payload = dict(payload)
        run_payload["run_id"] = run_id
        run_payload["saved_at"] = datetime.now(UTC).isoformat()
        run_path.write_text(
            json.dumps(run_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return PersistedRun(run_id=run_id, run_path=str(run_path), zoo_dir=str(self.zoo_dir))

    def update_run(self, run_id: str, payload: dict[str, Any]) -> None:
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        saved_at = None
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            saved_at = existing.get("saved_at")
        except Exception:
            saved_at = None
        run_payload = dict(payload)
        run_payload["run_id"] = run_id
        run_payload["saved_at"] = saved_at or datetime.now(UTC).isoformat()
        path.write_text(
            json.dumps(run_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        runs = []
        for path in sorted(self.runs_dir.glob("*.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            top = payload.get("top_results", [])
            best = max(top, key=lambda x: float(x.get("fitness", 0)), default=None) if top else None
            search_stats = payload.get("search_stats", {})
            timing = payload.get("timing", {})
            runs.append(
                {
                    "run_id": payload.get("run_id"),
                    "saved_at": payload.get("saved_at"),
                    "path": str(path),
                    "dataset": payload.get("dataset", {}),
                    "top_results": len(top),
                    "search_stats": search_stats,
                    "timing_seconds": timing.get("overall_seconds"),
                    "best_fitness": float(best["fitness"]) if best else None,
                    "best_sharpe": float(best.get("metrics", {}).get("sharpe", 0)) if best else None,
                    "best_ic": float(best.get("metrics", {}).get("rank_ic", 0)) if best else None,
                }
            )
            if len(runs) >= limit:
                break
        return runs

    def load_run(self, run_id: str) -> dict[str, Any]:
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def save_zoo_entries(self, entries: list[dict[str, Any]], run_id: str) -> list[str]:
        paths: list[str] = []
        for entry in entries:
            formula = entry.get("formula", "")
            expr_hash = entry.get("expr_hash") or hashlib.sha256(formula.encode("utf-8")).hexdigest()
            payload = dict(entry)
            payload["saved_at"] = datetime.now(UTC).isoformat()
            payload["run_id"] = run_id
            path = self.zoo_dir / f"alpha_{expr_hash[:16]}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            paths.append(str(path))
        return paths

    def list_zoo_entries(self, limit: int = 50) -> list[dict[str, Any]]:
        entries = []
        for path in sorted(self.zoo_dir.glob("alpha_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            payload["path"] = str(path)
            entries.append(payload)
        entries.sort(key=lambda item: float(item.get("fitness", 0.0)), reverse=True)
        return entries[:limit]

    def prune_runs(self, keep_latest: int) -> dict[str, Any]:
        paths = sorted(self.runs_dir.glob("*.json"), reverse=True)
        removed: list[str] = []
        for path in paths[max(keep_latest, 0) :]:
            path.unlink(missing_ok=True)
            removed.append(str(path))
        return {
            "kept": min(len(paths), max(keep_latest, 0)),
            "removed": len(removed),
            "removed_paths": removed,
        }

    def prune_zoo_entries(self, keep_top: int) -> dict[str, Any]:
        entries: list[tuple[float, Path]] = []
        for path in self.zoo_dir.glob("alpha_*.json"):
            fitness = 0.0
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                fitness = float(payload.get("fitness", 0.0))
            except Exception:
                fitness = float("-inf")
            entries.append((fitness, path))

        entries.sort(key=lambda item: item[0], reverse=True)
        removed: list[str] = []
        for _, path in entries[max(keep_top, 0) :]:
            path.unlink(missing_ok=True)
            removed.append(str(path))
        return {
            "kept": min(len(entries), max(keep_top, 0)),
            "removed": len(removed),
            "removed_paths": removed,
        }

    # -----------------------------------------------------------------------
    # Node / Zoo operations (from AlphaZooPersistence)
    # -----------------------------------------------------------------------

    def save_node(self, node: Any, metadata: Dict[str, Any] = None) -> None:
        """
        Save an AlphaNode to a JSON file.
        """
        formula = node.formula
        # Generate a unique ID based on the formula
        formula_id = hashlib.md5(formula.encode()).hexdigest()[:12]

        data = {
            "id": formula_id,
            "formula": formula,
            "metrics": node.metrics,
            "timestamp": datetime.now().isoformat(),
            "name": getattr(node, "name", "unknown"),
            "description": getattr(node, "description", ""),
            "metadata": metadata or {},
        }

        file_path = self.zoo_dir / f"alpha_{formula_id}.json"

        try:
            file_path.write_text(
                json.dumps(data, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug(f"Saved alpha factor {formula_id} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save alpha {formula_id}: {e}")

    def save_zoo(self, zoo: List[Any], task_name: str = "default") -> None:
        """
        Save the entire Alpha Zoo.
        """
        logger.info(f"Saving {len(zoo)} factors to Zoo...")
        for node in zoo:
            self.save_node(node, {"task": task_name})

    def load_all(self) -> List[Dict[str, Any]]:
        """
        Load all discovered alphas from the storage directory.
        """
        alphas = []
        zoo_str = str(self.zoo_dir)
        for filename in os.listdir(zoo_str):
            if filename.endswith(".json") and filename.startswith("alpha_"):
                path = os.path.join(zoo_str, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        alphas.append(json.load(f))
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

        # Sort by RankIC by default
        alphas.sort(key=lambda x: abs(x.get("metrics", {}).get("rank_ic", 0)), reverse=True)
        return alphas

    def export_to_csv(self, output_path: str = "") -> None:
        """
        Export factor summary to CSV for easy spreadsheet viewing.
        """
        if not output_path:
            from src.config.paths import DATA_DIR

            output_path = str(DATA_DIR / "alpha_zoo_summary.csv")
        import pandas as pd

        alphas = self.load_all()
        if not alphas:
            return

        flat_data = []
        for a in alphas:
            row = {
                "id": a.get("id"),
                "formula": a.get("formula"),
                "rank_ic": a.get("metrics", {}).get("rank_ic"),
                "ic_ir": a.get("metrics", {}).get("ic_ir"),
                "fitness": a.get("metrics", {}).get("fitness"),
                "timestamp": a.get("timestamp"),
            }
            flat_data.append(row)

        df = pd.DataFrame(flat_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported zoo summary to {output_path}")
