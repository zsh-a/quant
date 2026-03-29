from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class PersistedRun:
    run_id: str
    run_path: str
    zoo_dir: str


class AlphaLabPersistence:
    def __init__(self, root_dir: str = "data/alpha_lab"):
        self.root_dir = Path(root_dir)
        self.runs_dir = self.root_dir / "runs"
        self.zoo_dir = self.root_dir / "zoo"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.zoo_dir.mkdir(parents=True, exist_ok=True)

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
            runs.append(
                {
                    "run_id": payload.get("run_id"),
                    "saved_at": payload.get("saved_at"),
                    "path": str(path),
                    "dataset": payload.get("dataset", {}),
                    "top_results": len(payload.get("top_results", [])),
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
