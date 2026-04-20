from __future__ import annotations

import ast
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


# ---------------------------------------------------------------------------
# Canonical hash: AST-normalized, whitespace-free, commutative-op sorted
# ---------------------------------------------------------------------------

_COMMUTATIVE_OPS = {ast.Add, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor}


def _ast_to_canon(node: ast.AST) -> str:
    """Render an AST node as a normalized, deterministic string.

    Commutative binop children are sorted so ``a+b`` and ``b+a`` share
    the same canonical form. Function calls keep argument order (most
    alpha operators are position-sensitive).
    """
    if isinstance(node, ast.Expression):
        return _ast_to_canon(node.body)
    if isinstance(node, ast.BinOp):
        op = type(node.op).__name__
        lhs = _ast_to_canon(node.left)
        rhs = _ast_to_canon(node.right)
        if type(node.op) in _COMMUTATIVE_OPS:
            lhs, rhs = sorted([lhs, rhs])
        return f"({lhs} {op} {rhs})"
    if isinstance(node, ast.UnaryOp):
        return f"({type(node.op).__name__} {_ast_to_canon(node.operand)})"
    if isinstance(node, ast.BoolOp):
        op = type(node.op).__name__
        parts = sorted(_ast_to_canon(v) for v in node.values)
        return f"({op} {' '.join(parts)})"
    if isinstance(node, ast.Compare):
        left = _ast_to_canon(node.left)
        parts = []
        for op, cmp in zip(node.ops, node.comparators):
            parts.append(f"{type(op).__name__} {_ast_to_canon(cmp)}")
        return f"(Cmp {left} {' '.join(parts)})"
    if isinstance(node, ast.Call):
        func = _ast_to_canon(node.func)
        args = [_ast_to_canon(a) for a in node.args]
        kwargs = sorted(f"{kw.arg}={_ast_to_canon(kw.value)}" for kw in node.keywords if kw.arg is not None)
        return f"{func}({', '.join(args + kwargs)})"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_ast_to_canon(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Subscript):
        return f"{_ast_to_canon(node.value)}[{_ast_to_canon(node.slice)}]"
    if isinstance(node, ast.Tuple):
        return "(" + ", ".join(_ast_to_canon(e) for e in node.elts) + ")"
    if isinstance(node, ast.List):
        return "[" + ", ".join(_ast_to_canon(e) for e in node.elts) + "]"
    # Fallback: structural repr without line-number noise
    return ast.dump(node, annotate_fields=False, include_attributes=False)


def canonical_hash(formula: str) -> str:
    """SHA-256 of the canonicalised AST form. 64 hex chars.

    Falls back to hash-of-whitespace-stripped on parse failure so callers
    always get a stable key.
    """
    cleaned = formula.strip()
    try:
        tree = ast.parse(cleaned, mode="eval")
        canon = _ast_to_canon(tree)
    except SyntaxError:
        canon = "".join(cleaned.split())
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def full_expr_hash(formula: str) -> str:
    """Full 64-char SHA-256 of the raw formula text."""
    return hashlib.sha256(formula.encode("utf-8")).hexdigest()


def signature_hash(alpha_signature: list[float] | Any, bucket_size: int = 64) -> str:
    """Bucketed hash of an alpha signature vector for near-duplicate detection.

    We quantize each value into a small integer bucket and hash the
    result — nearly-identical signals land in the same bucket so ``==``
    on the hash approximates Pearson correlation > 0.99.
    """
    try:
        import numpy as np

        arr = np.asarray(alpha_signature, dtype=float).flatten()
    except Exception:
        return ""
    if arr.size == 0:
        return ""
    arr = arr[~(arr != arr)]  # drop NaN
    if arr.size == 0:
        return ""
    # Normalize to zero mean / unit std so scale doesn't matter
    mean = arr.mean()
    std = arr.std()
    if std <= 1e-12:
        return "zero_std"
    normed = (arr - mean) / std
    buckets = (normed.clip(-3, 3) * (bucket_size // 6)).astype("int32")
    return hashlib.sha256(buckets.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Alpha persistence: runs directory + zoo directory
# ---------------------------------------------------------------------------


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

    # -----------------------------------------------------------------------
    # Zoo: factor storage with semantic dedup + lineage
    # -----------------------------------------------------------------------

    @staticmethod
    def _zoo_filename(canonical: str) -> str:
        """Short, filesystem-friendly filename keyed on canonical hash."""
        return f"alpha_{canonical[:24]}.json"

    def _write_lineage_edge(
        self,
        *,
        parent_kind: str,
        parent_id: str,
        child_kind: str,
        child_id: str,
        relation: str,
        meta: Dict[str, Any] | None = None,
    ) -> None:
        """Best-effort lineage edge write; swallows errors (persistence must not fail)."""
        try:
            from session_db import SessionDB

            SessionDB().add_lineage_edge(
                parent_kind=parent_kind,
                parent_id=parent_id,
                child_kind=child_kind,
                child_id=child_id,
                relation=relation,
                meta=meta,
            )
        except Exception as exc:
            logger.debug("zoo lineage edge write failed: {}", exc)

    def upsert_zoo_entry(self, entry: dict[str, Any], *, run_id: str = "") -> dict[str, Any]:
        """Save or merge a single Zoo entry keyed on canonical hash.

        Accepted fields:
          formula, expr_hash, fitness, metrics, lineage, tags, source,
          note, validation, parent_expr_hashes, alpha_signature, live_metrics,
          source_job_id, source_run_id, source_cycle, source_strategy, source_round,
          auto_archived.
        """
        formula = str(entry.get("formula") or "").strip()
        if not formula:
            raise ValueError("upsert_zoo_entry: empty formula")

        canon = canonical_hash(formula)
        full_hash = entry.get("expr_hash") or full_expr_hash(formula)
        # Always upgrade to full 64-char hash
        if len(str(full_hash)) < 64:
            full_hash = full_expr_hash(formula)

        sig_hash: str = ""
        alpha_sig = entry.pop("alpha_signature", None)
        if alpha_sig is not None:
            sig_hash = signature_hash(alpha_sig)

        now = datetime.now(UTC).isoformat()
        path = self.zoo_dir / self._zoo_filename(canon)

        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}

        merged: Dict[str, Any] = dict(existing)
        merged.update(
            {
                "formula": formula,
                "expr_hash": full_hash,
                "canonical_hash": canon,
                "signature_hash": sig_hash or existing.get("signature_hash", ""),
                "fitness": max(
                    float(entry.get("fitness") or 0.0),
                    float(existing.get("fitness") or 0.0),
                ),
                "metrics": entry.get("metrics") or existing.get("metrics") or {},
                "lineage": entry.get("lineage") or existing.get("lineage") or {},
                "tags": sorted(set((entry.get("tags") or []) + (existing.get("tags") or []))),
                "note": entry.get("note") or existing.get("note"),
                "validation": entry.get("validation") or existing.get("validation"),
                "source": entry.get("source") or existing.get("source") or "search",
                "source_job_id": entry.get("source_job_id") or existing.get("source_job_id"),
                "source_run_id": entry.get("source_run_id") or existing.get("source_run_id") or run_id or None,
                "source_cycle": entry.get("source_cycle") or existing.get("source_cycle"),
                "source_strategy": entry.get("source_strategy") or existing.get("source_strategy"),
                "source_round": entry.get("source_round") or existing.get("source_round"),
                "parent_expr_hashes": list(
                    {*(entry.get("parent_expr_hashes") or []), *(existing.get("parent_expr_hashes") or [])}
                ),
                "auto_archived": bool(entry.get("auto_archived", existing.get("auto_archived", False))),
                "updated_at": now,
                "created_at": existing.get("created_at") or now,
                "saved_at": now,
                "run_id": run_id or existing.get("run_id"),
            }
        )

        # Live metrics: append time-series entry so we can track OOS performance.
        live_in: Any = entry.get("live_metrics")
        if live_in:
            live_series: List[Dict[str, Any]] = list(existing.get("live_metrics_series") or [])
            if isinstance(live_in, list):
                live_series.extend(live_in)
            else:
                item = dict(live_in)
                item.setdefault("recorded_at", now)
                live_series.append(item)
            merged["live_metrics_series"] = live_series
            merged["live_metrics"] = live_series[-1]

        path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        # Emit lineage edges for produced-by-search and derived-from relations.
        if merged.get("source_job_id"):
            self._write_lineage_edge(
                parent_kind="search_job",
                parent_id=str(merged["source_job_id"]),
                child_kind="zoo_factor",
                child_id=canon,
                relation="produced",
                meta={"run_id": merged.get("source_run_id"), "round": merged.get("source_round")},
            )
        for parent in merged["parent_expr_hashes"]:
            if parent and parent != canon:
                self._write_lineage_edge(
                    parent_kind="zoo_factor",
                    parent_id=str(parent),
                    child_kind="zoo_factor",
                    child_id=canon,
                    relation="derived_from",
                )

        merged["path"] = str(path)
        return merged

    def save_zoo_entries(self, entries: list[dict[str, Any]], run_id: str) -> list[str]:
        paths: list[str] = []
        for entry in entries:
            merged = self.upsert_zoo_entry(entry, run_id=run_id)
            paths.append(merged.get("path", ""))
        return [p for p in paths if p]

    def list_zoo_entries(self, limit: int = 50) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for path in self.zoo_dir.glob("alpha_*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            payload["path"] = str(path)
            entries.append(payload)
        entries.sort(key=lambda item: float(item.get("fitness", 0.0)), reverse=True)
        return entries[:limit]

    def get_zoo_entry(self, canonical_or_expr_hash: str) -> dict[str, Any] | None:
        """Look up a Zoo entry by canonical_hash (preferred) or expr_hash."""
        key = canonical_or_expr_hash
        # Canonical-hash fast path: filename prefix.
        candidate = self.zoo_dir / self._zoo_filename(key)
        if candidate.exists():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                payload["path"] = str(candidate)
                return payload
            except Exception:
                pass
        # Fallback: scan for matching expr_hash.
        for path in self.zoo_dir.glob("alpha_*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if payload.get("expr_hash") == key or payload.get("canonical_hash") == key:
                payload["path"] = str(path)
                return payload
        return None

    def append_live_metrics(
        self,
        canonical_or_expr_hash: str,
        metrics: dict[str, Any],
        *,
        source_run_id: str | None = None,
        source_simulation_job_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Record a live (out-of-sample) metrics snapshot for a Zoo factor."""
        entry = self.get_zoo_entry(canonical_or_expr_hash)
        if not entry:
            return None
        item = dict(metrics)
        item.setdefault("recorded_at", datetime.now(UTC).isoformat())
        if source_run_id:
            item["source_run_id"] = source_run_id
        if source_simulation_job_id:
            item["source_simulation_job_id"] = source_simulation_job_id
        merged = self.upsert_zoo_entry(
            {
                "formula": entry["formula"],
                "expr_hash": entry.get("expr_hash"),
                "live_metrics": item,
            },
            run_id=entry.get("run_id", ""),
        )
        if source_simulation_job_id:
            self._write_lineage_edge(
                parent_kind="simulation_run",
                parent_id=source_run_id or source_simulation_job_id,
                child_kind="zoo_factor",
                child_id=entry.get("canonical_hash") or canonical_or_expr_hash,
                relation="backtests",
                meta={"metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}},
            )
        return merged

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
    # Legacy AlphaNode API (kept for in-process callers only)
    # -----------------------------------------------------------------------

    def save_node(self, node: Any, metadata: Dict[str, Any] | None = None) -> None:
        formula = node.formula
        canon = canonical_hash(formula)
        data = {
            "id": canon[:12],
            "formula": formula,
            "canonical_hash": canon,
            "expr_hash": full_expr_hash(formula),
            "metrics": node.metrics,
            "timestamp": datetime.now().isoformat(),
            "name": getattr(node, "name", "unknown"),
            "description": getattr(node, "description", ""),
            "metadata": metadata or {},
        }
        file_path = self.zoo_dir / self._zoo_filename(canon)
        try:
            file_path.write_text(
                json.dumps(data, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug(f"Saved alpha factor {canon[:12]} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save alpha {canon[:12]}: {e}")

    def save_zoo(self, zoo: List[Any], task_name: str = "default") -> None:
        logger.info(f"Saving {len(zoo)} factors to Zoo...")
        for node in zoo:
            self.save_node(node, {"task": task_name})

    def load_all(self) -> List[Dict[str, Any]]:
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
        alphas.sort(key=lambda x: abs(x.get("metrics", {}).get("rank_ic", 0)), reverse=True)
        return alphas

    def export_to_csv(self, output_path: str = "") -> None:
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
                "id": a.get("id") or (a.get("canonical_hash") or "")[:12],
                "formula": a.get("formula"),
                "rank_ic": a.get("metrics", {}).get("rank_ic"),
                "ic_ir": a.get("metrics", {}).get("ic_ir"),
                "fitness": a.get("metrics", {}).get("fitness") or a.get("fitness"),
                "timestamp": a.get("timestamp") or a.get("saved_at"),
            }
            flat_data.append(row)

        df = pd.DataFrame(flat_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported zoo summary to {output_path}")
