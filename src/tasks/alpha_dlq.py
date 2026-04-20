"""Dead-letter queue for Alpha Lab automation.

Failed cycle payloads are written as JSON files under ``data/alpha/dlq/``
and (optionally) enqueued here for replay. The worker-side ``replay_dlq``
task re-submits the original automation cycle using the stored payload.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.config.paths import ALPHA_DLQ_DIR
from src.tasks.celery_app import app


def write_dlq_entry(
    *,
    source: str,
    error: str,
    payload: dict[str, Any],
    entry_id: str | None = None,
) -> Path:
    """Serialize a failed automation payload to the DLQ directory.

    ``source`` is a short tag (e.g. ``"auto_search_cycle"`` or
    ``"simulation_run"``). ``payload`` is everything needed to replay.
    Returns the absolute path of the written file.
    """
    ALPHA_DLQ_DIR.mkdir(parents=True, exist_ok=True)
    entry_id = entry_id or uuid.uuid4().hex[:16]
    record = {
        "entry_id": entry_id,
        "source": source,
        "error": error,
        "payload": payload,
        "created_at": datetime.now(UTC).isoformat(),
        "replays": 0,
    }
    path = ALPHA_DLQ_DIR / f"{source}_{entry_id}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.warning("alpha.dlq entry written source={} id={} path={}", source, entry_id, path)
    return path


def list_dlq_entries(limit: int = 50) -> list[dict[str, Any]]:
    if not ALPHA_DLQ_DIR.exists():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(ALPHA_DLQ_DIR.glob("*.json"), reverse=True):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            item["path"] = str(path)
            entries.append(item)
        except Exception:
            continue
        if len(entries) >= limit:
            break
    return entries


def pop_dlq_entry(entry_id: str) -> dict[str, Any] | None:
    for path in ALPHA_DLQ_DIR.glob(f"*_{entry_id}.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            path.unlink(missing_ok=True)
            return data
        except Exception:
            continue
    return None


@app.task(
    name="src.tasks.alpha_dlq.replay_entry",
    queue="alpha_dlq",
    acks_late=True,
)
def replay_dlq_entry(entry_id: str) -> dict[str, Any]:
    """Replay a DLQ entry by re-enqueueing the appropriate upstream task."""
    record = pop_dlq_entry(entry_id)
    if not record:
        return {"ok": False, "reason": "entry_not_found", "entry_id": entry_id}

    source = record.get("source", "")
    payload = record.get("payload") or {}
    replays = int(record.get("replays", 0)) + 1

    # Route by source tag. Add more branches as needed.
    if source == "simulation_run":
        from src.tasks.automation import run_simulation_job_task

        job_id = payload.get("job_id")
        if not job_id:
            return {"ok": False, "reason": "missing_job_id", "entry_id": entry_id}
        run_simulation_job_task.apply_async(
            args=[job_id],
            kwargs={"trigger_source": payload.get("trigger_source", "dlq_replay")},
            queue="automation",
        )
        return {"ok": True, "enqueued": "simulation", "job_id": job_id, "replays": replays}

    # Fallback: write a mirror record with incremented replay count for inspection.
    write_dlq_entry(
        source=f"{source}_unreplayable",
        error=f"No replay handler for source={source!r}",
        payload=payload,
        entry_id=entry_id,
    )
    return {"ok": False, "reason": "no_handler", "source": source, "entry_id": entry_id}
