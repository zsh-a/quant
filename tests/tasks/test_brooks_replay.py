"""End-to-end test for the historical-replay Celery task.

Drives the task body directly with an injected :class:`DataStream` of
synthetic bars and verifies:

* the session row is created with ``session_kind="replay"``;
* per-bar BarEvents land in ``session_logs`` (source ``brooks_bar``)
  so the timeline loader can rebuild the SessionTimeline;
* the session ends in ``stopped`` with progress 100.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pytest

from session_db import SessionDB
from src.core.base import Bar, DataStream


@pytest.fixture
def temp_session_db(monkeypatch, tmp_path):
    db_path = tmp_path / "sessions.db"
    monkeypatch.setenv("SESSION_DB_PATH", str(db_path))
    return SessionDB(str(db_path))


class _ReplayStream(DataStream):
    """Finite list-driven DataStream — exposes total_bars for progress tests."""

    is_live = False

    def __init__(self, batches: List[Dict[str, Bar]]):
        self._batches = list(batches)
        self._idx = 0

    def next_bar(self, timeout: Optional[float] = None):
        if self._idx >= len(self._batches):
            return None
        b = self._batches[self._idx]
        self._idx += 1
        return b

    def reset(self):
        self._idx = 0

    def total_bars(self) -> int:
        return len(self._batches)


def _make_bars(n: int = 24, symbol: str = "BTC/USDT") -> List[Dict[str, Bar]]:
    base = datetime(2026, 4, 26, 12, 0)
    out: List[Dict[str, Bar]] = []
    for i in range(n):
        price = 30_000 + i * 5.0
        out.append(
            {
                symbol: Bar(
                    symbol=symbol,
                    timestamp=base + timedelta(minutes=5 * i),
                    open=price,
                    high=price + 4.0,
                    low=price - 4.0,
                    close=price + 1.0,
                    volume=1.0,
                    amount=price,
                )
            }
        )
    return out


def test_replay_task_runs_to_completion(temp_session_db):
    from src.tasks.brooks_replay_task import brooks_replay_task

    session_id = "sess-replay-1"
    stream = _ReplayStream(_make_bars(24))
    result = brooks_replay_task.run(
        session_id,
        {
            "symbol": "BTC/USDT",
            "interval": "5m",
            "start": "2026-04-26T00:00:00Z",
            "end": "2026-04-26T23:59:00Z",
            "analyst": "rule",
            "stream": stream,
        },
    )
    assert result["session_id"] == session_id
    assert result["bars"] == 24
    assert "final_equity" in result

    row = temp_session_db.get_session(session_id)
    assert row is not None, "session row should be created"
    assert row["status"] == "stopped"
    assert row["progress"] == 100.0

    params = row.get("params") or {}
    if isinstance(params, str):
        params = json.loads(params)
    assert params.get("session_kind") == "replay"

    # Per-bar logs feed the timeline loader.
    rows = temp_session_db.get_session_logs(session_id, source="brooks_bar")
    assert len(rows) == 24


def test_replay_task_marks_session_failed_on_error(temp_session_db):
    """A broken stream should still flip the session to failed."""
    from src.tasks.brooks_replay_task import brooks_replay_task

    class _BrokenStream(DataStream):
        is_live = False

        def next_bar(self, timeout=None):
            raise RuntimeError("stream blew up")

        def reset(self):
            pass

    session_id = "sess-replay-fail"
    with pytest.raises(RuntimeError, match="stream blew up"):
        brooks_replay_task.run(
            session_id,
            {
                "symbol": "BTC/USDT",
                "interval": "5m",
                "start": "2026-04-26T00:00:00Z",
                "end": "2026-04-26T23:59:00Z",
                "stream": _BrokenStream(),
            },
        )
    row = temp_session_db.get_session(session_id)
    assert row is not None
    assert row["status"] == "failed"
