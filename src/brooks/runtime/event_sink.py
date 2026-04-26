"""Sinks for per-bar BarEvent payloads.

Live mode wants both: persist to ``session_logs`` (so the timeline loader
can rebuild the full SessionTimeline on reconnect) **and** broadcast on
the Studio WS channel (so the panel renders in real time).

Replay mode wants only persistence — the panel reads the timeline once
the task finishes; no streaming during the run.

Each sink is a tiny class with a single ``handle`` method so a composite
sink can fan out without coupling sinks to each other.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Protocol

from loguru import logger

from session_db import SessionDB
from src.api.events import emit_studio_bar_event


class BarEventSink(Protocol):
    """Consume a per-bar ``BarEvent`` dict."""

    def handle(self, bar_event: Dict[str, Any]) -> None: ...  # noqa: D401, E704


class PersistingSink:
    """Append the bar event to ``session_logs`` (source ``brooks_bar``).

    The :class:`~src.services.brooks_timeline_loader.BrooksTimelineLoader`
    reads exactly these rows when assembling a SessionTimeline, so this
    sink is what makes the panel render anything at all on a reload.
    """

    def __init__(self, session_id: str, session_db: SessionDB):
        self.session_id = session_id
        self.session_db = session_db

    def handle(self, bar_event: Dict[str, Any]) -> None:
        try:
            self.session_db.add_session_log(
                session_id=self.session_id,
                timestamp=datetime.now().isoformat(),
                level="DEBUG",
                source="brooks_bar",
                message=(f"bar_idx={bar_event.get('bar_idx')} ts_ns={bar_event.get('timestamp_ns')}"),
                extra=bar_event,
            )
        except Exception as e:  # pragma: no cover — logging is best-effort
            logger.debug("brooks_bar persist failed: {}", e)


class BroadcastingSink:
    """Push the bar event onto the Studio WS channel.

    ``schedule`` is the engine's hand-off into an async event loop —
    typically :func:`asyncio.run_coroutine_threadsafe` against the live
    task's event loop. Replay mode does not use this sink.
    """

    def __init__(self, session_id: str, schedule: Callable[[Any], None]):
        self.session_id = session_id
        self._schedule = schedule

    def handle(self, bar_event: Dict[str, Any]) -> None:
        self._schedule(emit_studio_bar_event(self.session_id, bar_event))


class CompositeSink:
    """Fan out to several sinks. A single sink failure does not block the rest."""

    def __init__(self, sinks: Iterable[BarEventSink]):
        self._sinks: List[BarEventSink] = list(sinks)

    def handle(self, bar_event: Dict[str, Any]) -> None:
        for sink in self._sinks:
            try:
                sink.handle(bar_event)
            except Exception as e:  # pragma: no cover — defensive
                logger.debug("sink {} handle failed: {}", type(sink).__name__, e)


__all__ = [
    "BarEventSink",
    "BroadcastingSink",
    "CompositeSink",
    "PersistingSink",
]
