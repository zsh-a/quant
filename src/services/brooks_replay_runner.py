"""Re-run analysts against a single bar for the Brooks Studio compare panel.

The MultiAnalystCompare side panel asks the backend for the output of N
analysts on the same bar so the user can diff their decisions while
replaying. This module builds a :class:`BrooksContext` from the data already
captured in ``session_db`` (per-bar OHLCV plus any HTF bars persisted by the
strategy) and runs each requested analyst against it.

Design notes:

* The runner is *best-effort*. LLM/VLM analysts make remote calls; if those
  fail (no API key, network down, etc.) we surface the error in the result
  payload rather than 500ing the whole request.
* The runner does **not** re-evaluate the EV gate or the risk layer — it
  exposes raw analyst signals plus, when available, the strategy's persisted
  decision for the chosen bar. That's the contract the frontend wants for a
  visual diff.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from session_db import SessionDB
from src.api.schemas.brooks_studio import SessionTimeline
from src.brooks.analyst.base import AnalystRegistry
from src.brooks.context import AccountSnapshot, Bar, BrooksContext, TFSnapshot
from src.brooks.schema import Decision, Signal
from src.services.brooks_timeline_loader import BrooksTimelineLoader

__all__ = [
    "AnalystResult",
    "BrooksReplayRunner",
    "DEFAULT_BAR_CONTEXT",
]

DEFAULT_BAR_CONTEXT = 200
DEFAULT_HTF_CONTEXT = 60


@dataclass
class AnalystResult:
    """Result for one analyst on one bar."""

    analyst: str
    bar_idx: int
    signals: List[Signal]
    decision: Optional[Decision]
    error: Optional[str] = None
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analyst": self.analyst,
            "bar_idx": self.bar_idx,
            "signals": [s.model_dump() for s in self.signals],
            "decision": self.decision.model_dump() if self.decision is not None else None,
            "error": self.error,
            "cached": self.cached,
        }


class BrooksReplayRunner:
    """Run the requested analysts against a chosen bar of a session."""

    def __init__(
        self,
        session_db: SessionDB,
        bar_context: int = DEFAULT_BAR_CONTEXT,
        htf_context: int = DEFAULT_HTF_CONTEXT,
    ):
        self._loader = BrooksTimelineLoader(session_db)
        self._bar_context = max(2, int(bar_context))
        self._htf_context = max(2, int(htf_context))

    # ------------------------------------------------------------------ public

    async def run(
        self,
        session_id: str,
        bar_idx: int,
        analysts: List[str],
    ) -> List[AnalystResult]:
        timeline = await asyncio.to_thread(self._loader.load, session_id)
        if not timeline.bars:
            raise ValueError(f"session {session_id!r} has no bars")
        if bar_idx < 0 or bar_idx >= len(timeline.bars):
            raise ValueError(
                f"bar_idx {bar_idx} out of range [0, {len(timeline.bars) - 1}]"
            )
        if not analysts:
            raise ValueError("analysts list is empty")

        ctx = self._build_context(timeline, bar_idx)
        persisted = self._persisted_event(timeline, bar_idx)

        coros = [self._run_one(name, ctx, persisted, bar_idx) for name in analysts]
        return await asyncio.gather(*coros)

    # ------------------------------------------------------------------ helpers

    async def _run_one(
        self,
        name: str,
        ctx: BrooksContext,
        persisted: Dict[str, Any],
        bar_idx: int,
    ) -> AnalystResult:
        if name not in AnalystRegistry.all():
            return AnalystResult(
                analyst=name,
                bar_idx=bar_idx,
                signals=[],
                decision=None,
                error=f"unknown analyst {name!r}",
            )
        try:
            analyst = AnalystRegistry.build(name)
            signals = await analyst.analyze(ctx)
        except Exception as e:  # network / config / runtime — surface to UI
            return AnalystResult(
                analyst=name,
                bar_idx=bar_idx,
                signals=[],
                decision=persisted.get("decision"),
                error=f"{type(e).__name__}: {e}",
            )

        # Use the persisted decision if it matches this analyst, otherwise
        # leave decision None (the side panel renders signal-level diffs).
        decision = None
        persisted_dec = persisted.get("decision")
        if isinstance(persisted_dec, Decision):
            if persisted_dec.source == name or persisted_dec.source.startswith(f"{name}:"):
                decision = persisted_dec

        return AnalystResult(
            analyst=name,
            bar_idx=bar_idx,
            signals=list(signals),
            decision=decision,
        )

    def _build_context(self, timeline: SessionTimeline, bar_idx: int) -> BrooksContext:
        start = max(0, bar_idx - self._bar_context + 1)
        primary_bars = [
            Bar(
                timestamp_ns=int(b.timestamp_ns),
                open=float(b.open),
                high=float(b.high),
                low=float(b.low),
                close=float(b.close),
                volume=float(b.volume),
            )
            for b in timeline.bars[start : bar_idx + 1]
        ]
        primary = TFSnapshot(interval=timeline.base_interval, bars=primary_bars)

        cursor_ns = primary_bars[-1].timestamp_ns if primary_bars else 0
        htf_snapshots: Dict[str, TFSnapshot] = {}
        for tf, bars in timeline.htf_bars.items():
            visible = [
                Bar(
                    timestamp_ns=int(b.timestamp_ns),
                    open=float(b.open),
                    high=float(b.high),
                    low=float(b.low),
                    close=float(b.close),
                    volume=float(b.volume),
                )
                for b in bars
                if int(b.timestamp_ns) <= cursor_ns
            ][-self._htf_context :]
            if visible:
                htf_snapshots[tf] = TFSnapshot(interval=tf, bars=visible)

        return BrooksContext(
            symbol=timeline.symbol or "",
            primary=primary,
            htf=htf_snapshots,
            account=AccountSnapshot(equity=0.0, cash=0.0),
            now_ns=cursor_ns,
        )

    def _persisted_event(
        self,
        timeline: SessionTimeline,
        bar_idx: int,
    ) -> Dict[str, Any]:
        for ev in timeline.events:
            if ev.bar_idx == bar_idx:
                return {
                    "decision": ev.decision,
                    "signals": list(ev.signals or []),
                }
        return {"decision": None, "signals": []}
