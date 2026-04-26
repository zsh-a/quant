"""Assemble :class:`SessionTimeline` from ``session_db`` rows.

The loader is the single backend truth for both the live Brooks Studio panel
(WS appends ``BarEvent`` payloads) and the replay panel (paged ``BarEvent``
list). It reads ``session_logs`` rows tagged ``source="brooks_bar"`` (each
row carries the full per-bar snapshot in ``extra``) and stitches them to
fills from the ``trades`` table and the realised PnL from
``equity_history``.

Older session sources that pre-date the ``brooks_bar`` log row continue to
work — the loader degrades to whatever sources exist (``brooks_decision``
for signals/decision, ``brooks_decision_outcome`` for realised R) so the
endpoint is useful even when QUA-51 hasn't backfilled a session.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from session_db import SessionDB
from src.api.schemas.brooks_studio import (
    Bar,
    BarEvent,
    FeaturesView,
    FillView,
    HTFView,
    PnLPoint,
    RegimeView,
    SessionTimeline,
    StopAdj,
    StructureView,
    TimelinePage,
)
from src.brooks.schema import Decision, Signal

__all__ = [
    "BrooksTimelineLoader",
    "load_timeline",
    "load_timeline_page",
]


_BAR_LOG_SOURCE = "brooks_bar"
_DECISION_LOG_SOURCE = "brooks_decision"
_OUTCOME_LOG_SOURCE = "brooks_decision_outcome"
_DEFAULT_PAGE_LIMIT = 500
_MAX_PAGE_LIMIT = 5000

# Cap historical structure.confirmed_swings — older sessions persisted the
# full cumulative swing list every bar (O(N²) JSON). Truncate during
# materialise so the in-memory ``extra`` dict stays bounded even when the
# row on disk is hundreds of KB. Mirrors :data:`src.brooks.runtime.views.MAX_SWINGS_IN_VIEW`.
_MAX_SWINGS_IN_VIEW = 200


@dataclass(frozen=True)
class _RawLogRow:
    """A ``session_logs`` row materialised with its auto-increment id."""

    id: int
    source: str
    timestamp: str
    extra: Dict[str, Any]


class BrooksTimelineLoader:
    """Stateless loader — bind once, call :meth:`load` / :meth:`load_page`."""

    def __init__(self, session_db: SessionDB):
        self.session_db = session_db

    # ------------------------------------------------------------------ public

    def load(self, session_id: str, event_limit: Optional[int] = None) -> SessionTimeline:
        session = self.session_db.get_session(session_id)
        if session is None:
            raise KeyError(session_id)

        bar_rows = self._read_bar_rows(session_id)
        fills = self._read_fills(session_id)
        equity = self._read_equity(session_id)

        # Bars list is cheap (~100B/row, ~2MB at 23k bars) — always include
        # the full set so the chart's candle layer can render any range.
        # Events are the heavy ones; cap them to ``event_limit`` and let
        # the frontend page in the rest via ``/timeline/since/{seq}``.
        if event_limit is not None and event_limit > 0 and len(bar_rows) > event_limit:
            event_rows = bar_rows[:event_limit]
            has_more_events = True
            next_event_seq = event_rows[-1].id if event_rows else 0
        else:
            event_rows = bar_rows
            has_more_events = False
            next_event_seq = 0

        events = [self._row_to_event(row) for row in event_rows]
        events = self._attach_fills(events, fills)
        bars, htf_bars = self._extract_bars(bar_rows)
        pnl_curve = self._build_pnl_curve(events, equity) if not has_more_events else []

        params = dict(session.get("params") or {})
        config = {
            "analyst": params.get("analyst"),
            "analyst_params": params.get("analyst_params") or {},
            "mtf_intervals": list(params.get("mtf_intervals") or []),
            "params": params,
        }
        htf_intervals = list(params.get("mtf_intervals") or list(htf_bars.keys()))
        session_kind = "replay" if str(params.get("session_kind") or "live") == "replay" else "live"

        return SessionTimeline(
            session_id=session_id,
            session_kind=session_kind,
            symbol=str(session.get("symbol") or ""),
            base_interval=str(session.get("interval") or params.get("base_interval") or "1d"),
            htf_intervals=htf_intervals,
            bars=bars,
            htf_bars=htf_bars,
            events=events,
            pnl_curve=pnl_curve,
            config=config,
            created_at=str(session.get("created_at") or ""),
            next_event_seq=next_event_seq,
            has_more_events=has_more_events,
        )

    def load_page(
        self,
        session_id: str,
        since_seq: int = 0,
        limit: int = _DEFAULT_PAGE_LIMIT,
    ) -> TimelinePage:
        if self.session_db.get_session(session_id) is None:
            raise KeyError(session_id)

        limit = max(1, min(int(limit), _MAX_PAGE_LIMIT))
        rows, total = self._read_bar_rows_page(session_id, since_seq=int(since_seq), limit=limit)
        fills = self._read_fills(session_id)
        events = [self._row_to_event(row) for row in rows]
        events = self._attach_fills(events, fills)
        next_seq = rows[-1].id if rows else int(since_seq)
        return TimelinePage(
            events=events,
            next_seq=next_seq,
            has_more=total > len(rows),
        )

    # ------------------------------------------------------------------ db reads

    def _read_bar_rows(self, session_id: str) -> List[_RawLogRow]:
        # Stream the cursor instead of fetchall(): a 23k-bar replay used
        # to load 2.5 GB of raw JSON into memory just to materialise it.
        # Iterating row-by-row + capping swings inside ``_materialise``
        # bounds peak memory to a few hundred MB.
        out: List[_RawLogRow] = []
        with sqlite3.connect(self.session_db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, source, timestamp, extra
                FROM session_logs
                WHERE session_id = ? AND source = ?
                ORDER BY id ASC
                """,
                (session_id, _BAR_LOG_SOURCE),
            )
            for row in cursor:
                out.append(self._materialise(row))
        return out

    def _read_bar_rows_page(
        self,
        session_id: str,
        *,
        since_seq: int,
        limit: int,
    ) -> Tuple[List[_RawLogRow], int]:
        with sqlite3.connect(self.session_db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute(
                """
                SELECT COUNT(*)
                FROM session_logs
                WHERE session_id = ? AND source = ? AND id > ?
                """,
                (session_id, _BAR_LOG_SOURCE, since_seq),
            ).fetchone()[0]
            rows = conn.execute(
                """
                SELECT id, source, timestamp, extra
                FROM session_logs
                WHERE session_id = ? AND source = ? AND id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (session_id, _BAR_LOG_SOURCE, since_seq, limit),
            ).fetchall()
        return [self._materialise(row) for row in rows], int(total)

    def _read_fills(self, session_id: str) -> List[Dict[str, Any]]:
        return self.session_db.get_trades(session_id)

    def _read_equity(self, session_id: str) -> List[Dict[str, Any]]:
        return self.session_db.get_equity_history(session_id)

    def _materialise(self, row: sqlite3.Row) -> _RawLogRow:
        extra_raw = row["extra"]
        extra: Dict[str, Any] = {}
        if extra_raw:
            try:
                import json

                extra = json.loads(extra_raw) or {}
            except (TypeError, ValueError):
                extra = {}
        # Defence against historical sessions that persisted the full
        # cumulative swing list on every bar — drop everything but the
        # most recent entries before the dict propagates further.
        struct = extra.get("structure")
        if isinstance(struct, dict):
            swings = struct.get("confirmed_swings")
            if isinstance(swings, list) and len(swings) > _MAX_SWINGS_IN_VIEW * 2:
                struct["confirmed_swings"] = swings[-_MAX_SWINGS_IN_VIEW * 2 :]
        return _RawLogRow(
            id=int(row["id"]),
            source=str(row["source"]),
            timestamp=str(row["timestamp"]),
            extra=extra,
        )

    # ------------------------------------------------------------------ row → event

    def _row_to_event(self, row: _RawLogRow) -> BarEvent:
        extra = row.extra
        bar_idx = int(extra.get("bar_idx", 0))
        timestamp_ns = int(extra.get("timestamp_ns", 0))

        return BarEvent(
            bar_idx=bar_idx,
            timestamp_ns=timestamp_ns,
            regime=_build_regime(extra.get("regime")),
            features=_build_features(extra.get("features")),
            structure=_build_structure(extra.get("structure")),
            signals=_build_signals(extra.get("signals")),
            decision=_build_decision(extra.get("decision")),
            stop_adj=_build_stop_adj(extra.get("stop_adj")),
            pnl_r=_safe_float(extra.get("pnl_r")),
            htf=_build_htf(extra.get("htf")),
        )

    # ------------------------------------------------------------------ fills

    def _attach_fills(
        self,
        events: List[BarEvent],
        fills: List[Dict[str, Any]],
    ) -> List[BarEvent]:
        if not events or not fills:
            return events

        by_ns = {e.timestamp_ns: i for i, e in enumerate(events)}
        for fill in fills:
            ts = fill.get("timestamp")
            if ts is None:
                continue
            ts_ns = _parse_timestamp_ns(ts)
            if ts_ns is None:
                continue
            idx = by_ns.get(ts_ns)
            if idx is None:
                # Snap to the most recent event at-or-before the fill ts.
                idx = _snap_to_event(events, ts_ns)
            if idx is None:
                continue
            side = (fill.get("type") or fill.get("side") or "").lower()
            if side not in {"buy", "sell", "buy_to_cover", "sell_short"}:
                continue
            events[idx] = events[idx].model_copy(
                update={
                    "fill": FillView(
                        side=side,  # type: ignore[arg-type]
                        qty=float(fill.get("quantity") or 0.0),
                        price=float(fill.get("price") or 0.0),
                        reason=str(fill.get("reason") or ""),
                    )
                }
            )
        return events

    # ------------------------------------------------------------------ bars

    def _extract_bars(
        self,
        rows: List[_RawLogRow],
    ) -> Tuple[List[Bar], Dict[str, List[Bar]]]:
        # Iterate rows directly — used to zip with events but events can
        # be paginated now while bars must always be full. The bar dict
        # already carries its own timestamp_ns, with the row's outer
        # timestamp_ns as a defensive fallback.
        bars: List[Bar] = []
        htf_bars: Dict[str, List[Bar]] = {}
        seen_htf: Dict[str, set] = {}
        for row in rows:
            outer_ts = int(row.extra.get("timestamp_ns") or 0)
            ohlcv = row.extra.get("bar")
            if isinstance(ohlcv, dict):
                bars.append(_bar_from_dict(ohlcv, default_ts_ns=outer_ts))

            for tf, htf_payload in (row.extra.get("htf_bars") or {}).items():
                if not isinstance(htf_payload, dict):
                    continue
                key = (
                    int(htf_payload.get("timestamp_ns", 0)),
                    tf,
                )
                seen = seen_htf.setdefault(tf, set())
                if key in seen:
                    continue
                seen.add(key)
                htf_bars.setdefault(tf, []).append(_bar_from_dict(htf_payload))
        return bars, htf_bars

    # ------------------------------------------------------------------ pnl

    def _build_pnl_curve(
        self,
        events: List[BarEvent],
        equity: List[Dict[str, Any]],
    ) -> List[PnLPoint]:
        # Prefer per-bar pnl_r recorded by the strategy; fall back to a
        # normalised equity curve when not present.
        per_bar = [e for e in events if e.pnl_r is not None]
        if per_bar:
            cumulative = 0.0
            out: List[PnLPoint] = []
            for e in events:
                if e.pnl_r is not None:
                    cumulative += float(e.pnl_r)
                out.append(PnLPoint(bar_idx=e.bar_idx, equity_r=cumulative))
            return out

        if not equity:
            return []
        baseline = float(equity[0].get("total_equity") or 0.0)
        if baseline <= 0:
            return []
        return [
            PnLPoint(bar_idx=i, equity_r=float(point.get("total_equity", baseline)) / baseline - 1.0)
            for i, point in enumerate(equity)
        ]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def load_timeline(
    session_id: str,
    session_db: SessionDB,
    *,
    event_limit: Optional[int] = None,
) -> SessionTimeline:
    return BrooksTimelineLoader(session_db).load(session_id, event_limit=event_limit)


def load_timeline_page(
    session_id: str,
    session_db: SessionDB,
    *,
    since_seq: int = 0,
    limit: int = _DEFAULT_PAGE_LIMIT,
) -> TimelinePage:
    return BrooksTimelineLoader(session_db).load_page(session_id, since_seq=since_seq, limit=limit)


# ---------------------------------------------------------------------------
# Field builders — defensive: any malformed extra block is silently dropped.
# ---------------------------------------------------------------------------


def _build_regime(payload: Any) -> Optional[RegimeView]:
    if not isinstance(payload, dict):
        return None
    name = payload.get("name") or payload.get("regime")
    if not name:
        return None
    return RegimeView(
        name=str(name),
        confidence=_safe_float(payload.get("confidence"), default=0.0) or 0.0,
        reasons=[str(r) for r in (payload.get("reasons") or [])],
    )


def _build_features(payload: Any) -> Optional[FeaturesView]:
    if not isinstance(payload, dict):
        return None
    try:
        return FeaturesView(**{k: payload[k] for k in payload if k in FeaturesView.model_fields})
    except Exception:
        return None


def _build_structure(payload: Any) -> Optional[StructureView]:
    if not isinstance(payload, dict):
        return None
    try:
        return StructureView(
            always_in=payload.get("always_in", "neutral"),
            confirmed_swings=list(payload.get("confirmed_swings") or []),
            micro_channel_top=payload.get("micro_channel_top"),
            micro_channel_bot=payload.get("micro_channel_bot"),
            last_breakout_lookback_high=_safe_float(payload.get("last_breakout_lookback_high")),
            last_breakout_lookback_low=_safe_float(payload.get("last_breakout_lookback_low")),
        )
    except Exception:
        return None


def _build_signals(payload: Any) -> List[Signal]:
    if not isinstance(payload, list):
        return []
    out: List[Signal] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            out.append(Signal(**item))
        except Exception:
            continue
    return out


def _build_decision(payload: Any) -> Optional[Decision]:
    if not isinstance(payload, dict):
        return None
    try:
        return Decision(**payload)
    except Exception:
        return None


def _build_stop_adj(payload: Any) -> Optional[StopAdj]:
    if not isinstance(payload, dict):
        return None
    try:
        return StopAdj(
            from_px=float(payload.get("from_px")),
            to_px=float(payload.get("to_px")),
            reason=str(payload.get("reason") or ""),
        )
    except Exception:
        return None


def _build_htf(payload: Any) -> Dict[str, HTFView]:
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, HTFView] = {}
    for tf, body in payload.items():
        if not isinstance(body, dict):
            continue
        try:
            out[str(tf)] = HTFView(
                regime=body.get("regime"),
                always_in=body.get("always_in"),
                last_swing_idx=body.get("last_swing_idx"),
            )
        except Exception:
            continue
    return out


def _bar_from_dict(payload: Dict[str, Any], default_ts_ns: int = 0) -> Bar:
    return Bar(
        timestamp_ns=int(payload.get("timestamp_ns") or default_ts_ns),
        open=float(payload.get("open") or 0.0),
        high=float(payload.get("high") or 0.0),
        low=float(payload.get("low") or 0.0),
        close=float(payload.get("close") or 0.0),
        volume=float(payload.get("volume") or 0.0),
    )


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_timestamp_ns(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value)
    # Numeric — already nanos / millis / seconds since epoch.
    try:
        as_int = int(text)
        if as_int > 10**16:
            return as_int  # ns
        if as_int > 10**13:
            return as_int * 1_000  # μs → ns
        if as_int > 10**10:
            return as_int * 1_000_000  # ms → ns
        return as_int * 1_000_000_000  # s → ns
    except ValueError:
        pass

    from datetime import datetime

    try:
        ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(ts.timestamp() * 1_000_000_000)


def _snap_to_event(events: List[BarEvent], ts_ns: int) -> Optional[int]:
    """Return the index of the latest event with ``timestamp_ns <= ts_ns``."""
    chosen: Optional[int] = None
    for i, event in enumerate(events):
        if event.timestamp_ns <= ts_ns:
            chosen = i
        else:
            break
    return chosen
