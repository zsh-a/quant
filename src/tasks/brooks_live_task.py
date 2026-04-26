"""Celery tasks — BrooksStrategy paper-trading session runner.

Two tasks live here:

* :func:`brooks_live_task` drives a :class:`BrooksStrategy` against a CCXT
  realtime stream (or any injected :class:`DataStream`) and a paper-mode
  broker produced by :func:`~src.core.live_broker.create_live_broker`.
  Per-bar logic is fully delegated to :class:`~src.brooks.runtime.BrooksCore`,
  which is shared with :mod:`src.tasks.brooks_replay_task`.

* :func:`brooks_live_close_task` runs end-of-day, reads the realized-R
  samples accumulated in ``session_logs`` during the day, and appends
  them to the incremental samples parquet (consumed by
  ``scripts/brooks_build_hit_rate.py``).

Live execution against real exchanges is intentionally rejected by
:func:`create_live_broker`; this module is paper-only.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from celery import Task
from loguru import logger

from session_db import SessionDB
from src.api.events import (
    emit_error,
    emit_session_completed,
    emit_session_failed,
    emit_session_started,
    emit_session_stopped,
    emit_strategy_step,
)
from src.brooks.runtime import (
    BrooksCore,
    CompositeSink,
    PersistingSink,
    Throttle,
    WallClock,
)
from src.brooks.runtime.analyst_wrap import build_throttled_analyst
from src.brooks.runtime.event_sink import BroadcastingSink
from src.brooks.strategy import BrooksStrategy
from src.config.settings import get_brooks_live_config
from src.core.base import DataStream
from src.core.live_broker import create_live_broker
from src.tasks.celery_app import app

__all__ = [
    "brooks_live_task",
    "brooks_live_close_task",
    "BrooksLiveRegistry",
    "BrooksLiveSession",
]


# ---------------------------------------------------------------------------
# In-process session registry
# ---------------------------------------------------------------------------


@dataclass
class BrooksLiveSession:
    """Live controls for an in-flight BrooksLive session."""

    session_id: str
    stop_event: threading.Event = field(default_factory=threading.Event)
    analyst_name: str = "rule"
    analyst_params: Dict[str, Any] = field(default_factory=dict)
    # Shared state snapshot — what the WS panel reads on reconnect.
    last_regime: Dict[str, Any] = field(default_factory=dict)
    last_signals: List[Dict[str, Any]] = field(default_factory=list)
    last_decision: Dict[str, Any] = field(default_factory=dict)
    equity_points: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    switch_requested: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)

    def request_stop(self) -> None:
        self.stop_event.set()

    def request_switch(self, analyst: str) -> None:
        self.switch_requested = analyst


class BrooksLiveRegistry:
    """Process-local registry for running sessions."""

    _sessions: Dict[str, BrooksLiveSession] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, session: BrooksLiveSession) -> None:
        with cls._lock:
            cls._sessions[session.session_id] = session

    @classmethod
    def unregister(cls, session_id: str) -> None:
        with cls._lock:
            cls._sessions.pop(session_id, None)

    @classmethod
    def get(cls, session_id: str) -> Optional[BrooksLiveSession]:
        with cls._lock:
            return cls._sessions.get(session_id)

    @classmethod
    def all(cls) -> List[BrooksLiveSession]:
        with cls._lock:
            return list(cls._sessions.values())


# ---------------------------------------------------------------------------
# Celery tasks
# ---------------------------------------------------------------------------


class BrooksLiveTask(Task):
    """Base task with progress meta — mirrors :class:`BacktestTask`."""

    def update_progress(self, session_id: str, message: str, **extra: Any) -> None:
        self.update_state(
            state="PROGRESS",
            meta={
                "session_id": session_id,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                **extra,
            },
        )


@app.task(
    bind=True,
    base=BrooksLiveTask,
    name="src.tasks.brooks_live_task.run_brooks_live",
)
def brooks_live_task(self, session_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Drive a BrooksLive paper-trading session until stopped."""
    live_cfg = get_brooks_live_config()
    mode = config.get("mode", "paper")
    if mode != "paper":
        raise ValueError(f"brooks_live_task refuses mode={mode!r}; only paper trading is allowed")

    symbol = config["symbol"]
    interval = config.get("interval", live_cfg.default_interval)
    exchange = config.get("exchange", live_cfg.default_exchange)
    analyst_name = config.get("analyst", live_cfg.default_analyst)
    analyst_params = dict(config.get("analyst_params") or {})

    session_db = SessionDB()
    session = BrooksLiveSession(
        session_id=session_id,
        analyst_name=analyst_name,
        analyst_params=analyst_params,
        config=dict(config),
    )
    BrooksLiveRegistry.register(session)

    try:
        return _run_session(
            task=self,
            session=session,
            session_db=session_db,
            symbol=symbol,
            interval=interval,
            exchange=exchange,
            config=config,
            live_cfg=live_cfg,
        )
    finally:
        BrooksLiveRegistry.unregister(session_id)


@app.task(
    bind=True,
    base=BrooksLiveTask,
    name="src.tasks.brooks_live_task.close_brooks_live_day",
)
def brooks_live_close_task(self, session_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Daily close: append today's realized-R samples to the hit-rate delta parquet."""
    from datetime import date

    import pandas as pd

    live_cfg = get_brooks_live_config()
    session_db = SessionDB()
    today_start = f"{date.today().isoformat()}T00:00:00"
    ids = session_ids or [s.session_id for s in BrooksLiveRegistry.all()]

    rows: List[Dict[str, Any]] = []
    for sid in ids:
        logs = session_db.get_session_logs(
            sid,
            source="brooks_decision_outcome",
            since=today_start,
            limit=5_000,
        )
        for item in logs:
            extra = item.get("extra") or {}
            if not extra:
                continue
            rows.append(
                {
                    "session_id": sid,
                    "timestamp": item.get("timestamp"),
                    "pattern": extra.get("pattern"),
                    "regime": extra.get("regime"),
                    "htf_aligned": bool(extra.get("htf_aligned", False)),
                    "side": extra.get("side"),
                    "realized_r": float(extra.get("realized_r", 0.0)),
                    "hit_1r": bool(extra.get("hit_1r", False)),
                    "hit_2r": bool(extra.get("hit_2r", False)),
                }
            )

    if not rows:
        logger.info("brooks_live_close: nothing to persist for {} sessions", len(ids))
        return {"appended": 0, "sessions": ids}

    path = Path(live_cfg.hit_rate_samples_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(path, index=False)
    logger.info("brooks_live_close: appended {} rows to {}", len(new_df), path)
    return {"appended": len(new_df), "sessions": ids, "path": str(path)}


# ---------------------------------------------------------------------------
# Live driver
# ---------------------------------------------------------------------------


def _run_session(
    task: BrooksLiveTask,
    session: BrooksLiveSession,
    session_db: SessionDB,
    symbol: str,
    interval: str,
    exchange: str,
    config: Dict[str, Any],
    live_cfg,
) -> Dict[str, Any]:
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(
        target=loop.run_forever,
        name=f"brooks-live-{session.session_id}",
        daemon=True,
    )
    loop_thread.start()

    def schedule(coro):
        try:
            asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("schedule() failed: {}", e)

    broker = create_live_broker(
        mode="paper",
        initial_cash=float(config.get("initial_cash", live_cfg.initial_cash)),
        commission=float(config.get("commission", live_cfg.commission)),
        slippage=float(config.get("slippage", live_cfg.slippage)),
        allow_short=True,
        session_id=session.session_id,
    )

    strategy = _build_strategy(session, config)

    clock = WallClock()
    key_fn = _make_throttle_key_fn(interval)
    strategy._analyst = build_throttled_analyst(
        strategy._analyst,
        min_gap_seconds=float(live_cfg.llm_min_interval_seconds),
        clock=clock,
        key_fn=key_fn,
    )

    stream = _build_stream(config, symbol=symbol, interval=interval, exchange=exchange)

    # Persist session record (re-used by /session/{id}/status and ws routing).
    try:
        session_db.create_session(
            session_id=session.session_id,
            strategy_name="brooks",
            symbol=symbol,
            mode="paper",
            start_date=datetime.now().date().isoformat(),
            end_date=None,
            params={
                "analyst": session.analyst_name,
                "analyst_params": session.analyst_params,
                "interval": interval,
                "exchange": exchange,
                "mode": "paper",
                "session_kind": "live",
                "source": "brooks_live",
            },
            market="crypto",
            interval=interval,
        )
    except Exception as e:
        # Likely duplicate session_id (resume) — continue.
        logger.warning("brooks_live: create_session swallowed error: {}", e)

    session_db.update_session_status(session.session_id, "running", progress=0.0)
    schedule(emit_session_started(session.session_id, "brooks", symbol))

    sink = CompositeSink(
        [
            PersistingSink(session.session_id, session_db),
            BroadcastingSink(session.session_id, schedule),
        ]
    )
    core = BrooksCore(
        strategy=strategy,
        broker=broker,
        sink=sink,
        session_id=session.session_id,
        session_db=session_db,
        analyst_name=session.analyst_name,
        clock=clock,
        equity_throttle=Throttle(min_gap_seconds=float(live_cfg.equity_sync_interval_seconds)),
        emit=schedule,
        recent_signals_window=int(live_cfg.recent_signals_window),
    )

    if hasattr(stream, "start"):
        stream.start()

    bar_timeout = float(live_cfg.bar_queue_timeout_seconds)
    try:
        while True:
            if session.stop_event.is_set():
                logger.info("brooks_live: stop requested for {}", session.session_id)
                schedule(emit_session_stopped(session.session_id))
                break

            bars = _next_bar(stream, timeout=bar_timeout)
            if bars is None:
                if not stream.is_live:
                    break  # injected replay-style stream exhausted
                continue

            _maybe_apply_analyst_switch(session, strategy, clock, key_fn, live_cfg, schedule)
            core.analyst_name = session.analyst_name

            core.process_bar(bars)
            _mirror_core_into_session(core, session)

        schedule(
            emit_session_completed(
                session.session_id,
                float(broker.get_account_info().get("total_equity", 0.0)),
                len(session.trades),
            )
        )
        session_db.update_session_status(session.session_id, "stopped", progress=100.0)
        return {
            "session_id": session.session_id,
            "trades": len(session.trades),
            "final_equity": float(broker.get_account_info().get("total_equity", 0.0)),
        }
    except Exception as exc:
        logger.exception("brooks_live_task failed: {}", exc)
        session_db.update_session_status(session.session_id, "failed", error=str(exc))
        schedule(emit_session_failed(session.session_id, str(exc)))
        schedule(emit_error(session.session_id, str(exc)))
        raise
    finally:
        if hasattr(stream, "stop"):
            try:
                stream.stop()
            except Exception:
                pass
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2.0)


def _build_strategy(session: BrooksLiveSession, config: Dict[str, Any]) -> BrooksStrategy:
    params: Dict[str, Any] = {
        "analyst": session.analyst_name,
        "analyst_params": dict(session.analyst_params),
        "base_interval": config.get("interval", "5m"),
        "mtf_intervals": list(config.get("mtf_intervals") or []),
    }
    if "min_expected_r" in config:
        params["min_expected_r"] = float(config["min_expected_r"])
    return BrooksStrategy(
        db_client=None,
        session_id=session.session_id,
        **params,
    )


def _build_stream(
    config: Dict[str, Any],
    *,
    symbol: str,
    interval: str,
    exchange: str,
) -> DataStream:
    """Return the stream to drive the engine.

    ``config["stream"]`` may carry a pre-built :class:`DataStream` instance
    for tests / autopilot replays; otherwise a live ccxt stream is built.
    """
    supplied = config.get("stream")
    if isinstance(supplied, DataStream):
        return supplied
    from src.core.ccxt_realtime_stream import CcxtRealtimeDataStream

    return CcxtRealtimeDataStream(symbols=[symbol], interval=interval, exchange_id=exchange)


def _next_bar(stream: DataStream, *, timeout: float):
    nb = getattr(stream, "next_bar", None)
    if nb is None:
        return None
    try:
        return nb(timeout=timeout)  # ccxt stream signature
    except TypeError:
        return nb()  # list / historical streams without timeout


def _make_throttle_key_fn(interval: str) -> Callable[[Any], str]:
    def key_fn(ctx: Any) -> str:
        return f"{getattr(ctx, 'symbol', '')}:{interval}"

    return key_fn


def _maybe_apply_analyst_switch(
    session: BrooksLiveSession,
    strategy: BrooksStrategy,
    clock: WallClock,
    key_fn: Callable[[Any], str],
    live_cfg,
    schedule: Callable[[Any], None],
) -> None:
    if not session.switch_requested:
        return
    target = session.switch_requested
    session.switch_requested = None
    try:
        from src.brooks.analyst.base import AnalystRegistry

        new_analyst = AnalystRegistry.build(target, **session.analyst_params)
        strategy._analyst = build_throttled_analyst(
            new_analyst,
            min_gap_seconds=float(live_cfg.llm_min_interval_seconds),
            clock=clock,
            key_fn=key_fn,
        )
        session.analyst_name = target
        schedule(
            emit_strategy_step(
                session.session_id,
                {"event": "analyst_switched", "analyst": target},
            )
        )
        logger.info("brooks_live: analyst switched → {}", target)
    except Exception as e:
        logger.error("analyst switch to {} failed: {}", target, e)
        schedule(emit_error(session.session_id, f"analyst switch failed: {e}"))


def _mirror_core_into_session(core: BrooksCore, session: BrooksLiveSession) -> None:
    """Copy the latest core state into the registry session for state endpoints."""
    session.last_regime = core.last_regime
    session.last_decision = core.last_decision
    session.last_signals = core.last_signals
    session.equity_points = list(core.equity_points)
    session.trades = list(core.tracked_trades)
