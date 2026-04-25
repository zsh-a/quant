"""Phase 4.6 Celery tasks — BrooksLive paper-trading session runner.

Two tasks live here:

* :func:`brooks_live_task` drives a :class:`BrooksStrategy` against
  :class:`CcxtRealtimeDataStream` (or an injected stream) and a paper-mode
  broker produced by :func:`~src.core.live_broker.create_live_broker`.
  Every bar surfaces regime / signal / decision / equity events through
  the WebSocket event bus and persists decisions to ``session_logs``.

* :func:`brooks_live_close_task` runs end-of-day, reads the realized-R
  samples accumulated in ``session_logs`` during the day, and appends
  them to the incremental samples parquet (consumed by
  ``scripts/brooks_build_hit_rate.py``).

Live-mode (real orders) is intentionally rejected by
:func:`create_live_broker`; Phase 4.6 is paper-only by design.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from celery import Task
from loguru import logger

from session_db import SessionDB
from src.api.events import (
    emit_equity_update,
    emit_error,
    emit_session_completed,
    emit_session_failed,
    emit_session_started,
    emit_session_stopped,
    emit_strategy_step,
    emit_studio_bar_event,
    emit_trade_executed,
)
from src.brooks.strategy import BrooksStrategy
from src.config.settings import get_brooks_live_config
from src.core.base import DataStream
from src.core.engine import TradingEngine
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
    # LLM/VLM rate-limit bookkeeping. Keyed by (symbol, interval).
    last_llm_call_ts: Dict[str, float] = field(default_factory=dict)

    def request_stop(self) -> None:
        self.stop_event.set()

    def request_switch(self, analyst: str) -> None:
        self.switch_requested = analyst


class BrooksLiveRegistry:
    """Process-local registry for running sessions.

    The Celery worker holds exactly one registry per worker process. The
    API router reaches into it via :func:`get_session` to surface a
    snapshot / send a stop signal. Cross-worker control (e.g. when the API
    and worker live in different processes) is handled at the API layer
    by delegating to Celery's revoke / custom Redis pubsub, both of which
    are out of scope for Phase 4.6.
    """

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
    """Drive a BrooksLive paper-trading session until stopped.

    ``config`` keys:

    * ``symbol`` (str, required) — e.g. ``"BTC/USDT"``
    * ``interval`` (str) — bar interval for the stream (default ``"5m"``)
    * ``exchange`` (str) — CCXT exchange id (default ``"binance"``)
    * ``analyst`` (str) — analyst selector (default from settings)
    * ``analyst_params`` (dict) — forwarded to ``AnalystRegistry.build``
    * ``initial_cash`` / ``commission`` / ``slippage`` — broker params
    * ``mode`` (str) — must be ``"paper"``; ``"live"`` raises ValueError
    * ``mtf_intervals`` (list[str]) — higher timeframes for HTF context
    * ``llm_min_interval_seconds`` / ``llm_daily_budget_usd`` — rate caps
    """
    live_cfg = get_brooks_live_config()
    mode = config.get("mode", "paper")
    if mode != "paper":
        raise ValueError(f"brooks_live_task refuses mode={mode!r}; only paper trading is allowed in Phase 4.6")

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
    """Daily close: append today's realized-R samples to the hit-rate delta parquet.

    Reads ``session_logs`` rows written by :func:`_persist_decision_outcome`
    for the target sessions (or every active session when ``session_ids``
    is ``None``) since the start of the local day, and appends them to the
    incremental samples parquet at ``live_cfg.hit_rate_samples_path``.
    The main hit-rate table is rebuilt from this delta offline by
    ``scripts/brooks_build_hit_rate.py``.
    """
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
# Core loop
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
    loop_thread = threading.Thread(target=loop.run_forever, name=f"brooks-live-{session.session_id}", daemon=True)
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

    strategy = _build_strategy(session, config, session_db)
    _install_llm_rate_limiter(strategy, session, live_cfg)

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

    engine = _BrooksLiveEngine(
        strategy=strategy,
        broker=broker,
        data_stream=stream,
        session=session,
        session_db=session_db,
        emit=schedule,
        live_cfg=live_cfg,
    )

    if hasattr(stream, "start"):
        stream.start()

    try:
        engine.run()
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


def _build_strategy(
    session: BrooksLiveSession,
    config: Dict[str, Any],
    session_db: SessionDB,
) -> BrooksStrategy:
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


def _build_stream(config: Dict[str, Any], *, symbol: str, interval: str, exchange: str) -> DataStream:
    """Return the stream to drive the engine.

    ``config["stream"]`` may carry a pre-built :class:`DataStream` instance
    for tests / autopilot replays; otherwise a live ccxt stream is built.
    """
    supplied = config.get("stream")
    if isinstance(supplied, DataStream):
        return supplied
    from src.core.ccxt_realtime_stream import CcxtRealtimeDataStream

    return CcxtRealtimeDataStream(symbols=[symbol], interval=interval, exchange_id=exchange)


def _install_llm_rate_limiter(strategy: BrooksStrategy, session: BrooksLiveSession, live_cfg) -> None:
    """Wrap the analyst's ``analyze`` so LLM/VLM calls respect rate limits.

    Rule analysts are left alone (they're free). For LLM/VLM analysts we
    enforce a minimum wall-clock interval per (symbol, interval) bucket,
    falling back to an empty signal list when the budget blocks the call.
    """
    analyst = strategy._analyst  # type: ignore[attr-defined]
    name = getattr(analyst, "name", "")
    if not (name.startswith("llm:") or name.startswith("vlm:")):
        return
    min_gap = float(live_cfg.llm_min_interval_seconds)
    original = analyst.analyze

    async def gated(ctx):
        key = f"{ctx.symbol}:{session.config.get('interval', '5m')}"
        now = time.monotonic()
        last = session.last_llm_call_ts.get(key, 0.0)
        if now - last < min_gap:
            logger.debug("LLM rate-limit skip: {} ({}s < {}s)", key, now - last, min_gap)
            return []
        session.last_llm_call_ts[key] = now
        return await original(ctx)

    analyst.analyze = gated  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Engine wrapper
# ---------------------------------------------------------------------------


class _BrooksLiveEngine:
    """Wrap :class:`TradingEngine` so we can intercept per-bar events.

    We can't simply install ``engine.on_step`` because we want the regime
    / signal / decision telemetry captured *while* the strategy is running
    its logic (so the UI reflects the same context the analyst saw, not
    post-trade state). The cleanest seam is to subclass at call-site:
    take over the bar loop and expose hooks pre-/post-strategy.
    """

    def __init__(
        self,
        strategy: BrooksStrategy,
        broker,
        data_stream: DataStream,
        session: BrooksLiveSession,
        session_db: SessionDB,
        emit: Callable[[Any], None],
        live_cfg,
    ):
        self.strategy = strategy
        self.broker = broker
        self.data_stream = data_stream
        self.session = session
        self.session_db = session_db
        self.emit = emit
        self.live_cfg = live_cfg
        self.engine = TradingEngine(strategy=strategy, broker=broker, data_stream=data_stream)
        self._step_index = 0
        self._last_equity_emit = 0.0
        # Hook trade callbacks onto the broker.
        original_on_order_submitted = getattr(broker, "on_order_submitted", None)

        def _on_order(order):
            if original_on_order_submitted:
                original_on_order_submitted(order)
            self._handle_order_submitted(order)

        broker.on_order_submitted = _on_order

    # ---- public loop --------------------------------------------------

    def run(self) -> None:
        timeout = float(self.live_cfg.bar_queue_timeout_seconds)
        self.engine.running = True
        while self.engine.running:
            if self.session.stop_event.is_set():
                logger.info("brooks_live: stop requested for {}", self.session.session_id)
                self.emit(emit_session_stopped(self.session.session_id))
                break
            bars = self._next_bar(timeout)
            if bars is None:
                # Stream exhausted (replay) or timed out — treat as end for replays,
                # continue waiting for live.
                if not _is_live_stream(self.data_stream):
                    break
                continue

            self._handle_analyst_switch()
            self.engine.current_bars = bars

            # 1. Broker step — fills pending NEXT_OPEN orders from prior bar.
            self.broker.step(bars)

            # 2. Strategy on_bar — build context, analyse, submit orders.
            if hasattr(self.strategy, "_update_current_date"):
                self.strategy._update_current_date(bars)
            for sym in list(bars.keys()):
                # Seed per-symbol state *before* on_bar so classify capture
                # covers the first bar, not the second.
                if hasattr(self.strategy, "_ensure_symbol_state"):
                    self.strategy._ensure_symbol_state(sym)
                self._ensure_regime_capture(sym)
            try:
                self.strategy.on_bar(bars)
            except Exception as e:
                logger.exception("strategy.on_bar raised: {}", e)
                self.emit(emit_error(self.session.session_id, str(e)))

            # 3. Same-bar orders (IMMEDIATE_*).
            self.broker.process_same_bar_orders(bars, "IMMEDIATE_OPEN")
            self.broker.process_same_bar_orders(bars, "IMMEDIATE_CLOSE")

            # 4. Telemetry.
            self._emit_regime_and_signals(bars)
            self._emit_equity_update(bars)
            self._persist_decision_outcomes(bars)

            self._step_index += 1
            self.engine.last_bars = bars
        self.engine.running = False

    def _next_bar(self, timeout: float):
        nb = getattr(self.data_stream, "next_bar", None)
        if nb is None:
            return None
        try:
            return nb(timeout=timeout)  # ccxt stream signature
        except TypeError:
            return nb()  # list-based test streams

    # ---- telemetry ----------------------------------------------------

    def _ensure_regime_capture(self, symbol: str) -> None:
        """Install a one-shot classify wrapper that caches the latest snapshot.

        :class:`BrooksRegimeClassifier` doesn't retain its output; for the UI
        we need the most recent snapshot per symbol. The wrapper is installed
        lazily the first time we see a symbol and is a no-op for subsequent
        calls thanks to the sentinel attribute.
        """
        classifier = (self.strategy._regimes or {}).get(symbol)  # type: ignore[attr-defined]
        if classifier is None or getattr(classifier, "_live_wrapped", False):
            return
        original = classifier.classify

        def wrapped(features, structure):
            snap = original(features, structure)
            classifier._live_last_snapshot = snap
            return snap

        classifier.classify = wrapped  # type: ignore[assignment]
        classifier._live_wrapped = True

    def _handle_analyst_switch(self) -> None:
        if not self.session.switch_requested:
            return
        target = self.session.switch_requested
        self.session.switch_requested = None
        try:
            from src.brooks.analyst.base import AnalystRegistry

            new_analyst = AnalystRegistry.build(target, **self.session.analyst_params)
            self.strategy._analyst = new_analyst  # type: ignore[attr-defined]
            self.session.analyst_name = target
            _install_llm_rate_limiter(self.strategy, self.session, self.live_cfg)
            self.emit(
                emit_strategy_step(
                    self.session.session_id,
                    {"event": "analyst_switched", "analyst": target},
                )
            )
            logger.info("brooks_live: analyst switched → {}", target)
        except Exception as e:
            logger.error("analyst switch to {} failed: {}", target, e)
            self.emit(emit_error(self.session.session_id, f"analyst switch failed: {e}"))

    def _emit_regime_and_signals(self, bars) -> None:
        symbol, bar = next(iter(bars.items()))
        self._ensure_regime_capture(symbol)
        regime_payload: Optional[Dict[str, Any]] = None
        classifier = (self.strategy._regimes or {}).get(symbol)  # type: ignore[attr-defined]
        snapshot = getattr(classifier, "_live_last_snapshot", None) if classifier else None
        if snapshot is not None:
            try:
                regime_payload = snapshot.to_dict()
            except Exception:
                regime_payload = None

        pending = getattr(self.strategy, "_pending_decisions", {}) or {}
        decision = pending.get(symbol)
        decision_payload = None
        if decision is not None:
            decision_payload = _decision_to_dict(decision)

        positions = {}
        for sym, pos in (getattr(self.strategy, "_positions", {}) or {}).items():
            positions[sym] = {
                "side": pos.side,
                "entry_px": pos.entry_px,
                "stop_px": pos.stop_px,
                "qty_open": pos.qty_open,
                "one_r": pos.one_r,
                "ladder_stage": getattr(pos, "ladder_stage", 0),
            }

        payload = {
            "event": "bar_closed",
            "symbol": symbol,
            "timestamp": bar.timestamp.isoformat(),
            "ohlcv": {
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            },
            "regime": regime_payload,
            "decision": decision_payload,
            "analyst": self.session.analyst_name,
            "positions": positions,
        }
        self.session.last_regime = regime_payload or {}
        if decision_payload is not None:
            self.session.last_decision = decision_payload
            self.session.last_signals.append(decision_payload)
            window = int(self.live_cfg.recent_signals_window)
            if len(self.session.last_signals) > window:
                del self.session.last_signals[:-window]
        self.emit(emit_strategy_step(self.session.session_id, payload))

        # Studio: persist a per-bar snapshot row + push the BarEvent payload
        # on the dedicated channel so the new panel can drive both replay and
        # live from the same schema.
        bar_event = self._build_studio_bar_event(symbol, bar, regime_payload, decision)
        self._persist_studio_bar_log(bar_event)
        self.emit(emit_studio_bar_event(self.session.session_id, bar_event))

    def _emit_equity_update(self, bars) -> None:
        now = time.monotonic()
        if now - self._last_equity_emit < float(self.live_cfg.equity_sync_interval_seconds):
            return
        self._last_equity_emit = now
        acct = self.broker.get_account_info()
        ts = next(iter(bars.values())).timestamp.isoformat()
        equity_pt = {
            "timestamp": ts,
            "total_equity": float(acct.get("total_equity", 0.0)),
            "cash": float(acct.get("cash", 0.0)),
            "positions": acct.get("detailed_positions", {}),
        }
        self.session.equity_points.append(equity_pt)
        try:
            self.session_db.add_equity_point(
                self.session.session_id,
                ts,
                equity_pt["total_equity"],
                cash=equity_pt["cash"],
            )
        except Exception as e:
            logger.debug("add_equity_point failed: {}", e)
        self.emit(emit_equity_update(self.session.session_id, equity_pt))

    def _handle_order_submitted(self, order) -> None:
        # Persist decision JSON keyed to the order so post-fill analysis can correlate.
        pending = getattr(self.strategy, "_pending_decisions", {}) or {}
        decision = pending.get(order.symbol)
        if decision is None:
            return
        payload = _decision_to_dict(decision)
        payload["order_id"] = order.id
        payload["order_type"] = order.type
        self.session_db.add_session_log(
            session_id=self.session.session_id,
            timestamp=datetime.now().isoformat(),
            level="INFO",
            source="brooks_decision",
            message=(
                f"{payload.get('side')} {order.symbol} @ {payload.get('entry_px')} "
                f"stop={payload.get('stop_px')} E={payload.get('expected_r')}"
            ),
            extra=payload,
        )

    # ---- studio per-bar plumbing -------------------------------------

    def _build_studio_bar_event(
        self,
        symbol: str,
        bar,
        regime_payload: Optional[Dict[str, Any]],
        decision,
    ) -> Dict[str, Any]:
        """Snapshot the analyst's per-bar context as a :class:`BarEvent` dict."""
        ts_ns = int(bar.timestamp.timestamp() * 1_000_000_000)

        feat_obj = None
        struct_obj = None
        history = (getattr(self.strategy, "_feature_history", None) or {}).get(symbol) or []
        if history:
            feat_obj = history[-1]
        struct_state = (getattr(self.strategy, "_structures", None) or {}).get(symbol)
        if struct_state is not None:
            struct_obj = getattr(struct_state, "state", None)

        bar_idx = int(getattr(feat_obj, "bar_idx", 0)) if feat_obj is not None else 0

        decision_dict: Optional[Dict[str, Any]] = None
        signals_list: list = []
        if decision is not None:
            decision_dict = _decision_to_full_dict(decision)
            signals_list = [s.model_dump() for s in (decision.signals or [])]

        htf_payload: Dict[str, Dict[str, Any]] = {}
        htf_bars: Dict[str, Dict[str, Any]] = {}
        for tf, st in ((getattr(self.strategy, "_htf_state", None) or {}).get(symbol) or {}).items():
            last_regime = getattr(st, "last_regime", None)
            last_struct = getattr(st, "last_struct", None)
            htf_payload[tf] = {
                "regime": getattr(getattr(last_regime, "regime", None), "value", None) if last_regime else None,
                "always_in": getattr(last_struct, "always_in", None) if last_struct else None,
                "last_swing_idx": (
                    getattr(last_struct, "confirmed_swing_highs", [-1])[-1].bar_idx
                    if last_struct and getattr(last_struct, "confirmed_swing_highs", None)
                    else None
                ),
            }
            recent = getattr(st, "recent_bars", []) or []
            if recent:
                latest = recent[-1]
                htf_bars[tf] = {
                    "timestamp_ns": int(latest.timestamp_ns),
                    "open": float(latest.open),
                    "high": float(latest.high),
                    "low": float(latest.low),
                    "close": float(latest.close),
                    "volume": float(latest.volume),
                }

        return {
            "bar_idx": bar_idx,
            "timestamp_ns": ts_ns,
            "bar": {
                "timestamp_ns": ts_ns,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            },
            "features": _feature_to_view_dict(feat_obj) if feat_obj is not None else None,
            "structure": _structure_to_view_dict(struct_obj) if struct_obj is not None else None,
            "regime": _regime_to_view_dict(regime_payload),
            "signals": signals_list,
            "decision": decision_dict,
            "htf": htf_payload,
            "htf_bars": htf_bars,
            "symbol": symbol,
            "analyst": self.session.analyst_name,
        }

    def _persist_studio_bar_log(self, bar_event: Dict[str, Any]) -> None:
        try:
            self.session_db.add_session_log(
                session_id=self.session.session_id,
                timestamp=datetime.now().isoformat(),
                level="DEBUG",
                source="brooks_bar",
                message=f"bar_idx={bar_event.get('bar_idx')} ts_ns={bar_event.get('timestamp_ns')}",
                extra=bar_event,
            )
        except Exception as e:  # pragma: no cover — logging is best-effort
            logger.debug("brooks_bar log persist failed: {}", e)

    def _persist_decision_outcomes(self, bars) -> None:
        # Compare broker.trades to what we've already tracked; for each new trade,
        # if it closes a position opened earlier we can compute realized R and
        # persist an outcome row (consumed by brooks_live_close_task).
        broker_trades = list(getattr(self.broker, "trades", []) or [])
        if len(broker_trades) <= len(self.session.trades):
            return
        for trade in broker_trades[len(self.session.trades) :]:
            self.session.trades.append(trade)
            self.emit(emit_trade_executed(self.session.session_id, trade))
            try:
                self.session_db.add_trade(self.session.session_id, trade)
            except Exception as e:
                logger.debug("add_trade failed: {}", e)
            # Realized R is recorded by the strategy's close path; we leave
            # the detailed outcome row to _close_position's own hook. For
            # Phase 4.6 we approximate with the most recent decision if any.
            if trade.get("type") in ("sell", "buy_to_cover"):
                decision = self.session.last_decision
                if not decision:
                    continue
                one_r = abs(float(decision.get("entry_px", 0)) - float(decision.get("stop_px", 0)))
                if one_r == 0:
                    continue
                fill_px = float(trade.get("price", 0))
                if decision.get("side") == "long":
                    realized_r = (fill_px - float(decision["entry_px"])) / one_r
                else:
                    realized_r = (float(decision["entry_px"]) - fill_px) / one_r
                self.session_db.add_session_log(
                    session_id=self.session.session_id,
                    timestamp=str(trade.get("timestamp") or datetime.now().isoformat()),
                    level="INFO",
                    source="brooks_decision_outcome",
                    message=f"R={realized_r:.3f} pattern={decision.get('pattern')}",
                    extra={
                        "pattern": decision.get("pattern"),
                        "regime": decision.get("regime"),
                        "htf_aligned": bool(decision.get("htf_aligned", False)),
                        "side": decision.get("side"),
                        "realized_r": float(realized_r),
                        "hit_1r": realized_r >= 1.0,
                        "hit_2r": realized_r >= 2.0,
                    },
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decision_to_dict(decision) -> Dict[str, Any]:
    """Flatten :class:`Decision` into a JSON-safe payload for WS / session_logs."""
    sig = decision.signals[0] if decision.signals else None
    return {
        "side": decision.side,
        "entry_px": float(decision.entry_px),
        "stop_px": float(decision.stop_px),
        "target_px": float(decision.target_px) if decision.target_px is not None else None,
        "quantity": float(decision.quantity),
        "probability": float(decision.probability),
        "expected_r": float(decision.expected_r),
        "regime": decision.regime,
        "htf_aligned": bool(decision.htf_aligned),
        "pattern": sig.pattern if sig is not None else "unknown",
        "source": decision.source,
        "reasoning": decision.reasoning,
    }


def _is_live_stream(stream: DataStream) -> bool:
    return stream.__class__.__name__ == "CcxtRealtimeDataStream"


def _decision_to_full_dict(decision) -> Dict[str, Any]:
    """Pydantic-friendly Decision dump (loader rebuilds via ``Decision(**dict)``)."""
    return decision.model_dump(mode="json")


def _feature_to_view_dict(feat) -> Dict[str, Any]:
    return {
        "is_bull": bool(getattr(feat, "is_bull", False)),
        "body_pct": int(getattr(feat, "body_pct", 0)),
        "close_position": getattr(feat, "close_position", "mid"),
        "ema_relation": getattr(feat, "ema_relation", "at"),
        "leg_dir": getattr(feat, "leg_dir", "flat"),
        "leg_length": int(getattr(feat, "leg_length", 0)),
        "is_doji": bool(getattr(feat, "is_doji", False)),
        "is_inside_bar": bool(getattr(feat, "is_inside_bar", False)),
    }


def _structure_to_view_dict(struct) -> Dict[str, Any]:
    swings = []
    for s in getattr(struct, "confirmed_swing_highs", []) or []:
        swings.append({"idx": s.bar_idx, "kind": "high", "price": s.price})
    for s in getattr(struct, "confirmed_swing_lows", []) or []:
        swings.append({"idx": s.bar_idx, "kind": "low", "price": s.price})
    top = getattr(struct, "micro_channel_top", None)
    bot = getattr(struct, "micro_channel_bot", None)
    return {
        "always_in": getattr(struct, "always_in", "neutral"),
        "confirmed_swings": swings,
        "micro_channel_top": top.to_dict() if top is not None else None,
        "micro_channel_bot": bot.to_dict() if bot is not None else None,
        "last_breakout_lookback_high": getattr(struct, "last_breakout_lookback_high", None),
        "last_breakout_lookback_low": getattr(struct, "last_breakout_lookback_low", None),
    }


def _regime_to_view_dict(regime_payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not regime_payload:
        return None
    return {
        "name": regime_payload.get("regime") or regime_payload.get("name"),
        "confidence": float(regime_payload.get("confidence") or 0.0),
        "reasons": list(regime_payload.get("reasons") or []),
    }
