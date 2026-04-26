"""Celery task — Brooks Studio historical replay.

Drives the same :class:`~src.brooks.runtime.BrooksCore` pipeline as
:func:`~src.tasks.brooks_live_task.brooks_live_task`, but pulls bars from
:class:`~src.core.historical_crypto_stream.HistoricalCryptoStream` instead
of CCXT and pairs the throttle with a :class:`~src.brooks.runtime.clock.BarClock`
so LLM rate limits are bar-time-driven rather than wall-clock-driven.

The task runs to completion synchronously: it walks every bar in the
window, persists per-bar :class:`BarEvent`-shaped rows into
``session_logs`` (source ``brooks_bar``), and updates the session row to
``stopped`` when finished. The Studio frontend reads the timeline once
the task ends — there is no live WS streaming during replay.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Optional

from celery import Task
from loguru import logger

from session_db import SessionDB
from src.brooks.runtime import (
    BarClock,
    BrooksCore,
    PersistingSink,
)
from src.brooks.runtime.analyst_wrap import build_throttled_analyst
from src.brooks.strategy import BrooksStrategy
from src.config.settings import get_brooks_live_config
from src.core.base import DataStream
from src.core.historical_crypto_stream import HistoricalCryptoStream
from src.core.live_broker import create_live_broker
from src.tasks.celery_app import app

__all__ = ["brooks_replay_task"]


class BrooksReplayTask(Task):
    """Base task with progress meta — mirrors :class:`BrooksLiveTask`."""

    def update_progress(self, session_id: str, **extra: Any) -> None:
        self.update_state(
            state="PROGRESS",
            meta={
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                **extra,
            },
        )


@app.task(
    bind=True,
    base=BrooksReplayTask,
    name="src.tasks.brooks_replay_task.run_brooks_replay",
)
def brooks_replay_task(self, session_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a Brooks historical replay session and persist its timeline.

    ``config`` keys:

    * ``symbol`` (required) — e.g. ``"BTC/USDT"`` or ``"BTCUSDT"``
    * ``interval`` — bar interval, default from live settings (``"5m"``)
    * ``start`` / ``end`` — ISO timestamps bounding the replay window
    * ``provider`` — ClickHouse ingest provider (default ``"bitget"``)
    * ``analyst`` — analyst selector (default from settings)
    * ``analyst_params`` — forwarded to ``AnalystRegistry.build``
    * ``mtf_intervals`` — higher timeframes for HTF context
    * ``initial_cash`` / ``commission`` / ``slippage`` — broker params
    * ``min_expected_r`` — EV gate threshold
    * ``llm_min_interval_seconds`` — bar-time gap between LLM calls
    * ``stream`` — pre-built :class:`DataStream` (used by tests)
    """
    live_cfg = get_brooks_live_config()
    symbol = config["symbol"]
    interval = config.get("interval", live_cfg.default_interval)
    analyst_name = config.get("analyst", live_cfg.default_analyst)
    analyst_params = dict(config.get("analyst_params") or {})
    provider = config.get("provider", "bitget")

    session_db = SessionDB()

    # The session row is created by the API endpoint that dispatched us
    # (so polling clients can reach /session/{id}/status immediately). If
    # the row is missing we still create it as a safety net — keeps tests
    # that drive the task body directly working.
    if session_db.get_session(session_id) is None:
        try:
            session_db.create_session(
                session_id=session_id,
                strategy_name="brooks",
                symbol=symbol,
                mode="paper",
                start_date=_iso_date(config.get("start")) or date.today().isoformat(),
                end_date=_iso_date(config.get("end")),
                params={
                    "analyst": analyst_name,
                    "analyst_params": analyst_params,
                    "interval": interval,
                    "provider": provider,
                    "mode": "paper",
                    "session_kind": "replay",
                    "source": "brooks_replay",
                    "mtf_intervals": list(config.get("mtf_intervals") or []),
                    "replay_start": str(config.get("start") or ""),
                    "replay_end": str(config.get("end") or ""),
                },
                market="crypto",
                interval=interval,
            )
        except Exception as e:
            logger.warning("brooks_replay: create_session swallowed error: {}", e)

    session_db.update_session_status(session_id, "running", progress=0.0)

    broker = create_live_broker(
        mode="paper",
        initial_cash=float(config.get("initial_cash", live_cfg.initial_cash)),
        commission=float(config.get("commission", live_cfg.commission)),
        slippage=float(config.get("slippage", live_cfg.slippage)),
        allow_short=True,
        session_id=session_id,
    )

    strategy = _build_strategy(
        analyst_name=analyst_name,
        analyst_params=analyst_params,
        config=config,
        session_id=session_id,
    )

    clock = BarClock()
    llm_min_gap = float(config.get("llm_min_interval_seconds", live_cfg.llm_min_interval_seconds))
    strategy._analyst = build_throttled_analyst(
        strategy._analyst,
        min_gap_seconds=llm_min_gap,
        clock=clock,
        key_fn=_make_key_fn(interval),
    )

    stream = _build_stream(config, symbol=symbol, interval=interval, provider=provider)

    sink = PersistingSink(session_id, session_db)
    core = BrooksCore(
        strategy=strategy,
        broker=broker,
        sink=sink,
        session_id=session_id,
        session_db=session_db,
        analyst_name=analyst_name,
        clock=clock,
        equity_throttle=None,  # replay records equity every bar
        emit=None,
        recent_signals_window=int(live_cfg.recent_signals_window),
    )

    total = _try_total(stream)
    processed = 0
    try:
        while True:
            bars = stream.next_bar()
            if bars is None:
                break

            # Advance the clock so the throttle compares bar timestamps.
            first_bar = next(iter(bars.values()))
            clock.advance_to(first_bar.timestamp.timestamp())

            core.process_bar(bars)
            processed += 1

            if total and processed % 50 == 0:
                progress = round(processed / max(total, 1) * 100.0, 2)
                self.update_progress(
                    session_id,
                    bar_idx=processed,
                    total=total,
                    progress=progress,
                )
                session_db.update_session_status(session_id, "running", progress=progress)
    except Exception as exc:
        logger.exception("brooks_replay_task failed: {}", exc)
        session_db.update_session_status(session_id, "failed", error=str(exc))
        raise
    finally:
        if hasattr(stream, "stop"):
            try:
                stream.stop()
            except Exception:
                pass

    session_db.update_session_status(session_id, "stopped", progress=100.0)
    final_equity = float(broker.get_account_info().get("total_equity", 0.0))
    logger.info(
        "brooks_replay_task done: session={} bars={} trades={} equity={}",
        session_id,
        processed,
        len(core.tracked_trades),
        final_equity,
    )
    return {
        "session_id": session_id,
        "bars": processed,
        "trades": len(core.tracked_trades),
        "final_equity": final_equity,
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_strategy(
    *,
    analyst_name: str,
    analyst_params: Dict[str, Any],
    config: Dict[str, Any],
    session_id: str,
) -> BrooksStrategy:
    params: Dict[str, Any] = {
        "analyst": analyst_name,
        "analyst_params": dict(analyst_params),
        "base_interval": config.get("interval", "5m"),
        "mtf_intervals": list(config.get("mtf_intervals") or []),
    }
    if "min_expected_r" in config:
        params["min_expected_r"] = float(config["min_expected_r"])
    return BrooksStrategy(
        db_client=None,
        session_id=session_id,
        **params,
    )


def _build_stream(
    config: Dict[str, Any],
    *,
    symbol: str,
    interval: str,
    provider: str,
) -> DataStream:
    supplied = config.get("stream")
    if isinstance(supplied, DataStream):
        return supplied
    start = config.get("start")
    end = config.get("end")
    if not start or not end:
        raise ValueError("brooks_replay_task requires both 'start' and 'end' in config")
    return HistoricalCryptoStream(
        symbol=symbol,
        interval=interval,
        start=_parse_dt(start),
        end=_parse_dt(end),
        provider=provider,
    )


def _make_key_fn(interval: str):
    def key_fn(ctx: Any) -> str:
        return f"{getattr(ctx, 'symbol', '')}:{interval}"

    return key_fn


def _try_total(stream: DataStream) -> Optional[int]:
    fn = getattr(stream, "total_bars", None)
    if fn is None:
        return None
    try:
        return int(fn())
    except Exception:
        return None


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _iso_date(value: Any) -> Optional[str]:
    if not value:
        return None
    try:
        return _parse_dt(value).date().isoformat()
    except Exception:
        return None
