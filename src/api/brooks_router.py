"""Phase 4.6 BrooksLive router — start/stop paper-trading + realtime WS.

Exposes:

* ``POST   /brooks-live/start``              start a new paper session
* ``POST   /brooks-live/{session_id}/stop``  stop a running session
* ``POST   /brooks-live/{session_id}/switch`` hot-swap the analyst
* ``GET    /brooks-live/{session_id}/state`` snapshot for panel bootstrap
* ``GET    /brooks-live/sessions``           list active paper sessions
* ``WS     /ws/brooks/{session_id}``         per-bar realtime stream

The WebSocket endpoint reuses the global
:class:`~src.api.websocket_manager.ConnectionManager` — events emitted by
:func:`~src.tasks.brooks_live_task.brooks_live_task` flow through the
existing event-bus → broadcast pipeline, so the only extra plumbing is
the connection accept loop here.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import orjson
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from src.api.websocket_manager import handle_websocket_message
from src.api.websocket_manager import manager as ws_manager
from src.brooks.analyst.base import AnalystRegistry
from src.config.settings import get_brooks_live_config
from src.tasks.brooks_live_task import (
    BrooksLiveRegistry,
    brooks_live_close_task,
    brooks_live_task,
)
from src.utils.logging_config import get_logger

try:
    from websockets.exceptions import ConnectionClosed as WsConnectionClosed
except ImportError:  # pragma: no cover — websockets may be missing in tests
    WsConnectionClosed = None

logger = get_logger(__name__)

router = APIRouter(prefix="/brooks-live", tags=["brooks_live"])
ws_router = APIRouter(tags=["brooks_live"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class StartRequest(BaseModel):
    symbol: str = Field(description="Trading symbol (e.g. 'BTC/USDT')")
    interval: Optional[str] = None
    exchange: Optional[str] = None
    analyst: Optional[str] = None
    analyst_params: Dict[str, Any] = Field(default_factory=dict)
    mtf_intervals: List[str] = Field(default_factory=list)
    initial_cash: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    min_expected_r: Optional[float] = None
    mode: str = Field(default="paper", description="Paper-only in Phase 4.6")


class StartResponse(BaseModel):
    session_id: str
    task_id: str
    status: str = "submitted"


class SwitchRequest(BaseModel):
    analyst: str


class BrooksSessionState(BaseModel):
    session_id: str
    analyst: str
    started_at: str
    config: Dict[str, Any]
    last_regime: Dict[str, Any]
    last_decision: Dict[str, Any]
    recent_signals: List[Dict[str, Any]]
    equity_points: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]


class SessionsResponse(BaseModel):
    sessions: List[BrooksSessionState]


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@router.get("/analysts")
async def list_analysts() -> Dict[str, List[str]]:
    """Return the names of every analyst currently registered."""
    return {"analysts": AnalystRegistry.all()}


@router.post("/start", response_model=StartResponse)
async def start_session(req: StartRequest) -> StartResponse:
    if req.mode != "paper":
        raise HTTPException(status_code=400, detail="Phase 4.6 supports mode='paper' only")
    cfg = get_brooks_live_config()
    if req.analyst and req.analyst not in AnalystRegistry.all():
        raise HTTPException(status_code=400, detail=f"Unknown analyst: {req.analyst!r}")

    session_id = str(uuid.uuid4())
    payload: Dict[str, Any] = {
        "symbol": req.symbol,
        "interval": req.interval or cfg.default_interval,
        "exchange": req.exchange or cfg.default_exchange,
        "analyst": req.analyst or cfg.default_analyst,
        "analyst_params": req.analyst_params,
        "mtf_intervals": req.mtf_intervals,
        "initial_cash": req.initial_cash if req.initial_cash is not None else cfg.initial_cash,
        "commission": req.commission if req.commission is not None else cfg.commission,
        "slippage": req.slippage if req.slippage is not None else cfg.slippage,
        "mode": "paper",
    }
    if req.min_expected_r is not None:
        payload["min_expected_r"] = req.min_expected_r

    task = brooks_live_task.apply_async(args=[session_id, payload], queue="automation")
    logger.info("brooks-live start: session={} task={} payload={}", session_id, task.id, payload)
    return StartResponse(session_id=session_id, task_id=task.id)


@router.post("/{session_id}/stop")
async def stop_session(session_id: str) -> Dict[str, str]:
    sess = BrooksLiveRegistry.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found or not running in this worker")
    sess.request_stop()
    return {"session_id": session_id, "status": "stop_requested"}


@router.post("/{session_id}/switch")
async def switch_analyst(session_id: str, req: SwitchRequest) -> Dict[str, str]:
    if req.analyst not in AnalystRegistry.all():
        raise HTTPException(status_code=400, detail=f"Unknown analyst: {req.analyst!r}")
    sess = BrooksLiveRegistry.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found or not running in this worker")
    sess.request_switch(req.analyst)
    return {"session_id": session_id, "requested_analyst": req.analyst, "status": "switch_requested"}


@router.get("/{session_id}/state", response_model=BrooksSessionState)
async def session_state(session_id: str) -> BrooksSessionState:
    sess = BrooksLiveRegistry.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found or not running in this worker")
    return BrooksSessionState(
        session_id=sess.session_id,
        analyst=sess.analyst_name,
        started_at=sess.started_at,
        config=sess.config,
        last_regime=sess.last_regime,
        last_decision=sess.last_decision,
        recent_signals=list(sess.last_signals),
        equity_points=list(sess.equity_points),
        trades=list(sess.trades),
    )


@router.get("/sessions", response_model=SessionsResponse)
async def list_sessions() -> SessionsResponse:
    sessions = [
        BrooksSessionState(
            session_id=s.session_id,
            analyst=s.analyst_name,
            started_at=s.started_at,
            config=s.config,
            last_regime=s.last_regime,
            last_decision=s.last_decision,
            recent_signals=list(s.last_signals),
            equity_points=list(s.equity_points),
            trades=list(s.trades),
        )
        for s in BrooksLiveRegistry.all()
    ]
    return SessionsResponse(sessions=sessions)


@router.post("/close-day")
async def close_day(session_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Trigger :func:`brooks_live_close_task` on demand (mirrors celery-beat job)."""
    task = brooks_live_close_task.apply_async(args=[session_ids], queue="automation")
    return {"task_id": task.id, "status": "submitted"}


# ---------------------------------------------------------------------------
# WebSocket endpoint — reuses the shared ConnectionManager
# ---------------------------------------------------------------------------


@ws_router.websocket("/ws/brooks/{session_id}")
async def brooks_live_ws(websocket: WebSocket, session_id: str) -> None:
    """Realtime channel for the BrooksLive panel.

    Reuses the global :class:`ConnectionManager`; events posted via
    ``emit_strategy_step`` / ``emit_equity_update`` / ``emit_trade_executed``
    from the Celery task reach every connected client.
    """
    await ws_manager.connect(websocket, session_id)
    closed_exc = (WebSocketDisconnect,)
    if WsConnectionClosed is not None:
        closed_exc = (WebSocketDisconnect, WsConnectionClosed)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = orjson.loads(raw)
            except Exception:
                continue
            await handle_websocket_message(websocket, data)
    except closed_exc:
        ws_manager.disconnect(websocket)
        logger.info("brooks-live ws disconnect: session={}", session_id)
    except Exception as e:
        logger.error("brooks-live ws error: {}", e)
        ws_manager.disconnect(websocket)


__all__ = ["router", "ws_router"]
