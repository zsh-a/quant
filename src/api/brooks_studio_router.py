"""Brooks Studio router — single backend truth for live + replay.

The router exposes the endpoints used by the Studio panel:

* ``GET  /brooks-studio/sessions/{session_id}/timeline``           — full snapshot
* ``GET  /brooks-studio/sessions/{session_id}/timeline/since/{seq}`` — incremental page
* ``POST /brooks-studio/sessions/{session_id}/replay-bar``         — re-run analysts on one bar
* ``WS   /ws/brooks-studio/{session_id}``                          — live BarEvent stream

All four speak the same :class:`SessionTimeline` / :class:`BarEvent`
contract from :mod:`src.api.schemas.brooks_studio`, so the frontend has
exactly one rendering path for both modes.
"""

from __future__ import annotations

import anyio
import orjson
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from session_db import SessionDB
from src.api.events import studio_channel_key
from src.api.schemas.brooks_studio import (
    ReplayBarAnalystResult,
    ReplayBarRequest,
    ReplayBarResponse,
    SessionTimeline,
    TimelinePage,
)
from src.api.websocket_manager import handle_websocket_message
from src.api.websocket_manager import manager as ws_manager
from src.services.brooks_replay_runner import BrooksReplayRunner
from src.services.brooks_timeline_loader import BrooksTimelineLoader
from src.utils.logging_config import get_logger

try:
    from websockets.exceptions import ConnectionClosed as WsConnectionClosed
except ImportError:  # pragma: no cover — websockets may be missing in tests
    WsConnectionClosed = None

logger = get_logger(__name__)

router = APIRouter(prefix="/brooks-studio", tags=["brooks_studio"])
ws_router = APIRouter(tags=["brooks_studio"])


def _loader() -> BrooksTimelineLoader:
    """Build a loader bound to the default :class:`SessionDB`.

    Each call makes a fresh ``SessionDB`` (which only holds a path, not a
    live connection) so the dependency is friendly to tests that monkey-
    patch ``SESSION_DB_PATH``.
    """
    return BrooksTimelineLoader(SessionDB())


@router.get("/sessions/{session_id}/timeline", response_model=SessionTimeline)
async def get_timeline(session_id: str) -> SessionTimeline:
    try:
        return await anyio.to_thread.run_sync(_loader().load, session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")


@router.get(
    "/sessions/{session_id}/timeline/since/{seq}",
    response_model=TimelinePage,
)
async def get_timeline_page(
    session_id: str,
    seq: int,
    limit: int = Query(500, ge=1, le=5000),
) -> TimelinePage:
    if seq < 0:
        raise HTTPException(status_code=422, detail="seq must be >= 0")
    try:
        return await anyio.to_thread.run_sync(
            _loader().load_page,
            session_id,
            seq,
            limit,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")


@router.post(
    "/sessions/{session_id}/replay-bar",
    response_model=ReplayBarResponse,
)
async def replay_bar(session_id: str, req: ReplayBarRequest) -> ReplayBarResponse:
    """Re-run a chosen set of analysts against a single bar of a session.

    Powers the MultiAnalystCompare side panel: the user selects a bar and a
    set of analysts (``rule``, ``llm:claude-opus-4-7``, ``vlm:gemini``,
    ``ensemble.critic``, …); each analyst's signals + decision (when one was
    persisted) are returned for direct comparison.
    """
    runner = BrooksReplayRunner(SessionDB())
    try:
        results = await runner.run(session_id, req.bar_idx, req.analysts)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return ReplayBarResponse(
        bar_idx=req.bar_idx,
        results=[ReplayBarAnalystResult(**r.to_dict()) for r in results],
    )


@ws_router.websocket("/ws/brooks-studio/{session_id}")
async def brooks_studio_ws(websocket: WebSocket, session_id: str) -> None:
    """Per-bar :class:`BarEvent` stream for the Studio panel."""
    channel = studio_channel_key(session_id)
    await ws_manager.connect(websocket, channel)
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
        logger.info("brooks-studio ws disconnect: session={}", session_id)
    except Exception as e:  # pragma: no cover — defensive
        logger.error("brooks-studio ws error: {}", e)
        ws_manager.disconnect(websocket)


__all__ = ["router", "ws_router"]
