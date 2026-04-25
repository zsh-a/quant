"""Brooks Studio router — single backend truth for live + replay.

The router exposes three endpoints used by the Studio panel:

* ``GET  /brooks-studio/sessions/{session_id}/timeline``           — full snapshot
* ``GET  /brooks-studio/sessions/{session_id}/timeline/since/{seq}`` — incremental page
* ``WS   /ws/brooks-studio/{session_id}``                          — live BarEvent stream

All three speak the same :class:`SessionTimeline` / :class:`BarEvent`
contract from :mod:`src.api.schemas.brooks_studio`, so the frontend has
exactly one rendering path for both modes.
"""

from __future__ import annotations

import anyio
import orjson
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from session_db import SessionDB
from src.api.events import studio_channel_key
from src.api.schemas.brooks_studio import SessionTimeline, TimelinePage
from src.api.websocket_manager import handle_websocket_message
from src.api.websocket_manager import manager as ws_manager
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
