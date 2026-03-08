"""
Logs API Router - API endpoints for session debug logs.
"""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from typing import Optional

from src.utils.session_logger import (
    get_all_logs,
    get_session_logs as fetch_session_logs,
    clear_session_logs as purge_session_logs,
    list_session_loggers,
)

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("")
async def list_sessions_with_logs():
    """List all sessions that have logs"""
    sessions = list_session_loggers()
    return {
        "sessions": sessions,
        "count": len(sessions)
    }


@router.get("/{session_id}")
async def get_session_logs(
    session_id: str,
    level: Optional[str] = None,
    source: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 500,
    format: str = "text"  # text or json
):
    """
    Get logs for a session.
    
    Args:
        session_id: Session ID
        level: Filter by log level (DEBUG, INFO, WARNING, ERROR)
        source: Filter by source (strategy, broker, engine)
        since: ISO timestamp to filter logs after
        limit: Maximum number of entries to return (default 500)
        format: Response format - 'text' for log viewer, 'json' for structured data
    """
    if format == "json":
        logs = fetch_session_logs(
            session_id,
            level=level,
            source=source,
            since=since,
            limit=limit,
        )
        if not logs:
            return {"logs": [], "count": 0}
        return {
            "session_id": session_id,
            "logs": logs,
            "count": len(logs)
        }
    else:
        # Return plain text for log viewer
        text = get_all_logs(session_id, level=level, source=source, since=since, limit=limit)
        return PlainTextResponse(content=text, media_type="text/plain")


@router.delete("/{session_id}")
async def clear_session_logs(session_id: str):
    """Clear logs for a session"""
    purge_session_logs(session_id)
    return {"message": f"Logs cleared for session {session_id}"}


@router.get("/{session_id}/stream")
async def stream_logs_info(session_id: str):
    """
    Return WebSocket URL for streaming logs.
    Actual streaming is handled by the main WebSocket endpoint.
    """
    return {
        "session_id": session_id,
        "ws_url": f"/ws/session/{session_id}/logs",
        "message": "Connect to WebSocket for live log streaming"
    }
