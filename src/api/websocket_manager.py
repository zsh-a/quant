"""
WebSocket connection manager for real-time session updates.
Handles connection lifecycle, broadcasting, heartbeat, and message throttling.
"""

import asyncio
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import orjson
from fastapi import WebSocket

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MessageThrottler:
    """Throttles and batches WebSocket messages to prevent flooding"""

    def __init__(
        self,
        min_interval_ms: int = 100,
        max_batch_size: int = 50,
        flush_interval_ms: int = 200,
    ):
        self.min_interval_ms = min_interval_ms
        self.max_batch_size = max_batch_size
        self.flush_interval_ms = flush_interval_ms

        self._queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._last_send: Dict[str, float] = {}
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._flush_tasks: Dict[str, asyncio.Task] = {}

    async def add_message(
        self, session_id: str, message: Dict[str, Any], callback
    ) -> None:
        """Add message to throttle queue"""
        async with self._locks[session_id]:
            self._queues[session_id].append(message)

            # Check if we should flush immediately
            if len(self._queues[session_id]) >= self.max_batch_size:
                await self._flush(session_id, callback)
                return

            # Schedule delayed flush if not already scheduled
            if (
                session_id not in self._flush_tasks
                or self._flush_tasks[session_id].done()
            ):
                self._flush_tasks[session_id] = asyncio.create_task(
                    self._delayed_flush(session_id, callback)
                )

    async def _delayed_flush(self, session_id: str, callback) -> None:
        """Flush queue after delay"""
        await asyncio.sleep(self.flush_interval_ms / 1000)
        async with self._locks[session_id]:
            await self._flush(session_id, callback)

    async def _flush(self, session_id: str, callback) -> None:
        """Flush all queued messages for a session"""
        if not self._queues[session_id]:
            return

        now = time.time() * 1000
        last = self._last_send.get(session_id, 0)

        if now - last < self.min_interval_ms:
            return

        messages = self._queues[session_id]
        self._queues[session_id] = []
        self._last_send[session_id] = now

        # Batch messages by type
        batched = self._batch_messages(messages)

        for batch in batched:
            await callback(session_id, batch)

    def _batch_messages(self, messages: List[Dict]) -> List[Dict]:
        """Batch multiple messages into fewer messages"""
        if len(messages) <= 1:
            return messages

        # Group by message type
        by_type: Dict[str, List[Dict]] = defaultdict(list)
        for msg in messages:
            msg_type = msg.get("type", "unknown")
            by_type[msg_type].append(msg)

        result = []

        for msg_type, msgs in by_type.items():
            if msg_type == "equity_update" and len(msgs) > 1:
                # Batch equity updates into single message
                result.append(
                    {
                        "type": "equity_batch",
                        "data": {
                            "updates": [
                                m.get("data", {}).get("equity", m.get("data", {}))
                                for m in msgs
                            ]
                        },
                        "count": len(msgs),
                    }
                )
            elif msg_type == "trade_executed" and len(msgs) > 1:
                # Batch trade updates
                result.append(
                    {
                        "type": "trades_batch",
                        "data": {
                            "trades": [
                                m.get("data", {}).get("trade", m.get("data", {}))
                                for m in msgs
                            ]
                        },
                        "count": len(msgs),
                    }
                )
            else:
                # Keep individual messages for other types
                result.extend(msgs)

        return result

    def clear(self, session_id: str = None) -> None:
        """Clear queued messages"""
        if session_id:
            self._queues[session_id] = []
            if session_id in self._flush_tasks:
                self._flush_tasks[session_id].cancel()
        else:
            self._queues.clear()
            for task in self._flush_tasks.values():
                task.cancel()
            self._flush_tasks.clear()


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_sessions: Dict[WebSocket, str] = {}
        self.heartbeat_interval = 30
        self.throttler = MessageThrottler(
            min_interval_ms=100, max_batch_size=50, flush_interval_ms=200
        )

    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()

        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()

        self.active_connections[session_id].add(websocket)
        self.connection_sessions[websocket] = session_id

        count = len(self.active_connections[session_id])
        if count == 1:
            logger.info(f"WebSocket connected: session={session_id}")

        asyncio.create_task(self._heartbeat(websocket))

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        session_id = self.connection_sessions.get(websocket)

        if session_id and session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)

            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
                self.throttler.clear(session_id)

        if websocket in self.connection_sessions:
            del self.connection_sessions[websocket]

        logger.info(f"WebSocket disconnected: session={session_id}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to a specific connection"""
        try:
            await websocket.send_text(orjson.dumps(message).decode())
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.disconnect(websocket)

    async def broadcast_to_session(self, session_id: str, message: dict):
        """Broadcast message to all connections of a session"""
        if session_id not in self.active_connections:
            return

        message["timestamp"] = datetime.now().isoformat()

        disconnected = []
        for connection in self.active_connections[session_id]:
            try:
                await connection.send_text(orjson.dumps(message).decode())
            except Exception as e:
                logger.error(f"Failed to broadcast to connection: {e}")
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)

        if disconnected:
            logger.warning(
                f"Removed {len(disconnected)} failed connections from session {session_id}"
            )

    async def broadcast_throttled(self, session_id: str, message: dict):
        """Broadcast with throttling to prevent message flooding"""
        await self.throttler.add_message(session_id, message, self._do_broadcast)

    async def _do_broadcast(self, session_id: str, message: dict):
        """Internal broadcast callback for throttler"""
        await self.broadcast_to_session(session_id, message)

    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all active connections"""
        message["timestamp"] = datetime.now().isoformat()

        for session_id in list(self.active_connections.keys()):
            await self.broadcast_to_session(session_id, message)

    async def _heartbeat(self, websocket: WebSocket):
        """Send periodic ping to keep connection alive"""
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval)

                if websocket not in self.connection_sessions:
                    break

                try:
                    await websocket.send_text(orjson.dumps(
                        {"type": "ping", "timestamp": datetime.now().isoformat()}
                    ).decode())
                except Exception:
                    break

        except asyncio.CancelledError:
            pass

    def get_connection_count(self, session_id: Optional[str] = None) -> int:
        """Get number of active connections"""
        if session_id:
            return len(self.active_connections.get(session_id, set()))
        return sum(len(conns) for conns in self.active_connections.values())

    def get_active_sessions(self) -> list:
        """Get list of sessions with active connections"""
        return list(self.active_connections.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": self.get_connection_count(),
            "active_sessions": len(self.active_connections),
            "sessions": {
                sid: len(conns) for sid, conns in self.active_connections.items()
            },
        }


manager = ConnectionManager()


async def handle_websocket_message(websocket: WebSocket, data: dict):
    """Handle incoming WebSocket messages from client"""
    msg_type = data.get("type")

    if msg_type == "pong":
        logger.debug("Received pong from client")

    elif msg_type == "subscribe":
        session_id = data.get("session_id")
        logger.debug(f"Client subscribed to session: {session_id}")

    elif msg_type == "unsubscribe":
        session_id = data.get("session_id")
        logger.info(f"Client unsubscribed from session: {session_id}")

    else:
        logger.warning(f"Unknown message type: {msg_type}")
