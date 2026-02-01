"""
WebSocket connection manager for real-time session updates.
Handles connection lifecycle, broadcasting, and heartbeat.
"""

import asyncio
import json
from typing import Dict, Set
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        # session_id -> set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # WebSocket -> session_id mapping for cleanup
        self.connection_sessions: Dict[WebSocket, str] = {}
        self.heartbeat_interval = 30  # seconds
        
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        
        self.active_connections[session_id].add(websocket)
        self.connection_sessions[websocket] = session_id
        
        count = len(self.active_connections[session_id])
        # Log only first connection per session to avoid log flood from reconnects/duplicates
        if count == 1:
            logger.info(f"WebSocket connected: session={session_id}")
        
        # Start heartbeat for this connection
        asyncio.create_task(self._heartbeat(websocket))
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        session_id = self.connection_sessions.get(websocket)
        
        if session_id and session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)
            
            # Clean up empty session
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        
        if websocket in self.connection_sessions:
            del self.connection_sessions[websocket]
        
        logger.info(f"WebSocket disconnected: session={session_id}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to a specific connection"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.disconnect(websocket)
    
    async def broadcast_to_session(self, session_id: str, message: dict):
        """Broadcast message to all connections of a session"""
        if session_id not in self.active_connections:
            return
        
        # Add timestamp
        message['timestamp'] = datetime.now().isoformat()
        
        disconnected = []
        for connection in self.active_connections[session_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to connection: {e}")
                disconnected.append(connection)
        
        # Clean up failed connections
        for connection in disconnected:
            self.disconnect(connection)
        
        if disconnected:
            logger.warning(f"Removed {len(disconnected)} failed connections from session {session_id}")
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all active connections"""
        message['timestamp'] = datetime.now().isoformat()
        
        for session_id in list(self.active_connections.keys()):
            await self.broadcast_to_session(session_id, message)
    
    async def _heartbeat(self, websocket: WebSocket):
        """Send periodic ping to keep connection alive"""
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Check if connection still exists
                if websocket not in self.connection_sessions:
                    break
                
                try:
                    await websocket.send_json({
                        'type': 'ping',
                        'timestamp': datetime.now().isoformat()
                    })
                except:
                    # Connection lost, will be cleaned up
                    break
                    
        except asyncio.CancelledError:
            pass
    
    def get_connection_count(self, session_id: str = None) -> int:
        """Get number of active connections"""
        if session_id:
            return len(self.active_connections.get(session_id, set()))
        return sum(len(conns) for conns in self.active_connections.values())
    
    def get_active_sessions(self) -> list:
        """Get list of sessions with active connections"""
        return list(self.active_connections.keys())


# Global connection manager instance
manager = ConnectionManager()


async def handle_websocket_message(websocket: WebSocket, data: dict):
    """Handle incoming WebSocket messages from client"""
    msg_type = data.get('type')
    
    if msg_type == 'pong':
        # Client responding to ping
        logger.debug("Received pong from client")
    
    elif msg_type == 'subscribe':
        session_id = data.get('session_id')
        logger.debug(f"Client subscribed to session: {session_id}")
    
    elif msg_type == 'unsubscribe':
        # Client unsubscribing
        session_id = data.get('session_id')
        logger.info(f"Client unsubscribed from session: {session_id}")
    
    else:
        logger.warning(f"Unknown message type: {msg_type}")
