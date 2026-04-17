"""
Event system for session state changes.
Triggers WebSocket broadcasts when session state updates.
"""

import asyncio
from datetime import datetime
from typing import Callable, Dict, List

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class EventType:
    """Event type constants"""

    SESSION_STARTED = "session_started"
    SESSION_PROGRESS = "session_progress"
    SESSION_COMPLETED = "session_completed"
    SESSION_FAILED = "session_failed"
    SESSION_STOPPED = "session_stopped"
    TRADE_EXECUTED = "trade_executed"
    EQUITY_UPDATE = "equity_update"
    ERROR_OCCURRED = "error_occurred"
    DATA_UPDATE_STARTED = "data_update_started"
    DATA_UPDATE_PROGRESS = "data_update_progress"
    DATA_UPDATE_COMPLETED = "data_update_completed"
    DATA_UPDATE_FAILED = "data_update_failed"
    SIMULATION_BATCH_STARTED = "simulation_batch_started"
    SIMULATION_BATCH_PROGRESS = "simulation_batch_progress"
    SIMULATION_BATCH_COMPLETED = "simulation_batch_completed"
    SIMULATION_BATCH_FAILED = "simulation_batch_failed"
    STRATEGY_STEP = "strategy_step"


class SessionEventBus:
    """Event bus for session state changes"""

    def __init__(self):
        # event_type -> list of callback functions
        self.listeners: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type"""
        if event_type not in self.listeners:
            self.listeners[event_type] = []

        self.listeners[event_type].append(callback)
        logger.debug(f"Subscribed to event: {event_type}")

    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from an event type"""
        if event_type in self.listeners:
            self.listeners[event_type].remove(callback)

    async def emit(self, event_type: str, session_id: str, data: dict):
        """Emit an event to all subscribers"""
        if event_type not in self.listeners:
            return

        event_data = {
            "type": event_type,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }

        logger.debug(f"Emitting event: {event_type} for session {session_id}")

        for callback in self.listeners[event_type]:
            try:
                # Support both sync and async callbacks
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_data)
                else:
                    callback(event_data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}", exc_info=True)

    def clear(self):
        """Clear all listeners"""
        self.listeners.clear()


# Global event bus instance
event_bus = SessionEventBus()


# Helper functions for common events
async def emit_session_started(session_id: str, strategy: str, symbol: str):
    """Emit session started event"""
    await event_bus.emit(EventType.SESSION_STARTED, session_id, {"strategy": strategy, "symbol": symbol})


async def emit_session_progress(session_id: str, progress: float, status: str):
    """Emit session progress event"""
    await event_bus.emit(EventType.SESSION_PROGRESS, session_id, {"progress": progress, "status": status})


async def emit_session_completed(session_id: str, final_equity: float, total_trades: int):
    """Emit session completed event"""
    await event_bus.emit(
        EventType.SESSION_COMPLETED, session_id, {"final_equity": final_equity, "total_trades": total_trades}
    )


async def emit_session_failed(session_id: str, error: str):
    """Emit session failed event"""
    await event_bus.emit(EventType.SESSION_FAILED, session_id, {"error": error})


async def emit_session_stopped(session_id: str):
    """Emit session stopped event"""
    await event_bus.emit(EventType.SESSION_STOPPED, session_id, {})


async def emit_trade_executed(session_id: str, trade: dict):
    """Emit trade executed event"""
    await event_bus.emit(EventType.TRADE_EXECUTED, session_id, {"trade": trade})


async def emit_equity_update(session_id: str, equity_point: dict):
    """Emit equity update event"""
    await event_bus.emit(EventType.EQUITY_UPDATE, session_id, {"equity": equity_point})


async def emit_error(session_id: str, error: str):
    """Emit error event"""
    await event_bus.emit(EventType.ERROR_OCCURRED, session_id, {"error": error})


async def emit_data_update_started(session_id: str, payload: dict):
    await event_bus.emit(EventType.DATA_UPDATE_STARTED, session_id, payload)


async def emit_data_update_progress(session_id: str, payload: dict):
    await event_bus.emit(EventType.DATA_UPDATE_PROGRESS, session_id, payload)


async def emit_data_update_completed(session_id: str, payload: dict):
    await event_bus.emit(EventType.DATA_UPDATE_COMPLETED, session_id, payload)


async def emit_data_update_failed(session_id: str, payload: dict):
    await event_bus.emit(EventType.DATA_UPDATE_FAILED, session_id, payload)


async def emit_simulation_batch_started(session_id: str, payload: dict):
    await event_bus.emit(EventType.SIMULATION_BATCH_STARTED, session_id, payload)


async def emit_simulation_batch_progress(session_id: str, payload: dict):
    await event_bus.emit(EventType.SIMULATION_BATCH_PROGRESS, session_id, payload)


async def emit_simulation_batch_completed(session_id: str, payload: dict):
    await event_bus.emit(EventType.SIMULATION_BATCH_COMPLETED, session_id, payload)


async def emit_simulation_batch_failed(session_id: str, payload: dict):
    await event_bus.emit(EventType.SIMULATION_BATCH_FAILED, session_id, payload)


async def emit_strategy_step(session_id: str, payload: dict):
    await event_bus.emit(EventType.STRATEGY_STEP, session_id, payload)
