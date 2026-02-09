from fastapi import (
    FastAPI,
    BackgroundTasks,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)

try:
    from websockets.exceptions import ConnectionClosed as WsConnectionClosed
except ImportError:
    WsConnectionClosed = None
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
import uuid
import threading
from datetime import datetime
import asyncio
import time
import pandas as pd
import anyio

# Patch requests timeout globally to prevent 20s stalls
import requests
from functools import partial
requests.get = partial(requests.get, timeout=5)
requests.post = partial(requests.get, timeout=5)

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.engine import TradingEngine
from src.core.backtest_broker import BacktestBroker
from src.core.live_broker import LiveBroker
from src.core.data_stream import DBDataStream, RealtimeDataStream
from src.strategies.registry import StrategyRegistry
from src.utils.cache import get_cache, get_backtest_cache
from src.config.settings import (
    get_data_stream_config,
    get_broker_config,
    get_api_config,
    get_settings,
)
from src.utils.logging_config import setup_logging, get_logger
from src.api.websocket_manager import manager as ws_manager, handle_websocket_message
from src.api.events import (
    event_bus,
    EventType,
    emit_equity_update,
    emit_session_progress,
    emit_trade_executed,
    emit_session_completed,
    emit_error,
)
from src.api.state_persistence import persistence
from src.analysis.backtest_metrics import calculate_metrics as calc_perf_metrics
from db import DB
from session_db import SessionDB
from src.api.tasks_router import router as tasks_router
from src.api.monitoring_router import router as monitoring_router
from src.api.portfolio_router import router as portfolio_router
from src.api.optimizer_router import router as optimizer_router
from src.api.analysis_router import router as analysis_router
from src.api.logs_router import router as logs_router
from src.api.market_router import router as market_router
from src.tasks.backtest import run_backtest_task

setup_logging()
logger = get_logger(__name__)

settings = get_settings()
api_config = get_api_config()
data_stream_config = get_data_stream_config()
broker_config = get_broker_config()

app = FastAPI()
session_db = SessionDB()

app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.cors_origins
    if hasattr(api_config, "cors_origins")
    else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(tasks_router)
app.include_router(monitoring_router)
app.include_router(portfolio_router)
app.include_router(optimizer_router)
app.include_router(analysis_router)
app.include_router(logs_router)
app.include_router(market_router)

logger.info(f"API Server starting with config: port={api_config.port}")


class SessionRequest(BaseModel):
    strategy: str
    symbol: str
    start_date: str
    end_date: Optional[str] = None
    mode: str = "backtest"  # backtest, simulation, live
    params: Optional[Dict[str, Any]] = {}


class Session:
    def __init__(
        self,
        session_id: str,
        strategy_name: str,
        symbol: str,
        mode: str,
        start_date: str,
        end_date: Optional[str],
        params: Dict[str, Any] = {},
    ):
        self.session_id = session_id
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.mode = mode
        self.start_date = start_date
        self.end_date = end_date
        self.params = params
        self.status = "starting"
        self.progress = 0.0
        # self.equity_history = [] # Removed to save memory, use DB
        # self.trades = [] # Removed to save memory, use DB
        self.positions = {}
        self.metrics = {}
        self.error = None
        self.engine = None
        self.broker = None


SESSIONS: Dict[str, Session] = {}

# Mock live server URL - in real scenario this might be config
LIVE_SERVER_URL = "http://localhost:11122"

# Register strategies once at startup
StrategyRegistry.register_all()


@app.get("/strategies")
async def get_strategies():
    return StrategyRegistry.list_strategies()


@app.post("/session/run")
async def run_session(req: SessionRequest, background_tasks: BackgroundTasks):
    print(f"Running session: {req}")
    session_id = str(uuid.uuid4())
    session = Session(
        session_id,
        req.strategy,
        req.symbol,
        req.mode,
        req.start_date,
        req.end_date,
        req.params,
    )
    SESSIONS[session_id] = session

    # Persist initial session state
    session_db.create_session(
        session_id, req.strategy, req.symbol, req.mode, req.start_date, req.end_date
    )
    # Note: params persistence in DB is not yet implemented in session_db, but it's okay for now.

    loop = asyncio.get_running_loop()

    def execute_session_task():
        try:
            db_client = DB()
            session.status = "running"
            session_db.update_session_status(session_id, "running")

            # Setup data stream (Chunked automatically by DBDataStream optimization)
            symbols = [req.symbol]

            if req.mode == "live":
                stream = RealtimeDataStream(
                    symbols,
                    interval_seconds=data_stream_config.realtime.interval_seconds,
                    data_source=data_stream_config.realtime.data_source,
                    enable_trading_hours_check=True,
                )
                total_bars = 0  # Live stream is indefinite
            else:
                stream = DBDataStream(
                    db_client,
                    symbols,
                    req.start_date,
                    req.end_date,
                    chunk_size_months=data_stream_config.chunk_size_months,
                )
                total_bars = getattr(stream, "total_bars", 1)

            # Setup broker
            if req.mode == "live":
                broker = LiveBroker(server_url=broker_config.live.server_url)
            else:
                broker = BacktestBroker(
                    db_client=db_client,
                    initial_cash=broker_config.backtest.initial_cash,
                    commission=broker_config.backtest.commission,
                )

            session.broker = broker

            # Setup strategy with params - use registry
            strategy = StrategyRegistry.create_strategy(
                req.strategy,
                db_client,
                session_id=session.session_id,
                **(req.params or {}),
            )

            if strategy is None:
                raise ValueError(f"Unknown strategy: {req.strategy}")


# ... inside execute_session_task ...
            # Progress callback
            last_db_write = time.time()

            def on_step(bars):
                nonlocal last_db_write
                
                # Update progress
                if req.mode == "backtest" and total_bars > 0:
                    session.progress = (stream.idx / total_bars) * 100
                else:
                    session.progress = 50.0

                # Throttle DB writes to every 2 seconds
                current_time = time.time()
                if current_time - last_db_write >= 2.0 or session.progress >= 100:
                    info = broker.get_account_info()

                    new_equity_points = info.get("equity_history", [])
                    if new_equity_points:
                        session_db.add_equity_points(session_id, new_equity_points)
                        for pt in new_equity_points:
                            # Keep latest positions in memory for quick access
                            session.positions = pt.get("positions", {})
                            # Broadcast to WebSocket
                            asyncio.run_coroutine_threadsafe(
                                emit_equity_update(session_id, pt), loop
                            )

                        if isinstance(broker, BacktestBroker):
                            broker.equity_history.clear()
                        
                        asyncio.run_coroutine_threadsafe(
                            emit_session_progress(
                                session_id, session.progress, session.status
                            ),
                            loop,
                        )

                    new_trades = info.get("trades", [])
                    if new_trades:
                        session_db.add_trades(session_id, new_trades)
                        for trade in new_trades:
                            asyncio.run_coroutine_threadsafe(
                                emit_trade_executed(session_id, trade), loop
                            )

                        if isinstance(broker, BacktestBroker):
                            broker.trades.clear()
                    
                    last_db_write = current_time

                # For simulation mode, we might want to slow down
                if req.mode == "simulation":
                    time.sleep(1)  # Simulate 1 second per bar

            # Initialize risk manager if enabled
            risk_manager = None
            if False:
                from src.core.risk_manager import RiskManager

                initial_capital = (
                    broker_config.backtest.initial_cash
                    if req.mode != "live"
                    else 1000000.0
                )
                risk_manager = RiskManager(initial_capital=initial_capital)
                logger.info(f"Risk manager initialized for session {session_id}")

                if isinstance(broker, BacktestBroker):
                    broker.risk_manager = risk_manager
                    logger.info(f"Risk manager attached to BacktestBroker")

            engine = TradingEngine(
                strategy, broker, stream, on_step=on_step, risk_manager=risk_manager
            )
            session.engine = engine
            engine.run()

            session.status = "completed"
            session.progress = 100.0
            session_db.update_session_status(session_id, "completed", 100.0)
            # Notify WebSocket clients so frontend marks session complete
            eq_hist = session_db.get_equity_history(session_id)
            final_equity = eq_hist[-1]["total_equity"] if eq_hist else 0.0
            trades_list = session_db.get_trades(session_id)
            asyncio.run_coroutine_threadsafe(
                emit_session_completed(session_id, final_equity, len(trades_list)), loop
            )

        except Exception as e:
            session.status = "failed"
            session.error = str(e)
            session_db.update_session_status(session_id, "failed", error=str(e))
            print(f"Session failed: {e}")
            asyncio.run_coroutine_threadsafe(emit_error(session_id, str(e)), loop)

    background_tasks.add_task(execute_session_task)
    return {"session_id": session_id}


@app.post("/session/run_async")
async def run_session_async(req: SessionRequest):
    """
    Submit a backtest session to Celery queue for async processing.
    Returns immediately with session_id and task_id for progress tracking.
    """
    session_id = str(uuid.uuid4())

    # Persist initial session state
    session_db.create_session(
        session_id, req.strategy, req.symbol, req.mode, req.start_date, req.end_date
    )

    # Build config for Celery task
    config = {
        "symbol": req.symbol,
        "strategy": req.strategy,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "params": req.params or {},
        "initial_cash": broker_config.backtest.initial_cash,
        "commission": broker_config.backtest.commission,
        "enable_risk_management": False,
        "chunk_size_months": data_stream_config.chunk_size_months,
    }

    # Submit to Celery
    task = run_backtest_task.apply_async(args=[session_id, config], queue="backtest")

    logger.info(f"Async backtest submitted: session={session_id}, task={task.id}")

    return {
        "session_id": session_id,
        "task_id": task.id,
        "status": "submitted",
        "message": "Backtest submitted to queue",
    }


@app.get("/sessions")
async def get_sessions():
    # Return sessions from DB (persistent)
    return await anyio.to_thread.run_sync(session_db.get_all_sessions)


@app.get("/session/{session_id}/risk")
async def get_session_risk(session_id: str):
    """Risk metrics and alerts for a session. Returns 404 if session not found."""
    s_mem = SESSIONS.get(session_id)
    s_db = await anyio.to_thread.run_sync(session_db.get_session, session_id) if not s_mem else None
    if not s_mem and not s_db:
        raise HTTPException(status_code=404, detail="Session not found")
    # Risk manager state is not persisted; return safe default so RiskPanel doesn't 404
    return {
        "enabled": bool(
            s_mem
            and getattr(s_mem, "engine", None)
            and getattr(s_mem.engine, "risk_manager", None)
        ),
        "metrics": {},
        "limits": {},
        "alerts": [],
    }


@app.get("/session/{session_id}/status")
async def get_session_status(
    session_id: str,
    since: Optional[str] = Query(None, description="Return data since this timestamp"),
):
    # Try to find in memory first for running status
    s_mem = SESSIONS.get(session_id)

    # Get basic info from DB or Memory
    if s_mem:
        status = s_mem.status
        mode = s_mem.mode
        progress = s_mem.progress
        error = s_mem.error
        start_date = s_mem.start_date
        end_date = s_mem.end_date
        positions = s_mem.positions
    else:
        # Fallback to DB
        s_db = await anyio.to_thread.run_sync(session_db.get_session, session_id)
        if not s_db:
            raise HTTPException(status_code=404, detail="Session not found")
        status = s_db["status"]
        mode = s_db["mode"]
        progress = s_db["progress"]
        error = s_db["error"]
        start_date = s_db["start_date"]
        end_date = s_db["end_date"]
        positions = {}  # Positions history not fully persisted in simple DB yet, only snapshots in equity?
        # Actually equity_history doesn't store full positions in DB in my schema (simplified).
        # So for finished sessions, positions might be empty unless we store final state.
        # For now, acceptable compromise.

    equity_history = await anyio.to_thread.run_sync(
        session_db.get_equity_history, session_id, since
    )
    trades = await anyio.to_thread.run_sync(session_db.get_trades, session_id, since)

    return {
        "status": status,
        "mode": mode,
        "progress": progress,
        "equity_history": equity_history,
        "trades": trades,
        "positions": positions,
        "error": error,
        "start_date": start_date,
        "end_date": end_date,
    }


@app.get("/session/{session_id}/metrics")
async def get_session_metrics(session_id: str):
    """
    Calculate and return performance metrics for a session.
    Uses unified metrics calculation for consistency with frontend.
    """
    # Get equity history and trades
    equity_history = await anyio.to_thread.run_sync(
        session_db.get_equity_history, session_id
    )
    trades = await anyio.to_thread.run_sync(session_db.get_trades, session_id)

    if not equity_history:
        raise HTTPException(
            status_code=404, detail="No equity history found for session"
        )

    # Calculate metrics using unified service
    metrics = calc_perf_metrics(equity_history, trades)

    return {"session_id": session_id, "metrics": metrics.to_dict()}


@app.post("/session/{session_id}/stop")
async def stop_session(session_id: str):
    if session_id in SESSIONS:
        s = SESSIONS[session_id]
        if s.engine:
            s.engine.stop()
        s.status = "stopped"
        session_db.update_session_status(session_id, "stopped")
        return {"status": "stopped"}

    # If not in memory (e.g. restarted), update DB just in case
    s_db = session_db.get_session(session_id)
    if s_db and s_db["status"] == "running":
        session_db.update_session_status(session_id, "stopped")
        return {"status": "stopped (db updated)"}

    raise HTTPException(status_code=404, detail="Session not found or not running")


@app.get("/market/benchmark")
async def get_benchmark(symbol: str, start_date: str, end_date: Optional[str] = None):
    try:
        db = DB()
        df = db.get_kline(symbol, start_date, end_date)
        if df.empty:
            return []

        result = []
        for ts, row in df.iterrows():
            result.append({"timestamp": str(ts), "value": row["close"]})
        return result
    except Exception as e:
        print(f"Error fetching benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time updates
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time session updates"""
    await ws_manager.connect(websocket, session_id)
    closed_exc = (WebSocketDisconnect,)
    if WsConnectionClosed is not None:
        closed_exc = (WebSocketDisconnect, WsConnectionClosed)
    try:
        while True:
            data = await websocket.receive_json()
            await handle_websocket_message(websocket, data)
    except closed_exc:
        ws_manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


# Setup event listeners to broadcast to WebSocket clients
async def broadcast_session_event(event_data: dict):
    """Broadcast session events to WebSocket clients"""
    session_id = event_data.get("session_id")
    if session_id:
        await ws_manager.broadcast_to_session(session_id, event_data)


# Register event listeners
event_bus.subscribe(EventType.SESSION_STARTED, broadcast_session_event)
event_bus.subscribe(EventType.SESSION_PROGRESS, broadcast_session_event)
event_bus.subscribe(EventType.SESSION_COMPLETED, broadcast_session_event)
event_bus.subscribe(EventType.SESSION_FAILED, broadcast_session_event)
event_bus.subscribe(EventType.TRADE_EXECUTED, broadcast_session_event)
event_bus.subscribe(EventType.EQUITY_UPDATE, broadcast_session_event)
event_bus.subscribe(EventType.ERROR_OCCURRED, broadcast_session_event)

logger.info("WebSocket event listeners registered")


# State persistence and recovery endpoints
@app.post("/session/{session_id}/checkpoint")
async def create_checkpoint(session_id: str):
    """Create a checkpoint for a session"""
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Collect session state
    state = {
        "session_id": session_id,
        "strategy": session.strategy_name,
        "symbol": session.symbol,
        "mode": session.mode,
        "status": session.status,
        "progress": session.progress,
        "equity_history": [
            {"timestamp": str(e["timestamp"]), "value": e["value"]}
            for e in session.equity_history
        ],
        "trades": session.trades,
        "positions": session.positions,
        "start_date": session.start_date,
        "end_date": session.end_date,
        "params": session.params,
    }

    metadata = {
        "checkpoint_type": "manual",
        "status": session.status,
        "progress": session.progress,
    }

    success = persistence.save_checkpoint(session_id, state, metadata)

    if success:
        return {"message": "Checkpoint created", "session_id": session_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to create checkpoint")


@app.get("/session/{session_id}/checkpoints")
async def list_checkpoints(session_id: str):
    """List all checkpoints for a session"""
    checkpoints = persistence.list_checkpoints(session_id)
    return {"session_id": session_id, "checkpoints": checkpoints}


@app.post("/session/{session_id}/restore")
async def restore_session(session_id: str):
    """Restore a session from the latest checkpoint"""
    checkpoint = persistence.load_latest_checkpoint(session_id)

    if not checkpoint:
        raise HTTPException(status_code=404, detail="No checkpoint found")

    state = checkpoint["state"]

    # Restore session
    session = Session(
        session_id=state["session_id"],
        strategy_name=state["strategy"],
        symbol=state["symbol"],
        mode=state["mode"],
        start_date=state["start_date"],
        end_date=state.get("end_date"),
        params=state.get("params", {}),
    )

    session.status = state["status"]
    session.progress = state["progress"]
    session.equity_history = state["equity_history"]
    session.trades = state["trades"]
    session.positions = state["positions"]

    SESSIONS[session_id] = session

    logger.info(
        f"Session restored: {session_id} from checkpoint {checkpoint['checkpoint_time']}"
    )

    return {
        "message": "Session restored",
        "session_id": session_id,
        "checkpoint_time": checkpoint["checkpoint_time"],
        "status": session.status,
        "progress": session.progress,
    }


@app.get("/persistence/stats")
async def persistence_stats():
    """Get persistence statistics"""
    stats = persistence.get_stats()
    return stats


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    cache = get_cache()
    backtest_cache = get_backtest_cache()
    return {"redis": cache.get_stats(), "backtest_cache": backtest_cache.get_stats()}


@app.post("/cache/clear")
async def cache_clear(pattern: str = "*"):
    """Clear cache by pattern"""
    cache = get_cache()
    deleted = cache.clear_pattern(pattern)
    return {"deleted": deleted, "pattern": pattern}


@app.get("/status")
async def status():
    ws_connections = ws_manager.get_connection_count()
    return {
        "status": "up",
        "active_sessions": len([s for s in SESSIONS.values() if s.status == "running"]),
        "websocket_connections": ws_connections,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
