from fastapi import (
    Depends,
    FastAPI,
    BackgroundTasks,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)

try:
    from websockets.exceptions import ConnectionClosed as WsConnectionClosed
except ImportError:
    WsConnectionClosed = None
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import orjson
import os
import sys
import uuid
import asyncio
import anyio


# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config.paths import ensure_data_dirs
ensure_data_dirs()

from src.strategies.registry import StrategyRegistry
from src.utils.cache import get_cache, get_backtest_cache
from src.config.settings import (
    get_data_stream_config,
    get_broker_config,
    get_api_config,
    get_settings,
)
from src.utils.logging_config import setup_logging, get_logger, logging_middleware
from src.api.websocket_manager import manager as ws_manager, handle_websocket_message
from src.api.events import (
    event_bus,
    EventType,
    emit_session_started,
    emit_equity_update,
    emit_session_progress,
    emit_trade_executed,
    emit_session_completed,
    emit_session_failed,
    emit_session_stopped,
    emit_error,
)
from src.api.state_persistence import persistence
from src.analysis.backtest_metrics import calculate_metrics as calc_perf_metrics
from src.market_data.db import DB
from session_db import SessionDB
from src.services import (
    SessionExecutionConfig,
    SessionExecutionHooks,
    SessionService,
    execute_session,
)
from src.api.auth import router as auth_router, require_auth
from src.api.tasks_router import router as tasks_router
from src.api.monitoring_router import router as monitoring_router
from src.api.portfolio_router import router as portfolio_router
from src.api.optimizer_router import router as optimizer_router
from src.api.analysis_router import router as analysis_router
from src.api.logs_router import router as logs_router
from src.api.market_router import router as market_router
from src.api.automation_router import router as automation_router
from src.api.market_admin_router import router as market_admin_router
from src.api.crypto_market_router import router as crypto_market_router

# Alpha Lab — optional, requires torch
try:
    from src.api.alpha_lab_router import router as alpha_lab_router
    _ALPHA_AVAILABLE = True
except Exception:
    _ALPHA_AVAILABLE = False
from src.tasks.backtest import run_backtest_task

setup_logging()
logger = get_logger(__name__)

settings = get_settings()
api_config = get_api_config()
data_stream_config = get_data_stream_config()
broker_config = get_broker_config()

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(default_response_class=ORJSONResponse)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

session_db = SessionDB()
session_service = SessionService(session_db, persistence)

app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(logging_middleware)

# Include routers
app.include_router(auth_router)          # /auth — always public
app.include_router(tasks_router, dependencies=[Depends(require_auth)])
app.include_router(monitoring_router)    # health/metrics — keep public
app.include_router(portfolio_router, dependencies=[Depends(require_auth)])
app.include_router(optimizer_router, dependencies=[Depends(require_auth)])
app.include_router(analysis_router, dependencies=[Depends(require_auth)])
app.include_router(logs_router, dependencies=[Depends(require_auth)])
app.include_router(market_router, dependencies=[Depends(require_auth)])
app.include_router(automation_router, dependencies=[Depends(require_auth)])
app.include_router(market_admin_router, dependencies=[Depends(require_auth)])
if _ALPHA_AVAILABLE:
    app.include_router(alpha_lab_router, dependencies=[Depends(require_auth)])
app.include_router(crypto_market_router, dependencies=[Depends(require_auth)])

logger.info(f"API Server starting with config: port={api_config.port}")


from src.api.validators import DateStr, SymbolStr, ModeStr, MarketStr, IntervalStr


class SessionRequest(BaseModel):
    strategy: str
    symbol: Optional[SymbolStr] = "sh.000300"
    start_date: DateStr
    end_date: Optional[DateStr] = None
    mode: ModeStr = "backtest"
    market: MarketStr = "a_share"
    interval: IntervalStr = "1d"
    params: Dict[str, Any] = Field(default_factory=dict)
    enable_risk_management: bool = True

# Register strategies once at startup
StrategyRegistry.register_all()

# Activate template-based strategies
try:
    from src.strategies.templates import activate_templates
    n = activate_templates()
    if n:
        logger.info(f"Activated {n} template strategies")
except Exception as e:
    logger.warning(f"Template activation skipped: {e}")


@app.get("/strategies", dependencies=[Depends(require_auth)])
async def get_strategies():
    return StrategyRegistry.list_strategies()


@app.post("/session/run", dependencies=[Depends(require_auth)])
@limiter.limit("5/minute")
async def run_session(req: SessionRequest, background_tasks: BackgroundTasks, request: Request = None):
    session_id = str(uuid.uuid4())
    runtime = session_service.create_session(
        session_id=session_id,
        strategy_name=req.strategy,
        symbol=req.symbol,
        mode=req.mode,
        start_date=req.start_date,
        end_date=req.end_date,
        params=req.params,
    )
    loop = asyncio.get_running_loop()

    def schedule(coro):
        asyncio.run_coroutine_threadsafe(coro, loop)

    def update_status(status: str, progress: float, error: Optional[str]):
        session_service.update_runtime(
            session_id,
            status=status,
            progress=progress,
            error=error,
        )

    def update_progress(progress: float, status: str):
        session_service.update_runtime(session_id, status=status, progress=progress)
        schedule(emit_session_progress(session_id, progress, status))

    def update_equity(point: Dict[str, Any]):
        session_service.update_runtime(session_id, positions=point.get("positions", {}))
        schedule(emit_equity_update(session_id, point))

    def update_trade(trade: Dict[str, Any]):
        schedule(emit_trade_executed(session_id, trade))

    def session_started(config: SessionExecutionConfig):
        schedule(emit_session_started(config.session_id, config.strategy, config.symbol))

    def engine_created(engine, broker):
        session_service.update_runtime(session_id, engine=engine, broker=broker)

    def session_completed(result):
        session_service.update_runtime(
            session_id,
            status="completed",
            progress=100.0,
            positions=result.positions,
            error=None,
        )
        schedule(
            emit_session_completed(
                result.session_id,
                result.final_equity,
                result.total_trades,
            )
        )

    def session_failed(error_message: str):
        session_service.update_runtime(
            session_id,
            status="failed",
            error=error_message,
        )
        schedule(emit_session_failed(session_id, error_message))
        schedule(emit_error(session_id, error_message))

    hooks = SessionExecutionHooks(
        on_session_started=session_started,
        on_engine_created=engine_created,
        on_status_change=update_status,
        on_progress=update_progress,
        on_equity_point=update_equity,
        on_trade=update_trade,
        on_completed=session_completed,
        on_failed=session_failed,
    )

    def execute_session_task():
        execute_session(
            SessionExecutionConfig(
                session_id=runtime.session_id,
                strategy=req.strategy,
                symbol=req.symbol,
                start_date=req.start_date,
                end_date=req.end_date,
                mode=req.mode,
                market=req.market,
                interval=req.interval,
                params=req.params,
                initial_cash=broker_config.backtest.initial_cash,
                commission=broker_config.backtest.commission,
                slippage=broker_config.backtest.slippage,
                enable_risk_management=req.enable_risk_management,
                chunk_size_months=data_stream_config.chunk_size_months,
            ),
            session_db=session_db,
            hooks=hooks,
        )

    background_tasks.add_task(execute_session_task)
    return {"session_id": session_id}


@app.post("/session/run_async", dependencies=[Depends(require_auth)])
@limiter.limit("10/minute")
async def run_session_async(req: SessionRequest, request: Request = None):
    """
    Submit a backtest session to Celery queue for async processing.
    Returns immediately with session_id and task_id for progress tracking.
    """
    session_id = str(uuid.uuid4())

    session_service.create_session(
        session_id=session_id,
        strategy_name=req.strategy,
        symbol=req.symbol,
        mode=req.mode,
        start_date=req.start_date,
        end_date=req.end_date,
        params=req.params,
        register_runtime=False,
    )

    # Build config for Celery task (include request_id for tracing)
    from src.utils.logging_config import request_id_ctx
    config = {
        "symbol": req.symbol,
        "strategy": req.strategy,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "market": req.market,
        "interval": req.interval,
        "params": req.params or {},
        "initial_cash": broker_config.backtest.initial_cash,
        "commission": broker_config.backtest.commission,
        "slippage": broker_config.backtest.slippage,
        "enable_risk_management": req.enable_risk_management,
        "chunk_size_months": data_stream_config.chunk_size_months,
        "request_id": request_id_ctx.get(),
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


@app.get("/sessions", dependencies=[Depends(require_auth)])
async def get_sessions():
    # Return sessions from DB (persistent)
    return await anyio.to_thread.run_sync(session_db.get_all_sessions)


@app.delete("/session/{session_id}", dependencies=[Depends(require_auth)])
async def delete_session(session_id: str):
    deleted = await anyio.to_thread.run_sync(session_service.delete_session, session_id)
    return {"session_id": session_id, "deleted": deleted}


@app.get("/session/{session_id}/risk", dependencies=[Depends(require_auth)])
async def get_session_risk(session_id: str):
    """Risk metrics and alerts for a session. Returns 404 if session not found."""
    runtime, persisted = await anyio.to_thread.run_sync(
        session_service.get_runtime_or_persisted, session_id
    )
    if not runtime and not persisted:
        raise HTTPException(status_code=404, detail="Session not found")
    # Risk manager state is not persisted; return safe default so RiskPanel doesn't 404
    return {
        "enabled": bool(
            runtime
            and getattr(runtime, "engine", None)
            and getattr(runtime.engine, "risk_manager", None)
        ),
        "metrics": {},
        "limits": {},
        "alerts": [],
    }


@app.get("/session/{session_id}/status", dependencies=[Depends(require_auth)])
async def get_session_status(
    session_id: str,
    since: Optional[str] = Query(None, description="Return data since this timestamp"),
):
    return await anyio.to_thread.run_sync(
        session_service.get_status_payload,
        session_id,
        since,
    )


@app.get("/sessions/{session_id}/equity", dependencies=[Depends(require_auth)])
async def get_session_equity(
    session_id: str,
    since: Optional[str] = Query(None, description="Return data since this timestamp"),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    session = await anyio.to_thread.run_sync(session_db.get_session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    page = await anyio.to_thread.run_sync(
        session_db.get_equity_history_page,
        session_id,
        since,
        limit,
        offset,
    )
    return {"session_id": session_id, **page}


@app.get("/sessions/{session_id}/trades", dependencies=[Depends(require_auth)])
async def get_session_trades(
    session_id: str,
    since: Optional[str] = Query(None, description="Return data since this timestamp"),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    session = await anyio.to_thread.run_sync(session_db.get_session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    page = await anyio.to_thread.run_sync(
        session_db.get_trades_page,
        session_id,
        since,
        limit,
        offset,
    )
    return {"session_id": session_id, **page}


@app.get("/session/{session_id}/metrics", dependencies=[Depends(require_auth)])
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


@app.post("/session/{session_id}/stop", dependencies=[Depends(require_auth)])
async def stop_session(session_id: str):
    result = await anyio.to_thread.run_sync(session_service.stop_session, session_id)
    await emit_session_stopped(session_id)
    await emit_session_progress(session_id, 0.0, "stopped")
    return result


@app.get("/market/benchmark", dependencies=[Depends(require_auth)])
async def get_benchmark(symbol: SymbolStr, start_date: DateStr, end_date: Optional[DateStr] = None):
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
        logger.error(f"Error fetching benchmark: {e}")
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
            data = orjson.loads(await websocket.receive_text())
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


# Register event listeners — broadcast all event types to WebSocket clients
for _evt in vars(EventType).values():
    if isinstance(_evt, str) and not _evt.startswith("_"):
        event_bus.subscribe(_evt, broadcast_session_event)

logger.info("WebSocket event listeners registered")


# State persistence and recovery endpoints
@app.post("/session/{session_id}/checkpoint", dependencies=[Depends(require_auth)])
async def create_checkpoint(session_id: str):
    """Create a checkpoint for a session"""
    state, metadata = await anyio.to_thread.run_sync(
        session_service.build_checkpoint_state, session_id
    )
    success = persistence.save_checkpoint(session_id, state, metadata)

    if success:
        return {"message": "Checkpoint created", "session_id": session_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to create checkpoint")


@app.get("/session/{session_id}/checkpoints", dependencies=[Depends(require_auth)])
async def list_checkpoints(session_id: str):
    """List all checkpoints for a session"""
    checkpoints = persistence.list_checkpoints(session_id)
    return {"session_id": session_id, "checkpoints": checkpoints}


@app.post("/session/{session_id}/restore", dependencies=[Depends(require_auth)])
async def restore_session(session_id: str):
    """Restore a session from the latest checkpoint"""
    checkpoint = persistence.load_latest_checkpoint(session_id)

    if not checkpoint:
        raise HTTPException(status_code=404, detail="No checkpoint found")

    state = checkpoint["state"]

    # Restore session
    session = await anyio.to_thread.run_sync(session_service.restore_session, state)

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


@app.get("/persistence/stats", dependencies=[Depends(require_auth)])
async def persistence_stats():
    """Get persistence statistics"""
    stats = persistence.get_stats()
    return stats


@app.get("/cache/stats", dependencies=[Depends(require_auth)])
async def cache_stats():
    """Get cache statistics"""
    cache = get_cache()
    backtest_cache = get_backtest_cache()
    return {"redis": cache.get_stats(), "backtest_cache": backtest_cache.get_stats()}


@app.post("/cache/clear", dependencies=[Depends(require_auth)])
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
        "active_sessions": session_service.count_running_sessions(),
        "websocket_connections": ws_connections,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
