from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
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

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.engine import TradingEngine
from src.core.backtest_broker import BacktestBroker
from src.core.live_broker import LiveBroker
from src.core.data_stream import DBDataStream, RealtimeDataStream
from src.strategies.jsg_strategy import JSGStrategy
from src.strategies.rotation_strategy import RotationStrategy
from src.utils.config import get, get_section
from src.utils.logging_config import setup_logging, get_logger
from db import DB
from session_db import SessionDB

# Initialize logging system
setup_logging()
logger = get_logger(__name__)

# Load configuration
api_config = get_section('api')
data_stream_config = get_section('data_stream')
broker_config = get_section('broker')

app = FastAPI()
session_db = SessionDB()

app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.get('cors_origins', ['*']),
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"API Server starting with config: port={api_config.get('port', 8000)}")

class SessionRequest(BaseModel):
    strategy: str
    symbol: str
    start_date: str
    end_date: Optional[str] = None
    mode: str = "backtest" # backtest, simulation, live
    params: Optional[Dict[str, Any]] = {}

class Session:
    def __init__(self, session_id: str, strategy_name: str, symbol: str, mode: str, start_date: str, end_date: Optional[str], params: Dict[str, Any] = {}):
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

@app.get("/strategies")
async def get_strategies():
    return [
        {"name": "jsg", "label": "JSG Quantitative", "params": JSGStrategy.get_parameters()},
        {"name": "rotation", "label": "Advanced Rotation", "params": RotationStrategy.get_parameters()}
    ]

@app.post("/session/run")
async def run_session(req: SessionRequest, background_tasks: BackgroundTasks):
    print(f"Running session: {req}")
    session_id = str(uuid.uuid4())
    session = Session(session_id, req.strategy, req.symbol, req.mode, req.start_date, req.end_date, req.params)
    SESSIONS[session_id] = session
    
    # Persist initial session state
    session_db.create_session(session_id, req.strategy, req.symbol, req.mode, req.start_date, req.end_date)
    # Note: params persistence in DB is not yet implemented in session_db, but it's okay for now.
    
    def execute_session_task():
        try:
            db_client = DB()
            session.status = "running"
            session_db.update_session_status(session_id, "running")
            
            # Setup data stream (Chunked automatically by DBDataStream optimization)
            symbols = [req.symbol]
            
            if req.mode == "live":
                # Use configuration for realtime stream
                rt_config = data_stream_config.get('realtime', {})
                stream = RealtimeDataStream(
                    symbols, 
                    interval_seconds=rt_config.get('interval_seconds', 60),
                    data_source=rt_config.get('data_source', 'akshare'),
                    enable_trading_hours_check=True
                )
                total_bars = 0 # Live stream is indefinite
            else:
                # Use configuration for backtest stream
                chunk_size = data_stream_config.get('chunk_size_months', None)
                stream = DBDataStream(
                    db_client, 
                    symbols, 
                    req.start_date, 
                    req.end_date,
                    chunk_size_months=chunk_size
                )
                # Use total_bars from stream if available (approximate)
                total_bars = getattr(stream, 'total_bars', 1)
            
            # Setup broker
            if req.mode == "live":
                live_config = broker_config.get('live', {})
                broker = LiveBroker(server_url=live_config.get('server_url', LIVE_SERVER_URL))
            else:
                # Backtest and Simulation (Paper) use BacktestBroker
                backtest_config = broker_config.get('backtest', {})
                broker = BacktestBroker(
                    db_client=db_client,
                    initial_cash=backtest_config.get('initial_cash', 1000000.0),
                    commission=backtest_config.get('commission', 0.0001)
                )
                
            session.broker = broker
            
            # Setup strategy with params
            if req.strategy == "jsg":
                strategy = JSGStrategy(db_client, **req.params)
            elif req.strategy == "rotation":
                strategy = RotationStrategy(db_client, **req.params)
            else:
                raise ValueError(f"Unknown strategy: {req.strategy}")
            
            # Progress callback
            def on_step(bars):
                # Update progress
                if req.mode == "backtest" and total_bars > 0:
                    session.progress = (stream.idx / total_bars) * 100
                    # Optimization: Don't update DB status every step for progress, maybe every 1%?
                    # For simplicity, we just keep in-memory status updated, and DB updated less frequently or at end.
                    # But if we want robust recovery, we should update DB occasionally.
                else:
                    session.progress = 50.0 
                
                info = broker.get_account_info()
                
                # Persist equity and trades
                # We assume info['equity_history'] contains NEW items if we clear them?
                # Actually, broker.get_account_info returns self.equity_history. 
                # If we clear self.equity_history in broker, we get what's accumulated since last clear.
                
                new_equity_points = info.get('equity_history', [])
                if new_equity_points:
                    for pt in new_equity_points:
                        session_db.add_equity_point(
                            session_id, 
                            pt['timestamp'], 
                            pt['total_equity'],
                            cash=pt.get('cash', 0.0),
                            daily_pnl=pt.get('daily_pnl', 0.0),
                            daily_return=pt.get('daily_return', 0.0),
                            positions=pt.get('positions', {})
                        )
                        # Keep latest positions in memory for quick access
                        session.positions = pt.get('positions', {})
                    
                    # Clear broker history to save memory
                    if isinstance(broker, BacktestBroker):
                        broker.equity_history.clear()

                new_trades = info.get('trades', [])
                if new_trades:
                    for trade in new_trades:
                        session_db.add_trade(session_id, trade)
                    
                    # Clear broker trades to save memory
                    if isinstance(broker, BacktestBroker):
                        broker.trades.clear()
                
                # For simulation mode, we might want to slow down
                if req.mode == "simulation":
                    time.sleep(1) # Simulate 1 second per bar

            engine = TradingEngine(strategy, broker, stream, on_step=on_step)
            session.engine = engine
            engine.run()
            
            session.status = "completed"
            session.progress = 100.0
            session_db.update_session_status(session_id, "completed", 100.0)
            
        except Exception as e:
            session.status = "failed"
            session.error = str(e)
            session_db.update_session_status(session_id, "failed", error=str(e))
            print(f"Session failed: {e}")

    background_tasks.add_task(execute_session_task)
    return {"session_id": session_id}

@app.get("/sessions")
async def get_sessions():
    # Return sessions from DB (persistent)
    return session_db.get_all_sessions()

@app.get("/session/{session_id}/status")
async def get_session_status(session_id: str, since: Optional[str] = Query(None, description="Return data since this timestamp")):
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
        s_db = session_db.get_session(session_id)
        if not s_db:
            raise HTTPException(status_code=404, detail="Session not found")
        status = s_db['status']
        mode = s_db['mode']
        progress = s_db['progress']
        error = s_db['error']
        start_date = s_db['start_date']
        end_date = s_db['end_date']
        positions = {} # Positions history not fully persisted in simple DB yet, only snapshots in equity?
                       # Actually equity_history doesn't store full positions in DB in my schema (simplified).
                       # So for finished sessions, positions might be empty unless we store final state.
                       # For now, acceptable compromise.

    equity_history = session_db.get_equity_history(session_id, since=since)
    trades = session_db.get_trades(session_id, since=since)
    
    return {
        "status": status,
        "mode": mode,
        "progress": progress,
        "equity_history": equity_history,
        "trades": trades, 
        "positions": positions,
        "error": error,
        "start_date": start_date,
        "end_date": end_date
    }

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
    if s_db and s_db['status'] == 'running':
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
            result.append({
                "timestamp": str(ts),
                "value": row['close']
            })
        return result
    except Exception as e:
        print(f"Error fetching benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    return {"status": "up", "active_sessions": len([s for s in SESSIONS.values() if s.status == "running"])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
