from fastapi import FastAPI, BackgroundTasks, HTTPException
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
from src.core.data_stream import DBDataStream
from src.strategies.jsg_strategy import JSGStrategy
from src.strategies.rotation_strategy import RotationStrategy
from db import DB

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SessionRequest(BaseModel):
    strategy: str
    symbol: str
    start_date: str
    end_date: Optional[str] = None
    mode: str = "backtest" # backtest, simulation, live

class Session:
    def __init__(self, session_id: str, strategy_name: str, symbol: str, mode: str, start_date: str, end_date: Optional[str]):
        self.session_id = session_id
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.mode = mode
        self.start_date = start_date
        self.end_date = end_date
        self.status = "starting"
        self.progress = 0.0
        self.equity_history = []
        self.trades = []
        self.positions = {}
        self.metrics = {}
        self.error = None
        self.engine = None
        self.broker = None

SESSIONS: Dict[str, Session] = {}

# Mock live server URL - in real scenario this might be config
LIVE_SERVER_URL = "http://localhost:11122" 

@app.post("/session/run")
async def run_session(req: SessionRequest, background_tasks: BackgroundTasks):
    session_id = str(uuid.uuid4())
    session = Session(session_id, req.strategy, req.symbol, req.mode, req.start_date, req.end_date)
    SESSIONS[session_id] = session
    
    def execute_session_task():
        try:
            db_client = DB()
            session.status = "running"
            
            # Setup data stream
            symbols = [req.symbol]
            stream = DBDataStream(db_client, symbols, req.start_date, req.end_date)
            total_bars = len(stream.timestamps) if stream.timestamps else 1
            
            # Setup broker
            if req.mode == "live":
                broker = LiveBroker(server_url=LIVE_SERVER_URL)
            else:
                # Backtest and Simulation (Paper) use BacktestBroker
                broker = BacktestBroker(db_client=db_client)
                
            session.broker = broker
            
            # Setup strategy
            if req.strategy == "jsg":
                strategy = JSGStrategy(db_client)
            elif req.strategy == "rotation":
                strategy = RotationStrategy(db_client)
            else:
                raise ValueError(f"Unknown strategy: {req.strategy}")
            
            # Progress callback
            def on_step(bars):
                # Update progress
                if req.mode == "backtest" and total_bars > 0:
                    session.progress = (stream.idx / total_bars) * 100
                else:
                    # For live/sim, progress is less meaningful, maybe just time elapsed?
                    session.progress = 50.0 
                
                info = broker.get_account_info()
                session.equity_history = info.get('equity_history', [])
                session.trades = info.get('trades', [])
                session.positions = info.get('detailed_positions', {})
                
                # For simulation mode, we might want to slow down
                if req.mode == "simulation":
                    time.sleep(1) # Simulate 1 second per bar

            engine = TradingEngine(strategy, broker, stream, on_step=on_step)
            session.engine = engine
            engine.run()
            
            session.status = "completed"
            session.progress = 100.0
        except Exception as e:
            session.status = "failed"
            session.error = str(e)
            print(f"Session failed: {e}")

    background_tasks.add_task(execute_session_task)
    return {"session_id": session_id}

@app.get("/sessions")
async def get_sessions():
    return [
        {
            "id": s.session_id,
            "strategy": s.strategy_name,
            "symbol": s.symbol,
            "status": s.status,
            "mode": s.mode,
            "progress": s.progress,
            "start_date": s.start_date,
            "end_date": s.end_date
        }
        for s in SESSIONS.values()
    ]

@app.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    
    s = SESSIONS[session_id]
    return {
        "status": s.status,
        "mode": s.mode,
        "progress": s.progress,
        "equity_history": s.equity_history,
        "trades": s.trades[-20:], # Top 20 for brief status
        "positions": s.positions,
        "error": s.error,
        "start_date": s.start_date,
        "end_date": s.end_date
    }

@app.post("/session/{session_id}/stop")
async def stop_session(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    s = SESSIONS[session_id]
    if s.engine:
        s.engine.stop()
    s.status = "stopped"
    return {"status": "stopped"}

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
