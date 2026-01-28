import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

class SessionDB:
    def __init__(self, db_path="sessions.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    strategy_name TEXT,
                    symbol TEXT,
                    mode TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    status TEXT,
                    progress REAL,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Equity history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS equity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp TEXT,
                    total_assets REAL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp TEXT,
                    symbol TEXT,
                    side TEXT,
                    price REAL,
                    quantity REAL,
                    commission REAL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_session ON equity_history(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_session ON trades(session_id)")
            conn.commit()

    def create_session(self, session_id, strategy_name, symbol, mode, start_date, end_date):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, strategy_name, symbol, mode, start_date, end_date, status, progress)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, strategy_name, symbol, mode, start_date, end_date, "starting", 0.0))

    def update_session_status(self, session_id, status, progress=None, error=None):
        with sqlite3.connect(self.db_path) as conn:
            updates = ["status = ?"]
            params = [status]
            if progress is not None:
                updates.append("progress = ?")
                params.append(progress)
            if error is not None:
                updates.append("error = ?")
                params.append(error)
            
            params.append(session_id)
            conn.execute(f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?", params)

    def add_equity_point(self, session_id, timestamp, total_assets):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO equity_history (session_id, timestamp, total_assets) VALUES (?, ?, ?)",
                (session_id, str(timestamp), total_assets)
            )

    def add_trade(self, session_id, trade: Dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades (session_id, timestamp, symbol, side, price, quantity, commission)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, 
                str(trade.get('timestamp')), 
                trade.get('symbol'), 
                trade.get('side'), 
                trade.get('price'), 
                trade.get('quantity'), 
                trade.get('commission', 0.0)
            ))

    def get_session(self, session_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d['id'] = d['session_id']
                return d
            return None

    def get_all_sessions(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM sessions ORDER BY created_at DESC")
            result = []
            for row in cursor.fetchall():
                d = dict(row)
                d['id'] = d['session_id']
                result.append(d)
            return result

    def get_equity_history(self, session_id, since=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT timestamp, total_assets FROM equity_history WHERE session_id = ?"
            params = [session_id]
            if since:
                query += " AND timestamp > ?"
                params.append(since)
            query += " ORDER BY timestamp"
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_trades(self, session_id, since=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM trades WHERE session_id = ?"
            params = [session_id]
            if since:
                query += " AND timestamp > ?"
                params.append(since)
            query += " ORDER BY timestamp"
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
