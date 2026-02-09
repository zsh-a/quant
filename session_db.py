import os
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional


def _default_db_path() -> str:
    return os.environ.get("SESSION_DB_PATH", "sessions.db")


class SessionDB:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path if db_path is not None else _default_db_path()
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
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

            # Migration: add columns if they don't exist
            self._add_column_if_not_exists(cursor, "equity_history", "cash", "REAL DEFAULT 0.0")
            self._add_column_if_not_exists(cursor, "equity_history", "daily_pnl", "REAL DEFAULT 0.0")
            self._add_column_if_not_exists(cursor, "equity_history", "daily_return", "REAL DEFAULT 0.0")
            self._add_column_if_not_exists(cursor, "equity_history", "positions", "TEXT")
            
            self._add_column_if_not_exists(cursor, "trades", "name", "TEXT")
            self._add_column_if_not_exists(cursor, "trades", "type", "TEXT")
            self._add_column_if_not_exists(cursor, "trades", "amount", "REAL")
            
            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_session ON equity_history(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_session ON trades(session_id)")
            conn.commit()

    def _add_column_if_not_exists(self, cursor, table, column, type_def):
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        if column not in columns:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_def}")

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def create_session(self, session_id, strategy_name, symbol, mode, start_date, end_date):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, strategy_name, symbol, mode, start_date, end_date, status, progress)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, strategy_name, symbol, mode, start_date, end_date, "starting", 0.0))

    def update_session_status(self, session_id, status, progress=None, error=None):
        with self._get_conn() as conn:
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

    def add_equity_points(self, session_id, points: List[Dict]):
        if not points: return
        data = [
            (session_id, str(p["timestamp"]), p["total_equity"], p.get("cash", 0.0), 
             p.get("daily_pnl", 0.0), p.get("daily_return", 0.0), json.dumps(p.get("positions", {})))
            for p in points
        ]
        with self._get_conn() as conn:
            conn.executemany(
                "INSERT INTO equity_history (session_id, timestamp, total_assets, cash, daily_pnl, daily_return, positions) VALUES (?, ?, ?, ?, ?, ?, ?)",
                data
            )

    def add_equity_point(self, session_id, timestamp, total_assets, cash=0.0, daily_pnl=0.0, daily_return=0.0, positions=None):
        self.add_equity_points(session_id, [{
            "timestamp": timestamp, "total_equity": total_assets, "cash": cash,
            "daily_pnl": daily_pnl, "daily_return": daily_return, "positions": positions
        }])

    def add_trades(self, session_id, trades: List[Dict]):
        if not trades: return
        data = [
            (session_id, str(t.get('timestamp')), t.get('symbol'), t.get('name', 'Unknown'),
             t.get('type') or t.get('side'), t.get('price'), t.get('quantity'), 
             t.get('amount') or (t.get('price', 0) * t.get('quantity', 0)), t.get('commission', 0.0))
            for t in trades
        ]
        with self._get_conn() as conn:
            conn.executemany("""
                INSERT INTO trades (session_id, timestamp, symbol, name, type, price, quantity, amount, commission)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)

    def add_trade(self, session_id, trade: Dict):
        self.add_trades(session_id, [trade])

    def get_session(self, session_id):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d['id'] = d['session_id']
                return d
            return None

    def get_all_sessions(self):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM sessions ORDER BY created_at DESC")
            result = []
            for row in cursor.fetchall():
                d = dict(row)
                d['id'] = d['session_id']
                result.append(d)
            return result

    def get_equity_history(self, session_id, since=None):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM equity_history WHERE session_id = ?"
            params = [session_id]
            if since:
                query += " AND timestamp > ?"
                params.append(since)
            query += " ORDER BY timestamp"
            
            cursor = conn.execute(query, params)
            result = []
            for row in cursor.fetchall():
                d = dict(row)
                d['total_equity'] = d.pop('total_assets')
                if d.get('positions'):
                    try:
                        d['positions'] = json.loads(d['positions'])
                    except:
                        d['positions'] = {}
                else:
                    d['positions'] = {}
                result.append(d)
            return result

    def get_trades(self, session_id, since=None):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM trades WHERE session_id = ?"
            params = [session_id]
            if since:
                query += " AND timestamp > ?"
                params.append(since)
            query += " ORDER BY timestamp"
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
