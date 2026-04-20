import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


def _default_db_path() -> str:
    from src.config.paths import SESSIONS_DB
    return os.environ.get("SESSION_DB_PATH", str(SESSIONS_DB))


class SessionDB:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path if db_path is not None else _default_db_path()
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            cursor = conn.cursor()

            cursor.execute(
                """
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
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS equity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp TEXT,
                    total_assets REAL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )

            cursor.execute(
                """
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
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS simulation_jobs (
                    job_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'simulation',
                    start_date TEXT NOT NULL,
                    end_date TEXT,
                    params TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'idle',
                    schedule TEXT NOT NULL DEFAULT 'daily',
                    last_processed_at TEXT,
                    last_update_at TEXT,
                    latest_session_id TEXT,
                    latest_run_id TEXT,
                    snapshot TEXT,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS simulation_runs (
                    run_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    session_id TEXT,
                    update_run_id TEXT,
                    trigger_source TEXT NOT NULL DEFAULT 'manual',
                    start_date TEXT,
                    end_date TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    progress REAL NOT NULL DEFAULT 0.0,
                    bars_processed INTEGER NOT NULL DEFAULT 0,
                    steps_recorded INTEGER NOT NULL DEFAULT 0,
                    summary TEXT,
                    error TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(job_id) REFERENCES simulation_jobs(job_id)
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS simulation_run_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    session_id TEXT,
                    step_index INTEGER NOT NULL,
                    timestamp TEXT,
                    event_type TEXT NOT NULL,
                    payload TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES simulation_runs(run_id)
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS data_update_runs (
                    update_run_id TEXT PRIMARY KEY,
                    trigger_source TEXT NOT NULL DEFAULT 'manual',
                    status TEXT NOT NULL DEFAULT 'pending',
                    has_new_data INTEGER NOT NULL DEFAULT 0,
                    details TEXT,
                    error TEXT,
                    started_at TEXT,
                    last_heartbeat_at TEXT,
                    completed_at TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS session_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    extra TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS lineage_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_kind TEXT NOT NULL,
                    parent_id TEXT NOT NULL,
                    child_kind TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    meta TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(parent_kind, parent_id, child_kind, child_id, relation)
                )
                """
            )

            self._add_column_if_not_exists(cursor, "equity_history", "cash", "REAL DEFAULT 0.0")
            self._add_column_if_not_exists(cursor, "equity_history", "daily_pnl", "REAL DEFAULT 0.0")
            self._add_column_if_not_exists(cursor, "equity_history", "daily_return", "REAL DEFAULT 0.0")
            self._add_column_if_not_exists(cursor, "equity_history", "positions", "TEXT")

            self._add_column_if_not_exists(cursor, "trades", "name", "TEXT")
            self._add_column_if_not_exists(cursor, "trades", "type", "TEXT")
            self._add_column_if_not_exists(cursor, "trades", "amount", "REAL")
            self._add_column_if_not_exists(cursor, "simulation_jobs", "end_date", "TEXT")
            self._add_column_if_not_exists(cursor, "simulation_jobs", "notification", "TEXT")
            self._add_column_if_not_exists(cursor, "simulation_jobs", "source_zoo_factor_id", "TEXT")
            self._add_column_if_not_exists(cursor, "sessions", "params", "TEXT")
            self._add_column_if_not_exists(cursor, "sessions", "source", "TEXT DEFAULT 'manual'")
            self._add_column_if_not_exists(cursor, "sessions", "job_id", "TEXT")
            self._add_column_if_not_exists(cursor, "sessions", "run_id", "TEXT")
            self._add_column_if_not_exists(cursor, "sessions", "last_processed_at", "TEXT")
            self._add_column_if_not_exists(cursor, "data_update_runs", "last_heartbeat_at", "TEXT")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_session ON equity_history(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_session ON trades(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_simulation_jobs_enabled ON simulation_jobs(enabled, updated_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_simulation_runs_job ON simulation_runs(job_id, created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_simulation_steps_run ON simulation_run_steps(run_id, step_index DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_update_runs_created ON data_update_runs(created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_logs_session_time ON session_logs(session_id, timestamp DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineage_parent ON lineage_edges(parent_kind, parent_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineage_child ON lineage_edges(child_kind, child_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineage_relation ON lineage_edges(relation)")
            conn.commit()

    def _add_column_if_not_exists(self, cursor, table, column, type_def):
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        if column not in columns:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_def}")

    def _json_dumps(self, value: Any) -> str:
        if value is None:
            value = {}
        return json.dumps(value, ensure_ascii=False, default=str)

    def _json_loads(self, value: Optional[str], default: Any):
        if not value:
            return default
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default

    def _get_conn(self, *, timeout: float = 10.0):
        conn = sqlite3.connect(self.db_path, timeout=timeout)
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def create_session(
        self,
        session_id,
        strategy_name,
        symbol,
        mode,
        start_date,
        end_date,
        params=None,
        source: str = "manual",
        job_id: Optional[str] = None,
        run_id: Optional[str] = None,
        last_processed_at: Optional[str] = None,
    ):
        params_json = self._json_dumps(params or {})
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, strategy_name, symbol, mode, start_date, end_date,
                    status, progress, params, source, job_id, run_id, last_processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    strategy_name,
                    symbol,
                    mode,
                    start_date,
                    end_date,
                    "starting",
                    0.0,
                    params_json,
                    source,
                    job_id,
                    run_id,
                    last_processed_at,
                ),
            )

    def update_session_status(self, session_id, status, progress=None, error=None, last_processed_at=None):
        with self._get_conn() as conn:
            updates = ["status = ?"]
            params = [status]
            if progress is not None:
                updates.append("progress = ?")
                params.append(progress)
            if error is not None:
                updates.append("error = ?")
                params.append(error)
            if last_processed_at is not None:
                updates.append("last_processed_at = ?")
                params.append(last_processed_at)
            params.append(session_id)
            conn.execute(f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?", params)

    def update_session(self, session_id: str, **fields):
        if not fields:
            return self.get_session(session_id)

        db_fields = {}
        for key, value in fields.items():
            if key == "params":
                db_fields[key] = self._json_dumps(value)
            else:
                db_fields[key] = value

        assignments = ", ".join(f"{key} = ?" for key in db_fields)
        values = list(db_fields.values()) + [session_id]
        with self._get_conn() as conn:
            conn.execute(f"UPDATE sessions SET {assignments} WHERE session_id = ?", values)
        return self.get_session(session_id)

    def clear_session_runtime_data(self, session_id: str, clear_logs: bool = True):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM equity_history WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM trades WHERE session_id = ?", (session_id,))
            if clear_logs:
                conn.execute("DELETE FROM session_logs WHERE session_id = ?", (session_id,))

    def delete_session(self, session_id: str) -> bool:
        with self._get_conn() as conn:
            conn.execute("DELETE FROM equity_history WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM trades WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_logs WHERE session_id = ?", (session_id,))
            conn.execute(
                "UPDATE simulation_jobs SET latest_session_id = NULL WHERE latest_session_id = ?",
                (session_id,),
            )
            conn.execute(
                "UPDATE simulation_runs SET session_id = NULL WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "UPDATE simulation_run_steps SET session_id = NULL WHERE session_id = ?",
                (session_id,),
            )
            result = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            return result.rowcount > 0

    def delete_simulation_job(self, job_id: str) -> bool:
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM simulation_run_steps WHERE run_id IN "
                "(SELECT run_id FROM simulation_runs WHERE job_id = ?)",
                (job_id,),
            )
            conn.execute("DELETE FROM simulation_runs WHERE job_id = ?", (job_id,))
            conn.execute("UPDATE sessions SET job_id = NULL WHERE job_id = ?", (job_id,))
            result = conn.execute("DELETE FROM simulation_jobs WHERE job_id = ?", (job_id,))
            return result.rowcount > 0

    def add_equity_points(self, session_id, points: List[Dict]):
        if not points:
            return
        data = [
            (
                session_id,
                str(p["timestamp"]),
                p["total_equity"],
                p.get("cash", 0.0),
                p.get("daily_pnl", 0.0),
                p.get("daily_return", 0.0),
                self._json_dumps(p.get("positions", {})),
            )
            for p in points
        ]
        with self._get_conn() as conn:
            conn.executemany(
                "INSERT INTO equity_history (session_id, timestamp, total_assets, cash, daily_pnl, daily_return, positions) VALUES (?, ?, ?, ?, ?, ?, ?)",
                data,
            )

    def add_equity_point(self, session_id, timestamp, total_assets, cash=0.0, daily_pnl=0.0, daily_return=0.0, positions=None):
        self.add_equity_points(
            session_id,
            [
                {
                    "timestamp": timestamp,
                    "total_equity": total_assets,
                    "cash": cash,
                    "daily_pnl": daily_pnl,
                    "daily_return": daily_return,
                    "positions": positions,
                }
            ],
        )

    def add_trades(self, session_id, trades: List[Dict]):
        if not trades:
            return
        data = [
            (
                session_id,
                str(t.get("timestamp")),
                t.get("symbol"),
                t.get("name", "Unknown"),
                t.get("type") or t.get("side"),
                t.get("price"),
                t.get("quantity"),
                t.get("amount") or (t.get("price", 0) * t.get("quantity", 0)),
                t.get("commission", 0.0),
            )
            for t in trades
        ]
        with self._get_conn() as conn:
            conn.executemany(
                """
                INSERT INTO trades (session_id, timestamp, symbol, name, type, price, quantity, amount, commission)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                data,
            )

    def add_trade(self, session_id, trade: Dict):
        self.add_trades(session_id, [trade])

    def persist_snapshot(
        self,
        session_id: str,
        equity_points: List[Dict],
        trades: List[Dict],
        status: str = "running",
        progress: Optional[float] = None,
    ):
        """Batch-write equity points, trades, and status in a single transaction."""
        with self._get_conn() as conn:
            if equity_points:
                data = [
                    (
                        session_id,
                        str(p["timestamp"]),
                        p["total_equity"],
                        p.get("cash", 0.0),
                        p.get("daily_pnl", 0.0),
                        p.get("daily_return", 0.0),
                        self._json_dumps(p.get("positions", {})),
                    )
                    for p in equity_points
                ]
                conn.executemany(
                    "INSERT INTO equity_history (session_id, timestamp, total_assets, cash, daily_pnl, daily_return, positions) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    data,
                )
            if trades:
                data = [
                    (
                        session_id,
                        str(t.get("timestamp")),
                        t.get("symbol"),
                        t.get("name", "Unknown"),
                        t.get("type") or t.get("side"),
                        t.get("price"),
                        t.get("quantity"),
                        t.get("amount") or (t.get("price", 0) * t.get("quantity", 0)),
                        t.get("commission", 0.0),
                    )
                    for t in trades
                ]
                conn.executemany(
                    "INSERT INTO trades (session_id, timestamp, symbol, name, type, price, quantity, amount, commission) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    data,
                )
            updates = ["status = ?"]
            params: list = [status]
            if progress is not None:
                updates.append("progress = ?")
                params.append(progress)
            params.append(session_id)
            conn.execute(f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?", params)

    def get_session(self, session_id):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            d["id"] = d["session_id"]
            d["strategy"] = d.get("strategy_name", "")
            d["params"] = self._json_loads(d.get("params"), {})
            return d

    def get_all_sessions(self):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM sessions ORDER BY created_at DESC")
            result = []
            for row in cursor.fetchall():
                d = dict(row)
                d["id"] = d["session_id"]
                d["strategy"] = d.get("strategy_name", "")
                d["params"] = self._json_loads(d.get("params"), {})
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
                d["total_equity"] = d.pop("total_assets")
                d["positions"] = self._json_loads(d.get("positions"), {})
                result.append(d)
            return result

    def get_equity_history_page(
        self,
        session_id: str,
        since: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> Dict[str, Any]:
        limit = max(1, int(limit))
        offset = max(0, int(offset))

        where = "WHERE session_id = ?"
        params: List[Any] = [session_id]
        if since:
            where += " AND timestamp > ?"
            params.append(since)

        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute(
                f"SELECT COUNT(*) FROM equity_history {where}",
                params,
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT *
                FROM equity_history
                {where}
                ORDER BY timestamp
                LIMIT ? OFFSET ?
                """,
                params + [limit, offset],
            ).fetchall()

        items = []
        for row in rows:
            item = dict(row)
            item["total_equity"] = item.pop("total_assets")
            item["positions"] = self._json_loads(item.get("positions"), {})
            items.append(item)

        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(items) < total,
        }

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

    def get_trades_page(
        self,
        session_id: str,
        since: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> Dict[str, Any]:
        limit = max(1, int(limit))
        offset = max(0, int(offset))

        where = "WHERE session_id = ?"
        params: List[Any] = [session_id]
        if since:
            where += " AND timestamp > ?"
            params.append(since)

        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute(
                f"SELECT COUNT(*) FROM trades {where}",
                params,
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT *
                FROM trades
                {where}
                ORDER BY timestamp
                LIMIT ? OFFSET ?
                """,
                params + [limit, offset],
            ).fetchall()

        items = [dict(row) for row in rows]
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(items) < total,
        }

    def add_session_logs(self, session_id: str, logs: List[Dict[str, Any]]):
        if not logs:
            return

        data = [
            (
                session_id,
                str(item.get("timestamp") or datetime.now().isoformat()),
                item.get("level", "INFO"),
                item.get("source", "system"),
                item.get("message", ""),
                self._json_dumps(item.get("extra") or {}),
            )
            for item in logs
        ]

        with self._get_conn() as conn:
            conn.executemany(
                """
                INSERT INTO session_logs (session_id, timestamp, level, source, message, extra)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                data,
            )

    def add_session_log(
        self,
        session_id: str,
        timestamp: str,
        level: str,
        source: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.add_session_logs(
            session_id,
            [
                {
                    "timestamp": timestamp,
                    "level": level,
                    "source": source,
                    "message": message,
                    "extra": extra or {},
                }
            ],
        )

    def get_session_logs(
        self,
        session_id: str,
        level: Optional[str] = None,
        source: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        query = "SELECT timestamp, level, source, message, extra FROM session_logs WHERE session_id = ?"
        params: List[Any] = [session_id]

        if level:
            query += " AND level = ?"
            params.append(level.upper())
        if source:
            query += " AND source = ?"
            params.append(source)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        result = []
        for row in reversed(rows):
            item = dict(row)
            item["extra"] = self._json_loads(item.get("extra"), {})
            result.append(item)
        return result

    def clear_session_logs(self, session_id: str):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM session_logs WHERE session_id = ?", (session_id,))

    def list_sessions_with_logs(self) -> List[str]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT session_id
                FROM session_logs
                GROUP BY session_id
                ORDER BY MAX(timestamp) DESC
                """
            ).fetchall()
        return [row[0] for row in rows]

    def create_simulation_job(
        self,
        name: str,
        strategy_name: str,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        notification: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        schedule: str = "daily",
    ) -> Dict[str, Any]:
        job_id = str(uuid.uuid4())
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO simulation_jobs (
                    job_id, name, strategy_name, symbol, start_date, end_date, params,
                    notification, enabled, schedule, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    name,
                    strategy_name,
                    symbol,
                    start_date,
                    end_date,
                    self._json_dumps(params or {}),
                    self._json_dumps(notification or {}),
                    1 if enabled else 0,
                    schedule,
                    datetime.now().isoformat(),
                ),
            )
        return self.get_simulation_job(job_id)

    def update_simulation_job(self, job_id: str, **fields):
        if not fields:
            return self.get_simulation_job(job_id)

        db_fields = {}
        for key, value in fields.items():
            if key in {"params", "snapshot", "notification"}:
                db_fields[key] = self._json_dumps(value)
            elif key == "enabled":
                db_fields[key] = 1 if value else 0
            else:
                db_fields[key] = value
        db_fields["updated_at"] = datetime.now().isoformat()

        assignments = ", ".join(f"{key} = ?" for key in db_fields)
        values = list(db_fields.values()) + [job_id]
        with self._get_conn() as conn:
            conn.execute(f"UPDATE simulation_jobs SET {assignments} WHERE job_id = ?", values)
        return self.get_simulation_job(job_id)

    def set_simulation_job_enabled(self, job_id: str, enabled: bool):
        return self.update_simulation_job(job_id=job_id, enabled=enabled, status="idle" if enabled else "disabled")

    def _decode_simulation_job_row(self, row: sqlite3.Row, include_snapshot: bool = True):
        job = dict(row)
        job["enabled"] = bool(job.get("enabled", 0))
        job["params"] = self._json_loads(job.get("params"), {})
        if include_snapshot:
            job["snapshot"] = self._json_loads(job.get("snapshot"), None)
        else:
            job.pop("snapshot", None)
        job["notification"] = self._json_loads(job.get("notification"), {})
        return job

    def get_simulation_job(self, job_id: str, include_snapshot: bool = True):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM simulation_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                return None
            return self._decode_simulation_job_row(row, include_snapshot=include_snapshot)

    def list_simulation_jobs(self, enabled_only: bool = False, include_snapshot: bool = False):
        query = """
            SELECT
                job_id, name, strategy_name, symbol, mode, start_date, end_date,
                params, notification, enabled, status, schedule, last_processed_at,
                last_update_at, latest_session_id, latest_run_id, error, created_at, updated_at
            FROM simulation_jobs
        """
        params: List[Any] = []
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY updated_at DESC, created_at DESC"

        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            result = []
            for row in rows:
                result.append(self._decode_simulation_job_row(row, include_snapshot=include_snapshot))
            return result

    def create_simulation_run(
        self,
        job_id: str,
        session_id: str,
        start_date: str,
        end_date: Optional[str],
        trigger_source: str,
        update_run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        run_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO simulation_runs (
                    run_id, job_id, session_id, update_run_id, trigger_source,
                    start_date, end_date, status, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, job_id, session_id, update_run_id, trigger_source, start_date, end_date, "running", now),
            )
        self.update_simulation_job(
            job_id,
            status="running",
            latest_session_id=session_id,
            latest_run_id=run_id,
            error=None,
        )
        return self.get_simulation_run(run_id)

    def update_simulation_run(self, run_id: str, **fields):
        if not fields:
            return self.get_simulation_run(run_id)
        db_fields = {}
        for key, value in fields.items():
            if key == "summary":
                db_fields[key] = self._json_dumps(value)
            else:
                db_fields[key] = value
        assignments = ", ".join(f"{key} = ?" for key in db_fields)
        values = list(db_fields.values()) + [run_id]
        with self._get_conn() as conn:
            conn.execute(f"UPDATE simulation_runs SET {assignments} WHERE run_id = ?", values)
        return self.get_simulation_run(run_id)

    def get_simulation_run(self, run_id: str):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM simulation_runs WHERE run_id = ?", (run_id,)).fetchone()
            if not row:
                return None
            run = dict(row)
            run["summary"] = self._json_loads(run.get("summary"), {})
            return run

    def list_simulation_runs(self, job_id: str, limit: int = 20):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM simulation_runs WHERE job_id = ? ORDER BY created_at DESC LIMIT ?",
                (job_id, limit),
            ).fetchall()
            result = []
            for row in rows:
                run = dict(row)
                run["summary"] = self._json_loads(run.get("summary"), {})
                result.append(run)
            return result

    def add_simulation_run_step(
        self,
        run_id: str,
        session_id: str,
        step_index: int,
        timestamp: Optional[str],
        event_type: str,
        payload: Dict[str, Any],
    ):
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO simulation_run_steps (run_id, session_id, step_index, timestamp, event_type, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, session_id, step_index, timestamp, event_type, self._json_dumps(payload)),
            )

    def list_simulation_run_steps(self, run_id: str, limit: int = 300, since_step: Optional[int] = None):
        query = "SELECT * FROM simulation_run_steps WHERE run_id = ?"
        params: List[Any] = [run_id]
        if since_step is not None:
            query += " AND step_index > ?"
            params.append(since_step)
        query += " ORDER BY step_index DESC LIMIT ?"
        params.append(limit)
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            result = []
            for row in reversed(rows):
                step = dict(row)
                step["payload"] = self._json_loads(step.get("payload"), {})
                result.append(step)
            return result

    def create_data_update_run(self, trigger_source: str = "manual") -> Dict[str, Any]:
        update_run_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO data_update_runs (update_run_id, trigger_source, status, started_at, last_heartbeat_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (update_run_id, trigger_source, "running", now, now),
            )
        return self.get_data_update_run(update_run_id)

    def update_data_update_run(self, update_run_id: str, **fields):
        if not fields:
            return self.get_data_update_run(update_run_id)
        db_fields = {}
        for key, value in fields.items():
            if key == "details":
                db_fields[key] = self._json_dumps(value)
            elif key == "has_new_data":
                db_fields[key] = 1 if value else 0
            else:
                db_fields[key] = value
        assignments = ", ".join(f"{key} = ?" for key in db_fields)
        values = list(db_fields.values()) + [update_run_id]
        with self._get_conn() as conn:
            conn.execute(f"UPDATE data_update_runs SET {assignments} WHERE update_run_id = ?", values)
        return self.get_data_update_run(update_run_id)

    def get_data_update_run(self, update_run_id: str):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM data_update_runs WHERE update_run_id = ?", (update_run_id,)).fetchone()
            if not row:
                return None
            item = dict(row)
            item["has_new_data"] = bool(item.get("has_new_data", 0))
            item["details"] = self._json_loads(item.get("details"), {})
            return item

    def list_data_update_runs(self, limit: int = 20):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM data_update_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["has_new_data"] = bool(item.get("has_new_data", 0))
                item["details"] = self._json_loads(item.get("details"), {})
                result.append(item)
            return result

    def get_latest_data_update_run(self):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM data_update_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            item = dict(row)
            item["has_new_data"] = bool(item.get("has_new_data", 0))
            item["details"] = self._json_loads(item.get("details"), {})
            return item

    def get_running_data_update_run(self):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT *
                FROM data_update_runs
                WHERE status = 'running'
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
            if not row:
                return None
            item = dict(row)
            item["has_new_data"] = bool(item.get("has_new_data", 0))
            item["details"] = self._json_loads(item.get("details"), {})
            return item

    # ------------------------------------------------------------------
    # Lineage edges: unified provenance graph across search/zoo/simulation
    # ------------------------------------------------------------------

    _LINEAGE_KINDS = {"search_job", "zoo_factor", "simulation_job", "simulation_run", "data_update_run"}
    _LINEAGE_RELATIONS = {
        "produced",         # search_job --produced--> zoo_factor
        "derived_from",     # zoo_factor --derived_from--> zoo_factor (parent formula)
        "promoted_to",      # zoo_factor --promoted_to--> simulation_job
        "backtests",        # simulation_run --backtests--> zoo_factor (records live metrics)
        "triggered_by",     # simulation_job --triggered_by--> data_update_run
    }

    def add_lineage_edge(
        self,
        parent_kind: str,
        parent_id: str,
        child_kind: str,
        child_id: str,
        relation: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if parent_kind not in self._LINEAGE_KINDS or child_kind not in self._LINEAGE_KINDS:
            raise ValueError(f"Unknown lineage kind: {parent_kind!r}/{child_kind!r}")
        if relation not in self._LINEAGE_RELATIONS:
            raise ValueError(f"Unknown lineage relation: {relation!r}")
        with self._get_conn() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO lineage_edges (parent_kind, parent_id, child_kind, child_id, relation, meta)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (parent_kind, parent_id, child_kind, child_id, relation, self._json_dumps(meta or {})),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def list_lineage_parents(self, kind: str, node_id: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT parent_kind, parent_id, relation, meta, created_at
                FROM lineage_edges
                WHERE child_kind = ? AND child_id = ?
                ORDER BY created_at DESC
                """,
                (kind, node_id),
            ).fetchall()
        return [
            {**dict(r), "meta": self._json_loads(r["meta"], {})}
            for r in rows
        ]

    def list_lineage_children(self, kind: str, node_id: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT child_kind, child_id, relation, meta, created_at
                FROM lineage_edges
                WHERE parent_kind = ? AND parent_id = ?
                ORDER BY created_at DESC
                """,
                (kind, node_id),
            ).fetchall()
        return [
            {**dict(r), "meta": self._json_loads(r["meta"], {})}
            for r in rows
        ]

    def get_lineage_graph(
        self,
        kind: str,
        node_id: str,
        *,
        max_depth: int = 4,
    ) -> Dict[str, Any]:
        """Return full upstream+downstream DAG within max_depth hops."""
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []
        seen_edges: set = set()

        def _node_key(k: str, i: str) -> str:
            return f"{k}:{i}"

        def _add_node(k: str, i: str):
            key = _node_key(k, i)
            if key not in nodes:
                nodes[key] = {"kind": k, "id": i}

        _add_node(kind, node_id)

        # Upstream
        frontier = [(kind, node_id, 0)]
        while frontier:
            k, i, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for parent in self.list_lineage_parents(k, i):
                edge_key = (parent["parent_kind"], parent["parent_id"], k, i, parent["relation"])
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                _add_node(parent["parent_kind"], parent["parent_id"])
                edges.append({
                    "parent_kind": parent["parent_kind"],
                    "parent_id": parent["parent_id"],
                    "child_kind": k,
                    "child_id": i,
                    "relation": parent["relation"],
                    "meta": parent["meta"],
                })
                frontier.append((parent["parent_kind"], parent["parent_id"], depth + 1))

        # Downstream
        frontier = [(kind, node_id, 0)]
        while frontier:
            k, i, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for child in self.list_lineage_children(k, i):
                edge_key = (k, i, child["child_kind"], child["child_id"], child["relation"])
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                _add_node(child["child_kind"], child["child_id"])
                edges.append({
                    "parent_kind": k,
                    "parent_id": i,
                    "child_kind": child["child_kind"],
                    "child_id": child["child_id"],
                    "relation": child["relation"],
                    "meta": child["meta"],
                })
                frontier.append((child["child_kind"], child["child_id"], depth + 1))

        return {
            "root": {"kind": kind, "id": node_id},
            "nodes": list(nodes.values()),
            "edges": edges,
        }
