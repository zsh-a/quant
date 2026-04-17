"""
Portfolio SQLite persistence.

Stores portfolio configurations as JSON in a 'portfolios' table alongside
the existing sessions.db (reuses the same DB file for simplicity).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from loguru import logger

_DB_PATH = Path("sessions.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            portfolio_id TEXT PRIMARY KEY,
            name         TEXT NOT NULL,
            config_json  TEXT NOT NULL,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_backtest_results (
            portfolio_id TEXT PRIMARY KEY,
            result_json  TEXT NOT NULL,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id)
        )
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_portfolio(portfolio_id: str, name: str, config: dict[str, Any]):
    conn = _get_conn()
    _ensure_table(conn)
    conn.execute(
        "INSERT OR REPLACE INTO portfolios (portfolio_id, name, config_json, updated_at) "
        "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
        (portfolio_id, name, json.dumps(config, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()
    logger.debug("Portfolio {} saved", portfolio_id)


def load_all_portfolios() -> list[dict[str, Any]]:
    conn = _get_conn()
    _ensure_table(conn)
    rows = conn.execute("SELECT * FROM portfolios ORDER BY created_at").fetchall()
    conn.close()
    return [
        {
            "portfolio_id": r["portfolio_id"],
            "name": r["name"],
            "config": json.loads(r["config_json"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def delete_portfolio(portfolio_id: str) -> bool:
    conn = _get_conn()
    _ensure_table(conn)
    cursor = conn.execute("DELETE FROM portfolios WHERE portfolio_id = ?", (portfolio_id,))
    conn.execute("DELETE FROM portfolio_backtest_results WHERE portfolio_id = ?", (portfolio_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def save_backtest_result(portfolio_id: str, result: dict[str, Any]):
    conn = _get_conn()
    _ensure_table(conn)
    conn.execute(
        "INSERT OR REPLACE INTO portfolio_backtest_results (portfolio_id, result_json) VALUES (?, ?)",
        (portfolio_id, json.dumps(result, ensure_ascii=False, default=str)),
    )
    conn.commit()
    conn.close()


def load_backtest_result(portfolio_id: str) -> dict[str, Any] | None:
    conn = _get_conn()
    _ensure_table(conn)
    row = conn.execute(
        "SELECT result_json FROM portfolio_backtest_results WHERE portfolio_id = ?",
        (portfolio_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return json.loads(row["result_json"])
