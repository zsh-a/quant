"""Regression test for the market/interval propagation bug.

Before the fix: ``/session/run_async`` sent market=crypto,interval=5m in the
Celery config dict, but ``run_backtest_task`` dropped both fields when
constructing ``SessionExecutionConfig``, causing crypto backtests to route
to the A-share ``DBDataStream`` and load zero bars. This test locks the
fix in three places:

  1. SessionDB persists market/interval columns.
  2. SessionService.create_session forwards them into the DB record.
  3. run_backtest_task forwards them into SessionExecutionConfig; when
     absent in the Celery payload, it falls back to the DB row.
"""

from __future__ import annotations

import inspect
import os
import tempfile

import pytest

from session_db import SessionDB


@pytest.fixture
def tmp_session_db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setenv("SESSION_DB_PATH", path)
    db = SessionDB(db_path=path)
    yield db
    os.unlink(path)


def test_session_db_persists_market_and_interval(tmp_session_db: SessionDB):
    tmp_session_db.create_session(
        session_id="s1",
        strategy_name="brooks_v2",
        symbol="BTCUSDT",
        mode="backtest",
        start_date="2025-01-01",
        end_date=None,
        market="crypto",
        interval="5m",
    )
    row = tmp_session_db.get_session("s1")
    assert row["market"] == "crypto"
    assert row["interval"] == "5m"


def test_session_db_defaults_preserve_backcompat(tmp_session_db: SessionDB):
    # Callers who don't pass market/interval (eg automation jobs) get
    # the A-share defaults so existing strategies continue to work.
    tmp_session_db.create_session(
        session_id="s2",
        strategy_name="jsg",
        symbol="sh.000300",
        mode="backtest",
        start_date="2024-01-01",
        end_date=None,
    )
    row = tmp_session_db.get_session("s2")
    assert row["market"] == "a_share"
    assert row["interval"] == "1d"


def test_session_service_forwards_market_interval(monkeypatch, tmp_session_db: SessionDB):
    from src.api.state_persistence import StatePersistence
    from src.services.session_service import SessionService

    svc = SessionService(tmp_session_db, StatePersistence())
    runtime = svc.create_session(
        session_id="s3",
        strategy_name="brooks_v2",
        symbol="BTCUSDT",
        mode="backtest",
        start_date="2025-01-01",
        end_date=None,
        market="crypto",
        interval="5m",
    )
    assert runtime.market == "crypto"
    assert runtime.interval == "5m"
    row = tmp_session_db.get_session("s3")
    assert row["market"] == "crypto" and row["interval"] == "5m"


def test_backtest_task_forwards_market_interval():
    """``SessionExecutionConfig`` must be constructed with the payload's
    market/interval. We inspect the function source to catch the bug without
    spinning up Celery."""
    from src.tasks import backtest

    src = inspect.getsource(backtest.run_backtest_task)
    assert "market=" in src, "run_backtest_task should forward market into SessionExecutionConfig"
    assert "interval=" in src, "run_backtest_task should forward interval into SessionExecutionConfig"


def test_resolve_market_interval_prefers_payload(tmp_session_db: SessionDB):
    from src.tasks.backtest import resolve_market_interval

    tmp_session_db.create_session(
        session_id="s4",
        strategy_name="brooks_v2",
        symbol="BTCUSDT",
        mode="backtest",
        start_date="2025-01-01",
        end_date=None,
        market="a_share",
        interval="1d",
    )
    market, interval = resolve_market_interval("s4", {"market": "crypto", "interval": "5m"}, tmp_session_db)
    assert market == "crypto" and interval == "5m"


def test_resolve_market_interval_falls_back_to_db(tmp_session_db: SessionDB):
    """When Celery payload omits market/interval (old payloads / reruns),
    we must read them from the persisted session record."""
    from src.tasks.backtest import resolve_market_interval

    tmp_session_db.create_session(
        session_id="s5",
        strategy_name="brooks_v2",
        symbol="BTCUSDT",
        mode="backtest",
        start_date="2025-01-01",
        end_date=None,
        market="crypto",
        interval="5m",
    )
    market, interval = resolve_market_interval("s5", {}, tmp_session_db)
    assert market == "crypto"
    assert interval == "5m"


def test_resolve_market_interval_defaults_when_missing_everywhere(tmp_session_db: SessionDB):
    from src.tasks.backtest import resolve_market_interval

    market, interval = resolve_market_interval("nonexistent", {}, tmp_session_db)
    assert market == "a_share"
    assert interval == "1d"
