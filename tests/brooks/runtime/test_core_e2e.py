"""BrooksCore — end-to-end per-bar pipeline test.

Drives a synthetic replay through ``BrooksCore`` with a fake sink and
verifies BarEvents are produced with regime + features populated. This
is the seam that both live and replay tasks rely on, so a green test
here is what makes a clean refactor confident.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pytest

from session_db import SessionDB
from src.brooks.runtime.clock import BarClock
from src.brooks.runtime.core import BrooksCore
from src.brooks.runtime.event_sink import BarEventSink, CompositeSink, PersistingSink
from src.brooks.strategy import BrooksStrategy
from src.core.base import Bar
from src.core.live_broker import create_live_broker


@pytest.fixture
def temp_session_db(monkeypatch, tmp_path):
    db_path = tmp_path / "sessions.db"
    monkeypatch.setenv("SESSION_DB_PATH", str(db_path))
    db = SessionDB(str(db_path))
    db.create_session(
        session_id="sess-core-1",
        strategy_name="brooks",
        symbol="BTC/USDT",
        mode="paper",
        start_date="2026-04-26",
        end_date=None,
        market="crypto",
        interval="5m",
    )
    return db


class _CollectingSink(BarEventSink):
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def handle(self, bar_event):
        self.events.append(bar_event)


def _make_bar(ts: datetime, price: float, symbol: str = "BTC/USDT") -> Dict[str, Bar]:
    return {
        symbol: Bar(
            symbol=symbol,
            timestamp=ts,
            open=price,
            high=price + 5.0,
            low=price - 5.0,
            close=price + 1.0,
            volume=1.0,
            amount=price,
        )
    }


def test_core_emits_bar_event_per_bar(temp_session_db):
    collecting = _CollectingSink()
    sink = CompositeSink([collecting, PersistingSink("sess-core-1", temp_session_db)])
    strategy = BrooksStrategy(
        db_client=None,
        session_id="sess-core-1",
        analyst="rule",
        analyst_params={},
        base_interval="5m",
        mtf_intervals=[],
    )
    broker = create_live_broker(
        mode="paper",
        initial_cash=100_000,
        commission=0.0003,
        slippage=0.001,
        allow_short=True,
        session_id="sess-core-1",
    )
    clock = BarClock()
    core = BrooksCore(
        strategy=strategy,
        broker=broker,
        sink=sink,
        session_id="sess-core-1",
        session_db=temp_session_db,
        analyst_name="rule",
        clock=clock,
        equity_throttle=None,
        emit=None,
    )

    base = datetime(2026, 4, 26, 12, 0)
    for i in range(20):
        bars = _make_bar(base + timedelta(minutes=5 * i), 30_000 + i)
        clock.advance_to((base + timedelta(minutes=5 * i)).timestamp())
        core.process_bar(bars)

    assert len(collecting.events) == 20
    # Each event must carry the BarEvent invariants the loader expects.
    for ev in collecting.events:
        assert "bar_idx" in ev
        assert "timestamp_ns" in ev
        assert "bar" in ev
        assert ev["symbol"] == "BTC/USDT"
        assert ev["analyst"] == "rule"

    # The persisting half — read back what the live timeline loader would see.
    rows = temp_session_db.get_session_logs("sess-core-1", source="brooks_bar")
    assert len(rows) == 20


def test_replay_clock_drives_throttle(temp_session_db):
    """Verify a bar-clock-driven analyst throttle behaves deterministically."""
    from src.brooks.runtime.analyst_wrap import build_throttled_analyst

    class _LlmStub:
        name = "llm:stub"

        def __init__(self):
            self.calls = 0

        async def analyze(self, ctx):
            self.calls += 1
            return []

    strategy = BrooksStrategy(
        db_client=None,
        session_id="sess-core-1",
        analyst="rule",
        analyst_params={},
        base_interval="5m",
        mtf_intervals=[],
    )
    inner = _LlmStub()
    clock = BarClock()
    strategy._analyst = build_throttled_analyst(
        inner,
        min_gap_seconds=600,  # 10 minutes — every other 5m bar should pass
        clock=clock,
    )

    broker = create_live_broker(
        mode="paper",
        initial_cash=100_000,
        commission=0.0003,
        slippage=0.001,
        allow_short=True,
        session_id="sess-core-1",
    )
    sink = _CollectingSink()
    core = BrooksCore(
        strategy=strategy,
        broker=broker,
        sink=sink,
        session_id="sess-core-1",
        session_db=temp_session_db,
        analyst_name="llm:stub",
        clock=clock,
        equity_throttle=None,
        emit=None,
    )

    base = datetime(2026, 4, 26, 12, 0)
    for i in range(8):
        bars = _make_bar(base + timedelta(minutes=5 * i), 30_000 + i)
        clock.advance_to((base + timedelta(minutes=5 * i)).timestamp())
        core.process_bar(bars)

    # 8 bars × 5m = 35m elapsed. With a 10-minute min gap, the analyst
    # should run on bars at t=0, 10, 20, 30 → 4 calls.
    assert inner.calls == 4
