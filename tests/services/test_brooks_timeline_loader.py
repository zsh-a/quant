"""Tests for :mod:`src.services.brooks_timeline_loader`.

The fixture session is the one called for in the QUA-61 acceptance gate:
30 bars, 5 signals, 2 fills. We seed ``session_logs`` with the same
``brooks_bar`` rows that the Studio expects in production, then drive
both the full-load and the incremental-page paths.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from session_db import SessionDB
from src.services.brooks_timeline_loader import BrooksTimelineLoader, load_timeline

SESSION_ID = "studio-fixture-1"
SYMBOL = "BTC/USDT"
INTERVAL = "5m"
HTF_INTERVALS = ["1h"]
BAR_NS_STEP = 5 * 60 * 1_000_000_000  # 5 minutes


@pytest.fixture
def session_db(tmp_path, monkeypatch) -> SessionDB:
    db_path = tmp_path / "sessions.db"
    monkeypatch.setenv("SESSION_DB_PATH", str(db_path))
    db = SessionDB(str(db_path))
    db.create_session(
        session_id=SESSION_ID,
        strategy_name="brooks",
        symbol=SYMBOL,
        mode="paper",
        start_date="2026-04-25",
        end_date=None,
        params={"analyst": "rule", "mtf_intervals": HTF_INTERVALS, "base_interval": INTERVAL},
        market="crypto",
        interval=INTERVAL,
    )
    _seed_bars(db)
    _seed_fills(db)
    _seed_equity(db)
    return db


def _seed_bars(db: SessionDB) -> None:
    """Insert 30 bars; bars 5, 10, 15, 20, 25 carry signals; 10/20 also a decision."""
    base_ns = 1_700_000_000_000_000_000
    base_price = 100.0
    for i in range(30):
        ts_ns = base_ns + i * BAR_NS_STEP
        price = base_price + i * 0.5
        bar_event: Dict[str, Any] = {
            "bar_idx": i,
            "timestamp_ns": ts_ns,
            "bar": {
                "timestamp_ns": ts_ns,
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price + 0.3,
                "volume": 10.0,
            },
            "features": {
                "is_bull": i % 2 == 0,
                "body_pct": 40,
                "close_position": "high" if i % 2 == 0 else "low",
                "ema_relation": "above",
                "leg_dir": "up",
                "leg_length": (i % 5) + 1,
                "is_doji": False,
                "is_inside_bar": False,
            },
            "structure": {
                "always_in": "long",
                "confirmed_swings": [],
                "last_breakout_lookback_high": price + 2.0,
                "last_breakout_lookback_low": price - 2.0,
            },
            "regime": {
                "name": "weak_bull_trend",
                "confidence": 0.6,
                "reasons": [f"bar={i}"],
            },
            "signals": [],
            "decision": None,
            "htf": {
                "1h": {"regime": "weak_bull_trend", "always_in": "long", "last_swing_idx": max(0, i - 1)},
            },
            "htf_bars": {
                "1h": {
                    "timestamp_ns": ts_ns,
                    "open": price - 0.5,
                    "high": price + 1.5,
                    "low": price - 1.5,
                    "close": price + 0.5,
                    "volume": 50.0,
                }
            },
            "pnl_r": 0.1 if (i in {11, 21}) else None,
        }
        if i in {5, 10, 15, 20, 25}:
            bar_event["signals"] = [
                {
                    "pattern": "h2",
                    "side": "long",
                    "signal_bar_idx": i,
                    "entry_px": price + 1.0,
                    "stop_px": price - 1.0,
                    "target_px": price + 3.0,
                    "probability": 0.55,
                    "quality": 0.7,
                    "reasoning": "test",
                    "source": "rule",
                    "meta": {},
                }
            ]
        if i in {10, 20}:
            bar_event["decision"] = {
                "symbol": SYMBOL,
                "side": "long",
                "entry_px": price + 1.0,
                "stop_px": price - 1.0,
                "target_px": price + 3.0,
                "quantity": 0.5,
                "probability": 0.55,
                "expected_r": 1.5,
                "regime": "weak_bull_trend",
                "htf_aligned": True,
                "signals": bar_event["signals"],
                "source": "rule",
                "reasoning": "decision",
            }
        db.add_session_log(
            session_id=SESSION_ID,
            timestamp=f"2026-04-25T00:{i:02d}:00",
            level="DEBUG",
            source="brooks_bar",
            message=f"bar={i}",
            extra=bar_event,
        )


def _seed_fills(db: SessionDB) -> None:
    base_ns = 1_700_000_000_000_000_000
    fills: List[Dict[str, Any]] = [
        {
            "timestamp": _ns_to_iso(base_ns + 11 * BAR_NS_STEP),
            "symbol": SYMBOL,
            "type": "buy",
            "price": 105.0,
            "quantity": 0.5,
        },
        {
            "timestamp": _ns_to_iso(base_ns + 21 * BAR_NS_STEP),
            "symbol": SYMBOL,
            "type": "sell",
            "price": 110.0,
            "quantity": 0.5,
        },
    ]
    db.add_trades(SESSION_ID, fills)


def _seed_equity(db: SessionDB) -> None:
    base_ns = 1_700_000_000_000_000_000
    points: List[Dict[str, Any]] = []
    eq = 100_000.0
    for i in range(30):
        eq += 5.0 if i in {12, 22} else 0.0
        points.append(
            {
                "timestamp": _ns_to_iso(base_ns + i * BAR_NS_STEP),
                "total_equity": eq,
            }
        )
    db.add_equity_points(SESSION_ID, points)


def _ns_to_iso(ts_ns: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadTimeline:
    def test_returns_full_session_metadata(self, session_db):
        timeline = load_timeline(SESSION_ID, session_db)
        assert timeline.session_id == SESSION_ID
        assert timeline.symbol == SYMBOL
        assert timeline.base_interval == INTERVAL
        assert timeline.htf_intervals == HTF_INTERVALS
        assert timeline.config["analyst"] == "rule"
        assert timeline.config["mtf_intervals"] == HTF_INTERVALS
        # No session_kind in params → default to "live".
        assert timeline.session_kind == "live"

    def test_event_limit_caps_events_but_not_bars(self, session_db):
        """Pagination — first call truncates events, ``next_event_seq`` and
        ``has_more_events`` flag tells the frontend where to resume. The
        full bars list still ships so the chart renders any range."""
        full = load_timeline(SESSION_ID, session_db)
        assert len(full.events) == 30
        assert full.has_more_events is False
        assert full.next_event_seq == 0

        page1 = load_timeline(SESSION_ID, session_db, event_limit=10)
        assert len(page1.events) == 10
        assert len(page1.bars) == 30  # bars are not paginated
        assert page1.has_more_events is True
        assert page1.next_event_seq > 0

    def test_loader_caps_legacy_unbounded_swings(self, tmp_path, monkeypatch):
        """Older session_logs persisted the full cumulative swing list every
        bar (O(N²) JSON). The loader must clamp them on read so memory
        stays bounded for tens of thousands of bars."""
        db_path = tmp_path / "fat.db"
        monkeypatch.setenv("SESSION_DB_PATH", str(db_path))
        db = SessionDB(str(db_path))
        db.create_session(
            session_id="fat-1",
            strategy_name="brooks",
            symbol=SYMBOL,
            mode="paper",
            start_date="2026-04-25",
            end_date=None,
            params={"analyst": "rule", "session_kind": "live", "base_interval": INTERVAL},
            market="crypto",
            interval=INTERVAL,
        )
        # One bar with 5000 swings persisted into the structure dict.
        ts_ns = 1_700_000_000_000_000_000
        big_swings = [{"idx": i, "kind": "high", "price": 100.0 + i} for i in range(5_000)]
        db.add_session_log(
            session_id="fat-1",
            timestamp="2026-04-25T00:00:00",
            level="DEBUG",
            source="brooks_bar",
            message="bar_idx=0",
            extra={
                "bar_idx": 0,
                "timestamp_ns": ts_ns,
                "bar": {"timestamp_ns": ts_ns, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0},
                "structure": {"always_in": "neutral", "confirmed_swings": big_swings},
            },
        )
        timeline = load_timeline("fat-1", db)
        assert len(timeline.events) == 1
        ev = timeline.events[0]
        assert ev.structure is not None
        # MAX_SWINGS_IN_VIEW * 2 covers highs + lows; cap should bite.
        assert len(ev.structure.confirmed_swings) <= 400

    def test_session_kind_replay_when_params_set(self, tmp_path, monkeypatch):
        """A session created with ``session_kind='replay'`` round-trips through the loader."""
        db_path = tmp_path / "replay.db"
        monkeypatch.setenv("SESSION_DB_PATH", str(db_path))
        db = SessionDB(str(db_path))
        db.create_session(
            session_id="replay-fixture-1",
            strategy_name="brooks",
            symbol=SYMBOL,
            mode="paper",
            start_date="2026-04-25",
            end_date=None,
            params={"analyst": "rule", "session_kind": "replay", "base_interval": INTERVAL},
            market="crypto",
            interval=INTERVAL,
        )
        timeline = load_timeline("replay-fixture-1", db)
        assert timeline.session_kind == "replay"

    def test_bars_match_log_count(self, session_db):
        timeline = load_timeline(SESSION_ID, session_db)
        assert len(timeline.bars) == 30
        assert len(timeline.events) == 30
        # bar_idx is monotonic and matches the log order.
        assert [e.bar_idx for e in timeline.events] == list(range(30))

    def test_signal_bars_carry_decisions_and_signals(self, session_db):
        timeline = load_timeline(SESSION_ID, session_db)
        signal_bars = [e for e in timeline.events if e.signals]
        assert {e.bar_idx for e in signal_bars} == {5, 10, 15, 20, 25}

        decisions = [e for e in timeline.events if e.decision is not None]
        assert {e.bar_idx for e in decisions} == {10, 20}
        assert decisions[0].decision.side == "long"

    def test_fills_attach_to_their_bar(self, session_db):
        timeline = load_timeline(SESSION_ID, session_db)
        fills = [e for e in timeline.events if e.fill is not None]
        assert {e.bar_idx for e in fills} == {11, 21}
        assert fills[0].fill.side == "buy"
        assert fills[1].fill.side == "sell"

    def test_pnl_curve_uses_per_bar_r(self, session_db):
        timeline = load_timeline(SESSION_ID, session_db)
        assert len(timeline.pnl_curve) == 30
        # Cumulative: 0 ... 0.1 (idx=11) ... 0.2 (idx=21+)
        last = timeline.pnl_curve[-1]
        assert last.equity_r == pytest.approx(0.2, abs=1e-6)

    def test_htf_bars_populated(self, session_db):
        timeline = load_timeline(SESSION_ID, session_db)
        assert "1h" in timeline.htf_bars
        # Each per-bar log adds a unique HTF row by timestamp_ns.
        assert len(timeline.htf_bars["1h"]) == 30

    def test_unknown_session_raises(self, session_db):
        with pytest.raises(KeyError):
            load_timeline("does-not-exist", session_db)


class TestLoadPage:
    def test_pagination_consistency_since_split_equals_full(self, session_db):
        loader = BrooksTimelineLoader(session_db)
        full = loader.load(SESSION_ID).events

        page_a = loader.load_page(SESSION_ID, since_seq=0, limit=10)
        page_b = loader.load_page(SESSION_ID, since_seq=page_a.next_seq, limit=10)
        page_c = loader.load_page(SESSION_ID, since_seq=page_b.next_seq, limit=50)

        merged_ids = (
            [e.bar_idx for e in page_a.events] + [e.bar_idx for e in page_b.events] + [e.bar_idx for e in page_c.events]
        )
        assert merged_ids == [e.bar_idx for e in full]
        assert page_a.has_more is True
        assert page_c.has_more is False

    def test_since_seq_zero_returns_first_chunk(self, session_db):
        page = BrooksTimelineLoader(session_db).load_page(SESSION_ID, since_seq=0, limit=5)
        assert [e.bar_idx for e in page.events] == [0, 1, 2, 3, 4]
        assert page.next_seq > 0
        assert page.has_more is True

    def test_since_seq_past_end_is_empty(self, session_db):
        page = BrooksTimelineLoader(session_db).load_page(SESSION_ID, since_seq=10**9, limit=10)
        assert page.events == []
        assert page.has_more is False
