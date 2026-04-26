"""Tests for the Brooks Studio router (REST + WebSocket).

We seed the same fixture session as the loader tests (30 bars, 5 signals,
2 fills) and exercise the three endpoints:

* full timeline GET
* incremental ``since/{seq}`` page GET
* WS subscription receives an ``emit_studio_bar_event`` payload
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

SESSION_ID = "studio-router-fixture-1"
SYMBOL = "BTC/USDT"
INTERVAL = "5m"
BAR_NS_STEP = 5 * 60 * 1_000_000_000


def _ns_to_iso(ts_ns: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).isoformat()


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """Start the FastAPI app pointing at an isolated session DB."""
    db_path = tmp_path / "sessions.db"
    monkeypatch.setenv("SESSION_DB_PATH", str(db_path))
    # Re-import server to honour the env override.
    import importlib

    from session_db import SessionDB

    db = SessionDB(str(db_path))
    db.create_session(
        session_id=SESSION_ID,
        strategy_name="brooks",
        symbol=SYMBOL,
        mode="paper",
        start_date="2026-04-25",
        end_date=None,
        params={"analyst": "rule", "mtf_intervals": ["1h"], "base_interval": INTERVAL},
        market="crypto",
        interval=INTERVAL,
    )

    base_ns = 1_700_000_000_000_000_000
    for i in range(30):
        ts_ns = base_ns + i * BAR_NS_STEP
        price = 100.0 + i * 0.5
        bar_event = {
            "bar_idx": i,
            "timestamp_ns": ts_ns,
            "bar": {
                "timestamp_ns": ts_ns,
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price + 0.3,
                "volume": 10.0,
            },
            "regime": {"name": "weak_bull_trend", "confidence": 0.5, "reasons": []},
            "features": {
                "is_bull": True,
                "body_pct": 30,
                "close_position": "high",
                "ema_relation": "above",
                "leg_dir": "up",
                "leg_length": 2,
                "is_doji": False,
                "is_inside_bar": False,
            },
            "signals": [],
            "decision": None,
            "htf": {},
            "htf_bars": {},
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
        db.add_session_log(
            session_id=SESSION_ID,
            timestamp=f"2026-04-25T00:{i:02d}:00",
            level="DEBUG",
            source="brooks_bar",
            message=f"bar={i}",
            extra=bar_event,
        )

    db.add_trades(
        SESSION_ID,
        [
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
        ],
    )

    # Reload the server module so it binds to the freshly-pointed SESSION_DB_PATH.
    import src.api.server as server_module

    importlib.reload(server_module)
    return TestClient(server_module.app)


class TestTimelineGet:
    def test_returns_full_timeline(self, app_client):
        resp = app_client.get(f"/brooks-studio/sessions/{SESSION_ID}/timeline")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["session_id"] == SESSION_ID
        assert body["symbol"] == SYMBOL
        assert body["base_interval"] == INTERVAL
        assert len(body["events"]) == 30
        assert len(body["bars"]) == 30
        # 5 signal bars + 2 fills landed on bars 11 and 21.
        sig_bars = [e for e in body["events"] if e["signals"]]
        assert {e["bar_idx"] for e in sig_bars} == {5, 10, 15, 20, 25}
        fill_bars = [e for e in body["events"] if e.get("fill")]
        assert {e["bar_idx"] for e in fill_bars} == {11, 21}

    def test_unknown_session_404(self, app_client):
        resp = app_client.get("/brooks-studio/sessions/no-such-id/timeline")
        assert resp.status_code == 404


class TestTimelinePage:
    def test_pagination_consistency(self, app_client):
        full = app_client.get(f"/brooks-studio/sessions/{SESSION_ID}/timeline").json()
        full_ids = [e["bar_idx"] for e in full["events"]]

        page_a = app_client.get(f"/brooks-studio/sessions/{SESSION_ID}/timeline/since/0?limit=10").json()
        page_b = app_client.get(
            f"/brooks-studio/sessions/{SESSION_ID}/timeline/since/{page_a['next_seq']}?limit=10"
        ).json()
        page_c = app_client.get(
            f"/brooks-studio/sessions/{SESSION_ID}/timeline/since/{page_b['next_seq']}?limit=50"
        ).json()

        merged = (
            [e["bar_idx"] for e in page_a["events"]]
            + [e["bar_idx"] for e in page_b["events"]]
            + [e["bar_idx"] for e in page_c["events"]]
        )
        assert merged == full_ids
        assert page_a["has_more"] is True
        assert page_c["has_more"] is False

    def test_negative_seq_rejected(self, app_client):
        resp = app_client.get(f"/brooks-studio/sessions/{SESSION_ID}/timeline/since/-1")
        assert resp.status_code == 422


class TestReplayBar:
    def test_runs_rule_analyst(self, app_client):
        resp = app_client.post(
            f"/brooks-studio/sessions/{SESSION_ID}/replay-bar",
            json={"bar_idx": 10, "analysts": ["rule"]},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["bar_idx"] == 10
        assert len(body["results"]) == 1
        result = body["results"][0]
        assert result["analyst"] == "rule"
        assert result["bar_idx"] == 10
        assert isinstance(result["signals"], list)
        # rule analyst is best-effort: error stays None unless something exploded
        assert result["error"] is None

    def test_unknown_analyst_returns_error_not_500(self, app_client):
        resp = app_client.post(
            f"/brooks-studio/sessions/{SESSION_ID}/replay-bar",
            json={"bar_idx": 0, "analysts": ["does-not-exist"]},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        result = body["results"][0]
        assert result["analyst"] == "does-not-exist"
        assert result["error"] is not None
        assert "unknown" in result["error"].lower()

    def test_unknown_session_404(self, app_client):
        resp = app_client.post(
            "/brooks-studio/sessions/no-such/replay-bar",
            json={"bar_idx": 0, "analysts": ["rule"]},
        )
        assert resp.status_code == 404

    def test_out_of_range_bar_idx_422(self, app_client):
        resp = app_client.post(
            f"/brooks-studio/sessions/{SESSION_ID}/replay-bar",
            json={"bar_idx": 9999, "analysts": ["rule"]},
        )
        assert resp.status_code == 422

    def test_empty_analyst_list_rejected(self, app_client):
        resp = app_client.post(
            f"/brooks-studio/sessions/{SESSION_ID}/replay-bar",
            json={"bar_idx": 0, "analysts": []},
        )
        assert resp.status_code == 422


class TestStudioWebSocket:
    def test_studio_ws_receives_bar_event(self, app_client):
        from src.api.events import emit_studio_bar_event

        payload = {
            "bar_idx": 7,
            "timestamp_ns": 1_700_000_000_000_000_000,
            "bar": {
                "timestamp_ns": 1_700_000_000_000_000_000,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 5.0,
            },
            "signals": [],
        }

        with app_client.websocket_connect(f"/ws/brooks-studio/{SESSION_ID}") as ws:
            asyncio.run(emit_studio_bar_event(SESSION_ID, payload))
            received = None
            for _ in range(5):
                msg = ws.receive_json()
                if msg.get("type") == "studio_bar_event":
                    received = msg
                    break
            assert received is not None, "studio bar event was not delivered"
            # event_bus wraps the payload under "data".
            bar_event = received.get("bar_event") or received.get("data", {}).get("bar_event")
            assert bar_event is not None, f"missing bar_event in {received}"
            assert bar_event["bar_idx"] == 7
