import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from session_db import SessionDB


def _seed_session(db: SessionDB, session_id: str = "session-1") -> str:
    db.create_session(
        session_id=session_id,
        strategy_name="jsg",
        symbol="sh.000300",
        mode="backtest",
        start_date="2024-01-01",
        end_date="2024-01-05",
        params={},
    )
    for i in range(5):
        date = f"2024-01-0{i + 1}T00:00:00"
        db.add_equity_point(
            session_id,
            date,
            100000.0 + i,
            positions={"sh.000300": {"qty": i + 1}},
        )
        db.add_trade(
            session_id,
            {
                "timestamp": date,
                "symbol": "sh.000300",
                "type": "buy" if i % 2 == 0 else "sell",
                "price": 1.0 + i,
                "quantity": 100 + i,
            },
        )
    return session_id


def test_get_equity_history_page(tmp_path):
    db = SessionDB(str(tmp_path / "pagination.sqlite"))
    session_id = _seed_session(db)

    page = db.get_equity_history_page(session_id, limit=2, offset=1)

    assert page["total"] == 5
    assert page["limit"] == 2
    assert page["offset"] == 1
    assert page["has_more"] is True
    assert len(page["items"]) == 2
    assert page["items"][0]["timestamp"] == "2024-01-02T00:00:00"
    assert page["items"][0]["positions"]["sh.000300"]["qty"] == 2


def test_get_trades_page_with_since(tmp_path):
    db = SessionDB(str(tmp_path / "trades.sqlite"))
    session_id = _seed_session(db)

    page = db.get_trades_page(
        session_id,
        since="2024-01-02T00:00:00",
        limit=10,
        offset=0,
    )

    assert page["total"] == 3
    assert page["has_more"] is False
    assert [item["timestamp"] for item in page["items"]] == [
        "2024-01-03T00:00:00",
        "2024-01-04T00:00:00",
        "2024-01-05T00:00:00",
    ]
