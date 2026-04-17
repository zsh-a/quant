import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from session_db import SessionDB
from src.services.session_execution import (
    SessionExecutionConfig,
    execute_session,
)
from src.services.session_service import SessionService


def test_session_service_checkpoint_uses_persisted_data(tmp_path):
    db = SessionDB(str(tmp_path / "sessions.sqlite"))

    class FakePersistence:
        def delete_checkpoints(self, session_id):
            return 0

    persistence = FakePersistence()
    service = SessionService(db, persistence)
    service.create_session(
        session_id="session-1",
        strategy_name="jsg",
        symbol="sh.000300",
        mode="backtest",
        start_date="2024-01-01",
        end_date="2024-01-05",
        params={"lookback": 20},
    )
    db.update_session("session-1", status="completed", progress=100.0)
    db.add_equity_point(
        "session-1",
        "2024-01-05T00:00:00",
        100000.0,
        positions={"sh.000300": {"qty": 100}},
    )
    db.add_trade(
        "session-1",
        {
            "timestamp": "2024-01-05T00:00:00",
            "symbol": "sh.000300",
            "type": "buy",
            "price": 1.0,
            "quantity": 100,
        },
    )

    state, metadata = service.build_checkpoint_state("session-1")

    assert state["equity_history"][0]["total_equity"] == 100000.0
    assert state["trades"][0]["symbol"] == "sh.000300"
    assert metadata["positions"] == 1


def test_execute_session_flushes_final_buffers(monkeypatch, tmp_path):
    db = SessionDB(str(tmp_path / "execution.sqlite"))
    db.create_session(
        session_id="session-2",
        strategy_name="jsg",
        symbol="sh.000300",
        mode="backtest",
        start_date="2024-01-01",
        end_date="2024-01-02",
        params={},
    )

    class FakeDB:
        pass

    class FakeStream:
        def __init__(self, *args, **kwargs):
            self.idx = 1
            self.total_bars = 2

    class FakeBroker:
        def __init__(self, *args, **kwargs):
            self.equity_history = []
            self.trades = []

        def get_account_info(self):
            return {
                "total_equity": 123456.0,
                "equity_history": self.equity_history,
                "trades": self.trades,
                "detailed_positions": {"sh.000300": {"qty": 100}},
            }

    class FakeStrategy:
        def set_engine(self, engine):
            self.engine = engine

        def on_bar(self, bars):
            return None

    class FakeEngine:
        def __init__(self, strategy, broker, data_stream, on_step=None, risk_manager=None):
            self.broker = broker
            self.on_step = on_step
            strategy.set_engine(self)

        def run(self):
            self.broker.equity_history.append(
                {
                    "timestamp": "2024-01-02T00:00:00",
                    "total_equity": 123456.0,
                    "positions": {"sh.000300": {"qty": 100}},
                }
            )
            self.broker.trades.append(
                {
                    "timestamp": "2024-01-02T00:00:00",
                    "symbol": "sh.000300",
                    "type": "buy",
                    "price": 1.0,
                    "quantity": 100,
                }
            )
            if self.on_step:
                self.on_step({})

    monkeypatch.setattr("src.services.session_execution.DB", FakeDB)
    monkeypatch.setattr("src.services.session_execution.DBDataStream", FakeStream)
    monkeypatch.setattr("src.services.session_execution.BacktestBroker", FakeBroker)
    monkeypatch.setattr("src.services.session_execution.TradingEngine", FakeEngine)
    monkeypatch.setattr(
        "src.services.session_execution.StrategyRegistry.create_strategy",
        lambda *args, **kwargs: FakeStrategy(),
    )

    result = execute_session(
        SessionExecutionConfig(
            session_id="session-2",
            strategy="jsg",
            symbol="sh.000300",
            start_date="2024-01-01",
            end_date="2024-01-02",
        ),
        session_db=db,
    )

    assert result.status == "completed"
    assert result.total_trades == 1
    assert db.get_equity_history("session-2")[0]["total_equity"] == 123456.0
    assert db.get_trades("session-2")[0]["symbol"] == "sh.000300"
