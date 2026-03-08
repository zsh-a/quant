from session_db import SessionDB
from datetime import datetime

from src.automation.service import AutomationService
from src.core.backtest_broker import BacktestBroker
from src.core.base import Bar
from src.core.base import Order
from src.tasks.automation import (
    _collect_simulation_order_notification,
    _send_batched_simulation_order_notifications,
    send_telegram_validation_notification_task,
)


def test_simulation_job_and_run_persistence(tmp_path):
    db = SessionDB(str(tmp_path / 'sessions.sqlite'))
    service = AutomationService(db)

    job = service.create_job(
        name='auto-jsg',
        strategy='jsg',
        symbol='sh.000300',
        start_date='2024-01-01',
        end_date='2024-01-31',
        params={'lookback': 20},
        notification={'telegram': {'enabled': True, 'chat_id': '10001'}},
    )
    assert job['enabled'] is True
    assert job['params']['lookback'] == 20
    assert job['notification']['telegram']['chat_id'] == '10001'
    assert job['end_date'] == '2024-01-31'

    db.update_simulation_job(job['job_id'], last_processed_at='2024-01-10')
    window = service.get_run_window(db.get_simulation_job(job['job_id']), latest_market_date='2024-01-12')
    assert window == {'start_date': '2024-01-11', 'end_date': '2024-01-12', 'force_full_replay': False}

    db.create_session(
        session_id='session-1',
        strategy_name='jsg',
        symbol='sh.000300',
        mode='simulation',
        start_date='2024-01-11',
        end_date='2024-01-12',
        params={'lookback': 20},
        source='automation',
        job_id=job['job_id'],
        run_id='run-1',
    )
    run = db.create_simulation_run(
        job_id=job['job_id'],
        session_id='session-1',
        start_date='2024-01-11',
        end_date='2024-01-12',
        trigger_source='data_update',
    )
    db.add_simulation_run_step(
        run['run_id'],
        'session-1',
        step_index=1,
        timestamp='2024-01-11T00:00:00',
        event_type='strategy_step',
        payload={'progress': 50, 'new_trades': []},
    )
    steps = db.list_simulation_run_steps(run['run_id'])

    assert len(steps) == 1
    assert steps[0]['payload']['progress'] == 50
    assert steps[0]['payload']['new_trades'] == []


def test_list_simulation_jobs_omits_snapshot_by_default(tmp_path):
    db = SessionDB(str(tmp_path / 'summary.sqlite'))
    job = db.create_simulation_job(
        name='summary-job',
        strategy_name='jsg',
        symbol='sh.000300',
        start_date='2024-01-01',
        params={'lookback': 20},
        notification={'telegram': {'enabled': True}},
    )
    db.update_simulation_job(job['job_id'], snapshot={'cash': 123456.0, 'positions': {'sh.000300': 100}})

    listed = db.list_simulation_jobs()
    detailed = db.get_simulation_job(job['job_id'])

    assert 'snapshot' not in listed[0]
    assert detailed['snapshot']['cash'] == 123456.0


def test_data_update_run_history(tmp_path):
    db = SessionDB(str(tmp_path / 'updates.sqlite'))
    created = db.create_data_update_run(trigger_source='schedule')
    db.update_data_update_run(
        created['update_run_id'],
        status='success',
        has_new_data=True,
        details={'steps': [{'step': 'kline_daily', 'status': 'success'}]},
    )

    history = db.list_data_update_runs(limit=5)
    assert len(history) == 1
    assert history[0]['has_new_data'] is True
    assert history[0]['details']['steps'][0]['step'] == 'kline_daily'


def test_get_run_window_skips_when_no_new_data(tmp_path):
    db = SessionDB(str(tmp_path / 'window.sqlite'))
    service = AutomationService(db)
    job = service.create_job(
        name='window-test',
        strategy='jsg',
        symbol='sh.000300',
        start_date='2024-01-01',
        params={},
    )

    db.update_simulation_job(job['job_id'], last_processed_at='2024-01-12')
    current_job = db.get_simulation_job(job['job_id'])

    assert service.get_run_window(current_job, latest_market_date='2024-01-12') is None


def test_get_run_window_force_full_replay_ignores_last_processed(tmp_path):
    db = SessionDB(str(tmp_path / 'force-window.sqlite'))
    service = AutomationService(db)
    job = service.create_job(
        name='force-window-test',
        strategy='jsg',
        symbol='sh.000300',
        start_date='2024-01-01',
        params={},
    )

    db.update_simulation_job(job['job_id'], last_processed_at='2024-01-12')
    current_job = db.get_simulation_job(job['job_id'])
    window = service.get_run_window(
        current_job,
        latest_market_date='2024-01-12',
        force_full_replay=True,
    )

    assert window == {
        'start_date': '2024-01-01',
        'end_date': '2024-01-12',
        'force_full_replay': True,
    }


def test_get_run_window_respects_job_end_date(tmp_path):
    db = SessionDB(str(tmp_path / 'end-date-window.sqlite'))
    service = AutomationService(db)
    job = service.create_job(
        name='end-date-window-test',
        strategy='jsg',
        symbol='sh.000300',
        start_date='2024-01-01',
        end_date='2024-01-10',
        params={},
    )

    window = service.get_run_window(job, latest_market_date='2024-01-20')

    assert window == {
        'start_date': '2024-01-01',
        'end_date': '2024-01-10',
        'force_full_replay': False,
    }


def test_get_run_window_continuation_ignores_initial_end_date(tmp_path):
    db = SessionDB(str(tmp_path / 'continue-latest.sqlite'))
    service = AutomationService(db)
    job = service.create_job(
        name='continue-latest-test',
        strategy='jsg',
        symbol='sh.000300',
        start_date='2024-01-01',
        end_date='2024-01-10',
        params={},
    )

    db.update_simulation_job(job['job_id'], last_processed_at='2024-01-05')
    current_job = db.get_simulation_job(job['job_id'])
    window = service.get_run_window(current_job, latest_market_date='2024-01-20')

    assert window == {
        'start_date': '2024-01-06',
        'end_date': '2024-01-20',
        'force_full_replay': False,
    }


def test_reuse_session_updates_context_and_can_reset_runtime_data(tmp_path):
    db = SessionDB(str(tmp_path / 'reuse-session.sqlite'))
    db.create_session(
        session_id='session-reuse',
        strategy_name='jsg',
        symbol='sh.000300',
        mode='simulation',
        start_date='2024-01-01',
        end_date='2024-01-05',
        params={'lookback': 20},
        source='automation',
        job_id='job-1',
        run_id='run-1',
        last_processed_at='2024-01-05',
    )
    db.add_equity_point('session-reuse', '2024-01-05T00:00:00', 100000.0)
    db.add_trade('session-reuse', {'timestamp': '2024-01-05T00:00:00', 'symbol': 'sh.000300', 'type': 'buy', 'price': 1.0, 'quantity': 100})
    db.add_session_log('session-reuse', '2024-01-05T00:00:00', 'INFO', 'strategy', 'hello', {'step': 1})

    db.update_session(
        'session-reuse',
        run_id='run-2',
        end_date='2024-01-10',
        status='running',
        progress=0.0,
        error=None,
    )
    session = db.get_session('session-reuse')
    assert session['run_id'] == 'run-2'
    assert session['end_date'] == '2024-01-10'

    db.clear_session_runtime_data('session-reuse')
    assert db.get_equity_history('session-reuse') == []
    assert db.get_trades('session-reuse') == []
    assert db.get_session_logs('session-reuse') == []


def test_trade_payload_keeps_snapshot_after_broker_buffer_clear():
    broker_trades = [
        {
            'timestamp': '2024-01-05T00:00:00',
            'symbol': 'sh.000300',
            'type': 'buy',
            'price': 1.0,
            'quantity': 100,
        }
    ]
    info = {'trades': broker_trades}

    new_trades = list(info.get('trades', []))
    broker_trades.clear()

    assert len(new_trades) == 1
    assert new_trades[0]['symbol'] == 'sh.000300'


def test_collect_simulation_order_notification_uses_job_chat_id(monkeypatch):
    monkeypatch.setattr("src.tasks.automation.notifications_config.telegram.enabled", True)
    monkeypatch.setattr("src.tasks.automation.notifications_config.telegram.default_chat_id", "fallback-chat")

    broker = type(
        "BrokerStub",
        (),
        {
            "current_bars": {},
            "last_prices": {"sh.000300": 3.18},
            "stock_names": {"sh.000300": "沪深300"},
        },
    )()

    notification = _collect_simulation_order_notification(
        job={
            "job_id": "job-1",
            "name": "daily-jsg",
            "strategy_name": "jsg",
            "symbol": "sh.000300",
            "notification": {"telegram": {"enabled": True, "chat_id": "job-chat"}},
        },
        session_id="session-1",
        run_id="run-1",
        broker=broker,
        order=Order("sh.000300", "buy", 100),
    )

    assert notification["chat_id"] == "job-chat"
    assert notification["job_name"] == "daily-jsg"
    assert notification["strategy_name"] == "jsg"
    assert "买入" in notification["order_summary"]
    assert "NEXT_OPEN" in notification["order_summary"]


def test_collect_simulation_order_notification_skips_disabled_jobs(monkeypatch):
    monkeypatch.setattr("src.tasks.automation.notifications_config.telegram.enabled", True)

    broker = type(
        "BrokerStub",
        (),
        {"current_bars": {}, "last_prices": {}, "stock_names": {}},
    )()

    notification = _collect_simulation_order_notification(
        job={
            "job_id": "job-1",
            "name": "daily-jsg",
            "strategy_name": "jsg",
            "symbol": "sh.000300",
            "notification": {"telegram": {"enabled": False}},
        },
        session_id="session-1",
        run_id="run-1",
        broker=broker,
        order=Order("sh.000300", "buy", 100),
    )

    assert notification is None


def test_send_batched_simulation_order_notifications_merges_messages(monkeypatch):
    sent_messages = []

    class DummyNotifier:
        def __init__(self, config):
            self.config = config

        def is_enabled(self):
            return True

        def send_message(self, chat_id, text):
            sent_messages.append({"chat_id": chat_id, "text": text})
            return True

    monkeypatch.setattr("src.tasks.automation.TelegramNotifier", DummyNotifier)
    monkeypatch.setattr("src.tasks.automation.notifications_config.telegram.enabled", True)
    monkeypatch.setattr("src.tasks.automation.notifications_config.telegram.bot_token", "token")

    _send_batched_simulation_order_notifications(
        [
            {
                "chat_id": "job-chat",
                "job_name": "daily-jsg",
                "strategy_name": "jsg",
                "session_id": "session-1",
                "run_id": "run-1",
                "bar_timestamp": "2024-01-05T09:30:00",
                "order_summary": "买入 `100` | `AAA` | 订单价 `市价` | 执行 `NEXT_OPEN` | 参考 `3.1800`",
            },
            {
                "chat_id": "job-chat",
                "job_name": "daily-jsg",
                "strategy_name": "jsg",
                "session_id": "session-1",
                "run_id": "run-1",
                "bar_timestamp": "2024-01-05T09:30:00",
                "order_summary": "卖出 `50` | `BBB` | 订单价 `市价` | 执行 `IMMEDIATE_CLOSE` | 参考 `2.4500`",
            },
        ]
    )

    assert len(sent_messages) == 1
    assert sent_messages[0]["chat_id"] == "job-chat"
    assert "模拟运行订单通知" in sent_messages[0]["text"]
    assert "*Bar 时间*: `2024-01-05T09:30:00`" in sent_messages[0]["text"]
    assert "*订单数*: `2`" in sent_messages[0]["text"]
    assert "1. 买入 `100`" in sent_messages[0]["text"]
    assert "2. 卖出 `50`" in sent_messages[0]["text"]


def test_backtest_broker_submit_order_uses_simulation_bar_timestamp():
    broker = BacktestBroker()
    simulation_ts = datetime(2024, 1, 5, 9, 30, 0)
    broker.current_bars = {
        "sh.000300": Bar(
            symbol="sh.000300",
            timestamp=simulation_ts,
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1000,
            amount=1000,
        )
    }

    order = Order("sh.000300", "buy", 100)
    broker.submit_order(order)

    assert order.created_at == simulation_ts


def test_send_telegram_validation_notification_uses_default_chat_id(monkeypatch):
    sent_messages = []

    class DummyNotifier:
        def __init__(self, config):
            self.config = config

        def is_enabled(self):
            return True

        def send_message(self, chat_id, text):
            sent_messages.append({"chat_id": chat_id, "text": text})
            return True

    monkeypatch.setattr("src.tasks.automation.TelegramNotifier", DummyNotifier)
    monkeypatch.setattr("src.tasks.automation.notifications_config.telegram.enabled", True)
    monkeypatch.setattr("src.tasks.automation.notifications_config.telegram.bot_token", "token")
    monkeypatch.setattr("src.tasks.automation.notifications_config.telegram.default_chat_id", "fallback-chat")

    result = send_telegram_validation_notification_task(context="test")

    assert result["status"] == "success"
    assert result["chat_id"] == "fallback-chat"
    assert len(sent_messages) == 1
    assert sent_messages[0]["chat_id"] == "fallback-chat"
    assert "Telegram 验证通知" in sent_messages[0]["text"]


def test_send_telegram_validation_notification_requires_chat_id(monkeypatch):
    class DummyNotifier:
        def __init__(self, config):
            self.config = config

        def is_enabled(self):
            return True

    monkeypatch.setattr("src.tasks.automation.TelegramNotifier", DummyNotifier)
    monkeypatch.setattr("src.tasks.automation.notifications_config.telegram.enabled", True)
    monkeypatch.setattr("src.tasks.automation.notifications_config.telegram.bot_token", "token")
    monkeypatch.setattr("src.tasks.automation.notifications_config.telegram.default_chat_id", "")

    result = send_telegram_validation_notification_task(context="test")

    assert result["status"] == "error"
    assert result["error"] == "telegram chat_id is missing"
