from session_db import SessionDB
from src.automation.service import AutomationService


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
    )
    assert job['enabled'] is True
    assert job['params']['lookback'] == 20
    assert job['end_date'] == '2024-01-31'

    db.update_simulation_job(job['job_id'], last_processed_at='2024-01-10')
    window = service.get_run_window(db.get_simulation_job(job['job_id']), latest_market_date='2024-01-12')
    assert window == {'start_date': '2024-01-11', 'end_date': '2024-01-12'}

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
