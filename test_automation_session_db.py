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
        params={'lookback': 20},
    )
    assert job['enabled'] is True
    assert job['params']['lookback'] == 20

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
