from src.utils.session_logger import (
    get_session_logger,
    get_session_logs,
    list_session_loggers,
    clear_session_logs,
    remove_session_logger,
    reset_session_log_store,
)


def test_session_logs_are_persisted_and_reloaded(tmp_path, monkeypatch):
    db_path = tmp_path / "session-logs.sqlite"
    monkeypatch.setenv("SESSION_DB_PATH", str(db_path))
    reset_session_log_store()

    session_id = "session-persisted"
    collector = get_session_logger(session_id)
    collector.info("strategy", "first message", step=1)
    collector.error("engine", "second message", step=2)
    collector.flush(force=True)
    remove_session_logger(session_id)

    reset_session_log_store()

    logs = get_session_logs(session_id, limit=10)
    assert len(logs) == 2
    assert logs[0]["message"] == "first message"
    assert logs[0]["extra"]["step"] == 1
    assert logs[1]["level"] == "ERROR"
    assert session_id in list_session_loggers()


def test_clear_session_logs_removes_memory_and_persistence(tmp_path, monkeypatch):
    db_path = tmp_path / "clear-logs.sqlite"
    monkeypatch.setenv("SESSION_DB_PATH", str(db_path))
    reset_session_log_store()

    session_id = "session-clear"
    collector = get_session_logger(session_id)
    collector.warning("broker", "will be cleared")
    collector.flush(force=True)

    clear_session_logs(session_id)
    remove_session_logger(session_id)

    assert get_session_logs(session_id) == []
    assert session_id not in list_session_loggers()

    reset_session_log_store()
