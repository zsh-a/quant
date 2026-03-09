import logging

from loguru import logger as loguru_logger

from src.config.settings import get_settings
from src.utils.logging_config import get_logger, setup_logging


def test_setup_logging_persists_stdlib_and_loguru_records(tmp_path, monkeypatch):
    log_path = tmp_path / "quant.log"
    monkeypatch.setenv("QUANT_LOGGING__FILE__PATH", str(log_path))
    monkeypatch.setenv("QUANT_LOGGING__CONSOLE__ENABLED", "false")
    monkeypatch.setenv("QUANT_LOGGING__LEVEL", "INFO")
    get_settings.cache_clear()

    setup_logging()

    get_logger("test.app").info("app log message", request_id="req-1")
    logging.getLogger("uvicorn.access").info(
        '127.0.0.1:12345 - "GET /health HTTP/1.1" 200'
    )
    loguru_logger.complete()

    content = log_path.read_text(encoding="utf-8")
    assert "app log message" in content
    assert 'GET /health HTTP/1.1" 200' in content
