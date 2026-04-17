"""
API endpoint integration tests using FastAPI TestClient.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    # Mock heavy imports before loading server
    with patch("src.api.server.session_db") as mock_db, \
         patch("src.api.server.session_service") as mock_svc, \
         patch("src.api.server.persistence"):
        mock_db.get_all_sessions.return_value = []
        mock_svc.count_running_sessions.return_value = 0
        from src.api.server import app
        yield TestClient(app)


class TestHealthEndpoints:
    """Public endpoints that don't require auth."""

    def test_status(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "up"
        assert "active_sessions" in data


class TestAuthEndpoints:
    def test_auth_me_no_token_when_disabled(self, client):
        """When auth is disabled, /auth/me should work without token."""
        resp = client.get("/auth/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["auth_enabled"] is False

    def test_login_fails_when_disabled(self, client):
        resp = client.post("/auth/login", json={"username": "admin", "password": "admin"})
        assert resp.status_code == 400  # auth not enabled


class TestAuthEnabled:
    """Tests with auth enabled."""

    @pytest.fixture
    def auth_client(self, client):
        """Enable auth and provide login helper."""
        from src.config.settings import get_settings
        settings = get_settings()
        original = settings.auth.enabled
        settings.auth.enabled = True
        yield client
        settings.auth.enabled = original

    def test_protected_endpoint_requires_token(self, auth_client):
        resp = auth_client.get("/sessions")
        assert resp.status_code == 401

    def test_login_and_access(self, auth_client):
        # Login
        resp = auth_client.post("/auth/login", json={"username": "admin", "password": "admin"})
        assert resp.status_code == 200
        token = resp.json()["access_token"]

        # Access protected endpoint
        resp = auth_client.get("/sessions", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_login_wrong_password(self, auth_client):
        resp = auth_client.post("/auth/login", json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401


class TestInputValidation:
    """Test that invalid inputs are rejected."""

    def test_session_invalid_date(self, client):
        resp = client.post("/session/run", json={
            "strategy": "test",
            "symbol": "sh.600000",
            "start_date": "not-a-date",
        })
        assert resp.status_code == 422  # Pydantic validation error

    def test_session_invalid_symbol(self, client):
        resp = client.post("/session/run", json={
            "strategy": "test",
            "symbol": "'; DROP TABLE --",
            "start_date": "2024-01-01",
        })
        assert resp.status_code == 422

    def test_session_invalid_mode(self, client):
        resp = client.post("/session/run", json={
            "strategy": "test",
            "symbol": "sh.600000",
            "start_date": "2024-01-01",
            "mode": "invalid_mode",
        })
        assert resp.status_code == 422

    def test_benchmark_invalid_date(self, client):
        resp = client.get("/market/benchmark?symbol=sh.000300&start_date=bad-date")
        assert resp.status_code == 422


class TestDBColumnValidation:
    """Test SQL injection prevention via column/table allowlists."""

    def test_valid_columns(self):
        from src.market_data.db import _VALID_STOCK_COLUMNS, _validate_columns
        result = _validate_columns(["close", "open", "volume"], _VALID_STOCK_COLUMNS)
        assert result == ["close", "open", "volume"]

    def test_invalid_columns_rejected(self):
        from src.market_data.db import _VALID_STOCK_COLUMNS, _validate_columns
        with pytest.raises(ValueError, match="不允许的列名"):
            _validate_columns(["close", "DROP TABLE"], _VALID_STOCK_COLUMNS)

    def test_valid_table(self):
        from src.market_data.db import _validate_table
        _validate_table("stock_data.stock_daily")

    def test_invalid_table_rejected(self):
        from src.market_data.db import _validate_table
        with pytest.raises(ValueError, match="不允许的表名"):
            _validate_table("system.users")

    def test_identifier_validation(self):
        from src.market_data.db import _validate_identifier
        assert _validate_identifier("code") == "code"
        with pytest.raises(ValueError):
            _validate_identifier("'; DROP TABLE --")
