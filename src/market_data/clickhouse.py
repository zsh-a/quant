from __future__ import annotations

from typing import Any, Dict

import clickhouse_connect

from src.config.settings import get_database_config


def get_clickhouse_connection_options() -> Dict[str, Any]:
    config = get_database_config()
    return {
        "host": config.host,
        "port": config.port,
        "username": config.username,
        "password": config.password,
    }


def create_clickhouse_client():
    return clickhouse_connect.get_client(**get_clickhouse_connection_options())
