"""
Pydantic Settings for the quantitative trading platform.
Modern configuration management with environment variable support.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RealtimeDataStreamConfig(BaseModel):
    """Realtime data stream configuration"""

    interval_seconds: int = 60
    data_source: str = "akshare"
    enable_websocket: bool = False


class DataStreamConfig(BaseModel):
    """Data stream configuration"""

    chunk_size_months: Optional[int] = None
    enable_memory_monitoring: bool = True
    realtime: RealtimeDataStreamConfig = Field(default_factory=RealtimeDataStreamConfig)


class BacktestBrokerConfig(BaseModel):
    """Backtest broker configuration"""

    initial_cash: float = 1000000.0
    commission: float = 0.0001
    slippage: float = 0.0


class LiveBrokerConfig(BaseModel):
    """Live broker configuration"""

    server_url: str = "http://localhost:11122"
    timeout_seconds: int = 30
    retry_attempts: int = 3


class BrokerConfig(BaseModel):
    """Broker configuration"""

    backtest: BacktestBrokerConfig = Field(default_factory=BacktestBrokerConfig)
    live: LiveBrokerConfig = Field(default_factory=LiveBrokerConfig)


class DatabaseConfig(BaseModel):
    """ClickHouse connection configuration."""

    host: str = "localhost"
    port: int = 8123
    username: str = "default"
    password: str = ""


class APISessionConfig(BaseModel):
    """API session configuration"""

    persist_to_db: bool = True
    clear_memory_on_persist: bool = True
    update_db_interval_bars: int = 10


class APIConfig(BaseModel):
    """API configuration"""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    session: APISessionConfig = Field(default_factory=APISessionConfig)
    enable_incremental_updates: bool = True
    max_response_size_mb: int = 10


class ConsoleConfig(BaseModel):
    """Console logging configuration"""

    enabled: bool = True
    colorize: bool = True


class FileConfig(BaseModel):
    """File logging configuration"""

    enabled: bool = True
    path: str = "logs/quant.log"


class LoggingConfig(BaseModel):
    """Logging configuration"""

    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    rotation: str = "100 MB"
    retention: str = "30 days"
    file: FileConfig = Field(default_factory=FileConfig)
    console: ConsoleConfig = Field(default_factory=ConsoleConfig)


class AlertsConfig(BaseModel):
    """Alerts configuration"""

    memory_usage_mb: int = 8000
    api_response_time_ms: int = 1000


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""

    enabled: bool = True
    metrics: List[str] = Field(
        default_factory=lambda: [
            "memory_usage",
            "api_response_time",
            "backtest_throughput",
        ]
    )
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)


class TelegramConfig(BaseModel):
    """Telegram notification configuration"""

    enabled: bool = False
    bot_token: str = ""
    default_chat_id: str = ""
    api_base_url: str = "https://api.telegram.org"
    timeout_seconds: int = 10


class NotificationsConfig(BaseModel):
    """Notification configuration"""

    telegram: TelegramConfig = Field(default_factory=TelegramConfig)


class FeaturesConfig(BaseModel):
    """Features configuration"""

    enable_websocket: bool = False
    enable_parameter_optimization: bool = False
    enable_multi_strategy: bool = False
    enable_risk_manager: bool = False


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = SettingsConfigDict(
        env_prefix="QUANT_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    data_stream: DataStreamConfig = Field(default_factory=DataStreamConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)

    @property
    def api_host(self) -> str:
        return self.api.host

    @property
    def api_port(self) -> int:
        return self.api.port

    @property
    def initial_cash(self) -> float:
        return self.broker.backtest.initial_cash

    @property
    def commission(self) -> float:
        return self.broker.backtest.commission


def _load_yaml_settings() -> Dict[str, Any]:
    """Load project defaults from YAML when available."""
    config_path = Path(os.getenv("QUANT_CONFIG_PATH", "config/system_config.yaml"))
    if not config_path.exists():
        return {}

    try:
        raw_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    return raw_data if isinstance(raw_data, dict) else {}


def _load_env_overrides() -> Dict[str, Any]:
    """Load QUANT_* environment variables as nested settings overrides."""
    prefix = "QUANT_"
    overrides: Dict[str, Any] = {}

    for env_key, value in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        path = env_key[len(prefix) :].lower().split("__")
        cursor = overrides
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = value

    return overrides


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two settings dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    merged_settings = _load_yaml_settings()
    merged_settings = _deep_merge(merged_settings, _load_env_overrides())
    return Settings(**merged_settings)


def get_data_stream_config() -> DataStreamConfig:
    """Get data stream configuration"""
    return get_settings().data_stream


def get_broker_config() -> BrokerConfig:
    """Get broker configuration"""
    return get_settings().broker


def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return get_settings().database


def get_api_config() -> APIConfig:
    """Get API configuration"""
    return get_settings().api


def get_logging_config() -> LoggingConfig:
    """Get logging configuration"""
    return get_settings().logging


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration"""
    return get_settings().monitoring


def get_notifications_config() -> NotificationsConfig:
    """Get notification configuration"""
    return get_settings().notifications


def get_features_config() -> FeaturesConfig:
    """Get features configuration"""
    return get_settings().features
