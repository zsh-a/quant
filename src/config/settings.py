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

from src.config import paths


class RealtimeDataStreamConfig(BaseModel):
    """Realtime data stream configuration"""

    interval_seconds: int = 60
    data_source: str = "akshare"


class DataStreamConfig(BaseModel):
    """Data stream configuration"""

    chunk_size_months: Optional[int] = None
    realtime: RealtimeDataStreamConfig = Field(default_factory=RealtimeDataStreamConfig)


class BacktestBrokerConfig(BaseModel):
    """Backtest broker configuration"""

    initial_cash: float = 1000000.0
    commission: float = 0.0001
    slippage: float = 0.001


class LiveBrokerConfig(BaseModel):
    """Live broker configuration"""

    server_url: str = "http://localhost:11122"


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


class APIConfig(BaseModel):
    """API configuration"""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


class ConsoleConfig(BaseModel):
    """Console logging configuration"""

    enabled: bool = True
    colorize: bool = True


class FileConfig(BaseModel):
    """File logging configuration"""

    enabled: bool = True
    path: str = str(paths.LOG_FILE_PATH)


class LoggingConfig(BaseModel):
    """Logging configuration"""

    level: str = "INFO"
    rotation: str = "100 MB"
    retention: str = "30 days"
    file: FileConfig = Field(default_factory=FileConfig)
    console: ConsoleConfig = Field(default_factory=ConsoleConfig)


class TelegramConfig(BaseModel):
    """Telegram notification configuration"""

    enabled: bool = False
    bot_token: str = ""
    default_chat_id: str = ""
    api_base_url: str = "https://api.telegram.org"
    timeout_seconds: int = 10


class WebhookConfig(BaseModel):
    """Generic webhook notification configuration (Slack/Discord/custom)."""

    enabled: bool = False
    url: str = ""
    format: str = "plain"  # slack | discord | plain
    timeout_seconds: int = 10


class NotificationsConfig(BaseModel):
    """Notification configuration"""

    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)


class AuthConfig(BaseModel):
    """Authentication configuration."""

    enabled: bool = False
    secret_key: str = "quent-change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24h
    default_username: str = "admin"
    default_password: str = "admin"


class AlphaLabConfig(BaseModel):
    """Configuration for the alpha lab research pipeline."""

    population_size: int = 64
    train_ratio: float = 0.5
    valid_ratio: float = 0.2
    test_ratio: float = 0.2
    llm_backend: str = "remote_api"
    max_ast_nodes: int = 24


class LiveTradingConfig(BaseModel):
    """Live trading operational parameters."""

    order_poll_interval_seconds: float = 1.0
    order_timeout_seconds: int = 30
    max_consecutive_errors: int = 5
    error_cooldown_seconds: int = 60
    auto_flatten_at_close: bool = False
    notify_on_fill: bool = True
    notify_on_rejection: bool = True
    equity_sync_interval_seconds: int = 10


class BitgetConfig(BaseModel):
    """Bitget market data defaults."""

    base_url: str = "https://api.bitget.com"
    product_type: str = "USDT-FUTURES"
    default_symbols: List[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    request_timeout_seconds: int = 10


class CryptoMarketConfig(BaseModel):
    """Unified crypto minute-data ingestion defaults."""

    default_provider: str = "bitget"
    default_interval: str = "1m"
    default_symbols: List[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"])
    default_lookback_hours: int = 24
    full_history_start: str = "2020-01-01T00:00:00+00:00"
    state_file: str = str(paths.CRYPTO_SYNC_STATE_PATH)


class BrooksLiveConfig(BaseModel):
    """Phase 4.6 BrooksStrategy paper-trading runtime parameters."""

    default_symbol: str = "BTC/USDT"
    default_interval: str = "5m"
    default_exchange: str = "binance"
    default_analyst: str = "rule"
    initial_cash: float = 100_000.0
    commission: float = 0.0003
    slippage: float = 0.001
    equity_sync_interval_seconds: float = 5.0
    bar_queue_timeout_seconds: float = 120.0
    # LLM/VLM rate limits — avoid exploding the budget.
    llm_daily_budget_usd: float = 10.0
    llm_min_interval_seconds: float = 60.0
    recent_signals_window: int = 50
    decision_history_window: int = 200
    # Where incremental hit-rate samples are appended by the daily close task.
    hit_rate_samples_path: str = "data/brooks/hit_rate_samples.parquet"
    hit_rate_table_path: str = "data/brooks/hit_rate_table.parquet"


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
    auth: AuthConfig = Field(default_factory=AuthConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    alpha_lab: AlphaLabConfig = Field(default_factory=AlphaLabConfig)
    live_trading: LiveTradingConfig = Field(default_factory=LiveTradingConfig)
    bitget: BitgetConfig = Field(default_factory=BitgetConfig)
    crypto_market: CryptoMarketConfig = Field(default_factory=CryptoMarketConfig)
    brooks: BrooksLiveConfig = Field(default_factory=BrooksLiveConfig)


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


def get_notifications_config() -> NotificationsConfig:
    """Get notification configuration"""
    return get_settings().notifications


def get_auth_config() -> AuthConfig:
    """Get authentication configuration."""
    return get_settings().auth


def get_alpha_lab_config() -> AlphaLabConfig:
    """Get alpha lab configuration."""
    return get_settings().alpha_lab


def get_bitget_config() -> BitgetConfig:
    """Get Bitget configuration."""
    return get_settings().bitget


def get_crypto_market_config() -> CryptoMarketConfig:
    """Get unified crypto market data configuration."""
    return get_settings().crypto_market


def get_brooks_live_config() -> BrooksLiveConfig:
    """Get Phase 4.6 BrooksLive configuration."""
    return get_settings().brooks
