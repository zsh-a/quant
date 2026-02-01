"""
Configuration loader for the quantitative trading platform.
Loads settings from YAML config file with environment variable overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger

class Config:
    """Singleton configuration manager"""
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from YAML file"""
        # Find config file
        config_paths = [
            Path(__file__).parent.parent / 'config' / 'system_config.yaml',
            Path('config/system_config.yaml'),
            Path('system_config.yaml')
        ]
        
        config_file = None
        for path in config_paths:
            if path.exists():
                config_file = path
                break
        
        if config_file is None:
            logger.warning("No config file found, using defaults")
            self._config = self._get_defaults()
            return
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"Configuration loaded from {config_file}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            self._config = self._get_defaults()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _apply_env_overrides(self):
        """Override config values from environment variables"""
        # Example: QUANT_DATABASE_HOST=localhost
        prefix = "QUANT_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert QUANT_DATABASE_HOST to database.host
                config_path = key[len(prefix):].lower().split('_')
                self._set_nested(config_path, value)
    
    def _set_nested(self, path: list, value: str):
        """Set nested config value from path list"""
        current = self._config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Try to convert value to appropriate type
        try:
            # Try int
            current[path[-1]] = int(value)
        except ValueError:
            try:
                # Try float
                current[path[-1]] = float(value)
            except ValueError:
                # Try bool
                if value.lower() in ('true', 'yes', '1'):
                    current[path[-1]] = True
                elif value.lower() in ('false', 'no', '0'):
                    current[path[-1]] = False
                else:
                    # Keep as string
                    current[path[-1]] = value
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'database': {
                'host': 'localhost',
                'port': 8123,
                'username': 'default',
                'password': '',
                'cache_enabled': True
            },
            'data_stream': {
                'chunk_size_months': None,
                'enable_memory_monitoring': True,
                'realtime': {
                    'interval_seconds': 60,
                    'data_source': 'akshare',
                    'enable_websocket': False
                }
            },
            'broker': {
                'backtest': {
                    'initial_cash': 1000000.0,
                    'commission': 0.0001,
                    'slippage': 0.0
                },
                'live': {
                    'server_url': 'http://localhost:11122',
                    'timeout_seconds': 30,
                    'retry_attempts': 3
                }
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'cors_origins': ['*'],
                'session': {
                    'persist_to_db': True,
                    'clear_memory_on_persist': True,
                    'update_db_interval_bars': 10
                },
                'enable_incremental_updates': True,
                'max_response_size_mb': 10
            },
            'logging': {
                'level': 'INFO',
                'format': '{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}',
                'rotation': '100 MB',
                'retention': '30 days',
                'file': {
                    'enabled': True,
                    'path': 'logs/quant_{time:YYYY-MM-DD}.log'
                },
                'console': {
                    'enabled': True,
                    'colorize': True
                }
            },
            'monitoring': {
                'enabled': True,
                'metrics': ['memory_usage', 'api_response_time', 'backtest_throughput'],
                'alerts': {
                    'memory_usage_mb': 8000,
                    'api_response_time_ms': 1000
                }
            },
            'features': {
                'enable_websocket': False,
                'enable_parameter_optimization': False,
                'enable_multi_strategy': False,
                'enable_risk_manager': False
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        Example: config.get('database.host')
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self._config.get(section, {})
    
    def reload(self):
        """Reload configuration from file"""
        self._load_config()
        logger.info("Configuration reloaded")
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self._config.copy()


# Global config instance
config = Config()


# Convenience functions
def get(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return config.get(key, default)


def get_section(section: str) -> Dict[str, Any]:
    """Get configuration section"""
    return config.get_section(section)


def reload():
    """Reload configuration"""
    config.reload()


if __name__ == '__main__':
    # Test configuration loading
    print("Configuration loaded:")
    print(f"Database host: {get('database.host')}")
    print(f"API port: {get('api.port')}")
    print(f"Chunk size: {get('data_stream.chunk_size_months')}")
    print(f"Initial cash: {get('broker.backtest.initial_cash')}")
    print(f"\nFull config:")
    import json
    print(json.dumps(config.all, indent=2))
