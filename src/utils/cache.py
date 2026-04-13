"""
Redis-based caching layer for backtest data.
Provides caching for session results, equity history, and trades.
"""

import orjson
import hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from functools import wraps

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class CacheConfig:
    """Cache configuration"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 3600,
        prefix: str = "quant:",
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.prefix = prefix


class RedisCache:
    """Redis cache implementation"""

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._client: Optional[redis.Redis] = None
        self._connected = False

    @property
    def client(self) -> Optional[redis.Redis]:
        if not REDIS_AVAILABLE:
            return None

        if self._client is None:
            try:
                self._client = redis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    password=self.config.password,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                )
                self._client.ping()
                self._connected = True
                logger.info(f"Redis connected: {self.config.host}:{self.config.port}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self._client = None
                self._connected = False

        return self._client

    @property
    def is_connected(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            self._connected = False
            return False

    def _make_key(self, key: str) -> str:
        return f"{self.config.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.client:
            return None

        try:
            full_key = self._make_key(key)
            value = self.client.get(full_key)
            if value:
                return orjson.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self.client:
            return False

        try:
            full_key = self._make_key(key)
            ttl = ttl or self.config.default_ttl
            serialized = orjson.dumps(value, default=str).decode()
            self.client.setex(full_key, ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.client:
            return False

        try:
            full_key = self._make_key(key)
            self.client.delete(full_key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self.client:
            return False

        try:
            full_key = self._make_key(key)
            return bool(self.client.exists(full_key))
        except Exception:
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        if not self.client:
            return 0

        try:
            full_pattern = self._make_key(pattern)
            keys = self.client.keys(full_pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache clear pattern error: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.client:
            return {"connected": False}

        try:
            info = self.client.info("memory")
            return {
                "connected": True,
                "used_memory": info.get("used_memory_human", "N/A"),
                "keys": self.client.dbsize(),
                "host": self.config.host,
                "port": self.config.port,
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}


class BacktestCache:
    """Specialized cache for backtest results"""

    def __init__(self, redis_cache: Optional[RedisCache] = None):
        self.cache = redis_cache or RedisCache()

    def _make_session_key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def _make_equity_key(self, session_id: str) -> str:
        return f"equity:{session_id}"

    def _make_trades_key(self, session_id: str) -> str:
        return f"trades:{session_id}"

    def _make_backtest_result_key(
        self,
        strategy: str,
        symbol: str,
        start_date: str,
        end_date: str,
        params: Dict[str, Any],
    ) -> str:
        """Generate unique key for backtest configuration"""
        params_str = orjson.dumps(params, option=orjson.OPT_SORT_KEYS, default=str).decode()
        key_data = f"{strategy}:{symbol}:{start_date}:{end_date}:{params_str}"
        hash_key = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"backtest:{strategy}:{symbol}:{hash_key}"

    def cache_session_result(
        self, session_id: str, result: Dict[str, Any], ttl: int = 7200
    ) -> bool:
        """Cache session result"""
        key = self._make_session_key(session_id)
        return self.cache.set(key, result, ttl)

    def get_session_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session result"""
        key = self._make_session_key(session_id)
        return self.cache.get(key)

    def cache_equity_history(
        self, session_id: str, equity_history: List[Dict[str, Any]], ttl: int = 7200
    ) -> bool:
        """Cache equity history"""
        key = self._make_equity_key(session_id)
        return self.cache.set(key, equity_history, ttl)

    def get_equity_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached equity history"""
        key = self._make_equity_key(session_id)
        return self.cache.get(key)

    def cache_trades(
        self, session_id: str, trades: List[Dict[str, Any]], ttl: int = 7200
    ) -> bool:
        """Cache trades"""
        key = self._make_trades_key(session_id)
        return self.cache.set(key, trades, ttl)

    def get_trades(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached trades"""
        key = self._make_trades_key(session_id)
        return self.cache.get(key)

    def cache_backtest_result(
        self,
        strategy: str,
        symbol: str,
        start_date: str,
        end_date: str,
        params: Dict[str, Any],
        result: Dict[str, Any],
        ttl: int = 86400,
    ) -> bool:
        """Cache backtest result by configuration (for deduplication)"""
        key = self._make_backtest_result_key(
            strategy, symbol, start_date, end_date, params
        )
        return self.cache.set(key, result, ttl)

    def get_backtest_result(
        self,
        strategy: str,
        symbol: str,
        start_date: str,
        end_date: str,
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Get cached backtest result by configuration"""
        key = self._make_backtest_result_key(
            strategy, symbol, start_date, end_date, params
        )
        return self.cache.get(key)

    def invalidate_session(self, session_id: str) -> int:
        """Invalidate all cache for a session"""
        count = 0
        for key_fn in [
            self._make_session_key,
            self._make_equity_key,
            self._make_trades_key,
        ]:
            if self.cache.delete(key_fn(session_id)):
                count += 1
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()


def cached(ttl: int = 3600, key_prefix: str = "func"):
    """Decorator for caching function results"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            if not cache.is_connected:
                return func(*args, **kwargs)

            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(a) for a in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_result

            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            logger.debug(f"Cache set: {cache_key}")
            return result

        return wrapper

    return decorator


_cache_instance: Optional[RedisCache] = None
_backtest_cache_instance: Optional[BacktestCache] = None


def get_cache() -> RedisCache:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance


def get_backtest_cache() -> BacktestCache:
    """Get global backtest cache instance"""
    global _backtest_cache_instance
    if _backtest_cache_instance is None:
        _backtest_cache_instance = BacktestCache(get_cache())
    return _backtest_cache_instance
