from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class CryptoSyncStateStore:
    def __init__(self, path: str = ""):
        from src.config.paths import CRYPTO_SYNC_STATE_PATH

        self.path = Path(path) if path else CRYPTO_SYNC_STATE_PATH

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "sync_points": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "sync_points": {}}
        if not isinstance(data, dict):
            return {"version": 1, "sync_points": {}}
        data.setdefault("version", 1)
        data.setdefault("sync_points", {})
        return data

    def get_sync_point(
        self,
        provider: str,
        market_type: str,
        symbol: str,
        interval: str,
    ) -> dict[str, Any] | None:
        return self.load()["sync_points"].get(self._key(provider, market_type, symbol, interval))

    def mark_started(
        self,
        provider: str,
        market_type: str,
        symbol: str,
        interval: str,
        start_time: datetime,
    ) -> None:
        data = self.load()
        key = self._key(provider, market_type, symbol, interval)
        point = data["sync_points"].get(key, {})
        point.update(
            {
                "provider": provider,
                "market_type": market_type,
                "symbol": symbol,
                "interval": interval,
                "status": "running",
                "run_started_at": datetime.now(UTC).isoformat(),
                "requested_start_time": start_time.astimezone(UTC).isoformat(),
            }
        )
        data["sync_points"][key] = point
        self._save(data)

    def mark_progress(
        self,
        provider: str,
        market_type: str,
        symbol: str,
        interval: str,
        last_open_time: datetime,
        fetched: int,
        inserted: int,
    ) -> None:
        data = self.load()
        key = self._key(provider, market_type, symbol, interval)
        point = data["sync_points"].get(key, {})
        point.update(
            {
                "provider": provider,
                "market_type": market_type,
                "symbol": symbol,
                "interval": interval,
                "status": "running",
                "last_open_time": last_open_time.astimezone(UTC).isoformat(),
                "last_heartbeat_at": datetime.now(UTC).isoformat(),
                "fetched": int(fetched),
                "inserted": int(inserted),
            }
        )
        data["sync_points"][key] = point
        self._save(data)

    def mark_completed(
        self,
        provider: str,
        market_type: str,
        symbol: str,
        interval: str,
        last_open_time: datetime | None,
        fetched: int,
        inserted: int,
    ) -> None:
        data = self.load()
        key = self._key(provider, market_type, symbol, interval)
        point = data["sync_points"].get(key, {})
        point.update(
            {
                "provider": provider,
                "market_type": market_type,
                "symbol": symbol,
                "interval": interval,
                "status": "success",
                "run_completed_at": datetime.now(UTC).isoformat(),
                "last_open_time": last_open_time.astimezone(UTC).isoformat()
                if last_open_time
                else point.get("last_open_time"),
                "fetched": int(fetched),
                "inserted": int(inserted),
                "error": None,
            }
        )
        data["sync_points"][key] = point
        self._save(data)

    def mark_failed(
        self,
        provider: str,
        market_type: str,
        symbol: str,
        interval: str,
        error: str,
    ) -> None:
        data = self.load()
        key = self._key(provider, market_type, symbol, interval)
        point = data["sync_points"].get(key, {})
        point.update(
            {
                "provider": provider,
                "market_type": market_type,
                "symbol": symbol,
                "interval": interval,
                "status": "error",
                "run_completed_at": datetime.now(UTC).isoformat(),
                "error": error,
            }
        )
        data["sync_points"][key] = point
        self._save(data)

    def _save(self, data: dict[str, Any]) -> None:
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _key(self, provider: str, market_type: str, symbol: str, interval: str) -> str:
        return f"{provider}:{market_type}:{symbol}:{interval}"
