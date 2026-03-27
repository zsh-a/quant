from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class BinanceKlineRecord:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume_base: float
    close_time_ms: int
    volume_quote: float
    trade_count: int


class BinanceSpotDataAdapter:
    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        timeout_seconds: int = 10,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.provider_name = "binance"
        self.market_type = "spot"

    def fetch_candles(
        self,
        symbol: str,
        interval: str = "1m",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[BinanceKlineRecord]:
        payload = self._get(
            "/api/v3/klines",
            {
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": limit,
            },
        )
        return [self._parse_kline(row) for row in payload]

    def normalize_candles(self, records: list[BinanceKlineRecord], symbol: str) -> list[dict[str, Any]]:
        return [
            {
                "provider": self.provider_name,
                "market_type": self.market_type,
                "symbol": symbol.upper(),
                "exchange_symbol": symbol.upper(),
                "interval": "1m",
                "open_time_ms": record.open_time_ms,
                "close_time_ms": record.close_time_ms,
                "open": record.open,
                "high": record.high,
                "low": record.low,
                "close": record.close,
                "volume_base": record.volume_base,
                "volume_quote": record.volume_quote,
                "trade_count": record.trade_count,
            }
            for record in records
        ]

    def _parse_kline(self, row: list[Any]) -> BinanceKlineRecord:
        return BinanceKlineRecord(
            open_time_ms=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume_base=float(row[5]),
            close_time_ms=int(row[6]),
            volume_quote=float(row[7]),
            trade_count=int(row[8]),
        )

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        filtered = {k: v for k, v in params.items() if v is not None}
        response = requests.get(
            f"{self.base_url}{path}",
            params=filtered,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
