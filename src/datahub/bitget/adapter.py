from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import requests


@dataclass
class CandleRecord:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume_base: float
    volume_quote: float
    exchange_symbol: str | None = None


@dataclass
class FundingRateRecord:
    timestamp_ms: int
    symbol: str
    funding_rate: float


class BitgetDataAdapter:
    def __init__(
        self,
        base_url: str = "https://api.bitget.com",
        product_type: str = "USDT-FUTURES",
        timeout_seconds: int = 10,
    ):
        self.base_url = base_url.rstrip("/")
        self.product_type = product_type
        self.timeout_seconds = timeout_seconds
        self.provider_name = "bitget"
        self.market_type = "perpetual"

    def fetch_candles(
        self,
        symbol: str,
        interval: str = "1m",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 200,
        granularity: str | None = None,
    ) -> list[CandleRecord]:
        interval = granularity or interval
        payload = self._get(
            "/api/v2/mix/market/candles",
            {
                "symbol": symbol,
                "productType": self.product_type.lower(),
                "granularity": interval,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": limit,
            },
        )
        rows = payload.get("data", [])
        return [self._parse_candle(row) for row in rows]

    def fetch_history_candles(
        self,
        symbol: str,
        interval: str = "1m",
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 200,
        granularity: str | None = None,
    ) -> list[CandleRecord]:
        interval = granularity or interval
        payload = self._get(
            "/api/v2/mix/market/history-candles",
            {
                "symbol": symbol,
                "productType": self.product_type.lower(),
                "granularity": interval,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": min(limit, 200),
            },
        )
        rows = payload.get("data", [])
        return [self._parse_candle(row) for row in rows]

    def fetch_funding_rate_history(
        self,
        symbol: str,
        page_no: int = 1,
        page_size: int = 20,
    ) -> list[FundingRateRecord]:
        payload = self._get(
            "/api/v2/mix/market/history-fund-rate",
            {
                "symbol": symbol,
                "productType": self.product_type.lower(),
                "pageNo": page_no,
                "pageSize": page_size,
            },
        )
        rows = payload.get("data", [])
        return [
            FundingRateRecord(
                timestamp_ms=int(item["fundingTime"]),
                symbol=item["symbol"],
                funding_rate=float(item["fundingRate"]),
            )
            for item in rows
        ]

    def normalize_candles(self, records: list[CandleRecord], symbol: str) -> list[dict[str, Any]]:
        return [
            {
                "provider": self.provider_name,
                "market_type": self.market_type,
                "symbol": symbol,
                "exchange_symbol": record.exchange_symbol or symbol,
                "timestamp_ms": record.timestamp_ms,
                "open": record.open,
                "high": record.high,
                "low": record.low,
                "close": record.close,
                "volume_base": record.volume_base,
                "volume_quote": record.volume_quote,
            }
            for record in records
        ]

    def to_clickhouse_rows(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return records

    def _parse_candle(self, row: list[str]) -> CandleRecord:
        return CandleRecord(
            timestamp_ms=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume_base=float(row[5]) if len(row) > 5 else 0.0,
            volume_quote=float(row[6]) if len(row) > 6 else 0.0,
        )

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        filtered = {k: v for k, v in params.items() if v is not None}
        response = requests.get(
            f"{self.base_url}{path}",
            params=filtered,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") not in {None, "00000"}:
            raise ValueError(f"Bitget API error: {payload}")
        return payload
