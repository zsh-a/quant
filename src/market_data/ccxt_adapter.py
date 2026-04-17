from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from loguru import logger

try:
    import ccxt
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    ccxt = None

from src.market_data.crypto_store import UnifiedMinuteBar

DEFAULT_CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]


@dataclass(frozen=True)
class CcxtProviderSpec:
    provider_name: str
    exchange_id: str
    market_type: str
    default_type: str
    batch_limit: int
    timeout_seconds: int = 10


PROVIDER_SPECS: dict[str, CcxtProviderSpec] = {
    "binance": CcxtProviderSpec(
        provider_name="binance",
        exchange_id="binance",
        market_type="spot",
        default_type="spot",
        batch_limit=1000,
    ),
    "bitget": CcxtProviderSpec(
        provider_name="bitget",
        exchange_id="bitget",
        market_type="perpetual",
        default_type="swap",
        batch_limit=200,
    ),
}


class CcxtCryptoDataAdapter:
    def __init__(self, spec: CcxtProviderSpec):
        if ccxt is None:
            raise RuntimeError("ccxt is required for crypto market sync. Install the 'ccxt' package first.")

        self.spec = spec
        self.provider_name = spec.provider_name
        self.market_type = spec.market_type
        self.batch_limit = spec.batch_limit
        exchange_cls = getattr(ccxt, spec.exchange_id)
        self.client = exchange_cls(
            {
                "enableRateLimit": True,
                "timeout": int(spec.timeout_seconds * 1000),
                "options": {"defaultType": spec.default_type},
            }
        )
        self._alias_to_market: dict[str, dict[str, Any]] | None = None

    @property
    def is_futures(self) -> bool:
        return self.market_type == "perpetual"

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "exchange_id": self.spec.exchange_id,
            "market_type": self.market_type,
            "intervals": ["1m"],
            "batch_limit": self.batch_limit,
        }

    def build_instruments(self, symbols: list[str]) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        instruments: list[dict[str, Any]] = []
        for raw_symbol in symbols:
            market = self.resolve_market(raw_symbol)
            instruments.append(
                {
                    "provider": self.provider_name,
                    "market_type": self.market_type,
                    "symbol": self.to_storage_symbol(market),
                    "exchange_symbol": str(market.get("id") or market.get("symbol") or raw_symbol).upper(),
                    "base_asset": str(market.get("base") or "").upper(),
                    "quote_asset": str(market.get("quote") or "").upper(),
                    "is_active": 1 if market.get("active", True) else 0,
                    "updated_at": now,
                }
            )
        return instruments

    def normalize_symbol(self, symbol: str) -> str:
        market = self.resolve_market(symbol)
        return self.to_storage_symbol(market)

    def fetch_bars(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: int | None = None,
    ) -> list[UnifiedMinuteBar]:
        market = self.resolve_market(symbol)
        storage_symbol = self.to_storage_symbol(market)
        since_ms = int(start_time.astimezone(UTC).timestamp() * 1000)
        end_ms = int(end_time.astimezone(UTC).timestamp() * 1000)
        step_ms = self.interval_to_milliseconds(interval)
        request_limit = max(1, min(int(limit or self.batch_limit), self.batch_limit))
        payload = self.client.fetch_ohlcv(
            market["symbol"],
            timeframe=interval,
            since=since_ms,
            limit=request_limit,
        )

        bars_by_open_time: dict[datetime, UnifiedMinuteBar] = {}
        for row in payload:
            open_time_ms = int(row[0])
            if open_time_ms < since_ms or open_time_ms > end_ms:
                continue

            open_time = datetime.fromtimestamp(open_time_ms / 1000, tz=UTC)
            close = float(row[4])
            volume_base = float(row[5] or 0.0)
            bars_by_open_time[open_time] = UnifiedMinuteBar(
                provider=self.provider_name,
                market_type=self.market_type,
                symbol=storage_symbol,
                exchange_symbol=str(market.get("id") or storage_symbol).upper(),
                interval=interval,
                open_time=open_time,
                close_time=datetime.fromtimestamp((open_time_ms + step_ms - 1) / 1000, tz=UTC),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=close,
                volume_base=volume_base,
                volume_quote=volume_base * close,
                trade_count=0,
            )

        return [bars_by_open_time[key] for key in sorted(bars_by_open_time)]

    # ------------------------------------------------------------------
    # Funding rate (perpetual/swap only)
    # ------------------------------------------------------------------

    def fetch_funding_rate_history(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 500,
    ) -> dict[datetime, float]:
        """Fetch historical funding rates and return {settlement_time: rate}.

        Only meaningful for perpetual/swap markets.  Returns an empty dict
        for spot markets or when the exchange does not support the endpoint.
        """
        if not self.is_futures:
            return {}

        market = self.resolve_market(symbol)
        since_ms = int(start_time.astimezone(UTC).timestamp() * 1000)
        end_ms = int(end_time.astimezone(UTC).timestamp() * 1000)

        result: dict[datetime, float] = {}
        try:
            if not hasattr(self.client, "fetch_funding_rate_history"):
                return result
            records = self.client.fetch_funding_rate_history(
                market["symbol"],
                since=since_ms,
                limit=limit,
            )
            for rec in records:
                ts_ms = int(rec.get("timestamp") or rec.get("datetime", 0))
                if ts_ms and since_ms <= ts_ms <= end_ms:
                    rate = float(rec.get("fundingRate", 0) or 0)
                    dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
                    result[dt] = rate
        except Exception as exc:
            logger.warning(
                "ccxt.funding_rate_history failed provider={} symbol={}: {}",
                self.provider_name,
                symbol,
                exc,
            )
        return result

    def fetch_funding_rate(self, symbol: str) -> float:
        """Fetch current (latest) funding rate for a perpetual symbol."""
        if not self.is_futures:
            return 0.0
        market = self.resolve_market(symbol)
        try:
            data = self.client.fetch_funding_rate(market["symbol"])
            return float(data.get("fundingRate", 0) or 0)
        except Exception as exc:
            logger.warning(
                "ccxt.fetch_funding_rate failed provider={} symbol={}: {}",
                self.provider_name,
                symbol,
                exc,
            )
            return 0.0

    # ------------------------------------------------------------------
    # Open interest (perpetual/swap only)
    # ------------------------------------------------------------------

    def fetch_open_interest(self, symbol: str) -> float:
        """Fetch latest open interest value (in base asset)."""
        if not self.is_futures:
            return 0.0
        market = self.resolve_market(symbol)
        try:
            data = self.client.fetch_open_interest(market["symbol"])
            return float(data.get("openInterestAmount", 0) or data.get("openInterest", 0) or 0)
        except Exception as exc:
            logger.warning(
                "ccxt.fetch_open_interest failed provider={} symbol={}: {}",
                self.provider_name,
                symbol,
                exc,
            )
            return 0.0

    def fetch_open_interest_history(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        timeframe: str = "5m",
        limit: int = 500,
    ) -> dict[datetime, float]:
        """Fetch historical open interest, returns {timestamp: oi_value}."""
        if not self.is_futures:
            return {}

        market = self.resolve_market(symbol)
        since_ms = int(start_time.astimezone(UTC).timestamp() * 1000)
        end_ms = int(end_time.astimezone(UTC).timestamp() * 1000)

        result: dict[datetime, float] = {}
        try:
            if not hasattr(self.client, "fetch_open_interest_history"):
                return result
            records = self.client.fetch_open_interest_history(
                market["symbol"],
                timeframe=timeframe,
                since=since_ms,
                limit=limit,
            )
            for rec in records:
                ts_ms = int(rec.get("timestamp", 0) or 0)
                if ts_ms and since_ms <= ts_ms <= end_ms:
                    oi = float(rec.get("openInterestAmount", 0) or rec.get("openInterest", 0) or 0)
                    dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
                    result[dt] = oi
        except Exception as exc:
            logger.warning(
                "ccxt.open_interest_history failed provider={} symbol={}: {}",
                self.provider_name,
                symbol,
                exc,
            )
        return result

    # ------------------------------------------------------------------
    # Enrichment — attach funding & OI to already-fetched bars
    # ------------------------------------------------------------------

    def enrich_bars(
        self,
        bars: list[UnifiedMinuteBar],
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[UnifiedMinuteBar]:
        """Attach funding_rate and open_interest to bars in-place.

        Funding rates are typically published every 8h, so the rate is
        forward-filled across minute bars until the next settlement.
        Open interest snapshots (5m granularity) are similarly forward-filled.
        """
        if not bars or not self.is_futures:
            return bars

        fr_map = self.fetch_funding_rate_history(symbol, start_time, end_time)
        oi_map = self.fetch_open_interest_history(symbol, start_time, end_time)

        sorted_fr = sorted(fr_map.items())
        sorted_oi = sorted(oi_map.items())
        fr_idx = 0
        oi_idx = 0
        current_fr = 0.0
        current_oi = 0.0

        for bar in bars:
            while fr_idx < len(sorted_fr) and sorted_fr[fr_idx][0] <= bar.open_time:
                current_fr = sorted_fr[fr_idx][1]
                fr_idx += 1
            while oi_idx < len(sorted_oi) and sorted_oi[oi_idx][0] <= bar.open_time:
                current_oi = sorted_oi[oi_idx][1]
                oi_idx += 1
            bar.funding_rate = current_fr
            bar.open_interest = current_oi

        return bars

    # ------------------------------------------------------------------
    # Market resolution helpers
    # ------------------------------------------------------------------

    def resolve_market(self, symbol: str) -> dict[str, Any]:
        alias_map = self._get_alias_to_market()
        normalized = self._normalize_alias(symbol)
        market = alias_map.get(normalized) or alias_map.get(str(symbol).upper())
        if market is None:
            raise ValueError(f"Unsupported symbol for {self.provider_name}: {symbol}")
        return market

    def to_storage_symbol(self, market: dict[str, Any]) -> str:
        base_asset = str(market.get("base") or "").upper()
        quote_asset = str(market.get("quote") or "").upper()
        if base_asset and quote_asset:
            return f"{base_asset}{quote_asset}"
        raw_value = str(market.get("id") or market.get("symbol") or "").upper()
        return raw_value.replace("/", "").replace(":", "").replace("-", "")

    def interval_to_milliseconds(self, interval: str) -> int:
        return int(self.client.parse_timeframe(interval) * 1000)

    def _get_alias_to_market(self) -> dict[str, dict[str, Any]]:
        if self._alias_to_market is not None:
            return self._alias_to_market

        if self.client.markets and self.client.markets_by_id:
            markets = self.client.markets
        else:
            markets = self.client.fetch_markets()
            markets = self.client.set_markets(markets)
        alias_to_market: dict[str, dict[str, Any]] = {}
        for market in markets.values():
            if not self._is_target_market(market):
                continue
            for alias in self._market_aliases(market):
                alias_to_market.setdefault(alias, market)
        self._alias_to_market = alias_to_market
        return alias_to_market

    def _is_target_market(self, market: dict[str, Any]) -> bool:
        if self.market_type == "spot":
            return bool(market.get("spot"))
        if self.market_type == "perpetual":
            return bool(market.get("swap") or market.get("future"))
        return True

    def _market_aliases(self, market: dict[str, Any]) -> set[str]:
        aliases = {
            str(market.get("symbol") or "").upper(),
            str(market.get("id") or "").upper(),
        }
        base_asset = str(market.get("base") or "").upper()
        quote_asset = str(market.get("quote") or "").upper()
        settle_asset = str(market.get("settle") or "").upper()
        if base_asset and quote_asset:
            aliases.add(f"{base_asset}{quote_asset}")
            aliases.add(f"{base_asset}/{quote_asset}")
            if settle_asset:
                aliases.add(f"{base_asset}/{quote_asset}:{settle_asset}")
        return {self._normalize_alias(alias) for alias in aliases if alias}

    def _normalize_alias(self, value: str) -> str:
        return str(value).upper().replace("/", "").replace(":", "").replace("-", "")


def build_default_providers() -> dict[str, CcxtCryptoDataAdapter]:
    return {provider_name: CcxtCryptoDataAdapter(spec) for provider_name, spec in PROVIDER_SPECS.items()}
