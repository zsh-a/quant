"""Historical crypto bar stream for replay sessions.

Reads OHLCV from ``crypto_data.minute_bars`` (the table populated by the
unified crypto ingestion pipeline) and yields :class:`Bar` objects in
chronological order. Used by ``brooks_replay_task`` to feed the same
:class:`BrooksCore` pipeline that drives live.

The ingestion pipeline only stores ``1m`` granularity, so any requested
interval coarser than 1m is resampled in-process via pandas (OHLCV
aggregation). The stream is a synchronous in-memory iterator: for typical
replay windows (a few weeks of 5m bars ≈ 6k rows after resampling) the
row volume is small enough that streaming is unnecessary. ``next_bar``
returns ``None`` when exhausted, mirroring the existing ``DataStream``
contract.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from src.core.base import Bar, DataStream
from src.market_data.crypto_pipeline import CryptoMinuteSyncService

# We always query 1m from ClickHouse and aggregate up — only 1m granularity
# is persisted by the unified crypto ingestion pipeline.
_STORAGE_INTERVAL = "1m"


def _normalize_symbol(symbol: str) -> str:
    """Drop the slash and uppercase — the storage layer keeps ``BTCUSDT``."""
    return symbol.replace("/", "").upper()


def _interval_minutes(interval: str) -> int:
    """Coerce ``"5m"`` / ``"15m"`` / ``"1h"`` / ``"4h"`` / ``"1d"`` to minutes.

    Raises :class:`ValueError` for unsupported units.
    """
    s = interval.strip().lower()
    m = re.fullmatch(r"(\d+)([mhd])", s)
    if not m:
        raise ValueError(f"unsupported interval {interval!r}")
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        return n
    if unit == "h":
        return n * 60
    if unit == "d":
        return n * 60 * 24
    raise ValueError(f"unsupported interval unit in {interval!r}")


class HistoricalCryptoStream(DataStream):
    """Replay :class:`Bar`s for ``(symbol, interval)`` over a closed time window.

    Parameters
    ----------
    symbol
        Trading symbol in either slash form (``"BTC/USDT"``) or compact
        form (``"BTCUSDT"``); both are accepted, normalised internally,
        and the original form is preserved on the emitted :class:`Bar`.
    interval
        Bar interval, e.g. ``"1m"``, ``"5m"``, ``"1h"``. Forwarded to the
        ingestion pipeline as-is.
    start, end
        Inclusive bar window (UTC).
    provider
        Source provider — defaults to ``bitget`` to match the unified
        ingestion default. The minute-bars table partitions on provider
        so this only narrows the query.
    """

    is_live = False

    def __init__(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        *,
        provider: str = "bitget",
        market_service: Optional[CryptoMinuteSyncService] = None,
    ):
        self.symbol = symbol
        self.interval = interval
        self.start = self._ensure_utc(start)
        self.end = self._ensure_utc(end)
        self.provider = provider
        self._service = market_service or CryptoMinuteSyncService()
        self._bars: List[Bar] = []
        self._idx = 0
        self._loaded = False

    # ---- public --------------------------------------------------------

    def total_bars(self) -> int:
        self._ensure_loaded()
        return len(self._bars)

    def next_bar(self) -> Optional[Dict[str, Bar]]:
        self._ensure_loaded()
        if self._idx >= len(self._bars):
            return None
        bar = self._bars[self._idx]
        self._idx += 1
        return {self.symbol: bar}

    def reset(self) -> None:
        self._idx = 0

    def remaining(self) -> int:
        self._ensure_loaded()
        return max(0, len(self._bars) - self._idx)

    # ---- loading -------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        # The pipeline only persists 1m granularity — query 1m from
        # ClickHouse and resample up to the requested interval in-process.
        target_minutes = _interval_minutes(self.interval)
        rows = self._service.query_bars(
            provider=self.provider,
            symbol=_normalize_symbol(self.symbol),
            start_time=self.start,
            end_time=self.end,
            interval=_STORAGE_INTERVAL,
        )
        if target_minutes == 1:
            self._bars = [self._row_to_bar(row, ts=self._ensure_utc(row["open_time"])) for row in rows]
        else:
            self._bars = self._resample_rows(rows, target_minutes)
        self._loaded = True
        logger.info(
            "HistoricalCryptoStream loaded {} bars for {} {} {}..{} (queried {} 1m rows, resampled to {}m)",
            len(self._bars),
            self.symbol,
            self.interval,
            self.start.isoformat(),
            self.end.isoformat(),
            len(rows),
            target_minutes,
        )

    def _resample_rows(self, rows: List[Dict[str, Any]], target_minutes: int) -> List[Bar]:
        """Aggregate 1m rows into ``target_minutes`` OHLCV bars."""
        if not rows:
            return []
        df = pd.DataFrame(rows)
        # Normalise the timestamp column to a tz-aware DatetimeIndex.
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time").sort_index()
        rule = f"{target_minutes}min"
        agg = (
            df.resample(rule, label="left", closed="left")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume_base": "sum",
                    "volume_quote": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
        )
        bars: List[Bar] = []
        for ts, row in agg.iterrows():
            bars.append(
                Bar(
                    symbol=self.symbol,
                    timestamp=ts.to_pydatetime(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume_base"]),
                    amount=float(row["volume_quote"]),
                )
            )
        return bars

    def _row_to_bar(self, row: Dict[str, Any], *, ts: datetime) -> Bar:
        return Bar(
            symbol=self.symbol,
            timestamp=ts,
            open=float(row.get("open") or 0.0),
            high=float(row.get("high") or 0.0),
            low=float(row.get("low") or 0.0),
            close=float(row.get("close") or 0.0),
            volume=float(row.get("volume_base") or 0.0),
            amount=float(row.get("volume_quote") or 0.0),
        )

    @staticmethod
    def _ensure_utc(value: Any) -> datetime:
        if value is None:
            raise ValueError("HistoricalCryptoStream timestamp is None")
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        # Allow ISO-format strings as a convenience for callers from REST.
        try:
            ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"unsupported timestamp value {value!r}") from exc
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)


__all__ = ["HistoricalCryptoStream"]
