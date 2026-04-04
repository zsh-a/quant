from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.market_data.clickhouse import create_clickhouse_client


@dataclass
class AlphaDataset:
    provider: str
    interval: str
    symbols: list[str]
    timestamps: list[str]
    fields: dict[str, np.ndarray]
    liquidity_mask: np.ndarray
    session_mask: np.ndarray

    def shape(self) -> tuple[int, int]:
        close = self.fields["close"]
        return int(close.shape[0]), int(close.shape[1])

    def slice_by_index(self, start_idx: int, end_idx: int) -> "AlphaDataset":
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(self.timestamps))
        sliced_fields = {name: values[start_idx:end_idx] for name, values in self.fields.items()}
        return AlphaDataset(
            provider=self.provider,
            interval=self.interval,
            symbols=list(self.symbols),
            timestamps=self.timestamps[start_idx:end_idx],
            fields=sliced_fields,
            liquidity_mask=self.liquidity_mask[start_idx:end_idx],
            session_mask=self.session_mask[start_idx:end_idx],
        )

    def take_indices(self, indices: list[int] | np.ndarray | tuple[int, ...]) -> "AlphaDataset":
        selected = np.asarray(indices, dtype=int)
        sliced_fields = {name: values[selected] for name, values in self.fields.items()}
        return AlphaDataset(
            provider=self.provider,
            interval=self.interval,
            symbols=list(self.symbols),
            timestamps=[self.timestamps[int(idx)] for idx in selected.tolist()],
            fields=sliced_fields,
            liquidity_mask=self.liquidity_mask[selected],
            session_mask=self.session_mask[selected],
        )


_FUTURES_TABLE = "crypto_data.futures_5m"
_BASE_INTERVAL_MINUTES = 5

_SELECT_COLUMNS = [
    "symbol", "open_time", "close_time",
    "open", "high", "low", "close",
    "volume", "quote_volume", "trade_count",
    "taker_buy_volume", "taker_buy_quote_volume",
    "mark_open", "mark_high", "mark_low", "mark_close",
    "premium_open", "premium_high", "premium_low", "premium_close",
    "open_interest", "open_interest_value",
    "top_trader_long_short_ratio", "top_trader_long_short_position_ratio",
    "long_short_ratio", "taker_long_short_vol_ratio",
    "funding_rate",
]

# Fields that should use "last" aggregation when resampling (snapshot values)
_RESAMPLE_LAST = {
    "symbol", "close_time", "close",
    "mark_close", "premium_close",
    "open_interest", "open_interest_value",
    "top_trader_long_short_ratio", "top_trader_long_short_position_ratio",
    "long_short_ratio", "taker_long_short_vol_ratio",
    "funding_rate",
}

# Fields that should be zero-filled when NaN
_ZERO_FILL_FIELDS = [
    "trade_count", "taker_buy_volume", "taker_buy_quote_volume",
    "mark_open", "mark_high", "mark_low", "mark_close",
    "premium_open", "premium_high", "premium_low", "premium_close",
    "open_interest", "open_interest_value",
    "top_trader_long_short_ratio", "top_trader_long_short_position_ratio",
    "long_short_ratio", "taker_long_short_vol_ratio",
    "funding_rate",
]


class CryptoMinuteDatasetLoader:
    """Load alpha datasets from Binance Vision futures_5m table in ClickHouse."""

    def __init__(self, client=None):
        self._client = client

    def _get_client(self):
        if self._client is None:
            self._client = create_clickhouse_client()
        return self._client

    def load(
        self,
        provider: str,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = "5m",
        min_quote_volume: float = 0.0,
        blocked_utc_hours: list[int] | set[int] | tuple[int, ...] | None = None,
    ) -> AlphaDataset:
        client = self._get_client()
        upper_symbols = [s.upper() for s in symbols]

        # Query from futures_5m
        symbols_clause = ",".join(f"'{s}'" for s in upper_symbols)
        start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        cols = ", ".join(_SELECT_COLUMNS)

        result = client.query(
            f"SELECT {cols} FROM {_FUTURES_TABLE} "
            f"WHERE symbol IN ({symbols_clause}) "
            f"AND open_time >= toDateTime64('{start_str}', 3, 'UTC') "
            f"AND open_time < toDateTime64('{end_str}', 3, 'UTC') "
            f"ORDER BY open_time, symbol"
        )

        if not result.result_rows:
            raise ValueError("No futures data found for the requested symbols/time range")

        df = pd.DataFrame(result.result_rows, columns=result.column_names)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

        # Resample if requested interval > 5m base
        requested_interval = interval
        if self._should_resample(requested_interval):
            frames = []
            for sym in df["symbol"].unique():
                frames.append(self._resample_frame(df[df["symbol"] == sym].copy(), requested_interval))
            df = pd.concat(frames, ignore_index=True)

        # Deduplicate and align across all symbols × timestamps
        sort_idx = np.lexsort([df["symbol"].values, df["open_time"].values])
        df = df.iloc[sort_idx].reset_index(drop=True)
        df = df.drop_duplicates(subset=["open_time", "symbol"], keep="last")
        timestamps = sorted(df["open_time"].drop_duplicates().tolist())
        resolved_symbols = sorted(df["symbol"].drop_duplicates().tolist())
        full_index = pd.MultiIndex.from_product([timestamps, resolved_symbols], names=["open_time", "symbol"])
        aligned = (
            df.set_index(["open_time", "symbol"])
            .reindex(full_index)
            .sort_index()
            .reset_index()
        )

        # Build fields dict
        fields: dict[str, np.ndarray] = {}

        # Core OHLCV
        for name in ["open", "high", "low", "close", "volume"]:
            fields[name] = self._pivot(aligned, name, timestamps, resolved_symbols)

        # Quote volume as turnover (for backward compatibility)
        fields["turnover"] = self._pivot(aligned, "quote_volume", timestamps, resolved_symbols)

        # Additional raw fields (zero-fill NaN)
        for name in _ZERO_FILL_FIELDS:
            fields[name] = np.nan_to_num(
                self._pivot(aligned, name, timestamps, resolved_symbols), nan=0.0,
            )

        # Derived fields
        fields["vwap"] = np.divide(fields["turnover"], fields["volume"] + 1e-12)
        close_ref = np.nan_to_num(np.abs(fields["close"]), nan=0.0, posinf=0.0, neginf=0.0)
        raw_spread = np.nan_to_num(np.abs(fields["high"] - fields["low"]) * 0.02, nan=0.0, posinf=0.0, neginf=0.0)
        min_spread = np.maximum(close_ref * 0.0001, 1e-6)
        max_spread = np.maximum(close_ref * 0.0025, min_spread)
        fields["bid_ask_spread"] = np.clip(raw_spread, min_spread, max_spread)

        # Masks
        liquidity_mask = fields["turnover"] > float(min_quote_volume)
        blocked_hours = {int(hour) % 24 for hour in (blocked_utc_hours or [])}
        tradable_by_row = np.array(
            [ts.tz_convert(UTC).hour not in blocked_hours for ts in timestamps],
            dtype=bool,
        )
        session_mask = np.broadcast_to(tradable_by_row[:, None], liquidity_mask.shape).copy()

        logger.info(
            "alpha.dataset loaded from {} symbols={} timestamps={} interval={}",
            _FUTURES_TABLE, len(resolved_symbols), len(timestamps), requested_interval,
        )

        return AlphaDataset(
            provider=provider,
            interval=requested_interval,
            symbols=resolved_symbols,
            timestamps=[ts.astimezone(UTC).isoformat() for ts in timestamps],
            fields=fields,
            liquidity_mask=liquidity_mask,
            session_mask=session_mask,
        )

    def _pivot(
        self,
        aligned: pd.DataFrame,
        column: str,
        timestamps: list[pd.Timestamp],
        symbols: list[str],
    ) -> np.ndarray:
        return (
            aligned.pivot(index="open_time", columns="symbol", values=column)
            .reindex(index=timestamps, columns=symbols)
            .to_numpy(dtype=float)
        )

    def _should_resample(self, interval: str) -> bool:
        minutes = self._interval_minutes(interval)
        return minutes is not None and minutes > _BASE_INTERVAL_MINUTES

    def _interval_minutes(self, interval: str) -> int | None:
        normalized = str(interval).strip().lower()
        if normalized.endswith("h"):
            try:
                return int(normalized[:-1]) * 60
            except ValueError:
                return None
        if not normalized.endswith("m"):
            return None
        try:
            return int(normalized[:-1])
        except ValueError:
            return None

    def _resample_frame(self, frame: pd.DataFrame, interval: str) -> pd.DataFrame:
        minutes = self._interval_minutes(interval)
        if minutes is None or minutes <= _BASE_INTERVAL_MINUTES:
            return frame
        if frame.empty:
            return frame

        symbol = str(frame["symbol"].iloc[0]).upper()
        rule = f"{minutes}min"
        ordered = frame.sort_values("open_time").set_index("open_time")

        agg_spec: dict[str, str] = {}
        for col in ordered.columns:
            if col in ("symbol",):
                agg_spec[col] = "first"
            elif col in ("close_time", "close", "mark_close", "premium_close"):
                agg_spec[col] = "last"
            elif col in ("open", "mark_open", "premium_open"):
                agg_spec[col] = "first"
            elif col in ("high", "mark_high", "premium_high"):
                agg_spec[col] = "max"
            elif col in ("low", "mark_low", "premium_low"):
                agg_spec[col] = "min"
            elif col in ("volume", "quote_volume", "trade_count",
                         "taker_buy_volume", "taker_buy_quote_volume"):
                agg_spec[col] = "sum"
            elif col in _RESAMPLE_LAST:
                agg_spec[col] = "last"

        aggregated = ordered.resample(rule, label="left", closed="left").agg(agg_spec)
        aggregated = aggregated.dropna(subset=["open", "high", "low", "close"], how="any").reset_index()
        aggregated["symbol"] = symbol

        logger.info(
            "alpha.dataset resampled symbol={} from=5m to={} rows_in={} rows_out={}",
            symbol, interval, len(frame), len(aggregated),
        )
        return aggregated


class StockDailyDatasetLoader:
    """Load A-share daily bar data and return as AlphaDataset (time x symbols numpy arrays)."""

    def __init__(self, db=None):
        self._db = db

    def _get_db(self):
        if self._db is None:
            from src.market_data.db import DB
            self._db = DB()
        return self._db

    def load(
        self,
        index_code: str = "000852",
        start_date: str = "2019-01-01",
        end_date: str = "2025-01-01",
        count: int = 1500,
    ) -> AlphaDataset:
        db = self._get_db()
        stocks = db.get_index_stocks(index_code, start_date)
        if not stocks:
            stocks = db.get_index_stocks(f"sh.{index_code}", start_date)
        if not stocks:
            raise ValueError(f"Could not find component stocks for index {index_code}")

        df = db.get_price(
            stocks,
            end_date,
            fields=["open", "high", "low", "close", "volume", "amount"],
            count=count,
            start_date=start_date,
        )
        if df.empty:
            raise ValueError("Data fetching returned empty DataFrame")

        df = df.reset_index()
        df.rename(columns={"code": "symbol"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])

        dates = sorted(df["date"].unique())
        symbols = sorted(df["symbol"].unique())

        fields: dict[str, np.ndarray] = {}
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                pivot = df.pivot(index="date", columns="symbol", values=col)
                fields[col] = pivot.reindex(index=dates, columns=symbols).to_numpy(dtype=float)

        if "amount" in fields and "volume" in fields:
            fields["vwap"] = fields["amount"] / (fields["volume"] + 1e-12)

        # Stock data doesn't have crypto-specific fields; provide zero placeholders
        n_t, n_s = len(dates), len(symbols)
        liquidity_mask = np.ones((n_t, n_s), dtype=bool)
        session_mask = np.ones((n_t, n_s), dtype=bool)

        logger.info(
            "alpha.dataset.stock loaded index={} symbols={} dates={}",
            index_code,
            n_s,
            n_t,
        )
        return AlphaDataset(
            provider="stock",
            interval="1d",
            symbols=list(symbols),
            timestamps=[str(d) for d in dates],
            fields=fields,
            liquidity_mask=liquidity_mask,
            session_mask=session_mask,
        )
