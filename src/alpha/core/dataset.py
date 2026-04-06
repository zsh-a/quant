from __future__ import annotations

import gc
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.market_data.clickhouse import create_clickhouse_client

# Default dtype for all numeric fields — float32 halves memory vs float64.
_DTYPE = np.float32


@dataclass
class AlphaDataset:
    interval: str
    symbols: list[str]
    timestamps: list[str]
    fields: dict[str, np.ndarray]
    liquidity_mask: np.ndarray
    session_mask: np.ndarray

    def shape(self) -> tuple[int, int]:
        close = self.fields["close"]
        return int(close.shape[0]), int(close.shape[1])

    def slice_by_index(self, start_idx: int, end_idx: int) -> AlphaDataset:
        """Contiguous slice — returns numpy views (zero-copy)."""
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(self.timestamps))
        sliced_fields = {name: values[start_idx:end_idx] for name, values in self.fields.items()}
        return AlphaDataset(
            interval=self.interval,
            symbols=list(self.symbols),
            timestamps=self.timestamps[start_idx:end_idx],
            fields=sliced_fields,
            liquidity_mask=self.liquidity_mask[start_idx:end_idx],
            session_mask=self.session_mask[start_idx:end_idx],
        )

    def take_indices(self, indices: list[int] | np.ndarray | tuple[int, ...]) -> AlphaDataset:
        """Index-based selection. Uses contiguous slice when indices are consecutive."""
        selected = np.asarray(indices, dtype=int)
        if selected.size == 0:
            empty_shape = (0, len(self.symbols))
            return AlphaDataset(
                interval=self.interval,
                symbols=list(self.symbols),
                timestamps=[],
                fields={name: np.empty(empty_shape, dtype=values.dtype) for name, values in self.fields.items()},
                liquidity_mask=np.empty(empty_shape, dtype=bool),
                session_mask=np.empty(empty_shape, dtype=bool),
            )

        # Check if indices form a contiguous range — use slice for zero-copy view
        if len(selected) > 1:
            diffs = np.diff(selected)
            if np.all(diffs == 1):
                return self.slice_by_index(int(selected[0]), int(selected[-1]) + 1)

        sliced_fields = {name: values[selected] for name, values in self.fields.items()}
        return AlphaDataset(
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

_RESAMPLE_LAST = {
    "symbol", "close_time", "close",
    "mark_close", "premium_close",
    "open_interest", "open_interest_value",
    "top_trader_long_short_ratio", "top_trader_long_short_position_ratio",
    "long_short_ratio", "taker_long_short_vol_ratio",
    "funding_rate",
}

_ZERO_FILL_FIELDS = [
    "trade_count", "taker_buy_volume", "taker_buy_quote_volume",
    "mark_open", "mark_high", "mark_low", "mark_close",
    "premium_open", "premium_high", "premium_low", "premium_close",
    "open_interest", "open_interest_value",
    "top_trader_long_short_ratio", "top_trader_long_short_position_ratio",
    "long_short_ratio", "taker_long_short_vol_ratio",
    "funding_rate",
]

# All numeric columns that need to be pivoted into fields.
_PIVOT_COLUMNS = [
    "open", "high", "low", "close", "volume", "quote_volume",
    *_ZERO_FILL_FIELDS,
]


class CryptoMinuteDatasetLoader:
    """Load alpha datasets from crypto_data.futures_5m in ClickHouse."""

    def __init__(self, client=None):
        self._client = client

    def _get_client(self):
        if self._client is None:
            self._client = create_clickhouse_client()
        return self._client

    def load(
        self,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = "5m",
        min_quote_volume: float = 0.0,
        blocked_utc_hours: list[int] | set[int] | tuple[int, ...] | None = None,
    ) -> AlphaDataset:
        client = self._get_client()
        upper_symbols = [s.upper() for s in symbols]

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

        requested_interval = interval
        if self._should_resample(requested_interval):
            frames = []
            for sym in df["symbol"].unique():
                frames.append(self._resample_frame(df[df["symbol"] == sym].copy(), requested_interval))
            df = pd.concat(frames, ignore_index=True)

        sort_idx = np.lexsort([df["symbol"].values, df["open_time"].values])
        df = df.iloc[sort_idx].reset_index(drop=True)
        df = df.drop_duplicates(subset=["open_time", "symbol"], keep="last")
        timestamps = sorted(df["open_time"].drop_duplicates().tolist())
        resolved_symbols = sorted(df["symbol"].drop_duplicates().tolist())

        # --- Build all field arrays directly without pandas pivot ---
        fields = self._build_fields_direct(df, timestamps, resolved_symbols)

        # Release the large DataFrame and force GC before downstream work.
        del df
        gc.collect()

        # Derived fields
        fields["vwap"] = np.divide(fields["turnover"], fields["volume"] + 1e-12)
        close_ref = np.nan_to_num(np.abs(fields["close"]), nan=0.0, posinf=0.0, neginf=0.0)
        raw_spread = np.nan_to_num(np.abs(fields["high"] - fields["low"]) * 0.02, nan=0.0, posinf=0.0, neginf=0.0)
        min_spread = np.maximum(close_ref * 0.0001, 1e-6)
        max_spread = np.maximum(close_ref * 0.0025, min_spread)
        fields["bid_ask_spread"] = np.clip(raw_spread, min_spread, max_spread)
        del close_ref, raw_spread, min_spread, max_spread

        liquidity_mask = fields["turnover"] > _DTYPE(min_quote_volume)
        blocked_hours = {int(hour) % 24 for hour in (blocked_utc_hours or [])}
        tradable_by_row = np.array(
            [ts.tz_convert(UTC).hour not in blocked_hours for ts in timestamps],
            dtype=bool,
        )
        session_mask = np.broadcast_to(tradable_by_row[:, None], liquidity_mask.shape).copy()

        logger.info(
            "alpha.dataset loaded symbols={} timestamps={} interval={} dtype={}",
            len(resolved_symbols), len(timestamps), requested_interval, _DTYPE.__name__,
        )

        return AlphaDataset(
            interval=requested_interval,
            symbols=resolved_symbols,
            timestamps=[ts.astimezone(UTC).isoformat() for ts in timestamps],
            fields=fields,
            liquidity_mask=liquidity_mask,
            session_mask=session_mask,
        )

    def _build_fields_direct(
        self,
        df: pd.DataFrame,
        timestamps: list,
        symbols: list[str],
    ) -> dict[str, np.ndarray]:
        """Build field arrays directly from DataFrame rows — avoids pandas pivot.

        Uses pd.Categorical for vectorized index mapping (no Python loop),
        then fills pre-allocated numpy arrays column by column.
        """
        n_time = len(timestamps)
        n_sym = len(symbols)

        # Vectorized index mapping via pd.Categorical — handles type
        # differences (Timestamp vs datetime64) automatically.
        row_codes = pd.Categorical(df["open_time"], categories=timestamps).codes
        col_codes = pd.Categorical(df["symbol"], categories=symbols).codes
        valid_mask = (row_codes >= 0) & (col_codes >= 0)
        valid_rows = row_codes[valid_mask]
        valid_cols = col_codes[valid_mask]

        # Determine which columns to extract
        unique_columns = list(dict.fromkeys(_PIVOT_COLUMNS))

        # Pre-allocate all field arrays and fill in one pass per column
        fields: dict[str, np.ndarray] = {}
        for col in unique_columns:
            arr = np.full((n_time, n_sym), np.nan, dtype=_DTYPE)
            if col in df.columns:
                np_vals = np.asarray(df[col].values[valid_mask], dtype=_DTYPE)
                arr[valid_rows, valid_cols] = np_vals
            fields[col] = arr

        # Apply zero-fill for specific fields
        for name in _ZERO_FILL_FIELDS:
            if name in fields:
                np.nan_to_num(fields[name], copy=False, nan=0.0)

        # Rename quote_volume → turnover
        if "quote_volume" in fields:
            fields["turnover"] = fields.pop("quote_volume")

        return fields

    def _pivot(self, aligned, column, timestamps, symbols) -> np.ndarray:
        """Legacy pivot — kept for compatibility but no longer used by load()."""
        return (
            aligned.pivot(index="open_time", columns="symbol", values=column)
            .reindex(index=timestamps, columns=symbols)
            .to_numpy(dtype=_DTYPE)
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
        return aggregated
