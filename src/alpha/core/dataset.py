from __future__ import annotations

import gc
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

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

    def build_universe_mask(
        self,
        adv_window: int = 288,
        top_n: int = 80,
        skip_top_n: int = 0,
        min_adv: float = 0.0,
    ) -> np.ndarray:
        """Build a dynamic universe mask based on rolling ADV.

        For each timestep, only the top-N symbols by rolling average daily
        turnover (quote volume) are included.  This implements the "dynamic
        universe filtering" needed for robust cross-sectional strategies.

        Args:
            adv_window: Lookback window in bars for rolling average
                        (default 288 = 1 day of 5m bars).
            top_n: Include top N symbols by ADV at each timestep.
            skip_top_n: Skip the top N most liquid symbols (for small-fund
                        strategies that avoid mega-caps).
            min_adv: Minimum ADV threshold (absolute, in quote currency).

        Returns:
            Boolean mask of shape (T, N) where True = symbol in universe.
        """
        turnover = self.fields.get("turnover")
        if turnover is None:
            turnover = self.fields.get("quote_volume")
        if turnover is None:
            # Fallback: all symbols included
            T, N = self.shape()
            return np.ones((T, N), dtype=bool)

        T, N = turnover.shape

        # Rolling average turnover (ADV proxy)
        # Use cumsum trick for fast rolling mean
        cumsum = np.nancumsum(turnover, axis=0)
        adv = np.full_like(turnover, np.nan)
        adv[adv_window:] = (cumsum[adv_window:] - cumsum[:-adv_window]) / adv_window
        # For the first `adv_window` bars, use expanding mean
        for t in range(1, min(adv_window, T)):
            adv[t] = cumsum[t] / (t + 1)
        adv[0] = turnover[0]

        # Build mask: per-timestep top-N by ADV (excluding top skip_top_n)
        mask = np.zeros((T, N), dtype=bool)
        for t in range(T):
            row = adv[t]
            valid = ~np.isnan(row)
            if not valid.any():
                continue
            # Apply min_adv threshold
            if min_adv > 0:
                valid = valid & (row >= min_adv)
            # Rank by ADV descending
            order = np.argsort(-np.nan_to_num(row, nan=-1))
            # Select range [skip_top_n, skip_top_n + top_n]
            count = 0
            for idx in order:
                if not valid[idx]:
                    continue
                count += 1
                if count <= skip_top_n:
                    continue
                if count > skip_top_n + top_n:
                    break
                mask[t, idx] = True

        included = mask.sum(axis=1).mean()
        logger.info(
            "universe_mask built: adv_window={} top_n={} skip_top={} "
            "avg_included={:.1f}/{} symbols",
            adv_window, top_n, skip_top_n, included, N,
        )
        return mask

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


@runtime_checkable
class DatasetLoader(Protocol):
    """Unified interface for loading alpha datasets from any market."""

    def load(
        self,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = "1d",
        **kwargs,
    ) -> AlphaDataset: ...


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

# ClickHouse-side aggregation rules per column (mirrors _resample_frame logic)
_CH_AGG: dict[str, str] = {}
for _col in _SELECT_COLUMNS:
    if _col in ("symbol", "open_time"):
        continue
    elif _col == "close_time":
        _CH_AGG[_col] = "max"
    elif _col in ("open", "mark_open", "premium_open"):
        _CH_AGG[_col] = "argMin"
    elif _col in ("high", "mark_high", "premium_high"):
        _CH_AGG[_col] = "max"
    elif _col in ("low", "mark_low", "premium_low"):
        _CH_AGG[_col] = "min"
    elif _col in ("volume", "quote_volume", "trade_count",
                   "taker_buy_volume", "taker_buy_quote_volume"):
        _CH_AGG[_col] = "sum"
    else:
        _CH_AGG[_col] = "argMax"  # last by time

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
        **kwargs,
    ) -> AlphaDataset:
        client = self._get_client()
        upper_symbols = [s.upper() for s in symbols]

        symbols_clause = ",".join(f"'{s}'" for s in upper_symbols)
        start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

        from time import perf_counter as _pc
        _t0 = _pc()

        requested_interval = interval
        resample_minutes = self._interval_minutes(requested_interval)
        if resample_minutes is not None and resample_minutes > _BASE_INTERVAL_MINUTES:
            query = self._build_ch_resample_query(
                symbols_clause, start_str, end_str, resample_minutes,
            )
        else:
            cols = ", ".join(_SELECT_COLUMNS)
            query = (
                f"SELECT {cols} FROM {_FUTURES_TABLE} "
                f"WHERE symbol IN ({symbols_clause}) "
                f"AND open_time >= toDateTime64('{start_str}', 3, 'UTC') "
                f"AND open_time < toDateTime64('{end_str}', 3, 'UTC') "
                f"ORDER BY open_time, symbol"
            )

        df = client.query_df(query)
        _t_query = _pc()

        if df.empty:
            raise ValueError("No futures data found for the requested symbols/time range")

        # Rename _ot → open_time when server-side resample was used
        if "_ot" in df.columns:
            df.rename(columns={"_ot": "open_time"}, inplace=True)

        # Downcast float64 → float32 early to halve DataFrame memory
        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].astype(_DTYPE)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

        sort_idx = np.lexsort([df["symbol"].values, df["open_time"].values])
        df = df.iloc[sort_idx].reset_index(drop=True)
        df = df.drop_duplicates(subset=["open_time", "symbol"], keep="last")
        timestamps = sorted(df["open_time"].drop_duplicates().tolist())
        resolved_symbols = sorted(df["symbol"].drop_duplicates().tolist())

        # --- Build all field arrays directly without pandas pivot ---
        fields = self._build_fields_direct(df, timestamps, resolved_symbols)
        _t_pivot = _pc()

        logger.info(
            "alpha.dataset.timing query={:.1f}s pivot={:.1f}s rows={}",
            _t_query - _t0, _t_pivot - _t_query, len(df),
        )

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

        mem_mb = sum(a.nbytes for a in fields.values()) / (1024 * 1024)
        logger.info(
            "alpha.dataset loaded symbols={} timestamps={} interval={} dtype={} fields={} mem={:.0f}MB",
            len(resolved_symbols), len(timestamps), requested_interval, _DTYPE.__name__,
            len(fields), mem_mb,
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

    @staticmethod
    def _build_ch_resample_query(
        symbols_clause: str, start_str: str, end_str: str, minutes: int,
    ) -> str:
        """Build a ClickHouse query that resamples 5m data server-side.

        Uses ``_ot`` as the interval alias to avoid ambiguity with the raw
        ``open_time`` column used inside ``argMin`` / ``argMax``.
        """
        interval_expr = f"toStartOfInterval(open_time, INTERVAL {minutes} MINUTE)"
        selects = ["symbol", f"{interval_expr} AS _ot"]
        for col, agg in _CH_AGG.items():
            if agg == "argMin":
                selects.append(f"argMin({col}, open_time) AS {col}")
            elif agg == "argMax":
                selects.append(f"argMax({col}, open_time) AS {col}")
            else:
                selects.append(f"{agg}({col}) AS {col}")
        return (
            f"SELECT {', '.join(selects)} FROM {_FUTURES_TABLE} "
            f"WHERE symbol IN ({symbols_clause}) "
            f"AND open_time >= toDateTime64('{start_str}', 3, 'UTC') "
            f"AND open_time < toDateTime64('{end_str}', 3, 'UTC') "
            f"GROUP BY symbol, {interval_expr} "
            f"ORDER BY _ot, symbol"
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
        ordered = frame.set_index("open_time").sort_index()

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


# ---------------------------------------------------------------------------
# A-share daily loader
# ---------------------------------------------------------------------------

_STOCK_TABLE = "stock_data.stock_daily"

_STOCK_SELECT_COLUMNS = [
    "date", "code",
    "open", "high", "low", "close", "preclose",
    "volume", "amount", "turn", "pctChg",
    "peTTM", "pbMRQ",
    "tradestatus", "isST", "adjfactor",
]

_STOCK_PRICE_COLS = ["open", "high", "low", "close", "preclose"]

_STOCK_PIVOT_COLUMNS = [
    "open", "high", "low", "close", "preclose",
    "volume", "amount", "turn", "pctChg",
    "peTTM", "pbMRQ", "isST", "adjfactor",
]


class AShareDailyDatasetLoader:
    """Load alpha datasets from stock_data.stock_daily in ClickHouse."""

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
        interval: str = "1d",
        exclude_st: bool = False,
        universe: str | None = None,
        **kwargs,
    ) -> AlphaDataset:
        if interval != "1d":
            raise ValueError(f"A-share data only supports daily interval ('1d'), got {interval!r}")

        client = self._get_client()

        # universe 优先：根据指数代码查成分股
        if universe:
            symbols = self._resolve_universe(client, universe)
            if not symbols:
                raise ValueError(f"No constituent stocks found for index {universe!r}")
            logger.info("alpha.dataset.a_share universe={} resolved {} symbols", universe, len(symbols))

        start_str = start_time.strftime("%Y-%m-%d")
        end_str = end_time.strftime("%Y-%m-%d")
        cols = ", ".join(_STOCK_SELECT_COLUMNS)

        if symbols:
            codes_clause = ",".join(f"'{s}'" for s in symbols)
            where_symbols = f"AND code IN ({codes_clause})"
        else:
            where_symbols = ""

        query = (
            f"SELECT {cols} FROM {_STOCK_TABLE} FINAL "
            f"WHERE date >= '{start_str}' AND date <= '{end_str}' "
            f"{where_symbols} "
            f"ORDER BY date, code"
        )
        df = client.query_df(query)

        if df.empty:
            raise ValueError("No A-share data found for the requested symbols/time range")

        df["date"] = pd.to_datetime(df["date"])

        # --- Forward-adjust prices (前复权) ---
        # Normalize adjfactor so that the latest date = 1.0
        latest_adj = df.groupby("code")["adjfactor"].transform("last")
        adj_ratio = df["adjfactor"] / latest_adj
        for col in _STOCK_PRICE_COLS:
            df[col] = df[col] * adj_ratio

        # Filter ST stocks if requested
        if exclude_st:
            df = df[df["isST"] != 1].reset_index(drop=True)

        # Sort and dedup
        sort_idx = np.lexsort([df["code"].values, df["date"].values])
        df = df.iloc[sort_idx].reset_index(drop=True)
        df = df.drop_duplicates(subset=["date", "code"], keep="last")

        timestamps = sorted(df["date"].drop_duplicates().tolist())
        resolved_symbols = sorted(df["code"].drop_duplicates().tolist())

        # --- Build field arrays ---
        fields = self._build_fields_direct(df, timestamps, resolved_symbols)

        tradestatus_arr = self._build_single_field(df, timestamps, resolved_symbols, "tradestatus")

        del df
        gc.collect()

        # Derived fields
        fields["vwap"] = np.divide(fields["amount"], fields["volume"] + 1e-12)
        fields["turnover"] = fields["amount"].copy()

        # isST as float
        np.nan_to_num(fields["isST"], copy=False, nan=0.0)

        # Masks
        volume_valid = ~np.isnan(fields["volume"]) & (fields["volume"] > 0)
        trading = ~np.isnan(tradestatus_arr) & (tradestatus_arr == 1)
        liquidity_mask = volume_valid & trading

        T, N = fields["close"].shape
        session_mask = np.ones((T, N), dtype=bool)

        logger.info(
            "alpha.dataset.a_share loaded symbols={} timestamps={} interval={} dtype={}",
            len(resolved_symbols), len(timestamps), interval, _DTYPE.__name__,
        )

        return AlphaDataset(
            interval=interval,
            symbols=resolved_symbols,
            timestamps=[ts.strftime("%Y-%m-%d") for ts in timestamps],
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
        """Build field arrays using vectorized pd.Categorical indexing."""
        n_time = len(timestamps)
        n_sym = len(symbols)

        row_codes = pd.Categorical(df["date"], categories=timestamps).codes
        col_codes = pd.Categorical(df["code"], categories=symbols).codes
        valid_mask = (row_codes >= 0) & (col_codes >= 0)
        valid_rows = row_codes[valid_mask]
        valid_cols = col_codes[valid_mask]

        fields: dict[str, np.ndarray] = {}
        for col in _STOCK_PIVOT_COLUMNS:
            arr = np.full((n_time, n_sym), np.nan, dtype=_DTYPE)
            if col in df.columns:
                np_vals = np.asarray(df[col].values[valid_mask], dtype=_DTYPE)
                arr[valid_rows, valid_cols] = np_vals
            fields[col] = arr

        return fields

    def _build_single_field(
        self,
        df: pd.DataFrame,
        timestamps: list,
        symbols: list[str],
        column: str,
    ) -> np.ndarray:
        """Build a single field array (for non-exported fields like tradestatus)."""
        n_time = len(timestamps)
        n_sym = len(symbols)
        row_codes = pd.Categorical(df["date"], categories=timestamps).codes
        col_codes = pd.Categorical(df["code"], categories=symbols).codes
        valid_mask = (row_codes >= 0) & (col_codes >= 0)
        arr = np.full((n_time, n_sym), np.nan, dtype=_DTYPE)
        if column in df.columns:
            np_vals = np.asarray(df[column].values[valid_mask], dtype=_DTYPE)
            arr[row_codes[valid_mask], col_codes[valid_mask]] = np_vals
        return arr

    @staticmethod
    def _resolve_universe(client, universe: str) -> list[str]:
        """Resolve index code(s) to constituent stock codes via stock_data.index_stocks."""
        codes = [c.strip() for c in universe.split(",") if c.strip()]
        codes_clause = ",".join(f"'{c}'" for c in codes)
        result = client.query(
            f"SELECT DISTINCT code FROM stock_data.index_stocks "
            f"WHERE index IN ({codes_clause})"
        )
        if not result.result_rows:
            return []
        return sorted(row[0] for row in result.result_rows)
