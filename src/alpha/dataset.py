from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.market_data.crypto_store import CryptoMinuteBarStore


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


class CryptoMinuteDatasetLoader:
    def __init__(self, store: CryptoMinuteBarStore | None = None):
        self.store = store

    def _store(self) -> CryptoMinuteBarStore:
        if self.store is None:
            self.store = CryptoMinuteBarStore()
        return self.store

    def load(
        self,
        provider: str,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
        min_quote_volume: float = 0.0,
        blocked_utc_hours: list[int] | set[int] | tuple[int, ...] | None = None,
    ) -> AlphaDataset:
        frames = []
        requested_interval = interval
        source_interval = "1m" if self._should_resample_interval(interval) else interval
        for symbol in symbols:
            rows = self._store().query_bars(
                provider=provider,
                symbol=symbol.upper(),
                start_time=start_time,
                end_time=end_time,
                interval=source_interval,
            )
            if not rows and source_interval != requested_interval:
                logger.warning(
                    "alpha.dataset resample_source_missing provider={} symbol={} source_interval={} fallback_interval={}",
                    provider,
                    symbol.upper(),
                    source_interval,
                    requested_interval,
                )
                rows = self._store().query_bars(
                    provider=provider,
                    symbol=symbol.upper(),
                    start_time=start_time,
                    end_time=end_time,
                    interval=requested_interval,
                )
            if not rows:
                continue
            frame = pd.DataFrame(rows)
            frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
            frame["symbol"] = symbol.upper()
            if self._should_resample_interval(requested_interval) and source_interval == "1m":
                frame = self._resample_symbol_frame(frame, requested_interval)
            frames.append(frame)

        if not frames:
            raise ValueError("No crypto minute-bar data found for the requested provider/symbols/time range")

        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=["open_time"]).reset_index(drop=True)
        if df.empty:
            raise ValueError("No valid rows after concat — all open_time values are NaN")
        sort_idx = np.lexsort([df["symbol"].values, df["open_time"].values])
        df = df.iloc[sort_idx].reset_index(drop=True).drop_duplicates(subset=["open_time", "symbol"], keep="last")
        timestamps = sorted(df["open_time"].drop_duplicates().tolist())
        resolved_symbols = sorted(df["symbol"].drop_duplicates().tolist())
        full_index = pd.MultiIndex.from_product([timestamps, resolved_symbols], names=["open_time", "symbol"])
        aligned = (
            df.set_index(["open_time", "symbol"])
            .reindex(full_index)
            .sort_index()
            .reset_index()
        )

        n_t, n_s = len(timestamps), len(resolved_symbols)

        fields = {
            "open": self._pivot_field(aligned, "open", timestamps, resolved_symbols),
            "high": self._pivot_field(aligned, "high", timestamps, resolved_symbols),
            "low": self._pivot_field(aligned, "low", timestamps, resolved_symbols),
            "close": self._pivot_field(aligned, "close", timestamps, resolved_symbols),
            "volume": self._pivot_field(aligned, "volume_base", timestamps, resolved_symbols),
            "turnover": self._pivot_field(aligned, "volume_quote", timestamps, resolved_symbols),
        }

        # Read real funding_rate / open_interest from DB; fall back to zeros
        if "funding_rate" in aligned.columns:
            fr = self._pivot_field(aligned, "funding_rate", timestamps, resolved_symbols)
            fields["funding_rate"] = np.nan_to_num(fr, nan=0.0)
        else:
            fields["funding_rate"] = np.zeros((n_t, n_s), dtype=float)

        if "open_interest" in aligned.columns:
            oi = self._pivot_field(aligned, "open_interest", timestamps, resolved_symbols)
            fields["open_interest"] = np.nan_to_num(oi, nan=0.0)
        else:
            fields["open_interest"] = np.zeros((n_t, n_s), dtype=float)

        fields["vwap"] = np.divide(
            fields["turnover"],
            fields["volume"] + 1e-12,
        )
        close_ref = np.nan_to_num(np.abs(fields["close"]), nan=0.0, posinf=0.0, neginf=0.0)
        raw_spread = np.nan_to_num(np.abs(fields["high"] - fields["low"]) * 0.02, nan=0.0, posinf=0.0, neginf=0.0)
        min_spread = np.maximum(close_ref * 0.0001, 1e-6)
        max_spread = np.maximum(close_ref * 0.0025, min_spread)
        fields["bid_ask_spread"] = np.clip(raw_spread, min_spread, max_spread)

        liquidity_mask = fields["turnover"] > float(min_quote_volume)
        blocked_hours = {int(hour) % 24 for hour in (blocked_utc_hours or [])}
        tradable_by_row = np.array(
            [ts.tz_convert(UTC).hour not in blocked_hours for ts in timestamps],
            dtype=bool,
        )
        session_mask = np.broadcast_to(tradable_by_row[:, None], liquidity_mask.shape).copy()

        return AlphaDataset(
            provider=provider,
            interval=requested_interval,
            symbols=resolved_symbols,
            timestamps=[ts.astimezone(UTC).isoformat() for ts in timestamps],
            fields=fields,
            liquidity_mask=liquidity_mask,
            session_mask=session_mask,
        )

    def _pivot_field(
        self,
        aligned: pd.DataFrame,
        column: str,
        timestamps: list[pd.Timestamp],
        symbols: list[str],
    ) -> np.ndarray:
        matrix = (
            aligned.pivot(index="open_time", columns="symbol", values=column)
            .reindex(index=timestamps, columns=symbols)
            .to_numpy(dtype=float)
        )
        return matrix

    def _should_resample_interval(self, interval: str) -> bool:
        minutes = self._interval_minutes(interval)
        return minutes is not None and minutes > 1

    def _interval_minutes(self, interval: str) -> int | None:
        normalized = str(interval).strip().lower()
        if not normalized.endswith("m"):
            return None
        try:
            return int(normalized[:-1])
        except ValueError:
            return None

    def _resample_symbol_frame(self, frame: pd.DataFrame, interval: str) -> pd.DataFrame:
        minutes = self._interval_minutes(interval)
        if minutes is None or minutes <= 1:
            return frame
        if frame.empty:
            return frame

        symbol = str(frame["symbol"].iloc[0]).upper()
        rule = f"{minutes}min"
        ordered = frame.iloc[frame["open_time"].values.argsort(kind="mergesort")].copy()
        ordered = ordered.set_index("open_time")
        agg_spec = {
            "provider": "first",
            "market_type": "first",
            "symbol": "first",
            "exchange_symbol": "first",
            "close_time": "last",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume_base": "sum",
            "volume_quote": "sum",
            "trade_count": "sum",
        }
        if "funding_rate" in ordered.columns:
            agg_spec["funding_rate"] = "last"
        if "open_interest" in ordered.columns:
            agg_spec["open_interest"] = "last"
        aggregated = ordered.resample(rule, label="left", closed="left").agg(agg_spec)
        aggregated = aggregated.dropna(subset=["open", "high", "low", "close"], how="any").reset_index()
        aggregated["symbol"] = symbol
        aggregated["interval"] = interval
        if "exchange_symbol" in aggregated.columns:
            aggregated["exchange_symbol"] = aggregated["exchange_symbol"].fillna(symbol)
        if "close_time" in aggregated.columns:
            fallback_close = aggregated["open_time"] + pd.to_timedelta(minutes, unit="min") - pd.to_timedelta(1, unit="ms")
            aggregated["close_time"] = aggregated["close_time"].fillna(fallback_close)
        logger.info(
            "alpha.dataset resampled symbol={} from_interval=1m to_interval={} rows_in={} rows_out={}",
            symbol,
            interval,
            len(frame),
            len(aggregated),
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
