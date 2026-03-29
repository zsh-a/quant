from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

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
        for symbol in symbols:
            rows = self._store().query_bars(
                provider=provider,
                symbol=symbol.upper(),
                start_time=start_time,
                end_time=end_time,
                interval=interval,
            )
            if not rows:
                continue
            frame = pd.DataFrame(rows)
            frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
            frame["symbol"] = symbol.upper()
            frames.append(frame)

        if not frames:
            raise ValueError("No crypto minute-bar data found for the requested provider/symbols/time range")

        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values(["open_time", "symbol"]).drop_duplicates(subset=["open_time", "symbol"], keep="last")
        timestamps = sorted(df["open_time"].drop_duplicates().tolist())
        resolved_symbols = sorted(df["symbol"].drop_duplicates().tolist())
        full_index = pd.MultiIndex.from_product([timestamps, resolved_symbols], names=["open_time", "symbol"])
        aligned = (
            df.set_index(["open_time", "symbol"])
            .reindex(full_index)
            .sort_index()
            .reset_index()
        )

        fields = {
            "open": self._pivot_field(aligned, "open", timestamps, resolved_symbols),
            "high": self._pivot_field(aligned, "high", timestamps, resolved_symbols),
            "low": self._pivot_field(aligned, "low", timestamps, resolved_symbols),
            "close": self._pivot_field(aligned, "close", timestamps, resolved_symbols),
            "volume": self._pivot_field(aligned, "volume_base", timestamps, resolved_symbols),
            "turnover": self._pivot_field(aligned, "volume_quote", timestamps, resolved_symbols),
            "funding_rate": np.zeros((len(timestamps), len(resolved_symbols)), dtype=float),
            "open_interest": np.zeros((len(timestamps), len(resolved_symbols)), dtype=float),
        }
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
            interval=interval,
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
