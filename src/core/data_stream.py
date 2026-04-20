import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import psutil

from src.utils.logging_config import get_logger

from .base import Bar, DataStream

logger = get_logger(__name__)


class CSVDataStream(DataStream):
    def __init__(self, csv_files: Dict[str, str], start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        csv_files: Dict mapping symbol to filepath
        """
        self.data: Dict[str, pd.DataFrame] = {}
        for symbol, path in csv_files.items():
            # Check if file has header or not. Based on inspection, it doesn't.
            df = pd.read_csv(path, header=None)
            if len(df.columns) >= 6:
                df.columns = ["timestamp", "open", "high", "low", "close", "volume", "amount"][: len(df.columns)]

            df["timestamp"] = pd.to_datetime(df["timestamp"])

            df = df.sort_values("timestamp")
            if start_date:
                df = df[df["timestamp"] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df["timestamp"] <= pd.to_datetime(end_date)]

            self.data[symbol] = df.reset_index(drop=True)

        self.idx = 0
        # Determine the union of all timestamps (or just use the first symbol if they are aligned)
        # Simplified for now: assume they are aligned by index
        self.max_idx = max(len(df) for df in self.data.values()) if self.data else 0

    def next_bar(self) -> Optional[Dict[str, Bar]]:
        if self.idx >= self.max_idx:
            return None

        bars = {}
        for symbol, df in self.data.items():
            if self.idx < len(df):
                row = df.iloc[self.idx]
                bars[symbol] = Bar(
                    symbol=symbol,
                    timestamp=row["timestamp"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row.get("volume", 0.0),
                    amount=row.get("amount", 0.0),
                )

        self.idx += 1
        return bars

    def reset(self):
        self.idx = 0


class DBDataStream(DataStream):
    def __init__(
        self,
        db_client,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        chunk_size_months: int = None,
    ):
        self.db_client = db_client
        self.symbols = symbols
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()

        # Dynamic chunk size based on symbol count
        # More symbols = smaller chunks to avoid memory overflow
        if chunk_size_months is None:
            if len(symbols) > 1000:
                chunk_size_months = 1  # 1 month for large portfolios
            elif len(symbols) > 100:
                chunk_size_months = 3  # 3 months for medium portfolios
            else:
                chunk_size_months = 12  # 1 year for small portfolios

        self.chunk_size_months = chunk_size_months

        # Memory monitoring
        self.process = psutil.Process(os.getpid())
        self.initial_memory_mb = self.process.memory_info().rss / 1024 / 1024

        logger.info(
            f"Initializing DBDataStream: {len(symbols)} symbols, "
            f"chunk_size={chunk_size_months} months, "
            f"initial_memory={self.initial_memory_mb:.2f}MB"
        )

        # Load master timeline (using the first symbol as reference or a market index)
        # This is lightweight compared to loading all columns for all stocks
        ref_symbol = symbols[0] if symbols else "sh.000001"
        try:
            # We fetch just dates if possible, but get_kline fetches all.
            # Optimization: In a real scenario, we'd add a get_trading_days method to DB.
            # For now, we assume fetching one symbol's full history is acceptable overhead
            # compared to fetching ALL symbols' full history.
            ref_df = self.db_client.get_kline(ref_symbol, start_date, end_date)
            if "date" in ref_df.columns:
                self.timestamps = pd.to_datetime(ref_df["date"]).sort_values().unique().tolist()
            elif "datetime" in ref_df.columns:
                self.timestamps = pd.to_datetime(ref_df["datetime"]).sort_values().unique().tolist()
            else:
                self.timestamps = pd.to_datetime(ref_df.index).sort_values().unique().tolist()
        except Exception as e:
            logger.warning(f"Failed to load timeline from {ref_symbol}: {e}, using fallback")
            # Fallback if reference symbol fails
            self.timestamps = pd.date_range(start=self.start_date, end=self.end_date, freq="B").tolist()

        self.total_bars = len(self.timestamps)
        self.global_idx = 0

        # Chunking
        self.current_chunk_data: Dict[str, pd.DataFrame] = {}
        self.current_chunk_start_idx = 0
        self.current_chunk_end_idx = 0
        self.chunks_loaded = 0

        logger.info(
            f"Timeline loaded: {self.total_bars} trading days from {self.start_date.date()} to {self.end_date.date()}"
        )

        self._load_next_chunk()

    def _load_next_chunk(self):
        if self.global_idx >= self.total_bars:
            self.current_chunk_data = {}
            logger.info(f"Reached end of data stream. Total chunks loaded: {self.chunks_loaded}")
            return

        chunk_start_ts = self.timestamps[self.global_idx]
        # Determine chunk end date using months instead of years
        chunk_end_date_limit = chunk_start_ts + pd.DateOffset(months=self.chunk_size_months)

        # Find the index in self.timestamps that corresponds to this limit
        # We want to load enough data to cover [chunk_start_ts, chunk_end_date_limit)

        # Filter timestamps for this chunk
        chunk_timestamps = [t for t in self.timestamps if t >= chunk_start_ts and t < chunk_end_date_limit]

        if not chunk_timestamps:
            # Should not happen unless global_idx is at end
            return

        chunk_end_ts = chunk_timestamps[-1]

        # Update chunk indices relative to global timestamps
        self.current_chunk_start_idx = self.global_idx
        self.current_chunk_end_idx = self.global_idx + len(chunk_timestamps)

        start_str = chunk_start_ts.strftime("%Y-%m-%d")
        end_str = chunk_end_ts.strftime("%Y-%m-%d")

        # Memory tracking before loading
        mem_before = self.process.memory_info().rss / 1024 / 1024

        # Load data for all symbols in this range
        self.current_chunk_data = {}
        symbols_loaded = 0
        for symbol in self.symbols:
            df = self.db_client.get_kline(symbol, start_str, end_str)
            if df.empty:
                continue

            df.columns = [c.lower() for c in df.columns]
            if "datetime" in df.columns:
                df["timestamp"] = pd.to_datetime(df["datetime"])
            elif "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
            else:
                df["timestamp"] = pd.to_datetime(df.index)

            if "adjfactor" in df.columns:
                for col in ["open", "high", "low", "close"]:
                    if col in df.columns:
                        df[col] = df[col] * df["adjfactor"]

            # Index by timestamp for faster lookup in next_bar
            self.current_chunk_data[symbol] = df.set_index("timestamp").sort_index()
            symbols_loaded += 1

        # Memory tracking after loading
        mem_after = self.process.memory_info().rss / 1024 / 1024
        mem_delta = mem_after - mem_before
        self.chunks_loaded += 1

        logger.info(
            f"Chunk {self.chunks_loaded} loaded: {start_str} to {end_str}, "
            f"{symbols_loaded}/{len(self.symbols)} symbols, "
            f"memory: {mem_after:.2f}MB (+{mem_delta:.2f}MB)"
        )

    def next_bar(self) -> Optional[Dict[str, Bar]]:
        if self.global_idx >= self.total_bars:
            return None

        # Check if we need to load next chunk
        if self.global_idx >= self.current_chunk_end_idx:
            self._load_next_chunk()
            if self.global_idx >= self.total_bars:  # Double check
                return None

        current_ts = self.timestamps[self.global_idx]
        bars = {}

        for symbol, df in self.current_chunk_data.items():
            if current_ts in df.index:
                row = df.loc[current_ts]
                # row might be a Series (single row) or DataFrame (duplicate timestamps)
                # handle duplicate timestamps if necessary, assume Series
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                bars[symbol] = Bar(
                    symbol=symbol,
                    timestamp=current_ts,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row.get("volume", 0.0),
                    amount=row.get("amount", 0.0),
                    extra={
                        k: v for k, v in row.items() if k not in ["open", "high", "low", "close", "volume", "amount"]
                    },
                )

        self.global_idx += 1
        # expose idx for progress tracking (mimicking old interface)
        self.idx = self.global_idx
        return bars

    def reset(self):
        self.global_idx = 0
        self.idx = 0
        self._load_next_chunk()


class CryptoDBDataStream(DataStream):
    """Chunked crypto data stream from ClickHouse ``crypto_data.futures_5m``."""

    _TABLE = "crypto_data.futures_5m"
    _BASE_MINUTES = 5

    _AGG = {
        "close_time": "max",
        "open": "argMin",
        "high": "max",
        "low": "min",
        "close": "argMax",
        "volume": "sum",
        "quote_volume": "sum",
        "trade_count": "sum",
    }

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1h",
        chunk_days: int = 90,
    ):
        from src.market_data.clickhouse import create_clickhouse_client

        self.ch = create_clickhouse_client()
        self.symbols = symbols
        self.interval = interval
        self.chunk_days = chunk_days

        self.start_dt = pd.to_datetime(start_date)
        self.end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.now(tz="UTC")
        if self.start_dt.tzinfo is None:
            self.start_dt = self.start_dt.tz_localize("UTC")
        if self.end_dt.tzinfo is None:
            self.end_dt = self.end_dt.tz_localize("UTC")

        # Parse interval minutes
        iv = interval.strip().lower()
        if iv.endswith("h"):
            self._minutes = int(iv[:-1]) * 60
        elif iv.endswith("m"):
            self._minutes = int(iv[:-1])
        elif iv == "1d":
            self._minutes = 1440
        else:
            self._minutes = 60

        # Load timeline by querying distinct timestamps for reference symbol
        self._load_timeline()
        self.total_bars = len(self.timestamps)
        self.global_idx = 0
        self.idx = 0

        self.current_chunk: Dict[str, pd.DataFrame] = {}
        self.chunk_end_idx = 0
        self.chunks_loaded = 0

        logger.info(
            f"CryptoDBDataStream: {len(symbols)} symbols, interval={interval}, "
            f"{self.total_bars} bars from {self.start_dt} to {self.end_dt}"
        )
        if self.total_bars > 0:
            self._load_next_chunk()

    @staticmethod
    def _fmt_dt(dt) -> str:
        # clickhouse_connect's DateTime binding accepts naive strings; strip tz.
        dt_naive = dt.tz_convert("UTC").tz_localize(None) if dt.tzinfo is not None else dt
        return dt_naive.strftime("%Y-%m-%d %H:%M:%S")

    def _load_timeline(self):
        """Get sorted unique timestamps for the first symbol."""
        ref = self.symbols[0] if self.symbols else "BTCUSDT"
        interval_expr = f"toStartOfInterval(open_time, INTERVAL {self._minutes} MINUTE)"
        q = (
            f"SELECT DISTINCT {interval_expr} AS t FROM {self._TABLE} "
            "WHERE symbol = {sym:String} "
            "AND open_time >= {s:DateTime} AND open_time < {e:DateTime} "
            "ORDER BY t"
        )
        result = self.ch.query(
            q,
            parameters={"sym": ref, "s": self._fmt_dt(self.start_dt), "e": self._fmt_dt(self.end_dt)},
        )
        self.timestamps = [r[0] for r in result.result_rows]

    def _build_query(self) -> str:
        interval_expr = f"toStartOfInterval(open_time, INTERVAL {self._minutes} MINUTE)"
        cols = ["symbol", f"{interval_expr} AS _ot"]
        for col, agg in self._AGG.items():
            if agg == "argMin":
                cols.append(f"argMin({col}, open_time) AS {col}")
            elif agg == "argMax":
                cols.append(f"argMax({col}, open_time) AS {col}")
            else:
                cols.append(f"{agg}({col}) AS {col}")
        return (
            f"SELECT {', '.join(cols)} FROM {self._TABLE} "
            "WHERE symbol IN {syms:Array(String)} "
            "AND open_time >= {s:DateTime} AND open_time < {e:DateTime} "
            f"GROUP BY symbol, {interval_expr} ORDER BY _ot, symbol"
        )

    def _load_next_chunk(self):
        if self.global_idx >= self.total_bars:
            self.current_chunk = {}
            return

        chunk_start = self.timestamps[self.global_idx]
        chunk_end_limit = chunk_start + pd.Timedelta(days=self.chunk_days)
        chunk_ts = [t for t in self.timestamps if chunk_start <= t < chunk_end_limit]
        if not chunk_ts:
            return
        self.chunk_end_idx = self.global_idx + len(chunk_ts)

        # ClickHouse timestamps come back naive (UTC); format bounds the same way.
        def _fmt(dt) -> str:
            if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
                dt = pd.Timestamp(dt).tz_convert("UTC").tz_localize(None)
            return pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M:%S")

        result = self.ch.query(
            self._build_query(),
            parameters={
                "syms": list(self.symbols),
                "s": _fmt(chunk_start),
                "e": _fmt(chunk_ts[-1] + pd.Timedelta(minutes=self._minutes)),
            },
        )
        col_names = list(result.column_names)
        df = pd.DataFrame(result.result_rows, columns=col_names)

        self.current_chunk = {}
        if not df.empty:
            for sym, gdf in df.groupby("symbol"):
                self.current_chunk[str(sym)] = gdf.set_index("_ot").sort_index()

        self.chunks_loaded += 1
        logger.info(
            f"Crypto chunk {self.chunks_loaded}: {len(chunk_ts)} bars, {len(self.current_chunk)} symbols loaded"
        )

    def next_bar(self) -> Optional[Dict[str, Bar]]:
        if self.global_idx >= self.total_bars:
            return None
        if self.global_idx >= self.chunk_end_idx:
            self._load_next_chunk()
            if self.global_idx >= self.total_bars:
                return None

        ts = self.timestamps[self.global_idx]
        bars = {}
        for symbol, df in self.current_chunk.items():
            if ts in df.index:
                row = df.loc[ts]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                bars[symbol] = Bar(
                    symbol=symbol,
                    timestamp=ts,
                    open=float(row.get("open", 0)),
                    high=float(row.get("high", 0)),
                    low=float(row.get("low", 0)),
                    close=float(row.get("close", 0)),
                    volume=float(row.get("volume", 0)),
                    amount=float(row.get("quote_volume", 0)),
                )
        self.global_idx += 1
        self.idx = self.global_idx
        return bars

    def reset(self):
        self.global_idx = 0
        self.idx = 0
        if self.total_bars > 0:
            self._load_next_chunk()


class RealtimeDataStream(DataStream):
    """
    Event-driven realtime data stream for live trading.
    Supports multiple data sources with automatic fallback.
    """

    def __init__(
        self,
        symbols: List[str],
        interval_seconds: int = 60,
        data_source: str = "akshare",
        enable_trading_hours_check: bool = True,
    ):
        """
        Initialize realtime data stream.

        Args:
            symbols: List of symbols to track
            interval_seconds: Update interval in seconds (default: 60 for 1-minute bars)
            data_source: Data source to use ('akshare', 'tushare', etc.)
            enable_trading_hours_check: Whether to check trading hours
        """
        self.symbols = symbols
        self.interval_seconds = interval_seconds
        self.data_source = data_source
        self.enable_trading_hours_check = enable_trading_hours_check

        self.last_fetch_time = time.time()
        self.consecutive_errors = 0
        self.max_retries = 3
        self.retry_delay = 5  # seconds

        # Initialize data source
        self._init_data_source()

        logger.info(
            f"RealtimeDataStream initialized: {len(symbols)} symbols, "
            f"interval={interval_seconds}s, source={data_source}"
        )

    def _init_data_source(self):
        """Initialize the data source client"""
        if self.data_source == "akshare":
            try:
                import akshare as ak

                self.ak = ak
                logger.info("AkShare data source initialized")
            except ImportError:
                logger.error("AkShare not installed, falling back to mock data")
                self.data_source = "mock"
        elif self.data_source == "tushare":
            try:
                import tushare as ts

                self.ts = ts
                logger.info("Tushare data source initialized")
            except ImportError:
                logger.error("Tushare not installed, falling back to akshare")
                self.data_source = "akshare"
                self._init_data_source()
        else:
            logger.warning(f"Unknown data source: {self.data_source}, using mock")
            self.data_source = "mock"

    def _is_trading_hours(self) -> bool:
        """
        Check if current time is within trading hours.
        China A-share market: 09:30-11:30, 13:00-15:00 (Mon-Fri)
        """
        if not self.enable_trading_hours_check:
            return True

        now = datetime.now()

        # Check if weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check trading hours
        current_time = now.time()
        morning_start = datetime.strptime("09:30", "%H:%M").time()
        morning_end = datetime.strptime("11:30", "%H:%M").time()
        afternoon_start = datetime.strptime("13:00", "%H:%M").time()
        afternoon_end = datetime.strptime("15:00", "%H:%M").time()

        is_morning = morning_start <= current_time <= morning_end
        is_afternoon = afternoon_start <= current_time <= afternoon_end

        return is_morning or is_afternoon

    def _wait_for_next_interval(self):
        """Wait until next data fetch interval"""
        time_to_wait = self.interval_seconds - (time.time() - self.last_fetch_time)
        if time_to_wait > 0:
            logger.debug(f"Waiting {time_to_wait:.1f}s for next interval")
            time.sleep(time_to_wait)

    def _wait_for_trading_hours(self):
        """Wait until market opens if outside trading hours"""
        while not self._is_trading_hours():
            now = datetime.now()
            logger.info(f"Outside trading hours ({now.strftime('%Y-%m-%d %H:%M:%S')}), waiting...")

            # Calculate time until next market open
            if now.weekday() >= 5:
                # Weekend, wait until Monday 09:30
                days_until_monday = (7 - now.weekday()) % 7
                if days_until_monday == 0:
                    days_until_monday = 1
                next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
                next_open = next_open + pd.Timedelta(days=days_until_monday)
            else:
                # Weekday, wait until next session
                current_time = now.time()
                morning_start = datetime.strptime("09:30", "%H:%M").time()
                afternoon_start = datetime.strptime("13:00", "%H:%M").time()

                if current_time < morning_start:
                    # Before morning session
                    next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
                elif current_time < afternoon_start:
                    # Lunch break
                    next_open = now.replace(hour=13, minute=0, second=0, microsecond=0)
                else:
                    # After market close, wait until next day
                    next_open = (now + pd.Timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)

            wait_seconds = (next_open - now).total_seconds()
            logger.info(
                f"Market opens at {next_open.strftime('%Y-%m-%d %H:%M:%S')}, waiting {wait_seconds / 60:.1f} minutes"
            )

            # Sleep in chunks to allow for interruption
            sleep_chunk = min(60, wait_seconds)  # Sleep max 1 minute at a time
            time.sleep(sleep_chunk)

    def _fetch_akshare_data(self) -> Dict[str, Bar]:
        """Fetch data from AkShare"""
        bars = {}
        current_ts = datetime.now()

        try:
            # Fetch ETF spot data
            df = self.ak.fund_etf_spot_em()

            for symbol in self.symbols:
                # Remove prefix if present (sh.510880 -> 510880)
                code = symbol.split(".")[-1]

                row = df[df["代码"] == code]
                if not row.empty:
                    data = row.iloc[0]
                    bars[symbol] = Bar(
                        symbol=symbol,
                        timestamp=current_ts,
                        open=float(data["开盘价"]),
                        high=float(data["最高价"]),
                        low=float(data["最低价"]),
                        close=float(data["最新价"]),
                        volume=float(data["成交量"]),
                        amount=float(data["成交额"]),
                        extra={"name": data["名称"]},
                    )
                else:
                    logger.warning(f"Symbol {symbol} not found in market data")

        except Exception as e:
            logger.error(f"AkShare fetch error: {e}")
            raise

        return bars

    def _fetch_mock_data(self) -> Dict[str, Bar]:
        """Generate mock data for testing"""
        import random

        bars = {}
        current_ts = datetime.now()

        for symbol in self.symbols:
            # Generate random OHLC data
            base_price = 100.0
            bars[symbol] = Bar(
                symbol=symbol,
                timestamp=current_ts,
                open=base_price + random.uniform(-5, 5),
                high=base_price + random.uniform(0, 10),
                low=base_price + random.uniform(-10, 0),
                close=base_price + random.uniform(-5, 5),
                volume=random.randint(1000000, 10000000),
                amount=random.randint(100000000, 1000000000),
                extra={"name": f"Mock {symbol}"},
            )

        return bars

    def next_bar(self) -> Optional[Dict[str, Bar]]:
        """
        Get next bar of realtime data.
        Blocks until data is available or returns None if stream should stop.
        """
        # Wait for next interval
        self._wait_for_next_interval()

        # Wait for trading hours if enabled
        if self.enable_trading_hours_check:
            self._wait_for_trading_hours()

        self.last_fetch_time = time.time()

        # Fetch data with retry logic
        for attempt in range(self.max_retries):
            try:
                if self.data_source == "akshare":
                    bars = self._fetch_akshare_data()
                elif self.data_source == "mock":
                    bars = self._fetch_mock_data()
                else:
                    logger.error(f"Unsupported data source: {self.data_source}")
                    return {}

                # Reset error counter on success
                if self.consecutive_errors > 0:
                    logger.info(f"Data fetch recovered after {self.consecutive_errors} errors")
                    self.consecutive_errors = 0

                logger.debug(f"Fetched {len(bars)} symbols at {datetime.now().strftime('%H:%M:%S')}")
                return bars

            except Exception as e:
                self.consecutive_errors += 1
                logger.error(f"Data fetch failed (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error("Max retries reached, returning empty data")
                    return {}

        return {}

    def reset(self):
        """Reset the stream (no-op for realtime stream)"""
        self.last_fetch_time = time.time()
        self.consecutive_errors = 0
        logger.info("RealtimeDataStream reset")
