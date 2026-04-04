"""
Sync USD-M futures data from data.binance.vision → ClickHouse for alpha mining.

Downloads and merges five data sources into a single wide table ``crypto_data.futures_5m``:

  ┌─────────────────────┬──────────────┬────────────────────────────────────────┐
  │ Data Source          │ Granularity  │ Fields                                 │
  ├─────────────────────┼──────────────┼────────────────────────────────────────┤
  │ Klines              │ 5m           │ OHLCV, quote_vol, trades, taker_buy    │
  │ Premium Index       │ 5m           │ premium OHLC (futures-spot basis)      │
  │ Mark Price          │ 5m           │ mark OHLC (fair price / liquidation)   │
  │ Metrics             │ 5m (daily*)  │ OI, long/short ratios, taker L/S vol  │
  │ Funding Rate        │ 8h (monthly*)│ funding_rate (forward-filled to 5m)    │
  └─────────────────────┴──────────────┴────────────────────────────────────────┘

  *Metrics only available as daily archives; Funding rate only as monthly archives.

Usage:
    python -m src.market_data.binance_vision sync
    python -m src.market_data.binance_vision sync --symbols BTCUSDT,ETHUSDT --start 2023-01-01 -v
    python -m src.market_data.binance_vision status
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

import pandas as pd
import requests
from loguru import logger

from src.market_data.clickhouse import create_clickhouse_client

BASE_URL = "https://data.binance.vision/data/futures/um"

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
DEFAULT_INTERVAL = "5m"
DEFAULT_START = "2020-01-01"

TABLE = "crypto_data.futures_5m"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    symbol             LowCardinality(String),
    open_time          DateTime64(3, 'UTC'),
    close_time         DateTime64(3, 'UTC'),
    -- kline OHLCV
    open               Float64,
    high               Float64,
    low                Float64,
    close              Float64,
    volume             Float64,
    quote_volume       Float64,
    trade_count        UInt32,
    taker_buy_volume   Float64,
    taker_buy_quote_volume Float64,
    -- mark price OHLC
    mark_open          Float64 DEFAULT 0,
    mark_high          Float64 DEFAULT 0,
    mark_low           Float64 DEFAULT 0,
    mark_close         Float64 DEFAULT 0,
    -- premium index OHLC  (futures − spot basis)
    premium_open       Float64 DEFAULT 0,
    premium_high       Float64 DEFAULT 0,
    premium_low        Float64 DEFAULT 0,
    premium_close      Float64 DEFAULT 0,
    -- metrics (5-min snapshots)
    open_interest             Float64 DEFAULT 0,
    open_interest_value       Float64 DEFAULT 0,
    top_trader_long_short_ratio          Float64 DEFAULT 0,
    top_trader_long_short_position_ratio Float64 DEFAULT 0,
    long_short_ratio                     Float64 DEFAULT 0,
    taker_long_short_vol_ratio           Float64 DEFAULT 0,
    -- funding rate (8 h → forward-filled to 5 m)
    funding_rate       Float64 DEFAULT 0,
    ingested_at        DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMM(open_time)
ORDER BY (symbol, open_time)
"""

# ---------------------------------------------------------------------------
# Syncer
# ---------------------------------------------------------------------------


class BinanceVisionSyncer:
    """Download, merge and ingest USD-M futures data from data.binance.vision."""

    def __init__(self, client=None):
        self.client = client or create_clickhouse_client()
        self._ensure_schema()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "quent/3.0"
        # (connect_timeout, read_timeout) – generous read timeout for large monthly ZIPs
        self._timeout = (10, 120)

    def _ensure_schema(self) -> None:
        self.client.command("CREATE DATABASE IF NOT EXISTS crypto_data")
        self.client.command(_CREATE_TABLE_SQL)

    # -- public API ---------------------------------------------------------

    def sync(
        self,
        symbols: list[str] | None = None,
        interval: str = DEFAULT_INTERVAL,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        progress: Callable[[dict], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Sync futures data. Auto-resumes from last synced point per symbol."""
        symbols = [s.upper() for s in (symbols or DEFAULT_SYMBOLS)]
        end_dt = _parse_dt(end) if end else datetime.now(UTC) - timedelta(hours=6)
        results = []
        for symbol in symbols:
            start_dt = self._resolve_start(symbol, start)
            if start_dt.date() >= end_dt.date():
                logger.info("{} already up to date", symbol)
                results.append({"symbol": symbol, "status": "up_to_date", "inserted": 0})
                continue
            results.append(self._sync_symbol(symbol, interval, start_dt, end_dt, progress))
        return results

    def status(
        self, symbols: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Show latest synced open_time per symbol."""
        symbols = [s.upper() for s in (symbols or DEFAULT_SYMBOLS)]
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            latest = self._latest_time(symbol)
            rows.append({
                "symbol": symbol,
                "latest": latest.isoformat() if latest else None,
            })
        return rows

    # -- sync internals -----------------------------------------------------

    def _resolve_start(self, symbol: str, explicit: str | datetime | None) -> datetime:
        if explicit:
            return _parse_dt(explicit)
        latest = self._latest_time(symbol)
        if latest:
            return (latest + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return _parse_dt(DEFAULT_START)

    def _latest_time(self, symbol: str) -> datetime | None:
        r = self.client.query(
            f"SELECT max(open_time) FROM {TABLE} WHERE symbol = '{symbol}'"
        )
        val = r.result_rows[0][0] if r.result_rows else None
        if val is None:
            return None
        return val.replace(tzinfo=UTC) if val.tzinfo is None else val

    def _sync_symbol(self, symbol, interval, start, end, progress) -> dict[str, Any]:
        total_inserted = 0
        errors: list[dict] = []
        periods = _build_periods(start, end)
        n_monthly = sum(1 for p in periods if p["type"] == "monthly")
        n_daily = len(periods) - n_monthly
        logger.info(
            "{} syncing {} → {}  ({} monthly + {} daily periods)",
            symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), n_monthly, n_daily,
        )

        for i, period in enumerate(periods):
            try:
                rows = self._download_and_merge(symbol, interval, period)
                if not rows:
                    continue
                inserted = self._insert_rows(symbol, rows)
                total_inserted += inserted
                if progress:
                    progress({
                        "symbol": symbol,
                        "period": period["label"],
                        "progress": f"{i + 1}/{len(periods)}",
                        "rows": len(rows),
                        "new": inserted,
                        "total_new": total_inserted,
                    })
            except Exception as exc:
                logger.warning("{} {} failed: {}", symbol, period["label"], exc)
                errors.append({"period": period["label"], "error": str(exc)})

        logger.info("{} done: inserted={} errors={}", symbol, total_inserted, len(errors))
        return {
            "symbol": symbol,
            "status": "ok" if not errors else "partial",
            "inserted": total_inserted,
            "periods": len(periods),
            "errors": errors[:10],
        }

    # -- download & merge ---------------------------------------------------

    def _download_and_merge(self, symbol: str, interval: str, period: dict) -> list[dict]:
        """Download all five data sources for one period and merge by open_time."""
        is_monthly = period["type"] == "monthly"
        label = period["label"]

        # 1) Kline-format downloads — parallel (klines / premiumIndex / markPrice)
        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_k = pool.submit(self._fetch_kline_csv, "klines", symbol, interval, period)
            fut_p = pool.submit(self._fetch_kline_csv, "premiumIndexKlines", symbol, interval, period)
            fut_m = pool.submit(self._fetch_kline_csv, "markPriceKlines", symbol, interval, period)
            klines = fut_k.result()
            premium = fut_p.result()
            mark = fut_m.result()

        if not klines:
            return []

        # 2) Metrics — daily archives only (parallel within month)
        metrics = (
            self._fetch_metrics_month(symbol, label) if is_monthly
            else self._fetch_metrics_day(symbol, label)
        )

        # 3) Funding rate — monthly archives only
        funding: dict[datetime, float] = {}
        if is_monthly:
            funding = self._fetch_funding_rate(symbol, label)

        return _merge_all(klines, premium, mark, metrics, funding)

    # -- downloaders --------------------------------------------------------

    def _fetch_kline_csv(
        self, data_type: str, symbol: str, interval: str, period: dict,
    ) -> list[dict]:
        """Download kline-format CSV (works for klines / premiumIndex / markPrice)."""
        label = period["label"]
        freq = "monthly" if period["type"] == "monthly" else "daily"
        url = (
            f"{BASE_URL}/{freq}/{data_type}/{symbol}/{interval}/"
            f"{symbol}-{interval}-{label}.zip"
        )
        raw = self._fetch_zip_csv(url)
        return _parse_kline_rows(raw) if raw is not None else []

    def _fetch_metrics_day(self, symbol: str, date_label: str, quiet: bool = False) -> list[dict]:
        url = f"{BASE_URL}/daily/metrics/{symbol}/{symbol}-metrics-{date_label}.zip"
        raw = self._fetch_zip_csv(url, _quiet=quiet)
        return _parse_metrics_rows(raw) if raw is not None else []

    def _fetch_metrics_month(self, symbol: str, month_label: str) -> list[dict]:
        """Download daily metrics for every day in a month (parallel)."""
        year, month = int(month_label[:4]), int(month_label[5:7])
        first = datetime(year, month, 1, tzinfo=UTC)
        dates = []
        day = first
        while day < _next_month(first):
            dates.append(day.strftime("%Y-%m-%d"))
            day += timedelta(days=1)

        all_rows: list[dict] = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(self._fetch_metrics_day, symbol, d, quiet=True): d for d in dates}
            for fut in as_completed(futs):
                try:
                    all_rows.extend(fut.result())
                except Exception:
                    pass  # individual day failures are non-fatal
        logger.info("  fetched metrics {} ({} days, {} rows)", month_label, len(dates), len(all_rows))
        return all_rows

    def _fetch_funding_rate(self, symbol: str, month_label: str) -> dict[datetime, float]:
        url = (
            f"{BASE_URL}/monthly/fundingRate/{symbol}/"
            f"{symbol}-fundingRate-{month_label}.zip"
        )
        raw = self._fetch_zip_csv(url)
        return _parse_funding_rate_rows(raw) if raw is not None else {}

    def _fetch_zip_csv(self, url: str, _retries: int = 2, _quiet: bool = False) -> bytes | None:
        """Download ZIP → extract first CSV → return raw bytes.  None on 404."""
        fname = url.split("/")[-1]
        for attempt in range(_retries + 1):
            try:
                resp = self.session.get(url, timeout=self._timeout)
            except requests.RequestException as exc:
                if attempt < _retries:
                    logger.debug("retry {}/{} for {}: {}", attempt + 1, _retries, fname, exc)
                    continue
                logger.warning("download failed {}: {}", fname, exc)
                return None
            if resp.status_code == 404:
                return None
            if resp.status_code >= 500 and attempt < _retries:
                logger.debug("retry {}/{} for {} (HTTP {})", attempt + 1, _retries, fname, resp.status_code)
                continue
            resp.raise_for_status()
            if not _quiet:
                logger.info("  fetched {} ({:.0f} KB)", fname, len(resp.content) / 1024)
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                return zf.read(zf.namelist()[0])
        return None

    # -- insert with dedup --------------------------------------------------

    def _insert_rows(self, symbol: str, rows: list[dict]) -> int:
        if not rows:
            return 0
        open_times = [r["open_time"] for r in rows]
        existing = self._existing_times(symbol, min(open_times), max(open_times))
        new_rows = [r for r in rows if r["open_time"] not in existing]
        if not new_rows:
            return 0

        now = datetime.now(UTC)
        for r in new_rows:
            r["symbol"] = symbol
            r["ingested_at"] = now

        df = pd.DataFrame(new_rows)
        self.client.insert_df(TABLE, df)
        return len(new_rows)

    def _existing_times(self, symbol: str, start: datetime, end: datetime) -> set[datetime]:
        s = start.strftime("%Y-%m-%d %H:%M:%S")
        e = end.strftime("%Y-%m-%d %H:%M:%S")
        r = self.client.query(
            f"SELECT DISTINCT open_time FROM {TABLE} "
            f"WHERE symbol = '{symbol}' "
            f"AND open_time >= toDateTime64('{s}', 3, 'UTC') "
            f"AND open_time <= toDateTime64('{e}', 3, 'UTC')"
        )
        return {
            (row[0].replace(tzinfo=UTC) if row[0].tzinfo is None else row[0])
            for row in r.result_rows
        }


# ---------------------------------------------------------------------------
# CSV parsers
# ---------------------------------------------------------------------------


def _parse_kline_rows(raw: bytes) -> list[dict]:
    """Parse kline-format CSV (12 columns: open_time … ignore)."""
    out: list[dict] = []
    for row in csv.reader(io.StringIO(raw.decode())):
        try:
            ot = _norm_ts(int(row[0]))
            ct = _norm_ts(int(row[6]))
            out.append({
                "open_time": datetime.fromtimestamp(ot / 1000, tz=UTC),
                "close_time": datetime.fromtimestamp(ct / 1000, tz=UTC),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "quote_volume": float(row[7]),
                "trade_count": int(row[8]),
                "taker_buy_volume": float(row[9]),
                "taker_buy_quote_volume": float(row[10]),
            })
        except (IndexError, ValueError):
            continue
    return out


def _parse_metrics_rows(raw: bytes) -> list[dict]:
    """Parse metrics CSV (8 columns: create_time, symbol, OI …)."""
    out: list[dict] = []
    for row in csv.reader(io.StringIO(raw.decode())):
        try:
            ot = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
            out.append({
                "open_time": ot,
                "open_interest": float(row[2]),
                "open_interest_value": float(row[3]),
                "top_trader_long_short_ratio": float(row[4]),
                "top_trader_long_short_position_ratio": float(row[5]),
                "long_short_ratio": float(row[6]),
                "taker_long_short_vol_ratio": float(row[7]),
            })
        except (IndexError, ValueError):
            continue
    return out


def _parse_funding_rate_rows(raw: bytes) -> dict[datetime, float]:
    """Parse funding-rate CSV (3 columns: calc_time, interval_hours, rate)."""
    rates: dict[datetime, float] = {}
    for row in csv.reader(io.StringIO(raw.decode())):
        try:
            ts = _norm_ts(int(row[0]))
            rates[datetime.fromtimestamp(ts / 1000, tz=UTC)] = float(row[2])
        except (IndexError, ValueError):
            continue
    return rates


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def _merge_all(
    klines: list[dict],
    premium: list[dict],
    mark: list[dict],
    metrics: list[dict],
    funding: dict[datetime, float],
) -> list[dict]:
    """Merge five sources by open_time.  Klines define the timeline."""
    rows: dict[datetime, dict] = {}

    # base: kline OHLCV
    for k in klines:
        ot = k["open_time"]
        rows[ot] = {
            "open_time": ot,
            "close_time": k["close_time"],
            "open": k["open"], "high": k["high"], "low": k["low"], "close": k["close"],
            "volume": k["volume"], "quote_volume": k["quote_volume"],
            "trade_count": k["trade_count"],
            "taker_buy_volume": k["taker_buy_volume"],
            "taker_buy_quote_volume": k["taker_buy_quote_volume"],
            # defaults for optional sources
            "mark_open": 0.0, "mark_high": 0.0, "mark_low": 0.0, "mark_close": 0.0,
            "premium_open": 0.0, "premium_high": 0.0, "premium_low": 0.0, "premium_close": 0.0,
            "open_interest": 0.0, "open_interest_value": 0.0,
            "top_trader_long_short_ratio": 0.0,
            "top_trader_long_short_position_ratio": 0.0,
            "long_short_ratio": 0.0, "taker_long_short_vol_ratio": 0.0,
            "funding_rate": 0.0,
        }

    # overlay mark price
    for m in mark:
        r = rows.get(m["open_time"])
        if r:
            r["mark_open"] = m["open"]
            r["mark_high"] = m["high"]
            r["mark_low"] = m["low"]
            r["mark_close"] = m["close"]

    # overlay premium index (basis)
    for p in premium:
        r = rows.get(p["open_time"])
        if r:
            r["premium_open"] = p["open"]
            r["premium_high"] = p["high"]
            r["premium_low"] = p["low"]
            r["premium_close"] = p["close"]

    # overlay metrics
    for m in metrics:
        r = rows.get(m["open_time"])
        if r:
            r["open_interest"] = m["open_interest"]
            r["open_interest_value"] = m["open_interest_value"]
            r["top_trader_long_short_ratio"] = m["top_trader_long_short_ratio"]
            r["top_trader_long_short_position_ratio"] = m["top_trader_long_short_position_ratio"]
            r["long_short_ratio"] = m["long_short_ratio"]
            r["taker_long_short_vol_ratio"] = m["taker_long_short_vol_ratio"]

    # forward-fill funding rate (8 h → 5 m)
    if funding:
        sorted_fr = sorted(funding.items())
        idx, rate = 0, 0.0
        for ot in sorted(rows):
            while idx < len(sorted_fr) and sorted_fr[idx][0] <= ot:
                rate = sorted_fr[idx][1]
                idx += 1
            rows[ot]["funding_rate"] = rate

    return [rows[ot] for ot in sorted(rows)]


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------


def _next_month(dt: datetime) -> datetime:
    return dt.replace(year=dt.year + 1, month=1, day=1) if dt.month == 12 else dt.replace(month=dt.month + 1, day=1)


def _build_periods(start: datetime, end: datetime) -> list[dict]:
    """Monthly periods for complete past months, daily for the rest."""
    periods: list[dict] = []
    now = datetime.now(UTC)
    current_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    yesterday = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    start_day = start.replace(hour=0, minute=0, second=0, microsecond=0)

    month_cursor = start_day.replace(day=1)
    while month_cursor <= end:
        next_m = _next_month(month_cursor)
        is_complete = next_m <= current_month
        covers_full = start_day <= month_cursor
        covers_to_end = end >= (next_m - timedelta(days=1))

        if is_complete and covers_full and covers_to_end:
            periods.append({"type": "monthly", "label": month_cursor.strftime("%Y-%m")})
        else:
            day = max(start_day, month_cursor)
            day_end = min(next_m - timedelta(days=1), end, yesterday)
            while day <= day_end:
                periods.append({"type": "daily", "label": day.strftime("%Y-%m-%d")})
                day += timedelta(days=1)

        month_cursor = next_m
    return periods


def _norm_ts(ts: int) -> int:
    """Normalise microsecond timestamps (spot ≥2025) back to milliseconds."""
    return ts // 1000 if ts > 1_000_000_000_000_000 else ts


def _parse_dt(value: str | datetime | None) -> datetime:
    if value is None:
        raise ValueError("datetime value required")
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if len(value) <= 10:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    dt = datetime.fromisoformat(value)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sync USD-M futures data from data.binance.vision → ClickHouse",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("sync", help="Download & ingest (incremental)")
    sp.add_argument("--symbols", help="Comma-separated, e.g. BTCUSDT,ETHUSDT")
    sp.add_argument("--interval", default=DEFAULT_INTERVAL, help="Kline interval (default: 5m)")
    sp.add_argument("--start", help="Start date, e.g. 2020-01-01")
    sp.add_argument("--end", help="End date")
    sp.add_argument("--verbose", "-v", action="store_true")

    st = sub.add_parser("status", help="Show latest synced time per symbol")
    st.add_argument("--symbols", help="Comma-separated")

    return p


def _split_symbols(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    syncer = BinanceVisionSyncer()
    symbols = _split_symbols(getattr(args, "symbols", None))

    if args.command == "sync":

        def _on_progress(e: dict) -> None:
            print(
                f"  [{e['symbol']}] {e['period']}  {e['progress']}"
                f"  rows={e['rows']}  +{e['new']}  total={e['total_new']}"
            )

        result = syncer.sync(
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            end=getattr(args, "end", None),
            progress=_on_progress if args.verbose else None,
        )
    elif args.command == "status":
        result = syncer.status(symbols=symbols)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
