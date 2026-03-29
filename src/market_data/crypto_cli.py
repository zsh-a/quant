"""
Local CLI for crypto minute-data initialization and synchronization.

Examples:
    python -m src.market_data.crypto_cli init-db
    python -m src.market_data.crypto_cli bootstrap --provider bitget --symbols BTCUSDT,ETHUSDT
    python -m src.market_data.crypto_cli sync --provider binance --symbols BTCUSDT --start 2026-03-27T00:00:00+00:00 --end 2026-03-28T00:00:00+00:00
    python -m src.market_data.crypto_cli sync-history-2020 --provider bitget --verbose
    python -m src.market_data.crypto_cli overview
    python -m src.market_data.crypto_cli coverage --interval 1m --limit 50
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

from src.market_data.crypto_pipeline import CryptoMinuteSyncService

DEFAULT_HISTORY_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
DEFAULT_HISTORY_START = "2020-01-01T00:00:00+00:00"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_symbols(value: str | None) -> list[str] | None:
    if not value:
        return None
    symbols = [item.strip().upper() for item in value.split(",") if item.strip()]
    return symbols or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crypto minute-data initialization and sync CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-db", help="Initialize ClickHouse crypto schema and instruments")
    init_parser.add_argument("--provider", default=None, help="Provider name, e.g. bitget/binance")
    init_parser.add_argument("--symbols", default=None, help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT")

    bootstrap_parser = subparsers.add_parser("bootstrap", help="Initialize and sync default minute bars")
    bootstrap_parser.add_argument("--provider", default=None, help="Provider name, e.g. bitget/binance")
    bootstrap_parser.add_argument("--symbols", default=None, help="Comma-separated symbols")
    bootstrap_parser.add_argument("--interval", default=None, help="Bar interval, default from config")

    backfill_parser = subparsers.add_parser("backfill", help="Initialize and backfill full history")
    backfill_parser.add_argument("--provider", default=None, help="Provider name, e.g. bitget/binance")
    backfill_parser.add_argument("--symbols", default=None, help="Comma-separated symbols")
    backfill_parser.add_argument("--interval", default=None, help="Bar interval, default from config")
    backfill_parser.add_argument("--start", default=None, help="ISO8601 start time, defaults to crypto_market.full_history_start")
    backfill_parser.add_argument("--end", default=None, help="ISO8601 end time")
    backfill_parser.add_argument("--verbose", action="store_true", help="Print per-window sync progress")

    history_parser = subparsers.add_parser(
        "sync-history-2020",
        aliases=["sync-2020"],
        help="Sync BTC/ETH/SOL/DOGE from 2020 and only fetch missing local ranges",
    )
    history_parser.add_argument("--provider", default=None, help="Provider name, e.g. bitget/binance")
    history_parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_HISTORY_SYMBOLS),
        help="Comma-separated symbols, default BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT",
    )
    history_parser.add_argument("--interval", default=None, help="Bar interval, default from config")
    history_parser.add_argument(
        "--start",
        default=DEFAULT_HISTORY_START,
        help="ISO8601 history start time, defaults to 2020-01-01T00:00:00+00:00",
    )
    history_parser.add_argument("--end", default=None, help="ISO8601 end time")
    history_parser.add_argument("--verbose", action="store_true", help="Print per-window sync progress")

    sync_parser = subparsers.add_parser("sync", help="Trigger minute-bar synchronization")
    sync_parser.add_argument("--provider", required=True, help="Provider name, e.g. bitget/binance")
    sync_parser.add_argument("--symbols", default=None, help="Comma-separated symbols")
    sync_parser.add_argument("--interval", default="1m", help="Bar interval")
    sync_parser.add_argument("--start", default=None, help="ISO8601 start time")
    sync_parser.add_argument("--end", default=None, help="ISO8601 end time")
    sync_parser.add_argument("--verbose", action="store_true", help="Print per-window sync progress")

    subparsers.add_parser("overview", help="Show crypto data overview in ClickHouse")

    coverage_parser = subparsers.add_parser("coverage", help="Show symbol coverage in ClickHouse")
    coverage_parser.add_argument("--interval", default="1m", help="Bar interval")
    coverage_parser.add_argument("--limit", type=int, default=100, help="Max rows to return")

    return parser


def run_command(args: argparse.Namespace, service: CryptoMinuteSyncService) -> Any:
    progress_callback = _build_progress_callback(verbose=getattr(args, "verbose", False))
    if args.command == "init-db":
        return service.initialize_database(
            provider=args.provider,
            symbols=_parse_symbols(args.symbols),
        )
    if args.command == "bootstrap":
        return service.bootstrap_default_dataset(
            provider=args.provider,
            symbols=_parse_symbols(args.symbols),
            interval=args.interval,
        )
    if args.command == "backfill":
        return service.backfill_history(
            provider=args.provider,
            symbols=_parse_symbols(args.symbols),
            interval=args.interval,
            start_time=_parse_iso(args.start),
            end_time=_parse_iso(args.end),
            progress_callback=progress_callback,
        )
    if args.command in {"sync-history-2020", "sync-2020"}:
        return service.backfill_history(
            provider=args.provider,
            symbols=_parse_symbols(args.symbols) or DEFAULT_HISTORY_SYMBOLS,
            interval=args.interval,
            start_time=_parse_iso(args.start) or _parse_iso(DEFAULT_HISTORY_START),
            end_time=_parse_iso(args.end),
            progress_callback=progress_callback,
        )
    if args.command == "sync":
        return service.sync_minute_bars(
            provider=args.provider,
            symbols=_parse_symbols(args.symbols),
            interval=args.interval,
            start_time=_parse_iso(args.start),
            end_time=_parse_iso(args.end),
            progress_callback=progress_callback,
        )
    if args.command == "overview":
        return service.get_overview()
    if args.command == "coverage":
        return {"coverage": service.get_coverage(interval=args.interval, limit=args.limit)}
    raise ValueError(f"Unsupported command: {args.command}")


def _build_progress_callback(verbose: bool):
    if not verbose:
        return None

    def callback(event: dict[str, Any]) -> None:
        print(
            "[sync] "
            f"{event['provider']}:{event['symbol']} {event['interval']} "
            f"cursor={event['cursor']} batch_end={event['batch_end']} "
            f"last_open={event['last_open_time']} fetched={event['fetched']} "
            f"inserted={event['inserted']} note={event.get('note', '')}"
        )

    return callback


def main(argv: list[str] | None = None, service: CryptoMinuteSyncService | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_command(args, service or CryptoMinuteSyncService())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
