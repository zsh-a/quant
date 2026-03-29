"""
CLI for the database-backed alpha_lab closed loop.

Examples:
    python -m src.alpha_lab.cli evaluate-db --formula "CSRank(ts_mean(close,5)-close)" --provider bitget --symbols BTCUSDT,ETHUSDT --start 2026-03-27T00:00:00+00:00 --end 2026-03-28T00:00:00+00:00
    python -m src.alpha_lab.cli search-db --provider bitget --symbols BTCUSDT,ETHUSDT,SOLUSDT --start 2026-03-27T00:00:00+00:00 --end 2026-03-28T00:00:00+00:00 --seed "CSRank(ts_mean(close,5)-close)" --seed "CSRank(ts_std(close,5))"
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

from src.alpha_lab.service import AlphaLabService


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_symbols(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Database-backed alpha_lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate a formula")
    validate.add_argument("--formula", required=True)

    compile_cmd = subparsers.add_parser("compile", help="Compile a formula")
    compile_cmd.add_argument("--formula", required=True)

    evaluate = subparsers.add_parser("evaluate-db", help="Evaluate a formula on ClickHouse minute data")
    evaluate.add_argument("--formula", required=True)
    evaluate.add_argument("--provider", required=True)
    evaluate.add_argument("--symbols", required=True, help="Comma-separated symbols")
    evaluate.add_argument("--start", required=True, help="ISO8601 start time")
    evaluate.add_argument("--end", required=True, help="ISO8601 end time")
    evaluate.add_argument("--interval", default="5m")
    evaluate.add_argument("--min-quote-volume", type=float, default=0.0)
    evaluate.add_argument("--blocked-utc-hours", default="")
    evaluate.add_argument("--summary-only", action="store_true")

    batch_evaluate = subparsers.add_parser("batch-evaluate-db", help="Evaluate multiple formulas on ClickHouse minute data")
    batch_evaluate.add_argument("--formula", action="append", required=True)
    batch_evaluate.add_argument("--provider", required=True)
    batch_evaluate.add_argument("--symbols", required=True, help="Comma-separated symbols")
    batch_evaluate.add_argument("--start", required=True, help="ISO8601 start time")
    batch_evaluate.add_argument("--end", required=True, help="ISO8601 end time")
    batch_evaluate.add_argument("--interval", default="5m")
    batch_evaluate.add_argument("--min-quote-volume", type=float, default=0.0)
    batch_evaluate.add_argument("--blocked-utc-hours", default="")
    batch_evaluate.add_argument("--summary-only", action="store_true")

    benchmark = subparsers.add_parser("benchmark-vm", help="Benchmark the local Stack VM with synthetic tensors")
    benchmark.add_argument("--formula", action="append", default=[])
    benchmark.add_argument("--rows", type=int, default=2048)
    benchmark.add_argument("--cols", type=int, default=16)
    benchmark.add_argument("--repeat", type=int, default=5)

    benchmark_db = subparsers.add_parser("benchmark-db", help="Benchmark the database-backed alpha evaluation pipeline")
    benchmark_db.add_argument("--provider", required=True)
    benchmark_db.add_argument("--symbols", required=True, help="Comma-separated symbols")
    benchmark_db.add_argument("--start", required=True, help="ISO8601 start time")
    benchmark_db.add_argument("--end", required=True, help="ISO8601 end time")
    benchmark_db.add_argument("--interval", default="5m")
    benchmark_db.add_argument("--min-quote-volume", type=float, default=0.0)
    benchmark_db.add_argument("--blocked-utc-hours", default="")
    benchmark_db.add_argument("--formula", action="append", default=[])
    benchmark_db.add_argument("--repeat", type=int, default=3)

    search = subparsers.add_parser("search-db", help="Run a simple population -> evaluate -> breed loop on ClickHouse minute data")
    search.add_argument("--provider", required=True)
    search.add_argument("--symbols", required=True, help="Comma-separated symbols")
    search.add_argument("--start", required=True, help="ISO8601 start time")
    search.add_argument("--end", required=True, help="ISO8601 end time")
    search.add_argument("--interval", default="5m")
    search.add_argument("--min-quote-volume", type=float, default=0.0)
    search.add_argument("--population-size", type=int, default=4)
    search.add_argument("--offspring-count", type=int, default=2)
    search.add_argument("--top-k", type=int, default=3)
    search.add_argument("--generations", type=int, default=2)
    search.add_argument("--run-name", default=None)
    search.add_argument("--no-persist", action="store_true")
    search.add_argument("--novelty-threshold", type=float, default=0.995)
    search.add_argument("--n-splits", type=int, default=5)
    search.add_argument("--purge-window", type=int, default=0)
    search.add_argument("--embargo-window", type=int, default=0)
    search.add_argument("--blocked-utc-hours", default="")
    search.add_argument("--llm-backend", default="auto", choices=["auto", "heuristic", "openai"])
    search.add_argument("--llm-model", default=None)
    search.add_argument("--llm-base-url", default=None)
    search.add_argument("--llm-api-key", default=None)
    search.add_argument("--seed", action="append", default=[])

    list_runs = subparsers.add_parser("list-runs", help="List persisted alpha_lab runs")
    list_runs.add_argument("--limit", type=int, default=20)

    show_run = subparsers.add_parser("show-run", help="Show one persisted alpha_lab run")
    show_run.add_argument("--run-id", required=True)

    list_zoo = subparsers.add_parser("list-zoo", help="List persisted alpha zoo entries")
    list_zoo.add_argument("--limit", type=int, default=50)

    lineage = subparsers.add_parser("lineage", help="Show persisted lineage for a run")
    lineage.add_argument("--run-id", required=True)

    return parser


def run_command(args: argparse.Namespace, service: AlphaLabService) -> Any:
    if args.command == "validate":
        return service.validate_formula(args.formula)
    if args.command == "compile":
        return service.compile_formula(args.formula)
    if args.command == "evaluate-db":
        return service.evaluate_formula_from_db(
            formula=args.formula,
            provider=args.provider,
            symbols=_parse_symbols(args.symbols),
            start_time=_parse_iso(args.start),
            end_time=_parse_iso(args.end),
            interval=args.interval,
            min_quote_volume=args.min_quote_volume,
            blocked_utc_hours=_parse_int_list(args.blocked_utc_hours),
            summary_only=args.summary_only,
        )
    if args.command == "batch-evaluate-db":
        return service.evaluate_formulas_from_db(
            formulas=args.formula,
            provider=args.provider,
            symbols=_parse_symbols(args.symbols),
            start_time=_parse_iso(args.start),
            end_time=_parse_iso(args.end),
            interval=args.interval,
            min_quote_volume=args.min_quote_volume,
            blocked_utc_hours=_parse_int_list(args.blocked_utc_hours),
            summary_only=args.summary_only,
        )
    if args.command == "benchmark-vm":
        return service.benchmark_vm(
            formulas=args.formula,
            rows=args.rows,
            cols=args.cols,
            repeat=args.repeat,
        )
    if args.command == "benchmark-db":
        return service.benchmark_db(
            provider=args.provider,
            symbols=_parse_symbols(args.symbols),
            start_time=_parse_iso(args.start),
            end_time=_parse_iso(args.end),
            interval=args.interval,
            min_quote_volume=args.min_quote_volume,
            blocked_utc_hours=_parse_int_list(args.blocked_utc_hours),
            formulas=args.formula,
            repeat=args.repeat,
        )
    if args.command == "search-db":
        return service.search_formulas_on_db(
            provider=args.provider,
            symbols=_parse_symbols(args.symbols),
            start_time=_parse_iso(args.start),
            end_time=_parse_iso(args.end),
            interval=args.interval,
            min_quote_volume=args.min_quote_volume,
            seeds=args.seed,
            population_size=args.population_size,
            offspring_count=args.offspring_count,
            top_k=args.top_k,
            generations=args.generations,
            run_name=args.run_name,
            persist=not args.no_persist,
            novelty_threshold=args.novelty_threshold,
            n_splits=args.n_splits,
            purge_window=args.purge_window,
            embargo_window=args.embargo_window,
            blocked_utc_hours=_parse_int_list(args.blocked_utc_hours),
        )
    if args.command == "list-runs":
        return {"runs": service.list_runs(limit=args.limit)}
    if args.command == "show-run":
        return service.load_run(args.run_id)
    if args.command == "list-zoo":
        return {"zoo": service.list_zoo(limit=args.limit)}
    if args.command == "lineage":
        return service.get_lineage(args.run_id)
    raise ValueError(f"Unsupported command: {args.command}")


def main(argv: list[str] | None = None, service: AlphaLabService | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    default_service = service or AlphaLabService(
        llm_backend_name=getattr(args, "llm_backend", "auto"),
        llm_model=getattr(args, "llm_model", None),
        llm_base_url=getattr(args, "llm_base_url", None),
        llm_api_key=getattr(args, "llm_api_key", None),
    )
    result = run_command(args, default_service)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
