"""
Alpha search CLI.

Data source: crypto_data.futures_5m (ClickHouse)

Examples:
    python -m src.alpha.cli search --symbols BTCUSDT,ETHUSDT --start 2026-03-27T00:00:00 --end 2026-03-28T00:00:00
    python -m src.alpha.cli evaluate --formula "cs_rank(ts_zscore(funding_rate, 20))" --symbols BTCUSDT --start 2026-03-27T00:00:00 --end 2026-03-28T00:00:00
    python -m src.alpha.cli auto --config config/alpha_lab/auto.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

from .risk import RiskConfig
from .service import AlphaService


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_symbols(value: str) -> list[str]:
    return [s.strip().upper() for s in value.split(",") if s.strip()]


def _parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(s.strip()) for s in value.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Shared CLI arguments
# ---------------------------------------------------------------------------

def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols (e.g. BTCUSDT,ETHUSDT)")
    parser.add_argument("--start", required=True, help="ISO8601 start time")
    parser.add_argument("--end", required=True, help="ISO8601 end time")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--min-quote-volume", type=float, default=0.0)
    parser.add_argument("--blocked-utc-hours", default="")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alpha factor search CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- validate / compile --
    validate = sub.add_parser("validate", help="Check formula syntax")
    validate.add_argument("--formula", required=True)

    compile_cmd = sub.add_parser("compile", help="Compile a formula to bytecode")
    compile_cmd.add_argument("--formula", required=True)

    # -- evaluate --
    evaluate = sub.add_parser("evaluate", help="Evaluate formula(s) on ClickHouse data")
    evaluate.add_argument("--formula", action="append", required=True)
    _add_data_args(evaluate)
    evaluate.add_argument("--summary-only", action="store_true")

    # -- search --
    search = sub.add_parser("search", help="Run evolutionary alpha search")
    _add_data_args(search)
    search.add_argument("--population-size", type=int, default=4)
    search.add_argument("--offspring-count", type=int, default=2)
    search.add_argument("--top-k", type=int, default=3)
    search.add_argument("--generations", type=int, default=2)
    search.add_argument("--novelty-threshold", type=float, default=0.995)
    search.add_argument("--n-splits", type=int, default=5)
    search.add_argument("--purge-window", type=int, default=0)
    search.add_argument("--embargo-window", type=int, default=0)
    search.add_argument("--seed", action="append", default=[])
    search.add_argument("--run-name", default=None)
    search.add_argument("--no-persist", action="store_true")
    search.add_argument("--llm-backend", default="auto", choices=["auto", "heuristic", "openai"])
    search.add_argument("--llm-model", default=None)

    # -- combine --
    combine = sub.add_parser("combine", help="Combine zoo factors into composite signal")
    _add_data_args(combine)
    combine.add_argument("--method", default="ic_weighted", choices=["equal", "ic_weighted", "ridge"])
    combine.add_argument("--max-factors", type=int, default=10)
    combine.add_argument("--zoo-limit", type=int, default=50)
    combine.add_argument("--vol-target", type=float, default=0.15)
    combine.add_argument("--max-drawdown", type=float, default=0.15)
    combine.add_argument("--trailing-stop", type=float, default=0.05)
    combine.add_argument("--summary-only", action="store_true")

    # -- auto --
    auto = sub.add_parser("auto", help="Run automated rolling search loop from config")
    auto.add_argument("--config", required=True, help="Path to YAML config")
    auto.add_argument("--once", action="store_true", help="Single cycle then exit")
    auto.add_argument("--max-cycles", type=int, default=None)

    # -- inspect --
    sub.add_parser("list-runs", help="List persisted runs").add_argument("--limit", type=int, default=20)
    sub.add_parser("list-zoo", help="List persisted zoo entries").add_argument("--limit", type=int, default=50)
    show = sub.add_parser("show-run", help="Show a persisted run")
    show.add_argument("--run-id", required=True)
    lineage = sub.add_parser("lineage", help="Show lineage for a run")
    lineage.add_argument("--run-id", required=True)

    return parser


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------

def run_command(args: argparse.Namespace, service: AlphaService) -> Any:
    cmd = args.command

    if cmd == "validate":
        return service.validate_formula(args.formula)

    if cmd == "compile":
        return service.compile_formula(args.formula)

    if cmd == "evaluate":
        formulas = args.formula
        if len(formulas) == 1:
            return service.evaluate_formula_from_db(
                formula=formulas[0],
                symbols=_parse_symbols(args.symbols),
                start_time=_parse_iso(args.start),
                end_time=_parse_iso(args.end),
                interval=args.interval,
                min_quote_volume=args.min_quote_volume,
                blocked_utc_hours=_parse_int_list(args.blocked_utc_hours),
                summary_only=args.summary_only,
            )
        return service.evaluate_formulas_from_db(
            formulas=formulas,
            symbols=_parse_symbols(args.symbols),
            start_time=_parse_iso(args.start),
            end_time=_parse_iso(args.end),
            interval=args.interval,
            min_quote_volume=args.min_quote_volume,
            blocked_utc_hours=_parse_int_list(args.blocked_utc_hours),
            summary_only=args.summary_only,
        )

    if cmd == "search":
        return service.search_formulas_on_db(
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

    if cmd == "combine":
        return service.combine_factors_from_db(
            symbols=_parse_symbols(args.symbols),
            start_time=_parse_iso(args.start),
            end_time=_parse_iso(args.end),
            interval=args.interval,
            min_quote_volume=args.min_quote_volume,
            blocked_utc_hours=_parse_int_list(args.blocked_utc_hours),
            method=args.method,
            max_factors=args.max_factors,
            zoo_limit=args.zoo_limit,
            risk_config=RiskConfig(
                vol_target=args.vol_target,
                max_drawdown=args.max_drawdown,
                trailing_stop_pct=args.trailing_stop,
            ),
            summary_only=args.summary_only,
        )

    if cmd == "auto":
        from .auto_runner import load_auto_search_config, run_auto_search_loop
        config = load_auto_search_config(args.config)
        return run_auto_search_loop(
            service=service, config=config,
            once=args.once, max_cycles=args.max_cycles,
        )

    if cmd == "list-runs":
        return {"runs": service.list_runs(limit=args.limit)}
    if cmd == "list-zoo":
        return {"zoo": service.list_zoo(limit=args.limit)}
    if cmd == "show-run":
        return service.load_run(args.run_id)
    if cmd == "lineage":
        return service.get_lineage(args.run_id)

    raise ValueError(f"Unknown command: {cmd}")


def main(argv: list[str] | None = None, service: AlphaService | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if service is None and args.command == "auto":
        from .auto_runner import build_service_from_auto_search_config, load_auto_search_config
        service = build_service_from_auto_search_config(load_auto_search_config(args.config))

    if service is None:
        service = AlphaService(
            llm_backend_name=getattr(args, "llm_backend", "auto"),
            llm_model=getattr(args, "llm_model", None),
        )

    result = run_command(args, service)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
