"""
Automated rolling alpha search loop.

Runs evolutionary search on rolling time windows, persists results,
and carries over top formulas across cycles.

Data source: crypto_data.futures_5m (ClickHouse)
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from .cli import _parse_iso
from .risk.models import RiskConfig
from .service import AlphaService


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class WindowConfig(BaseModel):
    lookback_minutes: int = 24 * 60
    lag_minutes: int = 5
    step_minutes: int = 60
    anchor_time: str | None = None
    start_time: str | None = None
    end_time: str | None = None


class SearchConfig(BaseModel):
    symbols: list[str]
    interval: str = "5m"
    min_quote_volume: float = 0.0
    blocked_utc_hours: list[int] = Field(default_factory=list)
    population_size: int = 8
    offspring_count: int = 4
    top_k: int = 5
    generations: int = 3
    novelty_threshold: float = 0.995
    n_splits: int = 5
    purge_window: int = 0
    embargo_window: int = 0
    persist: bool = True
    run_name: str = "auto_search"
    seeds: list[str] = Field(default_factory=list)
    seed_zoo_limit: int = 0
    feedback_seed_count: int = 0
    feedback_objective: str = "improve robustness and reduce turnover"
    carryover_top_k: int = 3
    llm_backend: str = "auto"
    llm_model: str | None = None


class RuntimeConfig(BaseModel):
    poll_interval_seconds: int = 60
    sleep_after_success_seconds: int = 0
    max_cycles: int | None = None
    continue_on_error: bool = True
    failure_backoff_seconds: int = 30
    max_failure_backoff_seconds: int = 900
    require_new_window: bool = True
    state_path: str = ""
    max_run_files: int | None = None
    max_zoo_entries: int | None = None


class CombinationConfig(BaseModel):
    enabled: bool = False
    method: str = "ic_weighted"
    max_factors: int = 10
    min_abs_ic: float = 0.01
    max_correlation: float = 0.70
    ic_lookback: int = 60
    zoo_limit: int = 50
    vol_target: float = 0.15
    max_drawdown: float = 0.15
    trailing_stop_pct: float = 0.05


class AutoSearchConfig(BaseModel):
    window: WindowConfig = Field(default_factory=WindowConfig)
    search: SearchConfig
    combination: CombinationConfig = Field(default_factory=CombinationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


def load_auto_search_config(path: str | Path) -> AutoSearchConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    search = raw.get("search")
    if isinstance(search, dict):
        if isinstance(search.get("symbols"), str):
            search["symbols"] = [s.strip().upper() for s in search["symbols"].split(",") if s.strip()]
        if isinstance(search.get("blocked_utc_hours"), str):
            search["blocked_utc_hours"] = [int(s.strip()) for s in search["blocked_utc_hours"].split(",") if s.strip()]
    return AutoSearchConfig.model_validate(raw)


def build_service_from_auto_search_config(config: AutoSearchConfig) -> AlphaService:
    return AlphaService(
        llm_backend_name=config.search.llm_backend,
        llm_model=config.search.llm_model,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_auto_search_loop(
    service: AlphaService,
    config: AutoSearchConfig,
    *,
    once: bool = False,
    max_cycles: int | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    _sp = config.runtime.state_path
    if not _sp:
        from src.config.paths import AUTO_SEARCH_STATE_PATH
        _sp = str(AUTO_SEARCH_STATE_PATH)
    state_path = Path(_sp)
    state = _load_state(state_path)
    resolved_max = max_cycles if max_cycles is not None else config.runtime.max_cycles
    effective_now = now_fn or (lambda: datetime.now(UTC))

    attempted = 0
    successful = int(state.get("successful_cycles", 0))
    failed = int(state.get("failed_cycles", 0))
    skipped = 0
    last_result: dict[str, Any] | None = None
    stopped_reason = "completed"

    logger.info(
        "alpha.auto_search start symbols={} interval={} once={} max_cycles={}",
        ",".join(config.search.symbols), config.search.interval, once, resolved_max,
    )

    while True:
        if resolved_max is not None and attempted >= resolved_max:
            stopped_reason = "max_cycles_reached"
            break

        try:
            window = _resolve_window(config, effective_now)
            last_end = (state.get("last_window") or {}).get("end")
            if config.runtime.require_new_window and last_end == window["end"]:
                skipped += 1
                if once:
                    stopped_reason = "window_not_advanced"
                    break
                sleep_fn(max(config.runtime.poll_interval_seconds, 1))
                continue

            attempted += 1

            # Wrap entire cycle in a single trace so seed generation
            # and search share the same trace_id in Langfuse.
            from .tracing import tracer
            with tracer.start_span(
                "auto_search_cycle", kind="search",
                cycle=attempted,
                window_start=window["start"],
                window_end=window["end"],
            ):
                seeds = _resolve_seeds(config, service, state)
                logger.info(
                    "alpha.auto_search cycle={} start={} end={} seeds={}",
                    attempted, window["start"], window["end"], len(seeds),
                )

                result = service.search_formulas_on_db(
                    symbols=list(config.search.symbols),
                    start_time=_parse_iso(window["start"]),
                    end_time=_parse_iso(window["end"]),
                    interval=config.search.interval,
                    min_quote_volume=config.search.min_quote_volume,
                    seeds=seeds,
                    population_size=config.search.population_size,
                    offspring_count=config.search.offspring_count,
                    top_k=config.search.top_k,
                    generations=config.search.generations,
                    run_name=config.search.run_name,
                    persist=config.search.persist,
                    novelty_threshold=config.search.novelty_threshold,
                    n_splits=config.search.n_splits,
                    purge_window=config.search.purge_window,
                    embargo_window=config.search.embargo_window,
                    blocked_utc_hours=config.search.blocked_utc_hours,
                )

                # Optional: combine factors
                combo_result = None
                if config.combination.enabled:
                    try:
                        c = config.combination
                        combo_result = service.combine_factors_from_db(
                            symbols=list(config.search.symbols),
                            start_time=_parse_iso(window["start"]),
                            end_time=_parse_iso(window["end"]),
                            interval=config.search.interval,
                            min_quote_volume=config.search.min_quote_volume,
                            blocked_utc_hours=config.search.blocked_utc_hours,
                            method=c.method,
                            max_factors=c.max_factors,
                            min_abs_ic=c.min_abs_ic,
                            max_correlation=c.max_correlation,
                            ic_lookback=c.ic_lookback,
                            zoo_limit=c.zoo_limit,
                            risk_config=RiskConfig(
                                vol_target=c.vol_target,
                                max_drawdown=c.max_drawdown,
                                trailing_stop_pct=c.trailing_stop_pct,
                            ),
                            summary_only=True,
                        )
                    except Exception as exc:
                        logger.warning("alpha.auto_search combination failed: {}", exc)

            # Update state
            successful += 1
            state.update({
                "successful_cycles": successful,
                "consecutive_failures": 0,
                "last_error": None,
                "last_window": window,
                "last_success_at": datetime.now(UTC).isoformat(),
                "last_run_id": result.get("persistence", {}).get("run_id"),
                "carryover_entries": _build_carryover(result, config.search.carryover_top_k),
                "updated_at": datetime.now(UTC).isoformat(),
            })
            if combo_result:
                state["last_combination"] = {
                    "metrics": combo_result.get("metrics"),
                    "factor_count": combo_result.get("combination", {}).get("factor_count"),
                }
            _save_state(state_path, state)
            _apply_retention(config, service)

            top = result.get("top_results") or []
            last_result = {
                "cycle": attempted,
                "window": window,
                "run_id": state["last_run_id"],
                "top_formula": top[0].get("formula") if top else None,
                "top_fitness": top[0].get("fitness") if top else None,
                "top_count": len(top),
            }
            logger.info(
                "alpha.auto_search cycle_done cycle={} run_id={} top={}",
                attempted, last_result["run_id"], last_result["top_count"],
            )

            if once:
                stopped_reason = "once_completed"
                break
            delay = config.runtime.sleep_after_success_seconds or config.runtime.poll_interval_seconds
            if delay > 0:
                sleep_fn(delay)

        except KeyboardInterrupt:
            stopped_reason = "interrupted"
            break
        except Exception as exc:
            failed += 1
            state["failed_cycles"] = failed
            state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
            state["last_error"] = str(exc)
            state["updated_at"] = datetime.now(UTC).isoformat()
            _save_state(state_path, state)
            logger.exception("alpha.auto_search cycle_failed: {}", exc)
            if once or not config.runtime.continue_on_error:
                raise
            backoff = min(
                config.runtime.failure_backoff_seconds * (2 ** max(state["consecutive_failures"] - 1, 0)),
                config.runtime.max_failure_backoff_seconds,
            )
            sleep_fn(max(backoff, 1))

    return {
        "attempted_cycles": attempted,
        "successful_cycles": successful,
        "failed_cycles": failed,
        "skipped_cycles": skipped,
        "stopped_reason": stopped_reason,
        "last_result": last_result,
        "last_run_id": state.get("last_run_id"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"successful_cycles": 0, "failed_cycles": 0, "consecutive_failures": 0}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_window(config: AutoSearchConfig, now_fn: Callable[[], datetime]) -> dict[str, str]:
    if config.window.start_time and config.window.end_time:
        return {
            "start": _parse_iso(config.window.start_time).isoformat(),
            "end": _parse_iso(config.window.end_time).isoformat(),
        }
    ref = _parse_iso(config.window.anchor_time) if config.window.anchor_time else now_fn().astimezone(UTC)
    end = ref - timedelta(minutes=config.window.lag_minutes)
    step = config.window.step_minutes * 60
    aligned_end = datetime.fromtimestamp((int(end.timestamp()) // step) * step, tz=UTC)
    start = aligned_end - timedelta(minutes=config.window.lookback_minutes)
    return {"start": start.isoformat(), "end": aligned_end.isoformat()}


def _resolve_seeds(
    config: AutoSearchConfig,
    service: AlphaService,
    state: dict[str, Any],
) -> list[str]:
    seeds: list[str] = []
    seen: set[str] = set()

    for f in config.search.seeds:
        if f and f not in seen:
            seeds.append(f)
            seen.add(f)

    # Carryover from previous cycle
    feedback_entries: list[dict[str, Any]] = []
    for entry in state.get("carryover_entries") or []:
        f = entry.get("formula")
        if f and f not in seen:
            seeds.append(f)
            seen.add(f)
            feedback_entries.append(entry)

    # Zoo top-k as seeds
    zoo = service.list_zoo(limit=max(config.search.seed_zoo_limit, config.search.feedback_seed_count, 0))
    for entry in zoo[:max(config.search.seed_zoo_limit, 0)]:
        f = entry.get("formula")
        if f and f not in seen:
            seeds.append(f)
            seen.add(f)
            feedback_entries.append(entry)

    # LLM feedback seeds
    if config.search.feedback_seed_count > 0:
        source = feedback_entries or zoo
        for f in _generate_feedback_seeds(service, source, config.search.feedback_seed_count, config.search.feedback_objective):
            if f not in seen:
                seeds.append(f)
                seen.add(f)

    return seeds


def _generate_feedback_seeds(
    service: AlphaService,
    entries: list[dict[str, Any]],
    count: int,
    objective: str,
) -> list[str]:
    from .search.evolution import BreedingSpec

    ranked = sorted(
        [e for e in entries if e.get("formula")],
        key=lambda e: float(e.get("fitness", 0) or 0),
        reverse=True,
    )
    if not ranked:
        return []
    parent_a = ranked[0]
    parent_b = ranked[1] if len(ranked) > 1 else None
    feedback = [{"formula": parent_a["formula"], "metrics": parent_a.get("metrics", {}), "rationale": "Carryover elite."}]
    if parent_b:
        feedback.append({"formula": parent_b["formula"], "metrics": parent_b.get("metrics", {}), "rationale": "Secondary elite."})
    formulas = service.llm_backend.generate_offspring(
        BreedingSpec(
            parent_a=parent_a["formula"],
            parent_b=parent_b["formula"] if parent_b else None,
            objective=objective,
            parent_feedback=feedback,
        ),
        count=count,
    )
    return [f for f in formulas if f][:count]


def _build_carryover(result: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    return [
        {"formula": r.get("formula"), "expr_hash": r.get("expr_hash"),
         "fitness": r.get("fitness"), "metrics": dict(r.get("metrics") or {})}
        for r in (result.get("top_results") or [])[:top_k]
    ]


def _apply_retention(config: AutoSearchConfig, service: AlphaService) -> None:
    if config.runtime.max_run_files is not None:
        service.persistence.prune_runs(config.runtime.max_run_files)
    if config.runtime.max_zoo_entries is not None:
        service.persistence.prune_zoo_entries(config.runtime.max_zoo_entries)
