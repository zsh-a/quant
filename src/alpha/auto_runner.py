from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from .risk import RiskConfig
from .service import AlphaService


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class AutoSearchWindowConfig(BaseModel):
    lookback_minutes: int = 24 * 60
    lag_minutes: int = 5
    step_minutes: int = 60
    anchor_time: str | None = None
    start_time: str | None = None
    end_time: str | None = None


class AutoSearchSearchConfig(BaseModel):
    provider: str
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
    run_name: str = "auto_search_db"
    seeds: list[str] = Field(default_factory=list)
    seed_zoo_limit: int = 0
    feedback_seed_count: int = 0
    feedback_objective: str = "improve robustness and reduce turnover"
    carryover_top_k: int = 3
    llm_backend: str = "auto"
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None


class AutoSearchRuntimeConfig(BaseModel):
    poll_interval_seconds: int = 60
    sleep_after_success_seconds: int = 0
    max_cycles: int | None = None
    continue_on_error: bool = True
    failure_backoff_seconds: int = 30
    max_failure_backoff_seconds: int = 900
    require_new_window: bool = True
    state_path: str = "data/alpha_lab/auto_search_state.json"
    max_run_files: int | None = None
    max_zoo_entries: int | None = None


class AutoSearchCombinationConfig(BaseModel):
    """Post-search factor combination settings."""

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
    window: AutoSearchWindowConfig = Field(default_factory=AutoSearchWindowConfig)
    search: AutoSearchSearchConfig
    combination: AutoSearchCombinationConfig = Field(default_factory=AutoSearchCombinationConfig)
    runtime: AutoSearchRuntimeConfig = Field(default_factory=AutoSearchRuntimeConfig)


def load_auto_search_config(path: str | Path) -> AutoSearchConfig:
    config_path = Path(path)
    raw_text = config_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw_text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Auto-search config must be a mapping: {config_path}")

    search_payload = payload.get("search")
    if isinstance(search_payload, dict):
        symbols = search_payload.get("symbols")
        if isinstance(symbols, str):
            search_payload["symbols"] = [item.strip().upper() for item in symbols.split(",") if item.strip()]
        blocked_hours = search_payload.get("blocked_utc_hours")
        if isinstance(blocked_hours, str):
            search_payload["blocked_utc_hours"] = [
                int(item.strip()) for item in blocked_hours.split(",") if item.strip()
            ]

    return AutoSearchConfig.model_validate(payload)


def build_service_from_auto_search_config(config: AutoSearchConfig) -> AlphaService:
    return AlphaService(
        llm_backend_name=config.search.llm_backend,
        llm_model=config.search.llm_model,
        llm_base_url=config.search.llm_base_url,
        llm_api_key=config.search.llm_api_key,
        program_cache_size=512,
    )


def run_auto_search_loop(
    service: AlphaService,
    config: AutoSearchConfig,
    *,
    once: bool = False,
    max_cycles: int | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    state_path = Path(config.runtime.state_path)
    state = _load_state(state_path)
    resolved_max_cycles = max_cycles if max_cycles is not None else config.runtime.max_cycles
    effective_now_fn = now_fn or (lambda: datetime.now(UTC))
    attempted_cycles = 0
    successful_cycles = int(state.get("successful_cycles", 0))
    failed_cycles = int(state.get("failed_cycles", 0))
    skipped_cycles = 0
    last_result: dict[str, Any] | None = None
    stopped_reason = "completed"

    logger.info(
        "alpha.auto_search start provider={} symbols={} interval={} once={} max_cycles={}",
        config.search.provider,
        ",".join(config.search.symbols),
        config.search.interval,
        once,
        resolved_max_cycles,
    )

    while True:
        if resolved_max_cycles is not None and attempted_cycles >= resolved_max_cycles:
            stopped_reason = "max_cycles_reached"
            break

        try:
            window = _resolve_window(config, effective_now_fn)
            last_window = state.get("last_window") or {}
            last_window_end = last_window.get("end")
            if config.runtime.require_new_window and last_window_end == window["end"]:
                skipped_cycles += 1
                logger.info(
                    "alpha.auto_search wait_for_new_window current_end={} poll_interval_seconds={}",
                    window["end"],
                    config.runtime.poll_interval_seconds,
                )
                if once:
                    stopped_reason = "window_not_advanced"
                    break
                sleep_fn(max(config.runtime.poll_interval_seconds, 1))
                continue

            attempted_cycles += 1
            cycle_started_at = datetime.now(UTC)
            seeds, feedback_entries = _resolve_cycle_seed_bundle(config, service, state)
            logger.info(
                "alpha.auto_search cycle_start cycle={} start={} end={} seed_count={} feedback_entry_count={} seed_zoo_limit={} feedback_seed_count={}",
                attempted_cycles,
                window["start"],
                window["end"],
                len(seeds),
                len(feedback_entries),
                config.search.seed_zoo_limit,
                config.search.feedback_seed_count,
            )
            result = service.search_formulas_on_db(
                provider=config.search.provider,
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
            # Post-search: combine factors from zoo if enabled
            combo_result: dict[str, Any] | None = None
            if config.combination.enabled:
                try:
                    combo_cfg = config.combination
                    risk_cfg = RiskConfig(
                        vol_target=combo_cfg.vol_target,
                        max_drawdown=combo_cfg.max_drawdown,
                        trailing_stop_pct=combo_cfg.trailing_stop_pct,
                    )
                    combo_result = service.combine_factors_from_db(
                        provider=config.search.provider,
                        symbols=list(config.search.symbols),
                        start_time=_parse_iso(window["start"]),
                        end_time=_parse_iso(window["end"]),
                        interval=config.search.interval,
                        min_quote_volume=config.search.min_quote_volume,
                        blocked_utc_hours=config.search.blocked_utc_hours,
                        method=combo_cfg.method,
                        max_factors=combo_cfg.max_factors,
                        min_abs_ic=combo_cfg.min_abs_ic,
                        max_correlation=combo_cfg.max_correlation,
                        ic_lookback=combo_cfg.ic_lookback,
                        zoo_limit=combo_cfg.zoo_limit,
                        risk_config=risk_cfg,
                        summary_only=True,
                    )
                    logger.info(
                        "alpha.auto_search combination_complete factors={} sharpe={:.4f} max_dd={:.4f}",
                        combo_result.get("combination", {}).get("factor_count", 0),
                        combo_result.get("metrics", {}).get("sharpe", 0),
                        combo_result.get("metrics", {}).get("max_drawdown", 0),
                    )
                except Exception as exc:
                    logger.warning("alpha.auto_search combination_failed: {}", exc)

            successful_cycles += 1
            state["successful_cycles"] = successful_cycles
            state["consecutive_failures"] = 0
            state["last_error"] = None
            state["last_window"] = window
            state["last_success_at"] = datetime.now(UTC).isoformat()
            state["last_run_id"] = result.get("persistence", {}).get("run_id")
            state["carryover_entries"] = _build_carryover_entries(result, config.search.carryover_top_k)
            if combo_result:
                state["last_combination"] = {
                    "metrics": combo_result.get("metrics"),
                    "factor_count": combo_result.get("combination", {}).get("factor_count"),
                }
            state["updated_at"] = datetime.now(UTC).isoformat()
            _save_state(state_path, state)
            retention = _apply_retention(config, service)

            top_results = result.get("top_results") or []
            last_result = {
                "cycle": attempted_cycles,
                "started_at": cycle_started_at.isoformat(),
                "completed_at": datetime.now(UTC).isoformat(),
                "window": window,
                "top_formula": top_results[0].get("formula") if top_results else None,
                "top_fitness": top_results[0].get("fitness") if top_results else None,
                "top_result_count": len(top_results),
                "run_id": result.get("persistence", {}).get("run_id"),
                "combination": combo_result.get("metrics") if combo_result else None,
                "timing": result.get("timing", {}),
                "validation": result.get("validation", {}),
                "retention": retention,
            }
            logger.info(
                "alpha.auto_search cycle_complete cycle={} run_id={} top_result_count={}",
                attempted_cycles,
                last_result["run_id"],
                last_result["top_result_count"],
            )

            if once:
                stopped_reason = "once_completed"
                break

            delay_seconds = config.runtime.sleep_after_success_seconds or config.runtime.poll_interval_seconds
            if delay_seconds > 0:
                sleep_fn(delay_seconds)
        except KeyboardInterrupt:
            stopped_reason = "keyboard_interrupt"
            logger.warning("alpha.auto_search interrupted by user")
            break
        except Exception as exc:
            failed_cycles += 1
            state["failed_cycles"] = failed_cycles
            state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
            state["last_error"] = str(exc)
            state["updated_at"] = datetime.now(UTC).isoformat()
            _save_state(state_path, state)
            logger.exception("alpha.auto_search cycle_failed error={}", exc)
            if once or not config.runtime.continue_on_error:
                raise
            backoff_seconds = min(
                config.runtime.failure_backoff_seconds * (2 ** max(state["consecutive_failures"] - 1, 0)),
                config.runtime.max_failure_backoff_seconds,
            )
            sleep_fn(max(backoff_seconds, 1))

    summary = {
        "state_path": str(state_path),
        "attempted_cycles": attempted_cycles,
        "successful_cycles": successful_cycles,
        "failed_cycles": failed_cycles,
        "skipped_cycles": skipped_cycles,
        "stopped_reason": stopped_reason,
        "last_result": last_result,
        "last_window": state.get("last_window"),
        "last_run_id": state.get("last_run_id"),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    logger.info(
        "alpha.auto_search stop attempted_cycles={} successful_cycles={} failed_cycles={} stopped_reason={}",
        attempted_cycles,
        successful_cycles,
        failed_cycles,
        stopped_reason,
    )
    return summary


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "successful_cycles": 0,
            "failed_cycles": 0,
            "consecutive_failures": 0,
            "last_error": None,
            "last_window": None,
            "last_run_id": None,
            "updated_at": None,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_window(
    config: AutoSearchConfig,
    now_fn: Callable[[], datetime],
) -> dict[str, str]:
    # Fixed window: explicit start_time / end_time
    if config.window.start_time and config.window.end_time:
        start_time = _parse_iso(config.window.start_time)
        end_time = _parse_iso(config.window.end_time)
        if start_time >= end_time:
            raise ValueError(f"start_time ({config.window.start_time}) must be before end_time ({config.window.end_time})")
        return {"start": start_time.isoformat(), "end": end_time.isoformat()}

    # Rolling window: anchor/now - lag - lookback
    if config.window.step_minutes <= 0:
        raise ValueError("window.step_minutes must be > 0")
    if config.window.lookback_minutes <= 0:
        raise ValueError("window.lookback_minutes must be > 0")

    reference_time = _parse_iso(config.window.anchor_time) if config.window.anchor_time else now_fn().astimezone(UTC)
    effective_end = reference_time - timedelta(minutes=config.window.lag_minutes)
    aligned_end = _floor_time(effective_end, config.window.step_minutes)
    start_time = aligned_end - timedelta(minutes=config.window.lookback_minutes)
    if start_time >= aligned_end:
        raise ValueError("Resolved auto-search window is empty")
    return {
        "start": start_time.isoformat(),
        "end": aligned_end.isoformat(),
    }


def _floor_time(value: datetime, step_minutes: int) -> datetime:
    step_seconds = step_minutes * 60
    timestamp = int(value.astimezone(UTC).timestamp())
    return datetime.fromtimestamp((timestamp // step_seconds) * step_seconds, tz=UTC)


def _generate_feedback_seeds(
    service: AlphaService,
    entries: list[dict[str, Any]],
    count: int,
    objective: str,
) -> list[str]:
    """Generate feedback seed formulas via the search engine's LLM backend."""
    from .evolution import BreedingSpec

    if count <= 0 or not entries:
        return []
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
    formulas = service.search_engine.llm_backend.generate_offspring(
        BreedingSpec(
            parent_a=parent_a["formula"],
            parent_b=parent_b["formula"] if parent_b else None,
            objective=objective,
            parent_feedback=feedback,
        ),
        count=count,
    )
    return [f for f in formulas if f][:count]


def _resolve_cycle_seed_bundle(
    config: AutoSearchConfig,
    service: AlphaService,
    state: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    seeds: list[str] = []
    seen: set[str] = set()
    feedback_entries: list[dict[str, Any]] = []
    for formula in config.search.seeds:
        if formula and formula not in seen:
            seeds.append(formula)
            seen.add(formula)
    carryover_entries = state.get("carryover_entries") or []
    for entry in carryover_entries:
        formula = entry.get("formula")
        if not formula or formula in seen:
            continue
        seeds.append(formula)
        seen.add(formula)
        feedback_entries.append(entry)

    zoo_entries = service.list_zoo(limit=max(config.search.seed_zoo_limit, config.search.feedback_seed_count, 0))
    for entry in zoo_entries[: max(config.search.seed_zoo_limit, 0)]:
        formula = entry.get("formula")
        if not formula or formula in seen:
            continue
        seeds.append(formula)
        seen.add(formula)
        feedback_entries.append(entry)

    source_entries = feedback_entries or zoo_entries or carryover_entries
    if config.search.feedback_seed_count > 0:
        for formula in _generate_feedback_seeds(
            service, source_entries,
            count=config.search.feedback_seed_count,
            objective=config.search.feedback_objective,
        ):
            if formula in seen:
                continue
            seeds.append(formula)
            seen.add(formula)
    return seeds, list(source_entries)


def _build_carryover_entries(result: dict[str, Any], carryover_top_k: int) -> list[dict[str, Any]]:
    if carryover_top_k <= 0:
        return []
    carryover_entries: list[dict[str, Any]] = []
    for item in (result.get("top_results") or [])[:carryover_top_k]:
        carryover_entries.append(
            {
                "formula": item.get("formula"),
                "expr_hash": item.get("expr_hash"),
                "fitness": item.get("fitness"),
                "metrics": dict(item.get("metrics") or {}),
            }
        )
    return carryover_entries


def _apply_retention(config: AutoSearchConfig, service: AlphaService) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if config.runtime.max_run_files is not None:
        summary["runs"] = service.persistence.prune_runs(config.runtime.max_run_files)
    if config.runtime.max_zoo_entries is not None:
        summary["zoo"] = service.persistence.prune_zoo_entries(config.runtime.max_zoo_entries)
    return summary
