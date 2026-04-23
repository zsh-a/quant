"""Weekly cross-model leaderboard Celery task.

Runs every Monday 03:00 UTC by default (see the beat schedule in
``src.tasks.celery_app``). Each run scores every analyst in the YAML
config against the golden dataset, writes an HTML report under the
config's ``output_dir``, and appends one parquet row per analyst to
the history log for trend queries.

LLM/VLM providers are resolved from model ids via
``_real_provider_factory``. The task never falls back to the mock
provider silently — callers that need offline runs should use
``scripts/brooks_leaderboard.py --mock-llm`` instead.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from src.brooks.eval.golden import GoldenDataset
from src.brooks.eval.leaderboard import (
    AnalystFactory,
    Leaderboard,
    LeaderboardConfig,
)
from src.tasks.celery_app import app

__all__ = ["run_weekly_leaderboard"]


DEFAULT_CONFIG_PATH = Path("config/brooks/leaderboard.yaml")


def _provider_factory():
    def factory(model: str):
        m = model.lower()
        if m.startswith("claude"):
            from src.alpha.llm.providers.anthropic import AnthropicProvider

            return AnthropicProvider(model=model)
        if m.startswith("gpt"):
            from src.alpha.llm.providers.openai import OpenAIProvider

            return OpenAIProvider(model=model)
        if m.startswith("gemini"):
            from src.alpha.llm.providers.gemini import GeminiProvider

            return GeminiProvider(model=model)
        raise ValueError(f"no provider mapping for model id: {model!r}")

    return factory


async def _execute(config_path: Path, output_override: Optional[Path]) -> Dict[str, Any]:
    cfg = LeaderboardConfig.load(config_path)
    dataset = GoldenDataset.load(cfg.dataset)
    if len(dataset) == 0:
        raise RuntimeError(f"leaderboard dataset is empty: {cfg.dataset}")

    factory = AnalystFactory(provider_factory=_provider_factory())
    board = Leaderboard(cfg, factory=factory)
    entries = await board.run_all(dataset)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_override or (cfg.output_dir / f"{ts}.html")
    board.to_html(out_path)
    history_path = board.persist()

    return {
        "status": "ok",
        "analysts": len(entries),
        "errors": [e.analyst_name for e in entries if e.error],
        "samples": entries[0].samples if entries else 0,
        "html": str(out_path),
        "history": str(history_path),
        "run_at": ts,
    }


@app.task(
    name="src.tasks.brooks_leaderboard_task.run_weekly_leaderboard",
    bind=True,
    soft_time_limit=3600,
    time_limit=5400,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
    max_retries=2,
)
def run_weekly_leaderboard(
    self,
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Entry point for the weekly cron.

    Parameters
    ----------
    config_path:
        Override for ``config/brooks/leaderboard.yaml``.
    output_path:
        Override for the HTML output path. When absent, defaults to
        ``<cfg.output_dir>/<timestamp>.html``.
    """
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    out_path = Path(output_path) if output_path else None
    logger.info("brooks_leaderboard: start config={} output={}", cfg_path, out_path)
    try:
        return asyncio.run(_execute(cfg_path, out_path))
    except Exception as exc:
        logger.exception("brooks_leaderboard: failed: {}", exc)
        raise
