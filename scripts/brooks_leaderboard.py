#!/usr/bin/env python3
"""CLI entry for the Brooks cross-analyst leaderboard.

Usage:

    python scripts/brooks_leaderboard.py --config config/brooks/leaderboard.yaml
    python scripts/brooks_leaderboard.py --config ... --mock-llm       # CI / offline

Writes ``<output_dir>/<timestamp>.html`` and appends one parquet row per
analyst to ``<history_path>``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow "python scripts/brooks_leaderboard.py" from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.brooks.eval.golden import GoldenDataset  # noqa: E402
from src.brooks.eval.leaderboard import (  # noqa: E402
    AnalystFactory,
    Leaderboard,
    LeaderboardConfig,
)

logger = logging.getLogger("brooks.leaderboard.cli")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--config", required=True, type=Path, help="Path to leaderboard.yaml")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output HTML path. Defaults to <output_dir>/<timestamp>.html.",
    )
    p.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use a deterministic in-process MockProvider for every LLM/VLM analyst "
        "(no network, no API keys). Required in CI and for smoke tests.",
    )
    p.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not append to the history parquet log.",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Override dataset path from the config file.",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def _mock_provider_factory():
    """Return a callable ``model_id -> MockProvider`` that synthesises a
    plausible ``LLMSignalBatch`` response.

    The signal echoes the most common (pattern, side) in the golden set
    — "h2" / "long" — with stable cost metadata, enough to drive the
    eval pipeline end-to-end without calling any real LLM.
    """
    from src.alpha.llm.provider import Response, Usage
    from src.brooks.analyst.llm import LLMSignalBatch
    from src.brooks.schema import Signal

    class _MockProvider:
        def __init__(self, model: str) -> None:
            self.name = f"mock:{model}"
            self._model = model

        async def complete(self, messages, schema, cache=None, seed=None, max_tokens=4096, temperature=0.0):
            parsed = schema(
                reasoning=f"mock-{self._model}",
                signals=[
                    Signal(
                        pattern="h2",
                        side="long",
                        signal_bar_idx=0,
                        entry_px=100.0,
                        stop_px=99.0,
                        target_px=102.0,
                        probability=0.55,
                        quality=0.6,
                        source=f"llm:{self._model}",
                    )
                ],
            ) if schema is LLMSignalBatch else schema()
            return Response(
                parsed=parsed,
                raw={"mock": True},
                usage=Usage(input_tokens=100, output_tokens=40),
                latency_ms=8.0,
                model=self._model,
                cache_hit=False,
            )

    return lambda model_id: _MockProvider(model_id)


def _real_provider_factory():
    """Pick a real provider by model-id prefix (claude → Anthropic, gpt → OpenAI,
    gemini → Gemini). Raises if the matching API key env var is not set."""

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


async def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = LeaderboardConfig.load(args.config)
    if args.dataset is not None:
        cfg.dataset = args.dataset

    if not cfg.dataset.exists():
        logger.error("dataset not found: %s", cfg.dataset)
        return 2

    dataset = GoldenDataset.load(cfg.dataset)
    logger.info("loaded %d golden samples from %s", len(dataset), cfg.dataset)
    if len(dataset) == 0:
        logger.error("dataset is empty; nothing to evaluate")
        return 2

    provider_factory = _mock_provider_factory() if args.mock_llm else _real_provider_factory()
    factory = AnalystFactory(provider_factory=provider_factory)
    board = Leaderboard(cfg, factory=factory)

    entries = await board.run_all(dataset)
    logger.info("scored %d analyst(s)", len(entries))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output if args.output is not None else cfg.output_dir / f"{ts}.html"
    board.to_html(out_path)
    logger.info("wrote leaderboard HTML → %s", out_path)

    if not args.no_persist:
        path = board.persist()
        logger.info("appended history → %s", path)

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
