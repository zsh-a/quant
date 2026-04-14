"""
自动化因子工厂 — 全自动 搜索 → 评估 → 入库 → 组合 流水线。

Usage:
    factory = FactorFactory()
    result = factory.run_pipeline(
        markets=["crypto"],
        symbols=["BTCUSDT", "ETHUSDT"],
        generations=5,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from loguru import logger


@dataclass
class FactoryConfig:
    """因子工厂配置。"""
    markets: list[str] = field(default_factory=lambda: ["crypto"])
    symbols_per_market: dict[str, list[str]] = field(default_factory=dict)
    generations: int = 5
    top_k: int = 30
    min_abs_ic: float = 0.02
    max_factors_combine: int = 10
    combine_method: str = "ic_weighted"
    zoo_retention_days: int = 90
    # 搜索参数透传
    strategy: str = ""
    n_splits: int = 5


@dataclass
class FactoryResult:
    """工厂流水线执行结果。"""
    search_results: dict[str, Any]  # market → search result
    zoo_count: int
    combination_result: dict[str, Any] | None
    decaying_factors: list[str]  # 衰减因子公式列表
    timing: dict[str, float]
    started_at: str
    completed_at: str


class FactorFactory:
    """全自动因子搜索 → 评估 → 入库 → 组合 → 健康检查。"""

    def __init__(self, **service_kwargs):
        self._service_kwargs = service_kwargs

    def _get_service(self, market: str):
        from src.alpha import AlphaService
        return AlphaService(market=market, **self._service_kwargs)

    def run_pipeline(self, config: FactoryConfig | None = None) -> FactoryResult:
        """执行完整流水线。"""
        if config is None:
            config = FactoryConfig()

        started_at = datetime.now(timezone.utc).isoformat()
        t0 = perf_counter()
        timing: dict[str, float] = {}
        search_results: dict[str, Any] = {}

        # --- Step 1: 多市场搜索 ---
        for market in config.markets:
            logger.info("factory.search market={} generations={}", market, config.generations)
            t_search = perf_counter()
            svc = self._get_service(market)
            symbols = config.symbols_per_market.get(market, [])

            search_kwargs: dict[str, Any] = {
                "generations": config.generations,
                "top_k": config.top_k,
                "persist": True,
            }
            if symbols:
                search_kwargs["symbols"] = symbols
            if config.strategy:
                search_kwargs["strategy"] = config.strategy
            if config.n_splits:
                search_kwargs["n_splits"] = config.n_splits

            try:
                result = svc.search_formulas_on_db(**search_kwargs)
                search_results[market] = {
                    "status": "completed",
                    "archive_size": len(result.get("archive", [])),
                    "total_evaluated": result.get("total_evaluated", 0),
                }
            except Exception as exc:
                logger.error("factory.search failed market={}: {}", market, exc)
                search_results[market] = {"status": "failed", "error": str(exc)}

            timing[f"search_{market}"] = perf_counter() - t_search

        # --- Step 2: Zoo 健康检查 ---
        t_health = perf_counter()
        primary_market = config.markets[0]
        primary_svc = self._get_service(primary_market)
        zoo = primary_svc.list_zoo(limit=500)
        zoo_count = len(zoo)

        decaying_factors: list[str] = []
        if hasattr(primary_svc, "search_engine") and primary_svc.search_engine:
            ctx = getattr(primary_svc.search_engine, "ctx", None)
            if ctx and hasattr(ctx, "factor_catalog") and ctx.factor_catalog:
                for entry in ctx.factor_catalog.decaying_factors():
                    decaying_factors.append(entry.formula)

        timing["health_check"] = perf_counter() - t_health

        # --- Step 3: 因子组合 ---
        combination_result = None
        if zoo_count >= 3:
            t_combine = perf_counter()
            try:
                symbols = config.symbols_per_market.get(primary_market, [])
                combine_kwargs: dict[str, Any] = {
                    "max_factors": config.max_factors_combine,
                    "min_abs_ic": config.min_abs_ic,
                    "method": config.combine_method,
                }
                if symbols:
                    combine_kwargs["symbols"] = symbols

                combo = primary_svc.combine_factors_from_db(**combine_kwargs)
                combination_result = {
                    "status": "completed",
                    "n_factors_selected": len(combo.get("selected_factors", [])),
                    "method": combo.get("method"),
                    "timing": combo.get("timing"),
                }
            except Exception as exc:
                logger.error("factory.combine failed: {}", exc)
                combination_result = {"status": "failed", "error": str(exc)}
            timing["combine"] = perf_counter() - t_combine

        timing["total"] = perf_counter() - t0
        completed_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "factory.pipeline completed markets={} zoo={} decaying={} total={:.1f}s",
            config.markets, zoo_count, len(decaying_factors), timing["total"],
        )

        return FactoryResult(
            search_results=search_results,
            zoo_count=zoo_count,
            combination_result=combination_result,
            decaying_factors=decaying_factors,
            timing=timing,
            started_at=started_at,
            completed_at=completed_at,
        )
