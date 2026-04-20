from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from time import perf_counter
from typing import Any

import numpy as np
from loguru import logger

from .core.compiler import BytecodeProgram, FormulaCompiler
from .core.dataset import AlphaDataset, AShareDailyDatasetLoader, CryptoMinuteDatasetLoader
from .core.dsl import TensorSchema
from .core.market import MarketType, get_market_profile
from .core.operators import OperatorRegistry
from .core.vm import StackVM, TensorStore
from .eval.validation import CPCVValidator, ValidationFold
from .infra.persistence import AlphaPersistence
from .knowledge.features import FeatureKitchen
from .knowledge.memory import StrategyMemory
from .knowledge.themes import FinancialKnowledgeBase
from .risk.combination import FactorCombiner
from .risk.models import (
    CostModel,
    ExecutionSimulator,
    MarketContext,
    PortfolioManager,
    RiskConfig,
    RuleOverlay,
    SignalTransformer,
)
from .search.evolution import EvalResult, Individual
from .search.orchestrator import SearchOrchestrator

try:
    import torch as _torch

except Exception:  # pragma: no cover
    _torch = None

DEFAULT_DB_SEEDS = [
    "cs_rank(ts_mean(close, 5) - close)",
    "cs_rank(ts_std(close, 10))",
    "cs_rank(volatility_n(close, 20))",
]


class AlphaService:
    def __init__(
        self,
        market: str = "crypto",
        schema: TensorSchema | None = None,
        llm_backend: Any | None = None,
        llm_backend_name: str = "auto",
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        program_cache_size: int = 512,
        strategy: str = "evolution",
        neural_sample_batch: int = 512,
        mcts_refinement_frequency: int = 3,
        strategy_memory_path: str | None = None,
        enum_max: int = 500,
        enum_top_k: int = 30,
    ):
        self.market_profile = get_market_profile(market)
        self.registry = OperatorRegistry()
        self.schema = schema or self.market_profile.schema
        self.compiler = FormulaCompiler(self.registry, field_aliases=self.market_profile.field_aliases)
        self.vm = StackVM()
        self.vm.enable_persistent_cache(max_entries=1024)

        # --- Enhanced modules ---
        self.knowledge_base = FinancialKnowledgeBase()
        self.feature_kitchen = FeatureKitchen(self.schema)
        if strategy_memory_path is None:
            from src.config.paths import STRATEGY_MEMORY_PATH

            strategy_memory_path = str(STRATEGY_MEMORY_PATH)
        self.strategy_memory = StrategyMemory(
            persistence_path=strategy_memory_path,
            all_theme_ids=self.knowledge_base.get_all_theme_ids(),
        )
        # Load persisted strategy memory from previous sessions
        self.strategy_memory.load()

        # --- Build LLM backend ---
        from .llm.backends import build_default_llm_backend

        resolved_llm = llm_backend or build_default_llm_backend(
            registry=self.registry,
            schema=self.schema,
            backend_name=llm_backend_name,
            model_name=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            strategy_memory=self.strategy_memory,
            knowledge_base=self.knowledge_base,
            feature_kitchen=self.feature_kitchen,
            market_profile=self.market_profile,
        )
        self.llm_backend = resolved_llm

        # --- Assemble strategies by mode ---
        strategies: list = self._build_strategies(
            strategy,
            resolved_llm,
            neural_sample_batch,
            mcts_refinement_frequency,
            enum_max=enum_max,
            enum_top_k=enum_top_k,
        )

        # --- Checkpoint manager ---
        from .search.checkpoint import CheckpointManager

        self.checkpoint_manager = CheckpointManager()

        # --- Search orchestrator ---
        self.search_engine = SearchOrchestrator(
            strategies=strategies,
            compiler=self.compiler,
            registry=self.registry,
            schema=self.schema,
            strategy_memory=self.strategy_memory,
            knowledge_base=self.knowledge_base,
            feature_kitchen=self.feature_kitchen,
            checkpoint_manager=self.checkpoint_manager,
        )

        self.signal_transformer = SignalTransformer()
        self.rule_overlay = RuleOverlay()
        self.portfolio_manager = PortfolioManager()
        self.execution = ExecutionSimulator()
        self.cost_model = CostModel(**(self.market_profile.cost_model_kwargs or {}))
        self.dataset_loader = self._create_loader()
        self.persistence = AlphaPersistence()
        self.combiner = FactorCombiner(compiler=self.compiler, vm=self.vm, schema=self.schema)
        self.validator = CPCVValidator()
        self._program_cache_size = max(int(program_cache_size), 0)
        self._program_cache: OrderedDict[str, BytecodeProgram] = OrderedDict()
        self._program_cache_lock = threading.Lock()
        self._complexity_cache: dict[str, dict[str, float]] = {}

        logger.info(
            "AlphaService init market={} vm_backend={} device={} triton={}",
            market,
            self.vm.backend,
            self.vm.device if self.vm.device is not None else "numpy",
            getattr(self.vm, "use_triton", False),
        )

    # ------------------------------------------------------------------
    # Loader factory
    # ------------------------------------------------------------------

    _LOADER_MAP = {
        MarketType.CRYPTO: CryptoMinuteDatasetLoader,
        MarketType.A_SHARE: AShareDailyDatasetLoader,
    }

    def _create_loader(self):
        cls = self._LOADER_MAP.get(self.market_profile.market_type)
        if cls is None:
            raise ValueError(f"No dataset loader for market: {self.market_profile.market_type}")
        return cls()

    # ------------------------------------------------------------------
    # Strategy assembly
    # ------------------------------------------------------------------

    def get_strategy_modes_info(self) -> list[dict[str, Any]]:
        """Return strategy modes for frontend StrategyModeInfo selector."""
        from .strategies.registry import get_all_modes

        return [
            {
                "mode": mode.name,
                "label": mode.label,
                "brief": mode.brief,
                "detail": mode.detail,
                "strategies": list(mode.strategies),
                "params": list(mode.params),
            }
            for mode in get_all_modes().values()
        ]

    def _build_strategies(
        self,
        strategy: str,
        llm_backend: Any,
        neural_sample_batch: int,
        mcts_frequency: int,
        enum_max: int = 500,
        enum_top_k: int = 30,
    ) -> list:
        """Build strategy list from mode name.

        ``strategy`` is a mode name (e.g. ``"evolution"``, ``"neural"``).
        Falls back to evolution if unknown.
        """
        from .strategies.registry import StrategyInfra, build_strategies, get_mode

        mode = get_mode(strategy or "evolution") or get_mode("evolution")
        names = set(mode.strategies)

        infra = StrategyInfra(
            compiler=self.compiler,
            vm=self.vm,
            schema=self.schema,
            registry=self.registry,
            llm_backend=llm_backend,
            neural_sample_batch=neural_sample_batch,
            mcts_frequency=mcts_frequency,
            enum_max=enum_max,
            enum_top_k=enum_top_k,
        )
        return build_strategies(names, infra)

    def list_operators(self) -> list[dict[str, Any]]:
        return [asdict(spec) for spec in self.registry.list_operators()]

    def validate_formula(self, formula: str) -> dict[str, Any]:
        report = self._validate_with_aliases(formula)
        return asdict(report)

    def compile_formula(self, formula: str) -> dict[str, Any]:
        program = self._compile_cached(formula)
        report = self._validate_with_aliases(formula)
        return {
            "validation": asdict(report),
            "program": program.to_dict(),
        }

    def _validate_with_aliases(self, formula: str):
        """Validate using the market-aware compiler (with correct field aliases)."""
        import hashlib

        from .core.dsl import FormulaParser, ValidationReport, normalize_formula

        parser = FormulaParser(self.registry)
        try:
            parsed = parser.parse(formula)
            typed = self.compiler.checker.infer(parsed, self.schema)
            normalized = normalize_formula(parsed)
            ast_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            return ValidationReport(
                ok=True, normalized_formula=normalized, errors=[], ast_hash=ast_hash, ast_tree=typed.to_dict()
            )
        except ValueError as exc:
            return ValidationReport(ok=False, normalized_formula=formula.strip(), errors=[str(exc)])

    def evaluate_formula(
        self,
        formula: str,
        fields: dict[str, list[list[float]]],
        liquidity_mask: list[list[bool]] | None = None,
        session_mask: list[list[bool]] | None = None,
    ) -> dict[str, Any]:
        store = TensorStore({name: np.asarray(values, dtype=float) for name, values in fields.items()})
        program = self._compile_cached(formula)
        alpha = self.vm.run(program, store)

        market_ctx = MarketContext(
            liquidity_mask=np.asarray(liquidity_mask, dtype=bool) if liquidity_mask is not None else None,
            session_mask=np.asarray(session_mask, dtype=bool) if session_mask is not None else None,
        )
        target_weights = self.signal_transformer.to_target_weights(alpha, market_ctx)
        wrapped_weights = self.rule_overlay.apply(target_weights, market_ctx)
        prices, funding = self._sim_args(store)
        result = self.execution.simulate(wrapped_weights, prices, self.cost_model, funding_rate=funding)
        return {
            "program": program.to_dict(),
            "alpha": self._to_serializable_list(alpha),
            "weights": self._to_serializable_list(wrapped_weights),
            "metrics": result.summary(),
        }

    def _prepare_store(self, dataset: AlphaDataset) -> TensorStore:
        """Create a TensorStore, converting to torch if VM uses GPU backend."""
        store = TensorStore(dataset.fields)
        if self.vm.backend == "torch" and self.vm.device is not None:
            store = self.vm._prepare_store(store)
        return store

    def _sim_args(self, store: TensorStore) -> tuple[dict[str, Any], Any]:
        """Build (prices, funding_rate) for ExecutionSimulator based on market schema."""
        prices: dict[str, Any] = {"close": store.get_field("close")}
        if "bid_ask_spread" in self.schema.fields:
            prices["bid_ask_spread"] = store.get_field("bid_ask_spread")
        funding = store.get_field("funding_rate") if "funding_rate" in self.schema.fields else None
        return prices, funding

    def evaluate_formula_from_dataset(
        self,
        formula: str,
        dataset: AlphaDataset,
        summary_only: bool = False,
    ) -> dict[str, Any]:
        store = self._prepare_store(dataset)
        program = self._compile_cached(formula)
        alpha = self.vm.run(program, store)
        evaluation = self._build_dataset_evaluation(program, alpha, dataset, store)
        result = {
            "dataset": self._dataset_summary(dataset),
            **evaluation,
        }
        return self._summarize_evaluation_payload(result) if summary_only else result

    def evaluate_formulas_from_dataset(
        self,
        formulas: list[str],
        dataset: AlphaDataset,
        summary_only: bool = False,
    ) -> dict[str, dict[str, Any]]:
        store = self._prepare_store(dataset)
        programs = [self._compile_cached(formula) for formula in formulas]
        alphas = self.vm.run_batch(programs, store)
        results: dict[str, dict[str, Any]] = {}
        for formula, program, alpha in zip(formulas, programs, alphas):
            evaluation = self._build_dataset_evaluation(program, alpha, dataset, store)
            results[formula] = {
                "dataset": self._dataset_summary(dataset),
                **evaluation,
            }
        if summary_only:
            return {formula: self._summarize_evaluation_payload(payload) for formula, payload in results.items()}
        return results

    def evaluate_formula_from_db(
        self,
        formula: str,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = "5m",
        min_quote_volume: float = 0.0,
        blocked_utc_hours: list[int] | None = None,
        summary_only: bool = False,
        **loader_kwargs,
    ) -> dict[str, Any]:
        t0 = perf_counter()
        dataset = self.dataset_loader.load(
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            min_quote_volume=min_quote_volume,
            blocked_utc_hours=blocked_utc_hours,
            **loader_kwargs,
        )
        t_load = perf_counter()
        result = self.evaluate_formula_from_dataset(formula=formula, dataset=dataset, summary_only=summary_only)
        t_eval = perf_counter()
        logger.info(
            "alpha.evaluate_db load={:.1f}s eval={:.1f}s total={:.1f}s backend={} device={}",
            t_load - t0,
            t_eval - t_load,
            t_eval - t0,
            self.vm.backend,
            self.vm.device or "numpy",
        )
        return result

    def evaluate_formulas_from_db(
        self,
        formulas: list[str],
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = "5m",
        min_quote_volume: float = 0.0,
        blocked_utc_hours: list[int] | None = None,
        summary_only: bool = False,
        **loader_kwargs,
    ) -> dict[str, dict[str, Any]]:
        dataset = self.dataset_loader.load(
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            min_quote_volume=min_quote_volume,
            blocked_utc_hours=blocked_utc_hours,
            **loader_kwargs,
        )
        return self.evaluate_formulas_from_dataset(formulas=formulas, dataset=dataset, summary_only=summary_only)

    def benchmark_vm(
        self,
        formulas: list[str],
        rows: int = 2048,
        cols: int = 16,
        repeat: int = 5,
    ) -> dict[str, Any]:
        formulas = formulas or DEFAULT_DB_SEEDS
        rng = np.random.default_rng(7)
        base_close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, size=(rows, cols)), axis=0)
        base_volume = np.abs(rng.normal(1000.0, 150.0, size=(rows, cols))) + 1.0
        base_turnover = base_close * base_volume
        base_oi = np.abs(rng.normal(1_000_000.0, 10_000.0, size=(rows, cols)))
        fields = {
            "open": base_close - rng.normal(0.0, 0.1, size=(rows, cols)),
            "high": base_close + np.abs(rng.normal(0.25, 0.1, size=(rows, cols))),
            "low": base_close - np.abs(rng.normal(0.25, 0.1, size=(rows, cols))),
            "close": base_close,
            "volume": base_volume,
            "turnover": base_turnover,
            "vwap": base_close + rng.normal(0.0, 0.03, size=(rows, cols)),
            "bid_ask_spread": np.abs(rng.normal(0.5, 0.05, size=(rows, cols))),
            # Volume details
            "trade_count": np.abs(rng.normal(500.0, 100.0, size=(rows, cols))),
            "taker_buy_volume": base_volume * np.abs(rng.normal(0.5, 0.1, size=(rows, cols))),
            "taker_buy_quote_volume": base_turnover * np.abs(rng.normal(0.5, 0.1, size=(rows, cols))),
            # Mark price
            "mark_open": base_close - rng.normal(0.0, 0.05, size=(rows, cols)),
            "mark_high": base_close + np.abs(rng.normal(0.2, 0.08, size=(rows, cols))),
            "mark_low": base_close - np.abs(rng.normal(0.2, 0.08, size=(rows, cols))),
            "mark_close": base_close + rng.normal(0.0, 0.02, size=(rows, cols)),
            # Premium index (basis)
            "premium_open": rng.normal(0.0, 0.001, size=(rows, cols)),
            "premium_high": np.abs(rng.normal(0.001, 0.0005, size=(rows, cols))),
            "premium_low": -np.abs(rng.normal(0.001, 0.0005, size=(rows, cols))),
            "premium_close": rng.normal(0.0, 0.0008, size=(rows, cols)),
            # Market metrics
            "funding_rate": rng.normal(0.0001, 0.0001, size=(rows, cols)),
            "open_interest": base_oi,
            "open_interest_value": base_oi * base_close,
            "top_trader_long_short_ratio": np.abs(rng.normal(1.0, 0.3, size=(rows, cols))),
            "top_trader_long_short_position_ratio": np.abs(rng.normal(1.0, 0.3, size=(rows, cols))),
            "long_short_ratio": np.abs(rng.normal(1.0, 0.2, size=(rows, cols))),
            "taker_long_short_vol_ratio": np.abs(rng.normal(1.0, 0.2, size=(rows, cols))),
        }
        store = TensorStore({name: np.asarray(values, dtype=float) for name, values in fields.items()})
        programs = [self._compile_cached(formula) for formula in formulas]

        warmup = self.vm.run_batch(programs, store)
        _ = [output.shape for output in warmup]

        batch_timings = []
        serial_timings = []
        for _ in range(max(repeat, 1)):
            start = perf_counter()
            outputs = self.vm.run_batch(programs, store)
            batch_timings.append(perf_counter() - start)
            _ = [output.shape for output in outputs]

            start = perf_counter()
            outputs = [self.vm.run(program, store) for program in programs]
            serial_timings.append(perf_counter() - start)
            _ = [output.shape for output in outputs]

        batch_avg_seconds = float(np.mean(batch_timings)) if batch_timings else 0.0
        serial_avg_seconds = float(np.mean(serial_timings)) if serial_timings else 0.0
        batch_throughput = (len(formulas) * rows * cols) / max(batch_avg_seconds, 1e-12)
        serial_throughput = (len(formulas) * rows * cols) / max(serial_avg_seconds, 1e-12)
        return {
            "backend": self.vm.backend,
            "device": str(self.vm.device) if self.vm.device is not None else "numpy",
            "rows": rows,
            "cols": cols,
            "formula_count": len(formulas),
            "repeat": repeat,
            "batch_avg_seconds": batch_avg_seconds,
            "batch_best_seconds": float(np.min(batch_timings)) if batch_timings else 0.0,
            "batch_throughput_cells_per_second": batch_throughput,
            "serial_avg_seconds": serial_avg_seconds,
            "serial_best_seconds": float(np.min(serial_timings)) if serial_timings else 0.0,
            "serial_throughput_cells_per_second": serial_throughput,
            "speedup_vs_serial": serial_avg_seconds / max(batch_avg_seconds, 1e-12),
            "programs": [program.to_dict() for program in programs],
        }

    def benchmark_db(
        self,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = "5m",
        min_quote_volume: float = 0.0,
        blocked_utc_hours: list[int] | None = None,
        formulas: list[str] | None = None,
        repeat: int = 3,
    ) -> dict[str, Any]:
        formulas = formulas or DEFAULT_DB_SEEDS

        load_start = perf_counter()
        dataset = self.dataset_loader.load(
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            min_quote_volume=min_quote_volume,
            blocked_utc_hours=blocked_utc_hours,
        )
        load_seconds = perf_counter() - load_start

        warmup = self.evaluate_formulas_from_dataset(formulas=formulas, dataset=dataset, summary_only=True)
        _ = list(warmup.keys())

        batch_timings = []
        serial_timings = []
        last_batch_result: dict[str, Any] = {}
        for _ in range(max(repeat, 1)):
            start = perf_counter()
            last_batch_result = self.evaluate_formulas_from_dataset(
                formulas=formulas,
                dataset=dataset,
                summary_only=True,
            )
            batch_timings.append(perf_counter() - start)

            start = perf_counter()
            for formula in formulas:
                self.evaluate_formula_from_dataset(formula=formula, dataset=dataset, summary_only=True)
            serial_timings.append(perf_counter() - start)

        batch_avg_seconds = float(np.mean(batch_timings)) if batch_timings else 0.0
        serial_avg_seconds = float(np.mean(serial_timings)) if serial_timings else 0.0
        rows, cols = dataset.shape()
        return {
            "backend": self.vm.backend,
            "device": str(self.vm.device) if self.vm.device is not None else "numpy",
            "dataset": self._dataset_summary(dataset),
            "load_seconds": load_seconds,
            "formula_count": len(formulas),
            "repeat": repeat,
            "batch_avg_seconds": batch_avg_seconds,
            "batch_best_seconds": float(np.min(batch_timings)) if batch_timings else 0.0,
            "batch_throughput_cells_per_second": (len(formulas) * rows * cols) / max(batch_avg_seconds, 1e-12),
            "serial_avg_seconds": serial_avg_seconds,
            "serial_best_seconds": float(np.min(serial_timings)) if serial_timings else 0.0,
            "serial_throughput_cells_per_second": (len(formulas) * rows * cols) / max(serial_avg_seconds, 1e-12),
            "speedup_vs_serial": serial_avg_seconds / max(batch_avg_seconds, 1e-12),
            "formula_summaries": last_batch_result,
        }

    def search_formulas_on_db(
        self,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = "5m",
        min_quote_volume: float = 0.0,
        seeds: list[str] | None = None,
        population_size: int = 4,
        offspring_count: int = 2,
        top_k: int = 3,
        generations: int = 2,
        run_name: str | None = None,
        persist: bool = True,
        novelty_threshold: float = 0.995,
        n_splits: int = 5,
        purge_window: int = 0,
        embargo_window: int = 0,
        blocked_utc_hours: list[int] | None = None,
        job_id: str = "",
        on_stage_complete: Any = None,
        on_round_complete: Any = None,
        should_abort: Any = None,
        prior_seen_hashes: Any = None,
        budget: Any = None,
        **loader_kwargs,
    ) -> dict[str, Any]:
        from .infra.tracing import tracer

        overall_start = perf_counter()
        timing: dict[str, Any] = {}

        # Top-level trace: all child spans nest under this
        with tracer.start_span(
            "search",
            kind="search",
            symbols=",".join(symbols),
            rounds=generations,
            batch_size=offspring_count,
        ) as search_span:
            # 1. Load dataset
            with tracer.start_span("load_dataset", kind="internal", symbols=len(symbols), interval=interval):
                dataset = self.dataset_loader.load(
                    symbols=symbols,
                    start_time=start_time,
                    end_time=end_time,
                    interval=interval,
                    min_quote_volume=min_quote_volume,
                    blocked_utc_hours=blocked_utc_hours,
                    **loader_kwargs,
                )

            # 2. Validation plan
            validation_plan = self._build_validation_plan(
                dataset,
                n_splits=n_splits,
                purge_window=purge_window,
                embargo_window=embargo_window,
            )
            folds = validation_plan["folds"]

            # 3. Evaluation callbacks
            def evaluate_fn(individuals):
                r = self.evaluate_population_on_validation_plan(individuals, folds)
                return EvalResult(
                    metrics_by_hash=r["metrics_by_hash"],
                    signatures_by_hash=r["signatures_by_hash"],
                    details_by_hash=r["details_by_hash"],
                    timing=r.get("timing", {}),
                )

            quick_fn = None
            if len(folds) > 1:

                def quick_fn(individuals):
                    r = self.evaluate_population_on_validation_plan(individuals, folds[:1])
                    return EvalResult(
                        metrics_by_hash=r["metrics_by_hash"],
                        signatures_by_hash=r["signatures_by_hash"],
                        details_by_hash=r["details_by_hash"],
                        timing=r.get("timing", {}),
                    )

            # 4. Run search (all round/breed/eval spans nest under this trace)
            search_result = self.search_engine.run(
                seeds=list(seeds or []),
                rounds=generations,
                batch_size=max(offspring_count, 2),
                top_k=top_k,
                novelty_threshold=novelty_threshold,
                evaluate_fn=evaluate_fn,
                quick_evaluate_fn=quick_fn,
                dataset=dataset,
                vm=self.vm,
                job_id=job_id,
                on_stage_complete=on_stage_complete,
                on_round_complete=on_round_complete,
                should_abort=should_abort,
                prior_seen_hashes=prior_seen_hashes,
                budget=budget,
            )

            # 4b. Persist strategy memory for cross-session learning
            self.strategy_memory.save()

            # 5. Build result
            timing.update(search_result.timing)
            result = {
                "dataset": {
                    "interval": dataset.interval,
                    "symbols": dataset.symbols,
                    "shape": dataset.shape(),
                },
                "llm": self._llm_backend_summary(),
                "timing": timing,
                "validation": validation_plan["summary"],
                "search_stats": {
                    "total_evaluations": search_result.total_evaluations,
                    "total_rejected": search_result.total_rejected,
                    "archive_size": len(search_result.archive),
                },
                "rounds": search_result.rounds,
                "lineage": [
                    {
                        "expr_hash": ind.expr_hash,
                        "formula": ind.formula,
                        "parent_a": ind.lineage.parent_a,
                        "parent_b": ind.lineage.parent_b,
                    }
                    for ind in search_result.all_evaluated
                    if ind.lineage.parent_a
                ],
                "top_results": [
                    {
                        "formula": ind.formula,
                        "expr_hash": ind.expr_hash,
                        "fitness": ind.fitness,
                        "lineage": ind.lineage.to_dict(),
                        "metrics": ind.metrics,
                    }
                    for ind in search_result.archive[:top_k]
                ],
                "evaluations": {
                    h: {
                        "formula": d.get("formula"),
                        "metrics": d.get("fitness_metrics", {}),
                        "split_metrics": d.get("split_metrics", {}),
                    }
                    for h, d in search_result.details_by_hash.items()
                },
                "pipeline": search_result.pipeline.to_dict() if search_result.pipeline else None,
            }

            # 6. Persist
            if persist:
                with tracer.start_span("persist", kind="internal"):
                    run = self.persistence.save_run(result, run_name=run_name or "search_db")
                    # Enrich top-K entries with source + signature hashes for
                    # automatic archival with full lineage.
                    zoo_payloads: list[dict[str, Any]] = []
                    for idx, top in enumerate(result["top_results"]):
                        detail = search_result.details_by_hash.get(top.get("expr_hash"), {})
                        payload = {
                            **top,
                            "source": "auto_archive",
                            "source_job_id": job_id or None,
                            "source_run_id": run.run_id,
                            "source_strategy": (top.get("lineage") or {}).get("origin"),
                            "source_round": (top.get("lineage") or {}).get("round_idx"),
                            "auto_archived": True,
                            "parent_expr_hashes": [
                                h
                                for h in [
                                    (top.get("lineage") or {}).get("parent_a"),
                                    (top.get("lineage") or {}).get("parent_b"),
                                ]
                                if h
                            ],
                            "alpha_signature": detail.get("alpha_signature"),
                        }
                        zoo_payloads.append(payload)
                    zoo_paths = self.persistence.save_zoo_entries(zoo_payloads, run.run_id)
                result["persistence"] = {
                    "run_id": run.run_id,
                    "run_path": run.run_path,
                    "zoo_paths": zoo_paths,
                }

            timing["overall_seconds"] = perf_counter() - overall_start

            # Log degenerate signal summary if any were suppressed
            if self._degenerate_count > 0:
                logger.info(
                    "alpha.eval degenerate_total={} (details at DEBUG level)",
                    self._degenerate_count,
                )
                self._degenerate_count = 0
                self._data_quality_logged = False

            # 7. Summary on top-level span (visible in Langfuse dashboard)
            search_span.set("total_evaluations", search_result.total_evaluations)
            search_span.set("total_rejected", search_result.total_rejected)
            search_span.set("archive_size", len(search_result.archive))
            if result["top_results"]:
                best = result["top_results"][0]
                search_span.set("best_fitness", round(best["fitness"], 4))
                search_span.set("best_formula", best["formula"][:60])

        tracer.flush()
        result["trace_id"] = search_span.trace_id

        if result.get("persistence", {}).get("run_id"):
            self.persistence.update_run(result["persistence"]["run_id"], result)
        return result

    def combine_factors_from_db(
        self,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = "5m",
        min_quote_volume: float = 0.0,
        blocked_utc_hours: list[int] | None = None,
        method: str = "ic_weighted",
        max_factors: int = 10,
        min_abs_ic: float = 0.01,
        max_correlation: float = 0.70,
        ic_lookback: int = 60,
        zoo_limit: int = 50,
        risk_config: RiskConfig | None = None,
        summary_only: bool = False,
        **loader_kwargs,
    ) -> dict[str, Any]:
        """
        Load market data, pick factors from zoo, combine, apply risk management, and evaluate.
        """
        overall_start = perf_counter()

        # 1. Load dataset
        dataset = self.dataset_loader.load(
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            min_quote_volume=min_quote_volume,
            **loader_kwargs,
            blocked_utc_hours=blocked_utc_hours,
        )
        load_seconds = perf_counter() - overall_start

        # 2. Load zoo entries
        zoo_entries = self.persistence.list_zoo_entries(limit=zoo_limit)
        if not zoo_entries:
            raise ValueError("Alpha zoo is empty — run search-db first to populate it")

        # 3. Select & combine
        combo_result = self.combiner.combine_from_zoo(
            zoo_entries=zoo_entries,
            dataset=dataset,
            method=method,
            max_factors=max_factors,
            min_abs_ic=min_abs_ic,
            max_correlation=max_correlation,
            ic_lookback=ic_lookback,
        )
        combined_signal = combo_result["combined_signal"]

        # 4. Signal → weights → rule overlay
        store = self._prepare_store(dataset)
        market_ctx = MarketContext(
            liquidity_mask=dataset.liquidity_mask,
            session_mask=dataset.session_mask,
        )
        target_weights = self.signal_transformer.to_target_weights(combined_signal, market_ctx)
        wrapped_weights = self.rule_overlay.apply(target_weights, market_ctx)

        # 5. Portfolio risk management (vol targeting, drawdown control, trailing stop)
        risk_cfg = risk_config or RiskConfig()
        close_np = self._to_numpy(store.get_field("close"))
        managed_weights = self.portfolio_manager.apply(self._to_numpy(wrapped_weights), close_np, risk_cfg)

        # 6. Execute
        prices, funding = self._sim_args(store)
        bt_result = self.execution.simulate(managed_weights, prices, self.cost_model, funding_rate=funding)

        # 7. Metrics
        from .eval.metrics import compute_forward_returns, compute_rank_ic

        fwd = compute_forward_returns(close_np)
        rank_ic = compute_rank_ic(combined_signal[:-1], fwd[:-1])

        result: dict[str, Any] = {
            "dataset": self._dataset_summary(dataset),
            "combination": {
                "method": method,
                "selected_factors": combo_result["selected_factors"],
                "factor_count": len(combo_result["selected_factors"]),
            },
            "risk_config": {
                "vol_target": risk_cfg.vol_target,
                "max_drawdown": risk_cfg.max_drawdown,
                "trailing_stop_pct": risk_cfg.trailing_stop_pct,
            },
            "metrics": {
                **bt_result.summary(),
                "rank_ic": float(rank_ic),
            },
            "timing": {
                "dataset_load_seconds": load_seconds,
                **combo_result["timing"],
                "overall_seconds": perf_counter() - overall_start,
            },
        }
        if not summary_only:
            result["equity_series"] = bt_result.equity_curve.tolist()[-20:]
            result["turnover_series"] = bt_result.turnover.tolist()[-20:]

        logger.info(
            "alpha.combine_factors complete method={} factors={} sharpe={:.4f} rank_ic={:.4f} "
            "max_dd={:.4f} overall={:.3f}s",
            method,
            len(combo_result["selected_factors"]),
            float(bt_result.summary().get("sharpe", 0)),
            float(rank_ic),
            float(bt_result.summary().get("max_drawdown", 0)),
            result["timing"]["overall_seconds"],
        )
        return result

    def generate_event_backtest_weights(
        self,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = "1d",
        min_quote_volume: float = 0.0,
        blocked_utc_hours: list[int] | None = None,
        method: str = "ic_weighted",
        position_method: str = "long_only",
        top_pct: float = 0.2,
        max_factors: int = 10,
        min_abs_ic: float = 0.01,
        max_correlation: float = 0.70,
        ic_lookback: int = 60,
        zoo_limit: int = 50,
        **loader_kwargs,
    ) -> dict[str, Any]:
        """
        生成适用于 event engine 的预计算权重表。

        流程:
        1. 加载数据集 (AlphaDataset)
        2. FactorCombiner.combine_from_zoo() -> 组合信号 (T x N)
        3. SignalTransformer.to_target_weights() -> 目标权重 (T x N)
        4. 将权重矩阵转为 {timestamp: {symbol: weight}} dict
        5. 返回 weight_map + 元数据
        """
        from .risk.models import EvalMethod

        overall_start = perf_counter()

        # 1. 加载数据集
        dataset = self.dataset_loader.load(
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            min_quote_volume=min_quote_volume,
            blocked_utc_hours=blocked_utc_hours,
            **loader_kwargs,
        )
        load_seconds = perf_counter() - overall_start

        # 2. 加载 zoo 并组合
        zoo_entries = self.persistence.list_zoo_entries(limit=zoo_limit)
        if not zoo_entries:
            raise ValueError("Alpha zoo is empty - run search-db first to populate it")

        combo_result = self.combiner.combine_from_zoo(
            zoo_entries=zoo_entries,
            dataset=dataset,
            method=method,
            max_factors=max_factors,
            min_abs_ic=min_abs_ic,
            max_correlation=max_correlation,
            ic_lookback=ic_lookback,
        )
        combined_signal = combo_result["combined_signal"]

        # 3. 信号 -> 目标权重
        market_ctx = MarketContext(
            liquidity_mask=dataset.liquidity_mask,
            session_mask=dataset.session_mask,
        )
        eval_method = EvalMethod.LONG_ONLY if position_method == "long_only" else EvalMethod.LONG_SHORT
        target_weights = self.signal_transformer.to_target_weights(
            combined_signal,
            market_ctx,
            method=eval_method,
            top_pct=top_pct,
        )
        weights_np = self._to_numpy(target_weights)

        # 4. 向量化参考指标
        store = self._prepare_store(dataset)
        prices, funding = self._sim_args(store)
        bt_result = self.execution.simulate(weights_np, prices, self.cost_model, funding_rate=funding)
        vectorized_metrics = bt_result.summary()

        # 5. 构建 weight_map: {date_str: {symbol: weight}}
        timestamps = dataset.timestamps
        ds_symbols = dataset.symbols
        weight_map: dict[str, dict[str, float]] = {}

        for t_idx in range(weights_np.shape[0]):
            ts = timestamps[t_idx]
            date_str = str(ts)[:10] if not isinstance(ts, str) else ts[:10]
            row = weights_np[t_idx]
            # 只保留非零权重，节省内存
            symbol_weights = {}
            for s_idx in range(len(ds_symbols)):
                w = float(row[s_idx])
                if abs(w) > 1e-8:
                    symbol_weights[ds_symbols[s_idx]] = w
            if symbol_weights:
                weight_map[date_str] = symbol_weights

        overall_seconds = perf_counter() - overall_start
        logger.info(
            "alpha.generate_event_weights method={} factors={} weight_dates={} overall={:.3f}s",
            method,
            len(combo_result["selected_factors"]),
            len(weight_map),
            overall_seconds,
        )

        return {
            "weight_map": weight_map,
            "symbols": ds_symbols,
            "combination": {
                "method": method,
                "selected_factors": combo_result["selected_factors"],
                "factor_count": len(combo_result["selected_factors"]),
            },
            "vectorized_metrics": vectorized_metrics,
            "dataset": self._dataset_summary(dataset),
            "timing": {
                "dataset_load_seconds": load_seconds,
                **combo_result["timing"],
                "overall_seconds": overall_seconds,
            },
        }

    def evaluate_formula_across_splits(
        self,
        formula: str,
        dataset_splits: dict[str, AlphaDataset],
    ) -> dict[str, Any]:
        program = self._compile_cached(formula)
        ind = Individual(
            formula=formula,
            program=program,
            expr_hash=program.expr_hash,
            lineage={"origin": "manual"},
        )
        population = [ind]
        split_results = self.evaluate_population_across_splits(population, dataset_splits)
        expr_hash = population[0].expr_hash
        return split_results["details_by_hash"][expr_hash]

    def evaluate_population_across_splits(
        self,
        population: list[Any],
        dataset_splits: dict[str, AlphaDataset],
    ) -> dict[str, dict[str, Any]]:
        fold = {
            "fold": ValidationFold(
                fold_id=0,
                train_indices=tuple(range(dataset_splits["train"].shape()[0])),
                valid_indices=tuple(range(dataset_splits["valid"].shape()[0])),
                test_indices=tuple(range(dataset_splits["test"].shape()[0])),
                valid_group=0,
                test_group=1,
                purge_window=0,
                embargo_window=0,
            ),
            "datasets": dataset_splits,
        }
        return self.evaluate_population_on_validation_plan(population, [fold])

    def evaluate_population_from_dataset(
        self,
        population: list[Any],
        dataset: AlphaDataset,
        timing_breakdown: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if not population:
            return {
                "metrics_by_hash": {},
                "signatures_by_hash": {},
                "details_by_hash": {},
            }

        if timing_breakdown is None:
            timing_breakdown = self._new_evaluation_timing()

        store_start = perf_counter()
        store = self._prepare_store(dataset)
        timing_breakdown["tensor_store_seconds"] += perf_counter() - store_start

        compile_start = perf_counter()
        programs = []
        for individual in population:
            program = individual.program
            if program is None:
                program = self._compile_cached(individual.formula)
                individual.program = program
            programs.append(program)
        timing_breakdown["compile_seconds"] += perf_counter() - compile_start

        vm_start = perf_counter()
        alphas = self.vm.run_batch(programs, store)
        _vm_ms = (perf_counter() - vm_start) * 1000
        timing_breakdown["vm_run_seconds"] += _vm_ms / 1000
        timing_breakdown["dataset_eval_calls"] += 1
        timing_breakdown["formula_evaluations"] += len(population)
        logger.debug("alpha.eval vm.run_batch  {:.0f}ms  n={}", _vm_ms, len(programs))

        # Use batched GPU pipeline when possible
        try:
            import torch as _torch

            if len(alphas) > 1 and isinstance(alphas[0], _torch.Tensor) and alphas[0].is_cuda:
                T, S = alphas[0].shape
                gpu_free = _torch.cuda.mem_get_info()[0] if _torch.cuda.is_available() else 0
                bytes_per_formula = T * S * 4 * 12
                chunk_size = max(8, min(len(alphas), int(gpu_free * 0.6 / max(bytes_per_formula, 1))))

                if chunk_size >= len(alphas):
                    return self._batch_evaluate_gpu(
                        population,
                        programs,
                        alphas,
                        dataset,
                        store,
                        timing_breakdown,
                    )
                merged: dict[str, dict] = {"metrics_by_hash": {}, "signatures_by_hash": {}, "details_by_hash": {}}
                for start in range(0, len(alphas), chunk_size):
                    end = min(start + chunk_size, len(alphas))
                    chunk_result = self._batch_evaluate_gpu(
                        population[start:end],
                        programs[start:end],
                        alphas[start:end],
                        dataset,
                        store,
                        timing_breakdown,
                    )
                    for key in merged:
                        merged[key].update(chunk_result[key])
                    _torch.cuda.empty_cache()
                return merged
        except Exception as exc:
            logger.warning("alpha.eval.gpu_batch FAILED, falling back to CPU: {}", exc)
            logger.debug("alpha.eval.gpu_batch traceback:", exc_info=True)

        # Fallback: per-formula evaluation
        metrics_by_hash: dict[str, dict[str, float]] = {}
        signatures_by_hash: dict[str, list[float]] = {}
        details_by_hash: dict[str, dict[str, Any]] = {}

        for individual, alpha in zip(population, alphas):
            evaluation = self._build_dataset_evaluation(
                individual.program,
                alpha,
                dataset,
                store,
                timing_breakdown=timing_breakdown,
            )
            metrics_by_hash[individual.expr_hash] = evaluation["metrics"]
            signatures_by_hash[individual.expr_hash] = evaluation["alpha_signature"]
            details_by_hash[individual.expr_hash] = {
                "formula": individual.formula,
                "metrics": evaluation["metrics"],
                "alpha_signature": evaluation["alpha_signature"],
            }

        return {
            "metrics_by_hash": metrics_by_hash,
            "signatures_by_hash": signatures_by_hash,
            "details_by_hash": details_by_hash,
        }

    def _batch_evaluate_gpu(
        self,
        population: list[Any],
        programs: list[Any],
        alphas: list[Any],
        dataset: Any,
        store: Any,
        timing_breakdown: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Batch-evaluate all formulas on GPU using 3D tensor ops."""
        import torch

        N = len(alphas)
        T, S = alphas[0].shape
        _gpu_t0 = perf_counter()
        stacked = torch.stack(alphas)  # (N, T, S)

        # --- 1. Batch signal transform: (N, T, S) → (N, T, S) weights ---
        signal_start = perf_counter()
        scores = stacked.clone()
        if dataset.liquidity_mask is not None:
            liq = torch.as_tensor(
                np.asarray(dataset.liquidity_mask, dtype=bool),
                device=scores.device,
                dtype=torch.bool,
            )
            scores = torch.where(liq.unsqueeze(0), scores, torch.full_like(scores, torch.nan))
        if dataset.session_mask is not None:
            sess = torch.as_tensor(
                np.asarray(dataset.session_mask, dtype=bool),
                device=scores.device,
                dtype=torch.bool,
            )
            scores = torch.where(sess.unsqueeze(0), scores, torch.zeros_like(scores))

        valid = ~torch.isnan(scores)
        valid_counts = valid.sum(dim=2, keepdim=True)  # (N, T, 1)
        safe_scores = torch.where(valid, scores, torch.zeros_like(scores))
        row_means = safe_scores.sum(dim=2, keepdim=True) / valid_counts.clamp(min=1).float()
        centered = scores - row_means
        empty = valid_counts == 0
        centered = torch.where(empty, torch.zeros_like(centered), centered)
        abs_c = torch.where(torch.isnan(centered), torch.zeros_like(centered), torch.abs(centered))
        denom = abs_c.sum(dim=2, keepdim=True)
        weights = centered / (denom + 1e-12)
        weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
        from .risk.models import MarketContext as _MC

        max_w = _MC().max_abs_weight
        max_turnover = _MC().max_turnover_per_bar
        weights = torch.clamp(weights, -max_w, max_w)
        timing_breakdown["signal_transform_seconds"] += perf_counter() - signal_start

        # --- 2. Batch rule overlay (turnover limit) ---
        # Use numba on CPU (compiled C loop is >>10x faster than Python for-loop on GPU)
        overlay_start = perf_counter()
        try:
            from .risk.models import _apply_turnover_limit_numba

            _has_numba = True
        except ImportError:
            _has_numba = False
        device = weights.device
        if _has_numba:
            w_cpu = weights.cpu().numpy()  # (N, T, S)
            for i in range(N):
                w_cpu[i] = _apply_turnover_limit_numba(
                    np.ascontiguousarray(w_cpu[i]),
                    max_turnover,
                )
            weights = torch.as_tensor(w_cpu, device=device, dtype=torch.float32)
        else:
            for t in range(1, T):
                delta = torch.clamp(
                    weights[:, t] - weights[:, t - 1],
                    -max_turnover,
                    max_turnover,
                )
                weights[:, t] = weights[:, t - 1] + delta
        timing_breakdown["rule_overlay_seconds"] += perf_counter() - overlay_start

        # --- 3. Batch simulate ---
        backtest_start = perf_counter()
        close = store.get_field("close")  # (T, S) GPU tensor
        forward_returns = torch.zeros_like(close)
        forward_returns[:-1] = close[1:] / (close[:-1] + 1e-12) - 1.0
        forward_returns = torch.clamp(forward_returns, -0.5, 0.5)

        # Position returns: (N, T)
        position_returns = torch.nansum(
            weights * forward_returns.unsqueeze(0),
            dim=2,
        ).clamp(-0.5, 0.5)

        # Turnover: (N, T)
        prev_weights = torch.cat([torch.zeros_like(weights[:, :1, :]), weights[:, :-1, :]], dim=1)
        trade_sizes = torch.abs(weights - prev_weights)  # (N, T, S)
        turnover = torch.nansum(trade_sizes, dim=2)  # (N, T)

        # Costs (inline to handle 3D correctly)
        cost = self.cost_model
        fees = turnover * (cost.taker_fee_bps / 10000.0)  # (N, T)
        slippage_base = trade_sizes * (cost.slippage_bps / 10000.0)  # (N, T, S)
        if "bid_ask_spread" in self.schema.fields:
            spread_field = store.get_field("bid_ask_spread")  # (T, S)
            spread_ratio = torch.abs(spread_field) / (torch.abs(close) + 1e-12)
            slippage_base = slippage_base + trade_sizes * spread_ratio.unsqueeze(0) * cost.spread_weight
        slippage_impact = trade_sizes.pow(2) * (cost.impact_coefficient_bps / 10000.0)
        total_slippage = torch.nansum(slippage_base + slippage_impact, dim=2)  # (N, T)
        if "funding_rate" in self.schema.fields:
            funding_rate = store.get_field("funding_rate")  # (T, S)
            funding = torch.nansum(torch.abs(weights) * funding_rate.unsqueeze(0), dim=2)
        else:
            funding = 0.0

        net_returns = position_returns - fees - total_slippage - funding  # (N, T)
        equity_curve = torch.cumprod(1.0 + torch.nan_to_num(net_returns, nan=0.0), dim=1)  # (N, T)
        timing_breakdown["backtest_seconds"] += perf_counter() - backtest_start

        # --- 4. Batch metrics on GPU ---
        fitness_start = perf_counter()
        # Summary stats: (N,)
        mean_ret = torch.nanmean(net_returns, dim=1)
        # nan-safe std
        nr_safe = torch.nan_to_num(net_returns, nan=0.0)
        std_ret = nr_safe.std(dim=1)
        sharpe = mean_ret / (std_ret + 1e-12)
        final_equity = equity_curve[:, -1]
        total_return = final_equity - 1.0
        avg_turnover = torch.nanmean(turnover, dim=1)
        peak = torch.cummax(equity_curve, dim=1).values
        drawdown = torch.where(peak > 1e-12, 1.0 - equity_curve / peak, torch.zeros_like(equity_curve))
        max_drawdown = torch.clamp(drawdown.max(dim=1).values, 0.0, 1.0)

        # Rank IC: batch cross-sectional correlation
        alpha_ic = stacked[:, :-1, :]  # (N, T-1, S)
        ret_ic = forward_returns[:-1, :].unsqueeze(0).expand_as(alpha_ic)  # (N, T-1, S)
        ic_mask = ~torch.isnan(alpha_ic) & ~torch.isnan(ret_ic)
        ic_counts = ic_mask.sum(dim=2)  # (N, T-1)
        safe_a = torch.where(ic_mask, alpha_ic, torch.zeros_like(alpha_ic))
        safe_r = torch.where(ic_mask, ret_ic, torch.zeros_like(ret_ic))
        ic_denom = ic_counts.clamp(min=1).float()
        a_mean = safe_a.sum(dim=2) / ic_denom  # (N, T-1)
        r_mean = safe_r.sum(dim=2) / ic_denom
        ca = torch.where(ic_mask, alpha_ic - a_mean.unsqueeze(2), torch.zeros_like(alpha_ic))
        cr = torch.where(ic_mask, ret_ic - r_mean.unsqueeze(2), torch.zeros_like(ret_ic))
        cov = (ca * cr).sum(dim=2)  # (N, T-1)
        va = (ca * ca).sum(dim=2)
        vr = (cr * cr).sum(dim=2)
        valid_rows = (ic_counts >= 3) & (va > 1e-12) & (vr > 1e-12)
        corr = cov / (torch.sqrt(va * vr) + 1e-24)
        corr = torch.clamp(corr, -1.0, 1.0)
        corr_safe = torch.where(valid_rows, corr, torch.zeros_like(corr))
        valid_row_cnt = valid_rows.float().sum(dim=1).clamp(min=1)
        rank_ic = corr_safe.sum(dim=1) / valid_row_cnt  # (N,)

        # Coverage & activity
        signal_coverage = (~stacked.isnan()).float().mean(dim=(1, 2))  # (N,)
        active_rows = weights.abs().sum(dim=2) > 1e-9  # (N, T)
        active_bar_ratio = active_rows.float().mean(dim=1)  # (N,)
        effective_bars = active_rows.float().sum(dim=1)  # (N,)

        # Transfer to CPU once (single batch transfer)
        cpu_data = (
            torch.stack(
                [
                    rank_ic,
                    sharpe,
                    total_return,
                    avg_turnover,
                    max_drawdown,
                    std_ret,
                    final_equity,
                    signal_coverage,
                    active_bar_ratio,
                    effective_bars,
                ]
            )
            .cpu()
            .numpy()
        )  # (10, N)
        timing_breakdown["fitness_seconds"] += perf_counter() - fitness_start

        # --- 5. Build per-formula result dicts ---
        metrics_by_hash: dict[str, dict[str, float]] = {}
        signatures_by_hash: dict[str, list[float]] = {}
        details_by_hash: dict[str, dict[str, Any]] = {}

        # Batch alpha signatures on CPU
        sig_start = perf_counter()
        stacked_cpu = stacked.cpu().numpy()  # single transfer
        timing_breakdown["signature_seconds"] += perf_counter() - sig_start

        for i, individual in enumerate(population):
            ric = float(cpu_data[0, i])
            sh = float(cpu_data[1, i])
            tr = float(cpu_data[2, i])
            at = float(cpu_data[3, i])
            md = float(cpu_data[4, i])
            vol = float(cpu_data[5, i])
            fe = float(cpu_data[6, i])
            sc = float(cpu_data[7, i])
            abr = float(cpu_data[8, i])
            eb = float(cpu_data[9, i])

            is_inactive = abr < 0.10 or at < 0.005
            pnl_per_turn = tr / (at + 1e-12) if not is_inactive else 0.0
            tail_adj = tr - md
            tail_ratio = tail_adj / max(vol, 1e-6)
            activity_score = self._clip_unit(min(abr / 0.60, 1.0) * min(at / 0.05, 1.0))
            turnover_penalty = self._clip_unit((at - 0.60) / 0.40)
            coverage_penalty = self._clip_unit((0.50 - sc) / 0.50)
            pnl_eff = self._clip_unit(float(np.log1p(max(pnl_per_turn, 0.0)) / np.log1p(10.0)))
            complexity = self._formula_complexity_metrics(programs[i])

            metrics: dict[str, float] = {
                "sharpe": sh,
                "total_return": tr,
                "avg_turnover": at,
                "volatility": vol,
                "max_drawdown": md,
                "final_equity": fe,
                "rank_ic": ric,
                "rank_ic_abs": abs(ric),
                "pnl_per_turnover": pnl_per_turn,
                "pnl_efficiency_score": pnl_eff,
                "activity_score": activity_score,
                "tail_ratio": tail_ratio,
                "tail_penalty_adjusted_return": tail_adj,
                "turnover_penalty": turnover_penalty,
                "coverage_penalty": coverage_penalty,
                "train_valid_gap_penalty": 0.0,
                "valid_test_gap_penalty": 0.0,
                "signal_coverage": sc,
                "active_bar_ratio": abr,
                "effective_bars": eb,
                "inactive": 1.0 if is_inactive else 0.0,
            }
            metrics.update(complexity)

            # Alpha signature from pre-transferred CPU array
            alpha_cpu = stacked_cpu[i]
            sig = self._build_alpha_signature_from_numpy(alpha_cpu)

            metrics_by_hash[individual.expr_hash] = metrics
            signatures_by_hash[individual.expr_hash] = sig
            details_by_hash[individual.expr_hash] = {
                "formula": individual.formula,
                "metrics": metrics,
                "alpha_signature": sig,
            }

        logger.debug("alpha.eval.gpu n={} {:.0f}ms", N, (perf_counter() - _gpu_t0) * 1000)
        return {
            "metrics_by_hash": metrics_by_hash,
            "signatures_by_hash": signatures_by_hash,
            "details_by_hash": details_by_hash,
        }

    def _build_alpha_signature_from_numpy(self, alpha_np: np.ndarray, max_points: int = 200) -> list[float]:
        """Build signature from a pre-transferred numpy array (avoids redundant GPU→CPU)."""
        data = np.nan_to_num(alpha_np, nan=0.0, posinf=0.0, neginf=0.0)
        if data.size == 0:
            return []
        row_mean = np.mean(data, axis=1)
        row_std = np.std(data, axis=1)
        col_mean = np.mean(data, axis=0)
        if len(row_mean) > max_points:
            step = len(row_mean) / max_points
            indices = np.clip((np.arange(max_points) * step).astype(int), 0, len(row_mean) - 1)
            row_mean = row_mean[indices]
            row_std = row_std[indices]
        return np.concatenate([row_mean, row_std, col_mean]).astype(np.float32).tolist()

    def evaluate_population_on_validation_plan(
        self,
        population: list[Any],
        validation_folds: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        timing_breakdown = self._new_evaluation_timing()
        split_metrics_by_hash: dict[str, dict[str, list[dict[str, float]]]] = {}
        split_signatures_by_hash: dict[str, dict[str, list[list[float]]]] = {}
        details_by_hash: dict[str, dict[str, Any]] = {}

        # --- Parallel fold evaluation ---
        # Each fold (with train/valid/test splits) is evaluated independently.
        # Uses ThreadPoolExecutor because VM releases the GIL during numpy/torch ops.
        def _eval_fold(fold_entry: dict[str, Any]) -> tuple[Any, dict[str, dict[str, Any]]]:
            fold = fold_entry["fold"]
            dataset_ref = fold_entry.get("dataset_ref")
            split_indices = {
                "train": fold.train_indices,
                "valid": fold.valid_indices,
                "test": fold.test_indices,
            }
            pre_materialized = fold_entry.get("datasets")
            fold_results: dict[str, dict[str, Any]] = {}
            for split_name in ("train", "valid", "test"):
                if pre_materialized is not None:
                    split_dataset = pre_materialized[split_name]
                else:
                    split_dataset = dataset_ref.take_indices(split_indices[split_name])
                split_results = self.evaluate_population_from_dataset(
                    population,
                    split_dataset,
                )
                fold_results[split_name] = split_results
                del split_dataset
            return fold, fold_results

        n_folds = len(validation_folds)
        # Scale workers by dataset size to cap total memory.
        # Each worker holds a TensorStore + alpha outputs (~30-50 MB for typical datasets).
        first_ref = validation_folds[0].get("dataset_ref") if validation_folds else None
        if first_ref is not None:
            t, s = first_ref.shape()
            n_fields = len(first_ref.fields)
            est_mb_per_worker = t * s * n_fields * 4 / (1024 * 1024)  # float32
            max_workers = max(1, min(n_folds, 4, int(512 / max(est_mb_per_worker, 1))))
        else:
            max_workers = min(n_folds, 2)
        if n_folds > 1 and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_eval_fold, fe): fe for fe in validation_folds}
                completed_folds = []
                for future in as_completed(futures):
                    completed_folds.append(future.result())
        else:
            completed_folds = [_eval_fold(fe) for fe in validation_folds]

        # Merge results from all folds
        for fold, fold_results in completed_folds:
            timing_breakdown["fold_count"] += 1
            for split_name in ("train", "valid", "test"):
                timing_breakdown["split_count"] += 1
                split_results = fold_results[split_name]
                for expr_hash, metrics in split_results["metrics_by_hash"].items():
                    split_metrics_by_hash.setdefault(expr_hash, {}).setdefault(split_name, []).append(metrics)
                for expr_hash, signature in split_results["signatures_by_hash"].items():
                    split_signatures_by_hash.setdefault(expr_hash, {}).setdefault(split_name, []).append(signature)

            for individual in population:
                expr_hash = individual.expr_hash
                details_by_hash.setdefault(
                    expr_hash,
                    {
                        "formula": individual.formula,
                        "split_metrics": {},
                        "fold_metrics": [],
                    },
                )
                details_by_hash[expr_hash]["fold_metrics"].append(
                    {
                        "fold": fold.to_dict(),
                        "metrics": {
                            split_name: fold_results[split_name]["metrics_by_hash"].get(expr_hash, {})
                            for split_name in ("train", "valid", "test")
                        },
                    }
                )

        aggregation_start = perf_counter()
        metrics_by_hash: dict[str, dict[str, float]] = {}
        signatures_by_hash: dict[str, list[float] | None] = {}
        for expr_hash, split_metrics in split_metrics_by_hash.items():
            train_records = split_metrics.get("train", [])
            valid_records = split_metrics.get("valid", [])
            test_records = split_metrics.get("test", [])
            train = self._aggregate_metric_records(train_records)
            valid = self._aggregate_metric_records(valid_records)
            test = self._aggregate_metric_records(test_records)
            gap_penalties = [
                self._train_valid_gap_penalty(train_record, valid_record)
                for train_record, valid_record in zip(train_records, valid_records)
            ]
            valid_test_gap_penalties = [
                self._valid_test_gap_penalty(valid_record, test_record)
                for valid_record, test_record in zip(valid_records, test_records)
            ]
            negative_test_ratio = (
                float(np.mean([1.0 if float(record.get("sharpe", 0.0)) < 0.0 else 0.0 for record in test_records]))
                if test_records
                else 0.0
            )
            inactive_valid_ratio = (
                float(np.mean([1.0 if float(record.get("inactive", 0.0)) >= 1.0 else 0.0 for record in valid_records]))
                if valid_records
                else 0.0
            )
            fitness_metrics = dict(valid)
            fitness_metrics.update(
                {
                    "rank_ic": float(valid.get("rank_ic", 0.0)),
                    "rank_ic_abs": float(valid.get("rank_ic_abs", abs(float(valid.get("rank_ic", 0.0))))),
                    "sharpe": float(valid.get("sharpe", 0.0)),
                    "pnl_per_turnover": float(valid.get("pnl_per_turnover", 0.0)),
                    "activity_score": float(valid.get("activity_score", 0.0)),
                    "tail_ratio": float(valid.get("tail_ratio", 0.0)),
                    "tail_penalty_adjusted_return": float(valid.get("tail_penalty_adjusted_return", 0.0)),
                    "turnover_penalty": float(valid.get("turnover_penalty", 0.0)),
                    "coverage_penalty": float(valid.get("coverage_penalty", 0.0)),
                    "complexity_penalty": float(valid.get("complexity_penalty", 0.0)),
                    "train_valid_gap_penalty": float(np.mean(gap_penalties)) if gap_penalties else 0.0,
                    "valid_test_gap_penalty": float(np.mean(valid_test_gap_penalties))
                    if valid_test_gap_penalties
                    else 0.0,
                    "test_sharpe": float(test.get("sharpe", 0.0)),
                    "train_sharpe": float(train.get("sharpe", 0.0)),
                    "valid_sharpe": float(valid.get("sharpe", 0.0)),
                    "train_rank_ic": float(train.get("rank_ic", 0.0)),
                    "valid_rank_ic": float(valid.get("rank_ic", 0.0)),
                    "test_rank_ic": float(test.get("rank_ic", 0.0)),
                    "train_rank_ic_abs": float(train.get("rank_ic_abs", abs(float(train.get("rank_ic", 0.0))))),
                    "valid_rank_ic_abs": float(valid.get("rank_ic_abs", abs(float(valid.get("rank_ic", 0.0))))),
                    "test_rank_ic_abs": float(test.get("rank_ic_abs", abs(float(test.get("rank_ic", 0.0))))),
                    "negative_test_ratio": negative_test_ratio,
                    "inactive_valid_ratio": inactive_valid_ratio,
                    "fold_count": float(len(valid_records)),
                }
            )
            metrics_by_hash[expr_hash] = fitness_metrics
            signatures_by_hash[expr_hash] = self._aggregate_signatures(
                split_signatures_by_hash.get(expr_hash, {}).get("valid", [])
            ) or self._aggregate_signatures(split_signatures_by_hash.get(expr_hash, {}).get("train", []))
            details_by_hash.setdefault(expr_hash, {})
            details_by_hash[expr_hash]["split_metrics"] = {
                "train": train,
                "valid": valid,
                "test": test,
            }
            details_by_hash[expr_hash]["fitness_metrics"] = fitness_metrics
            details_by_hash[expr_hash]["alpha_signature"] = signatures_by_hash[expr_hash]
        timing_breakdown["aggregation_seconds"] += perf_counter() - aggregation_start

        return {
            "metrics_by_hash": metrics_by_hash,
            "signatures_by_hash": signatures_by_hash,
            "details_by_hash": details_by_hash,
            "timing": timing_breakdown,
        }

    def _build_validation_plan(
        self,
        dataset: AlphaDataset,
        n_splits: int = 5,
        purge_window: int = 0,
        embargo_window: int = 0,
    ) -> dict[str, Any]:
        validator = CPCVValidator(
            purge_window=purge_window,
            embargo_window=embargo_window,
            min_train_size=3,
        )
        raw_folds = validator.generate_purged_splits(len(dataset.timestamps), n_splits=n_splits)
        mode = "cpcv"
        if not raw_folds:
            fallback = validator.generate_holdout_split(len(dataset.timestamps))
            if fallback is None:
                fallback = ValidationFold(
                    fold_id=0,
                    train_indices=tuple(range(len(dataset.timestamps))),
                    valid_indices=tuple(range(len(dataset.timestamps))),
                    test_indices=tuple(range(len(dataset.timestamps))),
                    valid_group=0,
                    test_group=0,
                    purge_window=0,
                    embargo_window=0,
                )
            raw_folds = [fallback]
            mode = "holdout"

        # Store only fold metadata — datasets are created lazily during
        # evaluation to avoid holding all 15 split copies in memory at once.
        folds = [{"fold": fold, "dataset_ref": dataset} for fold in raw_folds]
        return {
            "folds": folds,
            "summary": {
                "mode": mode,
                "n_splits": n_splits,
                "purge_window": purge_window,
                "embargo_window": embargo_window,
                "fold_count": len(raw_folds),
                "folds": [fold.to_dict() for fold in raw_folds],
            },
        }

    def _clip_unit(self, value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _formula_complexity_metrics(self, program: BytecodeProgram | None) -> dict[str, float]:
        if program is None:
            return {
                "ast_depth": 0.0,
                "node_count": 0.0,
                "operator_count": 0.0,
                "time_series_ops": 0.0,
                "cross_sectional_ops": 0.0,
                "domain_ops": 0.0,
                "branching_ops": 0.0,
                "complexity_penalty": 0.0,
            }
        cached = self._complexity_cache.get(program.expr_hash)
        if cached is not None:
            return dict(cached)

        report = self.registry.validate_formula(program.normalized_formula, self.schema)
        ast_tree = report.ast_tree or {}
        counts = {
            "time_series_ops": 0.0,
            "cross_sectional_ops": 0.0,
            "domain_ops": 0.0,
            "branching_ops": 0.0,
            "operator_count": 0.0,
        }
        node_count = 0
        ast_depth = 0

        def walk(node: dict[str, Any], depth: int = 1) -> None:
            nonlocal node_count, ast_depth
            if not node:
                return
            node_count += 1
            ast_depth = max(ast_depth, depth)
            if node.get("kind") == "call":
                counts["operator_count"] += 1.0
                op_name = str(node.get("value", ""))
                if self.registry.has(op_name):
                    category = self.registry.get(op_name).category
                    key = f"{category}_ops"
                    if key in counts:
                        counts[key] += 1.0
                    if category in {"conditional", "logical", "comparison"}:
                        counts["branching_ops"] += 1.0
            for child in node.get("children", []):
                if isinstance(child, dict):
                    walk(child, depth + 1)

        if isinstance(ast_tree, dict):
            walk(ast_tree)

        depth_excess = max(float(ast_depth) - 4.0, 0.0) / 3.0
        op_excess = max(counts["operator_count"] - 10.0, 0.0) / 10.0
        time_series_excess = max(counts["time_series_ops"] - 3.0, 0.0) / 3.0
        branching_excess = max(counts["branching_ops"] - 1.0, 0.0) / 2.0
        complexity_penalty = self._clip_unit(
            0.45 * depth_excess + 0.35 * op_excess + 0.15 * time_series_excess + 0.05 * branching_excess
        )
        metrics = {
            "ast_depth": float(ast_depth),
            "node_count": float(node_count),
            "operator_count": float(counts["operator_count"]),
            "time_series_ops": float(counts["time_series_ops"]),
            "cross_sectional_ops": float(counts["cross_sectional_ops"]),
            "domain_ops": float(counts["domain_ops"]),
            "branching_ops": float(counts["branching_ops"]),
            "complexity_penalty": complexity_penalty,
        }
        self._complexity_cache[program.expr_hash] = dict(metrics)
        return metrics

    def _train_valid_gap_penalty(self, train_record: dict[str, float], valid_record: dict[str, float]) -> float:
        sharpe_gap = self._clip_unit(
            abs(float(train_record.get("sharpe", 0.0)) - float(valid_record.get("sharpe", 0.0))) / 1.5
        )
        ic_gap = self._clip_unit(
            abs(float(train_record.get("rank_ic_abs", 0.0)) - float(valid_record.get("rank_ic_abs", 0.0))) / 0.05
        )
        activity_gap = self._clip_unit(
            abs(float(train_record.get("active_bar_ratio", 0.0)) - float(valid_record.get("active_bar_ratio", 0.0)))
            / 0.25
        )
        return 0.50 * sharpe_gap + 0.35 * ic_gap + 0.15 * activity_gap

    def _valid_test_gap_penalty(self, valid_record: dict[str, float], test_record: dict[str, float]) -> float:
        sharpe_decay = self._clip_unit(
            max(float(valid_record.get("sharpe", 0.0)) - float(test_record.get("sharpe", 0.0)), 0.0) / 1.0
        )
        ic_decay = self._clip_unit(
            max(float(valid_record.get("rank_ic_abs", 0.0)) - float(test_record.get("rank_ic_abs", 0.0)), 0.0) / 0.05
        )
        activity_decay = self._clip_unit(
            max(float(valid_record.get("activity_score", 0.0)) - float(test_record.get("activity_score", 0.0)), 0.0)
            / 0.5
        )
        return 0.55 * sharpe_decay + 0.30 * ic_decay + 0.15 * activity_decay

    def _build_fitness_metrics(
        self,
        program: BytecodeProgram | Any,
        alpha: Any,
        weights: Any,
        close: Any,
        summary: dict[str, float] | None = None,
    ) -> dict[str, float]:
        from .eval.metrics import compute_ic_metrics

        resolved_program: BytecodeProgram | None
        resolved_alpha: Any
        resolved_weights: Any
        resolved_close: Any
        resolved_summary: dict[str, float]

        if isinstance(program, BytecodeProgram):
            resolved_program = program
            resolved_alpha = alpha
            resolved_weights = weights
            resolved_close = close
            resolved_summary = dict(summary or {})
        else:
            resolved_program = None
            resolved_alpha = program
            resolved_weights = alpha
            resolved_close = weights
            resolved_summary = dict(close if isinstance(close, dict) else summary or {})

        alpha_np = self._to_numpy(resolved_alpha)
        weights_np = self._to_numpy(resolved_weights)
        close_np = self._to_numpy(resolved_close)

        # Unified IC metrics (rank_ic, ic_ir, ic_std, ic_decay, per-window ICs)
        ic_metrics = compute_ic_metrics(alpha_np, close_np)

        rank_ic = ic_metrics["rank_ic"]
        avg_turnover = float(resolved_summary.get("avg_turnover", 0.0))
        total_return = float(resolved_summary.get("total_return", 0.0))
        volatility = float(resolved_summary.get("volatility", 0.0))
        max_dd = float(resolved_summary.get("max_drawdown", 0.0))
        signal_coverage = float(np.mean(np.isfinite(alpha_np))) if alpha_np.size else 0.0
        active_rows = np.sum(np.abs(weights_np), axis=1) > 1e-9 if weights_np.size else np.array([], dtype=bool)
        active_bar_ratio = float(np.mean(active_rows)) if active_rows.size else 0.0
        effective_bars = float(np.sum(active_rows)) if active_rows.size else 0.0
        is_inactive = active_bar_ratio < 0.10 or avg_turnover < 0.005
        pnl_per_turnover = total_return / (avg_turnover + 1e-12) if not is_inactive else 0.0
        tail_adjusted_return = total_return - max_dd
        tail_ratio = tail_adjusted_return / max(volatility, 1e-6)
        activity_score = self._clip_unit(min(active_bar_ratio / 0.60, 1.0) * min(avg_turnover / 0.05, 1.0))
        turnover_penalty = self._clip_unit((avg_turnover - 0.60) / 0.40)
        coverage_penalty = self._clip_unit((0.50 - signal_coverage) / 0.50)
        pnl_efficiency_score = self._clip_unit(np.log1p(max(pnl_per_turnover, 0.0)) / np.log1p(10.0))
        complexity_metrics = self._formula_complexity_metrics(resolved_program)

        metrics = dict(resolved_summary)
        metrics.update(ic_metrics)  # rank_ic, ic_ir, ic_std, ic_decay, rank_ic_Nd, turnover_proxy
        metrics.update(
            {
                "rank_ic_abs": abs(rank_ic),
                "pnl_per_turnover": pnl_per_turnover,
                "pnl_efficiency_score": pnl_efficiency_score,
                "activity_score": activity_score,
                "tail_ratio": tail_ratio,
                "tail_penalty_adjusted_return": tail_adjusted_return,
                "turnover_penalty": turnover_penalty,
                "coverage_penalty": coverage_penalty,
                "train_valid_gap_penalty": 0.0,
                "valid_test_gap_penalty": 0.0,
                "signal_coverage": signal_coverage,
                "active_bar_ratio": active_bar_ratio,
                "effective_bars": effective_bars,
                "inactive": 1.0 if is_inactive else 0.0,
            }
        )
        metrics.update(complexity_metrics)
        return metrics

    def _aggregate_metric_records(self, records: list[dict[str, float]]) -> dict[str, float]:
        if not records:
            return {}
        keys = sorted({key for record in records for key in record})
        aggregated: dict[str, float] = {}
        for key in keys:
            values = [float(record[key]) for record in records if key in record]
            aggregated[key] = float(np.mean(values)) if values else 0.0
        return aggregated

    def _aggregate_signatures(self, signatures: list[list[float]]) -> list[float] | None:
        if not signatures:
            return None
        arrays = [np.asarray(signature, dtype=float) for signature in signatures if signature]
        if not arrays:
            return None
        min_len = min(array.size for array in arrays)
        if min_len == 0:
            return None
        stacked = np.stack([array[:min_len] for array in arrays], axis=0)
        return np.mean(stacked, axis=0).astype(float).tolist()

    def _compile_cached(self, formula: str) -> BytecodeProgram:
        with self._program_cache_lock:
            program = self._program_cache.get(formula)
            if program is not None:
                self._program_cache.move_to_end(formula)
                return program

        program = self.compiler.compile(formula, self.schema)
        if self._program_cache_size <= 0:
            return program
        with self._program_cache_lock:
            self._program_cache[formula] = program
            self._program_cache.move_to_end(formula)
            while len(self._program_cache) > self._program_cache_size:
                self._program_cache.popitem(last=False)
        return program

    def program_cache_stats(self) -> dict[str, int]:
        return {
            "size": len(self._program_cache),
            "capacity": self._program_cache_size,
        }

    def _dataset_summary(self, dataset: AlphaDataset) -> dict[str, Any]:
        return {
            "interval": dataset.interval,
            "symbols": dataset.symbols,
            "timestamps": dataset.timestamps[:5],
            "shape": dataset.shape(),
        }

    def _llm_backend_summary(self) -> dict[str, Any]:
        backend = self.llm_backend
        return {
            "backend": getattr(backend, "backend_name", backend.__class__.__name__),
            "model": getattr(backend, "model_name", None),
            "base_url": getattr(backend, "base_url", None),
            "call_stats": dict(getattr(backend, "call_stats", {})),
        }

    # Rate-limit degenerate signal warnings: log first N, then suppress until summary
    _degenerate_count: int = 0
    _degenerate_limit: int = 3
    _data_quality_logged: bool = False

    def _build_dataset_evaluation(
        self,
        program: BytecodeProgram,
        alpha: Any,
        dataset: AlphaDataset,
        store: TensorStore,
        timing_breakdown: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        alpha_np = self._to_numpy(alpha)
        nan_ratio = float(np.mean(np.isnan(alpha_np))) if alpha_np.size else 1.0
        alpha_std = float(np.nanstd(alpha_np)) if alpha_np.size else 0.0
        if nan_ratio > 0.95 or alpha_std < 1e-12:
            self._degenerate_count += 1
            if self._degenerate_count <= self._degenerate_limit:
                logger.debug(
                    "alpha.eval degenerate formula={} nan={:.0%} std={:.1e}",
                    program.normalized_formula[:50],
                    nan_ratio,
                    alpha_std,
                )
        close_np = self._to_numpy(store.get_field("close"))
        close_nan = float(np.mean(np.isnan(close_np))) if close_np.size else 1.0
        liq_true = (
            float(np.mean(np.asarray(dataset.liquidity_mask))) if np.asarray(dataset.liquidity_mask).size else 0.0
        )
        if (close_nan > 0.5 or liq_true < 0.5) and not self._data_quality_logged:
            self._data_quality_logged = True
            logger.warning(
                "alpha.eval data_quality close_nan={:.0%} liquidity={:.0%}",
                close_nan,
                liq_true,
            )

        from .risk.models import EvalMethod

        eval_method = EvalMethod(self.market_profile.default_eval_method)

        market_ctx = MarketContext(
            liquidity_mask=dataset.liquidity_mask,
            session_mask=dataset.session_mask,
        )
        signal_start = perf_counter()
        target_weights = self.signal_transformer.to_target_weights(alpha, market_ctx, method=eval_method)
        if timing_breakdown is not None:
            timing_breakdown["signal_transform_seconds"] += perf_counter() - signal_start

        overlay_start = perf_counter()
        wrapped_weights = self.rule_overlay.apply(target_weights, market_ctx)
        if timing_breakdown is not None:
            timing_breakdown["rule_overlay_seconds"] += perf_counter() - overlay_start

        backtest_start = perf_counter()
        prices, funding = self._sim_args(store)
        result = self.execution.simulate(wrapped_weights, prices, self.cost_model, funding_rate=funding)
        if timing_breakdown is not None:
            timing_breakdown["backtest_seconds"] += perf_counter() - backtest_start

        fitness_start = perf_counter()
        metrics = self._build_fitness_metrics(
            program, alpha, wrapped_weights, store.get_field("close"), result.summary()
        )
        metrics["eval_method"] = eval_method.value
        if timing_breakdown is not None:
            timing_breakdown["fitness_seconds"] += perf_counter() - fitness_start

        signature_start = perf_counter()
        alpha_signature = self._build_alpha_signature(alpha)
        if timing_breakdown is not None:
            timing_breakdown["signature_seconds"] += perf_counter() - signature_start
        equity_np = self._to_numpy(result.equity_curve)
        turnover_np = self._to_numpy(result.turnover)
        ts = dataset.timestamps
        equity_series = self._downsample_series(equity_np, ts, max_points=500)
        drawdown_series = self._build_drawdown_series(equity_np, ts, max_points=500)
        turnover_series = self._downsample_series(turnover_np, ts, max_points=500)

        payload: dict[str, Any] = {
            "program": program.to_dict(),
            "metrics": metrics,
            "alpha_signature": alpha_signature,
            "alpha_tail": self._to_serializable_list(alpha[-5:]),
            "weights_tail": self._to_serializable_list(wrapped_weights[-5:]),
            "equity_tail": self._to_serializable_list(result.equity_curve[-5:]),
            "equity_series": equity_series,
            "drawdown_series": drawdown_series,
            "turnover_series": turnover_series,
            "backend": self.vm.backend,
            "device": str(self.vm.device) if self.vm.device is not None else "numpy",
        }

        # 分层回测 (always computed — low cost, high diagnostic value)
        from .eval.metrics import compute_quantile_returns

        alpha_np = self._to_numpy(alpha)
        close_np = self._to_numpy(store.get_field("close"))
        payload["quantile_analysis"] = compute_quantile_returns(
            alpha_np,
            close_np,
            timestamps=dataset.timestamps,
        )

        return payload

    def _new_evaluation_timing(self) -> dict[str, Any]:
        return {
            "compile_seconds": 0.0,
            "tensor_store_seconds": 0.0,
            "vm_run_seconds": 0.0,
            "signal_transform_seconds": 0.0,
            "rule_overlay_seconds": 0.0,
            "backtest_seconds": 0.0,
            "fitness_seconds": 0.0,
            "signature_seconds": 0.0,
            "aggregation_seconds": 0.0,
            "fold_loop_seconds": 0.0,
            "fold_count": 0,
            "split_count": 0,
            "dataset_eval_calls": 0,
            "formula_evaluations": 0,
        }

    def _summarize_evaluation_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        program = payload.get("program", {})
        summary: dict[str, Any] = {
            "dataset": payload.get("dataset"),
            "backend": payload.get("backend"),
            "device": payload.get("device"),
            "expr_hash": program.get("expr_hash"),
            "normalized_formula": program.get("normalized_formula"),
            "metrics": payload.get("metrics", {}),
            "alpha_tail": payload.get("alpha_tail", []),
            "weights_tail": payload.get("weights_tail", []),
            "equity_tail": payload.get("equity_tail", []),
            "equity_series": payload.get("equity_series", []),
            "drawdown_series": payload.get("drawdown_series", []),
            "turnover_series": payload.get("turnover_series", []),
        }
        if "quantile_analysis" in payload:
            summary["quantile_analysis"] = payload["quantile_analysis"]
        return summary

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.persistence.list_runs(limit=limit)

    def load_run(self, run_id: str) -> dict[str, Any]:
        return self.persistence.load_run(run_id)

    def list_zoo(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.persistence.list_zoo_entries(limit=limit)

    def save_formula_to_zoo(
        self,
        formula: str,
        fitness: float | None = None,
        metrics: dict[str, Any] | None = None,
        lineage: dict[str, Any] | None = None,
        note: str | None = None,
        tags: list[str] | None = None,
        source: str = "manual",
    ) -> dict[str, Any]:
        validation = self.validate_formula(formula)
        if not validation.get("ok", False):
            errors = validation.get("errors") or ["Formula validation failed"]
            raise ValueError("; ".join(str(item) for item in errors))

        metrics_payload = dict(metrics or {})
        resolved_fitness = fitness
        if resolved_fitness is None:
            for key in (
                "sharpe",
                "rank_ic",
                "pnl_per_turnover",
                "tail_penalty_adjusted_return",
            ):
                candidate = metrics_payload.get(key)
                if candidate is None:
                    continue
                try:
                    resolved_fitness = float(candidate)
                    break
                except (TypeError, ValueError):
                    continue

        program = self._compile_cached(formula)
        payload = {
            "formula": formula,
            "expr_hash": program.expr_hash,
            "fitness": float(resolved_fitness) if resolved_fitness is not None else 0.0,
            "metrics": metrics_payload,
            "lineage": dict(lineage or {"origin": source}),
            "note": note,
            "tags": list(tags or []),
            "source": source,
            "validation": validation,
        }
        merged = self.persistence.upsert_zoo_entry(payload, run_id=source)
        return merged

    def get_lineage(self, run_id: str) -> dict[str, Any]:
        run = self.persistence.load_run(run_id)
        return {
            "run_id": run_id,
            "lineage": run.get("lineage", []),
            "top_results": run.get("top_results", []),
        }

    def _build_alpha_signature(self, alpha: Any, max_points: int = 200) -> list[float]:
        data = np.nan_to_num(self._to_numpy(alpha), nan=0.0, posinf=0.0, neginf=0.0)
        if data.size == 0:
            return []
        row_mean = np.mean(data, axis=1)  # (T,)
        row_std = np.std(data, axis=1)  # (T,)
        col_mean = np.mean(data, axis=0)  # (S,)
        # Downsample time-axis vectors to cap memory
        if len(row_mean) > max_points:
            step = len(row_mean) / max_points
            indices = np.arange(max_points) * step
            indices = np.clip(indices.astype(int), 0, len(row_mean) - 1)
            row_mean = row_mean[indices]
            row_std = row_std[indices]
        signature = np.concatenate([row_mean, row_std, col_mean])
        return signature.astype(np.float32).tolist()

    def _to_numpy(self, value: Any) -> np.ndarray:
        from .core.vm import to_numpy

        return to_numpy(value)

    def _to_serializable_list(self, value: Any) -> list[Any]:
        return self._to_numpy(value).tolist()

    def _downsample_series(
        self,
        arr: np.ndarray,
        timestamps: list[str] | None = None,
        max_points: int = 500,
    ) -> list[dict]:
        """Downsample a 1-D array to max_points via LTTB-like min/max bucketing."""
        flat = np.nan_to_num(arr.flatten() if arr.ndim > 1 else arr, nan=0.0)
        n = flat.size
        if n == 0:
            return []

        def _point(idx: int) -> dict:
            p: dict = {"i": idx, "v": round(float(flat[idx]), 6)}
            if timestamps and idx < len(timestamps):
                p["t"] = timestamps[idx]
            return p

        if n <= max_points:
            return [_point(i) for i in range(n)]
        step = n / max_points
        result: list[dict] = []
        for b in range(max_points):
            lo = int(b * step)
            hi = min(int((b + 1) * step), n)
            bucket = flat[lo:hi]
            idx_min = lo + int(np.argmin(bucket))
            idx_max = lo + int(np.argmax(bucket))
            first, second = (idx_min, idx_max) if idx_min <= idx_max else (idx_max, idx_min)
            result.append(_point(first))
            if first != second:
                result.append(_point(second))
        return result

    def _build_drawdown_series(
        self,
        equity: np.ndarray,
        timestamps: list[str] | None = None,
        max_points: int = 500,
    ) -> list[dict]:
        flat = np.nan_to_num(equity.flatten() if equity.ndim > 1 else equity, nan=1.0)
        if flat.size == 0:
            return []
        peak = np.maximum.accumulate(flat)
        dd = np.where(peak > 1e-12, 1.0 - flat / peak, 0.0)
        return self._downsample_series(dd, timestamps, max_points)
