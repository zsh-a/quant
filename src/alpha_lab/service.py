from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime
from time import perf_counter
from typing import Any

import numpy as np
from loguru import logger

from .compiler import BytecodeProgram, FormulaCompiler
from .dataset import AlphaDataset, CryptoMinuteDatasetLoader
from .dsl import DSLRegistry, TensorSchema
from .evolution import EvolutionEngine
from .persistence import AlphaLabPersistence
from .risk import CostModel, ExecutionSimulator, MarketContext, RuleOverlay, SignalTransformer
from .validation import CPCVValidator, ValidationFold
from .vm import StackVM, TensorStore

DEFAULT_DB_SEEDS = [
    "CSRank(ts_mean(close, 5) - close)",
    "CSRank(ts_std(close, 10))",
    "CSRank(volatility_n(close, 20))",
]


class AlphaLabService:
    def __init__(
        self,
        schema: TensorSchema | None = None,
        llm_backend: Any | None = None,
        llm_backend_name: str = "auto",
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        program_cache_size: int = 512,
    ):
        self.registry = DSLRegistry()
        self.schema = schema or TensorSchema.default_market_schema()
        self.compiler = FormulaCompiler(self.registry)
        self.vm = StackVM()
        self.evolution = EvolutionEngine(
            registry=self.registry,
            compiler=self.compiler,
            schema=self.schema,
            llm_backend=llm_backend,
            backend_name=llm_backend_name,
            model_name=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        self.signal_transformer = SignalTransformer()
        self.rule_overlay = RuleOverlay()
        self.execution = ExecutionSimulator()
        self.dataset_loader = CryptoMinuteDatasetLoader()
        self.persistence = AlphaLabPersistence()
        self.validator = CPCVValidator()
        self._program_cache_size = max(int(program_cache_size), 0)
        self._program_cache: OrderedDict[str, BytecodeProgram] = OrderedDict()

    def list_operators(self) -> list[dict[str, Any]]:
        return [asdict(spec) for spec in self.registry.list_operators()]

    def validate_formula(self, formula: str) -> dict[str, Any]:
        report = self.registry.validate_formula(formula, self.schema)
        return asdict(report)

    def compile_formula(self, formula: str) -> dict[str, Any]:
        program = self._compile_cached(formula)
        report = self.registry.validate_formula(formula, self.schema)
        return {
            "validation": asdict(report),
            "program": program.to_dict(),
        }

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
        result = self.execution.simulate(
            wrapped_weights,
            {
                "close": store.get_field("close"),
                "bid_ask_spread": store.get_field("bid_ask_spread"),
            },
            CostModel(),
        )
        return {
            "program": program.to_dict(),
            "alpha": self._to_serializable_list(alpha),
            "weights": self._to_serializable_list(wrapped_weights),
            "metrics": result.summary(),
        }

    def evaluate_formula_from_dataset(
        self,
        formula: str,
        dataset: AlphaDataset,
        summary_only: bool = False,
    ) -> dict[str, Any]:
        store = TensorStore(dataset.fields)
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
        store = TensorStore(dataset.fields)
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
        provider: str,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = "5m",
        min_quote_volume: float = 0.0,
        blocked_utc_hours: list[int] | None = None,
        summary_only: bool = False,
    ) -> dict[str, Any]:
        dataset = self.dataset_loader.load(
            provider=provider,
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            min_quote_volume=min_quote_volume,
            blocked_utc_hours=blocked_utc_hours,
        )
        return self.evaluate_formula_from_dataset(formula=formula, dataset=dataset, summary_only=summary_only)

    def evaluate_formulas_from_db(
        self,
        formulas: list[str],
        provider: str,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = "5m",
        min_quote_volume: float = 0.0,
        blocked_utc_hours: list[int] | None = None,
        summary_only: bool = False,
    ) -> dict[str, dict[str, Any]]:
        dataset = self.dataset_loader.load(
            provider=provider,
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            min_quote_volume=min_quote_volume,
            blocked_utc_hours=blocked_utc_hours,
        )
        return self.evaluate_formulas_from_dataset(formulas=formulas, dataset=dataset, summary_only=summary_only)

    def seed_population(self, seeds: list[str], population_size: int = 8) -> list[dict[str, Any]]:
        population = self.evolution.initialize(seeds, population_size)
        return [
            {
                "formula": individual.formula,
                "expr_hash": individual.expr_hash,
                "lineage": individual.lineage,
            }
            for individual in population
        ]

    def breed_population(self, formulas: list[str], offspring_count: int = 4) -> list[dict[str, Any]]:
        survivors = self.evolution.initialize(formulas, len(formulas))
        offspring = self.evolution.breed(survivors, offspring_count)
        return [
            {
                "formula": individual.formula,
                "expr_hash": individual.expr_hash,
                "lineage": individual.lineage,
            }
            for individual in offspring
        ]

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
        fields = {
            "open": base_close - rng.normal(0.0, 0.1, size=(rows, cols)),
            "high": base_close + np.abs(rng.normal(0.25, 0.1, size=(rows, cols))),
            "low": base_close - np.abs(rng.normal(0.25, 0.1, size=(rows, cols))),
            "close": base_close,
            "volume": base_volume,
            "turnover": base_close * base_volume,
            "vwap": base_close + rng.normal(0.0, 0.03, size=(rows, cols)),
            "funding_rate": np.zeros((rows, cols), dtype=float),
            "open_interest": np.abs(rng.normal(1_000_000.0, 10_000.0, size=(rows, cols))),
            "bid_ask_spread": np.abs(rng.normal(0.5, 0.05, size=(rows, cols))),
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
        provider: str,
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
            provider=provider,
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
        provider: str,
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
    ) -> dict[str, Any]:
        overall_start = perf_counter()
        timing: dict[str, Any] = {
            "overall_seconds": 0.0,
            "dataset_load_seconds": 0.0,
            "validation_plan_seconds": 0.0,
            "population_init_seconds": 0.0,
            "persistence_seconds": 0.0,
            "per_generation": [],
        }
        logger.info(
            "alpha_lab.search start provider={} symbols={} generations={} population_size={} offspring_count={} llm_backend={}",
            provider,
            ",".join(symbols),
            generations,
            population_size,
            offspring_count,
            getattr(self.evolution.llm_backend, "backend_name", self.evolution.llm_backend.__class__.__name__),
        )
        dataset_start = perf_counter()
        dataset = self.dataset_loader.load(
            provider=provider,
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            min_quote_volume=min_quote_volume,
            blocked_utc_hours=blocked_utc_hours,
        )
        timing["dataset_load_seconds"] = perf_counter() - dataset_start
        logger.info(
            "alpha_lab.search dataset_loaded shape={} load_seconds={:.4f}",
            dataset.shape(),
            timing["dataset_load_seconds"],
        )
        validation_start = perf_counter()
        validation_plan = self._build_validation_plan(
            dataset,
            n_splits=n_splits,
            purge_window=purge_window,
            embargo_window=embargo_window,
        )
        timing["validation_plan_seconds"] = perf_counter() - validation_start
        logger.info(
            "alpha_lab.search validation_ready mode={} fold_count={} plan_seconds={:.4f}",
            validation_plan["summary"]["mode"],
            validation_plan["summary"]["fold_count"],
            timing["validation_plan_seconds"],
        )
        seeds = list(seeds or [])
        init_start = perf_counter()
        population = self._deduplicate_population(self.evolution.initialize(seeds, population_size))
        timing["population_init_seconds"] = perf_counter() - init_start
        logger.info(
            "alpha_lab.search population_initialized requested_seeds={} actual_population={} init_seconds={:.4f}",
            len(seeds),
            len(population),
            timing["population_init_seconds"],
        )
        generation_summaries = []
        details_by_hash: dict[str, dict[str, Any]] = {}
        lineage_edges: list[dict[str, Any]] = []
        last_metrics_by_hash: dict[str, dict[str, float]] = {}
        final_population: list[Any] = population

        for generation in range(generations):
            generation_start = perf_counter()
            logger.info(
                "alpha_lab.search generation_start generation={} population_size={}",
                generation,
                len(population),
            )
            evaluation_start = perf_counter()
            split_results = self.evaluate_population_on_validation_plan(population, validation_plan["folds"])
            evaluation_seconds = perf_counter() - evaluation_start
            evaluation_timing = dict(split_results.get("timing", {}))
            metrics_by_hash = split_results["metrics_by_hash"]
            signatures_by_hash = split_results["signatures_by_hash"]
            details_by_hash.update(split_results["details_by_hash"])
            scoring_start = perf_counter()
            scored_population = self.evolution.attach_metrics(population, metrics_by_hash)
            scored_population = self._filter_by_novelty(
                scored_population,
                signatures_by_hash=signatures_by_hash,
                threshold=novelty_threshold,
            )
            survivors = self.evolution.select_survivors(
                scored_population,
                top_k=max(1, min(top_k, len(scored_population))),
            )
            scoring_seconds = perf_counter() - scoring_start
            last_metrics_by_hash = metrics_by_hash
            final_population = survivors
            offspring_count_actual = 0
            breeding_seconds = 0.0
            generation_summaries.append(
                {
                    "generation": generation,
                    "population_size": len(scored_population),
                    "novelty_threshold": novelty_threshold,
                    "validation_mode": validation_plan["summary"]["mode"],
                    "fold_count": validation_plan["summary"]["fold_count"],
                    "survivors": [
                        {
                            "formula": individual.formula,
                            "expr_hash": individual.expr_hash,
                            "fitness": individual.fitness,
                            "metrics": metrics_by_hash.get(individual.expr_hash, {}),
                        }
                        for individual in survivors
                    ],
                }
            )
            generation_timing = {
                "generation": generation,
                "evaluation_seconds": evaluation_seconds,
                "compile_seconds": float(evaluation_timing.get("compile_seconds", 0.0)),
                "tensor_store_seconds": float(evaluation_timing.get("tensor_store_seconds", 0.0)),
                "vm_run_seconds": float(evaluation_timing.get("vm_run_seconds", 0.0)),
                "signal_transform_seconds": float(evaluation_timing.get("signal_transform_seconds", 0.0)),
                "rule_overlay_seconds": float(evaluation_timing.get("rule_overlay_seconds", 0.0)),
                "backtest_seconds": float(evaluation_timing.get("backtest_seconds", 0.0)),
                "fitness_seconds": float(evaluation_timing.get("fitness_seconds", 0.0)),
                "signature_seconds": float(evaluation_timing.get("signature_seconds", 0.0)),
                "aggregation_seconds": float(evaluation_timing.get("aggregation_seconds", 0.0)),
                "fold_loop_seconds": float(evaluation_timing.get("fold_loop_seconds", 0.0)),
                "selection_seconds": scoring_seconds,
                "breeding_seconds": breeding_seconds,
                "total_seconds": 0.0,
                "population_size": len(population),
                "survivor_count": len(survivors),
                "offspring_count": offspring_count_actual,
                "fold_count": int(evaluation_timing.get("fold_count", 0)),
                "split_count": int(evaluation_timing.get("split_count", 0)),
                "dataset_eval_calls": int(evaluation_timing.get("dataset_eval_calls", 0)),
                "formula_evaluations": int(evaluation_timing.get("formula_evaluations", 0)),
            }
            if generation == generations - 1:
                generation_timing["total_seconds"] = perf_counter() - generation_start
                timing["per_generation"].append(generation_timing)
                logger.info(
                    "alpha_lab.search generation_complete generation={} survivors={} eval_seconds={:.4f} vm_seconds={:.4f} backtest_seconds={:.4f} fitness_seconds={:.4f} select_seconds={:.4f} total_seconds={:.4f}",
                    generation,
                    len(survivors),
                    evaluation_seconds,
                    generation_timing["vm_run_seconds"],
                    generation_timing["backtest_seconds"],
                    generation_timing["fitness_seconds"],
                    scoring_seconds,
                    generation_timing["total_seconds"],
                )
                break
            breeding_start = perf_counter()
            offspring = self._deduplicate_population(self.evolution.breed(survivors, offspring_count))
            breeding_seconds = perf_counter() - breeding_start
            offspring_count_actual = len(offspring)
            for child in offspring:
                lineage_edges.append(
                    {
                        "expr_hash": child.expr_hash,
                        "formula": child.formula,
                        "parent_a": child.lineage.get("parent_a"),
                        "parent_b": child.lineage.get("parent_b"),
                        "generation": generation + 1,
                    }
                )
            generation_timing["breeding_seconds"] = breeding_seconds
            generation_timing["offspring_count"] = offspring_count_actual
            if not offspring:
                final_population = survivors
                generation_timing["total_seconds"] = perf_counter() - generation_start
                timing["per_generation"].append(generation_timing)
                logger.info(
                    "alpha_lab.search generation_complete generation={} survivors={} offspring=0 eval_seconds={:.4f} vm_seconds={:.4f} backtest_seconds={:.4f} fitness_seconds={:.4f} select_seconds={:.4f} breed_seconds={:.4f} total_seconds={:.4f}",
                    generation,
                    len(survivors),
                    evaluation_seconds,
                    generation_timing["vm_run_seconds"],
                    generation_timing["backtest_seconds"],
                    generation_timing["fitness_seconds"],
                    scoring_seconds,
                    breeding_seconds,
                    generation_timing["total_seconds"],
                )
                break
            population = survivors + offspring
            generation_timing["total_seconds"] = perf_counter() - generation_start
            timing["per_generation"].append(generation_timing)
            logger.info(
                "alpha_lab.search generation_complete generation={} survivors={} offspring={} eval_seconds={:.4f} vm_seconds={:.4f} backtest_seconds={:.4f} fitness_seconds={:.4f} select_seconds={:.4f} breed_seconds={:.4f} total_seconds={:.4f}",
                generation,
                len(survivors),
                offspring_count_actual,
                evaluation_seconds,
                generation_timing["vm_run_seconds"],
                generation_timing["backtest_seconds"],
                generation_timing["fitness_seconds"],
                scoring_seconds,
                breeding_seconds,
                generation_timing["total_seconds"],
            )

        ranked_population = sorted(final_population, key=lambda item: item.fitness, reverse=True)
        result = {
            "dataset": {
                "provider": dataset.provider,
                "interval": dataset.interval,
                "symbols": dataset.symbols,
                "shape": dataset.shape(),
            },
            "llm": self._llm_backend_summary(),
            "timing": timing,
            "validation": validation_plan["summary"],
            "generations": generation_summaries,
            "lineage": lineage_edges,
            "top_results": [
                {
                    "formula": individual.formula,
                    "expr_hash": individual.expr_hash,
                    "fitness": individual.fitness,
                    "lineage": individual.lineage,
                    "metrics": last_metrics_by_hash.get(individual.expr_hash, {}),
                }
                for individual in ranked_population[:top_k]
            ],
            "evaluations": {
                expr_hash: {
                    "formula": next((item.formula for item in ranked_population if item.expr_hash == expr_hash), None),
                    "metrics": details["fitness_metrics"],
                    "split_metrics": details["split_metrics"],
                    "fold_metrics": details.get("fold_metrics", []),
                }
                for expr_hash, details in details_by_hash.items()
            },
        }
        persisted_run_id: str | None = None
        if persist:
            persistence_start = perf_counter()
            run = self.persistence.save_run(result, run_name=run_name or "search_db")
            zoo_paths = self.persistence.save_zoo_entries(result["top_results"], run.run_id)
            timing["persistence_seconds"] = perf_counter() - persistence_start
            persisted_run_id = run.run_id
            result["persistence"] = {
                "run_id": run.run_id,
                "run_path": run.run_path,
                "zoo_paths": zoo_paths,
            }
            logger.info(
                "alpha_lab.search persistence_complete run_id={} persistence_seconds={:.4f}",
                run.run_id,
                timing["persistence_seconds"],
            )
        timing["overall_seconds"] = perf_counter() - overall_start
        logger.info(
            "alpha_lab.search complete top_results={} overall_seconds={:.4f}",
            len(result["top_results"]),
            timing["overall_seconds"],
        )
        if persisted_run_id is not None:
            self.persistence.update_run(persisted_run_id, result)
        return result

    def evaluate_formula_across_splits(
        self,
        formula: str,
        dataset_splits: dict[str, AlphaDataset],
    ) -> dict[str, Any]:
        program = self._compile_cached(formula)
        population = [self.evolution._build_individual(formula, {"origin": "manual"})]
        if not population or population[0] is None:
            raise ValueError(f"Unable to compile formula: {formula}")
        population[0].program = program
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
        store = TensorStore(dataset.fields)
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
        timing_breakdown["vm_run_seconds"] += perf_counter() - vm_start
        timing_breakdown["dataset_eval_calls"] += 1
        timing_breakdown["formula_evaluations"] += len(population)
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
                "program": evaluation["program"],
                "metrics": evaluation["metrics"],
                "alpha_signature": evaluation["alpha_signature"],
            }

        return {
            "metrics_by_hash": metrics_by_hash,
            "signatures_by_hash": signatures_by_hash,
            "details_by_hash": details_by_hash,
        }

    def evaluate_population_on_validation_plan(
        self,
        population: list[Any],
        validation_folds: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        timing_breakdown = self._new_evaluation_timing()
        split_metrics_by_hash: dict[str, dict[str, list[dict[str, float]]]] = {}
        split_signatures_by_hash: dict[str, dict[str, list[list[float]]]] = {}
        details_by_hash: dict[str, dict[str, Any]] = {}

        for fold_entry in validation_folds:
            timing_breakdown["fold_count"] += 1
            fold = fold_entry["fold"]
            fold_results: dict[str, dict[str, dict[str, Any]]] = {}
            for split_name, split_dataset in fold_entry["datasets"].items():
                split_start = perf_counter()
                split_results = self.evaluate_population_from_dataset(
                    population,
                    split_dataset,
                    timing_breakdown=timing_breakdown,
                )
                timing_breakdown["fold_loop_seconds"] += perf_counter() - split_start
                timing_breakdown["split_count"] += 1
                fold_results[split_name] = split_results
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
                abs(float(train_record.get("sharpe", 0.0)) - float(valid_record.get("sharpe", 0.0)))
                for train_record, valid_record in zip(train_records, valid_records)
            ]
            fitness_metrics = dict(valid)
            fitness_metrics.update(
                {
                    "rank_ic": float(valid.get("rank_ic", 0.0)),
                    "sharpe": float(valid.get("sharpe", 0.0)),
                    "pnl_per_turnover": float(valid.get("pnl_per_turnover", 0.0)),
                    "stability": float(valid.get("stability", 0.0)),
                    "tail_penalty_adjusted_return": float(valid.get("tail_penalty_adjusted_return", 0.0)),
                    "turnover_penalty": float(valid.get("turnover_penalty", 0.0)),
                    "complexity_penalty": 0.0,
                    "train_valid_gap_penalty": float(np.mean(gap_penalties)) if gap_penalties else 0.0,
                    "test_sharpe": float(test.get("sharpe", 0.0)),
                    "train_sharpe": float(train.get("sharpe", 0.0)),
                    "valid_sharpe": float(valid.get("sharpe", 0.0)),
                    "fold_count": float(len(valid_records)),
                }
            )
            metrics_by_hash[expr_hash] = fitness_metrics
            signatures_by_hash[expr_hash] = (
                self._aggregate_signatures(split_signatures_by_hash.get(expr_hash, {}).get("valid", []))
                or self._aggregate_signatures(split_signatures_by_hash.get(expr_hash, {}).get("train", []))
            )
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

        folds = [
            {
                "fold": fold,
                "datasets": {
                    "train": dataset.take_indices(fold.train_indices),
                    "valid": dataset.take_indices(fold.valid_indices),
                    "test": dataset.take_indices(fold.test_indices),
                },
            }
            for fold in raw_folds
        ]
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

    def _build_fitness_metrics(
        self,
        alpha: Any,
        weights: Any,
        close: Any,
        summary: dict[str, float],
    ) -> dict[str, float]:
        alpha_np = self._to_numpy(alpha)
        weights_np = self._to_numpy(weights)
        close_np = self._to_numpy(close)
        forward_returns = np.zeros_like(close_np)
        forward_returns[:-1] = close_np[1:] / (close_np[:-1] + 1e-12) - 1.0
        rank_ic = self._mean_cross_sectional_correlation(alpha_np[:-1], forward_returns[:-1])
        avg_turnover = float(summary.get("avg_turnover", 0.0))
        total_return = float(summary.get("total_return", 0.0))
        volatility = float(summary.get("volatility", 0.0))
        signal_coverage = float(np.mean(np.isfinite(alpha_np))) if alpha_np.size else 0.0
        active_rows = np.sum(np.abs(weights_np), axis=1) > 1e-9 if weights_np.size else np.array([], dtype=bool)
        active_bar_ratio = float(np.mean(active_rows)) if active_rows.size else 0.0
        effective_bars = float(np.sum(active_rows)) if active_rows.size else 0.0
        is_inactive = active_bar_ratio <= 1e-6 or avg_turnover <= 1e-12
        metrics = dict(summary)
        pnl_per_turnover = total_return / (avg_turnover + 1e-12) if not is_inactive else 0.0
        stability = 0.0
        if not is_inactive and volatility > 1e-12:
            stability = min(1.0 / volatility, 10.0)
        metrics.update(
            {
                "rank_ic": rank_ic,
                "pnl_per_turnover": pnl_per_turnover,
                "stability": stability,
                "tail_penalty_adjusted_return": total_return - float(summary.get("max_drawdown", 0.0)),
                "turnover_penalty": avg_turnover,
                "complexity_penalty": 0.0,
                "train_valid_gap_penalty": 0.0,
                "signal_coverage": signal_coverage,
                "active_bar_ratio": active_bar_ratio,
                "effective_bars": effective_bars,
                "inactive": 1.0 if is_inactive else 0.0,
            }
        )
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
        program = self._program_cache.get(formula)
        if program is not None:
            self._program_cache.move_to_end(formula)
            return program

        program = self.compiler.compile(formula, self.schema)
        if self._program_cache_size <= 0:
            return program
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
            "provider": dataset.provider,
            "interval": dataset.interval,
            "symbols": dataset.symbols,
            "timestamps": dataset.timestamps[:5],
            "shape": dataset.shape(),
        }

    def _llm_backend_summary(self) -> dict[str, Any]:
        backend = self.evolution.llm_backend
        return {
            "backend": getattr(backend, "backend_name", backend.__class__.__name__),
            "model": getattr(backend, "model_name", None),
            "base_url": getattr(backend, "base_url", None),
            "call_stats": dict(getattr(backend, "call_stats", {})),
        }

    def _build_dataset_evaluation(
        self,
        program: BytecodeProgram,
        alpha: Any,
        dataset: AlphaDataset,
        store: TensorStore,
        timing_breakdown: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        market_ctx = MarketContext(
            liquidity_mask=dataset.liquidity_mask,
            session_mask=dataset.session_mask,
        )
        signal_start = perf_counter()
        target_weights = self.signal_transformer.to_target_weights(alpha, market_ctx)
        if timing_breakdown is not None:
            timing_breakdown["signal_transform_seconds"] += perf_counter() - signal_start

        overlay_start = perf_counter()
        wrapped_weights = self.rule_overlay.apply(target_weights, market_ctx)
        if timing_breakdown is not None:
            timing_breakdown["rule_overlay_seconds"] += perf_counter() - overlay_start

        backtest_start = perf_counter()
        result = self.execution.simulate(
            wrapped_weights,
            {
                "close": store.get_field("close"),
                "bid_ask_spread": store.get_field("bid_ask_spread"),
            },
            CostModel(),
            funding_rate=store.get_field("funding_rate"),
        )
        if timing_breakdown is not None:
            timing_breakdown["backtest_seconds"] += perf_counter() - backtest_start

        fitness_start = perf_counter()
        metrics = self._build_fitness_metrics(alpha, wrapped_weights, store.get_field("close"), result.summary())
        if timing_breakdown is not None:
            timing_breakdown["fitness_seconds"] += perf_counter() - fitness_start

        signature_start = perf_counter()
        alpha_signature = self._build_alpha_signature(alpha)
        if timing_breakdown is not None:
            timing_breakdown["signature_seconds"] += perf_counter() - signature_start
        return {
            "program": program.to_dict(),
            "metrics": metrics,
            "alpha_signature": alpha_signature,
            "alpha_tail": self._to_serializable_list(alpha[-5:]),
            "weights_tail": self._to_serializable_list(wrapped_weights[-5:]),
            "equity_tail": self._to_serializable_list(result.equity_curve[-5:]),
            "backend": self.vm.backend,
            "device": str(self.vm.device) if self.vm.device is not None else "numpy",
        }

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
        return {
            "dataset": payload.get("dataset"),
            "backend": payload.get("backend"),
            "device": payload.get("device"),
            "expr_hash": program.get("expr_hash"),
            "normalized_formula": program.get("normalized_formula"),
            "metrics": payload.get("metrics", {}),
            "alpha_tail": payload.get("alpha_tail", []),
            "weights_tail": payload.get("weights_tail", []),
            "equity_tail": payload.get("equity_tail", []),
        }

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
        paths = self.persistence.save_zoo_entries([payload], run_id=source)
        payload["path"] = paths[0] if paths else None
        return payload

    def get_lineage(self, run_id: str) -> dict[str, Any]:
        run = self.persistence.load_run(run_id)
        return {
            "run_id": run_id,
            "lineage": run.get("lineage", []),
            "top_results": run.get("top_results", []),
        }

    def _mean_cross_sectional_correlation(self, alpha: np.ndarray, returns: np.ndarray) -> float:
        alpha_np = np.asarray(alpha, dtype=float)
        returns_np = np.asarray(returns, dtype=float)
        mask = ~np.isnan(alpha_np) & ~np.isnan(returns_np)
        valid_counts = np.sum(mask, axis=1)
        if not np.any(valid_counts >= 2):
            return 0.0

        safe_alpha = np.where(mask, alpha_np, 0.0)
        safe_returns = np.where(mask, returns_np, 0.0)
        denom = np.maximum(valid_counts, 1)
        mean_alpha = np.sum(safe_alpha, axis=1) / denom
        mean_returns = np.sum(safe_returns, axis=1) / denom
        centered_alpha = np.where(mask, alpha_np - mean_alpha[:, None], 0.0)
        centered_returns = np.where(mask, returns_np - mean_returns[:, None], 0.0)

        cov = np.sum(centered_alpha * centered_returns, axis=1)
        var_alpha = np.sum(centered_alpha * centered_alpha, axis=1)
        var_returns = np.sum(centered_returns * centered_returns, axis=1)
        valid_rows = (valid_counts >= 2) & (var_alpha > 1e-24) & (var_returns > 1e-24)
        if not np.any(valid_rows):
            return 0.0

        correlations = cov[valid_rows] / np.sqrt(var_alpha[valid_rows] * var_returns[valid_rows])
        return float(np.mean(correlations)) if correlations.size else 0.0

    def _deduplicate_population(self, population: list[Any]) -> list[Any]:
        seen: set[str] = set()
        deduped = []
        for individual in population:
            if individual.expr_hash in seen:
                continue
            seen.add(individual.expr_hash)
            deduped.append(individual)
        return deduped

    def _filter_by_novelty(
        self,
        population: list[Any],
        signatures_by_hash: dict[str, list[float] | None],
        threshold: float,
    ) -> list[Any]:
        kept = []
        kept_signatures: list[np.ndarray] = []
        for individual in sorted(population, key=lambda item: item.fitness, reverse=True):
            signature = signatures_by_hash.get(individual.expr_hash)
            if not signature:
                kept.append(individual)
                kept_signatures.append(np.array([], dtype=float))
                continue
            candidate = np.asarray(signature, dtype=float)
            is_duplicate = False
            for existing in kept_signatures:
                if existing.size == 0 or candidate.size == 0:
                    continue
                corr = self._signature_corr(candidate, existing)
                if corr >= threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(individual)
                kept_signatures.append(candidate)
        return kept or population[:1]

    def _signature_corr(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        min_len = min(a.size, b.size)
        left = a[:min_len]
        right = b[:min_len]
        if np.std(left) < 1e-12 or np.std(right) < 1e-12:
            return 1.0 if np.allclose(left, right) else 0.0
        return float(abs(np.corrcoef(left, right)[0, 1]))

    def _build_alpha_signature(self, alpha: Any) -> list[float]:
        data = np.nan_to_num(self._to_numpy(alpha), nan=0.0, posinf=0.0, neginf=0.0)
        if data.size == 0:
            return []
        signature = np.concatenate(
            [
                np.mean(data, axis=1),
                np.std(data, axis=1),
                np.mean(data, axis=0),
            ]
        )
        return signature.astype(float).tolist()

    def _to_numpy(self, value: Any) -> np.ndarray:
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().numpy()
        return np.asarray(value, dtype=float)

    def _to_serializable_list(self, value: Any) -> list[Any]:
        return self._to_numpy(value).tolist()
