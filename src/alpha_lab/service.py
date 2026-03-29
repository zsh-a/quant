from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from time import perf_counter
from typing import Any

import numpy as np

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
    def __init__(self, schema: TensorSchema | None = None):
        self.registry = DSLRegistry()
        self.schema = schema or TensorSchema.default_market_schema()
        self.compiler = FormulaCompiler(self.registry)
        self.vm = StackVM()
        self.evolution = EvolutionEngine(registry=self.registry, compiler=self.compiler, schema=self.schema)
        self.signal_transformer = SignalTransformer()
        self.rule_overlay = RuleOverlay()
        self.execution = ExecutionSimulator()
        self.dataset_loader = CryptoMinuteDatasetLoader()
        self.persistence = AlphaLabPersistence()
        self.validator = CPCVValidator()
        self._program_cache: dict[str, BytecodeProgram] = {}

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
        interval: str = "1m",
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
        interval: str = "1m",
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
        interval: str = "1m",
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
        interval: str = "1m",
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
        dataset = self.dataset_loader.load(
            provider=provider,
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            min_quote_volume=min_quote_volume,
            blocked_utc_hours=blocked_utc_hours,
        )
        validation_plan = self._build_validation_plan(
            dataset,
            n_splits=n_splits,
            purge_window=purge_window,
            embargo_window=embargo_window,
        )
        seeds = seeds or DEFAULT_DB_SEEDS
        population = self._deduplicate_population(self.evolution.initialize(seeds, population_size))
        generation_summaries = []
        details_by_hash: dict[str, dict[str, Any]] = {}
        lineage_edges: list[dict[str, Any]] = []
        last_metrics_by_hash: dict[str, dict[str, float]] = {}
        final_population: list[Any] = population

        for generation in range(generations):
            split_results = self.evaluate_population_on_validation_plan(population, validation_plan["folds"])
            metrics_by_hash = split_results["metrics_by_hash"]
            signatures_by_hash = split_results["signatures_by_hash"]
            details_by_hash.update(split_results["details_by_hash"])
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
            last_metrics_by_hash = metrics_by_hash
            final_population = survivors
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
            if generation == generations - 1:
                break
            offspring = self._deduplicate_population(self.evolution.breed(survivors, offspring_count))
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
            if not offspring:
                final_population = survivors
                break
            population = survivors + offspring

        ranked_population = sorted(final_population, key=lambda item: item.fitness, reverse=True)
        result = {
            "dataset": {
                "provider": dataset.provider,
                "interval": dataset.interval,
                "symbols": dataset.symbols,
                "shape": dataset.shape(),
            },
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
        if persist:
            run = self.persistence.save_run(result, run_name=run_name or "search_db")
            zoo_paths = self.persistence.save_zoo_entries(result["top_results"], run.run_id)
            result["persistence"] = {
                "run_id": run.run_id,
                "run_path": run.run_path,
                "zoo_paths": zoo_paths,
            }
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
    ) -> dict[str, dict[str, Any]]:
        if not population:
            return {
                "metrics_by_hash": {},
                "signatures_by_hash": {},
                "details_by_hash": {},
            }

        store = TensorStore(dataset.fields)
        programs = [individual.program for individual in population]
        alphas = self.vm.run_batch(programs, store)
        metrics_by_hash: dict[str, dict[str, float]] = {}
        signatures_by_hash: dict[str, list[float]] = {}
        details_by_hash: dict[str, dict[str, Any]] = {}

        for individual, alpha in zip(population, alphas):
            evaluation = self._build_dataset_evaluation(
                individual.program,
                alpha,
                dataset,
                store,
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
        split_metrics_by_hash: dict[str, dict[str, list[dict[str, float]]]] = {}
        split_signatures_by_hash: dict[str, dict[str, list[list[float]]]] = {}
        details_by_hash: dict[str, dict[str, Any]] = {}

        for fold_entry in validation_folds:
            fold = fold_entry["fold"]
            fold_results: dict[str, dict[str, dict[str, Any]]] = {}
            for split_name, split_dataset in fold_entry["datasets"].items():
                split_results = self.evaluate_population_from_dataset(population, split_dataset)
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

        return {
            "metrics_by_hash": metrics_by_hash,
            "signatures_by_hash": signatures_by_hash,
            "details_by_hash": details_by_hash,
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
        close: Any,
        summary: dict[str, float],
    ) -> dict[str, float]:
        alpha_np = self._to_numpy(alpha)
        close_np = self._to_numpy(close)
        forward_returns = np.zeros_like(close_np)
        forward_returns[:-1] = close_np[1:] / (close_np[:-1] + 1e-12) - 1.0
        rank_ic = self._mean_cross_sectional_correlation(alpha_np[:-1], forward_returns[:-1])
        avg_turnover = float(summary.get("avg_turnover", 0.0))
        total_return = float(summary.get("total_return", 0.0))
        volatility = float(summary.get("volatility", 0.0))
        metrics = dict(summary)
        metrics.update(
            {
                "rank_ic": rank_ic,
                "pnl_per_turnover": total_return / (avg_turnover + 1e-12),
                "stability": 1.0 / (volatility + 1e-12),
                "tail_penalty_adjusted_return": total_return - float(summary.get("max_drawdown", 0.0)),
                "turnover_penalty": avg_turnover,
                "complexity_penalty": 0.0,
                "train_valid_gap_penalty": 0.0,
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
        if program is None:
            program = self.compiler.compile(formula, self.schema)
            self._program_cache[formula] = program
        return program

    def _dataset_summary(self, dataset: AlphaDataset) -> dict[str, Any]:
        return {
            "provider": dataset.provider,
            "interval": dataset.interval,
            "symbols": dataset.symbols,
            "timestamps": dataset.timestamps[:5],
            "shape": dataset.shape(),
        }

    def _build_dataset_evaluation(
        self,
        program: BytecodeProgram,
        alpha: Any,
        dataset: AlphaDataset,
        store: TensorStore,
    ) -> dict[str, Any]:
        market_ctx = MarketContext(
            liquidity_mask=dataset.liquidity_mask,
            session_mask=dataset.session_mask,
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
            funding_rate=store.get_field("funding_rate"),
        )
        metrics = self._build_fitness_metrics(alpha, store.get_field("close"), result.summary())
        return {
            "program": program.to_dict(),
            "metrics": metrics,
            "alpha_signature": self._build_alpha_signature(alpha),
            "alpha_tail": self._to_serializable_list(alpha[-5:]),
            "weights_tail": self._to_serializable_list(wrapped_weights[-5:]),
            "equity_tail": self._to_serializable_list(result.equity_curve[-5:]),
            "backend": self.vm.backend,
            "device": str(self.vm.device) if self.vm.device is not None else "numpy",
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

    def get_lineage(self, run_id: str) -> dict[str, Any]:
        run = self.persistence.load_run(run_id)
        return {
            "run_id": run_id,
            "lineage": run.get("lineage", []),
            "top_results": run.get("top_results", []),
        }

    def _mean_cross_sectional_correlation(self, alpha: np.ndarray, returns: np.ndarray) -> float:
        valid_rows = []
        for alpha_row, return_row in zip(alpha, returns):
            mask = ~np.isnan(alpha_row) & ~np.isnan(return_row)
            if mask.sum() < 2:
                continue
            a = alpha_row[mask]
            r = return_row[mask]
            if np.std(a) < 1e-12 or np.std(r) < 1e-12:
                continue
            valid_rows.append(float(np.corrcoef(a, r)[0, 1]))
        return float(np.mean(valid_rows)) if valid_rows else 0.0

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
