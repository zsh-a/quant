from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

import numpy as np

from .compiler import FormulaCompiler
from .dataset import AlphaDataset, CryptoMinuteDatasetLoader
from .dsl import DSLRegistry, TensorSchema
from .evolution import EvolutionEngine
from .persistence import AlphaLabPersistence
from .risk import CostModel, ExecutionSimulator, MarketContext, RuleOverlay, SignalTransformer
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

    def list_operators(self) -> list[dict[str, Any]]:
        return [asdict(spec) for spec in self.registry.list_operators()]

    def validate_formula(self, formula: str) -> dict[str, Any]:
        report = self.registry.validate_formula(formula, self.schema)
        return asdict(report)

    def compile_formula(self, formula: str) -> dict[str, Any]:
        program = self.compiler.compile(formula, self.schema)
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
        program = self.compiler.compile(formula, self.schema)
        alpha = np.asarray(self.vm.run(program, store), dtype=float)

        market_ctx = MarketContext(
            liquidity_mask=np.asarray(liquidity_mask, dtype=bool) if liquidity_mask is not None else None,
            session_mask=np.asarray(session_mask, dtype=bool) if session_mask is not None else None,
        )
        target_weights = self.signal_transformer.to_target_weights(alpha, market_ctx)
        wrapped_weights = self.rule_overlay.apply(target_weights, market_ctx)
        result = self.execution.simulate(wrapped_weights, {"close": store.get_field("close")}, CostModel())
        return {
            "program": program.to_dict(),
            "alpha": alpha.tolist(),
            "weights": wrapped_weights.tolist(),
            "metrics": result.summary(),
        }

    def evaluate_formula_from_dataset(
        self,
        formula: str,
        dataset: AlphaDataset,
    ) -> dict[str, Any]:
        store = TensorStore(dataset.fields)
        program = self.compiler.compile(formula, self.schema)
        alpha = np.asarray(self.vm.run(program, store), dtype=float)

        market_ctx = MarketContext(
            liquidity_mask=dataset.liquidity_mask,
            session_mask=dataset.session_mask,
        )
        target_weights = self.signal_transformer.to_target_weights(alpha, market_ctx)
        wrapped_weights = self.rule_overlay.apply(target_weights, market_ctx)
        result = self.execution.simulate(
            wrapped_weights,
            {"close": store.get_field("close")},
            CostModel(),
            funding_rate=store.get_field("funding_rate"),
        )
        metrics = self._build_fitness_metrics(alpha, store.get_field("close"), result.summary())
        return {
            "dataset": {
                "provider": dataset.provider,
                "interval": dataset.interval,
                "symbols": dataset.symbols,
                "timestamps": dataset.timestamps[:5],
                "shape": dataset.shape(),
            },
            "program": program.to_dict(),
            "metrics": metrics,
            "alpha_signature": self._build_alpha_signature(alpha),
            "alpha_tail": alpha[-5:].tolist(),
            "weights_tail": wrapped_weights[-5:].tolist(),
            "equity_tail": result.equity_curve[-5:].tolist(),
        }

    def evaluate_formula_from_db(
        self,
        formula: str,
        provider: str,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
        min_quote_volume: float = 0.0,
    ) -> dict[str, Any]:
        dataset = self.dataset_loader.load(
            provider=provider,
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            min_quote_volume=min_quote_volume,
        )
        return self.evaluate_formula_from_dataset(formula=formula, dataset=dataset)

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
    ) -> dict[str, Any]:
        dataset = self.dataset_loader.load(
            provider=provider,
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            min_quote_volume=min_quote_volume,
        )
        dataset_splits = self._split_dataset(dataset)
        seeds = seeds or DEFAULT_DB_SEEDS
        population = self._deduplicate_population(self.evolution.initialize(seeds, population_size))
        generation_summaries = []
        details_by_hash: dict[str, dict[str, Any]] = {}
        combined: list[Any] = []
        lineage_edges: list[dict[str, Any]] = []

        for generation in range(generations):
            metrics_by_hash = {}
            signatures_by_hash = {}
            for individual in population:
                split_eval = self.evaluate_formula_across_splits(individual.formula, dataset_splits)
                metrics_by_hash[individual.expr_hash] = split_eval["fitness_metrics"]
                signatures_by_hash[individual.expr_hash] = split_eval["alpha_signature"]
                details_by_hash[individual.expr_hash] = split_eval
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
            generation_summaries.append(
                {
                    "generation": generation,
                    "population_size": len(scored_population),
                    "novelty_threshold": novelty_threshold,
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
                combined = survivors
                break
            population = survivors + offspring
            combined = population

        combined = sorted(combined, key=lambda item: item.fitness, reverse=True)
        result = {
            "dataset": {
                "provider": dataset.provider,
                "interval": dataset.interval,
                "symbols": dataset.symbols,
                "shape": dataset.shape(),
            },
            "splits": {name: split.shape() for name, split in dataset_splits.items()},
            "generations": generation_summaries,
            "lineage": lineage_edges,
            "top_results": [
                {
                    "formula": individual.formula,
                    "expr_hash": individual.expr_hash,
                    "fitness": individual.fitness,
                    "lineage": individual.lineage,
                    "metrics": metrics_by_hash.get(individual.expr_hash, {}),
                }
                for individual in combined[:top_k]
            ],
            "evaluations": {
                expr_hash: {
                    "formula": next((item.formula for item in combined if item.expr_hash == expr_hash), None),
                    "metrics": details["fitness_metrics"],
                    "split_metrics": details["split_metrics"],
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
        split_metrics = {}
        split_signatures = {}
        for split_name, split_dataset in dataset_splits.items():
            evaluation = self.evaluate_formula_from_dataset(formula, split_dataset)
            split_metrics[split_name] = evaluation["metrics"]
            split_signatures[split_name] = evaluation["alpha_signature"]

        train = split_metrics.get("train", {})
        valid = split_metrics.get("valid", {})
        test = split_metrics.get("test", {})
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
                "train_valid_gap_penalty": abs(float(train.get("sharpe", 0.0)) - float(valid.get("sharpe", 0.0))),
                "test_sharpe": float(test.get("sharpe", 0.0)),
                "train_sharpe": float(train.get("sharpe", 0.0)),
                "valid_sharpe": float(valid.get("sharpe", 0.0)),
            }
        )
        return {
            "fitness_metrics": fitness_metrics,
            "split_metrics": split_metrics,
            "alpha_signature": split_signatures.get("valid") or split_signatures.get("train"),
        }

    def _split_dataset(
        self,
        dataset: AlphaDataset,
        train_ratio: float = 0.6,
        valid_ratio: float = 0.2,
    ) -> dict[str, AlphaDataset]:
        total = len(dataset.timestamps)
        if total < 6:
            return {
                "train": dataset,
                "valid": dataset,
                "test": dataset,
            }
        train_end = max(int(total * train_ratio), 1)
        valid_end = max(int(total * (train_ratio + valid_ratio)), train_end + 1)
        valid_end = min(valid_end, total)
        return {
            "train": dataset.slice_by_index(0, train_end),
            "valid": dataset.slice_by_index(train_end, valid_end),
            "test": dataset.slice_by_index(valid_end, total),
        }

    def _build_fitness_metrics(
        self,
        alpha: np.ndarray,
        close: np.ndarray,
        summary: dict[str, float],
    ) -> dict[str, float]:
        forward_returns = np.zeros_like(close)
        forward_returns[:-1] = close[1:] / (close[:-1] + 1e-12) - 1.0
        rank_ic = self._mean_cross_sectional_correlation(alpha[:-1], forward_returns[:-1])
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

    def _build_alpha_signature(self, alpha: np.ndarray) -> list[float]:
        data = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
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
