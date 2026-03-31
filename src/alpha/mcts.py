"""MCTS-based alpha search strategy.

Combines AlphaNode (tree nodes) with MCTSEngine (search algorithm).
Uses the compiled DSL/VM pipeline instead of eval().
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional

import numpy as np
from loguru import logger

from .compiler import FormulaCompiler
from .dataset import AlphaDataset
from .dsl import TensorSchema
from .evaluation import compute_forward_returns, compute_ic_metrics, compute_rank_ic
from .vm import StackVM, TensorStore


class AlphaNode:
    def __init__(self, formula: str, parent: Optional[AlphaNode] = None, c_puct: float = 1.0):
        self.formula = formula
        self.parent = parent
        self.children: list[AlphaNode] = []
        self.c_puct = c_puct

        # MCTS Statistics
        self.visits = 0
        self.value = 0.0
        self.max_value = 0.0
        self.value_sum = 0.0

        # Evaluation Metrics
        self.metrics: dict[str, float] = {}

        # Alpha Info
        self.name = ""
        self.description = ""

    def add_child(self, child: AlphaNode):
        self.children.append(child)

    def update(self, reward: float, use_moving_avg: bool = True):
        self.visits += 1
        self.value_sum += reward
        self.max_value = max(self.max_value, reward)
        if use_moving_avg:
            alpha = 0.1
            self.value = (1 - alpha) * self.value + alpha * reward
        else:
            self.value = max(self.value, reward)

    def get_uct_score(self, c: float | None = None) -> float:
        if self.visits == 0:
            return float("inf")
        c_value = c if c is not None else self.c_puct
        parent_visits = self.parent.visits if self.parent else 1
        exploitation = self.value
        exploration = c_value * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def __repr__(self):
        return f"<AlphaNode {self.formula[:20]}... V={self.value:.3f} N={self.visits}>"


class MCTSEngine:
    """Monte Carlo Tree Search for alpha discovery using compiled DSL/VM execution."""

    def __init__(
        self,
        compiler: FormulaCompiler,
        vm: StackVM,
        schema: TensorSchema,
        llm_agent: Any,
        c_puct: float = 1.0,
        max_iterations: int = 10,
        zoo_threshold: float = 0.05,
    ):
        self.compiler = compiler
        self.vm = vm
        self.schema = schema
        self.llm = llm_agent
        self.c_puct = c_puct
        self.max_iterations = max_iterations
        self.zoo_threshold = zoo_threshold
        self.root: AlphaNode | None = None
        self.alpha_zoo: list[AlphaNode] = []
        self._factor_cache: dict[str, np.ndarray] = {}

    def run(
        self,
        initial_formula: str,
        dataset: AlphaDataset,
        iterations: int | None = None,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ):
        iters = iterations or self.max_iterations
        n_t = dataset.shape()[0]
        train_end = int(n_t * train_ratio)
        val_end = int(n_t * (train_ratio + val_ratio))
        train_idx = list(range(0, train_end))
        val_idx = list(range(train_end, val_end))

        train_ds = dataset.take_indices(train_idx)
        val_ds = dataset.take_indices(val_idx)

        metrics = self._evaluate_formula(initial_formula, train_ds)
        score = self._calculate_score(metrics)

        self.root = AlphaNode(initial_formula, c_puct=self.c_puct)
        self.root.metrics = metrics
        self.root.update(score)

        for i in range(iters):
            logger.info(f"MCTS Iteration {i + 1}/{iters}")
            leaf = self._select(self.root)
            child = self._expand(leaf, train_ds)

            if child:
                self._backpropagate(child, child.value)
                train_ic = child.metrics.get("rank_ic", 0)
                if abs(train_ic) > self.zoo_threshold:
                    val_metrics = self._evaluate_formula(child.formula, val_ds)
                    val_ic = val_metrics.get("rank_ic", 0)
                    if abs(val_ic) > self.zoo_threshold * 0.4:
                        child.metrics["val_rank_ic"] = val_ic
                        self._add_to_zoo(child, dataset)
                        logger.info(f"Added to zoo: Train IC={train_ic:.4f}, Val IC={val_ic:.4f}")

            logger.info(f"MCTS Iteration {i + 1}/{iters} completed. Zoo size: {len(self.alpha_zoo)}")

    def _evaluate_formula(self, formula: str, dataset: AlphaDataset) -> dict[str, float]:
        try:
            program = self.compiler.compile(formula, self.schema)
            store = TensorStore(dataset.fields)
            alpha = self.vm.run(program, store)
            alpha_np = np.asarray(alpha, dtype=float)
            close = dataset.fields["close"]
            metrics = compute_ic_metrics(alpha_np, close, fwd_windows=[1, 5, 10])
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating {formula}: {e}")
            return {"rank_ic": 0.0, "ic_ir": 0.0, "error": str(e)}

    def _select(self, node: AlphaNode) -> AlphaNode:
        current = node
        while current.children:
            valid_children = [c for c in current.children if c.value > -0.5]
            if not valid_children:
                break
            current = max(valid_children, key=lambda c: c.get_uct_score(self.c_puct))
        return current

    def _expand(self, node: AlphaNode, train_ds: AlphaDataset) -> AlphaNode | None:
        ir = node.metrics.get("ic_ir", 0)
        rank_ic = node.metrics.get("rank_ic", 0)

        if random.random() < 0.3:
            dimension = random.choice([
                "Stability (use smoothing or longer windows)",
                "Effectiveness (use volume-price interaction or non-linear operators)",
                "Novelty (explore new alpha space with different operators)",
            ])
        elif ir < 0.4:
            dimension = "Stability (IR is low, use smoothing or longer windows)"
        elif abs(rank_ic) < 0.05:
            dimension = "Effectiveness (RankIC is low, use volume-price interaction or non-linear operators)"
        else:
            dimension = "Novelty (High performance but needs variation to explore new alpha space)"

        suggestion = self.llm.get_refinement_suggestion(node.formula, dimension, node.metrics)

        max_retries = 3
        error_msg = None

        for attempt in range(max_retries):
            new_formula = self.llm.refine_alpha(node.formula, suggestion, error_msg)
            if not new_formula or new_formula == node.formula:
                continue

            metrics = self._evaluate_formula(new_formula, train_ds)

            if "error" in metrics:
                error_msg = metrics["error"]
                logger.warning(f"Attempt {attempt + 1} failed: {error_msg}. Retrying...")
                continue

            child = AlphaNode(new_formula, parent=node, c_puct=self.c_puct)
            child.metrics = metrics
            child.value = self._calculate_score(metrics)
            node.add_child(child)
            logger.info(f"Expanded: {new_formula[:60]}... Score: {child.value:.4f}")
            return child

        dead_child = AlphaNode(f"FAILED_{node.formula[:10]}", parent=node)
        dead_child.value = -1.0
        node.add_child(dead_child)
        return None

    def _backpropagate(self, node: AlphaNode, reward: float):
        current: AlphaNode | None = node
        while current:
            current.update(reward)
            current = current.parent

    def _calculate_score(self, metrics: dict) -> float:
        if "error" in metrics:
            return -1.0
        rank_ic = metrics.get("rank_ic", 0)
        ic_ir = metrics.get("ic_ir", 0)
        ic_decay = metrics.get("ic_decay", 0)
        return rank_ic + 0.1 * ic_ir - 0.05 * max(ic_decay, 0)

    def _add_to_zoo(self, node: AlphaNode, dataset: AlphaDataset):
        if any(node.formula[:40] == existing.formula[:40] for existing in self.alpha_zoo):
            return

        try:
            program = self.compiler.compile(node.formula, self.schema)
            store = TensorStore(dataset.fields)
            factor_values = np.asarray(self.vm.run(program, store), dtype=float)
            factor_flat = factor_values.flatten()
            factor_flat = factor_flat[~np.isnan(factor_flat)]

            for existing in self.alpha_zoo:
                if existing.formula in self._factor_cache:
                    existing_flat = self._factor_cache[existing.formula]
                else:
                    ex_program = self.compiler.compile(existing.formula, self.schema)
                    ex_values = np.asarray(self.vm.run(ex_program, store), dtype=float)
                    existing_flat = ex_values.flatten()
                    existing_flat = existing_flat[~np.isnan(existing_flat)]
                    self._factor_cache[existing.formula] = existing_flat

                min_len = min(len(factor_flat), len(existing_flat))
                if min_len > 100:
                    corr = np.corrcoef(factor_flat[:min_len], existing_flat[:min_len])[0, 1]
                    if abs(corr) > 0.95:
                        return

            self._factor_cache[node.formula] = factor_flat
            self.alpha_zoo.append(node)
        except Exception as e:
            logger.warning(f"Deduplication check failed: {e}, adding anyway")
            self.alpha_zoo.append(node)


# Backward-compatible alias
AlphaMiningMCTS = MCTSEngine
