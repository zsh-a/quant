"""LLM-Guided MCTS framework for formulaic alpha factor mining.

Implements the algorithm from "Navigating the Alpha Jungle: An LLM-Powered
MCTS Framework for Formulaic Factor Mining" (Shi et al., 2025).

Key components:
  - AlphaNode: Tree node with multi-dimensional evaluation scores,
    refinement history, and per-action Q-values.
  - MCTSEngine: Full Algorithm 1 — UCT selection with virtual expansion
    action, dimension-targeted refinement, multi-dimensional evaluation
    with percentile ranking, max Q-value backpropagation, Frequent
    Subtree Avoidance (FSA), and dynamic search budget.
  - MCTSLLMAdapter: Bridges OpenAILLMBackend to the MCTSEngine interface
    with structured prompts for portrait generation, refinement, and
    overfitting risk assessment.
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from loguru import logger

from ...core.compiler import FormulaCompiler
from ...core.dataset import AlphaDataset
from ...core.dsl import TensorSchema
from ...eval.metrics import compute_forward_returns, compute_ic_metrics, compute_rank_ic
from ...core.vm import StackVM, TensorStore


# ---------------------------------------------------------------------------
# Evaluation dimensions (Section 3: Multi-Dimensional Alpha Evaluation)
# ---------------------------------------------------------------------------

EVAL_DIMENSIONS = ("effectiveness", "stability", "turnover", "diversity", "overfitting")
"""The 5 evaluation dimensions used for multi-dimensional feedback."""

E_MAX = 10  # max score per dimension (paper: e_max)


# ---------------------------------------------------------------------------
# AlphaNode — MCTS tree node
# ---------------------------------------------------------------------------


class AlphaNode:
    """A node in the MCTS search tree.

    Each node represents a candidate alpha formula. It stores:
      - Multi-dimensional evaluation scores E_s (Section 3)
      - Per-action Q-values Q(s, a) = max reward in subtree (Eq. 10)
      - Refinement history H(s) for the full path from root
    """

    __slots__ = (
        "formula", "parent", "children", "c_puct",
        "visits", "eval_scores", "alpha_score",
        "name", "description", "metrics",
        "refinement_history", "_child_q_values",
    )

    def __init__(
        self,
        formula: str,
        parent: Optional[AlphaNode] = None,
        c_puct: float = 1.0,
    ):
        self.formula = formula
        self.parent = parent
        self.children: list[AlphaNode] = []
        self.c_puct = c_puct

        # MCTS statistics
        self.visits: int = 0

        # Multi-dimensional evaluation scores E_s = [e_1, ..., e_q]
        # Each in [0, e_max].  Keys match EVAL_DIMENSIONS.
        self.eval_scores: dict[str, float] = {}

        # Aggregate alpha score S(f) = mean(E_s) (Eq. 8)
        self.alpha_score: float = 0.0

        # Raw backtesting metrics (rank_ic, ic_ir, turnover, etc.)
        self.metrics: dict[str, float] = {}

        # Alpha portrait info
        self.name: str = ""
        self.description: str = ""

        # Refinement history H(s): list of refinement step dicts
        # [{parent_formula, child_formula, dimension, suggestion,
        #   score_change, refinement_abstract}, ...]
        self.refinement_history: list[dict[str, Any]] = []

        # Q(s, a_k) for each child action k.
        # Key = child index in self.children, value = max reward.
        self._child_q_values: dict[int, float] = {}

    def add_child(self, child: AlphaNode) -> int:
        """Add a child node. Returns the action index."""
        idx = len(self.children)
        self.children.append(child)
        self._child_q_values[idx] = child.alpha_score
        return idx

    def get_q_value(self, action_idx: int) -> float:
        """Q(s, a) — max reward observed in subtree via this action."""
        return self._child_q_values.get(action_idx, 0.0)

    def update_q_value(self, action_idx: int, reward: float) -> None:
        """Q(s, a) <- max(Q(s, a), reward)  (Eq. 10)"""
        old = self._child_q_values.get(action_idx, -float("inf"))
        self._child_q_values[action_idx] = max(old, reward)

    def get_uct_score(self, action_idx: int, c: float | None = None) -> float:
        """UCT score for an existing child action (Eq. 2).

        UCT(s, a) = Q(s,a) + c * sqrt(ln(N_s) / N_s')
        """
        c_val = c if c is not None else self.c_puct
        q = self.get_q_value(action_idx)
        child = self.children[action_idx]
        if child.visits == 0:
            return float("inf")
        parent_visits = max(self.visits, 1)
        exploration = c_val * math.sqrt(math.log(parent_visits) / child.visits)
        return q + exploration

    def get_virtual_expansion_uct(self, c: float | None = None) -> float:
        """UCT score for the virtual expansion action a_e (Section 3).

        The virtual visit count is N_s' = 1 + |C(s)|, so the expansion
        action becomes less attractive as the node gains more children.
        """
        c_val = c if c is not None else self.c_puct
        parent_visits = max(self.visits, 1)
        virtual_visits = 1 + len(self.children)
        # Q-value for expansion: use the node's own score as estimate
        q = self.alpha_score
        exploration = c_val * math.sqrt(math.log(parent_visits) / virtual_visits)
        return q + exploration

    def build_refinement_history(self) -> list[dict[str, Any]]:
        """Build the full refinement history from root to this node."""
        if self.parent is None:
            return []
        parent_history = self.parent.build_refinement_history()
        return parent_history + self.refinement_history

    def depth(self) -> int:
        """Depth from root (root = 0)."""
        d = 0
        node = self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    @property
    def mean_value(self) -> float:
        return self.alpha_score

    def __repr__(self) -> str:
        return (
            f"<AlphaNode {self.formula[:30]}... "
            f"S={self.alpha_score:.2f} N={self.visits}>"
        )


# ---------------------------------------------------------------------------
# Frequent Subtree Avoidance (FSA) — Section 3
# ---------------------------------------------------------------------------


def _abstract_formula(formula: str) -> str:
    """Abstract away concrete parameter values from a formula.

    Abs(.) replaces numeric constants with a placeholder so that
    structurally identical formulas with different windows map to
    the same abstracted form.  E.g. ts_mean(close, 20) -> ts_mean(close, _).
    """
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return formula

    class _Abstracter(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            if isinstance(node.value, (int, float)):
                return ast.Constant(value="_")
            return node

    transformed = _Abstracter().visit(tree)
    return ast.unparse(transformed)


def _extract_root_genes(formula: str) -> list[str]:
    """Extract root genes G(f) from a formula.

    A root gene is a subtree whose leaves are exclusively raw input
    features (fields).  We extract all Call subtrees satisfying this.
    """
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return []

    genes: list[str] = []

    def _is_field_leaf(node: ast.AST) -> bool:
        """Check if a node is a raw field name."""
        return isinstance(node, ast.Name)

    def _all_leaves_are_fields(node: ast.AST) -> bool:
        """Recursively check if all leaves in this subtree are fields or constants."""
        if isinstance(node, ast.Name):
            return True
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Call):
            return all(_all_leaves_are_fields(a) for a in node.args)
        if isinstance(node, ast.BinOp):
            return _all_leaves_are_fields(node.left) and _all_leaves_are_fields(node.right)
        if isinstance(node, ast.UnaryOp):
            return _all_leaves_are_fields(node.operand)
        return False

    def _visit(node: ast.AST) -> None:
        if isinstance(node, ast.Call) and _all_leaves_are_fields(node):
            abstracted = _abstract_formula(ast.unparse(node))
            genes.append(abstracted)
        for child in ast.iter_child_nodes(node):
            _visit(child)

    _visit(tree.body)
    return genes


def compute_forbidden_subtrees(
    zoo_formulas: list[str],
    top_k: int = 3,
    min_support: float = 0.15,
) -> list[str]:
    """Identify the top-k most frequent closed root genes in the zoo (Eq. 11-12).

    A root gene g is "closed" if none of its immediate supertrees share
    the same support count.  For simplicity, we use frequency as proxy.
    """
    if not zoo_formulas:
        return []

    gene_counter: Counter[str] = Counter()
    for formula in zoo_formulas:
        unique_genes = set(_extract_root_genes(formula))
        for g in unique_genes:
            gene_counter[g] += 1

    n = len(zoo_formulas)
    # Filter by minimum support
    frequent = [
        (gene, count)
        for gene, count in gene_counter.most_common()
        if count / n >= min_support
    ]

    # Take top-k
    return [gene for gene, _ in frequent[:top_k]]


# ---------------------------------------------------------------------------
# Multi-dimensional evaluation (Section 3)
# ---------------------------------------------------------------------------


def _percentile_rank(value: float, zoo_values: list[float]) -> float:
    """R(f, m, F_zoo) = fraction of zoo members with metric < value (Eq. 6)."""
    if not zoo_values:
        return 0.5
    below = sum(1 for v in zoo_values if v < value)
    return below / len(zoo_values)


def compute_multi_dim_scores(
    metrics: dict[str, float],
    zoo_metrics: list[dict[str, float]],
    overfitting_score: float | None = None,
) -> dict[str, float]:
    """Compute multi-dimensional evaluation scores E_s (Eq. 7).

    For each dimension d in {Effectiveness, Stability, Turnover, Diversity}:
        e_d(f) = (1 - R(f, m_d, F_zoo)) * e_max

    Overfitting Risk is assessed separately via LLM (Appendix K).

    Lower turnover is better, so we invert the ranking for that dimension.
    """
    rank_ic = abs(metrics.get("rank_ic", 0))
    ic_ir = abs(metrics.get("ic_ir", 0))
    turnover = metrics.get("turnover_proxy", metrics.get("avg_turnover", 0.5))
    # Diversity: use max correlation with zoo (lower = more diverse)
    diversity_corr = metrics.get("max_zoo_corr", 0.0)

    scores: dict[str, float] = {}

    # Effectiveness: higher rank_ic is better
    zoo_ics = [abs(m.get("rank_ic", 0)) for m in zoo_metrics]
    scores["effectiveness"] = (1 - _percentile_rank(rank_ic, zoo_ics)) * E_MAX

    # Stability: higher ic_ir is better
    zoo_irs = [abs(m.get("ic_ir", 0)) for m in zoo_metrics]
    scores["stability"] = (1 - _percentile_rank(ic_ir, zoo_irs)) * E_MAX

    # Turnover: lower turnover is better, so rank is inverted
    zoo_turnovers = [
        m.get("turnover_proxy", m.get("avg_turnover", 0.5))
        for m in zoo_metrics
    ]
    scores["turnover"] = _percentile_rank(turnover, zoo_turnovers) * E_MAX

    # Diversity: lower correlation is better
    zoo_corrs = [m.get("max_zoo_corr", 0.0) for m in zoo_metrics]
    scores["diversity"] = _percentile_rank(diversity_corr, zoo_corrs) * E_MAX

    # Overfitting Risk: from LLM assessment (0-10 scale, higher = less risk)
    if overfitting_score is not None:
        scores["overfitting"] = float(overfitting_score)
    else:
        scores["overfitting"] = 5.0  # neutral default

    return scores


def aggregate_score(eval_scores: dict[str, float]) -> float:
    """S(f) = (1/|D|) * sum(e_i(f))  (Eq. 8)"""
    if not eval_scores:
        return 0.0
    return sum(eval_scores.values()) / len(eval_scores)


# ---------------------------------------------------------------------------
# MCTSEngine — core search algorithm (Algorithm 1)
# ---------------------------------------------------------------------------


class MCTSEngine:
    """LLM-Guided Monte Carlo Tree Search for alpha discovery.

    Implements Algorithm 1 from the paper with:
      - UCT selection with virtual expansion action (any node expandable)
      - Dimension-targeted refinement via softmax sampling (Eq. 3)
      - Multi-dimensional evaluation with percentile ranking (Eq. 6-8)
      - Backpropagation with max Q-values (Eq. 9-10)
      - Frequent Subtree Avoidance (FSA) (Eq. 11-12)
      - Dynamic search budget allocation
    """

    def __init__(
        self,
        compiler: FormulaCompiler,
        vm: StackVM,
        schema: TensorSchema,
        llm_agent: Any,
        c_puct: float = 1.0,
        max_iterations: int = 10,
        zoo_threshold: float = 0.015,
        # Paper hyperparameters (Section G)
        initial_budget: int = 3,
        budget_increment: int = 1,
        temperature: float = 1.0,
        fsa_top_k: int = 3,
        max_retries: int = 3,
        effectiveness_threshold: float = 0.3,
    ):
        self.compiler = compiler
        self.vm = vm
        self.schema = schema
        self.llm = llm_agent
        self.c_puct = c_puct
        self.max_iterations = max_iterations
        self.zoo_threshold = zoo_threshold

        # Dynamic budget (Section D)
        self.initial_budget = initial_budget
        self.budget_increment = budget_increment

        # Temperature for dimension selection softmax (Eq. 3)
        self.temperature = temperature

        # FSA parameters
        self.fsa_top_k = fsa_top_k

        # LLM retry limit
        self.max_retries = max_retries

        # Zoo effectiveness threshold (Section G)
        self.effectiveness_threshold = effectiveness_threshold

        # Search state
        self.root: AlphaNode | None = None
        self.alpha_zoo: list[AlphaNode] = []
        self._forbidden_subtrees: list[str] = []
        self._factor_cache: dict[str, np.ndarray] = {}
        self.evaluator: Any = None  # FormulaEvaluator, set by MCTSRefinementStrategy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        initial_formula: str,
        dataset: AlphaDataset,
        iterations: int | None = None,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> None:
        """Run the MCTS search starting from an initial formula.

        Implements Algorithm 1 lines 1-40.
        """
        from ...infra.tracing import tracer

        n_t = dataset.shape()[0]
        train_end = int(n_t * train_ratio)
        val_end = int(n_t * (train_ratio + val_ratio))
        train_ds = dataset.take_indices(list(range(0, train_end)))
        val_ds = dataset.take_indices(list(range(train_end, val_end)))

        with tracer.start_span(
            "mcts_paper_search", kind="search",
            dataset_shape=dataset.shape(),
            zoo_threshold=self.zoo_threshold,
            initial_budget=self.initial_budget,
        ) as search_span:
            # --- Initialization (lines 1-6) ---
            metrics = self._evaluate_formula(initial_formula, train_ds)
            if "error" in metrics:
                logger.error("Seed formula evaluation failed: {}", metrics["error"])
                return

            # Compute diversity metric for seed
            metrics["max_zoo_corr"] = 0.0
            zoo_metrics = [n.metrics for n in self.alpha_zoo]
            eval_scores = compute_multi_dim_scores(metrics, zoo_metrics)
            score = aggregate_score(eval_scores)

            self.root = AlphaNode(initial_formula, c_puct=self.c_puct)
            self.root.metrics = metrics
            self.root.eval_scores = eval_scores
            self.root.alpha_score = score
            self.root.visits = 1

            s_max = score  # Track highest score (line 6)
            budget = self.initial_budget  # Dynamic budget B (line 8)
            search_count = 0

            search_span.event(
                "seed_evaluated",
                formula=initial_formula[:60],
                score=round(score, 3),
                rank_ic=round(metrics.get("rank_ic", 0), 5),
            )

            # --- Main loop (lines 8-39) ---
            while search_count < budget:
                with tracer.start_span(
                    "mcts_iteration", kind="mcts",
                    iteration=search_count + 1,
                    budget=budget,
                ) as iter_span:
                    # Selection (line 9)
                    selected, path = self._select(self.root)

                    # Expansion (lines 10-19)
                    child = self._expand(selected, train_ds)

                    if child is None:
                        iter_span.set("expansion_failed", True)
                        search_count += 1
                        continue

                    # Multi-dimensional evaluation (lines 20-21)
                    child_metrics = child.metrics
                    child_metrics["max_zoo_corr"] = self._compute_max_zoo_corr(
                        child.formula, dataset,
                    )
                    zoo_metrics = [n.metrics for n in self.alpha_zoo]
                    overfitting_score = self.llm.assess_overfitting_risk(
                        child.formula,
                        child.build_refinement_history(),
                    )
                    child.eval_scores = compute_multi_dim_scores(
                        child_metrics, zoo_metrics, overfitting_score,
                    )
                    child.alpha_score = aggregate_score(child.eval_scores)

                    # Add to tree (lines 23-24)
                    action_idx = selected.add_child(child)

                    # Backpropagation (lines 26-30)
                    self._backpropagate(path, selected, action_idx, child.alpha_score)
                    child.visits = 1

                    iter_span.set("child_formula", child.formula[:60])
                    iter_span.set("child_score", round(child.alpha_score, 3))
                    iter_span.set("child_rank_ic", round(child_metrics.get("rank_ic", 0), 5))
                    iter_span.set("eval_scores", {
                        k: round(v, 2) for k, v in child.eval_scores.items()
                    })

                    # Zoo update (lines 31-34)
                    if self._passes_effectiveness_check(child, val_ds):
                        self._add_to_zoo(child, dataset)
                        # Update FSA forbidden subtrees
                        self._forbidden_subtrees = compute_forbidden_subtrees(
                            [n.formula for n in self.alpha_zoo],
                            top_k=self.fsa_top_k,
                        )
                        iter_span.event(
                            "zoo_add",
                            formula=child.formula[:60],
                            zoo_size=len(self.alpha_zoo),
                        )

                    # Dynamic budget (lines 35-38)
                    if child.alpha_score > s_max:
                        budget += self.budget_increment
                        s_max = child.alpha_score
                        iter_span.event(
                            "budget_increase",
                            new_budget=budget,
                            new_max=round(s_max, 3),
                        )

                    search_count += 1
                    iter_span.set("zoo_size", len(self.alpha_zoo))
                    iter_span.set("tree_depth", self._tree_depth())
                    iter_span.set("search_count", search_count)
                    iter_span.set("budget", budget)

            # Also check all tree nodes for zoo eligibility (Section G)
            self._scan_tree_for_zoo(self.root, val_ds, dataset)

            search_span.set("final_zoo_size", len(self.alpha_zoo))
            search_span.set("total_nodes", self._tree_size())
            search_span.set("final_budget", budget)

        tracer.flush()

    def get_refined_formulas(self) -> list[str]:
        """Return formulas discovered during the last run."""
        return [node.formula for node in self.alpha_zoo]

    # ------------------------------------------------------------------
    # Selection — UCT with virtual expansion action (Section 3)
    # ------------------------------------------------------------------

    def _select(self, root: AlphaNode) -> tuple[AlphaNode, list[tuple[AlphaNode, int]]]:
        """Select a node for expansion via UCT (line 9).

        Unlike standard MCTS that only expands leaf nodes, any node can be
        selected for expansion via the virtual expansion action a_e.

        Returns:
            (selected_node, path): where path is list of (parent, action_idx)
            pairs from root to selected_node's parent for backpropagation.
        """
        current = root
        path: list[tuple[AlphaNode, int]] = []

        while True:
            if not current.children:
                # Leaf node — must expand here
                break

            # Compute UCT for each existing child action
            best_child_uct = -float("inf")
            best_child_idx = -1
            for i in range(len(current.children)):
                uct = current.get_uct_score(i, self.c_puct)
                if uct > best_child_uct:
                    best_child_uct = uct
                    best_child_idx = i

            # Compute UCT for virtual expansion action a_e
            expansion_uct = current.get_virtual_expansion_uct(self.c_puct)

            # If expansion action wins, expand this node
            if expansion_uct >= best_child_uct:
                break

            # Otherwise descend to best child
            path.append((current, best_child_idx))
            current = current.children[best_child_idx]

        return current, path

    # ------------------------------------------------------------------
    # Expansion — dimension-targeted refinement (Section 3)
    # ------------------------------------------------------------------

    def _expand(
        self,
        node: AlphaNode,
        train_ds: AlphaDataset,
    ) -> AlphaNode | None:
        """Expand a node by generating a refined alpha formula (lines 10-19).

        1. Select target dimension via softmax (Eq. 3)
        2. LLM generates refinement suggestion + new formula
        3. Validate and retry if invalid
        """
        from ...infra.tracing import tracer

        # 1. Dimension-targeted refinement suggestion (lines 10-12)
        target_dim = self._sample_dimension(node)

        # Gather refinement context (line 13)
        context = self._get_refinement_context(node)

        # Select few-shot exemplars from zoo (line 14)
        exemplars = self._select_exemplars(node, target_dim)

        with tracer.start_span(
            "mcts_expand", kind="breed",
            parent_formula=node.formula[:60],
            parent_score=round(node.alpha_score, 3),
            target_dimension=target_dim,
        ) as span:
            # 2. LLM generates suggestion + formula (line 15)
            error_msg = None
            for attempt in range(self.max_retries):
                suggestion, new_formula = self.llm.generate_refined_alpha(
                    parent_formula=node.formula,
                    target_dimension=target_dim,
                    eval_scores=node.eval_scores,
                    context=context,
                    exemplars=exemplars,
                    forbidden_subtrees=self._forbidden_subtrees,
                    error_feedback=error_msg,
                )

                if not new_formula or new_formula == node.formula:
                    span.event("attempt_skip", attempt=attempt + 1, reason="unchanged")
                    error_msg = "Formula unchanged. Generate a DIFFERENT formula."
                    continue

                # 3. Validate formula (lines 16-19)
                metrics = self._evaluate_formula(new_formula, train_ds)

                if "error" in metrics:
                    error_msg = metrics["error"]
                    span.event("attempt_fail", attempt=attempt + 1, error=error_msg[:80])
                    continue

                # Success — create child node
                child = AlphaNode(new_formula, parent=node, c_puct=self.c_puct)
                child.metrics = metrics

                # Build refinement history entry
                score_change = {
                    dim: child.eval_scores.get(dim, 0) - node.eval_scores.get(dim, 0)
                    for dim in EVAL_DIMENSIONS
                } if node.eval_scores else {}

                child.refinement_history = [{
                    "parent_formula": node.formula,
                    "child_formula": new_formula,
                    "dimension": target_dim,
                    "suggestion": suggestion[:200] if suggestion else "",
                    "score_change": score_change,
                }]

                if suggestion:
                    child.name = suggestion.split("\n")[0][:60]

                span.set("child_formula", new_formula[:60])
                span.set("child_rank_ic", round(metrics.get("rank_ic", 0), 5))
                span.set("attempts_used", attempt + 1)
                span.set("success", True)
                return child

            # All retries exhausted
            span.set("success", False)
            span.set("attempts_used", self.max_retries)
            return None

    def _sample_dimension(self, node: AlphaNode) -> str:
        """Sample target dimension for refinement via softmax (Eq. 3).

        P(i*=i|s) = Softmax((e_max * 1_q - E_s) / T)_i

        Dimensions with lower scores have higher probability of being
        selected, guiding refinement toward areas of weakness.
        """
        if not node.eval_scores:
            return random.choice(list(EVAL_DIMENSIONS))

        gaps = []
        dims = []
        for dim in EVAL_DIMENSIONS:
            score = node.eval_scores.get(dim, E_MAX / 2)
            gap = E_MAX - score  # higher gap = more room for improvement
            gaps.append(gap)
            dims.append(dim)

        # Softmax with temperature
        gaps_arr = np.array(gaps) / max(self.temperature, 0.01)
        gaps_arr = gaps_arr - np.max(gaps_arr)  # numerical stability
        probs = np.exp(gaps_arr) / np.sum(np.exp(gaps_arr))

        return np.random.choice(dims, p=probs)

    def _get_refinement_context(self, node: AlphaNode) -> dict[str, Any]:
        """Get refinement context: parent, children, siblings history (line 13)."""
        context: dict[str, Any] = {
            "parent_formula": node.formula,
            "parent_scores": node.eval_scores,
            "refinement_history": node.build_refinement_history(),
            "depth": node.depth(),
        }

        # Sibling info (other children of parent)
        if node.parent is not None:
            siblings = [
                {
                    "formula": c.formula[:60],
                    "score": round(c.alpha_score, 2),
                    "dimension": (
                        c.refinement_history[-1]["dimension"]
                        if c.refinement_history else "unknown"
                    ),
                }
                for c in node.parent.children
                if c is not node
            ]
            context["siblings"] = siblings[:5]

        # Children info
        if node.children:
            context["children"] = [
                {
                    "formula": c.formula[:60],
                    "score": round(c.alpha_score, 2),
                }
                for c in node.children[:5]
            ]

        return context

    def _select_exemplars(
        self,
        node: AlphaNode,
        target_dim: str,
    ) -> list[dict[str, Any]]:
        """Select few-shot exemplars from zoo for the target dimension (line 14).

        For Effectiveness/Stability: filter by correlation, select top-k by score.
        For Diversity: select lowest correlation exemplars.
        For Turnover/Overfitting: zero-shot (no exemplars).
        """
        if not self.alpha_zoo:
            return []

        if target_dim in ("turnover", "overfitting"):
            # Zero-shot for these dimensions (Section D)
            return []

        # Compute correlations with current formula
        zoo_with_corr: list[tuple[AlphaNode, float]] = []
        for zoo_node in self.alpha_zoo:
            corr = self._get_formula_correlation(node.formula, zoo_node.formula)
            zoo_with_corr.append((zoo_node, corr))

        k = 1  # Paper uses 1 few-shot example

        if target_dim in ("effectiveness", "stability"):
            # Filter: remove top-eta% most correlated (Section D, eta=50%)
            zoo_with_corr.sort(key=lambda x: x[1], reverse=True)
            cutoff = max(1, int(len(zoo_with_corr) * 0.5))
            candidates = zoo_with_corr[cutoff:]

            # Select top-k by dimension score
            dim_key = "rank_ic" if target_dim == "effectiveness" else "ic_ir"
            candidates.sort(
                key=lambda x: abs(x[0].metrics.get(dim_key, 0)),
                reverse=True,
            )
            selected = candidates[:k]

        elif target_dim == "diversity":
            # Select lowest correlation (most diverse)
            zoo_with_corr.sort(key=lambda x: x[1])
            selected = zoo_with_corr[:k]

        else:
            selected = zoo_with_corr[:k]

        return [
            {
                "formula": zn.formula,
                "scores": zn.eval_scores,
                "metrics": {
                    k: round(v, 4)
                    for k, v in zn.metrics.items()
                    if k in ("rank_ic", "ic_ir", "turnover_proxy", "avg_turnover")
                },
            }
            for zn, _ in selected
        ]

    def _get_formula_correlation(self, formula_a: str, formula_b: str) -> float:
        """Estimate structural correlation between two formulas.

        Uses cached factor values if available, otherwise falls back to
        simple string similarity.
        """
        cache_a = self._factor_cache.get(formula_a)
        cache_b = self._factor_cache.get(formula_b)

        if cache_a is not None and cache_b is not None:
            min_len = min(len(cache_a), len(cache_b))
            if min_len > 100:
                corr = np.corrcoef(cache_a[:min_len], cache_b[:min_len])[0, 1]
                return abs(float(corr)) if not np.isnan(corr) else 0.0

        # Fallback: rough string-based similarity
        abs_a = _abstract_formula(formula_a)
        abs_b = _abstract_formula(formula_b)
        if abs_a == abs_b:
            return 0.9
        # Simple Jaccard on tokens
        tokens_a = set(re.findall(r"\w+", abs_a))
        tokens_b = set(re.findall(r"\w+", abs_b))
        if not tokens_a or not tokens_b:
            return 0.0
        inter = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return inter / union if union else 0.0

    # ------------------------------------------------------------------
    # Backpropagation (Section 3, lines 26-30)
    # ------------------------------------------------------------------

    def _backpropagate(
        self,
        path: list[tuple[AlphaNode, int]],
        selected: AlphaNode,
        action_idx: int,
        reward: float,
    ) -> None:
        """Backpropagate reward through the path (Eq. 9-10).

        N_s_k <- N_s_k + 1
        Q(s_k, a_k) <- max(Q(s_k, a_k), S(f_new))
        """
        # Update the selected node's action Q-value
        selected.update_q_value(action_idx, reward)
        selected.visits += 1

        # Walk up the path from selected to root
        for parent, act_idx in reversed(path):
            parent.visits += 1
            parent.update_q_value(act_idx, reward)

    # ------------------------------------------------------------------
    # Zoo management (Section 3 + Section G)
    # ------------------------------------------------------------------

    def _passes_effectiveness_check(
        self,
        node: AlphaNode,
        val_ds: AlphaDataset,
    ) -> bool:
        """Check if a node meets the effectiveness criteria for zoo entry (Section G).

        Basic criteria:
          - RankIC >= 0.015
          - RankIR >= 0.3
          - Validated on out-of-sample data
          - Max correlation with zoo < 0.8
        """
        train_ic = abs(node.metrics.get("rank_ic", 0))
        train_ir = abs(node.metrics.get("ic_ir", 0))

        if train_ic < self.zoo_threshold:
            return False
        if train_ir < self.effectiveness_threshold:
            return False

        # Validate on held-out data
        val_metrics = self._evaluate_formula(node.formula, val_ds)
        if "error" in val_metrics:
            return False
        val_ic = abs(val_metrics.get("rank_ic", 0))
        if val_ic < self.zoo_threshold * 0.5:
            return False

        node.metrics["val_rank_ic"] = val_metrics.get("rank_ic", 0)
        node.metrics["val_ic_ir"] = val_metrics.get("ic_ir", 0)

        return True

    def _prepare_store(self, dataset: AlphaDataset) -> TensorStore:
        """Create a TensorStore, converting to torch if VM uses GPU."""
        store = TensorStore(dataset.fields)
        if self.vm.backend == "torch" and self.vm.device is not None:
            store = self.vm._prepare_store(store)
        return store

    def _add_to_zoo(self, node: AlphaNode, dataset: AlphaDataset) -> None:
        """Add a node to the alpha zoo with diversity check (correlation < 0.8)."""
        from ...core.vm import to_numpy
        try:
            program = self.compiler.compile(node.formula, self.schema)
            store = self._prepare_store(dataset)
            factor_values = to_numpy(self.vm.run(program, store)).flatten()
            factor_flat = factor_values[~np.isnan(factor_values)]

            for existing in self.alpha_zoo:
                if existing.formula in self._factor_cache:
                    existing_flat = self._factor_cache[existing.formula]
                else:
                    ex_program = self.compiler.compile(existing.formula, self.schema)
                    existing_flat = to_numpy(self.vm.run(ex_program, store)).flatten()
                    existing_flat = existing_flat[~np.isnan(existing_flat)]
                    self._factor_cache[existing.formula] = existing_flat

                min_len = min(len(factor_flat), len(existing_flat))
                if min_len > 100:
                    corr = np.corrcoef(factor_flat[:min_len], existing_flat[:min_len])[0, 1]
                    if abs(corr) > 0.8:
                        return

            self._factor_cache[node.formula] = factor_flat
            self.alpha_zoo.append(node)
            logger.info(
                "mcts.zoo_add formula={} score={:.3f} zoo_size={}",
                node.formula[:50], node.alpha_score, len(self.alpha_zoo),
            )
        except Exception as e:
            logger.warning("Zoo deduplication check failed: {}, adding anyway", e)
            self.alpha_zoo.append(node)

    def _compute_max_zoo_corr(self, formula: str, dataset: AlphaDataset) -> float:
        """Compute max absolute correlation between formula and zoo members."""
        if not self.alpha_zoo:
            return 0.0

        try:
            from ...core.vm import to_numpy
            program = self.compiler.compile(formula, self.schema)
            store = self._prepare_store(dataset)
            factor_values = to_numpy(self.vm.run(program, store))
            factor_flat = factor_values.flatten()
            factor_flat = factor_flat[~np.isnan(factor_flat)]

            max_corr = 0.0
            for existing in self.alpha_zoo:
                if existing.formula in self._factor_cache:
                    existing_flat = self._factor_cache[existing.formula]
                    min_len = min(len(factor_flat), len(existing_flat))
                    if min_len > 100:
                        corr = np.corrcoef(
                            factor_flat[:min_len], existing_flat[:min_len],
                        )[0, 1]
                        max_corr = max(max_corr, abs(float(corr)) if not np.isnan(corr) else 0.0)
            return max_corr
        except Exception:
            return 0.0

    def _scan_tree_for_zoo(
        self,
        node: AlphaNode,
        val_ds: AlphaDataset,
        full_ds: AlphaDataset,
    ) -> None:
        """After search completes, scan all tree nodes for zoo eligibility (Section G)."""
        if node is None:
            return
        if node not in [n for n in self.alpha_zoo]:
            if self._passes_effectiveness_check(node, val_ds):
                self._add_to_zoo(node, full_ds)
        for child in node.children:
            self._scan_tree_for_zoo(child, val_ds, full_ds)

    # ------------------------------------------------------------------
    # Formula evaluation
    # ------------------------------------------------------------------

    def _evaluate_formula(
        self,
        formula: str,
        dataset: AlphaDataset,
    ) -> dict[str, float]:
        """Evaluate a formula on a dataset and return IC metrics."""
        if self.evaluator is not None:
            return self.evaluator.eval_metrics(formula, fwd_windows=[1, 5, 10])
        try:
            from ...core.vm import to_numpy
            program = self.compiler.compile(formula, self.schema)
            store = self._prepare_store(dataset)
            alpha_np = to_numpy(self.vm.run(program, store))
            close = dataset.fields["close"]
            metrics = compute_ic_metrics(alpha_np, close, fwd_windows=[1, 5, 10])
            return metrics
        except Exception as e:
            logger.debug("Error evaluating {}: {}", formula[:50], e)
            return {"rank_ic": 0.0, "ic_ir": 0.0, "error": str(e)}

    # ------------------------------------------------------------------
    # Tree utilities
    # ------------------------------------------------------------------

    def _tree_depth(self) -> int:
        if self.root is None:
            return 0

        def _depth(n: AlphaNode) -> int:
            if not n.children:
                return 1
            return 1 + max(_depth(c) for c in n.children)

        return _depth(self.root)

    def _tree_size(self) -> int:
        if self.root is None:
            return 0

        def _count(n: AlphaNode) -> int:
            return 1 + sum(_count(c) for c in n.children)

        return _count(self.root)


# ---------------------------------------------------------------------------
# MCTSLLMAdapter — bridges LLM backend to MCTSEngine interface
# ---------------------------------------------------------------------------

# Shared DSL reference for prompts
_FIELDS_REF = """\
open, high, low, close, volume, turnover, vwap, bid_ask_spread,
trade_count, taker_buy_volume, taker_buy_quote_volume,
mark_open, mark_high, mark_low, mark_close,
premium_open, premium_high, premium_low, premium_close,
funding_rate, open_interest, open_interest_value,
long_short_ratio, taker_long_short_vol_ratio,
top_trader_long_short_ratio, top_trader_long_short_position_ratio"""

_OPERATORS_REF = """\
Math: abs(x), log(x), sign(x), sqrt(x), sigmoid(x), neg(x), div(x,y), power(x,p)
Time-series: ts_mean(x,d), ts_std(x,d), ts_max(x,d), ts_min(x,d), ts_rank(x,d),
  ts_zscore(x,d), ts_ema(x,d), decay_linear(x,d), ts_argmax(x,d), ts_argmin(x,d),
  delay(x,d), delta(x,d), returns_n(x,d), log_return(x,d),
  ts_corr(x,y,d), ts_cov(x,y,d)
Cross-sectional: cs_rank(x), cs_zscore(x), cs_demean(x), cs_scale(x)
Domain: oi_delta(open_interest,d), funding_delta(funding_rate,d),
  spread_ratio(bid_ask_spread,close), adv_n(turnover,d),
  amihud(close,turnover,d), atr_n(high,low,close,d), volatility_n(close,d),
  hlc3(high,low,close), ohlc4(open,high,low,close), true_range(high,low,close)
Control: where(cond,x,y), clip(x,lo,hi), fillna(x,val), max(x,y), min(x,y)"""

_DIMENSION_DESCRIPTIONS = {
    "effectiveness": "Effectiveness measures the alpha's core predictive power (RankIC). "
        "Improve by incorporating stronger signals, better feature interactions, or "
        "more informative transformations.",
    "stability": "Stability assesses the consistency of predictive performance over time (IC IR). "
        "Improve by using smoothing (moving averages), longer windows, or noise-resistant "
        "transformations like z-score normalization.",
    "turnover": "Turnover evaluates the trading cost. Lower turnover means less frequent "
        "rebalancing. Improve by using slower-moving indicators, longer lookback windows, "
        "or applying smoothing operators.",
    "diversity": "Diversity quantifies novelty relative to the existing alpha repository. "
        "Improve by exploring different feature combinations, using uncommon operators, "
        "or capturing different market phenomena.",
    "overfitting": "Overfitting Risk assesses whether the formula is overly complex or "
        "tailored to training data. Improve by simplifying the expression, using "
        "well-motivated financial intuition, and avoiding excessive parameter tuning.",
}


class MCTSLLMAdapter:
    """Adapts OpenAILLMBackend to the MCTSEngine's interface.

    Implements the 4 LLM prompt types from the paper (Appendix K):
      1. Alpha portrait generation (Figure 15)
      2. Alpha refinement (Figure 18) — with few-shot exemplars + FSA
      3. Overfitting risk assessment (Figure 17)
      4. Formula correction on validation failure

    MCTSEngine calls:
      - generate_refined_alpha(parent_formula, target_dim, eval_scores,
            context, exemplars, forbidden_subtrees, error_feedback)
            -> (suggestion, formula)
      - assess_overfitting_risk(formula, refinement_history) -> float
    """

    def __init__(self, llm_backend: Any) -> None:
        self.llm_backend = llm_backend

    def _chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.8,
        timeout: int = 60,
    ) -> str:
        """Send a chat completion request to the LLM."""
        try:
            resp = self.llm_backend.client.chat.completions.create(
                model=self.llm_backend.model_name,
                messages=messages,
                temperature=temperature,
                stream=False,
                timeout=timeout,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("MCTSLLMAdapter chat failed: {}", e)
            return ""

    # ------------------------------------------------------------------
    # 1. Generate refined alpha (Figure 18 + Eq. 3-5)
    # ------------------------------------------------------------------

    def generate_refined_alpha(
        self,
        parent_formula: str,
        target_dimension: str,
        eval_scores: dict[str, float],
        context: dict[str, Any],
        exemplars: list[dict[str, Any]],
        forbidden_subtrees: list[str],
        error_feedback: str | None = None,
    ) -> tuple[str, str]:
        """Generate a refinement suggestion and a new formula.

        This is a two-step process (Eq. 4-5):
          1. d_{s,i*} ~ p_LLM(.|s, i*, F_zoo)  — refinement suggestion
          2. f_new ~ p_LLM(.|d_{s,i*}, f_s)    — concrete formula

        Returns (suggestion, formula).
        """
        dim_desc = _DIMENSION_DESCRIPTIONS.get(target_dimension, "")

        # Build score summary
        score_lines = []
        for dim in EVAL_DIMENSIONS:
            s = eval_scores.get(dim, 0)
            score_lines.append(f"  {dim}: {s:.1f}/{E_MAX}")
        scores_text = "\n".join(score_lines)

        # Build refinement history text
        history = context.get("refinement_history", [])
        history_text = ""
        if history:
            steps = []
            for i, h in enumerate(history[-5:]):  # Last 5 steps
                steps.append(
                    f"  Step {i+1}: {h.get('parent_formula', '?')[:40]} -> "
                    f"{h.get('child_formula', '?')[:40]} "
                    f"[{h.get('dimension', '?')}]"
                )
            history_text = "Refinement history:\n" + "\n".join(steps)

        # Build sibling info
        siblings_text = ""
        siblings = context.get("siblings", [])
        if siblings:
            sib_lines = [
                f"  - {s['formula']} (score={s['score']}, dim={s.get('dimension', '?')})"
                for s in siblings[:3]
            ]
            siblings_text = "Sibling attempts (avoid similar formulas):\n" + "\n".join(sib_lines)

        # Build exemplar text
        exemplar_text = ""
        if exemplars:
            ex_lines = []
            for ex in exemplars:
                ex_lines.append(
                    f"  Formula: {ex['formula']}\n"
                    f"  Metrics: {ex.get('metrics', {})}"
                )
            exemplar_text = (
                f"Reference alphas with high {target_dimension} scores:\n"
                + "\n".join(ex_lines)
            )

        # Build FSA constraint
        fsa_text = ""
        if forbidden_subtrees:
            fsa_text = (
                "IMPORTANT: When designing the formula, try to AVOID including "
                "the following frequent sub-expressions:\n"
                + "\n".join(f"  - {s}" for s in forbidden_subtrees)
            )

        # Error feedback for retry
        error_text = ""
        if error_feedback:
            error_text = (
                f"\nPrevious attempt failed with error: {error_feedback}\n"
                "Fix the error and generate a valid formula."
            )

        prompt = f"""\
Task: Refine an alpha factor for quantitative investment.

Available Data Fields: {_FIELDS_REF}

Available Operators: {_OPERATORS_REF}

Original alpha expression:
  {parent_formula}

Current evaluation scores (0-{E_MAX}, higher is better):
{scores_text}

Target dimension for improvement: {target_dimension}
{dim_desc}

{history_text}
{siblings_text}
{exemplar_text}
{fsa_text}
{error_text}

Alpha Requirements:
1. The alpha value should be dimensionless (unitless).
2. Use at least two distinct operators for sufficient complexity.
3. All lookback windows must be concrete integer values (e.g., 20, not a variable).
4. No more than 3 nested levels of operators.
5. Output must be a valid Python expression using ONLY the listed fields and operators.
6. Use descriptive variable logic that captures a clear financial intuition.

Provide your response in the following JSON format:
{{
  "name": "short_descriptive_name",
  "description": "One sentence explaining the investment intuition.",
  "suggestion": "1-2 sentence refinement suggestion for the {target_dimension} dimension.",
  "formula": "the_refined_alpha_formula_as_python_expression"
}}

Output ONLY the JSON, no markdown fences or extra text."""

        messages = [{"role": "user", "content": prompt}]
        raw = self._chat(messages, temperature=0.8)

        # Parse response
        suggestion, formula = self._parse_refinement_response(raw, parent_formula)
        return suggestion, formula

    def _parse_refinement_response(
        self,
        raw: str,
        fallback_formula: str,
    ) -> tuple[str, str]:
        """Parse JSON response from refinement LLM call."""
        if not raw:
            return "", ""

        # Try to extract JSON from response
        raw_clean = raw.strip()
        # Remove markdown fences if present
        if raw_clean.startswith("```"):
            lines = raw_clean.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw_clean = "\n".join(lines)

        try:
            data = json.loads(raw_clean)
            suggestion = data.get("suggestion", "")
            formula = data.get("formula", "")
            # Basic cleanup
            formula = formula.strip().strip("`\"' \n")
            return suggestion, formula
        except json.JSONDecodeError:
            # Fallback: try to find a formula-like expression
            # Look for formula field
            match = re.search(r'"formula"\s*:\s*"([^"]+)"', raw)
            if match:
                formula = match.group(1).strip()
                suggestion_match = re.search(r'"suggestion"\s*:\s*"([^"]+)"', raw)
                suggestion = suggestion_match.group(1) if suggestion_match else ""
                return suggestion, formula

            # Last resort: look for any parenthesized expression
            match = re.search(r'[a-z_]+\([^)]*\)', raw)
            if match:
                return "", match.group(0)

            return "", ""

    # ------------------------------------------------------------------
    # 2. Overfitting risk assessment (Figure 17)
    # ------------------------------------------------------------------

    def assess_overfitting_risk(
        self,
        formula: str,
        refinement_history: list[dict[str, Any]],
    ) -> float:
        """Assess the overfitting risk of an alpha formula via LLM (Appendix K).

        Returns a score from 0 (high risk) to 10 (low risk).
        """
        # Build refinement history text
        history_text = "No refinement history (root node)."
        if refinement_history:
            steps = []
            for i, h in enumerate(refinement_history[-5:]):
                steps.append(
                    f"Step {i+1}: {h.get('parent_formula', '?')[:50]} -> "
                    f"{h.get('child_formula', '?')[:50]} "
                    f"[dimension: {h.get('dimension', '?')}]"
                )
            history_text = "\n".join(steps)

        prompt = f"""\
Task: Critical Alpha Overfitting Risk Assessment

Critically evaluate the overfitting risk and generalization potential of this
quantitative investment alpha, based on its expression and refinement history.

Alpha Expression:
  {formula}

Refinement History:
  {history_text}

Evaluation Criteria:
1. Justified Rationale vs. Complexity: Is the complexity justified by a clear
   economic rationale, or does it seem arbitrary/excessive?
2. Principled Development vs. Data Dredging: Does the refinement history indicate
   hypothesis-driven improvements, or excessive optimization and parameter tweaks?
3. Transparency vs. Opacity: Is the logic reasonably interpretable?

Scoring:
- Assign a single Overfitting Risk Score from 0 to 10.
- 10 = Very Low Risk (high confidence in generalization)
- 0 = Very High Risk (low confidence in generalization)

Output ONLY valid JSON:
{{"reason": "one-sentence justification", "score": <integer 0-10>}}"""

        messages = [{"role": "user", "content": prompt}]
        raw = self._chat(messages, temperature=0.1, timeout=30)

        try:
            # Extract JSON
            raw_clean = raw.strip()
            if raw_clean.startswith("```"):
                lines = raw_clean.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw_clean = "\n".join(lines)
            data = json.loads(raw_clean)
            score = int(data.get("score", 5))
            return max(0, min(E_MAX, score))
        except (json.JSONDecodeError, ValueError, TypeError):
            # Try regex extraction
            match = re.search(r'"score"\s*:\s*(\d+)', raw)
            if match:
                return max(0, min(E_MAX, int(match.group(1))))
            return 5.0  # neutral default

