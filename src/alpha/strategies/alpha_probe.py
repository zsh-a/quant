"""
AlphaPROBE: DAG 贝叶斯检索 + 图感知进化策略.

Paper: "AlphaPROBE: Alpha Mining via Principled Retrieval and On-graph
       Biased Evolution" (Preprint 2026, Guo et al.)

核心思想:
  1. Factor DAG 跟踪因子进化谱系 (节点=因子, 边=父→子)
  2. Bayesian Factor Retriever 通过后验概率选择最优父代:
     - Prior: 归一化质量 × 深度惩罚 × 检索惩罚
     - Likelihood: 值多样性 × 语法多样性 (叶节点) / 性能增益 × 稀疏度 (非叶节点)
  3. DAG-aware Factor Generator 使用 LLM + 完整祖先路径生成子代
  4. 新因子验证后加入 DAG, 完成进化闭环

与 LLMEvolutionStrategy 的区别:
  - 有原则的贝叶斯父代选择 (vs. 随机锦标赛)
  - DAG 图结构感知 (vs. 无结构因子池)
  - 祖先路径上下文生成 (vs. 仅父代公式)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from loguru import logger

from ..search.context import SearchContext
from ..search.evolution import BreedingSpec, Individual
from ..search.pipeline import Lineage
from .base import BaseStrategy, StrategyMeta

# ---------------------------------------------------------------------------
# Factor DAG
# ---------------------------------------------------------------------------


@dataclass
class _FactorNode:
    """DAG 中的因子节点."""

    formula: str
    expr_hash: str
    fitness: float = 0.0
    rank_ic: float = 0.0
    depth: int = 0
    retrieval_count: int = 0
    parent_hashes: list[str] = field(default_factory=list)
    child_hashes: list[str] = field(default_factory=list)


class _FactorDAG:
    """有向无环图: 跟踪因子进化关系."""

    def __init__(self) -> None:
        self._nodes: dict[str, _FactorNode] = {}

    def __len__(self) -> int:
        return len(self._nodes)

    def add(self, ind: Individual) -> _FactorNode:
        """从 Individual 创建或更新节点."""
        h = ind.expr_hash
        if h in self._nodes:
            node = self._nodes[h]
            node.fitness = max(node.fitness, ind.fitness)
            node.rank_ic = ind.metrics.get("rank_ic", node.rank_ic)
            return node

        lineage = ind.lineage
        parent_hashes = []
        if isinstance(lineage, Lineage):
            if lineage.parent_a:
                parent_hashes.append(lineage.parent_a)
            if lineage.parent_b:
                parent_hashes.append(lineage.parent_b)

        depth = 0
        for ph in parent_hashes:
            if ph in self._nodes:
                depth = max(depth, self._nodes[ph].depth + 1)
                self._nodes[ph].child_hashes.append(h)

        node = _FactorNode(
            formula=ind.formula,
            expr_hash=h,
            fitness=ind.fitness,
            rank_ic=ind.metrics.get("rank_ic", 0.0),
            depth=depth,
            parent_hashes=parent_hashes,
        )
        self._nodes[h] = node
        return node

    def get(self, expr_hash: str) -> _FactorNode | None:
        return self._nodes.get(expr_hash)

    @property
    def nodes(self) -> list[_FactorNode]:
        return list(self._nodes.values())

    def is_leaf(self, node: _FactorNode) -> bool:
        return len(node.child_hashes) == 0

    def children(self, node: _FactorNode) -> list[_FactorNode]:
        return [self._nodes[h] for h in node.child_hashes if h in self._nodes]

    def ancestors(self, node: _FactorNode) -> list[_FactorNode]:
        """返回从根到当前节点的祖先路径."""
        path: list[_FactorNode] = []
        visited: set[str] = set()
        current = node
        while current.parent_hashes:
            ph = current.parent_hashes[0]
            if ph in visited or ph not in self._nodes:
                break
            visited.add(ph)
            current = self._nodes[ph]
            path.append(current)
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# Bayesian scoring (Section 4.1 of paper)
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))


def _bayesian_score(
    node: _FactorNode,
    dag: _FactorDAG,
    all_nodes: list[_FactorNode],
    *,
    gamma: float = 0.05,
    omega: float = 0.10,
) -> float:
    """计算因子作为父代的贝叶斯后验分数 (Eq. 4-13)."""
    if not all_nodes:
        return 0.0

    # --- Prior P(F_new): 质量 × 深度惩罚 × 检索惩罚 (Eq. 5) ---
    ics = [abs(n.rank_ic) for n in all_nodes]
    mu = np.mean(ics) if ics else 0.0
    sigma = max(np.std(ics), 1e-8)
    norm_quality = _sigmoid((abs(node.rank_ic) - mu) / sigma)
    depth_penalty = (1 - gamma) ** node.depth
    retrieval_penalty = (1 - omega) ** node.retrieval_count
    prior = norm_quality * depth_penalty * retrieval_penalty

    # --- Likelihood P(D|F_new) ---
    if dag.is_leaf(node):
        # 叶节点: 值多样性 × 语法多样性 (Eq. 6-9)
        val_div = _value_diversity(node, all_nodes)
        syn_div = _syntactic_diversity(node, all_nodes)
        likelihood = val_div * syn_div
    else:
        # 非叶节点: 性能增益 × 稀疏度 (Eq. 10-13)
        children = dag.children(node)
        if not children:
            likelihood = 0.5
        else:
            pg = _performance_gain(node, children)
            spar = _sparsity(node, children)
            likelihood = pg * spar

    return prior * likelihood


def _value_diversity(node: _FactorNode, all_nodes: list[_FactorNode]) -> float:
    """值多样性: 1 - |平均相关性| (Eq. 7 简化版, 用 IC 差异近似)."""
    if len(all_nodes) <= 1:
        return 1.0
    diffs = [abs(abs(node.rank_ic) - abs(n.rank_ic)) for n in all_nodes if n.expr_hash != node.expr_hash]
    return min(1.0, np.mean(diffs) / max(np.std([abs(n.rank_ic) for n in all_nodes]), 1e-8)) if diffs else 1.0


def _syntactic_diversity(node: _FactorNode, all_nodes: list[_FactorNode]) -> float:
    """语法多样性: 归一化编辑距离 (Eq. 9 简化版)."""
    if len(all_nodes) <= 1:
        return 1.0
    dists = []
    f1 = node.formula
    for n in all_nodes:
        if n.expr_hash == node.expr_hash:
            continue
        f2 = n.formula
        # 简化: 用 token 级 Jaccard 距离替代编辑距离
        s1 = set(f1.replace("(", " ").replace(")", " ").replace(",", " ").split())
        s2 = set(f2.replace("(", " ").replace(")", " ").replace(",", " ").split())
        union = s1 | s2
        if union:
            dists.append(1 - len(s1 & s2) / len(union))
    return float(np.mean(dists)) if dists else 1.0


def _performance_gain(parent: _FactorNode, children: list[_FactorNode]) -> float:
    """性能增益: 子代相对于父代的平均质量提升百分比 (Eq. 10)."""
    if not children:
        return 0.0
    parent_ic = abs(parent.rank_ic) + 1e-8
    gains = [(abs(c.rank_ic) - abs(parent.rank_ic)) / parent_ic for c in children]
    return _sigmoid(float(np.mean(gains)) * 5)  # scale to [0, 1]


def _sparsity(parent: _FactorNode, children: list[_FactorNode]) -> float:
    """子代稀疏度: 父子多样性 × 子间多样性 (Eq. 11-13)."""
    if len(children) <= 1:
        return 1.0
    # 父子多样性: 子代 IC 与父代 IC 的差异
    pc_diffs = [abs(abs(c.rank_ic) - abs(parent.rank_ic)) for c in children]
    pc_spar = min(1.0, float(np.mean(pc_diffs)) * 20)
    # 子间多样性
    cc_diffs = []
    for i, ci in enumerate(children):
        for cj in children[i + 1 :]:
            cc_diffs.append(abs(abs(ci.rank_ic) - abs(cj.rank_ic)))
    cc_spar = min(1.0, float(np.mean(cc_diffs)) * 20) if cc_diffs else 1.0
    return pc_spar * cc_spar


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class AlphaPROBEStrategy(BaseStrategy):
    """AlphaPROBE DAG 贝叶斯进化 (Preprint 2026).

    每轮:
    1. 从 DAG 中通过贝叶斯后验选择最优父代
    2. 提取完整祖先路径作为 LLM 上下文
    3. LLM 基于祖先路径生成改进的子代公式
    4. 编译并返回候选因子
    """

    meta: ClassVar[StrategyMeta] = StrategyMeta(
        registry_name="alpha_probe",
        label="AlphaPROBE DAG进化",
        brief="DAG 贝叶斯检索 + 祖先路径感知 LLM 生成 (AlphaPROBE)",
        detail=(
            "将因子池建模为有向无环图, 用贝叶斯后验选择最优父代 "
            "(质量 × 深度惩罚 × 检索惩罚 × 多样性), "
            "LLM 利用完整进化路径生成非冗余改进"
        ),
    )

    def __init__(
        self,
        llm_backend: Any,
        *,
        gamma: float = 0.05,
        omega: float = 0.10,
        top_k_parents: int = 3,
        children_per_parent: int = 5,
    ) -> None:
        self.llm_backend = llm_backend
        self.gamma = gamma
        self.omega = omega
        self.top_k_parents = top_k_parents
        self.children_per_parent = children_per_parent
        self.dag = _FactorDAG()
        self._stats = {"dag_size": 0, "parents_selected": 0}

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def should_activate(self, ctx: SearchContext) -> bool:
        return ctx.round_idx > 0 and (len(ctx.archive) + len(ctx.population)) >= 5

    def generate_candidates(self, ctx: SearchContext) -> list[Individual]:
        # 同步 DAG
        self._sync_dag(ctx)
        if len(self.dag) < 3:
            return []

        # 贝叶斯选择父代
        parents = self._select_parents()
        if not parents:
            return []
        self._stats["parents_selected"] = len(parents)

        # 对每个父代, 用祖先路径生成子代
        all_formulas: list[str] = []
        all_lineages: dict[str, Lineage] = {}
        for parent in parents:
            parent.retrieval_count += 1
            trace = self.dag.ancestors(parent)
            formulas = self._generate_from_parent(ctx, parent, trace)
            for f in formulas:
                all_lineages[f] = Lineage(
                    origin="alpha_probe",
                    parent_a=parent.expr_hash,
                )
            all_formulas.extend(formulas)

        logger.info(
            f"[AlphaPROBE] DAG={len(self.dag)}, "
            f"选择父代={len(parents)}, 生成={len(all_formulas)}"
        )
        return self.compile_and_dedup(
            ctx, all_formulas,
            lineage_fn=lambda f: all_lineages.get(f, Lineage(origin="alpha_probe")),
        )

    def on_evaluation_complete(
        self, ctx: SearchContext, evaluated: list[Individual],
    ) -> None:
        for ind in evaluated:
            self.dag.add(ind)
        self._stats["dag_size"] = len(self.dag)
        super().on_evaluation_complete(ctx, evaluated)

    def get_stats(self) -> dict[str, Any]:
        return {"strategy": self.name, **self._stats}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sync_dag(self, ctx: SearchContext) -> None:
        """从 archive + population 同步 DAG."""
        for ind in ctx.population:
            self.dag.add(ind)
        for ind in ctx.archive.values():
            self.dag.add(ind)

    def _select_parents(self) -> list[_FactorNode]:
        """贝叶斯后验选择 top-k 父代."""
        nodes = self.dag.nodes
        if not nodes:
            return []

        scored = [
            (node, _bayesian_score(node, self.dag, nodes, gamma=self.gamma, omega=self.omega))
            for node in nodes
            if abs(node.rank_ic) > 0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in scored[: self.top_k_parents]]

    def _generate_from_parent(
        self,
        ctx: SearchContext,
        parent: _FactorNode,
        ancestors: list[_FactorNode],
    ) -> list[str]:
        """DAG-aware LLM 生成: 基于祖先路径的定向改进."""
        # 构建祖先路径上下文
        trace_lines: list[str] = []
        for i, anc in enumerate(ancestors):
            trace_lines.append(
                f"  第{i}代: {anc.formula}  (IC={anc.rank_ic:.4f}, fitness={anc.fitness:.2f})"
            )
        trace_lines.append(
            f"  当前: {parent.formula}  (IC={parent.rank_ic:.4f}, fitness={parent.fitness:.2f})"
        )
        trace_text = "\n".join(trace_lines) if trace_lines else parent.formula

        # 已有子代 (避免冗余)
        existing_children = self.dag.children(parent)
        avoid_list = [c.formula for c in existing_children[:5]]
        avoid_text = "\n".join(f"  - {f}" for f in avoid_list) if avoid_list else "无"

        spec = BreedingSpec(
            parent_a=parent.formula,
            parent_b=None,
            objective=(
                f"基于以下因子进化路径, 生成改进版本. "
                f"进化路径:\n{trace_text}\n"
                f"已有变异 (请避免重复):\n{avoid_text}\n"
                f"要求: 保持可解释性, 改进预测能力或稳定性, 探索不同于已有子代的方向"
            ),
            parent_feedback=[
                {
                    "formula": parent.formula,
                    "metrics": {"rank_ic": parent.rank_ic, "fitness": parent.fitness},
                    "rationale": f"DAG深度={parent.depth}, 检索次数={parent.retrieval_count}",
                }
            ],
        )

        try:
            formulas = self.llm_backend.generate_offspring(
                spec, count=self.children_per_parent,
            )
            return formulas if formulas else []
        except Exception as e:
            logger.warning(f"[AlphaPROBE] LLM生成失败: {e}")
            return []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from .registry import StrategyInfra, register_strategy  # noqa: E402


@register_strategy(AlphaPROBEStrategy.meta)
def _build_alpha_probe(infra: StrategyInfra):
    return AlphaPROBEStrategy(llm_backend=infra.llm_backend)
