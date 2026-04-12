"""Strategy registry — maps string names to factory functions.

Each strategy file registers itself at import time via the ``@register_strategy``
decorator.  ``SearchMode`` defines which strategies compose each selectable mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .base import StrategyMeta


# ---------------------------------------------------------------------------
# Search modes — each mode explicitly lists its strategies
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchMode:
    """A selectable search mode shown in the frontend."""
    name: str
    label: str
    brief: str
    strategies: tuple[str, ...]
    params: tuple[str, ...] = ()
    detail: str = ""


_MODES: dict[str, SearchMode] = {}


def register_mode(mode: SearchMode) -> None:
    _MODES[mode.name] = mode


def get_mode(name: str) -> SearchMode | None:
    return _MODES.get(name)


def get_all_modes() -> dict[str, SearchMode]:
    return dict(_MODES)


# --- Built-in modes ---

register_mode(SearchMode(
    name="evolution",
    label="Evolution (LLM + Enum)",
    brief="LLM 驱动的进化搜索 + 程序化枚举",
    detail="Round 0: 枚举种子 → 后续轮次: LLM 进化 + CPCV 评估。",
    strategies=("enumeration", "llm_evolution"),
    params=("popSize", "offspring", "gens", "topK", "nSplits", "enumMax", "enumTopK"),
))

register_mode(SearchMode(
    name="neural",
    label="Neural (Transformer + RL)",
    brief="Transformer 自回归采样 + REINFORCE 策略梯度",
    detail="因果 Transformer 以 RPN 序列采样公式, rank-IC 作为 reward, 纯 neural 搜索。",
    strategies=("neural",),
    params=("gens", "topK", "nSplits", "neuralBatch"),
))

register_mode(SearchMode(
    name="mcts",
    label="MCTS (LLM-Guided Tree Search)",
    brief="枚举种子 + LLM 进化 + MCTS 精炼",
    detail="Round 0 枚举种子, LLM 进化扩充 archive, MCTS 从精英出发树搜索精炼。",
    strategies=("enumeration", "llm_evolution", "mcts"),
    params=("popSize", "offspring", "gens", "topK", "nSplits", "enumMax", "enumTopK"),
))

register_mode(SearchMode(
    name="alpha_forge",
    label="AlphaForge (Surrogate Model)",
    brief="代理模型预测 + Gumbel-Softmax 梯度生成",
    detail="Predictor 学习 IC 分布, Generator 梯度优化生成高质量公式。",
    strategies=("alpha_forge",),
    params=("gens", "topK", "nSplits"),
))

register_mode(SearchMode(
    name="alpha_probe",
    label="AlphaPROBE (DAG Evolution)",
    brief="枚举种子 + LLM 进化 + DAG 贝叶斯检索",
    detail="DAG 建模因子谱系, 贝叶斯后验选择父代, 祖先路径感知 LLM 生成后代。",
    strategies=("enumeration", "llm_evolution", "alpha_probe"),
    params=("popSize", "offspring", "gens", "topK", "nSplits", "enumMax", "enumTopK"),
))


# ---------------------------------------------------------------------------
# Strategy infrastructure & registry
# ---------------------------------------------------------------------------

@dataclass
class StrategyInfra:
    """Typed infrastructure bundle passed to all strategy factories.

    Every field is explicitly named — no ``**kwargs`` leakage.
    """

    compiler: Any       # FormulaCompiler
    vm: Any             # StackVM
    schema: Any         # TensorSchema
    registry: Any       # OperatorRegistry
    llm_backend: Any    # LLM backend instance
    # --- strategy-specific config (with defaults) ---
    neural_sample_batch: int = 4096
    mcts_frequency: int = 1
    enum_max: int = 500
    enum_top_k: int = 30


_REGISTRY: dict[str, tuple[Callable[[StrategyInfra], Any], StrategyMeta]] = {}


def register_strategy(meta: StrategyMeta) -> Callable:
    """Decorator: register a strategy factory under ``meta.registry_name``.

    The decorated function receives a single ``StrategyInfra`` argument.
    """
    def decorator(factory_fn: Callable[[StrategyInfra], Any]) -> Callable:
        _REGISTRY[meta.registry_name] = (factory_fn, meta)
        return factory_fn
    return decorator


def available_strategies() -> list[str]:
    """Return sorted list of registered strategy names."""
    return sorted(_REGISTRY.keys())


def get_strategy_meta(name: str) -> StrategyMeta | None:
    """Return metadata for a single strategy, or None if not found."""
    entry = _REGISTRY.get(name)
    return entry[1] if entry else None


def get_all_meta() -> dict[str, StrategyMeta]:
    """Return metadata for all registered strategies."""
    return {name: entry[1] for name, entry in _REGISTRY.items()}


def build_strategies(names: set[str], infra: StrategyInfra) -> list:
    """Build strategy instances from a set of names.

    All strategies receive the same typed ``StrategyInfra`` bundle.
    """
    strategies = []
    for name in sorted(names):  # deterministic order
        entry = _REGISTRY.get(name)
        if entry is None:
            raise ValueError(
                f"Unknown strategy {name!r}. Available: {available_strategies()}"
            )
        factory_fn, _meta = entry
        strategies.append(factory_fn(infra))
    return strategies
