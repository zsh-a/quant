"""
跨市场因子迁移 — 将一个市场的 alpha 因子映射到另一个市场并重新评估。

核心流程:
  1. 解析公式 AST，提取使用的字段列表
  2. 根据目标市场的 field_aliases 反向映射字段名
  3. 对无法映射的字段提供降级方案（替换为通用字段或标记不可迁移）
  4. 重新编译并评估
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from loguru import logger

from .dsl import Parser
from .market import get_market_profile

# ---------------------------------------------------------------------------
# 通用字段映射表 — 跨市场的语义等价字段
# ---------------------------------------------------------------------------

_CROSS_MARKET_EQUIVALENTS: dict[str, dict[str, str]] = {
    # crypto 字段 → A-share 等价字段
    "funding_rate": {"a_share": "pctChg"},  # 资金费率 → 日涨幅 (近似 sentiment)
    "open_interest": {"a_share": "volume"},  # 持仓量 → 成交量
    "trade_count": {"a_share": "turn"},  # 成交笔数 → 换手率
    "taker_buy_volume": {"a_share": "volume"},  # 主买量 → 总量
    "long_short_ratio": {"a_share": "turn"},  # 多空比 → 换手率
    "bid_ask_spread": {"a_share": "amount"},  # 买卖价差 → 成交额
    # A-share 字段 → crypto 等价字段
    "peTTM": {"crypto": "volume"},  # 市盈率 → 成交量 (无直接等价)
    "pbMRQ": {"crypto": "volume"},  # 市净率 → 成交量
    "turn": {"crypto": "trade_count"},  # 换手率 → 成交笔数
    "pctChg": {"crypto": "close"},  # 涨幅 → 收盘价 (需用 ts_return)
    "isST": {},  # ST 标记 — crypto 无等价
    "adjfactor": {},  # 复权因子 — crypto 无需
}

# 两个市场共有的基础字段
_UNIVERSAL_FIELDS = {"open", "high", "low", "close", "volume", "amount", "vwap"}


@dataclass
class MigrationResult:
    """因子迁移结果。"""

    original_formula: str
    migrated_formula: str | None
    source_market: str
    target_market: str
    field_mappings: dict[str, str]  # source_field → target_field
    unmappable_fields: list[str]
    is_viable: bool  # 是否所有字段都能映射


def _extract_fields_from_formula(formula: str) -> set[str]:
    """从公式中提取所有字段名（叶子节点标识符）。"""
    try:
        ast = Parser().parse(formula)
    except Exception:
        # 降级: 正则提取
        tokens = set(re.findall(r"\b([a-z_][a-z0-9_]*)\b", formula.lower()))
        # 排除运算符名
        operators = {
            "ts_mean",
            "ts_std",
            "ts_max",
            "ts_min",
            "ts_sum",
            "ts_rank",
            "ts_delta",
            "ts_delay",
            "ts_corr",
            "ts_cov",
            "ts_return",
            "cs_rank",
            "cs_zscore",
            "cs_demean",
            "decay_linear",
            "log",
            "abs",
            "sign",
            "rank",
            "sqrt",
            "power",
            "add",
            "sub",
            "mul",
            "div",
            "max",
            "min",
            "if_else",
        }
        return tokens - operators

    def _walk(node) -> set[str]:
        if node.kind == "field":
            return {node.value}
        result = set()
        for child in node.children or []:
            result |= _walk(child)
        return result

    return _walk(ast)


def _build_reverse_alias_map(market: str) -> dict[str, str]:
    """构建 标准字段名 → 别名 的反向映射。"""
    profile = get_market_profile(market)
    reverse = {}
    for alias, canonical in profile.field_aliases.items():
        if canonical not in reverse:
            reverse[canonical] = alias
    return reverse


def migrate_formula(
    formula: str,
    source_market: str,
    target_market: str,
) -> MigrationResult:
    """将因子公式从源市场迁移到目标市场。

    Args:
        formula: 原始因子公式
        source_market: 源市场 ("crypto" / "a_share")
        target_market: 目标市场

    Returns:
        MigrationResult 包含迁移后的公式和字段映射详情
    """
    fields = _extract_fields_from_formula(formula)
    field_mappings: dict[str, str] = {}
    unmappable: list[str] = []

    # 解析源市场 aliases 获取规范字段名
    source_profile = get_market_profile(source_market)
    source_aliases = source_profile.field_aliases

    for f in fields:
        canonical = source_aliases.get(f, f)  # 别名 → 规范名

        # 1. 通用字段 — 两个市场都有
        if canonical in _UNIVERSAL_FIELDS:
            field_mappings[f] = canonical
            continue

        # 2. 查跨市场等价表
        equiv = _CROSS_MARKET_EQUIVALENTS.get(canonical, {})
        target_field = equiv.get(target_market)
        if target_field:
            field_mappings[f] = target_field
            continue

        # 3. 无法映射
        unmappable.append(f)

    is_viable = len(unmappable) == 0

    # 构建迁移后的公式
    migrated = formula if is_viable else None
    if is_viable:
        migrated = formula
        for src_field, tgt_field in field_mappings.items():
            if src_field != tgt_field:
                migrated = re.sub(rf"\b{re.escape(src_field)}\b", tgt_field, migrated)

    result = MigrationResult(
        original_formula=formula,
        migrated_formula=migrated,
        source_market=source_market,
        target_market=target_market,
        field_mappings=field_mappings,
        unmappable_fields=unmappable,
        is_viable=is_viable,
    )
    logger.info(
        "factor_migration: {} → {} viable={} mappings={} unmappable={}",
        source_market,
        target_market,
        is_viable,
        len(field_mappings),
        unmappable,
    )
    return result
