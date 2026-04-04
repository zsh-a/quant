from __future__ import annotations

import ast
import json
import os
import re
import time
from time import perf_counter
from typing import Any, Dict, List

import httpx
from loguru import logger
from openai import OpenAI

from .compiler import FormulaCompiler
from .dsl import TensorSchema
from .evolution import BreedingSpec, HeuristicLLMBackend
from .operators import OperatorRegistry


GENESIS_SYSTEM_PROMPT = """You are a senior quantitative researcher specializing in crypto alpha discovery.
Strictly obey the provided field and operator library. Output JSON only."""

EVOLUTION_SYSTEM_PROMPT = """You are an automated alpha mutation engine.
Strictly obey the provided field and operator library. Output JSON only."""


class OpenAILLMBackend:
    def __init__(
        self,
        registry: OperatorRegistry | None = None,
        schema: TensorSchema | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature_genesis: float = 0.7,
        temperature_evolution: float = 0.4,
        max_ast_depth: int = 5,
        client: Any | None = None,
        fallback_backend: HeuristicLLMBackend | None = None,
    ):
        self.registry = registry or OperatorRegistry()
        self.schema = schema or TensorSchema.default_market_schema()
        self.compiler = FormulaCompiler(self.registry)
        self.model_name = model_name or os.getenv("ALPHA_LAB_LLM_MODEL", "gpt-4.1-mini")
        self.base_url = base_url or os.getenv("ALPHA_LAB_LLM_BASE_URL")
        self.api_key = api_key or os.getenv("ALPHA_LAB_LLM_API_KEY")
        if not self.api_key and client is None:
            raise ValueError("API key not found. Please set ALPHA_LAB_LLM_API_KEY.")
        self.temperature_genesis = temperature_genesis
        self.temperature_evolution = temperature_evolution
        self.max_ast_depth = max_ast_depth
        self.call_stats = {
            "total_calls": 0,
            "genesis_calls": 0,
            "evolution_calls": 0,
            "total_seconds": 0.0,
            "last_latency_seconds": 0.0,
        }
        self.fallback_backend = fallback_backend or HeuristicLLMBackend(
            registry=self.registry,
            schema=self.schema,
        )
        self.client = client or OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=5.0),
            max_retries=2,
        )

    @property
    def backend_name(self) -> str:
        return "openai"

    def generate_initial_population(self, count: int) -> list[str]:
        from .tracing import tracer

        with tracer.start_span("genesis_pipeline", kind="breed",
                               target_count=count, backend=self.backend_name) as span:
            prompt = self._build_genesis_prompt(count)
            response = self._call_llm(
                system_prompt=GENESIS_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=self.temperature_genesis,
                call_kind="genesis",
            )
            formulas = self._extract_formulas(response)
            fallback_used = len(formulas) < count
            finalized = self._finalize_formulas(
                formulas=formulas,
                target_count=count,
                fallback=lambda needed: self.fallback_backend.generate_initial_population(needed),
            )
            span.set("formulas_extracted", len(formulas))
            span.set("formulas_valid", len(finalized))
            span.set("fallback_used", fallback_used)
            span.set("target_count", count)
            return finalized

    def generate_offspring(self, spec: BreedingSpec, count: int) -> list[str]:
        from .tracing import tracer

        with tracer.start_span("evolution_pipeline", kind="breed",
                               target_count=count, backend=self.backend_name,
                               has_parent_b=bool(spec.parent_b)) as span:
            prompt = self._build_evolution_prompt(spec, count)
            response = self._call_llm(
                system_prompt=EVOLUTION_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=self.temperature_evolution,
                call_kind="evolution",
            )
            formulas = self._extract_formulas(response)
            fallback_used = len(formulas) < count
            finalized = self._finalize_formulas(
                formulas=formulas,
                target_count=count,
                fallback=lambda needed: self.fallback_backend.generate_offspring(spec, needed),
            )
            # Record parent fitness for correlation analysis
            parent_metrics = spec.parent_feedback[0].get("metrics", {}) if spec.parent_feedback else {}
            span.set("formulas_extracted", len(formulas))
            span.set("formulas_valid", len(finalized))
            span.set("fallback_used", fallback_used)
            span.set("parent_a_sharpe", parent_metrics.get("sharpe", 0))
            span.set("parent_a_rank_ic", parent_metrics.get("rank_ic_abs", 0))
            return finalized

    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float, call_kind: str) -> str:
        from .tracing import prompt_hash, tracer

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        with tracer.start_span(call_kind, kind="llm",
                               model=self.model_name,
                               backend=self.backend_name,
                               temperature=temperature,
                               prompt_chars=len(system_prompt) + len(user_prompt),
                               prompt_hash=prompt_hash(system_prompt + user_prompt),
                               input=messages) as span:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                stream=False,
                timeout=90,
            )
            content = response.choices[0].message.content or ""
            span.set_response(response)
            span.set("output", content)

            self.call_stats["total_calls"] += 1
            self.call_stats[f"{call_kind}_calls"] += 1
            self.call_stats["total_seconds"] += span.duration_ms / 1000
            self.call_stats["last_latency_seconds"] = span.duration_ms / 1000
            total_tokens = span.attributes.get("total_tokens") or 0
            self.call_stats.setdefault("total_tokens", 0)
            self.call_stats["total_tokens"] += total_tokens

            return content

    def _build_genesis_prompt(self, count: int) -> str:
        return f"""
# Role
你是一位就职于顶级 Crypto 自营机构的资深量化研究员。你的任务是挖掘高夏普、低相关性的 Alpha 因子公式。

# Environment & Constraints
你只能使用以下字段和算子。严禁发明未列出的变量或函数。

## Features — 核心行情
- Close, High, Low, Open, Volume, Turnover, VWAP, BidAskSpread

## Features — 成交量细节
- TradeCount（成交笔数）, TakerBuyVolume（主动买入量）, TakerBuyQuoteVolume（主动买入额）

## Features — 标记价格（公允价值）
- MarkOpen, MarkHigh, MarkLow, MarkClose

## Features — 期现基差（Premium Index = 期货 − 现货）
- PremiumOpen, PremiumHigh, PremiumLow, PremiumClose

## Features — 市场微观结构
- FundingRate（资金费率）, OI（持仓量）, OIValue（持仓价值 = OI × 价格）
- LongShortRatio（多空比）, TakerLongShortVolRatio（主动买卖比）
- TopTraderLongShortRatio（大户多空比）, TopTraderLongShortPositionRatio（大户持仓多空比）

## Operators
- Abs(x), Log(x), Sign(x), Sqrt(x), Div(x, y)
- Ts_Mean(x, d), Ts_Max(x, d), Ts_Min(x, d), Ts_Rank(x, d), StdDev(x, d), Delay(x, d), Delta(x, d), Corr(x, y, d)
- Ts_Zscore(x, d), Returns_N(x, d), Log_Return(x, d), Decay_Linear(x, d), Ts_EMA(x, d)
- CSRank(x), CSZScore(x), CSDemean(x)
- OIDelta(OI, d), FundingDelta(FundingRate, d), SpreadRatio(BidAskSpread, Close), AdvN(Turnover, d), Amihud(Close, Turnover, d), ATR_N(High, Low, Close, d)
- HLC3(High, Low, Close), OHLC4(Open, High, Low, Close), Where(cond, x, y), Clip(x, lo, hi), FillNA(x, value)

# Task
请生成 {count} 个截然不同的 Alpha 因子，重点关注：
1. 波动率压缩后的突破（ATR / StdDev 收敛 → 方向突破）。
2. 成交量/流动性与价格的背离（TakerBuyVolume vs Close, TradeCount 异常）。
3. 资金费率 + 持仓量 + 动量的三重共振（FundingRate × OIDelta × Returns）。
4. 期现基差（PremiumClose）与趋势/均值回归的关系。
5. 多空情绪极端化（LongShortRatio / TakerLongShortVolRatio 偏离均值时反转）。
6. 大户行为信号（TopTraderLongShortRatio 变化 → 领先指标）。

# Hard Rules
1. 公式必须能被 Python ast.parse(..., mode="eval") 解析。
2. AST 深度必须 <= {self.max_ast_depth}。
3. 优先使用 Div(x, y) 而不是裸除法，以避免 0 除。
4. 不要生成重复或仅参数微调的等价公式。
5. 最终只输出 JSON 数组，不要输出 Markdown。

# Output Format
[
  {{
    "rationale": "简述金融逻辑",
    "formula": "CSRank(Div(Close - Ts_Mean(Close, 10), StdDev(Close, 20)))"
  }}
]
""".strip()

    def _build_evolution_prompt(self, spec: BreedingSpec, count: int) -> str:
        parent_feedback = spec.parent_feedback or [
            {"formula": spec.parent_a, "metrics": {}, "rationale": "Parent A"},
            {"formula": spec.parent_b, "metrics": {}, "rationale": "Parent B"} if spec.parent_b else {},
        ]
        lines = []
        for idx, parent in enumerate([item for item in parent_feedback if item], start=1):
            metrics = parent.get("metrics", {})
            metrics_text = self._format_parent_metrics(metrics)
            diagnosis_text = self._diagnose_parent_metrics(metrics)
            lines.append(
                f"[Parent {idx}]\n"
                f"- Formula: `{parent.get('formula', '')}`\n"
                f"- Performance: {metrics_text}\n"
                f"- Diagnostics: {diagnosis_text}\n"
                f"- Rationale: {parent.get('rationale', 'Use available metrics to refine the factor.')}"
            )
        parent_block = "\n\n".join(lines) if lines else f"[Parent]\n- Formula: `{spec.parent_a}`"
        return f"""
# Context
上一代回测中，以下父代公式表现最好，它们是你的"父代基因"：

{parent_block}

# Task
基于以上父代公式，进行交叉（Crossover）和变异（Mutation），生成 {count} 个新的子代公式。

# Evaluation Priorities
优先保留并强化以下特征：
1. valid/test 维度都稳健，而不只是 train 内表现好。
2. Abs(RankIC)、Sharpe、TestSharpe、Tail-adjusted return 同时改善。
3. 追求"健康活跃度"而不是极低换手；避免 inactive、低覆盖、过低换手、极高换手、train-valid/test 衰减过大。
4. 如果父代 test_sharpe 弱于 valid_sharpe，优先做稳健化、降复杂度和增强泛化，而不是放大原信号。

# Batch Diversity Requirements
1. 这次是批量生成任务，请一次性给出 {count} 个候选。
2. 候选之间必须尽量分散到不同机制，避免只做微小参数改动。
3. 至少覆盖以下几类中的多类：动量、均值回复、波动率压缩、资金费率/OI、流动性/冲击成本、趋势过滤。
4. 如果某个父代存在明显缺陷，请只保留其有价值的局部结构，不要整段照抄。
5. 不要输出重复公式，不要输出仅窗口参数不同但结构几乎相同的一组公式。

# Mutation Rules
1. 参数变异：修改周期 d（如 3, 5, 10, 20, 30）。
2. 算子替换：在 Ts_Mean / Ts_Rank / StdDev / Ts_Zscore / Decay_Linear 之间切换。
3. 逻辑交叉：将动量、波动率、Funding/OI、Spread、ATR、Amihud 等不同基因拼接。
4. 平滑降噪：对高换手因子优先使用 Decay_Linear 或 Ts_Mean；对过低活跃度因子优先增加触发频率和信号覆盖。
5. 成本感知：鼓励显式使用 SpreadRatio、Amihud、ATR_N，但不要为了刷 PnL/Turnover 而把换手压到接近 0。

# Hard Rules
1. 只能使用当前字段和算子库。
2. 输出必须是 JSON 数组。
3. 公式必须能被 Python ast.parse(..., mode="eval") 解析。
4. AST 深度必须 <= {self.max_ast_depth}。
5. 优先使用 Div(x, y) 而不是裸除法。

# Output Format
[
  {{
    "mutation_type": "Crossover",
    "rationale": "为什么这样变异",
    "formula": "Ts_Mean(CSRank(Div(Close - Ts_Min(Low, 20), ATR_N(High, Low, Close, 14))), 3)"
  }}
]
""".strip()

    def _format_parent_metrics(self, metrics: dict[str, Any]) -> str:
        items = [
            ("Sharpe", metrics.get("sharpe", 0.0)),
            ("TestSharpe", metrics.get("test_sharpe", 0.0)),
            ("AbsRankIC", metrics.get("rank_ic_abs", abs(float(metrics.get("rank_ic", 0.0) or 0.0)))),
            ("PnLPerTurnover", metrics.get("pnl_per_turnover", 0.0)),
            ("Turnover", metrics.get("avg_turnover", metrics.get("turnover_penalty", 0.0))),
            ("ActivityScore", metrics.get("activity_score", 0.0)),
            ("TailRatio", metrics.get("tail_ratio", 0.0)),
            ("TailAdjReturn", metrics.get("tail_penalty_adjusted_return", 0.0)),
            ("SignalCoverage", metrics.get("signal_coverage", 0.0)),
            ("ActiveBarRatio", metrics.get("active_bar_ratio", 0.0)),
            ("TrainValidGap", metrics.get("train_valid_gap_penalty", 0.0)),
            ("ValidTestGap", metrics.get("valid_test_gap_penalty", 0.0)),
            ("Complexity", metrics.get("complexity_penalty", 0.0)),
            ("Inactive", metrics.get("inactive", 0.0)),
        ]
        return ", ".join(f"{name}={float(value):.4f}" for name, value in items)

    def _diagnose_parent_metrics(self, metrics: dict[str, Any]) -> str:
        issues: list[str] = []
        if float(metrics.get("inactive", 0.0)) >= 1.0:
            issues.append("inactive factor")
        if float(metrics.get("avg_turnover", 0.0)) < 0.005:
            issues.append("turnover too low")
        if float(metrics.get("train_valid_gap_penalty", 0.0)) > 0.5:
            issues.append("large train-valid gap")
        if float(metrics.get("test_sharpe", 0.0)) + 0.25 < float(metrics.get("sharpe", 0.0)):
            issues.append("test underperforms valid")
        if float(metrics.get("valid_test_gap_penalty", 0.0)) > 0.5:
            issues.append("large valid-test decay")
        if float(metrics.get("test_sharpe", 0.0)) < 0.0:
            issues.append("test sharpe negative")
        if float(metrics.get("turnover_penalty", 0.0)) > 0.5:
            issues.append("turnover too high")
        if float(metrics.get("signal_coverage", 1.0)) < 0.5:
            issues.append("signal coverage too low")
        if float(metrics.get("active_bar_ratio", 1.0)) < 0.2:
            issues.append("active bar ratio too low")
        if float(metrics.get("complexity_penalty", 0.0)) > 0.6:
            issues.append("formula too complex")
        if float(metrics.get("tail_penalty_adjusted_return", 0.0)) < 0.0:
            issues.append("tail-adjusted return negative")
        if float(metrics.get("rank_ic_abs", abs(float(metrics.get("rank_ic", 0.0) or 0.0)))) < 0.01:
            issues.append("cross-sectional IC too weak")
        if not issues:
            return "no major weakness detected"
        return "; ".join(issues)

    def _extract_formulas(self, raw_text: str) -> list[str]:
        formulas: list[str] = []
        cleaned = raw_text.strip()
        for candidate in self._extract_formula_keys(cleaned):
            normalized = self._normalize_formula(candidate)
            if normalized:
                formulas.append(normalized)
        if formulas:
            return formulas
        for line in cleaned.splitlines():
            normalized = self._normalize_formula(line.strip().strip("`"))
            if normalized:
                formulas.append(normalized)
        return formulas

    def _extract_formula_keys(self, raw_text: str) -> list[str]:
        formulas: list[str] = []
        try:
            payload = json.loads(self._strip_markdown_fence(raw_text))
            formulas.extend(self._extract_formulas_from_json(payload))
        except Exception:
            pass
        if formulas:
            return formulas
        pattern = re.compile(r'"formula"\s*:\s*"((?:[^"\\]|\\.)*)"')
        for match in pattern.finditer(raw_text):
            try:
                formulas.append(json.loads(f'"{match.group(1)}"'))
            except Exception:
                formulas.append(match.group(1))
        return formulas

    def _extract_formulas_from_json(self, payload: Any) -> list[str]:
        if isinstance(payload, dict):
            value = payload.get("formula")
            return [value] if isinstance(value, str) else []
        if isinstance(payload, list):
            results: list[str] = []
            for item in payload:
                results.extend(self._extract_formulas_from_json(item))
            return results
        return []

    def _normalize_formula(self, formula: str) -> str | None:
        candidate = formula.strip()
        if not candidate:
            return None
        candidate = re.sub(r"\bDiv\s*\(", "div(", candidate)
        candidate = re.sub(r"\bCSRank\s*\(", "cs_rank(", candidate)
        candidate = re.sub(r"\bCSZScore\s*\(", "cs_zscore(", candidate)
        candidate = re.sub(r"\bCSDemean\s*\(", "cs_demean(", candidate)
        if not self._is_valid_python_expr(candidate):
            return None
        if self._ast_depth(candidate) > self.max_ast_depth:
            return None
        try:
            self.compiler.compile(candidate, self.schema)
        except ValueError:
            return None
        return candidate

    def _finalize_formulas(
        self,
        formulas: list[str],
        target_count: int,
        fallback: Any,
    ) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for formula in formulas:
            if formula in seen:
                continue
            seen.add(formula)
            unique.append(formula)
            if len(unique) >= target_count:
                return unique
        if len(unique) < target_count:
            for formula in fallback(target_count - len(unique)):
                if formula in seen:
                    continue
                seen.add(formula)
                unique.append(formula)
                if len(unique) >= target_count:
                    break
        return unique

    def _is_valid_python_expr(self, formula: str) -> bool:
        try:
            ast.parse(formula, mode="eval")
            return True
        except SyntaxError:
            return False

    def _ast_depth(self, formula: str) -> int:
        tree = ast.parse(formula, mode="eval")

        def walk(node: ast.AST) -> int:
            if isinstance(node, ast.Expression):
                return walk(node.body)
            if isinstance(node, ast.Name | ast.Constant):
                return 1
            if isinstance(node, ast.Call):
                args = [walk(arg) for arg in node.args]
                return 1 + (max(args) if args else 0)
            if isinstance(node, ast.BinOp):
                return 1 + max(walk(node.left), walk(node.right))
            if isinstance(node, ast.UnaryOp):
                return 1 + walk(node.operand)
            if isinstance(node, ast.BoolOp):
                return 1 + max(walk(value) for value in node.values)
            if isinstance(node, ast.Compare):
                parts = [walk(node.left), *(walk(comp) for comp in node.comparators)]
                return 1 + max(parts)
            children = [walk(child) for child in ast.iter_child_nodes(node)]
            return 1 + (max(children) if children else 0)

        return walk(tree)

    def _strip_markdown_fence(self, raw_text: str) -> str:
        cleaned = raw_text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned


def build_default_llm_backend(
    registry: OperatorRegistry,
    schema: TensorSchema,
    backend_name: str = "auto",
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Any:
    requested = (backend_name or "auto").lower()
    if requested == "heuristic":
        return HeuristicLLMBackend(registry=registry, schema=schema)
    resolved_api_key = api_key or os.getenv("ALPHA_LAB_LLM_API_KEY")
    if requested == "openai" or (requested == "auto" and resolved_api_key):
        return OpenAILLMBackend(
            registry=registry,
            schema=schema,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
        )
    return HeuristicLLMBackend(registry=registry, schema=schema)


# ---------------------------------------------------------------------------
# LLMAgent (from alpha_mining) -- A-share / CSI-1000 focused agent
# ---------------------------------------------------------------------------

PROMPT_PORTRAIT_GENERATION = """
Task Description:
Design a high-performance alpha factor for CSI 1000 index.

MANDATORY Operator List (ONLY USE THESE):
- Unary: Abs(x), Log(x), Sign(x), Sqrt(x)
- Time-series: Ts_Mean(x, d), Ts_Std(x, d), Ts_Max(x, d), Ts_Min(x, d), Ts_Rank(x, d), Ts_Zscore(x, d), Ts_EMA(x, d), Ts_DecayLinear(x, d), Ts_Winsorize(x, d, n_std)
- Diff: Delta(x, d), Delay(x, d), Ts_Returns(x, d)
- Binary: Correlation(x, y, d), Covariance(x, y, d), Max(x, y), Min(x, y)
- Logic: Where(condition, x, y)  <-- Use this instead of If
- Cross-sectional: CSRank(x), Scale(x), Power(x, p), Sigmoid(x)
- Available fields: open, high, low, close, volume, amount, vwap

CRITICAL RULES:
1. Syntax MUST be valid Python with BALANCED parentheses.
2. Use '&' for AND, '|' for OR inside Where condition (e.g., Where((close > open) & (volume > 100), 1, 0)).
3. DO NOT use '&&' or 'AND' or 'If' or any undefined functions.
4. Ensure all Ts_* operators have a window 'd' parameter.
5. Always wrap the final result in CSRank().
6. Keep formula complexity reasonable (max 3-4 nested levels).
7. CSRank() takes EXACTLY 1 argument, not 2.

Output JSON: {"name": "...", "description": "...", "formula": "..."}
"""

PROMPT_REFINE_ALPHA = """
Improve this alpha: {formula}
Feedback: {suggestion}
{error_feedback}

MANDATORY SIGNATURES (ONLY USE THESE):
- Ts_Mean(x, d), Ts_Std(x, d), Ts_Rank(x, d), Ts_Zscore(x, d), Ts_EMA(x, d), Ts_DecayLinear(x, d), Ts_Winsorize(x, d, n_std)
- Delta(x, d), Delay(x, d), Correlation(x, y, d), Covariance(x, y, d)
- CSRank(x) - takes EXACTLY 1 argument
- Where(condition, x, y), Max(x, y), Min(x, y)
- Power(x, p), Sigmoid(x), Abs(x), Log(x), Sign(x), Sqrt(x)

CRITICAL SYNTAX RULES:
1. BALANCE all parentheses - count opening '(' and closing ')' must match.
2. Use '&' instead of 'AND' or '&&'.
3. Use '|' instead of 'OR' or '||'.
4. No 'If' statements, use Where().
5. CSRank(x) takes 1 argument, not 2 or more.
6. Keep complexity under control - MAX 3 nested levels, window size 5-60 days.
7. Only use functions listed above - no undefined functions.
8. AVOID excessive smoothing - don't combine Ts_DecayLinear + Ts_Mean together.
9. Prefer simple formulas over complex ones for better generalization.

Provide ONLY the improved formula string (no explanation, no markdown).
"""


class LLMAgent:
    def __init__(
        self,
        model_name: str = None,
        base_url: str = None,
        api_key: str = None
    ):
        self.model_name = model_name or os.getenv("ALPHA_MINING_MODEL", "deepseek-ai/DeepSeek-V3.2")
        base_url = base_url or os.getenv("ALPHA_MINING_BASE_URL", "https://api-inference.modelscope.cn/v1")
        api_key = api_key or os.getenv("ALPHA_MINING_API_KEY")

        if not api_key:
            raise ValueError("API key not found. Please set ALPHA_MINING_API_KEY in .env file")

        # Configure client with explicit timeouts
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(
                connect=10.0,   # 连接超时10秒
                read=60.0,      # 读取超时60秒
                write=10.0,     # 写入超时10秒
                pool=5.0        # 连接池超时5秒
            ),
            max_retries=2       # 最多重试2次
        )

    def generate_alpha(self, forbidden_structures: List[str] = []) -> Dict[str, str]:
        response = self._call_llm(PROMPT_PORTRAIT_GENERATION)
        return self._parse_json_response(response)

    def refine_alpha(self, formula: str, suggestion: str, error_msg: str = None) -> str:
        error_feedback = f"\nERROR IN PREVIOUS FORMULA: {error_msg}\nPlease fix the syntax or undefined name." if error_msg else ""
        prompt = PROMPT_REFINE_ALPHA.format(
            formula=formula,
            suggestion=suggestion,
            error_feedback=error_feedback
        )
        response = self._call_llm(prompt)
        # Robust cleaning
        res = response.strip().split('\n')[0].replace('`', '').replace('formula=', '')
        return res

    def get_refinement_suggestion(self, formula: str, dimension: str, metrics: dict) -> str:
        prompt = f"Alpha: {formula}\nMetrics: RankIC={metrics.get('rank_ic',0):.4f}, IR={metrics.get('ic_ir',0):.4f}\nImprove {dimension}. Give 1 short logic tip."
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> str:
        from .tracing import prompt_hash, tracer

        messages = [{"role": "user", "content": prompt}]
        with tracer.start_span("agent_call", kind="llm",
                               model=self.model_name,
                               backend="llm_agent",
                               temperature=0.1,
                               prompt_chars=len(prompt),
                               prompt_hash=prompt_hash(prompt),
                               input=messages) as span:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    stream=False,
                    timeout=60,
                )
                content = response.choices[0].message.content or ""
                span.set_response(response)
                span.set("output", content)
                return content

            except Exception as e:
                span.set_error(e)
                return ""

    def _parse_json_response(self, response: str) -> Dict[str, str]:
        try:
            clean_res = re.sub(r'```json\s*|\s*```', '', response).strip()
            start = clean_res.find('{')
            end = clean_res.rfind('}') + 1
            return json.loads(clean_res[start:end])
        except Exception:
            return {"name": "error", "description": "error", "formula": "CSRank(close)"}
