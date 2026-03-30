from __future__ import annotations

import ast
import json
import os
import re
from time import perf_counter
from typing import Any

import httpx
from loguru import logger
from openai import OpenAI

from .compiler import FormulaCompiler
from .dsl import DSLRegistry, TensorSchema
from .evolution import BreedingSpec, HeuristicLLMBackend


GENESIS_SYSTEM_PROMPT = """You are a senior quantitative researcher specializing in crypto alpha discovery.
Strictly obey the provided field and operator library. Output JSON only."""

EVOLUTION_SYSTEM_PROMPT = """You are an automated alpha mutation engine.
Strictly obey the provided field and operator library. Output JSON only."""


class OpenAILLMBackend:
    def __init__(
        self,
        registry: DSLRegistry | None = None,
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
        self.registry = registry or DSLRegistry()
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
        logger.info(
            "alpha_lab.llm genesis start backend={} model={} target_count={}",
            self.backend_name,
            self.model_name,
            count,
        )
        prompt = self._build_genesis_prompt(count)
        response = self._call_llm(
            system_prompt=GENESIS_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=self.temperature_genesis,
            call_kind="genesis",
        )
        formulas = self._extract_formulas(response)
        finalized = self._finalize_formulas(
            formulas=formulas,
            target_count=count,
            fallback=lambda needed: self.fallback_backend.generate_initial_population(needed),
        )
        logger.info(
            "alpha_lab.llm genesis complete backend={} extracted={} finalized={}",
            self.backend_name,
            len(formulas),
            len(finalized),
        )
        return finalized

    def generate_offspring(self, spec: BreedingSpec, count: int) -> list[str]:
        logger.info(
            "alpha_lab.llm evolution start backend={} model={} target_count={} has_parent_b={}",
            self.backend_name,
            self.model_name,
            count,
            bool(spec.parent_b),
        )
        prompt = self._build_evolution_prompt(spec, count)
        response = self._call_llm(
            system_prompt=EVOLUTION_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=self.temperature_evolution,
            call_kind="evolution",
        )
        formulas = self._extract_formulas(response)
        finalized = self._finalize_formulas(
            formulas=formulas,
            target_count=count,
            fallback=lambda needed: self.fallback_backend.generate_offspring(spec, needed),
        )
        logger.info(
            "alpha_lab.llm evolution complete backend={} extracted={} finalized={}",
            self.backend_name,
            len(formulas),
            len(finalized),
        )
        return finalized

    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float, call_kind: str) -> str:
        logger.info(
            "alpha_lab.llm request backend={} model={} kind={} temperature={} prompt_chars={}",
            self.backend_name,
            self.model_name,
            call_kind,
            temperature,
            len(system_prompt) + len(user_prompt),
        )
        start = perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            stream=False,
            timeout=90,
        )
        latency = perf_counter() - start
        self.call_stats["total_calls"] += 1
        self.call_stats[f"{call_kind}_calls"] += 1
        self.call_stats["total_seconds"] += latency
        self.call_stats["last_latency_seconds"] = latency
        logger.info(
            "alpha_lab.llm response backend={} model={} kind={} latency_seconds={:.4f}",
            self.backend_name,
            self.model_name,
            call_kind,
            latency,
        )
        return response.choices[0].message.content or ""

    def _build_genesis_prompt(self, count: int) -> str:
        features = [
            "Close, High, Low, Open, Volume, Turnover, VWAP",
            "FundingRate, OI, BidAskSpread",
        ]
        operators = [
            "Abs(x), Log(x), Sign(x), Sqrt(x), Div(x, y)",
            "Ts_Mean(x, d), Ts_Max(x, d), Ts_Min(x, d), Ts_Rank(x, d), StdDev(x, d), Delay(x, d), Delta(x, d), Corr(x, y, d)",
            "Ts_Zscore(x, d), Returns_N(x, d), Log_Return(x, d), Decay_Linear(x, d)",
            "CSRank(x), CSZScore(x), CSDemean(x)",
            "OIDelta(OI, d), FundingDelta(FundingRate, d), SpreadRatio(BidAskSpread, Close), AdvN(Turnover, d), Amihud(Close, Turnover, d), ATR_N(High, Low, Close, d)",
            "HLC3(High, Low, Close), OHLC4(Open, High, Low, Close), Where(cond, x, y), Clip(x, lo, hi), FillNA(x, value)",
        ]
        return f"""
# Role
你是一位就职于顶级 Crypto 自营机构的资深量化研究员。你的任务是挖掘高夏普、低相关性的 Alpha 因子公式。

# Environment & Constraints
你只能使用以下字段和算子。严禁发明未列出的变量或函数。

## Features
- {features[0]}
- {features[1]}

## Operators
- {operators[0]}
- {operators[1]}
- {operators[2]}
- {operators[3]}
- {operators[4]}
- {operators[5]}

# Task
请生成 {count} 个截然不同的 Alpha 因子，重点关注：
1. 波动率压缩后的突破。
2. 成交量/流动性与价格的背离。
3. 资金费率、持仓与价格动量的结合。
4. 价差、冲击成本与趋势的关系。

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
上一代回测中，以下父代公式表现最好，它们是你的“父代基因”：

{parent_block}

# Task
基于以上父代公式，进行交叉（Crossover）和变异（Mutation），生成 {count} 个新的子代公式。

# Evaluation Priorities
优先保留并强化以下特征：
1. valid/test 维度都稳健，而不只是 train 内表现好。
2. RankIC、Sharpe、PnL/Turnover、Tail-adjusted return 同时改善。
3. 避免 inactive、低覆盖、极高换手、train-valid gap 过大。
4. 如果父代 test_sharpe 弱于 valid_sharpe，优先做稳健化和降复杂度，而不是放大原信号。

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
4. 平滑降噪：对高换手因子优先使用 Decay_Linear 或 Ts_Mean。
5. 成本感知：鼓励显式使用 SpreadRatio、Amihud、ATR_N。

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
            ("RankIC", metrics.get("rank_ic", 0.0)),
            ("PnLPerTurnover", metrics.get("pnl_per_turnover", 0.0)),
            ("Turnover", metrics.get("avg_turnover", metrics.get("turnover_penalty", 0.0))),
            ("Stability", metrics.get("stability", 0.0)),
            ("TailAdjReturn", metrics.get("tail_penalty_adjusted_return", 0.0)),
            ("SignalCoverage", metrics.get("signal_coverage", 0.0)),
            ("ActiveBarRatio", metrics.get("active_bar_ratio", 0.0)),
            ("GapPenalty", metrics.get("train_valid_gap_penalty", 0.0)),
            ("Inactive", metrics.get("inactive", 0.0)),
        ]
        return ", ".join(f"{name}={float(value):.4f}" for name, value in items)

    def _diagnose_parent_metrics(self, metrics: dict[str, Any]) -> str:
        issues: list[str] = []
        if float(metrics.get("inactive", 0.0)) >= 1.0:
            issues.append("inactive factor")
        if float(metrics.get("train_valid_gap_penalty", 0.0)) > 0.5:
            issues.append("large train-valid gap")
        if float(metrics.get("test_sharpe", 0.0)) + 0.25 < float(metrics.get("sharpe", 0.0)):
            issues.append("test underperforms valid")
        if float(metrics.get("turnover_penalty", metrics.get("avg_turnover", 0.0))) > 0.5:
            issues.append("turnover too high")
        if float(metrics.get("signal_coverage", 1.0)) < 0.5:
            issues.append("signal coverage too low")
        if float(metrics.get("active_bar_ratio", 1.0)) < 0.2:
            issues.append("active bar ratio too low")
        if float(metrics.get("tail_penalty_adjusted_return", 0.0)) < 0.0:
            issues.append("tail-adjusted return negative")
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
    registry: DSLRegistry,
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
