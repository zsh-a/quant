"""
LLM backends for alpha formula generation.

OpenAILLMBackend talks to any OpenAI-compatible API (OpenAI, DeepSeek, vLLM, etc.)
to generate and mutate alpha factor formulas.  Prompts are written in the exact
snake_case DSL so the LLM output compiles with minimal normalisation.
"""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Any

import httpx
from loguru import logger
from openai import OpenAI

import hashlib

from ..core.compiler import FormulaCompiler
from ..core.dsl import TensorSchema
from ..search.evolution import BreedingSpec
from ..core.operators import OperatorRegistry

# ---------------------------------------------------------------------------
# Shared DSL reference (injected into every prompt)
# ---------------------------------------------------------------------------

_FIELDS = """\
## Fields — price
open, high, low, close, volume, turnover, vwap, bid_ask_spread

## Fields — volume detail
trade_count, taker_buy_volume, taker_buy_quote_volume

## Fields — mark price (fair value)
mark_open, mark_high, mark_low, mark_close

## Fields — premium index (futures − spot basis)
premium_open, premium_high, premium_low, premium_close

## Fields — market microstructure
funding_rate, open_interest, open_interest_value
long_short_ratio, taker_long_short_vol_ratio
top_trader_long_short_ratio, top_trader_long_short_position_ratio"""

_OPERATORS = """\
## Operators — math
abs(x), log(x), sign(x), sqrt(x), sigmoid(x), neg(x), div(x, y), power(x, p)

## Operators — time-series (x, window)
ts_mean(x, d), ts_std(x, d), ts_max(x, d), ts_min(x, d), ts_rank(x, d)
ts_zscore(x, d), ts_ema(x, d), decay_linear(x, d)
ts_argmax(x, d), ts_argmin(x, d), ts_winsorize(x, d, n_std)
delay(x, d), delta(x, d), returns_n(x, d), log_return(x, d)

## Operators — time-series (x, y, window)
ts_corr(x, y, d), ts_cov(x, y, d)

## Operators — cross-sectional
cs_rank(x), cs_zscore(x), cs_demean(x), cs_scale(x)

## Operators — domain
oi_delta(open_interest, d), funding_delta(funding_rate, d)
spread_ratio(bid_ask_spread, close), adv_n(turnover, d)
amihud(close, turnover, d), atr_n(high, low, close, d), volatility_n(close, d)
hlc3(high, low, close), ohlc4(open, high, low, close), true_range(high, low, close)

## Operators — control
where(cond, x, y), clip(x, lo, hi), fillna(x, value), max(x, y), min(x, y)"""

_HARD_RULES = """\
1. Formula must be a valid Python expression: ast.parse(formula, mode="eval").
2. AST depth ≤ {depth}.  Prefer simple, robust formulas.
3. Use div(x, y) instead of x/y to avoid division by zero.
4. Use only the fields and operators listed above — nothing else.
5. Output JSON array only, no markdown fences."""

# ---------------------------------------------------------------------------
# Normalisation: CamelCase → snake_case
# ---------------------------------------------------------------------------

# Maps CamelCase patterns to snake_case.  Order matters (longer prefixes first).
_CAMEL_TO_SNAKE: list[tuple[str, str]] = [
    # Cross-sectional
    ("CSRank", "cs_rank"), ("CSZScore", "cs_zscore"), ("CSDemean", "cs_demean"), ("CSScale", "cs_scale"),
    # Time-series (Ts_ prefix)
    ("Ts_DecayLinear", "decay_linear"), ("Ts_Winsorize", "ts_winsorize"),
    ("Ts_Zscore", "ts_zscore"), ("Ts_Argmax", "ts_argmax"), ("Ts_Argmin", "ts_argmin"),
    ("Ts_Returns", "returns_n"), ("Ts_Mean", "ts_mean"), ("Ts_Std", "ts_std"),
    ("Ts_Max", "ts_max"), ("Ts_Min", "ts_min"), ("Ts_Rank", "ts_rank"),
    ("Ts_EMA", "ts_ema"), ("Ts_Corr", "ts_corr"), ("Ts_Cov", "ts_cov"),
    # Diff
    ("Returns_N", "returns_n"), ("Log_Return", "log_return"),
    ("Decay_Linear", "decay_linear"),
    # Domain
    ("OIDelta", "oi_delta"), ("FundingDelta", "funding_delta"),
    ("SpreadRatio", "spread_ratio"), ("AdvN", "adv_n"),
    ("Amihud", "amihud"), ("ATR_N", "atr_n"), ("Volatility_N", "volatility_n"),
    ("HLC3", "hlc3"), ("OHLC4", "ohlc4"), ("TrueRange", "true_range"),
    # Math / control
    ("Div", "div"), ("Abs", "abs"), ("Log", "log"), ("Sign", "sign"),
    ("Sqrt", "sqrt"), ("Sigmoid", "sigmoid"), ("Power", "power"),
    ("Where", "where"), ("Clip", "clip"), ("FillNA", "fillna"),
    ("Max", "max"), ("Min", "min"),
    ("Correlation", "ts_corr"), ("Covariance", "ts_cov"),
    ("StdDev", "ts_std"), ("Corr", "ts_corr"),
    ("Delta", "delta"), ("Delay", "delay"), ("Scale", "cs_scale"),
    # Field aliases (CamelCase → snake_case)
    ("Close", "close"), ("High", "high"), ("Low", "low"), ("Open", "open"),
    ("Volume", "volume"), ("Turnover", "turnover"), ("VWAP", "vwap"),
    ("BidAskSpread", "bid_ask_spread"),
    ("TradeCount", "trade_count"),
    ("TakerBuyQuoteVolume", "taker_buy_quote_volume"),
    ("TakerBuyVolume", "taker_buy_volume"),
    ("MarkOpen", "mark_open"), ("MarkHigh", "mark_high"),
    ("MarkLow", "mark_low"), ("MarkClose", "mark_close"),
    ("PremiumOpen", "premium_open"), ("PremiumHigh", "premium_high"),
    ("PremiumLow", "premium_low"), ("PremiumClose", "premium_close"),
    ("FundingRate", "funding_rate"),
    ("OIValue", "open_interest_value"), ("OI", "open_interest"),
    ("OpenInterest", "open_interest"),
    ("LongShortRatio", "long_short_ratio"),
    ("TakerLongShortVolRatio", "taker_long_short_vol_ratio"),
    ("TopTraderLongShortPositionRatio", "top_trader_long_short_position_ratio"),
    ("TopTraderLongShortRatio", "top_trader_long_short_ratio"),
]

# Pre-compile regex patterns (word-boundary match to avoid partial replacement)
_CAMEL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{re.escape(src)}\b"), dst) for src, dst in _CAMEL_TO_SNAKE
]


def _normalize_to_snake(formula: str) -> str:
    """Convert any CamelCase identifiers in *formula* to the canonical snake_case DSL."""
    for pattern, replacement in _CAMEL_PATTERNS:
        formula = pattern.sub(replacement, formula)
    return formula


# ---------------------------------------------------------------------------
# Heuristic (local, deterministic) backend
# ---------------------------------------------------------------------------


class HeuristicLLMBackend:
    """Deterministic formula generation via mutation, crossover, and template wrapping.

    Lightweight fallback when no LLM API key is available, and also used
    by SearchOrchestrator to bootstrap initial populations.
    """

    def __init__(self, registry: OperatorRegistry | None = None, schema: TensorSchema | None = None):
        self.registry = registry or OperatorRegistry()
        self.schema = schema or TensorSchema.default_market_schema()
        self.call_stats = {"initial_population_calls": 0, "offspring_calls": 0}

    @property
    def backend_name(self) -> str:
        return "heuristic"

    def generate_offspring(self, spec: BreedingSpec, count: int) -> list[str]:
        self.call_stats["offspring_calls"] += 1
        offspring: list[str] = []
        bases = [spec.parent_a]
        if spec.parent_b:
            bases.append(spec.parent_b)
        attempts = 0
        max_attempts = max(count * 8, 8)
        while len(offspring) < count and attempts < max_attempts:
            base = bases[attempts % len(bases)]
            if spec.parent_b and attempts % 3 == 0:
                candidate = self._crossover_formula(spec.parent_a, spec.parent_b, attempts)
            elif attempts % 3 == 1:
                candidate = self._mutate_formula(base, attempts)
            else:
                candidate = self._wrap_formula(base, spec.objective, attempts)
            if candidate not in offspring:
                offspring.append(candidate)
            attempts += 1
        if not offspring:
            offspring.append(self._mutate_formula(spec.parent_a, 0))
        return offspring

    def generate_initial_population(self, count: int) -> list[str]:
        self.call_stats["initial_population_calls"] += 1
        seeds = [
            "cs_rank(ts_mean(close, 5) - close)",
            "cs_rank(ts_std(close, 10))",
            "cs_rank(delta(premium_close, 5))",
            "cs_rank(ts_zscore(funding_rate, 20))",
            "cs_rank(delta(open_interest, 10) - ts_mean(delta(open_interest, 10), 20))",
            "cs_rank(ts_zscore(long_short_ratio, 20))",
            "cs_rank(div(taker_buy_volume, volume + 1e-12) - 0.5)",
            "cs_rank(ts_corr(close, taker_buy_volume, 10))",
        ]
        generated: list[str] = []
        attempts = 0
        while len(generated) < count and attempts < max(count * 4, 8):
            base = seeds[attempts % len(seeds)]
            candidate = base if attempts < len(seeds) else self._wrap_formula(base, "bootstrap", attempts)
            if candidate not in generated:
                generated.append(candidate)
            attempts += 1
        return generated[:count]

    def _mutate_formula(self, formula: str, variant: int = 0) -> str:
        replacements = [
            ("ts_mean(", "ts_std("), ("ts_std(", "ts_mean("),
            ("ts_max(", "ts_rank("), ("ts_rank(", "ts_mean("),
            ("close", "vwap"), ("close", "mark_close"),
            ("volume", "turnover"), ("volume", "taker_buy_volume"),
            ("close", "hlc3(high, low, close)"), ("turnover", "adv_n(turnover, 5)"),
            ("close", "ohlc4(open, high, low, close)"),
            ("volatility_n(close, 20)", "atr_n(high, low, close, 14)"),
            ("funding_rate", "ts_zscore(funding_rate, 20)"),
            ("open_interest", "delta(open_interest, 5)"),
            ("close", "premium_close"), ("volume", "trade_count"),
        ]
        offset = self._stable_index(formula, len(replacements), salt=f"mutate:{variant}")
        for idx in range(len(replacements)):
            source, target = replacements[(offset + idx) % len(replacements)]
            if source in formula:
                return formula.replace(source, target, 1)
        return self._wrap_formula(formula, "mutation fallback", variant)

    def _wrap_formula(self, formula: str, objective: str, variant: int = 0) -> str:
        wrappers = [
            f"cs_rank(({formula}) - amihud(close, turnover, 5))",
            f"cs_rank(({formula}) + atr_n(high, low, close, 5))",
            f"cs_zscore(decay_linear(({formula}), 3))",
            f"cs_rank(fillna(({formula}), 0) + cs_demean(vwap))",
            f"cs_zscore(clip(({formula}), -3, 3) + adv_n(turnover, 10))",
            f"cs_rank(ts_zscore(({formula}), 5) - volatility_n(close, 10))",
            f"cs_rank(({formula}) + ts_corr(close, volume, 10))",
            f"cs_rank(({formula}) - ts_rank(turnover, 20))",
            f"cs_rank(decay_linear(({formula}), 5) + ts_mean(volume, 10))",
            f"cs_rank(({formula}) + delta(premium_close, 5))",
            f"cs_rank(({formula}) - ts_zscore(funding_rate, 20))",
            f"cs_rank(({formula}) + ts_zscore(long_short_ratio, 20))",
            f"cs_rank(({formula}) + delta(open_interest, 10))",
            f"cs_rank(({formula}) - ts_rank(taker_buy_volume, 10))",
        ]
        if "turnover" in objective.lower():
            wrappers.extend([
                f"cs_rank(decay_linear(({formula}), 5) - spread_ratio(bid_ask_spread, close))",
                f"cs_rank(fillna(({formula}), 0) - amihud(close, turnover, 10))",
            ])
        idx = self._stable_index(formula, len(wrappers), salt=f"wrap:{objective}:{variant}")
        return wrappers[idx]

    def _crossover_formula(self, parent_a: str, parent_b: str, variant: int = 0) -> str:
        templates = [
            f"cs_rank(({parent_a}) + ({parent_b}))",
            f"cs_rank(({parent_a}) - ({parent_b}))",
            f"cs_zscore(decay_linear((({parent_a}) + ({parent_b})), 3))",
            f"cs_rank(max(({parent_a}), ({parent_b})) - volatility_n(close, 10))",
            f"cs_rank(min(({parent_a}), ({parent_b})) + ts_corr(close, volume, 10))",
            f"cs_rank((({parent_a}) + atr_n(high, low, close, 5)) - (({parent_b}) + amihud(close, turnover, 5)))",
            f"cs_rank(ts_mean(({parent_a}), 3) - ts_mean(({parent_b}), 3))",
        ]
        idx = self._stable_index(f"{parent_a}|{parent_b}", len(templates), salt=f"cross:{variant}")
        return templates[idx]

    def _stable_index(self, value: str, modulo: int, salt: str = "") -> int:
        digest = hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()
        return int(digest[:12], 16) % max(modulo, 1)


# ---------------------------------------------------------------------------
# OpenAI-compatible LLM Backend
# ---------------------------------------------------------------------------


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
        strategy_memory: Any | None = None,
        knowledge_base: Any | None = None,
        feature_kitchen: Any | None = None,
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
        self.call_stats: dict[str, Any] = {
            "total_calls": 0, "genesis_calls": 0, "evolution_calls": 0,
            "total_seconds": 0.0, "total_tokens": 0,
        }
        self.fallback_backend = fallback_backend or HeuristicLLMBackend(
            registry=self.registry, schema=self.schema,
        )
        self.client = client or OpenAI(
            base_url=self.base_url, api_key=self.api_key,
            timeout=httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=5.0),
            max_retries=2,
        )

        # --- Enhanced modules (optional, gracefully degrade if None) ---
        self.strategy_memory = strategy_memory
        self.knowledge_base = knowledge_base
        self.feature_kitchen = feature_kitchen
        if self.feature_kitchen is not None:
            self.feature_kitchen.build_catalog()
        # Theme map: formula -> theme_id (populated during extraction)
        self._last_theme_map: dict[str, str] = {}

    @property
    def backend_name(self) -> str:
        return "openai"

    # -- public API ---------------------------------------------------------

    def generate_initial_population(self, count: int) -> list[str]:
        from ..infra.tracing import tracer

        # Over-generate: ask LLM for 2x then take best after validation.
        # This reduces fallback to heuristic and improves diversity.
        request_count = min(count * 2, 24)
        with tracer.start_span("genesis", kind="breed",
                               target_count=count, backend=self.backend_name) as span:
            self._last_theme_map.clear()
            prompt = self._build_genesis_prompt(request_count)
            raw = self._call_llm(prompt, self.temperature_genesis, "genesis")
            formulas = self._extract_and_validate(raw)
            finalized = self._finalize(formulas, count,
                                       lambda n: self.fallback_backend.generate_initial_population(n))
            span.set("extracted", len(formulas))
            span.set("finalized", len(finalized))
            return finalized

    def generate_offspring(self, spec: BreedingSpec, count: int) -> list[str]:
        from ..infra.tracing import tracer

        request_count = min(count * 2, 16)
        with tracer.start_span("evolution", kind="breed",
                               target_count=count, backend=self.backend_name) as span:
            self._last_theme_map.clear()
            prompt = self._build_evolution_prompt(spec, request_count)
            raw = self._call_llm(prompt, self.temperature_evolution, "evolution")
            formulas = self._extract_and_validate(raw)
            finalized = self._finalize(formulas, count,
                                       lambda n: self.fallback_backend.generate_offspring(spec, n))
            span.set("extracted", len(formulas))
            span.set("finalized", len(finalized))
            return finalized

    def get_theme_for_formula(self, formula: str) -> str | None:
        """Return the theme that the LLM assigned to a formula, if any."""
        return self._last_theme_map.get(formula)

    # -- prompt builders ----------------------------------------------------

    def _build_genesis_prompt(self, count: int) -> str:
        rules = _HARD_RULES.format(depth=self.max_ast_depth)

        # --- Financial knowledge (structured themes) ---
        if self.knowledge_base is not None and self.strategy_memory is not None:
            # UCB bandit selects themes balancing explore/exploit
            theme_ids = self.strategy_memory.select_themes_ucb(min(count, 6), c=1.0)
            theme_section = self.knowledge_base.build_theme_prompt(theme_ids)
        elif self.knowledge_base is not None:
            # No memory yet — sample broadly
            all_ids = self.knowledge_base.get_all_theme_ids()
            theme_section = self.knowledge_base.build_theme_prompt(all_ids[:6])
        else:
            theme_section = (
                "# Themes to explore (cover as many as possible)\n"
                "1. Volatility compression breakout — atr_n / ts_std contraction then expansion.\n"
                "2. Volume-price divergence — taker_buy_volume vs close, trade_count anomaly.\n"
                "3. Funding + OI + momentum triple resonance.\n"
                "4. Futures-spot basis — premium_close mean-reversion / trend.\n"
                "5. Sentiment extremes — long_short_ratio / taker_long_short_vol_ratio z-score reversal.\n"
                "6. Whale behaviour — delta(top_trader_long_short_ratio, d) as a leading signal."
            )

        # --- Feature groups (organized by financial meaning) ---
        if self.knowledge_base is not None:
            feature_section = self.knowledge_base.build_feature_groups_prompt()
        else:
            feature_section = _FIELDS

        # --- Derived features (building blocks) ---
        derived_section = ""
        if self.feature_kitchen is not None:
            derived_section = "\n" + self.feature_kitchen.get_catalog_as_prompt_section() + "\n"

        # --- Strategy memory feedback (RL loop) ---
        feedback_section = ""
        if self.strategy_memory is not None:
            feedback_section = "\n" + self.strategy_memory.build_feedback_summary() + "\n"

        return f"""\
You are a senior crypto quant researcher.  Generate {count} diverse alpha factor formulas.

{feature_section}

{_OPERATORS}
{derived_section}
## Financial Hypotheses to Explore
{theme_section}
{feedback_section}
# Rules
{rules}

For each formula, specify which theme it targets.
Output exactly {count} items:
[{{"theme": "...", "rationale": "...", "formula": "cs_rank(...)"}}]"""

    def _build_evolution_prompt(self, spec: BreedingSpec, count: int) -> str:
        rules = _HARD_RULES.format(depth=self.max_ast_depth)
        parents = self._format_parents(spec)

        # --- Feature reference ---
        if self.knowledge_base is not None:
            feature_section = self.knowledge_base.build_feature_groups_prompt()
        else:
            feature_section = _FIELDS

        # --- Strategy memory feedback ---
        feedback_section = ""
        if self.strategy_memory is not None:
            feedback_section = "\n" + self.strategy_memory.build_feedback_summary() + "\n"

        return f"""\
You are a quant formula mutation engine.  Given tournament-winning parent formulas with
their backtest metrics, generate {count} improved offspring.

{feature_section}

{_OPERATORS}

# Tournament winners
{parents}

# Mutation strategy
- Explore the **neighbourhood** of strong parents: swap operators, change windows, recombine sub-expressions.
- Each offspring must be structurally different — not just a window-size tweak.
- If a parent has weak IC: add cross-sectional structure (cs_rank, cs_zscore).
- If a parent has high turnover: wrap with decay_linear or ts_ema for smoothing.
- If a parent has large train-test gap: simplify (reduce depth, fewer operators).
- Actively incorporate under-used fields: premium_close, long_short_ratio, taker_buy_volume, mark_close.
{feedback_section}
# Rules
{rules}

For each formula, specify which theme it targets.
Output exactly {count} items:
[{{"theme": "...", "rationale": "...", "formula": "cs_rank(...)"}}]"""

    def _format_parents(self, spec: BreedingSpec) -> str:
        lines = []
        for i, fb in enumerate(spec.parent_feedback or [], start=1):
            m = fb.get("metrics", {})
            summary = (
                f"sharpe={m.get('sharpe', 0):.3f}  "
                f"test_sharpe={m.get('test_sharpe', 0):.3f}  "
                f"|IC|={abs(float(m.get('rank_ic', 0) or 0)):.4f}  "
                f"turnover={m.get('avg_turnover', 0):.4f}  "
                f"max_dd={m.get('max_drawdown', 0):.3f}"
            )
            issues = self._diagnose(m)
            lines.append(
                f"[Parent {i}]  `{fb.get('formula', spec.parent_a)}`\n"
                f"  metrics: {summary}\n"
                f"  issues:  {issues}"
            )
        return "\n".join(lines) if lines else f"[Parent 1]  `{spec.parent_a}`"

    @staticmethod
    def _diagnose(m: dict) -> str:
        issues = []
        if float(m.get("inactive", 0)) >= 1:
            issues.append("inactive")
        if float(m.get("avg_turnover", 0)) < 0.005:
            issues.append("low turnover")
        if float(m.get("avg_turnover", 0)) > 0.5:
            issues.append("high turnover")
        if float(m.get("test_sharpe", 0)) < 0:
            issues.append("negative test sharpe")
        gap = float(m.get("train_valid_gap_penalty", 0))
        if gap > 0.4:
            issues.append("overfitting")
        if abs(float(m.get("rank_ic", 0) or 0)) < 0.01:
            issues.append("weak IC")
        return "; ".join(issues) or "none"

    # -- LLM call -----------------------------------------------------------

    def _call_llm(self, user_prompt: str, temperature: float, kind: str) -> str:
        from ..infra.tracing import prompt_hash, tracer

        messages = [{"role": "user", "content": user_prompt}]
        with tracer.start_span(kind, kind="llm", model=self.model_name,
                               temperature=temperature,
                               prompt_hash=prompt_hash(user_prompt),
                               input=messages) as span:
            resp = self.client.chat.completions.create(
                model=self.model_name, messages=messages,
                temperature=temperature, stream=False, timeout=90,
            )
            content = resp.choices[0].message.content or ""
            span.set_response(resp)
            span.set("output", content)
            self.call_stats["total_calls"] += 1
            self.call_stats[f"{kind}_calls"] = self.call_stats.get(f"{kind}_calls", 0) + 1
            self.call_stats["total_seconds"] += span.duration_ms / 1000
            self.call_stats["total_tokens"] += span.attributes.get("total_tokens") or 0
            return content

    # -- RL feedback -------------------------------------------------------

    def record_evaluation_result(
        self,
        formula: str,
        theme_id: str | None,
        metrics: dict[str, float],
        is_novel: bool,
        round_idx: int = 0,
    ) -> None:
        """Record evaluation result to close the RL feedback loop."""
        if self.strategy_memory is None:
            return
        resolved_theme = theme_id or self._infer_theme(formula)
        self.strategy_memory.record(
            formula=formula,
            theme_id=resolved_theme,
            metrics=metrics,
            is_novel=is_novel,
            round_idx=round_idx,
            all_fields=self.schema.fields,
        )
        if self.feature_kitchen is not None:
            fitness = float(metrics.get("fitness", 0.0))
            self.feature_kitchen.track_feature_importance(formula, fitness)

    def _infer_theme(self, formula: str) -> str:
        """Infer theme from formula content when LLM didn't return one."""
        formula_lower = formula.lower()
        if "funding_rate" in formula_lower and "premium" in formula_lower:
            return "funding_basis_arb"
        if "open_interest" in formula_lower and ("return" in formula_lower or "close" in formula_lower):
            return "oi_momentum_divergence"
        if "taker_buy" in formula_lower:
            return "taker_flow_imbalance"
        if "spread" in formula_lower or "amihud" in formula_lower:
            return "microstructure_toxicity"
        if "long_short_ratio" in formula_lower or "taker_long_short" in formula_lower:
            return "sentiment_extreme_reversal"
        if "top_trader" in formula_lower:
            return "whale_positioning"
        if "volatility_n" in formula_lower or "atr_n" in formula_lower:
            return "volatility_regime_switch"
        if "premium" in formula_lower:
            return "premium_dynamics"
        if "funding_rate" in formula_lower:
            return "funding_basis_arb"
        if "mark_close" in formula_lower or "mark_open" in formula_lower:
            return "mark_spot_divergence"
        return "general"

    # -- parsing / validation -----------------------------------------------

    def _extract_and_validate(self, raw: str) -> list[str]:
        """Extract formula strings from LLM output, normalise, and compile-check."""
        candidates = self._extract_items(raw)
        valid: list[str] = []
        seen: set[str] = set()
        for formula, theme in candidates:
            normed = _normalize_to_snake(formula.strip())
            if not normed or normed in seen:
                continue
            if not self._is_valid_expr(normed):
                continue
            if self._ast_depth(normed) > self.max_ast_depth:
                continue
            try:
                self.compiler.compile(normed, self.schema)
            except ValueError:
                continue
            seen.add(normed)
            valid.append(normed)
            # Store theme mapping for lineage tracking
            if theme:
                self._last_theme_map[normed] = theme
        return valid

    def _extract_items(self, raw: str) -> list[tuple[str, str]]:
        """Pull (formula, theme) pairs from JSON or regex fallback."""
        items: list[tuple[str, str]] = []
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        # Try JSON parse
        try:
            payload = json.loads(cleaned)
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict) and "formula" in item:
                        items.append((
                            str(item["formula"]),
                            str(item.get("theme", "")),
                        ))
            elif isinstance(payload, dict) and "formula" in payload:
                items.append((
                    str(payload["formula"]),
                    str(payload.get("theme", "")),
                ))
        except Exception:
            pass
        if items:
            return items
        # Regex fallback (theme not available)
        for m in re.finditer(r'"formula"\s*:\s*"((?:[^"\\]|\\.)*)"', raw):
            try:
                items.append((json.loads(f'"{m.group(1)}"'), ""))
            except Exception:
                items.append((m.group(1), ""))
        return items

    def _extract_formula_strings(self, raw: str) -> list[str]:
        """Pull formula strings from JSON or regex fallback (backward compat)."""
        return [formula for formula, _ in self._extract_items(raw)]

    @staticmethod
    def _finalize(formulas: list[str], target: int, fallback) -> list[str]:
        unique: list[str] = list(dict.fromkeys(formulas))[:target]
        if len(unique) < target:
            for f in fallback(target - len(unique)):
                if f not in unique:
                    unique.append(f)
                if len(unique) >= target:
                    break
        return unique

    @staticmethod
    def _is_valid_expr(formula: str) -> bool:
        try:
            ast.parse(formula, mode="eval")
            return True
        except SyntaxError:
            return False

    @staticmethod
    def _ast_depth(formula: str) -> int:
        tree = ast.parse(formula, mode="eval")

        def walk(node: ast.AST) -> int:
            if isinstance(node, ast.Expression):
                return walk(node.body)
            if isinstance(node, (ast.Name, ast.Constant)):
                return 1
            children = [walk(c) for c in ast.iter_child_nodes(node)]
            return 1 + (max(children) if children else 0)

        return walk(tree)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_default_llm_backend(
    registry: OperatorRegistry,
    schema: TensorSchema,
    backend_name: str = "auto",
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    strategy_memory: Any | None = None,
    knowledge_base: Any | None = None,
    feature_kitchen: Any | None = None,
) -> Any:
    requested = (backend_name or "auto").lower()
    if requested == "heuristic":
        return HeuristicLLMBackend(registry=registry, schema=schema)
    resolved_key = api_key or os.getenv("ALPHA_LAB_LLM_API_KEY")
    if requested == "openai" or (requested == "auto" and resolved_key):
        return OpenAILLMBackend(
            registry=registry, schema=schema,
            model_name=model_name, base_url=base_url, api_key=api_key,
            strategy_memory=strategy_memory,
            knowledge_base=knowledge_base,
            feature_kitchen=feature_kitchen,
        )
    return HeuristicLLMBackend(registry=registry, schema=schema)
