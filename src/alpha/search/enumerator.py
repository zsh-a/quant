"""Programmatic batch formula enumeration.

Generates thousands of candidate alpha formulas by systematically combining
fields, time-series operators, cross-sectional wrappers, and windows — without
any LLM calls.  Deduplication is done via ``expr_hash`` so only structurally
unique formulas survive.
"""

from __future__ import annotations

import random

from loguru import logger

from ..core.compiler import FormulaCompiler
from ..core.dsl import TensorSchema

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

# Fields most useful for alpha construction (subset of full schema)
_CORE_FIELDS = [
    "close", "volume", "turnover", "vwap",
    "open_interest", "funding_rate",
    "taker_buy_volume", "long_short_ratio",
    "premium_close", "bid_ask_spread",
]

_PAIR_FIELDS = [
    ("close", "volume"),
    ("close", "turnover"),
    ("close", "open_interest"),
    ("close", "taker_buy_volume"),
    ("close", "funding_rate"),
    ("volume", "open_interest"),
    ("taker_buy_volume", "volume"),
    ("premium_close", "close"),
    ("funding_rate", "open_interest"),
]

_TS_OPS_UNARY = ["ts_mean", "ts_std", "ts_sum", "ts_rank", "ts_ema", "ts_zscore"]
_TS_OPS_PAIR = ["ts_corr", "ts_cov"]
_CS_WRAPPERS = ["cs_rank", "cs_zscore", "cs_demean"]
_WINDOWS = [5, 10, 20]
_DELTA_WINDOWS = [3, 5, 10]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class FormulaEnumerator:
    """Generate large batches of compilable alpha formulas programmatically."""

    def __init__(
        self,
        compiler: FormulaCompiler | None = None,
        schema: TensorSchema | None = None,
        seed: int = 42,
    ):
        from ..core.operators import OperatorRegistry

        registry = OperatorRegistry()
        self.compiler = compiler or FormulaCompiler(registry)
        self.schema = schema or TensorSchema.default_market_schema()
        self._rng = random.Random(seed)

    def generate(self, *, max_count: int = 500) -> list[str]:
        """Return up to *max_count* unique compilable formulas.

        Formulas are generated in priority order (simplest first) and
        deduplicated by ``expr_hash``.
        """
        seen_hashes: set[str] = set()
        results: list[str] = []

        # Available fields from schema
        available = self.schema.fields
        fields = [f for f in _CORE_FIELDS if f in available]

        for formula in self._iter_candidates(fields):
            if len(results) >= max_count:
                break
            try:
                prog = self.compiler.compile(formula, self.schema)
                if prog.expr_hash not in seen_hashes:
                    seen_hashes.add(prog.expr_hash)
                    results.append(formula)
            except Exception:
                continue

        logger.info(
            "enumerator.generate produced={} max_count={}", len(results), max_count,
        )
        return results

    # ------------------------------------------------------------------
    # Internal generators (yielded lazily)
    # ------------------------------------------------------------------

    def _iter_candidates(self, fields: list[str]):
        """Yield formula strings in priority order."""

        # --- Tier A: cs_rank(ts_op(field, window)) ---
        for cs in _CS_WRAPPERS:
            for op in _TS_OPS_UNARY:
                for field in fields:
                    for w in _WINDOWS:
                        yield f"{cs}({op}({field}, {w}))"

        # --- Tier B: cs_rank(field_a - ts_mean(field_a, w)) (mean-reversion) ---
        for field in fields:
            for w in _WINDOWS:
                yield f"cs_rank({field} - ts_mean({field}, {w}))"
                yield f"cs_rank(ts_mean({field}, {w}) - {field})"

        # --- Tier C: delta / returns patterns ---
        for field in fields:
            for w in _DELTA_WINDOWS:
                yield f"cs_rank(delta({field}, {w}))"
                yield f"cs_rank(returns_n({field}, {w}))"
                for w2 in _WINDOWS:
                    if w2 > w:
                        yield f"cs_rank(delta({field}, {w}) - ts_mean(delta({field}, {w}), {w2}))"

        # --- Tier D: pair correlations ---
        for f1, f2 in _PAIR_FIELDS:
            if f1 in fields and f2 in fields:
                for w in _WINDOWS:
                    yield f"cs_rank(ts_corr({f1}, {f2}, {w}))"

        # --- Tier E: ratio / interaction patterns ---
        for f1, f2 in _PAIR_FIELDS:
            if f1 in fields and f2 in fields:
                yield f"cs_rank({f1} / ({f2} + 1e-12))"
                yield f"cs_rank(ts_mean({f1}, 5) / (ts_mean({f2}, 5) + 1e-12))"
                yield f"cs_rank(delta({f1}, 5) - delta({f2}, 5))"

        # --- Tier F: composite (depth 3-4) ---
        for field in fields[:6]:  # limit to avoid explosion
            for w1 in [5, 10]:
                for w2 in [10, 20]:
                    if w2 > w1:
                        yield f"cs_rank(ts_zscore({field}, {w2}) - ts_zscore({field}, {w1}))"
                        yield f"cs_rank(ts_rank({field}, {w2}) - ts_rank({field}, {w1}))"

        # --- Tier G: domain-specific ---
        if "open_interest" in fields:
            for w in _DELTA_WINDOWS:
                yield f"cs_rank(oi_delta(open_interest, {w}))"
        if "funding_rate" in fields:
            for w in _DELTA_WINDOWS:
                yield f"cs_rank(funding_delta(funding_rate, {w}))"
        if "bid_ask_spread" in fields and "close" in fields:
            yield "cs_rank(spread_ratio(bid_ask_spread, close))"
        if "close" in fields and "turnover" in fields:
            for w in _WINDOWS:
                yield f"cs_rank(amihud(close, turnover, {w}))"

        # --- Tier H: shuffled mutations of top-tier formulas ---
        base_formulas = [
            "cs_rank(ts_mean(close, 5) - close)",
            "cs_rank(ts_std(close, 10))",
            "cs_rank(delta(premium_close, 5))",
            "cs_rank(ts_zscore(funding_rate, 20))",
            "cs_rank(ts_corr(close, volume, 10))",
        ]
        mutations = [
            ("close", "vwap"), ("close", "mark_close"),
            ("volume", "turnover"), ("volume", "taker_buy_volume"),
            ("ts_mean", "ts_ema"), ("ts_std", "volatility_n"),
            ("5", "10"), ("10", "20"), ("20", "40"),
        ]
        for base in base_formulas:
            for old, new in mutations:
                if old in base:
                    yield base.replace(old, new, 1)
