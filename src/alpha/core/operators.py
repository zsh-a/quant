"""Unified operator registry.

Merges the DSL OperatorSpec registry from alpha_lab with CamelCase alias
support for alpha_mining formula compatibility.  New operators added:
``ts_ema``, ``ts_winsorize``, ``power``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dsl import TensorSchema


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    min_args: int
    max_args: int
    output_kind: str
    category: str


class OperatorRegistry:
    def __init__(self):
        self._operators = self._build_defaults()
        self._aliases = self._build_aliases()

    def _build_defaults(self) -> dict[str, OperatorSpec]:
        specs = [
            # --- unary ---
            OperatorSpec("abs", 1, 1, "tensor", "unary"),
            OperatorSpec("log", 1, 1, "tensor", "unary"),
            OperatorSpec("sign", 1, 1, "tensor", "unary"),
            OperatorSpec("sqrt", 1, 1, "tensor", "unary"),
            OperatorSpec("sigmoid", 1, 1, "tensor", "unary"),
            OperatorSpec("neg", 1, 1, "tensor", "unary"),
            OperatorSpec("power", 2, 2, "tensor", "unary"),
            # --- binary ---
            OperatorSpec("add", 2, 2, "tensor", "binary"),
            OperatorSpec("sub", 2, 2, "tensor", "binary"),
            OperatorSpec("mul", 2, 2, "tensor", "binary"),
            OperatorSpec("div", 2, 2, "tensor", "binary"),
            OperatorSpec("max", 2, 2, "tensor", "binary"),
            OperatorSpec("min", 2, 2, "tensor", "binary"),
            OperatorSpec("pow", 2, 2, "tensor", "binary"),
            # --- conditional ---
            OperatorSpec("clip", 3, 3, "tensor", "conditional"),
            OperatorSpec("fillna", 2, 2, "tensor", "conditional"),
            OperatorSpec("where", 3, 3, "tensor", "conditional"),
            # --- comparison ---
            OperatorSpec("gt", 2, 2, "mask", "comparison"),
            OperatorSpec("ge", 2, 2, "mask", "comparison"),
            OperatorSpec("lt", 2, 2, "mask", "comparison"),
            OperatorSpec("le", 2, 2, "mask", "comparison"),
            OperatorSpec("eq", 2, 2, "mask", "comparison"),
            OperatorSpec("ne", 2, 2, "mask", "comparison"),
            # --- logical ---
            OperatorSpec("not", 1, 1, "mask", "logical"),
            OperatorSpec("and", 2, 2, "mask", "logical"),
            OperatorSpec("or", 2, 2, "mask", "logical"),
            # --- time_series ---
            OperatorSpec("delay", 2, 2, "tensor", "time_series"),
            OperatorSpec("delta", 2, 2, "tensor", "time_series"),
            OperatorSpec("returns_n", 2, 2, "tensor", "time_series"),
            OperatorSpec("log_return", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_mean", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_std", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_sum", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_max", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_min", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_rank", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_zscore", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_corr", 3, 3, "tensor", "time_series"),
            OperatorSpec("ts_cov", 3, 3, "tensor", "time_series"),
            OperatorSpec("decay_linear", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_argmax", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_argmin", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_ema", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_winsorize", 3, 3, "tensor", "time_series"),
            # --- cross_sectional ---
            OperatorSpec("cs_rank", 1, 1, "tensor", "cross_sectional"),
            OperatorSpec("cs_scale", 1, 1, "tensor", "cross_sectional"),
            OperatorSpec("cs_zscore", 1, 1, "tensor", "cross_sectional"),
            OperatorSpec("cs_demean", 1, 1, "tensor", "cross_sectional"),
            # --- domain (crypto) ---
            OperatorSpec("oi_delta", 2, 2, "tensor", "domain"),
            OperatorSpec("funding_delta", 2, 2, "tensor", "domain"),
            OperatorSpec("spread_ratio", 2, 2, "tensor", "domain"),
            OperatorSpec("adv_n", 2, 2, "tensor", "domain"),
            OperatorSpec("amihud", 3, 3, "tensor", "domain"),
            OperatorSpec("hlc3", 3, 3, "tensor", "domain"),
            OperatorSpec("ohlc4", 4, 4, "tensor", "domain"),
            OperatorSpec("true_range", 3, 3, "tensor", "domain"),
            OperatorSpec("atr_n", 4, 4, "tensor", "domain"),
            OperatorSpec("volatility_n", 2, 2, "tensor", "domain"),
        ]
        return {spec.name: spec for spec in specs}

    def list_operators(self) -> list[OperatorSpec]:
        return sorted(self._operators.values(), key=lambda spec: spec.name)

    def has(self, name: str) -> bool:
        return self.normalize_name(name) in self._operators

    def get(self, name: str) -> OperatorSpec:
        return self._operators[self.normalize_name(name)]

    def normalize_name(self, name: str) -> str:
        lowered = name.lower()
        if lowered in self._operators:
            return lowered
        compact = lowered.replace("_", "")
        return self._aliases.get(compact, lowered)

    def _build_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for canonical in self._operators:
            aliases[canonical.replace("_", "")] = canonical
        aliases.update(
            {
                # alpha_mining CamelCase → snake_case
                "csrank": "cs_rank",
                "csscale": "cs_scale",
                "scale": "cs_scale",
                "neutralize": "cs_demean",
                "cszscore": "cs_zscore",
                "csdemean": "cs_demean",
                "tsema": "ts_ema",
                "tswinsorize": "ts_winsorize",
                "tsreturns": "returns_n",
                "tsdecaylinear": "decay_linear",
                "correlation": "ts_corr",
                "covariance": "ts_cov",
                # convenience aliases
                "volatility": "volatility_n",
                "tsmax": "ts_max",
                "tsmin": "ts_min",
                "tsmean": "ts_mean",
                "tsstd": "ts_std",
                "tssum": "ts_sum",
                "tsrank": "ts_rank",
                "tszscore": "ts_zscore",
                "tscorr": "ts_corr",
                "tscov": "ts_cov",
                "returnsn": "returns_n",
                "logreturn": "log_return",
                "decaylinear": "decay_linear",
                "tsargmax": "ts_argmax",
                "tsargmin": "ts_argmin",
                "fillna": "fillna",
                "oidelta": "oi_delta",
                "fundingdelta": "funding_delta",
                "spreadratio": "spread_ratio",
                "adv": "adv_n",
                "advn": "adv_n",
                "truerange": "true_range",
                "atr": "atr_n",
                "atrn": "atr_n",
                "corr": "ts_corr",
                "cov": "ts_cov",
                "stddev": "ts_std",
                "diff": "delta",
            }
        )
        return aliases

    def validate_formula(self, formula: str, schema: "TensorSchema | None" = None):
        import hashlib

        from .dsl import FormulaParser, TensorSchema, TypeChecker, ValidationReport, normalize_formula

        parser = FormulaParser(self)
        checker = TypeChecker(self)
        try:
            parsed = parser.parse(formula)
            typed = checker.infer(parsed, schema or TensorSchema.default_market_schema())
            normalized = normalize_formula(parsed)
            ast_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            return ValidationReport(
                ok=True,
                normalized_formula=normalized,
                errors=[],
                ast_hash=ast_hash,
                ast_tree=typed.to_dict(),
            )
        except ValueError as exc:
            from .dsl import ValidationReport

            return ValidationReport(
                ok=False,
                normalized_formula=formula.strip(),
                errors=[str(exc)],
            )
