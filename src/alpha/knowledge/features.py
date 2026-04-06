"""
Systematic derived feature generation for alpha formula discovery.

Generates DSL formula strings (not pre-computed arrays) that the LLM can
reference as building blocks. The VM executes them on-demand — zero cost
for unused features.
"""

from __future__ import annotations

import ast
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.dsl import TensorSchema


@dataclass
class DerivedFeature:
    """A derived feature with its DSL formula and metadata."""

    name: str
    formula: str
    category: str  # ratio, delta, interaction
    base_fields: tuple[str, ...]
    financial_meaning: str


class FeatureKitchen:
    """Generates a catalog of derived features from raw dataset fields.

    Strategy: we generate DSL formula *strings* and put them in the LLM
    prompt as available building blocks. The VM computes them on-demand
    when they appear in generated formulas. Zero cost for unused features.
    """

    def __init__(self, schema: "TensorSchema") -> None:
        self.schema = schema
        self._catalog: list[DerivedFeature] = []
        self._importance: dict[str, _RunningMean] = defaultdict(_RunningMean)

    def build_catalog(self) -> list[DerivedFeature]:
        """Generate a curated catalog of derived features."""
        derived: list[DerivedFeature] = []
        derived.extend(self._generate_ratios())
        derived.extend(self._generate_deltas())
        derived.extend(self._generate_interactions())
        self._catalog = derived
        return derived

    @property
    def catalog(self) -> list[DerivedFeature]:
        if not self._catalog:
            self.build_catalog()
        return self._catalog

    # ------------------------------------------------------------------
    # Ratio features
    # ------------------------------------------------------------------

    def _generate_ratios(self) -> list[DerivedFeature]:
        return [
            DerivedFeature(
                "buy_pressure",
                "div(taker_buy_volume, volume + 1e-12)",
                "ratio", ("taker_buy_volume", "volume"),
                "Fraction of volume from aggressive buyers (0.5 = neutral)",
            ),
            DerivedFeature(
                "sell_pressure",
                "div(volume - taker_buy_volume, volume + 1e-12)",
                "ratio", ("taker_buy_volume", "volume"),
                "Fraction of volume from aggressive sellers",
            ),
            DerivedFeature(
                "avg_trade_size",
                "div(turnover, trade_count + 1e-12)",
                "ratio", ("turnover", "trade_count"),
                "Average notional per trade — whale detection proxy",
            ),
            DerivedFeature(
                "normalized_spread",
                "spread_ratio(bid_ask_spread, close)",
                "ratio", ("bid_ask_spread", "close"),
                "Bid-ask spread as fraction of price — liquidity proxy",
            ),
            DerivedFeature(
                "basis_per_vol",
                "div(premium_close, volatility_n(close, 20) + 1e-12)",
                "ratio", ("premium_close", "close"),
                "Basis normalized by realized volatility — regime-adjusted carry",
            ),
            DerivedFeature(
                "funding_oi_intensity",
                "div(funding_rate, ts_mean(open_interest, 20) + 1e-12)",
                "ratio", ("funding_rate", "open_interest"),
                "Funding rate per unit of OI — leverage cost intensity",
            ),
            DerivedFeature(
                "oi_per_volume",
                "div(open_interest, adv_n(turnover, 20) + 1e-12)",
                "ratio", ("open_interest", "turnover"),
                "Outstanding positions vs recent activity — position crowding proxy",
            ),
            DerivedFeature(
                "price_efficiency",
                "div(abs(returns_n(close, 5)), ts_sum(abs(returns_n(close, 1)), 5) + 1e-12)",
                "ratio", ("close",),
                "5-bar return / sum of 1-bar |returns| — trend efficiency (1 = straight line)",
            ),
            DerivedFeature(
                "close_vwap_ratio",
                "div(close, vwap + 1e-12)",
                "ratio", ("close", "vwap"),
                "Price vs volume-weighted avg — intrabar drift indicator",
            ),
            DerivedFeature(
                "mark_spot_spread",
                "div(close - mark_close, close + 1e-12)",
                "ratio", ("close", "mark_close"),
                "Normalized mark-spot deviation — premium/stress proxy",
            ),
            DerivedFeature(
                "high_low_range",
                "div(high - low, close + 1e-12)",
                "ratio", ("high", "low", "close"),
                "Intrabar range normalized by close — bar-level volatility",
            ),
            DerivedFeature(
                "upper_shadow",
                "div(high - max(open, close), high - low + 1e-12)",
                "ratio", ("high", "low", "open", "close"),
                "Upper shadow fraction — selling pressure at highs",
            ),
            DerivedFeature(
                "lower_shadow",
                "div(min(open, close) - low, high - low + 1e-12)",
                "ratio", ("high", "low", "open", "close"),
                "Lower shadow fraction — buying pressure at lows",
            ),
            DerivedFeature(
                "taker_ls_imbalance",
                "div(taker_long_short_vol_ratio - 1.0, taker_long_short_vol_ratio + 1.0 + 1e-12)",
                "ratio", ("taker_long_short_vol_ratio",),
                "Normalized taker long/short imbalance (-1 to 1 range)",
            ),
        ]

    # ------------------------------------------------------------------
    # Delta features
    # ------------------------------------------------------------------

    def _generate_deltas(self) -> list[DerivedFeature]:
        key_fields = [
            ("open_interest", "OI change — position build-up or unwinding"),
            ("funding_rate", "Funding rate change — leverage cost trend"),
            ("premium_close", "Basis change — derivatives sentiment shift"),
            ("long_short_ratio", "Long/short ratio change — positioning shift"),
            ("taker_long_short_vol_ratio", "Taker LS ratio change — aggression shift"),
        ]
        windows = (5, 10, 20)
        deltas: list[DerivedFeature] = []
        for field_name, meaning in key_fields:
            if field_name not in self.schema.fields:
                continue
            for w in windows:
                deltas.append(DerivedFeature(
                    f"{field_name}_delta_{w}",
                    f"delta({field_name}, {w})",
                    "delta", (field_name,),
                    f"{meaning} over {w} bars",
                ))
        return deltas

    # ------------------------------------------------------------------
    # Interaction features
    # ------------------------------------------------------------------

    def _generate_interactions(self) -> list[DerivedFeature]:
        return [
            DerivedFeature(
                "oi_price_correlation",
                "ts_corr(delta(open_interest, 10), returns_n(close, 10), 20)",
                "interaction", ("open_interest", "close"),
                "OI-price correlation: positive = trend confirmation, negative = divergence/squeeze",
            ),
            DerivedFeature(
                "flow_toxicity",
                "div(taker_buy_volume - div(volume, 2), volume + 1e-12)",
                "interaction", ("taker_buy_volume", "volume"),
                "Net taker imbalance as fraction of total volume",
            ),
            DerivedFeature(
                "funding_basis_spread",
                "ts_zscore(funding_rate, 20) - ts_zscore(premium_close, 20)",
                "interaction", ("funding_rate", "premium_close"),
                "Funding vs basis z-score divergence — arbitrage signal",
            ),
            DerivedFeature(
                "volume_price_divergence",
                "ts_corr(volume, abs(returns_n(close, 1)), 20)",
                "interaction", ("volume", "close"),
                "Volume-price correlation: low = stealth accumulation, high = normal trending",
            ),
            DerivedFeature(
                "smart_dumb_divergence",
                "delta(top_trader_long_short_ratio, 5) - delta(long_short_ratio, 5)",
                "interaction", ("top_trader_long_short_ratio", "long_short_ratio"),
                "Whale vs retail positioning delta — smart money leading signal",
            ),
            DerivedFeature(
                "oi_volume_divergence",
                "ts_zscore(delta(open_interest, 10), 20) - ts_zscore(delta(volume, 10), 20)",
                "interaction", ("open_interest", "volume"),
                "OI growing without volume = speculative positioning build-up",
            ),
            DerivedFeature(
                "momentum_quality",
                "ts_corr(returns_n(close, 1), volume, 20)",
                "interaction", ("close", "volume"),
                "Return-volume correlation — volume-confirmed momentum",
            ),
            DerivedFeature(
                "spread_vol_stress",
                "ts_corr(spread_ratio(bid_ask_spread, close), volatility_n(close, 10), 20)",
                "interaction", ("bid_ask_spread", "close"),
                "Spread-volatility correlation — market stress indicator",
            ),
            DerivedFeature(
                "funding_momentum",
                "ts_corr(delta(funding_rate, 5), returns_n(close, 5), 20)",
                "interaction", ("funding_rate", "close"),
                "Funding-return alignment — persistent when trend is real, divergent at exhaustion",
            ),
            DerivedFeature(
                "whale_activity_signal",
                "ts_zscore(div(turnover, trade_count + 1e-12), 20)",
                "interaction", ("turnover", "trade_count"),
                "Average trade size z-score — spikes indicate institutional activity",
            ),
        ]

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def get_catalog_as_prompt_section(self) -> str:
        """Format catalog for injection into LLM prompts."""
        sections: dict[str, list[str]] = defaultdict(list)
        for feat in self.catalog:
            sections[feat.category].append(
                f"  - **{feat.name}** = `{feat.formula}`\n    {feat.financial_meaning}"
            )

        parts: list[str] = []
        category_names = {"ratio": "Ratios", "delta": "Changes Over Time", "interaction": "Cross-Field Interactions"}
        for cat in ("ratio", "delta", "interaction"):
            items = sections.get(cat, [])
            if items:
                parts.append(f"### {category_names.get(cat, cat)}\n" + "\n".join(items))

        return "## Derived Features (use as sub-expressions or building blocks)\n\n" + "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Feature importance tracking
    # ------------------------------------------------------------------

    def track_feature_importance(self, formula: str, fitness: float) -> None:
        """Update importance scores based on which features appear in good formulas."""
        for feat in self.catalog:
            # Check if any of the feature's base fields appear in the formula
            for base_field in feat.base_fields:
                if base_field in formula:
                    self._importance[feat.name].update(fitness)
                    break

    def get_importance_ranking(self) -> list[tuple[str, float, int]]:
        """Return features ranked by mean fitness of formulas that use them.

        Returns list of (name, mean_fitness, count).
        """
        ranking = [
            (name, stats.mean, stats.count)
            for name, stats in self._importance.items()
            if stats.count > 0
        ]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def get_underexplored_features(self, k: int = 5) -> list[str]:
        """Return feature names with fewest appearances — for exploration."""
        all_names = {f.name for f in self.catalog}
        explored = {name for name, stats in self._importance.items() if stats.count > 0}
        unexplored = list(all_names - explored)
        if len(unexplored) >= k:
            return unexplored[:k]
        # Also include low-count features
        low_count = sorted(
            [(name, stats.count) for name, stats in self._importance.items() if stats.count > 0],
            key=lambda x: x[1],
        )
        result = unexplored + [name for name, _ in low_count]
        return result[:k]


class _RunningMean:
    """Incremental mean tracker."""

    __slots__ = ("mean", "count")

    def __init__(self) -> None:
        self.mean: float = 0.0
        self.count: int = 0

    def update(self, value: float) -> None:
        self.count += 1
        self.mean += (value - self.mean) / self.count
