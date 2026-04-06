"""
Structured financial domain knowledge for LLM prompt construction.

Provides FinancialTheme and FeatureGroup definitions that replace
hardcoded theme lines in llm.py with rich, structured financial
semantics for crypto perpetual futures alpha discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FinancialTheme:
    """A financial hypothesis that guides alpha formula generation."""

    theme_id: str
    name: str
    category: str  # derivatives, microstructure, momentum, mean_reversion, sentiment, flow, volatility
    hypothesis: str
    relevant_fields: tuple[str, ...]
    suggested_operators: tuple[str, ...]
    example_formulas: tuple[str, ...]
    anti_patterns: tuple[str, ...]
    window_guidance: str


@dataclass(frozen=True)
class FeatureGroup:
    """Semantically grouped features with interaction hints."""

    group_id: str
    name: str
    fields: tuple[str, ...]
    financial_meaning: str
    interaction_hints: tuple[str, ...]


class FinancialKnowledgeBase:
    """Repository of structured financial knowledge for prompt injection."""

    def __init__(self) -> None:
        self.themes: dict[str, FinancialTheme] = self._build_themes()
        self.feature_groups: dict[str, FeatureGroup] = self._build_feature_groups()

    # ------------------------------------------------------------------
    # Themes
    # ------------------------------------------------------------------

    @staticmethod
    def _build_themes() -> dict[str, FinancialTheme]:
        themes: list[FinancialTheme] = [
            # ---- derivatives ----
            FinancialTheme(
                theme_id="funding_basis_arb",
                name="Funding-Basis Arbitrage",
                category="derivatives",
                hypothesis=(
                    "When funding rate deviates significantly from premium (basis), "
                    "arbitrageurs step in causing mean reversion. High funding + low basis "
                    "= crowded long, expect reversal. Track delta(funding_rate) vs "
                    "delta(premium_close) divergence."
                ),
                relevant_fields=("funding_rate", "premium_close", "premium_open", "open_interest"),
                suggested_operators=("ts_zscore", "delta", "ts_corr", "cs_rank", "decay_linear"),
                example_formulas=(
                    "cs_rank(ts_zscore(funding_rate, 20) - ts_zscore(premium_close, 20))",
                    "cs_rank(delta(funding_rate, 5) - delta(premium_close, 5))",
                    "cs_rank(ts_corr(funding_rate, premium_close, 20))",
                ),
                anti_patterns=(
                    "Don't use funding_rate raw without normalization — scale varies across symbols",
                    "Don't use very short windows (< 5) on funding_rate — updates every 8h",
                ),
                window_guidance="Medium-term (12-48 bars at 5m = 1-4 hours)",
            ),
            FinancialTheme(
                theme_id="oi_momentum_divergence",
                name="OI-Price Momentum Divergence",
                category="derivatives",
                hypothesis=(
                    "When open interest increases in the same direction as price, it confirms "
                    "the trend. When OI rises but price stalls (or vice versa), it signals "
                    "potential reversal — the new positions are about to be squeezed."
                ),
                relevant_fields=("open_interest", "open_interest_value", "close", "volume"),
                suggested_operators=("delta", "ts_corr", "ts_zscore", "cs_rank", "returns_n"),
                example_formulas=(
                    "cs_rank(ts_corr(delta(open_interest, 10), returns_n(close, 10), 20))",
                    "cs_rank(ts_zscore(delta(open_interest, 5), 20) - ts_zscore(returns_n(close, 5), 20))",
                    "cs_rank(delta(open_interest_value, 10) - delta(close, 10))",
                ),
                anti_patterns=(
                    "Don't use open_interest raw — always use delta or ts_zscore for stationarity",
                ),
                window_guidance="Medium-term (10-40 bars)",
            ),
            # ---- flow ----
            FinancialTheme(
                theme_id="taker_flow_imbalance",
                name="Taker Flow Imbalance",
                category="flow",
                hypothesis=(
                    "Aggressive buyer/seller imbalance (taker_buy_volume / volume ratio) "
                    "predicts short-term price direction. Sustained imbalance without price "
                    "follow-through signals exhaustion and potential reversal."
                ),
                relevant_fields=("taker_buy_volume", "taker_buy_quote_volume", "volume", "turnover", "close"),
                suggested_operators=("div", "ts_zscore", "ts_mean", "cs_rank", "delta"),
                example_formulas=(
                    "cs_rank(ts_zscore(div(taker_buy_volume, volume + 1e-12), 20) - 0.5)",
                    "cs_rank(delta(div(taker_buy_volume, volume + 1e-12), 5))",
                    "cs_rank(ts_corr(div(taker_buy_volume, volume + 1e-12), returns_n(close, 1), 20))",
                ),
                anti_patterns=(
                    "Always add epsilon (1e-12) to divisor to avoid div-by-zero",
                ),
                window_guidance="Short-term (5-20 bars)",
            ),
            FinancialTheme(
                theme_id="volume_profile_anomaly",
                name="Volume Profile Anomaly",
                category="flow",
                hypothesis=(
                    "Abnormal volume (relative to rolling average) signals institutional "
                    "activity. Volume spikes without proportional price moves indicate "
                    "accumulation/distribution. Trade count divergence from volume reveals "
                    "whale activity (few large trades vs many small ones)."
                ),
                relevant_fields=("volume", "turnover", "trade_count", "close", "high", "low"),
                suggested_operators=("ts_zscore", "div", "cs_rank", "adv_n", "ts_mean", "delta"),
                example_formulas=(
                    "cs_rank(ts_zscore(div(turnover, trade_count + 1e-12), 20))",
                    "cs_rank(div(volume, adv_n(turnover, 20) + 1e-12))",
                    "cs_rank(ts_zscore(volume, 20) - ts_zscore(abs(returns_n(close, 1)), 20))",
                ),
                anti_patterns=(
                    "Don't use absolute volume — always normalize cross-sectionally or time-series",
                ),
                window_guidance="Short to medium (5-30 bars)",
            ),
            # ---- microstructure ----
            FinancialTheme(
                theme_id="microstructure_toxicity",
                name="Market Microstructure Toxicity",
                category="microstructure",
                hypothesis=(
                    "High bid-ask spread relative to normal levels signals low liquidity or "
                    "adverse selection. Combined with volume patterns, spread dynamics reveal "
                    "market maker stress and impending volatility."
                ),
                relevant_fields=("bid_ask_spread", "close", "volume", "trade_count", "turnover"),
                suggested_operators=("spread_ratio", "ts_zscore", "cs_rank", "div", "ts_mean"),
                example_formulas=(
                    "cs_rank(ts_zscore(spread_ratio(bid_ask_spread, close), 20))",
                    "cs_rank(neg(ts_zscore(div(turnover, trade_count + 1e-12), 20)))",
                    "cs_rank(ts_corr(spread_ratio(bid_ask_spread, close), volume, 20))",
                ),
                anti_patterns=(
                    "Don't use bid_ask_spread raw — always use spread_ratio(bid_ask_spread, close) for normalization",
                ),
                window_guidance="Short-term (5-20 bars)",
            ),
            # ---- sentiment ----
            FinancialTheme(
                theme_id="sentiment_extreme_reversal",
                name="Sentiment Extreme Reversal",
                category="sentiment",
                hypothesis=(
                    "Extreme values of long_short_ratio or taker_long_short_vol_ratio signal "
                    "crowded positioning. When the crowd is extremely bullish/bearish, "
                    "contrarian reversal becomes likely."
                ),
                relevant_fields=(
                    "long_short_ratio", "taker_long_short_vol_ratio",
                    "top_trader_long_short_ratio", "top_trader_long_short_position_ratio", "close",
                ),
                suggested_operators=("ts_zscore", "cs_rank", "neg", "delta", "decay_linear"),
                example_formulas=(
                    "cs_rank(neg(ts_zscore(long_short_ratio, 20)))",
                    "cs_rank(neg(ts_zscore(taker_long_short_vol_ratio, 20)))",
                    "cs_rank(delta(top_trader_long_short_ratio, 5) - delta(close, 5))",
                ),
                anti_patterns=(
                    "Don't use these ratios in momentum mode — they are contrarian indicators",
                    "Don't use very short windows — ratios update slowly",
                ),
                window_guidance="Medium-term (20-60 bars)",
            ),
            FinancialTheme(
                theme_id="whale_positioning",
                name="Whale Positioning Shift",
                category="sentiment",
                hypothesis=(
                    "Changes in top trader positioning (long_short_ratio, position_ratio) "
                    "lead retail by hours. Delta of these ratios is a leading indicator: "
                    "whales reducing longs before retail = impending sell-off."
                ),
                relevant_fields=(
                    "top_trader_long_short_ratio", "top_trader_long_short_position_ratio",
                    "long_short_ratio", "close", "open_interest",
                ),
                suggested_operators=("delta", "ts_zscore", "cs_rank", "ts_corr", "decay_linear"),
                example_formulas=(
                    "cs_rank(delta(top_trader_long_short_ratio, 10))",
                    "cs_rank(decay_linear(delta(top_trader_long_short_position_ratio, 5), 10))",
                    "cs_rank(ts_corr(delta(top_trader_long_short_ratio, 5), returns_n(close, 5), 20))",
                ),
                anti_patterns=(
                    "Don't use raw ratio levels — use delta for change detection",
                ),
                window_guidance="Medium-term (10-40 bars)",
            ),
            # ---- volatility ----
            FinancialTheme(
                theme_id="volatility_regime_switch",
                name="Volatility Regime Switch",
                category="volatility",
                hypothesis=(
                    "Volatility compression (ATR/std contraction) precedes expansion (breakout). "
                    "The ratio of short-term to long-term volatility detects regime transitions. "
                    "Low vol + high OI = energy building for directional move."
                ),
                relevant_fields=("high", "low", "close", "open_interest", "volume"),
                suggested_operators=("volatility_n", "atr_n", "ts_std", "div", "cs_rank", "ts_zscore"),
                example_formulas=(
                    "cs_rank(div(volatility_n(close, 5), volatility_n(close, 20) + 1e-12))",
                    "cs_rank(div(atr_n(high, low, close, 5), atr_n(high, low, close, 20) + 1e-12))",
                    "cs_rank(neg(ts_zscore(volatility_n(close, 20), 40)))",
                ),
                anti_patterns=(
                    "Don't use volatility as a directional signal directly — it's symmetric",
                ),
                window_guidance="Medium to long (10-60 bars)",
            ),
            FinancialTheme(
                theme_id="intraday_range",
                name="Intraday Range Dynamics",
                category="volatility",
                hypothesis=(
                    "The high-low range relative to close (true_range) captures intrabar "
                    "volatility. Expanding range with declining volume = exhaustion. "
                    "Contracting range with rising OI = coiling for breakout."
                ),
                relevant_fields=("high", "low", "close", "open", "volume", "open_interest"),
                suggested_operators=("true_range", "atr_n", "hlc3", "div", "cs_rank", "ts_zscore"),
                example_formulas=(
                    "cs_rank(ts_zscore(true_range(high, low, close), 20))",
                    "cs_rank(div(true_range(high, low, close), atr_n(high, low, close, 20) + 1e-12))",
                    "cs_rank(ts_corr(true_range(high, low, close), volume, 20))",
                ),
                anti_patterns=(),
                window_guidance="Short-term (5-20 bars)",
            ),
            # ---- momentum ----
            FinancialTheme(
                theme_id="momentum_decay",
                name="Momentum Decay & Reversal",
                category="momentum",
                hypothesis=(
                    "Short-term momentum (returns over 5 bars) that diverges from medium-term "
                    "trend (returns over 20 bars) signals exhaustion. Decaying momentum + "
                    "rising volume = distribution. Use decay_linear for recency weighting."
                ),
                relevant_fields=("close", "vwap", "volume", "turnover"),
                suggested_operators=("returns_n", "ts_zscore", "decay_linear", "cs_rank", "ts_mean", "delta"),
                example_formulas=(
                    "cs_rank(decay_linear(returns_n(close, 1), 10))",
                    "cs_rank(returns_n(close, 5) - returns_n(close, 20))",
                    "cs_rank(ts_zscore(returns_n(vwap, 5), 20))",
                ),
                anti_patterns=(
                    "Don't use pure momentum without smoothing — high turnover and whipsaw risk",
                ),
                window_guidance="Short to medium (5-30 bars)",
            ),
            FinancialTheme(
                theme_id="vwap_reversion",
                name="VWAP Mean Reversion",
                category="mean_reversion",
                hypothesis=(
                    "Price deviating from VWAP signals short-term overbought/oversold. "
                    "Close/VWAP ratio z-scored over rolling window is a classic intraday "
                    "mean-reversion signal. Works best in range-bound regimes."
                ),
                relevant_fields=("close", "vwap", "volume", "turnover"),
                suggested_operators=("div", "ts_zscore", "cs_rank", "neg", "ts_mean"),
                example_formulas=(
                    "cs_rank(neg(ts_zscore(div(close, vwap + 1e-12), 20)))",
                    "cs_rank(neg(close - ts_mean(vwap, 10)))",
                ),
                anti_patterns=(
                    "Don't forget to negate — this is a reversal signal, not momentum",
                ),
                window_guidance="Short-term (5-20 bars)",
            ),
            # ---- mean reversion ----
            FinancialTheme(
                theme_id="mean_reversion_squeeze",
                name="Mean Reversion After Squeeze",
                category="mean_reversion",
                hypothesis=(
                    "After a sharp move (high z-score of returns), price tends to revert. "
                    "The signal is stronger when accompanied by funding rate spike (liquidation "
                    "cascade) or OI drop (forced unwinding)."
                ),
                relevant_fields=("close", "funding_rate", "open_interest", "volume"),
                suggested_operators=("ts_zscore", "returns_n", "cs_rank", "neg", "delta", "ts_corr"),
                example_formulas=(
                    "cs_rank(neg(ts_zscore(returns_n(close, 5), 40)))",
                    "cs_rank(neg(ts_zscore(returns_n(close, 5), 20)) + ts_zscore(delta(open_interest, 5), 20))",
                ),
                anti_patterns=(
                    "Don't apply in trending markets — add vol-regime filter if possible",
                ),
                window_guidance="Medium-term (10-40 bars)",
            ),
            # ---- cross-metric ----
            FinancialTheme(
                theme_id="cross_metric_divergence",
                name="Cross-Metric Divergence",
                category="derivatives",
                hypothesis=(
                    "When multiple derivatives metrics diverge (e.g., funding up but premium down, "
                    "OI rising but volume falling), it signals structural imbalance. "
                    "Combine ts_corr of different metric pairs to detect multi-dimensional stress."
                ),
                relevant_fields=(
                    "funding_rate", "premium_close", "open_interest",
                    "volume", "close", "long_short_ratio",
                ),
                suggested_operators=("ts_corr", "delta", "ts_zscore", "cs_rank", "div"),
                example_formulas=(
                    "cs_rank(ts_corr(delta(funding_rate, 5), delta(premium_close, 5), 20))",
                    "cs_rank(ts_zscore(delta(open_interest, 10), 20) - ts_zscore(delta(volume, 10), 20))",
                    "cs_rank(ts_corr(delta(open_interest, 5), delta(long_short_ratio, 5), 20))",
                ),
                anti_patterns=(
                    "Don't use too many metrics in one formula — keep AST depth manageable",
                ),
                window_guidance="Medium-term (10-40 bars)",
            ),
            FinancialTheme(
                theme_id="premium_dynamics",
                name="Premium (Basis) Dynamics",
                category="derivatives",
                hypothesis=(
                    "Premium = futures - spot basis. Persistent positive premium = bullish "
                    "sentiment; mean-reverting premium after extreme = arbitrage opportunity. "
                    "Premium OHLC bars reveal intrabar basis volatility."
                ),
                relevant_fields=("premium_open", "premium_high", "premium_low", "premium_close", "close"),
                suggested_operators=("ts_zscore", "delta", "cs_rank", "ts_mean", "div", "decay_linear"),
                example_formulas=(
                    "cs_rank(ts_zscore(premium_close, 20))",
                    "cs_rank(decay_linear(delta(premium_close, 5), 10))",
                    "cs_rank(div(premium_high - premium_low, close + 1e-12))",
                ),
                anti_patterns=(
                    "Don't confuse premium_close (basis) with funding_rate (carry cost) — different signals",
                ),
                window_guidance="Medium-term (10-40 bars)",
            ),
            FinancialTheme(
                theme_id="mark_spot_divergence",
                name="Mark-Spot Price Divergence",
                category="derivatives",
                hypothesis=(
                    "Mark price is the fair value estimate. Divergence between mark_close "
                    "and close reflects funding/premium stress. Short-term mark-spot spread "
                    "changes predict funding rate moves."
                ),
                relevant_fields=("mark_close", "mark_open", "mark_high", "mark_low", "close", "funding_rate"),
                suggested_operators=("delta", "ts_zscore", "cs_rank", "div", "ts_corr"),
                example_formulas=(
                    "cs_rank(ts_zscore(close - mark_close, 20))",
                    "cs_rank(delta(close - mark_close, 5))",
                    "cs_rank(ts_corr(close - mark_close, funding_rate, 20))",
                ),
                anti_patterns=(),
                window_guidance="Short to medium (5-20 bars)",
            ),
            FinancialTheme(
                theme_id="liquidity_provision",
                name="Liquidity Provision Anomaly",
                category="microstructure",
                hypothesis=(
                    "When ADV (average daily volume) drops significantly but OI stays flat "
                    "or rises, market makers are withdrawing — expect volatility spike. "
                    "Amihud illiquidity ratio captures price impact per unit volume."
                ),
                relevant_fields=("turnover", "close", "volume", "open_interest", "bid_ask_spread"),
                suggested_operators=("amihud", "adv_n", "ts_zscore", "cs_rank", "div", "delta"),
                example_formulas=(
                    "cs_rank(ts_zscore(amihud(close, turnover, 20), 40))",
                    "cs_rank(neg(div(adv_n(turnover, 5), adv_n(turnover, 20) + 1e-12)))",
                    "cs_rank(delta(amihud(close, turnover, 10), 10))",
                ),
                anti_patterns=(
                    "Amihud is inversely related to liquidity — high amihud = illiquid = risky",
                ),
                window_guidance="Medium-term (10-40 bars)",
            ),
        ]
        return {t.theme_id: t for t in themes}

    # ------------------------------------------------------------------
    # Feature Groups
    # ------------------------------------------------------------------

    @staticmethod
    def _build_feature_groups() -> dict[str, FeatureGroup]:
        groups: list[FeatureGroup] = [
            FeatureGroup(
                group_id="price",
                name="Price Dynamics",
                fields=("open", "high", "low", "close", "vwap", "mark_close", "mark_open", "mark_high", "mark_low"),
                financial_meaning=(
                    "Core price information. HLC3/OHLC4 are noise-reduced alternatives. "
                    "Close vs VWAP = intrabar drift direction."
                ),
                interaction_hints=(
                    "high/low ratio = intrabar range (volatility proxy)",
                    "close vs vwap = intrabar momentum",
                    "close vs mark_close = basis/premium proxy",
                ),
            ),
            FeatureGroup(
                group_id="volume_flow",
                name="Volume & Order Flow",
                fields=("volume", "turnover", "trade_count", "taker_buy_volume", "taker_buy_quote_volume"),
                financial_meaning=(
                    "Trading activity and aggression measures. "
                    "Volume = contract count, turnover = quote volume (USDT). "
                    "Taker buys = aggressive buyers lifting asks."
                ),
                interaction_hints=(
                    "taker_buy_volume / volume = buy pressure ratio (0.5 = neutral)",
                    "turnover / trade_count = average trade size (whale detection)",
                    "volume divergence from price move = accumulation/distribution",
                ),
            ),
            FeatureGroup(
                group_id="derivatives_basis",
                name="Derivatives & Basis",
                fields=("premium_open", "premium_high", "premium_low", "premium_close", "funding_rate"),
                financial_meaning=(
                    "Futures-spot basis and funding cost. "
                    "Premium = basis (futures - spot), funding_rate = periodic carry cost. "
                    "These reflect leverage and directional sentiment in derivatives markets."
                ),
                interaction_hints=(
                    "premium_close = basis, funding_rate = carry cost",
                    "basis expansion + high funding = crowded leverage",
                    "funding vs premium divergence = arbitrage signal",
                ),
            ),
            FeatureGroup(
                group_id="positioning",
                name="Positioning & Sentiment",
                fields=(
                    "open_interest", "open_interest_value",
                    "long_short_ratio", "taker_long_short_vol_ratio",
                    "top_trader_long_short_ratio", "top_trader_long_short_position_ratio",
                ),
                financial_meaning=(
                    "Who is positioned how. OI = total outstanding contracts, "
                    "LS ratios = directional bias. "
                    "Top trader ratios lead retail — whale early-warning."
                ),
                interaction_hints=(
                    "delta(OI) with same-direction price = trend confirmation",
                    "extreme long_short_ratio = contrarian signal",
                    "top_trader vs overall LS divergence = smart money vs dumb money",
                ),
            ),
            FeatureGroup(
                group_id="microstructure",
                name="Market Microstructure",
                fields=("bid_ask_spread", "trade_count"),
                financial_meaning=(
                    "Execution environment quality. "
                    "Wide spread = low liquidity or adverse selection. "
                    "High trade_count with low volume = retail activity."
                ),
                interaction_hints=(
                    "spread_ratio(bid_ask_spread, close) = normalized liquidity proxy",
                    "trade_count / volume = average trade size (inverse)",
                ),
            ),
        ]
        return {g.group_id: g for g in groups}

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def get_theme(self, theme_id: str) -> FinancialTheme:
        return self.themes[theme_id]

    def get_themes_for_category(self, category: str) -> list[FinancialTheme]:
        return [t for t in self.themes.values() if t.category == category]

    def get_all_theme_ids(self) -> list[str]:
        return list(self.themes.keys())

    def build_theme_prompt(self, theme_ids: list[str]) -> str:
        """Build a detailed prompt section for selected themes."""
        sections: list[str] = []
        for i, tid in enumerate(theme_ids, 1):
            theme = self.themes.get(tid)
            if theme is None:
                continue
            examples = "\n".join(f"    {f}" for f in theme.example_formulas)
            anti = "\n".join(f"  - {a}" for a in theme.anti_patterns) if theme.anti_patterns else "  (none)"
            sections.append(
                f"### Theme {i}: {theme.name} [{theme.category}]\n"
                f"**Hypothesis:** {theme.hypothesis}\n"
                f"**Key fields:** {', '.join(theme.relevant_fields)}\n"
                f"**Suggested operators:** {', '.join(theme.suggested_operators)}\n"
                f"**Window guidance:** {theme.window_guidance}\n"
                f"**Examples:**\n{examples}\n"
                f"**Anti-patterns:**\n{anti}"
            )
        return "\n\n".join(sections)

    def build_feature_groups_prompt(self) -> str:
        """Build feature reference organized by financial meaning."""
        sections: list[str] = []
        for group in self.feature_groups.values():
            hints = "\n".join(f"  - {h}" for h in group.interaction_hints)
            sections.append(
                f"### {group.name}\n"
                f"Fields: {', '.join(group.fields)}\n"
                f"{group.financial_meaning}\n"
                f"Interaction hints:\n{hints}"
            )
        return "## Available Fields (by financial meaning)\n\n" + "\n\n".join(sections)
