"""
RL feedback loop via prompt engineering.

StrategyMemory tracks which generation strategies (themes, operators,
features) produce good alpha factors, then builds natural-language
feedback summaries that reshape the LLM's generation behaviour
round-over-round. This is "RL without gradient updates".
"""

from __future__ import annotations

import ast
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class StrategyRecord:
    """One generation attempt and its outcome."""

    round_idx: int
    theme_id: str
    operator_pattern: str
    features_used: list[str]
    formula: str
    fitness: float
    rank_ic: float
    sharpe: float
    turnover: float
    is_novel: bool


class StrategyMemory:
    """Tracks which generation strategies produce good results.

    Core of the "RL via prompt engineering" approach: instead of gradient
    updates, we maintain statistics that reshape LLM prompts each round.

    The UCB1 bandit in ``select_theme_ucb`` provides principled
    exploration/exploitation of financial themes.
    """

    def __init__(
        self,
        max_records: int = 2000,
        persistence_path: str | None = None,
        all_theme_ids: list[str] | None = None,
    ) -> None:
        self._records: deque[StrategyRecord] = deque(maxlen=max_records)
        self._theme_stats: dict[str, _ThemeStats] = defaultdict(_ThemeStats)
        self._operator_stats: dict[str, _RunningStats] = defaultdict(_RunningStats)
        self._feature_stats: dict[str, _RunningStats] = defaultdict(_RunningStats)
        # Per-strategy reward stats (e.g. "llm_evolution", "mcts_refinement")
        self._strategy_stats: dict[str, _RunningStats] = defaultdict(_RunningStats)
        self._persistence_path = persistence_path
        self._all_theme_ids: list[str] = all_theme_ids or []
        self._total_generated: int = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        formula: str,
        theme_id: str,
        metrics: dict[str, float],
        is_novel: bool,
        round_idx: int = 0,
        all_fields: frozenset[str] | None = None,
        strategy: str | None = None,
    ) -> None:
        """Record one evaluation result. Called after every full evaluation."""
        fitness = float(metrics.get("fitness", 0.0))
        rank_ic = float(metrics.get("rank_ic", 0.0) or 0.0)
        sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
        turnover = float(metrics.get("avg_turnover", 0.0) or 0.0)
        op_pattern = self.extract_operator_pattern(formula)
        features = self.extract_features_used(formula, all_fields) if all_fields else []

        rec = StrategyRecord(
            round_idx=round_idx,
            theme_id=theme_id,
            operator_pattern=op_pattern,
            features_used=features,
            formula=formula,
            fitness=fitness,
            rank_ic=rank_ic,
            sharpe=sharpe,
            turnover=turnover,
            is_novel=is_novel,
        )
        self._records.append(rec)
        self._total_generated += 1

        # Update theme stats
        ts = self._theme_stats[theme_id]
        ts.count += 1
        ts.fitness_sum += fitness
        if fitness > ts.best_fitness:
            ts.best_fitness = fitness
            ts.best_formula = formula
        if fitness > -5.0:  # not rejected
            ts.success_count += 1

        # Update operator stats
        if op_pattern:
            self._operator_stats[op_pattern].update(fitness)

        # Update feature stats
        for feat in features:
            self._feature_stats[feat].update(fitness)

        # Update per-strategy reward stats (enables adaptive scheduling)
        if strategy:
            self._strategy_stats[strategy].update(fitness)

    # ------------------------------------------------------------------
    # Per-strategy stats (for AdaptiveScheduler)
    # ------------------------------------------------------------------

    def get_strategy_stats(self) -> dict[str, dict[str, Any]]:
        return {
            name: {"mean_fitness": s.mean, "count": s.count} for name, s in self._strategy_stats.items() if s.count > 0
        }

    def suggest_strategy_mix(
        self,
        strategy_names: list[str],
        *,
        goal: str = "maximize_fitness",
        exploration: float = 1.0,
    ) -> dict[str, float]:
        """Return normalised UCB1 weights over the given strategy names.

        ``goal`` is reserved for future "diverse" / "reduce_turnover" modes;
        for now it's always "maximize_fitness" (the default reward signal).
        Unseen strategies get infinite UCB so at least one trial is
        guaranteed before any exploitation kicks in.
        """
        if not strategy_names:
            return {}
        total = max(self._total_generated, 1)
        raw_scores: dict[str, float] = {}
        for name in strategy_names:
            stats = self._strategy_stats.get(name)
            if stats is None or stats.count == 0:
                raw_scores[name] = math.inf
                continue
            mean = stats.mean
            bonus = exploration * math.sqrt(math.log(total) / stats.count)
            raw_scores[name] = mean + bonus

        # Convert infinite scores to flat-uniform + keep exploiters proportional.
        inf_names = [n for n, v in raw_scores.items() if math.isinf(v)]
        if inf_names:
            return {n: (1.0 / len(inf_names) if n in inf_names else 0.0) for n in strategy_names}

        # Softmax over finite scores with sane temperature to avoid collapse.
        mx = max(raw_scores.values())
        exps = {n: math.exp((v - mx) / max(exploration, 1e-3)) for n, v in raw_scores.items()}
        denom = sum(exps.values()) or 1.0
        return {n: exps[n] / denom for n in strategy_names}

    # ------------------------------------------------------------------
    # UCB1 Bandit — theme selection
    # ------------------------------------------------------------------

    def select_theme_ucb(self, c: float = 1.0) -> str:
        """UCB1 bandit selection of next theme to explore.

        Returns theme_id with highest UCB score.
        ``c`` controls exploration: higher = more exploration.
        """
        if not self._all_theme_ids:
            return "general"

        total = max(self._total_generated, 1)
        best_score = -math.inf
        best_theme = self._all_theme_ids[0]

        for theme_id in self._all_theme_ids:
            ts = self._theme_stats.get(theme_id)
            if ts is None or ts.count == 0:
                # Unvisited — infinite UCB (explore first)
                return theme_id
            mean_fitness = ts.fitness_sum / ts.count
            exploration = c * math.sqrt(math.log(total) / ts.count)
            score = mean_fitness + exploration
            if score > best_score:
                best_score = score
                best_theme = theme_id

        return best_theme

    def select_themes_ucb(self, n: int, c: float = 1.0) -> list[str]:
        """Select n diverse themes via UCB1 (no repeats)."""
        if not self._all_theme_ids:
            return ["general"] * n

        total = max(self._total_generated, 1)
        scores: list[tuple[float, str]] = []
        for theme_id in self._all_theme_ids:
            ts = self._theme_stats.get(theme_id)
            if ts is None or ts.count == 0:
                scores.append((math.inf, theme_id))
            else:
                mean_fitness = ts.fitness_sum / ts.count
                exploration = c * math.sqrt(math.log(total) / ts.count)
                scores.append((mean_fitness + exploration, theme_id))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [tid for _, tid in scores[:n]]

    # ------------------------------------------------------------------
    # Feedback summary — the "policy gradient"
    # ------------------------------------------------------------------

    def build_feedback_summary(self, max_themes: int = 5, max_ops: int = 5) -> str:
        """Build a text summary for injection into LLM prompts.

        This is the core of the RL feedback loop: the text changes every
        round based on accumulated evidence, steering the LLM toward
        productive regions of formula space.
        """
        if not self._records:
            return "## Learning from Past Generations\nNo data yet — explore broadly across all themes and operators."

        lines: list[str] = []
        lines.append(f"## Learning from Past Generations ({len(self._records)} formulas evaluated)\n")

        # --- Top themes ---
        theme_ranking = self.get_theme_ranking()
        good_themes = [(tid, ts) for tid, ts in theme_ranking if ts.count >= 2 and ts.mean_fitness > -3.0]
        bad_themes = [(tid, ts) for tid, ts in theme_ranking if ts.count >= 2 and ts.mean_fitness <= -3.0]

        if good_themes:
            lines.append("### High-performing themes (exploit these)")
            for i, (tid, ts) in enumerate(good_themes[:max_themes], 1):
                best = f"\n   Best: `{ts.best_formula}`" if ts.best_formula else ""
                lines.append(
                    f"{i}. **{tid}**: {ts.count} formulas, "
                    f"mean fitness {ts.mean_fitness:.2f}, "
                    f"success rate {ts.success_rate:.0%}{best}"
                )

        if bad_themes:
            lines.append("\n### Low-performing themes (avoid or modify approach)")
            for tid, ts in bad_themes[:3]:
                lines.append(
                    f"- **{tid}**: {ts.count} formulas, "
                    f"mean fitness {ts.mean_fitness:.2f} — needs different operator/window choices"
                )

        # --- Top operators ---
        top_ops = self.get_top_operators(max_ops)
        if top_ops:
            lines.append("\n### Operator patterns that work")
            for pattern, stats in top_ops:
                lines.append(f"- `{pattern}`: mean fitness {stats.mean:.2f} ({stats.count} uses)")

        # Bad operators
        bad_ops = self.get_worst_operators(3)
        if bad_ops:
            lines.append("\n### Operator patterns to avoid")
            for pattern, stats in bad_ops:
                lines.append(f"- `{pattern}`: mean fitness {stats.mean:.2f} ({stats.count} uses)")

        # --- Underexplored ---
        underexplored = self.get_underexplored_features(5)
        if underexplored:
            lines.append(f"\n### Underexplored features (try these for diversity)\n- {', '.join(underexplored)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Statistics queries
    # ------------------------------------------------------------------

    def get_theme_ranking(self) -> list[tuple[str, _ThemeStats]]:
        """Return themes sorted by mean fitness."""
        ranked = [(tid, ts) for tid, ts in self._theme_stats.items() if ts.count > 0]
        ranked.sort(key=lambda x: x[1].mean_fitness, reverse=True)
        return ranked

    def get_top_operators(self, k: int = 10) -> list[tuple[str, _RunningStats]]:
        """Return operator patterns sorted by mean fitness."""
        ranked = [(pattern, stats) for pattern, stats in self._operator_stats.items() if stats.count >= 2]
        ranked.sort(key=lambda x: x[1].mean, reverse=True)
        return ranked[:k]

    def get_worst_operators(self, k: int = 3) -> list[tuple[str, _RunningStats]]:
        """Return operator patterns with worst mean fitness."""
        ranked = [(pattern, stats) for pattern, stats in self._operator_stats.items() if stats.count >= 2]
        ranked.sort(key=lambda x: x[1].mean)
        return ranked[:k]

    def get_top_features(self, k: int = 10) -> list[tuple[str, float]]:
        """Return features sorted by mean fitness of formulas containing them."""
        ranked = [(field, stats.mean) for field, stats in self._feature_stats.items() if stats.count > 0]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def get_underexplored_features(self, k: int = 5) -> list[str]:
        """Return features with fewest appearances."""
        if not self._records:
            return []
        count_map = {field: stats.count for field, stats in self._feature_stats.items()}
        # Include fields with zero appearances
        all_relevant_fields = {
            "funding_rate",
            "premium_close",
            "open_interest",
            "open_interest_value",
            "long_short_ratio",
            "taker_long_short_vol_ratio",
            "top_trader_long_short_ratio",
            "top_trader_long_short_position_ratio",
            "mark_close",
            "taker_buy_volume",
            "bid_ask_spread",
            "trade_count",
        }
        scored = [(f, count_map.get(f, 0)) for f in all_relevant_fields]
        scored.sort(key=lambda x: x[1])
        return [f for f, _ in scored[:k]]

    # ------------------------------------------------------------------
    # Summaries for LLM context builder
    # ------------------------------------------------------------------

    def get_theme_summary(self) -> dict[str, dict[str, Any]]:
        """Return theme stats as plain dicts (for llm_context)."""
        result: dict[str, dict[str, Any]] = {}
        for tid, ts in self._theme_stats.items():
            if ts.count == 0:
                continue
            result[tid] = {
                "count": ts.count,
                "avg_fitness": ts.mean_fitness,
                "success_rate": ts.success_rate,
                "best_fitness": ts.best_fitness,
            }
        return result

    def get_operator_summary(self) -> dict[str, dict[str, Any]]:
        """Return operator stats as plain dicts (for llm_context)."""
        result: dict[str, dict[str, Any]] = {}
        for op, stats in self._operator_stats.items():
            if stats.count < 2:
                continue
            result[op] = {
                "count": stats.count,
                "avg_fitness": stats.mean,
            }
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist to JSON for cross-session learning."""
        if not self._persistence_path:
            return
        path = Path(self._persistence_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_generated": self._total_generated,
            "records": [asdict(r) for r in self._records],
            "theme_stats": {
                tid: {
                    "count": ts.count,
                    "fitness_sum": ts.fitness_sum,
                    "best_fitness": ts.best_fitness,
                    "best_formula": ts.best_formula,
                    "success_count": ts.success_count,
                }
                for tid, ts in self._theme_stats.items()
            },
            "operator_stats": {op: {"mean": s.mean, "count": s.count} for op, s in self._operator_stats.items()},
            "feature_stats": {f: {"mean": s.mean, "count": s.count} for f, s in self._feature_stats.items()},
            "strategy_stats": {n: {"mean": s.mean, "count": s.count} for n, s in self._strategy_stats.items()},
            "llm_insights": list(self._llm_insights),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def load(self) -> None:
        """Load from persistence."""
        if not self._persistence_path:
            return
        path = Path(self._persistence_path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return

        self._total_generated = data.get("total_generated", 0)

        for rec_dict in data.get("records", []):
            try:
                self._records.append(StrategyRecord(**rec_dict))
            except TypeError:
                continue

        for tid, ts_dict in data.get("theme_stats", {}).items():
            ts = _ThemeStats()
            ts.count = ts_dict.get("count", 0)
            ts.fitness_sum = ts_dict.get("fitness_sum", 0.0)
            ts.best_fitness = ts_dict.get("best_fitness", -999.0)
            ts.best_formula = ts_dict.get("best_formula", "")
            ts.success_count = ts_dict.get("success_count", 0)
            self._theme_stats[tid] = ts

        for op, s_dict in data.get("operator_stats", {}).items():
            s = _RunningStats()
            s.mean = s_dict.get("mean", 0.0)
            s.count = s_dict.get("count", 0)
            self._operator_stats[op] = s

        for f, s_dict in data.get("feature_stats", {}).items():
            s = _RunningStats()
            s.mean = s_dict.get("mean", 0.0)
            s.count = s_dict.get("count", 0)
            self._feature_stats[f] = s

        for name, s_dict in data.get("strategy_stats", {}).items():
            s = _RunningStats()
            s.mean = s_dict.get("mean", 0.0)
            s.count = s_dict.get("count", 0)
            self._strategy_stats[name] = s

        for item in data.get("llm_insights", []) or []:
            if isinstance(item, dict):
                self._llm_insights.append(item)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_operator_pattern(formula: str) -> str:
        """Extract structural fingerprint: sorted top-level operator names.

        Example: "cs_rank(ts_zscore(funding_rate, 20))" -> "cs_rank+ts_zscore"
        """
        try:
            tree = ast.parse(formula, mode="eval")
        except SyntaxError:
            return ""

        ops: list[str] = []

        def _walk(node: ast.AST, depth: int = 0) -> None:
            if depth > 3:
                return
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                ops.append(node.func.id)
            for child in ast.iter_child_nodes(node):
                _walk(child, depth + 1)

        _walk(tree)
        # Deduplicate but preserve order, then sort
        seen: set[str] = set()
        unique: list[str] = []
        for op in ops:
            if op not in seen:
                seen.add(op)
                unique.append(op)
        unique.sort()
        return "+".join(unique)

    @staticmethod
    def extract_features_used(formula: str, all_fields: frozenset[str] | None = None) -> list[str]:
        """Extract which base fields a formula references."""
        if all_fields is None:
            return []
        return [f for f in all_fields if f in formula]


# ------------------------------------------------------------------
# Internal data classes
# ------------------------------------------------------------------


class _ThemeStats:
    __slots__ = ("count", "fitness_sum", "best_fitness", "best_formula", "success_count")

    def __init__(self) -> None:
        self.count: int = 0
        self.fitness_sum: float = 0.0
        self.best_fitness: float = -999.0
        self.best_formula: str = ""
        self.success_count: int = 0

    @property
    def mean_fitness(self) -> float:
        return self.fitness_sum / self.count if self.count > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.count if self.count > 0 else 0.0


class _RunningStats:
    __slots__ = ("mean", "count")

    def __init__(self) -> None:
        self.mean: float = 0.0
        self.count: int = 0

    def update(self, value: float) -> None:
        self.count += 1
        self.mean += (value - self.mean) / self.count
