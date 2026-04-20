"""Adaptive per-round strategy scheduling.

Wraps a list of concrete ``SearchStrategy`` implementations and decides
which ones to run in each round based on accumulated ``StrategyMemory``
reward stats. Unseen strategies are forced at least once before any
exploitation (UCB1 semantics).

This replaces the static ``SearchMode`` → fixed-strategy-set binding:
old modes still work as presets (they initialise the list), but the
scheduler dynamically re-weights round-by-round based on observed fit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from loguru import logger

from ..infra.tracing import tracer


@dataclass
class AdaptiveSchedulerConfig:
    exploration: float = 1.0  # UCB1 exploration weight
    min_probability: float = 0.05  # floor so no strategy is ever starved
    cooldown_rounds: int = 0  # rounds to skip after a pick (0 = disabled)
    warmup_rounds: int = 2  # every strategy activates during warmup


class AdaptiveScheduler:
    """Select a subset of strategies to activate each round.

    Usage (inside the orchestrator or caller)::

        scheduler = AdaptiveScheduler(strategies, memory)
        for round_idx in range(rounds):
            activated = scheduler.pick(round_idx)
            for strat in activated:
                strat.generate_candidates(ctx)
            scheduler.observe(activated, rewards_by_name)
    """

    def __init__(
        self,
        strategies: list[Any],
        memory: Any | None = None,
        config: AdaptiveSchedulerConfig | None = None,
    ) -> None:
        self.strategies = list(strategies)
        self.memory = memory
        self.config = config or AdaptiveSchedulerConfig()
        self._cooldown: dict[str, int] = {}
        self._round_count: int = 0
        self._rng = random.Random(0xA1F4)

    def strategy_names(self) -> list[str]:
        return [getattr(s, "name", type(s).__name__) for s in self.strategies]

    def suggestions(self) -> dict[str, float]:
        """Return current mix weights (used by the UI/debug tools)."""
        names = self.strategy_names()
        if self.memory is None:
            return {n: 1.0 / max(len(names), 1) for n in names}
        return self.memory.suggest_strategy_mix(names, exploration=self.config.exploration)

    def pick(self, round_idx: int) -> list[Any]:
        """Return the concrete strategy objects activated for this round."""
        with tracer.start_span(
            "adaptive_scheduler.pick",
            kind="search",
            round=round_idx,
        ) as span:
            # Warmup + unseen-first: activate everything so every strategy
            # gets at least one trial before we start exploiting.
            if round_idx < self.config.warmup_rounds or self.memory is None:
                span.set("mode", "warmup")
                return list(self.strategies)

            weights = self.suggestions()
            span.set("weights", {k: round(v, 3) for k, v in weights.items()})

            # Cool-down decrement; strategies in cool-down are skipped.
            for name in list(self._cooldown.keys()):
                self._cooldown[name] -= 1
                if self._cooldown[name] <= 0:
                    del self._cooldown[name]

            eligible: list[Any] = []
            for strat in self.strategies:
                name = getattr(strat, "name", type(strat).__name__)
                if name in self._cooldown:
                    continue
                probability = max(weights.get(name, 0.0), self.config.min_probability)
                if self._rng.random() <= probability:
                    eligible.append(strat)

            # Never return empty — fall back to the current top-weighted strat.
            if not eligible:
                top_name = max(weights, key=weights.get)
                for strat in self.strategies:
                    if getattr(strat, "name", type(strat).__name__) == top_name:
                        eligible = [strat]
                        break

            for strat in eligible:
                name = getattr(strat, "name", type(strat).__name__)
                if self.config.cooldown_rounds > 0:
                    self._cooldown[name] = self.config.cooldown_rounds
            span.set("activated", [getattr(s, "name", type(s).__name__) for s in eligible])
            self._round_count += 1
            return eligible

    def observe(self, activated: list[Any], rewards_by_name: dict[str, float]) -> None:
        """Record per-strategy mean fitness after a round.

        The memory's internal UCB1 counters are updated through ``record``
        calls made by the orchestrator as each evaluation completes — this
        hook is a light-weight secondary log for debugging.
        """
        if not rewards_by_name:
            return
        logger.debug(
            "adaptive_scheduler.observe round={} rewards={}",
            self._round_count,
            {k: round(v, 3) for k, v in rewards_by_name.items()},
        )
