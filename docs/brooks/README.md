# Brooks Price-Action Architecture

The `src/brooks/` package houses the full Brooks-style trading pipeline —
features → structure → regime → patterns → analyst → aggregator → Trader's
Equation → EV gate → risk layer — behind a single public strategy,
`BrooksStrategy`, registered as `"brooks"` in `StrategyRegistry`.

## Package Map

| Layer | Module | Purpose |
|-------|--------|---------|
| L1 features | `src/brooks/features.py` | Streaming per-bar features + confirmed swings |
| L2 structure | `src/brooks/structure.py` | Always-in, leg dir/length, micro-channel fits |
| L3 regime | `src/brooks/regime.py` | 7-state Brooks regime classifier |
| L3 patterns | `src/brooks/patterns/` | H1..H4 / L1..L4 / H2/L2 / doubles / wedges / micro-channel / MTR / BP / ii-iii / final-flag / measured-move |
| L4 analyst | `src/brooks/analyst/` | `rule` + `llm:<model>` plug-ins (see below) |
| L5 decision | `src/brooks/decision/` | Signal aggregator + Trader's Equation + EV gate + hit-rate table |
| L6 risk | `src/brooks/risk/` | `KellySizer` / `FixedPercentSizer`, `StopLadder`, `TimeStop`, `PortfolioGuard` |
| Strategy | `src/brooks/strategy.py` | `BrooksStrategy` orchestrator (Phase 3.5) |
| Context | `src/brooks/context.py` | `BrooksContext` — single analyst input type |

## Extension Points

### Add a new Pattern Detector

1. Create a subclass of `PatternDetector` in `src/brooks/patterns/<name>.py`.
2. Register it with the decorator:

   ```python
   from src.brooks.patterns.base import DetectorContext, PatternDetector, PatternSignal
   from src.brooks.patterns.registry import PatternRegistry

   @PatternRegistry.register("my_pattern")
   class MyPatternDetector(PatternDetector):
       def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
           ...
   ```

3. Import the new module from `src/brooks/patterns/__init__.py` so the
   decorator actually runs at process start.
4. The `RuleAnalyst` (`src/brooks/analyst/rule.py`) picks up every registered
   detector automatically, so no other wiring is needed.
5. Add a unit test in `tests/brooks/test_patterns_<name>.py`.

### Add a new Analyst (Provider)

Analysts consume a `BrooksContext` and emit unified `Signal` records.

1. Implement the protocol in `src/brooks/analyst/<name>.py`:

   ```python
   from src.brooks.analyst.base import AnalystRegistry
   from src.brooks.context import BrooksContext
   from src.brooks.schema import Signal

   @AnalystRegistry.register("my_analyst")
   class MyAnalyst:
       name: str  # set by the decorator

       async def analyze(self, ctx: BrooksContext) -> list[Signal]:
           ...
   ```

2. Import it from `src/brooks/analyst/__init__.py` so the registration
   fires when the strategy package loads.
3. Selection happens at strategy-construction time:

   ```python
   BrooksStrategy(analyst="my_analyst", analyst_params={...})
   ```

The `llm:<model>` shape is a convention followed by
`LLMAnalyst` (`src/brooks/analyst/llm.py`); other analysts pick any
string they like.

### Add a new LLM Provider

LLM analysts delegate to a `Provider` from `src.alpha.llm.provider`.

1. Implement the `Provider` protocol (complete with `cache`, usage, and
   latency reporting) in a new module under `src/alpha/llm/providers/`.
2. Register it via whatever factory your deployment uses (see the
   existing Anthropic / OpenAI providers for reference).
3. Pass an instance in through `analyst_params`:

   ```python
   BrooksStrategy(
       analyst="llm:my-model",
       analyst_params={"provider": MyProvider(), "model": "my-model-1"},
   )
   ```

## Higher-Timeframe (HTF) Deep Integration (Phase 3.5)

`TFSnapshot` now carries `features` / `structure` / `regime` alongside
`bars`, and `BrooksContext` exposes three alignment helpers:

* `ctx.htf_alignment_for(side)` → `"aligned"`, `"conflict"`, or
  `"neutral"` — based on HTF regime trend vs. the candidate side.
* `ctx.htf_aligned_for(side)` → `bool` alias for `== "aligned"`.
* `ctx.htf_alignment_score(side)` → weighted score in `[-1, 1]` using
  each HTF's `RegimeSnapshot.confidence` as the weight.

The strategy feeds the tag through `EVGate.filter(..., htf_alignment=tag)`
which drives the Trader's Equation probability multiplier:

| Tag | Probability adjustment |
|-----|------------------------|
| `"aligned"`  | `p *= 1.2` (hard-capped at `0.95`) |
| `"conflict"` | `p *= 0.8` |
| `"neutral"` or `None` | pass-through |

The multipliers are tunable via `TraderEquation(htf_aligned_mult=...,
htf_conflict_mult=..., htf_prob_cap=...)`.

## Entry Point

```python
from src.brooks.strategy import BrooksStrategy

strategy = BrooksStrategy(
    analyst="rule",                  # "rule" | "llm:<model>" | "ensemble.<name>"
    analyst_params={...},
    aggregator_params={"confluence_n": 1},
    te_params={"cost_r": 0.05, "htf_aligned_mult": 1.2},
    min_expected_r=0.1,
    sizer_params={"kind": "kelly"},
    stop_ladder_params={},
    time_stop_params={},
    portfolio_params={"max_daily_risk_pct": 0.03},
    mtf_intervals=["1h"],            # turn on HTF pipeline
    base_interval="5m",
)
```

The strategy runs each bar through:

```
bar → features → structure → regime
                                ↓
                        BrooksContext (primary + HTF snapshots)
                                ↓
                        Analyst.analyze() → Signals
                                ↓
                        SignalAggregator.resolve()
                                ↓
                        EVGate.filter(htf_alignment=ctx.htf_alignment_for(side))
                                ↓
                        Sizer.size() + PortfolioGuard.can_open()
                                ↓
                        submit stop-entry; manage via StopLadder/TimeStop
```
