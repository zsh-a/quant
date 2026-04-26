# Brooks Price-Action Platform

Multi-analyst Brooks-style trading framework — pluggable rule / LLM / VLM /
ensemble analysts feed a unified `Signal` schema into an aggregator → Trader's
Equation → EV gate → risk layer → execution pipeline. Single public entry
point: `BrooksStrategy`, registered as `"brooks"` in `StrategyRegistry`.

## Overview

```
                                    ┌────────────────┐
bar ─► features ─► structure ─► regime ─► BrooksContext (LTF + HTF snapshots)
                                                │
                                                ▼
                                      ┌─────────────────┐
                                      │   Analyst       │  rule / llm / vlm
                                      │  .analyze(ctx)  │  ensemble.{vote,
                                      └────────┬────────┘  router, critic}
                                               │
                                               ▼  list[Signal]
                                  ┌────────────────────────┐
                                  │  SignalAggregator      │  confluence_n
                                  │  → AggregatedDecision  │  conservative entry/stop
                                  └────────────┬───────────┘
                                               │
                                               ▼
                                  ┌────────────────────────┐
                                  │  TraderEquation +      │  HitRateTable bucket
                                  │  EVGate.filter(...)    │  HTF prob multiplier
                                  └────────────┬───────────┘
                                               │
                                               ▼  list[Decision]
                                  ┌────────────────────────┐
                                  │  Sizer + StopLadder +  │  Kelly / Fixed / BE-ladder
                                  │  TimeStop +            │  N-bar invalidation
                                  │  PortfolioGuard        │  daily-risk / corr cap
                                  └────────────┬───────────┘
                                               │
                                               ▼
                                       broker stop-entry → fill → manage
```

Source-of-truth files:

| Concept | File |
|---------|------|
| Strategy orchestrator | `src/brooks/strategy.py` |
| Analyst input        | `src/brooks/context.py` (`BrooksContext`, `TFSnapshot`) |
| Schemas              | `src/brooks/schema.py` (`Signal`, `Decision`, `Order`) |
| Pattern detectors    | `src/brooks/patterns/` |
| Analyst plug-ins     | `src/brooks/analyst/` (`base`, `rule`, `llm`, `vlm`, `ensemble`) |
| Context renderers    | `src/brooks/render/text.py`, `src/brooks/render/chart.py` |
| Decision pipeline    | `src/brooks/decision/` (`aggregator`, `trader_equation`, `ev_gate`, `hit_rate`) |
| Risk layer           | `src/brooks/risk/` (`sizer`, `stop_ladder`, `time_stop`, `portfolio_guard`) |
| Eval pipeline        | `src/brooks/eval/` (`golden`, `auto_label`, `runner`, `report`, `leaderboard`) |
| Prompts              | `src/brooks/prompts.py` + `prompts/brooks/` |
| Live (paper) runner  | `src/tasks/brooks_live_task.py` |
| Leaderboard task     | `src/tasks/brooks_leaderboard_task.py` |
| Configs              | `config/brooks/{ensemble_router,leaderboard}.yaml`, `config/llm/registry.yaml` |
| Taxonomy             | `docs/brooks/taxonomy.yaml` (single source of truth for bar/pattern/regime vocab) |

## Quick Start

Minimum runnable backtest, rule analyst:

```python
from src.brooks.strategy import BrooksStrategy

strategy = BrooksStrategy(
    analyst="rule",
    aggregator_params={"confluence_n": 1},
    sizer_params={"kind": "kelly"},
    portfolio_params={"max_daily_risk_pct": 0.03},
    base_interval="5m",
    mtf_intervals=["1h"],
)
# Plug `strategy` into the standard backtest engine — it implements the
# Strategy ABC from src/core/base.py and auto-registers under name "brooks".
```

Switch analyst by changing one argument:

```python
# Pure rule baseline
BrooksStrategy(analyst="rule")

# LLM analyst (Anthropic Opus 4.7)
from src.alpha.llm.providers.anthropic import AnthropicProvider

BrooksStrategy(
    analyst="llm",
    analyst_params={
        "provider": AnthropicProvider(model="claude-opus-4-7"),
        "model": "claude-opus-4-7",
    },
)

# VLM analyst (Gemini 2.5 Pro on rendered chart PNG)
from src.alpha.llm.providers.gemini import GeminiProvider

BrooksStrategy(
    analyst="vlm",
    analyst_params={
        "provider": GeminiProvider(model="gemini-2.5-pro"),
        "model": "gemini-2.5-pro",
    },
)

# Ensemble — rule produces, LLM critiques (see ensemble.critic below)
from src.brooks.analyst.base import AnalystRegistry
from src.brooks.analyst.ensemble import CriticAnalyst

producer = AnalystRegistry.build("rule")
critic = AnalystRegistry.build(
    "llm",
    provider=AnthropicProvider(model="claude-sonnet-4-6"),
    model="claude-sonnet-4-6",
)
BrooksStrategy(
    analyst="ensemble.critic",
    analyst_params={"producer": producer, "critic": critic},
)
```

Live paper trading is just the same strategy under a Celery task — see
[Paper Trading](#paper-trading) below.

## Package Map

| Layer | Module | Purpose |
|-------|--------|---------|
| L1 features  | `src/brooks/features.py`   | Streaming per-bar features + confirmed swings |
| L2 structure | `src/brooks/structure.py`  | Always-in, leg dir/length, micro-channel fits |
| L3 regime    | `src/brooks/regime.py`     | 7-state Brooks regime classifier |
| L3 patterns  | `src/brooks/patterns/`     | H1..H4 / L1..L4 / H2/L2 / doubles / wedges / micro-channel / MTR / breakout-pullback / ii-iii / final-flag / measured-move |
| L4 analyst   | `src/brooks/analyst/`      | `rule` / `llm` / `vlm` / `ensemble.{vote,router,critic}` plug-ins |
| Render       | `src/brooks/render/`       | `text.py` (token-budgeted bar text), `chart.py` (deterministic OHLCV PNG) |
| Prompts      | `src/brooks/prompts.py` + `prompts/brooks/` | `PromptBundle.load()` / `.load_vlm()` — system + concept manual + few-shot |
| L5 decision  | `src/brooks/decision/`     | Aggregator + Trader's Equation + EV gate + HitRateTable |
| L6 risk      | `src/brooks/risk/`         | `KellySizer` / `FixedPercentSizer`, `StopLadder`, `TimeStop`, `PortfolioGuard` |
| Strategy     | `src/brooks/strategy.py`   | `BrooksStrategy` orchestrator |
| Context      | `src/brooks/context.py`    | `BrooksContext`, `TFSnapshot`, `AccountSnapshot` |
| Eval         | `src/brooks/eval/`         | `GoldenDataset`, `AutoLabeler`, `EvalRunner`, `EvalReport`, `Leaderboard` |
| Tasks        | `src/tasks/brooks_live_task.py`, `src/tasks/brooks_leaderboard_task.py` | Paper-trading runner + weekly leaderboard cron |
| API          | `src/api/brooks_router.py` | `/brooks-live` REST + WebSocket panel endpoints (**deprecated** since Phase S6 — migrate to `/brooks-studio/*`) |
| API (Studio) | `src/api/brooks_studio_router.py` | `/brooks-studio/*` timeline + replay-bar + WS bar-event stream |
| UI           | `ui/src/features/studio/` | **Brooks Studio** — unified live + replay K-line workspace (route `/studio`) |

## Analyst Routes

All analysts implement the same Protocol (`src/brooks/analyst/base.py`):

```python
class Analyst(Protocol):
    name: str
    async def analyze(self, ctx: BrooksContext) -> list[Signal]: ...
```

…and register themselves under a string key in `AnalystRegistry`. The
strategy resolves the key with `AnalystRegistry.build(name, **params)`.
Built-in keys: `"rule"`, `"llm"`, `"vlm"`, `"ensemble.vote"`,
`"ensemble.router"`, `"ensemble.critic"`.

### RuleAnalyst — `analyst="rule"`

Zero LLM calls; runs every detector in `PatternRegistry` once per bar (or a
filtered subset). Output `source` is `"rule:<detector_name>"`,
`probability` and `quality` default to `0.5`.

When to use: as a baseline, in regression suites, and as the deterministic
producer in `ensemble.critic` / `ensemble.router`.

```python
from src.brooks.analyst.base import AnalystRegistry

a = AnalystRegistry.build(
    "rule",
    detector_names=["h2", "l2"],            # default: every registered detector
    params={"h2": {"max_leg_bars": 12}},     # forwarded to PatternRegistry.build
    extractor_kwargs={"swing_k": 2, "atr_period": 5},
    structure_kwargs={"breakout_lookback": 8},
)
```

### LLMAnalyst — `analyst="llm"`

Drives a `Provider` (`src.alpha.llm.provider.Provider`) with a versioned
`PromptBundle` and parses the structured `LLMSignalBatch` response back into
`Signal` records. Provider, prompts, max_tokens, and cache TTL are all
injected — there are no hard-coded vendor SDK calls in the analyst itself.

`PromptBundle.load()` reads:

* `prompts/brooks/system_analyst.md` — analyst system prompt
* `prompts/brooks/concept_manual.md` — Brooks vocabulary (cached as
  `cache="concept_manual"`)
* `prompts/brooks/fewshot/analyst_examples.jsonl` — few-shot transcript
  (cached as `cache="fewshot"`)
* JSON schema derived from `Decision` via Pydantic — **never** hand-written
  in the prompt files

Cache hit rate is reported via `Signal.meta["cache_hit"]` and aggregated by
the eval pipeline.

Supported model ids live in `config/llm/registry.yaml` — currently
`claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5`, `gpt-4.1`,
`gpt-4.1-mini`, `gemini-2.5-pro`, `gemini-2.5-flash`.

```python
from src.alpha.llm.providers.anthropic import AnthropicProvider
from src.brooks.analyst.base import AnalystRegistry

a = AnalystRegistry.build(
    "llm",
    provider=AnthropicProvider(model="claude-opus-4-7"),
    model="claude-opus-4-7",
    cache_ttl_seconds=3600,
    max_tokens=4096,
    temperature=0.0,
    context_budget_tokens=2000,
)
# a.name == "llm:claude-opus-4-7"
```

### VLMAnalyst — `analyst="vlm"`

Renders `ctx.primary` as a deterministic OHLCV PNG via
`src/brooks/render/chart.py`, packs it as an `ImagePart` alongside a compact
text summary, and asks a multimodal `Provider` for the same
`LLMSignalBatch` schema (extended with chart `annotations` for re-overlay).

When VLM beats LLM: chart-shape patterns the text prompt struggles with —
wedges, broad ranges, micro-channels with subtle slope changes,
breakout-mode transitions. The default `RouterAnalyst` config routes
`breakout_mode` straight to VLM for that reason.

```python
from src.alpha.llm.providers.gemini import GeminiProvider
from src.brooks.analyst.base import AnalystRegistry
from src.brooks.render.chart import ChartStyle

a = AnalystRegistry.build(
    "vlm",
    provider=GeminiProvider(model="gemini-2.5-pro"),
    model="gemini-2.5-pro",
    include_htf=True,
    chart_style=ChartStyle(width=1280, height=720, show_volume=True),
    context_budget_tokens=1000,    # smaller; chart carries most of the info
)
# a.name == "vlm:gemini-2.5-pro"
```

### Ensemble Analysts

All three live in `src/brooks/analyst/ensemble.py` and wrap one or more
sub-analysts.

* **`ensemble.vote` (`VoteAnalyst`)** — fan out to N analysts concurrently
  via `asyncio.gather`, cluster signals by `(pattern, side)` tolerating ±1
  bar drift on `signal_bar_idx`, keep groups with at least
  `min_agree_count` distinct voters. Numeric fields are weighted averages
  (weights default to 1.0 per analyst).
* **`ensemble.router` (`RouterAnalyst`)** — pick a single sub-analyst based
  on `ctx.primary.regime.regime`. Each emitted signal is tagged with
  `meta["routed_by"] = <regime.value>`. Default routes live in
  `config/brooks/ensemble_router.yaml`:

  ```yaml
  default: rule
  routes:
    strong_bull_trend: rule
    weak_bull_trend: rule
    strong_bear_trend: rule
    weak_bear_trend: rule
    climax: rule
    tight_trading_range: "llm:claude-opus-4-7"
    broad_trading_range: "ensemble.critic"
    breakout_mode: "vlm:gemini-2.5-pro"
    unknown: rule
  ```

* **`ensemble.critic` (`CriticAnalyst`)** — producer/critic pipeline. The
  producer's signals are stashed on `ctx.candidates`; the critic returns
  one signal per candidate with `meta["confirms"] = candidate_idx` (or
  `-1` to reject) plus an adjusted `probability`. Surviving candidates
  inherit the critic's probability and reasoning under
  `source="ensemble.critic"`.

Construction goes through `AnalystRegistry.build` with the sub-analysts
already built, or — for the leaderboard YAML path — through
`AnalystFactory.build(spec)` which resolves nested specs like
`"llm:claude-opus-4-7"` into proper analyst instances.

## Signal & Decision Schema

`src/brooks/schema.py` defines the contract every layer shares.

```python
class Signal(BaseModel):
    pattern: str
    side: Literal["long", "short"]
    signal_bar_idx: int
    entry_px: float            # gt=0
    stop_px: float             # gt=0; must differ from entry_px
    target_px: float | None
    probability: float         # [0, 1]
    quality: float             # [0, 1]
    reasoning: str = ""
    source: str
    meta: dict = {}            # latency_ms / input_tokens / output_tokens / cache_hit / model / ...

    @property
    def one_r(self) -> float:  # |entry - stop|
```

```python
class Decision(BaseModel):
    symbol: str
    side: Literal["long", "short"]
    entry_px: float
    stop_px: float
    target_px: float
    quantity: float = 0.0
    probability: float         # post-TE, post-HTF-multiplier
    expected_r: float          # E[R] from Trader's Equation
    regime: str
    htf_aligned: bool
    signals: list[Signal]
    source: str
    reasoning: str             # "p=… E=… htf=…"
```

`source` naming convention (used everywhere — eval bucketing,
leaderboard, persisted hit-rate samples):

* `"rule:h2"`, `"rule:l2"`, `"rule:wedge_long"`, …
* `"llm:claude-opus-4-7"`, `"llm:gpt-4.1"`, …
* `"vlm:gemini-2.5-pro"`, …
* `"ensemble.vote"`, `"ensemble.router"`, `"ensemble.critic"`

## Decision Pipeline Deep-Dive

### SignalAggregator (`src/brooks/decision/aggregator.py`)

Combines raw `Signal`s into a single `AggregatedDecision`. Key parameter:
`confluence_n` — minimum number of agreeing-side signals to fire (`1` =
any-of, `≥2` = confluence). The "majority side" wins; ties go long.

Conservative entry/stop rule:

* **long** → `entry = max(s.entry_px)`, `stop = min(s.stop_px)`
* **short** → `entry = min(s.entry_px)`, `stop = max(s.stop_px)`

i.e. always use the latest breakout price and the tightest stop in the
agreeing group.

### Trader's Equation (`src/brooks/decision/trader_equation.py`)

```
E[R] = p * reward_R - (1 - p) * 1.0 - cost_R
```

* `p` ← `HitRateTable.lookup((pattern, regime, htf_aligned, side))` if the
  bucket has ≥ `MIN_SUFFICIENT_SAMPLES` (30) rows, else falls back to the
  analyst-supplied `Signal.probability` prior (default `0.55`).
* `reward_R` ← `target_px` distance in R units, or `default_reward_r`
  (default `2.0`) when `target_px is None`.
* `cost_R` ← fees + slippage in R, default `0.05`.
* HTF multiplier (Phase 3.5):

  | tag        | adjustment                           |
  |------------|---------------------------------------|
  | `aligned`  | `p *= htf_aligned_mult` (default 1.2; capped at `htf_prob_cap=0.95`) |
  | `conflict` | `p *= htf_conflict_mult` (default 0.8) |
  | `neutral`/`None` | pass-through                  |

  Multipliers tunable via `te_params={"htf_aligned_mult": ..., "htf_conflict_mult": ..., "htf_prob_cap": ...}`.

### EV Gate (`src/brooks/decision/ev_gate.py`)

Drops every signal whose `E < min_expected_r` (default `0.1`). Survivors
are promoted to `Decision`s with TE-derived `probability` and `expected_r`
baked in. This **replaces** the legacy `min_rr` hard threshold — set
`min_expected_r=0` to disable.

### HitRateTable (`src/brooks/decision/hit_rate.py`)

Parquet-backed lookup at `data/brooks/hit_rate_table.parquet` with columns
`(pattern, regime, htf_aligned, side, samples, hit_rate_1r, hit_rate_2r, avg_realized_r)`.

Built offline by `scripts/brooks_build_hit_rate.py` from
`session_logs` rows tagged with `event_type IN ('signal', 'fill_close')`.
Run with `--dry-run` to materialise an empty table for tests/CI.

## Risk Layer

| Component | File | Defaults |
|-----------|------|----------|
| `FixedPercentSizer` | `risk/sizer.py` | `risk_pct=0.005` |
| `KellySizer`        | `risk/sizer.py` | `max_risk_pct=0.02`, `fraction=0.5` (half-Kelly) |
| `StopLadder`        | `risk/stop_ladder.py` | `partial_trail_trigger_r=1.5`, `partial_trail_lookback=10`, `use_swing_trail=True` |
| `TimeStop`          | `risk/time_stop.py` | `max_bars_to_1r=10` |
| `PortfolioGuard`    | `risk/portfolio_guard.py` | `max_daily_risk_pct=0.03`, `max_symbol_positions=1` |

`KellySizer` formula: `f* = p − (1 − p) / b` where `b = expected_r`. The
effective risk fraction is `clip(f* * fraction, 0, max_risk_pct)`. Cash
cap (`available_cash / entry_px`) prevents leveraged sizes on spot.

`StopLadder` walks each position through monotonic upgrades (stops never
retreat):

```
initial → break_even (≥ 1R) → partial_trail (≥ trigger R) → swing_trail
```

`PortfolioGuard.can_open` enforces (1) per-symbol max, (2) aggregate daily
risk cap, (3) `correlation_groups` (one open position per named group).

## HTF Integration

`TFSnapshot` carries `features` / `structure` / `regime` alongside `bars`,
so detectors and the EV gate can read HTF state directly without replaying
the streaming extractors. `BrooksContext` exposes three alignment helpers:

* `ctx.htf_alignment_for(side)` → `"aligned"`, `"conflict"`, or `"neutral"`
* `ctx.htf_aligned_for(side)` → `bool` alias for `== "aligned"`
* `ctx.htf_alignment_score(side)` → weighted score in `[-1, 1]` using each
  HTF's `RegimeSnapshot.confidence` as the weight

The strategy passes the tag through `EVGate.filter(..., htf_alignment=tag)`
which drives the Trader's Equation probability multiplier described above.

Multi-timeframe is opt-in: pass `mtf_intervals=["1h", "4h"]` and a
`base_interval` to `BrooksStrategy`. The internal `TimeframeResampler`
aggregates the live LTF feed into closed HTF bars and runs the same
extractor/tracker/regime stack on each.

## Evaluation Pipeline

| Module | Class | Purpose |
|--------|-------|---------|
| `eval/golden.py`     | `GoldenSample`, `GoldenDataset` | Labeled (bars → expected signal) dataset, parquet/jsonl on disk |
| `eval/auto_label.py` | `AutoLabeler`                   | Rule + multi-LLM consensus → silver `GoldenSample`s |
| `eval/runner.py`     | `EvalRunner`, `SampleResult`    | Replay a dataset through one analyst |
| `eval/report.py`     | `EvalReport`, `BucketMetrics`   | Precision/recall/F1, hit_rate_1r/2r, expected_r vs realized_r, Wilson 95% CI, HTML/JSON/DataFrame |
| `eval/leaderboard.py`| `LeaderboardConfig`, `AnalystFactory`, `Leaderboard` | Multi-analyst comparison + Pareto chart |

`GoldenSample` schema (parquet rows are flat; `bars` and `meta` are
JSON-encoded strings for parquet, plain lists/dicts in JSONL):

```
id, symbol, interval, bars[], target_bar_idx,
expected_pattern, expected_side, expected_entry, expected_stop, expected_target,
regime, htf_aligned, source ("human" | "silver"), reasoning, meta
```

End-to-end:

```bash
# 1) seed silver labels from raw bars
python scripts/brooks_label_silver.py \
    --input data/brooks/raw/btcusdt_5m.parquet \
    --output data/brooks/silver/btcusdt_5m.parquet \
    --symbol BTCUSDT --interval 5m \
    --llm openai:gpt-4.1 --llm anthropic:claude-sonnet-4-6 \
    --min-agreement 2

# 2) score one analyst against the dataset → HTML / JSON
python -c "
import asyncio
from src.brooks.analyst.base import AnalystRegistry
from src.brooks.eval.golden import GoldenDataset
from src.brooks.eval.runner import EvalRunner

async def main():
    ds = GoldenDataset.load('data/brooks/silver/btcusdt_5m.parquet')
    runner = EvalRunner(analyst=AnalystRegistry.build('rule'), dataset=ds, max_concurrent=4)
    report = await runner.run()
    report.to_html('data/brooks/reports/rule_btc.html')
    print(report.overall().to_row())

asyncio.run(main())
"
```

## Leaderboard

`config/brooks/leaderboard.yaml` enumerates analysts to score weekly.
Adding a new model is a one-line YAML change; `AnalystFactory` resolves
each entry into a runnable analyst.

```yaml
dataset: data/brooks/golden/v1.parquet

analysts:
  - type: rule
  - type: llm
    model: claude-opus-4-7
  - type: llm
    model: claude-sonnet-4-6
  - type: vlm
    model: gemini-2.5-pro
  - type: ensemble.critic
    label: critic(rule+sonnet)
    producer: rule
    critic: llm:claude-sonnet-4-6

schedule: "0 3 * * MON"
output_dir: data/brooks/leaderboards
history_path: data/brooks/leaderboard.parquet
max_concurrent: 4
```

Run locally (offline, deterministic mock provider — no API keys, used in CI):

```bash
python scripts/brooks_leaderboard.py --config config/brooks/leaderboard.yaml --mock-llm
```

Run for real:

```bash
python scripts/brooks_leaderboard.py --config config/brooks/leaderboard.yaml
# Writes data/brooks/leaderboards/<timestamp>.html and appends one row per
# analyst to data/brooks/leaderboard.parquet (history log → trend charts).
```

Weekly cron is wired into Celery beat as `brooks-leaderboard-weekly` and
runs `src.tasks.brooks_leaderboard_task.run_weekly_leaderboard` every
Monday 03:00 UTC (see `src/tasks/celery_app.py`).

The HTML report includes an **Overall** table sorted by F1, a **Pareto**
scatter (F1 vs cost-per-run, USD), and **By regime / By pattern** bucket
breakdowns with Wilson CIs.

**Adding a new model** — three steps, no code change required for an LLM
on an already-supported provider:

1. Add the model entry to `config/llm/registry.yaml` (provider, api_id,
   pricing).
2. Add `- type: llm\n  model: <id>` to `config/brooks/leaderboard.yaml`.
3. Optionally add a row to `DEFAULT_COST_PER_MTOKEN` in
   `src/brooks/eval/leaderboard.py` so the Pareto chart costs are correct
   (or override per-leaderboard via `cost_per_mtoken:` in YAML).

A new provider family (e.g. Anthropic Bedrock, xAI) needs a `Provider`
implementation under `src/alpha/llm/providers/` plus a branch in
`scripts/brooks_leaderboard.py::_real_provider_factory`.

## Paper Trading

Single Celery task: `src.tasks.brooks_live_task.brooks_live_task`. It
drives a `BrooksStrategy` against a `CcxtRealtimeDataStream` (or any
injected `DataStream`) and a paper-mode broker built by
`src.core.live_broker.create_live_broker(mode="paper", ...)`. **Live mode
is intentionally rejected** — Phase 4.6 is paper-only by design.

Per-bar events surface through the WebSocket event bus and decisions are
persisted to `session_logs`:

* `regime` updates → WS panel
* `signal` / `decision` / `equity_update` / `trade_executed`
* `brooks_decision_outcome` rows → consumed by `brooks_live_close_task`

Daily close (`brooks-live-daily-close` cron, 23:55 local) reads the day's
`brooks_decision_outcome` rows and appends realized-R samples to
`data/brooks/hit_rate_samples.parquet`. The main table is rebuilt offline
by `scripts/brooks_build_hit_rate.py`, closing the feedback loop.

REST + WebSocket surface lives at `/brooks-live` (router in
`src/api/brooks_router.py`):

| Method | Path | Purpose |
|--------|------|---------|
| GET    | `/brooks-live/analysts`             | List registered analyst names |
| POST   | `/brooks-live/start`                | Start a session (analyst, params, symbol, interval, exchange, mtf) |
| POST   | `/brooks-live/{session_id}/stop`    | Cooperative stop signal |
| POST   | `/brooks-live/{session_id}/switch`  | Hot-swap analyst on a running session |
| GET    | `/brooks-live/{session_id}/state`   | Snapshot: last regime/signals/decision/equity/trades |
| GET    | `/brooks-live/sessions`             | List in-process sessions |
| POST   | `/brooks-live/close-day`            | Trigger the daily close manually |
| WS     | `/ws/brooks/{session_id}`           | Push regime/signal/decision/equity/trade events |

UI panel: **Brooks Studio** (`ui/src/features/studio/`, route `/studio`).
The legacy `/brooks-live` route now client-side redirects to `/studio`,
and the `BrooksLive` component / `useBrooksLive` hook have been removed.
See [studio.md](studio.md) for the user guide and
[studio-extension.md](studio-extension.md) for adding layers and panels.

The `/brooks-live/*` REST endpoints below remain available for one
release and are returning the `Deprecation: true` HTTP header — clients
should migrate to `/brooks-studio/*` (see
`src/api/brooks_studio_router.py`).

Cost / frequency control (configured under `BrooksLiveConfig` in
`src/config/settings.py`, override with `QUANT_BROOKS__*` env vars):

| Setting | Default | Purpose |
|---------|---------|---------|
| `default_symbol`               | `"BTC/USDT"` | Initial symbol if `start` request omits one |
| `default_interval`             | `"5m"`       | Initial bar interval |
| `default_exchange`             | `"binance"`  | CCXT exchange id |
| `default_analyst`              | `"rule"`     | Initial analyst name |
| `initial_cash`/`commission`/`slippage` | `100_000` / `0.0003` / `0.001` | Paper broker params |
| `llm_min_interval_seconds`     | `60.0`       | Min wall-clock gap per `(symbol, interval)` between LLM/VLM calls |
| `llm_daily_budget_usd`         | `10.0`       | Soft daily cap (advisory; rate limiter is the hard gate) |
| `hit_rate_samples_path`        | `data/brooks/hit_rate_samples.parquet` | Daily-close output |

Live `_install_llm_rate_limiter` wraps the analyst's `analyze` so calls
that arrive within `llm_min_interval_seconds` of the previous call return
an empty signal list (i.e. "skip this bar") — rule analysts are left
untouched.

## Extension Points

### Add a new Pattern Detector

1. Subclass `PatternDetector` in `src/brooks/patterns/<name>.py`.
2. Register it:

   ```python
   from src.brooks.patterns.base import DetectorContext, PatternDetector, PatternSignal
   from src.brooks.patterns.registry import PatternRegistry

   @PatternRegistry.register("my_pattern")
   class MyPatternDetector(PatternDetector):
       def on_bar(self, ctx: DetectorContext) -> PatternSignal | None: ...
   ```

3. Import the new module from `src/brooks/patterns/__init__.py` so the
   decorator runs at process start.
4. The `RuleAnalyst` picks up every registered detector automatically.
5. Add `tests/brooks/test_patterns_<name>.py` with the synthetic-bars
   pattern-fixture style (see existing pattern tests).

### Add a new Analyst

```python
from src.brooks.analyst.base import AnalystRegistry
from src.brooks.context import BrooksContext
from src.brooks.schema import Signal

@AnalystRegistry.register("my_analyst")
class MyAnalyst:
    name: str  # set by the decorator
    async def analyze(self, ctx: BrooksContext) -> list[Signal]: ...
```

Import it from `src/brooks/analyst/__init__.py`. Select via
`BrooksStrategy(analyst="my_analyst", analyst_params={...})`.

### Add a new LLM Provider

LLM/VLM analysts delegate to a `Provider` from `src.alpha.llm.provider`.

1. Implement the `Provider` protocol (with `cache`, usage, and latency
   reporting) under `src/alpha/llm/providers/`.
2. Register the model in `config/llm/registry.yaml`.
3. Add a branch in `scripts/brooks_leaderboard.py::_real_provider_factory`
   so leaderboard runs can resolve it.
4. Pass an instance via `analyst_params`:

   ```python
   BrooksStrategy(
       analyst="llm",
       analyst_params={"provider": MyProvider(), "model": "my-model-1"},
   )
   ```

### Add a new Ensemble strategy

Same protocol as any other analyst — register under
`"ensemble.<name>"` and accept its sub-analysts as constructor kwargs.
For YAML-driven leaderboard support, also add a branch in
`AnalystFactory.build` (`src/brooks/eval/leaderboard.py`).

### Add a new Regime state

Regime states are an enum (`BrooksRegime` in `src/brooks/regime.py`) but
the routing config and taxonomy treat them as the single source of truth:

1. Add the new value to `BrooksRegime` and update `BrooksRegimeClassifier`
   to set it.
2. Add a section to `docs/brooks/taxonomy.yaml`.
3. Optionally add a route in `config/brooks/ensemble_router.yaml`.

## Config & Assets

| Path | Purpose |
|------|---------|
| `config/brooks/ensemble_router.yaml` | Default `RouterAnalyst` regime → analyst map |
| `config/brooks/leaderboard.yaml`     | Cross-model leaderboard config |
| `config/llm/registry.yaml`           | Frontier-model registry (provider, api_id, pricing, capability flags) |
| `prompts/brooks/system_analyst.md`   | LLM analyst system prompt |
| `prompts/brooks/system_vlm.md`       | VLM analyst system prompt |
| `prompts/brooks/concept_manual.md`   | Brooks vocabulary (cached prefix) |
| `prompts/brooks/fewshot/analyst_examples.jsonl` | LLM few-shot transcript |
| `prompts/brooks/fewshot/vlm_examples.jsonl`     | VLM few-shot transcript |
| `prompts/brooks/sections/`           | Modular section drafts (intro, trader_equation, common_misreads) |
| `docs/brooks/taxonomy.yaml`          | Bar / pattern / regime definitions — single source of truth |

## Testing

```bash
pytest tests/brooks/                      # whole brooks suite
pytest tests/brooks/test_analyst_llm.py   # LLM analyst with MockProvider
pytest tests/brooks/test_leaderboard.py   # leaderboard + AnalystFactory
```

LLM/VLM tests use a `MockProvider` (see
`scripts/brooks_leaderboard.py::_mock_provider_factory` for the same
pattern) so the suite has zero network and zero API-key requirements.

## Further Reading

* [Brooks Studio user guide](studio.md) — `/studio` workspace, layers, keyboard shortcuts, URL params
* [Brooks Studio extension guide](studio-extension.md) — add a layer, side-panel tab, or `BarEvent` field
* [Usage recipes](usage.md) — eight runnable end-to-end recipes
* [Taxonomy](taxonomy.yaml) — bar / pattern / regime reference
* `prompts/brooks/concept_manual.md` — Brooks vocabulary used by analysts
