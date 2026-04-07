# Alpha Search System

The alpha search system discovers quantitative trading factors (alpha formulas) through a combination of LLM-driven evolution, LLM-guided MCTS, neural network generation, and programmatic enumeration.

## Module Structure

```
src/alpha/
  core/           — DSL, compiler, VM, operators, dataset, market profiles
  eval/           — Metrics, screening, validation, GPU acceleration
  search/         — Orchestrator, context, pipeline, evolution, checkpoints
  strategies/     — Pluggable strategy implementations
    mcts/         —   LLM-guided MCTS engine (paper algorithm)
  llm/            — LLM backends (Heuristic, OpenAI) and context
  knowledge/      — Financial themes, feature engineering, strategy memory
  risk/           — Risk models, signal transformation, factor combination
  infra/          — Persistence and tracing
  service.py      — AlphaService (top-level API)
  cli.py          — CLI commands
  auto_runner.py  — Auto-search runner
```

## Multi-Market Architecture

**File**: `src/alpha/core/market.py`

The system supports multiple markets via `MarketProfile`. Each profile bundles:
- `TensorSchema` — available fields for formula type-checking
- `field_aliases` — shorthand alias resolution (e.g., `pe` → `peTTM`)
- `field_descriptions` — LLM prompt field reference
- `seeds` — initial seed formulas for heuristic population
- `mutation_replacements` — formula mutation table for heuristic backend
- `persona` — LLM system prompt persona
- `supported_intervals` — valid data intervals
- `symbol_presets` — predefined stock universe presets (e.g., index constituents)

Available markets: `crypto` (5m–4h futures), `a_share` (daily).

### A-Share Universe Presets

A-share loader supports `universe` parameter — an index code that resolves to constituent stocks via `stock_data.index_stocks`:

| Preset | Index Code | Description |
|--------|-----------|-------------|
| 沪深 300 | `000300` | 大盘蓝筹，定价有效 |
| 中证 500 | `000905` | 中盘股，alpha 空间适中 |
| 中证 1000 | `000852` | 小盘股，因子收益高 |
| 中证全指 | `000985` | 全 A 股可投资范围 |
| 创业板 50 | `399673` | 科技成长板块 |

Usage: pass `universe="000300"` in API requests, or use `symbols` list for custom stock pools.

### Adding a New Market

1. Register a `MarketProfile` in `src/alpha/core/market.py`
2. Implement a `DatasetLoader` in `src/alpha/core/dataset.py`
3. Add loader class to `AlphaService._LOADER_MAP`

## Pipeline Overview

```
MarketProfile (schema, aliases, loader)
  ↓
DatasetLoader.load(symbols, time_range) → AlphaDataset
  ↓
Formula String
  → [FormulaParser] AST
  → [TypeChecker + field_aliases] validated AST
  → [FormulaCompiler] BytecodeProgram (instruction list + expr_hash)
  → [StackVM] + TensorStore → alpha signal matrix (time × symbols)
  → [Evaluation] → metrics (rank_ic, sharpe, turnover, fitness)
  → [SearchOrchestrator] round loop:
      ├── Strategies generate candidates
      ├── Quick screen (IC-only pass/fail)
      ├── Full evaluate (CPCV validation)
      ├── Fitness scoring
      ├── Archive update (MAP-Elites)
      └── RL feedback to strategies
  → [AlphaService] persists results, serves API
```

## DSL (Domain-Specific Language)

**File**: `src/alpha/core/dsl.py`, `src/alpha/core/operators.py`

Formulas are Python expressions operating on market data tensors. Example:

```python
cs_rank(ts_mean(close, 20) - ts_mean(close, 5))
```

### Available Fields (per market)

Fields are defined per market via `MarketProfile` (`src/alpha/core/market.py`). Each market registers a `TensorSchema` with its available fields.

**Crypto futures** (`market="crypto"`):

| Category | Fields |
|----------|--------|
| Price | `open`, `high`, `low`, `close`, `volume`, `turnover`, `vwap`, `bid_ask_spread` |
| Volume detail | `trade_count`, `taker_buy_volume`, `taker_buy_quote_volume` |
| Mark price | `mark_open`, `mark_high`, `mark_low`, `mark_close` |
| Premium index | `premium_open`, `premium_high`, `premium_low`, `premium_close` |
| Market metrics | `funding_rate`, `open_interest`, `open_interest_value`, `long_short_ratio`, `taker_long_short_vol_ratio`, `top_trader_long_short_ratio`, `top_trader_long_short_position_ratio` |

**A-share daily** (`market="a_share"`):

| Category | Fields |
|----------|--------|
| Price (前复权) | `open`, `high`, `low`, `close`, `preclose`, `vwap` |
| Volume / activity | `volume`, `amount`, `turnover` (= amount), `turn` (换手率 %) |
| Fundamental | `peTTM` (市盈率 TTM), `pbMRQ` (市净率 MRQ), `pctChg` (涨跌幅 %) |
| Status | `adjfactor` (复权因子), `isST` (ST 标记) |

Field aliases (e.g., `pe` → `peTTM`, `oi` → `open_interest`) are configured per market in `MarketProfile.field_aliases`.

### Operators (60+)

| Category | Operators |
|----------|-----------|
| Math | `abs`, `log`, `sign`, `sqrt`, `sigmoid`, `neg`, `div`, `power` |
| Time-series | `ts_mean`, `ts_std`, `ts_max`, `ts_min`, `ts_rank`, `ts_zscore`, `ts_ema`, `ts_corr`, `ts_cov`, `decay_linear`, `delay`, `delta`, `returns_n`, `log_return`, `ts_argmax`, `ts_argmin`, `ts_winsorize` |
| Cross-section | `cs_rank`, `cs_zscore`, `cs_demean`, `cs_scale` |
| Domain | `oi_delta`, `funding_delta`, `spread_ratio`, `adv_n`, `amihud`, `hlc3`, `ohlc4`, `true_range`, `atr_n`, `volatility_n` |
| Control | `where`, `clip`, `fillna`, `max`, `min` |

### Type System

- `tensor`: N-dimensional market data (time × symbols)
- `scalar`: Single numeric value (int/float)
- `mask`: Boolean tensor (output of comparisons)

## Compiler

**File**: `src/alpha/core/compiler.py`

```
FormulaCompiler.compile(formula_string) → BytecodeProgram
```

Process: Parse → Type-check → Emit bytecode. Each instruction has opcode, destination register, argument registers.

Outputs normalized formula hash (SHA256) for deduplication.

## StackVM

**File**: `src/alpha/core/vm.py`

Register-based virtual machine executing compiled bytecode on market data.

- `TensorStore`: In-memory field storage with persistent caching
- Instructions: `push_field`, `push_const`, operator opcodes
- Vectorized execution on NumPy arrays (or GPU tensors via Triton)
- LRU cache for repeated formula evaluation

## Evaluation

**File**: `src/alpha/eval/metrics.py`

### Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| `rank_ic` | Pearson corr(alpha_rank, forward_return) | Primary signal quality |
| `ic_ir` | mean(IC) / std(IC) | IC stability |
| `ic_std` | std(IC) | IC variance |
| `ic_decay` | IC_5d - IC_10d | Signal persistence |
| `turnover_proxy` | 1 - \|autocorrelation\| | Trading cost indicator |
| `fitness` | rank_ic² / ic_std | Archive ranking score |

CPCV (Combinatorial Purged Cross-Validation) via `src/alpha/eval/validation.py` with `n_splits`, `purge_window`, `embargo_window` for robust out-of-sample evaluation.

## Search Strategies

All strategies implement the `SearchStrategy` protocol (`src/alpha/search/context.py`):

```python
class SearchStrategy(Protocol):
    name: str
    def should_activate(self, ctx: SearchContext) -> bool: ...
    def generate_candidates(self, ctx: SearchContext) -> list[Individual]: ...
    def on_evaluation_complete(self, ctx: SearchContext, evaluated: list[Individual]): ...
```

### 1. LLM Evolution (`src/alpha/strategies/llm_evolution.py`)

Main search driver. Uses tournament selection + LLM mutation.

**Flow per round**:
1. Round 0: `llm.generate_initial_population()` (genesis)
2. Tournament selection (k=7) → top 2 parents
3. Package parents + metrics into `BreedingSpec`
4. `llm.generate_offspring(spec)` → mutated formulas
5. Compile, deduplicate, return candidates

**RL feedback loop**: After evaluation, `record_evaluation_result()` updates strategy memory (winning themes & operators). Memory fed back into prompts for next round.

### 2. MCTS Refinement (`src/alpha/strategies/mcts_refinement.py`, `src/alpha/strategies/mcts/`)

LLM-guided Monte Carlo Tree Search, implementing "Navigating the Alpha Jungle" (Shi et al., 2025).

Activates every N rounds. Refines archive elites via tree search with:

- **UCT Selection with Virtual Expansion**: Any node (not just leaves) can be expanded via virtual expansion action `a_e` with `N_s' = 1 + |C(s)|`
- **Dimension-Targeted Refinement**: Softmax sampling `P(i*=i|s) = Softmax((e_max - E_s)/T)` selects weakest dimension (Effectiveness, Stability, Turnover, Diversity, Overfitting) for improvement
- **Multi-Dimensional Evaluation**: Percentile ranking against zoo `e_i(f) = (1 - R(f, m_i, F_zoo)) * e_max`, with LLM overfitting risk assessment
- **Backpropagation**: `Q(s_k, a_k) ← max(Q(s_k, a_k), S(f_new))` (max reward, not average)
- **Frequent Subtree Avoidance (FSA)**: Mines frequent root genes from zoo formulas, instructs LLM to avoid common motifs for structural diversity
- **Dynamic Search Budget**: Budget increases when new high scores are found

Zoo criteria: RankIC ≥ 0.015, RankIR ≥ 0.3, correlation < 0.8.

### 3. Neural Formula (`src/alpha/strategies/neural_formula.py`)

Transformer-based RPN (Reverse Polish Notation) generator, AlphaGPT-style.

- **Vocabulary**: 60+ operators, fields, window sizes → tokens
- **Model**: Causal Transformer (64D, 4 heads, 2 layers, RMSNorm)
- **Generation**: Autoregressive sampling with vectorized action masking (GPU-friendly)
- **Training**: REINFORCE with normalized advantage
- **Rewards**: Invalid RPN → -5.0, constant signal → -2.0, valid → |rank_ic| × 20.0

### 4. Enumeration (`src/alpha/strategies/enumeration.py`)

Round 0 only. Programmatic formula generation via `FormulaEnumerator` + fast IC screening.

## SearchOrchestrator

**File**: `src/alpha/search/orchestrator.py`

Coordinates multiple strategies in a round-based loop:

```
for each round:
  for each active strategy:
    candidates = strategy.generate_candidates(ctx)
    quick_screen(candidates)  # IC-only filter
    evaluate(candidates)      # Full CPCV
    score(candidates)         # Fitness
    update_archive(candidates) # MAP-Elites
    strategy.on_evaluation_complete(ctx, candidates)
```

**MAP-Elites archive**: 2D bins by IC × turnover for diversity preservation.

**Pipeline overlap**: Prefetches next round's LLM candidates while current round evaluates.

**Checkpointing**: Periodic snapshots of strategy state + factor catalog via `src/alpha/search/checkpoint.py`.

## LLM Integration

**File**: `src/alpha/llm/backends.py`, `src/alpha/llm/context.py`

`OpenAILLMBackend` connects to any OpenAI-compatible API. `HeuristicLLMBackend` provides deterministic fallback.

**Prompt construction** (market-aware via `MarketProfile`):
1. Persona (crypto quant / A股量化 researcher) — from `profile.persona`
2. Field reference — from `profile.field_descriptions` or knowledge base
3. Operator catalog with signatures
4. Financial themes (market-specific)
5. Strategy memory feedback (RL: what worked in past rounds)
6. Feature kitchen (derived building blocks)
7. Rules (AST depth ≤ 5, valid Python, safe division)

**LLMContext**: Builds pipeline summaries for LLM analysis of search health and bottleneck detection.

## Adding a New Strategy

1. Create `src/alpha/strategies/my_strategy.py`
2. Implement the `SearchStrategy` protocol (4 methods: `name`, `should_activate`, `generate_candidates`, `on_evaluation_complete`)
3. Optionally implement `StatefulStrategy` for checkpoint/restore
4. Register in `src/alpha/strategies/__init__.py`
5. Add to `AlphaService._build_strategies()` in `src/alpha/service.py`

## Service Layer

**File**: `src/alpha/service.py`

`AlphaService(market="crypto")` provides the public API. The `market` parameter selects the `MarketProfile`, which configures the schema, compiler, dataset loader, and LLM persona.

| Method | Purpose |
|--------|---------|
| `list_operators()` | All 60+ available operators |
| `validate_formula(formula)` | Parse + type-check (market-aware aliases) |
| `compile_formula(formula)` | Return bytecode + validation |
| `evaluate_formula(formula, fields)` | Run on raw data, return weights + metrics |
| `evaluate_formula_from_db(formula, symbols, dates)` | Load from ClickHouse, evaluate |
| `search_formulas_on_db(symbols, dates, ...)` | Full search loop with CPCV |
| `combine_factors_from_db(symbols, dates, method)` | IC-weighted combination of zoo factors |
| `benchmark_vm(formulas)` | Throughput benchmarking |

**Dataset Loaders** (`src/alpha/core/dataset.py`):
- `CryptoMinuteDatasetLoader` — loads from `crypto_data.futures_5m`, supports 5m–4h intervals
- `AShareDailyDatasetLoader` — loads from `stock_data.stock_daily`, daily only, forward-adjusted prices. Accepts `universe` (index code) and `exclude_st` kwargs.

Both satisfy the `DatasetLoader` protocol: `load(symbols, start_time, end_time, interval, **kwargs) → AlphaDataset`.

## GPU Acceleration

**Files**: `src/alpha/eval/gpu_metrics.py`, `src/alpha/eval/gpu_ops.py`, `src/alpha/eval/triton_kernels.py`

Optional GPU acceleration via PyTorch and Triton custom kernels for:
- Vectorized operator execution
- Batch formula evaluation
- Parallel IC computation
