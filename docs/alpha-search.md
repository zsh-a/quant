# Alpha Search System

The alpha search system discovers quantitative trading factors (alpha formulas) through a combination of LLM-driven evolution, neural network generation, and Monte Carlo Tree Search.

## Pipeline Overview

```
Formula String
  → [FormulaParser] AST
  → [TypeChecker] validated AST
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

**File**: `src/alpha/dsl.py`, `src/alpha/operators.py`

Formulas are Python expressions operating on market data tensors. Example:

```python
cs_rank(ts_mean(close, 20) - ts_mean(close, 5))
```

### Available Fields

OHLCV: `open`, `high`, `low`, `close`, `volume`
Crypto: `mark_price`, `premium`, `open_interest`, `sentiment_*`

### Operators (50+)

| Category | Operators |
|----------|-----------|
| Unary | `abs`, `log`, `sign`, `sqrt`, `sigmoid`, `neg`, `power` |
| Binary | `add`, `sub`, `mul`, `div`, `max`, `min`, `pow` |
| Comparison | `gt`, `ge`, `lt`, `le`, `eq`, `ne` → mask |
| Logical | `and`, `or`, `not` (mask only) |
| Time-series | `ts_mean`, `ts_std`, `ts_max`, `ts_min`, `ts_rank`, `ts_zscore`, `ts_ema`, `ts_corr`, `ts_cov`, `decay_linear`, `delay`, `delta`, `returns_n`, `log_return`, `ts_argmax`, `ts_argmin`, `ts_winsorize` |
| Cross-section | `cs_rank`, `cs_zscore`, `cs_demean`, `cs_scale` |
| Domain | `oi_delta`, `funding_delta`, `spread_ratio`, `adv_n`, `amihud`, `hlc3`, `ohlc4`, `true_range`, `atr_n`, `volatility_n` |
| Control | `where`, `clip`, `fillna` |

### Type System

- `tensor`: N-dimensional market data (time × symbols)
- `scalar`: Single numeric value (int/float)
- `mask`: Boolean tensor (output of comparisons)

## Compiler

**File**: `src/alpha/compiler.py`

```
FormulaCompiler.compile(formula_string) → BytecodeProgram
```

Process: Parse → Type-check → Emit bytecode. Each instruction has opcode, destination register, argument registers.

Outputs normalized formula hash (SHA256) for deduplication.

## StackVM

**File**: `src/alpha/vm.py`

Register-based virtual machine executing compiled bytecode on market data.

- `TensorStore`: In-memory field storage with persistent caching
- Instructions: `push_field`, `push_const`, operator opcodes
- Vectorized execution on NumPy arrays (or GPU tensors via Triton)
- LRU cache for repeated formula evaluation

## Evaluation

**File**: `src/alpha/evaluation.py`

### Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| `rank_ic` | Pearson corr(alpha_rank, forward_return) | Primary signal quality |
| `ic_ir` | mean(IC) / std(IC) | IC stability |
| `ic_std` | std(IC) | IC variance |
| `ic_decay` | IC_5d - IC_10d | Signal persistence |
| `turnover_proxy` | 1 - \|autocorrelation\| | Trading cost indicator |
| `fitness` | rank_ic² / ic_std | Archive ranking score |

CPCV (Combinatorial Purged Cross-Validation) with `n_splits`, `purge_window`, `embargo_window` for robust out-of-sample evaluation.

## Search Strategies

All strategies implement the `SearchStrategy` protocol:

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

### 2. MCTS Refinement (`src/alpha/strategies/mcts_refinement.py`, `src/alpha/mcts.py`)

Activates every N rounds. Refines archive elites via UCT tree search.

**MCTS loop**: Select (UCT) → Expand (LLM suggests refinement) → Evaluate (VM + rank_ic) → Backpropagate

Zoo management: deduplicates via 0.95 correlation threshold. Adds high-IC candidates (train IC > 5%, val IC > 2%) to zoo.

### 3. Neural Formula (`src/alpha/strategies/neural_formula.py`)

Transformer-based RPN (Reverse Polish Notation) generator, AlphaGPT-style.

- **Vocabulary**: 50+ operators, 12 fields, 6 window sizes → tokens
- **Model**: Causal Transformer (64D, 4 heads, 2 layers, RMSNorm)
- **Generation**: Autoregressive sampling with vectorized action masking (GPU-friendly)
- **Training**: REINFORCE with normalized advantage
- **Rewards**: Invalid RPN → -5.0, constant signal → -2.0, valid → |rank_ic| × 20.0

### 4. Enumeration (`src/alpha/search_strategy.py`)

Round 0 only. Programmatic formula enumeration with IC-only fast screen.

## SearchOrchestrator

**File**: `src/alpha/search_strategy.py`

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

## LLM Integration

**File**: `src/alpha/llm.py`, `src/alpha/llm_context.py`

`OpenAILLMBackend` connects to any OpenAI-compatible API.

**Prompt construction**:
1. Field reference (OHLCV, crypto-specific)
2. Operator catalog with signatures
3. Financial themes (volatility compression, sentiment extremes, whale behavior)
4. Strategy memory feedback (RL: what worked in past rounds)
5. Feature kitchen (derived building blocks)
6. Rules (AST depth ≤ 5, valid Python, safe division)

**LLMContext**: Builds pipeline summaries for LLM analysis of search health and bottleneck detection.

## Service Layer

**File**: `src/alpha/service.py`

`AlphaService` provides the public API:

| Method | Purpose |
|--------|---------|
| `list_operators()` | All 50+ available operators |
| `validate_formula(formula)` | Parse + type-check |
| `compile_formula(formula)` | Return bytecode + validation |
| `evaluate_formula(formula, fields)` | Run on raw data, return weights + metrics |
| `evaluate_formula_from_db(formula, symbols, dates)` | Load from ClickHouse, evaluate |
| `search_formulas_on_db(symbols, dates, ...)` | Full search loop with CPCV |
| `combine_factors_from_db(symbols, dates, method)` | IC-weighted combination of zoo factors |
| `benchmark_vm(formulas)` | Throughput benchmarking |

## GPU Acceleration

**Files**: `src/alpha/gpu_evaluation.py`, `src/alpha/gpu_ops.py`, `src/alpha/triton_kernels.py`

Optional GPU acceleration via PyTorch and Triton custom kernels for:
- Vectorized operator execution
- Batch formula evaluation
- Parallel IC computation
