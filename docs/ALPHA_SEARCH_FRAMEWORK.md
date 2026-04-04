# Alpha Search Framework — 可插拔策略架构

> `src/alpha/` | 核心: `search_strategy.py`, `strategies/`, `financial_knowledge.py`, `feature_kitchen.py`, `strategy_memory.py`

## 架构总览

```
SearchOrchestrator
 |
 |-- SearchContext (共享状态: population, archive, evaluate_fn, ...)
 |
 |-- Strategy 1: LLMEvolutionStrategy   (每轮)
 |     |-- OpenAILLMBackend
 |     |     |-- FinancialKnowledgeBase  (16个金融主题)
 |     |     |-- FeatureKitchen          (39个衍生特征)
 |     |     |-- StrategyMemory          (UCB bandit + RL反馈)
 |     |
 |-- Strategy 2: MCTSRefinementStrategy  (每N轮, 可选)
 |     |-- MCTSEngine + MCTSLLMAdapter
 |
 |-- [未来] Strategy 3, 4, ...           (实现 SearchStrategy 协议即可接入)
 |
 |-- 共享: FormulaCompiler -> StackVM    (DSL编译 + 执行, 不变)
 |-- 共享: FitnessEngine                 (MAP-Elites archive)
```

每轮循环: 遍历所有策略 → `should_activate()` → `generate_candidates()` → quick screen → full evaluate → `on_evaluation_complete()` → archive update

---

## 1. 快速开始

### CLI

```bash
# 搜索
python -m src.alpha.cli search \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --start 2026-03-01T00:00:00 --end 2026-04-01T00:00:00 \
  --generations 10 --offspring-count 8 --top-k 5

# 评估单个公式
python -m src.alpha.cli evaluate \
  --formula "cs_rank(ts_zscore(funding_rate, 20))" \
  --symbols BTCUSDT,ETHUSDT --start 2026-03-27T00:00:00 --end 2026-03-28T00:00:00

# 组合 zoo 因子
python -m src.alpha.cli combine \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --start 2026-03-01T00:00:00 --end 2026-04-01T00:00:00

# 自动滚动搜索
python -m src.alpha.cli auto --config config/alpha_lab/auto.yaml --once

# 查看结果
python -m src.alpha.cli list-zoo --limit 10
python -m src.alpha.cli list-runs
```

### Python API

```python
from src.alpha import AlphaService

# 默认配置: LLM进化 + 金融语义 + RL反馈 (自动启用)
service = AlphaService()

# 启用 MCTS 精炼策略
service = AlphaService(enable_mcts=True, mcts_refinement_frequency=3)

# 运行搜索
result = service.search_formulas_on_db(
    provider="bitget",
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    start_time=start, end_time=end,
    generations=10, offspring_count=8, top_k=5,
)

for r in result["top_results"]:
    print(f'{r["fitness"]:.3f}  {r["formula"]}')
```

### 直接使用 SearchOrchestrator

```python
from src.alpha import (
    SearchOrchestrator, LLMEvolutionStrategy, MCTSRefinementStrategy,
    FinancialKnowledgeBase, FeatureKitchen, StrategyMemory,
)
from src.alpha.llm import build_default_llm_backend

kb = FinancialKnowledgeBase()
fk = FeatureKitchen(schema)
sm = StrategyMemory(
    persistence_path="data/alpha_lab/strategy_memory.json",
    all_theme_ids=kb.get_all_theme_ids(),
)
sm.load()

llm = build_default_llm_backend(
    registry=registry, schema=schema,
    strategy_memory=sm, knowledge_base=kb, feature_kitchen=fk,
)

orchestrator = SearchOrchestrator(
    strategies=[
        LLMEvolutionStrategy(llm_backend=llm),
        MCTSRefinementStrategy(mcts_engine=mcts, activation_frequency=3),
    ],
    compiler=compiler, registry=registry, schema=schema,
    strategy_memory=sm, knowledge_base=kb, feature_kitchen=fk,
)

result = orchestrator.run(
    seeds=["cs_rank(ts_zscore(funding_rate, 20))"],
    rounds=10, batch_size=8, top_k=5,
    novelty_threshold=0.995,
    evaluate_fn=my_eval_fn,
    dataset=dataset,
)
```

---

## 2. SearchStrategy 协议

所有搜索策略实现同一个接口:

```python
class SearchStrategy(Protocol):
    @property
    def name(self) -> str: ...

    def should_activate(self, ctx: SearchContext) -> bool: ...
    def generate_candidates(self, ctx: SearchContext) -> list[Individual]: ...
    def on_evaluation_complete(self, ctx: SearchContext, evaluated: list[Individual]) -> None: ...
```

| 方法 | 职责 |
|------|------|
| `name` | 策略名称, 用于日志和 tracing span |
| `should_activate` | 决定本轮是否运行 (例: MCTS 每3轮, ES 每轮) |
| `generate_candidates` | 生成已编译未评估的 `Individual` 列表 |
| `on_evaluation_complete` | 评估后回调, 用于策略内部学习 |

`SearchContext` 是���享状态容器 — 策略通过它访问 population, archive, compiler, evaluate_fn, strategy_memory 等.

### 实现一个新策略

```python
# src/alpha/strategies/my_strategy.py
from src.alpha.search_strategy import SearchContext, build_individual
from src.alpha.evolution import Individual

class MyCustomStrategy:
    name = "my_strategy"

    def should_activate(self, ctx: SearchContext) -> bool:
        return ctx.round_idx % 2 == 0  # 每隔一轮

    def generate_candidates(self, ctx: SearchContext) -> list[Individual]:
        formulas = self._my_generation_logic(ctx)
        candidates = []
        for f in formulas:
            ind = build_individual(ctx.compiler, ctx.schema, f, {"origin": "my_strategy"})
            if ind and ind.expr_hash not in ctx.seen_hashes:
                candidates.append(ind)
        return candidates

    def on_evaluation_complete(self, ctx, evaluated):
        # 可选: 记录结果用于内部学习
        pass
```

注册:
```python
orchestrator = SearchOrchestrator(
    strategies=[
        LLMEvolutionStrategy(llm_backend=llm),
        MyCustomStrategy(),
    ],
    ...
)
```

---

## 3. 内置策略

### LLMEvolutionStrategy

主力搜索策略. 每轮:
1. Tournament selection 从 population 选 2 个 parents
2. 构建 `BreedingSpec` (含 parent metrics + diagnostics)
3. 调用 `llm_backend.generate_offspring()` → 公式列表
4. 编译为 `Individual` 返回

**RL 反馈环**: `on_evaluation_complete` 调用 `llm_backend.record_evaluation_result()`, 将 fitness/theme/operators 记录到 `StrategyMemory`, 下一轮的 LLM prompt 自动包含反馈摘要.

```python
LLMEvolutionStrategy(
    llm_backend=llm,
    tournament_size=7,   # tournament 采样大小
    batch_size=8,        # 每轮生成公式数 (None = 使用 ctx.batch_size)
)
```

### MCTSRefinementStrategy

局部精炼策略. 周期性地对 archive 中的精英个体做 tree search 深挖:

```python
MCTSRefinementStrategy(
    mcts_engine=mcts_engine,
    activation_frequency=3,      # 每3轮触发一次
    top_k_to_refine=2,           # 精炼 archive 前2名
    iterations_per_refine=3,     # 每个个体3次 MCTS 迭代
)
```

`should_activate` 条件: `round_idx > 0`, `round_idx % frequency == 0`, `archive 非空`, `dataset 存在`

---

## 4. FinancialKnowledgeBase — 金融语义

16个结构化金融主题, 5个特征语义分组. 注入 LLM prompt 替代硬编码主题行.

```python
from src.alpha import FinancialKnowledgeBase

kb = FinancialKnowledgeBase()

# 查看所有主题
print(kb.get_all_theme_ids())
# ['funding_basis_arb', 'oi_momentum_divergence', 'taker_flow_imbalance', ...]

# 获取主题详情
theme = kb.get_theme("funding_basis_arb")
print(theme.hypothesis)
print(theme.relevant_fields)
print(theme.example_formulas)

# 生成 prompt 片段
prompt = kb.build_theme_prompt(["funding_basis_arb", "taker_flow_imbalance"])
```

**主题列表**:

| theme_id | 类别 | 描述 |
|----------|------|------|
| `funding_basis_arb` | derivatives | 资金费率-基差背离套利 |
| `oi_momentum_divergence` | derivatives | OI与价格动量背��� |
| `premium_dynamics` | derivatives | 基差期限结构与均值回复 |
| `mark_spot_divergence` | derivatives | 标记价-现货偏离 |
| `cross_metric_divergence` | derivatives | 跨衍生品指标背离 |
| `taker_flow_imbalance` | flow | 主动买卖力量失衡 |
| `volume_profile_anomaly` | flow | 成交量形态异常/大户行为 |
| `microstructure_toxicity` | microstructure | 订单流毒性/流动性异常 |
| `liquidity_provision` | microstructure | 流动性供给异常 |
| `sentiment_extreme_reversal` | sentiment | 情绪极端反转 |
| `whale_positioning` | sentiment | 大户持仓变化先行信号 |
| `volatility_regime_switch` | volatility | 波动率压缩-扩张regime切换 |
| `intraday_range` | volatility | ���内范围动态 |
| `momentum_decay` | momentum | 动量衰减与反转 |
| `vwap_reversion` | mean_reversion | VWAP均值回复 |
| `mean_reversion_squeeze` | mean_reversion | 急涨急跌后均值回归 |

每个主题包含: `hypothesis` (金融逻辑), `relevant_fields`, `suggested_operators`, `example_formulas`, `anti_patterns`, `window_guidance`

---

## 5. FeatureKitchen — 衍生特征

39个衍生特征以 DSL 公式形式存在, 注入 LLM prompt 作为可用积木. **零计算成本** — VM 仅在公式被使用时才执行.

```python
from src.alpha import FeatureKitchen, TensorSchema

fk = FeatureKitchen(TensorSchema.default_market_schema())
catalog = fk.build_catalog()

print(f"共 {len(catalog)} 个衍生特征")
for f in catalog[:5]:
    print(f"  {f.name}: {f.formula}")
    print(f"    {f.financial_meaning}")
```

**特征分类**:

| 类别 | 数量 | 示例 |
|------|------|------|
| ratio | 14 | `buy_pressure = div(taker_buy_volume, volume + 1e-12)` |
| delta | 15 | `open_interest_delta_5 = delta(open_interest, 5)` |
| interaction | 10 | `funding_basis_spread = ts_zscore(funding_rate, 20) - ts_zscore(premium_close, 20)` |

**特征重要性追踪**:
```python
# 评估后记录
fk.track_feature_importance(formula="cs_rank(div(taker_buy_volume, volume + 1e-12))", fitness=0.5)

# 查看排名
print(fk.get_importance_ranking())
# [('buy_pressure', 0.5, 1), ...]

# 获取未充分探索的特征
print(fk.get_underexplored_features(k=5))
```

---

## 6. StrategyMemory — RL 反馈环

核心思路: 不修改 LLM 权重, 通过统计追踪 + UCB bandit + 动态 prompt 实现 "无梯度RL".

```python
from src.alpha import StrategyMemory

sm = StrategyMemory(
    max_records=2000,
    persistence_path="data/alpha_lab/strategy_memory.json",
    all_theme_ids=["funding_basis_arb", "momentum_decay", ...],
)

# 加载历史记忆
sm.load()

# 记录评估结果 (每次评估后调用)
sm.record(
    formula="cs_rank(ts_zscore(funding_rate, 20))",
    theme_id="funding_basis_arb",
    metrics={"fitness": 0.5, "rank_ic": 0.03, "sharpe": 1.2, "avg_turnover": 0.1},
    is_novel=True,
    all_fields=schema.fields,
)

# UCB1 bandit 选择下一个探索主题
next_theme = sm.select_theme_ucb(c=1.0)  # c 越大越偏探索
themes = sm.select_themes_ucb(n=4, c=1.0)

# 构建反馈摘要 (注入 LLM prompt 的关键)
summary = sm.build_feedback_summary()
print(summary)

# 持久化 (跨session学习)
sm.save()
```

**反馈摘要示例** (搜索 30 个公式后):

```
## Learning from Past Generations (30 formulas evaluated)

### High-performing themes (exploit these)
1. **funding_basis_arb**: 8 formulas, mean fitness 0.38, success rate 62%
   Best: `cs_rank(ts_zscore(funding_rate, 20) - delta(premium_close, 5))`

### Low-performing themes (avoid or modify approach)
- **momentum_decay**: 6 formulas, mean fitness -4.20 — needs different operator/window choices

### Operator patterns that work
- `cs_rank+delta+ts_zscore`: mean fitness 0.41 (5 uses)

### Operator patterns to avoid
- `neg+ts_std+ts_std`: mean fitness -5.00 (2 uses)

### Underexplored features (try these for diversity)
- taker_long_short_vol_ratio, top_trader_long_short_position_ratio, mark_close
```

这段文本每轮动态生成, 注入 LLM 的 genesis/evolution prompt, 引导生成方向.

---

## 7. 数据流

一次完整搜索的数据流:

```
AlphaService.search_formulas_on_db()
  |
  |-- 1. 加载数据集 (ClickHouse → AlphaDataset)
  |-- 2. 构建 CPCV 验证计划 (train/valid/test folds)
  |-- 3. SearchOrchestrator.run()
  |     |
  |     |-- Init: seeds → compile → evaluate → populate archive
  |     |
  |     |-- Round 0..N:
  |     |     |
  |     |     |-- LLMEvolutionStrategy:
  |     |     |     |-- StrategyMemory.select_themes_ucb() → 选主题
  |     |     |     |-- FinancialKnowledgeBase.build_theme_prompt() → 金融语义
  |     |     |     |-- FeatureKitchen.get_catalog_as_prompt_section() ��� 衍生特征
  |     |     |     |-- StrategyMemory.build_feedback_summary() �� RL反馈
  |     |     |     |-- LLM API call → 公式列表 (含 theme 标注)
  |     |     |     |-- 编译 → Individual 列表
  |     |     |
  |     |     |-- [每3轮] MCTSRefinementStrategy:
  |     |     |     |-- 取 archive top-k → MCTSEngine.run() → 精炼公式
  |     |     |
  |     |     |-- Quick screen (1-fold, 快速淘汰 ~40-60%)
  |     |     |-- Full evaluate (CPCV)
  |     |     |-- FitnessEngine.score() → MAP-Elites archive update
  |     |     |-- on_evaluation_complete → StrategyMemory.record()
  |     |
  |     |-- Novelty filter → top_k 结果
  |
  |-- 4. StrategyMemory.save() (跨session持久化)
  |-- 5. Persist results (runs/ + zoo/)
```

---

## 8. 论文策略扩展路线

每个论文策略只需实现 `SearchStrategy` 的 4 个方法即可接入框架:

| 论文 | 文件 | generate_candidates | on_evaluation_complete |
|------|------|--------------------|-----------------------|
| [AlphaPROBE](docs/pdf/2602.11917v1.pdf) | `strategies/dag_evolution.py` | Bayesian 检索 parent → LLM 生成 (考虑全祖先路径) | 更新 DAG 拓扑 |
| [AlphaCFG](docs/pdf/Alpha%20Discovery%20via%20Grammar-Guided%20Learning%20and%20Search.pdf) | `strategies/grammar_guided.py` | CFG 约束 → Tree-LSTM → PUCT 选择 | RL 更新 policy/value |
| [Synergistic RL](docs/pdf/2306.12964.pdf) | `strategies/synergy_rl.py` | PPO agent 生成 RPN token (action masking) | 组合 IC 作为 reward |
| [AlphaForge](docs/pdf/2406.18394.pdf) | `strategies/neural_generative.py` | Generator 网络 + Predictor ��选 | 更新 G/P 网络 + diversity loss |
| [AlphaQCM](docs/pdf/1805_AlphaQCM_Alpha_Discovery_.pdf) | `strategies/distributional_rl.py` | IQN quantile regression + QCM 方差引导 | 更新 quantile 网络 |

所有策略共享:
- `SearchContext` 中的 compiler/vm/schema (编译执行)
- `StrategyMemory` (跨策略反馈)
- `FinancialKnowledgeBase` (金融语义)
- `FeatureKitchen` (衍生特征)
- `FitnessEngine` + MAP-Elites archive

---

## 9. 环境变量

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `ALPHA_LAB_LLM_API_KEY` | LLM API key | (必填, 或传入参数) |
| `ALPHA_LAB_LLM_BASE_URL` | OpenAI 兼容 API 地址 | OpenAI 官方 |
| `ALPHA_LAB_LLM_MODEL` | 模型名 | `gpt-4.1-mini` |

---

## 10. 文件索引

```
src/alpha/
├── search_strategy.py        # SearchStrategy Protocol + SearchContext + SearchOrchestrator
├── strategies/
│   ├─�� __init__.py
│   ├── llm_evolution.py      # LLMEvolutionStrategy (主策略)
│   └── mcts_refinement.py    # MCTSRefinementStrategy (精���策略)
├── financial_knowledge.py    # 16个金融主题 + 5个特征分组
├── feature_kitchen.py        # 39个衍生特征 DSL 公式
├── strategy_memory.py        # UCB bandit + RL反馈环 + 持久化
├── llm.py                    # OpenAILLMBackend (语义prompt + theme解析 + 反馈记录)
├── evolution.py              # Individual, BreedingSpec, FitnessEngine, HeuristicLLMBackend
├── mcts.py                   # MCTSEngine + MCTSLLMAdapter
├── service.py                # AlphaService (顶层编排)
├── compiler.py               # FormulaCompiler (DSL → 字节码)
├── vm.py                     # StackVM (numpy/torch 执行)
├── dsl.py                    # FormulaParser + TypeChecker
├── operators.py              # 95+ operator 注册表
├── evaluation.py             # IC 指标计算
├── validation.py             # CPCV 交叉验证
��── combination.py            # 多因子组合
├── risk.py                   # 风控/执行模拟
├── persistence.py            # 结果���久化
├── tracing.py                # Langfuse/loguru 可观测
├── auto_runner.py            # 滚动搜索循环
├── cli.py                    # 命令行接口
└── dataset.py                # 数据加载 (ClickHouse)
```
