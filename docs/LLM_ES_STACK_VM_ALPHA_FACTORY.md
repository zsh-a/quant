# 基于 LLM + ES + Stack VM 的自动化 Alpha 挖掘系统工程实现计划

## 1. 目标与边界

本文将仓库现有的 `FastAPI + Celery + ClickHouse + alpha_mining 原型` 收敛为一套可持续演进的自动化 Alpha 炼丹系统设计，并给出与当前工程的落位关系。

首版默认边界：

- 市场范围：`Bitget U 本位永续`
- 交付边界：`研究到仿真闭环`
- LLM 形态：`远程 API 优先`，同时保留本地推理后端接口
- 执行模型：`张量化批量回测`，不再继续沿用 `pandas + eval` 作为主执行内核

与当前仓库的关系：

- `src/alpha_mining` 仍保留为研究原型，不再作为主演进方向
- 新增 `src/alpha_lab` 承载 DSL、编译器、VM、进化与风险包装
- `src/core/risk_manager.py` 保留为交易/服务侧风控基础对象
- `trader/bitget_client.py` 作为兼容入口，内部转发到新的 Bitget 数据适配器

## 2. 技术栈选型与架构图

### 2.1 模块技术栈

- DSL/语法层：标准库 `ast` 解析 + 严格白名单校验
- 张量执行层：首版接口按 `GPU-first` 设计，当前仓库先落 `NumPy` 参考执行器，后续替换为 `PyTorch/CUDA/Triton`
- 数据层：`ClickHouse` 作为市场数据事实源
- 调度层：`Celery + Redis`
- 服务层：`FastAPI`
- Bitget 数据源：REST 历史拉取 + WebSocket 实时扩展位

### 2.2 数据流

```text
Bitget REST/WebSocket
    -> ClickHouse Raw Store
    -> TensorBuilder
    -> DSL Parser / Type Checker
    -> Bytecode Compiler
    -> Stack VM
    -> Fitness Engine
    -> Evolution Engine + LLM
    -> Alpha Zoo Registry
    -> Risk Wrapper / Paper Trading
```

### 2.3 工程落地原则

- 禁止使用 Python 原生循环逐元素解释公式
- 公式必须经过 `parse -> validate -> type-check -> compile -> execute`
- 所有评估默认包含手续费、滑点、换手惩罚与 funding
- `train / valid / test / final holdout` 严格隔离，`test` 不允许回流搜索

## 3. 核心模块设计

### 3.1 金融 DSL 与算子库

DSL 目标不是生成“任意 Python”，而是生成受限表达式语言。首版字段集：

- `open`
- `high`
- `low`
- `close`
- `volume`
- `turnover`
- `funding_rate`
- `open_interest`
- `bid_ask_spread`

首版算子集：

- 一元：`abs log sign sqrt sigmoid neg`
- 二元：`add sub mul div max min pow`
- 时间序列：`delay delta ts_mean ts_std ts_sum ts_max ts_min ts_rank`
- 截面：`cs_rank cs_scale`
- 条件：`where gt ge lt le and or`
- 领域别名：`volatility_n`

关键接口：

```python
class DSLRegistry:
    def validate_formula(self, formula: str) -> ValidationReport: ...

class FormulaParser:
    def parse(self, formula: str) -> ASTNode: ...

class TypeChecker:
    def infer(self, ast: ASTNode, schema: TensorSchema) -> TypedAST: ...
```

### 3.2 Stack VM

表达式编译路径：

1. 文本公式解析为 AST
2. AST 校验和类型推断
3. AST 编译为 RPN/Bytecode
4. Stack VM 在张量存储上执行

关键接口：

```python
class FormulaCompiler:
    def compile(self, formula: str, schema: TensorSchema) -> BytecodeProgram: ...

class StackVM:
    def run(self, program: BytecodeProgram, store: TensorStore) -> TensorLike: ...
    def run_batch(self, programs: list[BytecodeProgram], store: TensorStore) -> list[TensorLike]: ...
```

示例公式：

```text
CSRank((Ts_Max(high, 10) - close) / volatility_n(close, 20))
```

编译后的核心逻辑：

```text
PUSH_FIELD high
ROLLING_MAX 10
PUSH_FIELD close
SUB
PUSH_FIELD close
VOLATILITY_N 20
DIV
CS_RANK
```

### 3.3 ES / LLM Evolution

首版采用 `μ + λ` 进化循环替代当前 `MCTS refine` 原型。个体包含：

- `formula`
- `program`
- `expr_hash`
- `lineage`
- `metrics`

关键接口：

```python
class FitnessEngine:
    def score(self, metrics: dict[str, float]) -> float: ...

class EvolutionEngine:
    def initialize(self, seeds: list[str], population_size: int) -> list[Individual]: ...
    def select_survivors(self, pop: list[Individual]) -> list[Individual]: ...
    def breed(self, survivors: list[Individual], n_offspring: int) -> list[Individual]: ...
```

首版 mutation/crossover 通过结构化 breeding spec 驱动，不做自由文本拼接。

### 3.4 风控与执行包装层

Alpha 分数转仓位的流程：

1. 流动性 / 时段 mask
2. 截面归一化
3. 目标权重生成
4. 仓位变化约束
5. 止盈止损 / cooldown 覆盖
6. 手续费 / 滑点 / funding 进入 PnL

关键接口：

```python
class SignalTransformer:
    def to_target_weights(self, alpha, market_ctx) -> TensorLike: ...

class RuleOverlay:
    def apply(self, target_weights, market_ctx) -> TensorLike: ...

class ExecutionSimulator:
    def simulate(self, weights, prices, cost_model) -> BacktestResult: ...
```

## 4. 防止过拟合与 OOS

建议的时间切分：

- `train`: 50%
- `valid`: 20%
- `test`: 20%
- `purge/buffer`: 10%
- `final_holdout`: 最近完整 regime，仅用于最终验收

必须落实的防坑项：

- 禁止未来可得字段泄露到当前 bar
- 合约上市/下线必须进入样本管道
- 低流动性时段单独打标，避免伪收益
- 所有成本在评估阶段进入，不做事后补丁

Bitget 成本模型首版要求：

- fee：maker/taker 参数化
- slippage：`spread + participation_rate + volatility_proxy`
- funding：跨 funding 窗口的持仓现金流

## 5. 开发路径图

### Phase 1: Foundation

- 新建 `src/alpha_lab`
- 完成 DSL、校验、编译器、参考 VM
- 接入 Bitget 历史数据适配器

里程碑：

- 单公式可从文本编译为 bytecode
- 公式可在参考 VM 上跑通

### Phase 2: Tensor Execution

- 引入共享表达式缓存
- 完成 rolling/cs operator 的批量执行接口
- 建立风险包装与仿真执行器

里程碑：

- 公式评估输出净收益、换手、成本后 Sharpe

### Phase 3: Search Loop

- 引入 `EvolutionEngine`
- 形成 seed -> evaluate -> select -> breed 的代际循环
- 建立 alpha zoo 元数据与 lineage 记录

里程碑：

- 多代实验可自动执行并产出候选因子集

### Phase 4: API & Paper Trading

- 暴露 Alpha Lab API
- 打通实验提交、编译检查、公式评估、breed 调试
- 接通 paper trading 路径

里程碑：

- 从 Bitget 数据到 Alpha Zoo 再到仿真执行形成闭环

## 6. 后续子文档拆分约定

本文是总设计文档，不承载全部细节。后续建议继续拆分：

- `docs/ALPHA_DSL_SPEC.md`
- `docs/STACK_VM_EXECUTION_PLAN.md`
- `docs/BITGET_DATA_PIPELINE.md`
- `docs/EVOLUTION_ENGINE_SPEC.md`
- `docs/PAPER_TRADING_AND_RISK.md`
