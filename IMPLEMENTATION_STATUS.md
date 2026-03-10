# 实施进度总结（2026-03-11）

本文档记录“优化与扩展计划”的已实现与未实现内容，方便后续继续推进与回归验证。

## 已实现

### 后端与会话编排
- 抽出统一回测执行内核与会话服务：
  - `src/services/session_execution.py`：统一同步/异步回测执行路径，负责进度、落库、最终结果汇总。
  - `src/services/session_service.py`：会话运行态管理、状态查询、删除/停止、checkpoint 构建与恢复。
- `server.py` 去掉全局 `SESSIONS` 作为事实来源，运行态与持久化状态分离。
- `session/run` 与 Celery 任务复用同一执行逻辑（`src/tasks/backtest.py` 复用执行内核）。
- Checkpoint 逻辑改为基于持久化数据构建，不再依赖内存中不存在的权益/成交。

### API 能力扩展
- 新增资源风格别名接口，保留旧接口不破：
  - `/sessions/{id}`、`/sessions/{id}/equity`、`/sessions/{id}/trades`、
    `/sessions/{id}/metrics`、`/sessions/{id}/stop`、
    `/sessions/{id}/checkpoint`、`/sessions/{id}/checkpoints`、`/sessions/{id}/restore`
- 增加权益/成交分页查询能力（`session_db.py`：`get_equity_history_page`、`get_trades_page`）。
- 补全生命周期事件：`session_failed`、`session_stopped`，并在前端消费。

### 前端性能与数据路径
- 按 tab 懒加载重面板，减少首屏负载（`ui/src/App.tsx`）。
- Heatmap 仅加载必要的 ECharts 模块，减小图表包体（`ui/src/components/charts/HeatmapChart.tsx`）。
- `App.tsx` 全量会话加载改用分页接口，避免一次性拉大数据。
- Vite `manualChunks` 拆分 vendor / ui-kit / echarts / recharts；警告阈值调整到 700KB。

### 测试基线
- 默认 `pytest` 不再卡在收集阶段：
  - `pyproject.toml` 统一 pytest 配置（忽略 `db/` 与手工脚本型测试）。
- 新增核心单测：
  - `tests/test_session_services.py`、`tests/test_session_db_pagination.py`
- 修正“返回 bool”测试风格，统一为断言并对集成测试默认 skip。
- 运行结果（本地）：
  - `uv run pytest -q`：`55 passed, 7 skipped, 1 warning`
  - `npm run build`：通过，ECharts chunk 已降至 ~596KB。

## 未实现 / 待办

### 架构层面
- 会话模型的“Session / Run / Job”三层拆分与统一（目前仍是单一 session 语义）。
- 数据访问层抽象（Repository/Data Provider），ClickHouse 与 SQLite 仍是直接访问。
- 事件协议版本化（WebSocket/事件总线的 schema 仍是松散形式）。
- 系统级容量与并发边界（回测并发、缓存策略、任务优先级）。

### 产品扩展
- 策略插件化（策略包元数据、参数 schema、自描述、可插拔加载）。
- 多资产/多市场统一抽象（目前偏 A 股日线为主）。
- 实盘交易中台能力（订单生命周期、幂等、审计、失败补偿）。
- 多用户协作、权限与配额体系。
- 因子/特征工程平台化（feature store、实验跟踪、因子评估流水线）。

### 前端体验
- 更细粒度的图表包拆分（ECharts 仍为最大包，虽然已显著缩小）。
- 数据层进一步收口（比如新增“session summary/status-only”接口以减少轮询负载）。

## 验证与说明

- 默认自动化测试入口已稳定，但仍有 1 个第三方依赖警告（`py_mini_racer` 结构体布局）。
- 集成测试默认不跑，需设置 `RUN_INTEGRATION=1` 并启动 API / Redis / Celery。

## 备注

- `src/market_data/processors/baostock.py` 与 `src/market_data/update_pipeline.py` 有用户已有改动，未在本轮修改范围内。
- 前端构建警告已消除，当前最大的包是 ECharts 的精简 bundle。
