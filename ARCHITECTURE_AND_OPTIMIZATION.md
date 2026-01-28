# 架构分析与优化方案

## 1. 当前架构分析

### 1.1 核心模块 (Core & Backend)
*   **Engine & Broker 分离**: 采用了经典的 `Engine` (驱动), `Broker` (交易执行), `Strategy` (逻辑), `DataStream` (数据源) 分离的设计，符合回测框架的标准范式。
*   **Broker 实现**:
    *   `BacktestBroker`: 本地维护资金、持仓和订单状态，逻辑清晰。
    *   `LiveBroker`: 通过 HTTP 请求与外部实盘服务（如 `xiadan-client`）交互，实现了基本的下单和资产查询。
*   **DataStream**:
    *   `DBDataStream`: 目前在初始化时将所有历史数据加载到内存 (`self.data` 字典)。这对全市场回测或长周期回测是极大的内存瓶颈。
    *   **实盘缺失**: 目前没有真正的“实盘数据流” (`RealtimeDataStream`)。实盘模式下，代码似乎仍然在使用 `DBDataStream` 读取历史数据，这会导致实盘 session 瞬间跑完历史数据后结束，无法持续运行。
*   **Session 管理**:
    *   `server.py` 中使用 `SESSIONS` 全局字典存储所有状态。**严重问题**: 服务重启后，所有回测记录、实盘状态都会丢失。
*   **并发模型**:
    *   使用 `BackgroundTasks` 在线程池中运行回测。对于计算密集型任务（回测），这可能会阻塞 FastAPI 的主线程（如果未正确使用 `async` 或进程隔离）。

### 1.2 前端 (UI)
*   **技术栈**: React + Vite + Chakra UI + Recharts，现代且高效。
*   **数据交互**:
    *   采用 **轮询 (Polling)** 机制 (`setInterval` 每秒请求)。
    *   **性能瓶颈**: `/session/{id}/status` 接口每次返回**全量**的 `equity_history`。随着回测天数增加，响应体将越来越大，导致网络拥塞和前端渲染卡顿。
*   **展示**: 提供了基本的权益曲线、持仓和成交列表，功能完备但缺乏深度的交互分析（如：无法在图表上点击查看当日持仓快照，虽然代码有 `setSelectedDay` 逻辑，但数据流并未完全打通或优化）。

---

## 2. 优化方案建议

### 2.1 核心层 (Core) 优化

#### A. 解决内存瓶颈与数据流重构
**目标**: 支持全市场、长周期回测，支持真正的实盘事件驱动。
*   **Iterator-based DataStream**: 重写 `DBDataStream`，不再一次性 `load` 所有数据。
    *   **回测**: 使用数据库游标 (Generator) 或分块读取 (Chunking)。
    *   **实盘**: 实现 `RealtimeDataStream`。
        *   **机制**: 监控系统时间或订阅消息队列 (Redis/ZMQ)。
        *   **阻塞**: `next_bar()` 在实盘模式下应阻塞等待，直到新的 K 线生成或收到 Tick 数据，而不是返回 `None` 结束运行。

#### B. 持久化层 (Persistence)
**目标**: 服务重启不丢失数据，支持历史回看。
*   **引入 SQLite/PostgreSQL**:
    *   **Tables**: `sessions`, `orders`, `trades`, `equity_curve`.
    *   **ORM**: 使用 SQLAlchemy 或 Tortoise ORM。
    *   **流程**: `Session` 创建时写入 DB；`BacktestBroker` 的成交和权益更新实时（或批量）写入 DB。

#### C. 实盘健壮性
**目标**: 提高实盘稳定性和容错。
*   **状态恢复**: `TradingEngine` 启动时应从 DB 加载上次的持仓和净值。
*   **信号一致性**: 策略生成的信号需要与当前实际持仓对比（Diff），生成调仓指令，而非假设初始仓位为 0。
*   **风控层**: 在 Broker 前增加 `RiskManager`，拦截异常的大额下单或频繁撤单。

### 2.2 接口层 (API) 优化

#### A. 增量更新 (Delta Updates)
**目标**: 降低带宽消耗，提升 UI 响应速度。
*   **API 改造**: `/session/{id}/status` 增加参数 `since_timestamp`。
    *   前端维护 `last_updated_ts`。
    *   后端只返回 `timestamp > since_timestamp` 的权益数据和成交记录。

#### B. 异步与任务队列
*   **Celery / RQ**: 将回测任务从 API 进程中剥离。避免回测计算抢占 API 服务的 CPU 资源。

### 2.3 前端 (UI) 优化

#### A. 性能优化
*   **增量合并**: 前端收到增量数据后，追加到本地 State 数组中，而不是全量替换。
*   **图表降采样**: 如果数据点超过 2000 个，使用 LTTB (Largest-Triangle-Three-Buckets) 算法进行降采样展示，保持图表流畅。

#### B. 交互增强
*   **配置化启动**: 在“Start New Session”界面支持更多策略参数配置（不仅是日期，还包括资金、滑点、策略特定参数），通过 JSON Schema 动态生成表单。

---

## 3. 实施路线图 (Roadmap)

### 第一阶段：基础稳固 (Infrastructure)
1.  **数据库集成**: 引入 `sqlite.db`，替换内存中的 `SESSIONS` 字典。
2.  **增量 API**: 修改 `/session/{id}/status` 支持 `since` 参数。

### 第二阶段：实盘增强 (Live Trading)
1.  **RealtimeStream**: 实现基于系统时钟轮询的实盘数据流。
2.  **Broker 状态同步**: 实盘 Broker 启动时同步账户真实持仓到本地 `self.positions`。

### 第三阶段：性能与扩展 (Performance)
1.  **流式回测**: 重构 `DBDataStream` 为迭代器模式。
2.  **高级 UI**: 增加策略参数配置面板，增加多策略对比功能。
