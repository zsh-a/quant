# Architecture

## System Overview

```
                        ┌─────────────────────────────────────────┐
                        │              Docker Network              │
                        │                                         │
[Browser] ──:80──> [Nginx + React UI]                            │
                        │                                         │
                   HTTP/WS :8000                                  │
                        │                                         │
                   [FastAPI Server]                                │
                   ├── REST API (16 routers)                      │
                   ├── WebSocket (real-time events)               │
                   ├── Event Bus (session lifecycle)              │
                   │                                              │
              Redis :6379                                         │
                   │                                              │
              [Celery Workers ×4]                                 │
              ├── backtest queue                                  │
              ├── default queue                                   │
              └── automation queue                                │
                   │                                              │
              [ClickHouse :8123]                                  │
              ├── Market data (A-share, crypto)                   │
              ├── Factor storage (jointdata)                      │
              └── Alpha evaluation results                        │
                        │                                         │
              [SQLite sessions.db]                                │
              └── Session state persistence                       │
                        └─────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, TypeScript, Vite, Chakra UI, TailwindCSS, Zustand |
| API | FastAPI, Pydantic, WebSocket, SSE |
| Task Queue | Celery, Redis (broker + result backend) |
| Database | ClickHouse (columnar), SQLite (sessions) |
| Data Sources | Akshare, Baostock, TDX, CCXT, Binance Vision |
| ML/AI | LLM (OpenAI-compatible), PyTorch, Triton (GPU), MCTS |
| Monitoring | Prometheus, Loguru, Telegram alerts |
| Deployment | Docker Compose, Nginx, UV, Bun |

## Module Dependency Graph

```
src/api/          ← HTTP/WS interface
  ├── src/core/          ← Trading engine
  │   ├── src/strategies/    ← Strategy registry
  │   └── src/config/        ← Settings
  ├── src/alpha/         ← Alpha search system
  │   └── src/market_data/   ← Data access
  ├── src/portfolio/     ← Portfolio management
  ├── src/services/      ← Session orchestration
  ├── src/tasks/         ← Celery task definitions
  ├── src/analysis/      ← Metrics & reports
  ├── src/monitoring/    ← Health & metrics
  └── src/automation/    ← Simulation jobs
```

## Data Flow

### Backtesting Flow

```
SessionRequest → API → TradingService.create_session()
  → DataStream (CSV/DB/Realtime)
  → TradingEngine.run() [bar-by-bar loop]
    → Strategy.on_bar() → Order
    → RiskManager.check() → approved/rejected
    → Broker.step() → fill orders, update equity
    → EventBus → WebSocket → UI
  → SessionResult (metrics, equity, trades)
```

### Alpha Search Flow

```
SearchDbRequest → API → AlphaService.search_formulas_on_db()
  → Load dataset from ClickHouse
  → SearchOrchestrator (round-based loop)
    → Strategy.generate_candidates() → formulas
    → Compiler → BytecodeProgram
    → StackVM.execute() → alpha signals
    → Evaluation (rank_ic, sharpe, turnover)
    → Archive update (MAP-Elites)
    → SSE progress → UI
  → PipelineRecord (rounds, stages, top formulas)
```

### Market Data Flow

```
Data Sources (Akshare/Baostock/TDX/CCXT)
  → Processors (normalize, validate)
  → ClickHouse (partitioned by date, indexed by stock_code)
  → DBDataStream (chunked loading for backtests)
  → TradingEngine / AlphaService
```

## Event System

WebSocket events broadcast through `EventBus` (`src/api/events.py`):

| Event | Trigger |
|-------|---------|
| `SESSION_STARTED` | Session begins execution |
| `SESSION_PROGRESS` | Bar processed, progress % |
| `EQUITY_UPDATE` | Portfolio value change |
| `TRADE_EXECUTED` | Order filled |
| `SESSION_COMPLETED` | Session finished |
| `SESSION_FAILED` | Session error |
| `SESSION_STOPPED` | Manual stop |
| `DATA_UPDATE_*` | Market data pipeline events |
| `SIMULATION_BATCH_*` | Automation batch events |
| `STRATEGY_STEP` | Strategy-level events |

## Key Design Patterns

- **Factory Pattern**: `TradingService` creates strategies, brokers, and data streams via factories
- **Registry Pattern**: `StrategyRegistry` for plug-in strategy discovery (`@register` decorator)
- **Observer Pattern**: Event bus + WebSocket for real-time UI updates
- **Strategy Pattern**: Multiple broker implementations (backtest, live), multiple data streams (CSV, DB, realtime)
- **MAP-Elites**: Alpha search archive bins by IC and turnover for diversity
- **Pipeline Overlap**: LLM candidate generation overlaps with current round evaluation
