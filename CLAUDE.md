# Quent - AI-Powered Quantitative Trading Platform

> Crypto & A-share alpha factor discovery, backtesting, and live trading.

## Quick Reference

| Item | Value |
|------|-------|
| Version | 3.0.0 |
| Python | >= 3.10 |
| Backend | FastAPI + Celery + Redis |
| Frontend | React 19 + TypeScript + Vite |
| Database | ClickHouse (market data), SQLite (sessions) |
| Package Manager | UV (Python), Bun (JS) |
| Entry Point | `src/api/server.py:app` |

## Project Structure

```
src/
  api/            # FastAPI routers, WebSocket, event bus
  core/           # Trading engine, brokers, data streams, risk manager
  alpha/          # Alpha factor DSL, VM, search strategies, LLM integration
    strategies/   # Neural formula, LLM evolution, MCTS refinement
  strategies/     # Trading strategy registry and templates
  portfolio/      # Multi-strategy portfolio management
  market_data/    # ClickHouse DB, data processors (akshare, baostock, tdx, crypto)
  analysis/       # Performance metrics, attribution, report generators
  monitoring/     # Health checks, Prometheus metrics, alerts
  notifications/  # Telegram notifications
  tasks/          # Celery task definitions (backtest, automation, crypto, data)
  services/       # Session execution service
  automation/     # Simulation job automation
  config/         # Pydantic settings (env override: QUANT_*)
  utils/          # Cache, logging
ui/               # React 19 + Chakra UI + Zustand frontend
  src/components/ # Dashboard, AlphaLab, Portfolio, Session management
  src/hooks/      # useWebSocket, useChartData, useSearchSSE
  src/store/      # Zustand session store
jointdata/        # ClickHouse factor storage system (independent module)
data/             # Runtime data: checkpoints, logs, reports, alpha results
config/           # YAML configuration files
tests/            # pytest test suite
```

## Key Commands

```bash
# Development
./dev_local.sh up              # Start all services (API + Celery + Redis + UI)
./dev_local.sh down            # Stop all services
./dev_local.sh status          # Check service status
make dev_local                 # Alias for ./dev_local.sh up

# Backend only
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --ws-ping-timeout 60

# Celery worker
celery -A src.tasks.celery_app worker --loglevel=info --concurrency=4 \
  --queues=backtest,default,automation --max-tasks-per-child=100

# Frontend
cd ui && bun install && bun run dev    # Dev server on :5173

# Docker
docker compose up -d                   # Production deployment
docker compose down

# Testing
pytest                                 # Run test suite
pytest tests/test_alpha_lab.py         # Specific test

# Maintenance
make clean_all                         # Remove .pth, runs/, .log files
```

## Architecture Overview

```
[React UI :80] ──HTTP/WS──> [FastAPI :8000] ──Redis──> [Celery Workers]
                                  │                         │
                                  ├── ClickHouse (market data, factors)
                                  ├── SQLite sessions.db (session persistence)
                                  └── Event Bus (WebSocket broadcast)
```

### Services (docker-compose)

| Service | Port | Purpose |
|---------|------|---------|
| Redis | 6379 | Message broker + cache |
| API | 8000 | FastAPI application |
| Celery | - | Background tasks (backtest, automation) |
| Frontend | 80 | React app via Nginx |

### Core Modules

1. **Trading Engine** (`src/core/engine.py`): Bar-by-bar simulation loop. Order types: `NEXT_OPEN`, `IMMEDIATE_OPEN`, `IMMEDIATE_CLOSE`.
2. **Alpha Search** (`src/alpha/`): DSL compiler -> StackVM bytecode -> vectorized evaluation. Search strategies: LLM evolution, MCTS refinement, neural formula (transformer RPN), enumeration.
3. **Market Data** (`src/market_data/`): Multi-source pipeline (Akshare, Baostock, TDX, CCXT/Binance). ClickHouse storage with chunked loading.
4. **Task Queue** (`src/tasks/`): Celery queues - `backtest`, `default`, `automation`. Redis broker.

## Configuration

Settings via Pydantic (`src/config/settings.py`). Override with env vars prefixed `QUANT_`:

```bash
QUANT_BROKER__BACKTEST__INITIAL_CASH=5000000
QUANT_BROKER__BACKTEST__COMMISSION=0.0002
QUANT_DATABASE__HOST=localhost
QUANT_DATABASE__PORT=8123
QUANT_LOGGING__LEVEL=DEBUG
```

Or via `config/system_config.yaml`.

## API Structure

All endpoints served from FastAPI at `:8000`. Key route groups:

| Prefix | Router File | Purpose |
|--------|-------------|---------|
| `/` | `server.py` | Sessions CRUD, strategies, benchmarks |
| `/tasks` | `tasks_router.py` | Celery backtest task management |
| `/monitoring` | `monitoring_router.py` | Health, metrics, alerts |
| `/portfolio` | `portfolio_router.py` | Portfolio CRUD, backtesting |
| `/optimize` | `optimizer_router.py` | Parameter optimization |
| `/analysis` | `analysis_router.py` | Attribution, reports |
| `/logs` | `logs_router.py` | Session log access |
| `/market` | `market_router.py` | Industry breadth/amount |
| `/market-admin` | `market_admin_router.py` | Data update management |
| `/automation` | `automation_router.py` | Simulation jobs, data updates |
| `/alpha-lab` | `alpha_lab_router.py` | Factor search, zoo, evaluation |
| `/crypto-market` | `crypto_market_router.py` | Crypto data sync/query |
| `/ws/{session_id}` | `websocket_manager.py` | Real-time session events |

Health check: `GET /monitoring/health`

## Conventions

- **Language**: Chinese for user-facing text and comments where appropriate
- **Strategy registration**: Use `@StrategyRegistry.register("name")` decorator
- **Order execution**: Prefer `IMMEDIATE_CLOSE` for daily backtests to avoid look-ahead bias
- **Config access**: Use `get_settings()`, `get_broker_config()`, etc. from `src/config/settings.py`
- **Logging**: Use `loguru` via `src/utils/logging_config.py`
- **Session DB**: SQLite via `session_db.py` at project root
- **Alpha formulas**: Python expression syntax, compiled to bytecode via `src/alpha/compiler.py`

## Documentation

See [docs/README.md](docs/README.md) for detailed documentation:

- [Architecture](docs/architecture.md) - System design and data flow
- [API Reference](docs/api.md) - All endpoints with request/response models
- [Alpha Search](docs/alpha-search.md) - Factor discovery DSL, VM, strategies
- [Backtesting](docs/backtesting.md) - Engine, brokers, risk management
- [Data Pipeline](docs/data-pipeline.md) - Market data sources and storage
- [Frontend](docs/frontend.md) - React components and state management
- [Development](docs/development.md) - Setup, testing, deployment
