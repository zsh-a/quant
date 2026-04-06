# Development Guide

## Prerequisites

- Python >= 3.10
- Node.js / Bun
- Redis
- ClickHouse
- UV (Python package manager)
- Docker & Docker Compose (for containerized deployment)

## Local Setup

### 1. Python Environment

```bash
# Install UV if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Frontend

```bash
cd ui
bun install
```

### 3. Services

```bash
# Start Redis (required for Celery)
redis-server

# Start ClickHouse (required for market data)
# See ClickHouse docs for installation

# Or use Docker for infrastructure only:
docker compose up redis clickhouse -d
```

### 4. Run Everything

```bash
# All-in-one local dev
./dev_local.sh up

# Or manually:
# Terminal 1: API
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --ws-ping-timeout 60 --reload

# Terminal 2: Celery worker
celery -A src.tasks.celery_app worker --loglevel=info --concurrency=4 \
  --queues=backtest,default,automation

# Terminal 3: Frontend
cd ui && bun run dev

# Terminal 4 (optional): Flower monitoring
celery -A src.tasks.celery_app flower --port=5555
```

## Environment Variables

Key variables (set in `.env` or shell):

```bash
# Database
QUANT_DATABASE__HOST=localhost
QUANT_DATABASE__PORT=8123

# Celery
CELERY_BROKER_URL=redis://127.0.0.1:6379/0
CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/1

# Session DB
SESSION_DB_PATH=sessions.db

# API
API_PORT=8000
UI_PORT=5173

# Alpha Lab LLM
ALPHA_MINING_API_KEY=...
ALPHA_MINING_MODEL=...
ALPHA_MINING_BASE_URL=...

# Observability
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...

# Notifications
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

Full config override via `QUANT_*` env vars (see `src/config/settings.py`).

## Configuration

### Settings Hierarchy

1. Defaults in `src/config/settings.py`
2. Override with `config/system_config.yaml`
3. Override with `QUANT_*` environment variables

### Nested Config Override

Use `__` for nested keys:

```bash
QUANT_BROKER__BACKTEST__INITIAL_CASH=5000000
QUANT_BROKER__BACKTEST__COMMISSION=0.0002
QUANT_LOGGING__LEVEL=DEBUG
QUANT_ALPHA_LAB__POPULATION_SIZE=128
```

## Testing

```bash
# Run all tests
pytest

# Specific test file
pytest tests/test_alpha_lab.py

# With verbose output
pytest -v tests/test_session_services.py

# Skip slow tests (marked in pyproject.toml)
pytest -m "not slow"
```

### Test Files

| File | Coverage |
|------|----------|
| `tests/test_alpha_lab.py` | Alpha lab functionality |
| `tests/test_alpha_lab_db.py` | Alpha lab database ops |
| `tests/test_session_db_pagination.py` | Session pagination |
| `tests/test_session_services.py` | Session service layer |
| `tests/test_crypto_market_data.py` | Crypto data pipeline |
| `tests/test_triton_kernels.py` | GPU kernel tests |
| `src/tests/test_engine_manual.py` | Manual engine tests |
| `src/tests/test_migrated_strategies.py` | Strategy migration |

### Ignored Test Patterns

Configured in `pyproject.toml`:
- `alpha_mining`, `E2E`, `persistence`, `websocket` tests excluded by default

## Docker Deployment

### Production

```bash
# Build and start all services
docker compose up -d --build

# Check status
docker compose ps

# View logs
docker compose logs -f api
docker compose logs -f celery_worker

# Stop
docker compose down
```

### Services

| Service | Image | Port | Health Check |
|---------|-------|------|-------------|
| Redis | redis:7-alpine | 6379 | `redis-cli ping` |
| API | Custom (Python 3.14) | 8000 | `GET /monitoring/health` |
| Celery | Same as API | - | - |
| Frontend | Custom (Bun + Nginx) | 80 | - |

### Volumes

- `data/` - Runtime data (checkpoints, reports, logs)
- `config/` - Configuration files
- `sessions.db` - Session database (shared between API and workers)

## Make Targets

```bash
make dev_local          # Start local dev
make dev_local_down     # Stop local dev
make dev_local_status   # Check status
make clean_models       # Remove .pth files
make clean_runs         # Remove runs/ directory
make clean_logs         # Remove .log files
make clean_all          # All clean targets
```

## Project Scripts

| Script | Purpose |
|--------|---------|
| `dev_local.sh` | Local development orchestration |
| `deploy.sh` | Production deployment with health checks |
| `start_worker.sh` | Celery worker startup |
| `client.py` | API client wrapper |
| `session_db.py` | Session database management |
| `find_last_trading_day.py` | Trading calendar utility |

## Adding a New Strategy

1. Create strategy file in `src/strategies/`:

```python
from src.core.base import Strategy
from src.strategies.registry import StrategyRegistry

@StrategyRegistry.register("my_strategy", label="My Strategy", description="...")
class MyStrategy(Strategy):
    @classmethod
    def get_parameters(cls):
        return {
            "param1": {"type": "int", "default": 10, "min": 1, "max": 100}
        }

    def on_bar(self, bars):
        for symbol, bar in bars.items():
            # Trading logic
            if should_buy:
                self.buy(symbol, quantity, execution_type="IMMEDIATE_CLOSE")
            if should_sell:
                self.sell(symbol, quantity, execution_type="IMMEDIATE_CLOSE")
```

2. Import in `src/strategies/registry.py`'s `register_all()` method.

3. Strategy appears automatically in API (`GET /strategies`) and UI.

## Adding a New API Endpoint

1. Create or extend router in `src/api/`:

```python
from fastapi import APIRouter
router = APIRouter(prefix="/my-feature", tags=["my-feature"])

@router.get("/data")
async def get_data():
    return {"result": "..."}
```

2. Include in `src/api/server.py`:

```python
from src.api.my_router import router as my_router
app.include_router(my_router)
```

## Monitoring

- **Health**: `GET /monitoring/health` - checks Redis, ClickHouse, Celery
- **Metrics**: `GET /monitoring/metrics` - Prometheus format
- **Flower**: `http://localhost:5555` - Celery task monitoring
- **Logs**: Loguru to console + `logs/quant.log` (100MB rotation, 30 days retention)
- **Alerts**: Telegram notifications via `src/notifications/telegram.py`
