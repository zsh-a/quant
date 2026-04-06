# API Reference

Base URL: `http://localhost:8000`

## Session Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/strategies` | List registered strategies |
| POST | `/session/run` | Run session synchronously |
| POST | `/session/run_async` | Submit to Celery queue |
| GET | `/sessions` | List all sessions |
| DELETE | `/session/{session_id}` | Delete session |
| GET | `/session/{session_id}/status` | Session status (`?since=timestamp`) |
| GET | `/sessions/{session_id}` | Session details |
| GET | `/sessions/{session_id}/equity` | Equity history (`?since&limit&offset`) |
| GET | `/sessions/{session_id}/trades` | Trade history (`?since&limit&offset`) |
| GET | `/session/{session_id}/metrics` | Performance metrics |
| POST | `/session/{session_id}/stop` | Stop running session |
| POST | `/session/{session_id}/checkpoint` | Create checkpoint |
| GET | `/session/{session_id}/checkpoints` | List checkpoints |
| POST | `/session/{session_id}/restore` | Restore from checkpoint |
| GET | `/session/{session_id}/risk` | Risk metrics |
| GET | `/market/benchmark` | Benchmark data (`?symbol&start_date&end_date`) |
| GET | `/status` | Server status & active sessions |
| GET | `/persistence/stats` | Persistence statistics |
| GET | `/cache/stats` | Cache statistics |
| POST | `/cache/clear` | Clear cache (`?pattern`) |

### SessionRequest

```json
{
  "strategy": "momentum",
  "symbol": "sh.600000",
  "start_date": "2023-01-01",
  "end_date": "2024-01-01",
  "mode": "backtest",
  "params": { "lookback": 20, "top_n": 10 }
}
```

## Task Queue (`/tasks`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/tasks/backtest` | Submit backtest task |
| GET | `/tasks/backtest/{task_id}` | Task status |
| DELETE | `/tasks/backtest/{task_id}` | Cancel task |
| GET | `/tasks/backtest` | List recent tasks (`?limit`) |
| GET | `/tasks/workers` | Active Celery workers |

### BacktestTaskRequest

```json
{
  "session_id": "uuid",
  "symbol": "sh.600000",
  "strategy": "momentum",
  "start_date": "2023-01-01",
  "end_date": "2024-01-01",
  "params": {},
  "initial_cash": 1000000,
  "commission": 0.0003,
  "slippage": 0.001,
  "enable_risk_management": true,
  "chunk_size_months": 3
}
```

## Monitoring (`/monitoring`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/monitoring/health` | Health check (all subsystems) |
| GET | `/monitoring/metrics` | Prometheus metrics |
| GET | `/monitoring/status` | Detailed system status |
| POST | `/monitoring/alert/test` | Send test alert (`?title&message&severity`) |

## Portfolio (`/portfolio`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/portfolio` | Create portfolio |
| GET | `/portfolio` | List portfolios |
| GET | `/portfolio/{id}` | Portfolio details |
| PUT | `/portfolio/{id}/weights` | Update weights |
| POST | `/portfolio/{id}/backtest` | Run portfolio backtest |
| GET | `/portfolio/{id}/result` | Backtest result |

### CreatePortfolioRequest

```json
{
  "name": "multi-factor",
  "strategies": ["momentum", "mean_reversion"],
  "weight_method": "equal",
  "rebalance_frequency": "weekly"
}
```

## Optimization (`/optimize`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/optimize` | Submit optimization |
| GET | `/optimize/{task_id}` | Optimization status |
| GET | `/optimize/{task_id}/results` | Top results (`?top_n`) |
| GET | `/optimize/{task_id}/heatmap` | Parameter heatmap (`?param1&param2`) |
| DELETE | `/optimize/{task_id}` | Cancel optimization |

## Analysis (`/analysis`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/analysis/attribution/{session_id}` | Return attribution |
| GET | `/analysis/risk/{session_id}` | Risk attribution |
| GET | `/analysis/report/{session_id}` | Backtest report (`?format=markdown\|json`) |
| GET | `/analysis/summary/{session_id}` | Quick summary |

## Logs (`/logs`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/logs` | List sessions with logs |
| GET | `/logs/{session_id}` | Session logs (`?level&source&since&limit&format`) |
| DELETE | `/logs/{session_id}` | Clear session logs |

## Market Data (`/market`, `/market-admin`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/market/industry_breadth` | Industry breadth (`?start_date&end_date&index_code`) |
| GET | `/market/industry_amount` | Industry trading amount share |
| GET | `/market-admin/overview` | Database stats |
| GET | `/market-admin/update-capabilities` | Available update steps |
| GET | `/market-admin/update-runs` | Update run history (`?limit`) |
| POST | `/market-admin/update-runs` | Trigger data update |

## Automation (`/simulation-jobs`, `/data-update`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/simulation-jobs` | Create simulation job |
| GET | `/simulation-jobs` | List jobs (`?enabled_only`) |
| POST | `/simulation-jobs/run-enabled` | Run all enabled (`?force`) |
| GET | `/simulation-jobs/{id}` | Job details |
| POST | `/simulation-jobs/{id}/enable` | Enable job |
| POST | `/simulation-jobs/{id}/disable` | Disable job |
| POST | `/simulation-jobs/{id}/run` | Run specific job (`?force`) |
| GET | `/simulation-jobs/{id}/runs` | Job run history (`?limit`) |
| GET | `/simulation-runs/{run_id}` | Run details |
| GET | `/simulation-runs/{run_id}/steps` | Run steps (`?limit&since_step`) |
| POST | `/data-update/run` | Trigger data update |
| GET | `/data-update/history` | Update history (`?limit`) |

## Alpha Lab (`/alpha-lab`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/alpha-lab/workspace` | Workspace config (`?run_limit&zoo_limit`) |
| POST | `/alpha-lab/validate` | Validate formula syntax |
| POST | `/alpha-lab/compile` | Compile formula to bytecode |
| POST | `/alpha-lab/evaluate-db` | Evaluate formula on historical data |
| POST | `/alpha-lab/search-db` | Submit GA search job |
| GET | `/alpha-lab/search-jobs/{id}` | Poll search status |
| GET | `/alpha-lab/search-jobs` | List search jobs |
| GET | `/alpha-lab/search-jobs/{id}/events` | SSE progress stream |
| GET | `/alpha-lab/search-jobs/{id}/pipeline` | Pipeline record |
| POST | `/alpha-lab/search-jobs/{id}/analyze` | LLM analysis |
| GET | `/alpha-lab/runs` | List search runs (`?limit`) |
| GET | `/alpha-lab/runs/{run_id}` | Run details |
| GET | `/alpha-lab/zoo` | List zoo formulas (`?limit`) |
| POST | `/alpha-lab/zoo` | Save to zoo |
| POST | `/alpha-lab/combine-zoo` | Combine zoo factors |
| GET | `/alpha-lab/tracing/summary` | Tracing summary |
| GET | `/alpha-lab/tracing/spans` | Recent spans (`?kind&limit`) |
| GET | `/alpha-lab/neural/history` | Neural training history |
| GET | `/alpha-lab/neural/plot` | Training curves |
| GET | `/alpha-lab/strategy-state` | Strategy state |
| GET | `/alpha-lab/checkpoints` | List checkpoints |
| GET | `/alpha-lab/checkpoints/{job_id}` | Job checkpoints |
| GET | `/alpha-lab/factor-catalog` | Query factor catalog |
| GET | `/alpha-lab/factor-catalog/stats` | Catalog stats |

### SearchDbRequest

```json
{
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "start_time": "2024-01-01",
  "end_time": "2024-06-01",
  "interval": "1h",
  "seeds": ["ts_mean(close, 20)"],
  "population_size": 64,
  "generations": 50,
  "strategy": "llm_evolution",
  "run_name": "crypto_search_v1",
  "persist": true
}
```

### EvaluateDbRequest

```json
{
  "formula": "cs_rank(ts_mean(close, 20) - ts_mean(close, 5))",
  "symbols": ["BTCUSDT"],
  "start_time": "2024-01-01",
  "end_time": "2024-06-01",
  "interval": "1h"
}
```

## Crypto Market (`/crypto-market`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/crypto-market/providers` | Available data providers |
| GET | `/crypto-market/overview` | Market data overview |
| GET | `/crypto-market/coverage` | Data coverage (`?interval&limit`) |
| POST | `/crypto-market/init-db` | Initialize database |
| POST | `/crypto-market/bootstrap` | Bootstrap default data |
| POST | `/crypto-market/backfill` | Backfill historical data |
| POST | `/crypto-market/sync` | Sync minute bars |
| POST | `/crypto-market/sync-default` | Sync defaults (`?async_mode`) |
| GET | `/crypto-market/bars` | Query bars (`?provider&symbol&start_time&end_time&interval`) |

## WebSocket Protocol

**Endpoint**: `ws://localhost:8000/ws/{session_id}`

### Client -> Server

```json
{ "type": "pong" }
{ "type": "subscribe", "session_id": "..." }
{ "type": "unsubscribe", "session_id": "..." }
```

### Server -> Client

| Type | Fields | Description |
|------|--------|-------------|
| `ping` | - | Heartbeat (every 30s) |
| `session_started` | `session_id`, `data` | Session initialized |
| `session_progress` | `session_id`, `data.progress` | Progress update |
| `equity_update` | `session_id`, `data.equity` | Portfolio value |
| `equity_batch` | `session_id`, `data.updates[]` | Batched equity |
| `trade_executed` | `session_id`, `data.trade` | Order filled |
| `trades_batch` | `session_id`, `data.trades[]` | Batched trades |
| `session_completed` | `session_id`, `data.metrics` | Finished |
| `session_failed` | `session_id`, `data.error` | Error |
| `session_stopped` | `session_id` | Manually stopped |

All messages include `timestamp` (ISO 8601).
