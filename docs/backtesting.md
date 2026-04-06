# Backtesting Engine

## Engine

**File**: `src/core/engine.py`

`TradingEngine` runs bar-by-bar simulation:

```
for each bar:
  1. Fetch next bar from DataStream
  2. Broker.step(bars)          → process NEXT_OPEN orders at current open
  3. Strategy.on_bar(bars)      → generate signals/orders
  4. Broker.process_same_bar()  → execute IMMEDIATE_OPEN/CLOSE orders
  5. Progress callback          → EventBus → WebSocket → UI
```

### Order Execution Types

| Type | Fill Price | Use Case |
|------|-----------|----------|
| `NEXT_OPEN` | Next bar's open | Default, no look-ahead bias |
| `IMMEDIATE_OPEN` | Current bar's open | Intraday, caution with daily data |
| `IMMEDIATE_CLOSE` | Current bar's close | Recommended for daily backtests |

**Important**: `NEXT_OPEN` orders on the last bar never fill (no next bar exists).

## Strategy Interface

**File**: `src/core/base.py`

```python
class Strategy(ABC):
    def on_bar(self, bars: Dict[str, Bar]) -> None:
        """Called each bar. Generate orders here."""
        ...

    def buy(self, symbol, quantity, price=None,
            execution_type="NEXT_OPEN") -> Order: ...

    def sell(self, symbol, quantity, price=None,
             execution_type="NEXT_OPEN") -> Order: ...

    @classmethod
    def get_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """Parameter schema for UI/optimization."""
        return {}
```

### Bar Dataclass

```python
@dataclass
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    extra: Dict[str, Any]  # e.g., isst flag
```

## Strategy Registry

**File**: `src/strategies/registry.py`

```python
@StrategyRegistry.register("my_strategy", label="My Strategy", description="...")
class MyStrategy(Strategy):
    @classmethod
    def get_parameters(cls):
        return {
            "lookback": {"type": "int", "default": 20, "min": 5, "max": 100},
            "threshold": {"type": "float", "default": 0.5}
        }

    def on_bar(self, bars):
        # Trading logic here
        ...
```

**Registry methods**: `create_strategy(name, db_client, session_id, **kwargs)`, `list_strategies()`, `get_parameters(name)`

## Strategy Templates

**File**: `src/strategies/templates.py`

Built-in templates with parameter schemas:

| Template | Key Parameters |
|----------|---------------|
| `MOMENTUM` | lookback (5-252), top_n (1-50), rebalance_freq |
| `MEAN_REVERSION` | lookback (5-100), z_entry (-4 to 0), max_holding_days |
| `TREND_FOLLOWING` | fast_period (2-50), slow_period (10-200), ma_type, atr_multiplier |
| `FACTOR` | momentum/value/quality/volatility weights, top_n |

Categories: `MOMENTUM`, `MEAN_REVERSION`, `TREND_FOLLOWING`, `FACTOR`, `STATISTICAL`, `MACHINE_LEARNING`, `CUSTOM`

## Brokers

### BacktestBroker

**File**: `src/core/backtest_broker.py`

Paper trading broker with realistic simulation:

```python
BacktestBroker(
    initial_cash=1000000.0,
    commission=0.0003,    # 0.03% per trade
    slippage=0.001,       # 0.1% slippage
    risk_manager=None     # Optional
)
```

**Execution model**:
- Slippage: buy × (1 + slippage), sell × (1 - slippage)
- Commission deducted on both buy and sell
- Average cost basis includes commissions
- Limit up/down clamping: ST 5%, regular 10%, TECH (sh.68/sz.30) 20%

**State**: positions, position_costs, cash, equity_history, trades. Supports snapshot/restore for checkpoints.

### LiveBroker

**File**: `src/core/live_broker.py`

HTTP interface to external trading server:
- Server URL: `http://localhost:11122` (configurable)
- Endpoints: `/buy`, `/sell`, `/balance`, `/position`
- Position data in Chinese field names (证券代码, 股票余额, etc.)

## Risk Management

**File**: `src/core/risk_manager.py`

```python
@dataclass
class RiskLimits:
    max_position_pct: float = 0.10      # 10% max per position
    max_total_position: float = 0.95    # 95% max total exposure
    stop_loss_pct: float = 0.05         # 5% stop loss
    take_profit_pct: float = 0.15       # 15% take profit
    max_daily_loss_pct: float = 0.10    # 10% daily loss limit
    max_drawdown_pct: float = 0.20      # 20% max drawdown
```

**Checks**: position size limit, stop loss, take profit, daily loss limit, max drawdown. Auto-generates `IMMEDIATE_CLOSE` sell orders when limits trigger.

## Portfolio Management

**File**: `src/portfolio/portfolio_manager.py`, `src/portfolio/backtest.py`

Multi-strategy portfolio orchestration:

```python
PortfolioManager(
    strategies=[...],
    weight_method="equal",           # equal, volatility_inverse, sharpe_weighted, custom
    rebalance_frequency="weekly",    # daily, weekly, monthly
    min_weight=0.05,
    max_weight=0.40
)
```

**Signal combination**: Weighted aggregation of per-strategy signals. Buy/sell scores compared against 0.3 threshold.

**PortfolioBacktester**: Creates per-strategy brokers with proportional capital, runs coordinated backtest, calculates portfolio-level metrics (Sharpe, max drawdown, per-strategy attribution).

## Trading Service

**File**: `src/core/trading_service.py`

Decouples engine from API via factory pattern:

```python
service = TradingService(
    strategy_factory=...,
    broker_factory=...,
    data_stream_factory=...
)

session_id = service.create_session(config)
result = service.run_session(session_id, on_progress=callback)
```

**SessionConfig**: `strategy_name`, `symbol`, `start_date`, `end_date`, `mode` (backtest/paper/live), `initial_capital`, `params`, `risk_enabled`

**SessionResult**: `session_id`, `status`, `metrics`, `equity_history`, `trades`, `positions`, `duration_seconds`

## Analysis & Reports

**File**: `src/analysis/backtest_metrics.py`, `src/analysis/attribution.py`

**Performance metrics**: total return, Sharpe ratio, max drawdown, win rate, profit factor, etc.

**Report generators**:
- `src/analysis/reports/generator.py` - Markdown reports
- `src/analysis/reports/excel_generator.py` - Excel export
- `src/analysis/reports/html_generator.py` - HTML reports
