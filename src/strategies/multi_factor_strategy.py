"""
Multi-Factor Alpha Strategy

Uses alpha factors discovered by the MCTS alpha mining pipeline.
Computes a composite score from multiple factors, ranks the universe,
and periodically rebalances into the top-N stocks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from src.core.base import Strategy, Bar
from src.core.trading_calendar import TradingCalendar
from src.strategies.registry import StrategyRegistry
from src.alpha.infra.persistence import AlphaPersistence as AlphaZooPersistence
from src.alpha.core.operators import OperatorRegistry as _OperatorRegistry
from src.alpha.core.compiler import FormulaCompiler as _FormulaCompiler
from src.alpha.core.dsl import TensorSchema as _TensorSchema
from src.alpha.core.vm import StackVM as _StackVM, TensorStore as _TensorStore
from loguru import logger


# --------------- Factor computation helpers ---------------

_registry = _OperatorRegistry()
_compiler = _FormulaCompiler(_registry)
_vm = _StackVM(prefer_torch=False)
_stock_schema = _TensorSchema.default_stock_schema()


def _compute_factor(formula: str, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Evaluate a formula string via the compiled DSL/VM pipeline.
    Returns a Date × Symbol matrix of factor values.
    """
    dates = price_data["close"].index
    symbols = price_data["close"].columns

    fields: Dict[str, np.ndarray] = {}
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in price_data:
            fields[col] = price_data[col].to_numpy(dtype=float)

    if "amount" in fields and "volume" in fields:
        fields["vwap"] = fields["amount"] / (fields["volume"] + 1e-9)

    store = _TensorStore(fields)
    program = _compiler.compile(formula, _stock_schema)
    result = _vm.run(program, store)
    result_np = np.asarray(result, dtype=float)
    result_np = np.where(np.isinf(result_np), np.nan, result_np)
    return pd.DataFrame(result_np, index=dates, columns=symbols)


def _rank_cross_section(factor: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank across symbols for each date row."""
    return factor.rank(axis=1, pct=True)


# --------------- Strategy ---------------

@StrategyRegistry.register(
    name="multi_factor",
    label="Multi-Factor Alpha",
    description="多因子Alpha策略 - 基于MCTS挖掘的因子组合进行周度轮动选股",
)
class MultiFactorStrategy(Strategy):
    """
    Weekly rebalancing strategy driven by alpha factors from the alpha zoo.

    Flow (executed once per rebalance period):
      1. Pull trailing price history for the universe.
      2. Evaluate each factor formula → Date×Symbol matrix.
      3. Cross-sectionally rank each factor on the latest date.
      4. Compute a weighted composite score.
      5. Select top_n stocks and equal-weight rebalance.
    """

    def __init__(self, db_client, session_id: str = None, **kwargs):
        super().__init__(session_id=session_id)
        self.db_client = db_client
        self.calendar = TradingCalendar(db_client)

        params = self.get_parameters()
        self.top_n = int(kwargs.get("top_n", params["top_n"]["default"]))
        self.lookback = int(kwargs.get("lookback", params["lookback"]["default"]))
        self.rebalance_freq = str(kwargs.get("rebalance_freq", params["rebalance_freq"]["default"]))
        self.index_code = str(kwargs.get("index_code", params["index_code"]["default"]))
        zoo_dir = str(kwargs.get("zoo_dir", params["zoo_dir"]["default"]))

        self.factors = self._load_factors(zoo_dir)

        # State
        self._last_rebalance_date: Optional[str] = None
        self._bar_count = 0

        self._log(
            f"MultiFactorStrategy initialized: top_n={self.top_n}, "
            f"lookback={self.lookback}, rebalance={self.rebalance_freq}, "
            f"factors={len(self.factors)}"
        )
        for i, f in enumerate(self.factors):
            ic = f['metrics'].get('rank_ic', 0)
            direction = "正向" if ic > 0 else "反向"
            self._log(f"  Factor {i+1}: IC={ic:+.4f} ({direction})  {f['formula'][:60]}")

    # ---- Parameters ----

    @classmethod
    def get_parameters(cls) -> dict:
        return {
            "top_n": {
                "type": "int",
                "default": 10,
                "description": "持仓股票数量",
                "min": 3,
                "max": 50,
            },
            "lookback": {
                "type": "int",
                "default": 300,
                "description": "因子计算所需的历史天数",
                "min": 60,
                "max": 500,
            },
            "rebalance_freq": {
                "type": "str",
                "default": "weekly",
                "description": "调仓频率",
                "options": ["weekly", "biweekly", "monthly"],
            },
            "index_code": {
                "type": "str",
                "default": "000852",
                "description": "选股宇宙 (指数代码)",
                "options": ["000852", "000905", "399101"],
            },
            "zoo_dir": {
                "type": "str",
                "default": "data/alpha_zoo",
                "description": "Alpha因子库目录",
            },
        }

    # ---- Factor loading ----

    def _load_factors(self, zoo_dir: str) -> List[Dict[str, Any]]:
        """Load and filter factors from the alpha zoo."""
        zoo = AlphaZooPersistence(storage_dir=zoo_dir)
        all_factors = zoo.load_all()

        if not all_factors:
            # Fallback: use hardcoded default factors
            logger.warning("Alpha zoo is empty, using built-in default factors")
            all_factors = [
                {"formula": "CSRank(-Ts_Returns(close, 5))", "metrics": {"rank_ic": 0.03}},
                {"formula": "CSRank(Correlation(close, volume, 10))", "metrics": {"rank_ic": -0.03}},
                {"formula": "CSRank(Ts_Rank(volume, 20))", "metrics": {"rank_ic": 0.02}},
            ]

        # Keep top factors by absolute IC
        factors = sorted(all_factors, key=lambda x: abs(x["metrics"].get("rank_ic", 0)), reverse=True)
        return factors[:10]  # Max 10 factors

    # ---- Weight derivation ----

    def _get_factor_weights(self) -> List[float]:
        """IC-weighted factor combination. IC sign encodes factor direction."""
        ics = [f["metrics"].get("rank_ic", 0) for f in self.factors]
        abs_sum = sum(abs(ic) for ic in ics) or 1.0
        return [ic / abs_sum for ic in ics]

    # ---- Rebalance logic ----

    def _should_rebalance(self, today_str: str) -> bool:
        """Check if today is a rebalance date."""
        return self.calendar.is_rebalance_day(
            today_str,
            freq=self.rebalance_freq,
            last_rebalance=self._last_rebalance_date,
        )

    def _compute_composite_score(self, stocks: List[str], date_str: str) -> pd.Series:
        """
        Compute a composite factor score for each stock.
        Returns a Series indexed by stock symbol.
        """
        # 1. Fetch price history (lookback days for all stocks)
        price_df = self.db_client.get_price(
            stocks, date_str,
            fields=["open", "high", "low", "close", "volume", "amount"],
            count=self.lookback,
        )

        if price_df.empty:
            self._log("No price data available", level="WARNING")
            return pd.Series(dtype=float)

        # 2. Build pivot matrices (Date × Symbol)
        price_df = price_df.reset_index()
        price_df.rename(columns={"code": "symbol"}, inplace=True)
        price_df["date"] = pd.to_datetime(price_df["date"])

        price_data: Dict[str, pd.DataFrame] = {}
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in price_df.columns:
                price_data[col] = price_df.pivot(index="date", columns="symbol", values=col)

        # 3. Evaluate each factor and rank
        weights = self._get_factor_weights()
        composite = None

        for i, factor_info in enumerate(self.factors):
            formula = factor_info["formula"]
            w = weights[i]
            try:
                factor_matrix = _compute_factor(formula, price_data)
                # Take the latest row (today's factor value)
                latest = factor_matrix.iloc[-1]
                ranked = latest.rank(pct=True)
                # w carries IC sign: positive IC → positive w → high rank is good
                #                     negative IC → negative w → high rank is bad
                # This is correct because the IC sign already encodes whether
                # high factor values predict high or low future returns.

                if composite is None:
                    composite = ranked * w
                else:
                    composite = composite.add(ranked * w, fill_value=0)

            except Exception as e:
                self._log(f"Factor {i+1} evaluation failed: {e}", level="WARNING")
                continue

        if composite is None:
            return pd.Series(dtype=float)

        return composite.dropna().sort_values(ascending=False)

    # ---- Core on_bar ----

    def on_bar(self, bars: Dict[str, Bar]):
        if not bars:
            return

        today_str = self._current_date
        self._bar_count += 1

        if not self._should_rebalance(today_str):
            return

        self._log("========== 多因子调仓日 ==========")

        try:
            # 1. Get universe
            stocks = self.db_client.get_index_stocks(self.index_code, today_str)
            if not stocks:
                stocks = self.db_client.get_index_stocks(f"sh.{self.index_code}", today_str)
            if not stocks:
                self._log("Failed to fetch universe stocks", level="ERROR")
                return

            self._log(f"Universe: {len(stocks)} stocks from index {self.index_code}")

            # 2. Compute composite score
            scores = self._compute_composite_score(stocks, today_str)
            if scores.empty:
                self._log("Composite score is empty, skipping rebalance", level="WARNING")
                return

            # 3. Select top_n
            target_stocks = scores.head(self.top_n).index.tolist()
            self._log(f"Top {self.top_n} stocks selected")
            for i, s in enumerate(target_stocks[:5]):
                self._log(f"  {i+1}. {s}  score={scores[s]:.4f}")
            if len(target_stocks) > 5:
                self._log(f"  ... and {len(target_stocks) - 5} more")

            # 4. Rebalance
            self._rebalance(target_stocks, today_str)
            self._last_rebalance_date = today_str
        except Exception as e:
            self._log(f"调仓失败: {e}", level="ERROR")

    # ---- Rebalance execution ----

    def _rebalance(self, target_stocks: List[str], date_str: str):
        """Equal-weight rebalance into target stocks."""
        account = self.engine.broker.get_account_info()
        hold_list = list(account["positions"].keys())

        submitted_orders = []

        # Sell stocks not in target
        for stock in hold_list:
            if stock not in target_stocks:
                qty = account["positions"][stock]
                if qty > 0:
                    self._log(f"清仓 {stock}: qty={qty}", stock=stock)
                    self.sell(stock, qty)
                    submitted_orders.append(f"卖出 {stock} x{qty}")

        # Buy target stocks with equal weight
        total_equity = account["total_equity"]
        if not target_stocks:
            if submitted_orders:
                self._log(f"提交NEXT_OPEN订单({len(submitted_orders)}笔): {submitted_orders}")
            return

        val_per_stock = (total_equity * 0.95) / len(target_stocks)

        for code in target_stocks:
            price_df = self.db_client.get_price(code, date_str, ["close"], 1)
            if price_df.empty:
                self._log(f"跳过 {code}: 无价格数据", level="WARNING")
                continue

            price = price_df.iloc[0]["close"]
            if price <= 0:
                continue

            target_qty = int(val_per_stock / price // 100 * 100)
            curr_qty = account["positions"].get(code, 0)
            delta = target_qty - curr_qty

            if delta > 0:
                self._log(f"买入 {code}: {delta}股 @ {price:.2f}")
                self.buy(code, delta)
                submitted_orders.append(f"买入 {code} x{delta} @{price:.2f}")
            elif delta < 0:
                self._log(f"卖出 {code}: {abs(delta)}股 @ {price:.2f}")
                self.sell(code, abs(delta))
                submitted_orders.append(f"卖出 {code} x{abs(delta)} @{price:.2f}")

        if submitted_orders:
            self._log(f"提交NEXT_OPEN订单({len(submitted_orders)}笔): {submitted_orders}")
