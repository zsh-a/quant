import pandas as pd
import numpy as np
import talib as ta
from src.core.base import Strategy, Bar
from src.core.trading_calendar import TradingCalendar
import utils.utils as util
from src.strategies.registry import StrategyRegistry

PRICE_CHANGE_LIMIT = 0.098
NUM_STOCKS = 6


@StrategyRegistry.register(
    name="jsg",
    label="JSG Quantitative",
    description="JSG量化策略 - 基于行业轮动策略",
    requires_symbol=False,
)
class JSGStrategy(Strategy):

    def __init__(self, db_client, session_id: str = None, **kwargs):
        super().__init__(session_id=session_id)
        self.db_client = db_client
        self.calendar = TradingCalendar(db_client)

        params = self.get_parameters()
        self.max_stocks = int(kwargs.get("max_stocks", params["max_stocks"]["default"]))
        self.pool_size = int(kwargs.get("pool_size", params["pool_size"]["default"]))
        self.stock_sum = int(kwargs.get("stock_sum", params["stock_sum"]["default"]))
        self.stop_loss_pct = float(kwargs.get("stop_loss_pct", params["stop_loss_pct"]["default"]))
        self.trailing_stop_pct = float(kwargs.get("trailing_stop_pct", params["trailing_stop_pct"]["default"]))
        self.max_drawdown_pct = float(kwargs.get("max_drawdown_pct", params["max_drawdown_pct"]["default"]))

        self.black_industry_name = {"银行", "煤炭", "有色金属", "钢铁"}
        self.pass_month = []
        
        # Track stocks that hit limit-up yesterday
        self.prev_limit_up_stocks = set()

        # Risk control state
        self._peak_equity = 0.0  # 组合净值峰值
        self._trailing_highs = {}  # symbol -> 持仓期间最高价
        self._drawdown_triggered = False  # 回撤熔断标志

        self._log(
            f"JSGStrategy initialized: max_stocks={self.max_stocks}, pool_size={self.pool_size}, stock_sum={self.stock_sum}, "
            f"stop_loss={self.stop_loss_pct:.0%}, trailing_stop={self.trailing_stop_pct:.0%}, max_dd={self.max_drawdown_pct:.0%}"
        )

    @classmethod
    def get_parameters(cls) -> dict:
        return {
            "max_stocks": {
                "type": "int",
                "default": NUM_STOCKS,
                "description": "Maximum number of stocks to consider from index",
                "min": 1,
                "max": 100,
            },
            "pool_size": {
                "type": "int",
                "default": 20,
                "description": "Size of the candidate pool after financial filtering",
                "min": 5,
                "max": 100,
            },
            "stock_sum": {
                "type": "int",
                "default": 10,
                "description": "Maximum number of stocks to hold in portfolio",
                "min": 1,
                "max": 20,
            },
            "stop_loss_pct": {
                "type": "float",
                "default": 0,
                "description": "个股止损比例 (0=禁用, 0.08=跌8%止损)",
                "min": 0.0,
                "max": 0.30,
            },
            "trailing_stop_pct": {
                "type": "float",
                "default": 0,
                "description": "个股移动止盈比例 (0=禁用, 0.10=从最高点回落10%卖出)",
                "min": 0.0,
                "max": 0.30,
            },
            "max_drawdown_pct": {
                "type": "float",
                "default": 0,
                "description": "组合最大回撤熔断 (0=禁用, 0.15=回撤15%清仓)",
                "min": 0.0,
                "max": 0.50,
            },
        }

    def _is_limit_up(self, symbol: str, bar: Bar) -> bool:
        """Check if a stock is at limit-up price."""
        preclose = bar.extra.get('preclose', 0)
        if preclose <= 0:
            return False
            
        limit_pct = 0.10
        if bar.extra.get('isst') == 1:
            limit_pct = 0.05
        elif symbol.startswith('sh.68') or symbol.startswith('sz.30'):
            limit_pct = 0.20
            
        up_limit = round(preclose * (1 + limit_pct) + 0.0001, 2)
        return bar.close >= up_limit

    def _check_risk_controls(self, bars: dict[str, Bar]) -> bool:
        """每日风控检查。返回 True 表示触发了组合熔断，应跳过买入。"""
        account = self.engine.broker.get_account_info()
        total_equity = account["total_equity"]
        detailed = account.get("detailed_positions", {})

        # --- 组合回撤熔断 ---
        if self._peak_equity <= 0:
            self._peak_equity = total_equity
        if total_equity > self._peak_equity:
            self._peak_equity = total_equity

        if self.max_drawdown_pct > 0 and self._peak_equity > 0:
            drawdown = (self._peak_equity - total_equity) / self._peak_equity
            if drawdown >= self.max_drawdown_pct:
                if not self._drawdown_triggered:
                    self._drawdown_triggered = True
                    self._log(
                        f"⚠ 组合回撤熔断: 回撤={drawdown:.2%} >= 阈值={self.max_drawdown_pct:.0%}, "
                        f"峰值={self._peak_equity:.0f}, 当前={total_equity:.0f}, 全部清仓",
                        level="WARNING",
                    )
                    # 清仓所有持仓
                    for symbol, qty in list(account["positions"].items()):
                        if qty > 0:
                            self.sell(symbol, qty, execution_type="IMMEDIATE_CLOSE")
                    self._trailing_highs.clear()
                return True

        # 回撤恢复：在调仓日重置熔断标志 (由 on_bar 中调仓逻辑处理)

        # --- 个股止损 & 移动止盈 ---
        for symbol, info in detailed.items():
            qty = account["positions"].get(symbol, 0)
            if qty <= 0:
                continue

            current_price = info.get("price", 0)
            if current_price <= 0:
                continue

            # 更新移动止盈最高价
            if symbol not in self._trailing_highs:
                self._trailing_highs[symbol] = current_price
            elif current_price > self._trailing_highs[symbol]:
                self._trailing_highs[symbol] = current_price

            # 个股固定止损：基于成本价
            if self.stop_loss_pct > 0:
                pnl_pct = info.get("pnl_pct", 0)
                if pnl_pct <= -self.stop_loss_pct:
                    self._log(
                        f"✂ 止损卖出: {symbol} 亏损={pnl_pct:.2%} <= -{self.stop_loss_pct:.0%}, 收盘清仓",
                        level="WARNING",
                    )
                    self.sell(symbol, qty, execution_type="IMMEDIATE_CLOSE")
                    self._trailing_highs.pop(symbol, None)
                    continue

            # 个股移动止盈：从最高价回落超过阈值
            if self.trailing_stop_pct > 0:
                peak = self._trailing_highs.get(symbol, current_price)
                drop_from_peak = (peak - current_price) / peak if peak > 0 else 0
                if drop_from_peak >= self.trailing_stop_pct:
                    self._log(
                        f"✂ 移动止盈: {symbol} 从高点={peak:.2f}回落={drop_from_peak:.2%} >= {self.trailing_stop_pct:.0%}, 收盘清仓",
                        level="WARNING",
                    )
                    self.sell(symbol, qty, execution_type="IMMEDIATE_CLOSE")
                    self._trailing_highs.pop(symbol, None)
                    continue

        return False

    def on_bar(self, bars: dict[str, Bar]):
        if not bars:
            return

        today_str = self._current_date  # Auto-updated by engine

        # 1. Daily check for "limit-up break" sell rule
        account = self.engine.broker.get_account_info()
        hold_list = list(account["positions"].keys())

        current_limit_up_stocks = set()

        for symbol in bars:
            bar = bars[symbol]
            if self._is_limit_up(symbol, bar):
                current_limit_up_stocks.add(symbol)

        # Sell if it was limit-up yesterday but not today
        for stock in hold_list:
            if stock in self.prev_limit_up_stocks and stock not in current_limit_up_stocks:
                qty = account["positions"][stock]
                if qty > 0:
                    self._log(f"涨停打开: {stock} 昨日涨停, 今日未涨停, 收盘卖出", stock=stock)
                    self.sell(stock, qty, execution_type="IMMEDIATE_CLOSE")

        # Update state for tomorrow
        self.prev_limit_up_stocks = current_limit_up_stocks

        # 2. Daily risk control checks (stop-loss, trailing stop, drawdown)
        drawdown_triggered = self._check_risk_controls(bars)

        # 3. Rebalance Check (每周最后一个交易日)
        if not self.calendar.is_rebalance_day(today_str, freq="weekly"):
            return

        # 调仓日重置回撤熔断标志，允许重新建仓
        if self._drawdown_triggered:
            self._log("调仓日: 重置回撤熔断标志")
            self._drawdown_triggered = False
            self._peak_equity = self.engine.broker.get_account_info()["total_equity"]

        # 如果当日已触发熔断清仓，跳过买入
        if drawdown_triggered:
            self._log("调仓日: 回撤熔断中, 跳过建仓", level="WARNING")
            return

        self._log("========== 调仓日 ==========")

        try:
            # 1. Broad market check
            I = self.get_market_breadth(today_str)
            self._log(f"市场宽度(热门行业): {I}", breadth=I)

            # 2. Select stocks
            cand_stocks = self.stock_decider(I, today_str)
            self._log(f"候选股票: {cand_stocks}", candidates=len(cand_stocks))

            # 3. Adjust positions
            self.adjust(cand_stocks, today_str)
        except Exception as e:
            self._log(f"调仓失败: {e}", level="ERROR")

    def get_market_breadth(self, end_date):
        stocks = self.db_client.get_index_stocks("000985", end_date)
        df = self.db_client.get_price(stocks, end_date, ["close"], 20)
        if len(df) == 0:
            return []

        df["ma20"] = df.groupby(level="code")["close"].transform(
            lambda x: ta.MA(x, timeperiod=20)
        )
        df.dropna(inplace=True)
        df["bias"] = df["close"] > df["ma20"]

        if not df.empty and "date" in (
            df.index.names if hasattr(df.index, "names") else []
        ):
            df.reset_index(level="date", drop=True, inplace=True)

        industry_df = self.db_client.get_stock_industry_sw(df.index.to_list(), end_date)
        df["industry"] = industry_df["industry_name"]
        df = df[(df["industry"] != "")]
        df = df[["bias", "industry"]]

        ratio = (df.groupby("industry")["bias"].mean() * 100).round()
        return ratio.nlargest(1).index.to_list()

    def stock_decider(self, I, date_str):
        if not self.black_industry_name.intersection(set(I)):
            return self.select_stock(date_str)
        return []

    def filter_basic(self, stocks, date_str):
        df = self.db_client.get_price(stocks, date_str, ["isST"], 1)
        if not df.empty and "date" in (
            df.index.names if hasattr(df.index, "names") else []
        ):
            df.reset_index(level="date", drop=True, inplace=True)
        # Assuming tradestatus is also in the fields if needed,
        # but the original logic only asked for isST
        df = df[(df["isST"] == 0)]
        return df.index.to_list()

    def select_stock(self, date_str):
        stocks = self.db_client.get_index_stocks("399101", date_str)
        stocks = self.filter_basic(stocks, date_str)

        # Calculate fin_date as in original logic
        ts = pd.to_datetime(date_str)
        # This part might need the next trading day logic which is tricky in a standalone way
        # For now, we use a slightly simplified date for financial data
        fin_date = (ts - pd.DateOffset(days=1)).date()

        fin_db = self.db_client.get_stock_fincial(
            stocks, fields=["adjusted_profit_diff", "total_shares"], date=str(fin_date)
        )
        fin_db = fin_db[fin_db["adjusted_profit_diff"] > 0]

        df = self.db_client.get_price(
            fin_db.index.to_list(), date_str, ["close"], 1, price_adj=False
        )
        if not df.empty and "date" in (
            df.index.names if hasattr(df.index, "names") else []
        ):
            df.reset_index(level="date", drop=True, inplace=True)

        fin_db["close"] = df["close"]
        share_info = self.db_client.get_stock_shares_info(
            fin_db.index.to_list(), str(fin_date)
        )
        fin_db["total_shares"] = share_info["total_shares"]
        fin_db["market_cap"] = fin_db["close"] * fin_db["total_shares"]

        fin_dbContent = fin_db.sort_values(by="market_cap", ascending=True).iloc[
            : self.pool_size
        ]
        return fin_dbContent.index.to_list()

    def adjust(self, target_stocks, date_str):
        target = target_stocks[: min(len(target_stocks), self.stock_sum)]
        account = self.engine.broker.get_account_info()
        hold_list = list(account["positions"].keys())

        self._log(f"调仓: 目标={target}, 当前持仓={hold_list}")

        submitted_orders = []  # Track orders submitted during rebalance

        # Sell stocks not in target (full liquidation)
        for stock in hold_list:
            if stock not in target:
                qty = account["positions"][stock]
                self._log(
                    f"清仓 {stock}: qty={qty} (不在目标中)",
                    stock=stock,
                    action="sell_all",
                )
                if qty > 0:
                    order_id = self.sell(stock, qty)
                    submitted_orders.append(f"卖出 {stock} x{qty}")
                    self._trailing_highs.pop(stock, None)

        # Buy target stocks (or adjust position size)
        total_equity = account["total_equity"]
        if target:
            val_per_stock = (total_equity * 0.95) / len(target)  # 5% cash buffer
            self._log(
                f"权益={total_equity:.2f}, 每股分配={val_per_stock:.2f}",
                equity=total_equity,
            )

            for code in target:
                price_df = self.db_client.get_price(code, date_str, ["close"], 1)
                if price_df.empty:
                    self._log(f"跳过 {code}: 无价格数据", level="WARNING", stock=code)
                    continue
                price = price_df.iloc[0]["close"]
                target_qty = int(val_per_stock / price // 100 * 100)  # 100 share lot

                curr_qty = account["positions"].get(code, 0)
                delta = target_qty - curr_qty
                action = "买入" if delta > 0 else "卖出" if delta < 0 else "持有"
                self._log(
                    f"{action} {code}: 当前={curr_qty}, 目标={target_qty}, 差额={delta}, 价格={price:.2f}",
                    stock=code,
                    curr_qty=curr_qty,
                    target_qty=target_qty,
                    delta=delta,
                )
                if delta > 0:
                    order_id = self.buy(code, delta)
                    submitted_orders.append(f"买入 {code} x{delta} @{price:.2f}")
                elif delta < 0:
                    order_id = self.sell(code, abs(delta))
                    submitted_orders.append(f"卖出 {code} x{abs(delta)} @{price:.2f}")

        if submitted_orders:
            self._log(f"提交NEXT_OPEN订单({len(submitted_orders)}笔): {submitted_orders}")
