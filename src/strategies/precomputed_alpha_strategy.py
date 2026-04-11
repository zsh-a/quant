"""
预计算权重的多因子Alpha策略

接收由 AlphaService 预计算的目标权重表 (weight_map)，
在 event engine 中按调仓频率执行买卖操作。

核心流程:
  1. Alpha模块: 因子组合 → 组合信号 → SignalTransformer → 目标权重 (T×N)
  2. 权重矩阵转为 {日期: {symbol: weight}} 的 weight_map
  3. 本策略在 on_bar 中查找当前日期的目标权重，执行调仓
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from src.core.base import Bar, Strategy
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register(
    name="precomputed_alpha",
    label="多因子Alpha策略",
    description="基于Alpha因子组合的预计算权重策略，支持定期调仓",
)
class PrecomputedAlphaStrategy(Strategy):
    """
    Event engine 策略: 基于预计算的目标权重进行定期调仓。

    weight_map 中每个日期对应该日的目标持仓权重,
    策略按 rebalance_interval 的频率查找最近的权重并执行调仓。
    """

    def __init__(self, db_client, session_id: str = None, **kwargs):
        super().__init__(session_id=session_id)
        self.db_client = db_client

        params = self.get_parameters()
        self.weight_map: Dict[str, Dict[str, float]] = kwargs.get("weight_map", {})
        self.top_n = int(kwargs.get("top_n", params["top_n"]["default"]))
        self.rebalance_interval = int(kwargs.get("rebalance_interval", params["rebalance_interval"]["default"]))
        self.position_method = str(kwargs.get("position_method", params["position_method"]["default"]))
        self.cash_reserve_pct = float(kwargs.get("cash_reserve_pct", params["cash_reserve_pct"]["default"]))

        # 排序的权重日期列表，用于快速查找最近权重
        self._weight_dates = sorted(self.weight_map.keys())
        self._bar_count = 0
        self._last_rebalance_bar = -self.rebalance_interval  # 确保首次可以调仓

        self._log(
            f"PrecomputedAlphaStrategy initialized: "
            f"weight_dates={len(self._weight_dates)}, top_n={self.top_n}, "
            f"rebalance_interval={self.rebalance_interval}, "
            f"position_method={self.position_method}"
        )

    @classmethod
    def get_parameters(cls) -> dict:
        return {
            "top_n": {
                "type": "int",
                "default": 10,
                "description": "持仓股票/标的数量",
                "min": 1,
                "max": 100,
            },
            "rebalance_interval": {
                "type": "int",
                "default": 5,
                "description": "调仓间隔 (bars)",
                "min": 1,
                "max": 60,
            },
            "position_method": {
                "type": "str",
                "default": "long_only",
                "description": "持仓方式",
                "options": ["long_only", "long_short"],
            },
            "cash_reserve_pct": {
                "type": "float",
                "default": 0.05,
                "description": "现金保留比例",
                "min": 0.0,
                "max": 0.5,
            },
        }

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def on_bar(self, bars: Dict[str, Bar]):
        if not bars:
            return

        self._update_current_date(bars)
        self._bar_count += 1

        # 判断是否到达调仓日
        if self._bar_count - self._last_rebalance_bar < self.rebalance_interval:
            return

        today_str = self._current_date
        if not today_str:
            return

        # 查找当前日期对应的目标权重
        target_weights = self._lookup_weights(today_str)
        if target_weights is None:
            return

        self._last_rebalance_bar = self._bar_count
        self._execute_rebalance(target_weights, bars)

    # ------------------------------------------------------------------
    # Weight lookup
    # ------------------------------------------------------------------

    def _lookup_weights(self, date_str: str) -> Optional[Dict[str, float]]:
        """查找日期对应的权重，精确匹配或最近的历史权重。"""
        if date_str in self.weight_map:
            return self.weight_map[date_str]

        # 二分查找最近的历史日期
        import bisect
        idx = bisect.bisect_right(self._weight_dates, date_str)
        if idx == 0:
            return None  # 还没有可用的权重
        nearest_date = self._weight_dates[idx - 1]
        return self.weight_map[nearest_date]

    # ------------------------------------------------------------------
    # Rebalance execution
    # ------------------------------------------------------------------

    def _execute_rebalance(self, target_weights: Dict[str, float], bars: Dict[str, Bar]):
        """根据目标权重执行调仓。"""
        account = self.engine.broker.get_account_info()
        total_equity = account["total_equity"]
        current_positions = dict(account["positions"])

        if self.position_method == "long_only":
            target_portfolio = self._build_long_only_portfolio(target_weights, total_equity, bars)
        else:
            target_portfolio = self._build_weighted_portfolio(target_weights, total_equity, bars)

        if not target_portfolio and not current_positions:
            return

        self._log(f"========== 多因子调仓 ==========")
        self._log(f"总权益: {total_equity:,.0f}, 当前持仓: {len(current_positions)}, 目标持仓: {len(target_portfolio)}")

        submitted = []

        # 先卖出: 不在目标中的持仓 + 需要减仓的持仓
        for symbol, qty in current_positions.items():
            if qty <= 0:
                continue
            target_qty = target_portfolio.get(symbol, 0)
            if target_qty < qty:
                sell_qty = qty - target_qty
                if sell_qty > 0:
                    action = "清仓" if target_qty == 0 else "减仓"
                    self._log(f"{action} {symbol}: {sell_qty}股")
                    self.sell(symbol, sell_qty)
                    submitted.append(f"卖出 {symbol} x{sell_qty}")

        # 再买入: 新增持仓 + 需要加仓的持仓
        for symbol, target_qty in target_portfolio.items():
            if target_qty <= 0:
                continue
            current_qty = current_positions.get(symbol, 0)
            if target_qty > current_qty:
                buy_qty = target_qty - current_qty
                if buy_qty > 0:
                    action = "建仓" if current_qty == 0 else "加仓"
                    self._log(f"{action} {symbol}: {buy_qty}股")
                    self.buy(symbol, buy_qty)
                    submitted.append(f"买入 {symbol} x{buy_qty}")

        if submitted:
            self._log(f"提交NEXT_OPEN订单({len(submitted)}笔)")

    def _build_long_only_portfolio(
        self,
        weights: Dict[str, float],
        total_equity: float,
        bars: Dict[str, Bar],
    ) -> Dict[str, int]:
        """
        Long-only: 选取权重最大的 top_n 个 symbol，等权分配。
        返回 {symbol: target_quantity}。
        """
        # 按权重降序排列，选 top_n
        sorted_symbols = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        # 只选正权重
        selected = [(s, w) for s, w in sorted_symbols if w > 0][:self.top_n]

        if not selected:
            return {}

        # 等权分配可用资金
        available = total_equity * (1.0 - self.cash_reserve_pct)
        per_stock = available / len(selected)

        portfolio: Dict[str, int] = {}
        for symbol, _weight in selected:
            bar = bars.get(symbol)
            if bar is None or bar.close <= 0:
                continue
            # A股整手: 100股整数倍
            qty = int(per_stock / bar.close // 100 * 100)
            if qty > 0:
                portfolio[symbol] = qty

        return portfolio

    def _build_weighted_portfolio(
        self,
        weights: Dict[str, float],
        total_equity: float,
        bars: Dict[str, Bar],
    ) -> Dict[str, int]:
        """
        按权重比例分配 (支持 long-short 概念，但 A股只做多)。
        返回 {symbol: target_quantity}。
        """
        # 只保留正权重 (A股不支持做空)
        positive = {s: w for s, w in weights.items() if w > 0}
        if not positive:
            return {}

        # 按权重排序取 top_n
        sorted_symbols = sorted(positive.items(), key=lambda x: x[1], reverse=True)[:self.top_n]

        # 权重归一化
        total_weight = sum(w for _, w in sorted_symbols)
        if total_weight <= 0:
            return {}

        available = total_equity * (1.0 - self.cash_reserve_pct)

        portfolio: Dict[str, int] = {}
        for symbol, weight in sorted_symbols:
            bar = bars.get(symbol)
            if bar is None or bar.close <= 0:
                continue
            allocation = available * (weight / total_weight)
            qty = int(allocation / bar.close // 100 * 100)
            if qty > 0:
                portfolio[symbol] = qty

        return portfolio
