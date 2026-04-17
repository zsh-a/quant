"""
Brooks 价格行为策略 — Al Brooks 交易方法论的 Quent 实现。

从 ta_graph 的分析逻辑迁移，适配 Quent Strategy 接口。

特点:
  - L0 预处理: 纯 Python bar 特征提取 (零 API 成本)
  - 死市场过滤: ATR 过低时不交易
  - Brooks 信号: 趋势 bar + EMA 关系 + 摆动点
  - 交易过滤: 冷却期 + 日上限 + Barb Wire
  - 风险管理: ATR 止损 + 最小 RR 2:1

适用市场: crypto (高流动性品种), A 股 (日线)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.base import Bar, Strategy
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register("brooks")
class BrooksStrategy(Strategy):
    """Al Brooks 价格行为策略。"""

    def __init__(self, db_client=None, session_id: Optional[str] = None, **params):
        super().__init__(session_id)
        self.db_client = db_client
        # 参数 (可通过前端配置)
        self.atr_mult = params.get("atr_multiplier", 1.5)
        self.min_body_pct = params.get("min_body_pct", 50)
        self.min_rr = params.get("min_rr", 2.0)
        self.risk_pct = params.get("risk_percent", 1.0)
        self.lookback = params.get("lookback", 20)
        self.cooldown_bars = params.get("cooldown_bars", 3)
        self.use_llm = params.get("use_llm", False)
        self.llm_model = params.get("llm_model", "gpt-4o-mini")
        self.llm_api_key = params.get("llm_api_key", "")
        self.llm_base_url = params.get("llm_base_url", "")

        # LLM 管线 (按需创建)
        self._pipeline = None
        if self.use_llm:
            from src.strategies.brooks_llm_pipeline import BrooksLLMPipeline

            self._pipeline = BrooksLLMPipeline(
                model=self.llm_model,
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
            )

        # 状态
        self._bars_history: list[dict] = []
        self._bars_since_trade = 999
        self._in_position = False
        self._entry_price = 0.0
        self._stop_loss = 0.0
        self._position_side = ""  # "long" / "short"

    @classmethod
    def get_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "atr_multiplier": {
                "type": "float",
                "default": 1.5,
                "min": 0.5,
                "max": 5.0,
                "description": "ATR 止损倍数",
            },
            "min_body_pct": {
                "type": "int",
                "default": 50,
                "min": 20,
                "max": 80,
                "description": "最小趋势 bar 实体占比 (%)",
            },
            "min_rr": {
                "type": "float",
                "default": 2.0,
                "min": 1.0,
                "max": 5.0,
                "description": "最小风险回报比",
            },
            "risk_percent": {
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 5.0,
                "description": "每笔交易风险占比 (%)",
            },
            "lookback": {
                "type": "int",
                "default": 20,
                "min": 10,
                "max": 60,
                "description": "EMA / 摆动点回看期",
            },
            "cooldown_bars": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "交易间隔最少 bar 数",
            },
            "use_llm": {
                "type": "bool",
                "default": False,
                "description": "启用 LLM AI 分析 (需要 API Key)",
            },
            "llm_model": {
                "type": "str",
                "default": "gpt-4o-mini",
                "description": "LLM 模型名称",
            },
            "llm_api_key": {
                "type": "str",
                "default": "",
                "description": "LLM API Key",
            },
            "llm_base_url": {
                "type": "str",
                "default": "",
                "description": "LLM API Base URL (可选)",
            },
        }

    def on_bar(self, bars: Dict[str, Bar]):
        from src.analysis.bar_features import extract_features

        self._bars_since_trade += 1

        for symbol, bar in bars.items():
            bar_dict = {
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            self._bars_history.append(bar_dict)

            # 保持合理长度
            if len(self._bars_history) > 200:
                self._bars_history = self._bars_history[-200:]

            if len(self._bars_history) < self.lookback + 5:
                return

            # --- 仓位管理 ---
            if self._in_position:
                self._manage_position(symbol, bar)
                return

            # --- LLM 模式: 完整 AI 管线 ---
            if self._pipeline:
                self._run_llm_pipeline(symbol, bar)
                return

            # --- 纯规则模式: L0 预处理 ---
            features, ctx = extract_features(self._bars_history, ema_period=self.lookback)
            if not features:
                return

            # 死市场过滤
            if ctx.is_dead_market:
                self._log("死鱼盘，跳过", level="DEBUG")
                return

            # 冷却期
            if self._bars_since_trade < self.cooldown_bars:
                return

            # --- 信号判断 ---
            latest = features[-1]
            features[-2] if len(features) >= 2 else latest
            atr = ctx.atr_14

            # Brooks 做多信号: 牛趋势 bar + 收盘在高位 + 价格在 EMA 上方
            if (
                latest.bar_type == "bull_trend"
                and latest.body_pct >= self.min_body_pct
                and latest.close_position == "high"
                and latest.ema_relation in ("above", "at")
                and not latest.is_inside_bar
            ):
                entry = bar.close
                sl = entry - atr * self.atr_mult
                tp = entry + abs(entry - sl) * self.min_rr

                from src.core.price_calculator import enforce_min_rr, enforce_min_stop_distance

                sl = enforce_min_stop_distance(entry, sl, atr, self.atr_mult)
                tp = enforce_min_rr(entry, sl, tp, self.min_rr)

                qty = self._calc_position_size(entry, sl)
                if qty > 0:
                    self._log(f"做多信号: {symbol} entry={entry:.4f} sl={sl:.4f} tp={tp:.4f}")
                    self.buy(symbol, qty, execution_type="IMMEDIATE_CLOSE")
                    self._enter_position(symbol, "long", entry, sl)

            # Brooks 做空信号: 熊趋势 bar + 收盘在低位 + 价格在 EMA 下方
            elif (
                latest.bar_type == "bear_trend"
                and latest.body_pct >= self.min_body_pct
                and latest.close_position == "low"
                and latest.ema_relation in ("below", "at")
                and not latest.is_inside_bar
            ):
                entry = bar.close
                sl = entry + atr * self.atr_mult
                tp = entry - abs(sl - entry) * self.min_rr

                from src.core.price_calculator import enforce_min_rr, enforce_min_stop_distance

                sl = enforce_min_stop_distance(entry, sl, atr, self.atr_mult)
                tp = enforce_min_rr(entry, sl, tp, self.min_rr)

                qty = self._calc_position_size(entry, sl)
                if qty > 0:
                    self._log(f"做空信号: {symbol} entry={entry:.4f} sl={sl:.4f} tp={tp:.4f}")
                    self.sell(symbol, qty, execution_type="IMMEDIATE_CLOSE")
                    self._enter_position(symbol, "short", entry, sl)

    def _run_llm_pipeline(self, symbol: str, bar: Bar):
        """LLM AI 管线: Brooks 分析 → 决策 → 下单。"""
        result = self._pipeline.run(
            symbol=symbol,
            bars=self._bars_history,
            timeframe="1h",
        )

        if not result.decision or result.decision.operation == "hold":
            if result.decision:
                self._log(f"AI Hold: {result.decision.rationale}", level="DEBUG")
            return

        d = result.decision
        qty = self._calc_position_size(d.entry_price, d.stop_loss)
        if qty <= 0:
            return

        self._log(
            f"AI {d.operation}: {symbol} entry={d.entry_price:.4f} sl={d.stop_loss:.4f} "
            f"tp={d.take_profit:.4f} prob={d.probability:.0f}% — {d.rationale}"
        )

        if d.operation == "buy":
            self.buy(symbol, qty, execution_type="IMMEDIATE_CLOSE")
            self._enter_position(symbol, "long", d.entry_price, d.stop_loss)
        elif d.operation == "sell":
            self.sell(symbol, qty, execution_type="IMMEDIATE_CLOSE")
            self._enter_position(symbol, "short", d.entry_price, d.stop_loss)

    def _enter_position(self, symbol: str, side: str, entry: float, sl: float):
        self._in_position = True
        self._entry_price = entry
        self._stop_loss = sl
        self._position_side = side
        self._bars_since_trade = 0

    def _manage_position(self, symbol: str, bar: Bar):
        """仓位管理: 止损/保本/出场。"""
        if self._position_side == "long":
            if bar.low <= self._stop_loss:
                self._log(f"止损出场: {symbol} sl={self._stop_loss:.4f}")
                qty = self.engine.broker.positions.get(symbol, 0)
                if qty > 0:
                    self.sell(symbol, qty, execution_type="IMMEDIATE_CLOSE")
                self._reset_position()
                return

            # 保本: 盈利 >= 1R
            risk = self._entry_price - self._stop_loss
            if bar.close >= self._entry_price + risk and self._stop_loss < self._entry_price:
                self._stop_loss = self._entry_price
                self._log(f"移至保本: {symbol}")

        elif self._position_side == "short":
            if bar.high >= self._stop_loss:
                self._log(f"止损出场: {symbol} sl={self._stop_loss:.4f}")
                qty = abs(self.engine.broker.positions.get(symbol, 0))
                if qty > 0:
                    self.buy(symbol, qty, execution_type="IMMEDIATE_CLOSE")
                self._reset_position()
                return

            risk = self._stop_loss - self._entry_price
            if bar.close <= self._entry_price - risk and self._stop_loss > self._entry_price:
                self._stop_loss = self._entry_price
                self._log(f"移至保本: {symbol}")

    def _reset_position(self):
        self._in_position = False
        self._entry_price = 0.0
        self._stop_loss = 0.0
        self._position_side = ""

    def _calc_position_size(self, entry: float, sl: float) -> float:
        """基于风险百分比计算仓位大小。"""
        if not self.engine:
            return 0
        account = self.engine.broker.get_account_info()
        equity = account.get("total_equity", 0)
        if equity <= 0:
            return 0
        risk_usd = equity * (self.risk_pct / 100.0)
        risk_per_unit = abs(entry - sl)
        if risk_per_unit <= 0:
            return 0
        qty = risk_usd / risk_per_unit
        # 对 A 股取整到 100 的倍数
        if hasattr(self.engine.broker, "positions"):
            for sym in self.engine.broker.positions:
                if sym.startswith("sh.") or sym.startswith("sz."):
                    qty = int(qty / 100) * 100
                    break
        return max(0, qty)
