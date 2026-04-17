"""
Brooks LLM 交易管线 — VLM 图表分析 + AI 决策 + 风险检查。

不依赖 LangGraph，用轻量函数管线实现:
  market_data → L0 gate → Brooks VLM 分析 → AI 决策 → 风险检查 → 执行

状态持久化通过 Quent 现有的 session_db + checkpoint 系统实现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Decision Models (从 ta_graph 精简迁移)
# ---------------------------------------------------------------------------


class BrooksSignalBar(BaseModel):
    quality_score: int = Field(ge=0, le=10)
    bar_type: Literal["strong_bull", "weak_bull", "doji", "weak_bear", "strong_bear"]
    body_size_percent: float = 0
    closes_near: Literal["high", "mid", "low"] = "mid"


class BrooksAnalysisResult(BaseModel):
    """VLM Brooks 分析结果。"""

    market_cycle: Literal[
        "strong_bull_trend",
        "weak_bull_trend",
        "strong_bear_trend",
        "weak_bear_trend",
        "trading_range",
        "breakout_mode",
        "climax",
    ]
    always_in_direction: Literal["long", "short", "neutral"]
    signal_bar: BrooksSignalBar
    buying_pressure: int = Field(ge=0, le=10, default=5)
    selling_pressure: int = Field(ge=0, le=10, default=5)
    recommended_action: Literal["buy_setup", "sell_setup", "wait"]
    setup_quality: int = Field(ge=0, le=10, default=0)
    wait_reason: str | None = None
    context_summary: str = ""


class TradeDecision(BaseModel):
    """AI 交易决策。"""

    operation: Literal["buy", "sell", "hold"]
    symbol: str
    probability: float = 0.0
    rationale: str = ""
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_percent: float = 1.0


# ---------------------------------------------------------------------------
# Pipeline State
# ---------------------------------------------------------------------------


@dataclass
class PipelineState:
    """管线执行状态 — 通过各节点传递。"""

    symbol: str = ""
    timeframe: str = "1h"
    bars: list[dict] = field(default_factory=list)
    current_price: float = 0.0

    # L0 特征
    is_dead_market: bool = False
    bar_features: list[Any] = field(default_factory=list)
    market_context: Any = None

    # Brooks 分析
    brooks_analysis: BrooksAnalysisResult | None = None

    # 决策
    decision: TradeDecision | None = None

    # 执行
    executed: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Pipeline Nodes
# ---------------------------------------------------------------------------


def node_l0_gate(state: PipelineState) -> PipelineState:
    """L0 预处理 + 死市场过滤（零 API 成本）。"""
    from src.analysis.bar_features import extract_features

    if len(state.bars) < 25:
        state.is_dead_market = True
        state.error = "数据不足"
        return state

    features, ctx = extract_features(state.bars)
    state.bar_features = features
    state.market_context = ctx
    state.is_dead_market = ctx.is_dead_market
    state.current_price = state.bars[-1]["close"]

    if ctx.is_dead_market:
        logger.info("L0 gate: 死鱼盘 (ATR {:.4f}%), 跳过 AI 分析", ctx.atr_pct * 100)

    return state


def node_brooks_analyzer(
    state: PipelineState,
    llm_provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str = "",
    base_url: str = "",
) -> PipelineState:
    """Brooks VLM 分析 — 调用 LLM 进行价格行为分析。"""
    if state.is_dead_market:
        return state

    # 构建 bar 数据上下文
    recent = state.bars[-30:]
    bar_text = _format_bars_for_llm(recent)
    features_text = _format_features_for_llm(state.bar_features)

    system_prompt = _BROOKS_SYSTEM_PROMPT
    user_prompt = f"""分析以下 {state.symbol} {state.timeframe} K 线数据:

{bar_text}

L0 特征摘要:
{features_text}

请按照 Al Brooks 价格行为方法论分析当前市场状态，输出 JSON。"""

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url or None)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=120,
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        state.brooks_analysis = BrooksAnalysisResult(**data)
        logger.info(
            "Brooks 分析: cycle={} direction={} action={} quality={}",
            state.brooks_analysis.market_cycle,
            state.brooks_analysis.always_in_direction,
            state.brooks_analysis.recommended_action,
            state.brooks_analysis.setup_quality,
        )
    except Exception as exc:
        logger.error("Brooks 分析失败: {}", exc)
        # 降级: 保守 WAIT
        state.brooks_analysis = BrooksAnalysisResult(
            market_cycle="trading_range",
            always_in_direction="neutral",
            signal_bar=BrooksSignalBar(quality_score=0, bar_type="doji"),
            recommended_action="wait",
            wait_reason=f"分析异常: {exc}",
        )

    return state


def node_strategy_decision(
    state: PipelineState,
    llm_provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str = "",
    base_url: str = "",
) -> PipelineState:
    """AI 策略决策 — 基于 Brooks 分析生成交易决策。"""
    if state.is_dead_market or not state.brooks_analysis:
        state.decision = TradeDecision(operation="hold", symbol=state.symbol, rationale="无分析数据")
        return state

    ba = state.brooks_analysis

    # Brooks 强制 Hold 规则
    force_hold, reason = _should_force_hold(ba)
    if force_hold:
        state.decision = TradeDecision(operation="hold", symbol=state.symbol, rationale=reason)
        logger.info("Brooks 强制 Hold: {}", reason)
        return state

    # 动态 prompt 注入市场周期规则
    cycle_rules = _get_cycle_rules(ba.market_cycle)
    user_prompt = f"""Brooks 分析结果:
- 市场周期: {ba.market_cycle}
- Always-In: {ba.always_in_direction}
- 信号 bar 质量: {ba.signal_bar.quality_score}/10
- 买压/卖压: {ba.buying_pressure}/{ba.selling_pressure}
- 推荐动作: {ba.recommended_action}
- 当前价格: {state.current_price}

{cycle_rules}

请给出交易决策 (buy/sell/hold)，包含 entry_price, stop_loss, take_profit。输出 JSON。"""

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url or None)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _STRATEGY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=90,
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        state.decision = TradeDecision(symbol=state.symbol, **data)
        logger.info("AI 决策: {} {} prob={:.0f}%", state.decision.operation, state.symbol, state.decision.probability)
    except Exception as exc:
        logger.error("策略决策失败: {}", exc)
        state.decision = TradeDecision(operation="hold", symbol=state.symbol, rationale=f"决策异常: {exc}")

    return state


def node_risk_check(state: PipelineState) -> PipelineState:
    """风险检查 — 验证决策合理性。"""
    d = state.decision
    if not d or d.operation == "hold":
        return state

    from src.analysis.bar_features import calculate_atr
    from src.core.price_calculator import enforce_min_rr, enforce_min_stop_distance

    atr = calculate_atr(state.bars)

    if d.entry_price > 0 and d.stop_loss > 0:
        d.stop_loss = enforce_min_stop_distance(d.entry_price, d.stop_loss, atr)
        if d.take_profit > 0:
            d.take_profit = enforce_min_rr(d.entry_price, d.stop_loss, d.take_profit)

    # 概率过低 → hold
    if d.probability < 60:
        logger.info("概率不足 ({:.0f}% < 60%), 降级为 hold", d.probability)
        d.operation = "hold"
        d.rationale += " [概率不足]"

    return state


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------


class BrooksLLMPipeline:
    """Brooks LLM 交易管线 — 完整的分析→决策→执行流程。"""

    def __init__(
        self,
        llm_provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str = "",
    ):
        self.llm_provider = llm_provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def run(self, symbol: str, bars: list[dict], timeframe: str = "1h") -> PipelineState:
        """执行完整管线: L0 → Brooks 分析 → 策略决策 → 风险检查。"""
        state = PipelineState(symbol=symbol, timeframe=timeframe, bars=bars)

        # Step 1: L0 预处理 + 死市场过滤
        state = node_l0_gate(state)
        if state.is_dead_market:
            state.decision = TradeDecision(operation="hold", symbol=symbol, rationale="死鱼盘")
            return state

        # Step 2: Brooks VLM 分析
        state = node_brooks_analyzer(
            state,
            self.llm_provider,
            self.model,
            self.api_key,
            self.base_url,
        )

        # Step 3: AI 策略决策
        state = node_strategy_decision(
            state,
            self.llm_provider,
            self.model,
            self.api_key,
            self.base_url,
        )

        # Step 4: 风险检查
        state = node_risk_check(state)

        return state


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_BROOKS_SYSTEM_PROMPT = """你是 Al Brooks 价格行为分析专家。分析 K 线数据并输出 JSON:
{
  "market_cycle": "strong_bull_trend|weak_bull_trend|strong_bear_trend|weak_bear_trend|trading_range|breakout_mode|climax",
  "always_in_direction": "long|short|neutral",
  "signal_bar": {"quality_score": 0-10, "bar_type": "strong_bull|weak_bull|doji|weak_bear|strong_bear", "body_size_percent": 0-100, "closes_near": "high|mid|low"},
  "buying_pressure": 0-10,
  "selling_pressure": 0-10,
  "recommended_action": "buy_setup|sell_setup|wait",
  "setup_quality": 0-10,
  "wait_reason": "如果 wait，说明原因",
  "context_summary": "最近 20-50 bar 的市场叙事"
}
规则:
- EMA20 是唯一参考指标
- 趋势 bar = 实体占比 > 50%
- Trading Range 中只交易高质量信号 (>= 8/10)
- Climax 后预期回调
"""

_STRATEGY_SYSTEM_PROMPT = """你是量化交易策略生成器。基于 Brooks 分析生成交易决策，输出 JSON:
{
  "operation": "buy|sell|hold",
  "probability": 0-100,
  "rationale": "决策理由",
  "entry_price": 入场价,
  "stop_loss": 止损价,
  "take_profit": 止盈价,
  "risk_percent": 0.5-2.0
}
规则:
- 风险回报比 >= 2:1
- 止损必须放在结构位 (摆动点、信号 bar 极值)
- 概率 < 60% 时输出 hold
- 永远不要逆势交易 (always_in_direction)
"""


def _format_bars_for_llm(bars: list[dict]) -> str:
    lines = ["idx | open | high | low | close | vol"]
    for i, b in enumerate(bars):
        lines.append(
            f"{i - len(bars) + 1:+d} | {b['open']:.2f} | {b['high']:.2f} | {b['low']:.2f} | {b['close']:.2f} | {b.get('volume', 0):.0f}"
        )
    return "\n".join(lines)


def _format_features_for_llm(features: list) -> str:
    if not features:
        return "无特征数据"
    lines = []
    for f in features[-5:]:
        lines.append(
            f"  {f.bar_type} body={f.body_pct}% close={f.close_position} ema={f.ema_relation} inside={f.is_inside_bar} reversal={f.is_reversal_bar}"
        )
    return "\n".join(lines)


def _should_force_hold(ba: BrooksAnalysisResult) -> tuple[bool, str]:
    if ba.recommended_action == "wait":
        return True, ba.wait_reason or "分析建议等待"
    if ba.setup_quality < 6:
        return True, f"信号质量不足 ({ba.setup_quality}/10 < 6)"
    if ba.market_cycle == "trading_range" and ba.signal_bar.quality_score < 8:
        return True, f"Trading Range 中信号 bar 质量不足 ({ba.signal_bar.quality_score}/10 < 8)"
    if ba.market_cycle == "climax":
        return True, "Climax 阶段不建议入场"
    return False, ""


def _get_cycle_rules(cycle: str) -> str:
    rules = {
        "strong_bull_trend": "🟢 强牛趋势: 只做回调买入 (H1/H2), 不做空。入场: 信号 bar 高点上方 1 tick。止损: 回调低点下方。",
        "weak_bull_trend": "🟡 弱牛趋势: 谨慎做多，需更高质量信号。关注可能的趋势转换。",
        "strong_bear_trend": "🔴 强熊趋势: 只做反弹卖出 (L1/L2), 不做多。入场: 信号 bar 低点下方 1 tick。止损: 反弹高点上方。",
        "weak_bear_trend": "🟡 弱熊趋势: 谨慎做空，需更高质量信号。关注可能的底部。",
        "trading_range": "⬜ 横盘区间: 高抛低吸，需 >= 8/10 信号质量。中部不交易。",
        "breakout_mode": "⚡ 突破模式: 跟随突破方向，但注意假突破。需确认 bar。",
        "climax": "⚠️ Climax: 不入场新仓位，等待回调。",
    }
    return rules.get(cycle, "")
