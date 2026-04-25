# Brooks Usage Recipes

Eight runnable recipes — copy/paste into a script or a notebook in the repo
root. Every example uses real APIs from `src/brooks/` and `src/alpha/llm/`;
unverified parameters are not invented. For the architecture overview see
[README.md](README.md).

## Recipe 1 — Backtest with the rule analyst

零 LLM 调用，纯 Brooks 模式检测器作为基线。`BrooksStrategy` 注册名为
`"brooks"`，可以被 `StrategyRegistry` 取到，也可以直接构造。

```python
from src.brooks.strategy import BrooksStrategy

strategy = BrooksStrategy(
    analyst="rule",
    aggregator_params={"confluence_n": 1},
    te_params={"cost_r": 0.05},
    min_expected_r=0.1,
    sizer_params={"kind": "kelly", "max_risk_pct": 0.02, "fraction": 0.5},
    stop_ladder_params={"partial_trail_trigger_r": 1.5},
    time_stop_params={"max_bars_to_1r": 10},
    portfolio_params={"max_daily_risk_pct": 0.03, "max_symbol_positions": 1},
    base_interval="5m",
    mtf_intervals=["1h"],     # 打开 HTF 流水线
)
# Strategy 实现了 src/core/base.py:Strategy ABC，直接接到回测 engine 即可：
# engine.set_strategy(strategy); engine.run(...)
```

## Recipe 2 — LLM Analyst (Claude Opus 4.7)

用 Anthropic Opus 4.7 做语言模型分析师。Provider 实例由调用方注入，
PromptBundle 默认从 `prompts/brooks/` 加载，concept_manual + few-shot
会被 provider 走 prompt cache。

```python
from src.alpha.llm.providers.anthropic import AnthropicProvider
from src.brooks.strategy import BrooksStrategy

provider = AnthropicProvider(model="claude-opus-4-7")

strategy = BrooksStrategy(
    analyst="llm",
    analyst_params={
        "provider": provider,
        "model": "claude-opus-4-7",
        "cache_ttl_seconds": 3600,
        "max_tokens": 4096,
        "temperature": 0.0,
        "context_budget_tokens": 2000,    # 渲染 LTF/HTF bars 的 token 预算
    },
    base_interval="5m",
    mtf_intervals=["1h", "4h"],
)

# Signal.source 会是 "llm:claude-opus-4-7"；
# Signal.meta 携带 latency_ms / input_tokens / output_tokens / cache_hit / model。
```

## Recipe 3 — VLMAnalyst with Gemini 2.5 Pro

用渲染好的 OHLCV PNG 喂给多模态模型，适合形态偏图像（楔形 / 微通道
/ breakout-mode 转换）的场景。

```python
from src.alpha.llm.providers.gemini import GeminiProvider
from src.brooks.render.chart import ChartStyle
from src.brooks.strategy import BrooksStrategy

provider = GeminiProvider(model="gemini-2.5-pro")

strategy = BrooksStrategy(
    analyst="vlm",
    analyst_params={
        "provider": provider,
        "model": "gemini-2.5-pro",
        "include_htf": True,           # 在主图右上角嵌入 HTF inset
        "chart_style": ChartStyle(
            width=1280,
            height=720,
            show_volume=True,
            show_swing_markers=True,
        ),
        "context_budget_tokens": 1000,  # 文本预算偏小，主信息走图
    },
)
# Signal.source 会是 "vlm:gemini-2.5-pro"；
# Signal.meta["annotations"] 携带 VLM 返回的 box / line overlay，
# 可以喂回 render_annotated() 二次成图（用于 eval 报告或 golden review）。
```

## Recipe 4 — Ensemble Critic（Rule 出信号 + LLM 复核）

Producer 出候选，critic 决定保留/拒绝并调整 probability。LLM 调用次数
= 候选数（不是每根 bar 一次），成本可控。

```python
from src.alpha.llm.providers.anthropic import AnthropicProvider
from src.brooks.analyst.base import AnalystRegistry
from src.brooks.strategy import BrooksStrategy

producer = AnalystRegistry.build("rule")          # 零成本生成候选
critic = AnalystRegistry.build(
    "llm",
    provider=AnthropicProvider(model="claude-sonnet-4-6"),
    model="claude-sonnet-4-6",
)

strategy = BrooksStrategy(
    analyst="ensemble.critic",
    analyst_params={
        "producer": producer,
        "critic": critic,
        "critic_prompt_overlay": (
            "You are reviewing rule-engine candidates. For each candidate "
            "set meta.confirms = idx if you endorse, -1 to reject."
        ),
    },
)
# 通过 source="ensemble.critic" 与 meta["producer_source"] / meta["critic_source"]
# 可以追溯每个最终信号的来源。
```

## Recipe 5 — Add a custom Pattern Detector

新检测器自动被 `RuleAnalyst` 拾起 — 不需要改任何分发代码。

```python
# src/brooks/patterns/my_breakout.py
from typing import Optional

from src.brooks.patterns.base import DetectorContext, PatternDetector, PatternSignal
from src.brooks.patterns.registry import PatternRegistry


@PatternRegistry.register("my_breakout")
class MyBreakoutDetector(PatternDetector):
    """触发条件：当前 bar 收阳且突破最近 N 根 bar 的最高价。"""

    def __init__(self, lookback: int = 10) -> None:
        self.lookback = lookback

    def on_bar(self, ctx: DetectorContext) -> Optional[PatternSignal]:
        feats = ctx.recent_features
        if len(feats) <= self.lookback:
            return None
        cur = feats[-1]
        prior = feats[-1 - self.lookback : -1]
        prior_high = max(f.high for f in prior)
        if cur.is_bull and cur.close > prior_high:
            return PatternSignal(
                detector="my_breakout",
                side="long",
                signal_bar_idx=cur.bar_idx,
                entry_px=cur.high + 1e-6,
                stop_px=cur.low,
                timestamp_ns=cur.timestamp_ns,
                reason=f"close>{prior_high:.2f}",
            )
        return None
```

然后从 `src/brooks/patterns/__init__.py` 加一行 `from src.brooks.patterns
import my_breakout  # noqa: F401`，注册装饰器才会运行。`RuleAnalyst` 在
默认参数下会自动覆盖到新检测器。

## Recipe 6 — Run the Leaderboard locally

端到端：加载 golden dataset → 跑全部 analysts → 写 HTML + parquet 历史。
`--mock-llm` 用确定性的内嵌 MockProvider，CI 里跑也不需要 API key。

```bash
# 离线 / CI 烟测（不走任何网络，也不读取 API key）
python scripts/brooks_leaderboard.py \
    --config config/brooks/leaderboard.yaml \
    --mock-llm \
    --no-persist

# 真正跑（按需设置 ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY）
python scripts/brooks_leaderboard.py --config config/brooks/leaderboard.yaml

# 输出：
#   data/brooks/leaderboards/<timestamp>.html  ← 含 Pareto 散点 + bucket 表
#   data/brooks/leaderboard.parquet            ← 一行/分析师追加
```

代码内调用同样可行：

```python
import asyncio
from src.brooks.eval.golden import GoldenDataset
from src.brooks.eval.leaderboard import AnalystFactory, Leaderboard, LeaderboardConfig

async def main():
    cfg = LeaderboardConfig.load("config/brooks/leaderboard.yaml")
    dataset = GoldenDataset.load(cfg.dataset)
    factory = AnalystFactory(provider_factory=...)   # see _real_provider_factory in scripts/
    board = Leaderboard(cfg, factory=factory)
    entries = await board.run_all(dataset)
    board.to_html(cfg.output_dir / "latest.html")
    board.persist()
    for e in entries:
        print(e.analyst_name, "F1=", e.f1_pattern, "cost=$", e.cost_per_run_usd)

asyncio.run(main())
```

## Recipe 7 — Start a paper-trading session and view the UI

启动方式有两种：直接调 Celery 任务，或走 `/brooks-live` REST。两者最终都
会把 session 注册到 `BrooksLiveRegistry`，UI panel 通过
`/ws/brooks/{session_id}` 订阅事件。

```bash
# 1) 起后端 + worker（含 brooks 任务自动注册到 celery_app.imports）
./dev_local.sh up

# 2) REST 启一个 session
curl -X POST http://localhost:8000/brooks-live/start \
    -H "content-type: application/json" \
    -d '{
          "symbol": "BTC/USDT",
          "interval": "5m",
          "exchange": "binance",
          "analyst": "rule",
          "mode": "paper",
          "mtf_intervals": ["1h"]
        }'
# → {"session_id": "...", "task_id": "..."}

# 3) 浏览器打开 http://localhost:5173/brooks-live （dev）或 http://localhost/brooks-live （prod）
#    UI: ChartPanel + AnalystSwitch + DecisionLog + PnLCard
#    底层 hook: ui/src/hooks/useBrooksLive.ts
```

切换分析师不需要重启 session（`SwitchRequest` 仅接 `analyst` 名，参数走启动时的 `analyst_params`）：

```bash
curl -X POST http://localhost:8000/brooks-live/<session_id>/switch \
    -H "content-type: application/json" \
    -d '{"analyst": "rule"}'
```

成本控制（环境变量）：

```bash
QUANT_BROOKS__LLM_MIN_INTERVAL_SECONDS=60        # LLM/VLM 每对 (symbol,interval) 最小调用间隔
QUANT_BROOKS__LLM_DAILY_BUDGET_USD=10            # 软日预算（咨询用，rate limit 是硬门槛）
QUANT_BROOKS__INITIAL_CASH=100000
QUANT_BROOKS__COMMISSION=0.0003
```

⚠️ **仅 paper**：`brooks_live_task` 在收到 `mode != "paper"` 时会
`raise ValueError`。`create_live_broker(mode="live")` 同样被禁用 —
真实下单是 out-of-scope。

## Recipe 8 — Auto-label a silver dataset

把一段历史 OHLCV 拿来，跑 rule + N 个 LLM 求共识，写出 silver
`GoldenSample`。每根 bar 至多产出 1 条样本，至少 `--min-agreement` 个
分析师对 `(pattern, side)` 一致才会保留。

```bash
# 输入文件需要列：timestamp_ns, open, high, low, close, volume
python scripts/brooks_label_silver.py \
    --input data/brooks/raw/btcusdt_5m.parquet \
    --output data/brooks/silver/btcusdt_5m.parquet \
    --symbol BTCUSDT --interval 5m \
    --llm openai:gpt-4.1 \
    --llm anthropic:claude-sonnet-4-6 \
    --min-agreement 2 \
    --max-concurrent 4 \
    --entry-tolerance 0.005 \
    --min-bars 20

# Dry-run（只统计，不写文件，便于调参）
python scripts/brooks_label_silver.py \
    --input data/brooks/raw/sample.jsonl \
    --symbol BTC --interval 5m \
    --min-agreement 1 \
    --dry-run
```

代码内调用：

```python
import asyncio
from src.brooks.analyst.base import AnalystRegistry
from src.brooks.context import Bar
from src.brooks.eval.auto_label import AutoLabeler
from src.brooks.eval.golden import GoldenDataset

bars = [Bar(timestamp_ns=..., open=..., high=..., low=..., close=..., volume=...) for _ in range(...)]

labeler = AutoLabeler(
    rule_analyst=AnalystRegistry.build("rule"),
    llm_analysts=[],            # 全 rule 也行（min_agreement=1）
    min_agreement=1,
    min_bars_for_label=20,
)
samples = asyncio.run(labeler.label(bars=bars, symbol="BTCUSDT", interval="5m"))
GoldenDataset.from_samples(samples).save("data/brooks/silver/manual.parquet")
```

把 silver 文件喂回 leaderboard / EvalRunner 即可获得跨模型对比数据 —
完整闭环：raw bars → silver labels → eval → leaderboard → live → 反馈回
`hit_rate_table` → 影响新 EV gate 决策。
