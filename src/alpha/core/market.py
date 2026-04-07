"""Market type definitions and profiles.

Each market (crypto futures, A-shares, …) bundles its TensorSchema,
field aliases, LLM prompt descriptions, seed formulas, and persona
into a single ``MarketProfile``.  Adding a new market = registering
one more profile here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .dsl import TensorSchema


class MarketType(str, Enum):
    CRYPTO = "crypto"
    A_SHARE = "a_share"


@dataclass(frozen=True)
class MarketProfile:
    market_type: MarketType
    schema: TensorSchema
    field_aliases: dict[str, str]
    field_descriptions: str          # LLM prompt 中的可用字段说明
    seeds: tuple[str, ...]           # HeuristicBackend 种子公式
    supported_intervals: tuple[str, ...]
    persona: str                     # LLM system prompt 角色

    # HeuristicBackend 变异替换表: list of (source, target)
    mutation_replacements: tuple[tuple[str, str], ...] = ()

    # 前端 symbol preset 列表，每项 dict: key, label, brief, symbols/universe
    symbol_presets: tuple[dict, ...] = ()


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

_PROFILES: dict[MarketType, MarketProfile] = {}


def register_profile(profile: MarketProfile) -> None:
    _PROFILES[profile.market_type] = profile


def get_market_profile(market: MarketType | str) -> MarketProfile:
    if isinstance(market, str):
        market = MarketType(market)
    if market not in _PROFILES:
        raise ValueError(f"Unknown market: {market!r}. Available: {list(_PROFILES)}")
    return _PROFILES[market]


def list_market_types() -> list[str]:
    return [m.value for m in _PROFILES]


# ---------------------------------------------------------------------------
# Crypto futures profile
# ---------------------------------------------------------------------------

_CRYPTO_FIELD_ALIASES: dict[str, str] = {
    "oi": "open_interest",
    "openinterest": "open_interest",
    "oivalue": "open_interest_value",
    "openinterestvalue": "open_interest_value",
    "fundingrate": "funding_rate",
    "bidaskspread": "bid_ask_spread",
    "tradecount": "trade_count",
    "trades": "trade_count",
    "takerbuyvolume": "taker_buy_volume",
    "takerbuyquotevolume": "taker_buy_quote_volume",
    "markopen": "mark_open",
    "markhigh": "mark_high",
    "marklow": "mark_low",
    "markclose": "mark_close",
    "mark": "mark_close",
    "premiumopen": "premium_open",
    "premiumhigh": "premium_high",
    "premiumlow": "premium_low",
    "premiumclose": "premium_close",
    "premium": "premium_close",
    "basis": "premium_close",
    "lsratio": "long_short_ratio",
    "longshort": "long_short_ratio",
    "longshortratio": "long_short_ratio",
    "takerlsratio": "taker_long_short_vol_ratio",
    "takerlongshortvol": "taker_long_short_vol_ratio",
    "takerlongshortratio": "taker_long_short_vol_ratio",
    "takerlongshorvolratio": "taker_long_short_vol_ratio",
    "toptraderlongshortratio": "top_trader_long_short_ratio",
    "toptraderlongshortpositionratio": "top_trader_long_short_position_ratio",
}

_CRYPTO_FIELDS_DESC = """\
## Fields — price
open, high, low, close, volume, turnover, vwap, bid_ask_spread

## Fields — volume detail
trade_count, taker_buy_volume, taker_buy_quote_volume

## Fields — mark price (fair value)
mark_open, mark_high, mark_low, mark_close

## Fields — premium index (futures − spot basis)
premium_open, premium_high, premium_low, premium_close

## Fields — market microstructure
funding_rate, open_interest, open_interest_value
long_short_ratio, taker_long_short_vol_ratio
top_trader_long_short_ratio, top_trader_long_short_position_ratio"""

_CRYPTO_SEEDS = (
    "cs_rank(ts_mean(close, 5) - close)",
    "cs_rank(ts_std(close, 10))",
    "cs_rank(delta(premium_close, 5))",
    "cs_rank(ts_zscore(funding_rate, 20))",
    "cs_rank(delta(open_interest, 10) - ts_mean(delta(open_interest, 10), 20))",
    "cs_rank(ts_zscore(long_short_ratio, 20))",
    "cs_rank(div(taker_buy_volume, volume + 1e-12) - 0.5)",
    "cs_rank(ts_corr(close, taker_buy_volume, 10))",
)

_CRYPTO_MUTATIONS = (
    ("ts_mean(", "ts_std("), ("ts_std(", "ts_mean("),
    ("ts_max(", "ts_rank("), ("ts_rank(", "ts_mean("),
    ("close", "vwap"), ("close", "mark_close"),
    ("volume", "turnover"), ("volume", "taker_buy_volume"),
    ("close", "hlc3(high, low, close)"), ("turnover", "adv_n(turnover, 5)"),
    ("close", "ohlc4(open, high, low, close)"),
    ("volatility_n(close, 20)", "atr_n(high, low, close, 14)"),
    ("funding_rate", "ts_zscore(funding_rate, 20)"),
    ("open_interest", "delta(open_interest, 5)"),
    ("close", "premium_close"), ("volume", "trade_count"),
)

register_profile(MarketProfile(
    market_type=MarketType.CRYPTO,
    schema=TensorSchema.default_market_schema(),
    field_aliases=_CRYPTO_FIELD_ALIASES,
    field_descriptions=_CRYPTO_FIELDS_DESC,
    seeds=_CRYPTO_SEEDS,
    supported_intervals=("5m", "15m", "1h", "4h"),
    persona="You are a senior crypto quant researcher.",
    mutation_replacements=_CRYPTO_MUTATIONS,
))


# ---------------------------------------------------------------------------
# A-share daily profile
# ---------------------------------------------------------------------------

_ASTOCK_FIELD_ALIASES: dict[str, str] = {
    # 常见缩写 → 标准字段名
    "turnoverrate": "turn",
    "turnrate": "turn",
    "pe": "peTTM",
    "pettm": "peTTM",
    "pb": "pbMRQ",
    "pbmrq": "pbMRQ",
    "adj": "adjfactor",
    "adjfact": "adjfactor",
    "st": "isST",
    "ist": "isST",
    "pctchg": "pctChg",
    "ret": "pctChg",
    "return": "pctChg",
    "pre": "preclose",
    "prevclose": "preclose",
}

_ASTOCK_FIELDS_DESC = """\
## Fields — price (前复权)
open, high, low, close, preclose, vwap

## Fields — volume / activity
volume (成交量), amount (成交额), turnover (= amount), turn (换手率 %)

## Fields — fundamental
peTTM (市盈率 TTM), pbMRQ (市净率 MRQ), pctChg (涨跌幅 %)

## Fields — status
adjfactor (复权因子), isST (ST 标记, 0 或 1)"""

_ASTOCK_SEEDS = (
    "cs_rank(ts_mean(close, 5) - close)",
    "cs_rank(ts_std(close, 10))",
    "cs_rank(neg(ts_zscore(pctChg, 20)))",
    "cs_rank(div(1.0, peTTM + 1e-12))",
    "cs_rank(ts_corr(close, volume, 10))",
    "cs_rank(neg(ts_mean(turn, 5)))",
    "cs_rank(delta(close, 5) - ts_mean(delta(close, 5), 20))",
    "cs_rank(div(close - ts_min(low, 20), ts_max(high, 20) - ts_min(low, 20) + 1e-12))",
)

_ASTOCK_MUTATIONS = (
    ("ts_mean(", "ts_std("), ("ts_std(", "ts_mean("),
    ("ts_max(", "ts_rank("), ("ts_rank(", "ts_mean("),
    ("close", "vwap"), ("close", "preclose"),
    ("volume", "amount"), ("volume", "turn"),
    ("close", "hlc3(high, low, close)"),
    ("close", "ohlc4(open, high, low, close)"),
    ("volatility_n(close, 20)", "atr_n(high, low, close, 14)"),
    ("peTTM", "ts_zscore(peTTM, 20)"),
    ("pbMRQ", "ts_zscore(pbMRQ, 20)"),
    ("pctChg", "ts_mean(pctChg, 5)"),
    ("turn", "ts_zscore(turn, 10)"),
)

_ASTOCK_SYMBOL_PRESETS = (
    {
        "key": "hs300",
        "label": "沪深 300",
        "brief": "大盘蓝筹，定价有效，适合验证稳健因子",
        "universe": "000300",
    },
    {
        "key": "zz500",
        "label": "中证 500",
        "brief": "中盘股，流动性好，alpha 空间适中",
        "universe": "000905",
    },
    {
        "key": "zz1000",
        "label": "中证 1000",
        "brief": "小盘股，散户多，因子收益高但衰减快",
        "universe": "000852",
    },
    {
        "key": "zz_all",
        "label": "中证全指",
        "brief": "全 A 股可投资范围（中证全指 000985）",
        "universe": "000985",
    },
    {
        "key": "cy50",
        "label": "创业板 50",
        "brief": "科技成长板块，高波动高 beta",
        "universe": "399673",
    },
    {
        "key": "custom",
        "label": "自定义股票池",
        "brief": "手动指定 symbols 列表",
        "universe": None,
    },
)

register_profile(MarketProfile(
    market_type=MarketType.A_SHARE,
    schema=TensorSchema.default_stock_schema(),
    field_aliases=_ASTOCK_FIELD_ALIASES,
    field_descriptions=_ASTOCK_FIELDS_DESC,
    seeds=_ASTOCK_SEEDS,
    supported_intervals=("1d",),
    persona="You are a senior A-share (中国 A 股) quant researcher.",
    mutation_replacements=_ASTOCK_MUTATIONS,
    symbol_presets=_ASTOCK_SYMBOL_PRESETS,
))
