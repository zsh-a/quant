"""Brooks Studio API schemas — single contract shared by live & replay.

The backend assembles a :class:`SessionTimeline` from ``session_db.session_logs``
(and the per-session ``equity_history`` / ``trades`` tables) and serves it to
the Brooks Studio UI through both REST and WebSocket transports. Both modes
emit :class:`BarEvent` payloads, so the frontend renders them identically.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.brooks.schema import Decision, Signal


class Bar(BaseModel):
    timestamp_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class RegimeView(BaseModel):
    name: str
    confidence: float = 0.0
    reasons: List[str] = Field(default_factory=list)


class StructureView(BaseModel):
    always_in: Literal["long", "short", "neutral"] = "neutral"
    confirmed_swings: List[Dict[str, Any]] = Field(default_factory=list)
    micro_channel_top: Optional[Dict[str, Any]] = None
    micro_channel_bot: Optional[Dict[str, Any]] = None
    last_breakout_lookback_high: Optional[float] = None
    last_breakout_lookback_low: Optional[float] = None


class FeaturesView(BaseModel):
    is_bull: bool = False
    body_pct: int = 0
    close_position: Literal["high", "mid", "low"] = "mid"
    ema_relation: Literal["above", "at", "below"] = "at"
    leg_dir: Literal["up", "down", "flat"] = "flat"
    leg_length: int = 0
    is_doji: bool = False
    is_inside_bar: bool = False


class HTFView(BaseModel):
    regime: Optional[str] = None
    always_in: Optional[str] = None
    last_swing_idx: Optional[int] = None


class StopAdj(BaseModel):
    from_px: float
    to_px: float
    reason: str = ""


class FillView(BaseModel):
    side: Literal["buy", "sell", "buy_to_cover", "sell_short"]
    qty: float
    price: float
    reason: str = ""


class BarEvent(BaseModel):
    bar_idx: int
    timestamp_ns: int
    regime: Optional[RegimeView] = None
    features: Optional[FeaturesView] = None
    structure: Optional[StructureView] = None
    signals: List[Signal] = Field(default_factory=list)
    decision: Optional[Decision] = None
    fill: Optional[FillView] = None
    stop_adj: Optional[StopAdj] = None
    pnl_r: Optional[float] = None
    htf: Dict[str, HTFView] = Field(default_factory=dict)


class PnLPoint(BaseModel):
    bar_idx: int
    equity_r: float


class SessionTimeline(BaseModel):
    session_id: str
    session_kind: Literal["live", "replay"] = "live"
    symbol: str
    base_interval: str
    htf_intervals: List[str] = Field(default_factory=list)
    bars: List[Bar] = Field(default_factory=list)
    htf_bars: Dict[str, List[Bar]] = Field(default_factory=dict)
    events: List[BarEvent] = Field(default_factory=list)
    pnl_curve: List[PnLPoint] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class TimelinePage(BaseModel):
    """Incremental page returned by the ``/timeline/since/{seq}`` endpoint."""

    events: List[BarEvent] = Field(default_factory=list)
    next_seq: int
    has_more: bool


class ReplayBarRequest(BaseModel):
    """Body for ``POST /sessions/{id}/replay-bar``."""

    bar_idx: int = Field(ge=0)
    analysts: List[str] = Field(min_length=1)


class ReplayStartRequest(BaseModel):
    """Body for ``POST /brooks-studio/replay`` — historical replay session."""

    symbol: str = Field(description="Trading symbol, e.g. 'BTC/USDT'")
    interval: str = Field(default="5m", description="Bar interval, e.g. '1m', '5m', '1h'")
    start: str = Field(description="ISO timestamp for inclusive replay window start")
    end: str = Field(description="ISO timestamp for inclusive replay window end")
    analyst: str = Field(default="rule")
    analyst_params: Dict[str, Any] = Field(default_factory=dict)
    mtf_intervals: List[str] = Field(default_factory=list)
    provider: str = Field(default="bitget", description="ClickHouse ingest provider")
    initial_cash: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    min_expected_r: Optional[float] = None
    llm_min_interval_seconds: Optional[float] = Field(
        default=None,
        description=(
            "Min bar-time gap between LLM/VLM analyst calls. Defaults to "
            "the live session value (60s); set to 0 to disable."
        ),
    )


class ReplayStartResponse(BaseModel):
    session_id: str
    task_id: str
    status: str = "submitted"


class ReplayBarAnalystResult(BaseModel):
    analyst: str
    bar_idx: int
    signals: List[Signal] = Field(default_factory=list)
    decision: Optional[Decision] = None
    error: Optional[str] = None
    cached: bool = False


class ReplayBarResponse(BaseModel):
    bar_idx: int
    results: List[ReplayBarAnalystResult] = Field(default_factory=list)


__all__ = [
    "Bar",
    "RegimeView",
    "StructureView",
    "FeaturesView",
    "HTFView",
    "StopAdj",
    "FillView",
    "BarEvent",
    "PnLPoint",
    "SessionTimeline",
    "TimelinePage",
    "ReplayBarRequest",
    "ReplayBarAnalystResult",
    "ReplayBarResponse",
    "ReplayStartRequest",
    "ReplayStartResponse",
]
