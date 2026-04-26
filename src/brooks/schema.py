"""Unified Pydantic schemas for the Brooks pipeline.

These models form the single data contract shared by detectors, analysts,
aggregators, the Trader's Equation evaluator, the risk layer, and the
execution engine.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Signal(BaseModel):
    """A pattern-detection signal produced by a rule/LLM/VLM detector."""

    pattern: str
    pattern_type: Optional[str] = None
    side: Literal["long", "short"]
    signal_bar_idx: int
    entry_px: float = Field(gt=0)
    stop_px: float = Field(gt=0)
    target_px: Optional[float] = None
    probability: float = Field(ge=0, le=1)
    quality: float = Field(ge=0, le=1)
    reasoning: str = ""
    source: str
    meta: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _entry_differs_from_stop(self) -> "Signal":
        if self.entry_px == self.stop_px:
            raise ValueError("entry_px must differ from stop_px")
        return self

    @model_validator(mode="after")
    def _populate_pattern_type(self) -> "Signal":
        if self.pattern_type is None:
            from src.brooks.decision.context_filter import pattern_type_for

            object.__setattr__(self, "pattern_type", pattern_type_for(self.pattern))
        return self

    @property
    def one_r(self) -> float:
        """Absolute per-share risk (|entry − stop|)."""
        return abs(self.entry_px - self.stop_px)


class Decision(BaseModel):
    """An analyst's trading decision derived from one or more Signals."""

    symbol: str
    side: Literal["long", "short"]
    entry_px: float
    stop_px: float
    target_px: float
    quantity: float = 0.0
    probability: float = Field(ge=0, le=1)
    expected_r: float
    regime: str
    htf_aligned: bool
    signals: list[Signal] = Field(default_factory=list)
    source: str
    reasoning: str = ""


class Order(BaseModel):
    """A concrete broker order ready for execution."""

    symbol: str
    side: Literal["buy", "sell", "sell_short", "buy_to_cover"]
    quantity: float = Field(gt=0)
    type: Literal["market", "stop", "limit"]
    price: Optional[float] = None
    reason: str = ""
    decision_ref: Optional[str] = None
