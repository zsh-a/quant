"""Decision-layer plumbing — signal aggregation, the Trader's Equation
evaluator, and the EV-based filter that replaces the legacy ``min_rr``
hard threshold.
"""

from src.brooks.decision.aggregator import AggregatedDecision, SignalAggregator
from src.brooks.decision.context_filter import ContextDecision, ContextFilter
from src.brooks.decision.ev_gate import EVGate
from src.brooks.decision.hit_rate import HitRateKey, HitRateTable
from src.brooks.decision.trader_equation import TraderEquation

__all__ = [
    "AggregatedDecision",
    "ContextDecision",
    "ContextFilter",
    "EVGate",
    "HitRateKey",
    "HitRateTable",
    "SignalAggregator",
    "TraderEquation",
]
