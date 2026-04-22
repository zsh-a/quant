"""Decision-layer plumbing — signal aggregation, gating, and (later) the
Trader's Equation evaluator.
"""

from src.brooks.decision.aggregator import AggregatedDecision, SignalAggregator

__all__ = [
    "AggregatedDecision",
    "SignalAggregator",
]
