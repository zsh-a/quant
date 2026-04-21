"""Brooks price-action trading primitives.

Top-level package for the Brooks-style multi-stage trading pipeline
(detector → analyst → aggregator → TE → risk → execution).
"""

from src.brooks.schema import Decision, Order, Signal

__all__ = ["Signal", "Decision", "Order"]
