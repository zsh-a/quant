"""Brooks price-action trading primitives.

Top-level package for the Brooks-style multi-stage trading pipeline
(detector → analyst → aggregator → TE → risk → execution).
"""

from src.brooks.context import AccountSnapshot, Bar, BrooksContext, TFSnapshot
from src.brooks.features import (
    BarFeatureExtractor,
    ExtendedBarFeatures,
    SwingPoint,
)
from src.brooks.schema import Decision, Order, Signal
from src.brooks.structure import (
    ChannelFit,
    MarketStructure,
    MarketStructureTracker,
)

__all__ = [
    "AccountSnapshot",
    "Bar",
    "BarFeatureExtractor",
    "BrooksContext",
    "ChannelFit",
    "Decision",
    "ExtendedBarFeatures",
    "MarketStructure",
    "MarketStructureTracker",
    "Order",
    "Signal",
    "SwingPoint",
    "TFSnapshot",
]
