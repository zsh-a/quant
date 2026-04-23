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
from src.brooks.regime import BrooksRegime, BrooksRegimeClassifier, RegimeSnapshot
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
    "BrooksRegime",
    "BrooksRegimeClassifier",
    "BrooksStrategy",
    "ChannelFit",
    "Decision",
    "ExtendedBarFeatures",
    "MarketStructure",
    "MarketStructureTracker",
    "Order",
    "RegimeSnapshot",
    "Signal",
    "SwingPoint",
    "TFSnapshot",
]


def __getattr__(name: str):
    # Lazy import avoids a circular dependency at package import time:
    # ``src.brooks.strategy`` imports ``src.strategies.registry``, which
    # in turn imports the strategies that depend on this package.
    if name == "BrooksStrategy":
        from src.brooks.strategy import BrooksStrategy

        return BrooksStrategy
    raise AttributeError(f"module 'src.brooks' has no attribute {name!r}")
