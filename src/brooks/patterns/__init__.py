"""L3 pattern detectors.

Importing this module triggers registration of all built-in detectors
with :class:`PatternRegistry`.
"""

from src.brooks.patterns import (  # noqa: F401 — imported for side-effect
    double_tb,
    final_flag,
    h2_l2,
    measured_move,
    wedge,
)
from src.brooks.patterns.base import DetectorContext, PatternDetector, PatternSignal
from src.brooks.patterns.double_tb import DoubleBottomDetector, DoubleTopDetector
from src.brooks.patterns.final_flag import FinalFlagDetector
from src.brooks.patterns.h2_l2 import H2Detector, L2Detector
from src.brooks.patterns.measured_move import MeasuredMoveTargeter
from src.brooks.patterns.registry import PatternRegistry
from src.brooks.patterns.wedge import WedgeLongDetector, WedgeShortDetector

__all__ = [
    "DetectorContext",
    "DoubleBottomDetector",
    "DoubleTopDetector",
    "FinalFlagDetector",
    "H2Detector",
    "L2Detector",
    "MeasuredMoveTargeter",
    "PatternDetector",
    "PatternRegistry",
    "PatternSignal",
    "WedgeLongDetector",
    "WedgeShortDetector",
]
