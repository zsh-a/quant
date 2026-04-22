"""Brooks pattern detectors.

Importing this module triggers registration of all built-in detectors
with :class:`PatternRegistry`.
"""

from src.brooks.patterns import (  # noqa: F401 — imported for side-effect
    breakout_pullback,
    double_tb,
    final_flag,
    h2_l2,
    h_series,
    ii_iii,
    l_series,
    measured_move,
    micro_channel,
    mtr,
    wedge,
)
from src.brooks.patterns.base import DetectorContext, PatternDetector, PatternSignal
from src.brooks.patterns.breakout_pullback import (
    BreakoutPullbackLongDetector,
    BreakoutPullbackShortDetector,
    FailedBreakoutDetector,
)
from src.brooks.patterns.double_tb import DoubleBottomDetector, DoubleTopDetector
from src.brooks.patterns.final_flag import FinalFlagDetector
from src.brooks.patterns.h2_l2 import H2Detector, L2Detector
from src.brooks.patterns.h_series import H1Detector, H3Detector, H4Detector
from src.brooks.patterns.ii_iii import IIBreakoutDetector, IIIBreakoutDetector
from src.brooks.patterns.l_series import L1Detector, L3Detector, L4Detector
from src.brooks.patterns.measured_move import MeasuredMoveTargeter
from src.brooks.patterns.micro_channel import (
    MicroChannelLongDetector,
    MicroChannelShortDetector,
)
from src.brooks.patterns.mtr import MTRLongDetector, MTRShortDetector
from src.brooks.patterns.registry import PatternRegistry
from src.brooks.patterns.wedge import WedgeLongDetector, WedgeShortDetector

__all__ = [
    "BreakoutPullbackLongDetector",
    "BreakoutPullbackShortDetector",
    "DetectorContext",
    "DoubleBottomDetector",
    "DoubleTopDetector",
    "FailedBreakoutDetector",
    "FinalFlagDetector",
    "H1Detector",
    "H2Detector",
    "H3Detector",
    "H4Detector",
    "IIBreakoutDetector",
    "IIIBreakoutDetector",
    "L1Detector",
    "L2Detector",
    "L3Detector",
    "L4Detector",
    "MTRLongDetector",
    "MTRShortDetector",
    "MeasuredMoveTargeter",
    "MicroChannelLongDetector",
    "MicroChannelShortDetector",
    "PatternDetector",
    "PatternRegistry",
    "PatternSignal",
    "WedgeLongDetector",
    "WedgeShortDetector",
]
