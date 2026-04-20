"""L3 pattern detectors — each implements :class:`PatternDetector`.

Detectors are stateful FSMs that consume :class:`DetectorContext` each bar and
emit :class:`PatternSignal` when a setup forms.
"""

from .base import DetectorContext, PatternDetector, PatternSignal
from .double_tb import DoubleBottomDetector, DoubleTopDetector
from .final_flag import FinalFlagDetector
from .h2_l2 import H2Detector, L2Detector
from .measured_move import MeasuredMoveTargeter
from .wedge import WedgeDetector

__all__ = [
    "DetectorContext",
    "PatternDetector",
    "PatternSignal",
    "H2Detector",
    "L2Detector",
    "DoubleTopDetector",
    "DoubleBottomDetector",
    "WedgeDetector",
    "FinalFlagDetector",
    "MeasuredMoveTargeter",
]
