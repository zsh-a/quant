"""Brooks risk layer.

Composable components for the strategy → decision → risk → execution pipeline:

* :class:`FixedPercentSizer` / :class:`KellySizer` — position sizing.
* :class:`StopLadder` — initial → BE → partial_trail → swing_trail stop upgrades.
* :class:`TimeStop` — N-bar invalidation.
* :class:`PortfolioGuard` — daily / per-symbol / correlation-group gates.
* :class:`PositionState` — mutable per-position state threaded through all of
  the above.
"""

from src.brooks.risk.portfolio_guard import PortfolioGuard
from src.brooks.risk.sizer import FixedPercentSizer, KellySizer, Sizer
from src.brooks.risk.state import LadderStage, PositionState
from src.brooks.risk.stop_ladder import StopLadder
from src.brooks.risk.time_stop import TimeStop

__all__ = [
    "FixedPercentSizer",
    "KellySizer",
    "LadderStage",
    "PortfolioGuard",
    "PositionState",
    "Sizer",
    "StopLadder",
    "TimeStop",
]
