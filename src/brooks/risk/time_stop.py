"""N-bar invalidation.

If a trade fails to reach 1R within ``max_bars_to_1r`` bars of the entry, the
setup is treated as stale and exited at market. Tracks ``max_unrealized_r`` so
a trade that briefly touched 1R and pulled back is not forced to exit.
"""

from __future__ import annotations

from src.brooks.risk.state import PositionState


class TimeStop:
    def __init__(self, max_bars_to_1r: int = 10):
        if max_bars_to_1r < 1:
            raise ValueError("max_bars_to_1r must be >= 1")
        self.max_bars_to_1r = max_bars_to_1r

    def should_exit(
        self,
        pos: PositionState,
        bars_since_entry: int,
        max_unrealized_r: float,
    ) -> bool:
        return bars_since_entry >= self.max_bars_to_1r and max_unrealized_r < 1.0
