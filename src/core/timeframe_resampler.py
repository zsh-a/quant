"""In-strategy multi-timeframe resampling.

Aggregates lower-timeframe bars (e.g. 5m) into higher timeframes (15m/1h)
*without* ever emitting a partial bar. A higher-TF bar is only returned the
moment it closes; look-ahead is impossible because the close boundary is a
function of the incoming bar's timestamp.

Supported intervals: multiples of the base interval expressed in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.core.base import Bar

_INTERVAL_SECS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
}


@dataclass
class _AggState:
    bucket_start_s: int = -1
    o: float = 0.0
    h: float = 0.0
    l: float = 0.0
    c: float = 0.0
    v: float = 0.0
    amt: float = 0.0
    ts_open: object = None


class TimeframeResampler:
    def __init__(self, base: str = "5m", higher: Optional[List[str]] = None):
        if base not in _INTERVAL_SECS:
            raise ValueError(f"Unsupported base interval: {base}")
        self.base = base
        self.base_secs = _INTERVAL_SECS[base]
        self.higher: List[str] = []
        self._sec_per: Dict[str, int] = {}
        self._state: Dict[str, _AggState] = {}
        for h in higher or []:
            if h not in _INTERVAL_SECS:
                raise ValueError(f"Unsupported higher interval: {h}")
            if _INTERVAL_SECS[h] % self.base_secs != 0:
                raise ValueError(f"Higher TF {h} not a multiple of base {base}")
            self.higher.append(h)
            self._sec_per[h] = _INTERVAL_SECS[h]
            self._state[h] = _AggState()

    def update(self, bar: Bar) -> Dict[str, Optional[Bar]]:
        """Ingest a base-TF bar, return ``{interval: Bar or None}``.

        The returned Bar is the *just-closed* higher-TF bar. In-progress
        higher-TF bars are never returned.
        """
        out: Dict[str, Optional[Bar]] = {}
        ts = bar.timestamp
        open_s = int(ts.timestamp())
        for h in self.higher:
            st = self._state[h]
            sec = self._sec_per[h]
            # The bar *belongs* to the bucket that starts at or before its open.
            bucket_of_bar = open_s - (open_s % sec)

            if st.bucket_start_s == -1:
                st.bucket_start_s = bucket_of_bar
                st.o = bar.open
                st.h = bar.high
                st.l = bar.low
                st.c = bar.close
                st.v = bar.volume
                st.amt = bar.amount
                st.ts_open = ts
                out[h] = None
                continue

            if bucket_of_bar == st.bucket_start_s:
                st.h = max(st.h, bar.high)
                st.l = min(st.l, bar.low)
                st.c = bar.close
                st.v += bar.volume
                st.amt += bar.amount
                out[h] = None
                # Check if the bucket has just closed (this bar is the last of bucket)
                next_bar_bucket_start = (open_s + self.base_secs) - ((open_s + self.base_secs) % sec)
                if next_bar_bucket_start != st.bucket_start_s:
                    out[h] = Bar(
                        symbol=bar.symbol,
                        timestamp=st.ts_open,
                        open=st.o,
                        high=st.h,
                        low=st.l,
                        close=st.c,
                        volume=st.v,
                        amount=st.amt,
                    )
                    st.bucket_start_s = -1
            else:
                # New bucket started, emit prior closed bucket first.
                out[h] = Bar(
                    symbol=bar.symbol,
                    timestamp=st.ts_open,
                    open=st.o,
                    high=st.h,
                    low=st.l,
                    close=st.c,
                    volume=st.v,
                    amount=st.amt,
                )
                st.bucket_start_s = bucket_of_bar
                st.o = bar.open
                st.h = bar.high
                st.l = bar.low
                st.c = bar.close
                st.v = bar.volume
                st.amt = bar.amount
                st.ts_open = ts
        return out
