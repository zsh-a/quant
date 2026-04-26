"""Brooks runtime — engine internals shared by live + replay tasks.

Two Celery tasks (``brooks_live_task``, ``brooks_replay_task``) drive the
same per-bar pipeline. Anything time-sensitive or I/O-shaped lives here as
a pluggable component so the tasks differ only in configuration:

* ``Clock`` — wall-clock for live, bar-timestamp for replay.
* ``Throttle`` — min-gap rate limit, time-source agnostic.
* ``ThrottledAnalyst`` — DI replacement for the legacy ``analyze`` monkey-patch.
* ``RegimeCapturingClassifier`` — explicit decorator that retains last snapshot.
* ``BarEventSink`` — fanout target for the per-bar payload (persist + WS).
* ``BrooksCore`` — the shared per-bar pipeline. Live + replay both call ``process_bar``.
"""

from src.brooks.runtime.analyst_wrap import ThrottledAnalyst
from src.brooks.runtime.clock import BarClock, Clock, WallClock
from src.brooks.runtime.core import BrooksCore
from src.brooks.runtime.event_sink import (
    BarEventSink,
    BroadcastingSink,
    CompositeSink,
    PersistingSink,
)
from src.brooks.runtime.regime_capture import RegimeCapturingClassifier
from src.brooks.runtime.throttle import Throttle

__all__ = [
    "BarClock",
    "BarEventSink",
    "BroadcastingSink",
    "BrooksCore",
    "Clock",
    "CompositeSink",
    "PersistingSink",
    "RegimeCapturingClassifier",
    "ThrottledAnalyst",
    "Throttle",
    "WallClock",
]
