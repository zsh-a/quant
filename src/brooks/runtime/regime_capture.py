"""Decorator over :class:`BrooksRegimeClassifier` that retains the latest snapshot.

The classifier is stateless w.r.t. its return — for the Studio panel we
need to surface the most recent snapshot per symbol on every BarEvent.
The previous implementation monkey-patched ``classify`` on each instance
once it appeared; this decorator does the same thing through composition,
which is easier to reason about and test.

The wrapper forwards unknown attribute access to the inner classifier so
any code that pokes at config knobs (``classifier.tr_lookback`` etc.)
keeps working.
"""

from __future__ import annotations

from typing import Any, Optional

from src.brooks.regime import BrooksRegimeClassifier, RegimeSnapshot


class RegimeCapturingClassifier:
    """Hold onto the most recent ``classify`` output."""

    def __init__(self, inner: BrooksRegimeClassifier):
        self._inner = inner
        self.last_snapshot: Optional[RegimeSnapshot] = None

    def classify(self, *args: Any, **kwargs: Any) -> RegimeSnapshot:
        snap = self._inner.classify(*args, **kwargs)
        self.last_snapshot = snap
        return snap

    @property
    def inner(self) -> BrooksRegimeClassifier:
        return self._inner

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


__all__ = ["RegimeCapturingClassifier"]
