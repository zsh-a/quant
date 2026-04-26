"""RegimeCapturingClassifier holds onto the most recent classify() output.

The Studio panel reads ``last_snapshot`` after each bar to populate the
BarEvent's regime field. Without the wrapper, the classifier discards
its own output once it returns.
"""

from src.brooks.regime import BrooksRegime, BrooksRegimeClassifier, RegimeSnapshot
from src.brooks.runtime.regime_capture import RegimeCapturingClassifier


class _StubInner:
    def __init__(self):
        self.calls = 0
        self.config_value = 42

    def classify(self, *args, **kwargs) -> RegimeSnapshot:
        self.calls += 1
        return RegimeSnapshot(
            regime=BrooksRegime.UNKNOWN,
            confidence=0.5,
            reasons=[f"call {self.calls}"],
        )


def test_initial_snapshot_is_none():
    wrapped = RegimeCapturingClassifier(_StubInner())
    assert wrapped.last_snapshot is None


def test_classify_caches_snapshot():
    wrapped = RegimeCapturingClassifier(_StubInner())
    snap = wrapped.classify()
    assert wrapped.last_snapshot is snap
    assert snap.reasons == ["call 1"]


def test_subsequent_calls_overwrite():
    wrapped = RegimeCapturingClassifier(_StubInner())
    wrapped.classify()
    wrapped.classify()
    assert wrapped.last_snapshot.reasons == ["call 2"]


def test_attribute_forwarding():
    inner = _StubInner()
    wrapped = RegimeCapturingClassifier(inner)
    # Access an attr that lives on the inner classifier.
    assert wrapped.config_value == 42


def test_works_with_real_classifier():
    """Sanity check: wrapping the real BrooksRegimeClassifier still classifies."""
    real = BrooksRegimeClassifier()
    wrapped = RegimeCapturingClassifier(real)
    # Real classifier with no features returns UNKNOWN — exact regime depends
    # on internal heuristics; we only care that the wrapper captured something.
    snap = wrapped.classify([], None)
    assert wrapped.last_snapshot is snap
    assert isinstance(snap, RegimeSnapshot)
