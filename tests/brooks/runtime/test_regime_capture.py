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


def test_structure_view_caps_confirmed_swings():
    """structure_to_view_dict must cap confirmed_swings, otherwise per-bar
    BarEvent JSON grows O(N²) and tens of MB / GB session_logs."""
    from src.brooks.runtime.views import MAX_SWINGS_IN_VIEW, structure_to_view_dict

    class _Swing:
        def __init__(self, idx, price):
            self.bar_idx = idx
            self.price = price

    class _Struct:
        def __init__(self, n):
            self.always_in = "neutral"
            self.confirmed_swing_highs = [_Swing(i, 100 + i) for i in range(n)]
            self.confirmed_swing_lows = [_Swing(i, 50 + i) for i in range(n)]
            self.micro_channel_top = None
            self.micro_channel_bot = None
            self.last_breakout_lookback_high = None
            self.last_breakout_lookback_low = None

    view = structure_to_view_dict(_Struct(5_000))
    swings = view["confirmed_swings"]
    # At most MAX × 2 (highs + lows).
    assert len(swings) == MAX_SWINGS_IN_VIEW * 2
    # The kept entries are the *most recent* ones — first kept index =
    # 5000 - MAX_SWINGS_IN_VIEW.
    high_idxs = [s["idx"] for s in swings if s["kind"] == "high"]
    assert min(high_idxs) == 5_000 - MAX_SWINGS_IN_VIEW
    assert max(high_idxs) == 4_999


def test_works_with_real_classifier():
    """Sanity check: wrapping the real BrooksRegimeClassifier still classifies."""
    real = BrooksRegimeClassifier()
    wrapped = RegimeCapturingClassifier(real)
    # Real classifier with no features returns UNKNOWN — exact regime depends
    # on internal heuristics; we only care that the wrapper captured something.
    snap = wrapped.classify([], None)
    assert wrapped.last_snapshot is snap
    assert isinstance(snap, RegimeSnapshot)
