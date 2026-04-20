"""L1 BarFeatureExtractor tests — focus on look-ahead free swings."""

from __future__ import annotations

from src.analysis.brooks.features import BarFeatureExtractor


def test_swing_confirmed_after_k_bars_only():
    ext = BarFeatureExtractor(swing_k=2, atr_period=3, breakout_lookback=3)
    # a simple peak at idx 3
    bars = [
        (1, 1.0, 1.1, 0.9, 1.0),
        (2, 1.05, 1.2, 1.0, 1.15),
        (3, 1.15, 1.3, 1.1, 1.25),
        (4, 1.25, 1.6, 1.15, 1.4),  # pivot: high = 1.6
        (5, 1.35, 1.45, 1.2, 1.25),
        (6, 1.2, 1.3, 1.05, 1.1),
    ]
    for ts, o, h, l, c in bars:
        ext.on_bar(ts, o, h, l, c)
    # Only the swing at idx 3 should be confirmed, confirmed_at_idx == 5
    highs = [s for s in ext.confirmed_swings if s.kind == "high"]
    assert len(highs) == 1
    assert highs[0].bar_idx == 3
    assert highs[0].confirmed_at_idx == 5
    assert abs(highs[0].price - 1.6) < 1e-9


def test_no_swing_before_k_bars_lag():
    ext = BarFeatureExtractor(swing_k=3)
    # 5 bars shouldn't produce any swing (2*K+1 = 7 minimum needed)
    for i in range(5):
        ext.on_bar(i, 100.0 + i, 100.5 + i, 99.5 + i, 100.0 + i)
    assert ext.confirmed_swings == []


def test_breakout_flags():
    ext = BarFeatureExtractor(swing_k=2, atr_period=3, breakout_lookback=5)
    # 5 bars around 100, then a break-up
    for i in range(5):
        ext.on_bar(i, 100.0, 100.5, 99.5, 100.0)
    feat = ext.on_bar(5, 100.5, 102.0, 100.2, 101.9)
    assert feat.is_breakout_up_20 is True
    assert feat.is_breakout_down_20 is False


def test_consecutive_bars_counter():
    ext = BarFeatureExtractor(swing_k=2, atr_period=3, breakout_lookback=5)
    for i in range(3):
        ext.on_bar(i, 100.0, 101.0, 99.9, 100.5)  # bull
    f = ext.on_bar(3, 100.5, 101.5, 100.4, 101.0)
    assert f.consecutive_bull_bars == 4
    f2 = ext.on_bar(4, 101.0, 101.2, 100.0, 100.1)  # bear
    assert f2.consecutive_bear_bars == 1
    assert f2.consecutive_bull_bars == 0
