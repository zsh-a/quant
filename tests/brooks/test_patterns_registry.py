"""PatternRegistry sanity tests."""

from __future__ import annotations

import pytest

from src.brooks.patterns import PatternRegistry
from src.brooks.patterns.base import PatternDetector
from src.brooks.patterns.h2_l2 import H2Detector
from src.brooks.patterns.wedge import WedgeLongDetector, WedgeShortDetector

EXPECTED = {
    "h2",
    "l2",
    "double_top",
    "double_bottom",
    "wedge_long",
    "wedge_short",
    "final_flag",
    "measured_move",
}


def test_registry_contains_all_expected_detectors():
    assert EXPECTED.issubset(set(PatternRegistry.all()))


def test_registry_class_name_attribute_matches_key():
    # Decorator assigns klass.name to the registry key.
    assert H2Detector.name == "h2"
    assert WedgeLongDetector.name == "wedge_long"
    assert WedgeShortDetector.name == "wedge_short"


def test_registry_build_returns_instance_with_name():
    det = PatternRegistry.build("h2", max_leg_bars=12)
    assert isinstance(det, H2Detector)
    assert isinstance(det, PatternDetector)
    assert det.name == "h2"
    assert det.max_leg_bars == 12


def test_registry_build_unknown_raises():
    with pytest.raises(KeyError):
        PatternRegistry.build("nope")


def test_registry_duplicate_name_raises():
    with pytest.raises(ValueError):

        @PatternRegistry.register("h2")
        class _Dup(PatternDetector):
            def on_bar(self, ctx):
                return None
