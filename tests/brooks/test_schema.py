"""Tests for src/brooks/schema.py — Signal, Decision, Order contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.brooks.schema import Decision, Order, Signal


def _valid_signal(**overrides) -> Signal:
    base = dict(
        pattern="h2",
        side="long",
        signal_bar_idx=42,
        entry_px=100.0,
        stop_px=98.0,
        target_px=104.0,
        probability=0.6,
        quality=0.8,
        reasoning="pullback to EMA20",
        source="rule:h2",
        meta={"latency_ms": 3},
    )
    base.update(overrides)
    return Signal(**base)


def test_signal_valid_and_one_r():
    sig = _valid_signal()
    assert sig.one_r == pytest.approx(2.0)
    assert sig.side == "long"
    assert sig.source == "rule:h2"
    assert sig.meta["latency_ms"] == 3


@pytest.mark.parametrize(
    "field, value, err_fragment",
    [
        ("entry_px", 0.0, "greater than 0"),
        ("entry_px", -1.0, "greater than 0"),
        ("stop_px", 0.0, "greater than 0"),
        ("probability", 1.5, "less than or equal to 1"),
        ("probability", -0.1, "greater than or equal to 0"),
        ("quality", 2.0, "less than or equal to 1"),
        ("side", "flat", "Input should be"),
    ],
)
def test_signal_invalid_field_values(field, value, err_fragment):
    with pytest.raises(ValidationError) as exc:
        _valid_signal(**{field: value})
    assert err_fragment in str(exc.value)


def test_signal_entry_must_differ_from_stop():
    with pytest.raises(ValidationError) as exc:
        _valid_signal(entry_px=100.0, stop_px=100.0)
    assert "entry_px must differ from stop_px" in str(exc.value)


def test_signal_round_trip():
    sig = _valid_signal()
    dumped = sig.model_dump()
    restored = Signal(**dumped)
    assert restored == sig
    json_bytes = sig.model_dump_json()
    restored_json = Signal.model_validate_json(json_bytes)
    assert restored_json == sig


def test_signal_json_schema_export():
    schema = Signal.model_json_schema()
    assert schema["title"] == "Signal"
    props = schema["properties"]
    for key in (
        "pattern",
        "side",
        "signal_bar_idx",
        "entry_px",
        "stop_px",
        "target_px",
        "probability",
        "quality",
        "source",
        "meta",
    ):
        assert key in props
    assert props["entry_px"]["exclusiveMinimum"] == 0
    assert props["probability"]["minimum"] == 0
    assert props["probability"]["maximum"] == 1


def test_decision_valid_with_nested_signals_round_trip():
    sig = _valid_signal()
    dec = Decision(
        symbol="BTCUSDT",
        side="long",
        entry_px=100.0,
        stop_px=98.0,
        target_px=106.0,
        quantity=1.5,
        probability=0.55,
        expected_r=1.2,
        regime="bull_trend",
        htf_aligned=True,
        signals=[sig],
        source="analyst.trend_follower",
        reasoning="aligned H2 with HTF uptrend",
    )
    dumped = dec.model_dump()
    restored = Decision(**dumped)
    assert restored == dec
    assert restored.signals[0].one_r == pytest.approx(2.0)


def test_decision_probability_bounds():
    with pytest.raises(ValidationError):
        Decision(
            symbol="BTCUSDT",
            side="long",
            entry_px=100.0,
            stop_px=98.0,
            target_px=106.0,
            probability=1.01,
            expected_r=1.0,
            regime="bull_trend",
            htf_aligned=True,
            source="analyst.trend_follower",
        )


def test_order_valid_round_trip_and_defaults():
    order = Order(
        symbol="AAPL",
        side="buy",
        quantity=10.0,
        type="limit",
        price=150.0,
        reason="decision entry",
        decision_ref="7b3f1e62-5d4e-4f56-9a8d-2e9e4a3b1c77",
    )
    dumped = order.model_dump()
    assert Order(**dumped) == order

    market_order = Order(symbol="AAPL", side="sell", quantity=1.0, type="market")
    assert market_order.price is None
    assert market_order.reason == ""
    assert market_order.decision_ref is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("quantity", 0.0),
        ("quantity", -5.0),
        ("side", "hold"),
        ("type", "twap"),
    ],
)
def test_order_invalid_field_values(field, value):
    kwargs = dict(symbol="AAPL", side="buy", quantity=1.0, type="market")
    kwargs[field] = value
    with pytest.raises(ValidationError):
        Order(**kwargs)
