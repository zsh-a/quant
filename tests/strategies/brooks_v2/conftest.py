"""Shared fixtures for Brooks v2 tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List

import pytest

from src.core.base import Bar


def mkbar(i: int, o: float, h: float, l: float, c: float, *, symbol: str = "BTCUSDT") -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=datetime(2025, 1, 1, 9, 0) + timedelta(minutes=5 * i),
        open=o,
        high=h,
        low=l,
        close=c,
        volume=1.0,
        amount=c,
    )


def make_series(values: Iterable[tuple]) -> List[Bar]:
    """``values`` = iterable of (o, h, l, c)."""
    out: List[Bar] = []
    for i, (o, h, l, c) in enumerate(values):
        out.append(mkbar(i, o, h, l, c))
    return out


@pytest.fixture
def synthetic_h2_bars() -> List[Bar]:
    """Hand-crafted bars designed to produce a valid H2 pattern.

    Bars 0-14: steady uptrend (always-in long)
    Bars 15-17: leg-1 down (bearish pullback)
    Bars 18-19: small recovery
    Bars 20-21: leg-2 takes out leg-1 low — leg-2 last bar is a bull reversal bar
    """
    vals: list[tuple] = []
    # Uptrend: 15 bars bullish, each +1.0 close
    price = 100.0
    for i in range(15):
        o = price
        c = price + 1.0
        h = c + 0.1
        l = o - 0.1
        vals.append((o, h, l, c))
        price = c
    # leg1 pullback 3 bars
    for _ in range(3):
        o = price
        c = price - 0.8
        h = o + 0.05
        l = c - 0.1
        vals.append((o, h, l, c))
        price = c
    leg1_low = price - 0.1
    # recovery 2 bars
    for _ in range(2):
        o = price
        c = price + 0.6
        h = c + 0.05
        l = o - 0.05
        vals.append((o, h, l, c))
        price = c
    # leg2: one bear bar then a bull reversal bar that pokes below leg1 then closes up
    o = price
    c = price - 0.9
    h = o + 0.02
    l = c - 0.05
    vals.append((o, h, l, c))
    price = c
    # Reversal bar — tail below leg1_low, close bullish
    o = price
    l_r = leg1_low - 0.05  # tail below leg1 low
    c_r = price + 0.6
    h_r = c_r + 0.05
    vals.append((o, h_r, l_r, c_r))
    return make_series(vals)


@pytest.fixture
def synthetic_l2_bars() -> List[Bar]:
    """Mirror: steady downtrend then H2-equivalent bear setup."""
    vals: list[tuple] = []
    price = 100.0
    for i in range(15):
        o = price
        c = price - 1.0
        h = o + 0.1
        l = c - 0.1
        vals.append((o, h, l, c))
        price = c
    for _ in range(3):
        o = price
        c = price + 0.8
        h = c + 0.1
        l = o - 0.05
        vals.append((o, h, l, c))
        price = c
    leg1_high = price + 0.1
    for _ in range(2):
        o = price
        c = price - 0.6
        h = o + 0.05
        l = c - 0.05
        vals.append((o, h, l, c))
        price = c
    o = price
    c = price + 0.9
    h = c + 0.05
    l = o - 0.02
    vals.append((o, h, l, c))
    price = c
    o = price
    h_r = leg1_high + 0.05
    c_r = price - 0.6
    l_r = c_r - 0.05
    vals.append((o, h_r, l_r, c_r))
    return make_series(vals)
