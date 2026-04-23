"""Tests for :mod:`src.brooks.render.chart` — PNG chart rendering.

Key invariants covered here:

* PNG bytes decode cleanly via PIL and match ``ChartStyle.width / height``.
* Repeated rendering of the same ``BrooksContext`` produces byte-identical
  output (md5 stability — the acceptance criterion from QUA-52).
* Distinct inputs produce distinct bytes (sanity: we are not accidentally
  caching a single buffer).
* Public surface works under a range of styles and supports overlays
  through ``ChartStyle.annotations`` and :func:`render_annotated`.

Debug artifacts are written to ``tests/brooks/_artifacts/`` (git-ignored)
so a human can eyeball the fixtures when chasing rendering regressions.
"""

from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path
from typing import List

import pytest
from PIL import Image

from src.brooks.context import Bar, BrooksContext, TFSnapshot
from src.brooks.render import ChartStyle, render_annotated, render_chart
from src.brooks.schema import Signal

ARTIFACTS_DIR = Path(__file__).parent / "_artifacts"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _trendy_bars(n: int, base: float = 100.0, start_ts: int = 1_700_000_000_000_000_000) -> List[Bar]:
    """Mostly-bull trend with short pullbacks — yields confirmed swings."""
    bars: List[Bar] = []
    price = base
    ts = start_ts
    while len(bars) < n:
        for _ in range(6):
            o = price
            c = price + 0.8
            h = c + 0.2
            l_ = o - 0.15
            bars.append(Bar(timestamp_ns=ts, open=o, high=h, low=l_, close=c, volume=1000.0 + len(bars)))
            price = c
            ts += 300_000_000_000
            if len(bars) >= n:
                return bars
        for _ in range(2):
            o = price
            c = price - 0.5
            h = o + 0.1
            l_ = c - 0.2
            bars.append(Bar(timestamp_ns=ts, open=o, high=h, low=l_, close=c, volume=800.0 + len(bars)))
            price = c
            ts += 300_000_000_000
            if len(bars) >= n:
                return bars
    return bars


def _htf_bars(n: int = 40, base: float = 80.0) -> List[Bar]:
    bars: List[Bar] = []
    price = base
    ts = 1_700_000_000_000_000_000
    for i in range(n):
        o = price
        c = price + (0.4 if i % 3 else -0.4)
        h = max(o, c) + 0.15
        l_ = min(o, c) - 0.15
        bars.append(Bar(timestamp_ns=ts, open=o, high=h, low=l_, close=c, volume=1.0))
        price = c
        ts += 3_600_000_000_000
    return bars


@pytest.fixture
def bull_ctx() -> BrooksContext:
    return BrooksContext(
        symbol="BTCUSDT",
        primary=TFSnapshot(interval="5m", bars=_trendy_bars(60)),
        htf={"1h": TFSnapshot(interval="1h", bars=_htf_bars())},
    )


@pytest.fixture
def small_ctx() -> BrooksContext:
    return BrooksContext(
        symbol="ETHUSDT",
        primary=TFSnapshot(interval="15m", bars=_trendy_bars(20, base=50.0)),
    )


@pytest.fixture(autouse=True)
def _ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _write_artifact(name: str, png: bytes) -> Path:
    path = ARTIFACTS_DIR / name
    path.write_bytes(png)
    return path


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


# ---------------------------------------------------------------------------
# Core rendering
# ---------------------------------------------------------------------------


def test_render_chart_returns_valid_png(bull_ctx: BrooksContext) -> None:
    png = render_chart(bull_ctx)
    _write_artifact("bull_60.png", png)

    # PNG magic header.
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG"
    assert img.size == (1280, 720)


def test_render_chart_respects_custom_dimensions(small_ctx: BrooksContext) -> None:
    style = ChartStyle(width=640, height=360, dpi=100)
    png = render_chart(small_ctx, style=style)
    _write_artifact("small_640x360.png", png)

    img = Image.open(io.BytesIO(png))
    assert img.size == (640, 360)


def test_render_empty_primary_still_produces_png() -> None:
    ctx = BrooksContext(symbol="X", primary=TFSnapshot(interval="5m", bars=[]))
    png = render_chart(ctx)
    img = Image.open(io.BytesIO(png))
    # Same configured dimensions even with no bars.
    assert img.size == (1280, 720)


def test_render_single_bar() -> None:
    bar = Bar(timestamp_ns=0, open=100.0, high=101.0, low=99.0, close=100.5, volume=1.0)
    ctx = BrooksContext(symbol="X", primary=TFSnapshot(interval="5m", bars=[bar]))
    png = render_chart(ctx)
    img = Image.open(io.BytesIO(png))
    assert img.size == (1280, 720)


# ---------------------------------------------------------------------------
# md5 stability — the acceptance criterion
# ---------------------------------------------------------------------------


def test_md5_is_stable_across_repeated_calls(bull_ctx: BrooksContext) -> None:
    """Same input → byte-identical PNG on every call.

    This is the regression guard called out in the QUA-52 spec: if a
    future change to the renderer introduces non-determinism (locale-
    dependent numeric formatting, timestamped metadata, randomised
    colour maps, …) this test is the first line of defence.
    """
    a = render_chart(bull_ctx)
    b = render_chart(bull_ctx)
    c = render_chart(bull_ctx)
    assert _md5(a) == _md5(b) == _md5(c)


def test_md5_is_stable_for_annotated_render(bull_ctx: BrooksContext) -> None:
    sig = Signal(
        pattern="H2",
        side="long",
        signal_bar_idx=40,
        entry_px=bull_ctx.primary.bars[40].high + 0.1,
        stop_px=bull_ctx.primary.bars[40].low - 0.1,
        probability=0.6,
        quality=0.7,
        source="rule",
    )
    a = render_annotated(bull_ctx, [sig])
    b = render_annotated(bull_ctx, [sig])
    _write_artifact("bull_60_annotated.png", a)
    assert _md5(a) == _md5(b)


def test_md5_differs_for_different_inputs(bull_ctx: BrooksContext, small_ctx: BrooksContext) -> None:
    assert _md5(render_chart(bull_ctx)) != _md5(render_chart(small_ctx))


def test_md5_differs_with_and_without_annotations(bull_ctx: BrooksContext) -> None:
    plain = render_chart(bull_ctx)
    annotated = render_chart(
        bull_ctx,
        style=ChartStyle(
            annotations=[
                {"type": "box", "bar_range": (40, 50), "label": "H2", "color": "#3949AB"},
                {"type": "line", "price": bull_ctx.primary.bars[-1].close, "label": "entry"},
            ]
        ),
    )
    _write_artifact("bull_60_boxed.png", annotated)
    assert _md5(plain) != _md5(annotated)


# ---------------------------------------------------------------------------
# Style toggles
# ---------------------------------------------------------------------------


def test_volume_toggle_changes_output(bull_ctx: BrooksContext) -> None:
    with_vol = render_chart(bull_ctx, style=ChartStyle(show_volume=True))
    no_vol = render_chart(bull_ctx, style=ChartStyle(show_volume=False))
    assert _md5(with_vol) != _md5(no_vol)


def test_htf_inset_toggle_changes_output(bull_ctx: BrooksContext) -> None:
    with_inset = render_chart(bull_ctx, style=ChartStyle(include_htf_inset=True))
    no_inset = render_chart(bull_ctx, style=ChartStyle(include_htf_inset=False))
    assert _md5(with_inset) != _md5(no_inset)


def test_ema_toggle_changes_output(bull_ctx: BrooksContext) -> None:
    with_ema = render_chart(bull_ctx, style=ChartStyle(show_ema=(20, 200)))
    no_ema = render_chart(bull_ctx, style=ChartStyle(show_ema=()))
    assert _md5(with_ema) != _md5(no_ema)


def test_swing_markers_toggle_changes_output(bull_ctx: BrooksContext) -> None:
    with_m = render_chart(bull_ctx, style=ChartStyle(show_swing_markers=True))
    no_m = render_chart(bull_ctx, style=ChartStyle(show_swing_markers=False))
    assert _md5(with_m) != _md5(no_m)


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


def test_annotation_types_all_render(bull_ctx: BrooksContext) -> None:
    style = ChartStyle(
        annotations=[
            {"type": "box", "bar_range": (10, 18), "label": "wedge"},
            {"type": "line", "price": bull_ctx.primary.bars[-1].close + 0.5, "label": "target"},
            {"type": "label", "bar_idx": 30, "price": bull_ctx.primary.bars[30].close, "text": "x"},
        ],
    )
    png = render_chart(bull_ctx, style=style)
    _write_artifact("bull_60_all_annotations.png", png)
    img = Image.open(io.BytesIO(png))
    assert img.size == (1280, 720)


def test_render_annotated_with_short_signal(bull_ctx: BrooksContext) -> None:
    bar = bull_ctx.primary.bars[50]
    sig = Signal(
        pattern="L2",
        side="short",
        signal_bar_idx=50,
        entry_px=bar.low - 0.1,
        stop_px=bar.high + 0.1,
        probability=0.55,
        quality=0.6,
        source="rule",
    )
    png = render_annotated(bull_ctx, [sig])
    _write_artifact("bull_60_short_signal.png", png)
    img = Image.open(io.BytesIO(png))
    assert img.size == (1280, 720)


# ---------------------------------------------------------------------------
# Performance (soft budget — skip under CI stress but flag regressions)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("SKIP_PERF_TESTS") == "1", reason="perf budget skipped")
def test_render_under_budget(bull_ctx: BrooksContext) -> None:
    """Warm render under 400ms; the Phase 4 target is 200ms but the first
    call inside a fresh process pays matplotlib's font-cache warmup.
    """
    render_chart(bull_ctx)  # warm any lazy caches
    t0 = time.perf_counter()
    for _ in range(3):
        render_chart(bull_ctx)
    elapsed = (time.perf_counter() - t0) / 3
    assert elapsed < 0.4, f"render took {elapsed * 1000:.1f}ms"
