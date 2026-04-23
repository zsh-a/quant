"""OHLCV → PNG renderer for :class:`BrooksContext`.

Pure-matplotlib (``Agg`` backend) implementation — no ``mplfinance`` or
GUI dependency. The renderer is the single chart surface consumed by the
VLM analyst, UI previews, and eval reports, and is designed to be:

* **Headless** — sets the ``Agg`` backend on import so importing does not
  require a display server.
* **Deterministic** — PNG bytes are stable across runs for a given input.
  We suppress matplotlib's default ``Software`` / ``Creation Time`` PNG
  metadata and pin a handful of ``rcParams`` so font and locator choices
  do not drift with the caller's environment.
* **Composable** — per-render style is a :class:`ChartStyle` dataclass;
  free-form overlays (pattern boxes, entry/stop lines, labels) go through
  ``style.annotations`` and are shared with :func:`render_annotated`.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import matplotlib

# Lock backend *before* pyplot is imported elsewhere. Safe to call
# repeatedly; the Agg backend is always available and requires no display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PatchCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.brooks.context import Bar, BrooksContext, TFSnapshot  # noqa: E402
from src.brooks.features import BarFeatureExtractor  # noqa: E402
from src.brooks.schema import Signal  # noqa: E402

__all__ = ["ChartStyle", "render_chart", "render_annotated"]


@dataclass
class ChartStyle:
    """Per-render visual configuration.

    ``annotations`` is a list of dicts describing overlays on the main
    axes. Supported entries:

    * ``{"type": "box", "bar_range": (i, j), "label": "H2", "color": "#..."}``
      — axis-aligned rectangle spanning bars ``i..j`` (inclusive) at the
      bar-range low/high; label rendered above the box.
    * ``{"type": "line", "price": 101.5, "label": "entry", "color": "..."}``
      — horizontal line across the entire main axis.
    * ``{"type": "label", "bar_idx": 42, "price": 101.5, "text": "..."}``
      — free-form text anchored at a bar/price.
    """

    width: int = 1280
    height: int = 720
    dpi: int = 100
    candle_up: str = "#26A69A"
    candle_down: str = "#EF5350"
    ema_color: str = "#FB8C00"
    grid_alpha: float = 0.1
    show_ema: Tuple[int, ...] = (20, 200)
    show_volume: bool = True
    show_swing_markers: bool = True
    include_htf_inset: bool = True
    annotations: Optional[List[dict]] = None
    background: str = "#FFFFFF"
    text_color: str = "#222222"
    wick_color: str = "#555555"


@dataclass
class _BarSeries:
    """Lightweight working buffer used by the renderer."""

    idx: List[int] = field(default_factory=list)
    opens: List[float] = field(default_factory=list)
    highs: List[float] = field(default_factory=list)
    lows: List[float] = field(default_factory=list)
    closes: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.idx)

    @classmethod
    def from_bars(cls, bars: Sequence[Bar]) -> "_BarSeries":
        s = cls()
        for i, b in enumerate(bars):
            s.idx.append(i)
            s.opens.append(b.open)
            s.highs.append(b.high)
            s.lows.append(b.low)
            s.closes.append(b.close)
            s.volumes.append(b.volume)
        return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_chart(ctx: BrooksContext, style: Optional[ChartStyle] = None) -> bytes:
    """Render ``ctx.primary`` as a PNG.

    Returns the raw bytes; callers save to disk or forward to an LLM
    provider as base64 at their discretion.
    """
    return _render(ctx, signals=[], style=style or ChartStyle())


def render_annotated(
    ctx: BrooksContext,
    signals: Sequence[Signal],
    style: Optional[ChartStyle] = None,
) -> bytes:
    """Render ``ctx`` with ``signals`` overlaid as entry/stop levels.

    Each signal contributes:

    * an entry horizontal line at ``signal.entry_px``
    * a stop horizontal line at ``signal.stop_px``
    * a short text label above the signal bar with the pattern name

    ``style.annotations`` are applied on top of signal overlays, so
    callers can add pattern bounding boxes alongside the automatic
    entry/stop lines.
    """
    return _render(ctx, signals=list(signals), style=style or ChartStyle())


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------


def _render(
    ctx: BrooksContext,
    signals: Sequence[Signal],
    style: ChartStyle,
) -> bytes:
    series = _BarSeries.from_bars(ctx.primary.bars)

    # Build deterministic figure. We size by pixel dimensions via dpi so
    # the output is exactly ``width × height`` regardless of the caller's
    # default rcParams.
    fig = plt.figure(
        figsize=(style.width / style.dpi, style.height / style.dpi),
        dpi=style.dpi,
        facecolor=style.background,
    )
    try:
        with _deterministic_rc(style):
            _draw(fig, ctx, series, signals, style)
            return _savefig_png(fig, style)
    finally:
        plt.close(fig)


def _draw(
    fig,
    ctx: BrooksContext,
    series: _BarSeries,
    signals: Sequence[Signal],
    style: ChartStyle,
) -> None:
    if style.show_volume and series.volumes and any(v > 0 for v in series.volumes):
        gs = fig.add_gridspec(
            nrows=2,
            ncols=1,
            height_ratios=[4, 1],
            hspace=0.04,
            left=0.05,
            right=0.98,
            top=0.95,
            bottom=0.08,
        )
        ax_price = fig.add_subplot(gs[0, 0])
        ax_vol = fig.add_subplot(gs[1, 0], sharex=ax_price)
    else:
        ax_price = fig.add_axes([0.05, 0.08, 0.93, 0.87])
        ax_vol = None

    ax_price.set_facecolor(style.background)
    ax_price.set_title(f"{ctx.symbol} · {ctx.primary.interval}", loc="left", fontsize=11)
    ax_price.grid(True, alpha=style.grid_alpha, linestyle="--", linewidth=0.5)
    ax_price.tick_params(axis="both", labelsize=9, colors=style.text_color)

    if len(series) == 0:
        ax_price.text(
            0.5,
            0.5,
            "no bars",
            transform=ax_price.transAxes,
            ha="center",
            va="center",
            color=style.text_color,
            fontsize=12,
        )
        if ax_vol is not None:
            ax_vol.set_facecolor(style.background)
            ax_vol.set_xticks([])
            ax_vol.set_yticks([])
        return

    _plot_candles(ax_price, series, style)
    _plot_emas(ax_price, series, style)
    if style.show_swing_markers:
        _plot_swing_markers(ax_price, ctx.primary, series, style)

    if ax_vol is not None:
        _plot_volume(ax_vol, series, style)

    # Signal overlays first so caller annotations can layer on top.
    for sig in signals:
        _plot_signal(ax_price, sig, series, style)

    if style.annotations:
        for ann in style.annotations:
            _plot_annotation(ax_price, ann, series, style)

    if style.include_htf_inset and ctx.htf:
        _plot_htf_inset(fig, ax_price, ctx.htf, style)

    _tighten_axes(ax_price, series, style)


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def _plot_candles(ax, series: _BarSeries, style: ChartStyle) -> None:
    """Wicks via LineCollection, bodies via PatchCollection — one draw
    call per layer is ~10× faster than per-bar ``ax.plot`` / ``Rectangle``."""
    n = len(series)
    if n == 0:
        return

    # Width of each candle body on the x axis. 0.7 leaves visible gaps
    # between bars even at small figures.
    body_w = 0.7
    half = body_w / 2.0

    wick_segments: List[List[Tuple[float, float]]] = []
    up_patches: List[Rectangle] = []
    down_patches: List[Rectangle] = []

    for i in range(n):
        x = series.idx[i]
        o = series.opens[i]
        h = series.highs[i]
        l_ = series.lows[i]
        c = series.closes[i]
        wick_segments.append([(x, l_), (x, h)])

        bottom = min(o, c)
        height = max(abs(c - o), 1e-9)
        patch = Rectangle((x - half, bottom), body_w, height, linewidth=0.5)
        if c >= o:
            up_patches.append(patch)
        else:
            down_patches.append(patch)

    ax.add_collection(
        LineCollection(
            wick_segments,
            colors=style.wick_color,
            linewidths=0.9,
            antialiased=True,
        )
    )
    if up_patches:
        ax.add_collection(
            PatchCollection(
                up_patches,
                facecolors=style.candle_up,
                edgecolors=style.candle_up,
                linewidths=0.5,
                antialiased=True,
            )
        )
    if down_patches:
        ax.add_collection(
            PatchCollection(
                down_patches,
                facecolors=style.candle_down,
                edgecolors=style.candle_down,
                linewidths=0.5,
                antialiased=True,
            )
        )


def _plot_emas(ax, series: _BarSeries, style: ChartStyle) -> None:
    if not style.show_ema or len(series) < 2:
        return
    closes = series.closes
    base_color = style.ema_color
    shades = _ema_shades(base_color, len(style.show_ema))
    for period, color in zip(style.show_ema, shades):
        if period < 2:
            continue
        ema_vals = _ema(closes, period)
        ax.plot(
            series.idx,
            ema_vals,
            linewidth=1.2,
            color=color,
            label=f"EMA{period}",
            antialiased=True,
        )


def _plot_swing_markers(ax, snap: TFSnapshot, series: _BarSeries, style: ChartStyle) -> None:
    swings = _collect_swings(snap, series)
    if not swings:
        return
    high_xs, high_ys = [], []
    low_xs, low_ys = [], []
    for s in swings:
        if s.kind == "high":
            high_xs.append(s.bar_idx)
            high_ys.append(s.price)
        else:
            low_xs.append(s.bar_idx)
            low_ys.append(s.price)
    if high_xs:
        ax.scatter(
            high_xs,
            high_ys,
            marker="v",
            s=24,
            color=style.candle_down,
            edgecolors="none",
            zorder=5,
        )
    if low_xs:
        ax.scatter(
            low_xs,
            low_ys,
            marker="^",
            s=24,
            color=style.candle_up,
            edgecolors="none",
            zorder=5,
        )


def _plot_volume(ax, series: _BarSeries, style: ChartStyle) -> None:
    ax.set_facecolor(style.background)
    ax.grid(True, alpha=style.grid_alpha, linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=8, colors=style.text_color)
    colors = [style.candle_up if c >= o else style.candle_down for o, c in zip(series.opens, series.closes)]
    ax.bar(series.idx, series.volumes, width=0.7, color=colors, linewidth=0)
    ax.set_ylabel("vol", fontsize=8, color=style.text_color)
    ax.margins(x=0.01)
    # Keep y-axis minimal — the relative shape matters, not absolute levels.
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))


def _plot_signal(ax, sig: Signal, series: _BarSeries, style: ChartStyle) -> None:
    if not series:
        return
    entry_color = "#1E88E5" if sig.side == "long" else "#D81B60"
    stop_color = "#8E24AA"
    ax.axhline(sig.entry_px, color=entry_color, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.axhline(sig.stop_px, color=stop_color, linewidth=1.0, linestyle=":", alpha=0.8)

    # Label anchored at the signal bar.
    bar_x = _clamp(sig.signal_bar_idx, 0, len(series) - 1)
    label = sig.pattern
    y = series.highs[bar_x] if sig.side == "long" else series.lows[bar_x]
    dy = (max(series.highs) - min(series.lows)) * 0.01
    offset = dy if sig.side == "long" else -dy
    ax.text(
        bar_x,
        y + offset,
        label,
        fontsize=8,
        color=entry_color,
        ha="center",
        va="bottom" if sig.side == "long" else "top",
    )


def _plot_annotation(ax, ann: dict, series: _BarSeries, style: ChartStyle) -> None:
    kind = ann.get("type")
    if kind == "box":
        i, j = ann["bar_range"]
        i = _clamp(i, 0, len(series) - 1)
        j = _clamp(j, 0, len(series) - 1)
        if j < i:
            i, j = j, i
        lo = min(series.lows[i : j + 1])
        hi = max(series.highs[i : j + 1])
        color = ann.get("color", "#3949AB")
        rect = Rectangle(
            (i - 0.5, lo),
            (j - i + 1),
            hi - lo,
            facecolor="none",
            edgecolor=color,
            linewidth=1.2,
            linestyle="--",
        )
        ax.add_patch(rect)
        label = ann.get("label")
        if label:
            ax.text(
                (i + j) / 2,
                hi,
                label,
                color=color,
                fontsize=8,
                ha="center",
                va="bottom",
            )
    elif kind == "line":
        price = float(ann["price"])
        color = ann.get("color", "#424242")
        ax.axhline(price, color=color, linewidth=1.0, linestyle="-.", alpha=0.7)
        label = ann.get("label")
        if label:
            ax.text(
                len(series) - 1,
                price,
                f" {label}",
                color=color,
                fontsize=8,
                ha="left",
                va="center",
            )
    elif kind == "label":
        x = _clamp(int(ann["bar_idx"]), 0, len(series) - 1)
        price = float(ann["price"])
        ax.text(
            x,
            price,
            str(ann.get("text", "")),
            color=ann.get("color", style.text_color),
            fontsize=8,
            ha="center",
            va="center",
        )


def _plot_htf_inset(fig, ax_price, htf: dict, style: ChartStyle) -> None:
    """Embed the largest HTF snapshot as a thumbnail in the top-right
    corner of the main axes.
    """
    interval, snap = _pick_htf(htf)
    if snap is None or not snap.bars:
        return
    series = _BarSeries.from_bars(snap.bars)
    # Inset in axes coordinates: top-right, ~22% width × 30% height.
    inset = ax_price.inset_axes(
        [0.76, 0.66, 0.22, 0.32],
        facecolor=style.background,
    )
    inset.set_title(
        f"HTF {interval}",
        loc="left",
        fontsize=8,
        color=style.text_color,
    )
    inset.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    for spine in inset.spines.values():
        spine.set_edgecolor("#B0BEC5")
        spine.set_linewidth(0.8)
    _plot_candles(inset, series, style)
    _tighten_axes(inset, series, style, pad=0.01)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _tighten_axes(ax, series: _BarSeries, style: ChartStyle, pad: float = 0.02) -> None:
    if not series:
        return
    ax.set_xlim(-0.5, len(series) - 0.5)
    lo = min(series.lows)
    hi = max(series.highs)
    span = max(hi - lo, 1e-9)
    ax.set_ylim(lo - span * pad, hi + span * pad)


def _pick_htf(htf: dict[str, TFSnapshot]) -> Tuple[Optional[str], Optional[TFSnapshot]]:
    if not htf:
        return None, None
    # Pick the snapshot with the longest bar history deterministically:
    # ties broken by interval name.
    candidates = sorted(htf.items(), key=lambda kv: (-len(kv[1].bars), kv[0]))
    return candidates[0]


def _collect_swings(snap: TFSnapshot, series: _BarSeries):
    """Replay bars through the feature extractor to recover confirmed
    swings. If the snapshot already carries a structure object we use its
    cached swings; otherwise we rebuild them.
    """
    if snap.structure is not None:
        combined = list(snap.structure.confirmed_swing_highs) + list(snap.structure.confirmed_swing_lows)
        # Guard against swings whose bar_idx is out of range for our
        # pre-numbered series (caller-supplied structure may index
        # differently).
        return [s for s in combined if 0 <= s.bar_idx < len(series)]

    ext = BarFeatureExtractor()
    for bar in snap.bars:
        ext.on_bar(bar.timestamp_ns, bar.open, bar.high, bar.low, bar.close)
    return [s for s in ext.confirmed_swings if 0 <= s.bar_idx < len(series)]


def _ema(values: Sequence[float], period: int) -> List[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def _ema_shades(base: str, count: int) -> List[str]:
    """Return ``count`` visually distinct shades around ``base``.

    We keep this deterministic by returning the base colour first and
    then increasingly desaturated variants; for the default two-EMA case
    the colours are ``base`` and a darker derivative.
    """
    if count <= 0:
        return []
    r, g, b = _hex_to_rgb(base)
    shades = [base]
    for i in range(1, count):
        f = max(0.3, 1.0 - 0.35 * i)
        shades.append(_rgb_to_hex(r * f, g * f, b * f))
    return shades


def _hex_to_rgb(s: str) -> Tuple[float, float, float]:
    s = s.lstrip("#")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return r, g, b


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    def _c(x: float) -> int:
        return max(0, min(255, int(round(x * 255))))

    return f"#{_c(r):02X}{_c(g):02X}{_c(b):02X}"


def _clamp(v: int, lo: int, hi: int) -> int:
    if hi < lo:
        return lo
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------


class _deterministic_rc:
    """Context manager locking rc params that influence output bytes.

    Pinning these guarantees that :func:`render_chart` produces identical
    PNG bytes across environments, which is what the md5-stability test
    relies on.
    """

    _KEYS: dict[str, Any] = {
        "font.family": ["DejaVu Sans"],
        "font.size": 10.0,
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "svg.hashsalt": "brooks-render",
        "path.simplify": False,
    }

    def __init__(self, style: ChartStyle) -> None:
        self._style = style
        self._saved: dict[str, Any] = {}

    def __enter__(self) -> "_deterministic_rc":
        for k, v in self._KEYS.items():
            self._saved[k] = matplotlib.rcParams[k]
            matplotlib.rcParams[k] = v
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for k, v in self._saved.items():
            matplotlib.rcParams[k] = v


def _savefig_png(fig, style: ChartStyle) -> bytes:
    buf = io.BytesIO()
    # Explicitly null out the metadata keys matplotlib writes by default
    # (``Software`` = "Matplotlib version...", ``Creation Time`` = now())
    # so PNG bytes are stable across runs and environments.
    fig.savefig(
        buf,
        format="png",
        dpi=style.dpi,
        facecolor=fig.get_facecolor(),
        metadata={"Software": None, "Creation Time": None, "Source": None},
    )
    return buf.getvalue()


# Convenience for callers iterating many bars — not part of the public
# surface but exported as an implementation detail.
def _bar_iter(bars: Iterable[Bar]):  # pragma: no cover - not used directly
    for i, b in enumerate(bars):
        yield i, b
