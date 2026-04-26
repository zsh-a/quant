"""Render before/after comparison charts for QUA-70.

For a given replay session, runs :class:`BrooksStrategy` twice (CF off /
CF on), captures every entry timestamp+side+price, then renders a
candlestick PNG with the entry markers overlaid for each variant. The
two PNGs are then stitched horizontally into one ``.png`` for the issue
comment.

Run via::

    PYTHONPATH=. uv run --extra gpu python scripts/brooks_context_filter_chart.py \
        --session trend_2025_01_15 --out data/qua70_chart_trend.png
"""

from __future__ import annotations

import argparse
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PatchCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from src.brooks.strategy import BrooksStrategy  # noqa: E402
from src.core.backtest_broker import BacktestBroker  # noqa: E402
from src.core.base import Bar, DataStream  # noqa: E402
from src.core.engine import TradingEngine  # noqa: E402
from src.core.historical_crypto_stream import HistoricalCryptoStream  # noqa: E402


SESSIONS: Dict[str, Dict[str, Any]] = {
    "trend_2025_01_15": {
        "label": "Strong-trend day · 2025-01-15",
        "symbol": "BTCUSDT",
        "interval": "5m",
        "start": datetime(2025, 1, 15, 0, 0, tzinfo=timezone.utc),
        "end": datetime(2025, 1, 16, 0, 0, tzinfo=timezone.utc),
    },
    "range_2025_01_04": {
        "label": "Tight-range day · 2025-01-04",
        "symbol": "BTCUSDT",
        "interval": "5m",
        "start": datetime(2025, 1, 4, 0, 0, tzinfo=timezone.utc),
        "end": datetime(2025, 1, 5, 0, 0, tzinfo=timezone.utc),
    },
    "breakout_pullback_2025_01_20": {
        "label": "Breakout-and-fail day · 2025-01-20",
        "symbol": "BTCUSDT",
        "interval": "5m",
        "start": datetime(2025, 1, 20, 0, 0, tzinfo=timezone.utc),
        "end": datetime(2025, 1, 21, 0, 0, tzinfo=timezone.utc),
    },
}


class _CachedListStream(DataStream):
    is_live = False

    def __init__(self, bars: List[Bar]):
        self._bars = bars
        self._idx = 0

    def next_bar(self) -> Optional[Dict[str, Bar]]:
        if self._idx >= len(self._bars):
            return None
        b = self._bars[self._idx]
        self._idx += 1
        return {b.symbol: b}

    def reset(self) -> None:
        self._idx = 0


class _CapturingStrategy(BrooksStrategy):
    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.entries: List[Tuple[datetime, str, float]] = []
        self._current_bar_ts: Optional[datetime] = None

    def _process_symbol_bar(self, symbol, bar):  # type: ignore[override]
        self._current_bar_ts = bar.timestamp
        return super()._process_symbol_bar(symbol, bar)

    def _submit_entry(self, symbol, decision, risk_pct):  # type: ignore[override]
        ts = self._current_bar_ts or datetime.now(tz=timezone.utc)
        self.entries.append((ts, decision.side, float(decision.entry_px)))
        return super()._submit_entry(symbol, decision, risk_pct)


def _load_bars(session: Dict[str, Any]) -> List[Bar]:
    stream = HistoricalCryptoStream(
        symbol=session["symbol"],
        interval=session["interval"],
        start=session["start"],
        end=session["end"],
        provider="bitget",
    )
    out: List[Bar] = []
    while True:
        b = stream.next_bar()
        if b is None:
            break
        out.append(next(iter(b.values())))
    return out


def _make_strategy(*, context_filter_enabled: bool) -> _CapturingStrategy:
    return _CapturingStrategy(
        analyst="rule",
        analyst_params={
            "extractor_kwargs": {"breakout_lookback": 20, "swing_k": 2},
            "structure_kwargs": {"breakout_lookback": 20},
        },
        aggregator_params={"confluence_n": 1},
        context_filter_enabled=context_filter_enabled,
        te_params={"cost_r": 0.0},
        min_expected_r=0.1,
        sizer_params={"kind": "fixed", "risk_pct": 0.005},
    )


def _run(bars: List[Bar], *, context_filter_enabled: bool) -> _CapturingStrategy:
    strat = _make_strategy(context_filter_enabled=context_filter_enabled)
    broker = BacktestBroker(
        initial_cash=100_000, commission=0.0001, slippage=0.0, allow_short=True
    )
    engine = TradingEngine(strategy=strat, broker=broker, data_stream=_CachedListStream(bars))
    engine.run()
    return strat


def _draw_one(
    ax,
    bars: List[Bar],
    entries: List[Tuple[datetime, str, float]],
    title: str,
) -> None:
    times = [b.timestamp for b in bars]
    body_w_min = 4.0  # 5m bars; 4 minutes wide leaves gap
    body_w = body_w_min / (24.0 * 60.0)  # in matplotlib date units (days)
    half = body_w / 2.0

    wick_segments: List[List[Tuple[float, float]]] = []
    up_patches: List[Rectangle] = []
    down_patches: List[Rectangle] = []
    for b in bars:
        x = mdates.date2num(b.timestamp)
        wick_segments.append([(x, b.low), (x, b.high)])
        body_top = max(b.open, b.close)
        body_bot = min(b.open, b.close)
        body_h = max(body_top - body_bot, (b.high - b.low) * 1e-3)
        rect = Rectangle((x - half, body_bot), body_w, body_h)
        if b.close >= b.open:
            up_patches.append(rect)
        else:
            down_patches.append(rect)

    ax.add_collection(LineCollection(wick_segments, colors="#555555", linewidths=0.6, zorder=1))
    if up_patches:
        ax.add_collection(
            PatchCollection(
                up_patches,
                facecolor="#26A69A",
                edgecolor="#26A69A",
                linewidths=0.5,
                zorder=2,
            )
        )
    if down_patches:
        ax.add_collection(
            PatchCollection(
                down_patches,
                facecolor="#EF5350",
                edgecolor="#EF5350",
                linewidths=0.5,
                zorder=2,
            )
        )

    # Entry markers — placed slightly above (long) / below (short) the
    # signal bar so they don't disappear behind the candles. We use the
    # bar-range as the y reference rather than entry_px because the
    # stop-entry level is by construction at the candle's high/low and
    # would otherwise sit exactly on the wick.
    bar_by_ts = {b.timestamp: b for b in bars}
    ymin_chart = min(b.low for b in bars)
    ymax_chart = max(b.high for b in bars)
    pad = (ymax_chart - ymin_chart) * 0.012

    long_xs, long_ys = [], []
    short_xs, short_ys = [], []
    for t, side, _ in entries:
        b = bar_by_ts.get(t)
        x = mdates.date2num(t)
        if side == "long":
            y = (b.high if b else 0.0) + pad
            long_xs.append(x)
            long_ys.append(y)
        else:
            y = (b.low if b else 0.0) - pad
            short_xs.append(x)
            short_ys.append(y)
    if long_xs:
        ax.scatter(long_xs, long_ys, marker="^", s=120, color="#1565C0", edgecolor="white", linewidths=0.8, zorder=5, label=f"long entry × {len(long_xs)}")
    if short_xs:
        ax.scatter(short_xs, short_ys, marker="v", s=120, color="#AA00AA", edgecolor="white", linewidths=0.8, zorder=5, label=f"short entry × {len(short_xs)}")

    ax.set_title(title, fontsize=12, loc="left")
    ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.4)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(mdates.date2num(times[0]) - 2 * body_w, mdates.date2num(times[-1]) + 2 * body_w)
    ymin = min(b.low for b in bars)
    ymax = max(b.high for b in bars)
    pad = (ymax - ymin) * 0.03
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, choices=list(SESSIONS.keys()))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    session = SESSIONS[args.session]
    bars = _load_bars(session)

    old = _run(list(bars), context_filter_enabled=False)
    new = _run(list(bars), context_filter_enabled=True)

    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(15, 9), dpi=110, sharex=True, facecolor="white"
    )
    fig.suptitle(f"{session['label']}  ·  ContextFilter on/off entries", fontsize=14, y=0.995)
    _draw_one(
        axes[0],
        bars,
        old.entries,
        f"OLD — no ContextFilter — {len(old.entries)} entries submitted",
    )
    _draw_one(
        axes[1],
        bars,
        new.entries,
        f"NEW — ContextFilter ON — {len(new.entries)} entries submitted",
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}  (old entries={len(old.entries)}, new entries={len(new.entries)})")


if __name__ == "__main__":
    main()
