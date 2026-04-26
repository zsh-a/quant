"""ContextFilter on/off replay comparison on real BTCUSDT 5m sessions.

Runs :class:`BrooksStrategy` over three representative real-historical
sessions (strong trend / tight range / breakout-and-fail) twice each —
once with ``context_filter_enabled=False`` (legacy behaviour) and once
with the QUA-68 default ``context_filter_enabled=True`` — and reports a
metric table per session covering:

* total filled trades (buys + sell_shorts)
* aggregated signals (bars whose aggregator produced a decision)
* signals rejected by ContextFilter (only meaningful in the "new" run)
* win rate
* mean expected-R at entry
* max equity drawdown

The script writes a JSON report next to itself. It can be run via::

    uv run --extra gpu python scripts/brooks_context_filter_replay_compare.py \
        --out data/qua70_replay_compare.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.brooks.strategy import BrooksStrategy
from src.core.backtest_broker import BacktestBroker
from src.core.base import Bar, DataStream
from src.core.engine import TradingEngine
from src.core.historical_crypto_stream import HistoricalCryptoStream


SESSIONS: List[Dict[str, Any]] = [
    {
        "id": "trend_2025_01_15",
        "label": "Strong-trend day — 2025-01-15 (BTCUSDT, +4.1%)",
        "regime_hint": "strong bull trend, sustained leg up with shallow pullbacks",
        "symbol": "BTCUSDT",
        "interval": "5m",
        "start": datetime(2025, 1, 15, 0, 0, tzinfo=timezone.utc),
        "end": datetime(2025, 1, 16, 0, 0, tzinfo=timezone.utc),
    },
    {
        "id": "range_2025_01_04",
        "label": "Tight-range day — 2025-01-04 (BTCUSDT, ±0.6% chop)",
        "regime_hint": "tight trading range, no decisive break",
        "symbol": "BTCUSDT",
        "interval": "5m",
        "start": datetime(2025, 1, 4, 0, 0, tzinfo=timezone.utc),
        "end": datetime(2025, 1, 5, 0, 0, tzinfo=timezone.utc),
    },
    {
        "id": "breakout_pullback_2025_01_20",
        "label": "Breakout-and-fail day — 2025-01-20 (BTCUSDT, 110k spike → retrace)",
        "regime_hint": "broad range, strong morning breakout, afternoon failure-and-retrace",
        "symbol": "BTCUSDT",
        "interval": "5m",
        "start": datetime(2025, 1, 20, 0, 0, tzinfo=timezone.utc),
        "end": datetime(2025, 1, 21, 0, 0, tzinfo=timezone.utc),
    },
]


@dataclass
class _RunMetrics:
    label: str
    context_filter_enabled: bool
    bars: int = 0
    aggregated_signals: int = 0
    filtered_signals: int = 0
    entries_submitted: int = 0
    trades_filled: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_expected_r: float = 0.0
    max_drawdown_pct: float = 0.0
    final_equity: float = 0.0
    realised_pnl: float = 0.0
    expected_rs: List[float] = field(default_factory=list)
    pnl_per_trade: List[float] = field(default_factory=list)


class _CachedListStream(DataStream):
    """Replay a pre-loaded list of bars; lets us load once and run twice."""

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

    def total_bars(self) -> int:
        return len(self._bars)


class _InstrumentedStrategy(BrooksStrategy):
    """BrooksStrategy with hooks for counting signals / filtered / E[R]."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.aggregated_signals = 0
        self.filtered_signals = 0
        self.entries_submitted = 0
        self.entry_expected_rs: List[float] = []

        # Wrap aggregator.resolve so we count combined decisions per bar.
        _orig_resolve = self._aggregator.resolve

        def _resolve(signals):
            out = _orig_resolve(signals)
            if out is not None:
                self.aggregated_signals += 1
            return out

        self._aggregator.resolve = _resolve  # type: ignore[assignment]

        # Wrap context_filter.check so we count rejections (when enabled).
        if self._context_filter is not None:
            _orig_check = self._context_filter.check

            def _check(**kw):
                dec = _orig_check(**kw)
                if not dec.allow:
                    self.filtered_signals += 1
                return dec

            self._context_filter.check = _check  # type: ignore[assignment]

    def _submit_entry(self, symbol, decision, risk_pct):  # type: ignore[override]
        self.entries_submitted += 1
        self.entry_expected_rs.append(float(getattr(decision, "expected_r", 0.0) or 0.0))
        return super()._submit_entry(symbol, decision, risk_pct)


def _load_session_bars(session: Dict[str, Any]) -> List[Bar]:
    stream = HistoricalCryptoStream(
        symbol=session["symbol"],
        interval=session["interval"],
        start=session["start"],
        end=session["end"],
        provider="bitget",
    )
    bars: List[Bar] = []
    while True:
        b = stream.next_bar()
        if b is None:
            break
        bars.append(next(iter(b.values())))
    return bars


def _make_strategy(*, context_filter_enabled: bool) -> _InstrumentedStrategy:
    return _InstrumentedStrategy(
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


def _pair_trades(trades: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Pair entry/exit trades from broker.trades, FIFO, per symbol+side."""
    open_long: Dict[str, List[Dict[str, Any]]] = {}
    open_short: Dict[str, List[Dict[str, Any]]] = {}
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for t in trades:
        sym, ttype = t["symbol"], t["type"]
        if ttype == "buy":
            open_long.setdefault(sym, []).append(t)
        elif ttype == "sell_short":
            open_short.setdefault(sym, []).append(t)
        elif ttype == "sell":
            opens = open_long.get(sym, [])
            if opens:
                pairs.append((opens.pop(0), t))
        elif ttype == "buy_to_cover":
            opens = open_short.get(sym, [])
            if opens:
                pairs.append((opens.pop(0), t))
    return pairs


def _max_drawdown_pct(equity_history: List[Dict[str, Any]]) -> float:
    if not equity_history:
        return 0.0
    peak = -float("inf")
    max_dd = 0.0
    for row in equity_history:
        eq = float(row.get("total_equity", 0.0))
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd * 100.0


def _run_once(
    bars: List[Bar],
    *,
    context_filter_enabled: bool,
    label: str,
    initial_cash: float = 100_000.0,
) -> _RunMetrics:
    strat = _make_strategy(context_filter_enabled=context_filter_enabled)
    broker = BacktestBroker(
        initial_cash=initial_cash,
        commission=0.0001,
        slippage=0.0,
        allow_short=True,
    )
    engine = TradingEngine(strategy=strat, broker=broker, data_stream=_CachedListStream(bars))
    engine.run()

    pairs = _pair_trades(broker.trades)
    pnls: List[float] = []
    wins = losses = 0
    for entry, exit_ in pairs:
        if entry["type"] == "buy":
            pnl = (exit_["price"] - entry["price"]) * entry["quantity"]
        else:
            pnl = (entry["price"] - exit_["price"]) * entry["quantity"]
        pnl -= entry.get("commission", 0.0) + exit_.get("commission", 0.0)
        pnls.append(pnl)
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

    fill_count = sum(1 for t in broker.trades if t["type"] in ("buy", "sell_short"))
    win_rate = wins / max(1, wins + losses)
    avg_er = (sum(strat.entry_expected_rs) / len(strat.entry_expected_rs)) if strat.entry_expected_rs else 0.0
    final_equity = float(broker.get_account_info().get("total_equity", 0.0))

    return _RunMetrics(
        label=label,
        context_filter_enabled=context_filter_enabled,
        bars=len(bars),
        aggregated_signals=strat.aggregated_signals,
        filtered_signals=strat.filtered_signals,
        entries_submitted=strat.entries_submitted,
        trades_filled=fill_count,
        wins=wins,
        losses=losses,
        win_rate=round(win_rate, 4),
        avg_expected_r=round(avg_er, 4),
        max_drawdown_pct=round(_max_drawdown_pct(broker.equity_history), 4),
        final_equity=round(final_equity, 2),
        realised_pnl=round(sum(pnls), 2),
        expected_rs=[round(x, 4) for x in strat.entry_expected_rs],
        pnl_per_trade=[round(p, 2) for p in pnls],
    )


def _format_table(old: _RunMetrics, new: _RunMetrics) -> str:
    rows: List[Tuple[str, str, str]] = [
        ("Aggregated signals", str(old.aggregated_signals), str(new.aggregated_signals)),
        ("Filtered signals", "—", str(new.filtered_signals)),
        ("Entries submitted", str(old.entries_submitted), str(new.entries_submitted)),
        ("Trades filled", str(old.trades_filled), str(new.trades_filled)),
        ("Wins / Losses", f"{old.wins} / {old.losses}", f"{new.wins} / {new.losses}"),
        ("Win rate", f"{old.win_rate*100:.1f}%", f"{new.win_rate*100:.1f}%"),
        ("Avg expected R", f"{old.avg_expected_r:.3f}", f"{new.avg_expected_r:.3f}"),
        ("Max drawdown", f"{old.max_drawdown_pct:.2f}%", f"{new.max_drawdown_pct:.2f}%"),
        ("Realised PnL", f"{old.realised_pnl:.2f}", f"{new.realised_pnl:.2f}"),
        ("Final equity", f"{old.final_equity:.2f}", f"{new.final_equity:.2f}"),
    ]
    out: List[str] = []
    out.append("| Indicator | Old (no ContextFilter) | New (with ContextFilter) |")
    out.append("|---|---|---|")
    for name, a, b in rows:
        out.append(f"| {name} | {a} | {b} |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="QUA-70 ContextFilter replay comparison")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data") / "qua70_replay_compare.json",
        help="Where to write the JSON report",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "sessions": [],
    }

    for session in SESSIONS:
        print(f"\n=== {session['label']} ===")
        t0 = time.time()
        bars = _load_session_bars(session)
        print(f"loaded {len(bars)} bars in {time.time()-t0:.1f}s")

        old = _run_once(list(bars), context_filter_enabled=False, label=session["label"])
        new = _run_once(list(bars), context_filter_enabled=True, label=session["label"])

        print(_format_table(old, new))

        report["sessions"].append(
            {
                "id": session["id"],
                "label": session["label"],
                "regime_hint": session["regime_hint"],
                "symbol": session["symbol"],
                "interval": session["interval"],
                "start": session["start"].isoformat(),
                "end": session["end"].isoformat(),
                "bars": len(bars),
                "old": asdict(old),
                "new": asdict(new),
            }
        )

    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
