"""End-to-end integration test: strategy + broker + engine on synthetic bars."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.core.backtest_broker import BacktestBroker
from src.core.base import Bar, DataStream
from src.core.engine import TradingEngine
from src.strategies.brooks_v2.strategy import BrooksStrategyV2


class _ListStream(DataStream):
    def __init__(self, bars: List[Bar]):
        self._bars = bars
        self._i = 0

    def next_bar(self) -> Optional[Dict[str, Bar]]:
        if self._i >= len(self._bars):
            return None
        b = self._bars[self._i]
        self._i += 1
        return {b.symbol: b}

    def reset(self) -> None:
        self._i = 0


def _build_h2_scenario() -> List[Bar]:
    from datetime import datetime, timedelta

    base_ts = datetime(2025, 1, 1, 0, 0)

    def mk(i: int, o: float, h: float, l: float, c: float) -> Bar:
        return Bar(
            symbol="BTCUSDT",
            timestamp=base_ts + timedelta(minutes=5 * i),
            open=o,
            high=h,
            low=l,
            close=c,
            volume=1.0,
            amount=c,
        )

    vals: list[tuple] = []
    price = 100.0
    for _ in range(20):  # strong uptrend
        o = price
        c = price + 1.0
        h = c + 0.1
        l = o - 0.1
        vals.append((o, h, l, c))
        price = c
    for _ in range(3):
        o = price
        c = price - 0.8
        h = o + 0.05
        l = c - 0.1
        vals.append((o, h, l, c))
        price = c
    leg1_low = price - 0.1
    for _ in range(2):
        o = price
        c = price + 0.6
        h = c + 0.05
        l = o - 0.05
        vals.append((o, h, l, c))
        price = c
    o = price
    c = price - 0.9
    h = o + 0.02
    l = c - 0.05
    vals.append((o, h, l, c))
    price = c
    o = price
    l_r = leg1_low - 0.05
    c_r = price + 0.6
    h_r = c_r + 0.05
    vals.append((o, h_r, l_r, c_r))
    price = c_r
    # A follow-through bar that takes out the signal bar high (triggers entry)
    o = price
    c = price + 1.5
    h = c + 0.05
    l = o - 0.05
    vals.append((o, h, l, c))
    # Run a bit longer so position can be managed
    for _ in range(10):
        o = price
        c = price + 0.5
        h = c + 0.1
        l = o - 0.1
        vals.append((o, h, l, c))
        price = c

    return [mk(i, *vals[i]) for i in range(len(vals))]


def test_end_to_end_produces_trade():
    bars = _build_h2_scenario()
    strat = BrooksStrategyV2(
        db_client=None,
        session_id=None,
        breakout_lookback=10,
        swing_k=2,
        risk_pct=1.0,
        detectors_enabled=["h2"],
        confluence_n=1,
        use_mtf=False,
    )
    broker = BacktestBroker(initial_cash=100_000, commission=0.0001, slippage=0.0, allow_short=True)
    engine = TradingEngine(strategy=strat, broker=broker, data_stream=_ListStream(bars))
    engine.run()

    # Must have at least one entry trade.
    assert any(t["type"] == "buy" for t in broker.trades), f"no buy trade in {broker.trades}"


def test_reproducibility_same_seed_same_trades():
    bars = _build_h2_scenario()

    def run():
        strat = BrooksStrategyV2(
            db_client=None,
            session_id=None,
            seed=7,
            breakout_lookback=10,
            swing_k=2,
            risk_pct=1.0,
            detectors_enabled=["h2"],
            confluence_n=1,
            use_mtf=False,
        )
        broker = BacktestBroker(initial_cash=100_000, commission=0.0001, slippage=0.0, allow_short=True)
        engine = TradingEngine(strategy=strat, broker=broker, data_stream=_ListStream(bars))
        engine.run()
        return [(t["timestamp"], t["type"], round(t["price"], 4), round(t["quantity"], 4)) for t in broker.trades]

    assert run() == run()
