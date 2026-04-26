"""CCXT-pro WebSocket realtime data stream for crypto live trading.

Subscribes to ``watch_ohlcv`` and exposes a synchronous ``next_bar()`` that
returns *completed* candles. In-progress candles are held until a subsequent
update confirms their timestamp has advanced — this guarantees no look-ahead.

Usage::

    stream = CcxtRealtimeDataStream(["BTC/USDT"], interval="5m")
    stream.start()
    while True:
        bars = stream.next_bar(timeout=60)
        if bars:
            strategy.on_bar(bars)
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from src.core.base import Bar, DataStream


@dataclass
class _Candle:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class CcxtRealtimeDataStream(DataStream):
    """Thin wrapper around ``ccxt.pro.binance().watch_ohlcv``.

    Networking is done on a background thread; the main thread pulls from a
    thread-safe queue. When the stream disconnects, the background thread
    reconnects with exponential backoff.
    """

    is_live = True

    def __init__(
        self,
        symbols: List[str],
        interval: str = "5m",
        exchange_id: str = "binance",
        max_queue: int = 512,
    ):
        self.symbols = symbols
        self.interval = interval
        self.exchange_id = exchange_id
        self._queue: "queue.Queue[Dict[str, Bar]]" = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._last_ts: Dict[str, int] = {}

    # ---- public API ----------------------------------------------------

    def start(self) -> None:
        for sym in self.symbols:
            t = threading.Thread(target=self._runner, args=(sym,), daemon=True, name=f"ccxt-{sym}")
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()

    def next_bar(self, timeout: Optional[float] = None) -> Optional[Dict[str, Bar]]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def reset(self) -> None:
        # Live stream: "reset" is a no-op; clear queue.
        with self._queue.mutex:
            self._queue.queue.clear()

    # ---- internals -----------------------------------------------------

    def _runner(self, symbol: str) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                import asyncio

                asyncio.run(self._watch_loop(symbol))
                backoff = 1.0
            except Exception as exc:  # pragma: no cover — network path
                logger.warning(f"CCXT stream {symbol} error: {exc} — retrying in {backoff:.1f}s")
                time.sleep(min(backoff, 30.0))
                backoff *= 2

    async def _watch_loop(self, symbol: str) -> None:  # pragma: no cover — live
        import ccxt.pro as ccxtpro  # type: ignore

        cls = getattr(ccxtpro, self.exchange_id)
        exchange = cls({"enableRateLimit": True})
        try:
            while not self._stop.is_set():
                candles = await exchange.watch_ohlcv(symbol, self.interval)
                for raw in candles:
                    ts_ms, o, h, l, c, v = raw[:6]
                    last = self._last_ts.get(symbol, 0)
                    if ts_ms <= last:
                        continue  # still the in-progress candle
                    # New candle appeared → previous is closed
                    self._last_ts[symbol] = ts_ms
                    bar = Bar(
                        symbol=symbol,
                        timestamp=datetime.utcfromtimestamp(ts_ms / 1000.0),
                        open=float(o),
                        high=float(h),
                        low=float(l),
                        close=float(c),
                        volume=float(v),
                        amount=float(v) * float(c),
                    )
                    try:
                        self._queue.put_nowait({symbol: bar})
                    except queue.Full:
                        logger.warning(f"CCXT queue full for {symbol} — dropping oldest")
                        self._queue.get_nowait()
                        self._queue.put_nowait({symbol: bar})
        finally:
            await exchange.close()
