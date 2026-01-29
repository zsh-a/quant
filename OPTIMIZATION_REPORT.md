I have optimized the live trading system based on `ARCHITECTURE_AND_OPTIMIZATION.md`.

**Changes Implemented:**

1.  **Realtime Data Stream (`src/core/data_stream.py`):**
    *   Implemented `RealtimeDataStream` class.
    *   It polls market data (using `akshare`) and yields `Bar` objects.
    *   `next_bar()` blocks until the next minute interval to prevent tight loops, as recommended ("block in live mode").

2.  **Live Broker State Synchronization (`src/core/live_broker.py`):**
    *   Added `sync_state()` method to `LiveBroker`.
    *   Calls `sync_state()` in `__init__` to populate `self.positions` immediately upon startup.
    *   This ensures the broker (and strategy) is aware of the actual account positions before generating new signals, addressing the "Status recovery" and "Signal consistency" points.

3.  **Server Integration (`src/api/server.py`):**
    *   Updated `run_session` to use `RealtimeDataStream` when `mode="live"`.
    *   This replaces the previous behavior of using `DBDataStream` (historical data) for live sessions.

**Files Modified:**
*   `src/core/data_stream.py`
*   `src/core/live_broker.py`
*   `src/api/server.py`

These changes align the live trading backend with the architectural recommendations.