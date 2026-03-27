import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datahub.binance import BinanceSpotDataAdapter
from src.datahub.bitget import BitgetDataAdapter
from src.market_data.crypto_cli import build_parser, run_command
from src.market_data.crypto_pipeline import CryptoMinuteSyncService
from src.market_data.crypto_sync_state import CryptoSyncStateStore
from src.market_data.crypto_store import UnifiedMinuteBar


class FakeStore:
    def __init__(self):
        self.ensured = False
        self.inserted = []
        self.instrument_rows = []

    def ensure_schema(self):
        self.ensured = True

    def get_latest_open_time(self, provider, symbol, interval="1m", market_type=None):
        return None

    def insert_bars(self, bars):
        batch = list(bars)
        self.inserted.extend(batch)
        return len(batch)

    def upsert_instruments(self, rows):
        self.instrument_rows.extend(rows)
        return len(rows)

    def query_bars(self, provider, symbol, start_time, end_time, interval="1m"):
        return [{"provider": provider, "symbol": symbol, "interval": interval}]

    def get_overview(self):
        return {"row_count": len(self.inserted), "instrument_count": len(self.instrument_rows)}

    def get_coverage(self, interval="1m", limit=100):
        return [{"interval": interval, "row_count": len(self.inserted)}]


class FakeBitgetAdapter:
    provider_name = "bitget"
    market_type = "perpetual"

    def fetch_candles(self, symbol, interval="1m", start_time_ms=None, end_time_ms=None, limit=1000):
        return [
            type(
                "Record",
                (),
                {
                    "timestamp_ms": 1700000000000,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.5,
                    "close": 100.5,
                    "volume_base": 12.0,
                    "volume_quote": 1206.0,
                },
            )()
        ]

    def fetch_history_candles(self, symbol, interval="1m", start_time_ms=None, end_time_ms=None, limit=200):
        return self.fetch_candles(
            symbol=symbol,
            interval=interval,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            limit=limit,
        )


def test_binance_adapter_parses_kline():
    adapter = BinanceSpotDataAdapter()
    row = [
        1700000000000,
        "1.0",
        "1.2",
        "0.9",
        "1.1",
        "100.0",
        1700000059999,
        "110.0",
        42,
    ]
    record = adapter._parse_kline(row)

    assert record.open_time_ms == 1700000000000
    assert record.close == 1.1
    assert record.trade_count == 42


def test_bitget_adapter_accepts_interval_alias(monkeypatch):
    adapter = BitgetDataAdapter()

    def fake_get(path, params):
        assert params["granularity"] == "1m"
        return {"code": "00000", "data": [["1700000000000", "1", "2", "0.5", "1.5", "10", "15"]]}

    monkeypatch.setattr(adapter, "_get", fake_get)
    records = adapter.fetch_candles("BTCUSDT", interval="1m")

    assert len(records) == 1
    assert records[0].close == 1.5


def test_bitget_adapter_fetch_history_candles(monkeypatch):
    adapter = BitgetDataAdapter()

    def fake_get(path, params):
        assert path == "/api/v2/mix/market/history-candles"
        assert params["granularity"] == "1m"
        assert params["limit"] == 200
        return {"code": "00000", "data": [["1700000000000", "1", "2", "0.5", "1.5", "10", "15"]]}

    monkeypatch.setattr(adapter, "_get", fake_get)
    records = adapter.fetch_history_candles("BTCUSDT", interval="1m", limit=999)

    assert len(records) == 1
    assert records[0].close == 1.5


def test_crypto_normalization_falls_back_to_symbol_when_exchange_symbol_missing():
    service = CryptoMinuteSyncService(store=FakeStore())
    service.providers = {"bitget": FakeBitgetAdapter()}
    adapter = service.providers["bitget"]
    record = type(
        "Record",
        (),
        {
            "timestamp_ms": 1700000000000,
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume_base": 12.0,
            "volume_quote": 1206.0,
            "exchange_symbol": None,
        },
    )()

    bars = service._normalize_records(adapter, "BTCUSDT", "1m", [record])

    assert bars[0].exchange_symbol == "BTCUSDT"


def test_crypto_sync_service_normalizes_and_inserts():
    store = FakeStore()
    service = CryptoMinuteSyncService(store=store)
    service.providers = {"bitget": FakeBitgetAdapter()}

    result = service.sync_minute_bars(
        provider="bitget",
        symbols=["BTCUSDT"],
        start_time=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        end_time=datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
    )

    assert store.ensured is True
    assert result[0]["inserted"] == 1
    assert isinstance(store.inserted[0], UnifiedMinuteBar)
    assert store.inserted[0].symbol == "BTCUSDT"
    assert store.instrument_rows[0]["base_asset"] == "BTC"


def test_crypto_sync_service_exposes_query_path():
    store = FakeStore()
    service = CryptoMinuteSyncService(store=store)
    service.ensure_schema()
    rows = service.store.query_bars(
        provider="bitget",
        symbol="BTCUSDT",
        start_time=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        end_time=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
    )

    assert rows[0]["symbol"] == "BTCUSDT"


def test_crypto_sync_service_overview_and_default_sync():
    store = FakeStore()
    service = CryptoMinuteSyncService(store=store)
    service.providers = {"bitget": FakeBitgetAdapter()}
    service.config.default_provider = "bitget"
    service.config.default_symbols = ["ETHUSDT"]
    service.config.default_interval = "1m"
    service.config.default_lookback_hours = 1

    results = service.sync_default_minute_bars()
    overview = service.get_overview()
    coverage = service.get_coverage()

    assert results[0]["symbol"] == "ETHUSDT"
    assert overview["instrument_count"] >= 1
    assert coverage[0]["interval"] == "1m"


def test_crypto_initialize_and_bootstrap():
    store = FakeStore()
    service = CryptoMinuteSyncService(store=store)
    service.providers = {"bitget": FakeBitgetAdapter()}
    service.config.default_provider = "bitget"
    service.config.default_symbols = ["BTCUSDT", "ETHUSDT"]
    service.config.default_interval = "1m"
    service.config.default_lookback_hours = 1

    init_result = service.initialize_database(provider="bitget")
    bootstrap_result = service.bootstrap_default_dataset(provider="bitget", symbols=["BTCUSDT"])

    assert init_result["status"] == "initialized"
    assert init_result["instrument_rows_written"] >= 2
    assert bootstrap_result["status"] == "bootstrapped"
    assert bootstrap_result["sync_results"][0]["symbol"] == "BTCUSDT"


def test_crypto_cli_commands():
    store = FakeStore()
    service = CryptoMinuteSyncService(store=store)
    service.providers = {"bitget": FakeBitgetAdapter()}
    service.config.default_provider = "bitget"
    service.config.default_symbols = ["BTCUSDT"]
    parser = build_parser()

    init_args = parser.parse_args(["init-db", "--provider", "bitget", "--symbols", "BTCUSDT,ETHUSDT"])
    init_result = run_command(init_args, service)

    sync_args = parser.parse_args(
        [
            "sync",
            "--provider",
            "bitget",
            "--symbols",
            "BTCUSDT",
            "--interval",
            "1m",
            "--start",
            "2024-01-01T00:00:00+00:00",
            "--end",
            "2024-01-01T00:05:00+00:00",
        ]
    )
    sync_result = run_command(sync_args, service)

    overview_args = parser.parse_args(["overview"])
    overview = run_command(overview_args, service)

    assert init_result["status"] == "initialized"
    assert sync_result[0]["inserted"] == 1
    assert overview["row_count"] >= 1


def test_crypto_backfill_history_and_cli(tmp_path):
    store = FakeStore()
    service = CryptoMinuteSyncService(store=store)
    service.providers = {"bitget": FakeBitgetAdapter()}
    service.config.default_provider = "bitget"
    service.config.default_symbols = ["BTCUSDT"]
    service.config.default_interval = "1m"
    service.config.full_history_start = "2024-01-01T00:00:00+00:00"
    service.config.state_file = str(tmp_path / "test_crypto_sync_state.json")
    service.state_store = CryptoSyncStateStore(service.config.state_file)

    result = service.backfill_history(provider="bitget")
    parser = build_parser()
    cli_args = parser.parse_args(["backfill", "--provider", "bitget", "--start", "2024-01-01T00:00:00+00:00"])
    cli_result = run_command(cli_args, service)

    assert result["status"] == "backfilled"
    assert result["sync_results"][0]["inserted"] == 1
    assert cli_result["status"] == "backfilled"


def test_crypto_backfill_advances_on_empty_historical_windows(tmp_path):
    store = FakeStore()

    class EmptyThenDataBitgetAdapter(FakeBitgetAdapter):
        def __init__(self):
            self.calls = 0

        def fetch_history_candles(self, symbol, interval="1m", start_time_ms=None, end_time_ms=None, limit=200):
            self.calls += 1
            if self.calls == 1:
                return []
            return super().fetch_history_candles(symbol, interval, start_time_ms, end_time_ms, limit)

    service = CryptoMinuteSyncService(store=store)
    service.providers = {"bitget": EmptyThenDataBitgetAdapter()}
    service.config.default_provider = "bitget"
    service.config.default_symbols = ["BTCUSDT"]
    service.config.default_interval = "1m"
    service.config.full_history_start = "2024-01-01T00:00:00+00:00"
    service.config.state_file = str(tmp_path / "crypto_sync_state.json")
    service.state_store = CryptoSyncStateStore(service.config.state_file)

    result = service.backfill_history(provider="bitget")

    assert result["status"] == "backfilled"
    assert result["sync_results"][0]["inserted"] == 1


def test_crypto_sync_state_resume_and_progress(tmp_path):
    store = FakeStore()
    service = CryptoMinuteSyncService(store=store)
    service.providers = {"bitget": FakeBitgetAdapter()}
    service.config.default_symbols = ["BTCUSDT"]
    service.config.state_file = str(tmp_path / "crypto_sync_state.json")
    service.state_store = CryptoSyncStateStore(service.config.state_file)

    progress_events = []
    result = service.sync_minute_bars(
        provider="bitget",
        symbols=["BTCUSDT"],
        start_time=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        end_time=datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        progress_callback=progress_events.append,
    )
    state = service.state_store.get_sync_point("bitget", "perpetual", "BTCUSDT", "1m")

    assert result[0]["inserted"] == 1
    assert len(progress_events) == 1
    assert state["status"] == "success"
    assert "last_open_time" in state
    assert Path(service.config.state_file).exists()
