import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.market_data.ccxt_adapter import CcxtCryptoDataAdapter, PROVIDER_SPECS
from src.market_data.crypto_cli import build_parser, run_command
from src.market_data.crypto_pipeline import CryptoMinuteSyncService
from src.market_data.crypto_store import CryptoMinuteBarStore, UnifiedMinuteBar
from src.market_data.crypto_sync_state import CryptoSyncStateStore


class FakeStore:
    def __init__(self, existing_times=None):
        self.ensured = False
        self.inserted = []
        self.instrument_rows = []
        self.existing_times = set(existing_times or [])

    def ensure_schema(self):
        self.ensured = True

    def get_latest_open_time(self, provider, symbol, interval="1m", market_type=None):
        matching = sorted(self.existing_times)
        return matching[-1] if matching else None

    def iter_missing_windows(
        self,
        provider,
        symbol,
        start_time,
        end_time,
        interval="1m",
        market_type=None,
        batch_size=1000,
    ):
        step = timedelta(minutes=1)
        existing = sorted(ts for ts in self.existing_times if start_time <= ts <= end_time)
        missing_ranges = CryptoMinuteBarStore._find_missing_ranges(start_time, end_time, existing, step)
        return CryptoMinuteBarStore._split_missing_ranges(missing_ranges, step, batch_size)

    def insert_bars(self, bars):
        inserted = 0
        for bar in bars:
            if bar.open_time in self.existing_times:
                continue
            self.existing_times.add(bar.open_time)
            self.inserted.append(bar)
            inserted += 1
        return inserted

    def upsert_instruments(self, rows):
        self.instrument_rows.extend(rows)
        return len(rows)

    def query_bars(self, provider, symbol, start_time, end_time, interval="1m"):
        return [{"provider": provider, "symbol": symbol, "interval": interval}]

    def get_overview(self):
        return {"row_count": len(self.existing_times), "instrument_count": len(self.instrument_rows)}

    def get_coverage(self, interval="1m", limit=100):
        return [{"interval": interval, "row_count": len(self.existing_times)}]


class FakeBitgetAdapter:
    provider_name = "bitget"
    market_type = "perpetual"
    batch_limit = 200

    def __init__(self):
        self.fetch_calls = []

    def describe(self):
        return {
            "provider": self.provider_name,
            "market_type": self.market_type,
            "intervals": ["1m"],
            "batch_limit": self.batch_limit,
        }

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.upper()

    def build_instruments(self, symbols):
        now = datetime.now(UTC)
        rows = []
        for symbol in symbols:
            rows.append(
                {
                    "provider": self.provider_name,
                    "market_type": self.market_type,
                    "symbol": symbol.upper(),
                    "exchange_symbol": symbol.upper(),
                    "base_asset": symbol[:-4],
                    "quote_asset": symbol[-4:],
                    "is_active": 1,
                    "updated_at": now,
                }
            )
        return rows

    def fetch_bars(self, symbol, interval, start_time, end_time, limit=None):
        self.fetch_calls.append((symbol, start_time, end_time, limit))
        rows = []
        cursor = start_time
        while cursor <= end_time and len(rows) < int(limit or self.batch_limit):
            rows.append(
                UnifiedMinuteBar(
                    provider=self.provider_name,
                    market_type=self.market_type,
                    symbol=symbol.upper(),
                    exchange_symbol=symbol.upper(),
                    interval=interval,
                    open_time=cursor,
                    close_time=cursor + timedelta(minutes=1) - timedelta(milliseconds=1),
                    open=100.0,
                    high=101.0,
                    low=99.5,
                    close=100.5,
                    volume_base=12.0,
                    volume_quote=1206.0,
                    trade_count=0,
                )
            )
            cursor += timedelta(minutes=1)
        return rows


def test_ccxt_adapter_market_loading_skips_fetch_currencies():
    adapter = CcxtCryptoDataAdapter(PROVIDER_SPECS["bitget"])

    class FakeClient:
        def __init__(self):
            self.markets = {}
            self.markets_by_id = {}

        def load_markets(self):
            raise AssertionError("load_markets should not be used")

        def fetch_markets(self):
            return [
                {
                    "id": "BTCUSDT",
                    "symbol": "BTC/USDT:USDT",
                    "base": "BTC",
                    "quote": "USDT",
                    "settle": "USDT",
                    "swap": True,
                    "future": False,
                    "spot": False,
                    "active": True,
                }
            ]

        def set_markets(self, markets, currencies=None):
            indexed = {market["symbol"]: market for market in markets}
            self.markets = indexed
            self.markets_by_id = {market["id"]: market for market in markets}
            return indexed

    adapter.client = FakeClient()

    market = adapter.resolve_market("BTCUSDT")

    assert market["symbol"] == "BTC/USDT:USDT"

def test_crypto_store_missing_ranges_split():
    start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 1, 0, 5, tzinfo=UTC)
    existing = [
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
    ]

    missing_ranges = CryptoMinuteBarStore._find_missing_ranges(
        start_time=start,
        end_time=end,
        existing_times=existing,
        step=timedelta(minutes=1),
    )
    windows = list(
        CryptoMinuteBarStore._split_missing_ranges(
            missing_ranges=missing_ranges,
            step=timedelta(minutes=1),
            batch_size=2,
        )
    )

    assert missing_ranges == [
        (datetime(2024, 1, 1, 0, 1, tzinfo=UTC), datetime(2024, 1, 1, 0, 1, tzinfo=UTC)),
        (datetime(2024, 1, 1, 0, 3, tzinfo=UTC), datetime(2024, 1, 1, 0, 4, tzinfo=UTC)),
    ]
    assert windows[0].expected_points == 1
    assert windows[1].start_time == datetime(2024, 1, 1, 0, 3, tzinfo=UTC)
    assert windows[1].end_time == datetime(2024, 1, 1, 0, 4, tzinfo=UTC)


def test_crypto_sync_service_fetches_only_missing_windows():
    existing_times = [
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 3, tzinfo=UTC),
    ]
    store = FakeStore(existing_times=existing_times)
    adapter = FakeBitgetAdapter()
    service = CryptoMinuteSyncService(store=store)
    service.providers = {"bitget": adapter}

    result = service.sync_minute_bars(
        provider="bitget",
        symbols=["BTCUSDT"],
        start_time=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        end_time=datetime(2024, 1, 1, 0, 4, tzinfo=UTC),
    )

    assert store.ensured is True
    assert result[0]["inserted"] == 2
    assert len(adapter.fetch_calls) == 2
    assert adapter.fetch_calls[0][1] == datetime(2024, 1, 1, 0, 2, tzinfo=UTC)
    assert adapter.fetch_calls[1][1] == datetime(2024, 1, 1, 0, 4, tzinfo=UTC)


def test_crypto_sync_service_is_idempotent_for_existing_range():
    store = FakeStore()
    adapter = FakeBitgetAdapter()
    service = CryptoMinuteSyncService(store=store)
    service.providers = {"bitget": adapter}

    first = service.sync_minute_bars(
        provider="bitget",
        symbols=["BTCUSDT"],
        start_time=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        end_time=datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
    )
    second = service.sync_minute_bars(
        provider="bitget",
        symbols=["BTCUSDT"],
        start_time=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        end_time=datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
    )

    assert first[0]["inserted"] == 3
    assert second[0]["inserted"] == 0
    assert len(store.inserted) == 3


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

    results = service.sync_default_minute_bars(end_time=datetime(2024, 1, 1, 1, 0, tzinfo=UTC))
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
    bootstrap_result = service.bootstrap_default_dataset(
        provider="bitget",
        symbols=["BTCUSDT"],
        end_time=datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
    )

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
    assert sync_result[0]["inserted"] == 6
    assert overview["row_count"] >= 6


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

    result = service.backfill_history(
        provider="bitget",
        end_time=datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
    )
    parser = build_parser()
    cli_args = parser.parse_args(
        [
            "backfill",
            "--provider",
            "bitget",
            "--start",
            "2024-01-01T00:00:00+00:00",
            "--end",
            "2024-01-01T00:02:00+00:00",
        ]
    )
    cli_result = run_command(cli_args, service)

    assert result["status"] == "backfilled"
    assert result["sync_results"][0]["inserted"] == 3
    assert cli_result["status"] == "backfilled"


def test_crypto_backfill_skips_existing_windows(tmp_path):
    store = FakeStore(
        existing_times=[
            datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        ]
    )
    adapter = FakeBitgetAdapter()
    service = CryptoMinuteSyncService(store=store)
    service.providers = {"bitget": adapter}
    service.config.default_provider = "bitget"
    service.config.default_symbols = ["BTCUSDT"]
    service.config.default_interval = "1m"
    service.config.full_history_start = "2024-01-01T00:00:00+00:00"
    service.config.state_file = str(tmp_path / "crypto_sync_state.json")
    service.state_store = CryptoSyncStateStore(service.config.state_file)

    result = service.backfill_history(
        provider="bitget",
        end_time=datetime(2024, 1, 1, 0, 3, tzinfo=UTC),
    )

    assert result["status"] == "backfilled"
    assert result["sync_results"][0]["inserted"] == 2
    assert len(adapter.fetch_calls) == 1
    assert adapter.fetch_calls[0][1] == datetime(2024, 1, 1, 0, 2, tzinfo=UTC)


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
        end_time=datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
        progress_callback=progress_events.append,
    )
    state = service.state_store.get_sync_point("bitget", "perpetual", "BTCUSDT", "1m")

    assert result[0]["inserted"] == 3
    assert len(progress_events) == 1
    assert state["status"] == "success"
    assert "last_open_time" in state
    assert Path(service.config.state_file).exists()
