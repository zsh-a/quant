import os
import sys
from datetime import UTC, datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.alpha_lab.cli import build_parser, run_command
from src.alpha_lab.dataset import CryptoMinuteDatasetLoader
from src.alpha_lab.service import AlphaLabService


class FakeCryptoStore:
    def query_bars(self, provider, symbol, start_time, end_time, interval="1m"):
        base = [
            {"open_time": "2026-03-27T00:00:00+00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume_base": 10.0, "volume_quote": 1000.0, "trade_count": 10, "provider": provider, "market_type": "perpetual", "symbol": symbol, "exchange_symbol": symbol, "interval": interval, "close_time": "2026-03-27T00:00:59+00:00"},
            {"open_time": "2026-03-27T00:01:00+00:00", "open": 100.0, "high": 102.0, "low": 99.5, "close": 101.0, "volume_base": 11.0, "volume_quote": 1111.0, "trade_count": 11, "provider": provider, "market_type": "perpetual", "symbol": symbol, "exchange_symbol": symbol, "interval": interval, "close_time": "2026-03-27T00:01:59+00:00"},
            {"open_time": "2026-03-27T00:02:00+00:00", "open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0 if symbol == "BTCUSDT" else 99.0, "volume_base": 12.0, "volume_quote": 1224.0, "trade_count": 12, "provider": provider, "market_type": "perpetual", "symbol": symbol, "exchange_symbol": symbol, "interval": interval, "close_time": "2026-03-27T00:02:59+00:00"},
        ]
        return base


def test_dataset_loader_builds_tensor_matrices():
    loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    dataset = loader.load(
        provider="bitget",
        symbols=["BTCUSDT", "ETHUSDT"],
        start_time=datetime(2026, 3, 27, 0, 0, tzinfo=UTC),
        end_time=datetime(2026, 3, 27, 0, 3, tzinfo=UTC),
    )

    assert dataset.shape() == (3, 2)
    assert dataset.fields["close"].shape == (3, 2)
    assert dataset.symbols == ["BTCUSDT", "ETHUSDT"]


def test_alpha_lab_service_evaluates_formula_from_db():
    service = AlphaLabService()
    service.dataset_loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    result = service.evaluate_formula_from_db(
        formula="CSRank(ts_mean(close, 2) - close)",
        provider="bitget",
        symbols=["BTCUSDT", "ETHUSDT"],
        start_time=datetime(2026, 3, 27, 0, 0, tzinfo=UTC),
        end_time=datetime(2026, 3, 27, 0, 3, tzinfo=UTC),
    )

    assert "metrics" in result
    assert result["dataset"]["shape"] == (3, 2)
    assert "rank_ic" in result["metrics"]


def test_alpha_lab_cli_search_db():
    service = AlphaLabService()
    service.dataset_loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    parser = build_parser()
    args = parser.parse_args(
        [
            "search-db",
            "--provider",
            "bitget",
            "--symbols",
            "BTCUSDT,ETHUSDT",
            "--start",
            "2026-03-27T00:00:00+00:00",
            "--end",
            "2026-03-27T00:03:00+00:00",
            "--generations",
            "2",
            "--no-persist",
            "--seed",
            "CSRank(ts_mean(close, 2) - close)",
        ]
    )
    result = run_command(args, service)

    assert "top_results" in result
    assert len(result["top_results"]) >= 1
    assert "splits" in result


def test_alpha_lab_search_persistence(tmp_path):
    service = AlphaLabService()
    service.dataset_loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    service.persistence = __import__("src.alpha_lab.persistence", fromlist=["AlphaLabPersistence"]).AlphaLabPersistence(str(tmp_path / "alpha_lab"))
    result = service.search_formulas_on_db(
        provider="bitget",
        symbols=["BTCUSDT", "ETHUSDT"],
        start_time=datetime(2026, 3, 27, 0, 0, tzinfo=UTC),
        end_time=datetime(2026, 3, 27, 0, 3, tzinfo=UTC),
        seeds=["CSRank(ts_mean(close, 2) - close)"],
        generations=1,
        persist=True,
    )

    assert "persistence" in result
    assert os.path.exists(result["persistence"]["run_path"])
    assert len(result["persistence"]["zoo_paths"]) >= 1
    assert "lineage" in result


def test_alpha_lab_run_and_zoo_inspection(tmp_path):
    service = AlphaLabService()
    service.dataset_loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    service.persistence = __import__("src.alpha_lab.persistence", fromlist=["AlphaLabPersistence"]).AlphaLabPersistence(str(tmp_path / "alpha_lab"))
    result = service.search_formulas_on_db(
        provider="bitget",
        symbols=["BTCUSDT", "ETHUSDT"],
        start_time=datetime(2026, 3, 27, 0, 0, tzinfo=UTC),
        end_time=datetime(2026, 3, 27, 0, 3, tzinfo=UTC),
        seeds=["CSRank(ts_mean(close, 2) - close)"],
        generations=2,
        persist=True,
    )
    parser = build_parser()
    runs = run_command(parser.parse_args(["list-runs", "--limit", "5"]), service)
    loaded = run_command(parser.parse_args(["show-run", "--run-id", result["persistence"]["run_id"]]), service)
    zoo = run_command(parser.parse_args(["list-zoo", "--limit", "5"]), service)
    lineage = run_command(parser.parse_args(["lineage", "--run-id", result["persistence"]["run_id"]]), service)

    assert len(runs["runs"]) >= 1
    assert loaded["run_id"] == result["persistence"]["run_id"]
    assert len(zoo["zoo"]) >= 1
    assert lineage["run_id"] == result["persistence"]["run_id"]
