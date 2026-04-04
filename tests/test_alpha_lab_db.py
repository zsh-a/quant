import os
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.alpha.auto_runner import load_auto_search_config
from src.alpha.cli import build_parser, run_command
from src.alpha.dataset import CryptoMinuteDatasetLoader
from src.alpha.persistence import AlphaPersistence
from src.alpha.service import AlphaService

START = datetime(2026, 3, 27, 0, 0, tzinfo=UTC)
END = datetime(2026, 3, 27, 0, 19, tzinfo=UTC)


class FakeCryptoStore:
    def query_bars(self, provider, symbol, start_time, end_time, interval="1m"):
        rows = []
        start = pd.Timestamp("2026-03-27T00:00:00+00:00")
        for idx in range(20):
            open_time = start + pd.Timedelta(minutes=idx)
            close_base = 100.0 + idx if symbol == "BTCUSDT" else 100.0 - (idx * 0.5)
            open_price = close_base - 0.25
            high_price = close_base + 0.75
            low_price = close_base - 1.0
            volume_base = 10.0 + idx
            rows.append(
                {
                    "open_time": open_time.isoformat(),
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_base,
                    "volume_base": volume_base,
                    "volume_quote": close_base * volume_base,
                    "trade_count": 10 + idx,
                    "provider": provider,
                    "market_type": "perpetual",
                    "symbol": symbol,
                    "exchange_symbol": symbol,
                    "interval": interval,
                    "close_time": (open_time + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1)).isoformat(),
                }
            )
        start_ts = pd.Timestamp(start_time).tz_convert("UTC") if pd.Timestamp(start_time).tzinfo else pd.Timestamp(start_time, tz="UTC")
        end_ts = pd.Timestamp(end_time).tz_convert("UTC") if pd.Timestamp(end_time).tzinfo else pd.Timestamp(end_time, tz="UTC")
        return [row for row in rows if start_ts <= pd.Timestamp(row["open_time"]) <= end_ts]


def test_dataset_loader_builds_tensor_matrices():
    loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    dataset = loader.load(
        provider="bitget",
        symbols=["BTCUSDT", "ETHUSDT"],
        start_time=START,
        end_time=datetime(2026, 3, 27, 0, 2, tzinfo=UTC),
    )

    assert dataset.shape() == (3, 2)
    assert dataset.fields["close"].shape == (3, 2)
    assert dataset.symbols == ["BTCUSDT", "ETHUSDT"]
    assert np.all(dataset.session_mask)


def test_dataset_loader_resamples_to_5m():
    loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    dataset = loader.load(
        provider="bitget",
        symbols=["BTCUSDT", "ETHUSDT"],
        start_time=START,
        end_time=END,
        interval="5m",
    )

    assert dataset.shape() == (4, 2)
    assert dataset.interval == "5m"


def test_dataset_loader_applies_blocked_utc_hours():
    loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    dataset = loader.load(
        provider="bitget",
        symbols=["BTCUSDT", "ETHUSDT"],
        start_time=START,
        end_time=datetime(2026, 3, 27, 0, 2, tzinfo=UTC),
        blocked_utc_hours=[0],
    )

    assert not dataset.session_mask.any()


def test_alpha_lab_service_evaluates_formula_from_db():
    service = AlphaService()
    service.dataset_loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    result = service.evaluate_formula_from_db(
        formula="CSRank(ts_mean(close, 2) - close)",
        provider="bitget",
        symbols=["BTCUSDT", "ETHUSDT"],
        start_time=START,
        end_time=END,
    )

    assert "metrics" in result
    assert result["dataset"]["shape"] == (4, 2)
    assert "rank_ic" in result["metrics"]


def test_alpha_lab_service_evaluates_formula_from_db_summary_only():
    service = AlphaService()
    service.dataset_loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    result = service.evaluate_formula_from_db(
        formula="CSRank(ts_mean(close, 2) - close)",
        provider="bitget",
        symbols=["BTCUSDT", "ETHUSDT"],
        start_time=START,
        end_time=END,
        summary_only=True,
    )

    assert "metrics" in result
    assert "alpha_signature" not in result
    assert result["expr_hash"]


def test_alpha_lab_service_batch_evaluates_formulas_from_db():
    service = AlphaService()
    service.dataset_loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    result = service.evaluate_formulas_from_db(
        formulas=[
            "CSRank(ts_mean(close, 2) - close)",
            "CSRank(ts_std(close, 2))",
        ],
        provider="bitget",
        symbols=["BTCUSDT", "ETHUSDT"],
        start_time=START,
        end_time=END,
    )

    assert len(result) == 2
    assert "CSRank(ts_mean(close, 2) - close)" in result
    assert result["CSRank(ts_std(close, 2))"]["dataset"]["shape"] == (4, 2)


def test_alpha_lab_service_benchmark_db():
    service = AlphaService()
    service.dataset_loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    result = service.benchmark_db(
        provider="bitget",
        symbols=["BTCUSDT", "ETHUSDT"],
        start_time=START,
        end_time=END,
        formulas=["CSRank(ts_mean(close, 2) - close)"],
        repeat=1,
    )

    assert result["formula_count"] == 1
    assert result["dataset"]["shape"] == (4, 2)
    assert "formula_summaries" in result


def test_alpha_lab_cli_search_db():
    service = AlphaService()
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
            START.isoformat(),
            "--end",
            END.isoformat(),
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
    assert result["validation"]["mode"] in {"cpcv", "holdout"}
    assert result["validation"]["fold_count"] >= 1
    assert "timing" in result
    assert "overall_seconds" in result["timing"]
    assert result["timing"]["per_generation"]
    assert "vm_run_seconds" in result["timing"]["per_generation"][0]
    assert "backtest_seconds" in result["timing"]["per_generation"][0]
    assert "fitness_seconds" in result["timing"]["per_generation"][0]


def test_alpha_lab_cli_batch_evaluate_db():
    service = AlphaService()
    service.dataset_loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    parser = build_parser()
    args = parser.parse_args(
        [
            "batch-evaluate-db",
            "--formula",
            "CSRank(ts_mean(close, 2) - close)",
            "--formula",
            "CSRank(ts_std(close, 2))",
            "--provider",
            "bitget",
            "--symbols",
            "BTCUSDT,ETHUSDT",
            "--start",
            START.isoformat(),
            "--end",
            END.isoformat(),
            "--summary-only",
        ]
    )
    result = run_command(args, service)

    assert len(result) == 2
    assert "CSRank(ts_std(close, 2))" in result
    assert "alpha_signature" not in result["CSRank(ts_std(close, 2))"]


def test_alpha_lab_cli_benchmark_vm():
    service = AlphaService()
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-vm",
            "--formula",
            "CSRank(ts_mean(close, 2) - close)",
            "--rows",
            "16",
            "--cols",
            "3",
            "--repeat",
            "1",
        ]
    )
    result = run_command(args, service)

    assert result["rows"] == 16
    assert result["cols"] == 3
    assert result["formula_count"] == 1


def test_alpha_lab_cli_benchmark_db():
    service = AlphaService()
    service.dataset_loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-db",
            "--provider",
            "bitget",
            "--symbols",
            "BTCUSDT,ETHUSDT",
            "--start",
            START.isoformat(),
            "--end",
            END.isoformat(),
            "--formula",
            "CSRank(ts_mean(close, 2) - close)",
            "--repeat",
            "1",
        ]
    )
    result = run_command(args, service)

    assert result["formula_count"] == 1
    assert result["dataset"]["shape"] == (4, 2)


def test_alpha_lab_search_persistence(tmp_path):
    service = AlphaService()
    service.dataset_loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    service.persistence = __import__("src.alpha_lab.persistence", fromlist=["AlphaPersistence"]).AlphaPersistence(str(tmp_path / "alpha_lab"))
    result = service.search_formulas_on_db(
        provider="bitget",
        symbols=["BTCUSDT", "ETHUSDT"],
        start_time=START,
        end_time=END,
        seeds=["CSRank(ts_mean(close, 2) - close)"],
        generations=1,
        persist=True,
    )

    assert "persistence" in result
    assert os.path.exists(result["persistence"]["run_path"])
    assert len(result["persistence"]["zoo_paths"]) >= 1
    assert "lineage" in result
    assert "validation" in result
    assert result["timing"]["dataset_load_seconds"] >= 0.0
    persisted = service.persistence.load_run(result["persistence"]["run_id"])
    assert persisted["timing"]["overall_seconds"] > 0.0


def test_alpha_lab_run_and_zoo_inspection(tmp_path):
    service = AlphaService()
    service.dataset_loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    service.persistence = __import__("src.alpha_lab.persistence", fromlist=["AlphaPersistence"]).AlphaPersistence(str(tmp_path / "alpha_lab"))
    result = service.search_formulas_on_db(
        provider="bitget",
        symbols=["BTCUSDT", "ETHUSDT"],
        start_time=START,
        end_time=END,
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


def test_alpha_lab_search_top_results_are_scored():
    service = AlphaService()
    service.dataset_loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    result = service.search_formulas_on_db(
        provider="bitget",
        symbols=["BTCUSDT", "ETHUSDT"],
        start_time=START,
        end_time=END,
        seeds=["CSRank(ts_mean(close, 2) - close)"],
        generations=2,
        offspring_count=2,
        persist=False,
    )

    assert result["top_results"]
    assert all("sharpe" in item["metrics"] for item in result["top_results"])
    assert "call_stats" in result["llm"]


def test_auto_search_config_normalizes_symbols_and_blocked_hours(tmp_path):
    config_path = tmp_path / "auto_search.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "window": {
                    "lookback_minutes": 60,
                    "lag_minutes": 5,
                    "step_minutes": 5,
                },
                "search": {
                    "provider": "bitget",
                    "symbols": "btcusdt, ethusdt",
                    "blocked_utc_hours": "0, 1, 23",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_auto_search_config(config_path)

    assert config.search.symbols == ["BTCUSDT", "ETHUSDT"]
    assert config.search.blocked_utc_hours == [0, 1, 23]


def test_alpha_lab_cli_auto_search_db_once(tmp_path):
    service = AlphaService()
    service.dataset_loader = CryptoMinuteDatasetLoader(store=FakeCryptoStore())
    service.persistence = AlphaPersistence(root_dir=str(tmp_path / "alpha_lab"))
    state_path = tmp_path / "auto_search_state.json"
    config_path = tmp_path / "auto_search.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "window": {
                    "lookback_minutes": 20,
                    "lag_minutes": 0,
                    "step_minutes": 1,
                    "anchor_time": "2026-03-27T00:19:00+00:00",
                },
                "search": {
                    "provider": "bitget",
                    "symbols": ["BTCUSDT", "ETHUSDT"],
                    "interval": "5m",
                    "population_size": 2,
                    "offspring_count": 1,
                    "top_k": 1,
                    "generations": 1,
                    "persist": True,
                    "run_name": "test_auto_search",
                    "seeds": ["CSRank(ts_mean(close, 2) - close)"],
                    "seed_zoo_limit": 0,
                },
                "runtime": {
                    "state_path": str(state_path),
                    "poll_interval_seconds": 1,
                    "continue_on_error": False,
                },
            }
        ),
        encoding="utf-8",
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "auto-search-db",
            "--config",
            str(config_path),
            "--once",
        ]
    )
    result = run_command(args, service)

    assert result["successful_cycles"] == 1
    assert result["failed_cycles"] == 0
    assert result["stopped_reason"] == "once_completed"
    assert result["last_run_id"]
    assert state_path.exists()
    persisted = service.persistence.load_run(result["last_run_id"])
    assert persisted["run_id"] == result["last_run_id"]


def test_alpha_lab_persistence_prunes_runs_and_zoo(tmp_path):
    service = AlphaService()
    service.persistence = AlphaPersistence(root_dir=str(tmp_path / "alpha_lab"))

    for idx in range(3):
        run = service.persistence.save_run(
            {
                "dataset": {"shape": [1, 1]},
                "top_results": [{"formula": f"f{idx}", "fitness": float(idx)}],
            },
            run_name=f"run_{idx}",
        )
        service.persistence.save_zoo_entries(
            [
                {
                    "formula": f"CSRank(ts_mean(close, {idx + 2}) - close)",
                    "expr_hash": f"hash_{idx}",
                    "fitness": float(idx),
                }
            ],
            run.run_id,
        )

    runs_summary = service.persistence.prune_runs(keep_latest=2)
    zoo_summary = service.persistence.prune_zoo_entries(keep_top=2)

    assert runs_summary["removed"] == 1
    assert zoo_summary["removed"] == 1
    assert len(service.list_runs(limit=10)) == 2
    assert len(service.list_zoo(limit=10)) == 2
