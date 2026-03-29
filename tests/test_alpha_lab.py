import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.alpha_lab import AlphaLabService, FormulaCompiler, StackVM, TensorStore
from src.alpha_lab.evolution import BreedingSpec, HeuristicLLMBackend
from src.alpha_lab.risk import CostModel, ExecutionSimulator
from src.alpha_lab.validation import CPCVValidator
from src.datahub.bitget import BitgetDataAdapter


def test_formula_compile_and_vm_run():
    compiler = FormulaCompiler()
    program = compiler.compile("CSRank((Ts_Max(high, 3) - close) / volatility_n(close, 3))")

    store = TensorStore(
        {
            "high": np.array([[10, 12], [11, 13], [15, 12], [14, 16]], dtype=float),
            "close": np.array([[9, 11], [10, 12], [14, 11], [13, 15]], dtype=float),
            "open": np.array([[9, 11], [10, 12], [14, 11], [13, 15]], dtype=float),
            "low": np.array([[8, 10], [9, 11], [13, 10], [12, 14]], dtype=float),
            "volume": np.array([[100, 110], [120, 140], [150, 160], [180, 190]], dtype=float),
            "turnover": np.array([[900, 1210], [1200, 1680], [2100, 1760], [2340, 2850]], dtype=float),
            "vwap": np.array([[9, 11], [10, 12], [14, 11], [13, 15]], dtype=float),
            "funding_rate": np.zeros((4, 2), dtype=float),
            "open_interest": np.ones((4, 2), dtype=float),
            "bid_ask_spread": np.ones((4, 2), dtype=float) * 0.5,
        }
    )

    output = StackVM().run(program, store)
    assert output.shape == (4, 2)
    assert np.isnan(output[:2]).all()
    assert not np.isnan(output[3]).all()


def test_formula_compile_and_vm_run_batch():
    compiler = FormulaCompiler()
    programs = [
        compiler.compile("CSRank(ts_mean(close, 2) - close)"),
        compiler.compile("CSRank(ts_std(close, 2))"),
    ]
    store = TensorStore(
        {
            "high": np.array([[10, 12], [11, 13], [15, 12], [14, 16]], dtype=float),
            "close": np.array([[9, 11], [10, 12], [14, 11], [13, 15]], dtype=float),
            "open": np.array([[9, 11], [10, 12], [14, 11], [13, 15]], dtype=float),
            "low": np.array([[8, 10], [9, 11], [13, 10], [12, 14]], dtype=float),
            "volume": np.array([[100, 110], [120, 140], [150, 160], [180, 190]], dtype=float),
            "turnover": np.array([[900, 1210], [1200, 1680], [2100, 1760], [2340, 2850]], dtype=float),
            "vwap": np.array([[9, 11], [10, 12], [14, 11], [13, 15]], dtype=float),
            "funding_rate": np.zeros((4, 2), dtype=float),
            "open_interest": np.ones((4, 2), dtype=float),
            "bid_ask_spread": np.ones((4, 2), dtype=float) * 0.5,
        }
    )

    vm = StackVM()
    outputs = vm.run_batch(programs, store)

    assert len(outputs) == 2
    assert outputs[0].shape == (4, 2)
    assert outputs[1].shape == (4, 2)
    assert np.allclose(outputs[0], vm.run(programs[0], store), equal_nan=True)


def test_formula_compile_and_vm_run_extended_ops():
    compiler = FormulaCompiler()
    formula = (
        "CSZScore(fillna(clip(ts_zscore(log_return(close, 1), 2), -2, 2), 0) + "
        "ts_corr(close, volume, 2) + decay_linear(returns_n(close, 1), 2) + "
        "ts_argmax(high, 2) + ts_argmin(low, 2) + cs_demean(vwap))"
    )
    program = compiler.compile(formula)
    store = TensorStore(
        {
            "high": np.array([[10, 12], [11, 13], [15, 12], [14, 16]], dtype=float),
            "close": np.array([[9, 11], [10, 12], [14, 11], [13, 15]], dtype=float),
            "open": np.array([[9, 11], [10, 12], [14, 11], [13, 15]], dtype=float),
            "low": np.array([[8, 10], [9, 11], [13, 10], [12, 14]], dtype=float),
            "volume": np.array([[100, 110], [120, 140], [150, 160], [180, 190]], dtype=float),
            "turnover": np.array([[900, 1210], [1200, 1680], [2100, 1760], [2340, 2850]], dtype=float),
            "vwap": np.array([[9, 11], [10, 12], [14, 11], [13, 15]], dtype=float),
            "funding_rate": np.zeros((4, 2), dtype=float),
            "open_interest": np.ones((4, 2), dtype=float),
            "bid_ask_spread": np.ones((4, 2), dtype=float) * 0.5,
        }
    )

    output = StackVM().run(program, store)

    assert output.shape == (4, 2)
    assert np.isfinite(output[-1]).all()


def test_formula_compile_and_vm_run_crypto_domain_ops():
    compiler = FormulaCompiler()
    formula = (
        "cs_rank(oi_delta(open_interest, 1) + funding_delta(funding_rate, 1) - "
        "spread_ratio(bid_ask_spread, close) - amihud(close, turnover, 2) + "
        "adv_n(turnover, 2) + atr_n(high, low, close, 2) + "
        "hlc3(high, low, close) - ohlc4(open, high, low, close))"
    )
    program = compiler.compile(formula)
    store = TensorStore(
        {
            "open": np.array([[9, 11], [10, 12], [14, 11], [13, 15]], dtype=float),
            "high": np.array([[10, 12], [11, 13], [15, 12], [14, 16]], dtype=float),
            "low": np.array([[8, 10], [9, 11], [13, 10], [12, 14]], dtype=float),
            "close": np.array([[9, 11], [10, 12], [14, 11], [13, 15]], dtype=float),
            "volume": np.array([[100, 110], [120, 140], [150, 160], [180, 190]], dtype=float),
            "turnover": np.array([[900, 1210], [1200, 1680], [2100, 1760], [2340, 2850]], dtype=float),
            "vwap": np.array([[9, 11], [10, 12], [14, 11], [13, 15]], dtype=float),
            "funding_rate": np.array([[0.0, 0.0], [0.0001, -0.0002], [0.0003, -0.0001], [0.0002, 0.0002]], dtype=float),
            "open_interest": np.array([[1000, 900], [1010, 920], [1030, 910], [1040, 930]], dtype=float),
            "bid_ask_spread": np.array([[0.1, 0.2], [0.1, 0.15], [0.2, 0.1], [0.15, 0.12]], dtype=float),
        }
    )

    output = StackVM().run(program, store)

    assert output.shape == (4, 2)
    assert np.isfinite(output[-1]).all()


def test_alpha_lab_service_evaluate_formula():
    service = AlphaLabService()
    fields = {
        "open": [[10, 12], [11, 13], [12, 14], [13, 15]],
        "high": [[11, 13], [12, 14], [13, 15], [14, 16]],
        "low": [[9, 11], [10, 12], [11, 13], [12, 14]],
        "close": [[10, 12], [11, 13], [12, 14], [13, 15]],
        "volume": [[100, 120], [130, 140], [150, 160], [170, 180]],
        "turnover": [[1000, 1440], [1430, 1820], [1800, 2240], [2210, 2700]],
        "vwap": [[10, 12], [11, 13], [12, 14], [13, 15]],
        "funding_rate": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "open_interest": [[1, 1], [1, 1], [1, 1], [1, 1]],
        "bid_ask_spread": [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
    }

    result = service.evaluate_formula("CSRank(ts_mean(close, 2) - close)", fields)

    assert "metrics" in result
    assert "alpha" in result
    assert len(result["weights"]) == 4


def test_validate_formula_supports_new_boolean_and_fill_ops():
    service = AlphaLabService()
    report = service.validate_formula(
        "where(not(liquidity_mask) or (close == vwap), fillna(ts_cov(close, volume, 2), 0), cs_zscore(close))"
    )

    assert report["ok"] is True


def test_validate_formula_supports_crypto_domain_ops():
    service = AlphaLabService()
    report = service.validate_formula(
        "cs_rank(oi_delta(open_interest, 1) - spread_ratio(bid_ask_spread, close) + atr_n(high, low, close, 5))"
    )

    assert report["ok"] is True


def test_alpha_lab_service_benchmark_vm():
    service = AlphaLabService()
    result = service.benchmark_vm(
        formulas=["CSRank(ts_mean(close, 2) - close)"],
        rows=32,
        cols=4,
        repeat=2,
    )

    assert result["formula_count"] == 1
    assert result["rows"] == 32
    assert result["cols"] == 4
    assert result["backend"] in {"numpy", "torch"}
    assert "batch_avg_seconds" in result
    assert "serial_avg_seconds" in result


def test_population_seed_and_breed():
    service = AlphaLabService()
    seeds = [
        "CSRank(ts_mean(close, 5) - close)",
        "CSRank(ts_std(close, 5))",
    ]
    population = service.seed_population(seeds, population_size=2)
    offspring = service.breed_population([item["formula"] for item in population], offspring_count=2)

    assert len(population) == 2
    assert len(offspring) >= 1
    assert all(item["formula"] for item in offspring)


def test_heuristic_backend_generates_diverse_compilable_offspring():
    backend = HeuristicLLMBackend()
    compiler = FormulaCompiler()
    offspring = backend.generate_offspring(
        BreedingSpec(
            parent_a="cs_rank(ts_mean(close, 5) - close)",
            parent_b="cs_rank(ts_std(close, 5))",
            objective="improve robustness and reduce turnover",
        ),
        count=4,
    )

    assert len(offspring) == 4
    assert len(set(offspring)) >= 3
    for formula in offspring:
        program = compiler.compile(formula)
        assert program.expr_hash


def test_cpcv_validator_generates_purged_folds():
    validator = CPCVValidator(purge_window=1, embargo_window=1, min_train_size=2)
    folds = validator.generate_purged_splits(total_time_steps=12, n_splits=4)

    assert folds
    first_fold = folds[0]
    excluded = set(first_fold.valid_indices) | set(first_fold.test_indices)
    assert not (set(first_fold.train_indices) & excluded)


def test_execution_simulator_penalizes_wider_spread():
    weights = np.array(
        [
            [0.0, 0.0],
            [0.5, -0.5],
            [0.5, -0.5],
        ],
        dtype=float,
    )
    close = np.array(
        [
            [100.0, 100.0],
            [101.0, 99.0],
            [102.0, 98.0],
        ],
        dtype=float,
    )
    simulator = ExecutionSimulator()
    tight = simulator.simulate(
        weights,
        {"close": close, "bid_ask_spread": np.full_like(close, 0.01)},
        CostModel(),
    )
    wide = simulator.simulate(
        weights,
        {"close": close, "bid_ask_spread": np.full_like(close, 0.5)},
        CostModel(),
    )

    assert wide.summary()["total_return"] < tight.summary()["total_return"]


def test_bitget_adapter_normalization():
    adapter = BitgetDataAdapter()
    records = [
        adapter._parse_candle(["1700000000000", "1", "2", "0.5", "1.5", "10", "15"])
    ]
    normalized = adapter.normalize_candles(records, "BTCUSDT")

    assert normalized[0]["symbol"] == "BTCUSDT"
    assert normalized[0]["close"] == 1.5
