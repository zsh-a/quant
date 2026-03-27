import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.alpha_lab import AlphaLabService, FormulaCompiler, StackVM, TensorStore
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


def test_bitget_adapter_normalization():
    adapter = BitgetDataAdapter()
    records = [
        adapter._parse_candle(["1700000000000", "1", "2", "0.5", "1.5", "10", "15"])
    ]
    normalized = adapter.normalize_candles(records, "BTCUSDT")

    assert normalized[0]["symbol"] == "BTCUSDT"
    assert normalized[0]["close"] == 1.5
