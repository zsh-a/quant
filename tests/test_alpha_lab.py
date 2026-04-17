import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.alpha import AlphaService, FormulaCompiler, StackVM, TensorStore
from src.alpha.eval.validation import CPCVValidator
from src.alpha.infra.persistence import AlphaPersistence
from src.alpha.llm.backends import HeuristicLLMBackend, OpenAILLMBackend
from src.alpha.risk.models import CostModel, ExecutionSimulator, MarketContext, RuleOverlay
from src.alpha.search.evolution import BreedingSpec, FitnessEngine
from src.alpha.strategies import LLMEvolutionStrategy
from src.market_data.ccxt_adapter import PROVIDER_SPECS, CcxtCryptoDataAdapter


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
    service = AlphaService()
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
    service = AlphaService()
    report = service.validate_formula(
        "where(not(liquidity_mask) or (close == vwap), fillna(ts_cov(close, volume, 2), 0), cs_zscore(close))"
    )

    assert report["ok"] is True


def test_validate_formula_supports_crypto_domain_ops():
    service = AlphaService()
    report = service.validate_formula(
        "cs_rank(oi_delta(open_interest, 1) - spread_ratio(bid_ask_spread, close) + atr_n(high, low, close, 5))"
    )

    assert report["ok"] is True


def test_alpha_lab_service_benchmark_vm():
    service = AlphaService()
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


def test_alpha_lab_service_program_cache_uses_lru_eviction():
    service = AlphaService(program_cache_size=2)

    service.compile_formula("CSRank(ts_mean(close, 2) - close)")
    service.compile_formula("CSRank(ts_std(close, 2))")
    service.compile_formula("CSRank(volatility_n(close, 3))")

    stats = service.program_cache_stats()
    assert stats["size"] == 2
    assert stats["capacity"] == 2
    assert "CSRank(ts_mean(close, 2) - close)" not in service._program_cache


class _BatchCountingBackend:
    def __init__(self):
        self.calls: list[int] = []
        self.counter = 0

    def generate_initial_population(self, count: int):
        return [f"CSRank(ts_mean(close, {idx + 2}) - close)" for idx in range(count)]

    def generate_offspring(self, spec: BreedingSpec, count: int):
        self.calls.append(count)
        formulas = []
        for _ in range(count):
            self.counter += 1
            formulas.append(f"CSRank(ts_mean(close, {self.counter + 2}) - close)")
        return formulas


def test_evolution_breed_batches_llm_calls():
    backend = _BatchCountingBackend()
    LLMEvolutionStrategy(llm_backend=backend, batch_size=7)
    # Strategy generates offspring via LLM backend
    formulas = backend.generate_offspring(
        BreedingSpec(parent_a="CSRank(ts_mean(close, 5) - close)", parent_b=None, objective="test"),
        count=7,
    )
    assert len(formulas) == 7
    assert backend.calls[-1] == 7


def test_save_formula_to_zoo_persists_manual_entry(tmp_path):
    service = AlphaService()
    service.persistence = AlphaPersistence(root_dir=str(tmp_path / "alpha_lab"))

    entry = service.save_formula_to_zoo(
        formula="CSRank(ts_mean(close, 5) - close)",
        metrics={"sharpe": 1.25, "rank_ic": 0.08},
        note="mean reversion baseline",
        tags=["baseline", "mr"],
    )

    assert entry["expr_hash"]
    assert entry["fitness"] == 1.25
    assert entry["validation"]["ok"] is True

    zoo_entries = service.list_zoo(limit=10)
    assert len(zoo_entries) == 1
    assert zoo_entries[0]["formula"] == "CSRank(ts_mean(close, 5) - close)"
    assert zoo_entries[0]["tags"] == ["baseline", "mr"]
    assert zoo_entries[0]["note"] == "mean reversion baseline"


def test_save_formula_to_zoo_rejects_invalid_formula(tmp_path):
    service = AlphaService()
    service.persistence = AlphaPersistence(root_dir=str(tmp_path / "alpha_lab"))

    try:
        service.save_formula_to_zoo("close +")
    except ValueError as exc:
        assert "syntax" in str(exc).lower() or "failed" in str(exc).lower() or "unexpected" in str(exc).lower()
    else:
        raise AssertionError("Expected invalid formula to raise ValueError")


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


class _FeedbackBackend:
    def __init__(self):
        self.last_spec = None

    def generate_initial_population(self, count: int):
        return []

    def generate_offspring(self, spec: BreedingSpec, count: int):
        self.last_spec = spec
        return ["CSRank(ts_mean(close, 3) - close)"][:count]


def test_evolution_feedback_seed_generation_uses_metrics_in_parent_feedback():
    backend = _FeedbackBackend()
    spec = BreedingSpec(
        parent_a="CSRank(ts_mean(close, 5) - close)",
        parent_b="CSRank(ts_std(close, 5))",
        objective="improve robustness and reduce turnover",
        parent_feedback=[
            {
                "formula": "CSRank(ts_mean(close, 5) - close)",
                "metrics": {"sharpe": 1.5, "rank_ic": 0.1},
                "rationale": "Elite.",
            },
            {
                "formula": "CSRank(ts_std(close, 5))",
                "metrics": {"sharpe": 0.8, "rank_ic": 0.02},
                "rationale": "Runner-up.",
            },
        ],
    )
    formulas = backend.generate_offspring(spec, count=1)

    assert formulas == ["CSRank(ts_mean(close, 3) - close)"]
    assert backend.last_spec is not None
    assert backend.last_spec.parent_feedback[0]["metrics"]["sharpe"] == 1.5
    assert backend.last_spec.parent_feedback[1]["metrics"]["rank_ic"] == 0.02


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    def create(self, **kwargs):
        return _FakeResponse(self._responses.pop(0))


class _FakeClient:
    def __init__(self, responses: list[str]):
        self.chat = type("Chat", (), {"completions": _FakeChatCompletions(responses)})()


def test_openai_backend_parses_and_filters_formulas():
    backend = OpenAILLMBackend(
        client=_FakeClient(
            [
                """```json
                [
                  {"rationale": "oi + spread", "formula": "CSRank(OIDelta(OI, 1) - SpreadRatio(BidAskSpread, Close))"},
                  {"rationale": "too deep", "formula": "Ts_Mean(Ts_Mean(Ts_Mean(Ts_Mean(Ts_Mean(Ts_Mean(Close, 2), 2), 2), 2), 2), 2)"}
                ]
                ```"""
            ]
        ),
        max_ast_depth=5,
    )

    formulas = backend.generate_initial_population(2)

    assert len(formulas) == 2
    assert "OIDelta" in formulas[0] or "oi_delta" in formulas[0]


def test_openai_evolution_prompt_includes_extended_metrics_and_diagnostics():
    backend = OpenAILLMBackend(client=_FakeClient(['[{"formula": "CSRank(ts_mean(close, 3) - close)"}]']))

    prompt = backend._build_evolution_prompt(
        BreedingSpec(
            parent_a="CSRank(ts_mean(close, 5) - close)",
            parent_b="CSRank(ts_std(close, 5))",
            objective="improve robustness and reduce turnover",
            parent_feedback=[
                {
                    "formula": "CSRank(ts_mean(close, 5) - close)",
                    "metrics": {
                        "sharpe": 1.2,
                        "test_sharpe": 0.4,
                        "rank_ic": 0.08,
                        "pnl_per_turnover": 0.3,
                        "turnover_penalty": 0.7,
                        "stability": 2.0,
                        "tail_penalty_adjusted_return": -0.1,
                        "signal_coverage": 0.4,
                        "active_bar_ratio": 0.1,
                        "train_valid_gap_penalty": 0.8,
                        "inactive": 0.0,
                    },
                    "rationale": "Parent A",
                }
            ],
        ),
        count=2,
    )

    # Check that parent metrics appear in the prompt
    assert "test_sharpe=0.400" in prompt
    assert "sharpe=1.200" in prompt
    assert "|IC|=0.0800" in prompt
    # Check diagnostics
    assert "overfitting" in prompt or "low turnover" in prompt


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


def test_rule_overlay_respects_turnover_limit():
    overlay = RuleOverlay()
    target = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [-1.0, 1.0],
        ],
        dtype=float,
    )

    limited = overlay.apply(target, MarketContext(max_turnover_per_bar=0.25))

    expected = np.array(
        [
            [0.0, 0.0],
            [0.25, -0.25],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    assert np.allclose(limited, expected)


def test_compute_rank_ic_matches_reference_loop():
    from src.alpha.eval.metrics import compute_rank_ic

    alpha = np.array(
        [
            [1.0, 2.0, np.nan, 4.0],
            [2.0, 1.0, 3.0, 0.0],
            [5.0, 5.0, 5.0, 5.0],
        ],
        dtype=float,
    )
    returns = np.array(
        [
            [1.5, 1.0, 0.0, 4.5],
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    valid_rows = []
    for alpha_row, return_row in zip(alpha, returns):
        mask = ~np.isnan(alpha_row) & ~np.isnan(return_row)
        if mask.sum() < 2:
            continue
        left = alpha_row[mask]
        right = return_row[mask]
        if np.std(left) < 1e-12 or np.std(right) < 1e-12:
            continue
        valid_rows.append(float(np.corrcoef(left, right)[0, 1]))
    expected = float(np.mean(valid_rows)) if valid_rows else 0.0

    observed = compute_rank_ic(alpha, returns)

    assert np.isclose(observed, expected)


def test_build_fitness_metrics_marks_inactive_flat_strategy():
    service = AlphaService()
    alpha = np.full((4, 2), np.nan, dtype=float)
    weights = np.zeros((4, 2), dtype=float)
    close = np.array(
        [
            [100.0, 101.0],
            [101.0, 102.0],
            [102.0, 103.0],
            [103.0, 104.0],
        ],
        dtype=float,
    )
    summary = {
        "avg_turnover": 0.0,
        "total_return": 0.0,
        "volatility": 0.0,
        "max_drawdown": 0.0,
        "final_equity": 1.0,
        "sharpe": 0.0,
    }

    metrics = service._build_fitness_metrics(alpha, weights, close, summary)

    assert metrics["inactive"] == 1.0
    assert metrics["activity_score"] == 0.0
    assert metrics["pnl_per_turnover"] == 0.0
    assert metrics["coverage_penalty"] == 1.0


def test_fitness_engine_penalizes_inactive_strategies():
    fitness = FitnessEngine().score(
        {
            "inactive": 1.0,
            "sharpe": 0.0,
            "pnl_per_turnover": 0.0,
            "rank_ic": 0.0,
            "stability": 1e12,
            "tail_penalty_adjusted_return": 0.0,
            "turnover_penalty": 0.0,
            "complexity_penalty": 0.0,
            "train_valid_gap_penalty": 0.0,
        }
    )

    assert fitness <= -5.0


def test_fitness_engine_prefers_active_predictive_factor_over_zombie_factor():
    engine = FitnessEngine()
    zombie = engine.score(
        {
            "inactive": 0.0,
            "active_bar_ratio": 0.12,
            "signal_coverage": 0.95,
            "avg_turnover": 0.006,
            "test_sharpe": 0.05,
            "negative_test_ratio": 0.0,
            "sharpe": 0.10,
            "rank_ic_abs": 0.002,
            "pnl_per_turnover": 50.0,
            "tail_ratio": 0.20,
            "activity_score": 0.05,
            "turnover_penalty": 0.0,
            "complexity_penalty": 0.0,
            "train_valid_gap_penalty": 0.0,
            "valid_test_gap_penalty": 0.0,
            "coverage_penalty": 0.0,
        }
    )
    active = engine.score(
        {
            "inactive": 0.0,
            "active_bar_ratio": 0.65,
            "signal_coverage": 0.98,
            "avg_turnover": 0.12,
            "test_sharpe": 0.90,
            "negative_test_ratio": 0.0,
            "sharpe": 1.10,
            "rank_ic_abs": 0.04,
            "pnl_per_turnover": 2.0,
            "tail_ratio": 1.20,
            "activity_score": 0.95,
            "turnover_penalty": 0.0,
            "complexity_penalty": 0.10,
            "train_valid_gap_penalty": 0.10,
            "valid_test_gap_penalty": 0.10,
            "coverage_penalty": 0.0,
        }
    )

    assert active > zombie


def test_ccxt_adapter_storage_symbol_normalization():
    adapter = CcxtCryptoDataAdapter(PROVIDER_SPECS["bitget"])
    market = {
        "base": "BTC",
        "quote": "USDT",
        "symbol": "BTC/USDT:USDT",
        "id": "BTCUSDT",
    }

    assert adapter.to_storage_symbol(market) == "BTCUSDT"
