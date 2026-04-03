"""Diagnose seed formula signal quality on synthetic crypto data."""
import numpy as np
from src.alpha.operators import OperatorRegistry
from src.alpha.compiler import FormulaCompiler
from src.alpha.dsl import TensorSchema
from src.alpha.vm import StackVM, TensorStore
from src.alpha.risk import SignalTransformer, MarketContext

rng = np.random.default_rng(42)
T, N = 1050, 4
base_price = 100 + np.cumsum(rng.normal(0, 0.3, (T, N)), axis=0)
fields = {
    "open": base_price - rng.normal(0, 0.1, (T, N)),
    "high": base_price + np.abs(rng.normal(0.2, 0.1, (T, N))),
    "low": base_price - np.abs(rng.normal(0.2, 0.1, (T, N))),
    "close": base_price,
    "volume": np.abs(rng.normal(1000, 200, (T, N))) + 1,
    "turnover": np.abs(rng.normal(100000, 20000, (T, N))) + 1,
    "vwap": base_price + rng.normal(0, 0.02, (T, N)),
    "funding_rate": np.zeros((T, N)),
    "open_interest": np.zeros((T, N)),
    "bid_ask_spread": np.abs(rng.normal(0.5, 0.05, (T, N))),
}

registry = OperatorRegistry()
compiler = FormulaCompiler(registry)
vm = StackVM(prefer_torch=False)
schema = TensorSchema.default_market_schema()
store = TensorStore(fields)
transformer = SignalTransformer()

formulas = [
    "cs_rank(ts_mean(close, 5) - close)",
    "cs_rank(ts_std(close, 10))",
    "cs_rank(volatility_n(close, 20))",
    "cs_rank(adv_n(turnover, 10) - amihud(close, turnover, 5))",
    "cs_rank(atr_n(high, low, close, 14) - decay_linear(close, 5))",
    "cs_rank(ts_corr(close, volume, 10))",
]

print(f"Data: ({T}, {N})")
print(f"{'Formula':<55} {'NaN%':>6} {'Std':>8} {'Active%':>8}")
print("-" * 80)

for formula in formulas:
    program = compiler.compile(formula, schema)
    alpha = np.asarray(vm.run(program, store), dtype=float)
    nan_pct = np.mean(np.isnan(alpha)) * 100
    valid = alpha[~np.isnan(alpha)]
    std_val = float(np.std(valid)) if valid.size else 0.0
    ctx = MarketContext(
        liquidity_mask=np.ones((T, N), dtype=bool),
        session_mask=np.ones((T, N), dtype=bool),
    )
    weights = transformer.to_target_weights(alpha, ctx)
    active = float(np.mean(np.sum(np.abs(weights), axis=1) > 1e-9)) * 100
    print(f"{formula[:53]:<55} {nan_pct:>5.1f}% {std_val:>8.4f} {active:>7.1f}%")
