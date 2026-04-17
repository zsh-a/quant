"""Numerical correctness tests for Triton GPU kernels.

Each test compares the Triton kernel output against the existing
torch/numpy implementation to ensure identical results (within float32 tolerance).

Tests are skipped if Triton or CUDA is not available.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import triton  # noqa: F401

    from src.alpha.eval.gpu_ops import (
        TRITON_AVAILABLE,
        batch_rank_ic,
        cs_rank,
        decay_linear,
        factor_correlation_matrix,
        parallel_ema,
        rolling_corr_cov,
        rolling_mean_std,
        rolling_reduce,
    )
except ImportError:
    TRITON_AVAILABLE = False

requires_triton = pytest.mark.skipif(
    not (HAS_TORCH and TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="Triton + CUDA not available",
)


def _inject_nans(t: torch.Tensor, ratio: float = 0.1, seed: int = 42) -> torch.Tensor:
    rng = torch.Generator(device=t.device).manual_seed(seed)
    mask = torch.rand(t.shape, generator=rng, device=t.device) < ratio
    t = t.clone()
    t[mask] = float("nan")
    return t


def _torch_rolling(data: torch.Tensor, window: int, reducer: str) -> torch.Tensor:
    """Reference rolling implementation using torch.unfold (matches vm.py)."""
    result = torch.full_like(data, float("nan"))
    if window <= 0 or data.shape[0] < window:
        return result
    windows = data.unfold(0, window, 1)
    mask = ~torch.isnan(windows)
    safe = torch.where(mask, windows, torch.zeros_like(windows))
    count = mask.sum(dim=-1)
    safe_count = count.clamp(min=1).float()
    if reducer == "mean":
        reduced = safe.sum(dim=-1) / safe_count
    elif reducer == "std":
        mean = safe.sum(dim=-1) / safe_count
        centered = torch.where(mask, windows - mean.unsqueeze(-1), torch.zeros_like(windows))
        reduced = torch.sqrt((centered * centered).sum(dim=-1) / safe_count)
    elif reducer == "sum":
        reduced = safe.sum(dim=-1)
    elif reducer == "max":
        neg_inf = torch.where(mask, windows, torch.full_like(windows, float("-inf")))
        reduced = neg_inf.max(dim=-1).values
    elif reducer == "min":
        pos_inf = torch.where(mask, windows, torch.full_like(windows, float("inf")))
        reduced = pos_inf.min(dim=-1).values
    else:
        raise ValueError(f"Unknown reducer: {reducer}")
    reduced = torch.where(count > 0, reduced, torch.full_like(reduced, float("nan")))
    result[window - 1 :] = reduced
    return result


def _torch_ema(data: torch.Tensor, window: int) -> torch.Tensor:
    """Reference EMA implementation (sequential, matches vm.py)."""
    alpha = 2.0 / (window + 1)
    result = torch.full_like(data, float("nan"))
    if data.shape[0] == 0:
        return result
    result[0] = data[0]
    for i in range(1, data.shape[0]):
        prev = result[i - 1]
        cur = data[i]
        nan_prev = torch.isnan(prev)
        nan_cur = torch.isnan(cur)
        result[i] = torch.where(
            nan_cur,
            prev,
            torch.where(nan_prev, cur, alpha * cur + (1 - alpha) * prev),
        )
    return result


# ---------------------------------------------------------------------------
# Rolling mean + std
# ---------------------------------------------------------------------------


@requires_triton
@pytest.mark.parametrize("window", [2, 5, 10, 20])
@pytest.mark.parametrize("T,S", [(128, 32), (512, 64), (2048, 16)])
def test_rolling_mean_std(T, S, window):
    data = torch.randn(T, S, device="cuda", dtype=torch.float32)
    data = _inject_nans(data, ratio=0.05)

    mean_triton, std_triton = rolling_mean_std(data, window)
    mean_ref = _torch_rolling(data, window, "mean")
    std_ref = _torch_rolling(data, window, "std")

    assert torch.allclose(mean_triton, mean_ref, atol=1e-4, equal_nan=True)
    assert torch.allclose(std_triton, std_ref, atol=1e-4, equal_nan=True)


@requires_triton
def test_rolling_mean_std_window_larger_than_t():
    data = torch.randn(10, 8, device="cuda", dtype=torch.float32)
    mean, std = rolling_mean_std(data, window=20)
    assert torch.isnan(mean).all()
    assert torch.isnan(std).all()


@requires_triton
def test_rolling_mean_std_all_nan():
    data = torch.full((32, 8), float("nan"), device="cuda", dtype=torch.float32)
    mean, std = rolling_mean_std(data, window=5)
    assert torch.isnan(mean).all()
    assert torch.isnan(std).all()


# ---------------------------------------------------------------------------
# Rolling reduce (sum / max / min)
# ---------------------------------------------------------------------------


@requires_triton
@pytest.mark.parametrize("mode", ["sum", "max", "min"])
@pytest.mark.parametrize("window", [3, 10, 20])
def test_rolling_reduce(mode, window):
    T, S = 256, 32
    data = torch.randn(T, S, device="cuda", dtype=torch.float32)
    data = _inject_nans(data, ratio=0.05)

    result = rolling_reduce(data, window, mode)
    ref = _torch_rolling(data, window, mode)

    assert torch.allclose(result, ref, atol=1e-4, equal_nan=True)


# ---------------------------------------------------------------------------
# Parallel EMA
# ---------------------------------------------------------------------------


@requires_triton
@pytest.mark.parametrize("window", [5, 10, 20])
@pytest.mark.parametrize("T,S", [(64, 16), (512, 32), (2048, 8)])
def test_parallel_ema(T, S, window):
    data = torch.randn(T, S, device="cuda", dtype=torch.float32)
    data = _inject_nans(data, ratio=0.05)

    result = parallel_ema(data, window)
    ref = _torch_ema(data, window)

    assert torch.allclose(result, ref, atol=1e-4, equal_nan=True)


@requires_triton
def test_parallel_ema_empty():
    data = torch.empty(0, 8, device="cuda", dtype=torch.float32)
    result = parallel_ema(data, window=5)
    assert result.shape == (0, 8)


@requires_triton
def test_parallel_ema_all_nan():
    data = torch.full((32, 8), float("nan"), device="cuda", dtype=torch.float32)
    result = parallel_ema(data, window=5)
    assert torch.isnan(result).all()


# ---------------------------------------------------------------------------
# Rolling correlation / covariance
# ---------------------------------------------------------------------------


def _torch_rolling_corr_cov(x, y, window, mode):
    """Reference rolling corr/cov matching vm.py _rolling_pair_torch."""
    result = torch.full_like(x, float("nan"))
    if window <= 1 or x.shape[0] < window:
        return result
    x_win = x.unfold(0, window, 1)
    y_win = y.unfold(0, window, 1)
    valid = (~torch.isnan(x_win)) & (~torch.isnan(y_win))
    count = valid.sum(dim=-1)
    safe_count = count.clamp(min=1).float()
    x_safe = torch.where(valid, x_win, torch.zeros_like(x_win))
    y_safe = torch.where(valid, y_win, torch.zeros_like(y_win))
    x_mean = x_safe.sum(dim=-1) / safe_count
    y_mean = y_safe.sum(dim=-1) / safe_count
    cx = torch.where(valid, x_win - x_mean.unsqueeze(-1), torch.zeros_like(x_win))
    cy = torch.where(valid, y_win - y_mean.unsqueeze(-1), torch.zeros_like(y_win))
    cov = (cx * cy).sum(dim=-1) / safe_count
    if mode == "cov":
        reduced = cov
    else:
        x_std = torch.sqrt((cx * cx).sum(dim=-1) / safe_count)
        y_std = torch.sqrt((cy * cy).sum(dim=-1) / safe_count)
        reduced = cov / (x_std * y_std + 1e-12)
    reduced = torch.where(count > 0, reduced, torch.full_like(reduced, float("nan")))
    result[window - 1 :] = reduced
    return result


@requires_triton
@pytest.mark.parametrize("mode", ["corr", "cov"])
@pytest.mark.parametrize("window", [5, 10, 20])
def test_rolling_corr_cov(mode, window):
    T, S = 256, 32
    rng = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(T, S, device="cuda", dtype=torch.float32, generator=rng)
    y = torch.randn(T, S, device="cuda", dtype=torch.float32, generator=rng)
    x = _inject_nans(x, ratio=0.05, seed=1)
    y = _inject_nans(y, ratio=0.05, seed=2)

    result = rolling_corr_cov(x, y, window, mode)
    ref = _torch_rolling_corr_cov(x, y, window, mode)

    # NaN positions must match
    assert (torch.isnan(result) == torch.isnan(ref)).all()
    # Compare non-NaN values
    r = torch.nan_to_num(result, nan=0.0)
    f = torch.nan_to_num(ref, nan=0.0)
    assert torch.allclose(r, f, atol=1e-3)


# ---------------------------------------------------------------------------
# Cross-sectional rank
# ---------------------------------------------------------------------------


def _torch_cs_rank(data: torch.Tensor) -> torch.Tensor:
    """Reference cs_rank matching vm.py _cs_rank_torch."""
    nan_mask = torch.isnan(data)
    safe = torch.where(nan_mask, torch.full_like(data, float("inf")), data)
    order = torch.argsort(safe, dim=1, stable=True)
    ranks = torch.argsort(order, dim=1, stable=True).to(dtype=data.dtype) + 1.0
    denom = (~nan_mask).sum(dim=1, keepdim=True).to(dtype=data.dtype)
    scaled = ranks / denom.clamp(min=1.0)
    scaled = torch.where(nan_mask, torch.full_like(scaled, float("nan")), scaled)
    return scaled


@requires_triton
@pytest.mark.parametrize("T,S", [(64, 16), (256, 32), (512, 64)])
def test_cs_rank(T, S):
    data = torch.randn(T, S, device="cuda", dtype=torch.float32)
    data = _inject_nans(data, ratio=0.1)

    result = cs_rank(data)
    ref = _torch_cs_rank(data)

    # Rank values should match (rank ordering is stable)
    assert torch.allclose(result, ref, atol=1e-4, equal_nan=True)


@requires_triton
def test_cs_rank_single_symbol():
    data = torch.randn(32, 1, device="cuda", dtype=torch.float32)
    result = cs_rank(data)
    expected = torch.where(torch.isnan(data), data, torch.ones_like(data))
    assert torch.allclose(result, expected, atol=1e-5, equal_nan=True)


@requires_triton
def test_cs_rank_all_nan_row():
    data = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    data[3, :] = float("nan")
    result = cs_rank(data)
    assert torch.isnan(result[3]).all()
    assert not torch.isnan(result[0]).all()


# ---------------------------------------------------------------------------
# Decay linear
# ---------------------------------------------------------------------------


def _torch_decay_linear(data: torch.Tensor, window: int) -> torch.Tensor:
    """Reference decay_linear matching vm.py _decay_linear_torch."""
    result = torch.full_like(data, float("nan"))
    if window <= 0 or data.shape[0] < window:
        return result
    windows = data.unfold(0, window, 1)
    weights = torch.arange(1, window + 1, device=data.device, dtype=data.dtype)
    valid = ~torch.isnan(windows)
    weighted = torch.where(valid, windows * weights, torch.zeros_like(windows))
    denom = torch.where(valid, weights.expand_as(windows), torch.zeros_like(windows)).sum(dim=-1)
    reduced = weighted.sum(dim=-1) / denom.clamp(min=1e-12)
    reduced = torch.where(denom > 0, reduced, torch.full_like(reduced, float("nan")))
    result[window - 1 :] = reduced
    return result


@requires_triton
@pytest.mark.parametrize("window", [3, 10, 20])
def test_decay_linear(window):
    T, S = 256, 32
    data = torch.randn(T, S, device="cuda", dtype=torch.float32)
    data = _inject_nans(data, ratio=0.05)

    result = decay_linear(data, window)
    ref = _torch_decay_linear(data, window)

    assert torch.allclose(result, ref, atol=1e-4, equal_nan=True)


# ---------------------------------------------------------------------------
# Batch rank IC
# ---------------------------------------------------------------------------


def _numpy_rank_ic(alpha: np.ndarray, returns: np.ndarray) -> float:
    """Reference rank IC from evaluation.py."""
    mask = ~np.isnan(alpha) & ~np.isnan(returns)
    valid_counts = np.sum(mask, axis=1)
    if not np.any(valid_counts >= 2):
        return 0.0
    safe_a = np.where(mask, alpha, 0.0)
    safe_r = np.where(mask, returns, 0.0)
    denom = np.maximum(valid_counts, 1)
    mean_a = np.sum(safe_a, axis=1) / denom
    mean_r = np.sum(safe_r, axis=1) / denom
    ca = np.where(mask, alpha - mean_a[:, None], 0.0)
    cr = np.where(mask, returns - mean_r[:, None], 0.0)
    cov = np.sum(ca * cr, axis=1)
    va = np.sum(ca * ca, axis=1)
    vr = np.sum(cr * cr, axis=1)
    valid_rows = (valid_counts >= 2) & (va > 1e-24) & (vr > 1e-24)
    if not np.any(valid_rows):
        return 0.0
    corrs = cov[valid_rows] / np.sqrt(va[valid_rows] * vr[valid_rows])
    return float(np.mean(corrs))


@requires_triton
def test_batch_rank_ic():
    N, T, S = 5, 256, 32
    rng = np.random.default_rng(42)
    alphas_np = rng.standard_normal((N, T, S)).astype(np.float32)
    returns_np = rng.standard_normal((T, S)).astype(np.float32)
    # Inject some NaNs
    alphas_np[alphas_np < -2.0] = np.nan
    returns_np[returns_np < -2.5] = np.nan

    alphas_t = torch.tensor(alphas_np, device="cuda", dtype=torch.float32)
    returns_t = torch.tensor(returns_np, device="cuda", dtype=torch.float32)

    result = batch_rank_ic(alphas_t, returns_t)
    assert result.shape == (N,)

    for i in range(N):
        ref_ic = _numpy_rank_ic(alphas_np[i], returns_np)
        assert abs(result[i].item() - ref_ic) < 0.02, (
            f"Factor {i}: Triton IC={result[i].item():.6f}, NumPy IC={ref_ic:.6f}"
        )


# ---------------------------------------------------------------------------
# Factor correlation matrix
# ---------------------------------------------------------------------------


@requires_triton
def test_factor_correlation_matrix():
    N, D = 8, 1024
    rng = np.random.default_rng(42)
    factors_np = rng.standard_normal((N, D)).astype(np.float32)

    factors_t = torch.tensor(factors_np, device="cuda", dtype=torch.float32)
    result = factor_correlation_matrix(factors_t)
    assert result.shape == (N, N)

    # Compare with numpy pairwise correlation
    for i in range(N):
        for j in range(i, N):
            ref = abs(np.corrcoef(factors_np[i], factors_np[j])[0, 1])
            assert abs(result[i, j].item() - ref) < 0.01, (
                f"Pair ({i},{j}): Triton={result[i,j].item():.6f}, NumPy={ref:.6f}"
            )

    # Diagonal should be ~1.0
    for i in range(N):
        assert abs(result[i, i].item() - 1.0) < 0.01

    # Should be symmetric
    assert torch.allclose(result, result.T, atol=1e-5)


# ---------------------------------------------------------------------------
# VM integration (end-to-end)
# ---------------------------------------------------------------------------


@requires_triton
def test_vm_triton_backend_produces_valid_output():
    """Test that StackVM with Triton produces numerically valid output."""
    from src.alpha import FormulaCompiler, OperatorRegistry, StackVM, TensorStore

    registry = OperatorRegistry()
    compiler = FormulaCompiler(registry)
    vm = StackVM(prefer_torch=True, use_triton=True)

    T, S = 256, 16
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, (T, S)), axis=0)
    volume = np.abs(rng.normal(1000, 150, (T, S))) + 1

    store = TensorStore(
        {
            "close": np.asarray(close, dtype=float),
            "volume": np.asarray(volume, dtype=float),
        }
    )

    formulas = [
        "cs_rank(ts_mean(close, 5) - close)",
        "cs_rank(ts_std(close, 10))",
        "ts_ema(close, 20)",
        "ts_corr(close, volume, 10)",
        "decay_linear(close, 5)",
    ]

    from src.alpha.core.dsl import TensorSchema

    schema = TensorSchema(frozenset(["close", "volume"]))

    for formula in formulas:
        program = compiler.compile(formula, schema)
        result = vm.run(program, store)
        arr = np.asarray(result) if not isinstance(result, np.ndarray) else result
        if hasattr(result, "cpu"):
            arr = result.cpu().numpy()
        assert arr.shape == (T, S), f"Formula {formula}: expected ({T},{S}), got {arr.shape}"
        finite_ratio = np.isfinite(arr).mean()
        assert finite_ratio > 0.5, f"Formula {formula}: too many NaN ({1-finite_ratio:.1%})"


@requires_triton
def test_vm_triton_batch_matches_serial():
    """Test that batch execution matches serial execution with Triton."""
    from src.alpha import FormulaCompiler, OperatorRegistry, StackVM, TensorStore
    from src.alpha.core.dsl import TensorSchema

    registry = OperatorRegistry()
    compiler = FormulaCompiler(registry)
    vm = StackVM(prefer_torch=True, use_triton=True)

    T, S = 128, 16
    rng = np.random.default_rng(123)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, (T, S)), axis=0)
    volume = np.abs(rng.normal(1000, 150, (T, S))) + 1

    store = TensorStore(
        {
            "close": np.asarray(close, dtype=float),
            "volume": np.asarray(volume, dtype=float),
        }
    )

    schema = TensorSchema(frozenset(["close", "volume"]))
    formulas = [
        "cs_rank(ts_mean(close, 5))",
        "ts_std(close, 10)",
        "ts_sum(volume, 5)",
    ]
    programs = [compiler.compile(f, schema) for f in formulas]

    # Serial
    serial_results = [vm.run(p, store) for p in programs]

    # Batch
    batch_results = vm.run_batch(programs, store)

    for i, formula in enumerate(formulas):
        s = serial_results[i]
        b = batch_results[i]
        if hasattr(s, "cpu"):
            s = s.cpu().numpy()
        if hasattr(b, "cpu"):
            b = b.cpu().numpy()
        assert np.allclose(s, b, atol=1e-5, equal_nan=True), (
            f"Formula {formula}: batch/serial mismatch"
        )


# ---------------------------------------------------------------------------
# GPU evaluation integration
# ---------------------------------------------------------------------------


@requires_triton
def test_gpu_evaluation_ic_metrics():
    """Test GPU IC metrics computation."""
    from src.alpha.eval.gpu_metrics import compute_ic_metrics_gpu, compute_rank_ic_gpu
    from src.alpha.eval.metrics import compute_ic_metrics, compute_rank_ic

    T, S = 256, 32
    rng = np.random.default_rng(42)
    alpha_np = rng.standard_normal((T, S)).astype(np.float32)
    close_np = (100.0 + np.cumsum(rng.normal(0, 0.5, (T, S)), axis=0)).astype(np.float32)
    fwd_np = np.zeros_like(close_np)
    fwd_np[:-1] = close_np[1:] / (close_np[:-1] + 1e-12) - 1.0

    # CPU reference
    ic_cpu = compute_rank_ic(alpha_np, fwd_np)
    metrics_cpu = compute_ic_metrics(alpha_np, close_np)

    # GPU
    alpha_t = torch.tensor(alpha_np, device="cuda")
    close_t = torch.tensor(close_np, device="cuda")
    fwd_t = torch.tensor(fwd_np, device="cuda")

    ic_gpu = compute_rank_ic_gpu(alpha_t, fwd_t)
    metrics_gpu = compute_ic_metrics_gpu(alpha_t, close_t)

    assert abs(ic_cpu - ic_gpu) < 0.01, f"IC mismatch: CPU={ic_cpu}, GPU={ic_gpu}"
    assert abs(metrics_cpu["rank_ic"] - metrics_gpu["rank_ic"]) < 0.01
    assert abs(metrics_cpu["ic_std"] - metrics_gpu["ic_std"]) < 0.02
