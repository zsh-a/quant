"""GPU-accelerated operations using Triton kernels.

Provides drop-in replacements for StackVM's torch backend operations
with Triton-fused implementations. Falls back to pure torch when
Triton is unavailable.
"""

from __future__ import annotations

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    try:
        import triton

        from .triton_kernels import (
            _batch_rank_ic_kernel,
            _cs_rank_kernel,
            _decay_linear_kernel,
            _factor_corr_matrix_kernel,
            _parallel_ema_scan_kernel,
            _rolling_corr_cov_kernel,
            _rolling_mean_std_kernel,
            _rolling_reduce_kernel,
        )

        TRITON_AVAILABLE = True
    except ImportError:
        TRITON_AVAILABLE = False
else:
    TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return TRITON_AVAILABLE and torch.cuda.is_available()


def _select_block_s(S: int) -> int:
    if S <= 32:
        return 32
    if S <= 64:
        return 64
    if S <= 128:
        return 128
    return 256


def _ensure_f32_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.float32:
        t = t.to(dtype=torch.float32)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


# ---------------------------------------------------------------------------
# Rolling mean + std (fused)
# ---------------------------------------------------------------------------


def rolling_mean_std(
    data: torch.Tensor,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused rolling mean and std using Welford algorithm.

    Returns (mean, std) tensors of same shape as data, with NaN for rows < window-1.
    """
    data = _ensure_f32_contiguous(data)
    T, S = data.shape
    mean_out = torch.full_like(data, float("nan"))
    std_out = torch.full_like(data, float("nan"))

    if window <= 0 or T < window:
        return mean_out, std_out

    BLOCK_S = _select_block_s(S)
    grid = (T - window + 1, (S + BLOCK_S - 1) // BLOCK_S)

    _rolling_mean_std_kernel[grid](
        data,
        mean_out,
        std_out,
        data.stride(0),
        data.stride(1),
        T,
        S,
        window,
        BLOCK_S,
    )
    return mean_out, std_out


# ---------------------------------------------------------------------------
# Rolling reduce (sum / max / min)
# ---------------------------------------------------------------------------


def rolling_reduce(
    data: torch.Tensor,
    window: int,
    mode: str,
) -> torch.Tensor:
    """Rolling nansum/nanmax/nanmin over time axis.

    mode: 'sum', 'max', or 'min'.
    """
    data = _ensure_f32_contiguous(data)
    T, S = data.shape
    output = torch.full_like(data, float("nan"))

    if window <= 0 or T < window:
        return output

    mode_map = {"sum": 0, "max": 1, "min": 2}
    mode_int = mode_map[mode]

    BLOCK_S = _select_block_s(S)
    grid = (T - window + 1, (S + BLOCK_S - 1) // BLOCK_S)

    _rolling_reduce_kernel[grid](
        data,
        output,
        data.stride(0),
        data.stride(1),
        T,
        S,
        window,
        BLOCK_S,
        mode_int,
    )
    return output


# ---------------------------------------------------------------------------
# Parallel EMA
# ---------------------------------------------------------------------------


def parallel_ema(
    data: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """EMA with alpha = 2 / (window + 1). Handles NaN propagation."""
    data = _ensure_f32_contiguous(data)
    T, S = data.shape
    output = torch.full_like(data, float("nan"))

    if T == 0:
        return output

    alpha = 2.0 / (window + 1)
    BLOCK_S = _select_block_s(S)
    grid = ((S + BLOCK_S - 1) // BLOCK_S,)

    _parallel_ema_scan_kernel[grid](
        data,
        output,
        data.stride(0),
        data.stride(1),
        alpha,
        T,
        S,
        0,
        BLOCK_S,  # BLOCK_T unused but kept for signature compat
    )
    return output


# ---------------------------------------------------------------------------
# Rolling correlation / covariance (fused)
# ---------------------------------------------------------------------------


def rolling_corr_cov(
    x: torch.Tensor,
    y: torch.Tensor,
    window: int,
    mode: str,
) -> torch.Tensor:
    """Fused rolling correlation or covariance.

    mode: 'corr' or 'cov'. Single-pass computation, no intermediate arrays.
    """
    x = _ensure_f32_contiguous(x)
    y = _ensure_f32_contiguous(y)
    T, S = x.shape
    output = torch.full((T, S), float("nan"), dtype=torch.float32, device=x.device)

    if window <= 1 or T < window:
        return output

    is_corr = 1 if mode == "corr" else 0
    BLOCK_S = _select_block_s(S)
    grid = (T - window + 1, (S + BLOCK_S - 1) // BLOCK_S)

    _rolling_corr_cov_kernel[grid](
        x,
        y,
        output,
        x.stride(0),
        x.stride(1),
        T,
        S,
        window,
        BLOCK_S,
        is_corr,
    )
    return output


# ---------------------------------------------------------------------------
# Cross-sectional rank
# ---------------------------------------------------------------------------


def cs_rank(data: torch.Tensor) -> torch.Tensor:
    """Cross-sectional rank per row, normalized to [1/N, 1]. NaN-safe."""
    data = _ensure_f32_contiguous(data)
    T, S = data.shape
    output = torch.full_like(data, float("nan"))

    if T == 0 or S == 0:
        return output

    # BLOCK_S must be >= S since entire row is loaded per program
    BLOCK_S = max(triton.next_power_of_2(S), 32)
    grid = (T,)

    _cs_rank_kernel[grid](
        data,
        output,
        data.stride(0),
        data.stride(1),
        T,
        S,
        BLOCK_S,
    )
    return output


# ---------------------------------------------------------------------------
# Decay linear
# ---------------------------------------------------------------------------


def decay_linear(data: torch.Tensor, window: int) -> torch.Tensor:
    """Linearly weighted rolling average with weights [1, 2, ..., W]."""
    data = _ensure_f32_contiguous(data)
    T, S = data.shape
    output = torch.full_like(data, float("nan"))

    if window <= 0 or T < window:
        return output

    BLOCK_S = _select_block_s(S)
    grid = (T - window + 1, (S + BLOCK_S - 1) // BLOCK_S)

    _decay_linear_kernel[grid](
        data,
        output,
        data.stride(0),
        data.stride(1),
        T,
        S,
        window,
        BLOCK_S,
    )
    return output


# ---------------------------------------------------------------------------
# Batch rank IC
# ---------------------------------------------------------------------------


def batch_rank_ic(
    alphas: torch.Tensor,
    forward_returns: torch.Tensor,
) -> torch.Tensor:
    """Compute mean rank IC for a batch of factors.

    alphas: (N, T, S), forward_returns: (T, S)
    Returns: (N,) tensor of mean IC values.
    """
    alphas = _ensure_f32_contiguous(alphas)
    forward_returns = _ensure_f32_contiguous(forward_returns)

    N, T, S = alphas.shape
    BLOCK_S = max(triton.next_power_of_2(S), 32)

    ic_rows = torch.zeros((N, T), dtype=torch.float32, device=alphas.device)
    valid_counts = torch.zeros((N, T), dtype=torch.float32, device=alphas.device)

    grid = (N, T)

    _batch_rank_ic_kernel[grid](
        alphas,
        forward_returns,
        ic_rows,
        valid_counts,
        alphas.stride(0),
        alphas.stride(1),
        alphas.stride(2),
        forward_returns.stride(0),
        forward_returns.stride(1),
        N,
        T,
        S,
        BLOCK_S,
    )

    # Mean IC: average non-zero rows per factor
    valid_mask = valid_counts >= 2
    ic_rows = ic_rows * valid_mask.float()
    n_valid = valid_mask.float().sum(dim=1).clamp(min=1)
    mean_ic = ic_rows.sum(dim=1) / n_valid

    return mean_ic


# ---------------------------------------------------------------------------
# Factor correlation matrix
# ---------------------------------------------------------------------------


def factor_correlation_matrix(factors: torch.Tensor) -> torch.Tensor:
    """Compute full |Pearson correlation| matrix for N factor signals.

    factors: (N, D) where D = T * S (flattened signals).
    Returns: (N, N) symmetric matrix of |correlation| values.
    """
    factors = _ensure_f32_contiguous(factors)
    N, D = factors.shape
    corr_out = torch.zeros((N, N), dtype=torch.float32, device=factors.device)

    if N == 0:
        return corr_out

    BLOCK_D = min(triton.next_power_of_2(D), 4096)
    grid = (N, N)

    _factor_corr_matrix_kernel[grid](
        factors,
        corr_out,
        N,
        D,
        BLOCK_D,
    )
    return corr_out
