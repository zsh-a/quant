"""GPU-accelerated evaluation metrics using Triton kernels.

Drop-in replacements for evaluation.py functions when data is on GPU.
Falls back to CPU evaluation when Triton/CUDA is not available.
"""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from .gpu_ops import TRITON_AVAILABLE, batch_rank_ic
except Exception:  # pragma: no cover
    TRITON_AVAILABLE = False


def compute_forward_returns_gpu(
    close: torch.Tensor,
    periods: int = 1,
) -> torch.Tensor:
    """GPU version of compute_forward_returns. Input/output: CUDA float32 tensors."""
    fwd = torch.full_like(close, float("nan"))
    if periods < close.shape[0]:
        fwd[:-periods] = close[periods:] / (close[:-periods] + 1e-12) - 1.0
    return fwd


def compute_rank_ic_gpu(
    alpha: torch.Tensor,
    forward_returns: torch.Tensor,
) -> float:
    """GPU version of compute_rank_ic. Returns scalar mean IC."""
    mask = ~torch.isnan(alpha) & ~torch.isnan(forward_returns)
    valid_counts = mask.sum(dim=1)

    if not (valid_counts >= 2).any():
        return 0.0

    safe_alpha = torch.where(mask, alpha, torch.zeros_like(alpha))
    safe_returns = torch.where(mask, forward_returns, torch.zeros_like(forward_returns))
    denom = valid_counts.clamp(min=1).float()
    mean_alpha = safe_alpha.sum(dim=1) / denom
    mean_returns = safe_returns.sum(dim=1) / denom

    centered_alpha = torch.where(mask, alpha - mean_alpha[:, None], torch.zeros_like(alpha))
    centered_returns = torch.where(mask, forward_returns - mean_returns[:, None], torch.zeros_like(forward_returns))

    cov = (centered_alpha * centered_returns).sum(dim=1)
    var_alpha = (centered_alpha * centered_alpha).sum(dim=1)
    var_returns = (centered_returns * centered_returns).sum(dim=1)

    valid_rows = (valid_counts >= 2) & (var_alpha > 1e-24) & (var_returns > 1e-24)
    if not valid_rows.any():
        return 0.0

    correlations = cov[valid_rows] / torch.sqrt(var_alpha[valid_rows] * var_returns[valid_rows])
    return float(correlations.mean().item())


def compute_rank_ic_batch_gpu(
    alphas: torch.Tensor,
    forward_returns: torch.Tensor,
) -> torch.Tensor:
    """Compute IC for N factors simultaneously using Triton batch kernel.

    alphas: (N, T, S), forward_returns: (T, S) -> (N,) tensor of mean IC values.
    """
    if TRITON_AVAILABLE and alphas.is_cuda:
        return batch_rank_ic(alphas, forward_returns)

    # Fallback: loop over factors
    N = alphas.shape[0]
    ics = torch.zeros(N, dtype=torch.float32, device=alphas.device)
    for i in range(N):
        ics[i] = compute_rank_ic_gpu(alphas[i], forward_returns)
    return ics


def compute_ic_metrics_gpu(
    alpha: torch.Tensor,
    close: torch.Tensor,
    fwd_windows: list[int] | None = None,
) -> dict[str, float]:
    """GPU version of compute_ic_metrics.

    Returns dict with rank_ic, ic_ir, ic_std, rank_ic_{n}d, ic_decay,
    turnover_proxy, fitness.
    """
    fwd_windows = fwd_windows or [1, 5, 10]
    primary_window = fwd_windows[1] if len(fwd_windows) > 1 else fwd_windows[0]

    ics_by_window: dict[int, torch.Tensor] = {}
    for w in fwd_windows:
        fwd = compute_forward_returns_gpu(close, w)
        # Per-row IC
        mask = ~torch.isnan(alpha) & ~torch.isnan(fwd)
        valid_counts = mask.sum(dim=1)
        safe_a = torch.where(mask, alpha, torch.zeros_like(alpha))
        safe_r = torch.where(mask, fwd, torch.zeros_like(fwd))
        denom = valid_counts.clamp(min=1).float()
        mean_a = safe_a.sum(dim=1) / denom
        mean_r = safe_r.sum(dim=1) / denom
        ca = torch.where(mask, alpha - mean_a[:, None], torch.zeros_like(alpha))
        cr = torch.where(mask, fwd - mean_r[:, None], torch.zeros_like(fwd))
        cov_val = (ca * cr).sum(dim=1)
        va = (ca * ca).sum(dim=1)
        vr = (cr * cr).sum(dim=1)
        good = (valid_counts >= 2) & (va > 1e-24) & (vr > 1e-24)
        row_ics = torch.where(good, cov_val / (torch.sqrt(va * vr) + 1e-24), torch.full_like(cov_val, float("nan")))
        valid_ics = row_ics[~torch.isnan(row_ics)]
        ics_by_window[w] = valid_ics

    # Primary IC series
    primary_ics = ics_by_window.get(primary_window, torch.tensor([]))
    rank_ic = float(primary_ics.mean().item()) if primary_ics.numel() > 0 else 0.0
    ic_std = float(primary_ics.std().item()) if primary_ics.numel() > 0 else 0.0
    ic_ir = rank_ic / (ic_std + 1e-9)

    # Per-window mean IC
    per_window: dict[str, float] = {}
    for w in fwd_windows:
        arr = ics_by_window.get(w, torch.tensor([]))
        per_window[f"rank_ic_{w}d"] = float(arr.mean().item()) if arr.numel() > 0 else 0.0

    # IC decay
    if len(fwd_windows) >= 2:
        last_window = fwd_windows[-1]
        ic_decay = abs(rank_ic) - abs(per_window.get(f"rank_ic_{last_window}d", 0.0))
    else:
        ic_decay = 0.0

    # Turnover proxy
    if alpha.shape[0] > 1:
        a_curr = alpha[1:]
        a_prev = alpha[:-1]
        mask_tp = ~torch.isnan(a_curr) & ~torch.isnan(a_prev)
        if mask_tp.sum() > 10:
            ac = torch.where(mask_tp, a_curr, torch.zeros_like(a_curr)).flatten()
            ap = torch.where(mask_tp, a_prev, torch.zeros_like(a_prev)).flatten()
            ac_c = ac - ac.mean()
            ap_c = ap - ap.mean()
            denom_corr = torch.sqrt((ac_c**2).sum() * (ap_c**2).sum())
            autocorr = (ac_c * ap_c).sum() / (denom_corr + 1e-12) if denom_corr > 1e-12 else 0.0
            turnover_proxy = 1 - abs(float(autocorr))
        else:
            turnover_proxy = 1.0
    else:
        turnover_proxy = 1.0

    fitness = (rank_ic**2) / (ic_std + 1e-9)

    metrics: dict[str, float] = {
        "rank_ic": rank_ic,
        "ic_ir": ic_ir,
        "ic_std": ic_std,
        "ic_decay": ic_decay,
        "turnover_proxy": turnover_proxy,
        "fitness": fitness,
    }
    metrics.update(per_window)
    return metrics
