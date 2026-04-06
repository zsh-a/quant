"""Unified alpha evaluation metrics on numpy 2D arrays (time x symbols)."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_rank_ic(alpha: np.ndarray, forward_returns: np.ndarray) -> float:
    """Mean cross-sectional Pearson correlation between alpha and forward returns.

    Both inputs are 2D arrays of shape (time, symbols).
    Returns the average per-row correlation across time steps.
    """
    from ..core.vm import to_numpy
    alpha_np = to_numpy(alpha).astype(float)
    returns_np = to_numpy(forward_returns).astype(float)
    mask = ~np.isnan(alpha_np) & ~np.isnan(returns_np)
    valid_counts = np.sum(mask, axis=1)
    if not np.any(valid_counts >= 2):
        return 0.0

    safe_alpha = np.where(mask, alpha_np, 0.0)
    safe_returns = np.where(mask, returns_np, 0.0)
    denom = np.maximum(valid_counts, 1)
    mean_alpha = np.sum(safe_alpha, axis=1) / denom
    mean_returns = np.sum(safe_returns, axis=1) / denom
    centered_alpha = np.where(mask, alpha_np - mean_alpha[:, None], 0.0)
    centered_returns = np.where(mask, returns_np - mean_returns[:, None], 0.0)

    cov = np.sum(centered_alpha * centered_returns, axis=1)
    var_alpha = np.sum(centered_alpha * centered_alpha, axis=1)
    var_returns = np.sum(centered_returns * centered_returns, axis=1)
    valid_rows = (valid_counts >= 2) & (var_alpha > 1e-24) & (var_returns > 1e-24)
    if not np.any(valid_rows):
        return 0.0

    correlations = cov[valid_rows] / np.sqrt(var_alpha[valid_rows] * var_returns[valid_rows])
    return float(np.mean(correlations)) if correlations.size else 0.0


def compute_forward_returns(close: np.ndarray, periods: int = 1) -> np.ndarray:
    """Compute forward returns from close prices. Shape: (time, symbols)."""
    fwd = np.full_like(close, np.nan, dtype=float)
    if periods < close.shape[0]:
        fwd[:-periods] = close[periods:] / (close[:-periods] + 1e-12) - 1.0
    return fwd


def compute_ic_metrics(
    alpha: np.ndarray,
    close: np.ndarray,
    fwd_windows: list[int] | None = None,
) -> dict[str, float]:
    """Compute IC-based metrics for an alpha signal.

    Returns dict with rank_ic, ic_ir, ic_std, rank_ic_{n}d for each window,
    ic_decay, and turnover_proxy.
    """
    from ..core.vm import to_numpy
    alpha = to_numpy(alpha).astype(float)
    close = to_numpy(close).astype(float)
    fwd_windows = fwd_windows or [1, 5, 10]
    primary_window = fwd_windows[1] if len(fwd_windows) > 1 else fwd_windows[0]

    ics_by_window: dict[int, list[float]] = {}
    for w in fwd_windows:
        fwd = compute_forward_returns(close, w)
        # Compute per-row IC
        mask = ~np.isnan(alpha) & ~np.isnan(fwd)
        valid_counts = np.sum(mask, axis=1)
        safe_a = np.where(mask, alpha, 0.0)
        safe_r = np.where(mask, fwd, 0.0)
        denom = np.maximum(valid_counts, 1)
        mean_a = np.sum(safe_a, axis=1) / denom
        mean_r = np.sum(safe_r, axis=1) / denom
        ca = np.where(mask, alpha - mean_a[:, None], 0.0)
        cr = np.where(mask, fwd - mean_r[:, None], 0.0)
        cov_val = np.sum(ca * cr, axis=1)
        va = np.sum(ca * ca, axis=1)
        vr = np.sum(cr * cr, axis=1)
        good = (valid_counts >= 2) & (va > 1e-24) & (vr > 1e-24)
        row_ics = np.where(good, cov_val / np.sqrt(va * vr + 1e-24), np.nan)
        ics_by_window[w] = row_ics[~np.isnan(row_ics)].tolist()

    # Primary IC series
    primary_ics = np.array(ics_by_window.get(primary_window, []))
    rank_ic = float(np.mean(primary_ics)) if primary_ics.size else 0.0
    ic_std = float(np.std(primary_ics)) if primary_ics.size else 0.0
    ic_ir = rank_ic / (ic_std + 1e-9)

    # Per-window mean IC
    per_window: dict[str, float] = {}
    for w in fwd_windows:
        arr = np.array(ics_by_window.get(w, []))
        per_window[f"rank_ic_{w}d"] = float(np.mean(arr)) if arr.size else 0.0

    # IC decay
    if len(fwd_windows) >= 2:
        last_window = fwd_windows[-1]
        ic_decay = abs(rank_ic) - abs(per_window.get(f"rank_ic_{last_window}d", 0.0))
    else:
        ic_decay = 0.0

    # Turnover proxy: 1 - |autocorrelation of alpha|
    if alpha.shape[0] > 1:
        a_curr = alpha[1:]
        a_prev = alpha[:-1]
        mask_tp = ~np.isnan(a_curr) & ~np.isnan(a_prev)
        if np.sum(mask_tp) > 10:
            ac = np.where(mask_tp, a_curr, 0.0).flatten()
            ap = np.where(mask_tp, a_prev, 0.0).flatten()
            ac_centered = ac - np.mean(ac)
            ap_centered = ap - np.mean(ap)
            denom_corr = np.sqrt(np.sum(ac_centered**2) * np.sum(ap_centered**2))
            autocorr = np.sum(ac_centered * ap_centered) / (denom_corr + 1e-12) if denom_corr > 1e-12 else 0.0
            turnover_proxy = 1 - abs(autocorr)
        else:
            turnover_proxy = 1.0
    else:
        turnover_proxy = 1.0

    fitness = (rank_ic ** 2) / (ic_std + 1e-9)

    metrics = {
        "rank_ic": rank_ic,
        "ic_ir": ic_ir,
        "ic_std": ic_std,
        "ic_decay": ic_decay,
        "turnover_proxy": turnover_proxy,
        "fitness": fitness,
    }
    metrics.update(per_window)
    return metrics
