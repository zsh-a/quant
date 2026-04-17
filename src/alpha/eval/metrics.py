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
    """Compute forward returns from close prices. Shape: (time, symbols).

    Returns are clipped to [-0.5, 0.5] to cap extreme moves.
    """
    fwd = np.full_like(close, np.nan, dtype=float)
    if periods < close.shape[0]:
        fwd[:-periods] = np.clip(
            close[periods:] / (close[:-periods] + 1e-12) - 1.0,
            -0.5,
            0.5,
        )
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

    fitness = (rank_ic**2) / (ic_std + 1e-9)

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


def compute_ic_trend(ic_values: list[float]) -> dict[str, float]:
    """Analyze IC trend over time: slope, half-life, significance.

    Args:
        ic_values: Time-ordered IC measurements (e.g., per-round evaluations).

    Returns:
        dict with: trend_slope, half_life_rounds, is_decaying (bool as 0/1).
    """
    n = len(ic_values)
    if n < 3:
        return {"trend_slope": 0.0, "half_life_rounds": float("inf"), "is_decaying": 0.0}

    arr = np.array(ic_values, dtype=np.float64)
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = arr.mean()
    ss_xy = np.sum((x - x_mean) * (arr - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    slope = float(ss_xy / ss_xx) if ss_xx > 0 else 0.0

    # Half-life: rounds until IC drops to 50% of current (assuming linear decline)
    current_ic = abs(arr[-1])
    if slope < 0 and current_ic > 1e-6:
        half_life = current_ic / (2.0 * abs(slope))
    else:
        half_life = float("inf")

    return {
        "trend_slope": round(slope, 6),
        "half_life_rounds": round(half_life, 1) if half_life != float("inf") else float("inf"),
        "is_decaying": 1.0 if slope < -1e-4 else 0.0,
    }


def compute_quantile_returns(
    alpha: np.ndarray,
    close: np.ndarray,
    n_quantiles: int = 5,
    periods: int = 1,
    timestamps: list[str] | None = None,
) -> dict[str, Any]:
    """分层回测：按因子值分 N 组，计算每组的累计收益和统计指标。

    Args:
        alpha: (T, S) 因子值矩阵
        close: (T, S) 收盘价矩阵
        n_quantiles: 分组数量（默认 5 = 五分位）
        periods: 收益计算周期
        timestamps: 可选时间戳列表 (len = T)

    Returns:
        dict with:
          quantile_returns: list of per-group cumulative return series
          quantile_stats: per-group 年化收益/夏普/最大回撤
          long_short_series: Q_top - Q_bottom 累计收益
          monotonicity: 单调性得分 (1.0 = 完美单调)
    """
    fwd = compute_forward_returns(close, periods)
    T, S = alpha.shape

    # Per-timestep quantile assignment
    group_returns = [[] for _ in range(n_quantiles)]

    for t in range(T - periods):
        a_row = alpha[t]
        r_row = fwd[t]
        valid = ~np.isnan(a_row) & ~np.isnan(r_row)
        n_valid = valid.sum()
        if n_valid < n_quantiles:
            for g in range(n_quantiles):
                group_returns[g].append(0.0)
            continue

        # Rank → quantile assignment
        ranks = np.full(S, np.nan)
        valid_idx = np.where(valid)[0]
        order = np.argsort(a_row[valid_idx])
        ranks[valid_idx[order]] = np.linspace(0, 1, len(order))

        for g in range(n_quantiles):
            lo = g / n_quantiles
            hi = (g + 1) / n_quantiles
            if g == n_quantiles - 1:
                in_group = valid & (ranks >= lo) & (ranks <= hi)
            else:
                in_group = valid & (ranks >= lo) & (ranks < hi)
            n_in = in_group.sum()
            if n_in > 0:
                group_returns[g].append(float(np.nanmean(r_row[in_group])))
            else:
                group_returns[g].append(0.0)

    # Cumulative equity per group
    quantile_equity = []
    quantile_stats = []
    for g in range(n_quantiles):
        rets = np.array(group_returns[g])
        equity = np.cumprod(1.0 + np.nan_to_num(rets, nan=0.0))
        quantile_equity.append(equity.tolist())
        mean_r = float(np.mean(rets))
        std_r = float(np.std(rets))
        peak = np.maximum.accumulate(equity)
        dd = np.where(peak > 1e-12, 1.0 - equity / peak, 0.0)
        quantile_stats.append(
            {
                "group": g + 1,
                "total_return": float(equity[-1] - 1.0) if equity.size else 0.0,
                "annual_return": mean_r * 252,
                "annual_sharpe": mean_r / (std_r + 1e-12) * np.sqrt(252),
                "max_drawdown": float(np.max(dd)) if dd.size else 0.0,
            }
        )

    # Long-short: top group - bottom group
    top_rets = np.array(group_returns[-1])
    bot_rets = np.array(group_returns[0])
    ls_rets = top_rets - bot_rets
    ls_equity = np.cumprod(1.0 + np.nan_to_num(ls_rets, nan=0.0))

    # Monotonicity: Spearman correlation of group_index vs group_total_return
    group_total_returns = [s["total_return"] for s in quantile_stats]
    if len(group_total_returns) >= 3:
        from scipy.stats import spearmanr

        mono_corr, _ = spearmanr(range(n_quantiles), group_total_returns)
        monotonicity = float(mono_corr) if np.isfinite(mono_corr) else 0.0
    else:
        monotonicity = 0.0

    # Timestamps for the x-axis (one per return period, excluding last `periods` rows)
    ts = timestamps[: T - periods] if timestamps else None

    return {
        "n_quantiles": n_quantiles,
        "quantile_equity": quantile_equity,
        "quantile_stats": quantile_stats,
        "long_short_equity": ls_equity.tolist(),
        "monotonicity": monotonicity,
        "timestamps": ts,
    }
