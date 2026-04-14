"""
因子正交化 — 消除因子间共线性，提升组合多样性。

提供两种方法:
  1. QR 分解正交化: 保持因子排列顺序，每个因子投影到前序因子的正交补空间
  2. 残差法: 对目标因子回归已有因子，取残差作为正交化信号
"""

from __future__ import annotations

import numpy as np
from loguru import logger


def orthogonalize_qr(signals: np.ndarray) -> np.ndarray:
    """QR 分解正交化。

    Args:
        signals: shape (n_factors, n_obs) — 每行是一个因子在所有 (time*symbols) 上的平坦化信号。

    Returns:
        shape (n_factors, n_obs) — 正交化后的信号矩阵（Q 矩阵的转置）。
        因子顺序保持不变，第一个因子不变，后续因子逐个去除前序影响。
    """
    n_factors, n_obs = signals.shape
    if n_factors <= 1:
        return signals.copy()

    # 处理 NaN: 替换为 0 用于 QR 分解
    clean = np.nan_to_num(signals, nan=0.0)

    # QR 分解: signals.T = Q @ R  =>  Q.T 的行就是正交化后的因子
    q, r = np.linalg.qr(clean.T, mode="reduced")
    orth = q.T  # (n_factors, n_obs)

    # 恢复尺度: QR 正交化后信号是单位长度的，按原始标准差重新缩放
    for i in range(n_factors):
        orig_std = np.nanstd(signals[i])
        orth_std = np.std(orth[i])
        if orth_std > 1e-12 and orig_std > 1e-12:
            orth[i] = orth[i] * (orig_std / orth_std)

    logger.info("orthogonalize_qr: {} factors, shape={}", n_factors, signals.shape)
    return orth


def residualize(
    target: np.ndarray,
    references: np.ndarray,
) -> np.ndarray:
    """残差法正交化: 目标因子回归参考因子组，取残差。

    Args:
        target: shape (n_obs,) — 待正交化的因子信号
        references: shape (n_refs, n_obs) — 参考因子矩阵

    Returns:
        shape (n_obs,) — 残差信号（正交于所有 references）
    """
    if references.shape[0] == 0:
        return target.copy()

    # NaN 安全
    mask = ~np.isnan(target)
    for ref in references:
        mask &= ~np.isnan(ref)

    if mask.sum() < 10:
        return target.copy()

    y = target[mask]
    X = references[:, mask].T  # (n_valid, n_refs)

    # OLS: y = X @ beta + epsilon
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return target.copy()

    residual = target.copy()
    residual[mask] = y - X @ beta
    return residual


def orthogonalize_sequential(signals: np.ndarray) -> np.ndarray:
    """逐步残差法正交化（Gram-Schmidt 风格）。

    第 i 个因子对前 i-1 个已正交化因子做回归取残差。
    比 QR 更稳健（逐步处理 NaN），适合金融时序数据。

    Args:
        signals: shape (n_factors, n_obs)

    Returns:
        shape (n_factors, n_obs) — 正交化后的信号
    """
    n_factors = signals.shape[0]
    result = np.empty_like(signals)
    result[0] = signals[0].copy()

    for i in range(1, n_factors):
        result[i] = residualize(signals[i], result[:i])

    logger.info("orthogonalize_sequential: {} factors orthogonalized", n_factors)
    return result
