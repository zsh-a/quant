"""
Multi-factor combination: select diverse factors from zoo and blend into a composite signal.

Usage:
    combiner = FactorCombiner(compiler, vm, schema)
    selected = combiner.select_factors(zoo_entries, dataset, max_factors=10)
    combined = combiner.combine(selected, dataset, method="ic_weighted")
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from loguru import logger

from ..core.compiler import FormulaCompiler
from ..core.dataset import AlphaDataset
from ..core.dsl import TensorSchema
from ..core.operators import OperatorRegistry
from ..core.vm import StackVM, TensorStore

try:
    import torch

    from ..eval.gpu_ops import TRITON_AVAILABLE as _TRITON_OK
    from ..eval.gpu_ops import cs_rank as _triton_cs_rank
    from ..eval.gpu_ops import factor_correlation_matrix as _triton_factor_corr
except Exception:  # pragma: no cover
    torch = None
    _TRITON_OK = False


def _can_use_gpu() -> bool:
    return bool(torch is not None and _TRITON_OK and torch.cuda.is_available())


@dataclass
class FactorSignal:
    """One evaluated factor: formula + its 2D signal array."""

    formula: str
    expr_hash: str
    signal: np.ndarray  # shape (time, symbols)
    rank_ic: float = 0.0
    fitness: float = 0.0


class FactorCombiner:
    """Select and combine alpha factors from the zoo into a composite signal."""

    def __init__(
        self,
        compiler: FormulaCompiler | None = None,
        vm: StackVM | None = None,
        schema: TensorSchema | None = None,
    ):
        registry = OperatorRegistry()
        self.compiler = compiler or FormulaCompiler(registry)
        self.vm = vm or StackVM()
        self.schema = schema or TensorSchema.default_market_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def materialize_factors(
        self,
        zoo_entries: list[dict[str, Any]],
        dataset: AlphaDataset,
        min_abs_ic: float = 0.0,
    ) -> list[FactorSignal]:
        """Compile each zoo formula and run through VM to get signal arrays.

        When min_abs_ic > 0, skips storing the full signal for entries whose
        stored rank_ic is below the threshold, reducing peak memory.
        """
        store = TensorStore(dataset.fields)
        if self.vm.backend == "torch" and self.vm.device is not None:
            store = self.vm._prepare_store(store)
        # Batch-compile and run via VM shared cache for memory efficiency.
        programs = []
        entries_with_program = []
        for entry in zoo_entries:
            formula = entry.get("formula", "")
            if not formula:
                continue
            # Pre-filter: skip entries whose stored IC is below threshold
            stored_ic = abs(float(entry.get("metrics", {}).get("rank_ic", 0) or 0))
            if min_abs_ic > 0 and stored_ic < min_abs_ic:
                continue
            try:
                program = self.compiler.compile(formula, self.schema)
                programs.append(program)
                entries_with_program.append(entry)
            except Exception as exc:
                logger.debug("combination.materialize skip formula={}: {}", formula[:60], exc)

        if not programs:
            return []

        # Batch execution shares intermediate cache, then release VM cache
        raw_outputs = self.vm.run_batch(programs, store)

        signals: list[FactorSignal] = []
        from ..core.vm import to_numpy

        for entry, program, raw in zip(entries_with_program, programs, raw_outputs):
            arr = to_numpy(raw).astype(np.float32)
            signals.append(
                FactorSignal(
                    formula=entry.get("formula", ""),
                    expr_hash=entry.get("expr_hash", program.expr_hash),
                    signal=arr,
                    rank_ic=float(entry.get("metrics", {}).get("rank_ic", 0) or 0),
                    fitness=float(entry.get("fitness", 0) or 0),
                )
            )
        return signals

    def select_factors(
        self,
        factors: list[FactorSignal],
        *,
        max_factors: int = 10,
        min_abs_ic: float = 0.01,
        max_correlation: float = 0.70,
    ) -> list[FactorSignal]:
        """
        Greedy forward selection: pick diverse, high-IC factors.

        1. Sort by |rank_ic| descending.
        2. For each candidate, check pairwise |corr| with already-selected.
        3. Accept only if all pairwise correlations < max_correlation.
        """
        # Filter by minimum IC
        candidates = [f for f in factors if abs(f.rank_ic) >= min_abs_ic]
        candidates.sort(key=lambda f: abs(f.rank_ic), reverse=True)

        # GPU path: batch correlation matrix
        if _can_use_gpu() and len(candidates) > 3:
            selected = self._select_factors_gpu(candidates, max_factors, max_correlation)
        else:
            selected = self._select_factors_cpu(candidates, max_factors, max_correlation)

        logger.info(
            "combination.select candidates={} selected={} min_ic={} max_corr={}",
            len(candidates),
            len(selected),
            min_abs_ic,
            max_correlation,
        )
        return selected

    def _select_factors_cpu(
        self,
        candidates: list[FactorSignal],
        max_factors: int,
        max_correlation: float,
    ) -> list[FactorSignal]:
        selected: list[FactorSignal] = []
        selected_flat: list[np.ndarray] = []
        for factor in candidates:
            if len(selected) >= max_factors:
                break
            flat = factor.signal.ravel()
            mask = np.isfinite(flat)
            if mask.sum() < 10:
                continue
            flat_clean = flat.copy()
            flat_clean[~mask] = 0.0
            is_diverse = True
            for existing in selected_flat:
                corr = _abs_corr(flat_clean, existing)
                if corr >= max_correlation:
                    is_diverse = False
                    break
            if is_diverse:
                selected.append(factor)
                selected_flat.append(flat_clean)
        return selected

    def _select_factors_gpu(
        self,
        candidates: list[FactorSignal],
        max_factors: int,
        max_correlation: float,
    ) -> list[FactorSignal]:
        # Build flattened signal matrix on GPU
        flat_signals = []
        valid_indices = []
        for i, factor in enumerate(candidates):
            flat = factor.signal.ravel()
            mask = np.isfinite(flat)
            if mask.sum() < 10:
                continue
            flat_clean = flat.copy()
            flat_clean[~mask] = 0.0
            flat_signals.append(flat_clean)
            valid_indices.append(i)
        if not flat_signals:
            return []

        signals_tensor = torch.tensor(
            np.stack(flat_signals),
            dtype=torch.float32,
            device="cuda",
        )
        corr_matrix = _triton_factor_corr(signals_tensor).cpu().numpy()

        # Greedy forward selection using precomputed matrix
        selected: list[FactorSignal] = []
        selected_idx: list[int] = []
        for local_i, global_i in enumerate(valid_indices):
            if len(selected) >= max_factors:
                break
            is_diverse = True
            for sel_local in selected_idx:
                if corr_matrix[local_i, sel_local] >= max_correlation:
                    is_diverse = False
                    break
            if is_diverse:
                selected.append(candidates[global_i])
                selected_idx.append(local_i)
        return selected

    def combine(
        self,
        factors: list[FactorSignal],
        dataset: AlphaDataset,
        method: str = "ic_weighted",
        ic_lookback: int = 60,
    ) -> np.ndarray:
        """
        Combine factor signals into a single composite signal.

        Methods:
            "equal"       — simple average of cs_rank(signal)
            "ic_weighted" — weight by rolling |rank IC| (adaptive)
            "ridge"       — ridge regression on forward returns
        """
        if not factors:
            raise ValueError("No factors to combine")
        if len(factors) == 1:
            return factors[0].signal

        # Stack ranked signals: (n_factors, time, symbols)
        if _can_use_gpu():
            ranked = np.stack([_cs_rank_auto(f.signal) for f in factors], axis=0)
        else:
            ranked = np.stack([_cs_rank(f.signal) for f in factors], axis=0)

        if method == "equal":
            return _equal_combine(ranked)
        elif method == "ic_weighted":
            close = dataset.fields.get("close")
            if close is None:
                return _equal_combine(ranked)
            return _ic_weighted_combine(ranked, close, lookback=ic_lookback)
        elif method == "ridge":
            close = dataset.fields.get("close")
            if close is None:
                return _equal_combine(ranked)
            return _ridge_combine(ranked, close)
        else:
            raise ValueError(f"Unknown combination method: {method}")

    def combine_from_zoo(
        self,
        zoo_entries: list[dict[str, Any]],
        dataset: AlphaDataset,
        *,
        method: str = "ic_weighted",
        max_factors: int = 10,
        min_abs_ic: float = 0.01,
        max_correlation: float = 0.70,
        ic_lookback: int = 60,
    ) -> dict[str, Any]:
        """End-to-end: materialize → select → combine → return result dict."""
        t0 = perf_counter()
        all_factors = self.materialize_factors(zoo_entries, dataset, min_abs_ic=min_abs_ic)
        t_materialize = perf_counter() - t0

        t1 = perf_counter()
        selected = self.select_factors(
            all_factors,
            max_factors=max_factors,
            min_abs_ic=min_abs_ic,
            max_correlation=max_correlation,
        )
        t_select = perf_counter() - t1

        if not selected:
            raise ValueError(
                f"No factors passed selection (zoo={len(zoo_entries)}, "
                f"materialized={len(all_factors)}, min_ic={min_abs_ic})"
            )

        # Optional orthogonalization before combining
        t_orth_start = perf_counter()
        if len(selected) > 1:
            from .orthogonalization import orthogonalize_sequential

            stacked = np.stack([f.signal.reshape(-1) for f in selected])
            orth_flat = orthogonalize_sequential(stacked)
            for i, f in enumerate(selected):
                f.signal = orth_flat[i].reshape(f.signal.shape)
        t_orth = perf_counter() - t_orth_start

        t2 = perf_counter()
        combined = self.combine(selected, dataset, method=method, ic_lookback=ic_lookback)
        t_combine = perf_counter() - t2

        logger.info(
            "combination.combine_from_zoo zoo={} materialized={} selected={} method={} "
            "materialize={:.3f}s select={:.3f}s orth={:.3f}s combine={:.3f}s",
            len(zoo_entries),
            len(all_factors),
            len(selected),
            method,
            t_materialize,
            t_select,
            t_orth,
            t_combine,
        )

        return {
            "combined_signal": combined,
            "selected_factors": [
                {
                    "formula": f.formula,
                    "expr_hash": f.expr_hash,
                    "rank_ic": f.rank_ic,
                    "fitness": f.fitness,
                }
                for f in selected
            ],
            "method": method,
            "timing": {
                "materialize_seconds": t_materialize,
                "select_seconds": t_select,
                "orthogonalize_seconds": t_orth,
                "combine_seconds": t_combine,
                "total_seconds": perf_counter() - t0,
            },
        }


# ---------------------------------------------------------------------------
# Combination methods
# ---------------------------------------------------------------------------


def _equal_combine(ranked: np.ndarray) -> np.ndarray:
    """Simple mean of ranked signals. ranked shape: (n_factors, time, symbols)."""
    return np.nanmean(ranked, axis=0)


def _ic_weighted_combine(
    ranked: np.ndarray,
    close: np.ndarray,
    lookback: int = 60,
) -> np.ndarray:
    """
    Adaptive IC-weighted combination.

    For each time step t, compute trailing |rank IC| per factor over
    [t-lookback, t), then weight proportionally.
    """
    n_factors, n_time, n_symbols = ranked.shape
    # Forward returns for IC calculation
    fwd = np.zeros_like(close)
    fwd[:-1] = close[1:] / (close[:-1] + 1e-12) - 1.0
    fwd = np.clip(fwd, -0.5, 0.5)

    # Pre-compute per-bar IC for each factor
    ic_series = np.zeros((n_factors, n_time), dtype=float)
    for k in range(n_factors):
        for t in range(n_time - 1):
            ic_series[k, t] = _cross_sectional_corr(ranked[k, t], fwd[t])

    # Rolling mean |IC| as weights
    combined = np.zeros((n_time, n_symbols), dtype=float)
    for t in range(n_time):
        start = max(0, t - lookback)
        if t - start < 2:
            # Not enough history → equal weight
            weights = np.ones(n_factors) / n_factors
        else:
            weights = np.array([np.nanmean(np.abs(ic_series[k, start:t])) for k in range(n_factors)])
            total = weights.sum()
            weights = weights / total if total > 1e-12 else np.ones(n_factors) / n_factors
        for k in range(n_factors):
            combined[t] += weights[k] * ranked[k, t]

    return combined


def _ridge_combine(
    ranked: np.ndarray,
    close: np.ndarray,
    alpha: float = 1.0,
    train_ratio: float = 0.6,
) -> np.ndarray:
    """
    Ridge regression: learn weights on train portion, apply to full dataset.

    X = factor signals (time × n_factors), y = forward returns (time,)
    Fits per-symbol then averages weights for robustness.
    """
    n_factors, n_time, n_symbols = ranked.shape
    fwd = np.zeros_like(close)
    fwd[:-1] = close[1:] / (close[:-1] + 1e-12) - 1.0
    fwd = np.clip(fwd, -0.5, 0.5)

    train_end = int(n_time * train_ratio)
    if train_end < n_factors + 2:
        return _equal_combine(ranked)

    # Pool all symbols in training set for stability
    X_train = ranked[:, :train_end, :].reshape(n_factors, -1).T  # (train*symbols, n_factors)
    y_train = fwd[:train_end, :].ravel()  # (train*symbols,)

    # Remove NaN rows
    valid = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    X_clean = X_train[valid]
    y_clean = y_train[valid]

    if X_clean.shape[0] < n_factors + 2:
        return _equal_combine(ranked)

    # Ridge: w = (X'X + αI)^{-1} X'y
    XtX = X_clean.T @ X_clean
    Xty = X_clean.T @ y_clean
    w = np.linalg.solve(XtX + alpha * np.eye(n_factors), Xty)

    # Normalize weights to sum to 1 (absolute)
    w_abs = np.abs(w).sum()
    if w_abs > 1e-12:
        w = w / w_abs

    # Apply weights to full dataset
    combined = np.zeros((n_time, n_symbols), dtype=float)
    for k in range(n_factors):
        combined += w[k] * ranked[k]

    logger.info("combination.ridge weights={}", {f"f{k}": f"{w[k]:.4f}" for k in range(n_factors)})
    return combined


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _cs_rank(signal: np.ndarray) -> np.ndarray:
    """Cross-sectional rank normalization to [0, 1] per row."""
    out = np.full_like(signal, np.nan, dtype=float)
    for t in range(signal.shape[0]):
        row = signal[t]
        valid = np.isfinite(row)
        n_valid = valid.sum()
        if n_valid < 2:
            out[t, valid] = 0.5
            continue
        ranks = np.zeros(n_valid)
        order = np.argsort(row[valid])
        ranks[order] = np.linspace(0, 1, n_valid)
        out[t, valid] = ranks
    return out


def _cross_sectional_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation between two 1D arrays, NaN-safe."""
    valid = np.isfinite(x) & np.isfinite(y)
    n = valid.sum()
    if n < 3:
        return 0.0
    xv, yv = x[valid], y[valid]
    xm, ym = xv - xv.mean(), yv - yv.mean()
    denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    if denom < 1e-12:
        return 0.0
    return float((xm * ym).sum() / denom)


def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Pearson correlation between two flat arrays."""
    valid = np.isfinite(a) & np.isfinite(b)
    n = valid.sum()
    if n < 5:
        return 0.0
    av, bv = a[valid], b[valid]
    am, bm = av - av.mean(), bv - bv.mean()
    denom = np.sqrt((am * am).sum() * (bm * bm).sum())
    if denom < 1e-12:
        return 1.0
    return abs(float((am * bm).sum() / denom))


def _cs_rank_auto(signal: np.ndarray) -> np.ndarray:
    """Cross-sectional rank with GPU acceleration when available."""
    if _can_use_gpu():
        t = torch.tensor(signal, dtype=torch.float32, device="cuda")
        ranked = _triton_cs_rank(t)
        return ranked.cpu().numpy()
    return _cs_rank(signal)
